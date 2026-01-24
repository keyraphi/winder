#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cub/block/block_scan.cuh>
#include <cub/util_type.cuh>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_atomic_functions.h>
#include <driver_types.h>
#include <memory>
#include <stdexcept>
#include <sys/types.h>
#include <thrust/copy.h>
#include <thrust/detail/vector_base.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <vector_functions.h>
#include <vector_types.h>

#include "aabb.h"
#include "binary_node.h"
#include "bvh8.h"
#include "mat3x3.h"
#include "tailor_coefficients.cuh"
#include "tailor_coefficients.h"
#include "tensor3.h"
#include "utils.h"
#include "vec3.h"
#include "winder_cuda.h"

#define CUDA_CHECK(expr_to_check)                                              \
  do {                                                                         \
    cudaError_t result = expr_to_check;                                        \
    if (result != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA Runtime Error: %s:%i:%d = %s\n", __FILE__,         \
              __LINE__, result, cudaGetErrorString(result));                   \
    }                                                                          \
  } while (0)

void CudaDeleter::operator()(void *ptr) const { cudaFree(ptr); }

auto WinderBackend::get_bvh_view() -> BVH8View {
  return BVH8View{thrust::raw_pointer_cast(m_bvh8_nodes.data()),
                  thrust::raw_pointer_cast(m_sorted_geometry.data()),
                  (uint32_t)m_bvh8_nodes.size()};
}

auto WinderBackend::CreateFromMesh(const float *trisangles,
                                   size_t triangle_count, int device_id)
    -> std::unique_ptr<WinderBackend> {
  auto self = std::unique_ptr<WinderBackend>{
      new WinderBackend(WinderMode::Triangle, triangle_count, device_id)};

  self->initialize_mesh_data(trisangles); // Sorts and builds tree
  return self;
}

auto WinderBackend::CreateFromPoints(const float *points, const float *normals,
                                     size_t point_count, int device_id)
    -> std::unique_ptr<WinderBackend> {
  auto self = std::unique_ptr<WinderBackend>{
      new WinderBackend(WinderMode::Point, point_count, device_id)};
  self->initialize_point_data(points,
                              normals); // Interleaves, sorts, and builds tree
  return self;
}

auto WinderBackend::CreateForSolver(const float *points, size_t point_count,
                                    int device_id)
    -> std::unique_ptr<WinderBackend> {
  auto self = std::unique_ptr<WinderBackend>{
      new WinderBackend(WinderMode::Point, point_count, device_id)};
  self->initialize_point_data(
      points, nullptr); // Interleaves (with 0 normals), sorts, and builds tree
  return self;
}

WinderBackend::WinderBackend(WinderMode mode, size_t size, int device_id)
    : m_mode(mode), m_count(size), m_device(device_id) {
  ScopedCudaDevice device_scope(device_id);

  size_t floats_per_elem = (mode == WinderMode::Triangle) ? 9 : 6;
  size_t leaf_count = (size + LEAF_SIZE - 1) / LEAF_SIZE;

  // preallocate memory. No initialization!
  m_sorted_geometry.resize(size * floats_per_elem, thrust::no_init);
  m_to_internal.resize(size, thrust::no_init);
  m_to_canonical.resize(size, thrust::no_init);
  m_bvh8_nodes.resize(1.2 * leaf_count, thrust::no_init);
  m_leaf_coefficients.resize(leaf_count, thrust::no_init);
  m_morton_codes.resize(size, thrust::no_init);
  m_leaf_morton_codes.resize(leaf_count, thrust::no_init);
  m_binary_nodes.resize(leaf_count - 1, thrust::no_init);
  m_binary_parents.resize(2 * leaf_count - 1, thrust::no_init);
  m_binary_aabbs.resize(2 * leaf_count - 1, thrust::no_init);

  // counters are initialized to 0
  m_atomic_counters.resize(leaf_count - 1);
}

// Gather kernel that interleaves positions and normals in the sorted geometry
// array
__global__ void __launch_bounds__(256)
    interleave_gather_geometry(const float *__restrict__ points,
                               const float *__restrict__ normals,
                               const uint32_t *__restrict__ indices,
                               float *__restrict__ out_geometry,
                               const uint32_t count) {
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count)
    return;

  const uint32_t src_idx = indices[idx];
  const uint32_t src_offset = src_idx * 3;

  // Vectorized Loads (Uncoalesced, but fewer instructions)
  // We treat the float3 as a float2 + float1 to hit the 64-bit and 32-bit paths
  // This is faster than 3 individual float loads.
  float2 p_xy = reinterpret_cast<const float2 *>(points + src_offset)[0];
  float p_z = points[src_offset + 2];

  float2 n_xy = reinterpret_cast<const float2 *>(normals + src_offset)[0];
  float n_z = normals[src_offset + 2];

  // We write 24 bytes as 3 float2 transactions
  const uint32_t dst_offset = idx * 6;
  const float2 out1 = make_float2(p_xy.x, p_xy.y);
  const float2 out2 = make_float2(p_z, n_xy.x);
  const float2 out3 = make_float2(n_xy.y, n_z);

  float2 *f2_base_ptr = reinterpret_cast<float2 *>(out_geometry + dst_offset);
  f2_base_ptr[0] = out1;
  f2_base_ptr[1] = out2;
  f2_base_ptr[2] = out3;
}

// Longest Common Prefix for code[i] and code[j]
__device__ inline auto delta(int i, int j, const uint32_t *__restrict__ codes,
                             int codes_len) -> int {
  if (j < 0 || j >= codes_len) {
    return -1;
  }
  uint32_t x = codes[i];
  uint32_t y = codes[j];
  if (x == y) {
    // tie break
    constexpr int tiebreaker_offset = 32;
    return tiebreaker_offset + __clz(i ^ j);
  }
  return __clz((int)x ^ (int)y);
}

__global__ void build_binary_topology_kernel(
    const uint32_t
        *__restrict__ morton_codes, // morton code of first leaf entry
    BinaryNode *nodes, uint32_t *parents, const uint32_t leaf_count) {
  uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_idx >= leaf_count - 1) {
    return;
  }

  // Compute range direction (+1 or -1)
  int direction =
      (delta(thread_idx, thread_idx + 1, morton_codes, leaf_count) -
           delta(thread_idx, thread_idx - 1, morton_codes, leaf_count) >=
       0)
          ? 1
          : -1;

  // find bit position where prefix change happens (full node)
  int parent_prefix =
      delta(thread_idx, thread_idx - direction, morton_codes, leaf_count);
  int node_end_bit_mask = 2;
  while (delta(thread_idx, thread_idx + node_end_bit_mask * direction,
               morton_codes, leaf_count) > parent_prefix) {
    node_end_bit_mask *= 2;
  }

  // find the index in the morton code array where this bitflip happens
  int total_idx_range_length = 0;
  for (int search_step = node_end_bit_mask / 2; search_step >= 1;
       search_step /= 2) {
    if (delta(thread_idx,
              thread_idx + (total_idx_range_length + search_step) * direction,
              morton_codes, leaf_count) > parent_prefix) {
      total_idx_range_length += search_step;
    }
  }
  int range_end = (int)thread_idx + total_idx_range_length * direction;

  // Find the split point. Index where bit flip happens inside the node.
  int node_prefix = delta(thread_idx, range_end, morton_codes, leaf_count);
  int split_offset = 0;

  // start with largest power of 2 smaller than total_idx_range_length
  int search_step = 1 << (31 - __clz(total_idx_range_length));
  for (; search_step >= 1; search_step >>= 1) {
    int candidate_offset = split_offset + search_step;
    if (candidate_offset < total_idx_range_length) {
      if (delta(thread_idx, thread_idx + candidate_offset * direction,
                morton_codes, leaf_count) > node_prefix) {
        split_offset = candidate_offset;
      }
    }
  }
  // if direction == 1: spli_offset is where the left child ends.
  // if direction == -1: spli_offset is where the left child ends.
  int split_idx =
      (int)thread_idx + split_offset * direction + min(direction, 0);

  // determine child indices
  uint32_t left_node_idx =
      (min((int)thread_idx, range_end) == split_idx)
          ? (split_idx + (leaf_count - 1)) // its a leaf node
          : split_idx; // this node is responsible for the left side
  uint32_t right_node_idx =
      (max((int)thread_idx, range_end) == split_idx + 1)
          ? (split_idx + 1 + (leaf_count - 1)) // its a leaf node
          : (split_idx + 1); // this node is responsible for the right side

  // store childs of this node
  nodes[thread_idx].left_child = left_node_idx;
  nodes[thread_idx].right_child = right_node_idx;
  // for bottom up traversal tell those childs who therir parent is
  parents[left_node_idx] = thread_idx;
  parents[right_node_idx] = thread_idx;
}

__device__ __forceinline__ auto warp_reduce_add_down(float val) -> float {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  }
  return val;
}

__global__ void populate_binary_tree_aabb_and_leaf_coefficients(
    const float *__restrict__ sorted_geometry,
    TailorCoefficientsBf16 *leaf_coefficients, const uint32_t leaf_count,
    const BinaryNode *binary_nodes, AABB *binary_aabbs,
    const uint32_t *binary_parents, uint32_t *atomic_counters,
    const uint32_t point_count) {
  uint32_t point_idx = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t leaf_idx = point_idx / 32;
  if (leaf_idx >= leaf_count)
    return;

  uint32_t lane_id = point_idx % 32;

  Vec3 p, n;
  bool is_thread_active = point_idx < point_count;
  if (is_thread_active) {
    const auto *base =
        reinterpret_cast<const float2 *>(sorted_geometry + point_idx * 6);

    float2 chunk0 = base[0]; // px, py
    float2 chunk1 = base[1]; // pz, nx
    float2 chunk2 = base[2]; // ny, nz

    p = {chunk0.x, chunk0.y, chunk1.x};
    n = {chunk1.y, chunk2.x, chunk2.y};
  } else {
    p = {1e38F, 1e38F, 1e38F}; // neutral element for min reduction
    n = {0.F, 0.F, 0.F};
  }

  // Compute AABB
  Vec3 p_min = p;
  Vec3 p_max =
      is_thread_active
          ? p
          : Vec3{-1e38F, -1e38F, -1e38F}; // netural element for max reduction

#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    p_min.x = fminf(p_min.x, __shfl_xor_sync(0xFFFFFFFF, p_min.x, offset));
    p_min.y = fminf(p_min.y, __shfl_xor_sync(0xFFFFFFFF, p_min.y, offset));
    p_min.z = fminf(p_min.z, __shfl_xor_sync(0xFFFFFFFF, p_min.z, offset));

    p_max.x = fmaxf(p_max.x, __shfl_xor_sync(0xFFFFFFFF, p_max.x, offset));
    p_max.y = fmaxf(p_max.y, __shfl_xor_sync(0xFFFFFFFF, p_max.y, offset));
    p_max.z = fmaxf(p_max.z, __shfl_xor_sync(0xFFFFFFFF, p_max.z, offset));
  }
  // write aggregated AABB for leaf
  if (lane_id == 0) {
    binary_aabbs[leaf_idx + leaf_count - 1].min = p_min;
    binary_aabbs[leaf_idx + leaf_count - 1].max = p_max;
  }

  // Compute tailor coefficients
  // first and second order use aabb center
  Vec3 center = (p_min + p_max) * 0.5;

  // Zero order
  //\sum_{i=1}^m a_i n_i
  // for inactive threads neutral element wrt. +
  Vec3 zero_order = is_thread_active ? n : Vec3{0.F, 0.F, 0.F};
  zero_order.x = warp_reduce_add_down(zero_order.x);
  zero_order.y = warp_reduce_add_down(zero_order.y);
  zero_order.z = warp_reduce_add_down(zero_order.z);

  // write aggregated coefficient for leaf
  if (lane_id == 0) {
    leaf_coefficients[leaf_idx].zero_order = zero_order;
  }

  Vec3 d = p - center;
  // First order
  // \sum_{i=1}^m a_i(p_i-p_center)\otimes n_i
  // for inactive threads neutral element wrt. +
  Mat3x3 first_order =
      is_thread_active ? d.outer_product(n)
                       : Mat3x3{0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F};

#pragma unroll
  for (int i = 0; i < 9; ++i) {
    first_order.data[i] = warp_reduce_add_down(first_order.data[i]);
  }

  // write aggregated coefficient for leaf
  if (lane_id == 0) {
    leaf_coefficients[leaf_idx].first_order = first_order;
  }

  // second order
  // 1/2 (\sum_{i=1}^m a_i(p_i - p_center)\otimes (p_i-p_center) \otimes n)
  Tensor3_compressed second_order;
  if (is_thread_active) {
    second_order.data[0] = d.x * d.x * n.x;
    second_order.data[1] = d.x * d.x * n.y;
    second_order.data[2] = d.x * d.x * n.z;
    second_order.data[3] = d.x * d.y * n.x;
    second_order.data[4] = d.x * d.y * n.y;
    second_order.data[5] = d.x * d.y * n.z;
    second_order.data[6] = d.x * d.z * n.x;
    second_order.data[7] = d.x * d.z * n.y;
    second_order.data[8] = d.x * d.z * n.z;
    // t[-] = d.y*d.x*n.x = t[3]  not store twice
    // t[-] = d.y*d.x*n.y = t[4]
    // t[-] = d.y*d.x*n.z = t[5]
    second_order.data[9] = d.y * d.y * n.x;
    second_order.data[10] = d.y * d.y * n.y;
    second_order.data[11] = d.y * d.y * n.z;
    second_order.data[12] = d.y * d.z * n.x;
    second_order.data[13] = d.y * d.z * n.y;
    second_order.data[14] = d.y * d.z * n.z;
    // t[-] = d.z*d.x*n.x = t[6]
    // t[-] = d.z*d.x*n.y = t[7]
    // t[-] = d.z*d.x*n.z = t[8]
    // t[-] = d.z*d.y*n.x = t[12]
    // t[-] = d.z*d.y*n.y = t[13]
    // t[-] = d.z*d.y*n.z = t[14]
    second_order.data[15] = d.z * d.z * n.x;
    second_order.data[16] = d.z * d.z * n.y;
    second_order.data[17] = d.z * d.z * n.z;
  } else {
    // for inactive threads neutral element wrt. +
#pragma unroll
    for (int i = 0; i < 18; ++i) {
      second_order.data[i] = 0.F;
    }
  }
#pragma unroll
  for (int i = 0; i < 18; ++i) {
    second_order.data[i] = warp_reduce_add_down(second_order.data[i]);
  }

  // write aggregated coefficient for leaf
  if (lane_id == 0) {
    leaf_coefficients[leaf_idx].second_order = second_order;
  }

  // propagate the aabbs to the inner nodes (binary_aabbs)
  // lane 0 handles the Leaf-to-Root Race
  if (lane_id == 0) {
    uint32_t current_idx = leaf_idx + leaf_count - 1;
    uint32_t current_parent_idx = binary_parents[current_idx];

    // atomicAdd returns the value BEFORE the addition.
    // If it returns 0, we are the first child to arrive -> Terminate.
    // If it returns 1, we are the second child -> We own the parent.
    if (atomicAdd(&atomic_counters[current_parent_idx], 1) == 0) {
      return;
    }
    while (current_parent_idx != 0xFFFFFFFF) {
      BinaryNode parent_node = binary_nodes[current_parent_idx];

      // merge child aabbs
      binary_aabbs[current_parent_idx] =
          AABB::merge(binary_aabbs[parent_node.left_child],
                      binary_aabbs[parent_node.right_child]);
      // make sure that aabb has been written before the counter is incremented
      cuda::atomic_thread_fence(cuda::memory_order_seq_cst,
                                cuda::thread_scope_device);

      // Move up to the next level
      uint32_t next_parent = binary_parents[current_parent_idx];
      if (next_parent == 0xFFFFFFFF) {
        return;
      }

      if (atomicAdd(&atomic_counters[next_parent], 1) == 0) {
        return;
      }
      current_parent_idx = next_parent;
    }
  }
}

__global__ void convert_binary_tree_to_bvh8(
    const BinaryNode *binary_nodes, const AABB *binary_aabbs,
    BVH8Node *bvh8_nodes, uint32_t *bvh8_leaf_parents,
    uint32_t *bvh8_leaf_pointers, uint32_t *bvh8_internal_parents,
    const uint32_t *work_queue_in, const uint32_t work_queue_length,
    uint32_t *work_queue_out, uint32_t *global_counter,
    const uint32_t leaf_count, const uint32_t level_offset) {
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= work_queue_length) {
    return;
  }

  uint32_t children[8];
  uint32_t my_binary_idx = work_queue_in[idx];
  children[0] = my_binary_idx;
  uint32_t child_count = 1;
  for (int iter = 0; iter < 7; ++iter) {
    float max_diag = -1.F;
    int split_idx = -1;

    // Find the best candidate to split
    for (int i = 0; i < child_count; ++i) {
      uint32_t binary_idx = children[i];
      if (!BinaryNode::is_leaf(binary_idx, leaf_count)) {
        // find the inner node with the largest aabb
        float d = binary_aabbs[binary_idx].radius_sq();
        if (d > max_diag) {
          max_diag = d;
          split_idx = i;
        }
      }
    }

    if (split_idx == -1) {
      break; // No more internal nodes to split
    }

    // Split the winner
    uint32_t node_to_split = children[split_idx];
    children[split_idx] = binary_nodes[node_to_split].left_child;
    children[child_count] = binary_nodes[node_to_split].right_child;
    child_count++;
  }
  // count how many internal nodes (non leafs) were found
  uint32_t internal_node_count = 0;
  for (uint32_t i = 0; i < child_count; ++i) {
    if (!BinaryNode::is_leaf(children[i], leaf_count)) {
      internal_node_count++;
    }
  }

  // We need to know where in the bvh8_nodes array to store the children
  // (child_base) prefix sum over internal node count to find out the offset for
  // this thread, at least locally in this block.
  typedef cub::BlockScan<uint32_t, 256>
      BlockScan; // Assuming 256 threads per block
  __shared__ typename BlockScan::TempStorage temp_storage;

  uint32_t thread_offset;
  uint32_t total_internal_in_block;
  BlockScan(temp_storage)
      .ExclusiveSum(internal_node_count, thread_offset,
                    total_internal_in_block);

  // Get device wide offset of this block by claiming the current value from the
  // global counter. Only done once per block
  __shared__ uint32_t global_base;
  if (threadIdx.x == 0) {
    // next block that claims the counter will leaf space for all internal nodes
    // in this block.
    // This counter can be used to find out how long work_queue_out is in the
    // end.
    global_base = atomicAdd(global_counter, total_internal_in_block);
  }
  __syncthreads();

  // This thread will write its childs at its offset within the block
  uint32_t my_child_base = global_base + thread_offset + level_offset;

  BVH8Node out_node;
  uint32_t bvh8_idx = idx + level_offset;
  AABB parent_aabb = binary_aabbs[my_binary_idx];
  out_node.parent_aabb = parent_aabb;
  out_node.child_base = my_child_base;

  // AABB Quantization & Writing out the BVH8Node
  Vec3 parent_min = parent_aabb.min;
  Vec3 parent_inv_ext = 255.0f / (parent_aabb.max - parent_aabb.min);

  int internal_found = 0;
  uint32_t my_leaf_indices[8];

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    if (i < child_count) {
      uint32_t binary_child_idx = children[i];
      bool is_child_leaf = BinaryNode::is_leaf(binary_child_idx, leaf_count);

      // Quantize Child AABB
      AABB child_box = binary_aabbs[binary_child_idx];

      out_node.child_aabb_approx[i] =
          AABB8BitApprox::quantize_aabb(child_box, parent_min, parent_inv_ext);

      if (is_child_leaf) {
        out_node.child_meta[i] = ChildType::LEAF;
        uint32_t leaf_raw_idx = binary_child_idx - (leaf_count - 1);

        my_leaf_indices[i] = leaf_raw_idx;

        // Let the leaf know which BVH8Node is its parent
        bvh8_leaf_parents[leaf_raw_idx] = bvh8_idx;
      } else {
        out_node.child_meta[i] = ChildType::INTERNAL;
        // Calculate the specific global index for this child in the next level
        uint32_t next_bvh8_idx = my_child_base + internal_found;

        // The work_queue_out points to the inner BinaryNode that must be turned
        // into BVH8Node in the next kernel call
        work_queue_out[global_base + thread_offset + internal_found] =
            binary_child_idx;

        bvh8_internal_parents[next_bvh8_idx] = bvh8_idx;

        my_leaf_indices[i] = 0xFFFFFFFF;
        internal_found++; // Increment after use
      }
    } else {
      out_node.child_meta[i] = ChildType::EMPTY;
      my_leaf_indices[i] = 0xFFFFFFFF;
    }
  }

  // let the inner nodes know how many childs they have
  out_node.tailor_coefficients.set_expected_children(child_count);

  // Coalesced Writes
  bvh8_nodes[bvh8_idx] = out_node;

  // Parallel Leaf Pointers (32-byte vectorized write)
  auto *leaf_ptr_v4 = reinterpret_cast<uint4 *>(bvh8_leaf_pointers);
  leaf_ptr_v4[bvh8_idx * 2] =
      make_uint4(my_leaf_indices[0], my_leaf_indices[1], my_leaf_indices[2],
                 my_leaf_indices[3]);
  leaf_ptr_v4[bvh8_idx * 2 + 1] =
      make_uint4(my_leaf_indices[4], my_leaf_indices[5], my_leaf_indices[6],
                 my_leaf_indices[7]);
}

void WinderBackend::initialize_point_data(const float *points,
                                          const float *normals) {
  // compute scene bounds
  const thrust::device_ptr<const Vec3> points_begin(
      reinterpret_cast<const Vec3 *>(points));
  auto aabb_transform = thrust::make_transform_iterator(
      points_begin, [] __host__ __device__(const Vec3 &point) -> AABB {
        return AABB::from_point(point);
      });
  AABB scene_bounds = thrust::reduce(
      aabb_transform, aabb_transform + m_count, AABB::empty(),
      [] __host__ __device__(const AABB &a, const AABB &b) -> AABB {
        return AABB::merge(a, b);
      });

  // sort m_to_internal by morton codes of points
  Vec3 extent = scene_bounds.max - scene_bounds.min;
  float max_dim = fmaxf(extent.x, fmaxf(extent.y, extent.z));
  float scale = (max_dim > 1e-9f) ? 1.0f / max_dim : 0.0f;
  Vec3 min_p = scene_bounds.min;

  thrust::transform(
      points_begin, points_begin + m_count, m_morton_codes.begin(),
      [min_p, scale] __host__ __device__(const Vec3 &p) -> uint32_t {
        // Scale to range [0, 1]
        float tx = (p.x - min_p.x) * scale;
        float ty = (p.y - min_p.y) * scale;
        float tz = (p.z - min_p.z) * scale;

        // Scale to 10-bit integer range [0, 1023]
        auto x = static_cast<uint32_t>(fminf(fmaxf(tx * 1024.F, 0.F), 1023.F));
        auto y = static_cast<uint32_t>(fminf(fmaxf(ty * 1024.F, 0.F), 1023.F));
        auto z = static_cast<uint32_t>(fminf(fmaxf(tz * 1024.F, 0.F), 1023.F));

        // Expand bits (interleave x, y, z)
        return morton3D_30bit(x, y, z);
      });

  thrust::sequence(m_to_internal.begin(), m_to_internal.end());
  // sorts both morton_codes and m_to_internal
  thrust::sort_by_key(m_morton_codes.begin(), m_morton_codes.end(),
                      m_to_internal.begin());

  {
    const size_t threads = 256;
    const size_t block_size = (m_count + threads - 1) / threads;
    interleave_gather_geometry<<<block_size, threads>>>(
        points, normals, m_to_internal.data().get(),
        m_sorted_geometry.data().get(), m_count);
    CUDA_CHECK(cudaGetLastError());
  }

  uint32_t leaf_count = (m_morton_codes.size() + LEAF_SIZE - 1) / LEAF_SIZE;
  {
    auto morton_leaf_stride_idx = thrust::make_transform_iterator(
        thrust::make_counting_iterator<uint32_t>(0),
        [] __host__ __device__(uint32_t i) -> uint32_t {
          return i * LEAF_SIZE;
        });
    // thrust::make_strided_iterator<LEAF_SIZE>(m_morton_codes.begin());
    auto morton_leaf_stride = thrust::make_permutation_iterator(
        m_morton_codes.begin(), morton_leaf_stride_idx);
    thrust::copy(morton_leaf_stride, morton_leaf_stride + leaf_count,
                 m_leaf_morton_codes.begin());
    const size_t threads = 256;
    const size_t grid_size = (leaf_count - 1 + threads - 1) / threads;
    build_binary_topology_kernel<<<grid_size, threads>>>(
        m_leaf_morton_codes.data().get(), m_binary_nodes.data().get(),
        m_binary_parents.data().get(), leaf_count);
    CUDA_CHECK(cudaGetLastError());
    // the root node gets a unique value for its parent
    m_binary_parents[0] = 0xFFFFFFFF;
  }
  {
    const size_t threads = 256;
    const size_t grid_size = (leaf_count * LEAF_SIZE + threads - 1) / threads;
    populate_binary_tree_aabb_and_leaf_coefficients<<<grid_size, threads>>>(
        m_sorted_geometry.data().get(), m_leaf_coefficients.data().get(),
        leaf_count, m_binary_nodes.data().get(), m_binary_aabbs.data().get(),
        m_binary_parents.data().get(), m_atomic_counters.data().get(),
        m_sorted_geometry.size());
    CUDA_CHECK(cudaGetLastError());
  }
  // Convert binary LBVH tree into BVH8 tree
  {
    uint32_t max_bvh8_nodes = leaf_count * 0.2;
    thrust::device_vector<uint32_t> d_leaf_parents(leaf_count, thrust::no_init);
    thrust::device_vector<uint32_t> d_leaf_pointers(max_bvh8_nodes * 8,
                                                    thrust::no_init);
    thrust::device_vector<uint32_t> d_internal_parent_map(max_bvh8_nodes,
                                                          thrust::no_init);

    // Work queues (there are only leaf_count-1 internal binary nodes)
    thrust::device_vector<uint32_t> d_work_queue_A(leaf_count - 1,
                                                   thrust::no_init);
    thrust::device_vector<uint32_t> d_work_queue_B(leaf_count - 1,
                                                   thrust::no_init);
    uint32_t *q_in = thrust::raw_pointer_cast(d_work_queue_A.data());
    uint32_t *q_out = thrust::raw_pointer_cast(d_work_queue_B.data());

    // Global counter for the NEXT level's size
    thrust::device_vector<uint32_t> d_global_counter(1, thrust::no_init);

    // Add root to queue
    uint32_t root_binary_idx = 0;
    d_work_queue_A[0] = root_binary_idx;

    // Set root parent to 0xFFFFFFFF
    uint32_t root_parent = 0xFFFFFFFF;
    d_internal_parent_map[0] = root_parent;

    // Level Creation Loop
    uint32_t level_offset = 0;
    uint32_t current_level_width = 1; // start with only root
    uint32_t total_bvh8_nodes = 1;

    while (current_level_width > 0) {
      // Reset the counter for the children that will be discovered
      d_global_counter[0] = 0;

      uint32_t threads = 256;
      uint32_t blocks = (current_level_width + threads - 1) / threads;

      convert_binary_tree_to_bvh8<<<blocks, threads>>>(
          m_binary_nodes.data().get(), m_binary_aabbs.data().get(),
          m_bvh8_nodes.data().get(), d_leaf_parents.data().get(),
          d_leaf_pointers.data().get(), d_internal_parent_map.data().get(),
          q_in, current_level_width, q_out, d_global_counter.data().get(),
          leaf_count, level_offset);

      // Get the number of internal nodes found for the next level
      uint32_t next_level_width = d_global_counter[0];

      // Update offsets for the next kernel call
      level_offset = total_bvh8_nodes;
      total_bvh8_nodes += next_level_width;
      current_level_width = next_level_width;

      // Swap Work Queues
      std::swap(q_in, q_out);

      if (total_bvh8_nodes >= max_bvh8_nodes) {
        throw std::runtime_error(
            "This shouldn't happen. Maybe we need a floor to compute "
            "max_bvh8_nodes. Bug is more likely!");
      }
    }
  }

  // TODO:
  // 1. memory arena
  // 2. tests for everything!
  // 3. M2M for tailor coefficients
}
