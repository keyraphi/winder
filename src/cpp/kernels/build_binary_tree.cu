#include "aabb.h"
#include "binary_node.h"
#include "common.cuh"
#include "geometry.h"
#include "mat3x3.h"
#include "tailor_coefficients.h"
#include "tensor3.h"
#include "vec3.h"
#include <cmath>
#include <cstdint>
#include <cub/block/block_scan.cuh>
#include <cub/util_type.cuh>
#include <cuda/atomic>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_atomic_functions.h>
#include <driver_types.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <vector_functions.h>
#include <vector_types.h>

__global__ void __launch_bounds__(256)
    interleave_gather_geometry_kernel(const float *__restrict__ points,
                                      const float *__restrict__ normals,
                                      const uint32_t *__restrict__ indices,
                                      PointNormal *__restrict__ out_geometry,
                                      const uint32_t count) {
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  const uint32_t src_idx = indices[idx];
  const uint32_t src_offset = src_idx * 3;

  float p_x = points[src_offset];
  float p_y = points[src_offset + 1];
  float p_z = points[src_offset + 2];
  float n_x = normals[src_offset + 0];
  float n_y = normals[src_offset + 1];
  float n_z = normals[src_offset + 2];

  out_geometry[idx].p.x = p_x;
  out_geometry[idx].p.y = p_y;
  out_geometry[idx].p.z = p_z;
  out_geometry[idx].n.x = n_x;
  out_geometry[idx].n.y = n_y;
  out_geometry[idx].n.z = n_z;
}

void interleave_gather_geometry(const float *__restrict__ points,
                                const float *__restrict__ normals,
                                const uint32_t *__restrict__ indices,
                                PointNormal *__restrict__ out_geometry,
                                const uint32_t count,
                                const cudaStream_t &stream) {
  if (count < 1) {
    return;
  }
  // one thread per point
  const uint32_t threads = 256;
  const uint32_t blocks = (count + threads - 1) / threads;
  interleave_gather_geometry_kernel<<<blocks, threads, 0, stream>>>(
      points, normals, indices, out_geometry, count);
  CUDA_CHECK(cudaGetLastError());
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

void build_binary_topology(const uint32_t *__restrict__ morton_codes,
                           BinaryNode *nodes, uint32_t *parents,
                           const uint32_t leaf_count,
                           const cudaStream_t &stream) {

  if (leaf_count < 1) {
    return;
  }
  if (leaf_count > 1) {
    const uint32_t threads = 256;
    const uint32_t blocks = (leaf_count - 1 + threads - 1) / threads;
    build_binary_topology_kernel<<<blocks, threads, 0, stream>>>(
        morton_codes, nodes, parents, leaf_count);
    CUDA_CHECK(cudaGetLastError());
  }

  // the root node gets a unique value for its parent
  uint32_t root_parent = 0xFFFFFFFF;
  thrust::copy_n(&root_parent, 1, thrust::device_pointer_cast(parents));
}

__device__ __forceinline__ auto warp_reduce_add_down(float val) -> float {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  }
  return val;
}

template <typename Geometry>
__global__ void populate_binary_tree_aabb_and_leaf_coefficients_kernel(
    const Geometry *__restrict__ sorted_geometry,
    TailorCoefficientsBf16 *leaf_coefficients, const uint32_t leaf_count,
    const BinaryNode *binary_nodes, AABB *binary_aabbs,
    const uint32_t *binary_parents, uint32_t *atomic_counters,
    const uint32_t geometry_count) {
  uint32_t geometry_idx = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t leaf_idx = geometry_idx / 32;
  if (leaf_idx >= leaf_count)
    return;

  uint32_t lane_id = geometry_idx % 32;

  Geometry geometry =
      Geometry::load(sorted_geometry, geometry_idx, geometry_count);

  // Compute AABB
  bool is_thread_active = geometry_idx < geometry_count;
  AABB geometry_aabb;
  Vec3 center_of_mass;
  if (is_thread_active) {
    geometry_aabb = geometry.get_aabb();
    center_of_mass = geometry.centroid();
  } else {
    geometry_aabb.min = Vec3{1e38F, 1e38F, 1e38F};
    geometry_aabb.max = Vec3{-1e38F, -1e38F, -1e38F};
    center_of_mass = Vec3{0.F, 0.F, 0.F};
  }
  Vec3 &p_min = geometry_aabb.min;
  Vec3 &p_max = geometry_aabb.max;
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    p_min.x = fminf(p_min.x, __shfl_xor_sync(0xFFFFFFFF, p_min.x, offset));
    p_min.y = fminf(p_min.y, __shfl_xor_sync(0xFFFFFFFF, p_min.y, offset));
    p_min.z = fminf(p_min.z, __shfl_xor_sync(0xFFFFFFFF, p_min.z, offset));

    p_max.x = fmaxf(p_max.x, __shfl_xor_sync(0xFFFFFFFF, p_max.x, offset));
    p_max.y = fmaxf(p_max.y, __shfl_xor_sync(0xFFFFFFFF, p_max.y, offset));
    p_max.z = fmaxf(p_max.z, __shfl_xor_sync(0xFFFFFFFF, p_max.z, offset));

    center_of_mass.x += __shfl_xor_sync(0xFFFFFFFF, center_of_mass.x, offset);
    center_of_mass.y += __shfl_xor_sync(0xFFFFFFFF, center_of_mass.y, offset);
    center_of_mass.z += __shfl_xor_sync(0xFFFFFFFF, center_of_mass.z, offset);
  }
  // compute center of mass
  uint32_t geo_count_in_leaf = min(32U, geometry_count - (leaf_idx * 32U));
  float inv_geo_count_in_leaf = 1.F / (float)geo_count_in_leaf;
  center_of_mass *= inv_geo_count_in_leaf;
  // find max distance of center of mass to any element inside
  float dist_to_com = (geometry.centroid() - center_of_mass).length();
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    dist_to_com =
        fmaxf(dist_to_com, __shfl_xor_sync(0xFFFFFFFF, dist_to_com, offset));
  }
  // write aggregated AABB for leaf
  if (lane_id == 0) {
    binary_aabbs[leaf_idx + leaf_count - 1].min = p_min;
    binary_aabbs[leaf_idx + leaf_count - 1].max = p_max;
    binary_aabbs[leaf_idx + leaf_count - 1].setCenterOfMass(center_of_mass);
    binary_aabbs[leaf_idx + leaf_count - 1].setMaxDistanceToCenterOfMass(
        dist_to_com);
  }

  // Compute tailor coefficients
  // first and second order use center of mass

  // geometry dependent terms used to compute tailor coefficients
  Vec3 d, n;
  SymMat3x3 Ct;
  geometry.get_tailor_terms(center_of_mass, n, d, Ct);

  // Zero order
  //\sum_{i=1}^m a_i n_i
  // points: a_i*n_i is scaled normal
  // triangles: a_i*n_i is surface area * normal
  // for inactive threads neutral element wrt. +
  Vec3 zero_order = is_thread_active ? n : Vec3{0.F, 0.F, 0.F};
  zero_order.x = warp_reduce_add_down(zero_order.x);
  zero_order.y = warp_reduce_add_down(zero_order.y);
  zero_order.z = warp_reduce_add_down(zero_order.z);

  // write aggregated coefficient for leaf
  if (lane_id == 0) {
    leaf_coefficients[leaf_idx].zero_order = zero_order;
  }

  // First order
  // \sum_{i=1}^m a_i*d_i \otimes n_i
  // points: d_i = p_i-center
  // triangles: d_i = triangle_centroid_i - center
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
  // 1/2 (\sum_{i=1}^m Ct \otimes n)
  // points: Ct = a_i (p_i-center) \odot (p_i-center)
  // triangles: d_i = 1/3(1/2(x_i+xj-p')\odot(1/2(x_i+x_j)-p') +
  // 1/3(1/2(x_j+x_k)-p')\odot(1/2(x_j+x_k)-p') +
  // 1/3(1/2(x_k+x_i)-p')\odot(1/2(x_k+x_i)-p')
  // Note: Ct is symetric. It contains only 6 unique values
  // For that reason second_order also only contains 18 unique values. We don't
  // compute/store duplicates.
  Tensor3_compressed second_order;
  if (is_thread_active) {
    second_order.data[0] = Ct.data[0] * n.x;
    second_order.data[1] = Ct.data[0] * n.y;
    second_order.data[2] = Ct.data[0] * n.z;
    second_order.data[3] = Ct.data[1] * n.x;
    second_order.data[4] = Ct.data[1] * n.y;
    second_order.data[5] = Ct.data[1] * n.z;
    second_order.data[6] = Ct.data[2] * n.x;
    second_order.data[7] = Ct.data[2] * n.y;
    second_order.data[8] = Ct.data[2] * n.z;
    second_order.data[9] = Ct.data[3] * n.x;
    second_order.data[10] = Ct.data[3] * n.y;
    second_order.data[11] = Ct.data[3] * n.z;
    second_order.data[12] = Ct.data[4] * n.x;
    second_order.data[13] = Ct.data[4] * n.y;
    second_order.data[14] = Ct.data[4] * n.z;
    second_order.data[15] = Ct.data[5] * n.x;
    second_order.data[16] = Ct.data[5] * n.y;
    second_order.data[17] = Ct.data[5] * n.z;
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
    // If this leaf's parent is the root marker, we are done.
    if (current_parent_idx == 0xFFFFFFFF) {
      return;
    }

    // race to top
    // get number of elements in other child of current parent
    uint32_t geo_count_left_child =
        atomicAdd(&atomic_counters[current_parent_idx], geo_count_in_leaf);
    if (geo_count_left_child == 0) {
      return;
    }
    uint32_t geo_countr_right_child = geo_count_in_leaf;

    while (current_parent_idx != 0xFFFFFFFF) {
      BinaryNode parent_node = binary_nodes[current_parent_idx];

      if (parent_node.left_child == current_idx) {
        uint32_t tmp = geo_count_left_child;
        geo_count_left_child = geo_countr_right_child;
        geo_countr_right_child = tmp;
      }

      // merge child aabbs
      binary_aabbs[current_parent_idx] =
          AABB::merge(binary_aabbs[parent_node.left_child],
                      binary_aabbs[parent_node.right_child],
                      geo_count_left_child, geo_countr_right_child);
      // make sure that aabb has been written before the counter is incremented
      cuda::atomic_thread_fence(cuda::memory_order_seq_cst,
                                cuda::thread_scope_device);

      // Move up to the next level
      uint32_t next_parent = binary_parents[current_parent_idx];
      if (next_parent == 0xFFFFFFFF) {
        return;
      }
      geo_countr_right_child += geo_count_left_child; // new count for this node
      geo_count_left_child =
          atomicAdd(&atomic_counters[next_parent],
                    geo_countr_right_child); // count from other node
      if (geo_count_left_child == 0) {
        return;
      }
      current_parent_idx = next_parent;
    }
  }
}

template <typename Geometry>
void populate_binary_tree_aabb_and_leaf_coefficients(
    const Geometry *__restrict__ sorted_geometry,
    TailorCoefficientsBf16 *leaf_coefficients, const uint32_t leaf_count,
    const BinaryNode *binary_nodes, AABB *binary_aabbs,
    const uint32_t *binary_parents, uint32_t *atomic_counters,
    const uint32_t geometry_count, const cudaStream_t &stream) {
  if (geometry_count == 0) {
    return;
  }

  const uint32_t threads = 256;
  const uint32_t blocks = (leaf_count * 32 + threads - 1) / threads;
  populate_binary_tree_aabb_and_leaf_coefficients_kernel<Geometry>
      <<<blocks, threads, 0, stream>>>(
          sorted_geometry, leaf_coefficients, leaf_count, binary_nodes,
          binary_aabbs, binary_parents, atomic_counters, geometry_count);
  CUDA_CHECK(cudaGetLastError());
}

// Tell the compiler to generate the code for these types
template void populate_binary_tree_aabb_and_leaf_coefficients<PointNormal>(
    const PointNormal *__restrict__ sorted_geometry,
    TailorCoefficientsBf16 *leaf_coefficients, uint32_t leaf_count,
    const BinaryNode *binary_nodes, AABB *binary_aabbs,
    const uint32_t *binary_parents, uint32_t *atomic_counters,
    uint32_t geometry_count, const cudaStream_t &stream);

template void populate_binary_tree_aabb_and_leaf_coefficients<Triangle>(
    const Triangle *__restrict__ sorted_geometry,
    TailorCoefficientsBf16 *leaf_coefficients, uint32_t leaf_count,
    const BinaryNode *binary_nodes, AABB *binary_aabbs,
    const uint32_t *binary_parents, uint32_t *atomic_counters,
    uint32_t geometry_count, const cudaStream_t &stream);
