#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cub/util_type.cuh>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <memory>
#include <sys/types.h>
#include <thrust/copy.h>
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
#include "tensor3.h"
#include "utils.h"
#include "vec3.h"
#include "winder_cuda.h"

void CudaDeleter::operator()(void *ptr) const { cudaFree(ptr); }

auto WinderBackend::get_bvh_view() -> BVH8View {
  return BVH8View{thrust::raw_pointer_cast(m_nodes.data()),
                  thrust::raw_pointer_cast(m_sorted_geometry.data()),
                  (uint32_t)m_nodes.size()};
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

  // preallocate all memory used for tree construction
  m_sorted_geometry.resize(size * floats_per_elem);
  m_to_internal.resize(size);
  m_to_canonical.resize(size);
  m_nodes.resize(1.2 * leaf_count);
  m_leaf_coefficients.resize(leaf_count);
  m_leaf_aabb.resize(leaf_count);
  m_morton_codes.resize(size);
  m_leaf_morton_codes.resize(leaf_count);
  m_binary_nodes.resize(leaf_count - 1);
  m_binary_parents.resize(2 * leaf_count - 1);
  m_binary_aabb.resize(leaf_count - 1);
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

__global__ void compute_leaf_aabb_and_coefficients_kernel(
    const float *__restrict__ sorted_geometry, AABB *aabbs,
    const uint32_t leaf_count, const uint32_t point_count) {
  uint32_t point_idx = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t leaf_idx = point_idx / 32;
  if (leaf_idx >= leaf_count)
    return;

  // uint32_t lane_id = point_idx % 32;

  Vec3 p, n;
  if (point_idx < point_count) {
    const auto *base =
        reinterpret_cast<const float2 *>(sorted_geometry + point_idx * 6);

    float2 chunk0 = base[0]; // px, py
    float2 chunk1 = base[1]; // pz, nx
    float2 chunk2 = base[2]; // ny, nz

    p = {chunk0.x, chunk0.y, chunk1.x};
    n = {chunk1.y, chunk2.x, chunk2.y};
  } else {
    p = {1e38F, 1e38F, 1e38F};
    n = {0.F, 0.F, 0.F};
  }

  // Compute AABB
  Vec3 p_min = p;
  Vec3 p_max = point_idx < point_count ? p : Vec3{1e-38F, 1e-38F, 1e-38F};

#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    p_min.x = fminf(p_min.x, __shfl_down_sync(0xFFFFFFFF, p_min.x, offset));
    p_min.y = fminf(p_min.y, __shfl_down_sync(0xFFFFFFFF, p_min.y, offset));
    p_min.z = fminf(p_min.z, __shfl_down_sync(0xFFFFFFFF, p_min.z, offset));

    p_max.x = fmaxf(p_max.x, __shfl_down_sync(0xFFFFFFFF, p_max.x, offset));
    p_max.y = fmaxf(p_max.y, __shfl_down_sync(0xFFFFFFFF, p_max.y, offset));
    p_max.z = fmaxf(p_max.z, __shfl_down_sync(0xFFFFFFFF, p_max.z, offset));
  }

  // Compute tailor coefficients
  // w(q) \approx (\sum_{i=1}^m a_i n_i ) \cdot \nabla G(q, p_center) +
  // (\sum_{i=1}^m a_i(p_i-p_center)\otimes n_i) \cdot \nabla^2 G(q, p_center) +
  // 1/2 (\sum_{i=1}^m a_i(p_i - p_center)\otimes (p_i-p_center) \otimes n)
  // \cdot \nabla^3 G(q, p_center)
  Vec3 center = (p_min + p_max) * 0.5;

  // Zero order
  //\sum_{i=1}^m a_i n_i
  Vec3 zero_order = n;

#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    zero_order.x += __shfl_down_sync(0xFFFFFFFF, zero_order.x, offset);
    zero_order.y += __shfl_down_sync(0xFFFFFFFF, zero_order.y, offset);
    zero_order.z += __shfl_down_sync(0xFFFFFFFF, zero_order.z, offset);
  }

  // First order
  // \sum_{i=1}^m a_i(p_i-p_center)\otimes n_i
  Vec3 d = p - center;
  Mat3x3 first_order = d.outer_product(n);

#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < 9; ++i) {
      first_order.data[i] +=
          __shfl_down_sync(0xFFFFFFFF, first_order.data[i], offset);
    }
  }

  // second order
  // TODO. There should be 18 unique coefficients
  // 1/2 (\sum_{i=1}^m a_i(p_i - p_center)\otimes (p_i-p_center) \otimes n)
  // t[0] = d.x*d.x*n.x
  // t[1] = d.x*d.x*n.y
  // t[2] = d.x*d.x*n.z
  // t[3] = d.x*d.y*n.x
  // t[4] = d.x*d.y*n.y
  // t[5] = d.x*d.y*n.z
  // t[6] = d.x*d.z*n.x
  // t[7] = d.x*d.z*n.y
  // t[8] = d.x*d.z*n.z
  //
  // t[-] = d.y*d.x*n.x = t[3]
  // t[-] = d.y*d.x*n.y = t[4]
  // t[-] = d.y*d.x*n.z = t[5]
  // t[9] = d.y*d.y*n.x
  // t[10] = d.y*d.y*n.y
  // t[11] = d.y*d.y*n.z
  // t[12] = d.y*d.z*n.x
  // t[13] = d.y*d.z*n.y
  // t[14] = d.y*d.z*n.z
  //
  // t[-] = d.z*d.x*n.x = t[6]
  // t[-] = d.z*d.x*n.y = t[7]
  // t[-] = d.z*d.x*n.z = t[8]
  // t[-] = d.z*d.y*n.x = t[12]
  // t[-] = d.z*d.y*n.y = t[13]
  // t[-] = d.z*d.y*n.z = t[14]
  // t[18] = d.z*d.z*n.x
  // t[19] = d.z*d.z*n.y
  // t[20] = d.z*d.z*n.z
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
    check_launch_error("interleave_geometry");
  }

  uint32_t leaf_count = (m_morton_codes.size() + LEAF_SIZE + 1) / LEAF_SIZE;
  {
    auto morton_leaf_stride_idx = thrust::make_transform_iterator(
        thrust::make_counting_iterator<uint32_t>(0),
        [] __host__ __device__(uint32_t i) -> uint32_t {
          return i * LEAF_SIZE;
        });
    // thrust::make_strided_iterator<LEAF_SIZE>(m_morton_codes.begin());
    auto morton_leaf_stride = thrust::make_permutation_iterator(m_morton_codes.begin(), morton_leaf_stride_idx);
    thrust::copy(morton_leaf_stride, morton_leaf_stride + leaf_count,
                 m_leaf_morton_codes.begin());
    const size_t threads = 256;
    const size_t block_size = (leaf_count - 1 + threads - 1) / threads;
    build_binary_topology_kernel<<<block_size, threads>>>(
        m_leaf_morton_codes.data().get(), m_binary_nodes.data().get(),
        m_binary_parents.data().get(), leaf_count);
    check_launch_error("build_binary_topology_kernel");
  }
}
