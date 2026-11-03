#include "aabb.h"
#include "binary_node.h"
#include "common.cuh"
#include "cuda/__atomic/atomic.h"
#include "cuda/std/__atomic/order.h"
#include "cuda/std/__atomic/scopes.h"
#include "geometry.h"
#include "mat3x3.h"
#include "scene_normalization.h"
#include "soa.h"
#include "taylor_coefficients.h"
#include "tensor3.h"
#include "vec3.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cub/block/block_scan.cuh>
#include <cub/util_type.cuh>
#include <cuda/atomic>
#include <cuda_device_runtime_api.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_atomic_functions.h>
#include <driver_types.h>
#include <vector_functions.h>
#include <vector_types.h>

__global__ void __launch_bounds__(256)
    gather_queries_and_grad_outputs_soa_kernel(
        const float *__restrict__ queries_aos,
        const float *__restrict__ grad_outputs_aos,
        const uint32_t *__restrict__ indices,
        float *__restrict__ out_queries_soa,
        float *__restrict__ sorted_grad_outputs, const uint32_t count,
        const SceneNormalization norm) {
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  const uint32_t src_idx = indices[idx];
  const float *query_src_ptr = queries_aos + static_cast<size_t>(src_idx) * 3;

  const Vec3 query =
      norm.to_normalized(*reinterpret_cast<const Vec3 *>(query_src_ptr));

  out_queries_soa[0 * count + idx] = query.x;
  out_queries_soa[1 * count + idx] = query.y;
  out_queries_soa[2 * count + idx] = query.z;

  sorted_grad_outputs[idx] = grad_outputs_aos[src_idx];
}

void gather_queries_and_grad_outputs_soa(
    const float *__restrict__ queries, const float *__restrict__ grad_outputs,
    const uint32_t *__restrict__ indices,
    float *__restrict__ sorted_queries_soa,
    float *__restrict__ sorted_grad_outputs, const uint32_t count,
    const cudaStream_t &stream, const SceneNormalization &norm) {
  if (count < 1) {
    return;
  }
  // one thread per query
  const uint32_t threads = 256;
  const uint32_t blocks = (count + threads - 1) / threads;
  gather_queries_and_grad_outputs_soa_kernel<<<blocks, threads, 0, stream>>>(
      queries, grad_outputs, indices, sorted_queries_soa, sorted_grad_outputs,
      count, norm);
  CUDA_CHECK(cudaGetLastError());
}

__global__ void __launch_bounds__(256)
    gather_point_normals_soa_kernel(const float *__restrict__ points_aos,
                                    const float *__restrict__ normals_aos,
                                    const uint32_t *__restrict__ indices,
                                    float *__restrict__ out_point_normals_soa,
                                    const uint32_t count,
                                    const SceneNormalization norm) {
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  const uint32_t src_idx = indices[idx];
  const float *point_src_ptr = points_aos + static_cast<size_t>(src_idx) * 3;
  const float *normal_src_ptr = normals_aos + static_cast<size_t>(src_idx) * 3;

  // batch reads together
  PointNormal pn;
  auto *pn_ptr = reinterpret_cast<float *>(&pn);
#pragma unroll
  for (uint32_t i = 0; i < 3; ++i) {
    pn_ptr[i] = point_src_ptr[i];
  }
#pragma unroll
  for (uint32_t i = 0; i < 3; ++i) {
    pn_ptr[i + 3] = normal_src_ptr[i];
  }
  // normalize
  pn = norm.to_normalized(pn);

  // write into the SoA coalesced
  const auto *values = reinterpret_cast<const float *>(&pn);
#pragma unroll
  for (uint32_t i = 0; i < 6; ++i) {
    out_point_normals_soa[i * count + idx] = values[i];
  }
}

void gather_point_normals_soa(const float *__restrict__ points,
                              const float *__restrict__ normals,
                              const uint32_t *__restrict__ indices,
                              float *__restrict__ out_geometry,
                              const uint32_t count,
                              const SceneNormalization &norm,
                              const cudaStream_t &stream) {
  if (count < 1) {
    return;
  }
  // one thread per point
  const uint32_t threads = 256;
  const uint32_t blocks = (count + threads - 1) / threads;
  gather_point_normals_soa_kernel<<<blocks, threads, 0, stream>>>(
      points, normals, indices, out_geometry, count, norm);
  CUDA_CHECK(cudaGetLastError());
}

__global__ void __launch_bounds__(256) gather_triangles_aos_to_soa_kernel(
    const float *__restrict__ input_triangles_aos,
    const uint32_t *__restrict__ src_idxs,
    float *__restrict__ output_triangle_soa, uint32_t count,
    const SceneNormalization norm) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }
  uint32_t src_idx = src_idxs[idx];

  const float *src_ptr = input_triangles_aos + (src_idx * 9);
  const auto *t = reinterpret_cast<const Triangle *>(src_ptr);
  Triangle triangle = norm.to_normalized(*t);

  // write into the SoA coalesced
  const auto *value = reinterpret_cast<const float *>(&triangle);
#pragma unroll
  for (uint32_t i = 0; i < 9; ++i) {
    output_triangle_soa[i * count + idx] = value[i];
  }
}

void gather_triangles_soa(const float *__restrict__ input_triangles,
                          const uint32_t *__restrict__ to_internal_map,
                          float *__restrict__ output_triangles_soa,
                          uint32_t count, const SceneNormalization &norm,
                          const cudaStream_t &stream) {
  if (count < 1) {
    return;
  }

  uint32_t threads = 256;
  uint32_t blocks = (count + threads - 1) / threads;

  gather_triangles_aos_to_soa_kernel<<<blocks, threads, 0, stream>>>(
      input_triangles, to_internal_map, output_triangles_soa, count, norm);

  CUDA_CHECK(cudaGetLastError());
}

// Longest Common Prefix for 63-bit code[i] and code[j] with 32-bit index
// tiebreaker
__device__ inline auto delta_with_tiebreaker(int i, int j,
                                             const uint64_t *__restrict__ codes,
                                             int codes_len) -> int {
  if (j < 0 || j >= codes_len) {
    return -1;
  }

  uint64_t x_high = codes[i];
  uint64_t y_high = codes[j];

  // If the spatial Morton codes differ, they dictate the prefix length
  if (x_high != y_high) {
    return __clzll(x_high ^ y_high);
  }

  // Tie-breaker: If the codes match, append the 32-bit index to the key.
  // Left-shift by 32 bits so the index's MSB sits at the top of the lower word.
  uint64_t x_low = static_cast<uint64_t>(i) << 32;
  uint64_t y_low = static_cast<uint64_t>(j) << 32;

  // 64 bits from the identical high words + the matching prefix of the indices
  return 64 + __clzll(x_low ^ y_low);
}

__global__ void build_binary_topology_kernel(
    const uint64_t
        *__restrict__ morton_codes, // morton code of first leaf entry
    BinaryNode *nodes, uint32_t *parents, const uint32_t leaf_count) {
  uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_idx >= leaf_count - 1) {
    return;
  }

  // Compute range direction (+1 or -1)
  int direction = (delta_with_tiebreaker(thread_idx, thread_idx + 1,
                                         morton_codes, leaf_count) -
                       delta_with_tiebreaker(thread_idx, thread_idx - 1,
                                             morton_codes, leaf_count) >=
                   0)
                      ? 1
                      : -1;

  // find bit position where prefix change happens (full node)
  int parent_prefix = delta_with_tiebreaker(thread_idx, thread_idx - direction,
                                            morton_codes, leaf_count);
  int node_end_bit_mask = 2;
  while (delta_with_tiebreaker(thread_idx,
                               thread_idx + node_end_bit_mask * direction,
                               morton_codes, leaf_count) > parent_prefix) {
    node_end_bit_mask *= 2;
  }

  // find the index in the morton code array where this bitflip happens
  int total_idx_range_length = 0;
  for (int search_step = node_end_bit_mask / 2; search_step >= 1;
       search_step /= 2) {
    if (delta_with_tiebreaker(
            thread_idx,
            thread_idx + (total_idx_range_length + search_step) * direction,
            morton_codes, leaf_count) > parent_prefix) {
      total_idx_range_length += search_step;
    }
  }
  int range_end = (int)thread_idx + total_idx_range_length * direction;

  // Find the split point. Index where bit flip happens inside the node.
  int node_prefix =
      delta_with_tiebreaker(thread_idx, range_end, morton_codes, leaf_count);
  int split_offset = 0;

  // start with largest power of 2 smaller than total_idx_range_length
  int search_step = 1 << (31 - __clz(total_idx_range_length));
  for (; search_step >= 1; search_step >>= 1) {
    int candidate_offset = split_offset + search_step;
    if (candidate_offset < total_idx_range_length) {
      if (delta_with_tiebreaker(thread_idx,
                                thread_idx + candidate_offset * direction,
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

void build_binary_topology(const uint64_t *__restrict__ morton_codes,
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
  CUDA_CHECK(cudaMemsetAsync(parents, 0xFF, sizeof(uint32_t), stream));
}

__device__ __forceinline__ auto warp_reduce_add_down(float val) -> float {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  }
  return val;
}

template <IsGeometry Geometry>
__global__ void populate_binary_tree_aabb_and_leaf_coefficients_kernel(
    const SoAView<Geometry> sorted_geometry,
    TaylorCoefficientsF16 *leaf_coefficients, const uint32_t leaf_count,
    const BinaryNode *binary_nodes, AABB *binary_aabbs,
    const uint32_t *binary_parents, float *atomic_weights,
    const uint32_t geometry_count) {
  uint32_t geometry_idx = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t leaf_idx = geometry_idx / 32;
  if (leaf_idx >= leaf_count)
    return;

  uint32_t lane_id = geometry_idx % 32;

  Geometry geometry =
      Geometry::load(sorted_geometry, geometry_idx, geometry_count);

  // Compute AABB for leaf
  bool is_thread_active = geometry_idx < geometry_count;
  AABB geometry_aabb;
  Vec3 center_of_mass;
  float weight = 0.F;
  if (is_thread_active) {
    geometry_aabb = geometry.get_aabb();
    center_of_mass = geometry.centroid();
    weight = geometry.get_weight();
  } else {
    geometry_aabb.min = Vec3_f16{65504.F, 65504.F, 65504.F};
    geometry_aabb.max = Vec3_f16{-65504.F, -65504.F, -65504.F};
    center_of_mass = Vec3{0.F, 0.F, 0.F};
    weight = 0.F;
  }
  Vec3_f16 &p_min = geometry_aabb.min;
  Vec3_f16 &p_max = geometry_aabb.max;
  Vec3 weighted_com = center_of_mass * weight;

#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    p_min.x = __hmin(p_min.x, __shfl_xor_sync(0xFFFFFFFF, p_min.x, offset));
    p_min.y = __hmin(p_min.y, __shfl_xor_sync(0xFFFFFFFF, p_min.y, offset));
    p_min.z = __hmin(p_min.z, __shfl_xor_sync(0xFFFFFFFF, p_min.z, offset));

    p_max.x = __hmax(p_max.x, __shfl_xor_sync(0xFFFFFFFF, p_max.x, offset));
    p_max.y = __hmax(p_max.y, __shfl_xor_sync(0xFFFFFFFF, p_max.y, offset));
    p_max.z = __hmax(p_max.z, __shfl_xor_sync(0xFFFFFFFF, p_max.z, offset));

    weighted_com.x += __shfl_xor_sync(0xFFFFFFFF, weighted_com.x, offset);
    weighted_com.y += __shfl_xor_sync(0xFFFFFFFF, weighted_com.y, offset);
    weighted_com.z += __shfl_xor_sync(0xFFFFFFFF, weighted_com.z, offset);

    weight += __shfl_xor_sync(0xFFFFFFFF, weight, offset);
  }
  // compute center of mass for leaf
  center_of_mass = weight > 0.F ? (weighted_com / weight) : Vec3{0.F, 0.F, 0.F};
  // find max distance of center of mass to any element inside
  float dist_to_com =
      is_thread_active ? geometry.max_distance_to(center_of_mass) : 0.F;
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    dist_to_com =
        fmaxf(dist_to_com, __shfl_xor_sync(0xFFFFFFFF, dist_to_com, offset));
  }

  uint32_t leaf_node_idx = leaf_idx + leaf_count - 1;

  // write aggregated AABB for leaf (only lane 0)
  if (lane_id == 0) {
    binary_aabbs[leaf_idx + leaf_count - 1].min = p_min;
    binary_aabbs[leaf_idx + leaf_count - 1].max = p_max;

    binary_aabbs[leaf_idx + leaf_count - 1].center_of_mass = center_of_mass;
    binary_aabbs[leaf_idx + leaf_count - 1].max_distance =
        __float2half(dist_to_com);
  }

  // Compute tailor coefficients
  Vec3 zero_order;
  Mat3x3 first_order;
  Tensor3_compressed second_order;

  // Zero order
  //\sum_{i=1}^m a_i n_i
  // points: a_i*n_i is scaled normal
  // triangles: a_i*n_i is surface area * normal
  // First order
  // \sum_{i=1}^m a_i*d_i \otimes n_i
  // points: d_i = p_i-center
  // triangles: d_i = triangle_centroid_i - center
  // second order
  // 1/2 (\sum_{i=1}^m Ct \otimes n)
  // points: Ct = a_i (p_i-center) \odot (p_i-center)
  // triangles: d_i = 1/3(1/2(x_i+xj-p')\odot(1/2(x_i+x_j)-p') +
  // 1/3(1/2(x_j+x_k)-p')\odot(1/2(x_j+x_k)-p') +
  // 1/3(1/2(x_k+x_i)-p')\odot(1/2(x_k+x_i)-p')
  // Note: Ct is symmetric. It contains only 6 unique values
  // For that reason second_order also only contains 18 unique values. We don't
  // compute/store duplicates.
  // For inactive threads result is 0 (neutral wrt +)
  geometry.get_taylor_terms(center_of_mass, is_thread_active, zero_order,
                            first_order, second_order);

  // aggregate zero order
  zero_order.x = warp_reduce_add_down(zero_order.x);
  zero_order.y = warp_reduce_add_down(zero_order.y);
  zero_order.z = warp_reduce_add_down(zero_order.z);
  // write aggregated coefficient for leaf
  if (lane_id == 0) {
    leaf_coefficients[leaf_idx].zero_order = zero_order;
  }

// aggregate first order
#pragma unroll
  for (int i = 0; i < 9; ++i) {
    first_order.data[i] = warp_reduce_add_down(first_order.data[i]);
  }
  // write aggregated coefficient for leaf
  if (lane_id == 0) {
    leaf_coefficients[leaf_idx].first_order = first_order;
  }
// aggregate second order
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
    // prevent stall for degenerate leafs
    float my_weight = (weight == 0.F) ? 1e-10 : weight;

    uint32_t current_idx = leaf_node_idx;
    uint32_t current_parent_idx = binary_parents[current_idx];
    // If this leaf's parent is the root marker, we are done.
    if (current_parent_idx == 0xFFFFFFFF) {
      return;
    }

    // race to top
    cuda::atomic_thread_fence(cuda::memory_order_seq_cst,
                              cuda::thread_scope_device);
    // get number of elements in other child of current parent
    float other_weight =
        atomicAdd(&atomic_weights[current_parent_idx], my_weight);
    if (other_weight == 0.F) {
      return; // this subtree arrived first and is done
    }
    // The thread that arrived here has both weights in registers
    while (current_parent_idx != 0xFFFFFFFF) {
      BinaryNode parent_node = binary_nodes[current_parent_idx];

      float weight_left;
      float weight_right;
      // find out if my leaf is left or right
      if (parent_node.left_child == current_idx) {
        // I am the left child
        weight_left = my_weight;
        weight_right = other_weight;
      } else {
        // I am the right child
        weight_left = other_weight;
        weight_right = my_weight;
      }

      // merge child aabbs
      binary_aabbs[current_parent_idx] = AABB::merge_weighted(
          binary_aabbs[parent_node.left_child],
          binary_aabbs[parent_node.right_child], weight_left, weight_right);
      // make sure that aabb has been written before the counter is incremented
      __threadfence();

      // Move up to the next level
      uint32_t next_parent = binary_parents[current_parent_idx];
      if (next_parent == 0xFFFFFFFF) {
        break;
      }
      // the combined weight becomes the new weight for this threads future race
      my_weight = weight_left + weight_right;
      other_weight = atomicAdd(&atomic_weights[next_parent], my_weight);
      if (other_weight == 0.F) {
        break; // Thread arrived first at next level, let other thread finish it
      }
      current_idx = current_parent_idx;
      current_parent_idx = next_parent;
    }
  }
}

template <IsGeometry Geometry>
void populate_binary_tree_aabb_and_leaf_coefficients(
    const float *__restrict__ sorted_geometry,
    TaylorCoefficientsF16 *leaf_coefficients, const uint32_t leaf_count,
    const BinaryNode *binary_nodes, AABB *binary_aabbs,
    const uint32_t *binary_parents, float *atomic_counters,
    const uint32_t geometry_count, const cudaStream_t &stream) {
  if (geometry_count == 0) {
    return;
  }

  const uint32_t threads = 256;
  const uint32_t blocks = (leaf_count * 32 + threads - 1) / threads;
  populate_binary_tree_aabb_and_leaf_coefficients_kernel<Geometry>
      <<<blocks, threads, 0, stream>>>(
          SoAView<Geometry>{sorted_geometry, geometry_count}, leaf_coefficients,
          leaf_count, binary_nodes, binary_aabbs, binary_parents,
          atomic_counters, geometry_count);
  CUDA_CHECK(cudaGetLastError());
}

__global__ void populate_binary_tree_aabb_and_leaf_coefficients_backward_kernel(
    const SoAView<Vec3> sorted_queries,
    const float *__restrict__ sorted_grad_outputs,
    BackwardTaylorCoefficients *leaf_coefficients, const uint32_t leaf_count,
    const BinaryNode *binary_nodes, AABB *binary_aabbs,
    const uint32_t *binary_parents, float *atomic_weights,
    const uint32_t query_count) {
  uint32_t query_idx = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t leaf_idx = query_idx / 32;
  if (leaf_idx >= leaf_count) {
    return;
  }

  uint32_t lane_id = query_idx % 32;

  Vec3 query = Vec3::load(sorted_queries, query_idx, query_count);

  // Compute AABB for leaf
  bool is_thread_active = query_idx < query_count;
  AABB query_aabb;
  Vec3 center_of_mass;
  float weight = 0.F;
  if (is_thread_active) {
    query_aabb = query.get_aabb();
    center_of_mass = query.centroid();
    weight = 1.F;
  } else {
    query_aabb.min = Vec3_f16{65504.F, 65504.F, 65504.F};
    query_aabb.max = Vec3_f16{65504.F, 65504.F, 65504.F};
    center_of_mass = Vec3{0.F, 0.F, 0.F};
    weight = 0.F;
  }
  Vec3_f16 &p_min = query_aabb.min;
  Vec3_f16 &p_max = query_aabb.max;
  Vec3 weighted_com = center_of_mass * weight;

#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    p_min.x = __hmin(p_min.x, __shfl_xor_sync(0xFFFFFFFF, p_min.x, offset));
    p_min.y = __hmin(p_min.y, __shfl_xor_sync(0xFFFFFFFF, p_min.y, offset));
    p_min.z = __hmin(p_min.z, __shfl_xor_sync(0xFFFFFFFF, p_min.z, offset));

    p_max.x = __hmax(p_max.x, __shfl_xor_sync(0xFFFFFFFF, p_max.x, offset));
    p_max.y = __hmax(p_max.y, __shfl_xor_sync(0xFFFFFFFF, p_max.y, offset));
    p_max.z = __hmax(p_max.z, __shfl_xor_sync(0xFFFFFFFF, p_max.z, offset));

    weighted_com.x += __shfl_xor_sync(0xFFFFFFFF, weighted_com.x, offset);
    weighted_com.y += __shfl_xor_sync(0xFFFFFFFF, weighted_com.y, offset);
    weighted_com.z += __shfl_xor_sync(0xFFFFFFFF, weighted_com.z, offset);

    weight += __shfl_xor_sync(0xFFFFFFFF, weight, offset);
  }
  // compute center of mass for leaf
  center_of_mass = weight > 0.F ? (weighted_com / weight) : Vec3{0.F, 0.F, 0.F};
  // find max distance of center of mass to any element inside
  float dist_to_com =
      is_thread_active ? query.max_distance_to(center_of_mass) : 0.F;
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    dist_to_com =
        fmaxf(dist_to_com, __shfl_xor_sync(0xFFFFFFFF, dist_to_com, offset));
  }

  uint32_t leaf_node_idx = leaf_idx + leaf_count - 1;

  // write aggregated AABB for leaf (only lane 0)
  if (lane_id == 0) {
    binary_aabbs[leaf_idx + leaf_count - 1].min = p_min;
    binary_aabbs[leaf_idx + leaf_count - 1].max = p_max;

    binary_aabbs[leaf_idx + leaf_count - 1].center_of_mass = center_of_mass;
    binary_aabbs[leaf_idx + leaf_count - 1].max_distance =
        __float2half(dist_to_com);
  }

  // Compute backward tailor coefficients
  float zero_order = 0.F;
  Vec3 first_order = Vec3::zero();
  SymMat3x3 second_order = SymMat3x3::zero();

  // Zero order
  //\sum_{i=1}^m g_i
  //
  // First order
  // \sum_{i=1}^m g_i*(x_i - center)
  // second order
  // \sum_{i=1}^m g_i (x_i - center)\otimes(x_i - center)
  // For inactive threads result is 0 (neutral wrt +)
  if (query_idx < query_count) {
    zero_order = sorted_grad_outputs[query_idx];
    first_order = zero_order * (query - center_of_mass);
    second_order = zero_order * (query - center_of_mass).tensor_square();
  }

  // aggregate zero order
  zero_order = warp_reduce_add_down(zero_order);
  // write aggregated coefficient for leaf
  if (lane_id == 0) {
    leaf_coefficients[leaf_idx].zero_order = zero_order;
  }
  // aggregate first order
  first_order.x = warp_reduce_add_down(first_order.x);
  first_order.y = warp_reduce_add_down(first_order.y);
  first_order.z = warp_reduce_add_down(first_order.z);
  // write aggregated coefficient for leaf
  if (lane_id == 0) {
    leaf_coefficients[leaf_idx].first_order = first_order;
  }
// aggregate second order
#pragma unroll
  for (int i = 0; i < 9; ++i) {
    second_order.data[i] = warp_reduce_add_down(second_order.data[i]);
  }
  // write aggregated coefficient for leaf
  if (lane_id == 0) {
    leaf_coefficients[leaf_idx].second_order = second_order;
  }

  // propagate the aabbs to the inner nodes (binary_aabbs)
  // lane 0 handles the Leaf-to-Root Race
  if (lane_id == 0) {
    // prevent stall for degenerate leafs
    float my_weight = (weight == 0.F) ? 1e-10 : weight;

    uint32_t current_idx = leaf_node_idx;
    uint32_t current_parent_idx = binary_parents[current_idx];
    // If this leaf's parent is the root marker, we are done.
    if (current_parent_idx == 0xFFFFFFFF) {
      return;
    }

    // race to top
    cuda::atomic_thread_fence(cuda::memory_order_seq_cst,
                              cuda::thread_scope_device);
    // get number of elements in other child of current parent
    float other_weight =
        atomicAdd(&atomic_weights[current_parent_idx], my_weight);
    if (other_weight == 0.F) {
      return; // this subtree arrived first and is done
    }
    // The thread that arrived here has both weights in registers
    while (current_parent_idx != 0xFFFFFFFF) {
      BinaryNode parent_node = binary_nodes[current_parent_idx];

      float weight_left;
      float weight_right;
      // find out if my leaf is left or right
      if (parent_node.left_child == current_idx) {
        // I am the left child
        weight_left = my_weight;
        weight_right = other_weight;
      } else {
        // I am the right child
        weight_left = other_weight;
        weight_right = my_weight;
      }

      // merge child aabbs
      binary_aabbs[current_parent_idx] = AABB::merge_weighted(
          binary_aabbs[parent_node.left_child],
          binary_aabbs[parent_node.right_child], weight_left, weight_right);
      // make sure that aabb has been written before the counter is incremented
      __threadfence();

      // Move up to the next level
      uint32_t next_parent = binary_parents[current_parent_idx];
      if (next_parent == 0xFFFFFFFF) {
        break;
      }
      // the combined weight becomes the new weight for this threads future race
      my_weight = weight_left + weight_right;
      other_weight = atomicAdd(&atomic_weights[next_parent], my_weight);
      if (other_weight == 0.F) {
        break; // Thread arrived first at next level, let other thread finish it
      }
      current_idx = current_parent_idx;
      current_parent_idx = next_parent;
    }
  }
}

void populate_binary_tree_aabb_and_leaf_coefficients_backward(
    const float *__restrict__ sorted_queries,
    const float *__restrict__ sorted_grad_outputs,
    BackwardTaylorCoefficients *leaf_coefficients, const uint32_t leaf_count,
    const BinaryNode *binary_nodes, AABB *binary_aabbs,
    const uint32_t *binary_parents, float *atomic_counters,
    const uint32_t query_count, const cudaStream_t &stream) {
  if (query_count == 0) {
    return;
  }

  const uint32_t threads = 256;
  const uint32_t blocks = (leaf_count * 32 + threads - 1) / threads;
  populate_binary_tree_aabb_and_leaf_coefficients_backward_kernel<<<
      blocks, threads, 0, stream>>>(
      SoAView<Vec3>{sorted_queries, query_count}, sorted_grad_outputs,
      leaf_coefficients, leaf_count, binary_nodes, binary_aabbs, binary_parents,
      atomic_counters, query_count);
  CUDA_CHECK(cudaGetLastError());
}


template <IsGeometry Geometry, uint32_t D>
__global__ void populate_binary_tree_aabb_and_leaf_coefficients_payload_kernel(
    const SoAView<Geometry> sorted_geometry,
    const float *__restrict__ sorted_payload,
    PayloadTaylorCoefficientsF16<D> *leaf_coefficients,
    const uint32_t leaf_count, const BinaryNode *binary_nodes,
    AABB *binary_aabbs, const uint32_t *binary_parents, float *atomic_weights,
    const uint32_t geometry_count) {

  static_assert(D >= 1, "Payload dimensionality D must be >= 1");

  uint32_t geometry_idx = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t leaf_idx = geometry_idx / 32;
  if (leaf_idx >= leaf_count) {
    return;
  }
  uint32_t lane_id = geometry_idx % 32;

  Geometry geometry =
      Geometry::load(sorted_geometry, geometry_idx, geometry_count);
  bool is_thread_active = geometry_idx < geometry_count;

  AABB geometry_aabb;
  Vec3 center_of_mass;
  float weight = 0.F;
  if (is_thread_active) {
    geometry_aabb = geometry.get_aabb();
    center_of_mass = geometry.centroid();
    weight = geometry.get_weight();
  } else {
    geometry_aabb.min = Vec3_f16{65504.F, 65504.F, 65504.F};
    geometry_aabb.max = Vec3_f16{-65504.F, -65504.F, -65504.F};
    center_of_mass = Vec3{0.F, 0.F, 0.F};
    weight = 0.F;
  }
  Vec3_f16 &p_min = geometry_aabb.min;
  Vec3_f16 &p_max = geometry_aabb.max;
  Vec3 weighted_com = center_of_mass * weight;

#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    p_min.x = __hmin(p_min.x, __shfl_xor_sync(0xFFFFFFFF, p_min.x, offset));
    p_min.y = __hmin(p_min.y, __shfl_xor_sync(0xFFFFFFFF, p_min.y, offset));
    p_min.z = __hmin(p_min.z, __shfl_xor_sync(0xFFFFFFFF, p_min.z, offset));
    p_max.x = __hmax(p_max.x, __shfl_xor_sync(0xFFFFFFFF, p_max.x, offset));
    p_max.y = __hmax(p_max.y, __shfl_xor_sync(0xFFFFFFFF, p_max.y, offset));
    p_max.z = __hmax(p_max.z, __shfl_xor_sync(0xFFFFFFFF, p_max.z, offset));
    weighted_com.x += __shfl_xor_sync(0xFFFFFFFF, weighted_com.x, offset);
    weighted_com.y += __shfl_xor_sync(0xFFFFFFFF, weighted_com.y, offset);
    weighted_com.z += __shfl_xor_sync(0xFFFFFFFF, weighted_com.z, offset);
    weight += __shfl_xor_sync(0xFFFFFFFF, weight, offset);
  }
  center_of_mass = weight > 0.F ? (weighted_com / weight) : Vec3{0.F, 0.F, 0.F};

  float dist_to_com =
      is_thread_active ? geometry.max_distance_to(center_of_mass) : 0.F;
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    dist_to_com =
        fmaxf(dist_to_com, __shfl_xor_sync(0xFFFFFFFF, dist_to_com, offset));
  }

  const uint32_t leaf_node_idx = leaf_idx + leaf_count - 1;
  if (lane_id == 0) {
    binary_aabbs[leaf_node_idx].min = p_min;
    binary_aabbs[leaf_node_idx].max = p_max;
    binary_aabbs[leaf_node_idx].center_of_mass = center_of_mass;
    binary_aabbs[leaf_node_idx].max_distance = __float2half(dist_to_com);
  }

  // For each of the 30 slots and each D-chunk:
  //   acc[d] = v_i[d] * c_k_i                 (thread-local)
  //   acc[d] = warp_reduce_add_down(acc[d])   (32-lane sum, lane 0 gets total)
  //   lane 0 writes to leaf_coefficients[k][d0+d]
  //
  // The reduction uses __shfl_down_sync only (matching the scalar
  // warp_reduce_add_down pattern, which sums into lane 0). Slots 0..11 are
  // stored as half bits, slots 12..29 as bf16 bits.

  // Scalar Taylor terms of THIS thread's primitive (0 for inactive lanes).
  Vec3 zero_order;
  Mat3x3 first_order;
  Tensor3_compressed second_order;
  geometry.get_taylor_terms(center_of_mass, is_thread_active, zero_order,
                            first_order, second_order);

  const float *my_payload =
      is_thread_active ? sorted_payload + static_cast<size_t>(geometry_idx) * D
                       : sorted_payload;

  constexpr uint32_t CHUNK_SIZE = (D < 32U) ? D : 32U;
  constexpr uint32_t NUM_CHUNKS = (D + CHUNK_SIZE - 1U) / CHUNK_SIZE;

  for (uint32_t chunk_idx = 0; chunk_idx < NUM_CHUNKS; ++chunk_idx) {
    const uint32_t d0 = chunk_idx * CHUNK_SIZE;
    const uint32_t d_count = min(CHUNK_SIZE, D - d0);

    float v_slice[CHUNK_SIZE];
#pragma unroll
    for (uint32_t d = 0; d < CHUNK_SIZE; ++d) {
      v_slice[d] = (is_thread_active && d < d_count) ? my_payload[d0 + d] : 0.F;
    }

    // Warp-leader writes are the only writes; other lanes just participate in
    // the reduction. This helper is inlined. 
    auto aggregate_slot = [&](uint32_t slot_k, float c_k) -> void {
      // Thread-local: acc[d] = v_i[d] * c_k_i
      float acc[CHUNK_SIZE];
#pragma unroll
      for (uint32_t d = 0; d < CHUNK_SIZE; ++d) {
        acc[d] = v_slice[d] * c_k;
      }
      // Warp-reduce
#pragma unroll
      for (int offset = 16; offset > 0; offset /= 2) {
#pragma unroll
        for (uint32_t d = 0; d < CHUNK_SIZE; ++d) {
          acc[d] += __shfl_down_sync(0xFFFFFFFF, acc[d], offset);
        }
      }
      // Lane 0 writes.
      if (lane_id == 0) {
        const bool is_half_slot = (slot_k < 12U);
#pragma unroll
        for (uint32_t d = 0; d < CHUNK_SIZE; ++d) {
          if (d < d_count) {
            const uint16_t bits =
                is_half_slot
                    ? __half_as_ushort(__float2half_rn(acc[d]))
                    : __bfloat16_as_ushort(__float2bfloat16_rn(acc[d]));
            leaf_coefficients[leaf_idx].data[slot_k][d0 + d] = bits;
          }
        }
      }
    };

    // Zero order: slots 0..2 (half)
    aggregate_slot(0u, zero_order.x);
    aggregate_slot(1u, zero_order.y);
    aggregate_slot(2u, zero_order.z);

    // First order: slots 3..11 (half)
#pragma unroll
    for (int i = 0; i < 9; ++i) {
      aggregate_slot(3u + static_cast<uint32_t>(i), first_order.data[i]);
    }

    // Second order: slots 12..29 (bf16)
#pragma unroll
    for (int i = 0; i < 18; ++i) {
      aggregate_slot(12u + static_cast<uint32_t>(i), second_order.data[i]);
    }
  }

  // Race to top to merge aabbs
  if (lane_id == 0) {
    float my_weight = (weight == 0.F) ? 1e-10F : weight;

    uint32_t current_idx = leaf_node_idx;
    uint32_t current_parent_idx = binary_parents[current_idx];
    if (current_parent_idx == 0xFFFFFFFF) {
      return;
    }

    cuda::atomic_thread_fence(cuda::memory_order_seq_cst,
                              cuda::thread_scope_device);
    float other_weight =
        atomicAdd(&atomic_weights[current_parent_idx], my_weight);
    if (other_weight == 0.F) {
      return;
    }
    while (current_parent_idx != 0xFFFFFFFF) {
      BinaryNode parent_node = binary_nodes[current_parent_idx];

      float weight_left;
      float weight_right;
      if (parent_node.left_child == current_idx) {
        weight_left = my_weight;
        weight_right = other_weight;
      } else {
        weight_left = other_weight;
        weight_right = my_weight;
      }

      binary_aabbs[current_parent_idx] = AABB::merge_weighted(
          binary_aabbs[parent_node.left_child],
          binary_aabbs[parent_node.right_child], weight_left, weight_right);
      __threadfence();

      uint32_t next_parent = binary_parents[current_parent_idx];
      if (next_parent == 0xFFFFFFFF) {
        break;
      }
      my_weight = weight_left + weight_right;
      other_weight = atomicAdd(&atomic_weights[next_parent], my_weight);
      if (other_weight == 0.F) {
        break;
      }
      current_idx = current_parent_idx;
      current_parent_idx = next_parent;
    }
  }
}

template <IsGeometry Geometry, uint32_t D>
void populate_binary_tree_aabb_and_leaf_coefficients_payload(
    const float *sorted_geometry, const float *sorted_payload,
    PayloadTaylorCoefficientsF16<D> *leaf_coefficients, uint32_t leaf_count,
    const BinaryNode *binary_nodes, AABB *binary_aabbs,
    const uint32_t *binary_parents, float *atomic_weights,
    uint32_t geometry_count, cudaStream_t stream) {
  if (geometry_count == 0) {
    return;
  }
  constexpr uint32_t threads = 256;
  const uint32_t blocks = (leaf_count * 32 + threads - 1) / threads;
  populate_binary_tree_aabb_and_leaf_coefficients_payload_kernel<Geometry, D>
      <<<blocks, threads, 0, stream>>>(
          SoAView<Geometry>{sorted_geometry, geometry_count}, sorted_payload,
          leaf_coefficients, leaf_count, binary_nodes, binary_aabbs,
          binary_parents, atomic_weights, geometry_count);
  CUDA_CHECK(cudaGetLastError());
}

// Explicit instantiations
template void populate_binary_tree_aabb_and_leaf_coefficients<PointNormal>(
    const float *__restrict__ sorted_geometry,
    TaylorCoefficientsF16 *leaf_coefficients, uint32_t leaf_count,
    const BinaryNode *binary_nodes, AABB *binary_aabbs,
    const uint32_t *binary_parents, float *atomic_counters,
    uint32_t geometry_count, const cudaStream_t &stream);

template void populate_binary_tree_aabb_and_leaf_coefficients<Triangle>(
    const float *__restrict__ sorted_geometry,
    TaylorCoefficientsF16 *leaf_coefficients, uint32_t leaf_count,
    const BinaryNode *binary_nodes, AABB *binary_aabbs,
    const uint32_t *binary_parents, float *atomic_counters,
    uint32_t geometry_count, const cudaStream_t &stream);

// Explicit instantiations — one per (geometry, bucket) that
// PayloadWindingNumbersBackend exposes.
template void
populate_binary_tree_aabb_and_leaf_coefficients_payload<PointNormal, 1>(
    const float *, const float *, PayloadTaylorCoefficientsF16<1> *, uint32_t,
    const BinaryNode *, AABB *, const uint32_t *, float *, uint32_t,
    cudaStream_t);
template void
populate_binary_tree_aabb_and_leaf_coefficients_payload<PointNormal, 2>(
    const float *, const float *, PayloadTaylorCoefficientsF16<2> *, uint32_t,
    const BinaryNode *, AABB *, const uint32_t *, float *, uint32_t,
    cudaStream_t);
template void
populate_binary_tree_aabb_and_leaf_coefficients_payload<PointNormal, 3>(
    const float *, const float *, PayloadTaylorCoefficientsF16<3> *, uint32_t,
    const BinaryNode *, AABB *, const uint32_t *, float *, uint32_t,
    cudaStream_t);
template void
populate_binary_tree_aabb_and_leaf_coefficients_payload<PointNormal, 4>(
    const float *, const float *, PayloadTaylorCoefficientsF16<4> *, uint32_t,
    const BinaryNode *, AABB *, const uint32_t *, float *, uint32_t,
    cudaStream_t);
template void
populate_binary_tree_aabb_and_leaf_coefficients_payload<PointNormal, 8>(
    const float *, const float *, PayloadTaylorCoefficientsF16<8> *, uint32_t,
    const BinaryNode *, AABB *, const uint32_t *, float *, uint32_t,
    cudaStream_t);
template void
populate_binary_tree_aabb_and_leaf_coefficients_payload<PointNormal, 16>(
    const float *, const float *, PayloadTaylorCoefficientsF16<16> *, uint32_t,
    const BinaryNode *, AABB *, const uint32_t *, float *, uint32_t,
    cudaStream_t);
template void
populate_binary_tree_aabb_and_leaf_coefficients_payload<PointNormal, 32>(
    const float *, const float *, PayloadTaylorCoefficientsF16<32> *, uint32_t,
    const BinaryNode *, AABB *, const uint32_t *, float *, uint32_t,
    cudaStream_t);
template void
populate_binary_tree_aabb_and_leaf_coefficients_payload<PointNormal, 64>(
    const float *, const float *, PayloadTaylorCoefficientsF16<64> *, uint32_t,
    const BinaryNode *, AABB *, const uint32_t *, float *, uint32_t,
    cudaStream_t);
template void
populate_binary_tree_aabb_and_leaf_coefficients_payload<PointNormal, 128>(
    const float *, const float *, PayloadTaylorCoefficientsF16<128> *, uint32_t,
    const BinaryNode *, AABB *, const uint32_t *, float *, uint32_t,
    cudaStream_t);

// Same nine instantiations for Triangle
template void
populate_binary_tree_aabb_and_leaf_coefficients_payload<Triangle, 1>(
    const float *, const float *, PayloadTaylorCoefficientsF16<1> *, uint32_t,
    const BinaryNode *, AABB *, const uint32_t *, float *, uint32_t,
    cudaStream_t);
template void
populate_binary_tree_aabb_and_leaf_coefficients_payload<Triangle, 2>(
    const float *, const float *, PayloadTaylorCoefficientsF16<2> *, uint32_t,
    const BinaryNode *, AABB *, const uint32_t *, float *, uint32_t,
    cudaStream_t);
template void
populate_binary_tree_aabb_and_leaf_coefficients_payload<Triangle, 3>(
    const float *, const float *, PayloadTaylorCoefficientsF16<3> *, uint32_t,
    const BinaryNode *, AABB *, const uint32_t *, float *, uint32_t,
    cudaStream_t);
template void
populate_binary_tree_aabb_and_leaf_coefficients_payload<Triangle, 4>(
    const float *, const float *, PayloadTaylorCoefficientsF16<4> *, uint32_t,
    const BinaryNode *, AABB *, const uint32_t *, float *, uint32_t,
    cudaStream_t);
template void
populate_binary_tree_aabb_and_leaf_coefficients_payload<Triangle, 8>(
    const float *, const float *, PayloadTaylorCoefficientsF16<8> *, uint32_t,
    const BinaryNode *, AABB *, const uint32_t *, float *, uint32_t,
    cudaStream_t);
template void
populate_binary_tree_aabb_and_leaf_coefficients_payload<Triangle, 16>(
    const float *, const float *, PayloadTaylorCoefficientsF16<16> *, uint32_t,
    const BinaryNode *, AABB *, const uint32_t *, float *, uint32_t,
    cudaStream_t);
template void
populate_binary_tree_aabb_and_leaf_coefficients_payload<Triangle, 32>(
    const float *, const float *, PayloadTaylorCoefficientsF16<32> *, uint32_t,
    const BinaryNode *, AABB *, const uint32_t *, float *, uint32_t,
    cudaStream_t);
template void
populate_binary_tree_aabb_and_leaf_coefficients_payload<Triangle, 64>(
    const float *, const float *, PayloadTaylorCoefficientsF16<64> *, uint32_t,
    const BinaryNode *, AABB *, const uint32_t *, float *, uint32_t,
    cudaStream_t);
template void
populate_binary_tree_aabb_and_leaf_coefficients_payload<Triangle, 128>(
    const float *, const float *, PayloadTaylorCoefficientsF16<128> *, uint32_t,
    const BinaryNode *, AABB *, const uint32_t *, float *, uint32_t,
    cudaStream_t);
