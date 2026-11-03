#include "aabb.h"
#include "binary_node.h"
#include "bvh8.h"
#include "cuda/__functional/maximum.h"
#include "geometry.h"
#include "kernels/binary2bvh8.cuh"
#include "kernels/build_binary_tree.cuh"
#include "kernels/bvh8_m2m.cuh"
#include "kernels/common.cuh"
#include "kernels/mesh.cuh"
#include "kernels/payload.cuh"
#include "kernels/traversal.cuh"
#include "payload_backend.h"
#include "scene_normalization.h"
#include "soa.h"
#include "taylor_coefficients.h"
#include "thrust/detail/fill.inl"
#include "thrust/detail/reduce.inl"
#include "thrust/detail/sequence.inl"
#include "thrust/detail/sort.inl"
#include "thrust/execution_policy.h"
#include "thrust/system/cuda/detail/execution_policy.h"
#include "thrust/system/cuda/detail/fill.h"
#include "thrust/transform.h"
#include "utils.h"
#include "vec3.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <format>
#include <memory>
#include <stdexcept>
#include <string>
#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

using thrust::fill_n;

template <IsGeometry Geometry, uint32_t D>
PayloadWindingNumbersBackend<Geometry, D>::~PayloadWindingNumbersBackend() {
  int device = -1;
  if (cudaGetDevice(&device) != cudaSuccess) {
    return;
  }
  ScopedCudaDevice device_scope(m_device);
  CUDA_CHECK(cudaStreamSynchronize(m_build_stream));
  cudaGetLastError();

  if (m_start_tree_construction_event != nullptr) {
    CUDA_CHECK(cudaEventDestroy(m_start_tree_construction_event));
  }
  if (m_tree_construction_finished_event != nullptr) {
    CUDA_CHECK(cudaEventDestroy(m_tree_construction_finished_event));
  }

#define FREE_IF(x)                                                             \
  if (x) {                                                                     \
    CUDA_CHECK(cudaFreeAsync(x, m_build_stream));                              \
    x = nullptr;                                                               \
  }
  FREE_IF(m_to_internal)
  FREE_IF(m_sorted_geometry)
  FREE_IF(m_sorted_payload)
  FREE_IF(m_binary_aabbs)
  FREE_IF(m_bvh8_node_count)
  FREE_IF(m_bvh8_nodes)
  FREE_IF(m_tailor_coefficients)
  FREE_IF(m_leaf_coefficients)
  FREE_IF(m_bvh8_leaf_pointers)
#undef FREE_IF
}

// ===========================================================================
// Constructor — allocate payload coefficient arrays
// ===========================================================================
template <IsGeometry Geometry, uint32_t D>
PayloadWindingNumbersBackend<Geometry, D>::PayloadWindingNumbersBackend(
    size_t size, int device_id, uint64_t stream)
    : m_device{device_id}, m_count{size} {

  m_build_stream = reinterpret_cast<cudaStream_t>(stream);
  m_build_stream_policy = thrust::cuda::par.on(m_build_stream);

  CUDA_CHECK(cudaEventCreate(&m_start_tree_construction_event));
  CUDA_CHECK(cudaEventCreate(&m_tree_construction_finished_event));

  size_t leaf_count = (size + LEAF_SIZE - 1) / LEAF_SIZE;
  uint32_t max_bvh8_nodes = leaf_count - 1;

  CUDA_CHECK(
      cudaMallocAsync(&m_to_internal, size * sizeof(uint32_t), m_build_stream));
  CUDA_CHECK(cudaMallocAsync(&m_sorted_geometry, size * sizeof(Geometry),
                             m_build_stream));
  CUDA_CHECK(cudaMallocAsync(&m_sorted_payload, size * D * sizeof(float),
                             m_build_stream));
  CUDA_CHECK(cudaMallocAsync(
      &m_binary_aabbs, (2 * leaf_count - 1) * sizeof(AABB), m_build_stream));
  CUDA_CHECK(
      cudaMallocAsync(&m_bvh8_node_count, sizeof(uint32_t), m_build_stream));
  CUDA_CHECK(cudaMallocAsync(&m_bvh8_nodes, max_bvh8_nodes * sizeof(BVH8Node),
                             m_build_stream));

  CUDA_CHECK(
      cudaMallocAsync(&m_tailor_coefficients,
                      max_bvh8_nodes * sizeof(PayloadTaylorCoefficientsF16<D>),
                      m_build_stream));
  CUDA_CHECK(cudaMallocAsync(
      &m_leaf_coefficients,
      leaf_count * sizeof(PayloadTaylorCoefficientsF16<D>), m_build_stream));

  CUDA_CHECK(cudaMallocAsync(&m_bvh8_leaf_pointers,
                             max_bvh8_nodes * sizeof(LeafPointers),
                             m_build_stream));

  static_assert(4 * sizeof(PayloadTaylorCoefficientsF16<D>) <= 48 * 1024,
                "D too large for shared-memory node cache");
}

// ===========================================================================
// initialize_triangle_data
// ===========================================================================
template <IsGeometry Geometry, uint32_t D>
void PayloadWindingNumbersBackend<Geometry, D>::initialize_triangle_data(
    const float *triangles, const float *payload)
  requires std::is_same_v<Geometry, Triangle>
{
  const auto *triangles_tri = reinterpret_cast<const Triangle *>(triangles);

  CUDA_CHECK(cudaEventRecord(m_start_tree_construction_event, m_build_stream));

  // ---------- Morton + sort ----------
  uint64_t *geometry_morton_codes;
  CUDA_CHECK(cudaMallocAsync(&geometry_morton_codes, m_count * sizeof(uint64_t),
                             m_build_stream));
  SceneBounds scene_bounds;
  initializeMortonCodes(triangles_tri, geometry_morton_codes, m_count,
                        m_build_stream, &scene_bounds);
  m_norm = SceneNormalization::from_scene_bounds(scene_bounds);

  thrust::sequence(m_build_stream_policy, m_to_internal,
                   m_to_internal + m_count);
  thrust::stable_sort_by_key(m_build_stream_policy, geometry_morton_codes,
                             geometry_morton_codes + m_count, m_to_internal);

  gather_triangles_soa(triangles, m_to_internal, m_sorted_geometry, m_count,
                       m_norm, m_build_stream);
  gather_payload<D>(payload, m_to_internal, m_sorted_payload, m_count,
                    m_build_stream);

  // ---------- Leaf coefficients ----------
  uint32_t leaf_count = (m_count + LEAF_SIZE - 1) / LEAF_SIZE;
  uint32_t max_bvh8_nodes = leaf_count - 1;

  auto morton_leaf_stride_idx = thrust::make_transform_iterator(
      thrust::make_counting_iterator<uint64_t>(0),
      [] __host__ __device__(uint64_t i) -> uint64_t { return i * LEAF_SIZE; });
  auto morton_leaf_stride = thrust::make_permutation_iterator(
      geometry_morton_codes, morton_leaf_stride_idx);
  uint64_t *leaf_morton_codes;
  CUDA_CHECK(cudaMallocAsync(&leaf_morton_codes, m_count * sizeof(uint64_t),
                             m_build_stream));
  thrust::copy(m_build_stream_policy, morton_leaf_stride,
               morton_leaf_stride + leaf_count, leaf_morton_codes);
  CUDA_CHECK(cudaFreeAsync(geometry_morton_codes, m_build_stream));

  BinaryNode *binary_nodes;
  uint32_t *binary_parents;
  CUDA_CHECK(cudaMallocAsync(
      &binary_nodes, (leaf_count - 1) * sizeof(BinaryNode), m_build_stream));
  CUDA_CHECK(cudaMallocAsync(&binary_parents,
                             (2 * leaf_count - 1) * sizeof(uint32_t),
                             m_build_stream));
  build_binary_topology(leaf_morton_codes, binary_nodes, binary_parents,
                        leaf_count, m_build_stream);
  CUDA_CHECK(cudaFreeAsync(leaf_morton_codes, m_build_stream));

  float *atomic_weights;
  CUDA_CHECK(cudaMallocAsync(&atomic_weights, (leaf_count - 1) * sizeof(float),
                             m_build_stream));
  fill_n(m_build_stream_policy, atomic_weights, leaf_count - 1, 0.F);

  populate_binary_tree_aabb_and_leaf_coefficients_payload<Triangle, D>(
      m_sorted_geometry, m_sorted_payload, m_leaf_coefficients, leaf_count,
      binary_nodes, m_binary_aabbs, binary_parents, atomic_weights, m_count,
      m_build_stream);

  CUDA_CHECK(cudaFreeAsync(binary_parents, m_build_stream));
  CUDA_CHECK(cudaFreeAsync(atomic_weights, m_build_stream));

  // ---------- BVH8 conversion ----------
  uint32_t *bvh8_work_queue_A;
  uint32_t *bvh8_work_queue_B;
  uint32_t *bvh8_internal_parent_map;
  uint32_t *global_counter;
  uint32_t *bvh8_leaf_parents;
  uint32_t *bvh8_nodes_child_count;
  CUDA_CHECK(cudaMallocAsync(
      &bvh8_work_queue_A, (leaf_count - 1) * sizeof(uint32_t), m_build_stream));
  CUDA_CHECK(cudaMallocAsync(
      &bvh8_work_queue_B, (leaf_count - 1) * sizeof(uint32_t), m_build_stream));
  CUDA_CHECK(cudaMallocAsync(&bvh8_internal_parent_map,
                             max_bvh8_nodes * sizeof(uint32_t),
                             m_build_stream));
  CUDA_CHECK(
      cudaMallocAsync(&global_counter, sizeof(uint32_t), m_build_stream));
  CUDA_CHECK(cudaMallocAsync(&bvh8_leaf_parents, leaf_count * sizeof(uint32_t),
                             m_build_stream));
  CUDA_CHECK(cudaMallocAsync(&bvh8_nodes_child_count,
                             max_bvh8_nodes * sizeof(uint32_t),
                             m_build_stream));

  ConvertBinary2BVH8Params params{
      .work_queue_A = bvh8_work_queue_A,
      .work_queue_B = bvh8_work_queue_B,
      .bvh8_internal_parents = bvh8_internal_parent_map,
      .global_counter = global_counter,
      .leaf_count = leaf_count,
      .binary_aabbs = m_binary_aabbs,
      .binary_nodes = binary_nodes,
      .nodes_child_count = bvh8_nodes_child_count,
      .bvh8_leaf_parents = bvh8_leaf_parents,
      .bvh8_nodes = m_bvh8_nodes,
      .bvh8_leaf_pointers = m_bvh8_leaf_pointers,
      .bvh8_node_count = m_bvh8_node_count,
  };
  convert_binary_tree_to_bvh8(params, m_device, m_build_stream);

  CUDA_CHECK(cudaFreeAsync(bvh8_work_queue_A, m_build_stream));
  CUDA_CHECK(cudaFreeAsync(bvh8_work_queue_B, m_build_stream));
  CUDA_CHECK(cudaFreeAsync(global_counter, m_build_stream));
  CUDA_CHECK(cudaFreeAsync(binary_nodes, m_build_stream));

  // ---------- max distances ----------
  uint32_t bvh8_node_count;
  CUDA_CHECK(cudaMemcpyAsync(&bvh8_node_count, m_bvh8_node_count,
                             sizeof(uint32_t), cudaMemcpyDeviceToHost,
                             m_build_stream));
  float *tmp_max_distances;
  CUDA_CHECK(cudaMallocAsync(&tmp_max_distances,
                             bvh8_node_count * sizeof(float), m_build_stream));
  thrust::fill_n(m_build_stream_policy, tmp_max_distances, bvh8_node_count,
                 0.F);
  compute_max_distances<Triangle>(
      m_bvh8_nodes, m_sorted_geometry, bvh8_leaf_parents,
      bvh8_internal_parent_map, tmp_max_distances,
      static_cast<uint32_t>(m_count), bvh8_node_count, m_build_stream);
  CUDA_CHECK(cudaFreeAsync(tmp_max_distances, m_build_stream));

  // ---------- m2m ----------
  uint32_t *atomic_counters;
  CUDA_CHECK(cudaMallocAsync(
      &atomic_counters, (leaf_count - 1) * sizeof(uint32_t), m_build_stream));
  thrust::fill_n(m_build_stream_policy, atomic_counters, leaf_count - 1, 0);
  PayloadTaylorCoefficients<D> *m2m_f32_coefficients;
  CUDA_CHECK(cudaMallocAsync(
      &m2m_f32_coefficients,
      max_bvh8_nodes * sizeof(PayloadTaylorCoefficients<D>), m_build_stream));

  compute_internal_tailor_coefficients_m2m_payload<D>(
      m_bvh8_nodes, bvh8_internal_parent_map, m_binary_aabbs + leaf_count - 1,
      m_leaf_coefficients, bvh8_leaf_parents, m_bvh8_leaf_pointers,
      m_tailor_coefficients, m2m_f32_coefficients, bvh8_nodes_child_count,
      leaf_count, atomic_counters, m_build_stream);

  CUDA_CHECK(cudaFreeAsync(m2m_f32_coefficients, m_build_stream));
  CUDA_CHECK(cudaFreeAsync(atomic_counters, m_build_stream));
  CUDA_CHECK(cudaFreeAsync(bvh8_internal_parent_map, m_build_stream));
  CUDA_CHECK(cudaFreeAsync(bvh8_leaf_parents, m_build_stream));
  CUDA_CHECK(cudaFreeAsync(bvh8_nodes_child_count, m_build_stream));

  CUDA_CHECK(
      cudaEventRecord(m_tree_construction_finished_event, m_build_stream));
}

// ===========================================================================
// initialize_point_data
// ===========================================================================
template <IsGeometry Geometry, uint32_t D>
void PayloadWindingNumbersBackend<Geometry, D>::initialize_point_data(
    const float *points, const float *normals, const float *payload)
  requires std::is_same_v<Geometry, PointNormal>
{
  const auto *points_v3 = reinterpret_cast<const Vec3 *>(points);

  CUDA_CHECK(cudaEventRecord(m_start_tree_construction_event, m_build_stream));

  // ---------- Morton + sort ----------
  uint64_t *geometry_morton_codes;
  CUDA_CHECK(cudaMallocAsync(&geometry_morton_codes, m_count * sizeof(uint64_t),
                             m_build_stream));
  SceneBounds scene_bounds;
  initializeMortonCodes(points_v3, geometry_morton_codes, m_count,
                        m_build_stream, &scene_bounds);
  m_norm = SceneNormalization::from_scene_bounds(scene_bounds);

  thrust::sequence(m_build_stream_policy, m_to_internal,
                   m_to_internal + m_count);
  thrust::stable_sort_by_key(m_build_stream_policy, geometry_morton_codes,
                             geometry_morton_codes + m_count, m_to_internal);

  gather_point_normals_soa(points, normals, m_to_internal, m_sorted_geometry,
                           m_count, m_norm, m_build_stream);
  gather_payload<D>(payload, m_to_internal, m_sorted_payload, m_count,
                    m_build_stream);

  // ---------- Leaf coefficients ----------
  uint32_t leaf_count = (m_count + LEAF_SIZE - 1) / LEAF_SIZE;
  uint32_t max_bvh8_nodes = leaf_count - 1;

  auto morton_leaf_stride_idx = thrust::make_transform_iterator(
      thrust::make_counting_iterator<uint64_t>(0),
      [] __host__ __device__(uint64_t i) -> uint64_t { return i * LEAF_SIZE; });
  auto morton_leaf_stride = thrust::make_permutation_iterator(
      geometry_morton_codes, morton_leaf_stride_idx);
  uint64_t *leaf_morton_codes;
  CUDA_CHECK(cudaMallocAsync(&leaf_morton_codes, m_count * sizeof(uint64_t),
                             m_build_stream));
  thrust::copy(m_build_stream_policy, morton_leaf_stride,
               morton_leaf_stride + leaf_count, leaf_morton_codes);
  CUDA_CHECK(cudaFreeAsync(geometry_morton_codes, m_build_stream));

  BinaryNode *binary_nodes;
  uint32_t *binary_parents;
  CUDA_CHECK(cudaMallocAsync(
      &binary_nodes, (leaf_count - 1) * sizeof(BinaryNode), m_build_stream));
  CUDA_CHECK(cudaMallocAsync(&binary_parents,
                             (2 * leaf_count - 1) * sizeof(uint32_t),
                             m_build_stream));
  build_binary_topology(leaf_morton_codes, binary_nodes, binary_parents,
                        leaf_count, m_build_stream);
  CUDA_CHECK(cudaFreeAsync(leaf_morton_codes, m_build_stream));

  float *atomic_weights;
  CUDA_CHECK(cudaMallocAsync(&atomic_weights, (leaf_count - 1) * sizeof(float),
                             m_build_stream));
  thrust::fill_n(m_build_stream_policy, atomic_weights, leaf_count - 1, 0.F);

  populate_binary_tree_aabb_and_leaf_coefficients_payload<PointNormal, D>(
      m_sorted_geometry, m_sorted_payload, m_leaf_coefficients, leaf_count,
      binary_nodes, m_binary_aabbs, binary_parents, atomic_weights, m_count,
      m_build_stream);

  CUDA_CHECK(cudaFreeAsync(binary_parents, m_build_stream));
  CUDA_CHECK(cudaFreeAsync(atomic_weights, m_build_stream));

  // ---------- BVH8 conversion ----------
  uint32_t *bvh8_work_queue_A, *bvh8_work_queue_B;
  uint32_t *bvh8_internal_parent_map, *global_counter;
  uint32_t *bvh8_leaf_parents, *bvh8_nodes_child_count;
  CUDA_CHECK(cudaMallocAsync(
      &bvh8_work_queue_A, (leaf_count - 1) * sizeof(uint32_t), m_build_stream));
  CUDA_CHECK(cudaMallocAsync(
      &bvh8_work_queue_B, (leaf_count - 1) * sizeof(uint32_t), m_build_stream));
  CUDA_CHECK(cudaMallocAsync(&bvh8_internal_parent_map,
                             max_bvh8_nodes * sizeof(uint32_t),
                             m_build_stream));
  CUDA_CHECK(
      cudaMallocAsync(&global_counter, sizeof(uint32_t), m_build_stream));
  CUDA_CHECK(cudaMallocAsync(&bvh8_leaf_parents, leaf_count * sizeof(uint32_t),
                             m_build_stream));
  CUDA_CHECK(cudaMallocAsync(&bvh8_nodes_child_count,
                             max_bvh8_nodes * sizeof(uint32_t),
                             m_build_stream));

  ConvertBinary2BVH8Params params{
      .work_queue_A = bvh8_work_queue_A,
      .work_queue_B = bvh8_work_queue_B,
      .bvh8_internal_parents = bvh8_internal_parent_map,
      .global_counter = global_counter,
      .leaf_count = leaf_count,
      .binary_aabbs = m_binary_aabbs,
      .binary_nodes = binary_nodes,
      .nodes_child_count = bvh8_nodes_child_count,
      .bvh8_leaf_parents = bvh8_leaf_parents,
      .bvh8_nodes = m_bvh8_nodes,
      .bvh8_leaf_pointers = m_bvh8_leaf_pointers,
      .bvh8_node_count = m_bvh8_node_count,
  };
  convert_binary_tree_to_bvh8(params, m_device, m_build_stream);

  CUDA_CHECK(cudaFreeAsync(bvh8_work_queue_A, m_build_stream));
  CUDA_CHECK(cudaFreeAsync(bvh8_work_queue_B, m_build_stream));
  CUDA_CHECK(cudaFreeAsync(global_counter, m_build_stream));
  CUDA_CHECK(cudaFreeAsync(binary_nodes, m_build_stream));

  // ---------- max distances ----------
  uint32_t bvh8_node_count;
  CUDA_CHECK(cudaMemcpyAsync(&bvh8_node_count, m_bvh8_node_count,
                             sizeof(uint32_t), cudaMemcpyDeviceToHost,
                             m_build_stream));
  float *tmp_max_distances;
  CUDA_CHECK(cudaMallocAsync(&tmp_max_distances,
                             bvh8_node_count * sizeof(float), m_build_stream));
  thrust::fill_n(m_build_stream_policy, tmp_max_distances, bvh8_node_count,
                 0.F);
  compute_max_distances<PointNormal>(
      m_bvh8_nodes, m_sorted_geometry, bvh8_leaf_parents,
      bvh8_internal_parent_map, tmp_max_distances,
      static_cast<uint32_t>(m_count), bvh8_node_count, m_build_stream);
  CUDA_CHECK(cudaFreeAsync(tmp_max_distances, m_build_stream));

  // ---------- m2m ----------
  uint32_t *atomic_counters;
  CUDA_CHECK(cudaMallocAsync(
      &atomic_counters, (leaf_count - 1) * sizeof(uint32_t), m_build_stream));
  thrust::fill_n(m_build_stream_policy, atomic_counters, leaf_count - 1, 0);
  PayloadTaylorCoefficients<D> *m2m_f32_coefficients;
  CUDA_CHECK(cudaMallocAsync(
      &m2m_f32_coefficients,
      max_bvh8_nodes * sizeof(PayloadTaylorCoefficients<D>), m_build_stream));

  compute_internal_tailor_coefficients_m2m_payload<D>(
      m_bvh8_nodes, bvh8_internal_parent_map, m_binary_aabbs + leaf_count - 1,
      m_leaf_coefficients, bvh8_leaf_parents, m_bvh8_leaf_pointers,
      m_tailor_coefficients, m2m_f32_coefficients, bvh8_nodes_child_count,
      leaf_count, atomic_counters, m_build_stream);

  CUDA_CHECK(cudaFreeAsync(m2m_f32_coefficients, m_build_stream));
  CUDA_CHECK(cudaFreeAsync(atomic_counters, m_build_stream));
  CUDA_CHECK(cudaFreeAsync(bvh8_internal_parent_map, m_build_stream));
  CUDA_CHECK(cudaFreeAsync(bvh8_leaf_parents, m_build_stream));
  CUDA_CHECK(cudaFreeAsync(bvh8_nodes_child_count, m_build_stream));

  CUDA_CHECK(
      cudaEventRecord(m_tree_construction_finished_event, m_build_stream));
}

// ===========================================================================
// compute_payload
// ===========================================================================
template <IsGeometry Geometry, uint32_t D>
auto PayloadWindingNumbersBackend<Geometry, D>::compute_payload(
    const float *queries, size_t query_count, float *out_payload, float beta,
    float epsilon, size_t stream) const -> void {
  if (m_count == 0) {
    return;
  }

  ScopedCudaDevice device_scope{m_device};
  const Vec3 *queries_vec3 = reinterpret_cast<const Vec3 *>(queries);
  cudaStream_t compute_stream = reinterpret_cast<cudaStream_t>(stream);
  auto compute_stream_policy = thrust::cuda::par.on(compute_stream);

  uint32_t *queries_to_internal;
  uint64_t *queries_morton;
  CUDA_CHECK(cudaMallocAsync(&queries_to_internal,
                             query_count * sizeof(uint32_t), compute_stream));
  CUDA_CHECK(cudaMallocAsync(&queries_morton, query_count * sizeof(uint64_t),
                             compute_stream));
  thrust::transform(compute_stream_policy, queries_vec3,
                    queries_vec3 + query_count, queries_morton,
                    GeometryToMorton<Vec3>{m_norm.scale, m_norm.min_p});
  thrust::sequence(compute_stream_policy, queries_to_internal,
                   queries_to_internal + query_count);
  thrust::stable_sort_by_key(compute_stream_policy, queries_morton,
                             queries_morton + query_count, queries_to_internal);
  CUDA_CHECK(cudaFreeAsync(queries_morton, compute_stream));

  CUDA_CHECK(
      cudaStreamWaitEvent(compute_stream, m_tree_construction_finished_event));

  if (beta < 0.F) {
    beta = GeometryTraits<Geometry>::default_beta;
  }
  epsilon = std::max(0.F, epsilon);
  float scene_scale = 1.F;

  uint32_t leaf_count = (m_count + LEAF_SIZE - 1) / LEAF_SIZE;
  uint32_t *global_counter;
  CUDA_CHECK(
      cudaMallocAsync(&global_counter, sizeof(uint32_t), compute_stream));

  ComputePayloadWindingNumbersParams<Geometry, D> params{
      .queries = queries_vec3,
      .sort_indirections = queries_to_internal,
      .bvh8_nodes = m_bvh8_nodes,
      .bvh8_leaf_pointers = m_bvh8_leaf_pointers,
      .node_coefficients = m_tailor_coefficients,
      .leaf_coefficients = m_leaf_coefficients,
      .leaf_aabbs = m_binary_aabbs + leaf_count - 1,
      .sorted_geometry = SoAView<Geometry>{m_sorted_geometry, m_count},
      .sorted_payload = m_sorted_payload,
      .query_count = static_cast<uint32_t>(query_count),
      .geometry_count = static_cast<uint32_t>(m_count),
      .out_payload = out_payload,
      .global_device_counter = global_counter,
      .beta = beta,
      .epsilon = epsilon,
      .scene_scale = scene_scale,
      .norm = m_norm};

  compute_payload_winding_numbers<Geometry, D>(params, m_device,
                                               compute_stream);

  CUDA_CHECK(cudaFreeAsync(queries_to_internal, compute_stream));
  CUDA_CHECK(cudaFreeAsync(global_counter, compute_stream));
}

// ===========================================================================
// compute_payload_query_gradients
// ===========================================================================
template <IsGeometry Geometry, uint32_t D>
auto PayloadWindingNumbersBackend<Geometry, D>::compute_payload_query_gradients(
    const float *grad_output, const float *queries, size_t query_count,
    float *out_query_gradients, float beta, float epsilon, size_t stream) const
    -> void {
  if (m_count == 0) {
    return;
  }

  ScopedCudaDevice device_scope{m_device};
  const Vec3 *queries_vec3 = reinterpret_cast<const Vec3 *>(queries);
  cudaStream_t compute_stream = reinterpret_cast<cudaStream_t>(stream);
  auto compute_stream_policy = thrust::cuda::par.on(compute_stream);

  uint32_t *queries_to_internal;
  uint64_t *queries_morton;
  CUDA_CHECK(cudaMallocAsync(&queries_to_internal,
                             query_count * sizeof(uint32_t), compute_stream));
  CUDA_CHECK(cudaMallocAsync(&queries_morton, query_count * sizeof(uint64_t),
                             compute_stream));
  thrust::transform(compute_stream_policy, queries_vec3,
                    queries_vec3 + query_count, queries_morton,
                    GeometryToMorton<Vec3>{m_norm.scale, m_norm.min_p});
  thrust::sequence(compute_stream_policy, queries_to_internal,
                   queries_to_internal + query_count);
  thrust::stable_sort_by_key(compute_stream_policy, queries_morton,
                             queries_morton + query_count, queries_to_internal);
  CUDA_CHECK(cudaFreeAsync(queries_morton, compute_stream));

  CUDA_CHECK(
      cudaStreamWaitEvent(compute_stream, m_tree_construction_finished_event));

  if (beta < 0.F) {
    beta = GeometryTraits<Geometry>::default_query_beta;
  }
  epsilon = std::max(0.F, epsilon);
  float scene_scale = 1.F;
  uint32_t leaf_count = (m_count + LEAF_SIZE - 1) / LEAF_SIZE;
  uint32_t *global_counter;
  CUDA_CHECK(
      cudaMallocAsync(&global_counter, sizeof(uint32_t), compute_stream));

  ComputePayloadQueryGradientParams<Geometry, D> params{
      .queries = queries_vec3,
      .sort_indirections = queries_to_internal,
      .bvh8_nodes = m_bvh8_nodes,
      .bvh8_leaf_pointers = m_bvh8_leaf_pointers,
      .node_coefficients = m_tailor_coefficients,
      .leaf_coefficients = m_leaf_coefficients,
      .leaf_aabbs = m_binary_aabbs + leaf_count - 1,
      .sorted_geometry = SoAView<Geometry>{m_sorted_geometry, m_count},
      .sorted_payload = m_sorted_payload,
      .query_count = (uint32_t)query_count,
      .geometry_count = (uint32_t)m_count,
      .grad_output = grad_output,
      .out_query_gradients = out_query_gradients,
      .global_device_counter = global_counter,
      .beta = beta,
      .epsilon = epsilon,
      .scene_scale = scene_scale,
      .norm = m_norm};

  query_gradient_traversal_payload<Geometry, D>(params, m_device,
                                                compute_stream);

  CUDA_CHECK(cudaFreeAsync(queries_to_internal, compute_stream));
  CUDA_CHECK(cudaFreeAsync(global_counter, compute_stream));
}

// ===========================================================================
// Factories
// ===========================================================================
template <IsGeometry Geometry, uint32_t D>
auto PayloadWindingNumbersBackend<Geometry, D>::CreateFromTriangles(
    const float *triangles, const float *payload, size_t triangle_count,
    int device_id, uint64_t stream)
    -> std::unique_ptr<PayloadWindingNumbersBackend<Triangle, D>>
  requires std::is_same_v<Geometry, Triangle>
{
  ScopedCudaDevice device_scope(device_id);
  auto self = std::unique_ptr<PayloadWindingNumbersBackend>{
      new PayloadWindingNumbersBackend(triangle_count, device_id, stream)};
  self->initialize_triangle_data(triangles, payload);
  return self;
}

template <IsGeometry Geometry, uint32_t D>
auto PayloadWindingNumbersBackend<Geometry, D>::CreateFromMesh(
    const float *vertices, size_t vertex_count,
    const uint32_t *triangle_indices, const float *payload,
    size_t triangle_count, int device_id, uint64_t stream)
    -> std::unique_ptr<PayloadWindingNumbersBackend<Triangle, D>>
  requires std::is_same_v<Geometry, Triangle>
{
  ScopedCudaDevice device_scope(device_id);
  auto self = std::unique_ptr<PayloadWindingNumbersBackend>{
      new PayloadWindingNumbersBackend(triangle_count, device_id, stream)};

  if (triangle_count == 0) {
    return self;
  }

  // Verify that index range does not exceed vertex size
  uint32_t max_index = thrust::reduce(thrust::device, triangle_indices,
                                      triangle_indices + 3 * triangle_count, 0,
                                      cuda::maximum<uint32_t>());
  if (max_index >= vertex_count) {
    throw std::runtime_error(
        std::format("The triangle indices are not allowed to exceed the number "
                    "of vertices. Vertex count is {}, max index is {}.",
                    vertex_count, max_index));
  }

  // Create temporary triangles from indices. The payload is already
  // per-triangle, so it passes through unchanged.
  float *triangles;
  CUDA_CHECK(cudaMallocAsync(&triangles, triangle_count * sizeof(Triangle),
                             self->m_build_stream));

  gather_triangles(vertices, triangle_indices,
                   static_cast<uint32_t>(triangle_count), triangles,
                   self->m_build_stream);

  self->initialize_triangle_data(triangles, payload);

  CUDA_CHECK(cudaFreeAsync(triangles, self->m_build_stream));
  return self;
}

template <IsGeometry Geometry, uint32_t D>
auto PayloadWindingNumbersBackend<Geometry, D>::CreateFromPoints(
    const float *points, const float *scaled_normals, const float *payload,
    size_t point_count, int device_id, uint64_t stream)
    -> std::unique_ptr<PayloadWindingNumbersBackend<PointNormal, D>>
  requires std::is_same_v<Geometry, PointNormal>
{
  ScopedCudaDevice device_scope(device_id);
  auto self = std::unique_ptr<PayloadWindingNumbersBackend>{
      new PayloadWindingNumbersBackend(point_count, device_id, stream)};
  self->initialize_point_data(points, scaled_normals, payload);
  return self;
}

// ===========================================================================
// dump — prints component d=0 only
// ===========================================================================
template <IsGeometry Geometry, uint32_t D>
auto PayloadWindingNumbersBackend<Geometry, D>::dump() const -> std::string {
  return "digraph BVH8 { /* payload D=" + std::to_string(D) +
         " — see scalar backend */ }\n";
}

// ===========================================================================
// Explicit instantiations
// ===========================================================================
template class PayloadWindingNumbersBackend<PointNormal, 1>;
template class PayloadWindingNumbersBackend<PointNormal, 2>;
template class PayloadWindingNumbersBackend<PointNormal, 3>;
template class PayloadWindingNumbersBackend<PointNormal, 4>;
template class PayloadWindingNumbersBackend<PointNormal, 8>;
template class PayloadWindingNumbersBackend<PointNormal, 16>;
template class PayloadWindingNumbersBackend<PointNormal, 32>;
template class PayloadWindingNumbersBackend<PointNormal, 64>;
template class PayloadWindingNumbersBackend<PointNormal, 128>;

template class PayloadWindingNumbersBackend<Triangle, 1>;
template class PayloadWindingNumbersBackend<Triangle, 2>;
template class PayloadWindingNumbersBackend<Triangle, 3>;
template class PayloadWindingNumbersBackend<Triangle, 4>;
template class PayloadWindingNumbersBackend<Triangle, 8>;
template class PayloadWindingNumbersBackend<Triangle, 16>;
template class PayloadWindingNumbersBackend<Triangle, 32>;
template class PayloadWindingNumbersBackend<Triangle, 64>;
template class PayloadWindingNumbersBackend<Triangle, 128>;
