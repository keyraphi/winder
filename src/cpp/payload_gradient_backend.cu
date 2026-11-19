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
#include "mat3x3.h"
#include "payload_gradient_backend.h"
#include "scene_normalization.h"
#include "soa.h"
#include "taylor_coefficients.h"
#include "tensor3.h"
#include "thrust/detail/fill.inl"
#include "thrust/detail/reduce.inl"
#include "thrust/detail/sequence.inl"
#include "thrust/detail/sort.inl"
#include "thrust/execution_policy.h"
#include "thrust/iterator/counting_iterator.h"
#include "thrust/iterator/permutation_iterator.h"
#include "thrust/iterator/transform_iterator.h"
#include "thrust/system/cuda/detail/execution_policy.h"
#include "utils.h"
#include "vec3.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <format>
#include <queue>
#include <string>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <vector>

template <IsGeometry Geometry, uint32_t D>
PayloadGradientBackend<Geometry, D>::PayloadGradientBackend(
    size_t query_count, uint32_t payload_dim_user, int device_id,
    uint64_t stream)
    : m_device{device_id}, m_query_count{query_count},
      m_payload_dim_user{payload_dim_user} {

  m_build_stream = reinterpret_cast<cudaStream_t>(stream);

  uint32_t leaf_count = (m_query_count + LEAF_SIZE - 1) / LEAF_SIZE;
  uint32_t max_bvh8_nodes = leaf_count > 0 ? leaf_count - 1 : 0;

  CUDA_CHECK(cudaEventCreate(&m_tree_construction_finished_event));

  CUDA_CHECK(cudaMallocAsync(&m_to_internal, query_count * sizeof(uint32_t),
                             m_build_stream));
  CUDA_CHECK(cudaMallocAsync(&m_sorted_queries, query_count * sizeof(Vec3),
                             m_build_stream));
  CUDA_CHECK(cudaMallocAsync(&m_sorted_grad_outputs_bucket,
                             query_count * D * sizeof(float), m_build_stream));

  CUDA_CHECK(cudaMallocAsync(
      &m_binary_aabbs, (2 * leaf_count - 1) * sizeof(AABB), m_build_stream));
  CUDA_CHECK(
      cudaMallocAsync(&m_bvh8_node_count, sizeof(uint32_t), m_build_stream));
  CUDA_CHECK(cudaMallocAsync(&m_bvh8_nodes, max_bvh8_nodes * sizeof(BVH8Node),
                             m_build_stream));
  CUDA_CHECK(cudaMallocAsync(&m_bvh8_leaf_pointers,
                             max_bvh8_nodes * sizeof(LeafPointers),
                             m_build_stream));

  static_assert(4 * sizeof(BackwardPayloadTaylorCoefficients<D>) <= 48 * 1024,
                "D too large for shared-memory node cache");
}

template <IsGeometry Geometry, uint32_t D>
PayloadGradientBackend<Geometry, D>::~PayloadGradientBackend() {
  int device = -1;
  if (cudaGetDevice(&device) != cudaSuccess) {
    return;
  }
  ScopedCudaDevice device_scope(m_device);

  cudaStreamSynchronize(m_build_stream);
  cudaGetLastError();

  if (m_tree_construction_finished_event) {
    cudaEventDestroy(m_tree_construction_finished_event);
    m_tree_construction_finished_event = nullptr;
  }

#define FREE_IF(x)                                                             \
  if (x) {                                                                     \
    cudaFreeAsync(x, m_build_stream);                                          \
    x = nullptr;                                                               \
  }
  FREE_IF(m_to_internal)
  FREE_IF(m_sorted_queries)
  FREE_IF(m_sorted_grad_outputs_bucket)
  FREE_IF(m_binary_aabbs)
  FREE_IF(m_bvh8_node_count)
  FREE_IF(m_bvh8_nodes)
  FREE_IF(m_bvh8_leaf_pointers)
  FREE_IF(m_leaf_coefficients)
  FREE_IF(m_taylor_coefficients)
#undef FREE_IF

  cudaStreamSynchronize(m_build_stream);
}

// ===========================================================================
// init — build the query BVH
// ===========================================================================
template <IsGeometry Geometry, uint32_t D>
void PayloadGradientBackend<Geometry, D>::init(const float *queries,
                                               const float *grad_output) {
  if (m_query_count == 0) {
    return;
  }

  const auto *queries_v3 = reinterpret_cast<const Vec3 *>(queries);
  auto build_stream_policy = thrust::cuda::par.on(m_build_stream);

  uint32_t leaf_count = (m_query_count + LEAF_SIZE - 1) / LEAF_SIZE;
  uint32_t max_bvh8_nodes = leaf_count - 1;

  // ---------- Morton sort of queries ----------
  uint64_t *query_morton_codes;
  CUDA_CHECK(cudaMallocAsync(&query_morton_codes,
                             m_query_count * sizeof(uint64_t), m_build_stream));
  SceneBounds query_bounds;
  initializeMortonCodes<Vec3>(queries_v3, query_morton_codes, m_query_count,
                              m_build_stream, &query_bounds);
  m_norm = SceneNormalization::from_scene_bounds(query_bounds);

  // sort by morton codes; this rewrites m_to_internal as the permutation
  thrust::sequence(build_stream_policy, m_to_internal,
                   m_to_internal + m_query_count);
  thrust::stable_sort_by_key(build_stream_policy, query_morton_codes,
                             query_morton_codes + m_query_count, m_to_internal);

  // Gather sorted (normalized) queries and D-bucket grad outputs.
  gather_queries_and_payload_grad_outputs_soa<D>(
      queries, grad_output, m_to_internal, m_sorted_queries,
      m_sorted_grad_outputs_bucket, m_query_count, m_payload_dim_user,
      m_build_stream, m_norm);

  // ---------- Leaf morton codes ----------
  auto morton_leaf_stride_idx = thrust::make_transform_iterator(
      thrust::make_counting_iterator<uint64_t>(0),
      [] __host__ __device__(uint32_t i) -> uint64_t { return i * LEAF_SIZE; });
  auto morton_leaf_stride = thrust::make_permutation_iterator(
      query_morton_codes, morton_leaf_stride_idx);
  uint64_t *leaf_morton_codes;
  CUDA_CHECK(cudaMallocAsync(&leaf_morton_codes,
                             m_query_count * sizeof(uint64_t), m_build_stream));
  thrust::copy(build_stream_policy, morton_leaf_stride,
               morton_leaf_stride + leaf_count, leaf_morton_codes);
  CUDA_CHECK(cudaFreeAsync(query_morton_codes, m_build_stream));

  // ---------- Binary radix tree ----------
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

  // ---------- Leaf coefficients (D-channel backward payload) ----------
  CUDA_CHECK(
      cudaMallocAsync(&m_leaf_coefficients,
                      leaf_count * sizeof(BackwardPayloadTaylorCoefficients<D>),
                      m_build_stream));

  float *atomic_weights;
  CUDA_CHECK(cudaMallocAsync(&atomic_weights, (leaf_count - 1) * sizeof(float),
                             m_build_stream));
  thrust::fill_n(build_stream_policy, atomic_weights, leaf_count - 1, 0.F);

  populate_binary_tree_aabb_and_leaf_coefficients_backward_payload<D>(
      m_sorted_queries, m_sorted_grad_outputs_bucket, m_leaf_coefficients,
      leaf_count, binary_nodes, m_binary_aabbs, binary_parents, atomic_weights,
      m_query_count, m_build_stream);

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

  // ---------- Max distances ----------
  uint32_t bvh8_node_count;
  CUDA_CHECK(cudaMemcpyAsync(&bvh8_node_count, m_bvh8_node_count,
                             sizeof(uint32_t), cudaMemcpyDeviceToHost,
                             m_build_stream));
  float *tmp_max_distances;
  CUDA_CHECK(cudaMallocAsync(&tmp_max_distances,
                             bvh8_node_count * sizeof(float), m_build_stream));
  thrust::fill_n(build_stream_policy, tmp_max_distances, bvh8_node_count, 0.F);
  compute_max_distances<Vec3>(m_bvh8_nodes, m_sorted_queries, bvh8_leaf_parents,
                              bvh8_internal_parent_map, tmp_max_distances,
                              static_cast<uint32_t>(m_query_count),
                              bvh8_node_count, m_build_stream);

  CUDA_CHECK(cudaFreeAsync(tmp_max_distances, m_build_stream));

  // ---------- m2m aggregation ----------
  uint32_t *atomic_counters;
  CUDA_CHECK(cudaMallocAsync(
      &atomic_counters, (leaf_count - 1) * sizeof(uint32_t), m_build_stream));
  thrust::fill_n(build_stream_policy, atomic_counters, leaf_count - 1, 0);

  CUDA_CHECK(cudaMallocAsync(&m_taylor_coefficients,
                             bvh8_node_count *
                                 sizeof(BackwardPayloadTaylorCoefficients<D>),
                             m_build_stream));

  compute_internal_tailor_coefficients_m2m_backward_payload<D>(
      m_bvh8_nodes, bvh8_internal_parent_map, m_binary_aabbs + leaf_count - 1,
      m_leaf_coefficients, bvh8_leaf_parents, m_bvh8_leaf_pointers,
      m_taylor_coefficients, bvh8_nodes_child_count, leaf_count,
      atomic_counters, m_build_stream);

  CUDA_CHECK(cudaFreeAsync(atomic_counters, m_build_stream));
  CUDA_CHECK(cudaFreeAsync(bvh8_internal_parent_map, m_build_stream));
  CUDA_CHECK(cudaFreeAsync(bvh8_leaf_parents, m_build_stream));
  CUDA_CHECK(cudaFreeAsync(bvh8_nodes_child_count, m_build_stream));

  CUDA_CHECK(
      cudaEventRecord(m_tree_construction_finished_event, m_build_stream));
}

// ===========================================================================
// compute — PointNormal
// ===========================================================================
template <IsGeometry Geometry, uint32_t D>
auto PayloadGradientBackend<Geometry, D>::compute(
    const Vec3 *points, const Vec3 *scaled_normals, const float *payload,
    size_t geometry_count, float *gradients, float *epsilon_gradient,
    float beta, float epsilon, uint64_t stream, bool use_kahan) const -> void
  requires std::is_same_v<Geometry, PointNormal>
{

  ScopedCudaDevice device_scope{m_device};
  if (m_query_count == 0 || geometry_count == 0) {
    return;
  }

  cudaStream_t compute_stream = reinterpret_cast<cudaStream_t>(stream);
  auto compute_stream_policy = thrust::cuda::par.on(compute_stream);

  // ---------- Morton sort of geometry ----------
  uint64_t *geometry_morton_codes;
  CUDA_CHECK(cudaMallocAsync(&geometry_morton_codes,
                             geometry_count * sizeof(uint64_t),
                             compute_stream));
  uint32_t *geometry_to_internal;
  CUDA_CHECK(cudaMallocAsync(&geometry_to_internal,
                             geometry_count * sizeof(uint32_t),
                             compute_stream));
  SceneBounds scene_bounds;
  initializeMortonCodes(points, geometry_morton_codes, geometry_count,
                        compute_stream, &scene_bounds);
  float max_dim = SceneNormalization::effective_extent(scene_bounds);

  thrust::sequence(compute_stream_policy, geometry_to_internal,
                   geometry_to_internal + geometry_count);
  thrust::stable_sort_by_key(compute_stream_policy, geometry_morton_codes,
                             geometry_morton_codes + geometry_count,
                             geometry_to_internal);
  CUDA_CHECK(cudaFreeAsync(geometry_morton_codes, compute_stream));

  // ---------- Sort payload by Morton order ----------
  float *sorted_payload;
  CUDA_CHECK(cudaMallocAsync(
      &sorted_payload, geometry_count * D * sizeof(float), compute_stream));
  gather_payload<D>(payload, geometry_to_internal, sorted_payload,
                    geometry_count, m_payload_dim_user, compute_stream);
  // TODO Why though? Should we really do this - we don't do it with the
  // geometry. It is only read once in the traversal kerenel, so even a SoA view
  // wouldn't help - and we don't even use that. We just do the padding.
  // Task: Check the kernel - what would change if payload would not be aligned ot D?

  // ---------- Wait for tree ----------
  CUDA_CHECK(
      cudaStreamWaitEvent(compute_stream, m_tree_construction_finished_event));

  if (beta < 0.F) {
    beta = GeometryTraits<PointNormal>::default_gradient_beta;
  }
  epsilon = std::max(0.F, epsilon);
  float scene_scale = max_dim * m_norm.scale;

  uint32_t leaf_count = (m_query_count + LEAF_SIZE - 1) / LEAF_SIZE;
  uint32_t *global_counter;
  CUDA_CHECK(
      cudaMallocAsync(&global_counter, sizeof(uint32_t), compute_stream));

  float *epsilon_gradients;
  CUDA_CHECK(cudaMallocAsync(&epsilon_gradients, geometry_count * sizeof(float),
                             compute_stream));

  ComputePayloadGradientsPointNormalParams<D> params{
      .points = points,
      .normals = scaled_normals,
      .sort_indirections = geometry_to_internal,
      .bvh8_nodes = m_bvh8_nodes,
      .bvh8_leaf_pointers = m_bvh8_leaf_pointers,
      .leaf_coefficients = m_leaf_coefficients,
      .node_coefficients = m_taylor_coefficients,
      .leaf_aabbs = m_binary_aabbs + leaf_count - 1,
      .sorted_queries =
          SoAView<Vec3>{.base_ptr = m_sorted_queries, .stride = m_query_count},
      .sorted_grad_outputs = m_sorted_grad_outputs_bucket,
      .sorted_payload = sorted_payload,
      .query_count = static_cast<uint32_t>(m_query_count),
      .geometry_count = static_cast<uint32_t>(geometry_count),
      .payload_dim_user = m_payload_dim_user,
      .gradients = gradients,
      .epsilon_gradients = epsilon_gradients,
      .global_device_counter = global_counter,
      .beta = beta,
      .epsilon = epsilon,
      .scene_scale = scene_scale,
      .norm = m_norm};

  // NOTE: needs `compute_payload_point_normal_gradients<D>` dispatcher.
  compute_payload_point_normal_gradients<D>(params, m_device, compute_stream,
                                            use_kahan);

  CUDA_CHECK(cudaFreeAsync(sorted_payload, compute_stream));
  CUDA_CHECK(cudaFreeAsync(geometry_to_internal, compute_stream));
  CUDA_CHECK(cudaFreeAsync(global_counter, compute_stream));

  reduce_async(epsilon_gradients, epsilon_gradient, geometry_count,
               compute_stream);
  CUDA_CHECK(cudaFreeAsync(epsilon_gradients, compute_stream));
}

// ===========================================================================
// compute — Triangle soup
// ===========================================================================
template <IsGeometry Geometry, uint32_t D>
auto PayloadGradientBackend<Geometry, D>::compute(
    const Triangle *triangles, const float *payload, size_t geometry_count,
    float *gradients, float *epsilon_gradient, float beta, float epsilon,
    uint64_t stream, bool use_kahan) const -> void
  requires std::is_same_v<Geometry, Triangle>
{

  ScopedCudaDevice device_scope{m_device};
  if (m_query_count == 0 || geometry_count == 0) {
    return;
  }

  cudaStream_t compute_stream = reinterpret_cast<cudaStream_t>(stream);
  auto compute_stream_policy = thrust::cuda::par.on(compute_stream);

  uint64_t *geometry_morton_codes;
  CUDA_CHECK(cudaMallocAsync(&geometry_morton_codes,
                             geometry_count * sizeof(uint64_t),
                             compute_stream));
  uint32_t *geometry_to_internal;
  CUDA_CHECK(cudaMallocAsync(&geometry_to_internal,
                             geometry_count * sizeof(uint32_t),
                             compute_stream));
  SceneBounds scene_bounds;
  initializeMortonCodes(triangles, geometry_morton_codes, geometry_count,
                        compute_stream, &scene_bounds);
  float max_dim = SceneNormalization::effective_extent(scene_bounds);

  thrust::sequence(compute_stream_policy, geometry_to_internal,
                   geometry_to_internal + geometry_count);
  thrust::stable_sort_by_key(compute_stream_policy, geometry_morton_codes,
                             geometry_morton_codes + geometry_count,
                             geometry_to_internal);
  CUDA_CHECK(cudaFreeAsync(geometry_morton_codes, compute_stream));

  float *sorted_payload;
  CUDA_CHECK(cudaMallocAsync(
      &sorted_payload, geometry_count * D * sizeof(float), compute_stream));
  gather_payload<D>(payload, geometry_to_internal, sorted_payload,
                    geometry_count, m_payload_dim_user, compute_stream);

  CUDA_CHECK(
      cudaStreamWaitEvent(compute_stream, m_tree_construction_finished_event));

  if (beta < 0.F) {
    beta = GeometryTraits<Triangle>::default_gradient_beta;
  }
  epsilon = std::max(0.F, epsilon);
  float scene_scale = max_dim * m_norm.scale;

  uint32_t leaf_count = (m_query_count + LEAF_SIZE - 1) / LEAF_SIZE;
  uint32_t *global_counter;
  CUDA_CHECK(
      cudaMallocAsync(&global_counter, sizeof(uint32_t), compute_stream));

  float *epsilon_gradients;
  CUDA_CHECK(cudaMallocAsync(&epsilon_gradients, geometry_count * sizeof(float),
                             compute_stream));

  ComputePayloadGradientsTriangleParams<D> params{
      .triangles = triangles,
      .sort_indirections = geometry_to_internal,
      .bvh8_nodes = m_bvh8_nodes,
      .bvh8_leaf_pointers = m_bvh8_leaf_pointers,
      .leaf_coefficients = m_leaf_coefficients,
      .node_coefficients = m_taylor_coefficients,
      .leaf_aabbs = m_binary_aabbs + leaf_count - 1,
      .sorted_queries =
          SoAView<Vec3>{.base_ptr = m_sorted_queries, .stride = m_query_count},
      .sorted_grad_outputs = m_sorted_grad_outputs_bucket,
      .sorted_payload = sorted_payload,
      .query_count = static_cast<uint32_t>(m_query_count),
      .geometry_count = static_cast<uint32_t>(geometry_count),
      .payload_dim_user = m_payload_dim_user,
      .gradients = gradients,
      .epsilon_gradients = epsilon_gradients,
      .global_device_counter = global_counter,
      .beta = beta,
      .epsilon = epsilon,
      .scene_scale = scene_scale,
      .norm = m_norm};

  // NOTE: needs `compute_payload_triangle_gradients<D>` dispatcher.
  compute_payload_triangle_gradients<D>(params, m_device, compute_stream,
                                        use_kahan);

  CUDA_CHECK(cudaFreeAsync(sorted_payload, compute_stream));
  CUDA_CHECK(cudaFreeAsync(geometry_to_internal, compute_stream));
  CUDA_CHECK(cudaFreeAsync(global_counter, compute_stream));

  reduce_async(epsilon_gradients, epsilon_gradient, geometry_count,
               compute_stream);
  CUDA_CHECK(cudaFreeAsync(epsilon_gradients, compute_stream));
}

// ===========================================================================
// compute — Mesh (shared vertices + indexed triangles)
// ===========================================================================
template <IsGeometry Geometry, uint32_t D>
auto PayloadGradientBackend<Geometry, D>::compute(
    const float *vertices, const uint32_t *triangle_indices,
    const float *payload, size_t vertex_count, size_t triangle_count,
    float *vertex_gradients, float *epsilon_gradient, float beta, float epsilon,
    uint64_t stream, bool use_kahan) const -> void
  requires std::is_same_v<Geometry, Triangle>
{

  if (m_query_count == 0 || triangle_count == 0) {
    return;
  }
  ScopedCudaDevice device_scope{m_device};
  cudaStream_t compute_stream = reinterpret_cast<cudaStream_t>(stream);

  // Gather triangles from (vertices, indices).
  float *triangles;
  CUDA_CHECK(cudaMallocAsync(&triangles, triangle_count * sizeof(Triangle),
                             compute_stream));
  gather_triangles(vertices, triangle_indices,
                   static_cast<uint32_t>(triangle_count), triangles,
                   compute_stream);

  // Per-triangle gradients scratch buffer.
  float *triangle_gradients;
  CUDA_CHECK(cudaMallocAsync(
      &triangle_gradients, triangle_count * sizeof(Triangle), compute_stream));

  // Delegate to the triangle-soup overload.
  this->compute(reinterpret_cast<const Triangle *>(triangles), payload,
                triangle_count, triangle_gradients, epsilon_gradient, beta,
                epsilon, stream, use_kahan);

  CUDA_CHECK(cudaFreeAsync(triangles, compute_stream));

  // Accumulate per-triangle -> per-vertex.
  CUDA_CHECK(cudaMemsetAsync(vertex_gradients, 0,
                             vertex_count * 3 * sizeof(float), compute_stream));
  accumulate_vertex_gradients(triangle_gradients, triangle_indices,
                              static_cast<uint32_t>(triangle_count),
                              vertex_gradients, compute_stream);

  CUDA_CHECK(cudaFreeAsync(triangle_gradients, compute_stream));
}

// ===========================================================================
// dump — full query BVH + D-channel coefficients
//
// Displays channel d=0 verbatim (matching the winding-number dump layout)
// plus, for D > 1, a compact summary of the D-channel coefficients (L2 norm
// summed across channels) so that "did the adjoints make it into the tree?"
// can be checked at a glance.
// ===========================================================================
template <IsGeometry Geometry, uint32_t D>
auto PayloadGradientBackend<Geometry, D>::dump() const -> std::string {
  if (m_query_count == 0) {
    return "digraph BVH8 {\n}\n";
  }

  std::string result = "digraph BVH8 {\n";
  result += "  node [fontname=\"Arial\", fontsize=10];\n";
  result += "  rankdir=TB;\n\n";
  result += std::format("  // Payload BACKWARD: D = {}, query_count = {}\n\n",
                        D, m_query_count);

  auto sym_to_mat3 = [](const SymMat3x3 &s) -> Mat3x3 {
    return Mat3x3::from_sym(s);
  };

  auto coeff_norm_sq =
      [&](const BackwardPayloadTaylorCoefficients<D> &c) -> float {
    float sum = 0.F;
    for (uint32_t d = 0; d < D; ++d) {
      sum += c.zero_order[d] * c.zero_order[d];
      sum += c.first_order[d].length2();
      const auto &s = c.second_order[d];
      for (int k = 0; k < 6; ++k)
        sum += s.data[k] * s.data[k];
    }
    return sum;
  };

  // --- Single-leaf case ---
  if (m_query_count <= LEAF_SIZE) {
    std::vector<AABB> leaf_aabbs(1);
    CUDA_CHECK(cudaMemcpy(leaf_aabbs.data(), m_binary_aabbs, sizeof(AABB),
                          cudaMemcpyDeviceToHost));

    std::vector<Vec3> queries(m_query_count);
    CUDA_CHECK(cudaMemcpy(queries.data(), m_sorted_queries,
                          m_query_count * sizeof(Vec3),
                          cudaMemcpyDeviceToHost));
    SoAView<Vec3> query_view{.base_ptr =
                                 reinterpret_cast<float *>(queries.data()),
                             .stride = m_query_count};

    std::vector<BackwardPayloadTaylorCoefficients<D>> leaf_coefficients(1);
    CUDA_CHECK(cudaMemcpy(leaf_coefficients.data(), m_taylor_coefficients,
                          sizeof(BackwardPayloadTaylorCoefficients<D>),
                          cudaMemcpyDeviceToHost));

    const AABB &leaf_aabb = leaf_aabbs[0];
    const auto &coeffs = leaf_coefficients[0];
    const Vec3 leaf_com = leaf_aabb.center_of_mass;

    result += "  L0 [shape=none, label=<\n";
    result += "    <TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\" "
              "CELLPADDING=\"4\" BGCOLOR=\"#eaffea\">\n";
    result += std::format(
        "      <TR><TD COLSPAN=\"4\" BGCOLOR=\"#4CAF50\"><B><FONT "
        "COLOR=\"white\">LEAF 0 (Root Leaf, payload backward D={})</FONT></B>"
        "</TD></TR>\n",
        D);
    result += std::format(
        "      <TR><TD COLSPAN=\"4\" BGCOLOR=\"#c8e6c9\"><I>CoM: ({:.4f}, "
        "{:.4f}, {:.4f})</I></TD></TR>\n",
        leaf_com.x, leaf_com.y, leaf_com.z);
    result += std::format(
        "      <TR><TD COLSPAN=\"4\"><FONT POINT-SIZE=\"9\">AABB Min: "
        "({:.3f}, {:.3f}, {:.3f})<BR/>AABB Max: ({:.3f}, {:.3f}, "
        "{:.3f})</FONT></TD></TR>\n",
        __half2float(leaf_aabb.min.x), __half2float(leaf_aabb.min.y),
        __half2float(leaf_aabb.min.z), __half2float(leaf_aabb.max.x),
        __half2float(leaf_aabb.max.y), __half2float(leaf_aabb.max.z));

    if constexpr (D > 1) {
      result +=
          std::format("      <TR><TD COLSPAN=\"4\" BGCOLOR=\"#e8f5e9\"><FONT "
                      "POINT-SIZE=\"9\"><I>||coeff||² across all D = {:.6f}</I>"
                      "</FONT></TD></TR>\n",
                      coeff_norm_sq(coeffs));
    }

    // Channel d=0 detail
    result +=
        std::format("      <TR><TD BGCOLOR=\"#c8e6c9\"><B>Zero (d=0)</B></TD>"
                    "<TD>{:.4f}</TD><TD>{:.4f}</TD><TD>{:.4f}</TD></TR>\n",
                    coeffs.zero_order[0], D > 0 ? 0.F : 0.F,
                    D > 0 ? 0.F : 0.F); // placeholder columns
    result +=
        std::format("      <TR><TD BGCOLOR=\"#c8e6c9\"><B>First (d=0)</B></TD>"
                    "<TD>{:.4f}</TD><TD>{:.4f}</TD><TD>{:.4f}</TD></TR>\n",
                    coeffs.first_order[0].x, coeffs.first_order[0].y,
                    coeffs.first_order[0].z);

    const Mat3x3 m0 = sym_to_mat3(coeffs.second_order[0]);
    result += "      <TR><TD ROWSPAN=\"3\" BGCOLOR=\"#c8e6c9\"><B>Second "
              "(d=0)</B></TD>\n";
    for (int r = 0; r < 3; ++r) {
      if (r > 0)
        result += "      <TR>\n";
      result += std::format(
          "        <TD>{:.4f}</TD><TD>{:.4f}</TD><TD>{:.4f}</TD></TR>\n",
          m0.data[r * 3 + 0], m0.data[r * 3 + 1], m0.data[r * 3 + 2]);
    }

    result += std::format(
        "      <TR><TD COLSPAN=\"4\" BGCOLOR=\"#c8e6c9\"><I>Queries (Max "
        "{} per leaf)</I></TD></TR>\n",
        LEAF_SIZE);
    for (size_t q_id = 0; q_id < m_query_count; ++q_id) {
      const Vec3 &q = Vec3::load(query_view, q_id, m_query_count);
      result += std::format(
          "      <TR><TD>Q{}</TD><TD COLSPAN=\"3\" ALIGN=\"LEFT\"><FONT "
          "POINT-SIZE=\"9\">{}</FONT></TD></TR>\n",
          q_id, q.dump());
    }
    result += "    </TABLE>>];\n}\n";
    return result;
  }

  // --- Standard BVH8 multi-node traversal ---
  uint32_t node_count = 0;
  CUDA_CHECK(cudaMemcpy(&node_count, m_bvh8_node_count, sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));

  std::vector<BVH8Node> bvh8_nodes(node_count);
  CUDA_CHECK(cudaMemcpy(bvh8_nodes.data(), m_bvh8_nodes,
                        sizeof(BVH8Node) * node_count, cudaMemcpyDeviceToHost));

  uint32_t leaf_count = (m_query_count + LEAF_SIZE - 1) / LEAF_SIZE;
  std::vector<AABB> leaf_aabbs(leaf_count);
  CUDA_CHECK(cudaMemcpy(leaf_aabbs.data(), m_binary_aabbs + leaf_count - 1,
                        sizeof(AABB) * leaf_count, cudaMemcpyDeviceToHost));

  std::vector<Vec3> queries(m_query_count);
  CUDA_CHECK(cudaMemcpy(queries.data(), m_sorted_queries,
                        m_query_count * sizeof(Vec3), cudaMemcpyDeviceToHost));
  SoAView<Vec3> query_view{.base_ptr =
                               reinterpret_cast<float *>(queries.data()),
                           .stride = m_query_count};

  std::vector<LeafPointers> leaf_pointers(node_count);
  CUDA_CHECK(cudaMemcpy(leaf_pointers.data(), m_bvh8_leaf_pointers,
                        node_count * sizeof(LeafPointers),
                        cudaMemcpyDeviceToHost));

  std::vector<BackwardPayloadTaylorCoefficients<D>> node_coefficients(
      node_count);
  CUDA_CHECK(
      cudaMemcpy(node_coefficients.data(), m_taylor_coefficients,
                 sizeof(BackwardPayloadTaylorCoefficients<D>) * node_count,
                 cudaMemcpyDeviceToHost));

  // BFS traversal (matches winding dump layout).
  std::queue<uint32_t> queue;
  queue.push(0);
  int empty_counter = 0;

  while (!queue.empty()) {
    uint32_t current_id = queue.front();
    queue.pop();

    const BVH8Node &current_node = bvh8_nodes[current_id];
    const auto &coeffs = node_coefficients[current_id];
    AABB aabb = current_node.getAABB();
    Vec3 node_com = aabb.center_of_mass;

    result += std::format("  N{} [shape=none, label=<\n", current_id);
    result += "    <TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\" "
              "CELLPADDING=\"4\" BGCOLOR=\"#f0faff\">\n";
    result +=
        std::format("      <TR><TD COLSPAN=\"4\" BGCOLOR=\"#2196F3\"><B><FONT "
                    "COLOR=\"white\">NODE {} (payload backward "
                    "D={})</FONT></B></TD></TR>\n",
                    current_id, D);
    result += std::format(
        "      <TR><TD COLSPAN=\"4\" BGCOLOR=\"#bbdefb\"><I>CoM: ({:.4f}, "
        "{:.4f}, {:.4f})</I></TD></TR>\n",
        node_com.x, node_com.y, node_com.z);
    result += std::format(
        "      <TR><TD COLSPAN=\"4\"><FONT POINT-SIZE=\"9\">AABB Min: "
        "({:.3f}, {:.3f}, {:.3f})<BR/>AABB Max: ({:.3f}, {:.3f}, "
        "{:.3f})</FONT></TD></TR>\n",
        __half2float(aabb.min.x), __half2float(aabb.min.y),
        __half2float(aabb.min.z), __half2float(aabb.max.x),
        __half2float(aabb.max.y), __half2float(aabb.max.z));

    if constexpr (D > 1) {
      result +=
          std::format("      <TR><TD COLSPAN=\"4\" BGCOLOR=\"#e3f2fd\"><FONT "
                      "POINT-SIZE=\"9\"><I>||coeff||² across all D = {:.6f}</I>"
                      "</FONT></TD></TR>\n",
                      coeff_norm_sq(coeffs));
    }

    result +=
        std::format("      <TR><TD BGCOLOR=\"#bbdefb\"><B>Zero (d=0)</B></TD>"
                    "<TD>{:.4f}</TD><TD COLSPAN=\"2\"></TD></TR>\n",
                    coeffs.zero_order[0]);
    result +=
        std::format("      <TR><TD BGCOLOR=\"#bbdefb\"><B>First (d=0)</B></TD>"
                    "<TD>{:.4f}</TD><TD>{:.4f}</TD><TD>{:.4f}</TD></TR>\n",
                    coeffs.first_order[0].x, coeffs.first_order[0].y,
                    coeffs.first_order[0].z);

    const Mat3x3 m0 = sym_to_mat3(coeffs.second_order[0]);
    result += "      <TR><TD ROWSPAN=\"3\" BGCOLOR=\"#bbdefb\"><B>Second "
              "(d=0)</B></TD>\n";
    for (int r = 0; r < 3; ++r) {
      if (r > 0)
        result += "      <TR>\n";
      result += std::format(
          "        <TD>{:.4f}</TD><TD>{:.4f}</TD><TD>{:.4f}</TD></TR>\n",
          m0.data[r * 3 + 0], m0.data[r * 3 + 1], m0.data[r * 3 + 2]);
    }
    result += "    </TABLE>>];\n";

    uint32_t child_base = current_node.child_base;
    uint32_t child_offset = 0;
    LeafPointers current_leaf_pointers = leaf_pointers[current_id];

    for (size_t child_id = 0; child_id < 8; ++child_id) {
      ChildType child_type = current_node.getChildMeta(child_id);
      switch (child_type) {
      case ChildType::INTERNAL: {
        uint32_t next_idx = child_offset++ + child_base;
        queue.push(next_idx);
        result += std::format("  N{} -> N{} [label=\"{}\", weight=3];\n",
                              current_id, next_idx, child_id);
        break;
      }
      case ChildType::LEAF: {
        uint32_t l_id = current_leaf_pointers.indices[child_id];
        AABB leaf_aabb = leaf_aabbs[l_id];
        Vec3 leaf_com = leaf_aabb.center_of_mass;

        // Leaf coefficients are not stored; display the leaf AABB and queries.
        result += std::format("  L{} [shape=none, label=<\n", l_id);
        result += "    <TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\" "
                  "CELLPADDING=\"4\" BGCOLOR=\"#eaffea\">\n";
        result += std::format(
            "      <TR><TD COLSPAN=\"4\" BGCOLOR=\"#4CAF50\"><B><FONT "
            "COLOR=\"white\">LEAF {} (payload backward D={})</FONT></B>"
            "</TD></TR>\n",
            l_id, D);
        result += std::format(
            "      <TR><TD COLSPAN=\"4\" BGCOLOR=\"#c8e6c9\"><I>CoM: ({:.4f}, "
            "{:.4f}, {:.4f})</I></TD></TR>\n",
            leaf_com.x, leaf_com.y, leaf_com.z);
        result += std::format(
            "      <TR><TD COLSPAN=\"4\"><FONT POINT-SIZE=\"9\" "
            "COLOR=\"#333333\">AABB Min: ({:.2f}, {:.2f}, {:.2f}) Max: "
            "({:.2f}, {:.2f}, {:.2f})</FONT></TD></TR>\n",
            __half2float(leaf_aabb.min.x), __half2float(leaf_aabb.min.y),
            __half2float(leaf_aabb.min.z), __half2float(leaf_aabb.max.x),
            __half2float(leaf_aabb.max.y), __half2float(leaf_aabb.max.z));

        result += std::format(
            "      <TR><TD COLSPAN=\"4\" BGCOLOR=\"#c8e6c9\"><I>Queries (Max "
            "{} per leaf)</I></TD></TR>\n",
            LEAF_SIZE);
        size_t q_off = l_id * LEAF_SIZE;
        for (size_t q_id = 0; q_id < LEAF_SIZE; ++q_id) {
          size_t global_idx = q_off + q_id;
          if (global_idx >= m_query_count)
            break;
          const Vec3 &q = Vec3::load(query_view, global_idx, m_query_count);
          result += std::format(
              "      <TR><TD>Q{}</TD><TD COLSPAN=\"3\" ALIGN=\"LEFT\"><FONT "
              "POINT-SIZE=\"8\">{}</FONT></TD></TR>\n",
              q_id, q.dump());
        }

        result += "    </TABLE>>];\n";
        result += std::format(
            "  N{} -> L{} [label=\"{}\", color=\"#4CAF50\", penwidth=2];\n",
            current_id, l_id, child_id);
        break;
      }
      case ChildType::EMPTY: {
        int e_id = empty_counter++;
        result +=
            std::format("  E{} [label=\"\", shape=point, color=gray];\n", e_id);
        result += std::format(
            "  N{} -> E{} [style=dotted, color=gray, arrowhead=none];\n",
            current_id, e_id);
        break;
      }
      }
    }
  }

  result += "}\n";
  return result;
}

// ===========================================================================
// Explicit instantiations
// ===========================================================================
template class PayloadGradientBackend<PointNormal, 1>;
template class PayloadGradientBackend<PointNormal, 2>;
template class PayloadGradientBackend<PointNormal, 3>;
template class PayloadGradientBackend<PointNormal, 4>;
template class PayloadGradientBackend<PointNormal, 8>;
template class PayloadGradientBackend<PointNormal, 16>;
template class PayloadGradientBackend<PointNormal, 32>;
template class PayloadGradientBackend<PointNormal, 64>;
template class PayloadGradientBackend<PointNormal, 128>;

template class PayloadGradientBackend<Triangle, 1>;
template class PayloadGradientBackend<Triangle, 2>;
template class PayloadGradientBackend<Triangle, 3>;
template class PayloadGradientBackend<Triangle, 4>;
template class PayloadGradientBackend<Triangle, 8>;
template class PayloadGradientBackend<Triangle, 16>;
template class PayloadGradientBackend<Triangle, 32>;
template class PayloadGradientBackend<Triangle, 64>;
template class PayloadGradientBackend<Triangle, 128>;
