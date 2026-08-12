#include "aabb.h"
#include "binary_node.h"
#include "bvh8.h"
#include "geometry.h"
#include "gradient_backend.h"
#include "kernels/binary2bvh8.cuh"
#include "kernels/build_binary_tree.cuh"
#include "kernels/bvh8_m2m.cuh"
#include "kernels/common.cuh"
#include "kernels/mesh.cuh"
#include "kernels/traversal.cuh"
#include "soa.h"
#include "taylor_coefficients.h"
#include "utils.h"
#include "vec3.h"
#include <cstddef>
#include <cstdint>
#include <cuda_device_runtime_api.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_atomic_functions.h>
#include <driver_types.h>
#include <format>
#include <queue>
#include <string>
#include <thrust/copy.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/detail/vector_base.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/detail/execution_policy.h>
#include <thrust/transform.h>
#include <vector>

#define LEAF_SIZE 32

GradientBackend::GradientBackend(size_t query_count, int device_id)
    : m_device{device_id}, m_query_count{query_count} {
  uint32_t leaf_count = (m_query_count + LEAF_SIZE - 1) / LEAF_SIZE;

  uint32_t max_bvh8_nodes = leaf_count - 1; // TODO worst case scenario!!

  CUDA_CHECK(cudaEventCreate(&m_tree_construction_finished_event));

  // Allocate Persistent arrays
  CUDA_CHECK(cudaStreamCreate(&m_build_stream));
  CUDA_CHECK(cudaMallocAsync(&m_to_internal, query_count * sizeof(uint32_t),
                             m_build_stream));
  CUDA_CHECK(cudaMallocAsync(&m_sorted_queries, query_count * sizeof(Vec3),
                             m_build_stream));
  CUDA_CHECK(cudaMallocAsync(&m_sorted_grad_outputs,
                             query_count * sizeof(float), m_build_stream));

  CUDA_CHECK(cudaMallocAsync(
      &m_binary_aabbs, (2 * leaf_count - 1) * sizeof(AABB), m_build_stream));
  CUDA_CHECK(cudaMallocAsync(&m_bvh8_node_count, 1 * sizeof(uint32_t),
                             m_build_stream));
  CUDA_CHECK(cudaMallocAsync(&m_bvh8_nodes, max_bvh8_nodes * sizeof(BVH8Node),
                             m_build_stream));
  CUDA_CHECK(cudaMallocAsync(
      &m_taylor_coefficients,
      max_bvh8_nodes * sizeof(BackwardTaylorCoefficients), m_build_stream));
  CUDA_CHECK(cudaMallocAsync(&m_bvh8_leaf_pointers,
                             max_bvh8_nodes * sizeof(LeafPointers),
                             m_build_stream));
}

GradientBackend::~GradientBackend() {
  CUDA_CHECK(cudaStreamSynchronize(m_build_stream));

  CUDA_CHECK(cudaEventDestroy(m_tree_construction_finished_event));

  if (m_to_internal) {
    CUDA_CHECK(cudaFreeAsync(m_to_internal, m_build_stream));
  }
  if (m_sorted_queries) {
    CUDA_CHECK(cudaFreeAsync(m_sorted_queries, m_build_stream));
  }
  if (m_binary_aabbs) {
    CUDA_CHECK(cudaFreeAsync(m_binary_aabbs, m_build_stream));
  }
  if (m_bvh8_node_count) {
    CUDA_CHECK(cudaFreeAsync(m_bvh8_node_count, m_build_stream));
  }
  if (m_bvh8_nodes) {
    CUDA_CHECK(cudaFreeAsync(m_bvh8_nodes, m_build_stream));
  }
  if (m_taylor_coefficients) {
    CUDA_CHECK(cudaFreeAsync(m_taylor_coefficients, m_build_stream));
  }
  if (m_bvh8_leaf_pointers) {
    CUDA_CHECK(cudaFreeAsync(m_bvh8_leaf_pointers, m_build_stream));
  }
  CUDA_CHECK(cudaStreamSynchronize(m_build_stream));

  CUDA_CHECK(cudaStreamDestroy(m_build_stream));
}

// Build BVH8
void GradientBackend::init(const float *queries, const float *grad_outputs) {
  const auto *queries_v3 = reinterpret_cast<const Vec3 *>(queries);
  auto build_stream_policy = thrust::cuda::par.on(m_build_stream);

  uint32_t leaf_count = (m_query_count + LEAF_SIZE - 1) / LEAF_SIZE;
  uint32_t max_bvh8_nodes = leaf_count - 1; // TODO worst case scenario!!

  uint64_t *query_morton_codes;
  CUDA_CHECK(cudaMallocAsync(&query_morton_codes,
                             m_query_count * sizeof(uint64_t), m_build_stream));
  initializeMortonCodes<Vec3>(queries_v3, query_morton_codes, m_query_count,
                              m_build_stream);
  // sort by morton codes
  thrust::sequence(build_stream_policy, m_to_internal,
                   m_to_internal + m_query_count);
  // sorts both morton_codes and m_to_internal
  thrust::sort_by_key(build_stream_policy, query_morton_codes,
                      query_morton_codes + m_query_count, m_to_internal);

  gather_queries_and_grad_outputs_soa(queries, grad_outputs, m_to_internal,
                                      m_sorted_queries, m_sorted_grad_outputs,
                                      m_query_count, m_build_stream);

  auto morton_leaf_stride_idx = thrust::make_transform_iterator(
      thrust::make_counting_iterator<uint64_t>(0),
      [] __host__ __device__(uint32_t i) -> uint64_t { return i * LEAF_SIZE; });
  // thrust::make_strided_iterator<LEAF_SIZE>(query_morton_codes.begin());
  auto morton_leaf_stride = thrust::make_permutation_iterator(
      query_morton_codes, morton_leaf_stride_idx);
  uint64_t *leaf_morton_codes;
  CUDA_CHECK(cudaMallocAsync(&leaf_morton_codes,
                             m_query_count * sizeof(uint64_t), m_build_stream));
  thrust::copy(build_stream_policy, morton_leaf_stride,
               morton_leaf_stride + leaf_count, leaf_morton_codes);
  // build binary radix tree
  CUDA_CHECK(cudaFreeAsync(query_morton_codes, m_build_stream));
  // build binary radix tree
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

  // allocate leaf coefficients
  BackwardTaylorCoefficients *leaf_coefficients;
  CUDA_CHECK(cudaMallocAsync(&leaf_coefficients,
                             leaf_count * sizeof(BackwardTaylorCoefficients),
                             m_build_stream));
  // initialize the atomic weights to 0
  float *atomic_weights;

  CUDA_CHECK(cudaMallocAsync(&atomic_weights, (leaf_count - 1) * sizeof(float),
                             m_build_stream));
  thrust::fill_n(build_stream_policy, atomic_weights, leaf_count - 1, 0.F);
  populate_binary_tree_aabb_and_leaf_coefficients_backward(
      m_sorted_queries, m_sorted_grad_outputs, leaf_coefficients, leaf_count,
      binary_nodes, m_binary_aabbs, binary_parents, atomic_weights,
      m_query_count, m_build_stream);
  CUDA_CHECK(cudaFreeAsync(binary_parents, m_build_stream));
  CUDA_CHECK(cudaFreeAsync(atomic_weights, m_build_stream));

  // Convert binary LBVH tree into BVH8 tree
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
      cudaMallocAsync(&global_counter, 1 * sizeof(uint32_t), m_build_stream));
  CUDA_CHECK(cudaMallocAsync(&bvh8_leaf_parents, leaf_count * sizeof(uint32_t),
                             m_build_stream));
  CUDA_CHECK(cudaMallocAsync(&bvh8_nodes_child_count,
                             max_bvh8_nodes * sizeof(uint32_t),
                             m_build_stream));

  ConvertBinary2BVH8Params params{bvh8_work_queue_A,
                                  bvh8_work_queue_B,
                                  bvh8_internal_parent_map,
                                  global_counter,
                                  leaf_count,
                                  m_binary_aabbs,
                                  binary_nodes,
                                  bvh8_nodes_child_count,
                                  bvh8_leaf_parents,
                                  m_bvh8_nodes,
                                  m_bvh8_leaf_pointers,
                                  m_bvh8_node_count};
  convert_binary_tree_to_bvh8(params, m_device, m_build_stream);

  CUDA_CHECK(cudaFreeAsync(bvh8_work_queue_A, m_build_stream));
  CUDA_CHECK(cudaFreeAsync(bvh8_work_queue_B, m_build_stream));
  CUDA_CHECK(cudaFreeAsync(global_counter, m_build_stream));
  CUDA_CHECK(cudaFreeAsync(binary_nodes, m_build_stream));

  // Compute the max distances of queries from the center of mass for the nodes
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

  // populate BVH8 nodes with tailor coefficients using m2m
  // initialize atomic counters to 0
  uint32_t *atomic_counters;
  CUDA_CHECK(cudaMallocAsync(
      &atomic_counters, (leaf_count - 1) * sizeof(uint32_t), m_build_stream));
  thrust::fill_n(build_stream_policy, atomic_counters, leaf_count - 1, 0);
  compute_internal_tailor_coefficients_m2m_backward(
      m_bvh8_nodes, bvh8_internal_parent_map, m_binary_aabbs + leaf_count - 1,
      leaf_coefficients, bvh8_leaf_parents, m_bvh8_leaf_pointers,
      m_taylor_coefficients, bvh8_nodes_child_count, leaf_count,
      atomic_counters, m_build_stream);

  CUDA_CHECK(cudaFreeAsync(leaf_coefficients, m_build_stream));
  CUDA_CHECK(cudaFreeAsync(atomic_counters, m_build_stream));
  CUDA_CHECK(cudaFreeAsync(bvh8_internal_parent_map, m_build_stream));
  CUDA_CHECK(cudaFreeAsync(bvh8_leaf_parents, m_build_stream));
  CUDA_CHECK(cudaFreeAsync(bvh8_nodes_child_count, m_build_stream));

  CUDA_CHECK(
      cudaEventRecord(m_tree_construction_finished_event, m_build_stream));
}

auto GradientBackend::compute(const float *points, const float *scaled_normals,
                              size_t geometry_count, float beta, float epsilon,
                              uint64_t stream) -> CudaUniquePtr<float> {
  ScopedCudaDevice device_scope{m_device};
  // convert stream to cuda stream
  cudaStream_t compute_stream = reinterpret_cast<cudaStream_t>(stream);
  auto compute_stream_policy = thrust::cuda::par.on(compute_stream);

  const Vec3 *points_vec3 = reinterpret_cast<const Vec3 *>(points);
  const Vec3 *normals_vec3 = reinterpret_cast<const Vec3 *>(scaled_normals);

  // Allocate required buffer for result
  float *gradients; // result
  CUDA_CHECK(cudaMallocAsync(&gradients, geometry_count * sizeof(float) * 6,
                             compute_stream));

  // Sort geometry for cache coherence
  uint64_t *geometry_morton_codes;
  CUDA_CHECK(cudaMallocAsync(&geometry_morton_codes,
                             geometry_count * sizeof(uint64_t),
                             m_build_stream));
  uint32_t *geometry_to_internal;
  CUDA_CHECK(cudaMallocAsync(&geometry_to_internal,
                             geometry_count * sizeof(uint32_t),
                             compute_stream));
  initializeMortonCodes(points_vec3, geometry_morton_codes, geometry_count,
                        compute_stream);

  // sort by morton codes
  thrust::sequence(compute_stream_policy, geometry_to_internal,
                   geometry_to_internal + geometry_count);
  thrust::sort_by_key(compute_stream_policy, geometry_morton_codes,
                      geometry_morton_codes + geometry_count,
                      geometry_to_internal);

  // free morton code memory
  CUDA_CHECK(cudaFreeAsync(geometry_morton_codes, compute_stream));

  // make sure stream 0 has finished building the tree
  CUDA_CHECK(
      cudaStreamWaitEvent(compute_stream, m_tree_construction_finished_event));

  if (beta < 0.F) {
    beta = 2.0;
  }
  if (epsilon < 0.F) {
    epsilon = 1.F / 250.F;
  }

  uint32_t leaf_count = (m_query_count + LEAF_SIZE - 1) / LEAF_SIZE;
  uint32_t *global_counter;
  CUDA_CHECK(
      cudaMallocAsync(&global_counter, sizeof(uint32_t), compute_stream));

  ComputeGradientsPointNormalParams params{
      points_vec3,
      normals_vec3,
      geometry_to_internal,
      m_bvh8_nodes,
      m_bvh8_leaf_pointers,
      m_taylor_coefficients,
      m_binary_aabbs + leaf_count - 1,
      SoAView<Vec3>{m_sorted_queries, m_query_count},
      m_sorted_grad_outputs,
      (uint32_t)m_query_count,
      (uint32_t)geometry_count,
      gradients,
      global_counter,
      beta,
      epsilon};
  compute_point_normal_gradients(params, m_device, compute_stream);

  // free temporary memory
  CUDA_CHECK(cudaFreeAsync(geometry_to_internal, compute_stream));
  CUDA_CHECK(cudaFreeAsync(global_counter, compute_stream));

  CudaUniquePtr<float> result(
      gradients, CudaDeleter{reinterpret_cast<size_t>(compute_stream)});
  return result;
}

auto GradientBackend::compute(const float *triangles_float,
                              size_t geometry_count, float beta,
                              uint64_t stream) -> CudaUniquePtr<float> {
  ScopedCudaDevice device_scope{m_device};
  // convert stream to cuda stream
  cudaStream_t compute_stream = reinterpret_cast<cudaStream_t>(stream);
  auto compute_stream_policy = thrust::cuda::par.on(compute_stream);

  const auto *triangles = reinterpret_cast<const Triangle *>(triangles_float);

  // Allocate required buffer for result
  float *gradients; // result
  CUDA_CHECK(cudaMallocAsync(&gradients, geometry_count * sizeof(float) * 9,
                             compute_stream));

  // Sort geometry for cache coherence
  uint64_t *geometry_morton_codes;
  CUDA_CHECK(cudaMallocAsync(&geometry_morton_codes,
                             geometry_count * sizeof(uint64_t),
                             m_build_stream));
  uint32_t *geometry_to_internal;
  CUDA_CHECK(cudaMallocAsync(&geometry_to_internal,
                             geometry_count * sizeof(uint32_t),
                             compute_stream));
  initializeMortonCodes(triangles, geometry_morton_codes, geometry_count,
                        compute_stream);

  // sort by morton codes
  thrust::sequence(compute_stream_policy, geometry_to_internal,
                   geometry_to_internal + geometry_count);
  thrust::sort_by_key(compute_stream_policy, geometry_morton_codes,
                      geometry_morton_codes + geometry_count,
                      geometry_to_internal);

  // free morton code memory
  CUDA_CHECK(cudaFreeAsync(geometry_morton_codes, compute_stream));

  // make sure stream 0 has finished building the tree
  CUDA_CHECK(
      cudaStreamWaitEvent(compute_stream, m_tree_construction_finished_event));

  if (beta < 0.F) {
    beta = 2.3;
  }

  uint32_t leaf_count = (m_query_count + LEAF_SIZE - 1) / LEAF_SIZE;
  uint32_t *global_counter;
  CUDA_CHECK(
      cudaMallocAsync(&global_counter, sizeof(uint32_t), compute_stream));

  ComputeGradientsTriangleParams params{
      triangles,
      geometry_to_internal,
      m_bvh8_nodes,
      m_bvh8_leaf_pointers,
      m_taylor_coefficients,
      m_binary_aabbs + leaf_count - 1,
      SoAView<Vec3>{m_sorted_queries, m_query_count},
      m_sorted_grad_outputs,
      (uint32_t)m_query_count,
      (uint32_t)geometry_count,
      gradients,
      global_counter,
      beta};
  compute_triangle_gradients(params, m_device, compute_stream);

  // free temporary memory
  CUDA_CHECK(cudaFreeAsync(geometry_to_internal, compute_stream));
  CUDA_CHECK(cudaFreeAsync(global_counter, compute_stream));

  CudaUniquePtr<float> result(
      gradients, CudaDeleter{reinterpret_cast<size_t>(compute_stream)});
  return result;
}

auto GradientBackend::compute(const float *vertices,
                              const uint32_t *triangle_indices,
                              size_t vertex_count, size_t geometry_count,
                              float beta, uint64_t stream)
    -> CudaUniquePtr<float> {
  ScopedCudaDevice device_scope{m_device};
  // convert stream to cuda stream
  cudaStream_t compute_stream = reinterpret_cast<cudaStream_t>(stream);

  // Create temporary triangles from indices
  float *triangles;
  CUDA_CHECK(cudaMallocAsync(&triangles, geometry_count * sizeof(Triangle),
                             compute_stream));

  gather_triangles(vertices, triangle_indices,
                   static_cast<uint32_t>(geometry_count), triangles,
                   compute_stream);

  auto triangle_result = this->compute(triangles, geometry_count, beta, stream);
  CUDA_CHECK(cudaFreeAsync(triangles, compute_stream));

  float *vertex_gradients;
  CUDA_CHECK(cudaMallocAsync(&vertex_gradients,
                             vertex_count * 3 * sizeof(float), compute_stream));
  CUDA_CHECK(cudaMemsetAsync(vertex_gradients, 0,
                             vertex_count * 3 * sizeof(float), compute_stream));

  accumulate_vertex_gradients(triangle_result.get(), triangle_indices,
                              geometry_count, vertex_gradients, compute_stream);

  CudaUniquePtr<float> result(
      vertex_gradients, CudaDeleter{reinterpret_cast<size_t>(compute_stream)});
  return result;
}

auto GradientBackend::dump() const -> std::string {
  // Edge Case 1: No geometry at all
  if (m_query_count == 0) {
    return "digraph BVH8 {\n}\n";
  }

  std::string result = "digraph BVH8 {\n";
  result += "  node [fontname=\"Arial\", fontsize=10];\n";
  result += "  rankdir=TB;\n\n";

  // Edge Case 2: Only a single leaf exists (No internal nodes)
  if (m_query_count <= LEAF_SIZE) {
    std::vector<AABB> leaf_aabbs(1);
    CUDA_CHECK(cudaMemcpy(leaf_aabbs.data(), m_binary_aabbs, sizeof(AABB),
                          cudaMemcpyDeviceToHost));

    std::vector<Vec3> queries(m_query_count);
    CUDA_CHECK(cudaMemcpy(queries.data(), m_sorted_queries,
                          m_query_count * sizeof(Vec3),
                          cudaMemcpyDeviceToHost));
    SoAView<Vec3> query_view{reinterpret_cast<float *>(queries.data()),
                             m_query_count};

    AABB leaf_aabb = leaf_aabbs[0];
    Vec3 leaf_com = leaf_aabb.center_of_mass;

    result += "  L0 [shape=none, label=<\n";
    result += "    <TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\" "
              "CELLPADDING=\"4\" BGCOLOR=\"#eaffea\">\n";
    result += "      <TR><TD COLSPAN=\"4\" BGCOLOR=\"#4CAF50\"><B><FONT "
              "COLOR=\"white\">LEAF 0 (Root Leaf)</FONT></B></TD></TR>\n";
    result +=
        std::format("      <TR><TD COLSPAN=\"4\" BGCOLOR=\"#c8e6c9\"><I>CoM: "
                    "({:.4f}, {:.4f}, {:.4f})</I></TD></TR>\n",
                    leaf_com.x, leaf_com.y, leaf_com.z);
    result += std::format(
        "      <TR><TD COLSPAN=\"4\"><FONT POINT-SIZE=\"9\">AABB "
        "Min: ({:.3f}, {:.3f}, {:.3f})<BR/>AABB Max: ({:.3f}, "
        "{:.3f}, {:.3f})</FONT></TD></TR>\n",
        __half2float(leaf_aabb.min.x), __half2float(leaf_aabb.min.y),
        __half2float(leaf_aabb.min.z), __half2float(leaf_aabb.max.x),
        __half2float(leaf_aabb.max.y), __half2float(leaf_aabb.max.z));

    // Append child geometry primitives using SoA view
    for (size_t q_id = 0; q_id < m_query_count; q_id++) {
      const Vec3 &q = Vec3::load(query_view, q_id, m_query_count);
      result += std::format(
          "      <TR><TD>P{}</TD><TD COLSPAN=\"3\" ALIGN=\"LEFT\"><FONT "
          "POINT-SIZE=\"9\">{}</FONT></TD></TR>\n",
          q_id, q.dump());
    }
    result += "    </TABLE>>];\n}\n";
    return result;
  }

  // --- Standard BVH8 Multi-Node Multi-Leaf Gathering ---
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
  SoAView<Vec3> geometry_view{reinterpret_cast<float *>(queries.data()),
                              m_query_count};

  std::vector<LeafPointers> leaf_pointers(node_count);
  CUDA_CHECK(cudaMemcpy(leaf_pointers.data(), m_bvh8_leaf_pointers,
                        node_count * sizeof(LeafPointers),
                        cudaMemcpyDeviceToHost));

  // Pull unpacked values into your local host stack tracking vector
  std::vector<BackwardTaylorCoefficients> node_coefficients(node_count);
  CUDA_CHECK(cudaMemcpy(node_coefficients.data(), m_taylor_coefficients,
                        sizeof(BackwardTaylorCoefficients) * node_count,
                        cudaMemcpyDeviceToHost));

  // Breadth-First Search (BFS) matching your visualization layout
  std::queue<uint32_t> queue;
  queue.push(0); // Start at Root Internal Node
  int empty_counter = 0;

  while (!queue.empty()) {
    uint32_t current_id = queue.front();
    queue.pop();

    const BVH8Node &current_node = bvh8_nodes[current_id];
    BackwardTaylorCoefficients node_coeff = node_coefficients[current_id];
    AABB aabb = current_node.getAABB();
    Vec3 node_com = aabb.center_of_mass;

    // 1. Render Internal Node Layout Block
    result += std::format("  N{} [shape=none, label=<\n", current_id);
    result += "    <TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\" "
              "CELLPADDING=\"4\" BGCOLOR=\"#f0faff\">\n";
    result +=
        std::format("      <TR><TD COLSPAN=\"4\" BGCOLOR=\"#2196F3\"><B><FONT "
                    "COLOR=\"white\">NODE {}</FONT></B></TD></TR>\n",
                    current_id);
    result +=
        std::format("      <TR><TD COLSPAN=\"4\" BGCOLOR=\"#bbdefb\"><I>CoM: "
                    "({:.4f}, {:.4f}, {:.4f})</I></TD></TR>\n",
                    node_com.x, node_com.y, node_com.z);
    result += std::format(
        "      <TR><TD COLSPAN=\"4\"><FONT POINT-SIZE=\"9\">AABB Min: ({:.3f}, "
        "{:.3f}, {:.3f})<BR/>AABB Max: ({:.3f}, {:.3f}, "
        "{:.3f})</FONT></TD></TR>\n",
        __half2float(aabb.min.x), __half2float(aabb.min.y),
        __half2float(aabb.min.z), __half2float(aabb.max.x),
        __half2float(aabb.max.y), __half2float(aabb.max.z));

    // Render Internal Node Taylor Expansion Blocks
    result += std::format("      <TR><TD BGCOLOR=\"#bbdefb\"><B>1st "
                          "Order</B></TD><TD>{:.4f}</TD></TR>\n",
                          node_coeff.zero_order);
    result += std::format(
        "      <TR><TD BGCOLOR=\"#bbdefb\"><B>1st "
        "Order</B></TD><TD>{:.4f}</TD><TD>{:.4f}</TD><TD>{:.4f}</TD></TR>\n",
        node_coeff.first_order.x, node_coeff.first_order.y,
        node_coeff.first_order.z);

    result += "      <TR><TD ROWSPAN=\"3\" BGCOLOR=\"#bbdefb\"><B>2nd "
              "Order</B></TD>\n";
    Mat3x3 second_order = Mat3x3::from_sym(node_coeff.second_order);
    for (int r = 0; r < 3; ++r) {
      if (r > 0)
        result += "      <TR>\n";
      result += std::format(
          "        <TD>{:.4f}</TD><TD>{:.4f}</TD><TD>{:.4f}</TD></TR>\n",
          node_coeff.second_order.data[r * 3 + 0],
          node_coeff.second_order.data[r * 3 + 1],
          node_coeff.second_order.data[r * 3 + 2]);
    }
    result += "    </TABLE>>];\n";

    // 2. Parse Child Pointers Sequentially
    uint32_t child_base = current_node.child_base;
    uint32_t child_offset = 0;
    LeafPointers current_leaf_pointers = leaf_pointers[current_id];

    for (size_t child_id = 0; child_id < 8; child_id++) {
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

        // Render Leaf Block
        result += std::format("  L{} [shape=none, label=<\n", l_id);
        result += "    <TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\" "
                  "CELLPADDING=\"4\" BGCOLOR=\"#eaffea\">\n";
        result += std::format(
            "      <TR><TD COLSPAN=\"4\" BGCOLOR=\"#4CAF50\"><B><FONT "
            "COLOR=\"white\">LEAF {}</FONT></B></TD></TR>\n",
            l_id);
        result += std::format(
            "      <TR><TD COLSPAN=\"4\" BGCOLOR=\"#c8e6c9\"><I>CoM: ({:.4f}, "
            "{:.4f}, {:.4f})</I></TD></TR>\n",
            leaf_com.x, leaf_com.y, leaf_com.z);

        // Box Dimension Comparison (Reconstructed Compression vs True Bounds)
        result += std::format(
            "      <TR><TD COLSPAN=\"4\"><FONT POINT-SIZE=\"9\" "
            "COLOR=\"#333333\">"
            "AABB Min: ({:.2f}, {:.2f}, {:.2f}) Max: ({:.2f}, {:.2f}, {:.2f})"
            "</FONT></TD></TR>\n",
            __half2float(leaf_aabb.min.x), __half2float(leaf_aabb.min.y),
            __half2float(leaf_aabb.min.z), __half2float(leaf_aabb.max.x),
            __half2float(leaf_aabb.max.y), __half2float(leaf_aabb.max.z));

        // Extract and render local SoA query points bounded by the leaf
        result += std::format(
            "      <TR><TD COLSPAN=\"4\" BGCOLOR=\"#c8e6c9\"><I>Geometry (Max "
            "{} Packets)</I></TD></TR>\n",
            LEAF_SIZE);
        size_t g_off = l_id * LEAF_SIZE;
        for (size_t g_id = 0; g_id < LEAF_SIZE; g_id++) {
          size_t global_idx = g_off + g_id;
          if (global_idx >= m_query_count)
            break;

          const Vec3 &q = Vec3::load(geometry_view, global_idx, m_query_count);
          result += std::format(
              "      <TR><TD>Q{}</TD><TD COLSPAN=\"3\" ALIGN=\"LEFT\"><FONT "
              "POINT-SIZE=\"8\">{}</FONT></TD></TR>\n",
              g_id, q.dump());
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
