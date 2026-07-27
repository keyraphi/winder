#include "aabb.h"
#include "binary_node.h"
#include "bvh8.h"
#include "gradient_cuda.h"
#include "kernels/binary2bvh8.cuh"
#include "kernels/build_binary_tree.cuh"
#include "kernels/common.cuh"
#include "tailor_coefficients.h"
#include "utils.h"
#include "vec3.h"
#include <cstddef>
#include <cstdint>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_atomic_functions.h>
#include <driver_types.h>
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

#define LEAF_SIZE 32

GradientBackend::GradientBackend(const float *queries, const float* grad_outputs, size_t query_count,
                                 int device_id)
    : m_device{device_id}, m_query_count{query_count} {
  uint32_t leaf_count = (m_query_count + LEAF_SIZE - 1) / LEAF_SIZE;

  uint32_t max_bvh8_nodes = leaf_count - 1; // TODO worst case scenario!!

  // Allocate Persistent arrays
  CUDA_CHECK(cudaStreamCreate(&m_build_stream));
  CUDA_CHECK(cudaMallocAsync(&m_to_internal, query_count * sizeof(uint32_t),
                             m_build_stream));
  CUDA_CHECK(cudaMallocAsync(&m_sorted_queries, query_count * sizeof(float),
                             m_build_stream));

  CUDA_CHECK(cudaMallocAsync(
      &m_binary_aabbs, (2 * leaf_count - 1) * sizeof(AABB), m_build_stream));
  CUDA_CHECK(cudaMallocAsync(&m_bvh8_node_count, 1 * sizeof(uint32_t),
                             m_build_stream));
  CUDA_CHECK(cudaMallocAsync(&m_bvh8_nodes, max_bvh8_nodes * sizeof(BVH8Node),
                             m_build_stream));
  CUDA_CHECK(cudaMallocAsync(
      &m_tailor_coefficients,
      max_bvh8_nodes * sizeof(BackwardTailorCoefficientsF16), m_build_stream));
  CUDA_CHECK(cudaMallocAsync(&m_leaf_coefficients,
                             leaf_count * sizeof(BackwardTailorCoefficientsF16),
                             m_build_stream));
  CUDA_CHECK(cudaMallocAsync(&m_bvh8_leaf_pointers,
                             max_bvh8_nodes * sizeof(LeafPointers),
                             m_build_stream));

  // Build BVH8 ####################
  const auto *queries_v3 = reinterpret_cast<const Vec3 *>(queries);
  auto build_stream_policy = thrust::cuda::par.on(m_build_stream);

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

  gather_queries_soa(queries, m_to_internal, m_sorted_queries, m_query_count,
                     m_build_stream);

  auto morton_leaf_stride_idx = thrust::make_transform_iterator(
      thrust::make_counting_iterator<uint64_t>(0),
      [] __host__ __device__(uint32_t i) -> uint64_t { return i * LEAF_SIZE; });
  // thrust::make_strided_iterator<LEAF_SIZE>(geometry_morton_codes.begin());
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

  // initialize the atomic weights to 0
  float *atomic_weights;
  CUDA_CHECK(cudaMallocAsync(&atomic_weights, (leaf_count - 1) * sizeof(float),
                             m_build_stream));
  thrust::fill_n(m_build_stream_policy, atomic_weights, leaf_count - 1, 0.F);
  populate_binary_tree_aabb_and_leaf_coefficients_backward(
      m_sorted_queries, grad_outputs, m_leaf_coefficients, leaf_count, binary_nodes,
      m_binary_aabbs, binary_parents, atomic_weights, m_query_count,
      m_build_stream);
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
  thrust::fill_n(m_build_stream_policy, tmp_max_distances, bvh8_node_count,
                 0.F);
  Vec3 test;
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
  thrust::fill_n(m_build_stream_policy, atomic_counters, leaf_count - 1, 0);
  BackwardTailorCoefficients *m2m_f32_coefficients;
  CUDA_CHECK(cudaMallocAsync(&m2m_f32_coefficients,
                             max_bvh8_nodes * sizeof(TailorCoefficients),
                             m_build_stream));
  compute_internal_tailor_coefficients_m2m_backward( // TODO
      m_bvh8_nodes, bvh8_internal_parent_map, m_binary_aabbs + leaf_count - 1,
      m_leaf_coefficients, bvh8_leaf_parents, m_bvh8_leaf_pointers,
      m_tailor_coefficients, m2m_f32_coefficients, bvh8_nodes_child_count,
      leaf_count, atomic_counters, m_build_stream);

  CUDA_CHECK(cudaFreeAsync(m2m_f32_coefficients, m_build_stream));
  CUDA_CHECK(cudaFreeAsync(atomic_counters, m_build_stream));
  CUDA_CHECK(cudaFreeAsync(bvh8_internal_parent_map, m_build_stream));
  CUDA_CHECK(cudaFreeAsync(bvh8_leaf_parents, m_build_stream));
  CUDA_CHECK(cudaFreeAsync(bvh8_nodes_child_count, m_build_stream));
}

template GradientBackend::~GradientBackend() {
  CUDA_CHECK(cudaStreamSynchronize(m_build_stream));

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
  if (m_tailor_coefficients) {
    CUDA_CHECK(cudaFreeAsync(m_tailor_coefficients, m_build_stream));
  }
  if (m_leaf_coefficients) {
    CUDA_CHECK(cudaFreeAsync(m_leaf_coefficients, m_build_stream));
  }
  if (m_bvh8_leaf_pointers) {
    CUDA_CHECK(cudaFreeAsync(m_bvh8_leaf_pointers, m_build_stream));
  }
  CUDA_CHECK(cudaStreamSynchronize(m_build_stream));

  CUDA_CHECK(cudaStreamDestroy(m_build_stream));
}
