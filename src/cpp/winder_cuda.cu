#include <cmath>
#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cub/block/block_scan.cuh>
#include <cub/util_type.cuh>
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_atomic_functions.h>
#include <driver_types.h>
#include <format>
#include <memory>
#include <ostream>
#include <queue>
#include <stdexcept>
#include <string>
#include <sys/types.h>
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
#include <thrust/transform.h>
#include <vector>
#include <vector_functions.h>
#include <vector_types.h>

#include "aabb.h"
#include "binary_node.h"
#include "bvh8.h"
#include "geometry.h"
#include "kernels/binary2bvh8.cuh"
#include "kernels/brute_force.cuh"
#include "kernels/build_binary_tree.cuh"
#include "kernels/bvh8_m2m.cuh"
#include "kernels/common.cuh"
#include "kernels/mesh.cuh"
#include "kernels/traversal.cuh"
#include "tailor_coefficients.h"
#include "vec3.h"
#include "winder_cuda.h"

namespace cg = cooperative_groups;

#define CUDA_CHECK(expr_to_check)                                              \
  do {                                                                         \
    cudaError_t result = expr_to_check;                                        \
    if (result != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA Runtime Error: %s:%i:%d = %s\n", __FILE__,         \
              __LINE__, result, cudaGetErrorString(result));                   \
    }                                                                          \
  } while (0)

struct SceneParams {
  float scale;
  AABB bounds;
};
__constant__ SceneParams d_scene_params;

void CudaDeleter::operator()(void *ptr) const {
  if (ptr != nullptr) {
    auto compute_stream = reinterpret_cast<cudaStream_t>(stream);
    cudaFreeAsync(ptr, compute_stream);
  }
}

template <IsGeometry Geometry> WinderBackend<Geometry>::~WinderBackend() {
  CUDA_CHECK(cudaStreamSynchronize(m_build_stream));

  CUDA_CHECK(cudaEventDestroy(m_start_tree_construction_event));
  CUDA_CHECK(cudaEventDestroy(m_tree_construction_finished_event));

  if (m_to_internal) {
    CUDA_CHECK(cudaFreeAsync(m_to_internal, m_build_stream));
  }
  if (m_sorted_geometry) {
    CUDA_CHECK(cudaFreeAsync(m_sorted_geometry, m_build_stream));
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
  if (m_leaf_coefficients) {
    CUDA_CHECK(cudaFreeAsync(m_leaf_coefficients, m_build_stream));
  }
  if (m_bvh8_leaf_pointers) {
    CUDA_CHECK(cudaFreeAsync(m_bvh8_leaf_pointers, m_build_stream));
  }
  CUDA_CHECK(cudaStreamSynchronize(m_build_stream));

  CUDA_CHECK(cudaStreamDestroy(m_build_stream));
}

template <IsGeometry Geometry>
WinderBackend<Geometry>::WinderBackend(size_t size, int device_id)
    : m_count{size}, m_device{device_id} {

  // Setup streams
  CUDA_CHECK(cudaStreamCreate(&m_build_stream));
  // Policies for thrust to run async on streams
  m_build_stream_policy = thrust::cuda::par.on(m_build_stream);

  CUDA_CHECK(cudaEventCreate(&m_start_tree_construction_event));
  CUDA_CHECK(cudaEventCreate(&m_tree_construction_finished_event));

  // Allocate memory arena
  size_t leaf_count = (size + LEAF_SIZE - 1) / LEAF_SIZE;

  // uint32_t max_bvh8_nodes =
  //     (leaf_count <= 1) ? 0 : (uint32_t)ceil(leaf_count * 0.2F) + 1;
  uint32_t max_bvh8_nodes = leaf_count - 1; // DEBUG worst case scenario!!

  CUDA_CHECK(
      cudaMallocAsync(&m_to_internal, size * sizeof(uint32_t), m_build_stream));
  CUDA_CHECK(cudaMallocAsync(&m_sorted_geometry, size * sizeof(Geometry),
                             m_build_stream));
  CUDA_CHECK(cudaMallocAsync(
      &m_binary_aabbs, (2 * leaf_count - 1) * sizeof(AABB), m_build_stream));
  CUDA_CHECK(cudaMallocAsync(&m_bvh8_node_count, 1 * sizeof(uint32_t),
                             m_build_stream));
  CUDA_CHECK(cudaMallocAsync(&m_bvh8_nodes, max_bvh8_nodes * sizeof(BVH8Node),
                             m_build_stream));
  CUDA_CHECK(cudaMallocAsync(&m_leaf_coefficients,
                             leaf_count * sizeof(TailorCoefficientsF16),
                             m_build_stream));
  CUDA_CHECK(cudaMallocAsync(&m_bvh8_leaf_pointers,
                             max_bvh8_nodes * sizeof(LeafPointers),
                             m_build_stream));
}

template <IsPrimitiveGeometry PrimitiveGeometry> struct GeometryToAABB {
  __device__ auto operator()(const PrimitiveGeometry &g) const -> AABB {
    return g.get_aabb();
  }
};

struct MergeAABB {
  __device__ auto operator()(const AABB &a, const AABB &b) const -> AABB {
    return AABB::merge(a, b);
  }
};

template <IsPrimitiveGeometry PrimitiveGeometry> struct GeometryToMorton {

  __device__ auto operator()(const PrimitiveGeometry &g) const -> uint64_t {
    const float scale = d_scene_params.scale;
    const Vec3 min_p = Vec3::from_f16(d_scene_params.bounds.min);
    // Scale to range [0, 1]
    const Vec3 geometry_center = g.centroid();
    float tx = (geometry_center.x - min_p.x) * scale;
    float ty = (geometry_center.y - min_p.y) * scale;
    float tz = (geometry_center.z - min_p.z) * scale;

    // Fixed 21-bit integer quantization range [0, 2097151]
    auto x =
        static_cast<uint32_t>(fminf(fmaxf(tx * 2097151.F, 0.F), 2097151.F));
    auto y =
        static_cast<uint32_t>(fminf(fmaxf(ty * 2097151.F, 0.F), 2097151.F));
    auto z =
        static_cast<uint32_t>(fminf(fmaxf(tz * 2097151.F, 0.F), 2097151.F));

    // Expand bits (interleave x, y, z)
    return morton3D_63bit(x, y, z);
  }
};

template <IsGeometry Geometry>
template <IsPrimitiveGeometry PrimitiveGeometry>
auto WinderBackend<Geometry>::initializeMortonCodes(
    const PrimitiveGeometry *geometry, uint64_t *geometry_morton_codes)
    -> void {
  // compute scene bound
  auto aabb_transform = thrust::make_transform_iterator(
      geometry, GeometryToAABB<PrimitiveGeometry>{});

  AABB scene_bounds =
      thrust::reduce(m_build_stream_policy, aabb_transform,
                     aabb_transform + m_count, AABB::empty(), MergeAABB{});
  // create morton codes for each primitive
  Vec3 extent = Vec3::from_f16(scene_bounds.diagonal());
  float max_dim = fmaxf(extent.x, fmaxf(extent.y, extent.z));
  float scale = (max_dim > 1e-9F) ? 1.F / max_dim : 0.F;

  SceneParams scen_params{scale, scene_bounds};
  CUDA_CHECK(cudaMemcpyToSymbolAsync(d_scene_params, &scen_params,
                                     sizeof(SceneParams), 0,
                                     cudaMemcpyHostToDevice, m_build_stream));

  thrust::transform(m_build_stream_policy, geometry, geometry + m_count,
                    geometry_morton_codes,
                    GeometryToMorton<PrimitiveGeometry>{});
}

template <>
void WinderBackend<Triangle>::initialize_triangle_data(const float *triangles) {
  const auto *triangles_tri = reinterpret_cast<const Triangle *>(triangles);

  CUDA_CHECK(cudaEventRecord(m_start_tree_construction_event, m_build_stream));

  uint64_t *geometry_morton_codes;
  CUDA_CHECK(cudaMallocAsync(&geometry_morton_codes, m_count * sizeof(uint64_t),
                             m_build_stream));
  initializeMortonCodes(triangles_tri, geometry_morton_codes);

  // sort by morton codes
  thrust::sequence(m_build_stream_policy, m_to_internal,
                   m_to_internal + m_count);
  // sorts both morton_codes and m_to_internal
  thrust::sort_by_key(m_build_stream_policy, geometry_morton_codes,
                      geometry_morton_codes + m_count, m_to_internal);
  // sort triangles using m_to_internal
  gather_triangles_soa(triangles, m_to_internal, m_sorted_geometry, m_count,
                       m_build_stream);

  // each leaf contains 32 (LEAF_SIZE) elements
  uint32_t leaf_count = (m_count + LEAF_SIZE - 1) / LEAF_SIZE;
  uint32_t max_bvh8_nodes = leaf_count - 1; // DEBUG worst case scenario!!

  auto morton_leaf_stride_idx = thrust::make_transform_iterator(
      thrust::make_counting_iterator<uint64_t>(0),
      [] __host__ __device__(uint64_t i) -> uint64_t { return i * LEAF_SIZE; });
  // thrust::make_strided_iterator<LEAF_SIZE>(geometry_morton_codes.begin());
  // // requires cuda 13
  auto morton_leaf_stride = thrust::make_permutation_iterator(
      geometry_morton_codes, morton_leaf_stride_idx);

  uint64_t *leaf_morton_codes;
  CUDA_CHECK(cudaMallocAsync(&leaf_morton_codes, m_count * sizeof(uint64_t),
                             m_build_stream));
  thrust::copy(m_build_stream_policy, morton_leaf_stride,
               morton_leaf_stride + leaf_count, leaf_morton_codes);
  CUDA_CHECK(cudaFreeAsync(geometry_morton_codes, m_build_stream));
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
  populate_binary_tree_aabb_and_leaf_coefficients<Triangle>(
      m_sorted_geometry, m_leaf_coefficients, leaf_count, binary_nodes,
      m_binary_aabbs, binary_parents, atomic_weights, m_count, m_build_stream);
  CUDA_CHECK(cudaFreeAsync(binary_parents, m_build_stream));
  CUDA_CHECK(cudaFreeAsync(atomic_weights, m_build_stream));

  // Convert binary LBVH tree into BVH8 tree
  uint32_t *bvh8_work_queue_A;
  uint32_t *bvh8_work_queue_B;
  uint32_t *bvh8_internal_parent_map;
  uint32_t *global_counter;
  uint32_t *bvh8_leaf_parents;
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

  ConvertBinary2BVH8Params params{
      bvh8_work_queue_A,    bvh8_work_queue_B, bvh8_internal_parent_map,
      global_counter,       leaf_count,        m_binary_aabbs,
      binary_nodes,         bvh8_leaf_parents, m_bvh8_nodes,
      m_bvh8_leaf_pointers, m_bvh8_node_count};
  convert_binary_tree_to_bvh8(params, m_device, m_build_stream);

  CUDA_CHECK(cudaFreeAsync(bvh8_work_queue_A, m_build_stream));
  CUDA_CHECK(cudaFreeAsync(bvh8_work_queue_B, m_build_stream));
  CUDA_CHECK(cudaFreeAsync(global_counter, m_build_stream));
  CUDA_CHECK(cudaFreeAsync(binary_nodes, m_build_stream));

  // Compute the max distances of geometry from the center of mass for the nodes
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

  // populate BVH8 nodes with tailor coefficients using m2m
  // initialize atomic counters to 0 (TODO: good idea?)
  uint32_t *atomic_counters;
  CUDA_CHECK(cudaMallocAsync(
      &atomic_counters, (leaf_count - 1) * sizeof(uint32_t), m_build_stream));
  thrust::fill_n(m_build_stream_policy, atomic_counters, leaf_count - 1, 0);
  TailorCoefficients *m2m_f32_coefficients;
  CUDA_CHECK(cudaMallocAsync(&m2m_f32_coefficients,
                             max_bvh8_nodes * sizeof(TailorCoefficients),
                             m_build_stream));
  compute_internal_tailor_coefficients_m2m(
      m_bvh8_nodes, bvh8_internal_parent_map, m_binary_aabbs + leaf_count - 1,
      m_leaf_coefficients, bvh8_leaf_parents, m_bvh8_leaf_pointers,
      m2m_f32_coefficients, leaf_count, atomic_counters, m_build_stream);

  CUDA_CHECK(cudaFreeAsync(m2m_f32_coefficients, m_build_stream));
  CUDA_CHECK(cudaFreeAsync(atomic_counters, m_build_stream));
  CUDA_CHECK(cudaFreeAsync(bvh8_internal_parent_map, m_build_stream));
  CUDA_CHECK(cudaFreeAsync(bvh8_leaf_parents, m_build_stream));

  CUDA_CHECK(
      cudaEventRecord(m_tree_construction_finished_event, m_build_stream));
}

template <>
void WinderBackend<PointNormal>::initialize_point_data(const float *points,
                                                       const float *normals) {
  const auto *points_v3 = reinterpret_cast<const Vec3 *>(points);

  CUDA_CHECK(cudaEventRecord(m_start_tree_construction_event, m_build_stream));

  uint64_t *geometry_morton_codes;
  CUDA_CHECK(cudaMallocAsync(&geometry_morton_codes, m_count * sizeof(uint64_t),
                             m_build_stream));
  initializeMortonCodes<Vec3>(points_v3, geometry_morton_codes);

  // sort by morton codes
  thrust::sequence(m_build_stream_policy, m_to_internal,
                   m_to_internal + m_count);
  // sorts both morton_codes and m_to_internal
  thrust::sort_by_key(m_build_stream_policy, geometry_morton_codes,
                      geometry_morton_codes + m_count, m_to_internal);

  gather_point_normals_soa(points, normals, m_to_internal, m_sorted_geometry,
                           m_count, m_build_stream);

  // each leaf contains 32 (LEAF_SIZE) elements
  uint32_t leaf_count = (m_count + LEAF_SIZE - 1) / LEAF_SIZE;
  uint32_t max_bvh8_nodes =
      leaf_count - 1; // DEBUG worst case scenario - very pesimistic!!

  auto morton_leaf_stride_idx = thrust::make_transform_iterator(
      thrust::make_counting_iterator<uint64_t>(0),
      [] __host__ __device__(uint32_t i) -> uint64_t { return i * LEAF_SIZE; });
  // thrust::make_strided_iterator<LEAF_SIZE>(geometry_morton_codes.begin());
  auto morton_leaf_stride = thrust::make_permutation_iterator(
      geometry_morton_codes, morton_leaf_stride_idx);
  uint64_t *leaf_morton_codes;
  CUDA_CHECK(cudaMallocAsync(&leaf_morton_codes, m_count * sizeof(uint64_t),
                             m_build_stream));
  thrust::copy(m_build_stream_policy, morton_leaf_stride,
               morton_leaf_stride + leaf_count, leaf_morton_codes);
  // build binary radix tree
  CUDA_CHECK(cudaFreeAsync(geometry_morton_codes, m_build_stream));
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
  populate_binary_tree_aabb_and_leaf_coefficients<PointNormal>(
      m_sorted_geometry, m_leaf_coefficients, leaf_count, binary_nodes,
      m_binary_aabbs, binary_parents, atomic_weights, m_count, m_build_stream);
  CUDA_CHECK(cudaFreeAsync(binary_parents, m_build_stream));
  CUDA_CHECK(cudaFreeAsync(atomic_weights, m_build_stream));

  // Convert binary LBVH tree into BVH8 tree
  uint32_t *bvh8_work_queue_A;
  uint32_t *bvh8_work_queue_B;
  uint32_t *bvh8_internal_parent_map;
  uint32_t *global_counter;
  uint32_t *bvh8_leaf_parents;
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

  ConvertBinary2BVH8Params params{
      bvh8_work_queue_A,    bvh8_work_queue_B, bvh8_internal_parent_map,
      global_counter,       leaf_count,        m_binary_aabbs,
      binary_nodes,         bvh8_leaf_parents, m_bvh8_nodes,
      m_bvh8_leaf_pointers, m_bvh8_node_count};
  convert_binary_tree_to_bvh8(params, m_device, m_build_stream);

  CUDA_CHECK(cudaFreeAsync(bvh8_work_queue_A, m_build_stream));
  CUDA_CHECK(cudaFreeAsync(bvh8_work_queue_B, m_build_stream));
  CUDA_CHECK(cudaFreeAsync(global_counter, m_build_stream));
  CUDA_CHECK(cudaFreeAsync(binary_nodes, m_build_stream));

  // Compute the max distances of geometry from the center of mass for the nodes
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

  // populate BVH8 nodes with tailor coefficients using m2m
  // initialize atomic coutners to 0
  uint32_t *atomic_counters;
  CUDA_CHECK(cudaMallocAsync(&atomic_counters, (leaf_count-1)*sizeof(uint32_t), m_build_stream));
  thrust::fill_n(m_build_stream_policy, atomic_counters, leaf_count - 1, 0);
  TailorCoefficients *m2m_f32_coefficients;
  CUDA_CHECK(cudaMallocAsync(&m2m_f32_coefficients,
                             max_bvh8_nodes * sizeof(TailorCoefficients),
                             m_build_stream));
  compute_internal_tailor_coefficients_m2m(
      m_bvh8_nodes, bvh8_internal_parent_map, m_binary_aabbs + leaf_count - 1,
      m_leaf_coefficients, bvh8_leaf_parents, m_bvh8_leaf_pointers,
      m2m_f32_coefficients, leaf_count, atomic_counters, m_build_stream);

  CUDA_CHECK(cudaFreeAsync(m2m_f32_coefficients, m_build_stream));
  CUDA_CHECK(cudaFreeAsync(atomic_counters, m_build_stream));
  CUDA_CHECK(cudaFreeAsync(bvh8_internal_parent_map, m_build_stream));
  CUDA_CHECK(cudaFreeAsync(bvh8_leaf_parents, m_build_stream));

  CUDA_CHECK(
      cudaEventRecord(m_tree_construction_finished_event, m_build_stream));
}

template <typename T> struct GeometryTraits {
  static constexpr float default_beta = 2.3F;
};
template <> struct GeometryTraits<PointNormal> {
  static constexpr float default_beta = 2.0F;
};

template <IsGeometry Geometry>
auto WinderBackend<Geometry>::brute_force(const float *queries,
                                          size_t query_count, float epsilon,
                                          size_t stream) const
    -> CudaUniquePtr<float> {
  ScopedCudaDevice device_scope{m_device};
  const Vec3 *queries_vec3 = reinterpret_cast<const Vec3 *>(queries);

  cudaEvent_t start, finish;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&finish));

  // convert stream to cuda stream
  cudaStream_t compute_stream = reinterpret_cast<cudaStream_t>(stream);

  CUDA_CHECK(cudaEventRecord(start, compute_stream));

  // Allocate required buffers
  float *winding_numbers; // result
  CUDA_CHECK(cudaMallocAsync(&winding_numbers, query_count * sizeof(float),
                             compute_stream));

  if (epsilon < 0.F) {
    // default from 3D Reconstruction with Fast Dipole Sums
    epsilon = 1.F / 250.F;
  }

  compute_brute_force<Geometry>(queries_vec3, m_sorted_geometry,
                                (uint32_t)query_count, (uint32_t)m_count,
                                winding_numbers, epsilon, compute_stream);

  CUDA_CHECK(cudaEventRecord(finish, compute_stream));
  // free events
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(finish));

  CudaUniquePtr<float> result(
      winding_numbers, CudaDeleter{reinterpret_cast<size_t>(compute_stream)});
  return result;
}

template <IsGeometry Geometry>
auto WinderBackend<Geometry>::compute(const float *queries, size_t query_count,
                                      float beta, float epsilon,
                                      size_t stream) const
    -> CudaUniquePtr<float> {
  cudaEvent_t start, finish;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&finish));

  ScopedCudaDevice device_scope{m_device};
  const Vec3 *queries_vec3 = reinterpret_cast<const Vec3 *>(queries);

  // convert stream to cuda stream
  cudaStream_t compute_stream = reinterpret_cast<cudaStream_t>(stream);
  auto compute_stream_policy = thrust::cuda::par.on(compute_stream);
  CUDA_CHECK(cudaEventRecord(start, compute_stream));

  // Allocate required buffers
  float *winding_numbers; // result
  CUDA_CHECK(cudaMallocAsync(&winding_numbers, query_count * sizeof(float),
                             compute_stream));
  uint32_t *queries_to_internal;
  uint64_t *queries_morton;
  CUDA_CHECK(cudaMallocAsync(&queries_to_internal,
                             query_count * sizeof(uint32_t), compute_stream));
  CUDA_CHECK(cudaMallocAsync(&queries_morton, query_count * sizeof(uint64_t),
                             compute_stream));

  // sort queries by morton code, scaled with bvh8s aabb
  thrust::transform(compute_stream_policy, queries_vec3,
                    queries_vec3 + query_count, queries_morton,
                    GeometryToMorton<Vec3>{});

  thrust::sequence(compute_stream_policy, queries_to_internal,
                   queries_to_internal + query_count);
  thrust::sort_by_key(compute_stream_policy, queries_morton,
                      queries_morton + query_count, queries_to_internal);

  // free morton code memory
  CUDA_CHECK(cudaFreeAsync(queries_morton, compute_stream));

  // make sure stream 0 has finished building the tree
  CUDA_CHECK(
      cudaStreamWaitEvent(compute_stream, m_tree_construction_finished_event));

  if (beta < 0.F) {
    // defaults from Fast Winding Numbers paper
    beta = GeometryTraits<Geometry>::default_beta;
  }
  if (epsilon < 0.F) {
    // default from 3D Reconstruction with Fast Dipole Sums
    epsilon = 1.F / 250.F;
  }
  uint32_t *global_counter;
  CUDA_CHECK(
      cudaMallocAsync(&global_counter, sizeof(uint32_t), compute_stream));
  ComputeWindingNumbersParams<Geometry> params{
      queries_vec3,
      queries_to_internal,
      m_bvh8_nodes,
      m_bvh8_leaf_pointers,
      m_leaf_coefficients,
      SoAView<Geometry>{m_sorted_geometry, m_count},
      (uint32_t)query_count,
      (uint32_t)m_count,
      winding_numbers,
      global_counter,
      beta,
      epsilon};
  compute_winding_numbers<Geometry>(params, m_device, compute_stream);
  // free temporary memory
  CUDA_CHECK(cudaFreeAsync(queries_to_internal, compute_stream));
  CUDA_CHECK(cudaFreeAsync(global_counter, compute_stream));

  CUDA_CHECK(cudaEventRecord(finish, compute_stream));
  // free events
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(finish));
  CudaUniquePtr<float> result(
      winding_numbers, CudaDeleter{reinterpret_cast<size_t>(compute_stream)});
  return result;
}

template <>
auto WinderBackend<PointNormal>::CreateFromPoints(const float *points,
                                                  const float *normals,
                                                  size_t point_count,
                                                  int device_id)
    -> std::unique_ptr<WinderBackend<PointNormal>> {
  ScopedCudaDevice device_scope(device_id);

  printf("CreateFromPoints!\n");
  auto self = std::unique_ptr<WinderBackend<PointNormal>>{
      new WinderBackend<PointNormal>(point_count, device_id)};
  self->initialize_point_data(points, normals);
  return self;
}

template <>
auto WinderBackend<Triangle>::CreateFromTriangles(const float *triangles,
                                                  size_t triangle_count,
                                                  int device_id)
    -> std::unique_ptr<WinderBackend<Triangle>> {
  ScopedCudaDevice device_scope(device_id);
  printf("CreateFromTriangles!\n");
  auto self = std::unique_ptr<WinderBackend>{
      new WinderBackend<Triangle>(triangle_count, device_id)};

  self->initialize_triangle_data(triangles);
  return self;
}

template <>
auto WinderBackend<Triangle>::CreateFromMesh(const float *vertices,
                                             size_t vertex_count,
                                             const uint32_t *triangle_indices,
                                             size_t triangle_count,
                                             int device_id)
    -> std::unique_ptr<WinderBackend<Triangle>> {
  ScopedCudaDevice device_scope(device_id);
  printf("CreateFromMesh!\n");
  auto self = std::unique_ptr<WinderBackend>{
      new WinderBackend<Triangle>(triangle_count, device_id)};

  // Verify that index range does not exceed vertex size
  uint32_t max_index = thrust::reduce(thrust::device, triangle_indices,
                                      triangle_indices + 3 * triangle_count, 0,
                                      thrust::maximum<uint32_t>());
  if (max_index >= vertex_count) {
    throw std::runtime_error(
        std::format("The triangle indices are not allowed to exceed the number "
                    "of vertices. Vertex count is {}, max index is {}.",
                    vertex_count, max_index));
  }

  // Create temporary triangles from indices
  float *triangles;
  CUDA_CHECK(cudaMallocAsync(&triangles, triangle_count * sizeof(Triangle),
                             self->m_build_stream));

  gather_triangles(vertices, triangle_indices,
                   static_cast<uint32_t>(triangle_count), triangles,
                   self->m_build_stream);

  self->initialize_triangle_data(triangles);

  CUDA_CHECK(cudaFreeAsync(triangles, self->m_build_stream));
  return self;
}

template <>
auto WinderBackend<PointNormal>::CreateForSolver(const float *points,
                                                 size_t point_count,
                                                 int device_id)
    -> std::unique_ptr<WinderBackend<PointNormal>> {
  ScopedCudaDevice device_scope(device_id);
  auto self = std::unique_ptr<WinderBackend<PointNormal>>{
      new WinderBackend<PointNormal>(point_count, device_id)};

  // initialize unknown normals with 0
  thrust::device_vector<float> zero_normals(point_count, 0.F);

  self->initialize_point_data(points, zero_normals.data().get());
  return self;
}

template <>
auto WinderBackend<PointNormal>::get_normals() const -> CudaUniquePtr<float> {
  throw std::runtime_error("get_normals is not implemented yet!");
}
template <>
auto WinderBackend<Triangle>::get_normals() const -> CudaUniquePtr<float> {
  throw std::runtime_error("get_normals is not implemented yet!");
}

template <>
auto WinderBackend<PointNormal>::grad_normals(
    [[maybe_unused]] const float *grad_output,
    [[maybe_unused]] size_t n_queries) const -> CudaUniquePtr<float> {
  throw std::runtime_error("grad_normals not implemented yet!");
}
template <>
auto WinderBackend<Triangle>::grad_normals(
    [[maybe_unused]] const float *grad_output,
    [[maybe_unused]] size_t n_queries) const -> CudaUniquePtr<float> {
  throw std::runtime_error("grad_normals not implemented yet!");
}

template <>
auto WinderBackend<PointNormal>::grad_points(
    [[maybe_unused]] const float *grad_output,
    [[maybe_unused]] size_t n_queries) const -> CudaUniquePtr<float> {
  throw std::runtime_error("grad_points not implemented yet!");
}
template <>
auto WinderBackend<Triangle>::grad_points(
    [[maybe_unused]] const float *grad_output,
    [[maybe_unused]] size_t n_queries) const -> CudaUniquePtr<float> {
  throw std::runtime_error("grad_points not implemented yet!");
}

template <>
void WinderBackend<PointNormal>::solve_for_normals(
    [[maybe_unused]] const float *extra_p, [[maybe_unused]] size_t extra_count,
    [[maybe_unused]] const float *extra_wn, [[maybe_unused]] const float *pc_wn,
    [[maybe_unused]] float alpha) {
  throw std::runtime_error("solve_for_normals not implemented yet!");
}
template <>
void WinderBackend<Triangle>::solve_for_normals(
    [[maybe_unused]] const float *extra_p, [[maybe_unused]] size_t extra_count,
    [[maybe_unused]] const float *extra_wn, [[maybe_unused]] const float *pc_wn,
    [[maybe_unused]] float alpha) {
  throw std::runtime_error("solve_for_normals not implemented yet!");
}

__global__ void getTailorCoefficientsKernel(const BVH8Node *nodes,
                                             TailorCoefficientsF16 *result,
                                             uint32_t node_count) {
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < node_count) {
    const BVH8Node &node = nodes[tid];
    result[tid].zero_order = node.tailor_coefficients.get_tailor_zero_order();
    result[tid].first_order = node.tailor_coefficients.get_tailor_first_order();
    result[tid].second_order =
        node.tailor_coefficients.get_tailor_second_order();
  }
}

template <IsGeometry Geometry>
auto WinderBackend<Geometry>::dump() const -> std::string {
  // Edge Case 1: No geometry at all
  if (m_count == 0) {
    return "digraph BVH8 {\n}\n";
  }

  std::string result = "digraph BVH8 {\n";
  result += "  node [fontname=\"Arial\", fontsize=10];\n";
  result += "  rankdir=TB;\n\n";

  // Edge Case 2: Only a single leaf exists (No internal nodes)
  if (m_count <= LEAF_SIZE) {
    std::vector<AABB> leaf_aabbs(1);
    CUDA_CHECK(cudaMemcpy(leaf_aabbs.data(), m_binary_aabbs, sizeof(AABB),
                          cudaMemcpyDeviceToHost));

    std::vector<Geometry> geometry(m_count);
    CUDA_CHECK(cudaMemcpy(geometry.data(), m_sorted_geometry,
                          m_count * sizeof(Geometry), cudaMemcpyDeviceToHost));
    SoAView<Geometry> geometry_view{reinterpret_cast<float *>(geometry.data()),
                                    m_count};

    AABB leaf_aabb = leaf_aabbs[0];
    Vec3 leaf_com =
        leaf_aabb.center_of_mass.get(leaf_aabb.min, leaf_aabb.diagonal());

    result += "  L0 [shape=none, label=<\n";
    result += "    <TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\" "
              "CELLPADDING=\"4\" BGCOLOR=\"#eaffea\">\n";
    result += "      <TR><TD COLSPAN=\"4\" BGCOLOR=\"#4CAF50\"><B><FONT "
              "COLOR=\"white\">LEAF 0 (Root Leaf)</FONT></B></TD></TR>\n";
    result +=
        std::format("      <TR><TD COLSPAN=\"4\" BGCOLOR=\"#c8e6c9\"><I>CoM: "
                    "({:.4f}, {:.4f}, {:.4f})</I></TD></TR>\n",
                    leaf_com.x, leaf_com.y, leaf_com.z);
    result +=
        std::format("      <TR><TD COLSPAN=\"4\"><FONT POINT-SIZE=\"9\">AABB "
                    "Min: ({:.3f}, {:.3f}, {:.3f})<BR/>AABB Max: ({:.3f}, "
                    "{:.3f}, {:.3f})</FONT></TD></TR>\n",
                    leaf_aabb.min.x, leaf_aabb.min.y, leaf_aabb.min.z,
                    leaf_aabb.max.x, leaf_aabb.max.y, leaf_aabb.max.z);

    // Append child geometry primitives using SoA view
    for (size_t g_id = 0; g_id < m_count; g_id++) {
      const Geometry &g = Geometry::load(geometry_view, g_id, m_count);
      result += std::format(
          "      <TR><TD>P{}</TD><TD COLSPAN=\"3\" ALIGN=\"LEFT\"><FONT "
          "POINT-SIZE=\"9\">{}</FONT></TD></TR>\n",
          g_id, g.dump());
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

  uint32_t leaf_count = (m_count + LEAF_SIZE - 1) / LEAF_SIZE;
  std::vector<AABB> leaf_aabbs(leaf_count);
  CUDA_CHECK(cudaMemcpy(leaf_aabbs.data(), m_binary_aabbs + leaf_count - 1,
                        sizeof(AABB) * leaf_count, cudaMemcpyDeviceToHost));

  std::vector<Geometry> geometry(m_count);
  CUDA_CHECK(cudaMemcpy(geometry.data(), m_sorted_geometry,
                        m_count * sizeof(Geometry), cudaMemcpyDeviceToHost));
  SoAView<Geometry> geometry_view{reinterpret_cast<float *>(geometry.data()),
                                  m_count};

  std::vector<LeafPointers> leaf_pointers(node_count);
  CUDA_CHECK(cudaMemcpy(leaf_pointers.data(), m_bvh8_leaf_pointers,
                        node_count * sizeof(LeafPointers),
                        cudaMemcpyDeviceToHost));

  std::vector<TailorCoefficientsF16> leaf_coefficients(leaf_count);
  CUDA_CHECK(cudaMemcpy(leaf_coefficients.data(), m_leaf_coefficients,
                        leaf_count * sizeof(TailorCoefficientsF16),
                        cudaMemcpyDeviceToHost));

  TailorCoefficientsF16 *d_node_coefficients = nullptr;
  CUDA_CHECK(cudaMalloc(&d_node_coefficients,
                        sizeof(TailorCoefficientsF16) * node_count));

  // Calculate execution configuration and launch on the device
  uint32_t block = 128;
  uint32_t grid = (node_count + block - 1) / block;
  getTailorCoefficientsKernel<<<grid, block>>>(m_bvh8_nodes,
                                               d_node_coefficients, node_count);

  // Catch launch anomalies and wait for execution boundary
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Pull unpacked values into your local host stack tracking vector
  std::vector<TailorCoefficientsF16> node_coefficients(node_count);
  CUDA_CHECK(cudaMemcpy(node_coefficients.data(), d_node_coefficients,
                        sizeof(TailorCoefficientsF16) * node_count,
                        cudaMemcpyDeviceToHost));

  // Free the temporary GPU allocation
  CUDA_CHECK(cudaFree(d_node_coefficients));

  // Breadth-First Search (BFS) matching your visualization layout
  std::queue<uint32_t> queue;
  queue.push(0); // Start at Root Internal Node
  int empty_counter = 0;

  while (!queue.empty()) {
    uint32_t current_id = queue.front();
    queue.pop();

    const BVH8Node &current_node = bvh8_nodes[current_id];
    TailorCoefficients node_coeff =
        TailorCoefficients::from_f16(node_coefficients[current_id]);
    AABB aabb = current_node.parent_aabb;
    Vec3 node_com = aabb.center_of_mass.get(aabb.min, aabb.diagonal());

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
        aabb.min.x, aabb.min.y, aabb.min.z, aabb.max.x, aabb.max.y, aabb.max.z);

    // Render Internal Node Taylor Expansion Blocks
    result += std::format(
        "      <TR><TD BGCOLOR=\"#bbdefb\"><B>Zero "
        "Order</B></TD><TD>{:.4f}</TD><TD>{:.4f}</TD><TD>{:.4f}</TD></TR>\n",
        node_coeff.zero_order.x, node_coeff.zero_order.y,
        node_coeff.zero_order.z);

    result += "      <TR><TD ROWSPAN=\"3\" BGCOLOR=\"#bbdefb\"><B>1st "
              "Order</B></TD>\n";
    for (int r = 0; r < 3; ++r) {
      if (r > 0)
        result += "      <TR>\n";
      result += std::format(
          "        <TD>{:.4f}</TD><TD>{:.4f}</TD><TD>{:.4f}</TD></TR>\n",
          node_coeff.first_order.data[r * 3 + 0],
          node_coeff.first_order.data[r * 3 + 1],
          node_coeff.first_order.data[r * 3 + 2]);
    }

    const Tensor3 node_second_order = node_coeff.second_order.uncompress();
    result += "      <TR><TD COLSPAN=\"4\" BGCOLOR=\"#bbdefb\"><B>2nd "
              "Order</B></TD></TR>\n";
    for (int s = 0; s < 3; ++s) {
      result += std::format("      <TR><TD ROWSPAN=\"3\">Slice {}</TD>\n", s);
      for (int r = 0; r < 3; ++r) {
        if (r > 0)
          result += "      <TR>\n";
        int b = (s * 9) + (r * 3);
        result += std::format(
            "        <TD>{:.4f}</TD><TD>{:.4f}</TD><TD>{:.4f}</TD></TR>\n",
            node_second_order.data[b + 0], node_second_order.data[b + 1],
            node_second_order.data[b + 2]);
      }
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
        AABB leaf_aabb_approx = AABB::from_approximation(
            aabb, current_node.child_aabb_approx[child_id]);
        AABB leaf_aabb = leaf_aabbs[l_id];

        TailorCoefficients coeff =
            TailorCoefficients::from_f16(leaf_coefficients[l_id]);
        Vec3 leaf_com = leaf_coefficients[l_id].center_of_mass.get(
            leaf_aabb.min, leaf_aabb.diagonal());

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
            "Approx Min: ({:.2f}, {:.2f}, {:.2f}) Max: ({:.2f}, {:.2f}, "
            "{:.2f})<BR/>"
            "True Min: ({:.2f}, {:.2f}, {:.2f}) Max: ({:.2f}, {:.2f}, {:.2f})"
            "</FONT></TD></TR>\n",
            leaf_aabb_approx.min.x, leaf_aabb_approx.min.y,
            leaf_aabb_approx.min.z, leaf_aabb_approx.max.x,
            leaf_aabb_approx.max.y, leaf_aabb_approx.max.z, leaf_aabb.min.x,
            leaf_aabb.min.y, leaf_aabb.min.z, leaf_aabb.max.x, leaf_aabb.max.y,
            leaf_aabb.max.z);

        // Taylor Expansion Matrix
        result += std::format("      <TR><TD BGCOLOR=\"#c8e6c9\"><B>Zero "
                              "Order</B></TD><TD>{:.4f}</TD><TD>{:.4f}</"
                              "TD><TD>{:.4f}</TD></TR>\n",
                              coeff.zero_order.x, coeff.zero_order.y,
                              coeff.zero_order.z);
        result += "      <TR><TD ROWSPAN=\"3\" BGCOLOR=\"#c8e6c9\"><B>1st "
                  "Order</B></TD>\n";
        for (int r = 0; r < 3; ++r) {
          if (r > 0)
            result += "      <TR>\n";
          result += std::format(
              "        <TD>{:.4f}</TD><TD>{:.4f}</TD><TD>{:.4f}</TD></TR>\n",
              coeff.first_order.data[r * 3 + 0],
              coeff.first_order.data[r * 3 + 1],
              coeff.first_order.data[r * 3 + 2]);
        }

        // Extract and render local SoA points bounded by the leaf
        result += std::format(
            "      <TR><TD COLSPAN=\"4\" BGCOLOR=\"#c8e6c9\"><I>Geometry (Max "
            "{} Packets)</I></TD></TR>\n",
            LEAF_SIZE);
        size_t g_off = l_id * LEAF_SIZE;
        for (size_t g_id = 0; g_id < LEAF_SIZE; g_id++) {
          size_t global_idx = g_off + g_id;
          if (global_idx >= m_count)
            break;

          const Geometry &g =
              Geometry::load(geometry_view, global_idx, m_count);
          result += std::format(
              "      <TR><TD>P{}</TD><TD COLSPAN=\"3\" ALIGN=\"LEFT\"><FONT "
              "POINT-SIZE=\"8\">{}</FONT></TD></TR>\n",
              g_id, g.dump());
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

template class WinderBackend<PointNormal>;
template class WinderBackend<Triangle>;
