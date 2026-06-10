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

void CudaDeleter::operator()(void *ptr) const { cudaFree(ptr); }

template <IsGeometry Geometry> WinderBackend<Geometry>::~WinderBackend() {
  CUDA_CHECK(cudaStreamSynchronize(m_stream_0));
  CUDA_CHECK(cudaStreamSynchronize(m_stream_1));

  CUDA_CHECK(cudaEventDestroy(m_start_tree_construction_event));
  CUDA_CHECK(cudaEventDestroy(m_tree_construction_finished_event));

  if (m_to_internal) {
    CUDA_CHECK(cudaFreeAsync(m_to_internal, m_stream_0));
  }
  if (m_sorted_geometry) {
    CUDA_CHECK(cudaFreeAsync(m_sorted_geometry, m_stream_0));
  }
  if (m_binary_aabbs) {
    CUDA_CHECK(cudaFreeAsync(m_binary_aabbs, m_stream_0));
  }
  if (m_bvh8_node_count) {
    CUDA_CHECK(cudaFreeAsync(m_bvh8_node_count, m_stream_0));
  }
  if (m_bvh8_nodes) {
    CUDA_CHECK(cudaFreeAsync(m_bvh8_nodes, m_stream_0));
  }
  if (m_leaf_zero_order) {
    CUDA_CHECK(cudaFreeAsync(m_leaf_zero_order, m_stream_0));
  }
  if (m_bvh8_leaf_pointers) {
    CUDA_CHECK(cudaFreeAsync(m_bvh8_leaf_pointers, m_stream_0));
  }
  CUDA_CHECK(cudaStreamSynchronize(m_stream_0));

  CUDA_CHECK(cudaStreamDestroy(m_stream_0));
  CUDA_CHECK(cudaStreamDestroy(m_stream_1));
}

template <IsGeometry Geometry>
WinderBackend<Geometry>::WinderBackend(size_t size, int device_id)
    : m_count{size}, m_device{device_id} {

  // Setup streams
  CUDA_CHECK(cudaStreamCreate(&m_stream_0));
  CUDA_CHECK(cudaStreamCreate(&m_stream_1));
  // Policies for thrust to run async on streams
  m_stream_0_policy = thrust::cuda::par.on(m_stream_0);
  m_stream_1_policy = thrust::cuda::par.on(m_stream_1);

  CUDA_CHECK(cudaEventCreate(&m_start_tree_construction_event));
  CUDA_CHECK(cudaEventCreate(&m_tree_construction_finished_event));

  // Allocate memory arena
  size_t leaf_count = (size + LEAF_SIZE - 1) / LEAF_SIZE;

  // uint32_t max_bvh8_nodes =
  //     (leaf_count <= 1) ? 0 : (uint32_t)ceil(leaf_count * 0.2F) + 1;
  uint32_t max_bvh8_nodes = leaf_count - 1; // DEBUG worst case scenario!!

  CUDA_CHECK(
      cudaMallocAsync(&m_to_internal, size * sizeof(uint32_t), m_stream_0));
  CUDA_CHECK(
      cudaMallocAsync(&m_sorted_geometry, size * sizeof(Geometry), m_stream_0));
  CUDA_CHECK(cudaMallocAsync(&m_binary_aabbs,
                             (2 * leaf_count - 1) * sizeof(AABB), m_stream_0));
  CUDA_CHECK(
      cudaMallocAsync(&m_bvh8_node_count, 1 * sizeof(uint32_t), m_stream_0));
  CUDA_CHECK(cudaMallocAsync(&m_bvh8_nodes, max_bvh8_nodes * sizeof(BVH8Node),
                             m_stream_0));
  CUDA_CHECK(cudaMallocAsync(&m_leaf_zero_order, leaf_count * sizeof(Vec3),
                             m_stream_0));
  CUDA_CHECK(cudaMallocAsync(&m_bvh8_leaf_pointers,
                             max_bvh8_nodes * sizeof(LeafPointers),
                             m_stream_0));
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
    const Vec3 min_p = d_scene_params.bounds.min;
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
  // compute scene bounds
  auto aabb_transform = thrust::make_transform_iterator(
      geometry, GeometryToAABB<PrimitiveGeometry>{});

  AABB scene_bounds =
      thrust::reduce(m_stream_0_policy, aabb_transform,
                     aabb_transform + m_count, AABB::empty(), MergeAABB{});
  // create morton codes for each primitive
  Vec3 extent = scene_bounds.max - scene_bounds.min;
  float max_dim = fmaxf(extent.x, fmaxf(extent.y, extent.z));
  float scale = (max_dim > 1e-9F) ? 1.F / max_dim : 0.F;

  SceneParams scen_params{scale, scene_bounds};
  CUDA_CHECK(cudaMemcpyToSymbolAsync(d_scene_params, &scen_params,
                                     sizeof(SceneParams), 0,
                                     cudaMemcpyHostToDevice, m_stream_0));

  thrust::transform(m_stream_0_policy, geometry, geometry + m_count,
                    geometry_morton_codes,
                    GeometryToMorton<PrimitiveGeometry>{});
}

template <>
void WinderBackend<Triangle>::initialize_triangle_data(const float *triangles) {
  const auto *triangles_tri = reinterpret_cast<const Triangle *>(triangles);

  CUDA_CHECK(cudaEventRecord(m_start_tree_construction_event, m_stream_0));

  uint64_t *geometry_morton_codes;
  CUDA_CHECK(cudaMallocAsync(&geometry_morton_codes, m_count * sizeof(uint64_t),
                             m_stream_0));
  initializeMortonCodes(triangles_tri, geometry_morton_codes);

  // sort by morton codes
  thrust::sequence(m_stream_0_policy, m_to_internal, m_to_internal + m_count);
  // sorts both morton_codes and m_to_internal
  thrust::sort_by_key(m_stream_0_policy, geometry_morton_codes,
                      geometry_morton_codes + m_count, m_to_internal);
  // sort triangles using m_to_internal
  gather_triangles_soa(triangles, m_to_internal, m_sorted_geometry, m_count,
                       m_stream_0);

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
                             m_stream_0));
  thrust::copy(m_stream_0_policy, morton_leaf_stride,
               morton_leaf_stride + leaf_count, leaf_morton_codes);
  CUDA_CHECK(cudaFreeAsync(geometry_morton_codes, m_stream_0));
  // build binary radix tree
  BinaryNode *binary_nodes;
  uint32_t *binary_parents;
  CUDA_CHECK(cudaMallocAsync(
      &binary_nodes, (leaf_count - 1) * sizeof(BinaryNode), m_stream_0));
  CUDA_CHECK(cudaMallocAsync(
      &binary_parents, (2 * leaf_count - 1) * sizeof(uint32_t), m_stream_0));
  build_binary_topology(leaf_morton_codes, binary_nodes, binary_parents,
                        leaf_count, m_stream_0);
  CUDA_CHECK(cudaFreeAsync(leaf_morton_codes, m_stream_0));

  // initialize the atomic counters to 0
  uint32_t *atomic_counters;
  CUDA_CHECK(cudaMallocAsync(&atomic_counters,
                             (leaf_count - 1) * sizeof(uint32_t), m_stream_0));
  thrust::fill_n(m_stream_0_policy, atomic_counters, leaf_count - 1, 0);
  populate_binary_tree_aabb_and_leaf_coefficients<Triangle>(
      m_sorted_geometry, m_leaf_zero_order, leaf_count, binary_nodes,
      m_binary_aabbs, binary_parents, atomic_counters, m_count, m_stream_0);
  CUDA_CHECK(cudaFreeAsync(binary_parents, m_stream_0));

  // Convert binary LBVH tree into BVH8 tree
  uint32_t *bvh8_work_queue_A;
  uint32_t *bvh8_work_queue_B;
  uint32_t *bvh8_internal_parent_map;
  uint32_t *global_counter;
  uint32_t *bvh8_leaf_parents;
  uint32_t *bvh8_node_child_count;
  CUDA_CHECK(cudaMallocAsync(&bvh8_work_queue_A,
                             (leaf_count - 1) * sizeof(uint32_t), m_stream_0));
  CUDA_CHECK(cudaMallocAsync(&bvh8_work_queue_B,
                             (leaf_count - 1) * sizeof(uint32_t), m_stream_0));
  CUDA_CHECK(cudaMallocAsync(&bvh8_internal_parent_map,
                             max_bvh8_nodes * sizeof(uint32_t), m_stream_0));
  CUDA_CHECK(
      cudaMallocAsync(&global_counter, 1 * sizeof(uint32_t), m_stream_0));
  CUDA_CHECK(cudaMallocAsync(&bvh8_leaf_parents, leaf_count * sizeof(uint32_t),
                             m_stream_0));
  CUDA_CHECK(cudaMallocAsync(&bvh8_node_child_count,
                             max_bvh8_nodes * sizeof(uint32_t), m_stream_0));

  ConvertBinary2BVH8Params params{
      bvh8_work_queue_A,     bvh8_work_queue_B,    bvh8_internal_parent_map,
      global_counter,        leaf_count,           m_binary_aabbs,
      binary_nodes,          bvh8_leaf_parents,    m_bvh8_nodes,
      bvh8_node_child_count, m_bvh8_leaf_pointers, m_bvh8_node_count};
  convert_binary_tree_to_bvh8(params, m_device, m_stream_0);

  CUDA_CHECK(cudaFreeAsync(bvh8_work_queue_A, m_stream_0));
  CUDA_CHECK(cudaFreeAsync(bvh8_work_queue_B, m_stream_0));
  CUDA_CHECK(cudaFreeAsync(global_counter, m_stream_0));
  CUDA_CHECK(cudaFreeAsync(binary_nodes, m_stream_0));

  // Compute the max distances of geometry from the center of mass for the nodes
  compute_max_distances<Triangle>(m_bvh8_nodes, m_sorted_geometry,
                                  bvh8_leaf_parents, bvh8_internal_parent_map,
                                  static_cast<uint32_t>(m_count),
                                  m_bvh8_node_count, m_stream_0);
  // populate BVH8 nodes with tailor coefficients using m2m
  // initialize the atomic counters to 0
  thrust::fill_n(m_stream_0_policy, atomic_counters, leaf_count - 1, 0);
  compute_internal_tailor_coefficients_m2m(
      m_bvh8_nodes, bvh8_internal_parent_map, m_leaf_zero_order,
      bvh8_leaf_parents, m_bvh8_leaf_pointers, bvh8_node_child_count,
      leaf_count, atomic_counters, m_stream_0);

  CUDA_CHECK(cudaFreeAsync(bvh8_node_child_count, m_stream_0));
  CUDA_CHECK(cudaFreeAsync(atomic_counters, m_stream_0));
  CUDA_CHECK(cudaFreeAsync(bvh8_internal_parent_map, m_stream_0));
  CUDA_CHECK(cudaFreeAsync(bvh8_leaf_parents, m_stream_0));

  CUDA_CHECK(cudaEventRecord(m_tree_construction_finished_event, m_stream_0));
}

template <>
void WinderBackend<PointNormal>::initialize_point_data(const float *points,
                                                       const float *normals) {
  const auto *points_v3 = reinterpret_cast<const Vec3 *>(points);

  CUDA_CHECK(cudaEventRecord(m_start_tree_construction_event, m_stream_0));

  uint64_t *geometry_morton_codes;
  CUDA_CHECK(cudaMallocAsync(&geometry_morton_codes, m_count * sizeof(uint64_t),
                             m_stream_0));
  initializeMortonCodes<Vec3>(points_v3, geometry_morton_codes);

  // sort by morton codes
  thrust::sequence(m_stream_0_policy, m_to_internal, m_to_internal + m_count);
  // sorts both morton_codes and m_to_internal
  thrust::sort_by_key(m_stream_0_policy, geometry_morton_codes,
                      geometry_morton_codes + m_count, m_to_internal);

  gather_point_normals_soa(points, normals, m_to_internal, m_sorted_geometry,
                           m_count, m_stream_0);

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
                             m_stream_0));
  thrust::copy(m_stream_0_policy, morton_leaf_stride,
               morton_leaf_stride + leaf_count, leaf_morton_codes);
  // build binary radix tree
  CUDA_CHECK(cudaFreeAsync(geometry_morton_codes, m_stream_0));
  // build binary radix tree
  BinaryNode *binary_nodes;
  uint32_t *binary_parents;
  CUDA_CHECK(cudaMallocAsync(
      &binary_nodes, (leaf_count - 1) * sizeof(BinaryNode), m_stream_0));
  CUDA_CHECK(cudaMallocAsync(
      &binary_parents, (2 * leaf_count - 1) * sizeof(uint32_t), m_stream_0));
  build_binary_topology(leaf_morton_codes, binary_nodes, binary_parents,
                        leaf_count, m_stream_0);
  CUDA_CHECK(cudaFreeAsync(leaf_morton_codes, m_stream_0));

  // initialize the atomic counters to 0
  uint32_t *atomic_counters;
  CUDA_CHECK(cudaMallocAsync(&atomic_counters,
                             (leaf_count - 1) * sizeof(uint32_t), m_stream_0));
  thrust::fill_n(m_stream_0_policy, atomic_counters, leaf_count - 1, 0);
  populate_binary_tree_aabb_and_leaf_coefficients<PointNormal>(
      m_sorted_geometry, m_leaf_zero_order, leaf_count, binary_nodes,
      m_binary_aabbs, binary_parents, atomic_counters, m_count, m_stream_0);
  CUDA_CHECK(cudaFreeAsync(binary_parents, m_stream_0));

  // Convert binary LBVH tree into BVH8 tree
  uint32_t *bvh8_work_queue_A;
  uint32_t *bvh8_work_queue_B;
  uint32_t *bvh8_internal_parent_map;
  uint32_t *global_counter;
  uint32_t *bvh8_leaf_parents;
  uint32_t *bvh8_node_child_count;
  CUDA_CHECK(cudaMallocAsync(&bvh8_work_queue_A,
                             (leaf_count - 1) * sizeof(uint32_t), m_stream_0));
  CUDA_CHECK(cudaMallocAsync(&bvh8_work_queue_B,
                             (leaf_count - 1) * sizeof(uint32_t), m_stream_0));
  CUDA_CHECK(cudaMallocAsync(&bvh8_internal_parent_map,
                             max_bvh8_nodes * sizeof(uint32_t), m_stream_0));
  CUDA_CHECK(
      cudaMallocAsync(&global_counter, 1 * sizeof(uint32_t), m_stream_0));
  CUDA_CHECK(cudaMallocAsync(&bvh8_leaf_parents, leaf_count * sizeof(uint32_t),
                             m_stream_0));
  CUDA_CHECK(cudaMallocAsync(&bvh8_node_child_count,
                             max_bvh8_nodes * sizeof(uint32_t), m_stream_0));

  ConvertBinary2BVH8Params params{
      bvh8_work_queue_A,     bvh8_work_queue_B,    bvh8_internal_parent_map,
      global_counter,        leaf_count,           m_binary_aabbs,
      binary_nodes,          bvh8_leaf_parents,    m_bvh8_nodes,
      bvh8_node_child_count, m_bvh8_leaf_pointers, m_bvh8_node_count};
  convert_binary_tree_to_bvh8(params, m_device, m_stream_0);

  CUDA_CHECK(cudaFreeAsync(bvh8_work_queue_A, m_stream_0));
  CUDA_CHECK(cudaFreeAsync(bvh8_work_queue_B, m_stream_0));
  CUDA_CHECK(cudaFreeAsync(global_counter, m_stream_0));
  CUDA_CHECK(cudaFreeAsync(binary_nodes, m_stream_0));

  // Compute the max distances of geometry from the center of mass for the nodes
  compute_max_distances<PointNormal>(m_bvh8_nodes, m_sorted_geometry,
                                  bvh8_leaf_parents, bvh8_internal_parent_map,
                                  static_cast<uint32_t>(m_count),
                                  m_bvh8_node_count, m_stream_0);
  // populate BVH8 nodes with tailor coefficients using m2m
  // reset atomic counters to 0
  thrust::fill_n(m_stream_0_policy, atomic_counters, leaf_count - 1, 0);
  compute_internal_tailor_coefficients_m2m(
      m_bvh8_nodes, bvh8_internal_parent_map, m_leaf_zero_order,
      bvh8_leaf_parents, m_bvh8_leaf_pointers, bvh8_node_child_count,
      leaf_count, atomic_counters, m_stream_0);

  CUDA_CHECK(cudaFreeAsync(bvh8_node_child_count, m_stream_0));
  CUDA_CHECK(cudaFreeAsync(atomic_counters, m_stream_0));
  CUDA_CHECK(cudaFreeAsync(bvh8_internal_parent_map, m_stream_0));
  CUDA_CHECK(cudaFreeAsync(bvh8_leaf_parents, m_stream_0));

  CUDA_CHECK(cudaEventRecord(m_tree_construction_finished_event, m_stream_0));
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

  bool is_stream_0 = stream % 2 == 0;
  const auto &compute_stream = is_stream_0 ? m_stream_0 : m_stream_1;

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
  CudaUniquePtr<float> result(winding_numbers);

  // TODO better solution?
  CUDA_CHECK(cudaStreamSynchronize(compute_stream));
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

  // decide which stream to run on
  bool is_stream_0 = stream % 2 == 0;
  const auto &compute_stream = is_stream_0 ? m_stream_0 : m_stream_1;
  const auto &compute_stream_policy =
      is_stream_0 ? m_stream_0_policy : m_stream_1_policy;
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

  uint32_t leaf_count = (m_count + LEAF_SIZE - 1) / LEAF_SIZE;
  uint32_t *global_counter;
  CUDA_CHECK(
      cudaMallocAsync(&global_counter, sizeof(uint32_t), compute_stream));
  ComputeWindingNumbersParams<Geometry> params{
      queries_vec3,
      queries_to_internal,
      m_bvh8_nodes,
      m_bvh8_leaf_pointers,
      m_binary_aabbs + leaf_count -
          1, // the first leaf node is at m_binary_aabbs[leaf_count-1]
      m_leaf_zero_order,
      m_sorted_geometry,
      (uint32_t)query_count,
      (uint32_t)m_count,
      winding_numbers,
      global_counter,
      leaf_count,
      beta,
      epsilon};
  compute_winding_numbers<Geometry>(params, m_device, compute_stream);
  // free temporary memory
  CUDA_CHECK(cudaFreeAsync(queries_to_internal, compute_stream));

  CUDA_CHECK(cudaEventRecord(finish, compute_stream));

  // free events
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(finish));
  CudaUniquePtr<float> result(winding_numbers);

  // TODO better solution?
  // sync compute stream to make sure the result is there
  CUDA_CHECK(cudaStreamSynchronize(compute_stream));
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
                             self->m_stream_0));

  gather_triangles(vertices, triangle_indices,
                   static_cast<uint32_t>(triangle_count), triangles,
                   self->m_stream_0);

  self->initialize_triangle_data(triangles);

  CUDA_CHECK(cudaFreeAsync(triangles, self->m_stream_0));
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

template <IsGeometry Geometry>
auto WinderBackend<Geometry>::dump() const -> std::string {
  // Edge Case: No geometry at all
  if (m_count == 0) {
    return "";
  }

  // Edge Case: Only a single leaf exists (no BVH8 nodes to traverse)
  if (m_count <= LEAF_SIZE) {
    std::string result;

    // Copy only the single leaf's AABB and its geometry
    std::vector<AABB> leaf_aabbs(1);
    CUDA_CHECK(cudaMemcpy(leaf_aabbs.data(), m_binary_aabbs, sizeof(AABB),
                          cudaMemcpyDeviceToHost));

    std::vector<Geometry> geometry(m_count);
    CUDA_CHECK(cudaMemcpy(geometry.data(), m_sorted_geometry,
                          m_count * sizeof(Geometry), cudaMemcpyDeviceToHost));
    SoAViewConst<Geometry> geometry_view{
        reinterpret_cast<float *>(geometry.data()), m_count};

    result += "Leaf {\n";

    AABB leaf_aabb = leaf_aabbs[0];
    Vec3 leaf_com = leaf_aabb.center_of_mass;
    float leaf_max_distance = leaf_aabb.max_distance_to_center;

    result +=
        std::format("  AABB {{min: ({}, {}, {}), max: ({}, {}, {}), com: ({}, "
                    "{}, {}) max_distance: {}}}\n",
                    leaf_aabb.min.x, leaf_aabb.min.y, leaf_aabb.min.z,
                    leaf_aabb.max.x, leaf_aabb.max.y, leaf_aabb.max.z,
                    leaf_com.x, leaf_com.y, leaf_com.z, leaf_max_distance);

    for (size_t g_id = 0; g_id < m_count; g_id++) {
      const Geometry &g = Geometry::load(geometry_view, g_id, m_count);
      result += "  " + g.dump();
    }

    result += "}\n";
    return result;
  }
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
  SoAViewConst<Geometry> geometry_view{
      reinterpret_cast<float *>(geometry.data()), m_count};
  std::vector<LeafPointers> leaf_pointers(node_count);
  CUDA_CHECK(cudaMemcpy(leaf_pointers.data(), m_bvh8_leaf_pointers,
                        node_count * sizeof(LeafPointers),
                        cudaMemcpyDeviceToHost));
  std::vector<float> leaf_zero_order(leaf_count * 3);
  CUDA_CHECK(cudaMemcpy(leaf_zero_order.data(), m_leaf_zero_order,
                        leaf_count * sizeof(Vec3), cudaMemcpyDeviceToHost));
  std::string result;

  struct StackEntry {
    uint32_t node_id;
    bool is_closing;
    int depth;
  };

  std::vector<StackEntry> stack;
  stack.push_back({0, false, 0}); // Start with root

  auto get_indent = [](size_t depth) -> std::string {
    return std::string(depth * 2, ' ');
  };

  while (!stack.empty()) {
    StackEntry entry = stack.back();
    stack.pop_back();

    std::string indent = get_indent(entry.depth);

    if (entry.is_closing) {
      result += indent + "}\n";
      continue;
    }

    const BVH8Node &current_node = bvh8_nodes[entry.node_id];
    AABB aabb = current_node.parent_aabb;
    Vec3 node_com = aabb.center_of_mass;
    float max_dist = aabb.max_distance_to_center;

    result +=
        indent +
        std::format("BVH8Node {{ id: {}, com: ({:.4f}, {:.4f}, "
                    "{:.4f}), max_dist: {:.4f}, AABB: {{min: ({:.4f}, {:.4f}, "
                    "{:.4f}), max: ({:.4f}, {:.4f}, {:.4f})}}\n",
                    entry.node_id, node_com.x, node_com.y, node_com.z, max_dist,
                    aabb.min.x, aabb.min.y, aabb.min.z, aabb.max.x, aabb.max.y,
                    aabb.max.z);

    // Push the closing marker for THIS node
    stack.push_back({entry.node_id, true, entry.depth});

    //  Prepare children
    uint32_t child_base = current_node.child_base;

    // Count internal children first to handle child_offset correctly
    int internal_count = 0;
    for (int i = 0; i < 8; ++i) {
      if (current_node.child_meta[i] == ChildType::INTERNAL)
        internal_count++;
    }
    LeafPointers current_leaf_pointers = leaf_pointers[entry.node_id];

    // We iterate backwards through children to maintain correct stack order
    int current_internal_offset = internal_count - 1;
    for (int child_id = 7; child_id >= 0; --child_id) {
      ChildType type = current_node.child_meta[child_id];

      if (type == ChildType::INTERNAL) {
        uint32_t next_idx = child_base + current_internal_offset;
        stack.push_back({next_idx, false, entry.depth + 1});
        current_internal_offset--;
      } else if (type == ChildType::LEAF) {
        uint32_t l_id = current_leaf_pointers.indices[child_id];
        AABB leaf_aabb = leaf_aabbs[l_id];
        Vec3 leaf_com = leaf_aabb.center_of_mass;
        float leaf_max_distance = leaf_aabb.max_distance_to_center;
        result += indent + "    " +
                  std::format(
                      "AABB {{min: ({}, {}, {}), max: ({}, {}, {}), com: ({}, "
                      "{}, {}) max_distance: {}}}\n",
                      leaf_aabb.min.x, leaf_aabb.min.y, leaf_aabb.min.z,
                      leaf_aabb.max.x, leaf_aabb.max.y, leaf_aabb.max.z,
                      leaf_com.x, leaf_com.y, leaf_com.z, leaf_max_distance);
        size_t g_off = l_id * LEAF_SIZE;
        for (size_t g_id = 0; g_id < LEAF_SIZE; g_id++) {
          const Geometry &g =
              Geometry::load(geometry_view, g_off + g_id, m_count);
          result += indent + "    " + g.dump();
          if (g_off + g_id >= m_count) {
            break;
          }
        }
        result += indent + "  }\n";
      }
    }
  }
  return result;
}

template class WinderBackend<PointNormal>;
template class WinderBackend<Triangle>;
