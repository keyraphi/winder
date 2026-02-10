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
#include <memory>
#include <ostream>
#include <stdexcept>
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
#include <thrust/system/cuda/detail/par.h>
#include <thrust/transform.h>
#include <vector_functions.h>
#include <vector_types.h>

#include "aabb.h"
#include "binary_node.h"
#include "bvh8.h"
#include "geometry.h"
#include "kernels/binary2bvh8.cuh"
#include "kernels/build_binary_tree.cuh"
#include "kernels/bvh8_m2m.cuh"
#include "kernels/common.cuh"
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
  if (m_leaf_coefficients) {
    CUDA_CHECK(cudaFreeAsync(m_leaf_coefficients, m_stream_0));
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
  CUDA_CHECK(cudaMallocAsync(&m_leaf_coefficients,
                             leaf_count * sizeof(TailorCoefficientsBf16),
                             m_stream_0));
  CUDA_CHECK(cudaMallocAsync(&m_bvh8_leaf_pointers,
                             max_bvh8_nodes * sizeof(LeafPointers),
                             m_stream_0));

  // m_bvh8_nodes =
  //     get_aligned_ptr<BVH8Node>(ptr, max_bvh8_nodes, remaining_arena_size);
  // m_leaf_coefficients = get_aligned_ptr<TailorCoefficientsBf16>(
  //     ptr, leaf_count, remaining_arena_size);
  // m_bvh8_leaf_pointers =
  //     get_aligned_ptr<LeafPointers>(ptr, max_bvh8_nodes,
  //     remaining_arena_size);
  // m_binary_aabbs =
  //     get_aligned_ptr<AABB>(ptr, 2 * leaf_count - 1, remaining_arena_size);
  // m_binary_nodes =
  //     get_aligned_ptr<BinaryNode>(ptr, leaf_count - 1, remaining_arena_size);
  // m_sorted_geometry =
  //     get_aligned_ptr<Geometry>(ptr, size, remaining_arena_size);
  // m_to_internal = get_aligned_ptr<uint32_t>(ptr, size, remaining_arena_size);
  // m_to_canonical = get_aligned_ptr<uint32_t>(ptr, size,
  // remaining_arena_size); m_morton_codes = get_aligned_ptr<uint32_t>(ptr,
  // size, remaining_arena_size); m_leaf_morton_codes =
  //     get_aligned_ptr<uint32_t>(ptr, leaf_count, remaining_arena_size);
  // m_binary_parents =
  //     get_aligned_ptr<uint32_t>(ptr, 2 * leaf_count - 1,
  //     remaining_arena_size);
  // m_atomic_counters =
  //     get_aligned_ptr<uint32_t>(ptr, leaf_count - 1, remaining_arena_size);
  // m_bvh8_leaf_parents =
  //     get_aligned_ptr<uint32_t>(ptr, leaf_count, remaining_arena_size);
  // m_bvh8_internal_parent_map =
  //     get_aligned_ptr<uint32_t>(ptr, max_bvh8_nodes, remaining_arena_size);
  // m_bvh8_work_queue_A =
  //     get_aligned_ptr<uint32_t>(ptr, leaf_count - 1, remaining_arena_size);
  // m_bvh8_work_queue_B =
  //     get_aligned_ptr<uint32_t>(ptr, leaf_count - 1, remaining_arena_size);
  // m_bvh8_node_count = get_aligned_ptr<uint32_t>(
  //     ptr, 1, remaining_arena_size);
  // m_global_counter = get_aligned_ptr<uint32_t>(
  //     ptr, 1, remaining_arena_size); // always at the end!
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

  __device__ auto operator()(const PrimitiveGeometry &g) const -> uint32_t {
    const float scale = d_scene_params.scale;
    const Vec3 min_p = d_scene_params.bounds.min;
    // Scale to range [0, 1]
    const Vec3 geometry_center = g.centroid();
    float tx = (geometry_center.x - min_p.x) * scale;
    float ty = (geometry_center.y - min_p.y) * scale;
    float tz = (geometry_center.z - min_p.z) * scale;

    // Scale to 10-bit integer range [0, 1023]
    auto x = static_cast<uint32_t>(fminf(fmaxf(tx * 1024.F, 0.F), 1023.F));
    auto y = static_cast<uint32_t>(fminf(fmaxf(ty * 1024.F, 0.F), 1023.F));
    auto z = static_cast<uint32_t>(fminf(fmaxf(tz * 1024.F, 0.F), 1023.F));

    // Expand bits (interleave x, y, z)
    return morton3D_30bit(x, y, z);
  }
};

template <IsGeometry Geometry>
template <IsPrimitiveGeometry PrimitiveGeometry>
auto WinderBackend<Geometry>::initializeMortonCodes(
    const PrimitiveGeometry *geometry,
    uint32_t *geometry_morton_codes) -> void {
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
void WinderBackend<Triangle>::initialize_mesh_data(const float *triangles) {
  const auto *triangles_tri = reinterpret_cast<const Triangle *>(triangles);

  CUDA_CHECK(cudaEventRecord(m_start_tree_construction_event, m_stream_0));

  uint32_t *geometry_morton_codes;
  CUDA_CHECK(cudaMallocAsync(&geometry_morton_codes, m_count * sizeof(uint32_t),
                             m_stream_0));
  initializeMortonCodes(triangles_tri, geometry_morton_codes);

  // sort by morton codes
  thrust::sequence(m_stream_0_policy, m_to_internal, m_to_internal + m_count);
  // sorts both morton_codes and m_to_internal
  thrust::sort_by_key(m_stream_0_policy, geometry_morton_codes,
                      geometry_morton_codes + m_count, m_to_internal);
  // sort triangles using m_to_internal
  thrust::gather(m_stream_0_policy, m_to_internal, m_to_internal + m_count,
                 triangles_tri, m_sorted_geometry);

  // each leaf contains 32 (LEAF_SIZE) elements
  uint32_t leaf_count = (m_count + LEAF_SIZE - 1) / LEAF_SIZE;
  uint32_t max_bvh8_nodes = leaf_count - 1; // DEBUG worst case scenario!!

  auto morton_leaf_stride_idx = thrust::make_transform_iterator(
      thrust::make_counting_iterator<uint32_t>(0),
      [] __host__ __device__(uint32_t i) -> uint32_t { return i * LEAF_SIZE; });
  // thrust::make_strided_iterator<LEAF_SIZE>(geometry_morton_codes.begin());
  // // requires cuda 13
  auto morton_leaf_stride = thrust::make_permutation_iterator(
      geometry_morton_codes, morton_leaf_stride_idx);

  uint32_t *leaf_morton_codes;
  CUDA_CHECK(cudaMallocAsync(&leaf_morton_codes, m_count * sizeof(uint32_t),
                             m_stream_0));
  thrust::copy(m_stream_0_policy, morton_leaf_stride,
               morton_leaf_stride + leaf_count, leaf_morton_codes);
  CUDA_CHECK(cudaFreeAsync(geometry_morton_codes, m_stream_0));
  // build binary radix tree
  BinaryNode *binary_nodes;
  uint32_t *binary_parents;
  CUDA_CHECK(cudaMallocAsync(&binary_nodes, (leaf_count - 1) * sizeof(BinaryNode),
                             m_stream_0));
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
      m_sorted_geometry, m_leaf_coefficients, leaf_count, binary_nodes,
      m_binary_aabbs, binary_parents, atomic_counters, m_count, m_stream_0);
  CUDA_CHECK(cudaFreeAsync(binary_parents, m_stream_0));

  // Convert binary LBVH tree into BVH8 tree
  uint32_t *bvh8_work_queue_A;
  uint32_t *bvh8_work_queue_B;
  uint32_t *bvh8_internal_parent_map;
  uint32_t *global_counter;
  uint32_t *bvh8_leaf_parents;
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

  ConvertBinary2BVH8Params params{
      bvh8_work_queue_A,    bvh8_work_queue_B, bvh8_internal_parent_map,
      global_counter,       leaf_count,        m_binary_aabbs,
      binary_nodes,         bvh8_leaf_parents, m_bvh8_nodes,
      m_bvh8_leaf_pointers, m_bvh8_node_count};
  convert_binary_tree_to_bvh8(params, m_device, m_stream_0);

  CUDA_CHECK(cudaFreeAsync(bvh8_work_queue_A, m_stream_0));
  CUDA_CHECK(cudaFreeAsync(bvh8_work_queue_B, m_stream_0));
  CUDA_CHECK(cudaFreeAsync(global_counter, m_stream_0));
  CUDA_CHECK(cudaFreeAsync(binary_nodes, m_stream_0));

  // populate BVH8 nodes with tailor coefficients using m2m
  // initialize the atomic counters to 0
  thrust::fill_n(m_stream_0_policy, atomic_counters, leaf_count - 1, 0);
  compute_internal_tailor_coefficients_m2m(
      m_bvh8_nodes, bvh8_internal_parent_map, m_binary_aabbs + leaf_count - 1,
      m_leaf_coefficients, bvh8_leaf_parents, m_bvh8_leaf_pointers, leaf_count,
      atomic_counters, m_stream_0);

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

  uint32_t *geometry_morton_codes;
  CUDA_CHECK(cudaMallocAsync(&geometry_morton_codes, m_count * sizeof(uint32_t),
                             m_stream_0));
  initializeMortonCodes<Vec3>(points_v3, geometry_morton_codes);

  // sort by morton codes
  thrust::sequence(m_stream_0_policy, m_to_internal, m_to_internal + m_count);
  // sorts both morton_codes and m_to_internal
  thrust::sort_by_key(m_stream_0_policy, geometry_morton_codes,
                      geometry_morton_codes + m_count, m_to_internal);

  interleave_gather_geometry(points, normals, m_to_internal, m_sorted_geometry,
                             m_count, m_stream_0);

  // each leaf contains 32 (LEAF_SIZE) elements
  uint32_t leaf_count = (m_count + LEAF_SIZE - 1) / LEAF_SIZE;
  uint32_t max_bvh8_nodes = leaf_count - 1; // DEBUG worst case scenario!!

  auto morton_leaf_stride_idx = thrust::make_transform_iterator(
      thrust::make_counting_iterator<uint32_t>(0),
      [] __host__ __device__(uint32_t i) -> uint32_t { return i * LEAF_SIZE; });
  // thrust::make_strided_iterator<LEAF_SIZE>(geometry_morton_codes.begin());
  auto morton_leaf_stride = thrust::make_permutation_iterator(
      geometry_morton_codes, morton_leaf_stride_idx);
  uint32_t *leaf_morton_codes;
  CUDA_CHECK(cudaMallocAsync(&leaf_morton_codes, m_count * sizeof(uint32_t),
                             m_stream_0));
  thrust::copy(m_stream_0_policy, morton_leaf_stride,
               morton_leaf_stride + leaf_count, leaf_morton_codes);
  // build binary radix tree
  CUDA_CHECK(cudaFreeAsync(geometry_morton_codes, m_stream_0));
  // build binary radix tree
  BinaryNode *binary_nodes;
  uint32_t *binary_parents;
  CUDA_CHECK(cudaMallocAsync(&binary_nodes, (leaf_count - 1) * sizeof(BinaryNode),
                             m_stream_0));
  CUDA_CHECK(cudaMallocAsync(
      &binary_parents, (2 * leaf_count - 1) * sizeof(uint32_t), m_stream_0));
  build_binary_topology(leaf_morton_codes, binary_nodes, binary_parents,
                        leaf_count, m_stream_0);
  CUDA_CHECK(cudaFreeAsync(leaf_morton_codes, m_stream_0));

  // initialize the atomic counters to 0
  uint32_t *atomic_counters;
  CUDA_CHECK(cudaMallocAsync(&atomic_counters,
                             (leaf_count - 1) * sizeof(uint32_t), m_stream_0));
  populate_binary_tree_aabb_and_leaf_coefficients<PointNormal>(
      m_sorted_geometry, m_leaf_coefficients, leaf_count, binary_nodes,
      m_binary_aabbs, binary_parents, atomic_counters, m_count, m_stream_0);
  CUDA_CHECK(cudaFreeAsync(binary_parents, m_stream_0));

  // Convert binary LBVH tree into BVH8 tree
  uint32_t *bvh8_work_queue_A;
  uint32_t *bvh8_work_queue_B;
  uint32_t *bvh8_internal_parent_map;
  uint32_t *global_counter;
  uint32_t *bvh8_leaf_parents;
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

  ConvertBinary2BVH8Params params{
      bvh8_work_queue_A,    bvh8_work_queue_B, bvh8_internal_parent_map,
      global_counter,       leaf_count,        m_binary_aabbs,
      binary_nodes,         bvh8_leaf_parents, m_bvh8_nodes,
      m_bvh8_leaf_pointers, m_bvh8_node_count};
  convert_binary_tree_to_bvh8(params, m_device, m_stream_0);

  CUDA_CHECK(cudaFreeAsync(bvh8_work_queue_A, m_stream_0));
  CUDA_CHECK(cudaFreeAsync(bvh8_work_queue_B, m_stream_0));
  CUDA_CHECK(cudaFreeAsync(global_counter, m_stream_0));
  CUDA_CHECK(cudaFreeAsync(binary_nodes, m_stream_0));

  // populate BVH8 nodes with tailor coefficients using m2m
  // reset atomic counters to 0
  thrust::fill_n(m_stream_0_policy, atomic_counters, leaf_count - 1, 0);

  compute_internal_tailor_coefficients_m2m(
      m_bvh8_nodes, bvh8_internal_parent_map, m_binary_aabbs + leaf_count - 1,
      m_leaf_coefficients, bvh8_leaf_parents, m_bvh8_leaf_pointers,
      leaf_count, atomic_counters, m_stream_0);

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
auto WinderBackend<Geometry>::compute(
    const float *queries, size_t query_count, float beta, float epsilon,
    size_t stream) const -> CudaUniquePtr<float> {
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
  uint32_t *queries_morton;
  CUDA_CHECK(cudaMallocAsync(&queries_to_internal, query_count * sizeof(uint32_t),
                             compute_stream));
  CUDA_CHECK(cudaMallocAsync(&queries_morton, query_count * sizeof(uint32_t),
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

  // DEBUG optimize
  float construction_elapsed_time_ms;
  CUDA_CHECK(cudaEventElapsedTime(&construction_elapsed_time_ms,
                                  m_start_tree_construction_event,
                                  m_tree_construction_finished_event));
  printf("DEBUG OPTIM: tree construction took %f ms.\n",
         construction_elapsed_time_ms);
  // END DBUG optimize

  if (beta < 0.F) {
    // defaults from Fast Winding Numbers paper
    beta = GeometryTraits<Geometry>::default_beta;
  }
  if (epsilon < 0.F) {
    // default from 3D Reconstruction with Fast Dipole Sums
    epsilon = 1.F / 250.F;
  }
  uint32_t *global_counter;
  CUDA_CHECK(cudaMallocAsync(&global_counter, sizeof(uint32_t), compute_stream));
  ComputeWindingNumbersParams<Geometry> params{queries_vec3,
                                               queries_to_internal,
                                               m_bvh8_nodes,
                                               m_bvh8_leaf_pointers,
                                               m_leaf_coefficients,
                                               m_sorted_geometry,
                                               (uint32_t)query_count,
                                               (uint32_t)m_count,
                                               winding_numbers,
                                               global_counter,
                                               beta,
                                               epsilon};
  printf("DEBUG: calling compute_winding_numbers (beta = %f)\n", params.beta);
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
auto WinderBackend<PointNormal>::CreateFromPoints(
    const float *points, const float *normals, size_t point_count,
    int device_id) -> std::unique_ptr<WinderBackend<PointNormal>> {
  ScopedCudaDevice device_scope(device_id);

  auto self = std::unique_ptr<WinderBackend<PointNormal>>{
      new WinderBackend<PointNormal>(point_count, device_id)};
  self->initialize_point_data(points, normals);
  return self;
}

template <>
auto WinderBackend<Triangle>::CreateFromMesh(
    const float *trisangles, size_t triangle_count,
    int device_id) -> std::unique_ptr<WinderBackend<Triangle>> {
  ScopedCudaDevice device_scope(device_id);
  auto self = std::unique_ptr<WinderBackend>{
      new WinderBackend<Triangle>(triangle_count, device_id)};

  self->initialize_mesh_data(trisangles); // Sorts and builds tree
  return self;
}

template <>
auto WinderBackend<PointNormal>::CreateForSolver(
    const float *points, size_t point_count,
    int device_id) -> std::unique_ptr<WinderBackend<PointNormal>> {
  ScopedCudaDevice device_scope(device_id);
  auto self = std::unique_ptr<WinderBackend<PointNormal>>{
      new WinderBackend<PointNormal>(point_count, device_id)};

  // initialize unknown normals with 0
  thrust::device_vector<float> zero_normals(point_count, 0.F);

  self->initialize_point_data(points, zero_normals.data().get());
  return self;
}

template class WinderBackend<PointNormal>;
template class WinderBackend<Triangle>;
