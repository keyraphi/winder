#include <cmath>
#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cub/block/block_scan.cuh>
#include <cub/util_type.cuh>
#include <cuda/__ptx/ptx_dot_variants.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_atomic_functions.h>
#include <driver_types.h>
#include <memory>
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
#include "tailor_coefficients.h"
#include "utils.h"
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

auto WinderBackend::get_bvh_view() -> BVH8View {
  return BVH8View{m_bvh8_nodes, m_bvh8_leaf_pointers, m_sorted_geometry,
                  m_bvh8_node_count};
}

auto WinderBackend::CreateFromMesh(const float *trisangles,
                                   size_t triangle_count, int device_id)
    -> std::unique_ptr<WinderBackend> {
  ScopedCudaDevice device_scope(device_id);
  auto self = std::unique_ptr<WinderBackend>{
      new WinderBackend(WinderMode::Triangle, triangle_count, device_id)};

  self->initialize_mesh_data(trisangles); // Sorts and builds tree
  return self;
}

auto WinderBackend::CreateFromPoints(const float *points, const float *normals,
                                     size_t point_count, int device_id)
    -> std::unique_ptr<WinderBackend> {
  ScopedCudaDevice device_scope(device_id);

  auto self = std::unique_ptr<WinderBackend>{
      new WinderBackend(WinderMode::Point, point_count, device_id)};
  self->initialize_point_data(points, normals);
  return self;
}

auto WinderBackend::CreateForSolver(const float *points, size_t point_count,
                                    int device_id)
    -> std::unique_ptr<WinderBackend> {
  ScopedCudaDevice device_scope(device_id);
  auto self = std::unique_ptr<WinderBackend>{
      new WinderBackend(WinderMode::Point, point_count, device_id)};

  // initialize unknown normals with 0
  thrust::device_vector<float> zero_normals(point_count, 0.F);

  self->initialize_point_data(points, zero_normals.data().get());
  return self;
}

template <typename T>
T *get_aligned_ptr(void *&current_ptr, size_t count,
                   size_t &remaining_arena_size) {
  const size_t alignment = L2_ALIGN;
  size_t requested_bytes = count * sizeof(T);

  void *aligned =
      std::align(alignment, requested_bytes, current_ptr, remaining_arena_size);

  if (!aligned) {
    throw std::runtime_error("Arena overflow or alignment failure");
  }

  // Advance the global pointer for the next buffer
  current_ptr = static_cast<uint8_t *>(aligned) + requested_bytes;
  remaining_arena_size -= requested_bytes;

  return static_cast<T *>(aligned);
}

WinderBackend::~WinderBackend() {
  CUDA_CHECK(cudaStreamSynchronize(m_stream_0));
  CUDA_CHECK(cudaStreamSynchronize(m_stream_1));

  if (m_memory_arena) {
    CUDA_CHECK(cudaFreeAsync(m_memory_arena, m_stream_0));
  }
  CUDA_CHECK(cudaStreamSynchronize(m_stream_0));

  CUDA_CHECK(cudaStreamDestroy(m_stream_0));
  CUDA_CHECK(cudaStreamDestroy(m_stream_1));
}

WinderBackend::WinderBackend(WinderMode mode, size_t size, int device_id)
    : m_mode{mode}, m_count{size}, m_device{device_id} {

  // Setup streams
  CUDA_CHECK(cudaStreamCreate(&m_stream_0));
  CUDA_CHECK(cudaStreamCreate(&m_stream_1));
  // Policies for thrust to run async on streams
  m_stream_0_policy = thrust::cuda::par.on(m_stream_0);
  m_stream_1_policy = thrust::cuda::par.on(m_stream_1);

  // Allocate memory arena
  size_t floats_per_elem =
      (mode == WinderMode::Triangle) ? sizeof(Triangle) : sizeof(PointNormal);
  size_t leaf_count = (size + LEAF_SIZE - 1) / LEAF_SIZE;

  uint32_t max_bvh8_nodes = leaf_count * 0.2;

  // Compute how much memory is needed, to make every entry 128 byte aligned
  size_t total_required = 0;
  auto add_to_total = [&](size_t bytes, size_t alignment) {
    size_t padding = (alignment - (total_required % alignment)) % alignment;
    total_required += padding;
    total_required += bytes;
  };

  add_to_total(max_bvh8_nodes * sizeof(BVH8Node),
               L2_ALIGN); // m_bvh8_nodes are already 128 byte
  add_to_total(leaf_count * sizeof(TailorCoefficientsBf16),
               L2_ALIGN); // m_leaf_coefficients
  add_to_total(max_bvh8_nodes * sizeof(LeafPointers),
               L2_ALIGN); // m_bvh8_leaf_pointers
  add_to_total((2 * leaf_count - 1) * sizeof(AABB), L2_ALIGN); // m_binary_aabbs
  add_to_total((leaf_count - 1) * sizeof(BinaryNode),
               L2_ALIGN); // m_binary_nodes
  add_to_total(size * floats_per_elem * sizeof(float),
               L2_ALIGN);                                // m_sorted_geometry
  add_to_total(size * sizeof(uint32_t), L2_ALIGN);       // m_to_internal
  add_to_total(size * sizeof(uint32_t), L2_ALIGN);       // m_to_canonical
  add_to_total(size * sizeof(uint32_t), L2_ALIGN);       // m_morton_codes
  add_to_total(leaf_count * sizeof(uint32_t), L2_ALIGN); // m_leaf_morton_codes
  add_to_total((2 * leaf_count - 1) * sizeof(uint32_t),
               L2_ALIGN); // m_binary_parents
  add_to_total((leaf_count - 1) * sizeof(uint32_t),
               L2_ALIGN);                                // m_atomic_counters
  add_to_total(leaf_count * sizeof(uint32_t), L2_ALIGN); // m_bvh8_leaf_parents
  add_to_total(max_bvh8_nodes * sizeof(uint32_t),
               L2_ALIGN); // m_bvh8_internal_parent_map
  add_to_total((leaf_count - 1) * sizeof(uint32_t),
               L2_ALIGN); // m_bvh8_work_queue_A
  add_to_total((leaf_count - 1) * sizeof(uint32_t),
               L2_ALIGN);                       // m_bvh8_work_queue_B
  add_to_total(1 * sizeof(uint32_t), L2_ALIGN); // m_global_counter
  cudaMallocAsync(&m_memory_arena, total_required, m_stream_0);

  // Assign each member to its location in the arena
  void *ptr = reinterpret_cast<void *>(m_memory_arena);
  size_t remaining_arena_size = total_required;

  m_bvh8_nodes =
      get_aligned_ptr<BVH8Node>(ptr, max_bvh8_nodes, remaining_arena_size);
  m_leaf_coefficients = get_aligned_ptr<TailorCoefficientsBf16>(
      ptr, leaf_count, remaining_arena_size);
  m_bvh8_leaf_pointers =
      get_aligned_ptr<LeafPointers>(ptr, max_bvh8_nodes, remaining_arena_size);
  m_binary_aabbs =
      get_aligned_ptr<AABB>(ptr, 2 * leaf_count - 1, remaining_arena_size);
  m_binary_nodes =
      get_aligned_ptr<BinaryNode>(ptr, leaf_count - 1, remaining_arena_size);
  m_sorted_geometry =
      get_aligned_ptr<float>(ptr, size * floats_per_elem, remaining_arena_size);
  m_to_internal = get_aligned_ptr<uint32_t>(ptr, size, remaining_arena_size);
  m_to_canonical = get_aligned_ptr<uint32_t>(ptr, size, remaining_arena_size);
  m_morton_codes = get_aligned_ptr<uint32_t>(ptr, size, remaining_arena_size);
  m_leaf_morton_codes =
      get_aligned_ptr<uint32_t>(ptr, leaf_count, remaining_arena_size);
  m_binary_parents =
      get_aligned_ptr<uint32_t>(ptr, 2 * leaf_count - 1, remaining_arena_size);
  m_atomic_counters =
      get_aligned_ptr<uint32_t>(ptr, leaf_count - 1, remaining_arena_size);
  m_bvh8_leaf_parents =
      get_aligned_ptr<uint32_t>(ptr, leaf_count, remaining_arena_size);
  m_bvh8_internal_parent_map =
      get_aligned_ptr<uint32_t>(ptr, max_bvh8_nodes, remaining_arena_size);
  m_bvh8_work_queue_A =
      get_aligned_ptr<uint32_t>(ptr, leaf_count - 1, remaining_arena_size);
  m_bvh8_work_queue_B =
      get_aligned_ptr<uint32_t>(ptr, leaf_count - 1, remaining_arena_size);
  m_global_counter = get_aligned_ptr<uint32_t>(
      ptr, 1, remaining_arena_size); // always at the end!

  // initialize the atomic counters to 0
  thrust::fill_n(m_stream_0_policy, m_atomic_counters, leaf_count - 1, 0);
}

template <typename Geometry> struct GeometryToAABB {
  __host__ __device__ auto operator()(const Geometry &g) const -> AABB {
    return g.get_aabb();
  }
};

struct MergeAABB {
  __host__ __device__ auto operator()(const AABB &a, const AABB &b) const
      -> AABB {
    return AABB::merge(a, b);
  }
};

template <typename Geometry> struct GeometryToMorton {

  __device__ auto operator()(const Geometry &g) const -> uint32_t {
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

template <typename Geometry>
auto WinderBackend::initializeMortonCodes(const Geometry *geometry) -> void {
  // compute scene bounds
  auto aabb_transform =
      thrust::make_transform_iterator(geometry, GeometryToAABB<Geometry>{});

  AABB scene_bounds =
      thrust::reduce(m_stream_0_policy, aabb_transform,
                     aabb_transform + m_count, AABB::empty(), MergeAABB{});
  // create morton codes for each primitive
  Vec3 extent = scene_bounds.max - scene_bounds.min;
  float max_dim = fmaxf(extent.x, fmaxf(extent.y, extent.z));
  float scale = (max_dim > 1e-9f) ? 1.0f / max_dim : 0.0f;

  SceneParams scen_params{scale, scene_bounds};
  CUDA_CHECK(cudaMemcpyToSymbolAsync(d_scene_params, &scen_params,
                                     sizeof(SceneParams), 0,
                                     cudaMemcpyHostToDevice, m_stream_0));

  thrust::transform(m_stream_0_policy, geometry, geometry + m_count,
                    m_morton_codes, GeometryToMorton<Geometry>{});
}

void WinderBackend::initialize_mesh_data(const float *triangles) {
  const auto *triangles_tri = reinterpret_cast<const Triangle *>(triangles);
  initializeMortonCodes<Triangle>(triangles_tri);

  // sort by morton codes
  thrust::sequence(m_stream_0_policy, m_to_internal, m_to_internal + m_count);
  // sorts both morton_codes and m_to_internal
  thrust::sort_by_key(m_stream_0_policy, m_morton_codes,
                      m_morton_codes + m_count, m_to_internal);
  // create inverse mapping m_to_canonical
  thrust::scatter(m_stream_0_policy,
                  thrust::make_counting_iterator<uint32_t>(0),
                  thrust::make_counting_iterator<uint32_t>(m_count),
                  m_to_internal, m_to_canonical);
  // sort triangles using m_to_internal
  thrust::gather(m_stream_0_policy, m_to_internal, m_to_internal + m_count,
                 triangles_tri,
                 reinterpret_cast<Triangle *>(m_sorted_geometry));

  // each leaf contains 32 (LEAF_SIZE) elements
  uint32_t leaf_count = (m_count + LEAF_SIZE - 1) / LEAF_SIZE;

  {
    auto morton_leaf_stride_idx = thrust::make_transform_iterator(
        thrust::make_counting_iterator<uint32_t>(0),
        [] __host__ __device__(uint32_t i) -> uint32_t {
          return i * LEAF_SIZE;
        });
    // thrust::make_strided_iterator<LEAF_SIZE>(m_morton_codes.begin());
    auto morton_leaf_stride = thrust::make_permutation_iterator(
        m_morton_codes, morton_leaf_stride_idx);
    thrust::copy(m_stream_0_policy, morton_leaf_stride,
                 morton_leaf_stride + leaf_count, m_leaf_morton_codes);
    // build binary radix tree
    build_binary_topology(m_leaf_morton_codes, m_binary_nodes, m_binary_parents,
                          leaf_count, m_stream_0);
  }

  populate_binary_tree_aabb_and_leaf_coefficients<Triangle>(
      reinterpret_cast<const Triangle *>(m_sorted_geometry),
      m_leaf_coefficients, leaf_count, m_binary_nodes, m_binary_aabbs,
      m_binary_parents, m_atomic_counters, m_count, m_stream_0);
  // Convert binary LBVH tree into BVH8 tree
  ConvertBinary2BVH8Params params{
      m_bvh8_work_queue_A, m_bvh8_work_queue_B, m_bvh8_internal_parent_map,
      m_global_counter,    leaf_count,          m_binary_aabbs,
      m_binary_nodes,      m_bvh8_leaf_parents, m_bvh8_nodes,
      m_bvh8_leaf_pointers};
  convert_binary_tree_to_bvh8(params, m_device, m_stream_0);
  // populate BVH8 nodes with tailor coefficients using m2m
  // reset atomic counters to 0
  thrust::fill_n(m_stream_0_policy, m_atomic_counters, leaf_count - 1, 0);
  // one thread per leaf
  compute_internal_tailor_coefficients_m2m(
      m_bvh8_nodes, m_bvh8_internal_parent_map, m_binary_aabbs + leaf_count - 1,
      m_leaf_coefficients, m_bvh8_leaf_parents, m_bvh8_leaf_pointers,
      leaf_count, m_atomic_counters, m_stream_0);
}

void WinderBackend::initialize_point_data(const float *points,
                                          const float *normals) {
  const auto *points_v3 = reinterpret_cast<const Vec3 *>(points);
  initializeMortonCodes<Vec3>(points_v3);

  // sort by morton codes
  thrust::sequence(m_stream_0_policy, m_to_internal, m_to_internal + m_count);
  // sorts both morton_codes and m_to_internal
  thrust::sort_by_key(m_stream_0_policy, m_morton_codes,
                      m_morton_codes + m_count, m_to_internal);
  // create inverse mapping m_to_canonical
  thrust::scatter(m_stream_0_policy,
                  thrust::make_counting_iterator<uint32_t>(0),
                  thrust::make_counting_iterator<uint32_t>(m_count),
                  m_to_internal, m_to_canonical);

  interleave_gather_geometry(points, normals, m_to_internal, m_sorted_geometry,
                             m_count, m_stream_0);

  // each leaf contains 32 (LEAF_SIZE) elements
  uint32_t leaf_count = (m_count + LEAF_SIZE - 1) / LEAF_SIZE;
  {
    auto morton_leaf_stride_idx = thrust::make_transform_iterator(
        thrust::make_counting_iterator<uint32_t>(0),
        [] __host__ __device__(uint32_t i) -> uint32_t {
          return i * LEAF_SIZE;
        });
    // thrust::make_strided_iterator<LEAF_SIZE>(m_morton_codes.begin());
    auto morton_leaf_stride = thrust::make_permutation_iterator(
        m_morton_codes, morton_leaf_stride_idx);
    thrust::copy(m_stream_0_policy, morton_leaf_stride,
                 morton_leaf_stride + leaf_count, m_leaf_morton_codes);
    // build binary radix tree
    build_binary_topology(m_leaf_morton_codes, m_binary_nodes, m_binary_parents,
                          leaf_count, m_stream_0);
  }
  populate_binary_tree_aabb_and_leaf_coefficients<PointNormal>(
      reinterpret_cast<const PointNormal *>(m_sorted_geometry),
      m_leaf_coefficients, leaf_count, m_binary_nodes, m_binary_aabbs,
      m_binary_parents, m_atomic_counters, m_count, m_stream_0);
  // Convert binary LBVH tree into BVH8 tree
  ConvertBinary2BVH8Params params{
      m_bvh8_work_queue_A, m_bvh8_work_queue_B, m_bvh8_internal_parent_map,
      m_global_counter,    leaf_count,          m_binary_aabbs,
      m_binary_nodes,      m_bvh8_leaf_parents, m_bvh8_nodes,
      m_bvh8_leaf_pointers};
  convert_binary_tree_to_bvh8(params, m_device, m_stream_0);
  // populate BVH8 nodes with tailor coefficients using m2m
  // reset atomic counters to 0
  thrust::fill_n(m_stream_0_policy, m_atomic_counters, leaf_count - 1, 0);

  compute_internal_tailor_coefficients_m2m(
      m_bvh8_nodes, m_bvh8_internal_parent_map, m_binary_aabbs + leaf_count - 1,
      m_leaf_coefficients, m_bvh8_leaf_parents, m_bvh8_leaf_pointers,
      leaf_count, m_atomic_counters, m_stream_0);
}

auto WinderBackend::compute(const float *queries, size_t query_count) const
    -> CudaUniquePtr<float> {
  ScopedCudaDevice device_scope{m_device};
  const Vec3 *queries_vec3 = reinterpret_cast<const Vec3 *>(queries);
  thrust::device_vector<float> winding_numbers(query_count, 0);
  // Steps:
  // sort queries by morton code, scaled with bvh8s aabb
  // TODO arena and async everything! Careful with stream sync
  thrust::device_vector<uint32_t> queries_to_internal(query_count,
                                                      thrust::no_init);
  {
    thrust::device_vector<uint32_t> queries_morton(query_count,
                                                   thrust::no_init);

    thrust::transform(queries_vec3, queries_vec3 + query_count,
                      queries_morton.begin(), GeometryToMorton<Vec3>{});

    thrust::sequence(queries_to_internal.begin(), queries_to_internal.end());
    thrust::sort_by_key(queries_morton.begin(), queries_morton.end(),
                        queries_to_internal.begin());
  }
  // Persistent kernel with per warp work queues?
  // 1. Identify leafs in near-field
  // 2. Identify inner nodes used for far-field
  // 3. Compute near-field contribution (use regularized dipol sums from "3D
  // Reconstruction with Fast Dipole Sums for points")
  // 4. Compute far-field contribution (tensor cores)
  // 5. add contributions together
  return nullptr; // TODO
}
