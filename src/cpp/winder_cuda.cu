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
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <vector_functions.h>
#include <vector_types.h>

#include "aabb.h"
#include "utils.h"
#include "vec3.h"
#include "winder_cuda.h"

void CudaDeleter::operator()(void *ptr) const { cudaFree(ptr); }

auto WinderBackend::get_bvh_view() -> BVH8View {
  return BVH8View{thrust::raw_pointer_cast(m_nodes.data()),
                  thrust::raw_pointer_cast(m_leaf_info.data()),
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
  m_sorted_geometry.resize(size * floats_per_elem);

  size_t leaf_count_upper_bound = size / LEAF_MIN_SIZE;
  m_nodes.resize(1.2 * leaf_count_upper_bound);
  m_leaf_info.resize(leaf_count_upper_bound);

  m_to_internal.resize(size);
  m_to_canonical.resize(size);
}

// Gather kernel that interleaves positions and normals in the sorted geometry array
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

  // We write 24 bytes as a float4 + float2.
  uint32_t dst_offset = idx * 6;
  float4 out1 = make_float4(p_xy.x, p_xy.y, p_z, n_xy.x);
  float2 out2 = make_float2(n_xy.y, n_z);

  reinterpret_cast<float4 *>(out_geometry + dst_offset)[0] = out1;
  reinterpret_cast<float2 *>(out_geometry + dst_offset + 4)[0] = out2;
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

  thrust::device_vector<uint32_t> morton_codes(m_count);
  thrust::transform(
      points_begin, points_begin + m_count, morton_codes.begin(),
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
  thrust::sort_by_key(morton_codes.begin(), morton_codes.end(),
                      m_to_internal.begin());

  const size_t threads = 256;
  const size_t block_size = (m_count + threads - 1) / threads;
  interleave_gather_geometry<<<block_size, threads>>>(points, normals, m_to_internal.data().get(), m_sorted_geometry.data().get(), m_count);
  check_launch_error("interleave_geometry");

  // TODO continue here with the LBVH construction using Karras
}
