#include "aabb.h"
#include "geometry.h"
#include "utils.h"
#include "vec3.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/system/cuda/detail/execution_policy.h>
#include <vector_functions.h>

#define CUDA_CHECK(expr_to_check)                                              \
  do {                                                                         \
    cudaError_t result = expr_to_check;                                        \
    if (result != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA Runtime Error: %s:%i:%d = %s\n", __FILE__,         \
              __LINE__, result, cudaGetErrorString(result));                   \
    }                                                                          \
  } while (0)

ScopedCudaDevice::ScopedCudaDevice(int new_device) {
  cudaGetDevice(&original_device_);
  cudaSetDevice(new_device);
}
ScopedCudaDevice::~ScopedCudaDevice() { cudaSetDevice(original_device_); }

namespace winder_cuda {
// Helper to get CUDA device from nanobind ndarray
auto get_cuda_device_from_ndarray(const void *data_ptr) -> int {
  cudaPointerAttributes attributes;
  CUDA_CHECK(cudaPointerGetAttributes(&attributes, data_ptr));

  // attributes.device contains the device ID where the memory is allocated
  return attributes.device;
}

void *cuda_allocate(size_t size) {
  void *ptr = nullptr;
  CUDA_CHECK(cudaMalloc(&ptr, size));
  return ptr;
}

void cuda_free(void *ptr) {
  if (ptr != nullptr) {
    CUDA_CHECK(cudaFree(ptr));
  }
}
void cuda_free_async(void *ptr, cudaStream_t stream) {
  if (ptr != nullptr) {
    CUDA_CHECK(cudaFreeAsync(ptr, stream));
  }
}

void thrust_fill_float(float *ptr, size_t count, float value) {
  thrust::device_ptr<float> dev_ptr(ptr);
  thrust::fill(dev_ptr, dev_ptr + count, value);
}

} // namespace winder_cuda

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

__constant__ SceneParams d_scene_params;

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

template <IsPrimitiveGeometry PrimitiveGeometry>
auto initializeMortonCodes(const PrimitiveGeometry *geometry,
                           uint64_t *geometry_morton_codes, const size_t count,
                           cudaStream_t stream) -> void {

  auto build_stream_policy = thrust::cuda::par.on(stream);

  // compute scene bound
  auto aabb_transform = thrust::make_transform_iterator(
      geometry, GeometryToAABB<PrimitiveGeometry>{});

  AABB scene_bounds =
      thrust::reduce(build_stream_policy, aabb_transform,
                     aabb_transform + count, AABB::empty(), MergeAABB{});
  // create morton codes for each primitive
  Vec3 extent = Vec3::from_f16(scene_bounds.diagonal());
  float max_dim = fmaxf(extent.x, fmaxf(extent.y, extent.z));
  float scale = (max_dim > 1e-9F) ? 1.F / max_dim : 0.F;

  SceneParams scen_params{scale, scene_bounds};

  CUDA_CHECK(cudaMemcpyToSymbolAsync(d_scene_params, &scen_params,
                                     sizeof(SceneParams), 0,
                                     cudaMemcpyHostToDevice, stream));

  thrust::transform(build_stream_policy, geometry, geometry + count,
                    geometry_morton_codes,
                    GeometryToMorton<PrimitiveGeometry>{});
}
