#pragma once
#include "aabb.h"
#include "geometry.h"
#include "vec3.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <driver_types.h>

class ScopedCudaDevice {
private:
  int original_device_;

public:
  ScopedCudaDevice(int new_device);
  ~ScopedCudaDevice();

  // Disallow copying
  ScopedCudaDevice(const ScopedCudaDevice &) = delete;
  ScopedCudaDevice &operator=(const ScopedCudaDevice &) = delete;
};

namespace winder_cuda {

// CUDA memory management functions
void *cuda_allocate(size_t size);
void cuda_free(void *ptr);
void cuda_free_async(void *ptr, cudaStream_t stream);
bool cuda_memcpy(void *dest, void *src, size_t bytes);

void thrust_fill_float(float *ptr, size_t count, float value);
} // namespace winder_cuda

void reduce_async(const float *array, float *sum, uint32_t array_len,
                  cudaStream_t compute_stream);

struct SceneParams {
  float scale;
  AABB bounds;
};

__host__ __device__ __forceinline__ auto splitBy3(uint32_t v) -> uint64_t {
  // see
  // https://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/
  uint64_t x = v & 0x1fffff; // we only look at the first 21 bits
  x = (x | x << 32) &
      0x1f00000000ffff; // shift left 32 bits, OR with self, and
                        // 00011111000000000000000000000000000000001111111111111111
  x = (x | x << 16) &
      0x1f0000ff0000ff; // shift left 32 bits, OR with self, and
                        // 00011111000000000000000011111111000000000000000011111111
  x = (x | x << 8) &
      0x100f00f00f00f00f; // shift left 32 bits, OR with self, and
                          // 0001000000001111000000001111000000001111000000001111000000000000
  x = (x | x << 4) &
      0x10c30c30c30c30c3; // shift left 32 bits, OR with self, and
                          // 0001000011000011000011000011000011000011000011000011000100000000
  x = (x | x << 2) & 0x1249249249249249;
  return x;
}

__host__ __device__ __forceinline__ auto morton3D_63bit(uint32_t x, uint32_t y,
                                                        uint32_t z)
    -> uint64_t {
  // see
  // https://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/
  uint64_t answer = 0;
  answer |= splitBy3(x) | splitBy3(y) << 1 | splitBy3(z) << 2;
  return answer;
}

template <IsPrimitiveGeometry PrimitiveGeometry> struct GeometryToMorton {
  float scale;
  Vec3 min_p;

  __device__ auto operator()(const PrimitiveGeometry &g) const -> uint64_t {
    const Vec3 geometry_center = g.centroid();

    // Scale to range [0, 1]
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
    uint64_t result = morton3D_63bit(x, y, z);
    return result;
  }
};

template <IsPrimitiveGeometry PrimitiveGeometry>
auto initializeMortonCodes(const PrimitiveGeometry *geometry,
                           uint64_t *geometry_morton_codes, size_t count,
                           cudaStream_t stream,
                           SceneBounds *out_scene_bounds = nullptr) -> void;

template <IsPrimitiveGeometry PrimitiveGeometry>
auto computeSceneBounds(const PrimitiveGeometry *geometry, size_t count,
                        cudaStream_t stream = nullptr) -> SceneBounds;

template <typename T> struct GeometryTraits {
  static constexpr float default_beta = 2.3F;
  static constexpr float default_query_beta = 3.3F;
};
template <> struct GeometryTraits<Triangle> {
  static constexpr float default_beta = 2.0F;
  static constexpr float default_query_beta = 2.0F;
};
