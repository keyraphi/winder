#pragma once
#include "aabb.h"
#include "geometry.h"
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

struct SceneParams {
  float scale;
  AABB bounds;
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
};
template <> struct GeometryTraits<Triangle> {
  static constexpr float default_beta = 2.0F;
};
