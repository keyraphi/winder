#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <memory>
#include "aabb.h"
#include "geometry.h"

struct CudaDeleter {
  size_t stream = 0; // Plain integer data type, safe for pure C++

  // Constructor to make initialization clean
  explicit CudaDeleter(size_t stream_ptr = 0) : stream(stream_ptr) {}

  void operator()(void *ptr) const;
};

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

template <typename T> using CudaUniquePtr = std::unique_ptr<T[], CudaDeleter>;

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
                           cudaStream_t stream) -> void;

template <typename T> struct GeometryTraits {
  static constexpr float default_beta = 2.3F;
};
template <> struct GeometryTraits<PointNormal> {
  static constexpr float default_beta = 2.0F;
};
