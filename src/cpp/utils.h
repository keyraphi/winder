#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <optional>
#include <string>

namespace winder_cuda {

// Forward declarations
class ScopedCudaDevice;

// CUDA memory management functions
void *cuda_allocate(size_t size);
void cuda_free(void *ptr);
bool cuda_memcpy(void *dest, void *src, size_t bytes);

void thrust_fill_float(float *ptr, size_t count, float value);
} // namespace cuda
