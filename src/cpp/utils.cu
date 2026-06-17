#include "utils.h"
#include <cstddef>
#include <cstdio>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <vector_functions.h>

#define CUDA_CHECK(expr_to_check)                                              \
  do {                                                                         \
    cudaError_t result = expr_to_check;                                        \
    if (result != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA Runtime Error: %s:%i:%d = %s\n", __FILE__,         \
              __LINE__, result, cudaGetErrorString(result));                   \
    }                                                                          \
  } while (0)

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
