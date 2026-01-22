#include "utils.h"
#include <cstddef>
#include <cstdio>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <nanobind/ndarray.h>
#include <optional>
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

namespace nb = nanobind;

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

Scalar_t scalar_with_default(const ::std::optional<Scalar_t> &maybe_pc_wn,
                             size_t count, float default_value) {
  if (maybe_pc_wn.has_value()) {
    return maybe_pc_wn.value();
  }
  auto *data = static_cast<float *>(cuda_allocate(count * sizeof(float)));
  thrust::fill(data, data + count, default_value);
  nb::capsule owner(data, [](void *p) noexcept -> void { cuda_free(p); });
  return {data, {count}, owner};
}

} // namespace cuda
