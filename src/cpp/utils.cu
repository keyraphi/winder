#include "utils.h"
#include <cstddef>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <nanobind/ndarray.h>
#include <stdexcept>
#include <string>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <vector_functions.h>

namespace nb = nanobind;

namespace cuda {
// Helper to get CUDA device from nanobind ndarray
auto get_cuda_device_from_ndarray(const void *data_ptr) -> int {
  cudaPointerAttributes attributes;
  cudaError_t result = cudaPointerGetAttributes(&attributes, data_ptr);

  if (result != cudaSuccess) {
    throw ::std::runtime_error("Failed to get CUDA pointer attributes");
  }

  // attributes.device contains the device ID where the memory is allocated
  return attributes.device;
}

void *cuda_allocate(size_t size) {
  void *ptr = nullptr;
  cudaError_t err = cudaMalloc(&ptr, size);
  if (err != cudaSuccess) {
    return nullptr;
  }
  return ptr;
}

void cuda_free(void *ptr) {
  if (ptr != nullptr) {
    cudaFree(ptr);
  }
}

Scalar_t scalar_with_default(const ::std::optional<Scalar_t> &maybe_pc_wn, size_t count,
                             float default_value) {
  if (maybe_pc_wn.has_value()) {
    return maybe_pc_wn.value();
  }
  auto *data = static_cast<float *>(cuda_allocate(count * sizeof(float)));
  thrust::device_ptr<float> data_ptr(data);
  thrust::fill(data_ptr, data_ptr + count, default_value);
  nb::capsule owner(data, [](void *p) noexcept -> void { cuda_free(p); });
  return {data, {count}, owner};
}

} // namespace cuda

void check_launch_error(const std::string &kernel_name) {
  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    printf("%s: Launch Error: %s\n", kernel_name.c_str(),
           cudaGetErrorString(launch_err));
  }
}

