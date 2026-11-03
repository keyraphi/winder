#pragma once
#include <algorithm>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#define CUDA_CHECK(expr_to_check)                                              \
  do {                                                                         \
    cudaError_t result = expr_to_check;                                        \
    if (result != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA Runtime Error: %s:%i:%d = %s\n", __FILE__,         \
              __LINE__, result, cudaGetErrorString(result));                   \
      std::abort();                                                            \
    }                                                                          \
  } while (0)


__host__ __device__ __forceinline__ auto
quantize_value(float value, float p_min, float p_inv_ext) {
#ifdef __CUDA_ARCH__
  float normalized = __saturatef((value - p_min) * p_inv_ext) * 255.F;
  return (uint8_t)__float2uint_rn(normalized);
#else
  float val = (value - p_min) * p_inv_ext;
  float normalized = std::max(0.F, std::min(1.F, val)) * 255.F;
  return static_cast<uint8_t>(normalized + 0.5F); // round-to-nearest
#endif
}

