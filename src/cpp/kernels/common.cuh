#pragma once
#include <cstdint>
#include <cstdio>
#include <cuda_runtime_api.h>

#define CUDA_CHECK(expr_to_check)                                              \
  do {                                                                         \
    cudaError_t result = expr_to_check;                                        \
    if (result != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA Runtime Error: %s:%i:%d = %s\n", __FILE__,         \
              __LINE__, result, cudaGetErrorString(result));                   \
    }                                                                          \
  } while (0)

__host__ __device__ inline auto expand_bits(uint32_t v) -> uint32_t {
  // (v<<16+v<<0)
  v = (v * 0x00010001U) & 0xFF0000FFU;
  // (v<<8+v<<0)
  v = (v * 0x00000101U) & 0x0F00F00FU;
  // (v<<4+v<<0)
  v = (v * 0x00000011U) & 0xC30C30C3U;
  // (v<<2+v<<0)
  v = (v * 0x00000005U) & 0x49249249U;
  return v;
}

__host__ __device__ inline auto morton3D_30bit(uint32_t x, uint32_t y,
                                               uint32_t z) -> uint32_t {
  return (expand_bits(x) << 2) | (expand_bits(y) << 1) | expand_bits(z);
}
