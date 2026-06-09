#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <cstdio>

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

__host__ __device__ inline auto splitBy3(uint32_t v) -> uint64_t {
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

__host__ __device__ inline uint64_t morton3D_63bit(uint32_t x, uint32_t y, uint32_t z) {
  // see
  // https://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/
  uint64_t answer = 0;
  answer |= splitBy3(x) | splitBy3(y) << 1 | splitBy3(z) << 2;
  return answer;
}

template <typename T>
struct SoAViewConst {
  const float *base_ptr;
  size_t stride;
};

template <typename T>
struct SoAView {
  float *base_ptr;
  size_t stride;
};

