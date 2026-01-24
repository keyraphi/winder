#pragma once

#include "mat3x3.h"
#include "tensor3.h"
#include "vec3.h"
#include <cstdint>
#include <cuda_runtime_api.h>
#include <vector_types.h>

// tailor coefficients quantized to 11 bit each for inner BVH8 node
// with a shared 8 bit exponent
// first 4 bytes are reused for parent index
// cofficients: zero order (3), first order (9), second order compressed (18)
// total: 30 coefficients
// 44 bytes
struct TailorCoefficientsQuantized {
  uint32_t tailor_data[11];

  // During tree construction we use the tailor_data memory to temporarily store
  // how many children the node actually has.
  __device__ inline auto get_expected_children() const -> uint32_t;
  __device__ inline void set_expected_children(uint32_t idx);

  __device__ inline auto get_shared_scale_factor() const -> float;

  __device__ inline void
  set_tailor_coefficients(const Vec3 &zero_order, const Mat3x3 &first_order,
                          const Tensor3_compressed &second_order);

  __device__ inline auto
  get_tailor_zero_order(const float shared_scale_factor) const -> Vec3_bf16;

  __device__ inline auto
  get_tailor_first_order(const float shared_scale_factor) const -> Mat3x3_bf16;

  __device__ inline auto
  get_tailor_second_order(const float shared_scale_factor) const
      -> Tensor3_bf16_compressed;
};

// For leaf nodes
struct TailorCoefficientsBf16 {
  Vec3_bf16 zero_order;
  Mat3x3_bf16 first_order;
  Tensor3_bf16_compressed second_order;
};
