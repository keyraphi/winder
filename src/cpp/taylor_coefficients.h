#pragma once

#include "mat3x3.h"
#include "tensor3.h"
#include "vec3.h"
#include <cstddef>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <vector_types.h>

// For leaf nodes
// 60 byte aligned to 64 byte
struct alignas(64) TaylorCoefficientsF16 {
  Vec3_f16 zero_order;
  Mat3x3_f16 first_order;
  Tensor3_bf16_compressed second_order;
};

// For m2m
// 120 byte aligned to 128 bytes
struct alignas(128) TaylorCoefficients {
  Vec3 zero_order;
  Mat3x3 first_order;
  Tensor3_compressed second_order;

  __host__ __device__ static auto from_f16(const TaylorCoefficientsF16 &t)
      -> TaylorCoefficients {
    TaylorCoefficients result;
    result.zero_order = Vec3::from_f16(t.zero_order);
    result.first_order = Mat3x3::from_f16(t.first_order);
    result.second_order = Tensor3_compressed::from_bf16(t.second_order);
    return result;
  }
};

template <uint32_t D> struct alignas(64) PayloadTaylorCoefficientsF16 {
  // We need to store 30 value per dimension
  // zero order: 3, first_order: 9, second order: 18
  static constexpr uint32_t kSlots = 30;
  uint16_t data[kSlots][D];

  __device__ __forceinline__ auto getZeroOrder(uint32_t d) const -> Vec3_f16 {
    return Vec3_f16{
        .x = __ushort_as_half(data[0][d]),
        .y = __ushort_as_half(data[1][d]),
        .z = __ushort_as_half(data[2][d]),
    };
  }

  __device__ __forceinline__ auto getFirstOrder(uint32_t d) const -> Mat3x3_f16 {
    Mat3x3_f16 result;
#pragma unroll
    for (int i = 0; i < 9; ++i) {
      result.data[i] = __ushort_as_half(data[3 + i][d]);
    }
    return result;
  }

  __device__ __forceinline__ auto
  getSecondOrder(uint32_t d) const -> Tensor3_bf16_compressed {
    Tensor3_bf16_compressed result;
#pragma unroll
    for (int i = 0; i < 18; ++i) {
      result.data[i] = __ushort_as_bfloat16(data[12 + i][d]);
    }
    return result;
  }

  __device__ __forceinline__ void set_zero_order(uint32_t d,
                                                 const Vec3_f16 &c) {
    data[0][d] = __half_as_ushort(c.x);
    data[1][d] = __half_as_ushort(c.y);
    data[2][d] = __half_as_ushort(c.z);
  }
  __device__ __forceinline__ void set_first_order(uint32_t d,
                                                  const Mat3x3_f16 &m) {
#pragma unroll
    for (int i = 0; i < 9; ++i) {
      data[3 + i][d] = __half_as_ushort(m.data[i]);
    }
  }
  __device__ __forceinline__ void
  set_second_order(uint32_t d, const Tensor3_bf16_compressed &t) {
#pragma unroll
    for (int i = 0; i < 18; ++i) {
      data[12 + i][d] = __bfloat16_as_ushort(t.data[i]);
    }
  }
};

// M2M
template <uint32_t D> struct alignas(128) PayloadTaylorCoefficients {
  static constexpr uint32_t kSlots = 30;
  float data[kSlots][D];

  __device__ __forceinline__ auto getZeroOrder(uint32_t d) const -> Vec3 {
    return Vec3{data[0][d], data[1][d], data[2][d]};
  }
  __device__ __forceinline__ auto getFirstOrder(uint32_t d) const -> Mat3x3 {
    Mat3x3 m;
#pragma unroll
    for (int i = 0; i < 9; ++i)
      m.data[i] = data[3 + i][d];
    return m;
  }
  __device__ __forceinline__ auto getSecondOrder(uint32_t d) const
      -> Tensor3_compressed {
    Tensor3_compressed t;
#pragma unroll
    for (int i = 0; i < 18; ++i)
      t.data[i] = data[12 + i][d];
    return t;
  }
  __device__ __forceinline__ void set_zero_order(uint32_t d, const Vec3 &c) {
    data[0][d] = c.x;
    data[1][d] = c.y;
    data[2][d] = c.z;
  }
  __device__ __forceinline__ void set_first_order(uint32_t d, const Mat3x3 &m) {
#pragma unroll
    for (int i = 0; i < 9; ++i)
      data[3 + i][d] = m.data[i];
  }
  __device__ __forceinline__ void
  set_second_order(uint32_t d, const Tensor3_compressed &t) {
#pragma unroll
    for (int i = 0; i < 18; ++i)
      data[12 + i][d] = t.data[i];
  }
};

// ###### BACKWARD ########################
// 40 byte
struct BackwardTaylorCoefficients {
  float zero_order;
  Vec3 first_order;
  SymMat3x3 second_order;
};
