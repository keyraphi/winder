#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

struct Mat3x3;
struct Tensor3;

struct Mat3x3_bf16 {
  nv_bfloat16 data[9];

  __host__ __device__ __forceinline__ static auto from_float(const Mat3x3 &v)
      -> Mat3x3_bf16;
  __host__ __device__ __forceinline__ auto operator=(const Mat3x3 &m)
      -> Mat3x3_bf16;
};

struct Mat3x3 {
  float data[9];

  __host__ __device__ __forceinline__ static auto zero() -> Mat3x3 {
    return Mat3x3{0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F};
  }

  __host__ __device__ __forceinline__ static auto
  from_bf16(const Mat3x3_bf16 &m) -> Mat3x3 {
    Mat3x3 result;
    for (int i = 0; i < 9; i++) {
      result.data[i] = m.data[i];
    }
    return result;
  }

  __host__ __device__ __forceinline__ auto operator=(const Mat3x3_bf16 &m) {
    for (int i = 0; i < 9; i++) {
      data[i] = m.data[i];
    }
  }
  __host__ __device__ __forceinline__ Mat3x3 &operator+=(const Mat3x3 &m) {
    for (int i = 0; i < 9; i++) {
      data[i] += m.data[i];
    }
    return *this;
  }
  __host__ __device__ __forceinline__ auto operator+(const Mat3x3 &m) const {
    Mat3x3 result = *this;
    return result += m;
  }
  __host__ __device__ __forceinline__ auto operator*(const float f) const {
    Mat3x3 result;
    for (int i = 0; i < 9; i++) {
      result.data[i] = data[i] * f;
    }
    return result;
  }

};

__host__ __device__ __forceinline__ auto operator*(const float n,
                                                   const Mat3x3 &m) -> Mat3x3 {
  return m * n;
}

__host__ __device__ __forceinline__ auto
Mat3x3_bf16::from_float(const Mat3x3 &m) -> Mat3x3_bf16 {
  Mat3x3_bf16 result;
  for (int i = 0; i < 9; i++) {
    result.data[i] = m.data[i];
  }
  return result;
}

__host__ __device__ __forceinline__ auto Mat3x3_bf16::operator=(const Mat3x3 &m)
    -> Mat3x3_bf16 {

  for (int i = 0; i < 9; i++) {
    data[i] = m.data[i];
  }
  return *this;
}
