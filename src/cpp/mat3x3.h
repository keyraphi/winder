#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

struct Mat3x3_bf16 {
  nv_bfloat16 data[9];
};

struct Mat3x3 {
  float data[9];

  __host__ __device__ __forceinline__ static auto from_bf16(const Mat3x3_bf16 &m)
      -> Mat3x3 {
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
};
