#pragma once

#include "mat3x3.h"
#include "vec3.h"
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

struct Tensor3_bf16 {
  nv_bfloat16 data[27];
};
struct Tensor3 {
  float data[27];

  __host__ static auto from_outer_product(const Mat3x3 &m,
                                                     const Vec3 &v) -> Tensor3 {
    Tensor3 result;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        int idx = i * 3 + j;
        result.data[idx * 3] = m.data[idx] * v.x;
        result.data[idx * 3 + 1] = m.data[idx] * v.y;
        result.data[idx * 3 + 2] = m.data[idx] * v.z;
      }
    }
    return result;
  }
};

struct Tensor3_compressed;
// compressed tensor for second order coefficients
struct Tensor3_bf16_compressed {
  nv_bfloat16 data[18];

  __host__ __device__ __forceinline__ static auto
  from_float(const Tensor3_compressed &t) -> Tensor3_bf16_compressed;

  // get a full 3x3x3 tensor
  __host__ __device__ inline auto uncompress() const -> Tensor3_bf16 {
    Tensor3_bf16 result;
    result.data[0] = data[0];
    result.data[1] = data[1];
    result.data[2] = data[2];
    result.data[3] = data[3];
    result.data[4] = data[4];
    result.data[5] = data[5];
    result.data[6] = data[6];
    result.data[7] = data[7];
    result.data[8] = data[8];
    result.data[9] = data[3];
    result.data[10] = data[4];
    result.data[11] = data[5];
    result.data[12] = data[9];
    result.data[13] = data[10];
    result.data[14] = data[11];
    result.data[15] = data[12];
    result.data[16] = data[13];
    result.data[17] = data[14];
    result.data[18] = data[6];
    result.data[19] = data[7];
    result.data[20] = data[8];
    result.data[21] = data[12];
    result.data[22] = data[13];
    result.data[23] = data[14];
    result.data[24] = data[15];
    result.data[25] = data[16];
    result.data[26] = data[17];
    return result;
  }

  __host__ __device__ __forceinline__ auto
  operator=(const Tensor3_compressed &t) -> Tensor3_bf16_compressed;
};

struct Tensor3_compressed {
  float data[18];

  __host__ __device__ __forceinline__ static auto
  from_bf16(const Tensor3_bf16_compressed &t) -> Tensor3_compressed {
    Tensor3_compressed result;
    for (int i = 0; i < 18; ++i) {
      result.data[i] = t.data[i];
    }
    return result;
  }
  __host__ __device__ __forceinline__ auto
  operator=(const Tensor3_bf16_compressed &t) {
    for (int i = 0; i < 18; ++i) {
      data[i] = t.data[i];
    }
  }

  __host__ __device__ __forceinline__ Tensor3_compressed &
  operator+=(const Tensor3_compressed &t) {
    for (int i = 0; i < 18; i++) {
      data[i] += t.data[i];
    }
    return *this;
  }

  __host__ __device__ inline auto uncompress() const -> Tensor3 {
    Tensor3 result;
    result.data[0] = data[0];   // xxx = data[0]
    result.data[1] = data[1];   // xxy = data[1]
    result.data[2] = data[2];   // xxz = data[2]
    result.data[3] = data[3];   // xyx = data[3]
    result.data[4] = data[4];   // xyy = data[4]
    result.data[5] = data[5];   // xyz = data[5]
    result.data[6] = data[6];   // xzx = data[6]
    result.data[7] = data[7];   // xzy = data[7]
    result.data[8] = data[8];   // xzz = data[8]
    result.data[9] = data[3];   // yxx = data[3]
    result.data[10] = data[4];  // yxy = data[4]
    result.data[11] = data[5];  // yxz = data[5]
    result.data[12] = data[9];  // yyx = data[9]
    result.data[13] = data[10]; // yyy = data[10]
    result.data[14] = data[11]; // yyz = data[11]
    result.data[15] = data[12]; // yzx = data[12]
    result.data[16] = data[13]; // yzy = data[13]
    result.data[17] = data[14]; // yzz = data[14]
    result.data[18] = data[6];  // zxx = data[6]
    result.data[19] = data[7];  // zxy = data[7]
    result.data[20] = data[8];  // zxz = data[8]
    result.data[21] = data[12]; // zyx = data[12]
    result.data[22] = data[13]; // zyy = data[13]
    result.data[23] = data[14]; // zyz = data[14]
    result.data[24] = data[15]; // zzx = data[15]
    result.data[25] = data[16]; // zzy = data[16]
    result.data[26] = data[17]; // zzz  = data[17]
    return result;
  }
};

__host__ __device__ __forceinline__ static auto
from_float(const Tensor3_compressed &t) -> Tensor3_bf16_compressed {
  Tensor3_bf16_compressed result;
  for (int i = 0; i < 18; ++i) {
    result.data[i] = t.data[i];
  }
  return result;
}

__host__ __device__ __forceinline__ auto
Tensor3_bf16_compressed::operator=(const Tensor3_compressed &t)
    -> Tensor3_bf16_compressed {
  for (int i = 0; i < 18; ++i) {
    data[i] = t.data[i];
  }
  return *this;
}

__host__ __device__ __forceinline__ auto
Tensor3_bf16_compressed::from_float(const Tensor3_compressed &t)
    -> Tensor3_bf16_compressed {
  Tensor3_bf16_compressed result;
  for (int i = 0; i < 18; ++i) {
    result.data[i] = t.data[i];
  }
  return result;
}
