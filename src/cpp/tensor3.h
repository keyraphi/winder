#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

struct Tensor3_bf16 {
  nv_bfloat16 data[27];
};
struct Tensor3 {
  float data[27];
};

struct Tensor3_compressed;
// compressed tensor for second order coefficients
struct Tensor3_bf16_compressed {
  nv_bfloat16 data[18];

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
  operator=(const Tensor3_compressed &t);
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

  __host__ __device__ inline auto uncompress() const -> Tensor3 {
    Tensor3 result;
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
};

__host__ __device__ __forceinline__ auto
Tensor3_bf16_compressed::operator=(const Tensor3_compressed &t) {
  for (int i = 0; i < 18; ++i) {
    data[i] = t.data[i];
  }
}
