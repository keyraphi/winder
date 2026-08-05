#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

struct Mat3x3;
struct Vec3_f16;
struct Vec3_bf16;

// symmetric 3x3 matrix for tailor coefficient computation
struct SymMat3x3 {
  // 0:xx, 1:xy, 2:xz, 3:yy, 4:yz, 5:zz
  float data[6];

  __host__ __device__ __forceinline__ static auto zero() -> SymMat3x3 {
    return {0.F, 0.F, 0.F, 0.F, 0.F, 0.F};
  }

  __host__ __device__ __forceinline__ auto operator*(float s) const
      -> SymMat3x3 {
    return {data[0] * s, data[1] * s, data[2] * s,
            data[3] * s, data[4] * s, data[5] * s};
  }
  __host__ __device__ __forceinline__ SymMat3x3 &
  operator+=(const SymMat3x3 &m) {
    for (int i = 0; i < 6; i++) {
      data[i] += m.data[i];
    }
    return *this;
  }
  __host__ __device__ __forceinline__ auto operator+(const SymMat3x3 &m) const
      -> SymMat3x3 {
    SymMat3x3 result = *this;
    return result += m;
  }
  __host__ __device__ __forceinline__ SymMat3x3 &
  operator-=(const SymMat3x3 &m) {
    for (int i = 0; i < 6; i++) {
      data[i] -= m.data[i];
    }
    return *this;
  }
  __host__ __device__ __forceinline__ auto operator-(const SymMat3x3 &m) const
      -> SymMat3x3 {
    SymMat3x3 result = *this;
    return result -= m;
  }
  __host__ __device__ __forceinline__ auto trace() const -> float {
    return data[0] + data[3] + data[5];
  }
};

/**
 * @brief 3x3 FP16 Matrix structure with SIMD (half2) vector intrinsics.
 */
struct Mat3x3_f16 {
  half data[9]; // Stored in row-major order: [m00, m01, m02, m10, m11, m12,
                // m20, m21, m22]

  __host__ __device__ __forceinline__ auto operator*(half s) const
      -> Mat3x3_f16 {
    Mat3x3_f16 res;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
    half2 s2 = __half2half2(s);

    half2 p0 = __hmul2(make_half2(data[0], data[1]), s2);
    half2 p1 = __hmul2(make_half2(data[2], data[3]), s2);
    half2 p2 = __hmul2(make_half2(data[4], data[5]), s2);
    half2 p3 = __hmul2(make_half2(data[6], data[7]), s2);

    res.data[0] = p0.x;
    res.data[1] = p0.y;
    res.data[2] = p1.x;
    res.data[3] = p1.y;
    res.data[4] = p2.x;
    res.data[5] = p2.y;
    res.data[6] = p3.x;
    res.data[7] = p3.y;
    res.data[8] = __hmul(data[8], s);
#else
    for (int i = 0; i < 9; ++i) {
      res.data[i] = __hmul(data[i], s);
    }
#endif
    return res;
  }

  // Trace: Tr(M) = m00 + m11 + m22
  __host__ __device__ __forceinline__ auto trace() const -> half {
    return __hadd(__hadd(data[0], data[4]), data[8]);
  }

  // Quadratic Form: v^T * M * v
  // Packs Rows 0 and 1 to evaluate (M*v)_0 and (M*v)_1 in parallel using half2
  // FMA.
  __host__ __device__ __forceinline__ auto quadric_form(const Vec3_f16 &v) const
      -> half;

  __host__ __device__ __forceinline__ static auto from_float(const Mat3x3 &v)
      -> Mat3x3_f16;
  __host__ __device__ __forceinline__ auto operator=(const Mat3x3 &m)
      -> Mat3x3_f16;
};

/**
 * @brief 3x3 bfloat16 Matrix structure with SIMD (bfloat162) vector intrinsics.
 */
struct Mat3x3_bf16 {
  __nv_bfloat16 data[9]; // Stored in row-major order: [m00, m01, m02, m10, m11,
                         // m12, m20, m21, m22]

  __host__ __device__ __forceinline__ auto operator*(__nv_bfloat16 s) const
      -> Mat3x3_bf16 {
    Mat3x3_bf16 res;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __nv_bfloat162 s2 = __bfloat162bfloat162(s);

    __nv_bfloat162 p0 = __hmul2(make_bfloat162(data[0], data[1]), s2);
    __nv_bfloat162 p1 = __hmul2(make_bfloat162(data[2], data[3]), s2);
    __nv_bfloat162 p2 = __hmul2(make_bfloat162(data[4], data[5]), s2);
    __nv_bfloat162 p3 = __hmul2(make_bfloat162(data[6], data[7]), s2);

    res.data[0] = p0.x;
    res.data[1] = p0.y;
    res.data[2] = p1.x;
    res.data[3] = p1.y;
    res.data[4] = p2.x;
    res.data[5] = p2.y;
    res.data[6] = p3.x;
    res.data[7] = p3.y;
    res.data[8] = __hmul(data[8], s);
#else
    for (int i = 0; i < 9; ++i) {
      res.data[i] = __hmul(data[i], s);
    }
#endif
    return res;
  }
  __host__ __device__ __forceinline__ auto operator*(const Vec3_bf16 &v) const
      -> Vec3_bf16;

  // Trace: Tr(M) = m00 + m11 + m22
  __host__ __device__ __forceinline__ auto trace() const -> __nv_bfloat16 {
    return __hadd(__hadd(data[0], data[4]), data[8]);
  }

  // Quadratic Form: v^T * M * v
  // Packs Rows 0 and 1 to evaluate (M*v)_0 and (M*v)_1 in parallel using half2
  // FMA.
  __host__ __device__ __forceinline__ auto
  quadric_form(const Vec3_bf16 &v) const -> __nv_bfloat16;

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
  __host__ __device__ __forceinline__ static auto eye() -> Mat3x3 {
    return Mat3x3{1.F, 0.F, 0.F, 0.F, 1.F, 0.F, 0.F, 0.F, 1.F};
  }

  __host__ __device__ __forceinline__ static auto from_f16(const Mat3x3_f16 &m)
      -> Mat3x3 {
    Mat3x3 result;
    for (int i = 0; i < 9; i++) {
      result.data[i] = __half2float(m.data[i]);
    }
    return result;
  }

  __host__ __device__ __forceinline__ static auto
  from_bf16(const Mat3x3_bf16 &m) -> Mat3x3 {
    Mat3x3 result;
    for (int i = 0; i < 9; i++) {
      result.data[i] = __bfloat162float(m.data[i]);
    }
    return result;
  }

  __host__ __device__ __forceinline__ static auto from_sym(const SymMat3x3 &m) -> Mat3x3 {
    Mat3x3 result;
    result.data[0] = m.data[0];
    result.data[1] = m.data[1];
    result.data[2] = m.data[2];
    result.data[3] = m.data[1];
    result.data[4] = m.data[3];
    result.data[5] = m.data[4];
    result.data[6] = m.data[2];
    result.data[7] = m.data[4];
    result.data[8] = m.data[5];
    return result;
  }

  __host__ __device__ __forceinline__ auto operator=(const Mat3x3_f16 &m) {
    for (int i = 0; i < 9; i++) {
      data[i] = __half2float(m.data[i]);
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
  __host__ __device__ __forceinline__ Mat3x3 &operator-=(const Mat3x3 &m) {
    for (int i = 0; i < 9; i++) {
      data[i] -= m.data[i];
    }
    return *this;
  }
  __host__ __device__ __forceinline__ auto operator-(const Mat3x3 &m) const {
    Mat3x3 result = *this;
    return result -= m;
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
__host__ __device__ __forceinline__ auto operator*(const float f,
                                                   const SymMat3x3 &v)
    -> SymMat3x3 {
  return v * f;
}

__host__ __device__ __forceinline__ auto Mat3x3_f16::from_float(const Mat3x3 &m)
    -> Mat3x3_f16 {
  Mat3x3_f16 result;
  for (int i = 0; i < 9; i++) {
    result.data[i] = __float2half(m.data[i]);
  }
  return result;
}

__host__ __device__ __forceinline__ auto Mat3x3_f16::operator=(const Mat3x3 &m)
    -> Mat3x3_f16 {

  for (int i = 0; i < 9; i++) {
    data[i] = __float2half(m.data[i]);
  }
  return *this;
}
__host__ __device__ __forceinline__ auto
Mat3x3_bf16::from_float(const Mat3x3 &m) -> Mat3x3_bf16 {
  Mat3x3_bf16 result;
  for (int i = 0; i < 9; i++) {
    result.data[i] = __float2bfloat16(m.data[i]);
  }
  return result;
}

__host__ __device__ __forceinline__ auto Mat3x3_bf16::operator=(const Mat3x3 &m)
    -> Mat3x3_bf16 {

  for (int i = 0; i < 9; i++) {
    data[i] = __float2bfloat16(m.data[i]);
  }
  return *this;
}
