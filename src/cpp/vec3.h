#pragma once
#include "mat3x3.h"
#include "soa.h"
#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <format>
#include <string>
#include <vector_types.h>

struct Vec3;
struct AABB;

struct Vec3_f16 {
  half x, y, z;

  __host__ __device__ __forceinline__ static auto from_float(const Vec3 &v)
      -> Vec3_f16;

  // --- Vector Addition ---
  __host__ __device__ __forceinline__ auto operator+(const Vec3_f16 &b) const
      -> Vec3_f16 {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
    half2 xy = __hadd2(__halves2half2(x, y), __halves2half2(b.x, b.y));
    return {__low2half(xy), __high2half(xy), __hadd(z, b.z)};
#else
    return {x + b.x, y + b.y, z + b.z};
#endif
  }

  // --- Vector Subtraction ---
  __host__ __device__ __forceinline__ auto operator-(const Vec3_f16 &b) const
      -> Vec3_f16 {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
    half2 xy = __hsub2(__halves2half2(x, y), __halves2half2(b.x, b.y));
    return {__low2half(xy), __high2half(xy), __hsub(z, b.z)};
#else
    return {x - b.x, y - b.y, z - b.z};
#endif
  }

  // --- Elementwise Vector Multiplication ---
  __host__ __device__ __forceinline__ auto operator*(const Vec3_f16 &b) const
      -> Vec3_f16 {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
    half2 xy = __hmul2(__halves2half2(x, y), __halves2half2(b.x, b.y));
    return {__low2half(xy), __high2half(xy), __hmul(z, b.z)};
#else
    return {x * b.x, y * b.y, z * b.z};
#endif
  }

  // --- Scalar Multiplication ---
  __host__ __device__ __forceinline__ auto operator*(half s) const -> Vec3_f16 {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
    half2 xy = __hmul2(__halves2half2(x, y), __halves2half2(s, s));
    return {__low2half(xy), __high2half(xy), __hmul(z, s)};
#else
    return {x * s, y * s, z * s};
#endif
  }

  // --- Unary Negation ---
  __host__ __device__ __forceinline__ auto operator-() const -> Vec3_f16 {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
    half2 xy = __hneg2(__halves2half2(x, y));
    return {__low2half(xy), __high2half(xy), __hneg(z)};
#else
    return {-x, -y, -z};
#endif
  }

  // --- Dot Product ---
  __host__ __device__ __forceinline__ auto dot(const Vec3_f16 &b) const
      -> half {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
    half2 prod_xy = __hmul2(__halves2half2(x, y), __halves2half2(b.x, b.y));
    return __hadd(__hadd(__low2half(prod_xy), __high2half(prod_xy)),
                  __hmul(z, b.z));
#else
    return x * b.x + y * b.y + z * b.z;
#endif
  }

  // --- Cross Product ---
  __host__ __device__ __forceinline__ auto cross(const Vec3_f16 &b) const
      -> Vec3_f16 {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
    half2 a_yz = __halves2half2(y, z);
    half2 b_zy = __halves2half2(b.z, b.y);
    half2 p1 = __hmul2(a_yz, b_zy);
    half cx = __hsub(__low2half(p1), __high2half(p1));

    half2 a_zx = __halves2half2(z, x);
    half2 b_xz = __halves2half2(b.x, b.z);
    half2 p2 = __hmul2(a_zx, b_xz);
    half cy = __hsub(__low2half(p2), __high2half(p2));

    half2 a_xy = __halves2half2(x, y);
    half2 b_yx = __halves2half2(b.y, b.x);
    half2 p3 = __hmul2(a_xy, b_yx);
    half cz = __hsub(__low2half(p3), __high2half(p3));

    return {cx, cy, cz};
#else
    return {y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x};
#endif
  }

  // Fused Multiply-Add (v * s + a)
  __host__ __device__ __forceinline__ static auto fma(const Vec3_f16 &v, half s,
                                                      const Vec3_f16 &a)
      -> Vec3_f16 {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
    half2 v_xy = __halves2half2(v.x, v.y);
    half2 a_xy = __halves2half2(a.x, a.y);
    half2 s2 = __halves2half2(s, s);
    half2 res_xy = __hfma2(v_xy, s2, a_xy);
    return {__low2half(res_xy), __high2half(res_xy), __hfma(v.z, s, a.z)};
#else
    return {v.x * s + a.x, v.y * s + a.y, v.z * s + a.z};
#endif
  }

  __host__ __device__ __forceinline__ auto length2() const -> half {
    return dot(*this);
  }

  __device__ __forceinline__ auto length() const -> half {
#if defined(__CUDACC__) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
    return hsqrt(length2());
#else
    return sqrtf(float(length2()));
#endif
  }

  __host__ __device__ __forceinline__ auto
  outer_product(const Vec3_f16 &b) const -> Mat3x3_f16 {
    Mat3x3_f16 m;
    m.data[0] = x * b.x;
    m.data[1] = x * b.y;
    m.data[2] = x * b.z;
    m.data[3] = y * b.x;
    m.data[4] = y * b.y;
    m.data[5] = y * b.z;
    m.data[6] = z * b.x;
    m.data[7] = z * b.y;
    m.data[8] = z * b.z;
    return m;
  }

  __host__ __device__ __forceinline__ auto operator=(const Vec3 &v) -> Vec3_f16;
};

struct Vec3_bf16 {
  __nv_bfloat16 x, y, z;

  __host__ __device__ __forceinline__ static auto from_f16(const Vec3_f16 &v)
      -> Vec3_bf16 {
    Vec3_bf16 result;
    result.x = __nv_bfloat16(v.x);
    result.y = __nv_bfloat16(v.y);
    result.z = __nv_bfloat16(v.z);
    return result;
  }
  __host__ __device__ __forceinline__ static auto from_float(const Vec3 &v)
      -> Vec3_bf16;

  // --- Vector Addition ---
  __host__ __device__ __forceinline__ auto operator+(const Vec3_bf16 &b) const
      -> Vec3_bf16 {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __nv_bfloat162 xy =
        __hadd2(__halves2bfloat162(x, y), __halves2bfloat162(b.x, b.y));
    return {__low2bfloat16(xy), __high2bfloat16(xy), __hadd(z, b.z)};
#else
    return {x + b.x, y + b.y, z + b.z};
#endif
  }

  // --- Vector Subtraction ---
  __host__ __device__ __forceinline__ auto operator-(const Vec3_bf16 &b) const
      -> Vec3_bf16 {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __nv_bfloat162 xy =
        __hsub2(__halves2bfloat162(x, y), __halves2bfloat162(b.x, b.y));
    return {__low2bfloat16(xy), __high2bfloat16(xy), __hsub(z, b.z)};
#else
    return {x - b.x, y - b.y, z - b.z};
#endif
  }

  // --- Elementwise Vector Multiplication ---
  __host__ __device__ __forceinline__ auto operator*(const Vec3_bf16 &b) const
      -> Vec3_bf16 {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __nv_bfloat162 xy =
        __hmul2(__halves2bfloat162(x, y), __halves2bfloat162(b.x, b.y));
    return {__low2bfloat16(xy), __high2bfloat16(xy), __hmul(z, b.z)};
#else
    return {x * b.x, y * b.y, z * b.z};
#endif
  }

  // --- Scalar Multiplication ---
  __host__ __device__ __forceinline__ auto operator*(__nv_bfloat16 s) const
      -> Vec3_bf16 {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __nv_bfloat162 xy =
        __hmul2(__halves2bfloat162(x, y), __bfloat162bfloat162(s));
    return {__low2bfloat16(xy), __high2bfloat16(xy), __hmul(z, s)};
#else
    return {x * s, y * s, z * s};
#endif
  }

  // --- Unary Negation ---
  __host__ __device__ __forceinline__ auto operator-() const -> Vec3_bf16 {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __nv_bfloat162 xy = __hneg2(__halves2bfloat162(x, y));
    return {__low2bfloat16(xy), __high2bfloat16(xy), __hneg(z)};
#else
    return {-x, -y, -z};
#endif
  }

  // --- Dot Product ---
  __host__ __device__ __forceinline__ auto dot(const Vec3_bf16 &b) const
      -> __nv_bfloat16 {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __nv_bfloat162 prod_xy =
        __hmul2(__halves2bfloat162(x, y), __halves2bfloat162(b.x, b.y));
    return __hadd(__hadd(__low2bfloat16(prod_xy), __high2bfloat16(prod_xy)),
                  __hmul(z, b.z));
#else
    return x * b.x + y * b.y + z * b.z;
#endif
  }

  // --- Cross Product ---
  __host__ __device__ __forceinline__ auto cross(const Vec3_bf16 &b) const
      -> Vec3_bf16 {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __nv_bfloat162 a_yz = __halves2bfloat162(y, z);
    __nv_bfloat162 b_zy = __halves2bfloat162(b.z, b.y);
    __nv_bfloat162 p1 = __hmul2(a_yz, b_zy);
    __nv_bfloat16 cx = __hsub(__low2bfloat16(p1), __high2bfloat16(p1));

    __nv_bfloat162 a_zx = __halves2bfloat162(z, x);
    __nv_bfloat162 b_xz = __halves2bfloat162(b.x, b.z);
    __nv_bfloat162 p2 = __hmul2(a_zx, b_xz);
    __nv_bfloat16 cy = __hsub(__low2bfloat16(p2), __high2bfloat16(p2));

    __nv_bfloat162 a_xy = __halves2bfloat162(x, y);
    __nv_bfloat162 b_yx = __halves2bfloat162(b.y, b.x);
    __nv_bfloat162 p3 = __hmul2(a_xy, b_yx);
    __nv_bfloat16 cz = __hsub(__low2bfloat16(p3), __high2bfloat16(p3));

    return {cx, cy, cz};
#else
    return {y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x};
#endif
  }

  // Fused Multiply-Add (v * s + a)
  __host__ __device__ __forceinline__ static auto
  fma(const Vec3_bf16 &v, __nv_bfloat16 s, const Vec3_bf16 &a) -> Vec3_bf16 {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __nv_bfloat162 v_xy = __halves2bfloat162(v.x, v.y);
    __nv_bfloat162 a_xy = __halves2bfloat162(a.x, a.y);
    __nv_bfloat162 s2 = __halves2bfloat162(s, s);
    __nv_bfloat162 res_xy = __hfma2(v_xy, s2, a_xy);
    return {__low2bfloat16(res_xy), __high2bfloat16(res_xy),
            __hfma(v.z, s, a.z)};
#else
    return {v.x * s + a.x, v.y * s + a.y, v.z * s + a.z};
#endif
  }

  __host__ __device__ __forceinline__ auto length2() const -> __nv_bfloat16 {
    return dot(*this);
  }

  __device__ __forceinline__ auto length() const -> __nv_bfloat16 {
#if defined(__CUDACC__) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return hsqrt(length2());
#else
    return sqrtf(float(length2()));
#endif
  }

  __host__ __device__ __forceinline__ auto
  outer_product(const Vec3_bf16 &b) const -> Mat3x3_bf16 {
    Mat3x3_bf16 m;
    m.data[0] = x * b.x;
    m.data[1] = x * b.y;
    m.data[2] = x * b.z;
    m.data[3] = y * b.x;
    m.data[4] = y * b.y;
    m.data[5] = y * b.z;
    m.data[6] = z * b.x;
    m.data[7] = z * b.y;
    m.data[8] = z * b.z;
    return m;
  }

  __host__ __device__ __forceinline__ auto operator=(const Vec3 &v)
      -> Vec3_bf16;
};

struct Vec3 {
  float x, y, z;

  __host__ __device__ __forceinline__ static auto from_f16(const Vec3_f16 &v)
      -> Vec3 {
    Vec3 result;
    result.x = __half2float(v.x);
    result.y = __half2float(v.y);
    result.z = __half2float(v.z);
    return result;
  }

  __host__ __device__ __forceinline__ static auto zero() -> Vec3 {
    Vec3 result;
    result.x = 0.F;
    result.y = 0.F;
    result.z = 0.F;
    return result;
  }

  __host__ __device__ __forceinline__ auto dot(const Vec3 &v) const {
    return x * v.x + y * v.y + z * v.z;
  }

  __host__ __device__ __forceinline__ auto get_aabb() const -> AABB;

  __host__ __device__ __forceinline__ auto centroid() const -> Vec3 {
    return *this;
  }

  __host__ __device__ __forceinline__ auto length2() const -> float {
    return x * x + y * y + z * z;
  }
  __host__ __device__ __forceinline__ auto length() const -> float {
    return sqrtf(length2());
  }
  __host__ __device__ __forceinline__ auto inv_length() const -> float {

#if defined(__CUDA_ARCH__)
    return rsqrtf(length2());
#else
    return 1.F / length();
#endif
  }

  __host__ __device__ __forceinline__ auto
  max_distance_to(const Vec3 &pos) const -> float {
    return (*this - pos).length();
  }

  __host__ __device__ __forceinline__ static auto
  load(const SoAView<Vec3> &view, uint32_t idx, uint32_t count) -> Vec3 {
    if (idx < count) {
      return Vec3{.x = view.base_ptr[0 * view.stride + idx],
                  .y = view.base_ptr[1 * view.stride + idx],
                  .z = view.base_ptr[2 * view.stride + idx]};
    }
    return Vec3{0.F, 0.F, 0.F};
  }

  __host__ __device__ __forceinline__ auto operator+(const Vec3 &b) const
      -> Vec3 {
    return {x + b.x, y + b.y, z + b.z};
  }
  __host__ __device__ __forceinline__ auto operator+=(const Vec3 &b) -> Vec3 & {
    x += b.x;
    y += b.y;
    z += b.z;
    return *this;
  }
  __host__ __device__ __forceinline__ auto operator*=(const float f) -> Vec3 & {
    x *= f;
    y *= f;
    z *= f;
    return *this;
  }
  __host__ __device__ __forceinline__ auto operator+=(const Vec3_f16 &b)
      -> Vec3 & {
    return *this += Vec3::from_f16(b);
  }
  __host__ __device__ __forceinline__ auto operator-(const Vec3 &b) const
      -> Vec3 {
    return {x - b.x, y - b.y, z - b.z};
  }
  __host__ __device__ __forceinline__ auto operator-() const -> Vec3 {
    return {-x, -y, -z};
  }
  __host__ __device__ __forceinline__ auto operator*(const Vec3 &b) const
      -> Vec3 {
    return {x * b.x, y * b.y, z * b.z};
  }
  __host__ __device__ __forceinline__ auto operator*(float s) const -> Vec3 {
    return {x * s, y * s, z * s};
  }
  __host__ __device__ __forceinline__ auto operator/(float n) const -> Vec3 {
    float inv = 1.F / n;
    return {x * inv, y * inv, z * inv};
  }
  // elementwise division
  __host__ __device__ __forceinline__ auto operator/(const Vec3 &v) const
      -> Vec3 {
    return {x / v.x, y / v.x, z / v.x};
  }
  __host__ __device__ __forceinline__ auto outer_product(const Vec3 &b) const
      -> Mat3x3 {
    // x*b.x, x*b.y, x*b.z
    // y*b.x, y*b.y, y*b.z
    // z*b.x, z*b.y, z*b.z
    Mat3x3 m;
    m.data[0] = x * b.x;
    m.data[1] = x * b.y;
    m.data[2] = x * b.z;
    m.data[3] = y * b.x;
    m.data[4] = y * b.y;
    m.data[5] = y * b.z;
    m.data[6] = z * b.x;
    m.data[7] = z * b.y;
    m.data[8] = z * b.z;
    return m;
  }
  __host__ __device__ __forceinline__ auto tensor_square() const -> SymMat3x3 {
    // Outer product with itself
    SymMat3x3 m;
    m.data[0] = x*x;
    m.data[1] = x*y;
    m.data[2] = x*z;
    m.data[3] = y*y;
    m.data[4] = y*z;
    m.data[5] = z*z;
    return m;
  }
  __host__ __device__ __forceinline__ friend auto operator*(const Mat3x3 &lhs,
                                                            const Vec3 &v)
      -> Vec3 {
    Vec3 result;
    result.x = lhs.data[0] * v.x + lhs.data[1] * v.y + lhs.data[2] * v.z;
    result.y = lhs.data[3] * v.x + lhs.data[4] * v.y + lhs.data[5] * v.z;
    result.z = lhs.data[6] * v.x + lhs.data[7] * v.y + lhs.data[8] * v.z;
    return result;
  }
  __host__ __device__ __forceinline__ friend auto operator*(const SymMat3x3 &lhs,
                                                            const Vec3 &v)
      -> Vec3 {
    Vec3 result;
    result.x = lhs.data[0] * v.x + lhs.data[1] * v.y + lhs.data[2] * v.z;
    result.y = lhs.data[1] * v.x + lhs.data[3] * v.y + lhs.data[4] * v.z;
    result.z = lhs.data[2] * v.x + lhs.data[4] * v.y + lhs.data[5] * v.z;
    return result;
  }
  __host__ __device__ __forceinline__ static auto cross(const Vec3 &a,
                                                        const Vec3 &b) -> Vec3 {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
  }

  __host__ __device__ __forceinline__ auto operator=(const Vec3_f16 &v) {
    x = __half2float(v.x);
    y = __half2float(v.y);
    z = __half2float(v.z);
  }

  [[nodiscard]] auto dump() const -> std::string {
    // Returns a compact, single-line representation safe for HTML labels
    return std::format("({:.2f}, {:.2f}, {:.2f})",
                       x, y, z);
  }
};


__host__ __device__ __forceinline__ auto Vec3_f16::operator=(const Vec3 &v)
    -> Vec3_f16 {
  x = __float2half(v.x);
  y = __float2half(v.y);
  z = __float2half(v.z);
  return *this;
}

__host__ __device__ __forceinline__ auto Vec3_bf16::operator=(const Vec3 &v)
    -> Vec3_bf16 {
  x = __float2bfloat16(v.x);
  y = __float2bfloat16(v.y);
  z = __float2bfloat16(v.z);
  return *this;
}

__host__ __device__ __forceinline__ auto operator/(const float n, const Vec3 &v)
    -> Vec3 {
  return {n / v.x, n / v.y, n / v.z};
}
__host__ __device__ __forceinline__ auto operator*(const float n, const Vec3 &v)
    -> Vec3 {
  return {n * v.x, n * v.y, n * v.z};
}

__host__ __device__ __forceinline__ auto Vec3_f16::from_float(const Vec3 &v)
    -> Vec3_f16 {
  Vec3_f16 result;
  result.x = __float2half(v.x);
  result.y = __float2half(v.y);
  result.z = __float2half(v.z);
  return result;
}

__host__ __device__ __forceinline__ auto Vec3_bf16::from_float(const Vec3 &v)
    -> Vec3_bf16 {
  Vec3_bf16 result;
  result.x = __float2bfloat16(v.x);
  result.y = __float2bfloat16(v.y);
  result.z = __float2bfloat16(v.z);
  return result;
}

__host__ __device__ __forceinline__ auto
Mat3x3_f16::quadric_form(const Vec3_f16 &v) const -> half {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  // Broadcast vector components into half2 pairs
  half2 vx2 = __half2half2(v.x);
  half2 vy2 = __half2half2(v.y);
  half2 vz2 = __half2half2(v.z);

  // Pack column pairs for Rows 0 and 1: [m00, m10], [m01, m11], [m02, m12]
  half2 m01_x = make_half2(data[0], data[3]);
  half2 m01_y = make_half2(data[1], data[4]);
  half2 m01_z = make_half2(data[2], data[5]);

  // Compute (M*v)_0 and (M*v)_1 simultaneously in 2 SIMD instructions:
  // Mv_01 = m01_x * v.x + m01_y * v.y + m01_z * v.z
  half2 Mv_01 = __hfma2(m01_x, vx2, __hfma2(m01_y, vy2, __hmul2(m01_z, vz2)));

  // Compute (M*v)_2 scalar term
  half Mv_2 = __hfma(data[6], v.x, __hfma(data[7], v.y, __hmul(data[8], v.z)));

  // Inner product v^T * (M*v) = v_x*(Mv)_0 + v_y*(Mv)_1 + v_z*(Mv)_2
  half2 v_01 = make_half2(v.x, v.y);
  half2 prod_01 = __hmul2(v_01, Mv_01);
  half sum_01 = __hadd(prod_01.x, prod_01.y);

  return __hfma(v.z, Mv_2, sum_01);
#else
  half Mv_0 = __hadd(__hmul(data[0], v.x),
                     __hadd(__hmul(data[1], v.y), __hmul(data[2], v.z)));
  half Mv_1 = __hadd(__hmul(data[3], v.x),
                     __hadd(__hmul(data[4], v.y), __hmul(data[5], v.z)));
  half Mv_2 = __hadd(__hmul(data[6], v.x),
                     __hadd(__hmul(data[7], v.y), __hmul(data[8], v.z)));
  return __hadd(__hmul(v.x, Mv_0),
                __hadd(__hmul(v.y, Mv_1), __hmul(v.z, Mv_2)));
#endif
}

__host__ __device__ __forceinline__ auto
Mat3x3_bf16::quadric_form(const Vec3_bf16 &v) const -> __nv_bfloat16 {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  // Broadcast vector components into bfloat162 pairs
  __nv_bfloat162 vx2 = __bfloat162bfloat162(v.x);
  __nv_bfloat162 vy2 = __bfloat162bfloat162(v.y);
  __nv_bfloat162 vz2 = __bfloat162bfloat162(v.z);

  // Pack column pairs for Rows 0 and 1: [m00, m10], [m01, m11], [m02, m12]
  __nv_bfloat162 m01_x = __halves2bfloat162(data[0], data[3]);
  __nv_bfloat162 m01_y = __halves2bfloat162(data[1], data[4]);
  __nv_bfloat162 m01_z = __halves2bfloat162(data[2], data[5]);

  // Compute (M*v)_0 and (M*v)_1 simultaneously in 2 SIMD instructions:
  // Mv_01 = m01_x * v.x + m01_y * v.y + m01_z * v.z
  __nv_bfloat162 Mv_01 =
      __hfma2(m01_x, vx2, __hfma2(m01_y, vy2, __hmul2(m01_z, vz2)));

  // Compute (M*v)_2 scalar term
  __nv_bfloat16 Mv_2 =
      __hfma(data[6], v.x, __hfma(data[7], v.y, __hmul(data[8], v.z)));

  // Inner product v^T * (M*v) = v_x*(Mv)_0 + v_y*(Mv)_1 + v_z*(Mv)_2
  __nv_bfloat162 v_01 = __halves2bfloat162(v.x, v.y);
  __nv_bfloat162 prod_01 = __hmul2(v_01, Mv_01);
  __nv_bfloat16 sum_01 = __hadd(__low2bfloat16(prod_01), __high2bfloat16(prod_01));

  return __hfma(v.z, Mv_2, sum_01);
#else
  // Fallback for pre sm800
  const __nv_bfloat16 Mv_x = data[0] * v.x + data[1] * v.y + data[2] * v.z;
  const __nv_bfloat16 Mv_y = data[3] * v.x + data[4] * v.y + data[5] * v.z;
  const __nv_bfloat16 Mv_z = data[6] * v.x + data[7] * v.y + data[8] * v.z;

  return v.x * Mv_x + v.y * Mv_y + v.z * Mv_z;
#endif
}

__host__ __device__ __forceinline__ auto
Mat3x3_bf16::operator*(const Vec3_bf16 &v) const -> Vec3_bf16 {
  Vec3_bf16 res;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const __nv_bfloat162 v_xy = __halves2bfloat162(v.x, v.y);
  // --- Row 0 ---
  const __nv_bfloat162 r0_xy = __halves2bfloat162(data[0], data[1]);
  const __nv_bfloat162 p0 = __hmul2(r0_xy, v_xy);
  const __nv_bfloat16 xy_sum0 = __hadd(__low2bfloat16(p0), __high2bfloat16(p0));
  res.x = __hfma(data[2], v.z, xy_sum0);

  // --- Row 1 ---
  const __nv_bfloat162 r1_xy = __halves2bfloat162(data[3], data[4]);
  const __nv_bfloat162 p1 = __hmul2(r1_xy, v_xy);
  const __nv_bfloat16 xy_sum1 = __hadd(__low2bfloat16(p1), __high2bfloat16(p1));
  res.y = __hfma(data[5], v.z, xy_sum1);

  // --- Row 2 ---
  const __nv_bfloat162 r2_xy = __halves2bfloat162(data[6], data[7]);
  const __nv_bfloat162 p2 = __hmul2(r2_xy, v_xy);
  const __nv_bfloat16 xy_sum2 = __hadd(__low2bfloat16(p2), __high2bfloat16(p2));
  res.z = __hfma(data[8], v.z, xy_sum2);
#else
  res.x = data[0] * v.x + data[1] * v.y + data[2] * v.z;
  res.y = data[3] * v.x + data[4] * v.y + data[5] * v.z;
  res.z = data[6] * v.x + data[7] * v.y + data[8] * v.z;
#endif
  return res;
}
