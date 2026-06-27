#pragma once
#include "mat3x3.h"
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <vector_types.h>

struct Vec3;
struct AABB;

struct Vec3_f16 {
  half x, y, z;

  __host__ __device__ __forceinline__ static auto from_float(const Vec3 &v)
      -> Vec3_f16;

  __host__ __device__ __forceinline__ auto operator+(const Vec3_f16 &b) const
      -> Vec3_f16 {
    return {x + b.x, y + b.y, z + b.z};
  }
  __host__ __device__ __forceinline__ auto operator-(const Vec3_f16 &b) const
      -> Vec3_f16 {
    return {x - b.x, y - b.y, z - b.z};
  }
  __host__ __device__ __forceinline__ auto operator*(const Vec3_f16 &b) const
      -> Vec3_f16 {
    return {x * b.x, y * b.y, z * b.z};
  }
  __host__ __device__ __forceinline__ auto operator*(half s) const
      -> Vec3_f16 {
    return {x * s, y * s, z * s};
  }

  __host__ __device__ __forceinline__ auto length2() const -> half;
  __device__ __forceinline__ auto length() const -> half;

  __host__ __device__ __forceinline__ auto
  outer_product(const Vec3_f16 &b) const -> Mat3x3_f16 {
    // x*b.x, x*b.y, x*b.z
    // y*b.x, y*b.y, y*b.z
    // z*b.x, z*b.y, z*b.z
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

  __host__ __device__ __forceinline__ auto operator=(const Vec3 &v)
      -> Vec3_f16;
};

struct Vec3 {
  float x, y, z;

  __host__ __device__ __forceinline__ static auto from_f16(const Vec3_f16 &v)
      -> Vec3 {
    Vec3 result;
    result.x = v.x;
    result.y = v.y;
    result.z = v.z;
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

#ifdef __CUDACC__
    return rsqrtf(length2());
#else
    return 1.F / length();
#endif
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
  __host__ __device__ __forceinline__ static auto cross(const Vec3 &a,
                                                        const Vec3 &b) -> Vec3 {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
  }

  __host__ __device__ __forceinline__ auto operator=(const Vec3_f16 &v) {
    x = v.x;
    y = v.y;
    z = v.z;
  }
};

__host__ __device__ __forceinline__ auto Vec3_f16::operator=(const Vec3 &v)
    -> Vec3_f16 {
  x = v.x;
  y = v.y;
  z = v.z;
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

__host__ __device__ __forceinline__ auto Vec3_f16::length2() const -> half {
  return x * x + y * y + z * z;
}

// only for cuda compiler:
#ifdef __CUDACC__
__device__ __forceinline__ auto Vec3_f16::length() const -> half {
  return hsqrt(length2());
}
#endif
