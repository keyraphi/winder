#pragma once
#include "mat3x3.h"
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <vector_types.h>

struct Vec3;
struct AABB;

struct Vec3_bf16 {
  nv_bfloat16 x, y, z;

  __host__ __device__ __forceinline__ static auto from_float(const Vec3 &v)
      -> Vec3_bf16;

  __host__ __device__ __forceinline__ auto operator+(const Vec3_bf16 &b) const
      -> Vec3_bf16 {
    return {x + b.x, y + b.y, z + b.z};
  }
  __host__ __device__ __forceinline__ auto operator-(const Vec3_bf16 &b) const
      -> Vec3_bf16 {
    return {x - b.x, y - b.y, z - b.z};
  }
  __host__ __device__ __forceinline__ auto operator*(const Vec3_bf16 &b) const
      -> Vec3_bf16 {
    return {x * b.x, y * b.y, z * b.z};
  }
  __host__ __device__ __forceinline__ auto operator*(nv_bfloat16 s) const
      -> Vec3_bf16 {
    return {x * s, y * s, z * s};
  }

  __device__ __forceinline__ auto length2() const -> nv_bfloat16;
  __device__ __forceinline__ auto length() const -> nv_bfloat16;

  __host__ __device__ __forceinline__ auto
  outer_product(const Vec3_bf16 &b) const -> Mat3x3_bf16 {
    // x*b.x, x*b.y, x*b.z
    // y*b.x, y*b.y, y*b.z
    // z*b.x, z*b.y, z*b.z
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

  __host__ __device__ __forceinline__ auto operator=(const Vec3 &v);
};

struct Vec3 {
  float x, y, z;

  __host__ __device__ __forceinline__ static auto from_bf16(const Vec3_bf16 &v)
      -> Vec3 {
    Vec3 result;
    result.x = v.x;
    result.y = v.y;
    result.z = v.z;
    return result;
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

  __host__ __device__ __forceinline__ auto operator+(const Vec3 &b) const
      -> Vec3 {
    return {x + b.x, y + b.y, z + b.z};
  }
  __host__ __device__ __forceinline__ Vec3 &operator+=(const Vec3 &b) {
    x += b.x;
    y += b.y;
    z += b.z;
    return *this;
  }
  __host__ __device__ __forceinline__ Vec3 &operator+=(const Vec3_bf16 &b) {
    return *this += Vec3::from_bf16(b);
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
    return {a.y * b.z - a.z * b.x, a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
  }

  __host__ __device__ __forceinline__ auto operator=(const Vec3_bf16 &v) {
    x = v.x;
    y = v.y;
    z = v.z;
  }
};

__host__ __device__ __forceinline__ auto Vec3_bf16::operator=(const Vec3 &v) {
  x = v.x;
  y = v.y;
  z = v.z;
}

__host__ __device__ __forceinline__ auto operator/(const float n, const Vec3 &v)
    -> Vec3 {
  return {v.x / n, v.y / n, v.z / n};
}

__host__ __device__ __forceinline__ auto Vec3_bf16::from_float(const Vec3 &v)
    -> Vec3_bf16 {
  Vec3_bf16 result;
  result.x = v.x;
  result.y = v.y;
  result.z = v.z;
  return result;
}

// only for cuda compiler:
#ifdef __CUDACC__
__device__ __forceinline__ auto Vec3_bf16::length2() const -> nv_bfloat16 {
  return x * x + y * y + z * z;
}
__device__ __forceinline__ auto Vec3_bf16::length() const -> nv_bfloat16 {
  return hsqrt(length2());
}
#endif
