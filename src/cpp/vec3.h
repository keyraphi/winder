#pragma once
#include <cmath>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <vector_types.h>
#include "kernels/common.cuh"

struct Vec3;
struct AABB;

struct Vec3 {
  float x, y, z;

  __host__ __device__ __forceinline__ static auto
  load(const SoAViewConst<Vec3> &view, uint32_t idx, uint32_t count) -> Vec3 {
    if (idx < count) {
      return {.x = view.base_ptr[0 * view.stride + idx],
              .y = view.base_ptr[1 * view.stride + idx],
              .z = view.base_ptr[2 * view.stride + idx]};
    }
    return {0.F, 0.F, 0.F};
  }
  __host__ __device__ __forceinline__ static auto
  store(const Vec3 &value, SoAView<Vec3> &view, uint32_t idx) -> void {
    view.base_ptr[0 * view.stride + idx] = value.x;
    view.base_ptr[1 * view.stride + idx] = value.y;
    view.base_ptr[2 * view.stride + idx] = value.z;
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
  __host__ __device__ __forceinline__ static auto cross(const Vec3 &a,
                                                        const Vec3 &b) -> Vec3 {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
  }

};


__host__ __device__ __forceinline__ auto operator/(const float n, const Vec3 &v)
    -> Vec3 {
  return {n / v.x, n / v.y, n / v.z};
}
__host__ __device__ __forceinline__ auto operator*(const float n, const Vec3 &v)
    -> Vec3 {
  return {n * v.x, n * v.y, n * v.z};
}
