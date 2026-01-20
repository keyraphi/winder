#pragma once
#include <cuda_runtime_api.h>
#include <vector_types.h>
#include "mat3x3.h"

struct Vec3 {
  float x, y, z;

  __host__ __device__ __forceinline__ Vec3 operator+(const Vec3 &b) const {
    return {x + b.x, y + b.y, z + b.z};
  }
  __host__ __device__ __forceinline__ Vec3 operator-(const Vec3 &b) const {
    return {x - b.x, y - b.y, z - b.z};
  }
  __host__ __device__ __forceinline__ Vec3 operator*(const Vec3 &b) const {
    return {x * b.x, y * b.y, z * b.z};
  }
  __host__ __device__ __forceinline__ Vec3 operator*(float s) const {
    return {x * s, y * s, z * s};
  }
  __host__ __device__ __forceinline__ Mat3x3 outer_product(const Vec3 &b) {
  // x*b.x, x*b.y, x*b.z
  // y*b.x, y*b.y, y*b.z
  // z*b.x, z*b.y, z*b.z
    Mat3x3 m;
    m.data[0] = x*b.x;
    m.data[1] = x*b.y;
    m.data[2] = x*b.z;
    m.data[3] = y*b.x;
    m.data[4] = y*b.y;
    m.data[5] = y*b.z;
    m.data[6] = z*b.x;
    m.data[7] = z*b.y;
    m.data[8] = z*b.z;
    return m;
  }
};
