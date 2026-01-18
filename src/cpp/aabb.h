#pragma once
#include "vec3.h"
#include <cmath>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <vector_types.h>

struct AABB8BitApprox {
  uint8_t qmin[3];  // quantized min offset
  uint8_t qmax[3];  // quantized max offset
};


struct AABB {
  Vec3 min;
  Vec3 max;

  // factory for empty AABB
  static __host__ __device__ __forceinline__ auto empty() -> AABB {
    return {{INFINITY, INFINITY, INFINITY}, {-INFINITY, -INFINITY, -INFINITY}};
  }
  // factory for AABB for a single point
  static __host__ __device__ __forceinline__ auto from_point(const Vec3 &p)
      -> AABB {
    return {p, p};
  }

  __device__ __forceinline__ static auto
  from_approximation(const AABB &parent, const AABB8BitApprox &approx) -> AABB {
    Vec3 extent = parent.max - parent.min;
    constexpr float s = 1.0F / 255.0F;

    // This maps to 6 FFMA instructions
    return {parent.min + (Vec3{(float)approx.qmin[0], (float)approx.qmin[1],
                               (float)approx.qmin[2]} *
                          s * extent),
            parent.min + (Vec3{(float)approx.qmax[0], (float)approx.qmax[1],
                               (float)approx.qmax[2]} *
                          s * extent)};
  }

  __host__ __device__ __forceinline__ auto center() const -> Vec3 {
    return {(min.x + max.x) * 0.5F, (min.y + max.y) * 0.5F,
            (min.z + max.z) * 0.5F};
  }

  // radius = half-diagonal of the box
  __host__ __device__ __forceinline__ auto radius_sq() const -> float {
    float dx = max.x - min.x;
    float dy = max.y - min.y;
    float dz = max.z - min.z;
    return (dx * dx + dy * dy + dz * dz) * 0.5F;
  }

  // radius = half-diagonal of the box
  __host__ __device__ __forceinline__ auto radius() const -> float {
    return sqrtf(radius_sq());
  }

  // Union of two AABB
  __host__ __device__ __forceinline__ static auto merge(const AABB &a,
                                                        const AABB &b) -> AABB {
    return {{fminf(a.min.x, b.min.x), fminf(a.min.y, b.min.y),
             fminf(a.min.z, b.min.z)},
            {fmaxf(a.max.x, b.max.x), fmaxf(a.max.y, b.max.y),
             fmaxf(a.max.z, b.max.z)}};
  }
};
