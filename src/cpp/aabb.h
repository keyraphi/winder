#pragma once
#include "vec3.h"
#include <cmath>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <vector_types.h>

struct AABB;
struct AABB8BitApprox {
  uint8_t qmin[3]; // quantized min offset
  uint8_t qmax[3]; // quantized max offset

  __device__ __forceinline__ static auto
  quantize_aabb(const AABB &aabb, const Vec3 &parent_min,
                const Vec3 &parent_inv_extend) -> AABB8BitApprox;

};

struct AABB {
  Vec3 min;
  Vec3 max;

  // factory for empty AABB
  static __host__ __device__ __forceinline__ auto empty() -> AABB {
    return {{INFINITY, INFINITY, INFINITY}, {-INFINITY, -INFINITY, -INFINITY}};
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

__host__ __device__ __forceinline__ auto Vec3::get_aabb() const -> AABB {
  return {*this, *this};
}

// only for cuda compiler:
#ifdef __CUDACC__

__device__ __forceinline__ auto quantize_value(float value, float p_min,
                                               float p_inv_ext) {

  // catch infinite p_inv_ext. NaN results are flushed to positive zero.
  float normalized = __saturatef((value - p_min) * p_inv_ext) * 255.F;
  return (uint8_t)__float2uint_rn(normalized);
}

__device__ __forceinline__ auto
AABB8BitApprox::quantize_aabb(const AABB &aabb, const Vec3 &parent_min,
                              const Vec3 &parent_inv_extend) -> AABB8BitApprox {
  AABB8BitApprox result;
  result.qmin[0] =
      quantize_value(aabb.min.x, parent_min.x, parent_inv_extend.x);
  result.qmin[1] =
      quantize_value(aabb.min.y, parent_min.y, parent_inv_extend.y);
  result.qmin[2] =
      quantize_value(aabb.min.z, parent_min.z, parent_inv_extend.z);
  result.qmax[0] =
      quantize_value(aabb.max.x, parent_min.x, parent_inv_extend.x);
  result.qmax[1] =
      quantize_value(aabb.max.y, parent_min.y, parent_inv_extend.y);
  result.qmax[2] =
      quantize_value(aabb.max.z, parent_min.z, parent_inv_extend.z);
  return result;
}

#endif
