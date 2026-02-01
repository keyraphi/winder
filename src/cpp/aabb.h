#pragma once
#include "center_of_mass.h"
#include "kernels/common.cuh"
#include "vec3.h"
#include <cmath>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <sys/types.h>
#include <vector_types.h>

struct AABB;

// 6 Byte
struct AABB8BitApprox {
  uint8_t qmin[3]; // quantized min offset
  uint8_t qmax[3]; // quantized max offset

  __host__ __device__ __forceinline__ static auto
  quantize_aabb(const AABB &aabb, const Vec3 &parent_min,
                const Vec3 &parent_inv_extend) -> AABB8BitApprox;
};

struct AABB {
  Vec3 min;
  Vec3 max;

  CenterOfMass_quantized center_of_mass;

  // factory for empty AABB
  static __host__ __device__ __forceinline__ auto empty() -> AABB {
    AABB result;
    result.min = Vec3{INFINITY, INFINITY, INFINITY};
    result.max = Vec3{-INFINITY, -INFINITY, -INFINITY};
    return result;
  }

  // Note: no center of mass!
  __host__ __device__ __forceinline__ static auto
  from_approximation(const AABB &parent, const AABB8BitApprox &approx) -> AABB {
    Vec3 extent = parent.diagonal();
    constexpr float s = 1.0F / 255.0F;

    AABB result;
    result.min =
        parent.min + (Vec3{(float)approx.qmin[0], (float)approx.qmin[1],
                           (float)approx.qmin[2]} *
                      s * extent);
    result.max =
        parent.min + (Vec3{(float)approx.qmax[0], (float)approx.qmax[1],
                           (float)approx.qmax[2]} *
                      s * extent);
    return result;
  }

  __host__ __device__ __forceinline__ auto geometryc_center() const -> Vec3 {
    return {(min.x + max.x) * 0.5F, (min.y + max.y) * 0.5F,
            (min.z + max.z) * 0.5F};
  }

  __host__ __device__ __forceinline__ auto diagonal() const -> Vec3 {
    return max - min;
  }

  // radius = half-diagonal of the box
  __host__ __device__ __forceinline__ auto radius_sq() const -> float {
    return diagonal().length2() * 0.25F;
  }

  // radius = half-diagonal of the box
  __host__ __device__ __forceinline__ auto radius() const -> float {
    return sqrtf(radius_sq());
  }

  // Union of two AABB
  __host__ __device__ __forceinline__ static auto
  merge(const AABB &a, const AABB &b, const uint32_t a_element_count = 1,
        const uint32_t b_element_count = 1) -> AABB {
    AABB result;
    result.min = Vec3{fminf(a.min.x, b.min.x), fminf(a.min.y, b.min.y),
                      fminf(a.min.z, b.min.z)};
    result.max = Vec3{fmaxf(a.max.x, b.max.x), fmaxf(a.max.y, b.max.y),
                      fmaxf(a.max.z, b.max.z)};
    uint32_t total_element_count = a_element_count + b_element_count;
    float a_factor = (float)a_element_count / (float)total_element_count;
    float b_factor = (float)b_element_count / (float)total_element_count;
    Vec3 com_a = a.center_of_mass.get(a.min, a.diagonal());
    Vec3 com_b = b.center_of_mass.get(b.min, b.diagonal());
    Vec3 com_new = com_a * a_factor + com_b * b_factor;
    result.center_of_mass.set(com_new, result.min, 1.F / result.diagonal());
    // Calculate shift-adjusted radius
    // this is an upper bound
    float dist_a = (com_new - com_a).length() +
                   a.center_of_mass.getMaxDistance(a.diagonal().length());
    float dist_b = (com_new - com_b).length() +
                   b.center_of_mass.getMaxDistance(b.diagonal().length());

    result.center_of_mass.setMaxDistance(fmaxf(dist_a, dist_b),
                                         result.diagonal().inv_length());
    return result;
  }
};

__host__ __device__ __forceinline__ auto Vec3::get_aabb() const -> AABB {
  AABB result;
  result.min = *this;
  result.max = *this;
  result.center_of_mass.set(*this, *this, Vec3{INFINITY, INFINITY, INFINITY});
  result.center_of_mass.setMaxDistance(0.F, 0.F);
  return result;
}

__host__ __device__ __forceinline__ auto
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
