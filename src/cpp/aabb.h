#pragma once
#include "vec3.h"
#include <cmath>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <sys/types.h>
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

  // quantized center of mass and max distance of elements to center
  uint8_t _center_of_mass[3];
  uint8_t _max_distance_to_center;

  // factory for empty AABB
  static __host__ __device__ __forceinline__ auto empty() -> AABB {
    AABB result;
    result.min = Vec3{INFINITY, INFINITY, INFINITY};
    result.max = Vec3{-INFINITY, -INFINITY, -INFINITY};
    result.setCenterOfMass({0.F,0.F,0.F});
    result.setMaxDistanceToCenterOfMass(0);
    return result;
  }

  __device__ __forceinline__ static auto
  from_approximation(const AABB &parent, const AABB8BitApprox &approx) -> AABB {
    Vec3 extent = parent.diagonal();
    constexpr float s = 1.0F / 255.0F;

    // This maps to 6 FFMA instructions
    return {parent.min + (Vec3{(float)approx.qmin[0], (float)approx.qmin[1],
                               (float)approx.qmin[2]} *
                          s * extent),
            parent.min + (Vec3{(float)approx.qmax[0], (float)approx.qmax[1],
                               (float)approx.qmax[2]} *
                          s * extent)};
  }

  __host__ __device__ __forceinline__ auto getCenterOfMass() const -> Vec3 {
    Vec3 extent = diagonal();
    return {min.x + (static_cast<float>(_center_of_mass[0]) / 255.F) * extent.x,
            min.y + (static_cast<float>(_center_of_mass[1]) / 255.F) * extent.y,
            min.z +
                (static_cast<float>(_center_of_mass[2]) / 255.F) * extent.z};
  }

  __host__ __device__ __forceinline__ auto
  setCenterOfMass(const Vec3 &center_of_mass) -> void {
    Vec3 extent = diagonal();
    _center_of_mass[0] = static_cast<uint8_t>(fminf(
        255.F, fmaxf(0.F, ((center_of_mass.x - min.x) / extent.x) * 255.F)));
    _center_of_mass[1] = static_cast<uint8_t>(fminf(
        255.F, fmaxf(0.F, ((center_of_mass.y - min.y) / extent.y) * 255.F)));
    _center_of_mass[2] = static_cast<uint8_t>(fminf(
        255.F, fmaxf(0.F, ((center_of_mass.z - min.z) / extent.z) * 255.F)));
  }

  __host__ __device__ __forceinline__ auto getMaxDistanceToCenterOfMass() const
      -> float {
    return (static_cast<float>(_max_distance_to_center) / 255.F) *
           diagonal().length();
  }

  __host__ __device__ __forceinline__ auto
  setMaxDistanceToCenterOfMass(const float distance) -> void {
    _max_distance_to_center = static_cast<uint8_t>(
        fminf(255.F, fmaxf(0.F, distance / diagonal().length()) * 255.F));
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
    Vec3 com_a = a.getCenterOfMass();
    Vec3 com_b = b.getCenterOfMass();
    Vec3 com_new = com_a * a_factor + com_b * b_factor;
    result.setCenterOfMass(com_new);
    // Calculate shift-adjusted radius
    // this is an upper bound
    float dist_a =
        (com_new - com_a).length() + a.getMaxDistanceToCenterOfMass();
    float dist_b =
        (com_new - com_b).length() + b.getMaxDistanceToCenterOfMass();

    result.setMaxDistanceToCenterOfMass(fmaxf(dist_a, dist_b));
    return result;
  }
};

__host__ __device__ __forceinline__ auto Vec3::get_aabb() const -> AABB {
  AABB result;
  result.min = *this;
  result.max = *this;
  result.setCenterOfMass(*this);
  result.setMaxDistanceToCenterOfMass(0.F);
  return result;
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
