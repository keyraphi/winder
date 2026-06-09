#pragma once
#include "kernels/common.cuh"
#include "vec3.h"
#include <cmath>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <sys/types.h>
#include <vector_types.h>

struct AABB;

struct AABB {
  Vec3 min;                     // 12 byte
  Vec3 max;                     // 12 byte

  Vec3 center_of_mass;          // 12 byte
  float max_distance_to_center; // 4 byte
                                // total: 40 byte 
  

  // factory for empty AABB
  static __host__ __device__ __forceinline__ auto empty() -> AABB {
    AABB result;
    result.min = Vec3{INFINITY, INFINITY, INFINITY};
    result.max = Vec3{-INFINITY, -INFINITY, -INFINITY};
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
    // compute new center of mass
    uint32_t total_element_count = a_element_count + b_element_count;
    float a_factor = (float)a_element_count / (float)total_element_count;
    float b_factor = (float)b_element_count / (float)total_element_count;
    Vec3 com_a = a.center_of_mass;
    Vec3 com_b = b.center_of_mass;
    Vec3 com_new = com_a * a_factor + com_b * b_factor;

    result.center_of_mass = com_new;

    // Calculate shift-adjusted radius
    // this is an upper bound
    float dist_a = (com_new - com_a).length() +
                   a.max_distance_to_center;
    float dist_b = (com_new - com_b).length() +
                   b.max_distance_to_center;
    float new_dist = fmaxf(dist_a, dist_b);

    // Compute the maximum posible distance from the com in the aabb
    float dx = fmaxf(com_new.x - result.min.x, result.max.x - com_new.x);
    float dy = fmaxf(com_new.y - result.min.y, result.max.y - com_new.y);
    float dz = fmaxf(com_new.z - result.min.z, result.max.z - com_new.z);
    float max_possible_dist = sqrtf(dx * dx + dy * dy + dz * dz);

    new_dist = fminf(new_dist, max_possible_dist);

    result.max_distance_to_center = new_dist;
    return result;
  }
};

__host__ __device__ __forceinline__ auto Vec3::get_aabb() const -> AABB {
  AABB result;
  result.min = *this;
  result.max = *this;
  result.center_of_mass = Vec3{INFINITY, INFINITY, INFINITY};
  result.max_distance_to_center = 0.F;
  return result;
}
