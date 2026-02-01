#pragma once
#include "kernels/common.cuh"
#include "vec3.h"
#include <cstdint>
#include <cuda_runtime_api.h>

// 4 bytes
struct CenterOfMass_quantized {
  uint8_t _center_of_mass[3];
  uint8_t _max_distance_to_center;

  /**
   * Load quantized center of mass for a aabb.
   * @param aabb_min The aabb.min value of the aabb for which the com is
   * requested.
   * @param aabb_diagonal The aabb.diagonal() of the aabb for which the com is
   * requested.
   * @returns The center of mass as Vec3.
   */
  __host__ __device__ __forceinline__ auto get(const Vec3 &aabb_min,
                                               const Vec3 &aabb_diagonal) const
      -> Vec3;

  /**
   * Quantize center of mass relative to aabb.
   * @param center_of_mass The center of mass that should be quantized
   * @param aabb_min The aabb.min point of the aabb for which this center of
   * mass is stored.
   * @param aabb_inv_extend 1.F / aabb.diagonal() for the aabb for which the com
   * is stored.
   */
  __host__ __device__ __forceinline__ auto set(const Vec3 &center_of_mass,
                                               const Vec3 &aabb_min,
                                               const Vec3 &aabb_inv_extend)
      -> void;

  /**
   * Load quantized max distance of elements to the com in the aabb.
   * @param diagonal_length The aabb.diagonal().length() of the aabb for which
   * the max istance to the com is requested.
   * @returns The max distance of any element to the center of mass as float.
   */
  __host__ __device__ __forceinline__ auto
  getMaxDistance(float diagonal_length) const -> float;

  /**
   * Quantize center of mass relative to aabb.
   * @param distance The distance that should be quantized.
   * @param inv_diagonal_length 1.F / aabb.diagonal().length() for the aabb for
   * which the distance is stored.
   */
  __host__ __device__ __forceinline__ auto
  setMaxDistance(float distance, float inv_diagonal_length) -> void;
};

__host__ __device__ __forceinline__ auto
CenterOfMass_quantized::get(const Vec3 &aabb_min,
                            const Vec3 &aabb_diagonal) const -> Vec3 {
  float s = 1.F / 255.F;
  return {aabb_min.x +
              (static_cast<float>(_center_of_mass[0]) * s) * aabb_diagonal.x,
          aabb_min.y +
              (static_cast<float>(_center_of_mass[1]) * s) * aabb_diagonal.y,
          aabb_min.z +
              (static_cast<float>(_center_of_mass[2]) * s) * aabb_diagonal.z};
}

__host__ __device__ __forceinline__ auto
CenterOfMass_quantized::set(const Vec3 &center_of_mass, const Vec3 &aabb_min,
                            const Vec3 &aabb_inv_extend) -> void {
  _center_of_mass[0] =
      quantize_value(center_of_mass.x, aabb_min.x, aabb_inv_extend.x);
  _center_of_mass[1] =
      quantize_value(center_of_mass.y, aabb_min.y, aabb_inv_extend.y);
  _center_of_mass[2] =
      quantize_value(center_of_mass.z, aabb_min.z, aabb_inv_extend.z);
}

__host__ __device__ __forceinline__ auto
CenterOfMass_quantized::getMaxDistance(const float diagonal_length) const
    -> float {
  return (static_cast<float>(_max_distance_to_center) / 255.F) *
         diagonal_length;
}

__host__ __device__ __forceinline__ auto
CenterOfMass_quantized::setMaxDistance(float distance,
                                       float inv_diagonal_length) -> void {
#ifdef __CUDA_ARCH__
  float normalized = __saturatef(distance * inv_diagonal_length) * 255.F;
  _max_distance_to_center = (uint8_t)__float2uint_rn(normalized);
#else
  float val = distance * inv_diagonal_length;
  float normalized = std::max(0.0f, std::min(1.0f, val)) * 255.F;
  _max_distance_to_center = (uint8_t)(normalized + 0.5f);
#endif
}
