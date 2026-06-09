#pragma once
#include <cmath>
#include <concepts>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <format>
#include <math.h>
#include <string>
#include <vector_types.h>
#include <kernels/common.cuh>

#include "vec3.h"
#include "aabb.h"


struct Triangle {
  Vec3 v0, v1, v2;

  __host__ __device__ __forceinline__ auto get_aabb() const -> AABB {
    Vec3 min{
        fminf(v0.x, fminf(v1.x, v2.x)),
        fminf(v0.y, fminf(v1.y, v2.y)),
        fminf(v0.z, fminf(v1.z, v2.z)),
    };
    Vec3 max{
        fmaxf(v0.x, fmaxf(v1.x, v2.x)),
        fmaxf(v0.y, fmaxf(v1.y, v2.y)),
        fmaxf(v0.z, fmaxf(v1.z, v2.z)),
    };
    AABB result;
    result.min = min;
    result.max = max;
    result.center_of_mass = centroid();
    result.max_distance_to_center = 0.F;
    return result;
  }
  __host__ __device__ __forceinline__ auto centroid() const -> Vec3 {
    return (v0 + v1 + v2) / 3.F;
  }

  __host__ __device__ __forceinline__ static auto
  load(const SoAViewConst<Triangle> &view, uint32_t idx, uint32_t count) -> Triangle;
  __host__ __device__ __forceinline__ auto get_scaled_normal() const -> Vec3;

  __host__ __device__ __forceinline__ auto
  get_tailor_terms( bool is_active, Vec3 &zero_order) const
      -> void;

  __host__ __device__ __forceinline__ auto
  contributionToQuery(const Vec3 &query, float inf_epsilon) const -> float;

  [[nodiscard]] auto dump() const -> std::string {
    std::string result = std::format(
        "Triangle {{v0: ({}, {}, {}), v1: ({}, {}, {}), v2: ({}, {}, {})}}\n",
        v0.x, v0.y, v0.z, v1.x, v1.y, v1.z, v2.x, v2.y, v2.z);
    return result;
  }
};

struct PointNormal {
  Vec3 p;
  Vec3 n;

  __host__ __device__ __forceinline__ static auto
  load(const SoAViewConst<PointNormal> &view, uint32_t idx, uint32_t count)
      -> PointNormal;

  __host__ __device__ __forceinline__ auto get_aabb() const -> AABB;
  __host__ __device__ __forceinline__ auto centroid() const -> Vec3;
  __host__ __device__ __forceinline__ auto get_scaled_normal() const -> Vec3;
  __host__ __device__ __forceinline__ auto
  get_tailor_terms( bool is_active, Vec3 &zero_order) const
      -> void;

  __host__ __device__ __forceinline__ auto
  contributionToQuery(const Vec3 &query, float inv_epsilon) const -> float;

  [[nodiscard]] auto dump() const -> std::string {
    std::string result = std::format(
        "PointNormal {{point: ({}, {}, {}), normal: ({}, {}, {})}}\n", p.x, p.y,
        p.z, n.x, n.y, n.z);
    return result;
  }
};

__host__ __device__ __forceinline__ auto
Triangle::load(const SoAViewConst<Triangle> &view, uint32_t idx, uint32_t count)
    -> Triangle {
  if (idx < count) {
    return {.v0 = Vec3{.x = view.base_ptr[0 * view.stride + idx],
                       .y = view.base_ptr[1 * view.stride + idx],
                       .z = view.base_ptr[2 * view.stride + idx]},
            .v1 = Vec3{.x = view.base_ptr[3 * view.stride + idx],
                       .y = view.base_ptr[4 * view.stride + idx],
                       .z = view.base_ptr[5 * view.stride + idx]},
            .v2 = Vec3{.x = view.base_ptr[6 * view.stride + idx],
                       .y = view.base_ptr[7 * view.stride + idx],
                       .z = view.base_ptr[8 * view.stride + idx]}};
  }
  return {{1e38F, 1e38F, 1e38F}, {1e38F, 1e38F, 1e38F}, {1e38F, 1e38F, 1e38F}};
}

__host__ __device__ __forceinline__ auto Triangle::get_scaled_normal() const
    -> Vec3 {
  // Area-weighted normal for Winding Number approx
  return Vec3::cross(v1 - v0, v2 - v0) * 0.5F;
}

__host__ __device__ __forceinline__ auto
PointNormal::load(const SoAViewConst<PointNormal> &view, uint32_t idx,
                  uint32_t count) -> PointNormal {
  if (idx < count) {
    return {.p = Vec3{.x = view.base_ptr[0 * view.stride + idx],
                      .y = view.base_ptr[1 * view.stride + idx],
                      .z = view.base_ptr[2 * view.stride + idx]},
            .n = Vec3{.x = view.base_ptr[3 * view.stride + idx],
                      .y = view.base_ptr[4 * view.stride + idx],
                      .z = view.base_ptr[5 * view.stride + idx]}};
  }
  return {{1e38F, 1e38F, 1e38F}, {0.F, 0.F, 0.F}};
}

__host__ __device__ __forceinline__ auto PointNormal::get_aabb() const -> AABB {
  return p.get_aabb();
}
__host__ __device__ __forceinline__ auto PointNormal::centroid() const -> Vec3 {
  return p;
}
__host__ __device__ __forceinline__ auto PointNormal::get_scaled_normal() const
    -> Vec3 {
  return n;
}

__host__ __device__ __forceinline__ auto
Triangle::get_tailor_terms(bool is_active,
                           Vec3 &zero_order) const -> void {
  const Vec3 n = get_scaled_normal();
  zero_order = is_active ? n : Vec3{0.F, 0.F, 0.F};
}

__host__ __device__ __forceinline__ auto
PointNormal::get_tailor_terms(bool is_active,
                              Vec3 &zero_order) const -> void {
  // Zero Order: Just the normal
  Vec3 n = get_scaled_normal();
  zero_order = is_active ? n : Vec3{0.F, 0.F, 0.F};
}

#define TWO_OVER_SQRT_PI 1.1283791671F
#define FOUR_OVER_3SQRT_PI 0.75225277806F // (4 / (3 * sqrt(pi)))
#define INV_FOUR_PI 0.07957747154F
#define INV_TWO_PI 0.15915494309F

__host__ __device__ __forceinline__ auto
Triangle::contributionToQuery(const Vec3 &query,
                              [[maybe_unused]] const float inv_epsilon) const
    -> float {
  // TODO: for gradient: this has discontinuities on the edges. Use inv_epsilon
  // to create a regularized version:
  // if (min_dist2 < 4.0f * eps2) {a_len =
  //    sqrtf(a_l2 + eps2);...
  const Vec3 a = v0 - query;
  const Vec3 b = v1 - query;
  const Vec3 c = v2 - query;
  const float a_len = a.length();
  const float b_len = b.length();
  const float c_len = c.length();
  const float det = a.dot(Vec3::cross(b, c));
  const float div = a_len * b_len * c_len + a.dot(b) * c_len +
                    a.dot(c) * b_len + b.dot(c) * a_len;

  // Handle the singularity: atan2(0, 0) is undefined.
  // If div is 0, we are on the boundary.
  if (fabsf(div) < 1e-12f) {
    return 0.5f;
  }
  return atan2f(det, div) * INV_TWO_PI;
}

__host__ __device__ __forceinline__ auto S_regularization(const float t)
    -> float {
  // For small t use Taylor expansion to avoid numerical issues in subtraction
  // as both terms approach the same value.
  // S(t) \approx (4/(3*sqrt(pi))) * t^3
  if (t < 0.1F) {
    return FOUR_OVER_3SQRT_PI * (t * t * t);
  }
  // Standard evaluation for larger t
  // S(t) = erf(t) - (2t/sqrt(pi)) * exp(-t^2)
  return erff(t) - (TWO_OVER_SQRT_PI * t * expf(-t * t));
}

__host__ __device__ __forceinline__ auto
PointNormal::contributionToQuery(const Vec3 &query,
                                 const float inv_epsilon) const -> float {
  // Use regularized dipole potential from:
  // 3D reconstruction with fast dipole sums
  // Hanyu Chen, Bailey Miller, Ioannis Gkioulekas
  // ACM Transactions on Graphics (SIGGRAPH Asia) 2024
  // https://arxiv.org/pdf/2405.16788
  const Vec3 d = p - query;
  const float dist2 = d.length2();

  // Guard against exact zero distance to prevent NaN in dot(d)/dist3
  if (dist2 < 1e-18F) {
    return 0.F;
  }

  const float distance = sqrtf(dist2);
  const float t = distance * inv_epsilon;

  float s_over_dist3;

  // If t is very small, S(t) ~ t^3, which cancels the distance^3 in the
  // denominator.
  if (t < 2.F) {
    if (t < 0.1F) {
      // S(t)/dist^3 \approx (FOUR_OVER_3SQRT_PI * (dist/eps)^3) / distr^3
      // The distr^3 terms cancel out.
      // This is the finite value P_eps(y,y) mentioned in the paper
      s_over_dist3 =
          FOUR_OVER_3SQRT_PI * (inv_epsilon * inv_epsilon * inv_epsilon);
    } else {
      s_over_dist3 = S_regularization(t) / (dist2 * distance);
    }
  } else {
    // Standard Poisson kernel (S(t) is effectively 1.0)
    s_over_dist3 = 1.F / (dist2 * distance);
  }

  return n.dot(d) * INV_FOUR_PI * s_over_dist3;
}

// Concept for Geometry template
template <typename T>
concept IsGeometry =
    requires(T g, SoAViewConst<T> gp, Vec3 p, bool active, Vec3 &z,float f, uint32_t u) {
      { T::load(gp, u, u) } -> std::same_as<T>;
      { g.get_aabb() } -> std::same_as<AABB>;
      { g.centroid() } -> std::same_as<Vec3>;
      { g.get_tailor_terms(active, z) } -> std::same_as<void>;
      { g.contributionToQuery(p, f) } -> std::same_as<float>;
    };

template <typename T>
concept IsPrimitiveGeometry = requires(T g) {
  { g.get_aabb() } -> std::same_as<AABB>;
  { g.centroid() } -> std::same_as<Vec3>;
};
