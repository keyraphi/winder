#pragma once
#include "aabb.h"
#include "mat3x3.h"
#include "tensor3.h"
#include "vec3.h"
#include <cmath>
#include <concepts>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <math.h>
#include <vector_types.h>

// symetric 3x3 matrix for tailor coefficient computation
struct SymMat3x3 {
  // 0:xx, 1:xy, 2:xz, 3:yy, 4:yz, 5:zz
  float data[6];

  __host__ __device__ __forceinline__ static auto zero() -> SymMat3x3 {
    return {0.F, 0.F, 0.F, 0.F, 0.F, 0.F};
  }

  __host__ __device__ __forceinline__ auto
  operator+(const SymMat3x3 &other) const -> SymMat3x3 {
    return {data[0] + other.data[0], data[1] + other.data[1],
            data[2] + other.data[2], data[3] + other.data[3],
            data[4] + other.data[4], data[5] + other.data[5]};
  }

  __host__ __device__ __forceinline__ auto operator*(float s) const
      -> SymMat3x3 {
    return {data[0] * s, data[1] * s, data[2] * s,
            data[3] * s, data[4] * s, data[5] * s};
  }
};

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
    Vec3 diagonal = max - min;
    result.center_of_mass.set(centroid(), min, 1.F / diagonal);
    result.center_of_mass.setMaxDistance(0.F, diagonal.inv_length());
    return result;
  }
  __host__ __device__ __forceinline__ auto centroid() const -> Vec3 {
    return (v0 + v1 + v2) / 3.F;
  }

  __device__ __forceinline__ static auto load(const Triangle *base,
                                              uint32_t idx, uint32_t count)
      -> Triangle;
  __host__ __device__ __forceinline__ auto get_scaled_normal() const -> Vec3;

  __host__ __device__ __forceinline__ auto
  get_tailor_terms(const Vec3 &p_center, bool is_active, Vec3 &zero_order,
                   Mat3x3 &first_order, Tensor3_compressed &second_order) const
      -> void;

  __host__ __device__ __forceinline__ auto
  contributionToQuery(const Vec3 &query, float inf_epsilon) const -> float;
};

struct PointNormal {
  Vec3 p;
  Vec3 n;

  __device__ __forceinline__ static auto load(const PointNormal *base,
                                              uint32_t idx, uint32_t count)
      -> PointNormal;

  __host__ __device__ __forceinline__ auto get_aabb() const -> AABB;
  __host__ __device__ __forceinline__ auto centroid() const -> Vec3;
  __host__ __device__ __forceinline__ auto get_scaled_normal() const -> Vec3;
  __host__ __device__ __forceinline__ auto
  get_tailor_terms(const Vec3 &p_center, bool is_active, Vec3 &zero_order,
                   Mat3x3 &first_order, Tensor3_compressed &second_order) const
      -> void;

  __host__ __device__ __forceinline__ auto
  contributionToQuery(const Vec3 &query, float inv_epsilon) const -> float;
};

__device__ __forceinline__ auto Triangle::load(const Triangle *base,
                                               uint32_t idx, uint32_t count)
    -> Triangle {
  if (idx < count) {
    // Triangle is 9 floats (36 bytes).
    const Vec3 *ptr = reinterpret_cast<const Vec3 *>(base + idx);
    return {ptr[0], ptr[1], ptr[2]};
  }
  return {{1e38F, 1e38F, 1e38F}, {1e38F, 1e38F, 1e38F}, {1e38F, 1e38F, 1e38F}};
}

__device__ __forceinline__ auto Triangle::get_scaled_normal() const -> Vec3 {
  // Area-weighted normal for Winding Number approx
  return Vec3::cross(v1 - v0, v2 - v0) * 0.5f;
}

__device__ __forceinline__ auto PointNormal::load(const PointNormal *base,
                                                  uint32_t idx, uint32_t count)
    -> PointNormal {
  if (idx < count) {
    const auto *ptr = reinterpret_cast<const float2 *>(base + idx);
    float2 c0 = ptr[0]; // px, py
    float2 c1 = ptr[1]; // pz, nx
    float2 c2 = ptr[2]; // ny, nz
    return {{c0.x, c0.y, c1.x}, {c1.y, c2.x, c2.y}};
  }
  return {{1e38F, 1e38F, 1e38F}, {0.F, 0.F, 0.F}};
}

__device__ __forceinline__ auto PointNormal::get_aabb() const -> AABB {
  return p.get_aabb();
}
__device__ __forceinline__ auto PointNormal::centroid() const -> Vec3 {
  return p;
}
__device__ __forceinline__ auto PointNormal::get_scaled_normal() const -> Vec3 {
  return n;
}

__host__ __device__ __forceinline__ auto
Triangle::get_tailor_terms(const Vec3 &p_center, bool is_active,
                           Vec3 &zero_order, Mat3x3 &first_order,
                           Tensor3_compressed &second_order) const -> void {
  const Vec3 n = get_scaled_normal();
  const Vec3 d = centroid() - p_center;

  // Second Order term: The C_t matrix from Appendix B
  // Midpoints relative to p_center
  Vec3 m_ij = ((v0 + v1) * 0.5F) - p_center;
  Vec3 m_jk = ((v1 + v2) * 0.5F) - p_center;
  Vec3 m_ki = ((v2 + v0) * 0.5F) - p_center;

  // Ct = 1/3 * (m_ij \otimes m_ij + m_jk \otimes m_jk + m_ki \otimes m_ki)
  // This captures the "spread" of the triangle surface
  // Direct symetric 6-element construction
  auto outer_sym = [](const Vec3 &v) -> SymMat3x3 {
    return {v.x * v.x, v.x * v.y, v.x * v.z, v.y * v.y, v.y * v.z, v.z * v.z};
  };

  constexpr float one_over_three = 1.F / 3.F;
  const SymMat3x3 Ct = {(outer_sym(m_ij) + outer_sym(m_jk) + outer_sym(m_ki)) *
                        one_over_three};

  zero_order = is_active ? n : Vec3{0.F, 0.F, 0.F};
  first_order = is_active ? d.outer_product(n) : Mat3x3::zero();

  if (is_active) {
    second_order.data[0] = Ct.data[0] * n.x;
    second_order.data[1] = Ct.data[0] * n.y;
    second_order.data[2] = Ct.data[0] * n.z;
    second_order.data[3] = Ct.data[1] * n.x;
    second_order.data[4] = Ct.data[1] * n.y;
    second_order.data[5] = Ct.data[1] * n.z;
    second_order.data[6] = Ct.data[2] * n.x;
    second_order.data[7] = Ct.data[2] * n.y;
    second_order.data[8] = Ct.data[2] * n.z;
    second_order.data[9] = Ct.data[3] * n.x;
    second_order.data[10] = Ct.data[3] * n.y;
    second_order.data[11] = Ct.data[3] * n.z;
    second_order.data[12] = Ct.data[4] * n.x;
    second_order.data[13] = Ct.data[4] * n.y;
    second_order.data[14] = Ct.data[4] * n.z;
    second_order.data[15] = Ct.data[5] * n.x;
    second_order.data[16] = Ct.data[5] * n.y;
    second_order.data[17] = Ct.data[5] * n.z;
  } else {
    // for inactive threads neutral element wrt. +
    for (int i = 0; i < 18; ++i) {
      second_order.data[i] = 0.F;
    }
  }
}

__host__ __device__ __forceinline__ auto
PointNormal::get_tailor_terms(const Vec3 &p_center, bool is_active,
                              Vec3 &zero_order, Mat3x3 &first_order,
                              Tensor3_compressed &second_order) const -> void {
  // Zero Order: Just the normal
  Vec3 n = get_scaled_normal();
  zero_order = n;

  // First Order:
  Vec3 r = p - p_center;
  first_order = r.outer_product(n);

  // Second Order: For a point, the spatial distribution
  // tensor is just the outer product of the offset.
  // Ct = d \otimes d
  // direct symetric construction of outer product
  SymMat3x3 Ct = {r.x * r.x, r.x * r.y, r.x * r.z,
                  r.y * r.y, r.y * r.z, r.z * r.z};

  if (is_active) {
    second_order.data[0] = 0.5F * Ct.data[0] * n.x;
    second_order.data[1] = 0.5F * Ct.data[0] * n.y;
    second_order.data[2] = 0.5F * Ct.data[0] * n.z;
    second_order.data[3] = 0.5F * Ct.data[1] * n.x;
    second_order.data[4] = 0.5F * Ct.data[1] * n.y;
    second_order.data[5] = 0.5F * Ct.data[1] * n.z;
    second_order.data[6] = 0.5F * Ct.data[2] * n.x;
    second_order.data[7] = 0.5F * Ct.data[2] * n.y;
    second_order.data[8] = 0.5F * Ct.data[2] * n.z;
    second_order.data[9] = 0.5F * Ct.data[3] * n.x;
    second_order.data[10] = 0.5F * Ct.data[3] * n.y;
    second_order.data[11] = 0.5F * Ct.data[3] * n.z;
    second_order.data[12] = 0.5F * Ct.data[4] * n.x;
    second_order.data[13] = 0.5F * Ct.data[4] * n.y;
    second_order.data[14] = 0.5F * Ct.data[4] * n.z;
    second_order.data[15] = 0.5F * Ct.data[5] * n.x;
    second_order.data[16] = 0.5F * Ct.data[5] * n.y;
    second_order.data[17] = 0.5F * Ct.data[5] * n.z;
  } else {
    // for inactive threads neutral element wrt. +
    for (int i = 0; i < 18; ++i) {
      second_order.data[i] = 0.F;
    }
  }
}

#define TWO_OVER_SQRT_PI 1.1283791671F
#define FOUR_OVER_3SQRT_PI 0.75225277806F // (4 / (3 * sqrt(pi)))
#define INV_FOUR_PI 0.07957747154F
#define INV_TWO_PI 0.15915494309F

__device__ __forceinline__ auto Triangle::contributionToQuery(
    const Vec3 &query, [[maybe_unused]] const float unused) const -> float {
  const Vec3 a = v0 - query;
  const Vec3 b = v1 - query;
  const Vec3 c = v2 - query;
  const float a_len = a.length();
  const float b_len = b.length();
  const float c_len = c.length();
  const float det = a.dot(Vec3::cross(b, c));
  const float div = a_len * b_len * c_len + a.dot(b) * c_len +
                    a.dot(c) * b_len + b.dot(c) * a_len;
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

__device__ __forceinline__ auto
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
concept IsGeometry = requires(T g, Vec3 p, bool active, Vec3 &z, Mat3x3 &m,
                              Tensor3_compressed &t, float f) {
  { g.get_aabb() } -> std::same_as<AABB>;
  { g.centroid() } -> std::same_as<Vec3>;
  { g.get_tailor_terms(p, active, z, m, t) } -> std::same_as<void>;
  { g.contributionToQuery(p, f) } -> std::same_as<float>;
};

template <typename T>
concept IsPrimitiveGeometry = requires(T g) {
  { g.get_aabb() } -> std::same_as<AABB>;
  { g.centroid() } -> std::same_as<Vec3>;
};
