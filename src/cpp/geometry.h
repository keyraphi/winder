#pragma once
#include "aabb.h"
#include "mat3x3.h"
#include "vec3.h"
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <math.h>
#include <vector_types.h>

// symetric 3x3 matrix for tailor coefficient computation
struct SymMat3x3 {
  // 0:xx, 1:xy, 2:xz, 3:yy, 4:yz, 5:zz
  float data[6];

  __device__ __forceinline__ static auto zero() -> SymMat3x3 {
    return {0.F, 0.F, 0.F, 0.F, 0.F, 0.F};
  }

  __device__ __forceinline__ auto operator+(const SymMat3x3 &other) const
      -> SymMat3x3 {
    return {data[0] + other.data[0], data[1] + other.data[1],
            data[2] + other.data[2], data[3] + other.data[3],
            data[4] + other.data[4], data[5] + other.data[5]};
  }

  __device__ __forceinline__ auto operator*(float s) const -> SymMat3x3 {
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
    Vec3 diagonal = max-min;
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

  __device__ __forceinline__ auto get_scaled_normal() const -> Vec3;

  __device__ __forceinline__ auto get_tailor_terms(const Vec3 &p_center,
                                                   Vec3 &out_n, Vec3 &out_d,
                                                   SymMat3x3 &out_Ct) const
      -> void;

  __device__ __forceinline__ auto contributionToQuery(const Vec3 &query,
                                                      float inf_epsilon) const
      -> float;
};

struct PointNormal {
  Vec3 p;
  Vec3 n;

  __device__ __forceinline__ static auto load(const PointNormal *base,
                                              uint32_t idx, uint32_t count)
      -> PointNormal;

  __device__ __forceinline__ auto get_aabb() const -> AABB;
  __device__ __forceinline__ auto centroid() const -> Vec3;
  __device__ __forceinline__ auto get_scaled_normal() const -> Vec3;
  __device__ __forceinline__ auto get_tailor_terms(const Vec3 &p_center,
                                                   Vec3 &out_n, Vec3 &out_d,
                                                   SymMat3x3 &out_Ct) const
      -> void;

  __device__ __forceinline__ auto
  contributionToQuery(const Vec3 &query, const float inv_epsilon) const
      -> float;
};

#ifdef __CUDACC__
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

__device__ __forceinline__ auto
Triangle::get_tailor_terms(const Vec3 &p_center, Vec3 &out_n, Vec3 &out_d,
                           SymMat3x3 &out_Ct) const -> void {
  out_n = get_scaled_normal();

  out_d = centroid() - p_center;

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

  out_Ct = {(outer_sym(m_ij) + outer_sym(m_jk) + outer_sym(m_ki)) *
            (1.0F / 3.F)};
}
__device__ __forceinline__ auto
PointNormal::get_tailor_terms(const Vec3 &p_center, Vec3 &out_n, Vec3 &out_d,
                              SymMat3x3 &out_Ct) const -> void {
  // Zero Order: Just the normal
  out_n = get_scaled_normal();

  // First Order: Offset from center
  out_d = p - p_center;

  // Second Order: For a point, the spatial distribution
  // tensor is just the outer product of the offset.
  // Ct = d \otimes d
  // direct symetric construction of outer product
  out_Ct = {out_d.x * out_d.x, out_d.x * out_d.y, out_d.x * out_d.z,
            out_d.y * out_d.y, out_d.y * out_d.z, out_d.z * out_d.z};
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

__device__ __forceinline__ auto S_regularization(const float t) -> float {
  // For small t use Taylor expansion to avoid numerical issues in subtraction
  // as both terms approach the same value.
  // S(t) \approx (4/(3*sqrt(pi))) * t^3
  if (t < 0.1F) {
    return FOUR_OVER_3SQRT_PI * (t * t * t);
  }
  // Standard evaluation for larger t
  // S(t) = erf(t) - (2t/sqrt(pi)) * exp(-t^2)
  return erff(t) - (TWO_OVER_SQRT_PI * t * __expf(-t * t));
}

__device__ __forceinline__ auto
PointNormal::contributionToQuery(const Vec3 &query,
                                 const float inv_epsilon) const -> float {
  // Use regularized dipole potential from:
  // 3D reconstruction with fast dipole sums
  // Hanyu Chen, Bailey Miller, Ioannis Gkioulekas
  // ACM Transactions on Graphics (SIGGRAPH Asia) 2024
  // https://arxiv.org/pdf/2405.16788
  const Vec3 d = query - p;
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

#endif
