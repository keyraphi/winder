#pragma once
#include "aabb.h"
#include "mat3x3.h"
#include "soa.h"
#include "tensor3.h"
#include "vec3.h"
#include <cmath>
#include <concepts>
#include <cstdint>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <format>
#include <math.h>
#include <string>
#include <vector_types.h>


struct Triangle {
  Vec3 v0, v1, v2;

  __host__ __device__ __forceinline__ auto
  operator+(const Triangle &other) const -> Triangle {
    return Triangle{v0 + other.v0, v1 + other.v1, v2 + other.v2};
  }
  __host__ __device__ __forceinline__ auto
  operator-(const Triangle &other) const -> Triangle {
    return Triangle{v0 - other.v0, v1 - other.v1, v2 - other.v2};
  }
  __host__ __device__ __forceinline__ auto operator+=(const Triangle &other)
      -> Triangle & {
    v0 += other.v0;
    v1 += other.v1;
    v2 += other.v2;
    return *this;
  }

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
    Vec3 c = centroid();

    float d0 = (v0 - c).length2();
    float d1 = (v1 - c).length2();
    float d2 = (v2 - c).length2();
    float max_dist_sq = fmaxf(d0, fmaxf(d1, d2));
    float max_dist = sqrtf(max_dist_sq);

    result.center_of_mass = c;
    result.max_distance = __float2half(max_dist);
    return result;
  }

  __host__ __device__ __forceinline__ auto get_weight() const -> float {
    // weight is surface area
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 cp = Vec3{e1.y * e2.z - e1.z * e2.y, e1.z * e2.x - e1.x * e2.z,
                   e1.x * e2.y - e1.y * e2.x};
    return 0.5F * sqrtf(cp.x * cp.x + cp.y * cp.y + cp.z * cp.z);
  }
  __host__ __device__ __forceinline__ auto
  max_distance_to(const Vec3 &pos) const -> float {
    // A triangle is convex; its maximum distance to ANY point is always at one
    // of its vertices
    float d0 = (v0 - pos).length2();
    float d1 = (v1 - pos).length2();
    float d2 = (v2 - pos).length2();
    return sqrtf(fmaxf(d0, fmaxf(d1, d2)));
  }
  __host__ __device__ __forceinline__ auto centroid() const -> Vec3 {
    return (v0 + v1 + v2) / 3.F;
  }

  __host__ __device__ __forceinline__ static auto
  load(const SoAView<Triangle> &view, uint32_t idx, uint32_t count) -> Triangle;
  __host__ __device__ __forceinline__ auto get_scaled_normal() const -> Vec3;

  __host__ __device__ __forceinline__ auto
  get_taylor_terms(const Vec3 &p_center, bool is_active, Vec3 &zero_order,
                   Mat3x3 &first_order, Tensor3_compressed &second_order) const
      -> void;

  __host__ __device__ __forceinline__ auto
  contributionToQuery(const Vec3 &query, float inf_epsilon) const -> float;

  __host__ __device__ __forceinline__ auto
  gradContributionOfQuery(const Vec3 &q, float g) const -> Triangle;

  [[nodiscard]] auto dump() const -> std::string {
    // Returns a compact, single-line representation safe for HTML labels
    return std::format("v0:({:.2f}, {:.2f}, {:.2f}) | v1:({:.2f}, {:.2f}, "
                       "{:.2f}) | v2:({:.2f}, {:.2f}, {:.2f})",
                       v0.x, v0.y, v0.z, v1.x, v1.y, v1.z, v2.x, v2.y, v2.z);
  }
};

struct PointNormal {
  Vec3 p;
  Vec3 n;

  __host__ __device__ __forceinline__ auto
  operator+(const PointNormal &other) const -> PointNormal {
    return PointNormal{p + other.p, n + other.n};
  }
  __host__ __device__ __forceinline__ auto
  operator-(const PointNormal &other) const -> PointNormal {
    return PointNormal{p - other.p, n - other.n};
  }

  __host__ __device__ __forceinline__ auto operator+=(const PointNormal &other)
      -> PointNormal & {
    p += other.p;
    n += other.n;
    return *this;
  }

  __host__ __device__ __forceinline__ static auto
  load(const SoAView<PointNormal> &view, uint32_t idx, uint32_t count)
      -> PointNormal;

  __host__ __device__ __forceinline__ auto get_weight() const -> float {
    return 1.0F;
  }
  __host__ __device__ __forceinline__ auto
  max_distance_to(const Vec3 &pos) const -> float {
    return (p - pos).length();
  }

  __host__ __device__ __forceinline__ auto get_aabb() const -> AABB;
  __host__ __device__ __forceinline__ auto centroid() const -> Vec3;
  __host__ __device__ __forceinline__ auto get_scaled_normal() const -> Vec3;
  __host__ __device__ __forceinline__ auto
  get_taylor_terms(const Vec3 &p_center, bool is_active, Vec3 &zero_order,
                   Mat3x3 &first_order, Tensor3_compressed &second_order) const
      -> void;

  __host__ __device__ __forceinline__ auto
  contributionToQuery(const Vec3 &query, float inv_epsilon) const -> float;

  __host__ __device__ __forceinline__ auto gradContributionOfQuery(
      const Vec3 &q, const float g, const float inv_epsilon,
      const float reg_term_const,    // Precomputed: inv_epsilon3 * INV_PI_1_5
      const float near_field_g_denum // Precomputed: (1.f / (3.f * pi^1.5)) *
                                     // inv_epsilon3
  ) const -> PointNormal;

  [[nodiscard]] auto dump() const -> std::string {
    // Returns a compact, single-line representation safe for HTML labels
    return std::format(
        "Pos:({:.2f}, {:.2f}, {:.2f}) | Norm:({:.2f}, {:.2f}, {:.2f})", p.x,
        p.y, p.z, n.x, n.y, n.z);
  }
};

__host__ __device__ __forceinline__ auto
Triangle::load(const SoAView<Triangle> &view, uint32_t idx, uint32_t count)
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
PointNormal::load(const SoAView<PointNormal> &view, uint32_t idx,
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
Triangle::get_taylor_terms(const Vec3 &p_center, bool is_active,
                           Vec3 &zero_order, Mat3x3 &first_order,
                           Tensor3_compressed &second_order) const -> void {
  if (is_active) {
    const Vec3 n = get_scaled_normal();
    const Vec3 d = centroid() - p_center;

    // Second Order term: The C_t matrix from Appendix B
    // Midpoints relative to p_center
    Vec3 m_ij = ((v0 + v1) * 0.5F) - p_center;
    Vec3 m_jk = ((v1 + v2) * 0.5F) - p_center;
    Vec3 m_ki = ((v2 + v0) * 0.5F) - p_center;

    // Ct = 1/3 * (m_ij \otimes m_ij + m_jk \otimes m_jk + m_ki \otimes m_ki)
    // This captures the "spread" of the triangle surface
    // Direct symmetric 6-element construction
    auto outer_sym = [](const Vec3 &v) -> SymMat3x3 {
      return {v.x * v.x, v.x * v.y, v.x * v.z, v.y * v.y, v.y * v.z, v.z * v.z};
    };

    constexpr float one_over_three = 1.F / 3.F;
    const SymMat3x3 Ct = {
        (outer_sym(m_ij) + outer_sym(m_jk) + outer_sym(m_ki)) * one_over_three};

    zero_order = n;
    first_order = d.outer_product(n);

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
    zero_order = Vec3{0.F, 0.F, 0.F};
    first_order = Mat3x3::zero();
    for (int i = 0; i < 18; ++i) {
      second_order.data[i] = 0.F;
    }
  }
}

__host__ __device__ __forceinline__ auto
PointNormal::get_taylor_terms(const Vec3 &p_center, bool is_active,
                              Vec3 &zero_order, Mat3x3 &first_order,
                              Tensor3_compressed &second_order) const -> void {
  if (is_active) {
    // Zero Order: Just the normal
    Vec3 n = get_scaled_normal();
    zero_order = n;

    // First Order:
    Vec3 r = p - p_center;
    first_order = r.outer_product(n);

    // Second Order: For a point, the spatial distribution
    // tensor is just the outer product of the offset.
    // Ct = d \otimes d
    // direct symmetric construction of outer product
    SymMat3x3 Ct = {r.x * r.x, r.x * r.y, r.x * r.z,
                    r.y * r.y, r.y * r.z, r.z * r.z};

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
    zero_order = Vec3{0.F, 0.F, 0.F};
    first_order = Mat3x3::zero();
    for (int i = 0; i < 18; ++i) {
      second_order.data[i] = 0.F;
    }
  }
}

#define TWO_OVER_SQRT_PI 1.1283791671F
#define FOUR_OVER_3SQRT_PI 0.75225277806F // (4 / (3 * sqrt(pi)))
#define INV_FOUR_PI 0.07957747154F
#define INV_TWO_PI 0.15915494309F

__host__ __device__ __forceinline__ auto S_regularization(const float t)
    -> float {
  if (t < 0.1F) {
    return FOUR_OVER_3SQRT_PI * (t * t * t);
  }

#if defined(__CUDA_ARCH__)
  return erff(t) - (TWO_OVER_SQRT_PI * t * __expf(-t * t));
#else
  return erff(t) - (TWO_OVER_SQRT_PI * t * expf(-t * t));
#endif
}

__host__ __device__ __forceinline__ auto
Triangle::contributionToQuery(const Vec3 &query,
                              [[maybe_unused]] const float inv_epsilon) const
    -> float {
  const Vec3 a = v0 - query;
  const Vec3 b = v1 - query;
  const Vec3 c = v2 - query;

  const float a2 = a.length2() + 1e-20F;
  const float b2 = b.length2() + 1e-20F;
  const float c2 = c.length2() + 1e-20F;

#ifdef __CUDACC__
  const float inv_a = rsqrtf(a2);
  const float inv_b = rsqrtf(b2);
  const float inv_c = rsqrtf(c2);
#else
  const float inv_a = 1.F / sqrtf(a2);
  const float inv_b = 1.F / sqrtf(b2);
  const float inv_c = 1.F / sqrtf(c2);
#endif

  // Calculate dimensionless cosines
  const float cos_ab = a.dot(b) * inv_a * inv_b;
  const float cos_ac = a.dot(c) * inv_a * inv_c;
  const float cos_bc = b.dot(c) * inv_b * inv_c;

  // normalized determinant and denominator
  const float det_norm = a.dot(Vec3::cross(b, c)) * inv_a * inv_b * inv_c;
  const float div_norm = 1.F + cos_ab + cos_ac + cos_bc;

  // Scale-invariant singularity check
  if (fabsf(div_norm) < 1e-6F) {
    return 0.5F;
  }
  return atan2f(det_norm, div_norm) * INV_TWO_PI;
}

__host__ __device__ __forceinline__ auto
PointNormal::contributionToQuery(const Vec3 &query,
                                 const float inv_epsilon) const -> float {
  const Vec3 d = p - query;
  const float dist2 = d.length2();

#ifdef __CUDACC__
  const float inv_distance = rsqrtf(dist2 + 1e-20F);
#else
  const float inv_distance = 1.F / sqrtf(dist2 + 1e-20F);
#endif
  const float inv_dist2 = inv_distance * inv_distance;
  const float inv_dist3 = inv_dist2 * inv_distance;

  const float distance = dist2 * inv_distance;
  const float t = distance * inv_epsilon;

  float s_over_dist3;

  if (t < 2.F) {
    if (t < 0.1F) {
      s_over_dist3 =
          FOUR_OVER_3SQRT_PI * (inv_epsilon * inv_epsilon * inv_epsilon);
    } else {
      s_over_dist3 = S_regularization(t) * inv_dist3;
    }
  } else {
    s_over_dist3 = inv_dist3;
  }

  return n.dot(d) * INV_FOUR_PI * s_over_dist3;
}

__host__ __device__ __forceinline__ auto
Triangle::gradContributionOfQuery(const Vec3 &q, const float g) const
    -> Triangle {
  // Relative vectors from query point to vertices
  const Vec3 a = v0 - q;
  const Vec3 b = v1 - q;
  const Vec3 c = v2 - q;

  // Squared distances with a tiny epsilon to prevent division-by-zero on exact
  // singularities
  const float a2 = a.length2() + 1e-20F;
  const float b2 = b.length2() + 1e-20F;
  const float c2 = c.length2() + 1e-20F;

#if defined(__CUDA_ARCH__)
  const float inv_a = rsqrtf(a2);
  const float inv_b = rsqrtf(b2);
  const float inv_c = rsqrtf(c2);
#else
  const float inv_a = 1.F / sqrtf(a2);
  const float inv_b = 1.F / sqrtf(b2);
  const float inv_c = 1.F / sqrtf(c2);
#endif

  const float a_len = a2 * inv_a;
  const float b_len = b2 * inv_b;
  const float c_len = c2 * inv_c;

  // Precompute cosines for the dimensionless denominator
  const float cos_ab = a.dot(b) * inv_a * inv_b;
  const float cos_ac = a.dot(c) * inv_a * inv_c;
  const float cos_bc = b.dot(c) * inv_b * inv_c;

  // Normalized determinant (N_norm) and denominator (D_norm)
  const float det_norm = a.dot(Vec3::cross(b, c)) * inv_a * inv_b * inv_c;
  const float div_norm = 1.F + cos_ab + cos_ac + cos_bc;

  // Scale-invariant singular boundary check using the normalized denominator
  const float denom_norm = det_norm * det_norm + div_norm * div_norm;
  if (denom_norm < 1e-12F) {
    return Triangle{.v0 = Vec3{0.F, 0.F, 0.F},
                    .v1 = Vec3{0.F, 0.F, 0.F},
                    .v2 = Vec3{0.F, 0.F, 0.F}};
  }

  // 1. Numerator Derivatives: dN / dv_k
  const Vec3 dN_dv0 = Vec3::cross(b, c);
  const Vec3 dN_dv1 = Vec3::cross(c, a);
  const Vec3 dN_dv2 = Vec3::cross(a, b);

  // 2. Denominator Derivatives: dD / dv_k
  const Vec3 hat_a = a * inv_a;
  const Vec3 hat_b = b * inv_b;
  const Vec3 hat_c = c * inv_c;

  const float b_dot_c = b.dot(c);
  const float c_dot_a = c.dot(a);
  const float a_dot_b = a.dot(b);

  const Vec3 dD_dv0 = hat_a * (b_len * c_len + b_dot_c) + b * c_len + c * b_len;
  const Vec3 dD_dv1 = hat_b * (c_len * a_len + c_dot_a) + c * a_len + a * c_len;
  const Vec3 dD_dv2 = hat_c * (a_len * b_len + a_dot_b) + a * b_len + b * a_len;

  // 3. Assemble Gradients using the scale-invariant scaling factor
  const float inv_L = inv_a * inv_b * inv_c;
  const float factor = g * INV_TWO_PI * (inv_L / denom_norm);

  return Triangle{.v0 = (dN_dv0 * div_norm - dD_dv0 * det_norm) * factor,
                  .v1 = (dN_dv1 * div_norm - dD_dv1 * det_norm) * factor,
                  .v2 = (dN_dv2 * div_norm - dD_dv2 * det_norm) * factor};
}

__host__ __device__ __forceinline__ auto PointNormal::gradContributionOfQuery(
    const Vec3 &q, const float g, const float inv_epsilon,
    const float reg_term_const,     // Precomputed: inv_epsilon3 * INV_PI_1_5
    const float near_field_g_denum) // Precomputed: (1.f / (3.f * pi^1.5)) *
                                    // inv_epsilon3)
    const -> PointNormal {
  const Vec3 d = p - q;
  const float dist2 = d.x * d.x + d.y * d.y + d.z * d.z;

  // Fast reciprocal square root with NaN-safe offset
#ifdef __CUDACC__
  const float inv_dist = rsqrtf(dist2 + 1e-20F);
#else
  const float inv_dist = 1.F / sqrtf(dist2 + 1e-20F);
#endif
  const float inv_dist2 = inv_dist * inv_dist;
  const float inv_dist3 = inv_dist2 * inv_dist;

  const float distance = dist2 * inv_dist;
  const float t = distance * inv_epsilon;

  float scale_n;
  float scale_d;

  if (t < 0.1F) {
    scale_n = g * near_field_g_denum;
    scale_d = 0.F;
  } else {
    float reg_term = 0.F;
    float s_over_dist3;

    if (t < 2.F) {
      s_over_dist3 = S_regularization(t) * inv_dist3;
      const float t2 = t * t;
#if defined(__CUDA_ARCH__)
      const float exp_t2 = __expf(-t2);
#else
      const float exp_t2 = expf(-t2);
#endif
      reg_term = exp_t2 * reg_term_const;
    } else {
      s_over_dist3 = inv_dist3;
    }

    const float g_denum = INV_FOUR_PI * s_over_dist3;
    const float dot = n.x * d.x + n.y * d.y + n.z * d.z;
    const float shared_factor = dot * inv_dist2;

    scale_n = g * g_denum;
    scale_d = g * shared_factor * (reg_term - 3.F * g_denum);
  }

  PointNormal gradient{
      .p = Vec3{.x = scale_n * n.x + scale_d * d.x,
                .y = scale_n * n.y + scale_d * d.y,
                .z = scale_n * n.z + scale_d * d.z},
      .n = Vec3{.x = scale_n * d.x, .y = scale_n * d.y, .z = scale_n * d.z}};
  // Position gradient
  return gradient;
}

// Concept for Geometry template
template <typename T>
concept IsGeometry =
    requires(T g, SoAView<T> gp, Vec3 p, bool active, Vec3 &z, Mat3x3 &m,
             Tensor3_compressed &t, float f, uint32_t u) {
      { T::load(gp, u, u) } -> std::same_as<T>;
      { g.get_aabb() } -> std::same_as<AABB>;
      { g.centroid() } -> std::same_as<Vec3>;
      { g.get_weight() } -> std::same_as<float>;
      { g.max_distance_to(p) } -> std::same_as<float>;
      { g.get_taylor_terms(p, active, z, m, t) } -> std::same_as<void>;
      { g.contributionToQuery(p, f) } -> std::same_as<float>;
    };

template <typename T>
concept IsPrimitiveGeometry = requires(T g) {
  { g.get_aabb() } -> std::same_as<AABB>;
  { g.centroid() } -> std::same_as<Vec3>;
};
