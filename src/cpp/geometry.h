#pragma once
#include "aabb.h"
#include "mat3x3.h"
#include "soa.h"
#include "tensor3.h"
#include "vec3.h"
#include <cmath>
#include <concepts>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <format>
#include <numbers>
#include <string>
#include <type_traits>
#include <vector_types.h>

// =============================================================================
// Regularization strategies
//
//   RegSharp    r_a = |a|                     exact, no regularization
//   RegPlummer  r_a = sqrt(|a|^2 + eps^2)     Plummer softening
//   RegCompact  r_a = eps * g(|a|/eps)        compact-support polynomial
//                                             (quartic on [0,1], cubic
//                                              bridge on [1,2], identity
//                                              for t > 2)
//
// `eps = 0` at runtime falls back to the sharp kernel for RegPlummer and
// RegCompact. RegSharp ignores `eps` and takes the sharp branch at compile
// time.
//
// The library-wide default is `DefaultReg`. Override with
//   -DWINDER_DEFAULT_REG=RegCompact
// or by editing the typedef below.
// =============================================================================

struct RegSharp {};
struct RegPlummer {};
struct RegCompact {};

#ifndef WINDER_DEFAULT_REG
using DefaultReg = RegPlummer;
#else
using DefaultReg = WINDER_DEFAULT_REG;
#endif

#define TWO_OVER_SQRT_PI 1.1283791671F
#define FOUR_OVER_3SQRT_PI 0.75225277806F // (4 / (3 * sqrt(pi)))
#define INV_FOUR_PI 0.07957747154F
#define INV_TWO_PI 0.15915494309F

// =============================================================================
// Contexts (all values precomputed on host, passed by value to kernels)
// =============================================================================

template <typename Reg> struct TriangleContext;
struct PointNormalContext;

// =============================================================================
// Regularized edge -- internal helper shared by forward and backward
// =============================================================================
struct RegularizedEdge {
  float inv_r;     // 1 / r              (always)
  float r;         // r                  (only when NeedFull)
  float hat_scale; // dr/dv = a * hat_scale  (only when NeedFull)
};

namespace winder_reg_compact {
constexpr float SQRT2 = std::numbers::sqrt2_v<float>;
constexpr float INV_SQRT2 = 0.7071067811865476F;

// Quartic [0, 1]: g(t) = 1 + A2 t^2 + A4 t^4
constexpr float A2 = 0.4748737341529164F;
constexpr float A4 = -0.0606601717798213F;

// Cubic bridge [1, 2]: g(u) = V + S u + C u^2 + D u^3, u = t - 1
constexpr float V = SQRT2;
constexpr float S = INV_SQRT2;
constexpr float C = -0.6568542494923802F; //  5 - 4 sqrt(2)
constexpr float D = 0.5355339059327378F;  // -3 + 5/sqrt(2)
} // namespace winder_reg_compact

// PointNormal regularization
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

// =============================================================================
// Triangle contexts -- precompute everything once per launch
// =============================================================================

template <> struct TriangleContext<RegSharp> {
  __host__ __device__ __forceinline__ static auto make(float /*eps*/)
      -> TriangleContext {
    return {};
  }
};

template <> struct TriangleContext<RegPlummer> {
  float eps;  // softening length (needed for traversal criterion)
  float eps2; // softening length squared

  __host__ __device__ __forceinline__ static auto make(float eps)
      -> TriangleContext {
    return TriangleContext{.eps=eps, .eps2=eps * eps};
  }
};

template <> struct TriangleContext<RegCompact> {
  float eps;      // softening length (needed for `g * eps`)
  float eps2;     // eps^2 (for the sharp fallback test)
  float inv_eps;  // 1 / eps
  float inv_eps2; // 1 / eps^2

  __host__ __device__ __forceinline__ static auto make(float eps)
      -> TriangleContext {
    TriangleContext c{
        .eps = eps, .eps2 = eps * eps, .inv_eps = 0.F, .inv_eps2 = 0.F};
    if (eps > 0.F) {
      c.inv_eps = 1.F / eps;
      c.inv_eps2 = 1.F / c.eps2;
    }
    return c;
  }
};

// =============================================================================
// PointNormal context -- precompute all derived scalars
// =============================================================================
struct PointNormalContext {
  float inv_epsilon;        // 1 / epsilon
  float reg_term_const;     // inv_epsilon^3 * INV_PI_1_5
  float near_field_g_denum; // (INV_PI_1_5 / 3) * inv_epsilon^3

  __host__ __device__ __forceinline__ static auto make(float eps)
      -> PointNormalContext {
    constexpr float INV_PI_1_5 = 0.179587122F;
    const float inv_eps = 1.F / eps;
    const float inv_eps3 = inv_eps * inv_eps * inv_eps;
    return PointNormalContext{.inv_epsilon = inv_eps,
                              .reg_term_const = inv_eps3 * INV_PI_1_5,
                              .near_field_g_denum =
                                  (INV_PI_1_5 / 3.F) * inv_eps3};
  }
};

// =============================================================================
// Edge evaluators
//
// Only the fields under `if constexpr (NeedFull)` are written when NeedFull
// is false; the compiler eliminates the rest at each instantiation.
// =============================================================================

template <bool NeedFull>
__host__ __device__ __forceinline__ auto sharp_edge(const float a2)
    -> RegularizedEdge {
  RegularizedEdge out;
#if defined(__CUDA_ARCH__)
  out.inv_r = rsqrtf(a2);
#else
  out.inv_r = 1.F / sqrtf(a2);
#endif
  if constexpr (NeedFull) {
    out.r = a2 * out.inv_r;    // = |a|
    out.hat_scale = out.inv_r; // d|a|/dv = a / |a|
  }
  return out;
}

template <bool NeedFull>
__host__ __device__ __forceinline__ auto plummer_edge(const float a2,
                                                      const float eps2)
    -> RegularizedEdge {
  const float r2 = a2 + eps2;
  RegularizedEdge out;
#if defined(__CUDA_ARCH__)
  out.inv_r = rsqrtf(r2);
#else
  out.inv_r = 1.F / sqrtf(r2);
#endif
  if constexpr (NeedFull) {
    out.r = r2 * out.inv_r; // = sqrt(a2 + eps2)
    out.hat_scale = out.inv_r;
  }
  return out;
}

template <bool NeedFull>
__host__ __device__ __forceinline__ auto
compact_edge(const float a2, const float eps, const float inv_eps,
             const float inv_eps2) -> RegularizedEdge {
  using namespace winder_reg_compact;

  RegularizedEdge out;
  const float t2 = a2 * inv_eps2; // (|a| / eps)^2

  if (t2 <= 1.F) {
    // Quartic [0, 1].  No sqrt.
    const float t4 = t2 * t2;
    const float g = 1.F + A2 * t2 + A4 * t4;
    out.inv_r = inv_eps / g;
    if constexpr (NeedFull) {
      out.r = g * eps;
      out.hat_scale = (2.F * A2 + 4.F * A4 * t2) * inv_eps2;
    }
  } else if (t2 <= 4.F) {
    // Cubic bridge [1, 2].
    const float t = sqrtf(t2);
    const float u = t - 1.F;
    const float u2 = u * u;
    const float u3 = u2 * u;
    const float g = V + S * u + C * u2 + D * u3;
    out.inv_r = inv_eps / g;
    if constexpr (NeedFull) {
      const float gp = S + 2.F * C * u + 3.F * D * u2;
      out.r = g * eps;
      out.hat_scale = gp * inv_eps / t;
    }
  } else {
    // Sharp r = |a| exactly.
#if defined(__CUDA_ARCH__)
    out.inv_r = rsqrtf(a2);
#else
    out.inv_r = 1.F / sqrtf(a2);
#endif
    if constexpr (NeedFull) {
      out.r = a2 * out.inv_r;
      out.hat_scale = out.inv_r;
    }
  }
  return out;
}

template <typename Reg, bool NeedFull>
__host__ __device__ __forceinline__ auto
regularize_edge(const float a2, const TriangleContext<Reg> ctx)
    -> RegularizedEdge {
  if constexpr (std::is_same_v<Reg, RegSharp>) {
    return sharp_edge<NeedFull>(a2);
  } else if constexpr (std::is_same_v<Reg, RegPlummer>) {
    if (ctx.eps2 <= 0.F) {
      return sharp_edge<NeedFull>(a2);
    }
    return plummer_edge<NeedFull>(a2, ctx.eps2);
  } else {
    if (ctx.eps <= 0.F) {
      return sharp_edge<NeedFull>(a2);
    }
    return compact_edge<NeedFull>(a2, ctx.eps, ctx.inv_eps, ctx.inv_eps2);
  }
}

// =============================================================================
// Triangle
// =============================================================================
struct Triangle {
  using Context = TriangleContext<DefaultReg>;

  Vec3 v0, v1, v2;

  __host__ __device__ __forceinline__ auto
  operator+(const Triangle &other) const -> Triangle {
    return Triangle{
        .v0 = v0 + other.v0, .v1 = v1 + other.v1, .v2 = v2 + other.v2};
  }
  __host__ __device__ __forceinline__ auto
  operator-(const Triangle &other) const -> Triangle {
    return Triangle{
        .v0 = v0 - other.v0, .v1 = v1 - other.v1, .v2 = v2 - other.v2};
  }
  __host__ __device__ __forceinline__ auto operator+=(const Triangle &other)
      -> Triangle & {
    v0 += other.v0;
    v1 += other.v1;
    v2 += other.v2;
    return *this;
  }

  __host__ __device__ __forceinline__ auto get_bounds() const -> SceneBounds {
    Vec3 min{
        .x = fminf(v0.x, fminf(v1.x, v2.x)),
        .y = fminf(v0.y, fminf(v1.y, v2.y)),
        .z = fminf(v0.z, fminf(v1.z, v2.z)),
    };
    Vec3 max{
        .x = fmaxf(v0.x, fmaxf(v1.x, v2.x)),
        .y = fmaxf(v0.y, fmaxf(v1.y, v2.y)),
        .z = fmaxf(v0.z, fmaxf(v1.z, v2.z)),
    };
    SceneBounds result;
    result.min = min;
    result.max = max;
    return result;
  }

  __host__ __device__ __forceinline__ auto get_aabb() const -> AABB {
    Vec3 min{
        .x = fminf(v0.x, fminf(v1.x, v2.x)),
        .y = fminf(v0.y, fminf(v1.y, v2.y)),
        .z = fminf(v0.z, fminf(v1.z, v2.z)),
    };
    Vec3 max{
        .x = fmaxf(v0.x, fmaxf(v1.x, v2.x)),
        .y = fmaxf(v0.y, fmaxf(v1.y, v2.y)),
        .z = fmaxf(v0.z, fmaxf(v1.z, v2.z)),
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
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 cp = Vec3{.x = e1.y * e2.z - e1.z * e2.y,
                   .y = e1.z * e2.x - e1.x * e2.z,
                   .z = e1.x * e2.y - e1.y * e2.x};
    return 0.5F * sqrtf(cp.x * cp.x + cp.y * cp.y + cp.z * cp.z);
  }
  __host__ __device__ __forceinline__ auto distance_to2(const Vec3 &pos) const
      -> float {
    const Vec3 e10 = v1 - v0;
    const Vec3 e20 = v2 - v0;

    const Vec3 p0 = pos - v0;
    const float d1 = e10.dot(p0);
    const float d2 = e20.dot(p0);
    if (d1 <= 0 && d2 <= 0) {
      return (pos - v0).length2();
    }

    const Vec3 p1 = pos - v1;
    const float d3 = e10.dot(p1);
    const float d4 = e20.dot(p1);
    if (d3 >= 0 && d4 <= d3) {
      return (pos - v1).length2();
    }

    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) {
      const float v = d1 / (d1 - d3);
      return (pos - (v0 + v * e10)).length2();
    }

    const Vec3 p2 = pos - v2;
    const float d5 = e10.dot(p2);
    const float d6 = e20.dot(p2);
    if (d6 >= 0 && d5 <= d6) {
      return (pos - v2).length2();
    }

    const float vb = d5 * d2 - d1 * d6;
    if (vb < 0 && d2 >= 0 && d6 <= 0) {
      const float w = d2 / (d2 - d6);
      return (pos - (v0 + w * e20)).length2();
    }

    const float va = d3 * d6 - d5 * d4;
    if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
      const float w = (d4 - d3) / ((d5 - d3) + (d5 - d6));
      return (pos - (v1 + w * (v2 - v1))).length2();
    }

    const float denom = 1.F / (va + vb + vc);
    const float v = vb * denom;
    const float w = vc * denom;
    const Vec3 p_dash = v0 + v * e10 + w * e20;
    return (pos - p_dash).length2();
  }
  __host__ __device__ __forceinline__ auto distance_to(const Vec3 &pos) const
      -> float {
    return sqrtf(distance_to2(pos));
  }
  __host__ __device__ __forceinline__ auto
  min_vert_distance_to2(const Vec3 &pos) const -> float {
    float d0 = (v0 - pos).length2();
    float d1 = (v1 - pos).length2();
    float d2 = (v2 - pos).length2();
    return fminf(d0, fminf(d1, d2));
  }
  __host__ __device__ __forceinline__ auto
  max_distance_to2(const Vec3 &pos) const -> float {
    float d0 = (v0 - pos).length2();
    float d1 = (v1 - pos).length2();
    float d2 = (v2 - pos).length2();
    return fmaxf(d0, fmaxf(d1, d2));
  }
  __host__ __device__ __forceinline__ auto
  max_distance_to(const Vec3 &pos) const -> float {
    return sqrtf(max_distance_to2(pos));
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

  // -------------------------------------------------------------------------
  // Forward: contribution to winding number at query.
  // -------------------------------------------------------------------------
  template <typename Reg = DefaultReg>
  __host__ __device__ __forceinline__ auto
  contributionToQuery(const Vec3 &query, const TriangleContext<Reg> ctx) const
      -> float {
    const Vec3 a = v0 - query;
    const Vec3 b = v1 - query;
    const Vec3 c = v2 - query;

    const float a2 = a.length2() + 1e-20F;
    const float b2 = b.length2() + 1e-20F;
    const float c2 = c.length2() + 1e-20F;

    const float inv_a = regularize_edge<Reg, false>(a2, ctx).inv_r;
    const float inv_b = regularize_edge<Reg, false>(b2, ctx).inv_r;
    const float inv_c = regularize_edge<Reg, false>(c2, ctx).inv_r;

    const float cos_ab = a.dot(b) * inv_a * inv_b;
    const float cos_ac = a.dot(c) * inv_a * inv_c;
    const float cos_bc = b.dot(c) * inv_b * inv_c;

    const float det_norm = a.dot(Vec3::cross(b, c)) * inv_a * inv_b * inv_c;
    const float div_norm = 1.F + cos_ab + cos_ac + cos_bc;

    if (fabsf(div_norm) < 1e-6F) {
      return 0.5F;
    }
    return atan2f(det_norm, div_norm) * INV_TWO_PI;
  }

  // -------------------------------------------------------------------------
  // Backward: gradient of contributionToQuery with respect to v0, v1, v2.
  // -------------------------------------------------------------------------
  template <typename Reg = DefaultReg>
  __host__ __device__ __forceinline__ auto
  gradContributionOfQuery(const Vec3 &q, float g,
                          const TriangleContext<Reg> ctx) const -> Triangle {
    const Vec3 a = v0 - q;
    const Vec3 b = v1 - q;
    const Vec3 c = v2 - q;

    const float a2 = a.length2() + 1e-20F;
    const float b2 = b.length2() + 1e-20F;
    const float c2 = c.length2() + 1e-20F;

    const RegularizedEdge ra = regularize_edge<Reg, true>(a2, ctx);
    const RegularizedEdge rb = regularize_edge<Reg, true>(b2, ctx);
    const RegularizedEdge rc = regularize_edge<Reg, true>(c2, ctx);

    const float inv_a = ra.inv_r;
    const float inv_b = rb.inv_r;
    const float inv_c = rc.inv_r;
    const float a_len = ra.r;
    const float b_len = rb.r;
    const float c_len = rc.r;
    const Vec3 hat_a = a * ra.hat_scale;
    const Vec3 hat_b = b * rb.hat_scale;
    const Vec3 hat_c = c * rc.hat_scale;

    const float cos_ab = a.dot(b) * inv_a * inv_b;
    const float cos_ac = a.dot(c) * inv_a * inv_c;
    const float cos_bc = b.dot(c) * inv_b * inv_c;

    const float det_norm = a.dot(Vec3::cross(b, c)) * inv_a * inv_b * inv_c;
    const float div_norm = 1.F + cos_ab + cos_ac + cos_bc;

    const float denom_norm = det_norm * det_norm + div_norm * div_norm;
    if (denom_norm < 1e-12F) {
      return Triangle{.v0 = Vec3{.x = 0.F, .y = 0.F, .z = 0.F},
                      .v1 = Vec3{.x = 0.F, .y = 0.F, .z = 0.F},
                      .v2 = Vec3{.x = 0.F, .y = 0.F, .z = 0.F}};
    }

    const Vec3 dN_dv0 = Vec3::cross(b, c);
    const Vec3 dN_dv1 = Vec3::cross(c, a);
    const Vec3 dN_dv2 = Vec3::cross(a, b);

    const float b_dot_c = b.dot(c);
    const float c_dot_a = c.dot(a);
    const float a_dot_b = a.dot(b);

    const Vec3 dD_dv0 =
        hat_a * (b_len * c_len + b_dot_c) + b * c_len + c * b_len;
    const Vec3 dD_dv1 =
        hat_b * (c_len * a_len + c_dot_a) + c * a_len + a * c_len;
    const Vec3 dD_dv2 =
        hat_c * (a_len * b_len + a_dot_b) + a * b_len + b * a_len;

    const float inv_L = inv_a * inv_b * inv_c;
    const float factor = g * INV_TWO_PI * (inv_L / denom_norm);

    return Triangle{.v0 = (dN_dv0 * div_norm - dD_dv0 * det_norm) * factor,
                    .v1 = (dN_dv1 * div_norm - dD_dv1 * det_norm) * factor,
                    .v2 = (dN_dv2 * div_norm - dD_dv2 * det_norm) * factor};
  }

  [[nodiscard]] auto dump() const -> std::string {
    return std::format("v0:({:.2f}, {:.2f}, {:.2f}) | v1:({:.2f}, {:.2f}, "
                       "{:.2f}) | v2:({:.2f}, {:.2f}, {:.2f})",
                       v0.x, v0.y, v0.z, v1.x, v1.y, v1.z, v2.x, v2.y, v2.z);
  }
};

// =============================================================================
// PointNormal
// =============================================================================
struct PointNormal {
  using Context = PointNormalContext;

  Vec3 p;
  Vec3 n;

  __host__ __device__ __forceinline__ auto
  operator+(const PointNormal &other) const -> PointNormal {
    return PointNormal{.p = p + other.p, .n = n + other.n};
  }
  __host__ __device__ __forceinline__ auto
  operator-(const PointNormal &other) const -> PointNormal {
    return PointNormal{.p = p - other.p, .n = n - other.n};
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

  __host__ __device__ __forceinline__ auto get_bounds() const -> SceneBounds;
  __host__ __device__ __forceinline__ auto get_aabb() const -> AABB;
  __host__ __device__ __forceinline__ auto centroid() const -> Vec3;
  __host__ __device__ __forceinline__ auto get_scaled_normal() const -> Vec3;
  __host__ __device__ __forceinline__ auto
  get_taylor_terms(const Vec3 &p_center, bool is_active, Vec3 &zero_order,
                   Mat3x3 &first_order, Tensor3_compressed &second_order) const
      -> void;

  __host__ __device__ __forceinline__ auto
  contributionToQuery(const Vec3 &query, const PointNormalContext ctx) const
      -> float {
    const Vec3 d = p - query;
    const float dist2 = d.length2();

#if defined(__CUDA_ARCH__)
    const float inv_distance = rsqrtf(dist2 + 1e-20F);
#else
    const float inv_distance = 1.F / sqrtf(dist2 + 1e-20F);
#endif
    const float inv_dist2 = inv_distance * inv_distance;
    const float inv_dist3 = inv_dist2 * inv_distance;

    const float distance = dist2 * inv_distance;
    const float t = distance * ctx.inv_epsilon;

    float s_over_dist3;
    if (t < 2.F) {
      if (t < 0.1F) {
        s_over_dist3 = FOUR_OVER_3SQRT_PI *
                       (ctx.inv_epsilon * ctx.inv_epsilon * ctx.inv_epsilon);
      } else {
        s_over_dist3 = S_regularization(t) * inv_dist3;
      }
    } else {
      s_over_dist3 = inv_dist3;
    }
    return n.dot(d) * INV_FOUR_PI * s_over_dist3;
  }

  __host__ __device__ __forceinline__ auto
  gradContributionOfQuery(const Vec3 &q, const float g,
                          const PointNormalContext ctx) const -> PointNormal {
    const Vec3 d = p - q;
    const float dist2 = d.x * d.x + d.y * d.y + d.z * d.z;

#if defined(__CUDA_ARCH__)
    const float inv_dist = rsqrtf(dist2 + 1e-20F);
#else
    const float inv_dist = 1.F / sqrtf(dist2 + 1e-20F);
#endif
    const float inv_dist2 = inv_dist * inv_dist;
    const float inv_dist3 = inv_dist2 * inv_dist;

    const float distance = dist2 * inv_dist;
    const float t = distance * ctx.inv_epsilon;

    float scale_n;
    float scale_d;

    if (t < 0.1F) {
      scale_n = g * ctx.near_field_g_denum;
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
        reg_term = exp_t2 * ctx.reg_term_const;
      } else {
        s_over_dist3 = inv_dist3;
      }

      const float g_denum = INV_FOUR_PI * s_over_dist3;
      const float dot = n.x * d.x + n.y * d.y + n.z * d.z;
      const float shared_factor = dot * inv_dist2;

      scale_n = g * g_denum;
      scale_d = g * shared_factor * (reg_term - 3.F * g_denum);
    }

    return PointNormal{
        .p = Vec3{.x = scale_n * n.x + scale_d * d.x,
                  .y = scale_n * n.y + scale_d * d.y,
                  .z = scale_n * n.z + scale_d * d.z},
        .n = Vec3{.x = scale_n * d.x, .y = scale_n * d.y, .z = scale_n * d.z}};
  }

  [[nodiscard]] auto dump() const -> std::string {
    return std::format(
        "Pos:({:.2f}, {:.2f}, {:.2f}) | Norm:({:.2f}, {:.2f}, {:.2f})", p.x,
        p.y, p.z, n.x, n.y, n.z);
  }
};

// =============================================================================
// Triangle / PointNormal load and taylor-term implementations
// =============================================================================
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
  return {.v0 = {.x = 1e38F, .y = 1e38F, .z = 1e38F},
          .v1 = {.x = 1e38F, .y = 1e38F, .z = 1e38F},
          .v2 = {.x = 1e38F, .y = 1e38F, .z = 1e38F}};
}

__host__ __device__ __forceinline__ auto Triangle::get_scaled_normal() const
    -> Vec3 {
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
  return {.p = {.x = 1e38F, .y = 1e38F, .z = 1e38F},
          .n = {.x = 0.F, .y = 0.F, .z = 0.F}};
}

__host__ __device__ __forceinline__ auto PointNormal::get_bounds() const
    -> SceneBounds {
  return p.get_bounds();
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

    Vec3 m_ij = ((v0 + v1) * 0.5F) - p_center;
    Vec3 m_jk = ((v1 + v2) * 0.5F) - p_center;
    Vec3 m_ki = ((v2 + v0) * 0.5F) - p_center;

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
    zero_order = Vec3{.x = 0.F, .y = 0.F, .z = 0.F};
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
    Vec3 n = get_scaled_normal();
    zero_order = n;

    Vec3 r = p - p_center;
    first_order = r.outer_product(n);

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
    zero_order = Vec3{.x = 0.F, .y = 0.F, .z = 0.F};
    first_order = Mat3x3::zero();
    for (int i = 0; i < 18; ++i) {
      second_order.data[i] = 0.F;
    }
  }
}

// =============================================================================
// Concept -- both geometries must expose a `Context` type and take it by value
// =============================================================================
template <typename T>
concept IsGeometry =
    requires(T g, typename T::Context ctx, Vec3 p, bool active, Vec3 &z,
             Mat3x3 &m, Tensor3_compressed &t, float f, uint32_t u) {
      { T::load(SoAView<T>{}, u, u) } -> std::same_as<T>;
      { g.get_aabb() } -> std::same_as<AABB>;
      { g.centroid() } -> std::same_as<Vec3>;
      { g.get_weight() } -> std::same_as<float>;
      { g.max_distance_to(p) } -> std::same_as<float>;
      { g.get_taylor_terms(p, active, z, m, t) } -> std::same_as<void>;
      { g.contributionToQuery(p, ctx) } -> std::same_as<float>;
    };

template <typename T>
concept IsPrimitiveGeometry = requires(T g) {
  { g.get_aabb() } -> std::same_as<AABB>;
  { g.get_bounds() } -> std::same_as<SceneBounds>;
  { g.centroid() } -> std::same_as<Vec3>;
};
