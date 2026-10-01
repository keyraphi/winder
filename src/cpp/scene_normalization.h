#pragma once
#include "aabb.h"
#include "geometry.h"
#include "vec3.h"
#include <cmath>

struct SceneNormalization {
  Vec3 min_p;   // raw bbox min
  float scale;  // 1 / max_dim
  float scale2; // scale^2, for scaled-normal gradients

  static auto effective_extent(const AABB& b) -> float {
    Vec3 extent = Vec3::from_f16(b.diagonal());
    float max_dim = fmaxf(extent.x, fmaxf(extent.y, extent.z));
    // Guard against degenerate (zero-extent) AABBs. In that case, fall back
    // to identity so downstream math doesn't divide by zero.
    return (max_dim < 1e-20F || !isfinite(max_dim)) ? 1.F : max_dim;
  }

  static auto effective_extent(const SceneBounds& b) -> float {
    Vec3 extent = b.diagonal();
    float max_dim = fmaxf(extent.x, fmaxf(extent.y, extent.z));
    // Guard against degenerate (zero-extent) SceneBounds. In that case, fall back
    // to identity so downstream math doesn't divide by zero.
    return (max_dim < 1e-20F || !isfinite(max_dim)) ? 1.F : max_dim;
  }

  static auto from_scene_bounds(const SceneBounds &b) -> SceneNormalization {
    float max_dim = effective_extent(b);
    Vec3 raw_min = b.min;
    float s = 1.F / max_dim;
    return {.min_p = raw_min, .scale = s, .scale2 = s * s};
  }

  static auto identity() -> SceneNormalization {
    return {.min_p = Vec3{0.f, 0.f, 0.f}, .scale = 1.f, .scale2 = 1.f};
  }

  __host__ __device__ auto to_normalized(const Vec3 &p) const -> Vec3 {
    return (p - min_p) * scale;
  }

  __host__ __device__ auto to_normalized(const PointNormal &pn) const
      -> PointNormal {
    return {.p = to_normalized(pn.p), .n = pn.n * scale2};
  }

  __host__ __device__ auto to_normalized(const Triangle &t) const -> Triangle {
    return {.v0 = to_normalized(t.v0),
            .v1 = to_normalized(t.v1),
            .v2 = to_normalized(t.v2)};
  }
};
