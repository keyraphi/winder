#pragma once
#include "aabb.h"
#include "mat3x3.h"
#include "vec3.h"
#include <cmath>
#include <cstdint>
#include <cuda_runtime_api.h>
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
    return {min, max};
  }
  __host__ __device__ __forceinline__ auto centroid() const -> Vec3 {
    return (v0 + v1 + v2) / 3.F;
  }

  __device__ __forceinline__ static auto load(const Triangle *base, uint32_t idx,
                                              uint32_t count) -> Triangle;

  __device__ __forceinline__ auto get_scaled_normal() const -> Vec3;

  __device__ __forceinline__ auto get_tailor_terms(const Vec3 &p_center,
                                                   Vec3 &out_n, Vec3 &out_d,
                                                   SymMat3x3 &out_Ct) const
      -> void;
};

struct PointNormal {
  Vec3 p;
  Vec3 n;

  __device__ __forceinline__ static auto load(const PointNormal *base, uint32_t idx,
                                              uint32_t count) -> PointNormal;

  __device__ __forceinline__ auto get_aabb() const -> AABB;
  __device__ __forceinline__ auto centroid() const -> Vec3;
  __device__ __forceinline__ auto get_scaled_normal() const -> Vec3;
  __device__ __forceinline__ auto get_tailor_terms(const Vec3 &p_center,
                                                   Vec3 &out_n, Vec3 &out_d,
                                                   SymMat3x3 &out_Ct) const
      -> void;
};

#ifdef __CUDACC__
__device__ __forceinline__ auto Triangle::load(const Triangle *base, uint32_t idx,
                                               uint32_t count) -> Triangle {
  if (idx < count) {
    // Triangle is 9 floats (36 bytes).
    const Vec3 *ptr = reinterpret_cast<const Vec3 *>(base + idx);
    return {ptr[0], ptr[1], ptr[2]};
  }
  return {{1e38f, 1e38f, 1e38f}, {1e38f, 1e38f, 1e38f}, {1e38f, 1e38f, 1e38f}};
}

__device__ __forceinline__ auto Triangle::get_scaled_normal() const -> Vec3 {
  // Area-weighted normal for Winding Number
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
#endif
