#pragma once
#include "aabb.h"
#include "mat3x3.h"
#include "tailor_coefficients.h"
#include "tensor3.h"
#include "vec3.h"

// For leaf we don't have the center of mass or the max_distance
// We can still make conservative assumptions based on the aabb.
// If in doubt we don't approximate.
__device__ __forceinline__ auto
should_leaf_node_be_approximated(const Vec3 &query, const AABB &leaf_aabb,
                                 const float beta_2) -> bool {
  // Conservative assumption:
  // center of mass is at the aabb corner with the greatest distance.
  float dx =
      fmaxf(0.F, fmaxf(leaf_aabb.min.x - query.x, query.x - leaf_aabb.max.x));
  float dy =
      fmaxf(0.F, fmaxf(leaf_aabb.min.y - query.y, query.y - leaf_aabb.max.y));
  float dz =
      fmaxf(0.F, fmaxf(leaf_aabb.min.z - query.z, query.z - leaf_aabb.max.z));
  float dist_sq_to_box_corner = dx * dx + dy * dy + dz * dz;

  // The maximum possible radius within an AABB relative to ANY center of mass
  // is the full diagonal of the box.
  float max_possible_R_sq = leaf_aabb.diagonal().length2();

  return dist_sq_to_box_corner > max_possible_R_sq * beta_2;
}

__device__ __forceinline__ auto
should_inner_node_be_approximated(const Vec3 &query, const AABB &aabb,
                                  const float beta_2) -> bool {
  float max_distance_to_center =
      aabb.center_of_mass.getMaxDistance(aabb.diagonal().length());
  Vec3 com = aabb.center_of_mass.get(aabb.min, aabb.diagonal());
  float dist_query_to_com2 = (query - com).length2();
  return dist_query_to_com2 >
         max_distance_to_center * max_distance_to_center * beta_2;
}

__device__ __forceinline__ auto to_bits(const half2 f16x2) -> uint32_t {
  uint32_t result = *(uint32_t *)&f16x2;
  return result;
}

__device__ __forceinline__ auto to_bits(const half f16) -> short {
  short result = *(short *)&f16;
  return result;
}

/**
 * Computes the Zero Order Tailor contribution to the Winding Number.
 * This is the dot product between the normal and the Dipole Derivative.
 * Contribution = n * r / (4 * pi * ||r||^3)
 */
__device__ __forceinline__ auto
computeZeroOrderContribution(const Vec3_f16 &coeff, const Vec3_f16 &r,
                             const half inv_r3 // 1/(4 pi |r|^3)
                             ) -> float {
  // Pack f16s into 32-bit registers
  half2 c_xy = __halves2half2(coeff.x, coeff.y);
  half2 c_z = __halves2half2(coeff.z, __float2half(0.0f));

  half2 r_xy = __halves2half2(r.x, r.y);
  half2 r_z = __halves2half2(r.z, __float2half(0.0f));

  //  [cx*rx, cy*ry]
  half2 accumulator = __hmul2(c_xy, r_xy);
  // [ (cx*rx + cz*rz) | (cy*ry) ]
  accumulator = __hfma2(c_z, r_z, accumulator);
  half result =
      __hadd(__low2half(accumulator), __high2half(accumulator));
  result = __hmul(result, inv_r3);
  return __half2float(result);
}

/**
 * Computes the First Order Taylor contribution to the Winding Number.
 * * Mathematically, this is the Frobenius inner product (contraction)
 * between the first-order coefficient tensor C and the field gradient tensor G:
 * * Contribution = C : G = sum_{i,j} C_ij * G_ij
 * * where the field gradient tensor G at distance r is:
 * * G(r) = [ I / |r|^3 ] - [ (3 * r \otimes r) / |r|^5 ]
 * * (I is the Identity matrix, \otimes is the outer product).
 */
__device__ __forceinline__ auto
computeFirstOrderContribution(const Mat3x3_f16 &C, const Vec3_f16 &r,
                              const half inv_r3, // 1 / (4 pi |r|^3)
                              const half inv_r5  // 3 / (4 pi |r|^5)
                              ) -> float {
  half three = __float2half(3.F);
  // After unrolling and factoring we compute:
  // Contribution = (C_11+C_22+C_33)/(4pi||r||^3) -
  //                (C_11*r_1^2 + C_22*r_2^2 + C_33*r_3^2 +
  //                (C_12+C_21)*r_1*r_2 + (C_13+C_31)*r_1*r_3 +
  //                (C_23+C_32)*r_2*r_3)/(4pi||r||^5)

  // diagonal part
  half trace_C = __hadd(__hadd(C.data[0], C.data[4]), C.data[8]);

  // Pre-sum off-diagonals (symmetric parts)
  half C_xy_yx = __hadd(C.data[1], C.data[3]);
  half C_xz_zx = __hadd(C.data[2], C.data[6]);
  half C_yz_zy = __hadd(C.data[5], C.data[7]);

  // Prepare r_i * r_j pairs
  half2 r2_xy = __halves2half2(__hmul(r.x, r.x), __hmul(r.y, r.y));
  half2 r2_z_xy = __halves2half2(__hmul(r.z, r.z), __hmul(r.x, r.y));
  half2 r2_xz_yz = __halves2half2(__hmul(r.x, r.z), __hmul(r.y, r.z));

  // Pack G components
  half2 c_xy = __halves2half2(C.data[0],
                              C.data[4]);            // Cxx, Cyy
  half2 c_z_xy = __halves2half2(C.data[8], C_xy_yx); // Czz, Cxy+Cyx
  half2 c_xz_yz = __halves2half2(C_xz_zx, C_yz_zy);  // Cxz+zx, Cyz+Czy

  half2 zero = __halves2half2(0.F, 0.F);
  // accumulator = {Cxx*rx*rx + Czz*rz*rz + (Cxz+Czy)*rx*rz},
  //      {Cyy*ry*ry + (Cxy+Cyx)*rx*ry + (Cyz+Czy)*ry*rz}
  half2 accumulator = __hfma2(c_xy, r2_xy, zero);
  accumulator = __hfma2(c_z_xy, r2_z_xy, accumulator);
  accumulator = __hfma2(c_xz_yz, r2_xz_yz, accumulator);
  // result = (Cxx + Cyy + Czz) * inv_3 - (Cxx*rx*rx + Czz*rz*rz +
  // (Cxz+Czy)*rx*rz + Cyy*ry*ry + (Cxy+Cyx)*rx*ry + (Cyz+Czy)*ry*rz) * inv_5
  half result = __hadd(__low2half(accumulator), __high2half(accumulator));
  result = __hmul(result, inv_r5);
  half tmp = __hmul(trace_C, inv_r3);
  result = __hsub(tmp, result);
  return __half2float(result);
}

/**
 * Computes the Second Order Taylor contribution to the Winding Number.
 *
 * Mathematically, this is the contraction between the second-order
 * coefficient tensor C (rank 3) and the third-order field gradient tensor G
 * (rank 3):
 * * Contribution = C : G = sum_{i,j,k} C_ijk * G_ijk
 * * where the field gradient tensor G is:
 * G(r) = [ 15 * (r \otimes r \otimes r) / |r|^7 ] - [ 3 * (r \otimes I)_sym /
 * |r|^5 ]
 * * Here:
 * - I is the Identity matrix (I_jk = \delta_jk).
 * - \delta is the Kronecker delta (\delta_jk = 1 if j=k, else 0).
 * - \otimes is the outer product.
 * - (r \otimes I)_sym is the symmetric combination: (r_i \delta_jk + r_j
 * \delta_ik + r_k \delta_ij).
 *
 * Implementation note: Due to the partial symmetry of C (C_ijk = C_ikj) and the
 * cross-plane symmetries in this specific Taylor expansion, the 27-term
 * contraction reduces to 18 unique terms stored in the compressed format.
 * The second term (involving \delta) simplifies to a dot product between r and
 * the partial vector-trace of C.
 */
__device__ __forceinline__ auto
computeSecondOrderContribution(const Tensor3_f16_compressed &C,
                               const Vec3_f16 &r,
                               const half inv_r5, // 3.0f / (4pi * |r|^5)
                               const half inv_r7  // 15.0f / (4pi * |r|^7)
                               ) -> float {
  const half two = __float2half(2.0f);

  // Vector Trace V
  // Cxxx+Cxyy+Cxzz
  half vx = __hadd(__hadd(C.data[0], C.data[4]), C.data[8]);
  // Cyxx+Cyyy+Cyzz
  half vy = __hadd(__hadd(C.data[3], C.data[10]), C.data[14]);
  // Czxx+Czyy+Czzz
  half vz = __hadd(__hadd(C.data[6], C.data[13]), C.data[17]);
  // V dot r
  half v_dot_r =
      __hadd(__hadd(__hmul(vx, r.x), __hmul(vy, r.y)), __hmul(vz, r.z));

  // Pre-calculate baseline quadratic components
  half r_xx = __hmul(r.x, r.x);
  half r_yy = __hmul(r.y, r.y);
  half r_zz = __hmul(r.z, r.z);
  half r_xy = __hmul(r.x, r.y);
  half r_xz = __hmul(r.x, r.z);
  half r_yz = __hmul(r.y, r.z);

  // Accumulate on the fly
  // Chunk 0 & 1: xxx, xxy
  half2 accumulator =
      __hmul2(__halves2half2(C.data[0], C.data[1]),
              __halves2half2(__hmul(r_xx, r.x), __hmul(r_xx, r.y)));

  // Chunk 2 & 3: xxz, (yxx + xyx)*2
  accumulator =
      __hfma2(__halves2half2(C.data[2], C.data[3]),
              __halves2half2(__hmul(r_xx, r.z), __hmul(__hmul(r_xy, r.x), two)),
              accumulator);

  // Chunk 4 & 5: (xyy + yxy)*2, (xyz + yxz)*2
  accumulator = __hfma2(__halves2half2(C.data[4], C.data[5]),
                        __halves2half2(__hmul(__hmul(r_xy, r.y), two),
                                       __hmul(__hmul(r_xz, r.y), two)),
                        accumulator);

  // Chunk 6 & 7: (zxx + xzx)*2, (zxy + xzy)*2
  accumulator = __hfma2(__halves2half2(C.data[6], C.data[7]),
                        __halves2half2(__hmul(__hmul(r_xz, r.x), two),
                                       __hmul(__hmul(r_yz, r.x), two)),
                        accumulator);

  // Chunk 8 & 9: (zxz + xzz)*2, yyx
  accumulator =
      __hfma2(__halves2half2(C.data[8], C.data[9]),
              __halves2half2(__hmul(__hmul(r_xz, r.z), two), __hmul(r_yy, r.x)),
              accumulator);

  // Chunk 10 & 11: yyy, yyz
  accumulator = __hfma2(__halves2half2(C.data[10], C.data[11]),
                        __halves2half2(__hmul(r_yy, r.y), __hmul(r_yy, r.z)),
                        accumulator);

  // Chunk 12 & 13: (yzx + zyx)*2, (zyy + yzy)*2
  accumulator = __hfma2(__halves2half2(C.data[12], C.data[13]),
                        __halves2half2(__hmul(__hmul(r_yz, r.x), two),
                                       __hmul(__hmul(r_yz, r.y), two)),
                        accumulator);

  // Chunk 14 & 15: (yzz + zyz)*2, zzx
  accumulator =
      __hfma2(__halves2half2(C.data[14], C.data[15]),
              __halves2half2(__hmul(__hmul(r_yz, r.z), two), __hmul(r_zz, r.x)),
              accumulator);

  // Chunk 16 & 17: zzy, zzz
  accumulator = __hfma2(__halves2half2(C.data[16], C.data[17]),
                        __halves2half2(__hmul(r_zz, r.y), __hmul(r_zz, r.z)),
                        accumulator);

  // add together both results for the r^5 part
  half result =
      __hadd(__low2half(accumulator), __high2half(accumulator));
  // Final Scale: (result * inv_r7) - (v_dot_r * inv_r5)
  result = __hmul(result, inv_r7);
  result = __hsub(result, __hmul(v_dot_r, inv_r5));

  return __half2float(result);
}

__device__ __forceinline__ auto compute_node_approximation(
    const Vec3 &query, const Vec3 &center_of_mass,
    const Vec3_f16 &zero_order_coeff, const Mat3x3_f16 &first_order_coeff,
    const Tensor3_f16_compressed &second_order_coeff) -> float {
  Vec3 r = center_of_mass - query;
  float inv_norm_r = r.inv_length();
  float inv_norm_r3 = inv_norm_r * inv_norm_r * inv_norm_r;
  float inv_norm_r5 = inv_norm_r3 * inv_norm_r * inv_norm_r;
  float inv_norm_r7 = inv_norm_r5 * inv_norm_r * inv_norm_r;

  // 1/(4*pi)
  float inv_4pi = (0.07957747154F);

  float inv_4_pi_normr3 = inv_4pi * inv_norm_r3;
  float inv_4_pi_normr5 = (3.F) * inv_4pi * inv_norm_r5;
  float inv_4_pi_normr7 = (15.F) * inv_4pi * inv_norm_r7;

  Vec3_f16 r_f16 = Vec3_f16::from_float(r);
  float result = 0.F;
  // Zero Order
  result +=
      computeZeroOrderContribution(zero_order_coeff, r_f16, inv_4_pi_normr3);
  // First Order
  result += computeFirstOrderContribution(first_order_coeff, r_f16,
                                          inv_4_pi_normr3, inv_4_pi_normr5);
  // Second order
  result += computeSecondOrderContribution(second_order_coeff, r_f16,
                                           inv_4_pi_normr5, inv_4_pi_normr7);

  return result;
}
