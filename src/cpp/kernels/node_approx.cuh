#pragma once
#include "aabb.h"
#include "geometry.h"
#include "mat3x3.h"
#include "tensor3.h"
#include "vec3.h"
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

__device__ __forceinline__ auto
should_node_be_approximated(const Vec3 &query, const AABB &aabb,
                            const float beta_2, const float two_eps = 0.F)
    -> bool {
  float R = __half2float(aabb.max_distance);
  Vec3 com = aabb.center_of_mass;
  float dist_query_to_com2 = (query - com).length2();
  float threshold_r2 = R * R * beta_2;
    const float Re = R + two_eps;
    const float threshhold_e2 = Re * Re;
    return dist_query_to_com2 > fmaxf(threshold_r2, threshhold_e2);
}

__device__ __forceinline__ auto
should_node_be_approximated(const PointNormal &geometry, const AABB &aabb,
                            const float beta, const PointNormalContext &ctx)
    -> bool {
  const float R = __half2float(aabb.max_distance);
  const float dist_com2 = (geometry.centroid() - aabb.center_of_mass).length2();

  if (ctx.inv_eps_length <= 0.F) {
    const float one_plus_beta = 1.F + beta;
    return dist_com2 > (one_plus_beta * one_plus_beta) * (R * R);
  }
  const float thresh_r = (1.F + beta) * R;
  const float thresh_e = R + 2.F * ctx.eps_length;
  const float thresh = fmaxf(thresh_r, thresh_e);
  return dist_com2 > thresh * thresh;
}

template <typename Reg>
__device__ __forceinline__ auto
should_node_be_approximated(const Triangle &geometry, const AABB &aabb,
                            const float beta, const TriangleContext<Reg> ctx)
    -> bool {
  const Vec3 com = aabb.center_of_mass;
  const float R = __half2float(aabb.max_distance);
  const float dist_com2 = geometry.distance_to2(com);

  if constexpr (std::is_same_v<Reg, RegSharp>) {
    // Sharp: only the BH opening criterion applies.
    const float one_plus_beta = 1.F + beta;
    return dist_com2 > (one_plus_beta * one_plus_beta) * (R * R);
  } else {
    // Regularized: ball must clear both the band and the BH opening cone.
    const float thresh_r = (1.F + beta) * R;
    const float thresh_e = R + 2.F * ctx.eps_length;
    const float thresh = fmaxf(thresh_r, thresh_e);
    return dist_com2 > thresh * thresh;
  }
}

/**
 * @brief Computes the unscaled Zero-Order Taylor contraction for the Winding
 * Number.
 *
 * This calculates the directional projection of the zero-order multipole
 * coefficient (area-weighted normal) along the unit displacement vector r_hat.
 *
 * Mathematical formulation:
 * Contraction = coeff * r_hat
 *
 * The complete physical contribution is reconstructed by the caller via:
 * Contribution = Contraction * [1 / (4 * pi * ||r||^2)]
 *
 * @param coeff The zero-order multipole coefficient vector.
 * @param r_hat Normalized unit displacement vector pointing from query to the
 * cluster center (||r_hat|| = 1).
 * @return Unscaled contraction scalar evaluated in float16 precision.
 */
__device__ __forceinline__ auto
computeZeroOrderContribution(const Vec3_f16 &coeff, const Vec3_f16 &r_hat)
    -> float {
  half2 c_xy = __halves2half2(coeff.x, coeff.y);
  half2 c_z = __halves2half2(coeff.z, __float2half(0.0f));

  half2 r_xy = __halves2half2(r_hat.x, r_hat.y);
  half2 r_z = __halves2half2(r_hat.z, __float2half(0.0f));

  half2 accumulator = __hmul2(c_xy, r_xy);
  accumulator = __hfma2(c_z, r_z, accumulator);
  half result = __hadd(__low2half(accumulator), __high2half(accumulator));

  return __half2float(result);
}

/**
 * @brief Computes the unscaled First-Order Taylor contraction for the Winding
 * Number.
 *
 * Computes the Frobenius inner product (double contraction) between the
 * first-order multipole coefficient tensor C (rank 2) and the scale-invariant
 * field gradient component.
 *
 * Mathematical formulation:
 * Contraction = C : G_hat = trace(C) - 3 * sum_{i,j} (C_ij * r_hat_i * r_hat_j)
 *
 * where the normalized, unitless field gradient tensor component G_hat is
 * defined as: G_hat = I - 3 * (r_hat x r_hat)
 *
 * The complete physical contribution is reconstructed by the caller via:
 * Contribution = Contraction * [1 / (4 * pi * ||r||^3)]
 *
 * @param C The first-order rank-2 multipole coefficient tensor.
 * @param r_hat Normalized unit displacement vector pointing from query to the
 * cluster center (||r_hat|| = 1).
 * @return Unscaled contraction scalar evaluated in float16 precision.
 */
__device__ __forceinline__ auto
computeFirstOrderContribution(const Mat3x3_f16 &C, const Vec3_f16 &r_hat)
    -> float {
  // diagonal part
  half trace_C = __hadd(__hadd(C.data[0], C.data[4]), C.data[8]);

  // Pre-sum off-diagonals (symmetric parts)
  half C_xy_yx = __hadd(C.data[1], C.data[3]);
  half C_xz_zx = __hadd(C.data[2], C.data[6]);
  half C_yz_zy = __hadd(C.data[5], C.data[7]);

  // Prepare r_hat_i * r_hat_j pairs
  half2 r2_xy =
      __halves2half2(__hmul(r_hat.x, r_hat.x), __hmul(r_hat.y, r_hat.y));
  half2 r2_z_xy =
      __halves2half2(__hmul(r_hat.z, r_hat.z), __hmul(r_hat.x, r_hat.y));
  half2 r2_xz_yz =
      __halves2half2(__hmul(r_hat.x, r_hat.z), __hmul(r_hat.y, r_hat.z));

  // Pack G components
  half2 c_xy = __halves2half2(C.data[0], C.data[4]);
  half2 c_z_xy = __halves2half2(C.data[8], C_xy_yx);
  half2 c_xz_yz = __halves2half2(C_xz_zx, C_yz_zy);

  half2 zero = __halves2half2(0.F, 0.F);
  half2 accumulator = __hfma2(c_xy, r2_xy, zero);
  accumulator = __hfma2(c_z_xy, r2_z_xy, accumulator);
  accumulator = __hfma2(c_xz_yz, r2_xz_yz, accumulator);

  half result = __hadd(__low2half(accumulator), __high2half(accumulator));

  // Scale the tensor contraction by 3, then subtract from trace
  result = __hmul(result, __float2half(3.F));
  result = __hsub(trace_C, result);

  return __half2float(result);
}

/**
 * @brief Computes the unscaled Second-Order Taylor contraction for the Winding
 * Number.
 *
 * Performs a full contraction between the compressed second-order multipole
 * coefficient tensor C (rank 3) and the scale-invariant third-order field
 * gradient component.
 *
 * Mathematical formulation:
 * Contraction = C ::: G_hat
 * = 15 * sum_{i,j,k} (C_ijk * r_hat_i * r_hat_j * r_hat_k)
 * - [6 * (V1 · r_hat) + 3 * (V2 · r_hat)]
 *
 * where:
 * - G_hat_ijk = 15 * (r_hat_i * r_hat_j * r_hat_k) - 3 * (r_hat_i * delta_jk +
 * r_hat_j * delta_ik + r_hat_k * delta_ij)
 * - C possesses partial symmetry: C_ijk = C_jik (symmetric in the first two
 * indices).
 * - V1 is the first partial vector-trace of C, where V1_i = sum_j (C_ijj)
 * - V2 is the second partial vector-trace of C, where V2_k = sum_i (C_iik)
 *
 * Because C is not fully symmetric (C_ijk != C_kji), contracting against the
 * isotropic Kronecker delta components yields asymmetric trace multiplicities.
 * This splits the trace reduction into two distinct vector projections weighted
 * 6:3 instead of a single unified trace vector.
 *
 * Symmetries in the expansion reduce the 27-term tensor contraction to 18
 * unique terms unpacked directly from the layout matrix. The complete physical
 * contribution is reconstructed by the caller via:
 * Contribution = Contraction * [1 / (4 * pi * ||r||^4)]
 *
 * @param C The second-order rank-3 multipole coefficient tensor stored in
 * symmetric compressed format (18 float/bf16 elements).
 * @param r_hat_f16 Normalized unit displacement vector pointing from query to
 * the cluster center (||r_hat|| = 1).
 * @return Unscaled contraction scalar evaluated in float32 precision.
 */
__device__ __forceinline__ auto
computeSecondOrderContribution(const Tensor3_bf16_compressed &C,
                               const Vec3_f16 &r_hat_f16) -> float {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  const __nv_bfloat16 two = __float2bfloat16(2.0f);

  // Vector Trace V1
  __nv_bfloat16 v1x = __hadd(__hadd(C.data[0], C.data[4]), C.data[8]);
  __nv_bfloat16 v1y = __hadd(__hadd(C.data[3], C.data[10]), C.data[14]);
  __nv_bfloat16 v1z = __hadd(__hadd(C.data[6], C.data[13]), C.data[17]);

  // Vector Trace V2
  __nv_bfloat16 v2x = __hadd(__hadd(C.data[0], C.data[9]), C.data[15]);
  __nv_bfloat16 v2y = __hadd(__hadd(C.data[1], C.data[10]), C.data[16]);
  __nv_bfloat16 v2z = __hadd(__hadd(C.data[2], C.data[11]), C.data[17]);

  Vec3_bf16 r_hat = Vec3_bf16::from_f16(r_hat_f16);

  // V1 dot r_hat
  __nv_bfloat16 v1_dot_r = __hadd(
      __hadd(__hmul(v1x, r_hat.x), __hmul(v1y, r_hat.y)), __hmul(v1z, r_hat.z));
  // V2 dot r_hat
  __nv_bfloat16 v2_dot_r = __hadd(
      __hadd(__hmul(v2x, r_hat.x), __hmul(v2y, r_hat.y)), __hmul(v2z, r_hat.z));

  // Pre-calculate baseline quadratic components
  __nv_bfloat16 r_xx = __hmul(r_hat.x, r_hat.x);
  __nv_bfloat16 r_yy = __hmul(r_hat.y, r_hat.y);
  __nv_bfloat16 r_zz = __hmul(r_hat.z, r_hat.z);
  __nv_bfloat16 r_xy = __hmul(r_hat.x, r_hat.y);
  __nv_bfloat16 r_xz = __hmul(r_hat.x, r_hat.z);
  __nv_bfloat16 r_yz = __hmul(r_hat.y, r_hat.z);

  __nv_bfloat162 accumulator =
      __hmul2(__halves2bfloat162(C.data[0], C.data[1]),
              __halves2bfloat162(__hmul(r_xx, r_hat.x), __hmul(r_xx, r_hat.y)));
  accumulator = __hfma2(__halves2bfloat162(C.data[2], C.data[3]),
                        __halves2bfloat162(__hmul(r_xx, r_hat.z),
                                           __hmul(__hmul(r_xy, r_hat.x), two)),
                        accumulator);
  accumulator = __hfma2(__halves2bfloat162(C.data[4], C.data[5]),
                        __halves2bfloat162(__hmul(__hmul(r_xy, r_hat.y), two),
                                           __hmul(__hmul(r_xz, r_hat.y), two)),
                        accumulator);
  accumulator = __hfma2(__halves2bfloat162(C.data[6], C.data[7]),
                        __halves2bfloat162(__hmul(__hmul(r_xz, r_hat.x), two),
                                           __hmul(__hmul(r_yz, r_hat.x), two)),
                        accumulator);
  accumulator = __hfma2(__halves2bfloat162(C.data[8], C.data[9]),
                        __halves2bfloat162(__hmul(__hmul(r_xz, r_hat.z), two),
                                           __hmul(r_yy, r_hat.x)),
                        accumulator);
  accumulator =
      __hfma2(__halves2bfloat162(C.data[10], C.data[11]),
              __halves2bfloat162(__hmul(r_yy, r_hat.y), __hmul(r_yy, r_hat.z)),
              accumulator);
  accumulator = __hfma2(__halves2bfloat162(C.data[12], C.data[13]),
                        __halves2bfloat162(__hmul(__hmul(r_yz, r_hat.x), two),
                                           __hmul(__hmul(r_yz, r_hat.y), two)),
                        accumulator);
  accumulator = __hfma2(__halves2bfloat162(C.data[14], C.data[15]),
                        __halves2bfloat162(__hmul(__hmul(r_yz, r_hat.z), two),
                                           __hmul(r_zz, r_hat.x)),
                        accumulator);
  accumulator =
      __hfma2(__halves2bfloat162(C.data[16], C.data[17]),
              __halves2bfloat162(__hmul(r_zz, r_hat.y), __hmul(r_zz, r_hat.z)),
              accumulator);

  __nv_bfloat16 result =
      __hadd(__low2bfloat16(accumulator), __high2bfloat16(accumulator));

  // Apply final balanced weights: (15 * tensor_part) - (6 * V1 * r_hat + 3 * V2
  // * r_hat)
  result = __hmul(result, __float2bfloat16(15.0F));
  __nv_bfloat16 v_part = __hfma(v1_dot_r, __float2bfloat16(6.0F),
                                __hmul(v2_dot_r, __float2bfloat16(3.0F)));
  result = __hsub(result, v_part);

  return __bfloat162float(result);

#else
  // Fallback to float32 on old gpus without bfloat16 support
  Vec3_bf16 r_hat = Vec3_bf16::from_f16(r_hat_f16);
  const float rx = __bfloat162float(r_hat.x);
  const float ry = __bfloat162float(r_hat.y);
  const float rz = __bfloat162float(r_hat.z);

  // Unpack compressed tensor elements into float registers
  float c[18];
#pragma unroll
  for (int i = 0; i < 18; ++i) {
    c[i] = __bfloat162float(C.data[i]);
  }

  // Vector Trace V1 calculation in FP32
  const float v1x = c[0] + c[4] + c[8];
  const float v1y = c[3] + c[10] + c[14];
  const float v1z = c[6] + c[13] + c[17];

  // Vector Trace V2 calculation in FP32
  const float v2x = c[0] + c[9] + c[15];
  const float v2y = c[1] + c[10] + c[16];
  const float v2z = c[2] + c[11] + c[17];

  const float v1_dot_r = (v1x * rx) + (v1y * ry) + (v1z * rz);
  const float v2_dot_r = (v2x * rx) + (v2y * ry) + (v2z * rz);

  // Compute tensor contribution mirroring the packed __nv_bfloat162 pairings
  // exactly
  float tensor_part = 0.0f;
  tensor_part += c[0] * (rx * rx * rx);
  tensor_part += c[1] * (rx * rx * ry);
  tensor_part += c[2] * (rx * rx * rz);
  tensor_part += c[3] * (rx * ry * rx * 2.0f);
  tensor_part += c[4] * (rx * ry * ry * 2.0f);
  tensor_part += c[5] * (rx * rz * ry * 2.0f);
  tensor_part += c[6] * (rx * rz * rx * 2.0f);
  tensor_part += c[7] * (ry * rz * rx * 2.0f);
  tensor_part += c[8] * (rx * rz * rz * 2.0f);
  tensor_part += c[9] * (ry * ry * rx);
  tensor_part += c[10] * (ry * ry * ry);
  tensor_part += c[11] * (ry * ry * rz);
  tensor_part += c[12] * (ry * rz * rx * 2.0f);
  tensor_part += c[13] * (ry * rz * ry * 2.0f);
  tensor_part += c[14] * (ry * rz * rz * 2.0f);
  tensor_part += c[15] * (rz * rz * rx);
  tensor_part += c[16] * (rz * rz * ry);
  tensor_part += c[17] * (rz * rz * rz);

  // Balanced weights linear blending
  return (15.0f * tensor_part) - (6.0f * v1_dot_r + 3.0f * v2_dot_r);
#endif
}

__device__ __forceinline__ auto compute_node_approximation(
    const Vec3 &query, const Vec3 &center_of_mass,
    const Vec3_f16 &zero_order_coeff, const Mat3x3_f16 &first_order_coeff,
    const Tensor3_bf16_compressed &second_order_coeff) -> float {
  Vec3 r = center_of_mass - query;
  float inv_norm_r = r.inv_length();

  // Work a unit vectorfor float16 math
  Vec3_f16 r_hat_f16 = Vec3_f16::from_float(r * inv_norm_r);

  float inv_norm_r2 = inv_norm_r * inv_norm_r;
  float inv_norm_r3 = inv_norm_r2 * inv_norm_r;
  float inv_norm_r4 = inv_norm_r3 * inv_norm_r;

  float inv_4pi = 0.07957747154F;
  float factor_zero = inv_4pi * inv_norm_r2;
  float factor_first = inv_4pi * inv_norm_r3;
  float factor_second = inv_4pi * inv_norm_r4;

  float result = 0.F;

  // Compute contractions using the unit vector, then scale via float32 at the
  // end to prevent overflows of the float16
  result +=
      computeZeroOrderContribution(zero_order_coeff, r_hat_f16) * factor_zero;
  result += computeFirstOrderContribution(first_order_coeff, r_hat_f16) *
            factor_first;
  result += computeSecondOrderContribution(second_order_coeff, r_hat_f16) *
            factor_second;

  return result;
}

// General quadric form in bf16, using pre-packed monomials.
//   q(M) = m0 * P.a + m1 * P.b + m2 * Q.a + m3 * Q.b + m4 * S.a + m5 * S.b
// where P = (r_a^2, r_a r_b), Q = (r_a r_c, r_b^2), S = (r_b r_c, r_c^2)
// are the shared monomial packs.
__device__ __forceinline__ auto
quadric_bf16(__nv_bfloat16 m0, __nv_bfloat16 m1, __nv_bfloat16 m2,
             __nv_bfloat16 m3, __nv_bfloat16 m4, __nv_bfloat16 m5,
             __nv_bfloat162 P, __nv_bfloat162 Q, __nv_bfloat162 S)
    -> __nv_bfloat16 {
  __nv_bfloat162 acc = __hmul2(__halves2bfloat162(m0, m1), P);
  acc = __hfma2(__halves2bfloat162(m2, m3), Q, acc);
  acc = __hfma2(__halves2bfloat162(m4, m5), S, acc);
  return __hadd(__low2bfloat16(acc), __high2bfloat16(acc));
}

// =============================================================================
// Second-order (M2) contribution to the query gradient of one node.
//
// Field:   f2(q) = s / (4π R^4),  s = 15 Q_rrr - 6 (V1 . r_hat) - 3 (V2 .
// r_hat) Grad:    ∇_q f2 = [ -u + (u . r_hat + 4 s) r_hat ] / (4π R^5)
//
// where
//   u_l = 15 (2 A_l + C_l) - 6 V1_l - 3 V2_l
//   A_l = C_ljk r_hat_j r_hat_k        (free first index)
//   C_l = C_ijl r_hat_i r_hat_j        (free third index)
//
// M is symmetric in its first two indices, so A_l and C_l are two independent
// quadratic forms of the same tensor. All arithmetic is done in bf16 with the
// same packing as computeSecondOrderContribution; the final 1/(4π R^5) is
// applied in fp32 by the caller.
// =============================================================================
__device__ __forceinline__ auto
computeSecondOrderQueryGrad(const Tensor3_bf16_compressed &C,
                            const Vec3_f16 &r_hat_f16) -> Vec3 {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  using bf16 = __nv_bfloat16;
  using bf162 = __nv_bfloat162;

  const bf16 two = __float2bfloat16(2.0F);
  const bf16 three = __float2bfloat16(3.0F);
  const bf16 four = __float2bfloat16(4.0F);
  const bf16 six = __float2bfloat16(6.0F);
  const bf16 fifteen = __float2bfloat16(15.0F);

  const Vec3_bf16 r = Vec3_bf16::from_f16(r_hat_f16);
  const bf16 rx = r.x;
  const bf16 ry = r.y;
  const bf16 rz = r.z;

  // --- Vector traces (identical to computeSecondOrderContribution) ---
  const bf16 v1x = __hadd(__hadd(C.data[0], C.data[4]), C.data[8]);
  const bf16 v1y = __hadd(__hadd(C.data[3], C.data[10]), C.data[14]);
  const bf16 v1z = __hadd(__hadd(C.data[6], C.data[13]), C.data[17]);
  const bf16 v2x = __hadd(__hadd(C.data[0], C.data[9]), C.data[15]);
  const bf16 v2y = __hadd(__hadd(C.data[1], C.data[10]), C.data[16]);
  const bf16 v2z = __hadd(__hadd(C.data[2], C.data[11]), C.data[17]);

  const bf16 v1_dot_r =
      __hadd(__hadd(__hmul(v1x, rx), __hmul(v1y, ry)), __hmul(v1z, rz));
  const bf16 v2_dot_r =
      __hadd(__hadd(__hmul(v2x, rx), __hmul(v2y, ry)), __hmul(v2z, rz));

  // --- Quadratic monomials ---
  const bf16 rxx = __hmul(rx, rx);
  const bf16 ryy = __hmul(ry, ry);
  const bf16 rzz = __hmul(rz, rz);
  const bf16 rxy = __hmul(rx, ry);
  const bf16 rxz = __hmul(rx, rz);
  const bf16 ryz = __hmul(ry, rz);

  // --- Monomial packs (single for A, doubled for C) ---
  const bf162 mA_P = __halves2bfloat162(rxx, rxy);
  const bf162 mA_Q = __halves2bfloat162(rxz, ryy);
  const bf162 mA_S = __halves2bfloat162(ryz, rzz);
  const bf162 mC_P = __halves2bfloat162(rxx, __hadd(rxy, rxy));
  const bf162 mC_Q = __halves2bfloat162(__hadd(rxz, rxz), ryy);
  const bf162 mC_S = __halves2bfloat162(__hadd(ryz, ryz), rzz);

  // --- A_0, A_1, A_2 ---
  const bf16 A0 = quadric_bf16(
      C.data[0], __hadd(C.data[1], C.data[3]), __hadd(C.data[2], C.data[6]),
      C.data[4], __hadd(C.data[5], C.data[7]), C.data[8], mA_P, mA_Q, mA_S);

  const bf16 A1 = quadric_bf16(
      C.data[3], __hadd(C.data[4], C.data[9]), __hadd(C.data[5], C.data[12]),
      C.data[10], __hadd(C.data[11], C.data[13]), C.data[14], mA_P, mA_Q, mA_S);

  const bf16 A2 = quadric_bf16(
      C.data[6], __hadd(C.data[7], C.data[12]), __hadd(C.data[8], C.data[15]),
      C.data[13], __hadd(C.data[14], C.data[16]), C.data[17], mA_P, mA_Q, mA_S);

  // --- C_0, C_1, C_2 ---
  const bf16 C0 = quadric_bf16(C.data[0], C.data[3], C.data[6], C.data[9],
                               C.data[12], C.data[15], mC_P, mC_Q, mC_S);
  const bf16 C1 = quadric_bf16(C.data[1], C.data[4], C.data[7], C.data[10],
                               C.data[13], C.data[16], mC_P, mC_Q, mC_S);
  const bf16 C2 = quadric_bf16(C.data[2], C.data[5], C.data[8], C.data[11],
                               C.data[14], C.data[17], mC_P, mC_Q, mC_S);

  // --- u = 15 (2 A + C) - 6 V1 - 3 V2 ---
  const bf16 ux = __hsub(
      __hsub(__hmul(fifteen, __hadd(__hmul(two, A0), C0)), __hmul(six, v1x)),
      __hmul(three, v2x));
  const bf16 uy = __hsub(
      __hsub(__hmul(fifteen, __hadd(__hmul(two, A1), C1)), __hmul(six, v1y)),
      __hmul(three, v2y));
  const bf16 uz = __hsub(
      __hsub(__hmul(fifteen, __hadd(__hmul(two, A2), C2)), __hmul(six, v1z)),
      __hmul(three, v2z));

  // --- Scalar numerator s: identical packing to the field version ---
  const bf16 rxx_x = __hmul(rxx, rx);
  const bf16 rxx_y = __hmul(rxx, ry);
  const bf16 rxx_z = __hmul(rxx, rz);
  const bf16 rxy2_x = __hmul(__hmul(rxy, rx), two);
  const bf16 rxy2_y = __hmul(__hmul(rxy, ry), two);
  const bf16 rxz2_y = __hmul(__hmul(rxz, ry), two);
  const bf16 rxz2_x = __hmul(__hmul(rxz, rx), two);
  const bf16 ryz2_x = __hmul(__hmul(ryz, rx), two);
  const bf16 rxz2_z = __hmul(__hmul(rxz, rz), two);
  const bf16 ryy_x = __hmul(ryy, rx);
  const bf16 ryy_y = __hmul(ryy, ry);
  const bf16 ryy_z = __hmul(ryy, rz);
  const bf16 ryz2_y = __hmul(__hmul(ryz, ry), two);
  const bf16 ryz2_z = __hmul(__hmul(ryz, rz), two);
  const bf16 rzz_x = __hmul(rzz, rx);
  const bf16 rzz_y = __hmul(rzz, ry);
  const bf16 rzz_z = __hmul(rzz, rz);

  bf162 cubic_acc = __hmul2(__halves2bfloat162(C.data[0], C.data[1]),
                            __halves2bfloat162(rxx_x, rxx_y));
  cubic_acc = __hfma2(__halves2bfloat162(C.data[2], C.data[3]),
                      __halves2bfloat162(rxx_z, rxy2_x), cubic_acc);
  cubic_acc = __hfma2(__halves2bfloat162(C.data[4], C.data[5]),
                      __halves2bfloat162(rxy2_y, rxz2_y), cubic_acc);
  cubic_acc = __hfma2(__halves2bfloat162(C.data[6], C.data[7]),
                      __halves2bfloat162(rxz2_x, ryz2_x), cubic_acc);
  cubic_acc = __hfma2(__halves2bfloat162(C.data[8], C.data[9]),
                      __halves2bfloat162(rxz2_z, ryy_x), cubic_acc);
  cubic_acc = __hfma2(__halves2bfloat162(C.data[10], C.data[11]),
                      __halves2bfloat162(ryy_y, ryy_z), cubic_acc);
  cubic_acc = __hfma2(__halves2bfloat162(C.data[12], C.data[13]),
                      __halves2bfloat162(ryz2_x, ryz2_y), cubic_acc);
  cubic_acc = __hfma2(__halves2bfloat162(C.data[14], C.data[15]),
                      __halves2bfloat162(ryz2_z, rzz_x), cubic_acc);
  cubic_acc = __hfma2(__halves2bfloat162(C.data[16], C.data[17]),
                      __halves2bfloat162(rzz_y, rzz_z), cubic_acc);
  const bf16 cubic =
      __hadd(__low2bfloat16(cubic_acc), __high2bfloat16(cubic_acc));

  const bf16 s = __hsub(__hmul(fifteen, cubic),
                        __hfma(six, v1_dot_r, __hmul(three, v2_dot_r)));

  // --- Final combination: -u + (u.r_hat + 4 s) r_hat ---
  const bf16 u_dot_r =
      __hadd(__hadd(__hmul(ux, rx), __hmul(uy, ry)), __hmul(uz, rz));
  const bf16 coef = __hfma(four, s, u_dot_r);

  return Vec3{
      .x = __bfloat162float(__hfma(coef, rx, __hneg(ux))),
      .y = __bfloat162float(__hfma(coef, ry, __hneg(uy))),
      .z = __bfloat162float(__hfma(coef, rz, __hneg(uz))),
  };

#else
  // -------------------- fp32 fallback --------------------
  const Vec3 r_hat = Vec3::from_f16(r_hat_f16);
  const float rx = r_hat.x, ry = r_hat.y, rz = r_hat.z;
  const float rx2 = rx * rx, ry2 = ry * ry, rz2 = rz * rz;
  const float rxy = rx * ry, rxz = rx * rz, ryz = ry * rz;

  float c[18];
#pragma unroll
  for (int i = 0; i < 18; ++i) {
    c[i] = __bfloat162float(C.data[i]);
  }

  const float v1x = c[0] + c[4] + c[8];
  const float v1y = c[3] + c[10] + c[14];
  const float v1z = c[6] + c[13] + c[17];
  const float v2x = c[0] + c[9] + c[15];
  const float v2y = c[1] + c[10] + c[16];
  const float v2z = c[2] + c[11] + c[17];

  const float v1_dot_r = v1x * rx + v1y * ry + v1z * rz;
  const float v2_dot_r = v2x * rx + v2y * ry + v2z * rz;

  // A_l = C_ljk r_j r_k
  const float A0 = c[0] * rx2 + (c[1] + c[3]) * rxy + (c[2] + c[6]) * rxz +
                   c[4] * ry2 + (c[5] + c[7]) * ryz + c[8] * rz2;
  const float A1 = c[3] * rx2 + (c[4] + c[9]) * rxy + (c[5] + c[12]) * rxz +
                   c[10] * ry2 + (c[11] + c[13]) * ryz + c[14] * rz2;
  const float A2 = c[6] * rx2 + (c[7] + c[12]) * rxy + (c[8] + c[15]) * rxz +
                   c[13] * ry2 + (c[14] + c[16]) * ryz + c[17] * rz2;

  // C_l = C_ijl r_i r_j
  const float C0 = c[0] * rx2 + 2.f * c[3] * rxy + 2.f * c[6] * rxz +
                   c[9] * ry2 + 2.f * c[12] * ryz + c[15] * rz2;
  const float C1 = c[1] * rx2 + 2.f * c[4] * rxy + 2.f * c[7] * rxz +
                   c[10] * ry2 + 2.f * c[13] * ryz + c[16] * rz2;
  const float C2 = c[2] * rx2 + 2.f * c[5] * rxy + 2.f * c[8] * rxz +
                   c[11] * ry2 + 2.f * c[14] * ryz + c[17] * rz2;

  const float ux = 15.f * (2.f * A0 + C0) - 6.f * v1x - 3.f * v2x;
  const float uy = 15.f * (2.f * A1 + C1) - 6.f * v1y - 3.f * v2y;
  const float uz = 15.f * (2.f * A2 + C2) - 6.f * v1z - 3.f * v2z;

  const float cubic =
      c[0] * rx2 * rx + c[1] * rx2 * ry + c[2] * rx2 * rz +
      c[3] * 2.f * rxy * rx + c[4] * 2.f * rxy * ry + c[5] * 2.f * rxz * ry +
      c[6] * 2.f * rxz * rx + c[7] * 2.f * ryz * rx + c[8] * 2.f * rxz * rz +
      c[9] * ry2 * rx + c[10] * ry2 * ry + c[11] * ry2 * rz +
      c[12] * 2.f * ryz * rx + c[13] * 2.f * ryz * ry + c[14] * 2.f * ryz * rz +
      c[15] * rz2 * rx + c[16] * rz2 * ry + c[17] * rz2 * rz;
  const float s = 15.f * cubic - 6.f * v1_dot_r - 3.f * v2_dot_r;

  const float u_dot_r = ux * rx + uy * ry + uz * rz;
  const float coef = u_dot_r + 4.f * s;

  return Vec3{
      .x = -ux + coef * rx,
      .y = -uy + coef * ry,
      .z = -uz + coef * rz,
  };
#endif
}

// Far-field query gradient for one node.
__device__ __forceinline__ auto compute_node_query_grad_approximation(
    const Vec3 &query, const Vec3 &center_of_mass, const Vec3_f16 &M0_f16,
    const Mat3x3_f16 &M1_f16, const Tensor3_bf16_compressed &M2_bf16) -> Vec3 {

  const Vec3 r = center_of_mass - query;
  const float inv_R = r.inv_length();
  const float inv_R3 = inv_R * inv_R * inv_R;
  const float inv_R4 = inv_R3 * inv_R;
  const float inv_R5 = inv_R3 * inv_R * inv_R;

  const Vec3_f16 r_hat = Vec3_f16::from_float(r * inv_R);

  constexpr float INV_4PI = 0.07957747154594767F;
  const half three_h = __float2half(3.F);
  const half five_h = __float2half(5.F);
  const half two_h = __float2half(2.F);
  const half half_h = __float2half(0.5F);

  // Zero order --------------------------------------------------------------
  //   fma(r_hat, s, a) returns r_hat * s + a, so this is
  //   3 (M0 · r_hat) r_hat - M0.
  const Vec3_f16 grad0_h =
      Vec3_f16::fma(r_hat, __hmul(three_h, M0_f16.dot(r_hat)), -M0_f16);

  const float scale0 = INV_4PI * inv_R3;
  Vec3 grad{
      .x = __half2float(grad0_h.x) * scale0,
      .y = __half2float(grad0_h.y) * scale0,
      .z = __half2float(grad0_h.z) * scale0,
  };

  // First order -------------------------------------------------------------
  // Symmetrize M1 in half2. M1_sym[i][j] = (M1[i][j] + M1[j][i]) / 2.
  Mat3x3_f16 M1_sym;
  M1_sym.data[0] = M1_f16.data[0];
  M1_sym.data[4] = M1_f16.data[4];
  M1_sym.data[8] = M1_f16.data[8];
  M1_sym.data[1] = __hmul(__hadd(M1_f16.data[1], M1_f16.data[3]), half_h);
  M1_sym.data[3] = M1_sym.data[1];
  M1_sym.data[2] = __hmul(__hadd(M1_f16.data[2], M1_f16.data[6]), half_h);
  M1_sym.data[6] = M1_sym.data[2];
  M1_sym.data[5] = __hmul(__hadd(M1_f16.data[5], M1_f16.data[7]), half_h);
  M1_sym.data[7] = M1_sym.data[5];

  const Vec3_f16 v = M1_sym * r_hat; // M1_sym · r_hat
  const half q_rr = r_hat.dot(v);    // r_hat^T M1_sym r_hat
  const half trace =
      __hadd(__hadd(M1_f16.data[0], M1_f16.data[4]), M1_f16.data[8]);
  const half tr_5q = __hsub(trace, __hmul(five_h, q_rr));

  // (tr(M1) - 5 q_rr) r_hat + 2 M1_sym r_hat
  const Vec3_f16 grad1_h = Vec3_f16::fma(r_hat, tr_5q, v * two_h);

  const float scale1 = 3.F * INV_4PI * inv_R4;
  grad += Vec3::from_f16(grad1_h) * scale1;

  // Second order -------------------------------------------------------------
  const Vec3 grad2 = computeSecondOrderQueryGrad(M2_bf16, r_hat);
  const float scale2 = INV_4PI * inv_R5;
  grad += grad2 * scale2;

  return grad;
}

/**
 * @brief Computes the complete multi-order Taylor approximation of loss
 * gradients for a PointNormal primitive w.r.t. position (p) and normal dipole
 * (n).
 *
 * @param geometry Source primitive containing position p and normal dipole n.
 * @param center_of_mass Cluster centroid of the query group (q_tilde).
 * @param G_0 Scalar zero-order query moment (G0).
 * @param G_1 Vector first-order query moment (G1).
 * @param G_2 Matrix second-order query moment (G2).
 * @return PointNormalGrad Struct containing grad_p and grad_n evaluated in
 * FP32.
 */
__device__ __forceinline__ auto compute_node_gradient_approximation(
    const PointNormal &geometry, const Vec3 &center_of_mass, const float G_0,
    const Vec3 &G_1, const SymMat3x3 &G_2) -> PointNormal {

  const Vec3 r_vec = center_of_mass - geometry.p;
  const float R2 = r_vec.length2();

  // Singular condition check (R < 1e-10 -> R^2 < 1e-20)
  if (R2 < 1e-20F) {
    return PointNormal{Vec3::zero(), Vec3::zero()};
  }

  // Fast GPU reciprocal square root & power chain
  const float inv_R = rsqrtf(R2);
  const float inv_R2 = inv_R * inv_R;
  const float inv_R3 = inv_R2 * inv_R;
  const float inv_R4 = inv_R2 * inv_R2;
  const float inv_R5 = inv_R3 * inv_R2;

  const Vec3 r_hat = r_vec * inv_R;
  const Vec3 m = geometry.n;

  // Primary projections
  const float m_r = r_hat.dot(m);
  const float r_dot_G1 = r_hat.dot(G_1);
  const float G1_dot_m = G_1.dot(m);

  // Order 2 tensor-vector products and trace contractions
  const Vec3 G2_r = G_2 * r_hat;
  const Vec3 G2_m = G_2 * m;
  const float tr_G2 = G_2.trace();
  const float q_rr = r_hat.dot(G2_r);
  const float q_rm = r_hat.dot(G2_m);

  // Order 2 contraction scalar constants
  const float c_nr = 7.5F * q_rr - 1.5F * tr_G2;
  const float c_rp = 15.F * q_rm + (7.5F * tr_G2 - 52.5F * q_rr) * m_r;

  // -------------------------------------------------------------------------
  // Collect scalar coefficients for grad_m:
  // grad_m = r_hat * s_m_r + G_1 * s_m_g1 + G2_r * s_m_g2r
  // -------------------------------------------------------------------------
  const float s_m_r = G_0 * inv_R2 - 3.F * r_dot_G1 * inv_R3 + c_nr * inv_R4;
  const float s_m_g1 = inv_R3;
  const float s_m_g2r = -3.F * inv_R4;

  const Vec3 grad_m = r_hat * s_m_r + G_1 * s_m_g1 + G2_r * s_m_g2r;

  // -------------------------------------------------------------------------
  // Collect scalar coefficients for grad_p:
  // grad_p = r_hat * s_p_r + m * s_p_m + G_1 * s_p_g1 + G2_r * s_p_g2r + G2_m *
  // s_p_g2m
  // -------------------------------------------------------------------------
  const float s_p_r = 3.F * m_r * G_0 * inv_R3 +
                      (3.F * G1_dot_m - 15.F * r_dot_G1 * m_r) * inv_R4 -
                      c_rp * inv_R5;
  const float s_p_m = -G_0 * inv_R3 + 3.F * r_dot_G1 * inv_R4 - c_nr * inv_R5;
  const float s_p_g1 = 3.F * m_r * inv_R4;
  const float s_p_g2r = -15.F * m_r * inv_R5;
  const float s_p_g2m = 3.F * inv_R5;

  const Vec3 grad_p = r_hat * s_p_r + m * s_p_m + G_1 * s_p_g1 +
                      G2_r * s_p_g2r + G2_m * s_p_g2m;

  constexpr float INV_4PI = -0.07957747154594767F; // -1.0 / (4.0 * pi)

  return PointNormal{.p = grad_p * INV_4PI, .n = grad_m * INV_4PI};
}

/**
 * @brief Scale-invariant 0th + 1st + 2nd order Taylor approximation of loss
 * gradients for a Triangle primitive with respect to its vertices.
 */
__device__ __forceinline__ auto compute_node_gradient_approximation(
    const Triangle &geometry, const Vec3 &center_of_mass, const float G_0,
    const Vec3 &G_1, const SymMat3x3 &G_2) -> Triangle {

  // -------------------------------------------------------------------------
  // 1. Inputs & Non-Dimensionalization
  // -------------------------------------------------------------------------
  Vec3 r[3];
  float d_raw[3];

  r[0] = geometry.v0 - center_of_mass;
  r[1] = geometry.v1 - center_of_mass;
  r[2] = geometry.v2 - center_of_mass;
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    d_raw[i] = r[i].length();
  }

  float L_ref = fmaxf(fmaxf(d_raw[0], d_raw[1]), d_raw[2]);
  if (L_ref < 2e-12F) {
    L_ref = 1e-12F;
  }

  const float inv_L = 1.F / L_ref;
  const float inv_L2 = inv_L * inv_L;

  Vec3 r_bar[3];
  float d[3];
  Vec3 r_hat[3];

#pragma unroll
  for (int i = 0; i < 3; ++i) {
    r_bar[i] = r[i] * inv_L;
    d[i] = d_raw[i] * inv_L;
    r_hat[i] = r_bar[i] * (1.F / d[i]);
  }

  const float G0_bar = G_0;
  const Vec3 G1_bar = G_1 * inv_L;

  // SymMat3x3 indexing (0:xx, 1:xy, 2:xz, 3:yy, 4:yz, 5:zz)
  const float g00 = G_2.data[0] * inv_L2;
  const float g01 = G_2.data[1] * inv_L2;
  const float g20 = G_2.data[2] * inv_L2;
  const float g11 = G_2.data[3] * inv_L2;
  const float g12 = G_2.data[4] * inv_L2;
  const float g22 = G_2.data[5] * inv_L2;

  // -------------------------------------------------------------------------
  // 2. Base 0th-Order Field Evaluation (T0)
  // -------------------------------------------------------------------------
  Vec3 U[3];
  float alpha[3];
  Vec3 V[3];

#pragma unroll
  for (int p = 0; p < 3; ++p) {
    const int j = (p + 1) % 3;
    const int m = (p + 2) % 3;

    U[p] = Vec3::cross(r_bar[j], r_bar[m]);
    const float dot_jm = r_bar[j].dot(r_bar[m]);
    alpha[p] = d[j] * d[m] + dot_jm;
    V[p] = r_hat[p] * alpha[p] + r_bar[j] * d[m] + r_bar[m] * d[j];
  }

  const float N = r_bar[0].dot(U[0]);

  const float dot12 = r_bar[1].dot(r_bar[2]);
  const float dot20 = r_bar[2].dot(r_bar[0]);
  const float dot01 = r_bar[0].dot(r_bar[1]);
  const float D =
      d[0] * d[1] * d[2] + dot12 * d[0] + dot20 * d[1] + dot01 * d[2];

  const float S = N * N + D * D;
  const float inv_S = 1.F / S;
  const float inv_S2 = inv_S * inv_S;
  const float inv_S3 = inv_S2 * inv_S;

  Vec3 W[3];
  Vec3 T0[3];

#pragma unroll
  for (int p = 0; p < 3; ++p) {
    W[p] = U[p] * D - V[p] * N;
    T0[p] = W[p] * (G0_bar * (2.F * inv_S));
  }

  const Vec3 U_sum = U[0] + U[1] + U[2];
  const Vec3 V_sum = V[0] + V[1] + V[2];

  // -------------------------------------------------------------------------
  // 3. Exact 1st Directional Derivative Operator (T1)
  // -------------------------------------------------------------------------
  const Vec3 e = -G1_bar;

  float d_de[3];
  Vec3 r_hat_de[3];

#pragma unroll
  for (int p = 0; p < 3; ++p) {
    d_de[p] = r_hat[p].dot(e);
    r_hat_de[p] = (e - r_hat[p] * d_de[p]) * (1.F / d[p]);
  }

  Vec3 U_de[3];
  float alpha_de[3];
  Vec3 V_de[3];

#pragma unroll
  for (int p = 0; p < 3; ++p) {
    const int j = (p + 1) % 3;
    const int m = (p + 2) % 3;

    U_de[p] = Vec3::cross(r_bar[j] - r_bar[m], e);
    alpha_de[p] =
        d[m] * d_de[j] + d[j] * d_de[m] + (r_bar[j] + r_bar[m]).dot(e);
    V_de[p] = r_hat[p] * alpha_de[p] + r_hat_de[p] * alpha[p] +
              r_bar[j] * d_de[m] + r_bar[m] * d_de[j] + e * (d[j] + d[m]);
  }

  const float N_de = U_sum.dot(e);
  const float D_de = V_sum.dot(e);
  const float S_de = 2.F * N * N_de + 2.F * D * D_de;

  Vec3 T1[3];

#pragma unroll
  for (int p = 0; p < 3; ++p) {
    const Vec3 W_de_p =
        (U[p] * D_de + U_de[p] * D) - (V[p] * N_de + V_de[p] * N);
    T1[p] = W_de_p * (2.F * inv_S) - W[p] * (2.F * S_de * inv_S2);
  }

  // -------------------------------------------------------------------------
  // 4. Exact 2nd Directional Derivative Operator (T2 via Polarization)
  // -------------------------------------------------------------------------
  const float u_weights[6] = {0.5F * (g00 - g01 - g20),
                              0.5F * (g11 - g01 - g12),
                              0.5F * (g22 - g12 - g20),
                              0.5F * g01,
                              0.5F * g12,
                              0.5F * g20};

  const Vec3 u_dirs[6] = {// in local memory
                          {1.F, 0.F, 0.F}, {0.F, 1.F, 0.F}, {0.F, 0.F, 1.F},
                          {1.F, 1.F, 0.F}, {0.F, 1.F, 1.F}, {1.F, 0.F, 1.F}};

  Vec3 T2[3] = {Vec3::zero(), Vec3::zero(), Vec3::zero()};

  // Keep this loop to reuse working registers across directions
  for (int k = 0; k < 6; ++k) {
    const float weight = u_weights[k]; // TODO local memory
    if (weight == 0.F)
      continue;

    const Vec3 u = u_dirs[k]; // TODO leads to local memory access - use switch
                              // case or something like that
    const float u_norm2 = u.dot(u);

    float d_du[3];
    float d2_du2[3];
    Vec3 r_hat_du[3];
    Vec3 r_hat2_du2[3];

#pragma unroll
    for (int p = 0; p < 3; ++p) {
      d_du[p] = r_hat[p].dot(u);
      d2_du2[p] = (u_norm2 - d_du[p] * d_du[p]) / d[p];

      r_hat_du[p] = (u - r_hat[p] * d_du[p]) / d[p];
      r_hat2_du2[p] =
          (r_hat_du[p] * (-2.F * d_du[p]) - r_hat[p] * d2_du2[p]) / d[p];
    }

    Vec3 U_du[3];
    float alpha_du[3];
    float alpha2_du2[3];
    Vec3 V_du[3];
    Vec3 V2_du2[3];

#pragma unroll
    for (int p = 0; p < 3; ++p) {
      const int j = (p + 1) % 3;
      const int m = (p + 2) % 3;

      U_du[p] = Vec3::cross(r_bar[j] - r_bar[m], u);
      alpha_du[p] =
          d[m] * d_du[j] + d[j] * d_du[m] + (r_bar[j] + r_bar[m]).dot(u);
      alpha2_du2[p] = d2_du2[j] * d[m] + 2.F * d_du[j] * d_du[m] +
                      d[j] * d2_du2[m] + 2.F * u_norm2;

      V_du[p] = r_hat[p] * alpha_du[p] + r_hat_du[p] * alpha[p] +
                r_bar[j] * d_du[m] + r_bar[m] * d_du[j] + u * (d[j] + d[m]);

      V2_du2[p] = r_hat[p] * alpha2_du2[p] + r_hat_du[p] * (2.F * alpha_du[p]) +
                  r_hat2_du2[p] * alpha[p] + r_bar[j] * d2_du2[m] +
                  r_bar[m] * d2_du2[j] + u * (2.F * d_du[m] + 2.F * d_du[j]);
    }

    const float N_du = U_sum.dot(u);
    const float D_du = V_sum.dot(u);
    const float D2_du2 = V_du[0].dot(u) + V_du[1].dot(u) + V_du[2].dot(u);

    const float S_du = 2.F * N * N_du + 2.F * D * D_du;
    const float S2_du2 =
        2.F * (N_du * N_du) + 2.F * (D_du * D_du) + 2.F * D * D2_du2;

#pragma unroll
    for (int p = 0; p < 3; ++p) {
      const Vec3 W_du_p =
          (U[p] * D_du + U_du[p] * D) - (V[p] * N_du + V_du[p] * N);
      const Vec3 W2_du2_p = (U[p] * D2_du2 + U_du[p] * (2.F * D_du)) -
                            (V_du[p] * (2.F * N_du) + V2_du2[p] * N);

      const Vec3 Q_p =
          W2_du2_p * (2.F * inv_S) - W_du_p * (4.F * S_du * inv_S2) -
          W[p] * (2.F * S2_du2 * inv_S2) + W[p] * (4.F * S_du * S_du * inv_S3);

      T2[p] += Q_p * weight;
    }
  }

  // -------------------------------------------------------------------------
  // 5. Final Assembly
  // -------------------------------------------------------------------------
  constexpr float INV_4PI = 0.07957747154594767F; // 1.0 / (4.0 * pi)
  const float scale = INV_4PI * inv_L;

  Triangle result;
  result.v0 = (T0[0] + T1[0] + T2[0]) * scale;
  result.v1 = (T0[1] + T1[1] + T2[1]) * scale;
  result.v2 = (T0[2] + T1[2] + T2[2]) * scale;

  return result;
}
