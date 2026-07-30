#pragma once
#include "aabb.h"
#include "cuda/std/__cmath/isinf.h"
#include "cuda/std/__cmath/isnan.h"
#include "geometry.h"
#include "mat3x3.h"
#include "tensor3.h"
#include "vec3.h"
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

__device__ __forceinline__ auto should_node_be_approximated(const Vec3 &query,
                                                            const AABB &aabb,
                                                            const float beta_2)
    -> bool {
  float max_distance_to_center = __half2float(aabb.max_distance);
  Vec3 com = aabb.center_of_mass;
  float dist_query_to_com2 = (query - com).length2();
  return dist_query_to_com2 >
         max_distance_to_center * max_distance_to_center * beta_2;
}

__device__ __forceinline__ auto
should_node_be_approximated(const PointNormal &geometry, const AABB &aabb,
                            const float beta_2, const float inv_epsilon)
    -> bool {
  Vec3 com = aabb.center_of_mass;
  float dist_geometry_to_com2 = (geometry.centroid() - com).length2();
  const float min_far_field_dist2 = 4.0F / (inv_epsilon * inv_epsilon);
  if (dist_geometry_to_com2 < min_far_field_dist2) {
    return false;
  }
  float max_distance = __half2float(aabb.max_distance);
  float effective_R = fmaxf(max_distance, 2.0F / inv_epsilon);

  return dist_geometry_to_com2 > (effective_R * effective_R * beta_2);
}

__device__ __forceinline__ auto
should_node_be_approximated(const Triangle &geometry, const AABB &aabb,
                            const float beta_2) -> bool {
  float max_distance_to_center = __half2float(aabb.max_distance);
  Vec3 com = aabb.center_of_mass;
  float dist_geometry_to_com2 =
      (geometry.centroid() - com).length2(); // TODO consider large triangles
  return dist_geometry_to_com2 >
         max_distance_to_center * max_distance_to_center * beta_2;
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

/**
 * @brief Computes the unscaled Zero-Order Taylor contractions for the
 * PointNormal gradients.
 *
 * Mathematical formulation:
 * - grad_n_0 = G_0 * r_hat
 * - grad_p_0 = G_0 * [ 3 * (r_hat · m) * r_hat - m ]
 *
 * @param G_0 Scalar zero-order query moment.
 * @param r_hat Unit displacement vector from source position p to cluster
 * center (||r_hat|| = 1).
 * @param m Area-weighted normal / dipole moment vector of the point.
 * @param out_c0_n Output unscaled contraction for normal gradient (FP16).
 * @param out_c0_p Output unscaled contraction for position gradient (FP16).
 */
__device__ __forceinline__ void
computeZeroOrderGradientContribution(const half &G_0, const Vec3_f16 &r_hat,
                                     const Vec3_f16 &m, Vec3_f16 &out_c0_n,
                                     Vec3_f16 &out_c0_p) {
  // grad_n_0 = G_0 * r_hat
  out_c0_n.x = __hmul(G_0, r_hat.x);
  out_c0_n.y = __hmul(G_0, r_hat.y);
  out_c0_n.z = __hmul(G_0, r_hat.z);

  // Dot product: r_dot_m = r_hat . m
  half2 r_xy = __halves2half2(r_hat.x, r_hat.y);
  half2 m_xy = __halves2half2(m.x, m.y);
  half2 prod_xy = __hmul2(r_xy, m_xy);
  half r_dot_m = __hadd(__hadd(__low2half(prod_xy), __high2half(prod_xy)),
                        __hmul(r_hat.z, m.z));

  half three_r_dot_m = __hmul(r_dot_m, __float2half(3.F));

  // 3 * (r_hat . m) * r_hat - m
  half2 r_xy_scaled =
      __hmul2(r_xy, __halves2half2(three_r_dot_m, three_r_dot_m));
  half2 grad_p0_xy = __hsub2(r_xy_scaled, m_xy);
  half grad_p0_z = __hsub(__hmul(r_hat.z, three_r_dot_m), m.z);

  // Scale by G_0
  out_c0_p.x = __hmul(G_0, __low2half(grad_p0_xy));
  out_c0_p.y = __hmul(G_0, __high2half(grad_p0_xy));
  out_c0_p.z = __hmul(G_0, grad_p0_z);
}

/**
 * @brief Computes the unscaled First-Order Taylor contractions for the
 * PointNormal gradients.
 *
 * Mathematical formulation:
 * - grad_n_1 = G_1 - 3 * (r_hat · G_1) * r_hat
 * - grad_p_1 = 3 * (r_hat · m) * G_1 + 3 * (G_1 · m) * r_hat + 3 * (r_hat ·
 * G_1) * m
 *              - 15 * (r_hat · G_1) * (r_hat · m) * r_hat
 *
 * @param G_1 Vector first-order query moment.
 * @param r_hat Unit displacement vector from source position p to cluster
 * center.
 * @param m Area-weighted normal / dipole moment vector.
 * @param out_c1_n Output unscaled contraction for normal gradient (FP16).
 * @param out_c1_p Output unscaled contraction for position gradient (FP16).
 */
__device__ __forceinline__ void
computeFirstOrderGradientContribution(const Vec3_f16 &G_1,
                                      const Vec3_f16 &r_hat, const Vec3_f16 &m,
                                      Vec3_f16 &out_c1_n, Vec3_f16 &out_c1_p) {
  // Dot Products
  const half r_dot_q1 = r_hat.dot(G_1);
  const half r_dot_m = r_hat.dot(m);
  const half q1_dot_m = G_1.dot(m);

  // Normal Gradient
  const half three_r_dot_q1 = __hmul(r_dot_q1, __float2half(3.0F));
  out_c1_n = G_1 - (r_hat * three_r_dot_q1);

  // Position Gradient: Promote dot products to FP32 registers
  const float f_r_dot_q1 = __half2float(r_dot_q1);
  const float f_r_dot_m = __half2float(r_dot_m);
  const float f_q1_dot_m = __half2float(q1_dot_m);

  const float f_three_r_dot_m = 3.0F * f_r_dot_m;
  const float f_three_r_dot_q1 = 3.0F * f_r_dot_q1;
  const float f_r_coeff = 3.0F * f_q1_dot_m - 15.0F * f_r_dot_q1 * f_r_dot_m;

  // 3 * (r·m)*G_1 + 3 * (r·G_1)*m + coeff * r_hat
  const Vec3 grad_p_f32 =
      Vec3{__half2float(G_1.x), __half2float(G_1.y), __half2float(G_1.z)} *
          f_three_r_dot_m +
      Vec3{__half2float(m.x), __half2float(m.y), __half2float(m.z)} *
          f_three_r_dot_q1 +
      Vec3{__half2float(r_hat.x), __half2float(r_hat.y),
           __half2float(r_hat.z)} *
          f_r_coeff;

  out_c1_p = Vec3_f16::from_float(grad_p_f32);
}

/**
 * @brief Computes the unscaled Second-Order Taylor contractions for the
 * PointNormal gradients.
 *
 * Performs matrix-vector contractions between the rank-2 moment matrix G_2,
 * unit vector r_hat, and normal vector m.
 *
 * @param G_2 Symmetric second-order query moment tensor (3x3 matrix).
 * @param r_hat Unit displacement vector.
 * @param m Area-weighted normal / dipole moment vector.
 * @param out_c2_n Output unscaled contraction for normal gradient.
 * @param out_c2_p Output unscaled contraction for position gradient.
 */
__device__ __forceinline__ void computeSecondOrderGradientContribution(
    const Mat3x3_bf16 &G_2, const Vec3_bf16 &r_hat, const Vec3_bf16 &m,
    Vec3_bf16 &out_c2_n, Vec3_bf16 &out_c2_p) {

  // 1. Matrix-Vector Products & Trace
  const Vec3_bf16 G_2_r =
      G_2 * r_hat; // Assumes Mat3x3_bf16::operator*(Vec3_bf16)
  const Vec3_bf16 G_2_m = G_2 * m;
  const __nv_bfloat16 trace_G_2 = G_2.trace();

  // 2. Quadratic Forms via Vec3_bf16::dot
  const __nv_bfloat16 q_rr = r_hat.dot(G_2_r);
  const __nv_bfloat16 q_rm = r_hat.dot(G_2_m);
  const __nv_bfloat16 m_r = r_hat.dot(m);

  // 3. Scalar Prefactors evaluated in FP32 registers (prevents intermediate
  // precision loss)
  const float f_q_rr = __bfloat162float(q_rr);
  const float f_q_rm = __bfloat162float(q_rm);
  const float f_m_r = __bfloat162float(m_r);
  const float f_trace = __bfloat162float(trace_G_2);

  const float f_c_nr = 7.5F * f_q_rr - 1.5F * f_trace;
  const float f_c_rp =
      15.0F * f_q_rm + (7.5F * f_trace - 52.5F * f_q_rr) * f_m_r;
  const float f_15_m_r = 15.0F * f_m_r;

  // Convert scalar prefactors back to bfloat16 for vector scaling
  const __nv_bfloat16 c_nr = __float2bfloat16(f_c_nr);
  const __nv_bfloat16 c_rp = __float2bfloat16(f_c_rp);
  const __nv_bfloat16 bf_15m = __float2bfloat16(f_15_m_r);
  const __nv_bfloat16 bf_3 = __float2bfloat16(3.0F);

  // 4. Vector Contractions
  // grad_n_2 = c_nr * r_hat - 3 * (G_2 * r_hat)
  out_c2_n = (r_hat * c_nr) - (G_2_r * bf_3);

  // grad_p_2 = 15*m_r*(G_2*r) + c_r_p*r_hat + c_nr*m - 3*(G_2*m)
  out_c2_p = (G_2_r * bf_15m) + (r_hat * c_rp) + (m * c_nr) - (G_2_m * bf_3);
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
    const PointNormal &geometry, const Vec3 &center_of_mass, const half &G_0,
    const Vec3_f16 &G_1, const Mat3x3_bf16 &G_2) -> PointNormal {
  Vec3 r = center_of_mass - geometry.p;
  float inv_norm_r = r.inv_length();

  Vec3_f16 r_hat_f16 = Vec3_f16::from_float(r * inv_norm_r);
  Vec3_bf16 r_hat_bf16 = Vec3_bf16::from_float(r * inv_norm_r);
  Vec3_f16 m_f16 = Vec3_f16::from_float(geometry.n);

  float inv_norm_r2 = inv_norm_r * inv_norm_r;
  float inv_norm_r3 = inv_norm_r2 * inv_norm_r;
  float inv_norm_r4 = inv_norm_r3 * inv_norm_r;
  float inv_norm_r5 = inv_norm_r4 * inv_norm_r;

  constexpr float inv_4pi = 0.07957747154F;

  // Dipole scale factors: 1/(4pi*R^2), 1/(4pi*R^3), 1/(4pi*R^4)
  float factor_n_0 = inv_4pi * inv_norm_r2;
  float factor_n_1 = inv_4pi * inv_norm_r3;
  float factor_n_2 = inv_4pi * inv_norm_r4;

  // Position scale factors: 1/(4pi*R^3), 1/(4pi*R^4), 1/(4pi*R^5)
  float factor_p_0 = factor_n_1;
  float factor_p_1 = factor_n_2;
  float factor_p_2 = inv_4pi * inv_norm_r5;

  Vec3_f16 c0_n, c0_p;
  Vec3_f16 c1_n, c1_p;
  Vec3_bf16 c2_n, c2_p;

  computeZeroOrderGradientContribution(G_0, r_hat_f16, m_f16, c0_n, c0_p);
  computeFirstOrderGradientContribution(G_1, r_hat_f16, m_f16, c1_n, c1_p);
  computeSecondOrderGradientContribution(
      G_2, r_hat_bf16, Vec3_bf16::from_f16(m_f16), c2_n, c2_p);

  PointNormal grad;

  // Accumulate normal gradient in FP32
  grad.n.x = __half2float(c0_n.x) * factor_n_0 +
             __half2float(c1_n.x) * factor_n_1 +
             __bfloat162float(c2_n.x) * factor_n_2;
  grad.n.y = __half2float(c0_n.y) * factor_n_0 +
             __half2float(c1_n.y) * factor_n_1 +
             __bfloat162float(c2_n.y) * factor_n_2;
  grad.n.z = __half2float(c0_n.z) * factor_n_0 +
             __half2float(c1_n.z) * factor_n_1 +
             __bfloat162float(c2_n.z) * factor_n_2;

  // Accumulate position gradient in FP32
  grad.p.x = __half2float(c0_p.x) * factor_p_0 +
             __half2float(c1_p.x) * factor_p_1 +
             __bfloat162float(c2_p.x) * factor_p_2;
  grad.p.y = __half2float(c0_p.y) * factor_p_0 +
             __half2float(c1_p.y) * factor_p_1 +
             __bfloat162float(c2_p.y) * factor_p_2;
  grad.p.z = __half2float(c0_p.z) * factor_p_0 +
             __half2float(c1_p.z) * factor_p_1 +
             __bfloat162float(c2_p.z) * factor_p_2;

  if (cuda::std::isnan(grad.p.x) || cuda::std::isinf(grad.p.x) ||
      cuda::std::isnan(grad.p.y) || cuda::std::isinf(grad.p.y) ||
      cuda::std::isnan(grad.p.z) || cuda::std::isinf(grad.p.z) ||
      cuda::std::isnan(grad.n.x) || cuda::std::isinf(grad.n.x) ||
      cuda::std::isnan(grad.n.y) || cuda::std::isinf(grad.n.y) ||
      cuda::std::isnan(grad.n.z) || cuda::std::isinf(grad.n.z)) {
    float r_len = r.length();
    printf("[GRAD BREAKDOWN]\n"
           "  COM: (%.4f, %.4f, %.4f) | geom.p: (%.4f, %.4f, %.4f) | dist r: "
           "%.6e\n"
           "  inv_norm_r: %.6e | factor_n_2 (1/r^4): %.6e | factor_p_2 "
           "(1/r^5): %.6e\n"
           "  G0: %.4f | c2_p: (%.4f, %.4f, %.4f) | c2_n: (%.4f, %.4f, %.4f)\n",
           center_of_mass.x, center_of_mass.y, center_of_mass.z, geometry.p.x,
           geometry.p.y, geometry.p.z, r_len, inv_norm_r, factor_n_2,
           factor_p_2, __half2float(G_0), __bfloat162float(c2_p.x),
           __bfloat162float(c2_p.y), __bfloat162float(c2_p.z),
           __bfloat162float(c2_n.x), __bfloat162float(c2_n.y),
           __bfloat162float(c2_n.z));
  }

  return grad;
}

/**
 * @brief Helper to reconstruct an FP32 vertex gradient from scaled FP16 order
 * terms.
 */
__device__ __forceinline__ Vec3 reconstruct_vertex(
    const Vec3_f16 &term0, float scale_0th, const Vec3_f16 &term1,
    float scale_1st, const Vec3_f16 &term2, float scale_2nd) {
  return Vec3{
      __half2float(term0.x) * scale_0th + __half2float(term1.x) * scale_1st +
          __half2float(term2.x) * scale_2nd,
      __half2float(term0.y) * scale_0th + __half2float(term1.y) * scale_1st +
          __half2float(term2.y) * scale_2nd,
      __half2float(term0.z) * scale_0th + __half2float(term1.z) * scale_1st +
          __half2float(term2.z) * scale_2nd};
}

/**
 * @brief Scale-invariant 0th + 1st + 2nd order Taylor approximation of loss
 * gradients for a Triangle primitive with respect to its vertices.
 */
__device__ __forceinline__ auto compute_node_gradient_approximation(
    const Triangle &geometry, const Vec3 &center_of_mass, const half &G_0,
    const Vec3_f16 &G_1, const Mat3x3_bf16 &G_2) -> Triangle {

  // 1. Reference Scale Extraction (FP32)
  Vec3 r0_f = geometry.v0 - center_of_mass;
  Vec3 r1_f = geometry.v1 - center_of_mass;
  Vec3 r2_f = geometry.v2 - center_of_mass;

  float d0_f = r0_f.length();
  float d1_f = r1_f.length();
  float d2_f = r2_f.length();

  float L_ref = max(d0_f, max(d1_f, max(d2_f, 1e-7F)));
  float inv_L_ref = 1.F / L_ref;
  float inv_L_ref2 = inv_L_ref * inv_L_ref;

  // 2. Normalized Mapping to FP16 Domain
  Vec3_f16 r0 = Vec3_f16::from_float(r0_f * inv_L_ref);
  Vec3_f16 r1 = Vec3_f16::from_float(r1_f * inv_L_ref);
  Vec3_f16 r2 = Vec3_f16::from_float(r2_f * inv_L_ref);

  Vec3_f16 G1_bar = Vec3_f16::from_float(Vec3::from_f16(G_1) * inv_L_ref);
  Mat3x3_bf16 G2_bar = G_2 * __float2bfloat16(inv_L_ref2);

  half d0 = r0.length();
  half d1 = r1.length();
  half d2 = r2.length();

  half inv_d0 = hrcp(__hmax(d0, __float2half(1e-4F)));
  half inv_d1 = hrcp(__hmax(d1, __float2half(1e-4F)));
  half inv_d2 = hrcp(__hmax(d2, __float2half(1e-4F)));

  Vec3_f16 r0_hat = r0 * inv_d0;
  Vec3_f16 r1_hat = r1 * inv_d1;
  Vec3_f16 r2_hat = r2 * inv_d2;

  // 3. Pairwise & Geometric Quantities
  half r0_dot_r1 = r0.dot(r1);
  half r1_dot_r2 = r1.dot(r2);
  half r2_dot_r0 = r2.dot(r0);

  Vec3_f16 U0 = r1.cross(r2);
  Vec3_f16 U1 = r2.cross(r0);
  Vec3_f16 U2 = r0.cross(r1);

  half N = r0.dot(U0);

  half alpha0 = __hadd(__hmul(d1, d2), r1_dot_r2);
  half alpha1 = __hadd(__hmul(d2, d0), r2_dot_r0);
  half alpha2 = __hadd(__hmul(d0, d1), r0_dot_r1);

  half d012 = __hmul(d0, __hmul(d1, d2));
  half D = __hadd(d012,
                  __hadd(__hmul(r0_dot_r1, d2),
                         __hadd(__hmul(r1_dot_r2, d0), __hmul(r2_dot_r0, d1))));

  Vec3_f16 V0 = Vec3_f16::fma(r0_hat, alpha0, Vec3_f16::fma(r1, d2, r2 * d1));
  Vec3_f16 V1 = Vec3_f16::fma(r1_hat, alpha1, Vec3_f16::fma(r2, d0, r0 * d2));
  Vec3_f16 V2 = Vec3_f16::fma(r2_hat, alpha2, Vec3_f16::fma(r0, d1, r1 * d0));

  half S = __hadd(__hmul(N, N), __hmul(D, D));
  half inv_S = hrcp(__hmax(S, __float2half(1e-6F)));
  half inv_S2 = __hmul(inv_S, inv_S);
  half inv_S3 = __hmul(inv_S2, inv_S);

  half neg_N = __hneg(N);
  Vec3_f16 W0 = Vec3_f16::fma(U0, D, V0 * neg_N);
  Vec3_f16 W1 = Vec3_f16::fma(U1, D, V1 * neg_N);
  Vec3_f16 W2 = Vec3_f16::fma(U2, D, V2 * neg_N);

  // 4. 0th Order Evaluation
  half factor_2_over_S = __hmul(__float2half(2.F), inv_S);
  half coeff_W_0th = __hmul(G_0, factor_2_over_S);

  Vec3_f16 term0_v0 = W0 * coeff_W_0th;
  Vec3_f16 term0_v1 = W1 * coeff_W_0th;
  Vec3_f16 term0_v2 = W2 * coeff_W_0th;

  // 5. 1st Order Contractions with G1_bar
  half s_U0 = U0.dot(G1_bar);
  half s_U1 = U1.dot(G1_bar);
  half s_U2 = U2.dot(G1_bar);
  half s_V0 = V0.dot(G1_bar);
  half s_V1 = V1.dot(G1_bar);
  half s_V2 = V2.dot(G1_bar);
  half s_r0_hat = r0_hat.dot(G1_bar);
  half s_r1_hat = r1_hat.dot(G1_bar);
  half s_r2_hat = r2_hat.dot(G1_bar);
  half s_r0 = r0.dot(G1_bar);
  half s_r1 = r1.dot(G1_bar);
  half s_r2 = r2.dot(G1_bar);

  half S_U = __hadd(s_U0, __hadd(s_U1, s_U2));
  half S_V = __hadd(s_V0, __hadd(s_V1, s_V2));

  Vec3_f16 dU0_sum = G1_bar.cross(r2 - r1);
  Vec3_f16 dU1_sum = G1_bar.cross(r0 - r2);
  Vec3_f16 dU2_sum = G1_bar.cross(r1 - r0);

  half alpha0_over_d0 = __hmul(alpha0, inv_d0);
  half alpha1_over_d1 = __hmul(alpha1, inv_d1);
  half alpha2_over_d2 = __hmul(alpha2, inv_d2);

  half c_G1_0 = __hadd(alpha0_over_d0, __hadd(d1, d2));
  half c_G1_1 = __hadd(alpha1_over_d1, __hadd(d2, d0));
  half c_G1_2 = __hadd(alpha2_over_d2, __hadd(d0, d1));

  half c_r0_hat =
      __hadd(__hmul(__hneg(alpha0_over_d0), s_r0_hat),
             __hadd(__hmul(d2, s_r1_hat),
                    __hadd(__hmul(d1, s_r2_hat), __hadd(s_r1, s_r2))));
  half c_r1_hat =
      __hadd(__hmul(__hneg(alpha1_over_d1), s_r1_hat),
             __hadd(__hmul(d0, s_r2_hat),
                    __hadd(__hmul(d2, s_r0_hat), __hadd(s_r2, s_r0))));
  half c_r2_hat =
      __hadd(__hmul(__hneg(alpha2_over_d2), s_r2_hat),
             __hadd(__hmul(d1, s_r0_hat),
                    __hadd(__hmul(d0, s_r1_hat), __hadd(s_r0, s_r1))));

  Vec3_f16 dV0_sum =
      Vec3_f16::fma(G1_bar, c_G1_0,
                    Vec3_f16::fma(r0_hat, c_r0_hat,
                                  Vec3_f16::fma(r1, s_r2_hat, r2 * s_r1_hat)));
  Vec3_f16 dV1_sum =
      Vec3_f16::fma(G1_bar, c_G1_1,
                    Vec3_f16::fma(r1_hat, c_r1_hat,
                                  Vec3_f16::fma(r2, s_r0_hat, r0 * s_r2_hat)));
  Vec3_f16 dV2_sum =
      Vec3_f16::fma(G1_bar, c_G1_2,
                    Vec3_f16::fma(r2_hat, c_r2_hat,
                                  Vec3_f16::fma(r0, s_r1_hat, r1 * s_r0_hat)));

  half factor_4_over_S2 = __hmul(__float2half(4.F), inv_S2);
  half N_SU_plus_D_SV = __hadd(__hmul(N, S_U), __hmul(D, S_V));
  half coeff_W_1st = __hmul(N_SU_plus_D_SV, factor_4_over_S2);

  Vec3_f16 dW0_sum = (dV0_sum * N + V0 * S_U) - (dU0_sum * D + U0 * S_V);
  Vec3_f16 dW1_sum = (dV1_sum * N + V1 * S_U) - (dU1_sum * D + U1 * S_V);
  Vec3_f16 dW2_sum = (dV2_sum * N + V2 * S_U) - (dU2_sum * D + U2 * S_V);

  Vec3_f16 term1_v0 = Vec3_f16::fma(dW0_sum, factor_2_over_S, W0 * coeff_W_1st);
  Vec3_f16 term1_v1 = Vec3_f16::fma(dW1_sum, factor_2_over_S, W1 * coeff_W_1st);
  Vec3_f16 term1_v2 = Vec3_f16::fma(dW2_sum, factor_2_over_S, W2 * coeff_W_1st);

  // 6. 2nd Order Hessian Contractions with G2_bar (Mat3x3_bf16)
  Vec3_f16 grad_S_0 = (U0 * N + V0 * D) * __float2half(2.F);
  Vec3_f16 grad_S_1 = (U1 * N + V1 * D) * __float2half(2.F);
  Vec3_f16 grad_S_2 = (U2 * N + V2 * D) * __float2half(2.F);

  // Promote grad_S vectors to Vec3_bf16 for Mat3x3_bf16 quadric_form evaluation
  Vec3_bf16 grad_S_0_bf = Vec3_bf16::from_f16(grad_S_0);
  Vec3_bf16 grad_S_1_bf = Vec3_bf16::from_f16(grad_S_1);
  Vec3_bf16 grad_S_2_bf = Vec3_bf16::from_f16(grad_S_2);

  float gS_G2_gS_0 = __bfloat162float(G2_bar.quadric_form(grad_S_0_bf));
  float gS_G2_gS_1 = __bfloat162float(G2_bar.quadric_form(grad_S_1_bf));
  float gS_G2_gS_2 = __bfloat162float(G2_bar.quadric_form(grad_S_2_bf));

  float quad_S_sum = gS_G2_gS_0 + gS_G2_gS_1 + gS_G2_gS_2;

  half lap_S_0 = __hmul(__float2half(2.F), __hadd(U0.dot(U0), V0.dot(V0)));
  half lap_S_1 = __hmul(__float2half(2.F), __hadd(U1.dot(U1), V1.dot(V1)));
  half lap_S_2 = __hmul(__float2half(2.F), __hadd(U2.dot(U2), V2.dot(V2)));

  float lap_S_sum = __half2float(__hadd(lap_S_0, __hadd(lap_S_1, lap_S_2)));
  float lap_S_G2_sum = __bfloat162float(G2_bar.trace()) * lap_S_sum;

  // Evaluate scalar polynomial combination in FP32 registers
  float f_inv_S2 = __half2float(inv_S2);
  float f_inv_S3 = __half2float(inv_S3);
  float f_factor_4_over_S2 = 4.0F * f_inv_S2;
  float f_factor_8_over_S3 = 8.0F * f_inv_S3;

  float f_coeff_W_2nd =
      quad_S_sum * f_factor_8_over_S3 - lap_S_G2_sum * f_factor_4_over_S2;
  half coeff_W_2nd = __float2half(f_coeff_W_2nd);

  Vec3_f16 dW_G2_gS_0 = dW0_sum * grad_S_0.dot(G1_bar);
  Vec3_f16 dW_G2_gS_1 = dW1_sum * grad_S_1.dot(G1_bar);
  Vec3_f16 dW_G2_gS_2 = dW2_sum * grad_S_2.dot(G1_bar);

  Vec3_f16 cross_term_sum = dW_G2_gS_0 + dW_G2_gS_1 + dW_G2_gS_2;

  Vec3_f16 term2_v0 =
      Vec3_f16::fma(W0, coeff_W_2nd, cross_term_sum * factor_4_over_S2);
  Vec3_f16 term2_v1 =
      Vec3_f16::fma(W1, coeff_W_2nd, cross_term_sum * factor_4_over_S2);
  Vec3_f16 term2_v2 =
      Vec3_f16::fma(W2, coeff_W_2nd, cross_term_sum * factor_4_over_S2);

  // 7. Final Scale Reconstruction in FP32
  constexpr float inv_4pi = 0.07957747154F;
  float scale_0th = inv_4pi * inv_L_ref;
  float scale_1st = inv_4pi * inv_L_ref2;
  float scale_2nd = inv_4pi * (inv_L_ref2 * inv_L_ref);

  return Triangle{reconstruct_vertex(term0_v0, scale_0th, term1_v0, scale_1st,
                                     term2_v0, scale_2nd),
                  reconstruct_vertex(term0_v1, scale_0th, term1_v1, scale_1st,
                                     term2_v1, scale_2nd),
                  reconstruct_vertex(term0_v2, scale_0th, term1_v2, scale_2nd,
                                     term2_v2, scale_2nd)};
}
