#pragma once
#include "aabb.h"
#include "mat3x3.h"
#include "tailor_coefficients.h"
#include "tensor3.h"
#include "vec3.h"
#include <cuda_fp16.h>

__device__ __forceinline__ auto
should_node_be_approximated(const Vec3 &query, const AABB &aabb,
                                  const float beta_2) -> bool {
  float max_distance_to_center = __half2float(aabb.max_distance);
  Vec3 com = aabb.center_of_mass;
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
 * Contraction = C :: G_hat = 15 * sum_{i,j,k} (C_ijk * r_hat_i * r_hat_j *
 * r_hat_k) - 3 * (V · r_hat)
 *
 * where:
 * - G_hat_ijk = 15 * (r_hat_i * r_hat_j * r_hat_k) - 3 * (r_hat x I)_sym
 * - V is the partial vector-trace of the tensor C, where V_i = sum_j (C_ijj)
 * - (r_hat x I)_sym isolates Kronecker delta matches: (r_hat_i * delta_jk +
 * r_hat_j * delta_ik + r_hat_k * delta_ij)
 *
 * Symmetries in the expansion reduce the 27-term tensor contraction to 18
 * unique terms unpacked directly from the layout matrix. The complete physical
 * contribution is reconstructed by the caller via: Contribution = Contraction *
 * [1 / (4 * pi * ||r||^4)]
 *
 * @param C The second-order rank-3 multipole coefficient tensor stored in
 * symmetric compressed format.
 * @param r_hat Normalized unit displacement vector pointing from query to the
 * cluster center (||r_hat|| = 1).
 * @return Unscaled contraction scalar evaluated in float16 precision.
 */
__device__ __forceinline__ auto
computeSecondOrderContribution(const Tensor3_f16_compressed &C,
                               const Vec3_f16 &r_hat) -> float {
  const half two = __float2half(2.0f);

  // Vector Trace V
  half vx = __hadd(__hadd(C.data[0], C.data[4]), C.data[8]);
  half vy = __hadd(__hadd(C.data[3], C.data[10]), C.data[14]);
  half vz = __hadd(__hadd(C.data[6], C.data[13]), C.data[17]);

  // V dot r_hat
  half v_dot_r = __hadd(__hadd(__hmul(vx, r_hat.x), __hmul(vy, r_hat.y)),
                        __hmul(vz, r_hat.z));

  // Pre-calculate baseline quadratic components
  half r_xx = __hmul(r_hat.x, r_hat.x);
  half r_yy = __hmul(r_hat.y, r_hat.y);
  half r_zz = __hmul(r_hat.z, r_hat.z);
  half r_xy = __hmul(r_hat.x, r_hat.y);
  half r_xz = __hmul(r_hat.x, r_hat.z);
  half r_yz = __hmul(r_hat.y, r_hat.z);

  half2 accumulator =
      __hmul2(__halves2half2(C.data[0], C.data[1]),
              __halves2half2(__hmul(r_xx, r_hat.x), __hmul(r_xx, r_hat.y)));
  accumulator = __hfma2(
      __halves2half2(C.data[2], C.data[3]),
      __halves2half2(__hmul(r_xx, r_hat.z), __hmul(__hmul(r_xy, r_hat.x), two)),
      accumulator);
  accumulator = __hfma2(__halves2half2(C.data[4], C.data[5]),
                        __halves2half2(__hmul(__hmul(r_xy, r_hat.y), two),
                                       __hmul(__hmul(r_xz, r_hat.y), two)),
                        accumulator);
  accumulator = __hfma2(__halves2half2(C.data[6], C.data[7]),
                        __halves2half2(__hmul(__hmul(r_xz, r_hat.x), two),
                                       __hmul(__hmul(r_yz, r_hat.x), two)),
                        accumulator);
  accumulator = __hfma2(
      __halves2half2(C.data[8], C.data[9]),
      __halves2half2(__hmul(__hmul(r_xz, r_hat.z), two), __hmul(r_yy, r_hat.x)),
      accumulator);
  accumulator =
      __hfma2(__halves2half2(C.data[10], C.data[11]),
              __halves2half2(__hmul(r_yy, r_hat.y), __hmul(r_yy, r_hat.z)),
              accumulator);
  accumulator = __hfma2(__halves2half2(C.data[12], C.data[13]),
                        __halves2half2(__hmul(__hmul(r_yz, r_hat.x), two),
                                       __hmul(__hmul(r_yz, r_hat.y), two)),
                        accumulator);
  accumulator = __hfma2(
      __halves2half2(C.data[14], C.data[15]),
      __halves2half2(__hmul(__hmul(r_yz, r_hat.z), two), __hmul(r_zz, r_hat.x)),
      accumulator);
  accumulator =
      __hfma2(__halves2half2(C.data[16], C.data[17]),
              __halves2half2(__hmul(r_zz, r_hat.y), __hmul(r_zz, r_hat.z)),
              accumulator);

  half result = __hadd(__low2half(accumulator), __high2half(accumulator));

  // Apply final balanced weights: (15 * tensor_part) - (3 * vector_part)
  result = __hmul(result, __float2half(15.0f));
  half v_part = __hmul(v_dot_r, __float2half(3.0f));
  result = __hsub(result, v_part);

  return __half2float(result);
}

__device__ __forceinline__ auto compute_node_approximation(
    const Vec3 &query, const Vec3 &center_of_mass,
    const Vec3_f16 &zero_order_coeff, const Mat3x3_f16 &first_order_coeff,
    const Tensor3_f16_compressed &second_order_coeff) -> float {
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
