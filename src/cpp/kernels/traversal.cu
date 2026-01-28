#include "aabb.h"
#include "bvh8.h"
#include "common.cuh"
#include "geometry.h"
#include "mat3x3.h"
#include "tailor_coefficients.h"
#include "tensor3.h"
#include "vec3.h"
#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
#include <cstdint>
#include <cub/block/block_scan.cuh>
#include <cub/util_type.cuh>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_atomic_functions.h>
#include <driver_types.h>
#include <vector_functions.h>
#include <vector_types.h>

// Note instead of using the max distance from the aabb_center to any of its
// contained elements we use the aabbs radius (half the diagonal). This is a
// conservative overestimation, leading to slightly less approximation compared
// to the Fast Winding Numbers method. However it prevents an aditional 4 byte
// load and should not lead to significantly more computations.
__device__ __forceinline__ auto should_node_be_aproximated(const Vec3 &query,
                                                           const AABB &aabb,
                                                           const float beta_2)
    -> bool {
  return (query - aabb.center()).length2() > aabb.radius_sq() * beta_2;
}

__device__ __forceinline__ auto to_bits(const nv_bfloat162 bf16x2) -> uint32_t {
  uint32_t result = *(uint32_t *)&bf16x2;
  return result;
}

__device__ __forceinline__ auto to_bits(const nv_bfloat16 bf16) -> short {
  short result = *(short *)&bf16;
  return result;
}

// TODO: Write a test for this!
__device__ __forceinline__ auto
computeZeroOrderContribution(const Vec3_bf16 &coeff, const Vec3_bf16 &r,
                             const nv_bfloat16 inv_r3 // 1/(4 pi |r|^3)
                             ) -> float {
  // Pack bf16s into 32-bit registers
  nv_bfloat162 c_xy = __halves2bfloat162(coeff.x, coeff.y);
  nv_bfloat162 c_z = __halves2bfloat162(coeff.z, __float2bfloat16(0.0f));

  nv_bfloat162 r_xy = __halves2bfloat162(r.x, r.y);
  nv_bfloat162 r_z = __halves2bfloat162(r.z, __float2bfloat16(0.0f));

  uint32_t result;
  uint32_t zero = 0;

  // PTX for enforced packed fma logic
  asm("{\n\t"
      " .reg .b16 lo, hi;\n\t"
      // %0 = (cx*rx, cy*ry) + 0
      " fma.rn.bf16x2 %0, %1, %2, %6;\n\t"
      // %0 = (cz*rz, 0) + %0
      // %0 is [ (cx*rx + cz*rz) | (cy*ry) ]
      " fma.rn.bf16x2 %0, %3, %4, %0;\n\t"
      // Destructure %0 into two 16-bit regs
      " mov.b32 {lo, hi}, %0;\n\t"
      // hi = lo + hi = (cx*rx+cz*rz) + (cy*ry)
      " add.bf16 hi, lo, hi;\n\t"
      // hi = hi * scale
      " mul.bf16 hi, hi, %5;\n\t"
      // 'hi' in the bottom 16 bits
      // mov.b32 d {a, b}:
      // d = a.x | (a.y << 16)
      " mov.b32 %0, {hi, 0};\n\t"
      "}"
      : "=r"(result)          // %0
      : "r"(to_bits(c_xy)),   // %1
        "r"(to_bits(r_xy)),   // %2
        "r"(to_bits(c_z)),    // %3
        "r"(to_bits(r_z)),    // %4
        "h"(to_bits(inv_r3)), // %5
        "r"(zero));           // %6

  return __bfloat162float(*(nv_bfloat16 *)&result);
}

// TODO: Write a test for this!
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
computeFirstOrderContribution(const Mat3x3_bf16 &C, const Vec3_bf16 &r,
                              const nv_bfloat16 inv_r3, // 1 / (4 pi |r|^3)
                              const nv_bfloat16 inv_r5  // 3 / (4 pi |r|^5)
                              ) -> float {
  nv_bfloat16 three = __float2bfloat16(3.F);
  // After unrolling and factoring we compute:
  // Contribution = (C_11+C_22+C_33)/(4pi||r||^3) -
  //                (C_11*r_1^2 + C_22*r_2^2 + C_33*r_3^2 +
  //                (C_12+C_21)*r_1*r_2 + (C_13+C_31)*r_1*r_3 +
  //                (C_23+C_32)*r_2*r_3)/(4pi||r||^5)

  // diagonal part
  nv_bfloat16 trace_C = (C.data[0] + C.data[4]) + C.data[8];

  // Pre-sum off-diagonals (symetric parts)
  nv_bfloat16 C_xy_yx = C.data[1] + C.data[3];
  nv_bfloat16 C_xz_zx = C.data[2] + C.data[6];
  nv_bfloat16 C_yz_zy = C.data[5] + C.data[7];

  // Prepare r_i * r_j pairs
  nv_bfloat162 r2_xy = __halves2bfloat162(r.x * r.x, r.y * r.y);
  nv_bfloat162 r2_z_xy = __halves2bfloat162(r.z * r.z, r.x * r.y);
  nv_bfloat162 r2_xz_yz = __halves2bfloat162(r.x * r.z, r.y * r.z);

  // Pack G components
  nv_bfloat162 c_xy = __halves2bfloat162(C.data[0],
                                         C.data[4]);            // Cxx, Cyy
  nv_bfloat162 c_z_xy = __halves2bfloat162(C.data[8], C_xy_yx); // Czz, Cxy+Cyx
  nv_bfloat162 c_xz_yz =
      __halves2bfloat162(C_xz_zx, C_yz_zy); // Cxz+zx, Cyz+Czy

  uint32_t result;
  uint32_t zero = 0;

  // PTX for enforced packed fma logic
  asm("{\n\t"
      ".reg .b16 lo, hi, final_sum;\n\t"
      // %0 = {Cxx*rx*rx + Czz*rz*rz + (Cxz+Czy)*rx*rz},
      //      {Cyy*ry*ry + (Cxy+Cyx)*rx*ry + (Cyz+Czy)*ry*rz}
      "fma.rn.bf16x2 %0, %1, %2, %10;\n\t"
      "fma.rn.bf16x2 %0, %3, %4, %0;\n\t"
      "fma.rn.bf16x2 %0, %5, %6, %0;\n\t"

      // add the two bf16 together
      // final_sum = Cxx*rx*rx + Czz*rz*rz + (Cxz+Czy)*rx*rz +
      //             Cyy*ry*ry + (Cxy+Cyx)*rx*ry + (Cyz+Czy)*ry*rz
      "mov.b32 {lo, hi}, %0;\n\t"
      "add.bf16 final_sum, lo, hi;\n\t"

      // lo = (Cxx+Cyy+Czz) * inv_3
      "mul.bf16 lo, %7, %8;\n\t"
      // hi = final_sum * inv_5
      "mul.bf16 hi, final_sum, %9;\n\t"
      // final_sum = lo - hi
      "sub.bf16 final_sum, lo, hi;\n\t"

      // write out result (lowest 16 bit are the same for bf16)
      "mov.b32 %0, {final_sum, 0};\n\t"
      "}"
      : "=r"(result)            // %0
      : "r"(to_bits(c_xy)),     // %1
        "r"(to_bits(r2_xy)),    // %2
        "r"(to_bits(c_z_xy)),   // %3
        "r"(to_bits(r2_z_xy)),  // %4
        "r"(to_bits(c_xz_yz)),  // %5
        "r"(to_bits(r2_xz_yz)), // %6
        "h"(to_bits(trace_C)),  // %7
        "h"(to_bits(inv_r3)),   // %8
        "h"(to_bits(inv_r5)),   // %9
        "r"(zero)               // %10
  );

  return __bfloat162float(*(nv_bfloat16 *)&result);
}

// TODO: Write a test for this!
/**
 * Computes the Second Order Taylor contribution to the Winding Number.
 *
 * Mathematically, this is the contraction between the second-order 
 * coefficient tensor C (rank 3) and the third-order field gradient tensor G (rank 3):
 * * Contribution = C : G = sum_{i,j,k} C_ijk * G_ijk
 * * where the field gradient tensor G is:
 * G(r) = [ 15 * (r \otimes r \otimes r) / |r|^7 ] - [ 3 * (r \otimes I)_sym / |r|^5 ]
 * * Here:
 * - I is the Identity matrix (I_jk = \delta_jk).
 * - \delta is the Kronecker delta (\delta_jk = 1 if j=k, else 0).
 * - \otimes is the outer product.
 * - (r \otimes I)_sym is the symmetric combination: (r_i \delta_jk + r_j \delta_ik + r_k \delta_ij).
 *
 * Implementation note: Due to the partial symmetry of C (C_ijk = C_ikj) and the 
 * cross-plane symmetries in this specific Taylor expansion, the 27-term 
 * contraction reduces to 18 unique terms stored in the compressed format. 
 * The second term (involving \delta) simplifies to a dot product between r and 
 * the partial vector-trace of C.
 */
__device__ __forceinline__ auto
computeSecondOrderContribution(const Tensor3_bf16_compressed &C,
                               const Vec3_bf16 &r,
                               const nv_bfloat16 inv_r5, // 3.0f / (4pi * |r|^5)
                               const nv_bfloat16 inv_r7 // 15.0f / (4pi * |r|^7)
) -> float{
  const nv_bfloat16 two = __float2bfloat16(2.0f);

  // Vector Trace V
  // Cxxx+Cxyy+Cxzz
  nv_bfloat16 vx = (C.data[0] + C.data[4]) + C.data[8];
  // Cyxx+Cyyy+Cyzz
  nv_bfloat16 vy = (C.data[3] + C.data[10]) + C.data[14];
  // Czxx+Czyy+Czzz
  nv_bfloat16 vz = (C.data[6] + C.data[13]) + C.data[17];
  // V dot r
  nv_bfloat16 v_dot_r = vx * r.x + vy * r.y + vz * r.z;

  // 18 Weighted r-products
  // Pre-multiply by 2.0 where the coefficient is shared by two tensor slots
  nv_bfloat16 r_prods[18];
  r_prods[0] = r.x * r.x * r.x;          // data[0]: xxx
  r_prods[1] = r.x * r.x * r.y;          // data[1]: xxy
  r_prods[2] = r.x * r.x * r.z;          // data[2]: xxz
  r_prods[3] = (r.y * r.x * r.x) * two;  // data[3]: yxx + xyx
  r_prods[4] = (r.x * r.y * r.y) * two;  // data[4]: xyy + yxy
  r_prods[5] = (r.x * r.y * r.z) * two;  // data[5]: xyz + yxz
  r_prods[6] = (r.z * r.x * r.x) * two;  // data[6]: zxx + xzx
  r_prods[7] = (r.z * r.x * r.y) * two;  // data[7]: zxy + xzy
  r_prods[8] = (r.z * r.x * r.z) * two;  // data[8]: zxz + xzz
  r_prods[9] = r.y * r.y * r.x;          // data[9]: yyx
  r_prods[10] = r.y * r.y * r.y;         // data[10]: yyy
  r_prods[11] = r.y * r.y * r.z;         // data[11]: yyz
  r_prods[12] = (r.y * r.z * r.x) * two; // data[12]: yzx + zyx
  r_prods[13] = (r.z * r.y * r.y) * two; // data[13]: zyy + yzy
  r_prods[14] = (r.y * r.z * r.z) * two; // data[14]: yzz + zyz
  r_prods[15] = r.z * r.z * r.x;         // data[15]: zzx
  r_prods[16] = r.z * r.z * r.y;         // data[16]: zzy
  r_prods[17] = r.z * r.z * r.z;         // data[17]: zzz

  // FMA
  uint32_t res_raw;
  uint32_t zero = 0;

  asm("{\n\t"
      " .reg .b16 lo, hi;\n\t"

      // 9 Vector FMAs for 18 terms (8 instructions)
      " fma.rn.bf16x2 %0, %1,  %2,  %19;\n\t"
      " fma.rn.bf16x2 %0, %3,  %4,  %0;\n\t"
      " fma.rn.bf16x2 %0, %5,  %6,  %0;\n\t"
      " fma.rn.bf16x2 %0, %7,  %8,  %0;\n\t"
      " fma.rn.bf16x2 %0, %9,  %10, %0;\n\t"
      " fma.rn.bf16x2 %0, %11, %12, %0;\n\t"
      " fma.rn.bf16x2 %0, %13, %14, %0;\n\t"
      " fma.rn.bf16x2 %0, %15, %16, %0;\n\t"
      " fma.rn.bf16x2 %0, %17, %18, %0;\n\t"

      // add together both results for the r^3 part
      " mov.b32 {lo, hi}, %0;\n\t"
      " add.bf16 lo, lo, hi;\n\t"

      // Final Scale: (lo * inv_r7) - (v_dot_r * inv_r5)
      " mul.bf16 lo, lo, %20;\n\t"
      " mul.bf16 hi, %21, %22;\n\t"
      " sub.bf16 lo, lo, hi;\n\t"

      " mov.b32 %0, {lo, 0};\n\t"
      "}"
      : "=r"(res_raw)                                               // %0
      : "r"(to_bits(__halves2bfloat162(C.data[0], C.data[1]))),     // %1
        "r"(to_bits(__halves2bfloat162(r_prods[0], r_prods[1]))),   // %2
        "r"(to_bits(__halves2bfloat162(C.data[2], C.data[3]))),     // %3
        "r"(to_bits(__halves2bfloat162(r_prods[2], r_prods[3]))),   // %4
        "r"(to_bits(__halves2bfloat162(C.data[4], C.data[5]))),     // %5
        "r"(to_bits(__halves2bfloat162(r_prods[4], r_prods[5]))),   // %6
        "r"(to_bits(__halves2bfloat162(C.data[6], C.data[7]))),     // %7
        "r"(to_bits(__halves2bfloat162(r_prods[6], r_prods[7]))),   // %8
        "r"(to_bits(__halves2bfloat162(C.data[8], C.data[9]))),     // %9
        "r"(to_bits(__halves2bfloat162(r_prods[8], r_prods[9]))),   // %10
        "r"(to_bits(__halves2bfloat162(C.data[10], C.data[11]))),   // %11
        "r"(to_bits(__halves2bfloat162(r_prods[10], r_prods[11]))), // %12
        "r"(to_bits(__halves2bfloat162(C.data[12], C.data[13]))),   // %13
        "r"(to_bits(__halves2bfloat162(r_prods[12], r_prods[13]))), // %14
        "r"(to_bits(__halves2bfloat162(C.data[14], C.data[15]))),   // %15
        "r"(to_bits(__halves2bfloat162(r_prods[14], r_prods[15]))), // %16
        "r"(to_bits(__halves2bfloat162(C.data[16], C.data[17]))),   // %17
        "r"(to_bits(__halves2bfloat162(r_prods[16], r_prods[17]))), // %18
        "r"(zero),                                                  // %19
        "h"(to_bits(inv_r7)),                                       // %20
        "h"(to_bits(v_dot_r)),                                      // %21
        "h"(to_bits(inv_r5))                                        // %22
  );

  return __bfloat162float(*(nv_bfloat16 *)&res_raw);
}

template <typename Geometry>
__global__ void __launch_bounds__(128, 8) compute_winding_numbers_kernel(
    const Vec3 *queries, const uint32_t *sort_indirections,
    const BVH8Node *bvh8_nodes, const LeafPointers *bvh8_leaf_pointers,
    const TailorCoefficientsBf16 *leaf_coefficients,
    const Geometry *sorted_geometry, const uint32_t query_count,
    float *winding_numbers, const float beta_2) {

  const uint32_t warp_id = threadIdx.x / 32;
  const uint32_t lane_id = threadIdx.x % 32;

  // We split the queries into tiles of 128.
  // Each block of 128 threads works on one tile at a time.
  // 4 warps per 128 threads
  // Each warp has its own shared traversal stack.
  __shared__ uint32_t
      shared_stack[4][12]; // TODO find minimum required stack size
  __shared__ BVH8Node current_node_cache[4];

  static __device__ uint32_t global_device_idx = 0;
  __shared__ uint32_t tile_base;

  while (true) {
    // Claim a tile
    // Dynamic work balancing. Not all blocks will need the same amount of time
    // for their queries
    if (threadIdx.x == 0) {
      tile_base = atomicAdd(&global_device_idx, blockDim.x);
    }
    __syncthreads();

    uint32_t my_query_idx = tile_base + threadIdx.x;
    if (my_query_idx > query_count) {
      return; // TODO make sure full warp stays alife
    }
    // load query with sort indirections
    uint32_t original_query_idx = sort_indirections[my_query_idx];
    Vec3 my_query = queries[original_query_idx];
    float my_winding_number = 0.F;

    // Always start traversal on root
    uint32_t stack_ptr = 0;
    if (lane_id == 0) {
      shared_stack[warp_id][stack_ptr++] = 0;
    }

    // Do traversal
    while (true) {
      // Check if stack is empty
      uint32_t stack_not_empty_mask = __ballot_sync(0xFFFFFFFF, stack_ptr > 0);
      // stop if all warps are done
      if (stack_not_empty_mask == 0) {
        break;
      }
      // warp leader pops next node
      if (lane_id == 0) {
        uint32_t current_node_idx = shared_stack[warp_id][--stack_ptr];
        current_node_cache[warp_id] = bvh8_nodes[current_node_idx];
      }

      // process current node
      // 1. Check the nodes parent_aabb. If it is too far away
      if (should_node_be_aproximated(
              my_query, current_node_cache[warp_id].parent_aabb, beta_2)) {
        // compute approximation in bf16
        Vec3 p = current_node_cache[warp_id].parent_aabb.center();
        Vec3_bf16 r = Vec3_bf16::from_float(p - my_query);
        nv_bfloat16 norm_r = hrsqrt(r.length2());
        nv_bfloat16 norm_r3 = norm_r * norm_r * norm_r;
        nv_bfloat16 norm_r5 = norm_r3 * norm_r * norm_r;
        nv_bfloat16 norm_r7 = norm_r5 * norm_r * norm_r;

        // 1/(4*pi)
        nv_bfloat16 inv_4pi = __float2bfloat16(0.07957747154);

        nv_bfloat16 inv_4_pi_normr3 = inv_4pi / norm_r3;
        nv_bfloat16 inv_4_pi_normr5 = __float2bfloat16(3.F) * inv_4pi / norm_r5;
        nv_bfloat16 inv_4_pi_normr7 =
            __float2bfloat16(15.F) * inv_4pi / norm_r7;

        // Shared scale factor of quantized tailor coefficients
        float scale_factor = current_node_cache[warp_id]
                                 .tailor_coefficients.get_shared_scale_factor();
        // Zero Order
        Vec3_bf16 zero_order_coeff =
            current_node_cache[warp_id]
                .tailor_coefficients.get_tailor_zero_order(scale_factor);
        my_winding_number +=
            computeZeroOrderContribution(zero_order_coeff, r, inv_4_pi_normr3);
        // First Order
        Mat3x3_bf16 first_order_coeff =
            current_node_cache[warp_id]
                .tailor_coefficients.get_tailor_first_order(scale_factor);
        my_winding_number += computeFirstOrderContribution(
            first_order_coeff, r, inv_4_pi_normr3, inv_4_pi_normr5);
        // Second order
        Tensor3_bf16_compressed second_order_coeff =
            current_node_cache[warp_id]
                .tailor_coefficients.get_tailor_second_order(scale_factor);
        my_winding_number += computeSecondOrderContribution(
            second_order_coeff, r, inv_4_pi_normr5, inv_4_pi_normr7);
      } else {
        // TODO handle children in lock step!
        // continue here
      }
    }
    winding_numbers[original_query_idx] = my_winding_number;
  }
}

// defaults for beta
template <typename T> struct GeometryTraits {
  static constexpr float default_beta = 2.3F;
};
template <> struct GeometryTraits<PointNormal> {
  static constexpr float default_beta = 2.0F;
};

template <typename Geometry>
void compute_winding_numbers(const Vec3 *queries,
                             const uint32_t *sort_indirections,
                             const BVH8Node *bvh8_nodes,
                             const LeafPointers *bvh8_leaf_pointers,
                             const TailorCoefficientsBf16 *leaf_coefficients,
                             const Geometry *sorted_geometry,
                             const uint32_t query_count, float *winding_numbers,
                             const int device_id, const cudaStream_t &stream,
                             float beta) {
  if (query_count == 0) {
    return;
  }
  // if beta is negative initialize with default value
  float actual_beta =
      beta < 0.F ? GeometryTraits<Geometry>{}.default_beta : beta;
  float beta_2 = actual_beta * actual_beta;

  int threads = 128;
  int blocks_per_sm = 0;

  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, compute_winding_numbers_kernel<Geometry>, threads, 0);

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device_id);
  int blocks = blocks_per_sm * deviceProp.multiProcessorCount;

  compute_winding_numbers_kernel<Geometry><<<blocks, threads, 0, stream>>>(
      queries, sort_indirections, bvh8_nodes, bvh8_leaf_pointers,
      leaf_coefficients, sorted_geometry, query_count, winding_numbers, beta_2);
}

template void compute_winding_numbers<PointNormal>(
    const Vec3 *queries, const uint32_t *sort_indirections,
    const BVH8Node *bvh8_nodes, const LeafPointers *bvh8_leaf_pointers,
    const TailorCoefficientsBf16 *leaf_coefficients,
    const PointNormal *sorted_geometry, const uint32_t query_count,
    float *winding_numbers, const int device_id, const cudaStream_t &stream,
    float beta);
template void compute_winding_numbers<Triangle>(
    const Vec3 *queries, const uint32_t *sort_indirections,
    const BVH8Node *bvh8_nodes, const LeafPointers *bvh8_leaf_pointers,
    const TailorCoefficientsBf16 *leaf_coefficients,
    const Triangle *sorted_geometry, const uint32_t query_count,
    float *winding_numbers, const int device_id, const cudaStream_t &stream,
    float beta);
