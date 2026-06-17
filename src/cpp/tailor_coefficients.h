#pragma once

#include "center_of_mass.h"
#include "mat3x3.h"
#include "tensor3.h"
#include "vec3.h"
#include <cstdint>
#include <cuda_runtime_api.h>
#include <vector_types.h>

// tailor coefficients quantized to 11 bit each for inner BVH8 node
// with a shared 8 bit exponent
// first 4 bytes are reused for parent index
// coefficients: zero order (3), first order (9), second order compressed (18)
// total: 30 coefficients
// 44 bytes
struct TailorCoefficientsQuantized {
  uint32_t tailor_data[11];

  // During tree construction we use the tailor_data memory to temporarily store
  // how many children the node actually has.
  __device__ inline auto get_expected_children() const -> uint32_t;
  __device__ inline void set_expected_children(uint32_t idx);

  __device__ inline auto get_shared_scale_factor_low() const -> float;
  __device__ inline auto get_shared_scale_factor_second() const -> float;

  __device__ inline void
  set_tailor_coefficients(const Vec3 &zero_order, const Mat3x3 &first_order,
                          const Tensor3_compressed &second_order);

  __device__ inline auto get_tailor_zero_order() const -> Vec3_f16;

  __device__ inline auto get_tailor_first_order() const -> Mat3x3_f16;

  __device__ inline auto
  get_tailor_second_order() const -> Tesor3_f16_compressed;
};

// For leaf nodes
// 64 byte
struct TailorCoefficientsF16 {
  Vec3_f16 zero_order;
  Mat3x3_f16 first_order;
  Tesor3_f16_compressed second_order;
  // 60 bytes
  CenterOfMass_quantized center_of_mass; // 4 bytes
};

// For m2m
// 120 byte
struct TailorCoefficients {
  Vec3 zero_order;
  Mat3x3 first_order;
  Tensor3_compressed second_order;

  __host__ __device__ static auto
  from_f16(const TailorCoefficientsF16 &t) -> TailorCoefficients {
    TailorCoefficients result;
    result.zero_order = Vec3::from_f16(t.zero_order);
    result.first_order = Mat3x3::from_f16(t.first_order);
    result.second_order = Tensor3_compressed::from_f16(t.second_order);
    return result;
  }
};

// only for cuda compiler:
#ifdef __CUDACC__
#include <cmath>
#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <sys/types.h>
#include <vector_types.h>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

#define STR_HELPER(x) #x
#define TO_STR(x) STR_HELPER(x)

#define PACK_CORE(SRC, OFFSET, LOW)                                            \
  "cvt.rni.s32.f32 i_val, " SRC ";\n\t"                                        \
  "and.b32 i_val, i_val, 0x7ff;\n\t"                                           \
  "shl.b32 i_tmp, i_val, " TO_STR(OFFSET) ";\n\t"                              \
                                          "or.b32  " LOW ", " LOW              \
                                          ", i_tmp;\n\t"

#define PACK_SPILL_0(OFFSET, HIGH) ""
#define PACK_SPILL_1(OFFSET, HIGH)                                             \
  "shr.u32 i_tmp, i_val, " TO_STR(32 - (OFFSET)) ";\n\t"                       \
                                                 "or.b32  " HIGH ", " HIGH     \
                                                 ", i_tmp;\n\t"

#define PACK_11(SRC, OFFSET, LOW, HIGH, SHOULD_SPILL)                          \
  PACK_CORE(SRC, OFFSET, LOW)                                                  \
  PACK_SPILL_##SHOULD_SPILL(OFFSET, HIGH)

__device__ auto
TailorCoefficientsQuantized::get_expected_children() const -> uint32_t {
  // use first uint32_t for parent idx
  return tailor_data[0];
}

__device__ void
TailorCoefficientsQuantized::set_expected_children(uint32_t idx) {
  // use first uint32_t for parent idx
  tailor_data[0] = idx;
}

__device__ inline auto
TailorCoefficientsQuantized::get_shared_scale_factor_low() const -> float {
  // low_exponent is stored in bits 24-31 (Byte 3)
  const uint32_t shared_exponent = (tailor_data[10] >> 24) & 0xFF;
  float shared_scale_factor = __int_as_float(((int)shared_exponent - 9) << 23);
  return shared_scale_factor;
}

__device__ inline auto
TailorCoefficientsQuantized::get_shared_scale_factor_second() const -> float {
  // second_exponent is stored in bits 16-23 (Byte 2)
  const uint32_t shared_exponent = (tailor_data[10] >> 16) & 0xFF;
  float shared_scale_factor = __int_as_float(((int)shared_exponent - 9) << 23);
  return shared_scale_factor;
}

__device__ inline void TailorCoefficientsQuantized::set_tailor_coefficients(
    const Vec3 &zero_order, const Mat3x3 &first_order,
    const Tensor3_compressed &second_order) {
  // compute shared_exponent for zero order and first order
  float max_abs_val_low = fmaxf(fabsf(zero_order.x), fabsf(zero_order.y));
  max_abs_val_low = fmaxf(max_abs_val_low, fabsf(zero_order.z));
  for (int i = 0; i < 9; ++i) {
    max_abs_val_low = fmaxf(max_abs_val_low, fabsf(first_order.data[i]));
  }
  uint32_t max_abs_bits_low = __float_as_uint(max_abs_val_low);
  uint8_t shared_exponent_low = (max_abs_bits_low >> 23) & 0xFF;
  shared_exponent_low = max(shared_exponent_low, 9);
  float shared_scale_factor_low =
      __int_as_float(((int)shared_exponent_low - 9) << 23);
  float inv_shared_scale_factor_low = 1.F / shared_scale_factor_low;

  // compute shared_exponent for second_order
  float max_abs_val_second = 0.F;
  for (int i = 0; i < 18; ++i) {
    max_abs_val_second = fmaxf(max_abs_val_second, fabsf(second_order.data[i]));
  }
  uint32_t max_abs_bits_second = __float_as_uint(max_abs_val_second);
  uint8_t shared_exponent_second = (max_abs_bits_second >> 23) & 0xFF;

  shared_exponent_second = max(shared_exponent_second, 9);
  float shared_scale_factor_second =
      __int_as_float(((int)shared_exponent_second - 9) << 23);
  float inv_shared_scale_factor_second = 1.F / shared_scale_factor_second;

  // pack the coefficients
  asm volatile(
      ".reg .b32 i_val, i_tmp;\n\t"
      // first output bytes
      "mov.b32 %0, 0x0;\n\t"  // set output 0 to 0
      "mov.b32 %1, 0x0;\n\t"  // set output 1 to 0
      "mov.b32 %2, 0x0;\n\t"  // set output 2 to 0
      "mov.b32 %3, 0x0;\n\t"  // set output 3 to 0
      "mov.b32 %4, 0x0;\n\t"  // set output 4 to 0
      "mov.b32 %5, 0x0;\n\t"  // set output 5 to 0
      "mov.b32 %6, 0x0;\n\t"  // set output 6 to 0
      "mov.b32 %7, 0x0;\n\t"  // set output 7 to 0
      "mov.b32 %8, 0x0;\n\t"  // set output 8 to 0
      "mov.b32 %9, 0x0;\n\t"  // set output 9 to 0
      "mov.b32 %10, 0x0;\n\t" // set output 10 to 0
      //
      PACK_11("%11", 0, "%0", "%1", 0)  // zero 0
      PACK_11("%12", 11, "%0", "%1", 0) // zero 1
      PACK_11("%13", 22, "%0", "%1", 1) // zero 2

      PACK_11("%14", 1, "%1", "%2", 0)  // first 0
      PACK_11("%15", 12, "%1", "%2", 0) // first 1
      PACK_11("%16", 23, "%1", "%2", 1) // first 2
      PACK_11("%17", 2, "%2", "%3", 0)  // first 3
      PACK_11("%18", 13, "%2", "%3", 0) // first 4
      PACK_11("%19", 24, "%2", "%3", 1) // first 5
      PACK_11("%20", 3, "%3", "%4", 0)  // first 6
      PACK_11("%21", 14, "%3", "%4", 0) // first 7
      PACK_11("%22", 25, "%3", "%4", 1) // first 8

      PACK_11("%23", 4, "%4", "%5", 0)   // second 0
      PACK_11("%24", 15, "%4", "%5", 0)  // second 1
      PACK_11("%25", 26, "%4", "%5", 1)  // second 2
      PACK_11("%26", 5, "%5", "%6", 0)   // second 3
      PACK_11("%27", 16, "%5", "%6", 0)  // second 4
      PACK_11("%28", 27, "%5", "%6", 1)  // second 5
      PACK_11("%29", 6, "%6", "%7", 0)   // second 6
      PACK_11("%30", 17, "%6", "%7", 0)  // second 7
      PACK_11("%31", 28, "%6", "%7", 1)  // second 8
      PACK_11("%32", 7, "%7", "%8", 0)   // second 9
      PACK_11("%33", 18, "%7", "%8", 0)  // second 10
      PACK_11("%34", 29, "%7", "%8", 1)  // second 11
      PACK_11("%35", 8, "%8", "%9", 0)   // second 12
      PACK_11("%36", 19, "%8", "%9", 0)  // second 13
      PACK_11("%37", 30, "%8", "%9", 1)  // second 14
      PACK_11("%38", 9, "%9", "%10", 0)  // second 15
      PACK_11("%39", 20, "%9", "%10", 0) // second 16
      PACK_11("%40", 31, "%9", "%10", 1) // second 17

      // outputs
      : "+r"(tailor_data[0]), // %0
        "+r"(tailor_data[1]), // %1
        "+r"(tailor_data[2]), // %2
        "+r"(tailor_data[3]), // %3
        "+r"(tailor_data[4]), // %4
        "+r"(tailor_data[5]), // %5
        "+r"(tailor_data[6]), // %6
        "+r"(tailor_data[7]), // %7
        "+r"(tailor_data[8]), // %8
        "+r"(tailor_data[9]), // %9
        "+r"(tailor_data[10]) // %10
      // inputs
      : "f"(inv_shared_scale_factor_low * zero_order.x),             // %11
        "f"(inv_shared_scale_factor_low * zero_order.y),             // %12
        "f"(inv_shared_scale_factor_low * zero_order.z),             // %13
        "f"(inv_shared_scale_factor_low * first_order.data[0]),      // %14
        "f"(inv_shared_scale_factor_low * first_order.data[1]),      // %15
        "f"(inv_shared_scale_factor_low * first_order.data[2]),      // %16
        "f"(inv_shared_scale_factor_low * first_order.data[3]),      // %17
        "f"(inv_shared_scale_factor_low * first_order.data[4]),      // %18
        "f"(inv_shared_scale_factor_low * first_order.data[5]),      // %19
        "f"(inv_shared_scale_factor_low * first_order.data[6]),      // %20
        "f"(inv_shared_scale_factor_low * first_order.data[7]),      // %21
        "f"(inv_shared_scale_factor_low * first_order.data[8]),      // %22
        "f"(inv_shared_scale_factor_second * second_order.data[0]),  // %23
        "f"(inv_shared_scale_factor_second * second_order.data[1]),  // %24
        "f"(inv_shared_scale_factor_second * second_order.data[2]),  // %25
        "f"(inv_shared_scale_factor_second * second_order.data[3]),  // %26
        "f"(inv_shared_scale_factor_second * second_order.data[4]),  // %27
        "f"(inv_shared_scale_factor_second * second_order.data[5]),  // %28
        "f"(inv_shared_scale_factor_second * second_order.data[6]),  // %29
        "f"(inv_shared_scale_factor_second * second_order.data[7]),  // %30
        "f"(inv_shared_scale_factor_second * second_order.data[8]),  // %31
        "f"(inv_shared_scale_factor_second * second_order.data[9]),  // %32
        "f"(inv_shared_scale_factor_second * second_order.data[10]), // %33
        "f"(inv_shared_scale_factor_second * second_order.data[11]), // %34
        "f"(inv_shared_scale_factor_second * second_order.data[12]), // %35
        "f"(inv_shared_scale_factor_second * second_order.data[13]), // %36
        "f"(inv_shared_scale_factor_second * second_order.data[14]), // %37
        "f"(inv_shared_scale_factor_second * second_order.data[15]), // %38
        "f"(inv_shared_scale_factor_second * second_order.data[16]), // %39
        "f"(inv_shared_scale_factor_second * second_order.data[17])  // %40
  );

  // Store the two exponents in the last two bytes of tailor_data[10]
  // Bits 0-9 contain the spilled bits from second_order[17].
  // Bits 10-15 are empty.
  // Pack second_exponent into bits 16-23 (Byte 2)
  // Pack low_exponent into bits 24-31 (Byte 3)
  tailor_data[10] |= (static_cast<uint32_t>(shared_exponent_second) << 16) | 
                     (static_cast<uint32_t>(shared_exponent_low) << 24);
}

__device__ __forceinline__ auto unpack_f16(uint32_t selector, uint32_t shift,
                                            uint32_t in_a, uint32_t in_b,
                                            float scale) -> half {
  uint16_t out_reg; // Correct 16-bit target for "=h"
  asm volatile("{\n\t"
               ".reg .b32 m_tmp;\n\t"
               ".reg .f32 f_tmp;\n\t"
               "prmt.b32 m_tmp, %1, %2, %3;\n\t"
               "shr.b32  m_tmp, m_tmp, %4;\n\t"
               "and.b32  m_tmp, m_tmp, 0x7ff;\n\t"
               "shl.b32  m_tmp, m_tmp, 21;\n\t"
               "shr.s32  m_tmp, m_tmp, 21;\n\t"
               "cvt.rn.f32.s32 f_tmp, m_tmp;\n\t"
               "mul.f32  f_tmp, f_tmp, %5;\n\t"
               "cvt.rn.f16.f32 %0, f_tmp;\n\t"
               "}"
               : "=h"(out_reg)
               : "r"(in_a), "r"(in_b), "r"(selector), "r"(shift), "f"(scale));
  return reinterpret_cast<half &>(out_reg);
}

__device__ inline auto
TailorCoefficientsQuantized::get_tailor_zero_order() const -> Vec3_f16 {
  const auto *d = reinterpret_cast<const uint32_t *>(tailor_data);
  float shared_scale_factor = get_shared_scale_factor_low();

  // [          raw1,                                 raw0               ]
  // [{stuff.stuff.stuff.-------z}, {zzzzzzzz.zzyyyyyy.yyyyyxxx.xxxxxxxx}]
  //    7      6    5        4            3       2      1          0
  Vec3_f16 result;
  result.x = unpack_f16(0x0010, 0, d[0], d[1], shared_scale_factor);
  result.y = unpack_f16(0x0021, 3, d[0], d[1], shared_scale_factor);
  result.z = unpack_f16(0x5432, 6, d[0], d[1], shared_scale_factor);

  return result;
}


__device__ inline auto TailorCoefficientsQuantized::get_tailor_first_order(
) const -> Mat3x3_f16 {
  const auto *d = reinterpret_cast<const uint32_t *>(tailor_data);
  float shared_scale_factor = get_shared_scale_factor_low();
  Mat3x3_f16 result;
  // tailor_data[1]
  // [22222222.21111111.11110000.0000000-]
  // tailor_data[2]
  // [55555555.44444444.44433333.33333322]
  // tailor_data[3]
  // [88888887.77777777.77666666.66666555]
  // tailor_data[4]
  // [--------.--------.--------.----8888]

  result.data[0] =
      unpack_f16(0x0010, 1, d[1], d[2], shared_scale_factor); // xx
  result.data[1] =
      unpack_f16(0x0021, 4, d[1], d[2], shared_scale_factor); // xy
  result.data[2] =
      unpack_f16(0x0432, 7, d[1], d[2], shared_scale_factor); // xz

  result.data[3] =
      unpack_f16(0x0010, 2, d[2], d[3], shared_scale_factor); // yx
  result.data[4] =
      unpack_f16(0x0021, 5, d[2], d[3], shared_scale_factor); // yy
  result.data[5] =
      unpack_f16(0x0043, 0, d[2], d[3], shared_scale_factor); // yz

  result.data[6] =
      unpack_f16(0x0054, 3, d[2], d[3], shared_scale_factor); // zx
  result.data[7] =
      unpack_f16(0x0765, 6, d[2], d[3], shared_scale_factor); // zy
  result.data[8] =
      unpack_f16(0x0043, 1, d[3], d[4], shared_scale_factor); // zz

  return result;
}

__device__ inline auto TailorCoefficientsQuantized::get_tailor_second_order(
) const -> Tensor3_f16_compressed {
  const auto *d = reinterpret_cast<const uint32_t *>(tailor_data);
  float shared_scale_factor = get_shared_scale_factor_second();

  Tensor3_f16_compressed result;

  // tailor_data[4]
  // [22222211.11111111.10000000.0000----]
  // tailor_data[5]
  // [55555444.44444444.33333333.33322222]
  // tailor_data[6]
  // [88887777.77777776.66666666.66555555]
  // tailor_data[7]
  // [11100000.00000099.99999999.98888888]
  // tailor_data[8]
  // [44333333.33333222.22222222.11111111]
  // tailor_data[9]
  // [76666666.66665555.55555554.44444444]
  // tailor_data[10]
  // [eeeeeeee.--------.------77.77777777]
  result.data[0] = unpack_f16(0x0010, 4, d[4], d[5], shared_scale_factor);
  result.data[1] = unpack_f16(0x0321, 7, d[4], d[5], shared_scale_factor);
  result.data[2] = unpack_f16(0x0043, 2, d[4], d[5], shared_scale_factor);
  result.data[3] = unpack_f16(0x0054, 5, d[4], d[5], shared_scale_factor);
  result.data[4] = unpack_f16(0x0076, 0, d[4], d[5], shared_scale_factor);

  result.data[5] = unpack_f16(0x0043, 3, d[5], d[6], shared_scale_factor);
  result.data[6] = unpack_f16(0x0654, 6, d[5], d[6], shared_scale_factor);
  result.data[7] = unpack_f16(0x0076, 1, d[5], d[6], shared_scale_factor);

  result.data[8] = unpack_f16(0x0043, 4, d[6], d[7], shared_scale_factor);

  result.data[9] = unpack_f16(0x0210, 7, d[7], d[8], shared_scale_factor);
  result.data[10] = unpack_f16(0x0032, 2, d[7], d[8], shared_scale_factor);
  result.data[11] = unpack_f16(0x0043, 5, d[7], d[8], shared_scale_factor);
  result.data[12] = unpack_f16(0x0065, 0, d[7], d[8], shared_scale_factor);
  result.data[13] = unpack_f16(0x0076, 3, d[7], d[8], shared_scale_factor);

  result.data[14] = unpack_f16(0x0543, 6, d[8], d[9], shared_scale_factor);
  result.data[15] = unpack_f16(0x0065, 1, d[8], d[9], shared_scale_factor);
  result.data[16] = unpack_f16(0x0076, 4, d[8], d[9], shared_scale_factor);

  result.data[17] = unpack_f16(0x0543, 7, d[9], d[10], shared_scale_factor);

  return result;
}

#endif
