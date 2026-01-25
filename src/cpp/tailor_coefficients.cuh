#pragma once

#include "mat3x3.h"
#include "tailor_coefficients.h"
#include "tensor3.h"
#include "vec3.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <sys/types.h>
#include <vector_types.h>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
// Native Ampere/Hopper conversion
#define BF16_CONVERT_STR(OUT_IDX) "cvt.rn.bf16.f32 " OUT_IDX ", f_tmp;\n\t"
#else
// Emulated Turing/Pascal conversion
#define BF16_CONVERT_STR(OUT_IDX)                                              \
  "mov.b32 m_tmp, f_tmp;\n\t"                                                  \
  "add.u32 m_tmp, m_tmp, 0x8000;\n\t"                                          \
  "shr.u32 m_tmp, m_tmp, 16;\n\t"                                              \
  "cvt.u16.u32 " OUT_IDX ", m_tmp;\n\t"
#endif

#define UNPACK_SCALE_STR(SELECTOR, SHIFT, OUT, IN_A, IN_B, SCALE)              \
  "prmt.b32 m_tmp, " IN_A ", " IN_B ", " SELECTOR ";\n\t"                      \
  "shr.b32  m_tmp, m_tmp, " SHIFT ";\n\t"                                      \
  "and.b32  m_tmp, m_tmp, 0x7ff;\n\t"                                          \
  "shl.b32  m_tmp, m_tmp, 21;\n\t"                                             \
  "shr.s32  m_tmp, m_tmp, 21;\n\t"                                             \
  "cvt.rn.f32.s32 f_tmp, m_tmp;\n\t"                                           \
  "mul.f32  f_tmp, f_tmp, " SCALE ";\n\t" BF16_CONVERT_STR(OUT)

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

__device__ auto TailorCoefficientsQuantized::get_expected_children() const
    -> uint32_t {
  // use first uint32_t for parent idx
  return tailor_data[0];
}

__device__ void TailorCoefficientsQuantized::set_expected_children(uint32_t idx) {
  // use first uint32_t for parent idx
  tailor_data[0] = idx;
}

__device__ inline auto
TailorCoefficientsQuantized::get_shared_scale_factor() const -> float {
  const uint8_t *last_word =
      reinterpret_cast<const uint8_t *>(&tailor_data[10]);
  const uint32_t shared_exponent = last_word[3];
  float shared_scale_factor = __int_as_float(((int)shared_exponent - 9) << 23);
  return shared_scale_factor;
}

__device__ inline void TailorCoefficientsQuantized::set_tailor_coefficients(
    const Vec3 &zero_order, const Mat3x3 &first_order,
    const Tensor3_compressed &second_order) {
  // compute shared_exponent
  float max_abs_val = fmaxf(fabsf(zero_order.x), fabsf(zero_order.y));
  max_abs_val = fmaxf(max_abs_val, fabsf(zero_order.z));
  for (int i = 0; i < 9; ++i) {
    max_abs_val = fmaxf(max_abs_val, fabsf(first_order.data[i]));
  }
  for (int i = 0; i < 18; ++i) {
    max_abs_val = fmaxf(max_abs_val, fabsf(second_order.data[i]));
  }
  uint32_t max_abs_bits = __float_as_uint(max_abs_val);
  uint8_t shared_exponent = (max_abs_bits >> 23) & 0xFF;

  shared_exponent = max(shared_exponent, 9);
  float shared_scale_factor = __int_as_float(((int)shared_exponent - 9) << 23);
  float inv_shared_scale_factor = 1.F / shared_scale_factor;

  // pack the coefficients
  asm volatile(".reg .b32 i_val, i_tmp;\n\t"
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
               : "f"(inv_shared_scale_factor * zero_order.x),          // %11
                 "f"(inv_shared_scale_factor * zero_order.y),          // %12
                 "f"(inv_shared_scale_factor * zero_order.z),          // %13
                 "f"(inv_shared_scale_factor * first_order.data[0]),   // %14
                 "f"(inv_shared_scale_factor * first_order.data[1]),   // %15
                 "f"(inv_shared_scale_factor * first_order.data[2]),   // %16
                 "f"(inv_shared_scale_factor * first_order.data[3]),   // %17
                 "f"(inv_shared_scale_factor * first_order.data[4]),   // %18
                 "f"(inv_shared_scale_factor * first_order.data[5]),   // %19
                 "f"(inv_shared_scale_factor * first_order.data[6]),   // %20
                 "f"(inv_shared_scale_factor * first_order.data[7]),   // %21
                 "f"(inv_shared_scale_factor * first_order.data[8]),   // %22
                 "f"(inv_shared_scale_factor * second_order.data[0]),  // %23
                 "f"(inv_shared_scale_factor * second_order.data[1]),  // %24
                 "f"(inv_shared_scale_factor * second_order.data[2]),  // %25
                 "f"(inv_shared_scale_factor * second_order.data[3]),  // %26
                 "f"(inv_shared_scale_factor * second_order.data[4]),  // %27
                 "f"(inv_shared_scale_factor * second_order.data[5]),  // %28
                 "f"(inv_shared_scale_factor * second_order.data[6]),  // %29
                 "f"(inv_shared_scale_factor * second_order.data[7]),  // %30
                 "f"(inv_shared_scale_factor * second_order.data[8]),  // %31
                 "f"(inv_shared_scale_factor * second_order.data[9]),  // %32
                 "f"(inv_shared_scale_factor * second_order.data[10]), // %33
                 "f"(inv_shared_scale_factor * second_order.data[11]), // %34
                 "f"(inv_shared_scale_factor * second_order.data[12]), // %35
                 "f"(inv_shared_scale_factor * second_order.data[13]), // %36
                 "f"(inv_shared_scale_factor * second_order.data[14]), // %37
                 "f"(inv_shared_scale_factor * second_order.data[15]), // %38
                 "f"(inv_shared_scale_factor * second_order.data[16]), // %39
                 "f"(inv_shared_scale_factor * second_order.data[17])  // %40
  );

  auto *last_word = reinterpret_cast<uint8_t *>(&tailor_data[10]);
  last_word[3] = shared_exponent;
  printf("packing done\n");
}

__device__ inline auto TailorCoefficientsQuantized::get_tailor_zero_order(
    const float shared_scale_factor) const -> Vec3_bf16 {
  const auto *uint32_data = reinterpret_cast<const uint32_t *>(
      tailor_data); // works because tailor_data is b-byte aligned
  const uint32_t raw_0 = uint32_data[0];
  const uint32_t raw_1 = uint32_data[1];

  Vec3_bf16 result;

  // [          raw1,                                 raw0               ]
  // [{stuff.stuff.stuff.-------z}, {zzzzzzzz.zzyyyyyy.yyyyyxxx.xxxxxxxx}]
  //    7      6    5        4            3       2      1          0
  asm volatile(
      // unpack and convert to bfloat16s
      "{\n\t"
      // temporary registers
      ".reg .s32 m_tmp;\n\t"
      ".reg .f32 f_tmp;\n\t"

      UNPACK_SCALE_STR("0x0010", "0", "%0", "%3", "%4", "%5") // x
      UNPACK_SCALE_STR("0x0021", "3", "%1", "%3", "%4", "%5") // y
      UNPACK_SCALE_STR("0x5432", "6", "%2", "%3", "%4", "%5") // z
      "}"

      : "=h"(reinterpret_cast<unsigned short &>(result.x)), //%0
        "=h"(reinterpret_cast<unsigned short &>(result.y)), //%1
        "=h"(reinterpret_cast<unsigned short &>(result.z))  //%2
      : "r"(raw_0), "r"(raw_1),
        "f"(shared_scale_factor) // %3 %4 and %5 (Inputs)
  );

  return result;
}

__device__ inline auto TailorCoefficientsQuantized::get_tailor_first_order(
    const float shared_scale_factor) const -> Mat3x3_bf16 {
  // tailor_data[1]
  // [22222222.21111111.11110000.0000000-]
  // tailor_data[2]
  // [55555555.44444444.44433333.33333322]
  // tailor_data[3]
  // [88888887.77777777.77666666.66666555]
  // tailor_data[4]
  // [--------.--------.--------.----8888]
  Mat3x3_bf16 result;
  asm volatile("{\n\t"
               ".reg .s32 m_tmp;\n\t"
               ".reg .f32 f_tmp;\n\t"

               UNPACK_SCALE_STR("0x0010", "1", "%0", "%9", "%10", "%13")  // xx
               UNPACK_SCALE_STR("0x0021", "4", "%1", "%9", "%10", "%13")  // xy
               UNPACK_SCALE_STR("0x0432", "7", "%2", "%9", "%10", "%13")  // xz
               UNPACK_SCALE_STR("0x0010", "2", "%3", "%10", "%11", "%13") // yx
               UNPACK_SCALE_STR("0x0021", "5", "%4", "%10", "%11", "%13") // yy
               UNPACK_SCALE_STR("0x0043", "0", "%5", "%10", "%11", "%13") // yz
               UNPACK_SCALE_STR("0x0054", "3", "%6", "%10", "%11", "%13") // zx
               UNPACK_SCALE_STR("0x0765", "6", "%7", "%10", "%11", "%13") // zy
               UNPACK_SCALE_STR("0x0043", "1", "%8", "%11", "%12", "%13") // zz
               "}"

               // ouputs
               : "=h"(reinterpret_cast<unsigned short &>(result.data[0])), // %0
                 "=h"(reinterpret_cast<unsigned short &>(result.data[1])), // %1
                 "=h"(reinterpret_cast<unsigned short &>(result.data[2])), // %2
                 "=h"(reinterpret_cast<unsigned short &>(result.data[3])), // %3
                 "=h"(reinterpret_cast<unsigned short &>(result.data[4])), // %4
                 "=h"(reinterpret_cast<unsigned short &>(result.data[5])), // %5
                 "=h"(reinterpret_cast<unsigned short &>(result.data[6])), // %6
                 "=h"(reinterpret_cast<unsigned short &>(result.data[7])), // %7
                 "=h"(reinterpret_cast<unsigned short &>(result.data[8]))  // %8
               // inputs
               : "r"(tailor_data[1]),     // %9
                 "r"(tailor_data[2]),     // %10
                 "r"(tailor_data[3]),     // %11
                 "r"(tailor_data[4]),     // %12
                 "f"(shared_scale_factor) // %13
  );
  return result;
}

__device__ inline auto TailorCoefficientsQuantized::get_tailor_second_order(
    const float shared_scale_factor) const -> Tensor3_bf16_compressed {
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
  Tensor3_bf16_compressed result;

  asm volatile(
      "{\n\t"
      ".reg .s32 m_tmp;\n\t"
      ".reg .f32 f_tmp;\n\t"

      UNPACK_SCALE_STR("0x0010", "4", "%0", "%18", "%19", "%25")  // 0
      UNPACK_SCALE_STR("0x0321", "7", "%1", "%18", "%19", "%25")  // 1
      UNPACK_SCALE_STR("0x0043", "2", "%2", "%18", "%19", "%25")  // 2
      UNPACK_SCALE_STR("0x0054", "5", "%3", "%18", "%19", "%25")  // 3
      UNPACK_SCALE_STR("0x0076", "0", "%4", "%18", "%19", "%25")  // 4
      UNPACK_SCALE_STR("0x0043", "3", "%5", "%19", "%20", "%25")  // 5
      UNPACK_SCALE_STR("0x0654", "6", "%6", "%19", "%20", "%25")  // 6
      UNPACK_SCALE_STR("0x0076", "1", "%7", "%19", "%20", "%25")  // 7
      UNPACK_SCALE_STR("0x0043", "4", "%8", "%20", "%21", "%25")  // 8
      UNPACK_SCALE_STR("0x0210", "7", "%9", "%21", "%22", "%25")  // 9
      UNPACK_SCALE_STR("0x0032", "2", "%10", "%21", "%22", "%25") // 10
      UNPACK_SCALE_STR("0x0043", "5", "%11", "%21", "%22", "%25") // 11
      UNPACK_SCALE_STR("0x0065", "0", "%12", "%21", "%22", "%25") // 12
      UNPACK_SCALE_STR("0x0076", "3", "%13", "%21", "%22", "%25") // 13
      UNPACK_SCALE_STR("0x0543", "6", "%14", "%22", "%23", "%25") // 14
      UNPACK_SCALE_STR("0x0065", "1", "%15", "%22", "%23", "%25") // 15
      UNPACK_SCALE_STR("0x0076", "4", "%16", "%22", "%23", "%25") // 16
      UNPACK_SCALE_STR("0x0543", "7", "%17", "%23", "%24", "%25") // 17
      "}"
      // ouputs
      : "=h"(reinterpret_cast<unsigned short &>(result.data[0])),  // %0
        "=h"(reinterpret_cast<unsigned short &>(result.data[1])),  // %1
        "=h"(reinterpret_cast<unsigned short &>(result.data[2])),  // %2
        "=h"(reinterpret_cast<unsigned short &>(result.data[3])),  // %3
        "=h"(reinterpret_cast<unsigned short &>(result.data[4])),  // %4
        "=h"(reinterpret_cast<unsigned short &>(result.data[5])),  // %5
        "=h"(reinterpret_cast<unsigned short &>(result.data[6])),  // %6
        "=h"(reinterpret_cast<unsigned short &>(result.data[7])),  // %7
        "=h"(reinterpret_cast<unsigned short &>(result.data[8])),  // %8
        "=h"(reinterpret_cast<unsigned short &>(result.data[9])),  // %9
        "=h"(reinterpret_cast<unsigned short &>(result.data[10])), // %10
        "=h"(reinterpret_cast<unsigned short &>(result.data[11])), // %11
        "=h"(reinterpret_cast<unsigned short &>(result.data[12])), // %12
        "=h"(reinterpret_cast<unsigned short &>(result.data[13])), // %13
        "=h"(reinterpret_cast<unsigned short &>(result.data[14])), // %14
        "=h"(reinterpret_cast<unsigned short &>(result.data[15])), // %15
        "=h"(reinterpret_cast<unsigned short &>(result.data[16])), // %16
        "=h"(reinterpret_cast<unsigned short &>(result.data[17]))  // %17
      // inputs
      : "r"(tailor_data[4]),     // %18
        "r"(tailor_data[5]),     // %19
        "r"(tailor_data[6]),     // %20
        "r"(tailor_data[7]),     // %21
        "r"(tailor_data[8]),     // %22
        "r"(tailor_data[9]),     // %23
        "r"(tailor_data[10]),    // %24
        "f"(shared_scale_factor) // %25
  );
  return result;
}
