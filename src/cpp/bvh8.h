#pragma once
#include "aabb.h"
#include "mat3x3.h"
#include "tensor3.h"
#include "vec3.h"
#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
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

// Info what each of the childs are
enum class ChildType : uint8_t { EMPTY = 0, INTERNAL = 1, LEAF = 2 };

// If a node is a leafe it points to a range of points&normals or triangles
struct LeafInfo {
  uint32_t range_start;
  uint32_t range_end;
};

// A Node or the BVH8 tree
// Aligned to 128 byte cache lines
struct BVH8Node {
  AABB parent_aabb;               // 24 bytes
  uint32_t child_base;            // 4 bytes
  ChildType child_meta[8];        // 8 bytes
  AABB8BitApprox child_approx[8]; // 48 bytes (6*8)
  // 84 bytes so far => 44 left for tailor coefficients

  // tailor coefficients quantized to 11 bit
  // with a shared 8 bit exponent
  // first 8 bytes are reused for parent index
  // cofficients: zero order (3), first order (9), second order compressed (18)
  // total: 30
  uint8_t tailor_data[44];
  // total 128 bytes

  // During construction we use the tailor_data memory for a parent pointer
  // which is used to spread the aabb and tailor coefficients to the inner
  // nodes.
  __host__ __device__ auto get_parent_idx() const -> uint32_t {
    // Treat the first two bfloat16s as raw uint16_t storage
    const auto *uint32_data = reinterpret_cast<const uint32_t *>(tailor_data);
    return uint32_data[0];
  }

  // Set the parent idx. Reuses tailor coefficient data. Only call this before
  // the tailor coefficients and aabb are computed for the nodes.
  __host__ __device__ void set_parent_idx(uint32_t idx) {
    auto *uint32_data = reinterpret_cast<uint32_t *>(tailor_data);
    uint32_data[0] = idx;
  }

  __host__ __device__ inline auto get_shared_scale_factor() const -> float {
    uint32_t shared_exponent = tailor_data[43];
    float scale_factor;
    asm volatile(
        // compute shared scale factor
        // scale = (shared_exponent + 127 - 10) << 23;
        ".reg .u32 exp_bits;\n\t"
        "add.u32 exp_bits, %1, 117;\n\t"
        "shl.b32 exp_bits, exp_bits, 23;\n\t"
        "mov.b32 %0, exp_bits;\n\t"
        : "=f"(scale_factor)   // %0
        : "r"(shared_exponent) // %1
    );
    return scale_factor;
  }

  __device__ inline auto
  set_tailor_coefficients(const Vec3 &zero_order, const Mat3x3 &first_order,
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

    uint32_t shared_scale_factor_bits = ((uint32_t)shared_exponent + 117) << 23;
    float shared_scale_factor =
        reinterpret_cast<float &>(shared_scale_factor_bits);
    float inv_shared_scale_factor = 1.F / shared_scale_factor;

    // pack the coefficients
    auto *uint32_data = reinterpret_cast<uint32_t *>(
        tailor_data); // works because tailor_data is b-byte aligned

    asm volatile(".reg .b32 i_val, i_tmp;\n\t"
                 // first output bytes
                 "mov.b32 %0, 0x0;\n\t"  // set output to 0
                 "mov.b32 %1, 0x0;\n\t"  // set output to 0
                 "mov.b32 %2, 0x0;\n\t"  // set output to 0
                 "mov.b32 %3, 0x0;\n\t"  // set output to 0
                 "mov.b32 %4, 0x0;\n\t"  // set output to 0
                 "mov.b32 %5, 0x0;\n\t"  // set output to 0
                 "mov.b32 %6, 0x0;\n\t"  // set output to 0
                 "mov.b32 %7, 0x0;\n\t"  // set output to 0
                 "mov.b32 %8, 0x0;\n\t"  // set output to 0
                 "mov.b32 %9, 0x0;\n\t"  // set output to 0
                 "mov.b32 %10, 0x0;\n\t" // set output to 0
                 //
                 PACK_11("%11", 0, "%0", "%1", 0)   //
                 PACK_11("%12", 11, "%0", "%1", 0)  //
                 PACK_11("%13", 22, "%0", "%1", 1)  //
                 PACK_11("%14", 1, "%1", "%2", 0)   //
                 PACK_11("%15", 12, "%1", "%2", 0)  //
                 PACK_11("%16", 23, "%1", "%2", 1)  //
                 PACK_11("%17", 2, "%2", "%3", 0)   //
                 PACK_11("%18", 13, "%2", "%3", 0)  //
                 PACK_11("%19", 24, "%2", "%3", 1)  //
                 PACK_11("%20", 3, "%3", "%4", 0)   //
                 PACK_11("%21", 14, "%3", "%4", 0)  //
                 PACK_11("%22", 25, "%3", "%4", 1)  //
                 PACK_11("%23", 4, "%4", "%5", 0)   //
                 PACK_11("%24", 15, "%4", "%5", 0)  //
                 PACK_11("%25", 26, "%4", "%5", 1)  //
                 PACK_11("%26", 5, "%5", "%6", 0)   //
                 PACK_11("%27", 16, "%5", "%6", 0)  //
                 PACK_11("%28", 27, "%5", "%6", 1)  //
                 PACK_11("%29", 6, "%6", "%7", 0)   //
                 PACK_11("%30", 17, "%6", "%7", 0)  //
                 PACK_11("%31", 28, "%6", "%7", 1)  //
                 PACK_11("%32", 7, "%7", "%8", 0)   //
                 PACK_11("%33", 18, "%7", "%8", 0)  //
                 PACK_11("%34", 29, "%7", "%8", 1)  //
                 PACK_11("%35", 8, "%8", "%9", 0)   //
                 PACK_11("%36", 19, "%8", "%9", 0)  //
                 PACK_11("%37", 30, "%8", "%9", 1)  //
                 PACK_11("%38", 9, "%9", "%10", 0)  //
                 PACK_11("%39", 20, "%9", "%10", 0) //
                 PACK_11("%40", 31, "%9", "%10", 1) //

                 // outputs
                 : "+r"(uint32_data[0]), // %0
                   "+r"(uint32_data[1]), // %1
                   "+r"(uint32_data[2]), // %2
                   "+r"(uint32_data[3]), // %3
                   "+r"(uint32_data[4]), // %4
                   "+r"(uint32_data[5]), // %5
                   "+r"(uint32_data[6]), // %6
                   "+r"(uint32_data[7]), // %7
                   "+r"(uint32_data[8]), // %8
                   "+r"(uint32_data[9]), // %9
                   "+r"(uint32_data[10]) // %10
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

    tailor_data[43] = shared_exponent;
  }

  __host__ __device__ inline auto
  get_tailor_zero_order(const float shared_scale_factor) const -> Vec3_bf16 {
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

  __host__ __device__ auto
  get_tailor_first_order(const float shared_scale_factor) const -> Mat3x3_bf16 {
    const auto *uint32_data = reinterpret_cast<const uint32_t *>(tailor_data);
    // uint32_data[1] ~ tailor_data[4-7]
    // [22222222.21111111.11110000.0000000-]
    // uint32_data[2] ~ tailor_data[8-11]
    // [55555555.44444444.44433333.33333322]
    // uint32_data[3] ~ tailor_data[12-15]
    // [88888887.77777777.77666666.66666555]
    // uint32_data[4] ~ tailor_data[16-19]
    // [--------.--------.--------.----8888]
    Mat3x3_bf16 result;
    asm volatile(
        "{\n\t"
        ".reg .s32 m_tmp;\n\t"
        ".reg .f32 f_tmp;\n\t"

        UNPACK_SCALE_STR("0x0010", "1", "%0", "%9", "%10", "%13")  // xx
        UNPACK_SCALE_STR("0x0021", "4", "%1", "%9", "%10", "%13")  // xy
        UNPACK_SCALE_STR("0x0432", "7", "%2", "%9", "%10", "%13")  // xz
        UNPACK_SCALE_STR("0x0010", "2", "%3", "%10", "%11", "%13") // yx
        UNPACK_SCALE_STR("0x0021", "5", "%4", "%10", "%11", "%13") // yy
        UNPACK_SCALE_STR("0x0043", "0", "%5", "%10", "%11", "%13") // yz
        UNPACK_SCALE_STR("0x0054", "3", "%6", "%10", "%11", "%13") // zx
        UNPACK_SCALE_STR("0x0765", "7", "%7", "%10", "%11", "%13") // zy
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
        : "r"(uint32_data[1]),     // %9
          "r"(uint32_data[2]),     // %10
          "r"(uint32_data[3]),     // %11
          "r"(uint32_data[4]),     // %12
          "f"(shared_scale_factor) // %13
    );
    return result;
  }

  __host__ __device__ auto
  get_tailor_second_order(const float shared_scale_factor) const
      -> Tensor3_bf16_compressed {
    const auto *uint32_data = reinterpret_cast<const uint32_t *>(tailor_data);
    // uint32_data[4] ~ tailor_data[16-19]
    // [22222211.11111111.10000000.0000----]
    // uint32_data[5] ~ tailor_data[20-23]
    // [55555444.44444444.33333333.33322222]
    // uint32_data[6] ~ tailor_data[24-27]
    // [88887777.77777776.66666666.66555555]
    // uint32_data[7] ~ tailor_data[28-31]
    // [11100000.00000099.99999999.98888888]
    // uint32_data[8] ~ tailor_data[32-35]
    // [44333333.33333222.22222222.11111111]
    // uint32_data[9] ~ tailor_data[36-39]
    // [76666666.66665555.55555554.44444444]
    // uint32_data[10] ~ tailor_data[40-44]
    // [eeeeeeee.--------.------77.77777777]
    Tensor3_bf16_compressed result;

    asm volatile(
        "{\n\t"
        ".reg .s32 m_tmp;\n\t"
        ".reg .f32 f_tmp;\n\t"

        UNPACK_SCALE_STR("0x0010", "4", "%0", "%18", "%19", "%25")  // 0
        UNPACK_SCALE_STR("0x0032", "7", "%1", "%18", "%19", "%25")  // 1
        UNPACK_SCALE_STR("0x0043", "2", "%2", "%18", "%19", "%25")  // 2
        UNPACK_SCALE_STR("0x0054", "5", "%3", "%18", "%19", "%25")  // 3
        UNPACK_SCALE_STR("0x0076", "0", "%4", "%18", "%19", "%25")  // 4
        UNPACK_SCALE_STR("0x0043", "3", "%5", "%19", "%20", "%25")  // 5
        UNPACK_SCALE_STR("0x0654", "6", "%6", "%19", "%20", "%25")  // 6
        UNPACK_SCALE_STR("0x0076", "1", "%7", "%19", "%20", "%25")  // 7
        UNPACK_SCALE_STR("0x0043", "4", "%8", "%20", "%21", "%25")  // 8
        UNPACK_SCALE_STR("0x0210", "7", "%9", "%21", "%22", "%25")  // 9
        UNPACK_SCALE_STR("0x0032", "2", "%10", "%21", "%22", "%25") // 0
        UNPACK_SCALE_STR("0x0043", "5", "%11", "%21", "%22", "%25") // 1
        UNPACK_SCALE_STR("0x0065", "0", "%12", "%21", "%22", "%25") // 2
        UNPACK_SCALE_STR("0x0076", "3", "%13", "%21", "%22", "%25") // 3
        UNPACK_SCALE_STR("0x0543", "6", "%14", "%22", "%23", "%25") // 4
        UNPACK_SCALE_STR("0x0076", "1", "%15", "%22", "%23", "%25") // 5
        UNPACK_SCALE_STR("0x0076", "4", "%16", "%22", "%23", "%25") // 6
        UNPACK_SCALE_STR("0x0543", "7", "%17", "%23", "%24", "%25") // 7
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
        : "r"(uint32_data[4]),     // %18
          "r"(uint32_data[5]),     // %19
          "r"(uint32_data[6]),     // %20
          "r"(uint32_data[7]),     // %21
          "r"(uint32_data[8]),     // %22
          "r"(uint32_data[9]),     // %23
          "r"(uint32_data[10]),    // %24
          "f"(shared_scale_factor) // %25
    );
    return result;
  }
};
