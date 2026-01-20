#pragma once
#include "aabb.h"
#include "mat3x3.h"
#include "tensor3.h"
#include "vec3.h"
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
    // [88888877.77777777.76666666.66665555]
    // uint32_data[4] ~ tailor_data[16-19]
    // [--------.--------.--------.---88888]
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
        UNPACK_SCALE_STR("0x0054", "4", "%6", "%10", "%11", "%13") // zx
        UNPACK_SCALE_STR("0x0765", "7", "%7", "%10", "%11", "%13") // zy
        UNPACK_SCALE_STR("0x0043", "2", "%8", "%11", "%12", "%13") // zz
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
    // [22222111.11111111.00000000.000-----]
    // uint32_data[5] ~ tailor_data[20-23]
    // [55554444.44444443.33333333.33222222]
    // uint32_data[6] ~ tailor_data[24-27]
    // [88877777.77777766.66666666.65555555]
    // uint32_data[7] ~ tailor_data[28-31]
    // [11000000.00000999.99999999.88888888]
    // uint32_data[8] ~ tailor_data[32-35]
    // [43333333.33332222.22222221.11111111]
    // uint32_data[9] ~ tailor_data[36-39]
    // [66666666.66655555.55555544.44444444]
    // uint32_data[10] ~ tailor_data[40-44]
    // [eeeeeeee.--------.-----777.77777777]
    Tensor3_bf16_compressed result;

    asm volatile(
        "{\n\t"
        ".reg .s32 m_tmp;\n\t"
        ".reg .f32 f_tmp;\n\t"

        UNPACK_SCALE_STR("0x0010", "5", "%0", "%18", "%19", "%25")  // 0
        UNPACK_SCALE_STR("0x0032", "0", "%1", "%18", "%19", "%25")  // 1
        UNPACK_SCALE_STR("0x0043", "3", "%2", "%18", "%19", "%25")  // 2
        UNPACK_SCALE_STR("0x0654", "6", "%3", "%18", "%19", "%25")  // 3
        UNPACK_SCALE_STR("0x0076", "1", "%4", "%18", "%19", "%25")  // 4
        UNPACK_SCALE_STR("0x0043", "4", "%5", "%19", "%20", "%25")  // 5
        UNPACK_SCALE_STR("0x0654", "7", "%6", "%19", "%20", "%25")  // 6
        UNPACK_SCALE_STR("0x0076", "2", "%7", "%19", "%20", "%25")  // 7
        UNPACK_SCALE_STR("0x0043", "5", "%8", "%20", "%21", "%25")  // 8
        UNPACK_SCALE_STR("0x0021", "0", "%9", "%21", "%22", "%25")  // 9
        UNPACK_SCALE_STR("0x0032", "3", "%10", "%21", "%22", "%25") // 0
        UNPACK_SCALE_STR("0x0543", "6", "%11", "%21", "%22", "%25") // 1
        UNPACK_SCALE_STR("0x0065", "1", "%12", "%21", "%22", "%25") // 2
        UNPACK_SCALE_STR("0x0076", "4", "%13", "%21", "%22", "%25") // 3
        UNPACK_SCALE_STR("0x0543", "7", "%14", "%22", "%23", "%25") // 4
        UNPACK_SCALE_STR("0x0076", "2", "%15", "%22", "%23", "%25") // 5
        UNPACK_SCALE_STR("0x0076", "5", "%16", "%22", "%23", "%25") // 6
        UNPACK_SCALE_STR("0x0054", "0", "%17", "%23", "%24", "%25") // 7
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
