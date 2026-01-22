#pragma once
#include "aabb.h"
#include "tailor_coefficients.h"
#include <cstdint>

// Info what each of the childs are
enum class ChildType : uint8_t { EMPTY = 0, INTERNAL = 1, LEAF = 2 };

// A Node or the BVH8 tree
// Aligned to 128 byte cache lines
struct BVH8Node {
  AABB parent_aabb;               // 24 bytes
  uint32_t child_base;            // 4 bytes
  ChildType child_meta[8];        // 8 bytes
  AABB8BitApprox child_approx[8]; // 48 bytes (6*8)
  // 84 bytes so far => 44 left for tailor coefficients

  // Quantized Tailor coefficients: 44 bytes
  TailorCoefficientsQuantized tailor_coefficients;
};
