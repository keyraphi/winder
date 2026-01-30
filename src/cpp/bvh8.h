#pragma once
#include "aabb.h"
#include "tailor_coefficients.h"
#include <cstdint>

// Leafs have an extra LeafPointer to the actual leaf indices
struct LeafPointers {
  uint32_t indices[8];
};

// Info what each of the childs are
enum class ChildType : uint8_t { EMPTY = 0, INTERNAL = 1, LEAF = 2 };

// A Node or the BVH8 tree
// Aligned to 128 byte cache lines
struct BVH8Node {
  AABB parent_aabb;                    // 29 bytes
  uint32_t child_base;                 // 4 bytes
  uint16_t _child_meta_raw;            // 2 bytes
  AABB8BitApprox child_aabb_approx[8]; // 48 bytes (6*8)
  // 83 bytes so far => 45 left for tailor coefficients

  // Quantized Tailor coefficients: 44 bytes
  TailorCoefficientsQuantized tailor_coefficients;


  // Helpers for ChildType
  __host__ __device__ __forceinline__ auto getChildMeta(const uint32_t child_idx) const
      -> ChildType {
    // Shift by child_idx * 2, mask the bottom 2 bits
    return static_cast<ChildType>((_child_meta_raw >> (child_idx * 2)) & 0x3);
  }
// Host/Device setter
  __host__ __device__ __forceinline__ auto setChildMeta(const uint32_t child_idx, const ChildType type) -> void {
    uint32_t clear_mask = 0x3 << (child_idx * 2);
    uint32_t value_mask = static_cast<uint32_t>(type) << (child_idx * 2);
    _child_meta_raw = static_cast<uint16_t>((_child_meta_raw & ~clear_mask) | value_mask);
  }
};
