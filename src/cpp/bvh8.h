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

// A Node of the BVH8 tree
// Hard-aligned to 128-byte cache lines
struct alignas(128) BVH8Node {
    AABB parent_aabb;                    // 28 bytes (0 -> 28)
    uint32_t child_base;                 // 4 bytes  (28 -> 32)
    uint32_t _child_meta_raw;            // 4 bytes  (32 -> 36) (only 2 used)
    AABB8BitApprox child_aabb_approx[8]; // 48 bytes (36 -> 84)
    
    // Quantized Tailor coefficients: 44 bytes (84 -> 128)
    TailorCoefficientsQuantized tailor_coefficients;

    __host__ __device__ __forceinline__ auto getChildMeta(const uint32_t child_idx) const
        -> ChildType {
        return static_cast<ChildType>((_child_meta_raw >> (child_idx * 2)) & 0x3);
    }

    __host__ __device__ __forceinline__ auto setChildMeta(const uint32_t child_idx, const ChildType type) -> void {
        uint32_t clear_mask = 0x3U << (child_idx * 2);
        uint32_t value_mask = static_cast<uint32_t>(type) << (child_idx * 2);
        _child_meta_raw = (_child_meta_raw & ~clear_mask) | value_mask;
    }
};
