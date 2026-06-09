#pragma once
#include "aabb.h"
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
  AABB parent_aabb;                    // 40 bytes
  uint32_t child_base;                 // 4 bytes
  ChildType child_meta[8];             // 8 bytes
  Vec3 zero_order_coefficients;       // 12 bytes
                                      // total: 64 bytes
};
