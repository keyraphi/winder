#pragma once
#include "aabb.h"
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <vector_types.h>

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
  uint8_t tailor_data[44];
  // total 128 bytes

  // During construction we use the tailor_data memory for a parent pointer
  // which is used to spread the aabb and tailor coefficients to the inner nodes.
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
};
