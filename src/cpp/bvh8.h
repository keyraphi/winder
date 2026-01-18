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

  __nv_bfloat16 first_order[3];  // 6 bytes
  __nv_bfloat16 second_order[9]; // 18 bytes
  __nv_bfloat16 third_order[10]; // 20 bytes
  // total 128 bytes


  // During construction we use the first_order memory for a parent pointer which is used to compute the tailor coefficients and then overwritten
  // Retrieve the 32-bit parent index by stitching two 16-bit chunks
  __host__ __device__ auto get_parent_idx() const -> uint32_t {
    // Treat the first two bfloat16s as raw uint16_t storage
    const auto *raw = reinterpret_cast<const uint16_t *>(first_order);
    auto low = static_cast<uint32_t>(raw[0]);
    auto high = static_cast<uint32_t>(raw[1]);
    return (high << 16) | low;
  }

  // Set the 32-bit parent index by splitting it into two 16-bit chunks
  __host__ __device__ void set_parent_idx(uint32_t idx) {
    auto *raw = reinterpret_cast<uint16_t *>(first_order);
    raw[0] = static_cast<uint16_t>(idx & 0xFFFF);         // LSB
    raw[1] = static_cast<uint16_t>((idx >> 16) & 0xFFFF); // MSB
  }
};
