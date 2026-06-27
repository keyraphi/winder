#pragma once
#include "aabb.h"
#include "vec3.h"
#include <__clang_cuda_runtime_wrapper.h>
#include <cuda_fp16.h>
#include <cstdint>

// Leafs have an extra LeafPointer to the actual leaf indices
struct LeafPointers {
  uint32_t indices[8];
};

// Info what each of the childs are
enum class ChildType : uint8_t { EMPTY = 0, INTERNAL = 1, LEAF = 2 };

struct alignas(32) BVH8Node {
  uint32_t child_base;      // 4 bytes (0 -> 4)
  Vec3_f16 _aabb_min;       // 6 bytes (4 -> 10)
  Vec3_f16 _aabb_max;       // 6 bytes (10 -> 16)
  Vec3 _aabb_com;          // 12 bytes (16 -> 28)
  half _aabb_max_dist;      // 2 bytes (28 -> 30)
  uint16_t _child_meta_raw; // 2 bytes (30 -> 32)

  __host__ __device__ __forceinline__ auto getAABB() const -> AABB {
    return AABB{
      .min=_aabb_min,
      .max=_aabb_max,
      .center_of_mass=_aabb_com,
      .max_distance=_aabb_max_dist
    };
  }

  __host__ __device__ __forceinline__ auto setAABB(const AABB &aabb) -> void {
    _aabb_min = aabb.min;
    _aabb_max = aabb.max;
    _aabb_com = aabb.center_of_mass;
    _aabb_max_dist = aabb.max_distance;
  }

  __host__ __device__ __forceinline__ auto
  getChildMeta(const uint32_t child_idx) const -> ChildType {
    return static_cast<ChildType>((_child_meta_raw >> (child_idx * 2)) & 0x3);
  }

  __host__ __device__ __forceinline__ auto
  setChildMeta(const uint32_t child_idx, const ChildType type) -> void {
    uint32_t clear_mask = 0x3U << (child_idx * 2);
    uint32_t value_mask = static_cast<uint32_t>(type) << (child_idx * 2);
    _child_meta_raw = (_child_meta_raw & ~clear_mask) | value_mask;
  }
};
