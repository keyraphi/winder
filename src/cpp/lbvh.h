#pragma once
#include "aabb.h"
#include <cstdint>
#include <cuda_runtime_api.h>
#include <vector_types.h>

// 32 byte node
struct BinaryLBVHNode {
  AABB aabb;            // 24 bytes (Vec3 min, Vec3 max)
  uint32_t left_child;  // Index to left child. Right is left_child + 1.
  uint32_t range_start; // First point in Morton order

  // Helper to identify leaves during the collapse
  __host__ __device__ auto is_leaf() const -> bool {
    return left_child == 0xFFFFFFFF;
  }
  __host__ __device__ void set_leaf() { left_child = 0xFFFFFFFF; }
};
