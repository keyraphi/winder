#pragma once
#include <cstdint>
#include <cuda_runtime_api.h>
#include <vector_types.h>

// 8 byte node
struct BinaryNode {
  uint32_t left_child;
  uint32_t right_child;

  __host__ __device__ static auto is_leaf(uint32_t idx, uint32_t leaf_count)
      -> bool {
    return idx >= (leaf_count - 1);
  }
};
