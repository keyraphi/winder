#pragma once

#include "aabb.h"
#include "binary_node.h"
#include "bvh8.h"
#include "winder_cuda.h"
#include <cstdint>

struct ConvertBinary2BVH8Params {
  uint32_t *work_queue_A;
  uint32_t *work_queue_B;
  uint32_t *bvh8_internal_parents;
  uint32_t *global_counter;
  const uint32_t leaf_count;
  const AABB *binary_aabbs;
  const BinaryNode *binary_nodes;
  uint32_t *bvh8_leaf_parents;
  BVH8Node *bvh8_nodes;
  LeafPointers *bvh8_leaf_pointers;
};

void convert_binary_tree_to_bvh8(ConvertBinary2BVH8Params params, int device_id);
