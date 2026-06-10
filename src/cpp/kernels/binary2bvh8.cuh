#pragma once

#include "aabb.h"
#include "binary_node.h"
#include "bvh8.h"
#include "geometry.h"
#include "winder_cuda.h"
#include <cstdint>
#include <driver_types.h>

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
  uint32_t *bvh8_node_child_count;
  LeafPointers *bvh8_leaf_pointers;
  uint32_t *bvh8_node_count;
};

void convert_binary_tree_to_bvh8(ConvertBinary2BVH8Params params, int device_id,
                                 const cudaStream_t &stream = 0);

template <IsGeometry Geometry>
void compute_max_distances(BVH8Node *__restrict__ nodes, const float *geometry,
                           const uint32_t *__restrict__ leaf_parents,
                           const uint32_t *__restrict__ internal_parent_map,
                           uint32_t geometry_count, const uint32_t *bvh8_node_count,
                           const cudaStream_t &stream = 0);

extern template void compute_max_distances<PointNormal>(
    BVH8Node *__restrict__ nodes, const float *geometry,
    const uint32_t *__restrict__ leaf_parents,
    const uint32_t *__restrict__ internal_parent_map,
    const uint32_t geometry_count, const uint32_t *bvh8_node_count,
    const cudaStream_t &stream);
extern template void compute_max_distances<Triangle>(
    BVH8Node *__restrict__ nodes, const float *geometry,
    const uint32_t *__restrict__ leaf_parents,
    const uint32_t *__restrict__ internal_parent_map,
    const uint32_t geometry_count, const uint32_t *bvh8_node_count,
    const cudaStream_t &stream);
