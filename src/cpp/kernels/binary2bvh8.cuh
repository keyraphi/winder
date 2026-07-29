#pragma once

#include "aabb.h"
#include "binary_node.h"
#include "bvh8.h"
#include "vec3.h"
#include "soa.h"
#include "geometry.h"
#include <concepts>
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
  uint32_t *nodes_child_count;
  uint32_t *bvh8_leaf_parents;
  BVH8Node *bvh8_nodes;
  LeafPointers *bvh8_leaf_pointers;
  uint32_t *bvh8_node_count;
};

void convert_binary_tree_to_bvh8(ConvertBinary2BVH8Params params, int device_id,
                                 const cudaStream_t &stream = 0);

// Concept for max distance computation
template <typename T>
concept HasMaxDistance = requires(T g, SoAView<T> gp, Vec3 p, uint32_t u) {
  { T::load(gp, u, u) } -> std::same_as<T>;
  { g.max_distance_to(p) } -> std::same_as<float>;
};

template <HasMaxDistance Geometry>
void compute_max_distances(BVH8Node *__restrict__ nodes, const float *geometry,
                           const uint32_t *__restrict__ leaf_parents,
                           const uint32_t *__restrict__ internal_parent_map,
                           float *__restrict__ tmp_max_distance,
                           uint32_t geometry_count, uint32_t bvh8_node_count,
                           const cudaStream_t &stream = 0);
