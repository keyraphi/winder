#pragma once

#include "aabb.h"
#include "bvh8.h"
#include "geometry.h"
#include "scene_normalization.h"
#include "soa.h"
#include "taylor_coefficients.h"
#include "vec3.h"
#include <cstdint>

template <IsGeometry Geometry, uint32_t D>
struct ComputePayloadWindingNumbersParams {
  const Vec3 *queries;
  const uint32_t *sort_indirections;
  const BVH8Node *bvh8_nodes;
  const LeafPointers *bvh8_leaf_pointers;
  const PayloadTaylorCoefficientsF16<D> *node_coefficients;
  const PayloadTaylorCoefficientsF16<D> *leaf_coefficients;
  const AABB *leaf_aabbs;
  const SoAView<Geometry> *sorted_geometry;
  const float *sorted_payload;
  uint32_t query_count;
  uint32_t geometry_count;
  float *out_payload;
  uint32_t *global_device_counter;
  const float beta;
  const float epsilon;
  const float scene_scale;
  const SceneNormalization norm;
};
