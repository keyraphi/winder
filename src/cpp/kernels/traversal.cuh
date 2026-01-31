#pragma once
#include "bvh8.h"
#include "tailor_coefficients.h"
#include "vec3.h"
#include <cstdint>
#include <driver_types.h>

template <typename Geometry> struct ComputeWindingNumbersParams {
  const Vec3 *queries;
  const uint32_t *sort_indirections;
  const BVH8Node *bvh8_nodes;
  const LeafPointers *bvh8_leaf_pointers;
  const TailorCoefficientsBf16 *leaf_coefficients;
  const Geometry *sorted_geometry;
  uint32_t query_count;
  uint32_t geometry_count;
  float *winding_numbers;
  float beta;
  float epsilon;
};

template <typename Geometry>
void compute_winding_numbers(
    const ComputeWindingNumbersParams<Geometry> &params, int device_id,
    const cudaStream_t &stream);
