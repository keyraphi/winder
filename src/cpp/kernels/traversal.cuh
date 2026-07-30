#pragma once
#include "bvh8.h"
#include "geometry.h"
#include "soa.h"
#include "taylor_coefficients.h"
#include "vec3.h"
#include <cstdint>
#include <driver_types.h>

template <IsGeometry Geometry> struct ComputeWindingNumbersParams {
  const Vec3 *queries;
  const uint32_t *sort_indirections;
  const BVH8Node *bvh8_nodes;
  const LeafPointers *bvh8_leaf_pointers;
  const TaylorCoefficientsF16 *node_coefficients;
  const TaylorCoefficientsF16 *leaf_coefficients;
  const AABB *leaf_aabbs;
  const SoAView<Geometry> sorted_geometry;
  uint32_t query_count;
  uint32_t geometry_count;
  float *winding_numbers;
  uint32_t *global_device_counter;
  const float beta;
  const float epsilon;
};

template <IsGeometry Geometry>
void compute_winding_numbers(
    const ComputeWindingNumbersParams<Geometry> &params, int device_id,
    const cudaStream_t &stream);

struct ComputeGradientsPointNormalParams {
  const Vec3 *points;
  const Vec3 *normals;
  const uint32_t *sort_indirections;
  const BVH8Node *bvh8_nodes;
  const LeafPointers *bvh8_leaf_pointers;
  const BackwardTaylorCoefficientsF16 *node_coefficients;
  const BackwardTaylorCoefficientsF16 *leaf_coefficients;
  const AABB *leaf_aabbs;
  const SoAView<Vec3> sorted_queries;
  const float* sorted_grad_outputs;
  uint32_t query_count;
  uint32_t geometry_count;
  float *gradients;
  uint32_t *global_device_counter;
  const float beta;
  const float epsilon;
};

void compute_point_normal_gradients(
    const ComputeGradientsPointNormalParams &params, int device_id,
    const cudaStream_t &stream);

struct ComputeGradientsTriangleParams {
  const Triangle *triangles;
  const uint32_t *sort_indirections;
  const BVH8Node *bvh8_nodes;
  const LeafPointers *bvh8_leaf_pointers;
  const BackwardTaylorCoefficientsF16 *node_coefficients;
  const BackwardTaylorCoefficientsF16 *leaf_coefficients;
  const AABB *leaf_aabbs;
  const SoAView<Vec3> sorted_queries;
  const float* sorted_grad_outputs;
  uint32_t query_count;
  uint32_t geometry_count;
  float *gradients;
  uint32_t *global_device_counter;
  const float beta;
};

void compute_triangle_gradients(const ComputeGradientsTriangleParams &params,
                                int device_id, const cudaStream_t &stream);

extern template void compute_winding_numbers<PointNormal>(
    const ComputeWindingNumbersParams<PointNormal> &params, int device_id,
    const cudaStream_t &stream);
extern template void compute_winding_numbers<Triangle>(
    const ComputeWindingNumbersParams<Triangle> &params, int device_id,
    const cudaStream_t &stream);
