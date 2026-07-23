#pragma once

#include "aabb.h"
#include "bvh8.h"
#include "utils.h"
#include <cstddef>
#include <cstdint>
class GradientBackend {
public:
  GradientBackend(const float *queries, size_t query_count, int device_id);
  ~GradientBackend();

  auto compute(const float *grad_output, const float *points,
               const float *scaled_normals, size_t grad_count,
               size_t geometry_count, float beta = -1, float epsilon = -1,
               uint64_t stream = 0) -> CudaUniquePtr<float>;
  auto compute(const float *grad_output, const float *vertices,
               const uint32_t *triangle_indices, size_t grad_count,
               size_t vertex_count, size_t geometry_count, float beta = -1,
               uint64_t stream = 0) -> CudaUniquePtr<float>;
  auto compute(const float *grad_output, const float *triangles,
               size_t grad_count, size_t geometry_count, float beta = -1,
               uint64_t stream = 0) -> CudaUniquePtr<float>;

private:
  const int m_device;
  const size_t m_query_count;

  uint32_t *m_to_internal;
  float *m_sorted_queries;

  AABB *m_binary_aabbs;
  DualBVH8Node *m_bvh8_nodes;
  // TODO Tailor coefficients
  LeafPointers *m_bvh8_leaf_pointers;
  
};
