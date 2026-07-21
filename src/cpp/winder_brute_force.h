#pragma once

#include "utils.h"
#include <cstddef>
#include <cstdint>

auto brute_force_point_normal_impl(const float *points,
                                   const float *scaled_normals,
                                   const float *queries, size_t geometry_count,
                                   size_t query_count, float epsilon,
                                   int device_id, uint64_t stream)
    -> CudaUniquePtr<float>;

auto brute_force_mesh_impl(const float *vertices,
                           const uint32_t *triangle_indices,
                           const float *queries, size_t geometry_count,
                           size_t vertex_count, size_t query_count,
                           int device_id, uint64_t stream)
    -> CudaUniquePtr<float>;

auto brute_force_triangle_impl(const float *triangles, const float *queries,
                               size_t geometry_count, size_t query_count,
                               int device_id, uint64_t stream)
    -> CudaUniquePtr<float>;
