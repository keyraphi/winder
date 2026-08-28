#pragma once

#include <cstddef>
#include <cstdint>

auto brute_force_point_normal_impl(const float *points,
                                   const float *scaled_normals,
                                   const float *queries, size_t geometry_count,
                                   size_t query_count, float *out_winding_numbers,
                                   float epsilon, int device_id,
                                   uint64_t stream = 0) -> void;

auto brute_force_mesh_impl(const float *vertices,
                           const uint32_t *triangle_indices,
                           const float *queries, size_t geometry_count,
                           size_t vertex_count, size_t query_count,
                           float *out_winding_numbers, int device_id,
                           uint64_t stream = 0) -> void;

auto brute_force_triangle_impl(const float *triangles, const float *queries,
                               size_t geometry_count, size_t query_count,
                               float *out_winding_numbers, int device_id,
                               uint64_t stream = 0) -> void;

auto brute_force_point_normal_gradient_impl(
    const float *grad_output, const float *points, const float *scaled_normals,
    const float *queries, size_t geometry_count, size_t query_count,
    float *output_gradients, float epsilon, int device_id, uint64_t stream = 0)
    -> void;

auto brute_force_mesh_gradient_impl(const float *grad_output,
                                    const float *vertices,
                                    const uint32_t *triangle_indices,
                                    const float *queries, size_t geometry_count,
                                    size_t query_count, size_t vertex_count,
                                    float *output_gradients, int device_id,
                                    uint64_t stream = 0) -> void;

auto brute_force_triangle_gradient_impl(
    const float *grad_output, const float *triangles, const float *queries,
    size_t geometry_count, size_t query_count, float *output_gradients,
    int device_id, uint64_t stream = 0) -> void;
