#pragma once

#include "geometry.h"
#include "vec3.h"
#include <cstdint>
#include <driver_types.h>

void compute_brute_force_point_normal(
    const Vec3 *queries_vec3, const Vec3 *points_vec3, const Vec3 *normals_vec3,
    uint32_t query_count, uint32_t geometry_count, float *winding_numbers,
    float epsilon, cudaStream_t compute_stream);

void compute_brute_force_triangle(const Vec3 *queries_vec3,
                                  const Triangle *triangles,
                                  uint32_t query_count, uint32_t geometry_count,
                                  float *winding_numbers,
                                  cudaStream_t compute_stream);

void compute_brute_force_gradients_point_normals(
    const float *grad_output, const Vec3 *points, const Vec3 *scaled_normals,
    const Vec3 *queries_vec3, uint32_t geometry_count, uint32_t query_count,
    float epsilon, float *gradients, cudaStream_t compute_stream);

void compute_brute_force_gradients_triangles(
    const float *grad_output, const Triangle *triangles,
    const Vec3 *queries_vec3, uint32_t geometry_count, uint32_t query_count,
    float *gradients, cudaStream_t compute_stream);

void accumulate_vertice_gradients(const float *triangle_results,
                                  const uint32_t *triangle_indices,
                                  uint32_t triangle_count,
                                  float *vertice_grads,
                                  cudaStream_t compute_stream);
