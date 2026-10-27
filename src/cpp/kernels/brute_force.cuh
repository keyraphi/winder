#pragma once

#include "geometry.h"
#include "vec3.h"
#include <cstdint>
#include <driver_types.h>

void compute_brute_force_point_normal(
    const Vec3 *queries_vec3, const Vec3 *points_vec3, const Vec3 *normals_vec3,
    uint32_t query_count, uint32_t geometry_count, float *winding_numbers,
    float epsilon, float scene_scale, cudaStream_t compute_stream,
    bool use_kahan = true);

void compute_brute_force_triangle(
    const Vec3 *queries_vec3, const Triangle *triangles, uint32_t query_count,
    uint32_t geometry_count, float *winding_numbers, float epsilon,
    float scene_scale, cudaStream_t compute_stream, bool use_kahan = true);

void compute_brute_force_point_normal_with_payload(
    const Vec3 *queries_vec3, const Vec3 *points_vec3, const Vec3 *normals_vec3,
    const float *payload, uint32_t query_count, uint32_t geometry_count,
    uint32_t payload_size, float *out_payload, float epsilon, float scene_scale,
    cudaStream_t compute_stream, bool use_kahan = true,
    bool use_vectorized_loads = false);

void compute_brute_force_triangle_with_payload(
    const Vec3 *queries_vec3, const Triangle *triangles, const float *payload,
    uint32_t query_count, uint32_t geometry_count, uint32_t payload_size,
    float *out_payload, float epsilon, float scene_scale,
    cudaStream_t compute_stream, bool use_kahan = false,
    bool use_vectorized_loads = false);

void compute_brute_force_gradients_point_normals(
    const float *grad_output, const Vec3 *points, const Vec3 *scaled_normals,
    const Vec3 *queries_vec3, uint32_t geometry_count, uint32_t query_count,
    float *gradients, float *epsilon_gradients, float epsilon,
    float scene_scale, cudaStream_t compute_stream, bool use_kahan = false);

void compute_brute_force_query_gradients_point_normals(
    const float *grad_output, const Vec3 *points_vec3,
    const Vec3 *scaled_normals_vec3, const Vec3 *queries_vec3,
    uint32_t geometry_count, uint32_t query_count, float *query_gradients,
    float epsilon, float scene_scale, cudaStream_t compute_stream,
    bool use_kahan = true);

void compute_brute_force_gradients_triangles(
    const float *grad_output, const Triangle *triangles,
    const Vec3 *queries_vec3, uint32_t geometry_count, uint32_t query_count,
    float *gradients, float *epsilon_gradients, float epsilon,
    float scene_scale, cudaStream_t compute_stream, bool use_kahan = false);

void compute_brute_force_query_gradients_triangles(
    const float *grad_output, const Triangle *triangles,
    const Vec3 *queries_vec3, uint32_t geometry_count, uint32_t query_count,
    float *query_gradients, float epsilon, float scene_scale,
    cudaStream_t compute_stream, bool use_kahan = true);

// Payload gradients
void compute_brute_force_payload_gradients_point_normal(
    const Vec3 *queries, const Vec3 *points, const Vec3 *normals,
    const float *grad_output, uint32_t query_count, uint32_t geometry_count,
    uint32_t payload_size, float *out_grad, float epsilon, float scene_scale,
    cudaStream_t compute_stream, bool use_kahan = false,
    bool use_vectorized_loads = false);

void compute_brute_force_payload_gradients_triangle(
    const Vec3 *queries, const Triangle *triangles, const float *grad_output,
    uint32_t query_count, uint32_t geometry_count, uint32_t payload_size,
    float *out_grad, float epsilon, float scene_scale,
    cudaStream_t compute_stream, bool use_kahan = false,
    bool use_vectorized_loads = false);

// Geometry gradients with payload
void compute_brute_force_geometry_gradients_with_payload_point_normal(
    const float *grad_output, const Vec3 *points, const Vec3 *normals,
    const float *payload, const Vec3 *queries, uint32_t geometry_count,
    uint32_t query_count, uint32_t payload_size, float *gradients,
    float *epsilon_gradients, float epsilon, float scene_scale,
    cudaStream_t compute_stream, bool use_kahan = false);

void compute_brute_force_geometry_gradients_with_payload_triangle(
    const float *grad_output, const Triangle *triangles, const float *payload,
    const Vec3 *queries, uint32_t geometry_count, uint32_t query_count,
    uint32_t payload_size, float *gradients, float *epsilon_gradients,
    float epsilon, float scene_scale, cudaStream_t compute_stream,
    bool use_kahan = false);

// Query gradients with payload
void compute_brute_force_query_gradients_with_payload_point_normal(
    const float *grad_output, const Vec3 *points, const Vec3 *normals,
    const float *payload, const Vec3 *queries, uint32_t geometry_count,
    uint32_t query_count, uint32_t payload_size, float *query_gradients,
    float epsilon, float scene_scale, cudaStream_t compute_stream,
    bool use_kahan = true);

void compute_brute_force_query_gradients_with_payload_triangle(
    const float *grad_output, const Triangle *triangles, const float *payload,
    const Vec3 *queries, uint32_t geometry_count, uint32_t query_count,
    uint32_t payload_size, float *query_gradients, float epsilon,
    float scene_scale, cudaStream_t compute_stream, bool use_kahan = true);


// Accumulator for mesh
void accumulate_vertice_gradients(const float *triangle_results,
                                  const uint32_t *triangle_indices,
                                  uint32_t triangle_count, float *vertice_grads,
                                  cudaStream_t compute_stream
                                  );
