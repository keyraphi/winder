#pragma once

#include "geometry.h"
#include "vec3.h"
#include <cstdint>
#include <driver_types.h>

template <IsGeometry Geometry>
void compute_brute_force(const Vec3 *queries_vec3, const float *geometry,
                         uint32_t query_count, uint32_t geometry_count,
                         float *winding_numbers, float epsilon,
                         cudaStream_t compute_stream);

extern template void
compute_brute_force<Triangle>(const Vec3 *queries_vec3,
                              const float *geometry, uint32_t query_count,
                              uint32_t geometry_count, float *winding_numbers,
                              float epsilon, cudaStream_t compute_stream);
extern template void compute_brute_force<PointNormal>(
    const Vec3 *queries_vec3, const float *geometry, uint32_t query_count,
    uint32_t geometry_count, float *winding_numbers, float epsilon,
    cudaStream_t compute_stream);
