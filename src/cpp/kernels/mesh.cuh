#pragma once
#include <cstdint>
#include <driver_types.h>

void gather_triangles(const float *__restrict__ vertices,
                      const uint32_t *__restrict__ triangle_indices,
                      uint32_t triangle_count, float *__restrict__ triangles,
                      const cudaStream_t &stream);

void accumulate_vertex_gradients(
    const float *__restrict__ triangle_gradients,
    const uint32_t *__restrict__ triangle_indices,
    uint32_t triangle_count,
    float *__restrict__ vertice_gradients,
    cudaStream_t stream);
