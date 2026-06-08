#pragma once
#include "aabb.h"
#include "binary_node.h"
#include "geometry.h"
#include "tailor_coefficients.h"
#include <cstdint>
#include <driver_types.h>

// Gather kernel to read PointNormals into SoA format
void gather_point_normals_soa(const float *__restrict__ points,
                             const float *__restrict__ normals,
                             const uint32_t *__restrict__ indices,
                             float *__restrict__ out_geometry, uint32_t count,
                             const cudaStream_t &stream = 0);

// Gather kernel to read Triangles into SoA format
void gather_triangles_soa(const float *__restrict__ input_triangles,
                          const uint32_t *__restrict__ to_internal_map,
                          float *__restrict__ output_triangles_soa,
                          uint32_t count, const cudaStream_t &stream = 0);

void build_binary_topology(const uint64_t *__restrict__ morton_codes,
                           BinaryNode *nodes, uint32_t *parents,
                           uint32_t leaf_count, const cudaStream_t &stream = 0);

template <IsGeometry Geometry>
void populate_binary_tree_aabb_and_leaf_coefficients(
    const float *__restrict__ sorted_geometry,
    TailorCoefficientsBf16 *leaf_coefficients, uint32_t leaf_count,
    const BinaryNode *binary_nodes, AABB *binary_aabbs,
    const uint32_t *binary_parents, uint32_t *atomic_counters,
    uint32_t point_count, const cudaStream_t &stream = 0);
