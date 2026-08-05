#pragma once
#include "aabb.h"
#include "bvh8.h"
#include "taylor_coefficients.h"
#include <cstdint>
#include <driver_types.h>

void compute_internal_tailor_coefficients_m2m(
    BVH8Node *nodes, const uint32_t *internal_parent_map,
    const AABB *leaf_aabbs, const TaylorCoefficientsF16 *leaf_coefficients,
    const uint32_t *leaf_parents, const LeafPointers *leaf_pointers,
    TaylorCoefficientsF16 *node_tailor_coefficients,
    TaylorCoefficients *m2m_f32_coefficients, const uint32_t *nodes_child_count,
    uint32_t leaf_count, uint32_t *atomic_counters,
    const cudaStream_t &stream = 0);

void compute_internal_tailor_coefficients_m2m_backward(
    BVH8Node *nodes, const uint32_t *internal_parent_map,
    const AABB *leaf_aabbs,
    const BackwardTaylorCoefficients *leaf_coefficients,
    const uint32_t *leaf_parents, const LeafPointers *leaf_pointers,
    BackwardTaylorCoefficients *node_tailor_coefficients,
    const uint32_t *nodes_child_count, uint32_t leaf_count,
    uint32_t *atomic_counters, const cudaStream_t &stream = 0);
