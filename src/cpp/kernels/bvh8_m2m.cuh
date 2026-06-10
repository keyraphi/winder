#pragma once
#include "bvh8.h"
#include <cstdint>
#include <driver_types.h>

void compute_internal_tailor_coefficients_m2m(
    BVH8Node *nodes, const uint32_t *internal_parent_map,
    const float *leaf_zero_order, const uint32_t *leaf_parents,
    const LeafPointers *leaf_pointers, const uint32_t *node_child_count,
    uint32_t leaf_count, uint32_t *atomic_counters,
    const cudaStream_t &stream = 0);
