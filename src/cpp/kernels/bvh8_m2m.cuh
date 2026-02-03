#pragma once
#include "aabb.h"
#include "bvh8.h"
#include "tailor_coefficients.h"
#include <cstdint>
#include <driver_types.h>

void compute_internal_tailor_coefficients_m2m(
    BVH8Node *nodes, const uint32_t *internal_parent_map,
    const AABB *leaf_aabbs, const TailorCoefficientsBf16 *leaf_coefficients,
    const uint32_t *leaf_parents, const LeafPointers *leaf_pointers,
    uint32_t leaf_count, uint32_t *atomic_counters, const cudaStream_t &stream = 0);
