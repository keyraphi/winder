#pragma once
#include "scene_normalization.h"
#include "vec3.h"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <driver_types.h>

// Gather the (N, D) payload array along a permutation:
//     sorted_payload[i, :] = payload[sort_indices[i], :]
template <uint32_t D>
void gather_payload(const float *payload, const uint32_t *sort_indices,
                    float *sorted_payload, size_t n_rows,
                    uint32_t payload_dim_user, cudaStream_t stream);

// Runtime-D dispatcher. Picks the compile-time `D` that matches the bucket
// used by PayloadWindingNumbersBackend. Throws for `d > 128`.
void gather_payload_dynamic(const float *payload, const uint32_t *sort_indices,
                            float *sorted_payload, size_t n_rows, uint32_t d,
                            cudaStream_t stream);

template <uint32_t D>
void gather_queries_and_payload_grad_outputs_soa(
    const float *queries, const float *grad_output,
    const uint32_t *sort_indices, float *sorted_queries,
    float *sorted_grad_outputs_bucket, size_t query_count,
    uint32_t payload_dim_user, cudaStream_t stream, SceneNormalization norm);

// Returns the smallest bucket >= `d` from {1, 2, 3, 4, 8, 16, 32, 64, 128}.
// Throws std::runtime_error for d > 128 or d == 0.
auto payload_d_bucket(uint32_t d) -> uint32_t;
