#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <driver_types.h>

// Gather the (N, D) payload array along a permutation:
//     sorted_payload[i, :] = payload[sort_indices[i], :]
template <uint32_t D>
void gather_payload(const float *payload, const uint32_t *sort_indices,
                    float *sorted_payload, size_t n_rows, cudaStream_t stream);

// Runtime-D dispatcher. Picks the compile-time `D` that matches the bucket
// used by PayloadWindingNumbersBackend. Throws for `d > 128`.
void gather_payload_dynamic(const float *payload, const uint32_t *sort_indices,
                            float *sorted_payload, size_t n_rows, uint32_t d,
                            cudaStream_t stream);

// Returns the smallest bucket >= `d` from {1, 2, 3, 4, 8, 16, 32, 64, 128}.
// Throws std::runtime_error for d > 128 or d == 0.
auto payload_d_bucket(uint32_t d) -> uint32_t;
