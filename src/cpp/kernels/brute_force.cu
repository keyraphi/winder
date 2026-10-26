#include "brute_force.cuh"
#include "geometry.h"
#include "kernels/common.cuh"
#include "vec3.h"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <vector_types.h>

// ===========================================================================
// Numerical accumulation helpers
//
// Every kernel takes a `UseKahan` template parameter. When true, outer-loop
// accumulation uses Kahan-compensated summation (~2x instructions, +1 register
// per accumulator, but error drops from O(N*eps) to O(eps)). When false,
// accumulators use plain `+=`. Vector3/PointNormal/Triangle accumulators use
// the existing `kahan_add` overloads and `operator+=`.
// ===========================================================================

template <bool UseKahan>
__device__ __forceinline__ void accum_float(float &sum, float value,
                                            float &comp) {
  if constexpr (UseKahan) {
    kahan_add(sum, value, comp);
  } else {
    (void)comp;
    sum += value;
  }
}

template <bool UseKahan>
__device__ __forceinline__ void accum_vec3(Vec3 &sum, const Vec3 &value,
                                           Vec3 &comp) {
  if constexpr (UseKahan) {
    kahan_add(sum, value, comp);
  } else {
    (void)comp;
    sum += value;
  }
}

template <bool UseKahan>
__device__ __forceinline__ void accum_point_normal(PointNormal &sum,
                                                   const PointNormal &value,
                                                   PointNormal &comp) {
  if constexpr (UseKahan) {
    kahan_add(sum, value, comp);
  } else {
    (void)comp;
    sum += value;
  }
}

template <bool UseKahan>
__device__ __forceinline__ void
accum_triangle(Triangle &sum, const Triangle &value, Triangle &comp) {
  if constexpr (UseKahan) {
    kahan_add(sum, value, comp);
  } else {
    (void)comp;
    sum += value;
  }
}

// Single-component FMA of the form `acc += w * p`, optionally
// Kahan-compensated. Fast-math-safe: uses __fmaf_rn for the fused product+sub,
// and ieee_add/ieee_sub for every correction step, so the compensation is not
// reassociated away.
template <bool UseKahan>
__device__ __forceinline__ void kahan_fma_scalar(float &acc, float &comp,
                                                 float w, float p) {
  if constexpr (UseKahan) {
    const float y = __fmaf_rn(w, p, -comp);
    const float t = ieee_add(acc, y);
    comp = ieee_sub(ieee_sub(t, acc), y);
    acc = t;
  } else {
    (void)comp;
    acc = __fmaf_rn(w, p, acc);
  }
}

// ===========================================================================
// Payload accumulator micro-kernels
//
// `_full` variants assume the accumulator chunk is at its compile-time max
// (`MaxAcc`). `_tail` variants handle the final (possibly partial) chunk with
// a runtime count. `_vec4_*` variants use `float4` loads for the payload row
// and require 16-byte alignment (guaranteed by the caller when `vec=true`).
// ===========================================================================

template <uint32_t MaxAcc, bool UseKahan>
__device__ __forceinline__ void
payload_accumulate_vec4_full(const float *__restrict__ payload_row, float w_j,
                             float (&acc)[MaxAcc], float (&comp)[MaxAcc]) {
  static_assert(MaxAcc % 4 == 0, "Vectorized loads require MaxAcc % 4 == 0");
  const auto *pv = reinterpret_cast<const float4 *>(payload_row);
  constexpr uint32_t NV = MaxAcc / 4;
#pragma unroll
  for (uint32_t k4 = 0; k4 < NV; ++k4) {
    const float4 p = __ldg(pv + k4);
    kahan_fma_scalar<UseKahan>(acc[4 * k4 + 0], comp[4 * k4 + 0], w_j, p.x);
    kahan_fma_scalar<UseKahan>(acc[4 * k4 + 1], comp[4 * k4 + 1], w_j, p.y);
    kahan_fma_scalar<UseKahan>(acc[4 * k4 + 2], comp[4 * k4 + 2], w_j, p.z);
    kahan_fma_scalar<UseKahan>(acc[4 * k4 + 3], comp[4 * k4 + 3], w_j, p.w);
  }
}

template <uint32_t MaxAcc, bool UseKahan>
__device__ __forceinline__ void
payload_accumulate_scalar_full(const float *__restrict__ payload_row, float w_j,
                               float (&acc)[MaxAcc], float (&comp)[MaxAcc]) {
#pragma unroll
  for (uint32_t k = 0; k < MaxAcc; ++k) {
    kahan_fma_scalar<UseKahan>(acc[k], comp[k], w_j, __ldg(payload_row + k));
  }
}

template <bool UseKahan>
__device__ __forceinline__ void
payload_accumulate_vec4_tail(const float *__restrict__ payload_row, float w_j,
                             uint32_t d_count, float *acc, float *comp) {
  const auto *pv = reinterpret_cast<const float4 *>(payload_row);
  const uint32_t num_vec4 = d_count / 4;
  for (uint32_t k4 = 0; k4 < num_vec4; ++k4) {
    const float4 p = __ldg(pv + k4);
    kahan_fma_scalar<UseKahan>(acc[4 * k4 + 0], comp[4 * k4 + 0], w_j, p.x);
    kahan_fma_scalar<UseKahan>(acc[4 * k4 + 1], comp[4 * k4 + 1], w_j, p.y);
    kahan_fma_scalar<UseKahan>(acc[4 * k4 + 2], comp[4 * k4 + 2], w_j, p.z);
    kahan_fma_scalar<UseKahan>(acc[4 * k4 + 3], comp[4 * k4 + 3], w_j, p.w);
  }
  for (uint32_t k = num_vec4 * 4; k < d_count; ++k) {
    kahan_fma_scalar<UseKahan>(acc[k], comp[k], w_j, __ldg(payload_row + k));
  }
}

template <bool UseKahan>
__device__ __forceinline__ void
payload_accumulate_scalar_tail(const float *__restrict__ payload_row, float w_j,
                               uint32_t d_count, float *acc, float *comp) {
  for (uint32_t k = 0; k < d_count; ++k) {
    kahan_fma_scalar<UseKahan>(acc[k], comp[k], w_j, __ldg(payload_row + k));
  }
}

// ===========================================================================
// Forward winding: point-normal
// ===========================================================================

template <uint32_t BlockSize = 256, bool UseKahan = true>
__global__ void compute_winding_numbers_brute_force_point_normal_kernel(
    const Vec3 *__restrict__ queries, const Vec3 *__restrict__ points,
    const Vec3 *__restrict__ normals, const uint32_t query_count,
    const uint32_t geometry_count, float *__restrict__ winding_numbers,
    const PointNormalContext ctx) {

  uint32_t q_idx = blockIdx.x * BlockSize + threadIdx.x;

  extern __shared__ char shared_mem[];
  Vec3 *point_tile = reinterpret_cast<Vec3 *>(shared_mem);
  Vec3 *normal_tile = point_tile + BlockSize;

  float my_wn = 0.0F;
  float c = 0.F;
  Vec3 my_q = (q_idx < query_count) ? queries[q_idx] : Vec3::zero();

  for (uint32_t i = 0; i < geometry_count; i += BlockSize) {
    uint32_t load_idx = i + threadIdx.x;

    if (load_idx < geometry_count) {
      point_tile[threadIdx.x] = points[load_idx];
      normal_tile[threadIdx.x] = normals[load_idx];
    }
    __syncthreads();

    if (q_idx < query_count) {
      uint32_t num_elements_in_tile = min(BlockSize, geometry_count - i);

      if (num_elements_in_tile == BlockSize) {
#pragma unroll 4
        for (uint32_t j = 0; j < BlockSize; ++j) {
          PointNormal pn{.p = point_tile[j], .n = normal_tile[j]};
          const float contrib = pn.contributionToQuery(my_q, ctx);
          accum_float<UseKahan>(my_wn, contrib, c);
        }
      } else {
        for (uint32_t j = 0; j < num_elements_in_tile; ++j) {
          PointNormal pn{.p = point_tile[j], .n = normal_tile[j]};
          const float contrib = pn.contributionToQuery(my_q, ctx);
          accum_float<UseKahan>(my_wn, contrib, c);
        }
      }
    }
    __syncthreads();
  }

  if (q_idx < query_count) {
    winding_numbers[q_idx] = my_wn;
  }
}

// ===========================================================================
// Forward winding: triangle soup
// ===========================================================================

template <uint32_t BlockSize = 256, bool UseKahan = true>
__global__ void compute_winding_numbers_brute_force_triangle_kernel(
    const Vec3 *__restrict__ queries, const Triangle *__restrict__ triangles,
    const uint32_t query_count, const uint32_t geometry_count,
    const typename Triangle::Context reg_ctx,
    float *__restrict__ winding_numbers) {

  uint32_t q_idx = blockIdx.x * BlockSize + threadIdx.x;

  extern __shared__ char shared_mem[];
  auto *tile = reinterpret_cast<Triangle *>(shared_mem);

  float my_wn = 0.0F;
  float c = 0.F;
  Vec3 my_q = (q_idx < query_count) ? queries[q_idx] : Vec3::zero();

  for (uint32_t i = 0; i < geometry_count; i += BlockSize) {
    uint32_t load_idx = i + threadIdx.x;

    if (load_idx < geometry_count) {
      tile[threadIdx.x] = triangles[load_idx];
    }
    __syncthreads();

    if (q_idx < query_count) {
      uint32_t num_elements_in_tile = min(BlockSize, geometry_count - i);

      if (num_elements_in_tile == BlockSize) {
#pragma unroll 4
        for (uint32_t j = 0; j < BlockSize; ++j) {
          const float contrib = tile[j].contributionToQuery(my_q, reg_ctx);
          accum_float<UseKahan>(my_wn, contrib, c);
        }
      } else {
        for (uint32_t j = 0; j < num_elements_in_tile; ++j) {
          const float contrib = tile[j].contributionToQuery(my_q, reg_ctx);
          accum_float<UseKahan>(my_wn, contrib, c);
        }
      }
    }
    __syncthreads();
  }

  if (q_idx < query_count) {
    winding_numbers[q_idx] = my_wn;
  }
}

// ===========================================================================
// Forward winding dispatchers
// ===========================================================================

void compute_brute_force_point_normal(
    const Vec3 *queries_vec3, const Vec3 *points_vec3, const Vec3 *normals_vec3,
    const uint32_t query_count, const uint32_t geometry_count,
    float *winding_numbers, float epsilon, float scene_scale,
    cudaStream_t compute_stream, bool use_kahan) {

  if (query_count == 0) {
    return;
  }

  const auto ctx = PointNormalContext::make(epsilon, scene_scale);
  constexpr uint32_t threads = 256;
  const uint32_t blocks = (query_count + threads - 1) / threads;
  const size_t smem_size = static_cast<size_t>(threads) * 2 * sizeof(Vec3);

  if (use_kahan) {
    compute_winding_numbers_brute_force_point_normal_kernel<threads, true>
        <<<blocks, threads, smem_size, compute_stream>>>(
            queries_vec3, points_vec3, normals_vec3, query_count,
            geometry_count, winding_numbers, ctx);
  } else {
    compute_winding_numbers_brute_force_point_normal_kernel<threads, false>
        <<<blocks, threads, smem_size, compute_stream>>>(
            queries_vec3, points_vec3, normals_vec3, query_count,
            geometry_count, winding_numbers, ctx);
  }
  CUDA_CHECK(cudaGetLastError());
}

void compute_brute_force_triangle(const Vec3 *queries_vec3,
                                  const Triangle *triangles,
                                  const uint32_t query_count,
                                  const uint32_t geometry_count,
                                  float *winding_numbers, const float epsilon,
                                  const float scene_scale,
                                  cudaStream_t compute_stream, bool use_kahan) {
  if (query_count == 0) {
    return;
  }

  constexpr uint32_t threads = 256;
  const uint32_t blocks = (query_count + threads - 1) / threads;
  const size_t smem_size = threads * sizeof(Triangle);
  const auto ctx = Triangle::Context::make(epsilon, scene_scale);

  if (use_kahan) {
    compute_winding_numbers_brute_force_triangle_kernel<threads, true>
        <<<blocks, threads, smem_size, compute_stream>>>(
            queries_vec3, triangles, query_count, geometry_count, ctx,
            winding_numbers);
  } else {
    compute_winding_numbers_brute_force_triangle_kernel<threads, false>
        <<<blocks, threads, smem_size, compute_stream>>>(
            queries_vec3, triangles, query_count, geometry_count, ctx,
            winding_numbers);
  }
  CUDA_CHECK(cudaGetLastError());
}

// ===========================================================================
// Payload brute-force: point-normal
// ===========================================================================

template <uint32_t BlockSize = 256, uint32_t MaxAcc = 32, bool UseKahan = false,
          bool UseVectorizedLoads = false>
__global__ void
compute_winding_numbers_brute_force_point_normal_with_payload_kernel(
    const Vec3 *__restrict__ queries, const Vec3 *__restrict__ points,
    const Vec3 *__restrict__ normals, const float *__restrict__ payload,
    const uint32_t query_count, const uint32_t geometry_count,
    const uint32_t payload_size, float *__restrict__ out_payload,
    const PointNormalContext ctx) {

  const uint32_t q_idx = blockIdx.x * BlockSize + threadIdx.x;
  const bool valid_query = q_idx < query_count;
  const Vec3 my_q = valid_query ? queries[q_idx] : Vec3::zero();

  extern __shared__ char shared_mem[];
  Vec3 *point_tile = reinterpret_cast<Vec3 *>(shared_mem);
  Vec3 *normal_tile = point_tile + BlockSize;

  const uint32_t num_chunks = (payload_size + MaxAcc - 1) / MaxAcc;

  for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
    const uint32_t base_d = chunk * MaxAcc;
    const uint32_t d_count = min(MaxAcc, payload_size - base_d);

    float acc[MaxAcc];
    float comp[MaxAcc];
#pragma unroll
    for (uint32_t k = 0; k < MaxAcc; ++k) {
      acc[k] = 0.F;
      comp[k] = 0.F;
    }

    for (uint32_t i = 0; i < geometry_count; i += BlockSize) {
      const uint32_t load_idx = i + threadIdx.x;

      if (load_idx < geometry_count) {
        point_tile[threadIdx.x] = points[load_idx];
        normal_tile[threadIdx.x] = normals[load_idx];
      }
      __syncthreads();

      if (valid_query) {
        const uint32_t num_elements = min(BlockSize, geometry_count - i);

        if (num_elements == BlockSize && d_count == MaxAcc) {
          // HOT: full tile, full chunk. Compile-time loop bounds.
#pragma unroll 2
          for (uint32_t j = 0; j < BlockSize; ++j) {
            const PointNormal pn{.p = point_tile[j], .n = normal_tile[j]};
            const float w_j = pn.contributionToQuery(my_q, ctx);
            const float *pr =
                payload + static_cast<size_t>(i + j) * payload_size + base_d;
            if constexpr (UseVectorizedLoads) {
              payload_accumulate_vec4_full<MaxAcc, UseKahan>(pr, w_j, acc,
                                                             comp);
            } else {
              payload_accumulate_scalar_full<MaxAcc, UseKahan>(pr, w_j, acc,
                                                               comp);
            }
          }
        } else if (d_count == MaxAcc) {
          // Ragged tile, full chunk. Runtime j-bound, compile-time chunk.
          for (uint32_t j = 0; j < num_elements; ++j) {
            const PointNormal pn{.p = point_tile[j], .n = normal_tile[j]};
            const float w_j = pn.contributionToQuery(my_q, ctx);
            const float *pr =
                payload + static_cast<size_t>(i + j) * payload_size + base_d;
            if constexpr (UseVectorizedLoads) {
              payload_accumulate_vec4_full<MaxAcc, UseKahan>(pr, w_j, acc,
                                                             comp);
            } else {
              payload_accumulate_scalar_full<MaxAcc, UseKahan>(pr, w_j, acc,
                                                               comp);
            }
          }
        } else {
          // Tail chunk (at most one per thread). Runtime d_count.
          for (uint32_t j = 0; j < num_elements; ++j) {
            const PointNormal pn{.p = point_tile[j], .n = normal_tile[j]};
            const float w_j = pn.contributionToQuery(my_q, ctx);
            const float *pr =
                payload + static_cast<size_t>(i + j) * payload_size + base_d;
            if constexpr (UseVectorizedLoads) {
              payload_accumulate_vec4_tail<UseKahan>(pr, w_j, d_count, acc,
                                                     comp);
            } else {
              payload_accumulate_scalar_tail<UseKahan>(pr, w_j, d_count, acc,
                                                       comp);
            }
          }
        }
      }
      __syncthreads();
    }

    if (valid_query) {
      float *out_row =
          out_payload + static_cast<size_t>(q_idx) * payload_size + base_d;
      for (uint32_t k = 0; k < d_count; ++k) {
        out_row[k] = acc[k];
      }
    }
    __syncthreads();
  }
}

// ===========================================================================
// Payload brute-force: triangle soup
// ===========================================================================

template <uint32_t BlockSize = 256, uint32_t MaxAcc = 32, bool UseKahan = false,
          bool UseVectorizedLoads = false>
__global__ void
compute_winding_numbers_brute_force_triangle_with_payload_kernel(
    const Vec3 *__restrict__ queries, const Triangle *__restrict__ triangles,
    const float *__restrict__ payload, const uint32_t query_count,
    const uint32_t geometry_count, const uint32_t payload_size,
    float *__restrict__ out_payload, const typename Triangle::Context reg_ctx) {

  const uint32_t q_idx = blockIdx.x * BlockSize + threadIdx.x;
  const bool valid_query = q_idx < query_count;
  const Vec3 my_q = valid_query ? queries[q_idx] : Vec3::zero();

  extern __shared__ char shared_mem[];
  auto *tile = reinterpret_cast<Triangle *>(shared_mem);

  const uint32_t num_chunks = (payload_size + MaxAcc - 1) / MaxAcc;

  for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
    const uint32_t base_d = chunk * MaxAcc;
    const uint32_t d_count = min(MaxAcc, payload_size - base_d);

    float acc[MaxAcc];
    float comp[MaxAcc];
#pragma unroll
    for (uint32_t k = 0; k < MaxAcc; ++k) {
      acc[k] = 0.F;
      comp[k] = 0.F;
    }

    for (uint32_t i = 0; i < geometry_count; i += BlockSize) {
      const uint32_t load_idx = i + threadIdx.x;

      if (load_idx < geometry_count) {
        tile[threadIdx.x] = triangles[load_idx];
      }
      __syncthreads();

      if (valid_query) {
        const uint32_t num_elements = min(BlockSize, geometry_count - i);

        if (num_elements == BlockSize && d_count == MaxAcc) {
#pragma unroll 2
          for (uint32_t j = 0; j < BlockSize; ++j) {
            const float w_j = tile[j].contributionToQuery(my_q, reg_ctx);
            const float *pr =
                payload + static_cast<size_t>(i + j) * payload_size + base_d;
            if constexpr (UseVectorizedLoads) {
              payload_accumulate_vec4_full<MaxAcc, UseKahan>(pr, w_j, acc,
                                                             comp);
            } else {
              payload_accumulate_scalar_full<MaxAcc, UseKahan>(pr, w_j, acc,
                                                               comp);
            }
          }
        } else if (d_count == MaxAcc) {
          for (uint32_t j = 0; j < num_elements; ++j) {
            const float w_j = tile[j].contributionToQuery(my_q, reg_ctx);
            const float *pr =
                payload + static_cast<size_t>(i + j) * payload_size + base_d;
            if constexpr (UseVectorizedLoads) {
              payload_accumulate_vec4_full<MaxAcc, UseKahan>(pr, w_j, acc,
                                                             comp);
            } else {
              payload_accumulate_scalar_full<MaxAcc, UseKahan>(pr, w_j, acc,
                                                               comp);
            }
          }
        } else {
          for (uint32_t j = 0; j < num_elements; ++j) {
            const float w_j = tile[j].contributionToQuery(my_q, reg_ctx);
            const float *pr =
                payload + static_cast<size_t>(i + j) * payload_size + base_d;
            if constexpr (UseVectorizedLoads) {
              payload_accumulate_vec4_tail<UseKahan>(pr, w_j, d_count, acc,
                                                     comp);
            } else {
              payload_accumulate_scalar_tail<UseKahan>(pr, w_j, d_count, acc,
                                                       comp);
            }
          }
        }
      }
      __syncthreads();
    }

    if (valid_query) {
      float *out_row =
          out_payload + static_cast<size_t>(q_idx) * payload_size + base_d;
      for (uint32_t k = 0; k < d_count; ++k) {
        out_row[k] = acc[k];
      }
    }
    __syncthreads();
  }
}

// ===========================================================================
// Payload dispatch helpers
// ===========================================================================

template <uint32_t BlockSize, bool UseKahan, bool UseVectorized>
static void dispatch_payload_point_normal(
    const Vec3 *q, const Vec3 *p, const Vec3 *n, const float *payload,
    uint32_t query_count, uint32_t geometry_count, uint32_t payload_size,
    float *out_payload, PointNormalContext ctx, uint32_t blocks,
    size_t smem_size, cudaStream_t stream) {

#define LAUNCH(ACC)                                                            \
  compute_winding_numbers_brute_force_point_normal_with_payload_kernel<        \
      BlockSize, ACC, UseKahan, UseVectorized>                                 \
      <<<blocks, BlockSize, smem_size, stream>>>(                              \
          q, p, n, payload, query_count, geometry_count, payload_size,         \
          out_payload, ctx)

  if constexpr (UseVectorized) {
    // Vec path already requires payload_size % 4 == 0 at the call site,
    // so 1, 2, 3 are unreachable. Round up to {4, 8, 16, 32}.
    const uint32_t b = (payload_size <= 4)    ? 4
                       : (payload_size <= 8)  ? 8
                       : (payload_size <= 16) ? 16
                                              : 32;
    switch (b) {
    case 4:
      LAUNCH(4);
      break;
    case 8:
      LAUNCH(8);
      break;
    case 16:
      LAUNCH(16);
      break;
    default:
      LAUNCH(32);
      break;
    }
  } else {
    // Scalar path: D=5..7 -> 8, D=9..15 -> 16. Both are 40-register
    // instantiations (see ptxas), same chunk count as 32, better occupancy.
    const uint32_t b = (payload_size <= 1)    ? 1
                       : (payload_size <= 2)  ? 2
                       : (payload_size <= 3)  ? 3
                       : (payload_size <= 4)  ? 4
                       : (payload_size <= 8)  ? 8
                       : (payload_size <= 16) ? 16
                                              : 32;
    switch (b) {
    case 1:
      LAUNCH(1);
      break;
    case 2:
      LAUNCH(2);
      break;
    case 3:
      LAUNCH(3);
      break;
    case 4:
      LAUNCH(4);
      break;
    case 8:
      LAUNCH(8);
      break;
    case 16:
      LAUNCH(16);
      break;
    default:
      LAUNCH(32);
      break;
    }
  }
#undef LAUNCH
}

template <uint32_t BlockSize, bool UseKahan, bool UseVectorized>
static void dispatch_payload_triangle(
    const Vec3 *q, const Triangle *triangles, const float *payload,
    uint32_t query_count, uint32_t geometry_count, uint32_t payload_size,
    float *out_payload, typename Triangle::Context ctx, uint32_t blocks,
    size_t smem_size, cudaStream_t stream) {

#define LAUNCH(ACC)                                                            \
  compute_winding_numbers_brute_force_triangle_with_payload_kernel<            \
      BlockSize, ACC, UseKahan, UseVectorized>                                 \
      <<<blocks, BlockSize, smem_size, stream>>>(                              \
          q, triangles, payload, query_count, geometry_count, payload_size,    \
          out_payload, ctx)

  if constexpr (UseVectorized) {
    // Vec path already requires payload_size % 4 == 0 at the call site,
    // so 1, 2, 3 are unreachable. Round up to {4, 8, 16, 32}.
    const uint32_t b = (payload_size <= 4)    ? 4
                       : (payload_size <= 8)  ? 8
                       : (payload_size <= 16) ? 16
                                              : 32;
    switch (b) {
    case 4:
      LAUNCH(4);
      break;
    case 8:
      LAUNCH(8);
      break;
    case 16:
      LAUNCH(16);
      break;
    default:
      LAUNCH(32);
      break;
    }
  } else {
    // Scalar path: D=5..7 -> 8, D=9..15 -> 16. Both are 40-register
    // instantiations (see ptxas), same chunk count as 32, better occupancy.
    const uint32_t b = (payload_size <= 1)    ? 1
                       : (payload_size <= 2)  ? 2
                       : (payload_size <= 3)  ? 3
                       : (payload_size <= 4)  ? 4
                       : (payload_size <= 8)  ? 8
                       : (payload_size <= 16) ? 16
                                              : 32;
    switch (b) {
    case 1:
      LAUNCH(1);
      break;
    case 2:
      LAUNCH(2);
      break;
    case 3:
      LAUNCH(3);
      break;
    case 4:
      LAUNCH(4);
      break;
    case 8:
      LAUNCH(8);
      break;
    case 16:
      LAUNCH(16);
      break;
    default:
      LAUNCH(32);
      break;
    }
  }
#undef LAUNCH
}

void compute_brute_force_point_normal_with_payload(
    const Vec3 *queries_vec3, const Vec3 *points_vec3, const Vec3 *normals_vec3,
    const float *payload, uint32_t query_count, uint32_t geometry_count,
    uint32_t payload_size, float *out_payload, float epsilon, float scene_scale,
    cudaStream_t compute_stream, bool use_kahan, bool use_vectorized_loads) {

  if (query_count == 0) {
    return;
  }

  const auto ctx = PointNormalContext::make(epsilon, scene_scale);
  constexpr uint32_t threads = 256;
  const uint32_t blocks = (query_count + threads - 1) / threads;
  const size_t smem_size = static_cast<size_t>(threads) * 2 * sizeof(Vec3);

  // Vectorized path requires 16-byte row alignment on top of the payload-size
  // condition.  cudaMalloc always returns 256-byte aligned pointers, so the
  // base-pointer check is only a defensive guard for exotic user-supplied
  // slices; in practice it should always pass.
  const bool can_vec = (payload_size % 4 == 0) &&
                       (reinterpret_cast<uintptr_t>(payload) % 16 == 0);
  const bool vec = use_vectorized_loads && can_vec;

  if (use_kahan) {
    if (vec) {
      dispatch_payload_point_normal<threads, true, true>(
          queries_vec3, points_vec3, normals_vec3, payload, query_count,
          geometry_count, payload_size, out_payload, ctx, blocks, smem_size,
          compute_stream);
    } else {
      dispatch_payload_point_normal<threads, true, false>(
          queries_vec3, points_vec3, normals_vec3, payload, query_count,
          geometry_count, payload_size, out_payload, ctx, blocks, smem_size,
          compute_stream);
    }
  } else {
    if (vec) {
      dispatch_payload_point_normal<threads, false, true>(
          queries_vec3, points_vec3, normals_vec3, payload, query_count,
          geometry_count, payload_size, out_payload, ctx, blocks, smem_size,
          compute_stream);
    } else {
      dispatch_payload_point_normal<threads, false, false>(
          queries_vec3, points_vec3, normals_vec3, payload, query_count,
          geometry_count, payload_size, out_payload, ctx, blocks, smem_size,
          compute_stream);
    }
  }
  CUDA_CHECK(cudaGetLastError());
}

void compute_brute_force_triangle_with_payload(
    const Vec3 *queries_vec3, const Triangle *triangles, const float *payload,
    uint32_t query_count, uint32_t geometry_count, uint32_t payload_size,
    float *out_payload, float epsilon, float scene_scale,
    cudaStream_t compute_stream, bool use_kahan, bool use_vectorized_loads) {

  if (query_count == 0) {
    return;
  }

  constexpr uint32_t threads = 256;
  const uint32_t blocks = (query_count + threads - 1) / threads;
  const size_t smem_size = threads * sizeof(Triangle);
  const auto ctx = Triangle::Context::make(epsilon, scene_scale);

  const bool can_vec = (payload_size % 4 == 0) &&
                       (reinterpret_cast<uintptr_t>(payload) % 16 == 0);
  const bool vec = use_vectorized_loads && can_vec;

  if (use_kahan) {
    if (vec) {
      dispatch_payload_triangle<threads, true, true>(
          queries_vec3, triangles, payload, query_count, geometry_count,
          payload_size, out_payload, ctx, blocks, smem_size, compute_stream);
    } else {
      dispatch_payload_triangle<threads, true, false>(
          queries_vec3, triangles, payload, query_count, geometry_count,
          payload_size, out_payload, ctx, blocks, smem_size, compute_stream);
    }
  } else {
    if (vec) {
      dispatch_payload_triangle<threads, false, true>(
          queries_vec3, triangles, payload, query_count, geometry_count,
          payload_size, out_payload, ctx, blocks, smem_size, compute_stream);
    } else {
      dispatch_payload_triangle<threads, false, false>(
          queries_vec3, triangles, payload, query_count, geometry_count,
          payload_size, out_payload, ctx, blocks, smem_size, compute_stream);
    }
  }
  CUDA_CHECK(cudaGetLastError());
}

// ===========================================================================
// Backward: geometry gradients (point-normal)
// ===========================================================================

template <uint32_t BlockSize = 256, bool UseKahan = true>
__global__ void gradients_brute_force_point_normals_kernel(
    const float *__restrict__ in_gradients, const Vec3 *__restrict__ points,
    const Vec3 *__restrict__ normals, const Vec3 *__restrict__ queries,
    const uint32_t geometry_count, const uint32_t query_count,
    const PointNormalContext ctx, float *__restrict__ out_gradients,
    float *__restrict__ epsilon_gradients) {

  uint32_t pn_idx = blockIdx.x * BlockSize + threadIdx.x;

  extern __shared__ char shared_mem[];
  auto *tile_q = reinterpret_cast<Vec3 *>(shared_mem);
  auto *tile_g =
      reinterpret_cast<float *>(shared_mem + BlockSize * sizeof(Vec3));

  const auto *queries_float = reinterpret_cast<const float *>(queries);
  auto *tile_q_float = reinterpret_cast<float *>(tile_q);

  PointNormal my_grad{.p = Vec3::zero(), .n = Vec3::zero()};
  float my_eps = 0.F;
  PointNormal geo_c{.p = Vec3::zero(), .n = Vec3::zero()};
  float eps_c = 0.F;

  PointNormal my_point_normal;
  if (pn_idx < geometry_count) {
    my_point_normal = PointNormal{.p = points[pn_idx], .n = normals[pn_idx]};
  }

  for (uint32_t i = 0; i < query_count; i += BlockSize) {
    uint32_t num_elements_in_tile = min(BlockSize, query_count - i);

    if (num_elements_in_tile == BlockSize) {
#pragma unroll
      for (uint32_t f_idx = threadIdx.x; f_idx < BlockSize * 3;
           f_idx += BlockSize) {
        tile_q_float[f_idx] = queries_float[i * 3 + f_idx];
      }
      tile_g[threadIdx.x] = in_gradients[i + threadIdx.x];
      __syncthreads();

      if (pn_idx < geometry_count) {
#pragma unroll 4
        for (uint32_t j = 0; j < BlockSize; ++j) {
          GeometryGradient<PointNormal> contribution =
              my_point_normal.gradContributionOfQuery(tile_q[j], tile_g[j],
                                                      ctx);
          accum_point_normal<UseKahan>(my_grad, contribution.geometry, geo_c);
          accum_float<UseKahan>(my_eps, contribution.epsilon, eps_c);
        }
      }
    } else {
      uint32_t total_floats_to_copy = num_elements_in_tile * 3;
      for (uint32_t f_idx = threadIdx.x; f_idx < total_floats_to_copy;
           f_idx += BlockSize) {
        tile_q_float[f_idx] = queries_float[i * 3 + f_idx];
      }
      uint32_t load_idx = i + threadIdx.x;
      if (load_idx < query_count) {
        tile_g[threadIdx.x] = in_gradients[load_idx];
      }
      __syncthreads();

      if (pn_idx < geometry_count) {
        for (uint32_t j = 0; j < num_elements_in_tile; ++j) {
          GeometryGradient<PointNormal> contribution =
              my_point_normal.gradContributionOfQuery(tile_q[j], tile_g[j],
                                                      ctx);
          accum_point_normal<UseKahan>(my_grad, contribution.geometry, geo_c);
          accum_float<UseKahan>(my_eps, contribution.epsilon, eps_c);
        }
      }
    }
    __syncthreads();
  }

  if (pn_idx < geometry_count) {
    uint32_t out_base = 6 * pn_idx;
    out_gradients[out_base] = my_grad.n.x;
    out_gradients[out_base + 1] = my_grad.n.y;
    out_gradients[out_base + 2] = my_grad.n.z;
    out_gradients[out_base + 3] = my_grad.p.x;
    out_gradients[out_base + 4] = my_grad.p.y;
    out_gradients[out_base + 5] = my_grad.p.z;
    epsilon_gradients[pn_idx] = my_eps;
  }
}

void compute_brute_force_gradients_point_normals(
    const float *grad_output, const Vec3 *points_vec3,
    const Vec3 *scaled_normals_vec3, const Vec3 *queries_vec3,
    const uint32_t geometry_count, const uint32_t query_count, float *gradients,
    float *epsilon_gradients, const float epsilon, const float scene_scale,
    cudaStream_t compute_stream, bool use_kahan) {

  constexpr uint32_t threads = 256;
  uint32_t geom_blocks = (geometry_count + threads - 1) / threads;

  const auto ctx = PointNormalContext::make(epsilon, scene_scale);
  size_t smem_size = threads * (sizeof(Vec3) + sizeof(float));

  if (use_kahan) {
    gradients_brute_force_point_normals_kernel<threads, true>
        <<<geom_blocks, threads, smem_size, compute_stream>>>(
            grad_output, points_vec3, scaled_normals_vec3, queries_vec3,
            geometry_count, query_count, ctx, gradients, epsilon_gradients);
  } else {
    gradients_brute_force_point_normals_kernel<threads, false>
        <<<geom_blocks, threads, smem_size, compute_stream>>>(
            grad_output, points_vec3, scaled_normals_vec3, queries_vec3,
            geometry_count, query_count, ctx, gradients, epsilon_gradients);
  }
  CUDA_CHECK(cudaGetLastError());
}

// ===========================================================================
// Backward: query gradients (point-normal)
// ===========================================================================

template <uint32_t BlockSize = 256, bool UseKahan = true>
__global__ void gradients_wrt_query_brute_force_point_normals_kernel(
    const float *__restrict__ in_gradients, const Vec3 *__restrict__ points,
    const Vec3 *__restrict__ normals, const Vec3 *__restrict__ queries,
    const uint32_t geometry_count, const uint32_t query_count,
    const PointNormalContext ctx, float *__restrict__ out_query_gradients) {

  const uint32_t q_idx = blockIdx.x * BlockSize + threadIdx.x;

  const bool active = (q_idx < query_count);
  const Vec3 my_query = active ? queries[q_idx] : Vec3::zero();
  const float my_g = active ? in_gradients[q_idx] : 0.F;

  Vec3 q_grad = Vec3::zero();
  Vec3 compensator = Vec3::zero();

  extern __shared__ char shared_mem[];
  auto *tile_p = reinterpret_cast<Vec3 *>(shared_mem);
  auto *tile_n =
      reinterpret_cast<Vec3 *>(shared_mem + BlockSize * sizeof(Vec3));
  auto *tile_p_f = reinterpret_cast<float *>(tile_p);
  auto *tile_n_f = reinterpret_cast<float *>(tile_n);

  const auto *points_f = reinterpret_cast<const float *>(points);
  const auto *normals_f = reinterpret_cast<const float *>(normals);

  for (uint32_t i = 0; i < geometry_count; i += BlockSize) {
    const uint32_t num = min(BlockSize, geometry_count - i);
    const uint32_t total_floats = num * 3;

    for (uint32_t f = threadIdx.x; f < total_floats; f += BlockSize) {
      tile_p_f[f] = points_f[i * 3 + f];
      tile_n_f[f] = normals_f[i * 3 + f];
    }
    __syncthreads();

    if (active) {
      if (num == BlockSize) {
#pragma unroll 4
        for (uint32_t j = 0; j < BlockSize; ++j) {
          const PointNormal pn{.p = tile_p[j], .n = tile_n[j]};
          const Vec3 contrib = pn.queryGradContribution(my_query, my_g, ctx);
          accum_vec3<UseKahan>(q_grad, contrib, compensator);
        }
      } else {
        for (uint32_t j = 0; j < num; ++j) {
          const PointNormal pn{.p = tile_p[j], .n = tile_n[j]};
          const Vec3 contrib = pn.queryGradContribution(my_query, my_g, ctx);
          accum_vec3<UseKahan>(q_grad, contrib, compensator);
        }
      }
    }
    __syncthreads();
  }

  if (active) {
    const uint32_t out_base = 3 * q_idx;
    out_query_gradients[out_base + 0] = q_grad.x;
    out_query_gradients[out_base + 1] = q_grad.y;
    out_query_gradients[out_base + 2] = q_grad.z;
  }
}

void compute_brute_force_query_gradients_point_normals(
    const float *grad_output, const Vec3 *points_vec3,
    const Vec3 *scaled_normals_vec3, const Vec3 *queries_vec3,
    const uint32_t geometry_count, const uint32_t query_count,
    float *query_gradients, const float epsilon, const float scene_scale,
    cudaStream_t compute_stream, bool use_kahan) {

  constexpr uint32_t threads = 256;
  const uint32_t query_blocks = (query_count + threads - 1) / threads;
  const auto ctx = PointNormalContext::make(epsilon, scene_scale);
  const size_t smem_size = threads * (sizeof(Vec3) + sizeof(Vec3));

  if (use_kahan) {
    gradients_wrt_query_brute_force_point_normals_kernel<threads, true>
        <<<query_blocks, threads, smem_size, compute_stream>>>(
            grad_output, points_vec3, scaled_normals_vec3, queries_vec3,
            geometry_count, query_count, ctx, query_gradients);
  } else {
    gradients_wrt_query_brute_force_point_normals_kernel<threads, false>
        <<<query_blocks, threads, smem_size, compute_stream>>>(
            grad_output, points_vec3, scaled_normals_vec3, queries_vec3,
            geometry_count, query_count, ctx, query_gradients);
  }
  CUDA_CHECK(cudaGetLastError());
}

// ===========================================================================
// Backward: geometry gradients (triangle soup)
// ===========================================================================

template <uint32_t BlockSize = 256, bool UseKahan = true>
__global__ void gradients_brute_force_triangles_kernel(
    const float *__restrict__ in_gradients,
    const Triangle *__restrict__ triangles, const Vec3 *__restrict__ queries,
    const uint32_t geometry_count, const uint32_t query_count,
    const typename Triangle::Context reg_ctx, float *__restrict__ out_gradients,
    float *__restrict__ epsilon_gradients) {

  uint32_t triangle_idx = blockIdx.x * BlockSize + threadIdx.x;

  extern __shared__ char shared_mem[];
  auto *tile_q = reinterpret_cast<Vec3 *>(shared_mem);
  auto *tile_g =
      reinterpret_cast<float *>(shared_mem + BlockSize * sizeof(Vec3));

  const auto *queries_float = reinterpret_cast<const float *>(queries);
  auto *tile_q_float = reinterpret_cast<float *>(tile_q);

  Triangle my_grad{.v0 = Vec3::zero(), .v1 = Vec3::zero(), .v2 = Vec3::zero()};
  float my_eps = 0.F;
  Triangle geo_c{.v0 = Vec3::zero(), .v1 = Vec3::zero(), .v2 = Vec3::zero()};
  float eps_c = 0.F;

  Triangle my_triangle;
  if (triangle_idx < geometry_count) {
    my_triangle = triangles[triangle_idx];
  }

  for (uint32_t i = 0; i < query_count; i += BlockSize) {
    uint32_t num_elements_in_tile = min(BlockSize, query_count - i);

    if (num_elements_in_tile == BlockSize) {
#pragma unroll
      for (uint32_t f_idx = threadIdx.x; f_idx < BlockSize * 3;
           f_idx += BlockSize) {
        tile_q_float[f_idx] = queries_float[i * 3 + f_idx];
      }
      tile_g[threadIdx.x] = in_gradients[i + threadIdx.x];
      __syncthreads();

      if (triangle_idx < geometry_count) {
#pragma unroll 4
        for (uint32_t j = 0; j < BlockSize; ++j) {
          GeometryGradient<Triangle> contribution =
              my_triangle.gradContributionOfQuery(tile_q[j], tile_g[j],
                                                  reg_ctx);
          accum_triangle<UseKahan>(my_grad, contribution.geometry, geo_c);
          accum_float<UseKahan>(my_eps, contribution.epsilon, eps_c);
        }
      }
    } else {
      uint32_t total_floats_to_copy = num_elements_in_tile * 3;
      for (uint32_t f_idx = threadIdx.x; f_idx < total_floats_to_copy;
           f_idx += BlockSize) {
        tile_q_float[f_idx] = queries_float[i * 3 + f_idx];
      }
      uint32_t load_idx = i + threadIdx.x;
      if (load_idx < query_count) {
        tile_g[threadIdx.x] = in_gradients[load_idx];
      }
      __syncthreads();

      if (triangle_idx < geometry_count) {
        for (uint32_t j = 0; j < num_elements_in_tile; ++j) {
          GeometryGradient<Triangle> contribution =
              my_triangle.gradContributionOfQuery(tile_q[j], tile_g[j],
                                                  reg_ctx);
          accum_triangle<UseKahan>(my_grad, contribution.geometry, geo_c);
          accum_float<UseKahan>(my_eps, contribution.epsilon, eps_c);
        }
      }
    }
    __syncthreads();
  }

  if (triangle_idx < geometry_count) {
    uint32_t out_base = 9 * triangle_idx;
    out_gradients[out_base + 0] = my_grad.v0.x;
    out_gradients[out_base + 1] = my_grad.v0.y;
    out_gradients[out_base + 2] = my_grad.v0.z;
    out_gradients[out_base + 3] = my_grad.v1.x;
    out_gradients[out_base + 4] = my_grad.v1.y;
    out_gradients[out_base + 5] = my_grad.v1.z;
    out_gradients[out_base + 6] = my_grad.v2.x;
    out_gradients[out_base + 7] = my_grad.v2.y;
    out_gradients[out_base + 8] = my_grad.v2.z;
    epsilon_gradients[triangle_idx] = my_eps;
  }
}

void compute_brute_force_gradients_triangles(
    const float *grad_output, const Triangle *triangles,
    const Vec3 *queries_vec3, const uint32_t geometry_count,
    const uint32_t query_count, float *gradients, float *epsilon_gradients,
    const float epsilon, const float scene_scale, cudaStream_t compute_stream,
    bool use_kahan) {

  constexpr uint32_t threads = 256;
  uint32_t geom_blocks = (geometry_count + threads - 1) / threads;
  size_t smem_size = threads * (sizeof(Vec3) + sizeof(float));
  const auto ctx = Triangle::Context::make(epsilon, scene_scale);

  if (use_kahan) {
    gradients_brute_force_triangles_kernel<threads, true>
        <<<geom_blocks, threads, smem_size, compute_stream>>>(
            grad_output, triangles, queries_vec3, geometry_count, query_count,
            ctx, gradients, epsilon_gradients);
  } else {
    gradients_brute_force_triangles_kernel<threads, false>
        <<<geom_blocks, threads, smem_size, compute_stream>>>(
            grad_output, triangles, queries_vec3, geometry_count, query_count,
            ctx, gradients, epsilon_gradients);
  }
  CUDA_CHECK(cudaGetLastError());
}

// ===========================================================================
// Backward: query gradients (triangle soup)
// ===========================================================================

template <uint32_t BlockSize = 256, bool UseKahan = true>
__global__ void gradients_wrt_query_brute_force_triangles_kernel(
    const float *__restrict__ in_gradients,
    const Triangle *__restrict__ triangles, const Vec3 *__restrict__ queries,
    const uint32_t geometry_count, const uint32_t query_count,
    const typename Triangle::Context reg_ctx,
    float *__restrict__ out_query_gradients) {

  const uint32_t q_idx = blockIdx.x * BlockSize + threadIdx.x;

  const bool active = (q_idx < query_count);
  const Vec3 my_query = active ? queries[q_idx] : Vec3::zero();
  const float my_g = active ? in_gradients[q_idx] : 0.F;

  Vec3 q_grad = Vec3::zero();
  Vec3 compensator = Vec3::zero();

  extern __shared__ char shared_mem[];
  auto *tile = reinterpret_cast<Triangle *>(shared_mem);

  for (uint32_t i = 0; i < geometry_count; i += BlockSize) {
    const uint32_t num = min(BlockSize, geometry_count - i);

    if (threadIdx.x < num) {
      tile[threadIdx.x] = triangles[i + threadIdx.x];
    }
    __syncthreads();

    if (active) {
      if (num == BlockSize) {
#pragma unroll 4
        for (uint32_t j = 0; j < BlockSize; ++j) {
          const Vec3 contrib =
              tile[j].queryGradContribution(my_query, my_g, reg_ctx);
          accum_vec3<UseKahan>(q_grad, contrib, compensator);
        }
      } else {
        for (uint32_t j = 0; j < num; ++j) {
          const Vec3 contrib =
              tile[j].queryGradContribution(my_query, my_g, reg_ctx);
          accum_vec3<UseKahan>(q_grad, contrib, compensator);
        }
      }
    }
    __syncthreads();
  }

  if (active) {
    const uint32_t out_base = 3 * q_idx;
    out_query_gradients[out_base + 0] = q_grad.x;
    out_query_gradients[out_base + 1] = q_grad.y;
    out_query_gradients[out_base + 2] = q_grad.z;
  }
}

void compute_brute_force_query_gradients_triangles(
    const float *grad_output, const Triangle *triangles,
    const Vec3 *queries_vec3, const uint32_t geometry_count,
    const uint32_t query_count, float *query_gradients, const float epsilon,
    const float scene_scale, cudaStream_t compute_stream, bool use_kahan) {

  constexpr uint32_t threads = 256;
  const uint32_t query_blocks = (query_count + threads - 1) / threads;
  const auto ctx = Triangle::Context::make(epsilon, scene_scale);
  const size_t smem_size = threads * sizeof(Triangle);

  if (use_kahan) {
    gradients_wrt_query_brute_force_triangles_kernel<threads, true>
        <<<query_blocks, threads, smem_size, compute_stream>>>(
            grad_output, triangles, queries_vec3, geometry_count, query_count,
            ctx, query_gradients);
  } else {
    gradients_wrt_query_brute_force_triangles_kernel<threads, false>
        <<<query_blocks, threads, smem_size, compute_stream>>>(
            grad_output, triangles, queries_vec3, geometry_count, query_count,
            ctx, query_gradients);
  }
  CUDA_CHECK(cudaGetLastError());
}

// ===========================================================================
// Vertex-gradient accumulation (unchanged)
// ===========================================================================

__global__ void accumulate_vertice_gradients_kernel(
    const float *__restrict__ triangle_results,
    const uint32_t *__restrict__ triangle_indices,
    const uint32_t total_local_vertices, float *__restrict__ vertice_grads) {
  const uint32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t stride = blockDim.x * gridDim.x;

  const auto *triangle_results_float3 =
      reinterpret_cast<const float3 *>(triangle_results);

  for (uint32_t i = gtid; i < total_local_vertices; i += stride) {
    const uint32_t v_idx = triangle_indices[i];
    const float3 grad = triangle_results_float3[i];
    float *dst = &vertice_grads[static_cast<size_t>(v_idx) * 3];

    atomicAdd(&dst[0], grad.x);
    atomicAdd(&dst[1], grad.y);
    atomicAdd(&dst[2], grad.z);
  }
}

void accumulate_vertice_gradients(const float *triangle_results,
                                  const uint32_t *triangle_indices,
                                  const uint32_t triangle_count,
                                  float *vertice_grads,
                                  cudaStream_t compute_stream) {
  if (triangle_count == 0) {
    return;
  }

  const size_t triangle_vertex_count = static_cast<size_t>(triangle_count) * 3;
  const size_t threads_per_block = 256;

  const size_t blocks_needed =
      (triangle_vertex_count + threads_per_block - 1) / threads_per_block;
  const size_t max_blocks = 65535;
  const size_t actual_blocks =
      (blocks_needed < max_blocks) ? blocks_needed : max_blocks;

  accumulate_vertice_gradients_kernel<<<actual_blocks, threads_per_block, 0,
                                        compute_stream>>>(
      triangle_results, triangle_indices, triangle_vertex_count, vertice_grads);
}
