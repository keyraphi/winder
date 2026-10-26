#include "brute_force.cuh"
#include "geometry.h"
#include "kernels/common.cuh"
#include "vec3.h"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <driver_types.h>

template <uint32_t BlockSize = 256>
__global__ void compute_winding_numbers_brute_force_point_normal_kernel(
    const Vec3 *__restrict__ queries, const Vec3 *__restrict__ points,
    const Vec3 *__restrict__ normals, const uint32_t query_count,
    const uint32_t geometry_count, float *__restrict__ winding_numbers,
    const PointNormalContext ctx) {

  // One thread per query
  uint32_t q_idx = blockIdx.x * BlockSize + threadIdx.x;

  // Shared memory allocation
  extern __shared__ char shared_mem[];
  Vec3 *point_tile = reinterpret_cast<Vec3 *>(shared_mem);
  Vec3 *normal_tile = point_tile + BlockSize;

  float my_wn = 0.0F;
  float c = 0.F;
  Vec3 my_q = (q_idx < query_count) ? queries[q_idx] : Vec3::zero();

  // Loop over geometry in tiles of BlockSize
  for (uint32_t i = 0; i < geometry_count; i += BlockSize) {
    uint32_t load_idx = i + threadIdx.x;

    // Cooperatively load geometry into shared memory
    if (load_idx < geometry_count) {
      point_tile[threadIdx.x] = points[load_idx];
      normal_tile[threadIdx.x] = normals[load_idx];
    }
    __syncthreads();

    // Accumulate contribution if query is in bounds
    if (q_idx < query_count) {
      uint32_t num_elements_in_tile = min(BlockSize, geometry_count - i);

      if (num_elements_in_tile == BlockSize) {
        // full loop unrolling
#pragma unroll 4
        for (uint32_t j = 0; j < BlockSize; ++j) {
          PointNormal pn{.p = point_tile[j], .n = normal_tile[j]};
          kahan_add(my_wn, pn.contributionToQuery(my_q, ctx), c);
        }
      } else {
        // Fallback for trailing partial tiles
        for (uint32_t j = 0; j < num_elements_in_tile; ++j) {
          PointNormal pn{.p = point_tile[j], .n = normal_tile[j]};
          kahan_add(my_wn, pn.contributionToQuery(my_q, ctx), c);
        }
      }
    }
    __syncthreads();
  }

  if (q_idx < query_count) {
    winding_numbers[q_idx] = my_wn;
  }
}

template <uint32_t BlockSize = 256>
__global__ void compute_winding_numbers_brute_force_triangle_kernel(
    const Vec3 *__restrict__ queries, const Triangle *__restrict__ triangles,
    const uint32_t query_count, const uint32_t geometry_count,
    const typename Triangle::Context reg_ctx,
    float *__restrict__ winding_numbers) {

  // One thread per query
  uint32_t q_idx = blockIdx.x * BlockSize + threadIdx.x;

  // Shared memory allocation
  extern __shared__ char shared_mem[];
  Triangle *tile = reinterpret_cast<Triangle *>(shared_mem);

  float my_wn = 0.0F;
  float c = 0.F;
  Vec3 my_q = (q_idx < query_count) ? queries[q_idx] : Vec3::zero();

  // Loop over geometry in tiles of BlockSize
  for (uint32_t i = 0; i < geometry_count; i += BlockSize) {
    uint32_t load_idx = i + threadIdx.x;

    // Cooperatively load geometry into shared memory
    if (load_idx < geometry_count) {
      tile[threadIdx.x] = triangles[load_idx];
    }
    __syncthreads();

    // Accumulate contribution if query is in bounds
    if (q_idx < query_count) {
      uint32_t num_elements_in_tile = min(BlockSize, geometry_count - i);

      if (num_elements_in_tile == BlockSize) {
        // full loop unrolling
#pragma unroll 4
        for (uint32_t j = 0; j < BlockSize; ++j) {
          kahan_add(my_wn, tile[j].contributionToQuery(my_q, reg_ctx), c);
        }
      } else {
        // Fallback for trailing partial tiles
        for (uint32_t j = 0; j < num_elements_in_tile; ++j) {
          kahan_add(my_wn, tile[j].contributionToQuery(my_q, reg_ctx), c);
        }
      }
    }
    __syncthreads();
  }

  if (q_idx < query_count) {
    winding_numbers[q_idx] = my_wn;
  }
}

void compute_brute_force_point_normal(
    const Vec3 *queries_vec3, const Vec3 *points_vec3, const Vec3 *normals_vec3,
    const uint32_t query_count, const uint32_t geometry_count,
    float *winding_numbers, float epsilon, float scene_scale,
    cudaStream_t compute_stream) {

  if (query_count == 0) {
    return;
  }

  const auto ctx = PointNormalContext::make(epsilon, scene_scale);

  constexpr uint32_t threads = 256;
  uint32_t blocks = (query_count + threads - 1) / threads;
  size_t smem_size = threads * 2 * sizeof(Vec3); // for points and normals

  compute_winding_numbers_brute_force_point_normal_kernel<threads>
      <<<blocks, threads, smem_size, compute_stream>>>(
          queries_vec3, points_vec3, normals_vec3, query_count, geometry_count,
          winding_numbers, ctx);

  CUDA_CHECK(cudaGetLastError());
}

void compute_brute_force_triangle(const Vec3 *queries_vec3,
                                  const Triangle *triangles,
                                  const uint32_t query_count,
                                  const uint32_t geometry_count,
                                  float *winding_numbers, const float epsilon,
                                  const float scene_scale,
                                  cudaStream_t compute_stream) {
  if (query_count == 0) {
    return;
  }

  constexpr uint32_t threads = 256;
  uint32_t blocks = (query_count + threads - 1) / threads;
  size_t smem_size = threads * sizeof(Triangle);

  const auto ctx = Triangle::Context::make(epsilon, scene_scale);

  compute_winding_numbers_brute_force_triangle_kernel<threads>
      <<<blocks, threads, smem_size, compute_stream>>>(
          queries_vec3, triangles, query_count, geometry_count, ctx,
          winding_numbers);

  CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Payload brute-force: point-normal
// ---------------------------------------------------------------------------
template <uint32_t BlockSize = 256, uint32_t MaxAcc = 32>
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

  // handle payload in chunks
  const uint32_t num_chunks = (payload_size + MaxAcc - 1) / MaxAcc;

  for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
    const uint32_t base_d = chunk * MaxAcc;
    const uint32_t d_count = min(MaxAcc, payload_size - base_d);

    // Register accumulators — laid out so unused tail (d_count<MaxAcc) is free.
    float acc[MaxAcc];
#pragma unroll
    for (uint32_t k = 0; k < MaxAcc; ++k) {
      acc[k] = 0.F;
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
          // Hot path: full tile, full accumulator chunk.
#pragma unroll 2
          for (uint32_t j = 0; j < BlockSize; ++j) {
            const PointNormal pn{.p = point_tile[j], .n = normal_tile[j]};
            const float w_j = pn.contributionToQuery(my_q, ctx);

            const float *payload_row =
                payload + static_cast<size_t>(i + j) * payload_size + base_d;

#pragma unroll
            for (uint32_t k = 0; k < MaxAcc; ++k) {
              acc[k] = fmaf(w_j, __ldg(payload_row + k), acc[k]);
            }
          }
        } else if (d_count == MaxAcc) {
          // Full accumulator chunk, ragged tile.
          for (uint32_t j = 0; j < num_elements; ++j) {
            const PointNormal pn{.p = point_tile[j], .n = normal_tile[j]};
            const float w_j = pn.contributionToQuery(my_q, ctx);

            const float *payload_row =
                payload + static_cast<size_t>(i + j) * payload_size + base_d;

#pragma unroll
            for (uint32_t k = 0; k < MaxAcc; ++k) {
              acc[k] = fmaf(w_j, __ldg(payload_row + k), acc[k]);
            }
          }
        } else {
          // Tail chunk (only the very last chunk of the last D-segment).
          for (uint32_t j = 0; j < num_elements; ++j) {
            const PointNormal pn{.p = point_tile[j], .n = normal_tile[j]};
            const float w_j = pn.contributionToQuery(my_q, ctx);

            const float *payload_row =
                payload + static_cast<size_t>(i + j) * payload_size + base_d;

            for (uint32_t k = 0; k < d_count; ++k) {
              acc[k] = fmaf(w_j, __ldg(payload_row + k), acc[k]);
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

// ---------------------------------------------------------------------------
// Payload brute-force: triangle soup
// ---------------------------------------------------------------------------
template <uint32_t BlockSize = 256, uint32_t MaxAcc = 32>
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
#pragma unroll
    for (uint32_t k = 0; k < MaxAcc; ++k)
      acc[k] = 0.F;

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

            const float *payload_row =
                payload + static_cast<size_t>(i + j) * payload_size + base_d;

#pragma unroll
            for (uint32_t k = 0; k < MaxAcc; ++k) {
              acc[k] = fmaf(w_j, __ldg(payload_row + k), acc[k]);
            }
          }
        } else if (d_count == MaxAcc) {
          for (uint32_t j = 0; j < num_elements; ++j) {
            const float w_j = tile[j].contributionToQuery(my_q, reg_ctx);

            const float *payload_row =
                payload + static_cast<size_t>(i + j) * payload_size + base_d;

#pragma unroll
            for (uint32_t k = 0; k < MaxAcc; ++k) {
              acc[k] = fmaf(w_j, __ldg(payload_row + k), acc[k]);
            }
          }
        } else {
          for (uint32_t j = 0; j < num_elements; ++j) {
            const float w_j = tile[j].contributionToQuery(my_q, reg_ctx);

            const float *payload_row =
                payload + static_cast<size_t>(i + j) * payload_size + base_d;

            for (uint32_t k = 0; k < d_count; ++k) {
              acc[k] = fmaf(w_j, __ldg(payload_row + k), acc[k]);
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

void compute_brute_force_point_normal_with_payload(
    const Vec3 *queries_vec3, const Vec3 *points_vec3, const Vec3 *normals_vec3,
    const float *payload, uint32_t query_count, uint32_t geometry_count,
    uint32_t payload_size, float *out_payload, float epsilon, float scene_scale,
    cudaStream_t compute_stream) {

  if (query_count == 0) {
    return;
  }

  const auto ctx = PointNormalContext::make(epsilon, scene_scale);

  constexpr uint32_t threads = 256;
  uint32_t blocks = (query_count + threads - 1) / threads;
  size_t smem_size = threads * 2 * sizeof(Vec3); // for points and normals

  compute_winding_numbers_brute_force_point_normal_with_payload_kernel<threads>
      <<<blocks, threads, smem_size, compute_stream>>>(
          queries_vec3, points_vec3, normals_vec3, payload, query_count,
          geometry_count, payload_size, out_payload, ctx);

  CUDA_CHECK(cudaGetLastError());
}

void compute_brute_force_triangle_with_payload(
    const Vec3 *queries_vec3, const Triangle *triangles, const float *payload,
    uint32_t query_count, uint32_t geometry_count, uint32_t payload_size,
    float *out_payload, float epsilon, float scene_scale,
    cudaStream_t compute_stream) {
  if (query_count == 0) {
    return;
  }

  constexpr uint32_t threads = 256;
  uint32_t blocks = (query_count + threads - 1) / threads;
  size_t smem_size = threads * sizeof(Triangle);

  const auto ctx = Triangle::Context::make(epsilon, scene_scale);

  compute_winding_numbers_brute_force_triangle_with_payload_kernel<threads>
      <<<blocks, threads, smem_size, compute_stream>>>(
          queries_vec3, triangles, payload, query_count, geometry_count,
          payload_size, out_payload, ctx);

  CUDA_CHECK(cudaGetLastError());
}

template <uint32_t BlockSize = 256>
__global__ void gradients_brute_force_point_normals_kernel(
    const float *__restrict__ in_gradients, const Vec3 *__restrict__ points,
    const Vec3 *__restrict__ normals, const Vec3 *__restrict__ queries,
    const uint32_t geometry_count, const uint32_t query_count,
    const PointNormalContext ctx, float *__restrict__ out_gradients,
    float *__restrict__ epsilon_gradients) {

  uint32_t pn_idx = blockIdx.x * BlockSize + threadIdx.x;

  // Shared memory allocation
  extern __shared__ char shared_mem[];
  auto *tile_q = reinterpret_cast<Vec3 *>(shared_mem);
  auto *tile_g =
      reinterpret_cast<float *>(shared_mem + BlockSize * sizeof(Vec3));

  // Cast pointers for coalesced cooperative copying
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

  // Accumulate gradient contributions tile by tile
  for (uint32_t i = 0; i < query_count; i += BlockSize) {
    uint32_t num_elements_in_tile = min(BlockSize, query_count - i);

    if (num_elements_in_tile == BlockSize) {
      // Fully unroll BlockSized loops
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

          kahan_add(my_grad, contribution.geometry, geo_c);
          kahan_add(my_eps, contribution.epsilon, eps_c);
        }
      }
    } else {
      // Fallback for the trailing partial tile
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

      // Fallback for the trailing partial tile
      if (pn_idx < geometry_count) {
        for (uint32_t j = 0; j < num_elements_in_tile; ++j) {
          GeometryGradient<PointNormal> contribution =
              my_point_normal.gradContributionOfQuery(tile_q[j], tile_g[j],
                                                      ctx);

          kahan_add(my_grad, contribution.geometry, geo_c);
          kahan_add(my_eps, contribution.epsilon, eps_c);
        }
      }
    }
    __syncthreads();
  }

  // Write out result
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
    cudaStream_t compute_stream) {

  constexpr uint32_t threads = 256;
  uint32_t geom_blocks = (geometry_count + threads - 1) / threads;

  const auto ctx = PointNormalContext::make(epsilon, scene_scale);
  size_t smem_size = threads * (sizeof(Vec3) + sizeof(float));

  gradients_brute_force_point_normals_kernel<threads>
      <<<geom_blocks, threads, smem_size, compute_stream>>>(
          grad_output, points_vec3, scaled_normals_vec3, queries_vec3,
          geometry_count, query_count, ctx, gradients, epsilon_gradients);

  CUDA_CHECK(cudaGetLastError());
}

template <uint32_t BlockSize = 256>
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
          kahan_add(q_grad, contrib, compensator);
        }
      } else {
        for (uint32_t j = 0; j < num; ++j) {
          const PointNormal pn{.p = tile_p[j], .n = tile_n[j]};
          const Vec3 contrib = pn.queryGradContribution(my_query, my_g, ctx);
          kahan_add(q_grad, contrib, compensator);
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
    cudaStream_t compute_stream) {

  constexpr uint32_t threads = 256;
  const uint32_t query_blocks = (query_count + threads - 1) / threads;

  const auto ctx = PointNormalContext::make(epsilon, scene_scale);
  const size_t smem_size = threads * (sizeof(Vec3) + sizeof(Vec3));

  gradients_wrt_query_brute_force_point_normals_kernel<threads>
      <<<query_blocks, threads, smem_size, compute_stream>>>(
          grad_output, points_vec3, scaled_normals_vec3, queries_vec3,
          geometry_count, query_count, ctx, query_gradients);

  CUDA_CHECK(cudaGetLastError());
}

template <uint32_t BlockSize = 256>
__global__ void gradients_brute_force_triangles_kernel(
    const float *__restrict__ in_gradients,
    const Triangle *__restrict__ triangles, const Vec3 *__restrict__ queries,
    const uint32_t geometry_count, const uint32_t query_count,
    const typename Triangle::Context reg_ctx, float *__restrict__ out_gradients,
    float *__restrict__ epsilon_gradients) {

  uint32_t triangle_idx = blockIdx.x * BlockSize + threadIdx.x;

  // Shared memory allocation
  extern __shared__ char shared_mem[];
  auto *tile_q = reinterpret_cast<Vec3 *>(shared_mem);
  auto *tile_g =
      reinterpret_cast<float *>(shared_mem + BlockSize * sizeof(Vec3));

  // Cast pointers for coalesced cooperative copying
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

  // Accumulate gradient contributions tile by tile
  for (uint32_t i = 0; i < query_count; i += BlockSize) {
    uint32_t num_elements_in_tile = min(BlockSize, query_count - i);

    if (num_elements_in_tile == BlockSize) {
      // Fully unroll BlockSized loops
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
          kahan_add(my_grad, contribution.geometry, geo_c);
          kahan_add(my_eps, contribution.epsilon, eps_c);
        }
      }
    } else {
      // Fallback for the trailing partial tile
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

      // Fallback for the trailing partial tile
      if (triangle_idx < geometry_count) {
        for (uint32_t j = 0; j < num_elements_in_tile; ++j) {
          GeometryGradient<Triangle> contribution =
              my_triangle.gradContributionOfQuery(tile_q[j], tile_g[j],
                                                  reg_ctx);
          kahan_add(my_grad, contribution.geometry, geo_c);
          kahan_add(my_eps, contribution.epsilon, eps_c);
        }
      }
    }
    __syncthreads();
  }

  // Write out result
  if (triangle_idx < geometry_count) {
    uint32_t out_base = 9 * triangle_idx;
    out_gradients[out_base] = my_grad.v0.x;
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
    const float epsilon, const float scene_scale, cudaStream_t compute_stream) {

  constexpr uint32_t threads = 256;
  uint32_t geom_blocks = (geometry_count + threads - 1) / threads;

  size_t smem_size = threads * (sizeof(Vec3) + sizeof(float));

  const auto ctx = Triangle::Context::make(epsilon, scene_scale);

  gradients_brute_force_triangles_kernel<threads>
      <<<geom_blocks, threads, smem_size, compute_stream>>>(
          grad_output, triangles, queries_vec3, geometry_count, query_count,
          ctx, gradients, epsilon_gradients);

  CUDA_CHECK(cudaGetLastError());
}

template <uint32_t BlockSize = 256>
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

    // All threads participate. For the full-tile case, threads each load
    // one Triangle; for the partial tile, only threads < num load.
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
          kahan_add(q_grad, contrib, compensator);
        }
      } else {
        for (uint32_t j = 0; j < num; ++j) {
          const Vec3 contrib =
              tile[j].queryGradContribution(my_query, my_g, reg_ctx);
          kahan_add(q_grad, contrib, compensator);
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
    const float scene_scale, cudaStream_t compute_stream) {

  constexpr uint32_t threads = 256;
  const uint32_t query_blocks = (query_count + threads - 1) / threads;

  const auto ctx = Triangle::Context::make(epsilon, scene_scale);
  const size_t smem_size = threads * sizeof(Triangle);

  gradients_wrt_query_brute_force_triangles_kernel<threads>
      <<<query_blocks, threads, smem_size, compute_stream>>>(
          grad_output, triangles, queries_vec3, geometry_count, query_count,
          ctx, query_gradients);

  CUDA_CHECK(cudaGetLastError());
}

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
    // (dx, dy, dz) gradient vector for triangle vertex
    const float3 grad = triangle_results_float3[i];
    // Pointer to target vertex gradient in output array
    float *dst = &vertice_grads[static_cast<size_t>(v_idx) * 3];

    // Hardware-accelerated L2 atomic additions
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

  const size_t triangle_vertex_count = triangle_count * 3;
  const size_t threads_per_block = 256;

  // Calculate grid size
  const size_t blocks_needed =
      (triangle_vertex_count + threads_per_block - 1) / threads_per_block;
  const size_t max_blocks = 65535;
  const size_t actual_blocks =
      (blocks_needed < max_blocks) ? blocks_needed : max_blocks;

  accumulate_vertice_gradients_kernel<<<actual_blocks, threads_per_block, 0,
                                        compute_stream>>>(
      triangle_results, triangle_indices, triangle_vertex_count, vertice_grads);
}
