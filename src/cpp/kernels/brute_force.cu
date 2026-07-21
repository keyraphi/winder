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
    const float inv_epsilon) {

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
          float contrib = pn.contributionToQuery(my_q, inv_epsilon) - c;
          float t = my_wn + contrib;
          c = (t - my_wn) - contrib;
          my_wn = t;
        }
      } else {
        // Fallback for trailing partial tiles
        for (uint32_t j = 0; j < num_elements_in_tile; ++j) {
          PointNormal pn{.p = point_tile[j], .n = normal_tile[j]};
          float contrib = pn.contributionToQuery(my_q, inv_epsilon) - c;
          float t = my_wn + contrib;
          c = (t - my_wn) - contrib;
          my_wn = t;
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
          float contrib = tile[j].contributionToQuery(my_q, 0.F) - c;
          float t = my_wn + contrib;
          c = (t - my_wn) - contrib;
          my_wn = t;
        }
      } else {
        // Fallback for trailing partial tiles
        for (uint32_t j = 0; j < num_elements_in_tile; ++j) {
          float contrib = tile[j].contributionToQuery(my_q, 0.F) - c;
          float t = my_wn + contrib;
          c = (t - my_wn) - contrib;
          my_wn = t;
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
    float *winding_numbers, float epsilon, cudaStream_t compute_stream) {

  if (query_count == 0) {
    return;
  }

  float inv_epsilon = 1.F / epsilon;

  constexpr uint32_t threads = 256;
  uint32_t blocks = (query_count + threads - 1) / threads;
  size_t smem_size = threads * 2 * sizeof(Vec3); // for points and normals

  compute_winding_numbers_brute_force_point_normal_kernel<threads>
      <<<blocks, threads, smem_size, compute_stream>>>(
          queries_vec3, points_vec3, normals_vec3, query_count, geometry_count,
          winding_numbers, inv_epsilon);

  CUDA_CHECK(cudaGetLastError());
}

void compute_brute_force_triangle(const Vec3 *queries_vec3,
                                  const Triangle *triangles,
                                  const uint32_t query_count,
                                  const uint32_t geometry_count,
                                  float *winding_numbers,
                                  cudaStream_t compute_stream) {
  if (query_count == 0) {
    return;
  }

  constexpr uint32_t threads = 256;
  uint32_t blocks = (query_count + threads - 1) / threads;
  size_t smem_size = threads * sizeof(Triangle);

  compute_winding_numbers_brute_force_triangle_kernel<threads>
      <<<blocks, threads, smem_size, compute_stream>>>(
          queries_vec3, triangles, query_count, geometry_count,
          winding_numbers);

  CUDA_CHECK(cudaGetLastError());
}

template <uint32_t BlockSize = 256>
__global__ void gradients_brute_force_point_normals_kernel(
    const Vec3 *__restrict__ queries, const float *__restrict__ in_gradients,
    SoAView<PointNormal> geometry,
    const uint32_t *__restrict__ mapping_to_internal,
    const uint32_t query_count, const uint32_t geometry_count,
    const float inv_epsilon, float *__restrict__ out_gradients) {

  uint32_t pn_idx = blockIdx.x * BlockSize + threadIdx.x;

  // Shared memory allocation
  extern __shared__ char shared_mem[];
  auto *tile_q = reinterpret_cast<Vec3 *>(shared_mem);
  auto *tile_g =
      reinterpret_cast<float *>(shared_mem + BlockSize * sizeof(Vec3));

  // Cast pointers for coalesced cooperative copying
  const float *queries_float = reinterpret_cast<const float *>(queries);
  float *tile_q_float = reinterpret_cast<float *>(tile_q);

  Vec3 my_p_grad = Vec3::zero();
  Vec3 my_n_grad = Vec3::zero();

  Vec3 c_p = Vec3::zero();
  Vec3 c_n = Vec3::zero();

  // Loop-Invariant Constants
  constexpr float INV_PI_1_5 = 0.179587122F; // 1.0 / (pi^1.5)
  const float inv_epsilon3 = inv_epsilon * inv_epsilon * inv_epsilon;
  const float reg_term_const = inv_epsilon3 * INV_PI_1_5;
  const float near_field_g_denum = (INV_PI_1_5 / 3.F) * inv_epsilon3;

  PointNormal my_point_normal =
      PointNormal::load(geometry, pn_idx, geometry_count);

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
          Vec3 contrib_p;
          Vec3 contrib_n;

          my_point_normal.gradContributionOfQuery(
              tile_q[j], tile_g[j], inv_epsilon, reg_term_const,
              near_field_g_denum, contrib_p, contrib_n);

          contrib_p = contrib_p - c_p;
          contrib_n = contrib_n - c_n;
          Vec3 t_p = my_p_grad + contrib_p;
          Vec3 t_n = my_n_grad + contrib_n;
          c_p = (t_p - my_p_grad) - contrib_p;
          c_n = (t_n - my_n_grad) - contrib_n;
          my_p_grad = t_p;
          my_n_grad = t_n;
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
          Vec3 contrib_p;
          Vec3 contrib_n;

          my_point_normal.gradContributionOfQuery(
              tile_q[j], tile_g[j], inv_epsilon, reg_term_const,
              near_field_g_denum, contrib_p, contrib_n);

          contrib_p = contrib_p - c_p;
          contrib_n = contrib_n - c_n;
          Vec3 t_p = my_p_grad + contrib_p;
          Vec3 t_n = my_n_grad + contrib_n;
          c_p = (t_p - my_p_grad) - contrib_p;
          c_n = (t_n - my_n_grad) - contrib_n;
          my_p_grad = t_p;
          my_n_grad = t_n;
        }
      }
    }
    __syncthreads();
  }

  // Write out result
  if (pn_idx < geometry_count) {
    uint32_t orig_idx = mapping_to_internal[pn_idx];
    uint32_t out_base = 6 * orig_idx;
    out_gradients[out_base] = my_n_grad.x;
    out_gradients[out_base + 1] = my_n_grad.y;
    out_gradients[out_base + 2] = my_n_grad.z;
    out_gradients[out_base + 3] = my_p_grad.x;
    out_gradients[out_base + 4] = my_p_grad.y;
    out_gradients[out_base + 5] = my_p_grad.z;
  }
}

void compute_brute_force_gradients_point_normals(
    const Vec3 *queries_vec3, const float *grad_output, const float *geometry,
    const uint32_t *mapping_to_internal, const uint32_t query_count,
    const uint32_t geometry_count, const float epsilon, float *gradients,
    cudaStream_t compute_stream) {

  constexpr uint32_t threads = 256;
  uint32_t geom_blocks = (geometry_count + threads - 1) / threads;

  const float inv_epsilon = 1.F / epsilon;
  size_t smem_size = threads * (sizeof(Vec3) + sizeof(float));

  gradients_brute_force_point_normals_kernel<threads>
      <<<geom_blocks, threads, smem_size, compute_stream>>>(
          queries_vec3, grad_output,
          SoAView<PointNormal>{geometry, geometry_count}, mapping_to_internal,
          query_count, geometry_count, inv_epsilon, gradients);

  CUDA_CHECK(cudaGetLastError());
}

template <uint32_t BlockSize = 256>
__global__ void gradients_brute_force_triangles_kernel(
    const Vec3 *__restrict__ queries, const float *__restrict__ in_gradients,
    SoAView<Triangle> geometry,
    const uint32_t *__restrict__ mapping_to_internal,
    const uint32_t query_count, const uint32_t geometry_count,
    float *__restrict__ out_gradients) {

  uint32_t pn_idx = blockIdx.x * BlockSize + threadIdx.x;

  // Shared memory allocation
  extern __shared__ char shared_mem[];
  auto *tile_q = reinterpret_cast<Vec3 *>(shared_mem);
  auto *tile_g =
      reinterpret_cast<float *>(shared_mem + BlockSize * sizeof(Vec3));

  // Cast pointers for coalesced cooperative copying
  const float *queries_float = reinterpret_cast<const float *>(queries);
  float *tile_q_float = reinterpret_cast<float *>(tile_q);

  Vec3 my_v0_grad = Vec3::zero();
  Vec3 my_v1_grad = Vec3::zero();
  Vec3 my_v2_grad = Vec3::zero();

  Vec3 c_v0 = Vec3::zero();
  Vec3 c_v1 = Vec3::zero();
  Vec3 c_v2 = Vec3::zero();

  Triangle my_triangle = Triangle::load(geometry, pn_idx, geometry_count);

  // Accumulate gradient contributions tile by tile
  for (uint32_t i = 0; i < query_count; i += BlockSize) {
    uint32_t num_elements_in_tile = min(BlockSize, query_count - i);

    if (num_elements_in_tile == BlockSize) {
      // Fully unroll BlockSized loops
      //
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
          Vec3 contrib_v0;
          Vec3 contrib_v1;
          Vec3 contrib_v2;

          my_triangle.gradContributionOfQuery(tile_q[j], tile_g[j], contrib_v0,
                                              contrib_v1, contrib_v2);

          contrib_v0 = contrib_v0 - c_v0;
          contrib_v1 = contrib_v1 - c_v1;
          contrib_v2 = contrib_v2 - c_v2;
          Vec3 t_v0 = my_v0_grad + contrib_v0;
          Vec3 t_v1 = my_v1_grad + contrib_v1;
          Vec3 t_v2 = my_v2_grad + contrib_v2;
          c_v0 = (t_v0 - my_v0_grad) - contrib_v0;
          c_v1 = (t_v1 - my_v1_grad) - contrib_v1;
          c_v2 = (t_v2 - my_v2_grad) - contrib_v2;
          my_v0_grad = t_v0;
          my_v1_grad = t_v1;
          my_v2_grad = t_v2;
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
          Vec3 contrib_v0;
          Vec3 contrib_v1;
          Vec3 contrib_v2;

          my_triangle.gradContributionOfQuery(tile_q[j], tile_g[j], contrib_v0,
                                              contrib_v1, contrib_v2);

          contrib_v0 = contrib_v0 - c_v0;
          contrib_v1 = contrib_v1 - c_v1;
          contrib_v2 = contrib_v2 - c_v2;
          Vec3 t_v0 = my_v0_grad + contrib_v0;
          Vec3 t_v1 = my_v1_grad + contrib_v1;
          Vec3 t_v2 = my_v2_grad + contrib_v2;
          c_v0 = (t_v0 - my_v0_grad) - contrib_v0;
          c_v1 = (t_v1 - my_v1_grad) - contrib_v1;
          c_v2 = (t_v2 - my_v2_grad) - contrib_v2;
          my_v0_grad = t_v0;
          my_v1_grad = t_v1;
          my_v2_grad = t_v2;
        }
      }
    }
    __syncthreads();
  }

  // Write out result
  if (pn_idx < geometry_count) {
    uint32_t orig_idx = mapping_to_internal[pn_idx];
    uint32_t out_base = 9 * orig_idx;
    out_gradients[out_base] = my_v0_grad.x;
    out_gradients[out_base + 1] = my_v0_grad.y;
    out_gradients[out_base + 2] = my_v0_grad.z;
    out_gradients[out_base + 3] = my_v1_grad.x;
    out_gradients[out_base + 4] = my_v1_grad.y;
    out_gradients[out_base + 5] = my_v1_grad.z;
    out_gradients[out_base + 6] = my_v2_grad.x;
    out_gradients[out_base + 7] = my_v2_grad.y;
    out_gradients[out_base + 8] = my_v2_grad.z;
  }
}

void compute_brute_force_gradients_triangles(
    const Vec3 *queries_vec3, const float *grad_output, const float *geometry,
    const uint32_t *mapping_to_internal, const uint32_t query_count,
    const uint32_t geometry_count, float *gradients,
    cudaStream_t compute_stream) {

  constexpr uint32_t threads = 256;
  uint32_t geom_blocks = (geometry_count + threads - 1) / threads;

  size_t smem_size = threads * (sizeof(Vec3) + sizeof(float));

  gradients_brute_force_triangles_kernel<threads>
      <<<geom_blocks, threads, smem_size, compute_stream>>>(
          queries_vec3, grad_output,
          SoAView<Triangle>{geometry, geometry_count}, mapping_to_internal,
          query_count, geometry_count, gradients);

  CUDA_CHECK(cudaGetLastError());
}
