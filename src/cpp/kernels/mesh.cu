#include "geometry.h"
#include "kernels/common.cuh"
#include "mesh.cuh"
#include "vec3.h"
#include <cstddef>
#include <cstdint>
#include <driver_types.h>
#include <sys/types.h>

__global__ void
gather_triangles_kernel(const float *__restrict__ vertices,
                        const uint32_t *__restrict__ triangle_indices,
                        float *__restrict__ triangles,
                        uint32_t triangle_count) {
  extern __shared__ uint32_t smem_indices[];

  uint32_t tid = threadIdx.x;
  uint32_t tri_idx = blockIdx.x * blockDim.x + tid;

  uint32_t base_idx = blockIdx.x * blockDim.x * 3;
  uint32_t total_indices = 3 * triangle_count;

  // Coalesced load of indices into shared memory
  if (base_idx + tid < total_indices) {
    smem_indices[tid] = triangle_indices[base_idx + tid];
  }
  if (base_idx + blockDim.x + tid < total_indices) {
    smem_indices[tid + blockDim.x] =
        triangle_indices[base_idx + blockDim.x + tid];
  }
  if (base_idx + 2 * blockDim.x + tid < total_indices) {
    smem_indices[tid + 2 * blockDim.x] =
        triangle_indices[base_idx + 2 * blockDim.x + tid];
  }
  __syncthreads();

  if (tri_idx >= triangle_count) {
    return;
  }

  uint32_t i0 = smem_indices[tid * 3 + 0];
  uint32_t i1 = smem_indices[tid * 3 + 1];
  uint32_t i2 = smem_indices[tid * 3 + 2];

  const size_t offset_v0 = static_cast<size_t>(i0) * 3;
  const size_t offset_v1 = static_cast<size_t>(i1) * 3;
  const size_t offset_v2 = static_cast<size_t>(i2) * 3;

  const size_t float_count = 9;
  float values[float_count];

  values[0] = vertices[offset_v0 + 0];
  values[1] = vertices[offset_v0 + 1];
  values[2] = vertices[offset_v0 + 2];
  values[3] = vertices[offset_v1 + 0];
  values[4] = vertices[offset_v1 + 1];
  values[5] = vertices[offset_v1 + 2];
  values[6] = vertices[offset_v2 + 0];
  values[7] = vertices[offset_v2 + 1];
  values[8] = vertices[offset_v2 + 2];

  float *out = triangles + (tri_idx * float_count);

#pragma unroll
  for (size_t i = 0; i < float_count; ++i) {
    out[i] = values[i];
  }
}

void gather_triangles(const float *__restrict__ vertices,
                      const uint32_t *__restrict__ triangle_indices,
                      const uint32_t triangle_count,
                      float *__restrict__ triangles,
                      const cudaStream_t &stream) {
  if (triangle_count == 0) {
    return;
  }
  const uint32_t threads = 256;
  const uint32_t blocks = (triangle_count + threads - 1) / threads;
  const uint32_t smem_size = threads * 3 * sizeof(uint32_t);

  gather_triangles_kernel<<<blocks, threads, smem_size, stream>>>(
      vertices, triangle_indices, triangles, triangle_count);
  CUDA_CHECK(cudaGetLastError());
}

__forceinline__ __device__ Vec3 warp_reduce_vec3(unsigned mask, Vec3 val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val.x += __shfl_down_sync(mask, val.x, offset);
    val.y += __shfl_down_sync(mask, val.y, offset);
    val.z += __shfl_down_sync(mask, val.z, offset);
  }
  return val;
}

template <uint32_t BLOCK_SIZE = 256>
__global__ void accumulate_vertex_gradients_kernel(
    const float *__restrict__ triangle_gradients,
    const uint32_t *__restrict__ triangle_indices,
    const uint32_t triangle_count,
    float *__restrict__ vertice_gradients)
{
  __shared__ float smem_grads[BLOCK_SIZE * 9];
  __shared__ uint32_t smem_indices[BLOCK_SIZE * 3];

  const uint32_t block_tri_start = blockIdx.x * BLOCK_SIZE;
  const uint32_t thread_tri_idx = block_tri_start + threadIdx.x;
  const bool is_valid_thread = thread_tri_idx < triangle_count;

  const uint32_t valid_tris_in_block =
      (block_tri_start < triangle_count)
          ? min(BLOCK_SIZE, triangle_count - block_tri_start)
          : 0;

  // Coalesced Global Load through sm staging
  const uint32_t total_grads_to_load = valid_tris_in_block * 9;
  const float *block_grad_ptr = &triangle_gradients[block_tri_start * 9];
#pragma unroll
  for (uint32_t i = threadIdx.x; i < BLOCK_SIZE * 9; i += BLOCK_SIZE) {
    smem_grads[i] = (i < total_grads_to_load) ? block_grad_ptr[i] : 0.0f;
  }

  const uint32_t total_indices_to_load = valid_tris_in_block * 3;
  const uint32_t *block_idx_ptr = &triangle_indices[block_tri_start * 3];
#pragma unroll
  for (uint32_t i = threadIdx.x; i < BLOCK_SIZE * 3; i += BLOCK_SIZE) {
    smem_indices[i] =
        (i < total_indices_to_load) ? block_idx_ptr[i] : 0xFFFFFFFF;
  }

  __syncthreads();

  // bank conflict free loads from sm
  uint32_t v_idx[3] = {0, 0, 0};
  Vec3 g[3] = {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}};

  if (is_valid_thread) {
    v_idx[0] = smem_indices[threadIdx.x * 3 + 0];
    v_idx[1] = smem_indices[threadIdx.x * 3 + 1];
    v_idx[2] = smem_indices[threadIdx.x * 3 + 2];

    const float *thread_g = &smem_grads[threadIdx.x * 9];
    g[0] = {thread_g[0], thread_g[1], thread_g[2]};
    g[1] = {thread_g[3], thread_g[4], thread_g[5]};
    g[2] = {thread_g[6], thread_g[7], thread_g[8]};
  }

  auto *vert_grads = reinterpret_cast<Vec3 *>(vertice_gradients);
  #pragma unroll
  for (int corner = 0; corner < 3; ++corner) {
    unsigned active_mask = __activemask();
    uint32_t target_vert = v_idx[corner];

    // Group threads writing to the same vertex within the warp
    unsigned match_mask =
        __match_any_sync(active_mask, is_valid_thread ? target_vert : 0xFFFFFFFF);

    if (is_valid_thread) {
      int leader = __ffs(match_mask) - 1;
      int lane_id = threadIdx.x % 32;

      Vec3 aggregated = Vec3::zero();
      unsigned m = match_mask;
      while (m != 0) {
        int src_lane = __ffs(m) - 1;
        aggregated.x += __shfl_sync(match_mask, g[corner].x, src_lane);
        aggregated.y += __shfl_sync(match_mask, g[corner].y, src_lane);
        aggregated.z += __shfl_sync(match_mask, g[corner].z, src_lane);
        m &= m - 1;
      }

      // Only leader of each match performs global atomic add
      if (lane_id == leader) {
        atomicAdd(&vert_grads[target_vert].x, aggregated.x);
        atomicAdd(&vert_grads[target_vert].y, aggregated.y);
        atomicAdd(&vert_grads[target_vert].z, aggregated.z);
      }
    }
  }
}

void accumulate_vertex_gradients(const float *__restrict__ triangle_gradients,
                                 const uint32_t *__restrict__ triangle_indices,
                                 const uint32_t triangle_count,
                                 float *__restrict__ vertice_gradients,
                                 cudaStream_t stream) {
  if (triangle_count == 0) {
    return;
  }
  constexpr uint32_t threads = 256;
  const uint32_t blocks = (triangle_count + threads - 1) / threads;

  accumulate_vertex_gradients_kernel<threads><<<blocks, threads, 0, stream>>>(
      triangle_gradients, triangle_indices, triangle_count, vertice_gradients);
  CUDA_CHECK(cudaGetLastError());
}
