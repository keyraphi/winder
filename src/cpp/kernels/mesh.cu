#include "kernels/common.cuh"
#include "mesh.cuh"
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

  // FIX: Change destination pointer to write sequentially (AoS Layout)
  float *out = triangles + (tri_idx * float_count);

#pragma unroll
  for (size_t i = 0; i < float_count; ++i) {
    out[i] = values[i]; // Flat sequential write
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
