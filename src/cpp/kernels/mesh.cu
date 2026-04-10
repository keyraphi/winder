#include "kernels/common.cuh"
#include "mesh.cuh"
#include <cstdint>
#include <driver_types.h>

__global__ void
gather_triangles_kernel(const float *__restrict__ vertices,
                        const uint32_t *__restrict__ triangle_indices,
                        float *__restrict__ triangles,
                        uint32_t triangle_count) {
  extern __shared__ uint32_t smem_indices[]; // size = blockDim.x * 3

  uint32_t tid = threadIdx.x;
  uint32_t tri_idx = blockIdx.x * blockDim.x + tid;

  uint32_t base_idx = blockIdx.x * blockDim.x * 3;
  // Coalesced load of indices into shared memory
  if (tri_idx < triangle_count) {
    smem_indices[tid] = triangle_indices[base_idx + tid];
    smem_indices[tid + blockDim.x] = triangle_indices[base_idx + blockDim.x + tid];
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

  float *out = triangles + tri_idx * 9;

  out[0] = vertices[i0 * 3 + 0];
  out[1] = vertices[i0 * 3 + 1];
  out[2] = vertices[i0 * 3 + 2];
  out[3] = vertices[i1 * 3 + 0];
  out[4] = vertices[i1 * 3 + 1];
  out[5] = vertices[i1 * 3 + 2];
  out[6] = vertices[i2 * 3 + 0];
  out[7] = vertices[i2 * 3 + 1];
  out[8] = vertices[i2 * 3 + 2];
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
