#include "brute_force.cuh"
#include "geometry.h"
#include "kernels/common.cuh"
#include "vec3.h"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <driver_types.h>

template <IsGeometry Geometry>
__global__ void compute_winding_numbers_brute_force_kernel(
    const Vec3 *queries, const Geometry *geometry, const uint32_t query_count,
    const uint32_t geometry_count, float *winding_numbers,
    const float inv_epsilon) {
  // One thread per query
  uint32_t q_idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Use shared memory to cache a tile of geometry for the whole block
  extern __shared__ char shared_mem[];
  Geometry *tile = reinterpret_cast<Geometry *>(shared_mem);

  float my_wn = 0.0F;
  float c = 0.F;
  Vec3 my_q = (q_idx < query_count) ? queries[q_idx] : Vec3{0, 0, 0};

  // Loop over geometry in tiles of blockDim.x
  for (uint32_t i = 0; i < geometry_count; i += blockDim.x) {
    uint32_t load_idx = i + threadIdx.x;

    // Cooperatively load geometry into shared memory
    if (load_idx < geometry_count) {
      tile[threadIdx.x] = Geometry::load(geometry, load_idx, geometry_count);
    }
    __syncthreads();

    // Accumulate contribution if query is in bounds
    if (q_idx < query_count) {
      uint32_t num_elements_in_tile = min(blockDim.x, geometry_count - i);
      for (uint32_t j = 0; j < num_elements_in_tile; ++j) {
        // Kahan summation
        float contrib = tile[j].contributionToQuery(my_q, inv_epsilon) - c;
        float t = my_wn + contrib;
        c = (t - my_wn) - contrib;
        my_wn = t;
      }
    }
    __syncthreads();
  }

  if (q_idx < query_count) {
    winding_numbers[q_idx] = my_wn;
  }
}

template <IsGeometry Geometry>
void compute_brute_force(const Vec3 *queries_vec3, const Geometry *geometry,
                          uint32_t query_count,
                          uint32_t geometry_count, float *winding_numbers,
                          float epsilon, cudaStream_t compute_stream) {
  if (query_count == 0) {
    return;
  }

  float inv_epsilon = 1.F / epsilon;

  uint32_t threads = 256;
  uint32_t blocks = (query_count + threads - 1) / threads;
  size_t smem_size = threads * sizeof(Geometry);
  compute_winding_numbers_brute_force_kernel<Geometry>
      <<<blocks, threads, smem_size, compute_stream>>>(
          queries_vec3, geometry, query_count, geometry_count, winding_numbers,
          inv_epsilon);
  CUDA_CHECK(cudaGetLastError());

}

template void compute_brute_force<PointNormal>(
    const Vec3 *queries_vec3, const PointNormal *geometry,
     uint32_t query_count,  uint32_t geometry_count,
    float *winding_numbers,  float epsilon, cudaStream_t compute_stream);
template void compute_brute_force<Triangle>(
    const Vec3 *queries_vec3, const Triangle *geometry,
     uint32_t query_count,  uint32_t geometry_count,
    float *winding_numbers,  float epsilon, cudaStream_t compute_stream);
