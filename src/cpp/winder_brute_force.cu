#include "geometry.h"
#include "kernels/brute_force.cuh"
#include "kernels/mesh.cuh"
#include "utils.h"
#include "vec3.h"
#include "winder_brute_force.h"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <format>
#include <stdexcept>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

#define CUDA_CHECK(expr_to_check)                                              \
  do {                                                                         \
    cudaError_t result = expr_to_check;                                        \
    if (result != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA Runtime Error: %s:%i:%d = %s\n", __FILE__,         \
              __LINE__, result, cudaGetErrorString(result));                   \
    }                                                                          \
  } while (0)

auto brute_force_point_normal_impl(
    const float *points, const float *scaled_normals, const float *queries,
    size_t geometry_count, const size_t query_count, float *winding_numbers,
    float epsilon, const int device_id, const uint64_t stream) -> void {
  ScopedCudaDevice device_scope{device_id};
  const auto *queries_vec3 = reinterpret_cast<const Vec3 *>(queries);
  const auto *points_vec3 = reinterpret_cast<const Vec3 *>(points);
  const auto *normals_vec3 = reinterpret_cast<const Vec3 *>(scaled_normals);

  cudaEvent_t start, finish;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&finish));

  // convert stream to cuda stream
  cudaStream_t compute_stream = reinterpret_cast<cudaStream_t>(stream);

  CUDA_CHECK(cudaEventRecord(start, compute_stream));

  if (epsilon < 0.F) {
    // default from 3D Reconstruction with Fast Dipole Sums
    epsilon = 1.F / 250.F;
  }

  compute_brute_force_point_normal(
      queries_vec3, points_vec3, normals_vec3, (uint32_t)query_count,
      (uint32_t)geometry_count, winding_numbers, epsilon, compute_stream);

  CUDA_CHECK(cudaEventRecord(finish, compute_stream));
  // free events
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(finish));
}

auto brute_force_mesh_impl(const float *vertices,
                           const uint32_t *triangle_indices,
                           const float *queries, const size_t geometry_count,
                           const size_t vertex_count, const size_t query_count,
                           float *winding_numbers, const int device_id,
                           const uint64_t stream) -> void {
  ScopedCudaDevice device_scope{device_id};

  cudaStream_t compute_stream = reinterpret_cast<cudaStream_t>(stream);
  auto compute_stream_policy = thrust::cuda::par.on(compute_stream);

  // Verify that index range does not exceed vertex size
  uint32_t max_index = thrust::reduce(compute_stream_policy, triangle_indices,
                                      triangle_indices + 3 * geometry_count, 0,
                                      cuda::maximum<uint32_t>());
  if (max_index >= vertex_count) {
    throw std::runtime_error(
        std::format("The triangle indices are not allowed to exceed the number "
                    "of vertices. Vertex count is {}, max index is {}.",
                    vertex_count, max_index));
  }
  // Create temporary triangles from indices
  float *triangles;
  CUDA_CHECK(cudaMallocAsync(&triangles, geometry_count * sizeof(Triangle),
                             compute_stream));

  gather_triangles(vertices, triangle_indices,
                   static_cast<uint32_t>(geometry_count), triangles,
                   compute_stream);

  // use regular triangle computation
  brute_force_triangle_impl(triangles, queries, geometry_count, query_count,
                            winding_numbers, device_id, stream);
  // release temporary triangle array
  CUDA_CHECK(cudaFreeAsync(triangles, compute_stream));
}

auto brute_force_triangle_impl(const float *triangles_float,
                               const float *queries, size_t geometry_count,
                               size_t query_count, float *winding_numbers,
                               int device_id, uint64_t stream) -> void {
  ScopedCudaDevice device_scope{device_id};
  const auto *queries_vec3 = reinterpret_cast<const Vec3 *>(queries);
  const auto *triangles = reinterpret_cast<const Triangle *>(triangles_float);

  cudaEvent_t start, finish;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&finish));

  // convert stream to cuda stream
  cudaStream_t compute_stream = reinterpret_cast<cudaStream_t>(stream);

  CUDA_CHECK(cudaEventRecord(start, compute_stream));

  compute_brute_force_triangle(queries_vec3, triangles, (uint32_t)query_count,
                               (uint32_t)geometry_count, winding_numbers,
                               compute_stream);

  CUDA_CHECK(cudaEventRecord(finish, compute_stream));
  // free events
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(finish));
}

auto brute_force_point_normal_gradient_impl(
    const float *grad_output, const float *points, const float *scaled_normals,
    const float *queries, const size_t geometry_count, const size_t query_count,
    float *gradients, float epsilon, const int device_id, const uint64_t stream)
    -> void {
  ScopedCudaDevice device_scope{device_id};

  const Vec3 *points_vec3 = reinterpret_cast<const Vec3 *>(points);
  const Vec3 *scaled_normals_vec3 =
      reinterpret_cast<const Vec3 *>(scaled_normals);
  const Vec3 *queries_vec3 = reinterpret_cast<const Vec3 *>(queries);

  if (epsilon < 0.F) {
    // default from 3D Reconstruction with Fast Dipole Sums
    epsilon = 1.F / 250.F;
  }

  cudaEvent_t start, finish;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&finish));

  // convert stream to cuda stream
  cudaStream_t compute_stream = reinterpret_cast<cudaStream_t>(stream);

  CUDA_CHECK(cudaEventRecord(start, compute_stream));

  compute_brute_force_gradients_point_normals(
      grad_output, points_vec3, scaled_normals_vec3, queries_vec3,
      (uint32_t)geometry_count, (uint32_t)query_count, epsilon, gradients,
      compute_stream);

  CUDA_CHECK(cudaEventRecord(finish, compute_stream));
  // free events
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(finish));
}

auto brute_force_triangle_gradient_impl(
    const float *grad_output, const float *triangles_float,
    const float *queries, const size_t geometry_count, const size_t query_count,
    float *gradients, const int device_id, const uint64_t stream) -> void {

  const Vec3 *queries_vec3 = reinterpret_cast<const Vec3 *>(queries);
  const auto *triangles = reinterpret_cast<const Triangle *>(triangles_float);
  ScopedCudaDevice device_scope{device_id};

  cudaEvent_t start, finish;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&finish));

  // convert stream to cuda stream
  cudaStream_t compute_stream = reinterpret_cast<cudaStream_t>(stream);

  CUDA_CHECK(cudaEventRecord(start, compute_stream));

  compute_brute_force_gradients_triangles(
      grad_output, triangles, queries_vec3, (uint32_t)geometry_count,
      (uint32_t)query_count, gradients, compute_stream);

  CUDA_CHECK(cudaEventRecord(finish, compute_stream));
  // free events
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(finish));
}

auto brute_force_mesh_gradient_impl(
    const float *grad_output, const float *vertices,
    const uint32_t *triangle_indices, const float *queries,
    const size_t geometry_count, const size_t query_count,
    const size_t vertex_count, float *vertice_grads, const int device_id,
    const uint64_t stream) -> void {

  cudaStream_t compute_stream = reinterpret_cast<cudaStream_t>(stream);
  auto compute_stream_policy = thrust::cuda::par.on(compute_stream);

  // Verify that index range does not exceed vertex size
  uint32_t max_index = thrust::reduce(compute_stream_policy, triangle_indices,
                                      triangle_indices + 3 * geometry_count, 0,
                                      cuda::maximum<uint32_t>());
  if (max_index >= vertex_count) {
    throw std::runtime_error(
        std::format("The triangle indices are not allowed to exceed the number "
                    "of vertices. Vertex count is {}, max index is {}.",
                    vertex_count, max_index));
  }
  // Create temporary triangles from indices
  float *triangles;
  CUDA_CHECK(cudaMallocAsync(&triangles, geometry_count * sizeof(Triangle),
                             compute_stream));

  gather_triangles(vertices, triangle_indices,
                   static_cast<uint32_t>(geometry_count), triangles,
                   compute_stream);

  float *triangle_gradients;
  CUDA_CHECK(cudaMallocAsync(
      &triangle_gradients, geometry_count * sizeof(Triangle), compute_stream));

  // use regular triangle computation
  brute_force_triangle_gradient_impl(grad_output, triangles, queries,
                                     geometry_count, query_count,
                                     triangle_gradients, device_id, stream);
  // release temporary triangle array
  CUDA_CHECK(cudaFreeAsync(triangles, compute_stream));

  CUDA_CHECK(cudaMemsetAsync(vertice_grads, 0, vertex_count * 3 * sizeof(float),
                             compute_stream));
  accumulate_vertice_gradients(triangle_gradients, triangle_indices,
                               static_cast<uint32_t>(geometry_count),
                               vertice_grads, compute_stream);
  CUDA_CHECK(cudaFreeAsync(triangle_gradients, compute_stream));
}
