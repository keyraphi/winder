#include "bvh8.h"
#include "geometry.h"
#include "kernels/common.cuh"
#include "vec3.h"
#include "winder_cuda.h"
#include <cstddef>
#include <cstdint>
#include <driver_types.h>
#include <gtest/gtest.h>
#include <random>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

__global__ void get_root_zero_order_kernel(const BVH8Node *nodes,
                                          Vec3 *root_zero_order) {
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx == 0) {
    BVH8Node root = nodes[0];
    const float scale_factor =
        root.tailor_coefficients.get_shared_scale_factor();
    Vec3_bf16 zero_order =
        root.tailor_coefficients.get_tailor_zero_order(scale_factor);
    *root_zero_order = Vec3::from_bf16(zero_order);
  }
}

TEST(M2M, ZeroOrder) {
  size_t count = 1000000;
  thrust::host_vector<PointNormal> geometry_h(count);
  thrust::host_vector<Vec3> points_h(count);
  thrust::host_vector<Vec3> scaled_normals_h(count);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist_pos(-50,
                                                 50);
  std::normal_distribution<float> norm_dis(0.0F, 5);

  for (size_t i = 0; i < count; ++i) {
    geometry_h[i].p = Vec3{dist_pos(gen), dist_pos(gen), dist_pos(gen)};
    geometry_h[i].n = Vec3{norm_dis(gen), norm_dis(gen), norm_dis(gen)};
    points_h[i] = geometry_h[i].p;
    scaled_normals_h[i] = geometry_h[i].n;
  }

  // create tree
  thrust::device_vector<Vec3> points_d = points_h;
  thrust::device_vector<Vec3> scaled_normals_d = scaled_normals_h;
  std::unique_ptr<WinderBackend<PointNormal>> backend;
  backend = WinderBackend<PointNormal>::CreateFromPoints(
      (float *)points_d.data().get(), (float *)scaled_normals_d.data().get(),
      points_d.size(), 0);

  uint32_t bvh8_node_count;
  CUDA_CHECK(cudaMemcpy(&bvh8_node_count, backend->m_bvh8_node_count,
                        sizeof(uint32_t), cudaMemcpyDeviceToHost));

  Vec3 normal_sum = {0.F, 0.F, 0.F};
  for (const Vec3 &n : scaled_normals_h) {
    normal_sum += n;
  }

  thrust::device_vector<Vec3> root_zero_order_d(1);
  get_root_zero_order_kernel<<<1, 1>>>(backend->m_bvh8_nodes,
                                       root_zero_order_d.data().get());

  thrust::host_vector<Vec3> root_zero_order_h = root_zero_order_d;

  EXPECT_NEAR(root_zero_order_h[0].x, normal_sum.x, 1e-5);
  EXPECT_NEAR(root_zero_order_h[0].y, normal_sum.y, 1e-5);
  EXPECT_NEAR(root_zero_order_h[0].z, normal_sum.z, 1e-5);
}
