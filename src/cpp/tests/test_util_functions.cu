#include "aabb.h"
#include "center_of_mass.h"
#include "kernels/common.cuh"
#include "vec3.h"
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>

TEST(AABB, Size) {
  EXPECT_EQ(sizeof(CenterOfMass_quantized), 4);
  EXPECT_EQ(sizeof(AABB), 28);
  EXPECT_EQ(sizeof(AABB8BitApprox), 6);
}

__global__ void aabb8bitapprox_kernel(const AABB *aabb, AABB *aabb_approx,
                                      AABB *result_parent, uint32_t count) {
  uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id >= count) {
    return;
  }

  AABB parent = aabb[0];
  for (uint32_t i = 1; i < count; ++i) {
    parent = AABB::merge(parent, aabb[i]);
  }

  AABB8BitApprox aabb8bit = AABB8BitApprox::quantize_aabb(
      aabb[id], parent.min, 1.F / parent.diagonal());
  aabb_approx[id] = AABB::from_approximation(parent, aabb8bit);

  if (id == 0) {
    result_parent[0] = parent;
  }
}

TEST(AABB, AABB8BitApprox) {
  AABB aabb1;
  aabb1.min = Vec3{-1.F, -2.F, -3.F};
  aabb1.max = Vec3{3.F, 2.F, 1.F};
  aabb1.center_of_mass = {{0, 0, 0}, 0};
  AABB aabb2;
  aabb2.min = Vec3{4.F, 3.F, 2.F};
  aabb2.max = Vec3{5.F, 6.F, 7.F};
  aabb2.center_of_mass = {{0, 0, 0}, 0};

  AABB parent1 = AABB::merge(aabb1, aabb2);

  EXPECT_FLOAT_EQ(parent1.min.x, -1.F);
  EXPECT_FLOAT_EQ(parent1.min.y, -2.F);
  EXPECT_FLOAT_EQ(parent1.min.z, -3.F);
  EXPECT_FLOAT_EQ(parent1.max.x, 5.F);
  EXPECT_FLOAT_EQ(parent1.max.y, 6.F);
  EXPECT_FLOAT_EQ(parent1.max.z, 7.F);

  AABB8BitApprox approx1 = AABB8BitApprox::quantize_aabb(
      aabb1, parent1.min, 1.F / parent1.diagonal());
  AABB8BitApprox approx2 = AABB8BitApprox::quantize_aabb(
      aabb2, parent1.min, 1.F / parent1.diagonal());

  AABB aabb1_approx = AABB::from_approximation(parent1, approx1);
  AABB aabb2_approx = AABB::from_approximation(parent1, approx2);

  Vec3 expected_error = parent1.diagonal() / (2.F * 255.F);
  EXPECT_NEAR(aabb1_approx.min.x, aabb1.min.x, expected_error.x);
  EXPECT_NEAR(aabb1_approx.min.y, aabb1.min.y, expected_error.y);
  EXPECT_NEAR(aabb1_approx.min.z, aabb1.min.z, expected_error.z);
  EXPECT_NEAR(aabb2_approx.max.x, aabb2.max.x, expected_error.x);
  EXPECT_NEAR(aabb2_approx.max.y, aabb2.max.y, expected_error.y);
  EXPECT_NEAR(aabb2_approx.max.z, aabb2.max.z, expected_error.z);

  thrust::device_vector<AABB> d_aabbs(2);
  d_aabbs[0] = aabb1;
  d_aabbs[1] = aabb2;
  thrust::device_vector<AABB> d_approx(2);
  thrust::device_vector<AABB> d_parent(1);

  aabb8bitapprox_kernel<<<1, 32>>>(d_aabbs.data().get(), d_approx.data().get(),
                                   d_parent.data().get(), 2);
  CUDA_CHECK(cudaGetLastError());

  thrust::host_vector<AABB> h_approx = d_approx;
  thrust::host_vector<AABB> h_parent = d_parent;

  // Make sure device and host get same results
  EXPECT_NEAR(h_approx[0].min.x, aabb1_approx.min.x, 1e-6f);
  EXPECT_NEAR(h_approx[0].min.y, aabb1_approx.min.y, 1e-6f);
  EXPECT_NEAR(h_approx[0].min.z, aabb1_approx.min.z, 1e-6f);
  EXPECT_NEAR(h_approx[0].max.x, aabb1_approx.max.x, 1e-6f);
  EXPECT_NEAR(h_approx[0].max.y, aabb1_approx.max.y, 1e-6f);
  EXPECT_NEAR(h_approx[0].max.z, aabb1_approx.max.z, 1e-6f);
  EXPECT_NEAR(h_approx[1].min.x, aabb2_approx.min.x, 1e-6f);
  EXPECT_NEAR(h_approx[1].min.y, aabb2_approx.min.y, 1e-6f);
  EXPECT_NEAR(h_approx[1].min.z, aabb2_approx.min.z, 1e-6f);
  EXPECT_NEAR(h_approx[1].max.x, aabb2_approx.max.x, 1e-6f);
  EXPECT_NEAR(h_approx[1].max.y, aabb2_approx.max.y, 1e-6f);
  EXPECT_NEAR(h_approx[1].max.z, aabb2_approx.max.z, 1e-6f);
  EXPECT_NEAR(h_parent[0].min.x, parent1.min.x, 1e-6f);
  EXPECT_NEAR(h_parent[0].min.y, parent1.min.y, 1e-6f);
  EXPECT_NEAR(h_parent[0].min.z, parent1.min.z, 1e-6f);
  EXPECT_NEAR(h_parent[0].max.x, parent1.max.x, 1e-6f);
  EXPECT_NEAR(h_parent[0].max.y, parent1.max.y, 1e-6f);
  EXPECT_NEAR(h_parent[0].max.z, parent1.max.z, 1e-6f);
}

__global__ void
center_of_mass_quant_kernel(const AABB *aabb, const Vec3 *center_of_mass,
                            const float *max_dist, Vec3 *center_of_mass_approx,
                            float *max_dist_approx, uint32_t count) {
  uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id >= count) {
    return;
  }
  AABB box = aabb[id];
  box.center_of_mass.set(center_of_mass[id], box.min, 1.F / box.diagonal());
  box.center_of_mass.setMaxDistance(max_dist[id], box.diagonal().inv_length());

  center_of_mass_approx[id] = box.center_of_mass.get(box.min, box.diagonal());
  max_dist_approx[id] =
      box.center_of_mass.getMaxDistance(box.diagonal().length());
}

TEST(AABB, CenterOfMassQuantization) {
  AABB aabb;
  aabb.min = Vec3{-5, -5, -5};
  aabb.max = Vec3{5, 5, 5};
  Vec3 com = Vec3{0.2F, -0.5F, 3.F};
  float max_dist = 6.F;

  aabb.center_of_mass.set(com, aabb.min, 1.F / aabb.diagonal());
  aabb.center_of_mass.set(com, aabb.min, 1.F / aabb.diagonal());
  aabb.center_of_mass.setMaxDistance(max_dist, aabb.diagonal().inv_length());

  Vec3 com_approx = aabb.center_of_mass.get(aabb.min, aabb.diagonal());
  float max_dist_approx =
      aabb.center_of_mass.getMaxDistance(aabb.diagonal().length());

  Vec3 expected_error = aabb.diagonal() / (2.F * 255.F);
  EXPECT_NEAR(com_approx.x, com.x, expected_error.x);
  EXPECT_NEAR(com_approx.y, com.y, expected_error.y);
  EXPECT_NEAR(com_approx.z, com.z, expected_error.z);

  float expected_dist_error = aabb.diagonal().length() / (2.F * 255.F);
  EXPECT_NEAR(max_dist_approx, max_dist, expected_dist_error);

  thrust::device_vector<AABB> d_aabb(1);
  d_aabb[0] = aabb;
  thrust::device_vector<Vec3> d_com(1);
  d_com[0] = com;
  thrust::device_vector<float> d_max_dist(1);
  d_max_dist[0] = max_dist;
  thrust::device_vector<Vec3> d_com_approx(1);
  thrust::device_vector<float> d_max_dist_approx(1);

  center_of_mass_quant_kernel<<<1, 32>>>(
      d_aabb.data().get(), d_com.data().get(), d_max_dist.data().get(),
      d_com_approx.data().get(), d_max_dist_approx.data().get(), 1);

  thrust::host_vector<Vec3> h_com_approx = d_com_approx;
  thrust::host_vector<float> h_max_dist_approx = d_max_dist_approx;

  EXPECT_NEAR(h_com_approx[0].x, com_approx.x, 1e-6f);
  EXPECT_NEAR(h_com_approx[0].y, com_approx.y, 1e-6f);
  EXPECT_NEAR(h_com_approx[0].z, com_approx.z, 1e-6f);
  EXPECT_NEAR(h_max_dist_approx[0], max_dist_approx, 1e-6f);
}

__global__ void merge_unbalance_kernel(const AABB *a, const AABB *b,
                                       const uint32_t *n_a, const uint32_t *n_b,
                                       AABB *merged, uint32_t count) {
  uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;

  if (id >= count) {
    return;
  }

  merged[id] = AABB::merge(a[id], b[id], n_a[id], n_b[id]);
}

TEST(AABB, MergeUnbalanced) {
  uint32_t a_count = 10;
  uint32_t b_count = 1;
  AABB a;
  AABB b;

  a.min = Vec3{-10, -10, -10};
  a.max = Vec3{0, 0, 0};
  Vec3 a_com{-5, -5, -5};
  float a_max_dist = 5.F;
  a.center_of_mass.set(a_com, a.min, 1.F / a.diagonal());
  a.center_of_mass.setMaxDistance(a_max_dist, a.diagonal().inv_length());

  b.min = Vec3{0, 0, 0};
  b.max = Vec3{10, 10, 10};
  Vec3 b_com{5, 5, 5};
  float b_max_dist = 3.F;
  b.center_of_mass.set(b_com, b.min, 1.F / b.diagonal());
  b.center_of_mass.setMaxDistance(b_max_dist, b.diagonal().inv_length());

  AABB merged = AABB::merge(a, b, a_count, b_count);
  EXPECT_FLOAT_EQ(merged.min.x, -10.F);
  EXPECT_FLOAT_EQ(merged.min.y, -10.F);
  EXPECT_FLOAT_EQ(merged.min.z, -10.F);
  EXPECT_FLOAT_EQ(merged.max.x, 10.F);
  EXPECT_FLOAT_EQ(merged.max.y, 10.F);
  EXPECT_FLOAT_EQ(merged.max.z, 10.F);
  Vec3 merged_com = merged.center_of_mass.get(merged.min, merged.diagonal());
  float merged_max_dist =
      merged.center_of_mass.getMaxDistance(merged.diagonal().length());

  Vec3 expected_com = (a_com * a_count + b_com * b_count) / (a_count + b_count);
  float expected_merged_max_dist =
      fmaxf((merged_com - a_com).length() + a_max_dist,
            (merged_com - b_com).length() + b_max_dist);

  // Error propagation in merge through quantization
  Vec3 err_a = a.diagonal() / (2.F * 255.F);
  Vec3 err_b = b.diagonal() / (2.F * 255.F);
  Vec3 err_merged = merged.diagonal() / (2.F * 255.F);
  float a_factor = (float)a_count / (float)(a_count + b_count);
  float b_factor = (float)b_count / (float)(a_count + b_count);
  Vec3 expected_error = (err_a * a_factor) + (err_b * b_factor) + err_merged;
  // same for max_distance
  float expected_dist_error =
      (a.diagonal().length() / (2.F * 255.F)) * a_factor +
      (b.diagonal().length() / (2.F * 255.F)) * b_factor +
      (merged.diagonal().length() / (2.F * 255.F));
  EXPECT_NEAR(merged_com.x, expected_com.x, expected_error.x);
  EXPECT_NEAR(merged_com.y, expected_com.y, expected_error.y);
  EXPECT_NEAR(merged_com.z, expected_com.z, expected_error.z);
  EXPECT_NEAR(merged_max_dist, expected_merged_max_dist, expected_dist_error);

  // test device code
  thrust::device_vector<AABB> d_a(1);
  thrust::device_vector<AABB> d_b(1);
  thrust::device_vector<AABB> d_merged(1);
  thrust::device_vector<uint32_t> d_n_a(1);
  thrust::device_vector<uint32_t> d_n_b(1);
  d_a[0] = a;
  d_b[0] = b;
  d_n_a[0] = a_count;
  d_n_b[0] = b_count;
  merge_unbalance_kernel<<<1, 32>>>(d_a.data().get(), d_b.data().get(),
                                    d_n_a.data().get(), d_n_b.data().get(),
                                    d_merged.data().get(), 1);
  thrust::host_vector<AABB> h_merged = d_merged;
  EXPECT_NEAR(h_merged[0].min.x, merged.min.x, 1e-6f);
  EXPECT_NEAR(h_merged[0].min.y, merged.min.y, 1e-6f);
  EXPECT_NEAR(h_merged[0].min.z, merged.min.z, 1e-6f);
  EXPECT_NEAR(h_merged[0].max.x, merged.max.x, 1e-6f);
  EXPECT_NEAR(h_merged[0].max.y, merged.max.y, 1e-6f);
  EXPECT_NEAR(h_merged[0].max.z, merged.max.z, 1e-6f);
  // ensure quantized data is equal
  EXPECT_EQ(h_merged[0].center_of_mass._center_of_mass[0],
            merged.center_of_mass._center_of_mass[0]);
  EXPECT_EQ(h_merged[0].center_of_mass._center_of_mass[1],
            merged.center_of_mass._center_of_mass[1]);
  EXPECT_EQ(h_merged[0].center_of_mass._center_of_mass[2],
            merged.center_of_mass._center_of_mass[2]);
  EXPECT_EQ(h_merged[0].center_of_mass._max_distance_to_center,
            merged.center_of_mass._max_distance_to_center);
}
