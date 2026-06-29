#include "aabb.h"
#include "binary_node.h"
#include "bvh8.h"
#include "geometry.h"
#include "kernels/build_binary_tree.cuh"
#include "kernels/common.cuh"
#include "kernels/node_approx.cuh"
#include "mat3x3.h"
#include "tailor_coefficients.h"
#include "tensor3.h"
#include "vec3.h"
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <gtest/gtest.h>
#include <random>
#include <sys/types.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>

TEST(Sizes, AABB) {
  EXPECT_EQ(sizeof(AABB), 28);
}
TEST(Sizes, BinaryNode) { EXPECT_EQ(sizeof(BinaryNode), 8); }
TEST(Sizes, BVH8Node) { EXPECT_EQ(sizeof(BVH8Node), 32); }
TEST(Sizes, TailorCoefficients) {
  EXPECT_EQ(sizeof(TailorCoefficientsF16), 64);
}
TEST(Sizes, Geometry) {
  EXPECT_EQ(sizeof(Triangle), 36);
  EXPECT_EQ(sizeof(PointNormal), 24);
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

  a.min = Vec3_f16{-10, -10, -10};
  a.max = Vec3_f16{0, 0, 0};
  Vec3 a_com{-5, -5, -5};
  half a_max_dist = 5.F;
  a.center_of_mass = a_com;
  a.max_distance = a_max_dist;

  b.min = Vec3_f16{0, 0, 0};
  b.max = Vec3_f16{10, 10, 10};
  Vec3 b_com{5, 5, 5};
  half b_max_dist = 3.F;
  b.center_of_mass = b_com;
  b.max_distance = b_max_dist;

  AABB merged = AABB::merge(a, b, a_count, b_count);
  EXPECT_FLOAT_EQ(merged.min.x, -10.F);
  EXPECT_FLOAT_EQ(merged.min.y, -10.F);
  EXPECT_FLOAT_EQ(merged.min.z, -10.F);
  EXPECT_FLOAT_EQ(merged.max.x, 10.F);
  EXPECT_FLOAT_EQ(merged.max.y, 10.F);
  EXPECT_FLOAT_EQ(merged.max.z, 10.F);
  Vec3 merged_com = merged.center_of_mass;

  Vec3 expected_com = (a_com * a_count + b_com * b_count) / (a_count + b_count);

  // Error propagation in merge through quantization
  Vec3 err_a = Vec3::from_f16(a.diagonal()) / (2.F * 255.F);
  Vec3 err_b = Vec3::from_f16(b.diagonal()) / (2.F * 255.F);
  Vec3 err_merged = Vec3::from_f16(merged.diagonal()) / (2.F * 255.F);
  float a_factor = (float)a_count / (float)(a_count + b_count);
  float b_factor = (float)b_count / (float)(a_count + b_count);
  Vec3 expected_error = (err_a * a_factor) + (err_b * b_factor) + err_merged;
  // same for max_distance
  EXPECT_NEAR(merged_com.x, expected_com.x, expected_error.x);
  EXPECT_NEAR(merged_com.y, expected_com.y, expected_error.y);
  EXPECT_NEAR(merged_com.z, expected_com.z, expected_error.z);

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
  CUDA_CHECK(cudaGetLastError());
  thrust::host_vector<AABB> h_merged = d_merged;
  EXPECT_NEAR(h_merged[0].min.x, merged.min.x, 1e-6f);
  EXPECT_NEAR(h_merged[0].min.y, merged.min.y, 1e-6f);
  EXPECT_NEAR(h_merged[0].min.z, merged.min.z, 1e-6f);
  EXPECT_NEAR(h_merged[0].max.x, merged.max.x, 1e-6f);
  EXPECT_NEAR(h_merged[0].max.y, merged.max.y, 1e-6f);
  EXPECT_NEAR(h_merged[0].max.z, merged.max.z, 1e-6f);
}

template <IsGeometry Geometry>
__global__ void tailor_terms_kernel(const Geometry *g, const Vec3 *center,
                                    Vec3 *zero_order, Mat3x3 *first_order,
                                    Tensor3_compressed *second_order,
                                    const uint32_t count) {
  uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id >= count) {
    return;
  }
  g[id].get_tailor_terms(center[id], true, zero_order[id], first_order[id],
                         second_order[id]);
}

TEST(TailorTerms, PointNormal) {
  PointNormal pn1{{1, 2, 3}, {5, 6, 7}};
  PointNormal pn2{{5, 6, 7}, {1, 2, 3}};
  Vec3 center1{0, 0, 0};
  Vec3 center2{10, -10, 4};

  thrust::device_vector<Vec3> zero_order(2);
  thrust::device_vector<Mat3x3> first_order(2);
  thrust::device_vector<Tensor3_compressed> second_order(2);
  thrust::device_vector<PointNormal> pn(2);
  thrust::device_vector<Vec3> d_center(2);
  pn[0] = pn1;
  pn[1] = pn2;
  d_center[0] = center1;
  d_center[1] = center2;

  tailor_terms_kernel<PointNormal><<<1, 32>>>(
      pn.data().get(), d_center.data().get(), zero_order.data().get(),
      first_order.data().get(), second_order.data().get(), 2);
  CUDA_CHECK(cudaGetLastError());

  thrust::host_vector<Vec3> h_zero = zero_order;
  thrust::host_vector<Mat3x3> h_first = first_order;
  thrust::host_vector<Tensor3_compressed> h_second = second_order;

  // zero
  EXPECT_FLOAT_EQ(h_zero[0].x, pn1.n.x);
  EXPECT_FLOAT_EQ(h_zero[0].y, pn1.n.y);
  EXPECT_FLOAT_EQ(h_zero[0].z, pn1.n.z);
  EXPECT_FLOAT_EQ(h_zero[1].x, pn2.n.x);
  EXPECT_FLOAT_EQ(h_zero[1].y, pn2.n.y);
  EXPECT_FLOAT_EQ(h_zero[1].z, pn2.n.z);

  // first
  Vec3 d1 = pn1.p - center1;
  Vec3 d2 = pn2.p - center2;
  Mat3x3 first_gt1 = d1.outer_product(pn1.n);
  Mat3x3 first_gt2 = d2.outer_product(pn2.n);
  for (int i = 0; i < 9; ++i) {
    EXPECT_FLOAT_EQ(h_first[0].data[i], first_gt1.data[i]);
    EXPECT_FLOAT_EQ(h_first[1].data[i], first_gt2.data[i]);
  }

  // second
  Tensor3 second_gt1 =
      Tensor3::from_outer_product(1.F / 2.F * d1.outer_product(d1), pn1.n);
  Tensor3 second_gt2 =
      Tensor3::from_outer_product(1.F / 2.F * d2.outer_product(d2), pn2.n);
  Tensor3 second1 = h_second[0].uncompress();
  Tensor3 second2 = h_second[1].uncompress();
  for (int i = 0; i < 27; ++i) {
    EXPECT_FLOAT_EQ(second1.data[i], second_gt1.data[i]);
    EXPECT_FLOAT_EQ(second2.data[i], second_gt2.data[i]);
  }
}

TEST(TailorTerms, Triangle) {
  Triangle t1{{1, 2, 3}, {5, 6, 7}, {-1, -2, -3}};
  Triangle t2{{5, 6, 7}, {1, 2, 3}, {-5, -6, -7}};
  Vec3 center1{0, 0, 0};
  Vec3 center2{10, -10, 4};

  thrust::device_vector<Vec3> zero_order(2);
  thrust::device_vector<Mat3x3> first_order(2);
  thrust::device_vector<Tensor3_compressed> second_order(2);
  thrust::device_vector<Triangle> t(2);
  thrust::device_vector<Vec3> d_center(2);
  t[0] = t1;
  t[1] = t2;
  d_center[0] = center1;
  d_center[1] = center2;

  tailor_terms_kernel<Triangle><<<1, 32>>>(
      t.data().get(), d_center.data().get(), zero_order.data().get(),
      first_order.data().get(), second_order.data().get(), 2);
  CUDA_CHECK(cudaGetLastError());

  thrust::host_vector<Vec3> h_zero = zero_order;
  thrust::host_vector<Mat3x3> h_first = first_order;
  thrust::host_vector<Tensor3_compressed> h_second = second_order;

  // zero
  Vec3 n1 = Vec3::cross(t1.v1 - t1.v0, t1.v2 - t1.v0) * 0.5f;
  Vec3 n2 = Vec3::cross(t2.v1 - t2.v0, t2.v2 - t2.v0) * 0.5f;
  EXPECT_FLOAT_EQ(h_zero[0].x, n1.x);
  EXPECT_FLOAT_EQ(h_zero[0].y, n1.y);
  EXPECT_FLOAT_EQ(h_zero[0].z, n1.z);
  EXPECT_FLOAT_EQ(h_zero[1].x, n2.x);
  EXPECT_FLOAT_EQ(h_zero[1].y, n2.y);
  EXPECT_FLOAT_EQ(h_zero[1].z, n2.z);

  // first
  Vec3 t1_center = (t1.v0 + t1.v1 + t1.v2) / 3.F;
  Vec3 t2_center = (t2.v0 + t2.v1 + t2.v2) / 3.F;
  Vec3 d1 = t1_center - center1;
  Vec3 d2 = t2_center - center2;
  Mat3x3 first_gt1 = d1.outer_product(n1);
  Mat3x3 first_gt2 = d2.outer_product(n2);
  for (int i = 0; i < 9; ++i) {
    EXPECT_FLOAT_EQ(h_first[0].data[i], first_gt1.data[i]);
    EXPECT_FLOAT_EQ(h_first[1].data[i], first_gt2.data[i]);
  }

  // second
  Mat3x3 Ct_1 = 1.F / 3.F *
                    (1.F / 2.F * (t1.v0 + t1.v1) - center1)
                        .outer_product(1.F / 2.F * (t1.v0 + t1.v1) - center1) +
                1.F / 3.F *
                    (1.F / 2.F * (t1.v1 + t1.v2) - center1)
                        .outer_product(1.F / 2.F * (t1.v1 + t1.v2) - center1) +
                1.F / 3.F *
                    (1.F / 2.F * (t1.v2 + t1.v0) - center1)
                        .outer_product(1.F / 2.F * (t1.v2 + t1.v0) - center1);
  Mat3x3 Ct_2 = 1.F / 3.F *
                    (1.F / 2.F * (t2.v0 + t2.v1) - center2)
                        .outer_product(1.F / 2.F * (t2.v0 + t2.v1) - center2) +
                1.F / 3.F *
                    (1.F / 2.F * (t2.v1 + t2.v2) - center2)
                        .outer_product(1.F / 2.F * (t2.v1 + t2.v2) - center2) +
                1.F / 3.F *
                    (1.F / 2.F * (t2.v2 + t2.v0) - center2)
                        .outer_product(1.F / 2.F * (t2.v2 + t2.v0) - center2);
  Tensor3 second_gt1 = Tensor3::from_outer_product(Ct_1, n1);
  Tensor3 second_gt2 = Tensor3::from_outer_product(Ct_2, n2);
  for (int i = 0; i < 27; ++i) {
    second_gt1.data[i] *= 0.5f;
    second_gt2.data[i] *= 0.5f;
  }
  Tensor3 second1 = h_second[0].uncompress();
  Tensor3 second2 = h_second[1].uncompress();
  for (int i = 0; i < 27; ++i) {
    EXPECT_FLOAT_EQ(second1.data[i], second_gt1.data[i]);
    EXPECT_FLOAT_EQ(second2.data[i], second_gt2.data[i]);
  }
}

TEST(GeometryWindingNumberContribution, PointNormal) {
  // 1. Setup a PointNormal at the origin, pointing UP
  PointNormal pn;
  pn.p = Vec3{0, 0, 0};
  pn.n = Vec3{0, 0, 1};      // Unit normal
  float inv_epsilon = 10.0f; // epsilon = 0.1

  // 2. Trivial Test: Query point far away on the Z-axis
  // For r >> epsilon, contribution should be n.dot(d) / (4 * pi * r^3)
  // r = p - query = {0, 0, 0} - {0, 0, 10}  = {0, 0, -10}
  // expected = -10 / (4 * pi * 1000) = -1 / (400 * pi)
  Vec3 far_query{0, 0, 10};
  float far_contrib = pn.contributionToQuery(far_query, inv_epsilon);
  float expected_far = (-1.0f / (4.0f * M_PI * 100.0f));
  EXPECT_NEAR(far_contrib, expected_far, 1e-6f);

  // 3. Singularity Test: Query exactly at the source
  // According to the paper/code, if d is 0, we return 0.
  float zero_contrib = pn.contributionToQuery(Vec3{0, 0, 0}, inv_epsilon);
  EXPECT_FLOAT_EQ(zero_contrib, 0.0f);

  // 4. Regularization Test: Query very close (t < 0.1)
  // s_over_dist3 should approach 4 / (3 * sqrt(pi) * epsilon^3)
  // result = n.dot(d) * (1/4pi) * (4 / (3 * sqrt(pi) * eps^3))
  float small_dist = 0.001f; // t = 0.01
  Vec3 near_query{0, 0, small_dist};
  float near_contrib = pn.contributionToQuery(near_query, inv_epsilon);

  float expected_near =
      (pn.n.dot(pn.p - near_query)) * (1.0f / (4.0f * M_PI)) *
      (FOUR_OVER_3SQRT_PI * (inv_epsilon * inv_epsilon * inv_epsilon));
  EXPECT_NEAR(near_contrib, expected_near, 1e-7f);
}

TEST(GeometryWindingNumberContribution, Triangle) {
  Triangle tri;
  // An axis-aligned right triangle
  // This represents 1/8th of a sphere's surface area when viewed from origin
  tri.v0 = Vec3{1.0f, 0.0f, 0.0f};
  tri.v1 = Vec3{0.0f, 1.0f, 0.0f};
  tri.v2 = Vec3{0.0f, 0.0f, 1.0f};

  // 1. Query from origin: Should be exactly 1/8
  // Solid Angle = pi/4.  Contribution = (pi/4) / (2*pi) = 0.125
  Vec3 origin{0, 0, 0};
  float contrib = tri.contributionToQuery(origin, 0.0f);
  EXPECT_NEAR(contrib, 0.125f, 1e-6f);

  // 2. Test Winding/Sign: Query from the "back" side
  // If we move the query to (1, 1, 1), we are looking at the back of the
  // triangle The sign should flip.
  Vec3 back_query{1.0f, 1.0f, 1.0f};
  float back_contrib = tri.contributionToQuery(back_query, 0.0f);
  EXPECT_LT(back_contrib, 0.0f);

  // 3. Robust "Flat" Test:
  // Instead of exactly 0, test a point very far away on the plane of the
  // triangle. The triangle plane for these vertices is x + y + z = 1. A point
  // like (10, 10, -19) is on the plane.
  Vec3 far_plane_query{10.0f, 10.0f, -19.0f};
  float far_flat_contrib = tri.contributionToQuery(far_plane_query, 0.0f);
  // On the plane (but outside the triangle), det should be 0.
  EXPECT_NEAR(far_flat_contrib, 0.0f, 1e-7f);
}

__global__ void test_approximation_kernel(
    const Vec3 query, const Vec3 com, const Vec3_f16 zero_coeff,
    const Mat3x3_f16 first_coeff, const Tensor3_bf16_compressed second_coeff,
    float *out_result) {
  *out_result = compute_node_approximation(query, com, zero_coeff, first_coeff,
                                           second_coeff);
}

TEST(TaylorApproximation, Triangle) {
  // 1. Create a "Leaf" of 4 triangles randomized inside a small box [0, 1]^3
  std::vector<Triangle> leaf_triangles = {
      Triangle{{0.1, 0.1, 0.1}, {0.4, 0.1, 0.1}, {0.1, 0.4, 0.1}},
      Triangle{{0.5, 0.5, 0.5}, {0.8, 0.5, 0.5}, {0.5, 0.8, 0.5}},
      Triangle{{0.1, 0.6, 0.1}, {0.3, 0.8, 0.2}, {0.1, 0.9, 0.1}},
      Triangle{{0.6, 0.1, 0.6}, {0.9, 0.2, 0.7}, {0.7, 0.3, 0.8}}};

  // std::vector<Triangle> leaf_triangles = {
  //     Triangle{{0.1, 0.1, 0.1}, {0.4, 0.1, 0.1}, {0.1, 0.4, 0.1}}};

  // 2. Compute Center of Mass (COM) for the node
  Vec3 com{0, 0, 0};
  float total_area = 0.0f;
  for (const auto &tri : leaf_triangles) {
    float area = tri.get_scaled_normal().length();
    com = com + tri.centroid() * area;
    total_area += area;
  }
  com = com / total_area; // TODO in construction code we should also use total
                          // area instead of triangle count!

  // 3. Aggregate Coefficients
  Vec3 zero_order_sum{0, 0, 0};
  Mat3x3 first_order_sum = Mat3x3::zero();
  Tensor3_compressed second_order_sum;
  for (int i = 0; i < 18; ++i)
    second_order_sum.data[i] = 0.0f;

  for (const auto &tri : leaf_triangles) {
    Vec3 zero_order;
    Mat3x3 first_order;
    Tensor3_compressed second_order;
    tri.get_tailor_terms(com, true, zero_order, first_order, second_order);

    zero_order_sum += zero_order;
    first_order_sum += first_order;
    second_order_sum += second_order;
  }

  float *d_result;
  cudaMalloc(&d_result, sizeof(float));
  Vec3_f16 h_zero = Vec3_f16::from_float(zero_order_sum);
  Mat3x3_f16 h_first = Mat3x3_f16::from_float(first_order_sum);
  Tensor3_bf16_compressed h_second =
      ::Tensor3_bf16_compressed::from_float(second_order_sum);

  // 4. Test at multiple distances to observe error decay
  // Query points moving away along the diagonal
  float distances[] = {5.0f, 10.0f, 20.0f};

  for (float dist : distances) {
    Vec3 query = com + Vec3{dist, dist, dist};

    // A. Ground Truth: Sum of individual triangle contributions
    float ground_truth = 0.0f;
    for (const auto &tri : leaf_triangles) {
      ground_truth += tri.contributionToQuery(query, 0.0f);
    }

    test_approximation_kernel<<<1, 1>>>(query, com, h_zero, h_first, h_second,
                                        d_result);

    float approx_h;
    cudaMemcpy(&approx_h, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    printf("GT: %f, approx: %f\n", ground_truth, approx_h);
    float error = std::abs(ground_truth - approx_h);

    // Expected behavior: error drops significantly as 1/r^p
    if (dist == 5.0f)
      EXPECT_LT(error, 1e-3);
    if (dist == 50.0f)
      EXPECT_LT(error, 1e-6);

    printf("Dist: %4.1f | Truth: %10.8f | Approx: %10.8f | Error: %.4e\n", dist,
           ground_truth, approx_h, error);
  }
  cudaFree(d_result);
}

TEST(TaylorApproximation, TriangleRandomized) {
  const int N_TRIANGLES = 32; // Number of triangles in the leaf
  const int M_QUERIES = 10;   // Number of random directions per distance
  const float LEAF_SIZE = 1.0f;

  std::mt19937 gen(42); // Fixed seed for reproducibility
  std::uniform_real_distribution<float> dist_pos(0.0f, LEAF_SIZE);
  std::uniform_real_distribution<float> dist_offset(-0.1f, 0.1f);

  // 1. Create N_TRIANGLES randomized triangles inside [0, LEAF_SIZE]^3
  std::vector<Triangle> leaf_triangles;
  for (int i = 0; i < N_TRIANGLES; ++i) {
    Vec3 v0{dist_pos(gen), dist_pos(gen), dist_pos(gen)};
    Vec3 v1 = v0 + Vec3{dist_offset(gen), dist_offset(gen), dist_offset(gen)};
    Vec3 v2 = v0 + Vec3{dist_offset(gen), dist_offset(gen), dist_offset(gen)};
    leaf_triangles.push_back(Triangle{v0, v1, v2});
  }

  // 2. Compute Center of Mass (COM)
  Vec3 com{0, 0, 0};
  float total_area = 0.0f;
  for (const auto &tri : leaf_triangles) {
    float area = tri.get_scaled_normal().length();
    com = com + tri.centroid() * area;
    total_area += area;
  }
  com = com / total_area;

  // 3. Aggregate Coefficients
  Vec3 zero_order_sum{0, 0, 0};
  Mat3x3 first_order_sum = Mat3x3::zero();
  Tensor3_compressed second_order_sum;
  std::fill(std::begin(second_order_sum.data), std::end(second_order_sum.data),
            0.0f);

  for (const auto &tri : leaf_triangles) {
    Vec3 zero_order;
    Mat3x3 first_order;
    Tensor3_compressed second_order;
    tri.get_tailor_terms(com, true, zero_order, first_order, second_order);

    zero_order_sum += zero_order;
    first_order_sum += first_order;
    second_order_sum += second_order;
  }

  // Prepare GPU data
  float *d_result;
  cudaMalloc(&d_result, sizeof(float));
  Vec3_f16 h_zero = Vec3_f16::from_float(zero_order_sum);
  Mat3x3_f16 h_first = Mat3x3_f16::from_float(first_order_sum);
  Tensor3_bf16_compressed h_second =
      ::Tensor3_bf16_compressed::from_float(second_order_sum);

  // 4. Test at multiple distances with M random directions
  float distances[] = {5.0f, 10.0f, 20.0f};

  for (float dist_mag : distances) {
    float max_error = 0.0f;

    for (int m = 0; m < M_QUERIES; ++m) {
      // Generate a random direction on the unit sphere
      // If you don't have C++20, simple rejection sampling or Gaussian works:
      std::normal_distribution<float> norm_dist(0.0f, 1.0f);
      Vec3 dir{norm_dist(gen), norm_dist(gen), norm_dist(gen)};
      dir = dir / dir.length();

      Vec3 query = com + dir * dist_mag;

      float ground_truth = 0.0f;
      for (const auto &tri : leaf_triangles) {
        ground_truth += tri.contributionToQuery(query, 0.0f);
      }

      test_approximation_kernel<<<1, 1>>>(query, com, h_zero, h_first, h_second,
                                          d_result);

      float approx_h;
      cudaMemcpy(&approx_h, d_result, sizeof(float), cudaMemcpyDeviceToHost);
      printf("ground_truth: %f, approx_h: %f\n", ground_truth, approx_h);

      float error = std::abs(ground_truth - approx_h);
      max_error = std::max(max_error, error);
    }

    printf("Dist: %4.1f | Max Error over %d directions: %.4e\n", dist_mag,
           M_QUERIES, max_error);

    if (dist_mag == 5.0f)
      EXPECT_LT(max_error, 5e-3); // Slightly looser for random orientations
    if (dist_mag == 50.0f)
      EXPECT_LT(max_error, 1e-5);
  }

  cudaFree(d_result);
}

TEST(TaylorApproximation, PointNormal) {
  // 1. Create a "Leaf" of 4 PointNormals inside [0, 1]^3
  // PointNormal(position, normal)
  std::vector<PointNormal> leaf_points = {
      PointNormal{{0.1, 0.1, 0.1}, {0.0, 0.0, 0.045}},
      PointNormal{{0.5, 0.5, 0.5}, {0.0, 0.0, 0.045}},
      PointNormal{{0.1, 0.6, 0.1}, {-0.015, 0.0, 0.030}},
      PointNormal{{0.6, 0.1, 0.6}, {0.0, -0.025, 0.025}}};

  // 2. Compute Center of Mass (COM)
  Vec3 com{0, 0, 0};
  float total_area = 0.0f;
  for (const auto &pn : leaf_points) {
    float area = pn.get_scaled_normal().length();
    com = com + pn.centroid() * area;
    total_area += area;
  }
  com = com / total_area;

  // 3. Aggregate Coefficients
  Vec3 zero_order_sum{0, 0, 0};
  Mat3x3 first_order_sum = Mat3x3::zero();
  Tensor3_compressed second_order_sum;
  std::fill(std::begin(second_order_sum.data), std::end(second_order_sum.data),
            0.0f);

  for (const auto &pn : leaf_points) {
    Vec3 zero_order;
    Mat3x3 first_order;
    Tensor3_compressed second_order;
    pn.get_tailor_terms(com, true, zero_order, first_order, second_order);

    zero_order_sum += zero_order;
    first_order_sum += first_order;
    second_order_sum += second_order;
  }

  // Prepare GPU data
  float *d_result;
  cudaMalloc(&d_result, sizeof(float));
  Vec3_f16 h_zero = Vec3_f16::from_float(zero_order_sum);
  Mat3x3_f16 h_first = Mat3x3_f16::from_float(first_order_sum);
  Tensor3_bf16_compressed h_second =
      ::Tensor3_bf16_compressed::from_float(second_order_sum);

  float distances[] = {5.0f, 10.0f, 20.0f};
  for (float dist : distances) {
    Vec3 query = com + Vec3{dist, dist, dist};

    float ground_truth = 0.0f;
    for (const auto &pn : leaf_points) {
      ground_truth += pn.contributionToQuery(query, 200.0f);
    }

    test_approximation_kernel<<<1, 1>>>(query, com, h_zero, h_first, h_second,
                                        d_result);

    float approx_h;
    cudaMemcpy(&approx_h, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    printf("GT: %f, approx: %f\n", ground_truth, approx_h);
    float error = std::abs(ground_truth - approx_h);
    EXPECT_LT(error, (dist < 10.0f ? 1e-3 : 1e-6));

    printf("PointNormal Fixed | Dist: %4.1f | Truth: %10.8f | Approx: %10.8f | "
           "Error: %.4e\n",
           dist, ground_truth, approx_h, error);
  }
  cudaFree(d_result);
}

TEST(TaylorApproximation, PointNormalRandomized) {
  const int N_POINTS = 64;
  const int M_QUERIES = 10;
  const float LEAF_SIZE = 1.0f;

  std::mt19937 gen(1337);
  std::uniform_real_distribution<float> dist_pos(0.0f, LEAF_SIZE);
  std::uniform_real_distribution<float> dist_normal(-0.1f, 0.1f);
  std::normal_distribution<float> norm_dist(0.0f, 1.0f);

  // 1. Create N_POINTS randomized PointNormals
  std::vector<PointNormal> leaf_points;
  for (int i = 0; i < N_POINTS; ++i) {
    Vec3 p{dist_pos(gen), dist_pos(gen), dist_pos(gen)};
    Vec3 n{dist_normal(gen), dist_normal(gen), dist_normal(gen)};
    leaf_points.push_back(PointNormal{p, n});
  }

  // 2. Compute COM
  Vec3 com{0, 0, 0};
  float total_area = 0.0f;
  for (const auto &pn : leaf_points) {
    float area = pn.get_scaled_normal().length();
    com = com + pn.centroid() * area;
    total_area += area;
  }
  com = com / total_area;

  // 3. Aggregate Coefficients
  Vec3 zero_order_sum{0, 0, 0};
  Mat3x3 first_order_sum = Mat3x3::zero();
  Tensor3_compressed second_order_sum;
  std::fill(std::begin(second_order_sum.data), std::end(second_order_sum.data),
            0.0f);

  for (const auto &pn : leaf_points) {
    Vec3 zero_order;
    Mat3x3 first_order;
    Tensor3_compressed second_order;
    pn.get_tailor_terms(com, true, zero_order, first_order, second_order);

    zero_order_sum += zero_order;
    first_order_sum += first_order;
    second_order_sum += second_order;
  }

  float *d_result;
  cudaMalloc(&d_result, sizeof(float));
  Vec3_f16 h_zero = Vec3_f16::from_float(zero_order_sum);
  Mat3x3_f16 h_first = Mat3x3_f16::from_float(first_order_sum);
  Tensor3_bf16_compressed h_second =
      ::Tensor3_bf16_compressed::from_float(second_order_sum);

  float distances[] = {5.0f, 10.0f, 20.0f};
  for (float dist_mag : distances) {
    float max_error = 0.0f;
    for (int m = 0; m < M_QUERIES; ++m) {
      Vec3 dir{norm_dist(gen), norm_dist(gen), norm_dist(gen)};
      dir = dir / dir.length();
      Vec3 query = com + dir * dist_mag;

      float ground_truth = 0.0f;
      for (const auto &pn : leaf_points) {
        ground_truth += pn.contributionToQuery(query, 200.0f);
      }

      test_approximation_kernel<<<1, 1>>>(query, com, h_zero, h_first, h_second,
                                          d_result);
      float approx_h;
      cudaMemcpy(&approx_h, d_result, sizeof(float), cudaMemcpyDeviceToHost);
      printf("GT: %f, approx: %f\n", ground_truth, approx_h);

      max_error = std::max(max_error, std::abs(ground_truth - approx_h));
    }

    printf("PointNormal Random | Dist: %4.1f | Max Error: %.4e\n", dist_mag,
           max_error);
    EXPECT_LT(max_error, (dist_mag < 10.0f ? 5e-3 : 5e-5));
  }

  cudaFree(d_result);
}
