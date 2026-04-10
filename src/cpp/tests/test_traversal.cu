#include "geometry.h"
#include "vec3.h"
#include "winder_cuda.h"
#include <cstddef>
#include <cstdio>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <memory>
#include <numbers>
#include <random>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

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

struct WinderTestParams {
  std::string name;
  size_t count;
  size_t query_count;
  float pos_dist_radius;
  float norm_dist_variance;
  float query_dist_radius;
  float epsilon;
  float beta;
  double allowed_rms;
};

class WindingNumberTest : public ::testing::TestWithParam<WinderTestParams> {
  // You can put helper methods here if needed
};

class WindingNumberPointTest : public WindingNumberTest {};
class WindingNumberTriangleTest : public WindingNumberTest {};

template <typename T> void RunAccuracyTest(const WinderTestParams &params) {
  // Replace your hardcoded constants with params.X
  size_t count = params.count;
  float epsilon = params.epsilon;
  size_t query_count = params.query_count;
  float radius = params.pos_dist_radius;
  float query_dist_radius = params.query_dist_radius;
  double allowed_rms = params.allowed_rms;
  float beta = params.beta;

  thrust::host_vector<Vec3> queries_h(query_count);
  thrust::host_vector<Vec3> points_h;
  thrust::host_vector<Vec3> scaled_normals_h;
  thrust::host_vector<T> geometry_h;

  std::mt19937 gen(42);
  std::normal_distribution<float> dist_pos(0.F, 1.F);

  const float pi = std::numbers::pi_v<float>;
  const float total_sphere_area = 4.F * pi * radius;
  const float area_per_point = total_sphere_area / static_cast<float>(count);

  auto get_pos = [&](float theta, float phi) {
    return Vec3{radius * sinf(theta) * cosf(phi),
                radius * sinf(theta) * sinf(phi), radius * cosf(theta)};
  };

  auto rings =
      static_cast<size_t>((1.F + sqrtf(1.F + static_cast<float>(count))) / 2.F);
  size_t sectors = count / (2 * rings - 1);

  for (size_t i = 0; i < count; ++i) {
    if constexpr (std::is_same_v<T, PointNormal>) {
      Vec3 p{dist_pos(gen), dist_pos(gen), dist_pos(gen)};
      float normalization_factor = radius / p.length();
      p = p * normalization_factor; // point on sphere
      Vec3 n = area_per_point * p / radius;
      PointNormal pn{p, n};
      geometry_h.push_back(pn);
      points_h.push_back(p);
      scaled_normals_h.push_back(n);
    } else if constexpr (std::is_same_v<T, Triangle>) {
      if (count == 1) {
        Triangle t;
        t.v0 = Vec3{0, 0, 0};
        t.v1 = Vec3{1, 0, 0};
        t.v2 = Vec3{0, 1, 0};
        geometry_h.push_back(t);
        break;
      }
      size_t ring = i / sectors;
      size_t sector = i % sectors;
      // Define the 4 corners of the current "quad" on the sphere grid
      float theta0 = pi * float(ring) / float(rings);
      float theta1 = pi * float(ring + 1) / float(rings);
      float phi0 = 2.0f * pi * float(sector) / float(sectors);
      float phi1 = 2.0f * pi * float(sector + 1) / float(sectors);
      Vec3 v_top_left = get_pos(theta0, phi0);
      Vec3 v_top_right = get_pos(theta0, phi1);
      Vec3 v_bot_left = get_pos(theta1, phi0);
      Vec3 v_bot_right = get_pos(theta1, phi1);
      // The two triangles
      if (ring != 0) {
        Triangle t1;
        t1.v0 = v_top_left;
        t1.v1 = v_bot_left;
        t1.v2 = v_top_right;
        geometry_h.push_back(t1);
      }
      if (ring != rings - 1) {
        Triangle t2;
        t2.v0 = v_top_right;
        t2.v1 = v_bot_left;
        t2.v2 = v_bot_right;
        geometry_h.push_back(t2);
      }
    }
  }
  std::uniform_real_distribution<float> dist_q(-query_dist_radius,
                                               query_dist_radius);

  for (size_t i = 0; i < query_count; ++i) {
    Vec3 p{dist_q(gen), dist_q(gen), dist_q(gen)};
    queries_h[i] = p;
  }

  float inv_epsilon = 1.F / epsilon;

  // thrust::host_vector<double> cpu_wn(query_count);
  // for (size_t i = 0; i < query_count; ++i) {
  //   Vec3 q = queries_h[i];
  //   cpu_wn[i] = 0.0;
  //   for (size_t j = 0; j < count; ++j) {
  //     T g = geometry_h[j];
  //     cpu_wn[i] += static_cast<double>(g.contributionToQuery(q,
  //     inv_epsilon));
  //   }
  // }

  // compute gt using brute force
  thrust::device_vector<Vec3> queries_d = queries_h;
  thrust::device_vector<T> geom_d = geometry_h;
  thrust::device_vector<float> gt_wn_d(params.query_count);

  int threads = 256;
  int blocks = (params.query_count + threads - 1) / threads;
  size_t smem_size = threads * sizeof(T);

  printf("DEBUG: geometry_h.size(): %lu\n", geometry_h.size());
  compute_winding_numbers_brute_force_kernel<<<blocks, threads, smem_size>>>(
      (Vec3 *)thrust::raw_pointer_cast(queries_d.data()),
      (T *)thrust::raw_pointer_cast(geom_d.data()), params.query_count,
      geometry_h.size(), thrust::raw_pointer_cast(gt_wn_d.data()),
      1.0f / params.epsilon);
  CUDA_CHECK(cudaGetLastError());

  thrust::host_vector<float> gt_h = gt_wn_d;

  // Ensure CPU wn = brute force gpu wn
  // for (size_t i = 0; i < query_count; ++i) {
  //   EXPECT_NEAR(gt_h[i], static_cast<float>(cpu_wn[i]), 1e-5F);
  // }

  // use winder backend instead
  thrust::device_vector<Vec3> points_d = points_h;
  thrust::device_vector<Vec3> scaled_normals_d = scaled_normals_h;

  std::unique_ptr<WinderBackend<T>> backend;

  if constexpr (std::is_same_v<T, PointNormal>) {
    backend = WinderBackend<T>::CreateFromPoints(
        (float *)points_d.data().get(), (float *)scaled_normals_d.data().get(),
        points_d.size(), 0);
  } else {
    backend = WinderBackend<T>::CreateFromTriangles((float *)geom_d.data().get(),
                                               geom_d.size(), 0);
  }

  // DEBUG
  printf("Waiting until tree construction is actually done\n");
  cudaDeviceSynchronize();
  printf("Tree construction is done!\n");

  printf("DEBUG: calling compute with beta: %f\n", beta);
  auto wn = backend->compute((float *)queries_d.data().get(), queries_d.size(),
                             beta, epsilon, 0);

  std::vector<float> wn_h(query_count);
  cudaMemcpy(&wn_h[0], wn.get(), query_count * sizeof(float),
             cudaMemcpyDeviceToHost);

  double mse = 0;
  double max_error = 0;
  double mae = 0;
  double mean_diff = 0;

  for (size_t i = 0; i < query_count; ++i) {
    double diff = static_cast<double>(wn_h[i]) - static_cast<double>(gt_h[i]);
    mse += (diff * diff - mse) / (i + 1);
    mae += (std::abs(diff) - mae) / (i + 1);
    mean_diff += (diff - mean_diff) / (i + 1);
    max_error = std::max(max_error, std::abs(diff));
  }
  double rms_error = std::sqrt(mse);
  double variance = mse - mean_diff * mean_diff;
  double std_dev = std::sqrt(std::max(0.0, variance));
  printf(
      "Mean error:%f | stdDev: %f | MSE: %f | MAE: %f | RMS: %f | MAX: %f \n",
      mean_diff, std_dev, mse, mae, rms_error, max_error);
  EXPECT_LE(rms_error, allowed_rms);
}

TEST_P(WindingNumberPointTest, Accuracy) {
  RunAccuracyTest<PointNormal>(GetParam());
}

TEST_P(WindingNumberTriangleTest, Accuracy) {
  RunAccuracyTest<Triangle>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    PointVariations, WindingNumberPointTest,
    ::testing::Values(
        // Name, Count, Q_Count, PosR, NormV, Q_Rad, Eps, Beta, allowed_rms
        WinderTestParams{"SinglePointSingleQuery", 1, 1, 1.F, 2.F, 20.F,
                         1.F / 250.F, 2.3F, 1e-2},
        WinderTestParams{"SmallLeafSingleQuery", 32, 1, 1.F, 2.F, 20.F,
                         1.F / 250.F, 2.3F, 1e-2},
        WinderTestParams{"UnderfullLeafSingleQuery", 10, 1, 0.5f, 1.F, 10.F,
                         1.F / 100.F, 2.3F, 1e-2},
        WinderTestParams{"SmallLeaf", 32, 100, 1.F, 2.F, 20.F, 1.F / 250.F,
                         2.3F, 1e-2},
        WinderTestParams{"UnderfullLeaf", 10, 100, 0.5f, 1.F, 10.F, 1.F / 100.F,
                         2.3F, 1e-2},
        WinderTestParams{"MediumSceneSingleQuery", 1024, 1, 5.F, 5.F, 5.F,
                         1.F / 250.F, 2.3F, 1e-2},
        WinderTestParams{"MediumScene", 1024, 100, 5.F, 5.F, 50.F, 1.F / 250.F,
                         2.3F, 1e-2},
        WinderTestParams{"MediumSceneSmallEpsilon", 1024, 100, 5.F, 5.F, 50.F,
                         1.F / 1000.F, 2.3F, 1e-2},
        WinderTestParams{"MediumSceneBigEpsilon", 1024, 100, 5.F, 5.F, 50.F,
                         1.F / 2.F, 2.3F, 1e-2},
        WinderTestParams{"LargeSceneSingleQuery", 1000000, 1, 50.F, 5.F, 60.F,
                         1.F / 250.F, 2.3F, 1e-2},
        WinderTestParams{"LargeScene", 1000000, 1000, 50.F, 5.F, 60.F,
                         1.F / 250.F, 2.3F, 1e-2},
        WinderTestParams{"LargeSceneSingleQueryLargeBeta", 1000000, 1, 50.F,
                         5.F, 60.F, 1.F / 250.F, 5.F, 1e-2},
        WinderTestParams{"LargeSceneLargeBeta", 1000000, 1000, 50.F, 5.F, 60.F,
                         1.F / 250.F, 5.F, 1e-2}),
    [](const ::testing::TestParamInfo<WindingNumberTest::ParamType> &info) {
      return info.param.name; // This makes the test output readable
    });

INSTANTIATE_TEST_SUITE_P(
    TriangleVariations, WindingNumberTriangleTest,
    ::testing::Values(
        // Name, Count, Q_Count, PosR, NormV, Q_Rad, Eps, Beta, allowed_rms
        WinderTestParams{"SinglePointSingleQuery", 1, 1, 1.F, 2.F, 20.F,
                         1.F / 250.F, 2.F, 1e-3},
        WinderTestParams{"SmallLeafSingleQuery", 32, 1, 1.F, 2.F, 20.F,
                         1.F / 250.F, 2.F, 1e-3},
        WinderTestParams{"UnderfullLeafSingleQuery", 10, 1, 0.5f, 1.F, 10.F,
                         1.F / 100.F, 2.F, 1e-3},
        WinderTestParams{"SmallLeaf", 32, 100, 1.F, 2.F, 20.F, 1.F / 250.F, 2.F,
                         1e-3},
        WinderTestParams{"UnderfullLeaf", 10, 100, 0.5f, 1.F, 10.F, 1.F / 100.F,
                         2.F, 1e-3},
        WinderTestParams{"MediumSceneSingleQuery", 1024, 1, 5.F, 5.F, 5.F,
                         1.F / 250.F, 2.F, 1e-3},
        WinderTestParams{"MediumScene", 1024, 100, 5.F, 5.F, 50.F, 1.F / 250.F,
                         2.F, 1e-3},
        WinderTestParams{"MediumSceneSmallEpsilon", 1024, 100, 5.F, 5.F, 50.F,
                         1.F / 1000.F, 2.F, 1e-3},
        WinderTestParams{"MediumSceneBigEpsilon", 1024, 100, 5.F, 5.F, 50.F,
                         1.F / 2.F, 2.F, 1e-3},
        WinderTestParams{"LargeSceneSingleQuery", 1000000, 1, 50.F, 5.F, 60.F,
                         1.F / 250.F, 2.F, 1e-3},
        WinderTestParams{"LargeScene", 1000000, 1000, 50.F, 5.F, 60.F,
                         1.F / 250.F, 2.F, 1e-3},
        WinderTestParams{"LargeSceneSingleQueryLargeBeta", 1000000, 1, 50.F,
                         5.F, 60.F, 1.F / 250.F, 4.F, 1e-3},
        WinderTestParams{"LargeSceneLargeBeta", 1000000, 1000, 50.F, 5.F, 60.F,
                         1.F / 250.F, 4.F, 1e-3}),
    [](const ::testing::TestParamInfo<WindingNumberTest::ParamType> &info) {
      return info.param.name; // This makes the test output readable
    });
