#include "geometry.h"
#include "vec3.h"
#include "winder_cuda.h"
#include <cstddef>
#include <cstdio>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <memory>
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

  float my_wn = 0.0f;
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
        my_wn += tile[j].contributionToQuery(my_q, inv_epsilon);
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
  float pos_dist_radius = params.pos_dist_radius;
  float norm_dist_variance = params.norm_dist_variance;
  float query_dist_radius = params.query_dist_radius;

  thrust::host_vector<Vec3> queries_h(query_count);
  thrust::host_vector<Vec3> points_h(count);
  thrust::host_vector<Vec3> scaled_normals_h(count);
  thrust::host_vector<T> geometry_h(count);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist_pos(-pos_dist_radius,
                                                 pos_dist_radius);
  std::normal_distribution<float> norm_dis(0.0F, norm_dist_variance);

  for (size_t i = 0; i < count; ++i) {
    if constexpr (std::is_same_v<T, PointNormal>) {
      geometry_h[i].p = Vec3{dist_pos(gen), dist_pos(gen), dist_pos(gen)};
      geometry_h[i].n = Vec3{norm_dis(gen), norm_dis(gen), norm_dis(gen)};
      points_h[i] = geometry_h[i].p;
      scaled_normals_h[i] = geometry_h[i].n;
    } else if constexpr (std::is_same_v<T, Triangle>) {
      Vec3 offset = {dist_pos(gen), dist_pos(gen), dist_pos(gen)};
      Vec3 v0 = Vec3{dist_pos(gen) * 0.04F, dist_pos(gen) * 0.04F,
                 dist_pos(gen) * 0.04F};
      Vec3 v1 = Vec3{dist_pos(gen) * 0.04F, dist_pos(gen) * 0.04F,
                 dist_pos(gen) * 0.04F};
      Vec3 v2 = Vec3{dist_pos(gen) * 0.04F, dist_pos(gen) * 0.04F,
                 dist_pos(gen) * 0.04F};
      geometry_h[i].v0 = offset + v0;
      geometry_h[i].v1 = offset + v1;
      geometry_h[i].v2 = offset + v2;
    }
  }
  std::uniform_real_distribution<float> dist_q(-query_dist_radius,
                                               query_dist_radius);
  for (size_t i = 0; i < query_count; ++i) {
    Vec3 p{dist_q(gen), dist_q(gen), dist_q(gen)};
    queries_h[i] = p;
  }

  // compute gt using brute force
  thrust::device_vector<Vec3> queries_d = queries_h;
  thrust::device_vector<T> geom_d = geometry_h;
  thrust::device_vector<float> gt_wn_d(params.query_count);

  int threads = 256;
  int blocks = (params.query_count + threads - 1) / threads;
  size_t smem_size = threads * sizeof(PointNormal);

  compute_winding_numbers_brute_force_kernel<<<blocks, threads, smem_size>>>(
      (Vec3 *)thrust::raw_pointer_cast(queries_d.data()),
      (T *)thrust::raw_pointer_cast(geom_d.data()), params.query_count,
      params.count, thrust::raw_pointer_cast(gt_wn_d.data()),
      1.0f / params.epsilon);
  CUDA_CHECK(cudaGetLastError());

  thrust::host_vector<float> gt_h = gt_wn_d;

  // use winder backend instead
  thrust::device_vector<Vec3> points_d = points_h;
  thrust::device_vector<Vec3> scaled_normals_d = scaled_normals_h;

  std::unique_ptr<WinderBackend<T>> backend;

  if constexpr (std::is_same_v<T, PointNormal>) {
    backend = WinderBackend<T>::CreateFromPoints(
        (float *)points_d.data().get(), (float *)scaled_normals_d.data().get(),
        points_d.size(), 0);
  } else {
    backend = WinderBackend<T>::CreateFromMesh((float *)geom_d.data().get(),
                                               points_d.size(), 0);
  }

  // DEBUG
  printf("Waiting until tree construction is actually done\n");
  cudaDeviceSynchronize();
  printf("Tree construction is done!\n");

  auto wn = backend->compute((float *)queries_d.data().get(), queries_d.size(),
                             2.F, epsilon, 0);

  std::vector<float> wn_h(query_count);
  cudaMemcpy(&wn_h[0], wn.get(), query_count * sizeof(float),
             cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < query_count; ++i) {
    EXPECT_NEAR(wn_h[i], gt_h[i], std::abs(gt_h[i] * 1e-5f));
  }
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
        // Name, Count, Q_Count, PosR, NormV, Q_Rad, Eps, Beta
        WinderTestParams{"SinglePointSingleQuery", 1, 1, 1.0f, 2.0f, 20.0f,
                         1.0f / 250.0f, 2.0f},
        WinderTestParams{"SmallLeaf", 32, 100, 1.0f, 2.0f, 20.0f, 1.0f / 250.0f,
                         2.0f},
        WinderTestParams{"SmallLeafSingleQuery", 32, 1, 1.0f, 2.0f, 20.0f,
                         1.0f / 250.0f, 2.0f},
        WinderTestParams{"UnderfullLeaf", 10, 100, 0.5f, 1.0f, 10.0f,
                         1.0f / 100.0f, 2.0f},
        WinderTestParams{"UnderfullLeafSingleQuery", 10, 1, 0.5f, 1.0f, 10.0f,
                         1.0f / 100.0f, 2.0f},
        WinderTestParams{"MediumScene", 1024, 100, 5.0f, 5.0f, 50.0f,
                         1.0f / 250.0f, 2.0f},
        WinderTestParams{"MediumSceneSmallEpsilon", 1024, 100, 5.0f, 5.0f,
                         50.0f, 1.0f / 1000.0f, 2.0f},
        WinderTestParams{"MediumSceneBigEpsilon", 1024, 100, 5.0f, 5.0f, 50.0f,
                         1.0f / 2.0f, 2.0f},
        WinderTestParams{"LargeScene", 1000000, 1000, 50.0f, 5.0f, 60.0f,
                         1.0f / 250.0f, 2.0f}),
    [](const ::testing::TestParamInfo<WindingNumberTest::ParamType> &info) {
      return info.param.name; // This makes the test output readable
    });

INSTANTIATE_TEST_SUITE_P(
    TriangleVariations, WindingNumberTriangleTest,
    ::testing::Values(
        // Name, Count, Q_Count, PosR, NormV, Q_Rad, Eps, Beta
        WinderTestParams{"SinglePointSingleQuery", 1, 1, 1.0f, 2.0f, 20.0f,
                         1.0f / 250.0f, 2.0f},
        WinderTestParams{"SmallLeaf", 32, 100, 1.0f, 2.0f, 20.0f, 1.0f / 250.0f,
                         2.0f},
        WinderTestParams{"SmallLeafSingleQuery", 32, 1, 1.0f, 2.0f, 20.0f,
                         1.0f / 250.0f, 2.0f},
        WinderTestParams{"UnderfullLeaf", 10, 100, 0.5f, 1.0f, 10.0f,
                         1.0f / 100.0f, 2.0f},
        WinderTestParams{"UnderfullLeafSingleQuery", 10, 1, 0.5f, 1.0f, 10.0f,
                         1.0f / 100.0f, 2.0f},
        WinderTestParams{"MediumScene", 1024, 100, 5.0f, 5.0f, 50.0f,
                         1.0f / 250.0f, 2.0f},
        WinderTestParams{"MediumSceneSmallEpsilon", 1024, 100, 5.0f, 5.0f,
                         50.0f, 1.0f / 1000.0f, 2.0f},
        WinderTestParams{"MediumSceneBigEpsilon", 1024, 100, 5.0f, 5.0f, 50.0f,
                         1.0f / 2.0f, 2.0f},
        WinderTestParams{"LargeScene", 1000000, 1000, 50.0f, 5.0f, 60.0f,
                         1.0f / 250.0f, 2.0f}),
    [](const ::testing::TestParamInfo<WindingNumberTest::ParamType> &info) {
      return info.param.name; // This makes the test output readable
    });
