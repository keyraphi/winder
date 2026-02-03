#include "geometry.h"
#include "vec3.h"
#include "winder_cuda.h"
#include <cstddef>
#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

TEST(WindingNumber, PointNormal32) {
  size_t count = 32;
  size_t query_count = 100;
  thrust::host_vector<float> queries_h(query_count * 3);

  thrust::host_vector<float> points_h(count * 3);
  thrust::host_vector<float> scaled_normals_h(count * 3);
  thrust::host_vector<PointNormal> point_normals(count);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist_pos(-1.F, 1.F);
  std::normal_distribution<float> norm_dis(0.0F, 2.0F);

  for (size_t i = 0; i < count; ++i) {
    Vec3 p{dist_pos(gen), dist_pos(gen), dist_pos(gen)};
    Vec3 n{norm_dis(gen), norm_dis(gen), norm_dis(gen)};
    points_h[3 * i] = p.x;
    points_h[3 * i + 1] = p.y;
    points_h[3 * i + 2] = p.z;
    scaled_normals_h[3 * i] = n.x;
    scaled_normals_h[3 * i + 1] = n.y;
    scaled_normals_h[3 * i + 2] = n.z;
    point_normals[i].p = p;
    point_normals[i].n = n;
  }
  std::uniform_real_distribution<float> dist_q(-10.F, 10.F);
  for (size_t i = 0; i < query_count; ++i) {
    Vec3 p{dist_q(gen), dist_q(gen), dist_q(gen)};
    queries_h[3 * i] = p.x;
    queries_h[3 * i + 1] = p.y;
    queries_h[3 * i + 2] = p.z;
  }

  thrust::device_vector<float> points_d = points_h;
  thrust::device_vector<float> scaled_normals_d = scaled_normals_h;

  std::unique_ptr<WinderBackend<PointNormal>> backend =
      WinderBackend<PointNormal>::CreateFromPoints(
          points_d.data().get(), scaled_normals_d.data().get(), points_d.size(),
          0);


  // Ground Truth
}
