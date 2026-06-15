#include "bvh8.h"
#include "geometry.h"
#include "kernels/common.cuh"
#include "tailor_coefficients.h"
#include "vec3.h"
#include "winder_cuda.h"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime_api.h>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <ostream>
#include <queue>
#include <random>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


TEST(TreeConstruction, TreeStructure) {
  size_t leaf_per_dim = 2;
  size_t leaf_size = 32;

  thrust::host_vector<Vec3> points;
  thrust::host_vector<Vec3> normals;
  thrust::host_vector<PointNormal> point_normals;
  // fill each leaf with equal points
  // fill each leaf with equal points
  for (size_t z = 1; z <= leaf_per_dim; z++) {
    for (size_t y = 1; y <= leaf_per_dim; y++) {
      for (size_t x = 1; x <= leaf_per_dim; x++) {

        // Add a tiny unique perturbation per cell to break axis-sorting ties
        float shift_x = static_cast<float>(x) + static_cast<float>(y) * 0.01f;
        float shift_y = static_cast<float>(y) + static_cast<float>(z) * 0.01f;
        float shift_z = static_cast<float>(z) + static_cast<float>(x) * 0.01f;

        Vec3 p{shift_x, shift_y, shift_z};
        Vec3 n = p / (p.length() + 1e-8F);
        PointNormal pn = {p, n};

        for (size_t i = 0; i < leaf_size; i++) {
          point_normals.push_back(pn);
          points.push_back(pn.p);
          normals.push_back(pn.n);
        }
      }
    }
  }

  thrust::device_vector<PointNormal> d_point_normals = point_normals;
  thrust::device_vector<Vec3> d_points = points;
  thrust::device_vector<Vec3> d_normals = normals;

  std::unique_ptr<WinderBackend<PointNormal>> backend =
      WinderBackend<PointNormal>::CreateFromPoints(
          (float *)d_points.data().get(), (float *)d_normals.data().get(),
          d_points.size(), 0);

  // Geometry count
  EXPECT_EQ(backend->m_count, point_normals.size());
  // Node count
  uint32_t bvh8_node_count;
  cudaMemcpy(&bvh8_node_count, backend->m_bvh8_node_count, sizeof(uint32_t),
             cudaMemcpyDeviceToHost);
  size_t leaf_count = (point_normals.size() + 31) / 32;
  printf("INFO: leaf count: %lu\n", leaf_count);
  printf("INFO: bvh8_node_count: %u\n", bvh8_node_count);

  // sorted geometry - each leaf should contain the exact same points
  thrust::host_vector<PointNormal> sorted_geometry(point_normals.size());
  SoAView<PointNormal> geometry_view{
      reinterpret_cast<float *>(sorted_geometry.data()),
      sorted_geometry.size()};
  cudaMemcpy(sorted_geometry.data(), backend->m_sorted_geometry,
             sizeof(PointNormal) * backend->m_count, cudaMemcpyDeviceToHost);
  CUDA_CHECK(cudaGetLastError());
  for (size_t i = 0; i < leaf_per_dim * leaf_per_dim * leaf_per_dim; i++) {
    size_t first_idx = i * leaf_size;
    PointNormal first_pn =
        PointNormal::load(geometry_view, first_idx, sorted_geometry.size());
    for (size_t j = 0; j < leaf_size; j++) {
      size_t idx = first_idx + j;
      PointNormal pn =
          PointNormal::load(geometry_view, idx, sorted_geometry.size());
      ASSERT_EQ(pn.p.x, first_pn.p.x);
      ASSERT_EQ(pn.p.y, first_pn.p.y);
      ASSERT_EQ(pn.p.z, first_pn.p.z);
      ASSERT_EQ(pn.n.x, first_pn.n.x);
      ASSERT_EQ(pn.n.y, first_pn.n.y);
      ASSERT_EQ(pn.n.z, first_pn.n.z);
    }
  }

  std::string tree_representation = backend->dump();

  std::string dot_rep = backend->dump();
  std::ofstream out("tree.dot");
  if (!out.is_open()) {
    std::cerr << "Error: Could not open file " << "backend_dump.dot" << " for writing."
              << std::endl;
    return;
  }

  out << dot_rep;
  out.close();
}

TEST(TreeConstruction, RandomTreeStructure) {
  size_t gen_leafs = 1000000 / 32;
  size_t leaf_size = 32;

  thrust::host_vector<Vec3> points;
  thrust::host_vector<Vec3> normals;
  thrust::host_vector<PointNormal> point_normals;

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist_pos(-10, 10);
  std::normal_distribution<float> norm_dis(0.0F, 1);
  // fill each leaf with equal points
  for (size_t i = 0; i < gen_leafs; i++) {
    for (size_t i = 0; i < leaf_size; i++) {
      Vec3 p = Vec3{dist_pos(gen), dist_pos(gen), dist_pos(gen)};
      Vec3 n = Vec3{norm_dis(gen), norm_dis(gen), norm_dis(gen)};
      PointNormal pn = {p, n};
      point_normals.push_back(pn);
      points.push_back(pn.p);
      normals.push_back(pn.n);
    }
  }

  thrust::device_vector<PointNormal> d_point_normals = point_normals;
  thrust::device_vector<Vec3> d_points = points;
  thrust::device_vector<Vec3> d_normals = normals;

  std::unique_ptr<WinderBackend<PointNormal>> backend =
      WinderBackend<PointNormal>::CreateFromPoints(
          (float *)d_points.data().get(), (float *)d_normals.data().get(),
          d_points.size(), 0);

  // Geometry count
  EXPECT_EQ(backend->m_count, point_normals.size());
  // Node count
  uint32_t bvh8_node_count;
  cudaMemcpy(&bvh8_node_count, backend->m_bvh8_node_count, sizeof(uint32_t),
             cudaMemcpyDeviceToHost);
  size_t leaf_count = (point_normals.size() + 31) / 32;
  printf("INFO: leaf count: %lu\n", leaf_count);
  printf("INFO: bvh8_node_count: %u\n", bvh8_node_count);

  // get sorted geometry
  thrust::host_vector<PointNormal> sorted_geometry(point_normals.size());
  cudaMemcpy(sorted_geometry.data(), backend->m_sorted_geometry,
             sizeof(PointNormal) * backend->m_count, cudaMemcpyDeviceToHost);
  CUDA_CHECK(cudaGetLastError());

  std::string dot_rep = backend->dump();
  std::ofstream out("rand_tree.dot");
  if (!out.is_open()) {
    std::cerr << "Error: Could not open file " << "backend_dump_rand.dot" << " for writing."
              << std::endl;
    return;
  }

  out << dot_rep;
  out.close();
}
