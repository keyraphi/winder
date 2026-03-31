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
#include <string>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

void exportToDot(
    const std::string &filepath,
    const thrust::host_vector<BVH8Node> &bvh8_nodes,
    const thrust::host_vector<TailorCoefficientsBf16> &node_coefficients,
    const thrust::host_vector<LeafPointers> &leaf_pointers,
    const thrust::host_vector<PointNormal> &geometry,
    const thrust::host_vector<TailorCoefficientsBf16> &leaf_coefficients) {

  std::ofstream out(filepath.c_str());
  if (!out.is_open()) {
    std::cerr << "Error: Could not open file " << filepath << " for writing."
              << std::endl;
    return;
  }

  out << "digraph BVH8 {\n";
  out << "    node [fontname=\"Arial\", fontsize=10];\n";
  out << "    rankdir=TB;\n\n";

  std::queue<uint32_t> stack;
  stack.push(0);
  int empty_counter = 0;

  while (!stack.empty()) {
    uint32_t current_id = stack.front();
    const BVH8Node &current_node = bvh8_nodes[current_id];
    stack.pop();

    // --- Internal Node Logic ---
    TailorCoefficients node_coeff =
        TailorCoefficients::from_bf16(node_coefficients[current_id]);
    AABB aabb = current_node.parent_aabb;
    Vec3 node_com = aabb.center_of_mass.get(aabb.min, aabb.diagonal());

    out << "    N" << current_id << " [shape=none, label=<\n"
        << "      <TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\" "
           "CELLPADDING=\"4\" BGCOLOR=\"#f0faff\">\n"
        << "        <TR><TD COLSPAN=\"4\" BGCOLOR=\"#2196F3\"><B><FONT "
           "COLOR=\"white\">NODE "
        << current_id << "</FONT></B></TD></TR>\n"
        << "        <TR><TD COLSPAN=\"4\" BGCOLOR=\"#bbdefb\"><I>CoM: {"
        << node_com.x << ", " << node_com.y << ", " << node_com.z
        << "}</I></TD></TR>\n";

    // Node Coefficients (Zero, First, Second Order)
    out << "        <TR><TD BGCOLOR=\"#bbdefb\"><B>Zero Order</B></TD><TD>"
        << node_coeff.zero_order.x << "</TD><TD>" << node_coeff.zero_order.y
        << "</TD><TD>" << node_coeff.zero_order.z << "</TD></TR>\n";
    out << "        <TR><TD ROWSPAN=\"3\" BGCOLOR=\"#bbdefb\"><B>1st "
           "Order</B></TD>\n";
    for (int r = 0; r < 3; ++r) {
      if (r > 0)
        out << "        <TR>\n";
      out << "          <TD>" << node_coeff.first_order.data[r * 3 + 0]
          << "</TD><TD>" << node_coeff.first_order.data[r * 3 + 1]
          << "</TD><TD>" << node_coeff.first_order.data[r * 3 + 2]
          << "</TD></TR>\n";
    }
    const Tensor3 node_second_order = node_coeff.second_order.uncompress();
    out << "        <TR><TD COLSPAN=\"4\" BGCOLOR=\"#bbdefb\"><B>2nd "
           "Order</B></TD></TR>\n";
    for (int s = 0; s < 3; ++s) {
      out << "        <TR><TD ROWSPAN=\"3\">Slice " << s << "</TD>\n";
      for (int r = 0; r < 3; ++r) {
        if (r > 0)
          out << "        <TR>\n";
        int b = (s * 9) + (r * 3);
        out << "          <TD>" << node_second_order.data[b + 0] << "</TD><TD>"
            << node_second_order.data[b + 1] << "</TD><TD>"
            << node_second_order.data[b + 2] << "</TD></TR>\n";
      }
    }
    out << "      </TABLE>>];\n";

    uint32_t child_base = current_node.child_base;
    uint32_t child_offset = 0;
    LeafPointers current_leaf_pointers = leaf_pointers[current_id];

    for (size_t child_id = 0; child_id < 8; child_id++) {
      ChildType child_type = current_node.getChildMeta(child_id);
      switch (child_type) {
      case ChildType::INTERNAL: {
        uint32_t next_idx = child_offset++ + child_base;
        stack.push(next_idx);
        out << "    N" << current_id << " -> N" << next_idx << " [label=\""
            << child_id << "\"];\n";
        break;
      }
      case ChildType::LEAF: {
        AABB leaf_aabb = AABB::from_approximation(
            aabb, current_node.child_aabb_approx[child_id]);
        uint32_t l_id = current_leaf_pointers.indices[child_id];
        const TailorCoefficientsBf16 &coeffs_bf16 = leaf_coefficients[l_id];
        const TailorCoefficients coeff =
            TailorCoefficients::from_bf16(coeffs_bf16);
        Vec3 leaf_com =
            coeffs_bf16.center_of_mass.get(leaf_aabb.min, leaf_aabb.diagonal());

        out << "    L" << l_id << " [shape=none, label=<\n"
            << "      <TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\" "
               "CELLPADDING=\"4\" BGCOLOR=\"#eaffea\">\n"
            << "        <TR><TD COLSPAN=\"4\" BGCOLOR=\"#4CAF50\"><B>LEAF "
            << l_id << "</B></TD></TR>\n"
            << "        <TR><TD COLSPAN=\"4\" BGCOLOR=\"#c8e6c9\"><I>CoM: {"
            << leaf_com.x << ", " << leaf_com.y << ", " << leaf_com.z
            << "}</I></TD></TR>\n";

        // Leaf Coefficients
        out << "        <TR><TD BGCOLOR=\"#c8e6c9\"><B>Zero Order</B></TD><TD>"
            << coeff.zero_order.x << "</TD><TD>" << coeff.zero_order.y
            << "</TD><TD>" << coeff.zero_order.z << "</TD></TR>\n";
        out << "        <TR><TD ROWSPAN=\"3\" BGCOLOR=\"#c8e6c9\"><B>1st "
               "Order</B></TD>\n";
        for (int r = 0; r < 3; ++r) {
          if (r > 0)
            out << "        <TR>\n";
          out << "          <TD>" << coeff.first_order.data[r * 3 + 0]
              << "</TD><TD>" << coeff.first_order.data[r * 3 + 1] << "</TD><TD>"
              << coeff.first_order.data[r * 3 + 2] << "</TD></TR>\n";
        }
        // ... (Second order and Geometry logic remains the same as your current
        // implementation) ... [Included below for completeness in your logic]
        const Tensor3 second_order = coeff.second_order.uncompress();
        out << "        <TR><TD COLSPAN=\"4\" BGCOLOR=\"#c8e6c9\"><B>2nd "
               "Order</B></TD></TR>\n";
        for (int s = 0; s < 3; ++s) {
          out << "        <TR><TD ROWSPAN=\"3\">Slice " << s << "</TD>\n";
          for (int r = 0; r < 3; ++r) {
            if (r > 0)
              out << "        <TR>\n";
            int b = (s * 9) + (r * 3);
            out << "          <TD>" << second_order.data[b + 0] << "</TD><TD>"
                << second_order.data[b + 1] << "</TD><TD>"
                << second_order.data[b + 2] << "</TD></TR>\n";
          }
        }
        out << "        <TR><TD COLSPAN=\"4\" BGCOLOR=\"#c8e6c9\"><I>Geometry "
               "(32 Points)</I></TD></TR>\n";
        size_t g_off = l_id * 32;
        for (size_t g_id = 0; g_id < 32; g_id++) {
          const PointNormal &g = geometry[g_off + g_id];
          out << "        <TR><TD> P" << g_id << "</TD><TD COLSPAN=\"3\">P:{"
              << g.p.x << "," << g.p.y << "," << g.p.z << "} N:{" << g.n.x
              << "," << g.n.y << "," << g.n.z << "}</TD></TR>\n";
        }
        out << "      </TABLE>>];\n";
        out << "    N" << current_id << " -> L" << l_id << " [label=\""
            << child_id << "\", color=green, penwidth=2];\n";
        break;
      }
      case ChildType::EMPTY: {
        int e_id = empty_counter++;
        out << "    E" << e_id << " [label=\"\", shape=point, color=gray];\n";
        out << "    N" << current_id << " -> E" << e_id
            << " [style=dotted, color=gray];\n";
        break;
      }
      }
    }
  }
  out << "}\n";
  out.close();
}

__global__ void dequantizeTailorCoefficients(const BVH8Node *nodes,
                                             TailorCoefficientsBf16 *result,
                                             uint32_t node_count) {
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < node_count) {
    BVH8Node node = nodes[tid];
    float scalar_factor = node.tailor_coefficients.get_shared_scale_factor();
    result[tid].zero_order =
        node.tailor_coefficients.get_tailor_zero_order(scalar_factor);
    result[tid].first_order =
        node.tailor_coefficients.get_tailor_first_order(scalar_factor);
    result[tid].second_order =
        node.tailor_coefficients.get_tailor_second_order(scalar_factor);
  }
}

TEST(TreeConstruction, TreeStructure) {
  size_t leaf_per_dim = 2;
  size_t leaf_size = 32;

  thrust::host_vector<Vec3> points;
  thrust::host_vector<Vec3> normals;
  thrust::host_vector<PointNormal> point_normals;
  // fill each leaf with equal points
  for (size_t z = 0; z < leaf_per_dim; z++) {
    for (size_t y = 0; y < leaf_per_dim; y++) {
      for (size_t x = 0; x < leaf_per_dim; x++) {
        Vec3 p{static_cast<float>(x), static_cast<float>(y),
               static_cast<float>(z)};
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
  cudaMemcpy(sorted_geometry.data(), backend->m_sorted_geometry,
             sizeof(PointNormal) * backend->m_count, cudaMemcpyDeviceToHost);
  CUDA_CHECK(cudaGetLastError());
  for (size_t i = 0; i < leaf_per_dim * leaf_per_dim * leaf_per_dim; i++) {
    size_t first_idx = i * leaf_size;
    PointNormal first_pn = sorted_geometry[first_idx];
    for (size_t j = 0; j < leaf_size; j++) {
      size_t idx = first_idx + j;
      PointNormal pn = sorted_geometry[idx];
      ASSERT_EQ(pn.p.x, first_pn.p.x);
      ASSERT_EQ(pn.p.y, first_pn.p.y);
      ASSERT_EQ(pn.p.z, first_pn.p.z);
      ASSERT_EQ(pn.n.x, first_pn.n.x);
      ASSERT_EQ(pn.n.y, first_pn.n.y);
      ASSERT_EQ(pn.n.z, first_pn.n.z);
    }
  }

  // DOWNLOAD BVH8 DATA TO HOST
  // leafs
  thrust::host_vector<LeafPointers> leaf_pointers(leaf_count);
  thrust::host_vector<TailorCoefficientsBf16> leaf_coefficients(leaf_count);
  cudaMemcpy(leaf_pointers.data(), backend->m_bvh8_leaf_pointers,
             sizeof(LeafPointers) * bvh8_node_count, cudaMemcpyDeviceToHost);
  CUDA_CHECK(cudaGetLastError());
  cudaMemcpy(leaf_coefficients.data(), backend->m_leaf_coefficients,
             sizeof(TailorCoefficientsBf16) * leaf_count,
             cudaMemcpyDeviceToHost);
  CUDA_CHECK(cudaGetLastError());
  // nodes
  thrust::host_vector<BVH8Node> bvh8_nodes(bvh8_node_count);
  cudaMemcpy(bvh8_nodes.data(), backend->m_bvh8_nodes,
             sizeof(BVH8Node) * bvh8_node_count, cudaMemcpyDeviceToHost);
  CUDA_CHECK(cudaGetLastError());
  // node coefficients
  thrust::device_vector<TailorCoefficientsBf16> d_node_coefficients(
      bvh8_node_count);

  uint32_t block = 128;
  uint32_t grid = (bvh8_node_count + block - 1) / block;
  dequantizeTailorCoefficients<<<grid, block>>>(
      backend->m_bvh8_nodes, d_node_coefficients.data().get(), bvh8_node_count);
  CUDA_CHECK(cudaGetLastError());

  thrust::host_vector<TailorCoefficientsBf16> node_coefficients =
      d_node_coefficients;
  // Print the BVH8 structure
  exportToDot("tree.dot", bvh8_nodes, node_coefficients, leaf_pointers,
              sorted_geometry, leaf_coefficients);
}
