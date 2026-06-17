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

struct UnpackedTailorCoefficients {
  Vec3 zero_order;
  Mat3x3 first_order;
  Tensor3_compressed second_order;
  float scale_factor_low;
  float scale_factor_second;
};

__global__ void
get_root_coefficients_kernel(const BVH8Node *nodes,
                             UnpackedTailorCoefficients *out_data) {
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx == 0) {
    BVH8Node root = nodes[0];

    // 1. Extract scale factors
    out_data->scale_factor_low =
        root.tailor_coefficients.get_shared_scale_factor_low();
    out_data->scale_factor_second =
        root.tailor_coefficients.get_shared_scale_factor_second();

    // 2. Unpack and dequantize Zero Order
    Vec3_f16 z_f16 = root.tailor_coefficients.get_tailor_zero_order();
    out_data->zero_order = Vec3::from_f16(z_f16);

    // 3. Unpack and dequantize First Order
    // (Assuming you have a matching getter implemented for first order)
    out_data->first_order = root.tailor_coefficients.get_tailor_first_order();

    // 4. Unpack and dequantize Second Order
    // (Assuming you have a matching getter implemented for second order)
    out_data->second_order = root.tailor_coefficients.get_tailor_second_order();
  }
}

TEST(M2M, AllOrdersQuantizationAware) {
  const size_t count = 8 * 32 + 1;
  thrust::host_vector<PointNormal> geometry_h(count);
  thrust::host_vector<Vec3> points_h(count);
  thrust::host_vector<Vec3> scaled_normals_h(count);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist_pos(-20.0f, 20.0f);
  std::normal_distribution<float> norm_dis(0.0f, 5.0f);

  for (size_t i = 0; i < count; ++i) {
    geometry_h[i].p = Vec3{dist_pos(gen), dist_pos(gen), dist_pos(gen)};
    geometry_h[i].n = Vec3{norm_dis(gen), norm_dis(gen), norm_dis(gen)};
    points_h[i] = geometry_h[i].p;
    scaled_normals_h[i] = geometry_h[i].n;
  }

  // Build the tree hierarchy using your backend
  thrust::device_vector<Vec3> points_d = points_h;
  thrust::device_vector<Vec3> scaled_normals_d = scaled_normals_h;
  auto backend = WinderBackend<PointNormal>::CreateFromPoints(
      (float *)points_d.data().get(), (float *)scaled_normals_d.data().get(),
      points_d.size(), 0);

  CUDA_CHECK(cudaDeviceSynchronize());

  // printf("%s\n", backend->dump().c_str());

  // 1. Read the root node's bounding box from the device to find the reference
  // parent center
  BVH8Node root_node_h;
  CUDA_CHECK(cudaMemcpy(&root_node_h, backend->m_bvh8_nodes, sizeof(BVH8Node),
                        cudaMemcpyDeviceToHost));
  Vec3 root_parent_center = root_node_h.parent_aabb.center_of_mass.get(
      root_node_h.parent_aabb.min, root_node_h.parent_aabb.diagonal());

  // 2. Fetch leaf configurations back to host to compute an exact mathematical
  // reference
  uint32_t leaf_count = (backend->m_count + LEAF_SIZE - 1) / LEAF_SIZE;
  std::vector<AABB> leaf_aabbs_h(leaf_count);
  CUDA_CHECK(cudaMemcpy(leaf_aabbs_h.data(),
                        backend->m_binary_aabbs + leaf_count - 1,
                        sizeof(AABB) * leaf_count, cudaMemcpyDeviceToHost));

  std::vector<TailorCoefficientsF16> leaf_coeffs_f16_h(leaf_count);
  CUDA_CHECK(cudaMemcpy(leaf_coeffs_f16_h.data(), backend->m_leaf_coefficients,
                        leaf_count * sizeof(TailorCoefficientsF16),
                        cudaMemcpyDeviceToHost));

  // 3. Compute unquantized reference values on the host using a direct flat M2M
  // shift
  Vec3 ref_zero = {0.0f, 0.0f, 0.0f};
  Mat3x3 ref_first = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  Tensor3_compressed ref_second = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                   0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                   0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  for (size_t i = 0; i < leaf_count; ++i) {
    Vec3 leaf_center = leaf_aabbs_h[i].center_of_mass.get(
        leaf_aabbs_h[i].min, leaf_aabbs_h[i].diagonal());
    TailorCoefficients child_coefficients =
        TailorCoefficients::from_f16(leaf_coeffs_f16_h[i]);

    Vec3 shift = leaf_center - root_parent_center;
    const Vec3 &zero_child = child_coefficients.zero_order;
    const Mat3x3 &child_first = child_coefficients.first_order;
    const Tensor3_compressed &child_second = child_coefficients.second_order;

    // Zero order accumulator
    ref_zero += zero_child;

    // First order accumulator
    ref_first += child_first + shift.outer_product(zero_child);

    // Second order accumulator (Mirrors your device-side mapping layout)
    ref_second.data[0] += child_second.data[0] + shift.x * child_first.data[0] +
                          0.5f * shift.x * shift.x * zero_child.x;
    ref_second.data[1] += child_second.data[1] + shift.x * child_first.data[1] +
                          0.5f * shift.x * shift.x * zero_child.y;
    ref_second.data[2] += child_second.data[2] + shift.x * child_first.data[2] +
                          0.5f * shift.x * shift.x * zero_child.z;
    ref_second.data[3] +=
        child_second.data[3] +
        0.5f * (shift.x * child_first.data[3] + shift.y * child_first.data[0]) +
        0.5f * shift.x * shift.y * zero_child.x;
    ref_second.data[4] +=
        child_second.data[4] +
        0.5f * (shift.x * child_first.data[4] + shift.y * child_first.data[1]) +
        0.5f * shift.x * shift.y * zero_child.y;
    ref_second.data[5] +=
        child_second.data[5] +
        0.5f * (shift.x * child_first.data[5] + shift.y * child_first.data[2]) +
        0.5f * shift.x * shift.y * zero_child.z;
    ref_second.data[6] +=
        child_second.data[6] +
        0.5f * (shift.x * child_first.data[6] + shift.z * child_first.data[0]) +
        0.5f * shift.x * shift.z * zero_child.x;
    ref_second.data[7] +=
        child_second.data[7] +
        0.5f * (shift.x * child_first.data[7] + shift.z * child_first.data[1]) +
        0.5f * shift.x * shift.z * zero_child.y;
    ref_second.data[8] +=
        child_second.data[8] +
        0.5f * (shift.x * child_first.data[8] + shift.z * child_first.data[2]) +
        0.5f * shift.x * shift.z * zero_child.z;
    ref_second.data[9] += child_second.data[9] + shift.y * child_first.data[3] +
                          0.5f * shift.y * shift.y * zero_child.x;
    ref_second.data[10] += child_second.data[10] +
                           shift.y * child_first.data[4] +
                           0.5f * shift.y * shift.y * zero_child.y;
    ref_second.data[11] += child_second.data[11] +
                           shift.y * child_first.data[5] +
                           0.5f * shift.y * shift.y * zero_child.z;
    ref_second.data[12] +=
        child_second.data[12] +
        0.5f * (shift.y * child_first.data[6] + shift.z * child_first.data[3]) +
        0.5f * shift.y * shift.z * zero_child.x;
    ref_second.data[13] +=
        child_second.data[13] +
        0.5f * (shift.y * child_first.data[7] + shift.z * child_first.data[4]) +
        0.5f * shift.y * shift.z * zero_child.y;
    ref_second.data[14] +=
        child_second.data[14] +
        0.5f * (shift.y * child_first.data[8] + shift.z * child_first.data[5]) +
        0.5f * shift.y * shift.z * zero_child.z;
    ref_second.data[15] += child_second.data[15] +
                           shift.z * child_first.data[6] +
                           0.5f * shift.z * shift.z * zero_child.x;
    ref_second.data[16] += child_second.data[16] +
                           shift.z * child_first.data[7] +
                           0.5f * shift.z * shift.z * zero_child.y;
    ref_second.data[17] += child_second.data[17] +
                           shift.z * child_first.data[8] +
                           0.5f * shift.z * shift.z * zero_child.z;
  }

  // 4. Run extraction kernel to collect the dequantized GPU results
  thrust::device_vector<UnpackedTailorCoefficients> unpacked_d(1);
  get_root_coefficients_kernel<<<1, 1>>>(backend->m_bvh8_nodes,
                                         unpacked_d.data().get());

  UnpackedTailorCoefficients unpacked_h = unpacked_d[0];

  // 5. Establish Quantization Error Bounds
  // Allow a tiny margin (e.g., 5e-2f) for floating-point accumulation sequence
  // alterations
  const float tol_low = (0.5f * unpacked_h.scale_factor_low) + 5e-2f;
  const float tol_second = (0.5f * unpacked_h.scale_factor_second) + 1e-1f;

  // --- ASSERTIONS ---
  // Establish Quantization Error Bounds + 1% relative drift margin for float32
  // tree reassociation
  const float relative_margin = 0.01f;

  // Verify Zero Order
  EXPECT_NEAR(unpacked_h.zero_order.x, ref_zero.x,
              (0.5f * unpacked_h.scale_factor_low) +
                  (relative_margin * std::abs(ref_zero.x)));
  EXPECT_NEAR(unpacked_h.zero_order.y, ref_zero.y,
              (0.5f * unpacked_h.scale_factor_low) +
                  (relative_margin * std::abs(ref_zero.y)));
  EXPECT_NEAR(unpacked_h.zero_order.z, ref_zero.z,
              (0.5f * unpacked_h.scale_factor_low) +
                  (relative_margin * std::abs(ref_zero.z)));

  // Verify First Order
  for (int i = 0; i < 9; ++i) {
    float tol = (0.5f * unpacked_h.scale_factor_low) +
                (relative_margin * std::abs(ref_first.data[i]));
    EXPECT_NEAR(unpacked_h.first_order.data[i], ref_first.data[i], tol);
  }

  // Verify Second Order
  for (int i = 0; i < 18; ++i) {
    float tol = (0.5f * unpacked_h.scale_factor_second) +
                (relative_margin * std::abs(ref_second.data[i]));
    EXPECT_NEAR(unpacked_h.second_order.data[i], ref_second.data[i], tol);
  }
}
