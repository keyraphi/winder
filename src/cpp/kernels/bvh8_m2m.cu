#include "aabb.h"
#include "bvh8.h"
#include "common.cuh"
#include "mat3x3.h"
#include "tailor_coefficients.h"
#include "tensor3.h"
#include "vec3.h"
#include "winder_cuda.h"
#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
#include <cstdint>
#include <cub/block/block_scan.cuh>
#include <cub/util_type.cuh>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_atomic_functions.h>
#include <driver_types.h>
#include <vector_functions.h>
#include <vector_types.h>

__global__ void compute_internal_tailor_coefficients_m2m_kernel(
    const BVH8Node *nodes, const uint32_t *internal_parent_map,
    const AABB *leaf_aabbs, const TailorCoefficientsBf16 *leaf_coefficients,
    const uint32_t *leaf_parents, const LeafPointers *leaf_pointers,
    const uint32_t leaf_count, uint32_t *atomic_counters) {

  uint32_t leaf_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (leaf_idx >= leaf_count) {
    return;
  }

  uint32_t current_node_idx = leaf_parents[leaf_idx];

  while (current_node_idx != 0xFFFFFFFF) {
    BVH8Node node = nodes[current_node_idx];
    // race to the top
    uint32_t expected_children =
        nodes[current_node_idx].tailor_coefficients.get_expected_children();
    __threadfence(); // ensure the tailor coefficients of childs are written
    if (atomicAdd(&atomic_counters[current_node_idx], 1) <
        expected_children - 1) {
      return;
    }

    // only one thread for current node survives
    Vec3 parent_center = node.parent_aabb.center();

    Vec3 zero_order = {0.F, 0.F, 0.F};
    Mat3x3 first_order = {0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F};
    Tensor3_compressed second_order = {0.F, 0.F, 0.F, 0.F, 0.F, 0.F,
                                       0.F, 0.F, 0.F, 0.F, 0.F, 0.F,
                                       0.F, 0.F, 0.F, 0.F, 0.F, 0.F};

    uint32_t internal_child_count = 0;
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
      if (node.child_meta[i] == ChildType::EMPTY) {
        continue;
      }
      Vec3 child_center;
      TailorCoefficientsBf16 child_coefficients;
      if (node.child_meta[i] == ChildType::LEAF) {
        uint32_t leaf_idx = leaf_pointers[current_node_idx].indices[i];
        child_center = leaf_aabbs[leaf_idx].center();
        child_coefficients = leaf_coefficients[leaf_idx];
      } else {
        uint32_t child_idx = node.child_base + internal_child_count;
        internal_child_count++;
        BVH8Node child_node = nodes[child_idx];
        child_center = child_node.parent_aabb.center();
        float shared_scale =
            child_node.tailor_coefficients.get_shared_scale_factor();
        child_coefficients.zero_order =
            child_node.tailor_coefficients.get_tailor_zero_order(shared_scale);
        child_coefficients.first_order =
            child_node.tailor_coefficients.get_tailor_first_order(shared_scale);
        child_coefficients.second_order =
            child_node.tailor_coefficients.get_tailor_second_order(
                shared_scale);
      }

      // Merge child coeficients into parent tailor by recentering child
      Vec3 shift_vector = child_center - parent_center; // v in equations
      // zero order doesn't change
      // M_0' = M_0
      Vec3 zero_child = Vec3::from_bf16(child_coefficients.zero_order);
      zero_order += zero_child;
      // first order
      // M_1' = M_1 + v \otimes M_0
      Mat3x3 child_first = Mat3x3::from_bf16(child_coefficients.first_order);
      first_order += child_first + shift_vector.outer_product(zero_child);
      // second order
      // M_2' = M_2 + 1/2 * (v \times M_1 + perm(v \times M_1)) + 1/2 * (v
      // \times v \times M_0)
      // M2_jkl′ = M2_jkl + 1/2*(v_j * M1_kl + v_k * M1_jl) + 1/2 v_j * v_k *
      //           M0_l
      Tensor3_compressed child_second =
          Tensor3_compressed::from_bf16(child_coefficients.second_order);
      second_order.data[0] +=
          child_second.data[0] + shift_vector.x * child_first.data[0] +
          0.5F * shift_vector.x * shift_vector.x * zero_child.x;
      second_order.data[1] +=
          child_second.data[1] + shift_vector.x * child_first.data[1] +
          0.5F * shift_vector.x * shift_vector.x * zero_child.y;
      second_order.data[2] +=
          child_second.data[2] + shift_vector.x * child_first.data[2] +
          0.5F * shift_vector.x * shift_vector.x * zero_child.z;
      second_order.data[3] +=
          child_second.data[3] +
          0.5F * (shift_vector.x * child_first.data[3] +
                  shift_vector.y * child_first.data[0]) +
          0.5F * shift_vector.x * shift_vector.y * zero_child.x;
      second_order.data[4] +=
          child_second.data[4] +
          0.5F * (shift_vector.x * child_first.data[4] +
                  shift_vector.y * child_first.data[1]) +
          0.5F * shift_vector.x * shift_vector.y * zero_child.y;
      second_order.data[5] +=
          child_second.data[5] +
          0.5F * (shift_vector.x * child_first.data[5] +
                  shift_vector.y * child_first.data[2]) +
          0.5F * shift_vector.x * shift_vector.y * zero_child.z;
      second_order.data[6] +=
          child_second.data[6] +
          0.5F * (shift_vector.x * child_first.data[6] +
                  shift_vector.z * child_first.data[0]) +
          0.5F * shift_vector.x * shift_vector.z * zero_child.x;
      second_order.data[7] +=
          child_second.data[7] +
          0.5F * (shift_vector.x * child_first.data[7] +
                  shift_vector.z * child_first.data[1]) +
          0.5F * shift_vector.x * shift_vector.z * zero_child.y;
      second_order.data[8] +=
          child_second.data[8] +
          0.5F * (shift_vector.x * child_first.data[8] +
                  shift_vector.z * child_first.data[2]) +
          0.5F * shift_vector.x * shift_vector.z * zero_child.z;
      // second_order.data[-] += child_second.data[-] + 0.5F * (shift_vector.y *
      // child_first.data[0] + shift_vector.x * child_first.data[3]) + 0.5F *
      // shift_vector.y * shift_vector.x *zero_child.x // not stored
      // second_order.data[-] += child_second.data[-] + 0.5F * (shift_vector.y *
      // child_first.data[1] + shift_vector.x * child_first.data[4]) + 0.5F *
      // shift_vector.y * shift_vector.x *zero_child.y // not stored
      // second_order.data[-] += child_second.data[-] + 0.5F * (shift_vector.y *
      // child_first.data[2] + shift_vector.x * child_first.data[5]) + 0.5F *
      // shift_vector.y * shift_vector.x *zero_child.z // not stored
      second_order.data[9] +=
          child_second.data[9] + shift_vector.y * child_first.data[3] +
          0.5F * shift_vector.y * shift_vector.y * zero_child.x;
      second_order.data[10] +=
          child_second.data[10] + shift_vector.y * child_first.data[4] +
          0.5F * shift_vector.y * shift_vector.y * zero_child.y;
      second_order.data[11] +=
          child_second.data[11] + shift_vector.y * child_first.data[5] +
          0.5F * shift_vector.y * shift_vector.y * zero_child.z;
      second_order.data[12] +=
          child_second.data[12] +
          0.5F * (shift_vector.y * child_first.data[6] +
                  shift_vector.z * child_first.data[3]) +
          0.5F * shift_vector.y * shift_vector.z * zero_child.x;
      second_order.data[13] +=
          child_second.data[13] +
          0.5F * (shift_vector.y * child_first.data[7] +
                  shift_vector.z * child_first.data[4]) +
          0.5F * shift_vector.y * shift_vector.z * zero_child.y;
      second_order.data[14] +=
          child_second.data[14] +
          0.5F * (shift_vector.y * child_first.data[8] +
                  shift_vector.z * child_first.data[5]) +
          0.5F * shift_vector.y * shift_vector.z * zero_child.z;
      // second_order.data[-] += child_second.data[-] + 0.5F * (shift_vector.z *
      // child_first.data[0] + shift_vector.x * child_first.data[6]) + 0.5F *
      // shift_vector.z * shift_vector.x *zero_child.x // not stored
      // second_order.data[-] += child_second.data[-] + 0.5F * (shift_vector.z *
      // child_first.data[1] + shift_vector.x * child_first.data[7]) + 0.5F *
      // shift_vector.z * shift_vector.x *zero_child.y // not stored
      // second_order.data[-] += child_second.data[-] + 0.5F * (shift_vector.z *
      // child_first.data[2] + shift_vector.x * child_first.data[8]) + 0.5F *
      // shift_vector.z * shift_vector.x *zero_child.z // not stored
      // second_order.data[-] += child_second.data[-] + 0.5F * (shift_vector.z *
      // child_first.data[3] + shift_vector.y * child_first.data[6]) + 0.5F *
      // shift_vector.z * shift_vector.y *zero_child.x // not stored
      // second_order.data[-] += child_second.data[-] + 0.5F * (shift_vector.z *
      // child_first.data[4] + shift_vector.y * child_first.data[7]) + 0.5F *
      // shift_vector.z * shift_vector.y *zero_child.y // not stored
      // second_order.data[-] += child_second.data[-] + 0.5F * (shift_vector.z *
      // child_first.data[5] + shift_vector.y * child_first.data[8]) + 0.5F *
      // shift_vector.z * shift_vector.y *zero_child.z // not stored
      second_order.data[15] +=
          child_second.data[15] + shift_vector.z * child_first.data[6] +
          0.5F * shift_vector.z * shift_vector.z * zero_child.x;
      second_order.data[16] +=
          child_second.data[16] + shift_vector.z * child_first.data[7] +
          0.5F * shift_vector.z * shift_vector.z * zero_child.y;
      second_order.data[17] +=
          child_second.data[17] + shift_vector.z * child_first.data[8] +
          0.5F * shift_vector.z * shift_vector.z * zero_child.z;
    }
    // store accumulated coefficients in current node (quantized)
    node.tailor_coefficients.set_tailor_coefficients(zero_order, first_order,
                                                     second_order);

    current_node_idx = internal_parent_map[current_node_idx];
  }
}

void compute_internal_tailor_coefficients_m2m(
    const BVH8Node *nodes, const uint32_t *internal_parent_map,
    const AABB *leaf_aabbs, const TailorCoefficientsBf16 *leaf_coefficients,
    const uint32_t *leaf_parents, const LeafPointers *leaf_pointers,
    const uint32_t leaf_count, uint32_t *atomic_counters, const cudaStream_t &stream) {
  uint32_t threads = 256;
  uint32_t blocks = (leaf_count + threads - 1) / threads;
  compute_internal_tailor_coefficients_m2m_kernel<<<blocks, threads, 0, stream>>>(
      nodes, internal_parent_map, leaf_aabbs, leaf_coefficients, leaf_parents,
      leaf_pointers, leaf_count, atomic_counters);
  CUDA_CHECK(cudaGetLastError());
}
