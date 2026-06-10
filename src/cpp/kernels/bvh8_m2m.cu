#include "aabb.h"
#include "bvh8.h"
#include "common.cuh"
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
    BVH8Node *nodes, const uint32_t *internal_parent_map,
    SoAViewConst<Vec3> leaf_zero_order, const uint32_t *leaf_parents,
    const LeafPointers *leaf_pointers, const uint32_t *node_child_count,
    const uint32_t leaf_count, uint32_t *atomic_counters) {

  uint32_t leaf_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (leaf_idx >= leaf_count) {
    return;
  }

  uint32_t current_node_idx = leaf_parents[leaf_idx];

  while (current_node_idx != 0xFFFFFFFF) {
    BVH8Node node = nodes[current_node_idx];
    // race to the top
    uint32_t expected_children = node_child_count[current_node_idx];

    if (atomicAdd(&atomic_counters[current_node_idx], 1) <
        expected_children - 1) {
      return;
    }

    Vec3 zero_order = {0.F, 0.F, 0.F};
    uint32_t internal_child_count = 0;
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
      if (node.child_meta[i] == ChildType::EMPTY) {
        continue;
      }
      Vec3 child_zero_order;
      if (node.child_meta[i] == ChildType::LEAF) {
        uint32_t leaf_idx = leaf_pointers[current_node_idx].indices[i];
        child_zero_order = Vec3::load(leaf_zero_order, leaf_idx, leaf_count);
      } else {
        uint32_t child_idx = node.child_base + internal_child_count;
        internal_child_count++;
        BVH8Node child_node = nodes[child_idx];
        child_zero_order = child_node.zero_order_coefficients;
      }

      // zero order doesn't change
      // M_0' = M_0
      const Vec3 &zero_child = child_zero_order;
      zero_order += zero_child;
    }
    // store accumulated coefficients in current node (quantized)
    node.zero_order_coefficients = zero_order;
    nodes[current_node_idx] = node;

    __threadfence(); // ensure the tailor coefficients of childs are written
                     // before continuing with the next
    current_node_idx = __ldcg(internal_parent_map + current_node_idx);
  }
}

void compute_internal_tailor_coefficients_m2m(
    BVH8Node *nodes, const uint32_t *internal_parent_map,
    const float *leaf_zero_order, const uint32_t *leaf_parents,
    const LeafPointers *leaf_pointers, const uint32_t *node_child_count,
    const uint32_t leaf_count, uint32_t *atomic_counters,
    const cudaStream_t &stream) {
  uint32_t threads = 256;
  uint32_t blocks = (leaf_count + threads - 1) / threads;
  compute_internal_tailor_coefficients_m2m_kernel<<<blocks, threads, 0,
                                                    stream>>>(
      nodes, internal_parent_map,
      SoAViewConst<Vec3>{const_cast<float *>(leaf_zero_order), leaf_count},
      leaf_parents, leaf_pointers, node_child_count, leaf_count,
      atomic_counters);
  CUDA_CHECK(cudaGetLastError());
}
