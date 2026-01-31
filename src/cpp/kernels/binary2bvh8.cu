#include "aabb.h"
#include "binary2bvh8.cuh"
#include "binary_node.h"
#include "bvh8.h"
#include "common.cuh"
#include "vec3.h"
#include <cmath>
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
#include <stdexcept>
#include <vector_functions.h>
#include <vector_types.h>

namespace cg = cooperative_groups;

__global__ void
convert_binary_tree_to_bvh8_kernel(ConvertBinary2BVH8Params params) {

  cg::grid_group grid = cg::this_grid();
  cg::thread_block block = cg::this_thread_block();

  // These stay in registers, consistent across the whole grid
  uint32_t current_level_width = 1;
  uint32_t current_level_offset = 0;
  uint32_t level_iteration = 0;

  // Initialization only first thread in grid
  if (grid.thread_rank() == 0) {
    params.work_queue_A[0] = 0;
    params.bvh8_internal_parents[0] = 0xFFFFFFFF;
    *(params.global_counter) = 0;
  }
  grid.sync();

  while (true) {
    // pick the correct queue based on the level
    uint32_t *work_queue_in =
        level_iteration % 2 == 0 ? params.work_queue_A : params.work_queue_B;
    uint32_t *work_queue_out =
        level_iteration % 2 == 0 ? params.work_queue_B : params.work_queue_A;
    grid.sync();

    // level_width could be > number of threads in grid
    // for (uint32_t idx = grid.thread_rank(); idx < current_level_width;
    //      idx += grid.size()) {
    uint32_t level_width_blocks =
        (current_level_width + blockDim.x - 1) / blockDim.x;
    for (uint32_t idx = grid.thread_rank();
         idx < level_width_blocks * blockDim.x; idx += grid.size()) {

      uint32_t internal_node_count = 0;
      uint32_t my_binary_idx = 0;
      uint32_t child_count = 1;
      uint32_t children[8];
      if (idx < current_level_width) {
        my_binary_idx = work_queue_in[idx];
        children[0] = my_binary_idx;
        for (int iter = 0; iter < 7; ++iter) {
          float max_diag = -1.F;
          int split_idx = -1;

          // Find the best candidate to split
          for (int i = 0; i < child_count; ++i) {
            uint32_t binary_idx = children[i];
            if (!BinaryNode::is_leaf(binary_idx, params.leaf_count)) {
              // find the inner node with the largest aabb
              float d = params.binary_aabbs[binary_idx].radius_sq();
              if (d > max_diag) {
                max_diag = d;
                split_idx = i;
              }
            }
          }

          if (split_idx == -1) {
            break; // No more internal nodes to split
          }

          // Split the winner
          uint32_t node_to_split = children[split_idx];
          children[split_idx] = params.binary_nodes[node_to_split].left_child;
          children[child_count] =
              params.binary_nodes[node_to_split].right_child;
          child_count++;
        }
        // count how many internal nodes (non leafs) were found
        for (uint32_t i = 0; i < child_count; ++i) {
          if (!BinaryNode::is_leaf(children[i], params.leaf_count)) {
            internal_node_count++;
          }
        }
      }

      // At this point it is guaranteed that a full block is running
      // We need to know where in the bvh8_nodes array to store the children
      // (child_base) prefix sum over internal node count to find out the offset
      // for this thread, at least locally in this block.
      typedef cub::BlockScan<uint32_t, 256>
          BlockScan; // Assuming 256 threads per block
      __shared__ typename BlockScan::TempStorage temp_storage;

      uint32_t thread_offset;
      uint32_t total_internal_in_block;
      BlockScan(temp_storage)
          .ExclusiveSum(internal_node_count, thread_offset,
                        total_internal_in_block);

      // Get device wide offset of this block by claiming the current value from
      // the global counter. Only done once per block
      __shared__ uint32_t global_base;
      if (threadIdx.x == 0) {
        // next block that claims the counter will leaf space for all internal
        // nodes in this block. This counter can be used to find out how long
        // work_queue_out is in the end.
        global_base = atomicAdd(params.global_counter, total_internal_in_block);
      }
      block.sync();

      // we don't need the full block in this for loop anymore
      if (idx >= current_level_width) {
        break;
      }

      // index for this bvh8 node
      uint32_t bvh8_idx = current_level_offset + idx;
      // This thread will write its childs at its offset within the block
      // go to start of level -> skip inner nodes of this level ->  skip childs
      // of earlier blocks -> skip childs earlier threas in this block
      uint32_t my_child_base = current_level_offset + current_level_width +
                               global_base + thread_offset;

      BVH8Node out_node;
      AABB parent_aabb = params.binary_aabbs[my_binary_idx];
      out_node.parent_aabb = parent_aabb;
      out_node.child_base = my_child_base;

      // AABB Quantization & Writing out the BVH8Node
      Vec3 parent_min = parent_aabb.min;
      // Note: can be inf - doesn't matter because of implementation of
      // quantize_aabb
      Vec3 parent_inv_ext = 1.F / (parent_aabb.max - parent_aabb.min);

      int internal_found = 0;
      uint32_t my_leaf_indices[8];

#pragma unroll
      for (int i = 0; i < 8; ++i) {
        if (i < child_count) {
          uint32_t binary_child_idx = children[i];
          bool is_child_leaf =
              BinaryNode::is_leaf(binary_child_idx, params.leaf_count);

          // Quantize Child AABB
          AABB child_box = params.binary_aabbs[binary_child_idx];

          out_node.child_aabb_approx[i] = AABB8BitApprox::quantize_aabb(
              child_box, parent_min, parent_inv_ext);

          if (is_child_leaf) {
            out_node.setChildMeta(i, ChildType::LEAF);
            uint32_t leaf_raw_idx = binary_child_idx - (params.leaf_count - 1);

            my_leaf_indices[i] = leaf_raw_idx;

            // Let the leaf know which BVH8Node is its parent
            params.bvh8_leaf_parents[leaf_raw_idx] = bvh8_idx;
          } else {
            out_node.setChildMeta(i, ChildType::INTERNAL);
            // Calculate the specific global index for this child in the next
            // level
            uint32_t next_bvh8_idx = my_child_base + internal_found;

            // The work_queue_out points to the inner BinaryNode that must be
            // turned into BVH8Node in the next kernel call
            work_queue_out[global_base + thread_offset + internal_found] =
                binary_child_idx;

            params.bvh8_internal_parents[next_bvh8_idx] = bvh8_idx;

            my_leaf_indices[i] = 0xFFFFFFFF;
            internal_found++;
          }
        } else {
          out_node.setChildMeta(i, ChildType::EMPTY);
          my_leaf_indices[i] = 0xFFFFFFFF;
        }
      }

      // let the inner nodes know how many childs they have
      out_node.tailor_coefficients.set_expected_children(child_count);

      // Coalesced Writes
      params.bvh8_nodes[bvh8_idx] = out_node;

      // Parallel Leaf Pointers (32-byte vectorized write)
      auto *leaf_ptr_v4 = reinterpret_cast<uint4 *>(params.bvh8_leaf_pointers);
      leaf_ptr_v4[bvh8_idx * 2] =
          make_uint4(my_leaf_indices[0], my_leaf_indices[1], my_leaf_indices[2],
                     my_leaf_indices[3]);
      leaf_ptr_v4[bvh8_idx * 2 + 1] =
          make_uint4(my_leaf_indices[4], my_leaf_indices[5], my_leaf_indices[6],
                     my_leaf_indices[7]);
    }
    // level synchronization
    grid.sync();
    uint32_t next_level_width = *(params.global_counter);
    current_level_offset += current_level_width;
    current_level_width = next_level_width;
    level_iteration++;
    grid.sync(); // ensure global_counter has been read by everyone
    if (grid.thread_rank() == 0) {
      *(params.global_counter) = 0;
    }
    grid.sync();

    if (current_level_width == 0) {
      break;
    }
  }
}

void convert_binary_tree_to_bvh8(ConvertBinary2BVH8Params params,
                                 const int device_id,
                                 const cudaStream_t &stream) {
  int threads = 256;
  int blocks_per_sm = 0;

  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, convert_binary_tree_to_bvh8_kernel, threads, 0);

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device_id);
  int blocks = blocks_per_sm * deviceProp.multiProcessorCount;

  // Check if the device actually supports cooperative launch
  if (!deviceProp.cooperativeLaunch) {
    throw std::runtime_error("Device does not support Cooperative Launch");
  }

  void *args[] = {&params};
  cudaLaunchCooperativeKernel((void *)convert_binary_tree_to_bvh8_kernel,
                              blocks, threads, args, 0, stream);
  CUDA_CHECK(cudaGetLastError());
}
