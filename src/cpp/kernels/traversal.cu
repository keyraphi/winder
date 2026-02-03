#include "aabb.h"
#include "bvh8.h"
#include "common.cuh"
#include "geometry.h"
#include "mat3x3.h"
#include "node_approx.cuh"
#include "tailor_coefficients.h"
#include "tensor3.h"
#include "traversal.cuh"
#include "vec3.h"
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
#include <limits>
#include <vector_functions.h>
#include <vector_types.h>

__device__ __forceinline__ auto warp_reduce_add_xor(float val) -> float {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
  }
  return val;
}

template <typename T>
__device__ __forceinline__ void
load_shared_cooperative(T *shared_dst, const T *global_src, uint32_t lane_id) {
  // Treat the structure as an array of uint32_t
  // BVH8Node is 128 bytes = 32 x 4 bytes. Perfect mapping for 32 threads.
  static_assert(sizeof(T) % 4 == 0, "Structure must be 4-byte aligned size");
  const uint32_t words = sizeof(T) / 4;

  auto *dst_u32 = reinterpret_cast<uint32_t *>(shared_dst);
  const auto *src_u32 = reinterpret_cast<const uint32_t *>(global_src);

  // If structure is exactly 128 bytes (BVH8Node)
  if (words == 32) {
    dst_u32[lane_id] = src_u32[lane_id];
  }
  // If structure is 32 bytes (LeafPointers)
  else if (words == 8) {
    if (lane_id < 8) {
      dst_u32[lane_id] = src_u32[lane_id];
    }
  } else {
    // Generic fallback
    for (uint32_t i = lane_id; i < words; i += 32) {
      dst_u32[i] = src_u32[i];
    }
  }
}

template <IsGeometry Geometry>
__global__ void __launch_bounds__(128, 8) compute_winding_numbers_kernel(
    const Vec3 *queries, const uint32_t *sort_indirections,
    const BVH8Node *bvh8_nodes, const LeafPointers *bvh8_leaf_pointers,
    const TailorCoefficientsBf16 *leaf_coefficients,
    const Geometry *sorted_geometry, const uint32_t query_count,
    const uint32_t geometry_count, float *winding_numbers, const float beta_2,
    const float inv_epsilon) {

  const uint32_t warp_id = threadIdx.x / 32;
  const uint32_t lane_id = threadIdx.x % 32;

  // We split the queries into tiles of 128.
  // Each block of 128 threads works on one tile at a time.
  // 4 warps per 128 threads
  // Each warp has its own shared traversal stack.
  __shared__ uint32_t shared_stack[4][64];
  __shared__ BVH8Node current_node_cache[4];
  __shared__ LeafPointers shared_leaf_ptrs[4];

  static __device__ uint32_t global_device_idx = 0;
  __shared__ uint32_t tile_base;

  while (true) {
    // Claim a tile
    // Dynamic work balancing. Not all blocks will need the same amount of time
    // for their queries
    if (threadIdx.x == 0) {
      tile_base = atomicAdd(&global_device_idx, blockDim.x);
    }
    __syncthreads();

    // keep track of what subtrees have been approximated for this query
    int my_required_stack_depth = std::numeric_limits<int>::max();

    uint32_t my_query_idx = tile_base + threadIdx.x;
    Vec3 my_query{0.F, 0.F, 0.F};
    uint32_t original_query_idx = 0xFFFFFFFF;
    if (my_query_idx >= query_count) {
      uint32_t query_count_warpstep = 32 * ((query_count + 32 - 1) / 32);
      if (my_query_idx >= query_count_warpstep) {
        // The entire warp is out of bounds and can safely return.
        return;
      }
      // This thread has no query, but still needs to help the others in the
      // warp with their computations
      my_required_stack_depth = -1;
    } else {
      // load query with sort indirections
      original_query_idx = sort_indirections[my_query_idx];
      my_query = queries[original_query_idx];
    }
    float my_winding_number = 0.F;

    // Always start traversal on root
    int stack_ptr = 0;
    if (lane_id == 0) {
      shared_stack[warp_id][stack_ptr++] = 0;
    }

    // Do traversal
    while (true) {
      // Check if stack is empty
      uint32_t stack_not_empty_mask = __ballot_sync(0xFFFFFFFF, stack_ptr > 0);
      // stop if all warps are done
      if (stack_not_empty_mask == 0) {
        break; // this tile is done
      }
      // warp leader pops next node into cache
      uint32_t current_node_idx;
      if (lane_id == 0) {
        current_node_idx = shared_stack[warp_id][--stack_ptr];
      }
      // load current node to shared cache
      load_shared_cooperative<BVH8Node>(&current_node_cache[warp_id],
                                        bvh8_nodes + current_node_idx, lane_id);
      __syncwarp();

      // share current_node_idx with the rest
      current_node_idx = __shfl_sync(0xFFFFFFFF, current_node_idx, 0);
      const BVH8Node &current_node = current_node_cache[warp_id];

      // Check if this thread needs to process the current node
      bool is_active = stack_ptr < my_required_stack_depth;
      if (is_active) {
        // this node is interested in the current subtree and will also have to
        // look at everything that will be put onto the stack to traverse this
        // subtree.
        my_required_stack_depth = std::numeric_limits<int>::max();
      }

      // process current node
      // Check the nodes parent_aabb. If it is too far away
      if (is_active && should_inner_node_be_aproximated(
                           my_query, current_node.parent_aabb, beta_2)) {
        // Get tailor_coefficients
        float scale_factor =
            current_node.tailor_coefficients.get_shared_scale_factor();
        // Zero Order
        Vec3_bf16 zero_order_coeff =
            current_node.tailor_coefficients.get_tailor_zero_order(
                scale_factor);
        // First Order
        Mat3x3_bf16 first_order_coeff =
            current_node.tailor_coefficients.get_tailor_first_order(
                scale_factor);
        // Second order
        Tensor3_bf16_compressed second_order_coeff =
            current_node.tailor_coefficients.get_tailor_second_order(
                scale_factor);

        // Do actual approximation
        my_winding_number += compute_node_approximation(
            my_query,
            current_node.parent_aabb.center_of_mass.get(
                current_node.parent_aabb.min,
                current_node.parent_aabb.diagonal()),
            zero_order_coeff, first_order_coeff, second_order_coeff);

        // Remember that I have the full contribution of this node already.
        my_required_stack_depth = stack_ptr;
      }
      bool is_still_active = my_required_stack_depth > stack_ptr;
      uint32_t interest_mask = __ballot_sync(0xFFFFFFFF, is_still_active);
      if (interest_mask == 0) {
        // This subtree is done for all threads in the warp.
        continue;
      }
      // load leaf ptrs to shared memory
      bool is_leaf = lane_id < 8
                         ? current_node.getChildMeta(lane_id) == ChildType::LEAF
                         : false;
      bool leaf_mask = __ballot_sync(0x000000FF, is_leaf);
      if (leaf_mask > 0) {
        load_shared_cooperative(&shared_leaf_ptrs[warp_id],
                                &bvh8_leaf_pointers[current_node_idx], lane_id);
        __syncwarp();
      }

      // Go through all childs together
      uint32_t added_inner_node_counter = 0;
#pragma unroll
      for (uint32_t child_idx = 0; child_idx < 8; ++child_idx) {
        ChildType child_type = current_node.getChildMeta(child_idx);
        if (child_type == ChildType::EMPTY) {
          continue;
        }
        if (child_type == ChildType::LEAF) {
          uint32_t leaf_idx = shared_leaf_ptrs[warp_id].indices[child_idx];
          AABB child_aabb = AABB::from_approximation(
              current_node.parent_aabb,
              current_node.child_aabb_approx[child_idx]);
          bool is_detail_eval_needed = true;
          if (is_still_active &&
              should_leaf_node_be_aproximated(my_query, child_aabb, beta_2)) {
            // load leaf tailor coefficient from global memory
            // 60 bytes
            // TODO test if it is worth approximating this.
            const TailorCoefficientsBf16 &current_leaf_coefficients =
                leaf_coefficients[leaf_idx];
            const Vec3 leaf_center_of_mass =
                current_leaf_coefficients.center_of_mass.get(
                    child_aabb.min, child_aabb.diagonal());
            my_winding_number += compute_node_approximation(
                my_query, leaf_center_of_mass,
                current_leaf_coefficients.zero_order,
                current_leaf_coefficients.first_order,
                current_leaf_coefficients.second_order);
            is_detail_eval_needed = false;
          }
          uint32_t detailed_leaf_evaluation_mask = __ballot_sync(
              0xFFFFFFFF, is_detail_eval_needed && is_still_active);

          if (detailed_leaf_evaluation_mask == 0) {
            // leaf contribution was approximated by all interested threads.
            // Continue with next child.
            continue;
          }
          // Detailed evaluation of leaf is needed for some threads.
          uint32_t my_geometry_idx = leaf_idx * 32 + lane_id;
          bool is_my_geometry_in_bounds = my_geometry_idx < geometry_count;
          Geometry my_geometry;
          my_geometry =
              Geometry::load(sorted_geometry, my_geometry_idx, geometry_count);
          // Use full warp to compute contributions of interested queries one by
          // one
          Vec3 shared_query;
          while (detailed_leaf_evaluation_mask > 0) {
            int current_leader = __ffs(detailed_leaf_evaluation_mask) - 1;
            // set current leader bit to 0
            detailed_leaf_evaluation_mask =
                detailed_leaf_evaluation_mask & (~(1u << current_leader));
            // Get the query from leader
            shared_query.x =
                __shfl_sync(0xFFFFFFFF, my_query.x, current_leader);
            shared_query.y =
                __shfl_sync(0xFFFFFFFF, my_query.y, current_leader);
            shared_query.z =
                __shfl_sync(0xFFFFFFFF, my_query.z, current_leader);
            // compute the winding number contribution from my geometry
            float my_contribution = 0.F;
            if (is_my_geometry_in_bounds) {
              my_contribution =
                  my_geometry.contributionToQuery(shared_query, inv_epsilon);
            }
            // sum up all contributions in warp
            float total_contribution = warp_reduce_add_xor(my_contribution);
            // only leader adds that contribution
            if ((int)lane_id == current_leader) {
              my_winding_number += total_contribution;
            }
          }
        } else {
          // Child is inner node
          if (lane_id == 0) {
            uint32_t child_node_idx =
                current_node.child_base + added_inner_node_counter;
            added_inner_node_counter++;
            shared_stack[warp_id][stack_ptr++] = child_node_idx;
          }
        }
      } // child for
    } // traversal while
    // winding number computation is complete
    if (my_required_stack_depth >= 0) {
      winding_numbers[original_query_idx] = my_winding_number;
    }
  } // grid while
}

template <IsGeometry Geometry>
__global__ void compute_winding_numbers_single_leaf_kernel(
    const Vec3 *queries, const uint32_t *sort_indirections,
    const Geometry *sorted_geometry, const uint32_t query_count,
    const uint32_t geometry_count, float *winding_numbers,
    const float inv_epsilon) {
  // Global index of the query this thread is responsible for
  uint32_t my_query_idx = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ Geometry shared_geometry[32];
  if (threadIdx.x < geometry_count) {
    shared_geometry[threadIdx.x].load(sorted_geometry, threadIdx.x, geometry_count);
  }
  __syncthreads();

  if (my_query_idx >= query_count)
    return;

  // Load query
  uint32_t original_idx = sort_indirections[my_query_idx];
  Vec3 my_query = queries[original_idx];
  float my_winding_number = 0.0f;

  // Since there is only one leaf, all geometry (1-32 elements)
  // is stored at the beginning of sorted_geometry.
  // Every thread iterates through all available geometry for its specific
  // query.
  for (uint32_t i = 0; i < geometry_count; ++i) {
    my_winding_number += shared_geometry[i].contributionToQuery(my_query, inv_epsilon);
  }

  // Write out results
  winding_numbers[original_idx] = my_winding_number;
}

template <IsGeometry Geometry>
void compute_winding_numbers(
    const ComputeWindingNumbersParams<Geometry> &params, int device_id,
    const cudaStream_t &stream) {
  if (params.query_count == 0) {
    return;
  }

  float inv_epsilon = 1.F / params.epsilon;
  // There i no tree if there is only one leaf
  if (params.geometry_count <= 32) {
    uint32_t threads = 256;
    uint32_t blocks = (params.query_count + threads - 1) / threads;

    compute_winding_numbers_single_leaf_kernel<Geometry>
        <<<blocks, threads, 0, stream>>>(
            params.queries, params.sort_indirections, params.sorted_geometry,
            params.query_count, params.geometry_count, params.winding_numbers,
            inv_epsilon);
    return;
  }

  float beta_2 = params.beta * params.beta;

  int threads = 128;
  int blocks_per_sm = 0;

  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, compute_winding_numbers_kernel<Geometry>, threads, 0);

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device_id);
  int blocks = blocks_per_sm * deviceProp.multiProcessorCount;

  compute_winding_numbers_kernel<Geometry><<<blocks, threads, 0, stream>>>(
      params.queries, params.sort_indirections, params.bvh8_nodes,
      params.bvh8_leaf_pointers, params.leaf_coefficients,
      params.sorted_geometry, params.query_count, params.geometry_count,
      params.winding_numbers, beta_2, inv_epsilon);
  CUDA_CHECK(cudaGetLastError());
}

template void compute_winding_numbers<PointNormal>(
    const ComputeWindingNumbersParams<PointNormal> &params, int device_id,
    const cudaStream_t &stream);
template void compute_winding_numbers<Triangle>(
    const ComputeWindingNumbersParams<Triangle> &params, int device_id,
    const cudaStream_t &stream);
