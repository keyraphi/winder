#include "aabb.h"
#include "bvh8.h"
#include "common.cuh"
#include "geometry.h"
#include "mat3x3.h"
#include "node_approx.cuh"
#include "scene_normalization.h"
#include "soa.h"
#include "taylor_coefficients.h"
#include "tensor3.h"
#include "traversal.cuh"
#include "vec3.h"
#include "winding_numbers_backend.h"
#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cub/block/block_scan.cuh>
#include <cub/util_type.cuh>
#include <cuda_device_runtime_api.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_atomic_functions.h>
#include <driver_types.h>
#include <limits>
#include <type_traits>
#include <vector_functions.h>
#include <vector_types.h>

__device__ __forceinline__ auto warp_reduce_add_xor(float val) -> float {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
  }
  return val;
}

template <bool UseKahan>
__device__ __forceinline__ void payload_accum_scalar(float &acc, float &comp,
                                                     float v) {
  if constexpr (UseKahan) {
    kahan_add(acc, v, comp);
  } else {
    (void)comp;
    acc += v;
  }
}

template <bool UseKahan>
__device__ __forceinline__ void payload_accum_fma(float &acc, float &comp,
                                                  float w, float v) {
  if constexpr (UseKahan) {
    const float y = __fmaf_rn(w, v, -comp);
    const float t = ieee_add(acc, y);
    comp = ieee_sub(ieee_sub(t, acc), y);
    acc = t;
  } else {
    (void)comp;
    acc = __fmaf_rn(w, v, acc);
  }
}

template <typename T>
__device__ __forceinline__ void
load_shared_cooperative(T *shared_dst, const T *global_src, uint32_t lane_id) {
  static_assert(sizeof(T) % 4 == 0, "Structure must be 4-byte aligned size");
  constexpr uint32_t words = sizeof(T) / 4;

  auto *dst_u32 = reinterpret_cast<uint32_t *>(shared_dst);
  const auto *src_u32 = reinterpret_cast<const uint32_t *>(global_src);

  if constexpr (words == 16) { // TaylorCoefficientsF16 (64 bytes)
    if (lane_id < 16) {
      dst_u32[lane_id] = src_u32[lane_id];
    }
  } else if constexpr (words == 8) { // BVH8Node / LeafPointers /
                                     // BackwardTaylorCoefficientsF16 (32 bytes)
    if (lane_id < 8) {
      dst_u32[lane_id] = src_u32[lane_id];
    }
  } else { // Generic fallback
    for (uint32_t i = lane_id; i < words; i += 32) {
      dst_u32[i] = src_u32[i];
    }
  }
}

// Forward traversal kernel for computation of winding numbers is almost
// identical to the traversal kernel for the computation of the query gradients.
// We merged them into one kernel to make maintenance simpler.
// The differences are guarded with constexpr checks on the TraversalOutput type
enum class TraversalOutput { WindingNumbersField, QueryGradient };

template <IsGeometry Geometry, TraversalOutput Out>
__global__ void __launch_bounds__(128) traverse_geometry_tree_kernel(
    const Vec3 *__restrict__ queries,
    const uint32_t *__restrict__ sort_indirections,
    const BVH8Node *__restrict__ bvh8_nodes,
    const LeafPointers *__restrict__ bvh8_leaf_pointers,
    const TaylorCoefficientsF16 *__restrict__ node_coefficients,
    const TaylorCoefficientsF16 *__restrict__ leaf_coefficients,
    const AABB *__restrict__ leaf_aabbs,
    const SoAView<Geometry> sorted_geometry,
    const float *__restrict__ grad_output, // null for WindingNumbersField
    const uint32_t query_count, const uint32_t geometry_count,
    float *__restrict__ out_winding_numbers_field, // nullptr for QueryGradient
    float *__restrict__ out_query_gradients, // null for WindingNumbersField
    uint32_t *__restrict__ global_device_counter, const float beta_2,
    const typename Geometry::Context ctx, const SceneNormalization norm) {

  using Accum = std::conditional_t<Out == TraversalOutput::WindingNumbersField,
                                   float, Vec3>;

  const uint32_t warp_id = threadIdx.x / 32;
  const uint32_t lane_id = threadIdx.x % 32;

  // We split the queries into tiles of 32 threads.
  // Each warp of 32 threads works on one tile at a time.
  // 4 warps per block
  // Each warp has its own shared traversal stack.
  __shared__ uint32_t shared_stack[4][64];
  __shared__ BVH8Node current_node_cache[4];
  __shared__ TaylorCoefficientsF16 current_taylor_coefficients_cache[4];
  __shared__ LeafPointers shared_leaf_ptrs[4];
  uint32_t warp_tile_base;

  while (true) {
    // Claim a tile
    // Dynamic work balancing. Not all blocks will need the same amount of time
    // for their queries
    if (lane_id == 0) {
      warp_tile_base = atomicAdd(global_device_counter, 32);
    }
    warp_tile_base = __shfl_sync(0xFFFFFFFF, warp_tile_base, 0);

    if (warp_tile_base >= query_count) {
      // No more work for this warp
      break;
    }

    // keep track of what subtrees have been approximated for this query
    int my_required_stack_depth = std::numeric_limits<int>::max();

    uint32_t my_query_idx = warp_tile_base + lane_id;
    Vec3 my_query{0.F, 0.F, 0.F};
    uint32_t original_query_idx = 0xFFFFFFFF;
    if (my_query_idx >= query_count) {
      // This thread has no query, but still needs to help the others in the
      // warp with their computations
      my_required_stack_depth = -1;
    } else {
      // load query with sort indirections
      original_query_idx = sort_indirections[my_query_idx];
      my_query = norm.to_normalized(queries[original_query_idx]);
    }
    Accum my_result;
    Accum my_comp; // for kahan add
    if constexpr (Out == TraversalOutput::WindingNumbersField) {
      my_result = 0.F;
      my_comp = 0.F;
    } else {
      my_result = Vec3::zero();
      my_comp = Vec3::zero();
    }

    // Always start traversal on root
    int stack_ptr = 0;
    if (lane_id == 0) {
      shared_stack[warp_id][stack_ptr] = 0;
    }
    stack_ptr++; // increment stack pointer for all lanes

    float two_eps = 0.F;
    two_eps = 2.F * ctx.eps_length;
    // Do traversal as full warps
    while (true) {
      // Check if stack is empty
      uint32_t stack_not_empty_mask = __ballot_sync(0xFFFFFFFF, stack_ptr > 0);
      // stop if all warps are done
      if (stack_not_empty_mask == 0) {
        break; // this tile is done
      }
      // warp leader pops next node into cache
      uint32_t current_node_idx;
      --stack_ptr; // decrement stack pointer for all lanes
      if (lane_id == 0) {
        current_node_idx = shared_stack[warp_id][stack_ptr];
      }
      // share current_node_idx with the rest
      current_node_idx = __shfl_sync(0xFFFFFFFF, current_node_idx, 0);
      // load current node to shared cache
      load_shared_cooperative<BVH8Node>(&current_node_cache[warp_id],
                                        bvh8_nodes + current_node_idx, lane_id);
      __syncwarp();

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
      // Check the nodes parent_aabb. If it is too far away approximate using
      // taylor coefficients
      bool need_taylor_coefficients =
          is_active && should_node_be_approximated(
                           my_query, current_node.getAABB(), beta_2, two_eps);

      uint32_t load_taylor_coefficients_mask =
          __ballot_sync(0xFFFFFFFF, need_taylor_coefficients);
      if (load_taylor_coefficients_mask > 0) {
        load_shared_cooperative<TaylorCoefficientsF16>(
            &current_taylor_coefficients_cache[warp_id],
            node_coefficients + current_node_idx, lane_id);
        __syncwarp();
      }
      if (need_taylor_coefficients) {
        // taylor_coefficients dequantization
        TaylorCoefficientsF16 &current_node_coefficients =
            current_taylor_coefficients_cache[warp_id];
        // Zero Order
        Vec3_f16 zero_order_coeff = current_node_coefficients.zero_order;
        // First Order
        Mat3x3_f16 first_order_coeff = current_node_coefficients.first_order;
        // Second order
        Tensor3_bf16_compressed second_order_coeff =
            current_node_coefficients.second_order;

        AABB parent_aabb = current_node.getAABB();

        // Do approximation
        if constexpr (Out == TraversalOutput::WindingNumbersField) {
          float approx_contribution = compute_node_approximation(
              my_query, parent_aabb.center_of_mass, zero_order_coeff,
              first_order_coeff, second_order_coeff);
          kahan_add(my_result, approx_contribution, my_comp);
        } else {
          Vec3 approx_grad = compute_node_query_grad_approximation(
              my_query, parent_aabb.center_of_mass,
              current_node_coefficients.zero_order,
              current_node_coefficients.first_order,
              current_node_coefficients.second_order);
          kahan_add(my_result, approx_grad, my_comp);
        }

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
      uint32_t leaf_mask = __ballot_sync(0xFFFFFFFF, is_leaf);
      if (leaf_mask > 0) {
        load_shared_cooperative(&shared_leaf_ptrs[warp_id],
                                &bvh8_leaf_pointers[current_node_idx], lane_id);
        __syncwarp();
      }

      // Go through all childs together
      uint32_t added_inner_node_counter = 0;
      for (uint32_t child_idx = 0; child_idx < 8; ++child_idx) {
        ChildType child_type = current_node.getChildMeta(child_idx);
        if (child_type == ChildType::EMPTY) {
          continue;
        }
        if (child_type == ChildType::LEAF) {
          uint32_t leaf_idx = shared_leaf_ptrs[warp_id].indices[child_idx];
          AABB child_aabb = leaf_aabbs[leaf_idx];

          bool need_taylor_coefficients =
              is_still_active && should_node_be_approximated(
                                     my_query, child_aabb, beta_2, two_eps);
          uint32_t load_taylor_coefficients_mask =
              __ballot_sync(0xFFFFFFFF, need_taylor_coefficients);
          if (load_taylor_coefficients_mask > 0) {
            load_shared_cooperative<TaylorCoefficientsF16>(
                &current_taylor_coefficients_cache[warp_id],
                leaf_coefficients + leaf_idx, lane_id);
            __syncwarp();
          }
          bool is_detail_eval_needed = true;
          if (need_taylor_coefficients) {
            const TaylorCoefficientsF16 &current_leaf_coefficients =
                current_taylor_coefficients_cache[warp_id];
            const Vec3 leaf_center_of_mass = child_aabb.center_of_mass;
            if constexpr (Out == TraversalOutput::WindingNumbersField) {
              float approx_contribution = compute_node_approximation(
                  my_query, leaf_center_of_mass,
                  current_leaf_coefficients.zero_order,
                  current_leaf_coefficients.first_order,
                  current_leaf_coefficients.second_order);
              kahan_add(my_result, approx_contribution, my_comp);
            } else {
              Vec3 approx_gradient = compute_node_query_grad_approximation(
                  my_query, leaf_center_of_mass,
                  current_leaf_coefficients.zero_order,
                  current_leaf_coefficients.first_order,
                  current_leaf_coefficients.second_order);
              kahan_add(my_result, approx_gradient, my_comp);
            }
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
          // Use full warp to compute contributions of interested queries one
          // by one
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

            if constexpr (Out == TraversalOutput::WindingNumbersField) {
              // compute the winding number contribution from my geometry
              float my_contribution = 0.F;
              if (is_my_geometry_in_bounds) {
                my_contribution =
                    my_geometry.contributionToQuery(shared_query, ctx);
              }
              // sum up all contributions in warp
              float total_contribution = warp_reduce_add_xor(my_contribution);
              // only leader adds that contribution
              if ((int)lane_id == current_leader) {
                kahan_add(my_result, total_contribution, my_comp);
              }
            } else {
              Vec3 my_grad = Vec3::zero();
              if (is_my_geometry_in_bounds) {
                my_grad =
                    my_geometry.queryGradContribution(shared_query, 1.F, ctx);
              }
              Vec3 total_grad{.x = warp_reduce_add_xor(my_grad.x),
                              .y = warp_reduce_add_xor(my_grad.y),
                              .z = warp_reduce_add_xor(my_grad.z)};
              if ((int)lane_id == current_leader) {
                kahan_add(my_result, total_grad, my_comp);
              }
            }
          }
        } else {
          // Child is inner node
          uint32_t child_node_idx =
              current_node.child_base + added_inner_node_counter;
          added_inner_node_counter++;
          if (lane_id == 0) {
            shared_stack[warp_id][stack_ptr] = child_node_idx;
          }
          stack_ptr++; // increment stack pointer for all lanes
        }
      } // child for
    } // traversal while
    // winding number computation is complete
    if (my_required_stack_depth >= 0) {
      if constexpr (Out == TraversalOutput::WindingNumbersField) {
        out_winding_numbers_field[original_query_idx] = my_result;
      } else {
        const float g_j_scaled = grad_output[original_query_idx] * norm.scale;
        size_t out_idx = static_cast<size_t>(original_query_idx) * 3;
        out_query_gradients[out_idx] = my_result.x * g_j_scaled;
        out_query_gradients[out_idx + 1] = my_result.y * g_j_scaled;
        out_query_gradients[out_idx + 2] = my_result.z * g_j_scaled;
      }
    }
  } // grid while
}

// Single-leaf traversal. Used when geometry_count <= 32, i.e. there is no
// BVH8 tree and every query thread simply iterates over the whole geometry
// set. Shares the TraversalOutput template with the main traversal kernel
// so both output kinds live in the same code path.
template <IsGeometry Geometry, TraversalOutput Out>
__global__ void traverse_single_leaf_kernel(
    const Vec3 *__restrict__ queries,
    const uint32_t *__restrict__ sort_indirections,
    const SoAView<Geometry> sorted_geometry,
    const float *__restrict__ grad_output, // nullptr for WindingNumbersField
    const uint32_t query_count, const uint32_t geometry_count,
    float *__restrict__ out_winding_numbers_field, // nullptr for QueryGradient
    float *__restrict__ out_query_gradients, // nullptr for WindingNumbersField
    const typename Geometry::Context ctx, const SceneNormalization norm) {

  const uint32_t my_query_idx = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ Geometry shared_geometry[32];
  if (threadIdx.x < geometry_count) {
    shared_geometry[threadIdx.x] =
        Geometry::load(sorted_geometry, threadIdx.x, geometry_count);
  }
  __syncthreads();

  if (my_query_idx >= query_count)
    return;

  const uint32_t original_idx = sort_indirections[my_query_idx];
  const Vec3 my_query = norm.to_normalized(queries[original_idx]);

  if constexpr (Out == TraversalOutput::WindingNumbersField) {
    float my_winding_number = 0.0F;
    for (uint32_t i = 0; i < geometry_count; ++i) {
      my_winding_number +=
          shared_geometry[i].contributionToQuery(my_query, ctx);
    }
    out_winding_numbers_field[original_idx] = my_winding_number;
  } else {
    // Query-gradient accumulation uses Kahan since it is new code and the
    // extra accuracy is free.
    Vec3 my_grad = Vec3::zero();
    for (uint32_t i = 0; i < geometry_count; ++i) {
      const Vec3 contrib =
          shared_geometry[i].queryGradContribution(my_query, 1.F, ctx);
      my_grad += contrib;
    }
    // Adjoint is applied once at the end, matching the main traversal.
    const float g_j_scaled = grad_output[original_idx] * norm.scale;
    const size_t out_idx = static_cast<size_t>(original_idx) * 3;
    out_query_gradients[out_idx] = my_grad.x * g_j_scaled;
    out_query_gradients[out_idx + 1] = my_grad.y * g_j_scaled;
    out_query_gradients[out_idx + 2] = my_grad.z * g_j_scaled;
  }
}

template <IsGeometry Geometry>
void compute_winding_numbers(
    const ComputeWindingNumbersParams<Geometry> &params, int device_id,
    const cudaStream_t &stream) {
  if (params.query_count == 0) {
    return;
  }

  auto ctx = Geometry::Context::make(params.epsilon, params.scene_scale);
  // There i no tree if there is only one leaf
  if (params.geometry_count <= 32) {
    uint32_t threads = 256;
    uint32_t blocks = (params.query_count + threads - 1) / threads;
    traverse_single_leaf_kernel<Geometry, TraversalOutput::WindingNumbersField>
        <<<blocks, threads, 0, stream>>>(
            params.queries, params.sort_indirections, params.sorted_geometry,
            /*grad_output=*/nullptr, params.query_count, params.geometry_count,
            params.winding_numbers,
            /*out_query_gradients=*/nullptr, ctx, params.norm);
    CUDA_CHECK(cudaGetLastError());
    return;
  }

  float beta_2 = params.beta * params.beta;

  int threads = 128;
  int blocks_per_sm = 0;

  // Set global counter to 0
  uint32_t global_device_counter_reset = 0;
  CUDA_CHECK(cudaMemcpyAsync(params.global_device_counter,
                             &global_device_counter_reset, sizeof(uint32_t),
                             cudaMemcpyHostToDevice, stream));

  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm,
      traverse_geometry_tree_kernel<Geometry,
                                    TraversalOutput::WindingNumbersField>,
      threads, 0);

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device_id);
  int blocks = blocks_per_sm * deviceProp.multiProcessorCount;

  traverse_geometry_tree_kernel<Geometry, TraversalOutput::WindingNumbersField>
      <<<blocks, threads, 0, stream>>>(
          params.queries, params.sort_indirections, params.bvh8_nodes,
          params.bvh8_leaf_pointers, params.node_coefficients,
          params.leaf_coefficients, params.leaf_aabbs, params.sorted_geometry,
          /*grad_output=*/nullptr, params.query_count, params.geometry_count,
          params.winding_numbers, /*out_query_gradients=*/nullptr,
          params.global_device_counter, beta_2, ctx, params.norm);
  CUDA_CHECK(cudaGetLastError());
}

template <IsGeometry Geometry>
void query_gradient_traversal(
    const ComputeQueryGradientParams<Geometry> &params, int device_id,
    const cudaStream_t &stream) {
  if (params.query_count == 0) {
    return;
  }

  auto ctx = Geometry::Context::make(params.epsilon, params.scene_scale);
  // There i no tree if there is only one leaf
  if (params.geometry_count <= 32) {
    uint32_t threads = 256;
    uint32_t blocks = (params.query_count + threads - 1) / threads;
    traverse_single_leaf_kernel<Geometry,
                                TraversalOutput::QueryGradient> // TODO
        <<<blocks, threads, 0, stream>>>(
            params.queries, params.sort_indirections, params.sorted_geometry,
            params.grad_output, params.query_count, params.geometry_count,
            /*winding_numbers=*/nullptr, params.out_query_gradients, ctx,
            params.norm);
    CUDA_CHECK(cudaGetLastError());
    return;
  }

  float beta_2 = params.beta * params.beta;

  int threads = 128;
  int blocks_per_sm = 0;

  // Set global counter to 0
  uint32_t global_device_counter_reset = 0;
  CUDA_CHECK(cudaMemcpyAsync(params.global_device_counter,
                             &global_device_counter_reset, sizeof(uint32_t),
                             cudaMemcpyHostToDevice, stream));

  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm,
      traverse_geometry_tree_kernel<Geometry, TraversalOutput::QueryGradient>,
      threads, 0);

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device_id);
  int blocks = blocks_per_sm * deviceProp.multiProcessorCount;

  traverse_geometry_tree_kernel<Geometry, TraversalOutput::QueryGradient>
      <<<blocks, threads, 0, stream>>>(
          params.queries, params.sort_indirections, params.bvh8_nodes,
          params.bvh8_leaf_pointers, params.node_coefficients,
          params.leaf_coefficients, params.leaf_aabbs, params.sorted_geometry,
          params.grad_output, params.query_count, params.geometry_count,
          /*winding_numbers=*/nullptr, params.out_query_gradients,
          params.global_device_counter, beta_2, ctx, params.norm);
  CUDA_CHECK(cudaGetLastError());
}

__global__ void compute_point_normal_gradient_single_leaf_kernel(
    const Vec3 *__restrict__ points, const Vec3 *__restrict__ normals,
    const uint32_t *__restrict__ sort_indirections,
    const SoAView<Vec3> sorted_queries,
    const float *__restrict__ sorted_grad_outputs, const uint32_t query_count,
    const uint32_t geometry_count, float *__restrict__ gradients,
    float *__restrict__ epsilon_gradients, const PointNormalContext ctx,
    const SceneNormalization norm) {
  // Global index of the geometry this thread is responsible for
  uint32_t my_geometry_idx = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ Vec3 shared_query[32];
  __shared__ float shared_grad_output[32];
  if (threadIdx.x < query_count) {
    shared_query[threadIdx.x] =
        Vec3::load(sorted_queries, threadIdx.x, query_count);
    shared_grad_output[threadIdx.x] = sorted_grad_outputs[threadIdx.x];
  }
  __syncthreads();

  if (my_geometry_idx >= geometry_count) {
    return;
  }

  // Load geometry
  uint32_t original_idx = sort_indirections[my_geometry_idx];
  const auto my_geometry = norm.to_normalized(
      PointNormal{.p = points[original_idx], .n = normals[original_idx]});
  PointNormal my_grad{.p = Vec3::zero(), .n = Vec3::zero()};
  float my_epsilon = 0.F;

  // Since there is only one leaf, all queries (1-32 elements)
  // are stored at the beginning of sorted_queries.
  // Every thread iterates through all available queries for its
  // PointNormal.
  for (uint32_t i = 0; i < query_count; ++i) {
    GeometryGradient<PointNormal> contrib = my_geometry.gradContributionOfQuery(
        shared_query[i], shared_grad_output[i], ctx);
    my_grad += contrib.geometry;
    my_epsilon += contrib.epsilon;
  }

  // Write out results
  gradients[original_idx * 6] = my_grad.n.x * norm.scale2;
  gradients[original_idx * 6 + 1] = my_grad.n.y * norm.scale2;
  gradients[original_idx * 6 + 2] = my_grad.n.z * norm.scale2;
  gradients[original_idx * 6 + 3] = my_grad.p.x * norm.scale;
  gradients[original_idx * 6 + 4] = my_grad.p.y * norm.scale;
  gradients[original_idx * 6 + 5] = my_grad.p.z * norm.scale;
  epsilon_gradients[original_idx] = my_epsilon;
}

// Parameter struct to save registers
struct PointNormalGradientKernelParams {
  const Vec3 *points;
  const Vec3 *normals;
  const uint32_t *sort_indirections;
  const BVH8Node *bvh8_nodes;
  const LeafPointers *bvh8_leaf_pointers;
  const BackwardTaylorCoefficients *node_coefficients;
  const AABB *leaf_aabbs;
  const float *sorted_grad_outputs;
  float *gradients;
  float *epsilon_gradients;
  uint32_t *global_device_counter;
  SoAView<Vec3> sorted_queries;
  uint32_t query_count;
  uint32_t geometry_count;
  float beta;
  PointNormalContext reg_context;
  SceneNormalization norm;
};

__global__ void __launch_bounds__(128) compute_point_normal_gradient_kernel(
    __grid_constant__ const PointNormalGradientKernelParams params) {
  const uint32_t warp_id = threadIdx.x / 32;
  const uint32_t lane_id = threadIdx.x % 32;

  // We split the geometry into tiles of 32 threads.
  // Each warp of 32 threads works on one tile at a time.
  // 4 warps per block
  // Each warp has its own shared traversal stack.
  __shared__ uint32_t shared_stack[4][64];
  __shared__ BVH8Node current_node_cache[4];
  __shared__ BackwardTaylorCoefficients current_taylor_coefficients_cache[4];
  __shared__ LeafPointers shared_leaf_ptrs[4];
  uint32_t warp_tile_base;

  while (true) {
    // Claim a tile
    // Dynamic work balancing. Not all blocks will need the same amount of time
    // for their queries
    if (lane_id == 0) {
      warp_tile_base = atomicAdd(params.global_device_counter, 32);
    }
    warp_tile_base = __shfl_sync(0xFFFFFFFF, warp_tile_base, 0);

    if (warp_tile_base >= params.geometry_count) {
      // No more work for this warp
      break;
    }

    // keep track of what subtrees have been approximated for this geometry item
    int my_required_stack_depth = std::numeric_limits<int>::max();

    uint32_t my_geometry_idx = warp_tile_base + lane_id;

    PointNormal my_geometry{Vec3::zero(), Vec3::zero()};
    uint32_t original_geometry_idx = 0xFFFFFFFF;
    if (my_geometry_idx >= params.geometry_count) {
      // This thread has no geometry, but still needs to help the others in the
      // warp with their computations
      my_required_stack_depth = -1;
    } else {
      // load geometry with sort indirections
      original_geometry_idx = params.sort_indirections[my_geometry_idx];
      my_geometry = params.norm.to_normalized(
          PointNormal{.p = params.points[original_geometry_idx],
                      .n = params.normals[original_geometry_idx]});
    }
    PointNormal my_gradient{.p = Vec3::zero(), .n = Vec3::zero()};
    float my_epsilon = 0.F;
    PointNormal geo_compensation{.p = Vec3::zero(),
                                 .n = Vec3::zero()}; // for Kahn summation
    float eps_compensation = 0.F;                    // for Kahan summation

    // Always start traversal on root
    int stack_ptr = 0;
    if (lane_id == 0) {
      shared_stack[warp_id][stack_ptr] = 0;
    }
    stack_ptr++; // increment stack pointer for all lanes

    // Do traversal as full warps
    while (true) {
      // Check if stack is empty
      uint32_t stack_not_empty_mask = __ballot_sync(0xFFFFFFFF, stack_ptr > 0);
      // stop if all warps are done
      if (stack_not_empty_mask == 0) {
        break; // this tile is done
      }
      // warp leader pops next node into cache
      uint32_t current_node_idx;
      --stack_ptr; // decrement stack pointer for all lanes
      if (lane_id == 0) {
        current_node_idx = shared_stack[warp_id][stack_ptr];
      }
      // share current_node_idx with the rest
      current_node_idx = __shfl_sync(0xFFFFFFFF, current_node_idx, 0);
      // load current node to shared cache
      load_shared_cooperative<BVH8Node>(&current_node_cache[warp_id],
                                        params.bvh8_nodes + current_node_idx,
                                        lane_id);
      __syncwarp();

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
      // Check the nodes parent_aabb. If it is too far away approximate using
      // taylor coefficients
      bool need_taylor_coefficients =
          is_active &&
          should_node_be_approximated(my_geometry, current_node.getAABB(),
                                      params.beta, params.reg_context);

      uint32_t load_taylor_coefficients_mask =
          __ballot_sync(0xFFFFFFFF, need_taylor_coefficients);
      if (load_taylor_coefficients_mask > 0) {
        load_shared_cooperative<BackwardTaylorCoefficients>(
            &current_taylor_coefficients_cache[warp_id],
            params.node_coefficients + current_node_idx, lane_id);
        __syncwarp();
      }
      if (need_taylor_coefficients) {
        BackwardTaylorCoefficients &current_node_coefficients =
            current_taylor_coefficients_cache[warp_id];
        AABB parent_aabb = current_node.getAABB();
        // Do approximation
        PointNormal approx_grad_contribution =
            compute_node_gradient_approximation(
                my_geometry, parent_aabb.center_of_mass,
                current_node_coefficients.zero_order,
                current_node_coefficients.first_order,
                current_node_coefficients.second_order);
        kahan_add(my_gradient, approx_grad_contribution, geo_compensation);

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
      uint32_t leaf_mask = __ballot_sync(0xFFFFFFFF, is_leaf);
      if (leaf_mask > 0) {
        load_shared_cooperative(&shared_leaf_ptrs[warp_id],
                                &params.bvh8_leaf_pointers[current_node_idx],
                                lane_id);
        __syncwarp();
      }

      // Go through all childs together
      uint32_t added_inner_node_counter = 0;
      for (uint32_t child_idx = 0; child_idx < 8; ++child_idx) {
        ChildType child_type = current_node.getChildMeta(child_idx);
        if (child_type == ChildType::EMPTY) {
          continue;
        }
        if (child_type == ChildType::LEAF) {
          uint32_t leaf_idx = shared_leaf_ptrs[warp_id].indices[child_idx];

          if (is_still_active) {
            for (uint32_t query_counter = 0; query_counter < LEAF_SIZE;
                 ++query_counter) {
              uint32_t query_idx = query_counter + leaf_idx * LEAF_SIZE;
              if (query_idx >= params.query_count) {
                break;
              }
              Vec3 query = Vec3::load(params.sorted_queries, query_idx,
                                      params.query_count);
              float grad_output = params.sorted_grad_outputs[query_idx];
              GeometryGradient<PointNormal> grad_contribution =
                  my_geometry.gradContributionOfQuery(query, grad_output,
                                                      params.reg_context);

              kahan_add(my_gradient, grad_contribution.geometry,
                        geo_compensation);
              kahan_add(my_epsilon, grad_contribution.epsilon,
                        eps_compensation);
            }
          }
        }

        else {
          // Child is inner node
          uint32_t child_node_idx =
              current_node.child_base + added_inner_node_counter;
          added_inner_node_counter++;
          if (lane_id == 0) {
            shared_stack[warp_id][stack_ptr] = child_node_idx;
          }
          stack_ptr++; // increment stack pointer for all lanes
        }
      } // child for
    } // traversal while
    // winding number computation is complete
    if (my_required_stack_depth >= 0) {
      params.gradients[original_geometry_idx * 6] =
          my_gradient.n.x * params.norm.scale2;
      params.gradients[original_geometry_idx * 6 + 1] =
          my_gradient.n.y * params.norm.scale2;
      params.gradients[original_geometry_idx * 6 + 2] =
          my_gradient.n.z * params.norm.scale2;
      params.gradients[original_geometry_idx * 6 + 3] =
          my_gradient.p.x * params.norm.scale;
      params.gradients[original_geometry_idx * 6 + 4] =
          my_gradient.p.y * params.norm.scale;
      params.gradients[original_geometry_idx * 6 + 5] =
          my_gradient.p.z * params.norm.scale;
      params.epsilon_gradients[original_geometry_idx] = my_epsilon;
    }
  } // grid while
}

void compute_point_normal_gradients(
    const ComputeGradientsPointNormalParams &params, int device_id,
    const cudaStream_t &stream) {
  if (params.geometry_count == 0) {
    return;
  }

  const auto ctx = PointNormalContext::make(params.epsilon, params.scene_scale);
  // There is no tree if there is only one leaf
  if (params.query_count <= 32) {
    uint32_t threads = 256;
    uint32_t blocks = (params.geometry_count + threads - 1) / threads;
    compute_point_normal_gradient_single_leaf_kernel<<<blocks, threads, 0,
                                                       stream>>>(
        params.points, params.normals, params.sort_indirections,
        params.sorted_queries, params.sorted_grad_outputs, params.query_count,
        params.geometry_count, params.gradients, params.epsilon_gradients, ctx,
        params.norm);
    CUDA_CHECK(cudaGetLastError());
    return;
  }

  int threads = 128;
  int blocks_per_sm = 0;

  // Set global counter to 0
  uint32_t global_device_counter_reset = 0;
  CUDA_CHECK(cudaMemcpyAsync(params.global_device_counter,
                             &global_device_counter_reset, sizeof(uint32_t),
                             cudaMemcpyHostToDevice, stream));

  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, compute_point_normal_gradient_kernel, threads, 0);

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device_id);
  int blocks = blocks_per_sm * deviceProp.multiProcessorCount;

  PointNormalGradientKernelParams kernel_params{
      .points = params.points,
      .normals = params.normals,
      .sort_indirections = params.sort_indirections,
      .bvh8_nodes = params.bvh8_nodes,
      .bvh8_leaf_pointers = params.bvh8_leaf_pointers,
      .node_coefficients = params.node_coefficients,
      .leaf_aabbs = params.leaf_aabbs,
      .sorted_grad_outputs = params.sorted_grad_outputs,
      .gradients = params.gradients,
      .epsilon_gradients = params.epsilon_gradients,
      .global_device_counter = params.global_device_counter,
      .sorted_queries = params.sorted_queries,
      .query_count = params.query_count,
      .geometry_count = params.geometry_count,
      .beta = params.beta,
      .reg_context = ctx,
      .norm = params.norm};
  compute_point_normal_gradient_kernel<<<blocks, threads, 0, stream>>>(
      kernel_params);
  CUDA_CHECK(cudaGetLastError());
}

__global__ void compute_triangle_gradient_single_leaf_kernel(
    const Triangle *__restrict__ triangles,
    const uint32_t *__restrict__ sort_indirections,
    const SoAView<Vec3> sorted_queries,
    const float *__restrict__ sorted_grad_outputs, const uint32_t query_count,
    const uint32_t geometry_count, float *__restrict__ gradients,
    float *__restrict__ epsilon_gradients, const typename Triangle::Context ctx,
    const SceneNormalization norm) {
  // Global index of the geometry this thread is responsible for
  uint32_t my_geometry_idx = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ Vec3 shared_query[32];
  __shared__ float shared_grad_output[32];
  if (threadIdx.x < query_count) {
    shared_query[threadIdx.x] =
        Vec3::load(sorted_queries, threadIdx.x, query_count);
    shared_grad_output[threadIdx.x] = sorted_grad_outputs[threadIdx.x];
  }
  __syncthreads();

  if (my_geometry_idx >= geometry_count) {
    return;
  }

  // Load geometry
  uint32_t original_idx = sort_indirections[my_geometry_idx];
  const Triangle my_geometry = norm.to_normalized(triangles[original_idx]);

  // Accumulators
  Triangle my_grad{.v0 = Vec3::zero(), .v1 = Vec3::zero(), .v2 = Vec3::zero()};
  float my_epsilon = 0.F;

  // Since there is only one leaf, all queries (1-32 elements)
  // are stored at the beginning of sorted_queries.
  // Every thread iterates through all available queries for its
  // PointNormal.
  for (uint32_t i = 0; i < query_count; ++i) {
    GeometryGradient<Triangle> contrib = my_geometry.gradContributionOfQuery(
        shared_query[i], shared_grad_output[i], ctx);
    my_grad += contrib.geometry;
    my_epsilon += contrib.epsilon;
  }

  // Write out results
  gradients[original_idx * 9] = my_grad.v0.x * norm.scale;
  gradients[original_idx * 9 + 1] = my_grad.v0.y * norm.scale;
  gradients[original_idx * 9 + 2] = my_grad.v0.z * norm.scale;
  gradients[original_idx * 9 + 3] = my_grad.v1.x * norm.scale;
  gradients[original_idx * 9 + 4] = my_grad.v1.y * norm.scale;
  gradients[original_idx * 9 + 5] = my_grad.v1.z * norm.scale;
  gradients[original_idx * 9 + 6] = my_grad.v2.x * norm.scale;
  gradients[original_idx * 9 + 7] = my_grad.v2.y * norm.scale;
  gradients[original_idx * 9 + 8] = my_grad.v2.z * norm.scale;
  epsilon_gradients[original_idx] = my_epsilon;
}

// Parameter struct to save registers
struct TriangleGradientKernelParams {
  const Triangle *triangles;
  const uint32_t *sort_indirections;
  const BVH8Node *bvh8_nodes;
  const LeafPointers *bvh8_leaf_pointers;
  const BackwardTaylorCoefficients *node_coefficients;
  const AABB *leaf_aabbs;
  const float *sorted_grad_outputs;
  float *gradients;
  float *epsilon_gradients;
  uint32_t *global_device_counter;
  SoAView<Vec3> sorted_queries;
  uint32_t query_count;
  uint32_t geometry_count;
  float beta;
  typename Triangle::Context reg_context;
  SceneNormalization norm;
};

// Kernel signature using __grid_constant__
__global__ void __launch_bounds__(128) compute_triangle_gradient_kernel(
    __grid_constant__ const TriangleGradientKernelParams params) {
  const uint32_t warp_id = threadIdx.x / 32;
  const uint32_t lane_id = threadIdx.x % 32;

  // We split the geometry into tiles of 32 threads.
  // Each warp of 32 threads works on one tile at a time.
  // 4 warps per block
  // Each warp has its own shared traversal stack.
  __shared__ uint32_t shared_stack[4][64];
  __shared__ BVH8Node current_node_cache[4];
  __shared__ BackwardTaylorCoefficients current_taylor_coefficients_cache[4];
  __shared__ LeafPointers shared_leaf_ptrs[4];
  __shared__ Triangle shared_warp_geometry[4][32]; // Here not as registers
  uint32_t warp_tile_base;

  while (true) {
    // Claim a tile
    // Dynamic work balancing. Not all blocks will need the same amount of time
    // for their queries
    if (lane_id == 0) {
      warp_tile_base = atomicAdd(params.global_device_counter, 32);
    }
    warp_tile_base = __shfl_sync(0xFFFFFFFF, warp_tile_base, 0);

    if (warp_tile_base >= params.geometry_count) {
      // No more work for this warp
      break;
    }

    // keep track of what subtrees have been approximated for this geometry item
    int my_required_stack_depth = std::numeric_limits<int>::max();

    uint32_t my_geometry_idx = warp_tile_base + lane_id;

    if (my_geometry_idx >= params.geometry_count) {
      // This thread has no geometry, but still needs to help the others in the
      // warp with their computations
      my_required_stack_depth = -1;
      shared_warp_geometry[warp_id][lane_id] =
          Triangle{.v0 = Vec3::zero(), .v1 = Vec3::zero(), .v2 = Vec3::zero()};
    } else {
      // load geometry with sort indirections
      uint32_t original_geometry_idx =
          params.sort_indirections[my_geometry_idx];
      shared_warp_geometry[warp_id][lane_id] =
          params.norm.to_normalized(params.triangles[original_geometry_idx]);
    }
    Triangle my_gradient{
        .v0 = Vec3::zero(), .v1 = Vec3::zero(), .v2 = Vec3::zero()};
    Triangle geo_compensation{
        .v0 = Vec3::zero(), .v1 = Vec3::zero(), .v2 = Vec3::zero()};
    float my_epsilon = 0.F;
    float eps_compensation = 0.F;

    __syncwarp();

    // Always start traversal on root
    int stack_ptr = 0;
    if (lane_id == 0) {
      shared_stack[warp_id][stack_ptr] = 0;
    }
    stack_ptr++; // increment stack pointer for all lanes

    // Do traversal as full warps
    while (true) {
      // Check if stack is empty
      uint32_t stack_not_empty_mask = __ballot_sync(0xFFFFFFFF, stack_ptr > 0);
      // stop if all warps are done
      if (stack_not_empty_mask == 0) {
        break; // this tile is done
      }
      // warp leader pops next node into cache
      uint32_t current_node_idx;
      --stack_ptr; // decrement stack pointer for all lanes
      if (lane_id == 0) {
        current_node_idx = shared_stack[warp_id][stack_ptr];
      }
      // share current_node_idx with the rest
      current_node_idx = __shfl_sync(0xFFFFFFFF, current_node_idx, 0);
      // load current node to shared cache
      load_shared_cooperative<BVH8Node>(&current_node_cache[warp_id],
                                        params.bvh8_nodes + current_node_idx,
                                        lane_id);
      __syncwarp();

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
      // Check the nodes parent_aabb. If it is too far away approximate using
      // taylor coefficients
      bool need_taylor_coefficients =
          is_active &&
          should_node_be_approximated(shared_warp_geometry[warp_id][lane_id],
                                      current_node.getAABB(), params.beta,
                                      params.reg_context);

      uint32_t load_taylor_coefficients_mask =
          __ballot_sync(0xFFFFFFFF, need_taylor_coefficients);
      if (load_taylor_coefficients_mask > 0) {
        load_shared_cooperative<BackwardTaylorCoefficients>(
            &current_taylor_coefficients_cache[warp_id],
            params.node_coefficients + current_node_idx, lane_id);
        __syncwarp();
      }
      if (need_taylor_coefficients) {
        BackwardTaylorCoefficients &current_node_coefficients =
            current_taylor_coefficients_cache[warp_id];
        AABB parent_aabb = current_node.getAABB();
        // Do approximation
        Triangle approx_grad_contribution = compute_node_gradient_approximation(
            shared_warp_geometry[warp_id][lane_id], parent_aabb.center_of_mass,
            current_node_coefficients.zero_order,
            current_node_coefficients.first_order,
            current_node_coefficients.second_order);
        kahan_add(my_gradient, approx_grad_contribution, geo_compensation);

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
      uint32_t leaf_mask = __ballot_sync(0xFFFFFFFF, is_leaf);
      if (leaf_mask > 0) {
        load_shared_cooperative(&shared_leaf_ptrs[warp_id],
                                &params.bvh8_leaf_pointers[current_node_idx],
                                lane_id);
        __syncwarp();
      }

      // Go through all childs together
      uint32_t added_inner_node_counter = 0;
      for (uint32_t child_idx = 0; child_idx < 8; ++child_idx) {
        ChildType child_type = current_node.getChildMeta(child_idx);
        if (child_type == ChildType::EMPTY) {
          continue;
        }
        if (child_type == ChildType::LEAF) {
          uint32_t leaf_idx = shared_leaf_ptrs[warp_id].indices[child_idx];

          if (is_still_active) {
            for (uint32_t query_counter = 0; query_counter < LEAF_SIZE;
                 ++query_counter) {
              uint32_t query_idx = query_counter + leaf_idx * LEAF_SIZE;
              if (query_idx >= params.query_count) {
                break;
              }
              Vec3 query = Vec3::load(params.sorted_queries, query_idx,
                                      params.query_count);
              float grad_output = params.sorted_grad_outputs[query_idx];
              const GeometryGradient<Triangle> grad_contribution =
                  shared_warp_geometry[warp_id][lane_id]
                      .gradContributionOfQuery(query, grad_output,
                                               params.reg_context);
              kahan_add(my_gradient, grad_contribution.geometry,
                        geo_compensation);
              kahan_add(my_epsilon, grad_contribution.epsilon,
                        eps_compensation);
            }
          }
        } else {
          // Child is inner node
          uint32_t child_node_idx =
              current_node.child_base + added_inner_node_counter;
          added_inner_node_counter++;
          if (lane_id == 0) {
            shared_stack[warp_id][stack_ptr] = child_node_idx;
          }
          stack_ptr++; // increment stack pointer for all lanes
        }
      } // child for
    } // traversal while
    // winding number computation is complete
    if (my_required_stack_depth >= 0) {
      uint32_t original_geometry_idx =
          params.sort_indirections[my_geometry_idx];
      float *out_ptr = params.gradients + original_geometry_idx * 9;
      const auto *grad_ptr = reinterpret_cast<const float *>(&my_gradient);
#pragma unroll
      for (int i = 0; i < 9; ++i) {
        out_ptr[i] = grad_ptr[i] * params.norm.scale;
      }
      params.epsilon_gradients[original_geometry_idx] = my_epsilon;
    }
  } // grid while
}

void compute_triangle_gradients(const ComputeGradientsTriangleParams &params,
                                int device_id, const cudaStream_t &stream) {
  if (params.geometry_count == 0) {
    return;
  }

  const auto ctx = Triangle::Context::make(params.epsilon, params.scene_scale);

  // There is no tree if there is only one leaf
  if (params.query_count <= 32) {
    uint32_t threads = 256;
    uint32_t blocks = (params.geometry_count + threads - 1) / threads;
    compute_triangle_gradient_single_leaf_kernel<<<blocks, threads, 0,
                                                   stream>>>(
        params.triangles, params.sort_indirections, params.sorted_queries,
        params.sorted_grad_outputs, params.query_count, params.geometry_count,
        params.gradients, params.epsilon_gradients, ctx, params.norm);
    return;
  }

  int threads = 128;
  int blocks_per_sm = 0;

  // Set global counter to 0
  uint32_t global_device_counter_reset = 0;
  CUDA_CHECK(cudaMemcpyAsync(params.global_device_counter,
                             &global_device_counter_reset, sizeof(uint32_t),
                             cudaMemcpyHostToDevice, stream));

  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, compute_triangle_gradient_kernel, threads, 0);

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device_id);
  int blocks = blocks_per_sm * deviceProp.multiProcessorCount;

  TriangleGradientKernelParams kernel_params{
      .triangles = params.triangles,
      .sort_indirections = params.sort_indirections,
      .bvh8_nodes = params.bvh8_nodes,
      .bvh8_leaf_pointers = params.bvh8_leaf_pointers,
      .node_coefficients = params.node_coefficients,
      .leaf_aabbs = params.leaf_aabbs,
      .sorted_grad_outputs = params.sorted_grad_outputs,
      .gradients = params.gradients,
      .epsilon_gradients = params.epsilon_gradients,
      .global_device_counter = params.global_device_counter,
      .sorted_queries = params.sorted_queries,
      .query_count = params.query_count,
      .geometry_count = params.geometry_count,
      .beta = params.beta,
      .reg_context = ctx,
      .norm = params.norm};
  compute_triangle_gradient_kernel<<<blocks, threads, 0, stream>>>(
      kernel_params);
  CUDA_CHECK(cudaGetLastError());
}

template void compute_winding_numbers<PointNormal>(
    const ComputeWindingNumbersParams<PointNormal> &params, int device_id,
    const cudaStream_t &stream);
template void compute_winding_numbers<Triangle>(
    const ComputeWindingNumbersParams<Triangle> &params, int device_id,
    const cudaStream_t &stream);
template void query_gradient_traversal<PointNormal>(
    const ComputeQueryGradientParams<PointNormal> &params, int device_id,
    const cudaStream_t &stream);
template void query_gradient_traversal<Triangle>(
    const ComputeQueryGradientParams<Triangle> &params, int device_id,
    const cudaStream_t &stream);

template <uint32_t D>
constexpr uint32_t components_per_thread = (D < 8U) ? D : 8U;

template <uint32_t D>
constexpr uint32_t threads_per_query =
    (D + components_per_thread<D> - 1U) / components_per_thread<D>;

template <uint32_t D>
constexpr uint32_t queries_per_warp = 32U / threads_per_query<D>;

static_assert(threads_per_query<128> <= 32,
              "D too large for warp-cooperative traversal");

//
// Forward: payload winding field
template <IsGeometry Geometry, uint32_t D, bool UseKahan>
__global__ void __launch_bounds__(128) traverse_payload_winding_numbers_kernel(
    const Vec3 *__restrict__ queries,
    const uint32_t *__restrict__ sort_indirections,
    const BVH8Node *__restrict__ bvh8_nodes,
    const LeafPointers *__restrict__ bvh8_leaf_pointers,
    const PayloadTaylorCoefficientsF16<D> *__restrict__ node_coefficients,
    const PayloadTaylorCoefficientsF16<D> *__restrict__ leaf_coefficients,
    const AABB *__restrict__ leaf_aabbs,
    const SoAView<Geometry> sorted_geometry,
    const float *__restrict__ sorted_payload, const uint32_t payload_dim_user,
    const uint32_t query_count, const uint32_t geometry_count,
    float *__restrict__ out_payload,
    uint32_t *__restrict__ global_device_counter, const float beta_2,
    const typename Geometry::Context ctx, const SceneNormalization norm) {

  constexpr uint32_t COMPONENTS_PER_THREAD = components_per_thread<D>;
  constexpr uint32_t THREADS_PER_QUERY = threads_per_query<D>;
  constexpr uint32_t QUERIES_PER_WARP = queries_per_warp<D>;

  const uint32_t warp_id = threadIdx.x / 32;
  const uint32_t lane_id = threadIdx.x % 32;
  const uint32_t my_query_slot = lane_id / THREADS_PER_QUERY;
  const uint32_t my_dblock = lane_id % THREADS_PER_QUERY;

  const uint32_t my_d0 = my_dblock * COMPONENTS_PER_THREAD;
  const uint32_t my_d_count =
      (my_d0 >= payload_dim_user)
          ? 0U
          : min(COMPONENTS_PER_THREAD, payload_dim_user - my_d0);

  __shared__ uint32_t shared_stack[4][64];
  __shared__ BVH8Node current_node_cache[4];
  __shared__ PayloadTaylorCoefficientsF16<D> current_coeff_cache[4];
  __shared__ LeafPointers shared_leaf_ptrs[4];
  __shared__ float w_leaf_values[4][32];

  uint32_t warp_tile_base;

  while (true) {
    if (lane_id == 0) {
      warp_tile_base = atomicAdd(global_device_counter, QUERIES_PER_WARP);
    }
    warp_tile_base = __shfl_sync(0xFFFFFFFF, warp_tile_base, 0);
    if (warp_tile_base >= query_count)
      break;

    int my_required_stack_depth = std::numeric_limits<int>::max();
    const uint32_t my_query_idx = warp_tile_base + my_query_slot;
    Vec3 my_query{.x = 0.F, .y = 0.F, .z = 0.F};
    uint32_t original_query_idx = 0xFFFFFFFF;
    if (my_query_idx >= query_count) {
      my_required_stack_depth = -1;
    } else {
      original_query_idx = sort_indirections[my_query_idx];
      my_query = norm.to_normalized(queries[original_query_idx]);
    }

    float acc[COMPONENTS_PER_THREAD];
    float acc_comp[COMPONENTS_PER_THREAD];
#pragma unroll
    for (uint32_t c = 0; c < COMPONENTS_PER_THREAD; ++c) {
      acc[c] = 0.F;
      acc_comp[c] = 0.F;
    }

    int stack_ptr = 0;
    if (lane_id == 0)
      shared_stack[warp_id][stack_ptr] = 0;
    stack_ptr++;
    const float two_eps = 2.F * ctx.eps_length;

    while (true) {
      uint32_t stack_not_empty_mask = __ballot_sync(0xFFFFFFFF, stack_ptr > 0);
      if (stack_not_empty_mask == 0)
        break;

      uint32_t current_node_idx;
      --stack_ptr;
      if (lane_id == 0)
        current_node_idx = shared_stack[warp_id][stack_ptr];
      current_node_idx = __shfl_sync(0xFFFFFFFF, current_node_idx, 0);
      load_shared_cooperative<BVH8Node>(&current_node_cache[warp_id],
                                        bvh8_nodes + current_node_idx, lane_id);
      __syncwarp();

      const BVH8Node &current_node = current_node_cache[warp_id];

      bool is_active = stack_ptr < my_required_stack_depth;
      if (is_active)
        my_required_stack_depth = std::numeric_limits<int>::max();

      bool need_coeffs =
          is_active && should_node_be_approximated(
                           my_query, current_node.getAABB(), beta_2, two_eps);

      uint32_t load_mask = __ballot_sync(0xFFFFFFFF, need_coeffs);
      if (load_mask > 0) {
        load_shared_cooperative<PayloadTaylorCoefficientsF16<D>>(
            &current_coeff_cache[warp_id], node_coefficients + current_node_idx,
            lane_id);
        __syncwarp();
      }

      if (need_coeffs) {
        const AABB node_aabb = current_node.getAABB();
        const Vec3 com = node_aabb.center_of_mass;
        const Vec3 r = com - my_query;
        const float inv_R = r.inv_length();
        const Vec3_f16 r_hat = Vec3_f16::from_float(r * inv_R);
        const float inv_R2 = inv_R * inv_R;
        const float inv_R3 = inv_R2 * inv_R;
        const float inv_R4 = inv_R3 * inv_R;
        constexpr float INV_4PI = 0.07957747154F;
        const float f0 = INV_4PI * inv_R2;
        const float f1 = INV_4PI * inv_R3;
        const float f2 = INV_4PI * inv_R4;

#pragma unroll
        for (uint32_t c = 0; c < COMPONENTS_PER_THREAD; ++c) {
          if (c >= my_d_count)
            break;
          const uint32_t d = my_d0 + c;
          const auto &coeffs = current_coeff_cache[warp_id];
          float v =
              computeZeroOrderContribution(coeffs.getZeroOrder(d), r_hat) * f0;
          v += computeFirstOrderContribution(coeffs.getFirstOrder(d), r_hat) *
               f1;
          v += computeSecondOrderContribution(coeffs.getSecondOrder(d), r_hat) *
               f2;
          payload_accum_scalar<UseKahan>(acc[c], acc_comp[c], v);
        }
        my_required_stack_depth = stack_ptr;
      }

      bool is_still_active = my_required_stack_depth > stack_ptr;
      uint32_t interest_mask = __ballot_sync(0xFFFFFFFF, is_still_active);
      if (interest_mask == 0)
        continue;

      bool is_leaf = lane_id < 8
                         ? current_node.getChildMeta(lane_id) == ChildType::LEAF
                         : false;
      uint32_t leaf_mask = __ballot_sync(0xFFFFFFFF, is_leaf);
      if (leaf_mask > 0) {
        load_shared_cooperative(&shared_leaf_ptrs[warp_id],
                                &bvh8_leaf_pointers[current_node_idx], lane_id);
        __syncwarp();
      }

      uint32_t added_inner_node_counter = 0;
      for (uint32_t child_idx = 0; child_idx < 8; ++child_idx) {
        ChildType child_type = current_node.getChildMeta(child_idx);
        if (child_type == ChildType::EMPTY)
          continue;

        if (child_type == ChildType::LEAF) {
          const uint32_t leaf_idx =
              shared_leaf_ptrs[warp_id].indices[child_idx];
          const AABB child_aabb = leaf_aabbs[leaf_idx];

          bool need_leaf_coeffs =
              is_still_active && should_node_be_approximated(
                                     my_query, child_aabb, beta_2, two_eps);
          uint32_t load_mask_l = __ballot_sync(0xFFFFFFFF, need_leaf_coeffs);
          if (load_mask_l > 0) {
            load_shared_cooperative<PayloadTaylorCoefficientsF16<D>>(
                &current_coeff_cache[warp_id], leaf_coefficients + leaf_idx,
                lane_id);
            __syncwarp();
          }

          bool detail_needed = true;
          if (need_leaf_coeffs) {
            const Vec3 com = child_aabb.center_of_mass;
            const Vec3 r = com - my_query;
            const float inv_R = r.inv_length();
            const Vec3_f16 r_hat = Vec3_f16::from_float(r * inv_R);
            const float inv_R2 = inv_R * inv_R;
            const float inv_R3 = inv_R2 * inv_R;
            const float inv_R4 = inv_R3 * inv_R;
            constexpr float INV_4PI = 0.07957747154F;
            const float f0 = INV_4PI * inv_R2;
            const float f1 = INV_4PI * inv_R3;
            const float f2 = INV_4PI * inv_R4;
#pragma unroll
            for (uint32_t c = 0; c < COMPONENTS_PER_THREAD; ++c) {
              if (c >= my_d_count) {
                break;
              }
              const uint32_t d = my_d0 + c;
              const auto &coeffs = current_coeff_cache[warp_id];
              float v =
                  computeZeroOrderContribution(coeffs.getZeroOrder(d), r_hat) *
                  f0;
              v += computeFirstOrderContribution(coeffs.getFirstOrder(d),
                                                 r_hat) *
                   f1;
              v += computeSecondOrderContribution(coeffs.getSecondOrder(d),
                                                  r_hat) *
                   f2;
              payload_accum_scalar<UseKahan>(acc[c], acc_comp[c], v);
            }
            detail_needed = false;
          }

          uint32_t detail_mask =
              __ballot_sync(0xFFFFFFFF, detail_needed && is_still_active);
          if (detail_mask == 0) {
            continue;
          }

          const uint32_t my_geom_idx = leaf_idx * 32 + lane_id;
          const bool my_geom_in_bounds = my_geom_idx < geometry_count;
          Geometry my_geometry =
              Geometry::load(sorted_geometry, my_geom_idx, geometry_count);

          while (detail_mask > 0) {
            const int leader_lane = __ffs(detail_mask) - 1;
            const uint32_t leader_slot = leader_lane / THREADS_PER_QUERY;
            const uint32_t group_clear =
                ((THREADS_PER_QUERY == 32U)
                     ? 0xFFFFFFFFU
                     : (((1U << THREADS_PER_QUERY) - 1U)
                        << (leader_slot * THREADS_PER_QUERY)));
            detail_mask &= ~group_clear;

            const Vec3 shared_query{
                .x = __shfl_sync(0xFFFFFFFF, my_query.x, leader_lane),
                .y = __shfl_sync(0xFFFFFFFF, my_query.y, leader_lane),
                .z = __shfl_sync(0xFFFFFFFF, my_query.z, leader_lane),
            };

            float my_w = 0.F;
            if (my_geom_in_bounds) {
              my_w = my_geometry.contributionToQuery(shared_query, ctx);
            }
            w_leaf_values[warp_id][lane_id] = my_w;
            __syncwarp();

            // CHANGED — v_j reads bucket-stride zero-padded sorted_payload,
            //           inner loop bounded by my_d_count
            if (my_query_slot == leader_slot) {
              const float *my_v_base = sorted_payload + my_d0;
              for (uint32_t j = 0; j < LEAF_SIZE; ++j) {
                if (leaf_idx * LEAF_SIZE + j >= geometry_count) {
                  break;
                }
                const float w_j = w_leaf_values[warp_id][j];
                const float *v_j =
                    my_v_base +
                    (static_cast<size_t>(leaf_idx * LEAF_SIZE + j) * D);
#pragma unroll
                for (uint32_t c = 0; c < COMPONENTS_PER_THREAD; ++c) {
                  if (c >= my_d_count) {
                    break;
                  }
                  payload_accum_fma<UseKahan>(acc[c], acc_comp[c], w_j, v_j[c]);
                }
              }
            }
            __syncwarp();
          }
        } else {
          const uint32_t child_node_idx =
              current_node.child_base + added_inner_node_counter;
          added_inner_node_counter++;
          if (lane_id == 0) {
            shared_stack[warp_id][stack_ptr] = child_node_idx;
          }
          stack_ptr++;
        }
      }
    }

    if (my_required_stack_depth >= 0 && my_d_count > 0) {
      float *out = out_payload +
                   static_cast<size_t>(original_query_idx) * payload_dim_user +
                   my_d0;
#pragma unroll
      for (uint32_t c = 0; c < COMPONENTS_PER_THREAD; ++c) {
        if (c >= my_d_count) {
          break;
        }
        out[c] = acc[c];
      }
    }
  }
}

// ===========================================================================
// Single-leaf fallback (geometry_count <= 32)
// ===========================================================================
template <IsGeometry Geometry, uint32_t D, bool UseKahan>
__global__ void traverse_payload_single_leaf_kernel(
    const Vec3 *__restrict__ queries,
    const uint32_t *__restrict__ sort_indirections,
    const SoAView<Geometry> sorted_geometry,
    const float *__restrict__ sorted_payload,
    const uint32_t payload_dim_user, // NEW
    const uint32_t query_count, const uint32_t geometry_count,
    float *__restrict__ out_payload, const typename Geometry::Context ctx,
    const SceneNormalization norm) {

  constexpr uint32_t COMPONENTS_PER_THREAD = components_per_thread<D>;
  constexpr uint32_t THREADS_PER_QUERY = threads_per_query<D>;
  constexpr uint32_t QUERIES_PER_WARP = 32U / THREADS_PER_QUERY;

  const uint32_t warp_id = threadIdx.x / 32;
  const uint32_t lane_id = threadIdx.x % 32;
  const uint32_t my_query_slot = lane_id / THREADS_PER_QUERY;
  const uint32_t my_d0 = (lane_id % THREADS_PER_QUERY) * COMPONENTS_PER_THREAD;
  const uint32_t my_d_count =
      (my_d0 >= payload_dim_user)
          ? 0U
          : min(COMPONENTS_PER_THREAD, payload_dim_user - my_d0);

  __shared__ Geometry shared_geometry[32];
  __shared__ float w_leaf_values[4][32];

  if (threadIdx.x < geometry_count) {
    shared_geometry[threadIdx.x] =
        Geometry::load(sorted_geometry, threadIdx.x, geometry_count);
  }
  __syncthreads();

  uint32_t warp_tile_base = blockIdx.x * (blockDim.x / 32) * QUERIES_PER_WARP +
                            warp_id * QUERIES_PER_WARP;
  const uint32_t my_query_idx = warp_tile_base + my_query_slot;
  if (my_query_idx >= query_count) {
    return;
  }

  const uint32_t original_idx = sort_indirections[my_query_idx];
  const Vec3 my_query = norm.to_normalized(queries[original_idx]);

  float my_w = 0.F;
  if (lane_id < geometry_count) {
    my_w = shared_geometry[lane_id].contributionToQuery(my_query, ctx);
  }
  w_leaf_values[warp_id][lane_id] = my_w;
  __syncwarp();

  float acc[COMPONENTS_PER_THREAD];
  float acc_comp[COMPONENTS_PER_THREAD];
#pragma unroll
  for (uint32_t c = 0; c < COMPONENTS_PER_THREAD; ++c) {
    acc[c] = 0.F;
    acc_comp[c] = 0.F;
  }

  const float *my_v_base = sorted_payload + my_d0;
  for (uint32_t j = 0; j < geometry_count; ++j) {
    const float w_j = w_leaf_values[warp_id][j];
    const float *v_j = my_v_base + static_cast<size_t>(j * D);
#pragma unroll
    for (uint32_t c = 0; c < COMPONENTS_PER_THREAD; ++c) {
      if (c >= my_d_count) {
        break;
      }
      payload_accum_fma<UseKahan>(acc[c], acc_comp[c], w_j, v_j[c]);
    }
  }

  if (my_d_count > 0) {
    float *out = out_payload +
                 static_cast<size_t>(original_idx) * payload_dim_user + my_d0;
#pragma unroll
    for (uint32_t c = 0; c < COMPONENTS_PER_THREAD; ++c) {
      if (c >= my_d_count)
        break;
      out[c] = acc[c];
    }
  }
}

template <IsGeometry Geometry, uint32_t D>
void compute_payload_winding_numbers(
    const ComputePayloadWindingNumbersParams<Geometry, D> &params,
    int device_id, cudaStream_t stream, bool use_kahan) {
  if (params.query_count == 0) {
    return;
  }

  const auto ctx = Geometry::Context::make(params.epsilon, params.scene_scale);

  if (params.geometry_count <= 32) {
    constexpr uint32_t threads = 256;
    constexpr uint32_t Q = queries_per_warp<D>;
    const uint32_t warps_per_block = threads / 32;
    const uint32_t queries_per_block = warps_per_block * Q;
    const uint32_t blocks =
        (params.query_count + queries_per_block - 1) / queries_per_block;
    if (use_kahan) {
      traverse_payload_single_leaf_kernel<Geometry, D, true>
          <<<blocks, threads, 0, stream>>>(
              params.queries, params.sort_indirections, params.sorted_geometry,
              params.sorted_payload, params.payload_dim_user,
              params.query_count, params.geometry_count, params.out_payload,
              ctx, params.norm);
    } else {
      traverse_payload_single_leaf_kernel<Geometry, D, false>
          <<<blocks, threads, 0, stream>>>(
              params.queries, params.sort_indirections, params.sorted_geometry,
              params.sorted_payload, params.payload_dim_user,
              params.query_count, params.geometry_count, params.out_payload,
              ctx, params.norm);
    }
    CUDA_CHECK(cudaGetLastError());
    return;
  }

  const float beta_2 = params.beta * params.beta;
  constexpr int threads = 128;

  uint32_t global_device_counter_reset = 0;
  CUDA_CHECK(cudaMemcpyAsync(params.global_device_counter,
                             &global_device_counter_reset, sizeof(uint32_t),
                             cudaMemcpyHostToDevice, stream));

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device_id);

  int blocks_per_sm = 0;
  if (use_kahan) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm,
        traverse_payload_winding_numbers_kernel<Geometry, D, true>, threads, 0);
    int blocks = blocks_per_sm * deviceProp.multiProcessorCount;
    traverse_payload_winding_numbers_kernel<Geometry, D, true>
        <<<blocks, threads, 0, stream>>>(
            params.queries, params.sort_indirections, params.bvh8_nodes,
            params.bvh8_leaf_pointers, params.node_coefficients,
            params.leaf_coefficients, params.leaf_aabbs, params.sorted_geometry,
            params.sorted_payload, params.payload_dim_user, params.query_count,
            params.geometry_count, params.out_payload,
            params.global_device_counter, beta_2, ctx, params.norm);
  } else {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm,
        traverse_payload_winding_numbers_kernel<Geometry, D, false>, threads,
        0);
    int blocks = blocks_per_sm * deviceProp.multiProcessorCount;
    traverse_payload_winding_numbers_kernel<Geometry, D, false>
        <<<blocks, threads, 0, stream>>>(
            params.queries, params.sort_indirections, params.bvh8_nodes,
            params.bvh8_leaf_pointers, params.node_coefficients,
            params.leaf_coefficients, params.leaf_aabbs, params.sorted_geometry,
            params.sorted_payload, params.payload_dim_user, params.query_count,
            params.geometry_count, params.out_payload,
            params.global_device_counter, beta_2, ctx, params.norm);
  }
  CUDA_CHECK(cudaGetLastError());
}

template <IsGeometry Geometry, uint32_t D, bool UseKahan>
__global__ void __launch_bounds__(128) traverse_payload_query_gradient_kernel(
    const Vec3 *__restrict__ queries,
    const uint32_t *__restrict__ sort_indirections,
    const BVH8Node *__restrict__ bvh8_nodes,
    const LeafPointers *__restrict__ bvh8_leaf_pointers,
    const PayloadTaylorCoefficientsF16<D> *__restrict__ node_coefficients,
    const PayloadTaylorCoefficientsF16<D> *__restrict__ leaf_coefficients,
    const AABB *__restrict__ leaf_aabbs,
    const SoAView<Geometry> sorted_geometry,
    const float *__restrict__ sorted_payload,
    const uint32_t payload_dim_user, // NEW
    const float *__restrict__ grad_output, const uint32_t query_count,
    const uint32_t geometry_count, float *__restrict__ out_query_gradients,
    uint32_t *__restrict__ global_device_counter, const float beta_2,
    const typename Geometry::Context ctx, const SceneNormalization norm) {

  constexpr uint32_t COMPONENT_PER_THREAD = components_per_thread<D>;
  constexpr uint32_t THREADS_PER_QUERY = threads_per_query<D>;
  constexpr uint32_t QUERIES_PER_WARP = queries_per_warp<D>;

  const uint32_t warp_id = threadIdx.x / 32;
  const uint32_t lane_id = threadIdx.x % 32;
  const uint32_t my_query_slot = lane_id / THREADS_PER_QUERY;
  const uint32_t my_dblock = lane_id % THREADS_PER_QUERY;
  const uint32_t my_d0 = my_dblock * COMPONENT_PER_THREAD;
  const uint32_t my_d_count =
      (my_d0 >= payload_dim_user)
          ? 0U
          : min(COMPONENT_PER_THREAD, payload_dim_user - my_d0);

  __shared__ uint32_t shared_stack[4][64];
  __shared__ BVH8Node current_node_cache[4];
  __shared__ PayloadTaylorCoefficientsF16<D> current_coeff_cache[4];
  __shared__ LeafPointers shared_leaf_ptrs[4];
  __shared__ Vec3 grad_leaf_values[4][32];

  uint32_t warp_tile_base;

  while (true) {
    if (lane_id == 0) {
      warp_tile_base = atomicAdd(global_device_counter, QUERIES_PER_WARP);
    }
    warp_tile_base = __shfl_sync(0xFFFFFFFF, warp_tile_base, 0);
    if (warp_tile_base >= query_count)
      break;

    int my_required_stack_depth = std::numeric_limits<int>::max();
    const uint32_t my_query_idx = warp_tile_base + my_query_slot;
    Vec3 my_query{.x = 0.F, .y = 0.F, .z = 0.F};
    uint32_t original_query_idx = 0xFFFFFFFF;
    float g_block[COMPONENT_PER_THREAD];
#pragma unroll
    for (uint32_t c = 0; c < COMPONENT_PER_THREAD; ++c)
      g_block[c] = 0.F;

    if (my_query_idx >= query_count) {
      my_required_stack_depth = -1;
    } else {
      original_query_idx = sort_indirections[my_query_idx];
      my_query = norm.to_normalized(queries[original_query_idx]);
      // CHANGED — user stride for grad_output; loop bound already respects
      //           my_d_count so we never read past D_user
      const float *g_full =
          grad_output +
          static_cast<size_t>(original_query_idx) * payload_dim_user + my_d0;
#pragma unroll
      for (uint32_t c = 0; c < COMPONENT_PER_THREAD; ++c) {
        if (c < my_d_count) {
          g_block[c] = g_full[c];
        }
      }
    }

    Vec3 acc_grad = Vec3::zero();
    Vec3 acc_grad_comp = Vec3::zero();

    int stack_ptr = 0;
    if (lane_id == 0)
      shared_stack[warp_id][stack_ptr] = 0;
    stack_ptr++;
    const float two_eps = 2.F * ctx.eps_length;

    while (true) {
      uint32_t stack_not_empty_mask = __ballot_sync(0xFFFFFFFF, stack_ptr > 0);
      if (stack_not_empty_mask == 0) {
        break;
      }

      uint32_t current_node_idx;
      --stack_ptr;
      if (lane_id == 0) {
        current_node_idx = shared_stack[warp_id][stack_ptr];
      }
      current_node_idx = __shfl_sync(0xFFFFFFFF, current_node_idx, 0);
      load_shared_cooperative<BVH8Node>(&current_node_cache[warp_id],
                                        bvh8_nodes + current_node_idx, lane_id);
      __syncwarp();
      const BVH8Node &current_node = current_node_cache[warp_id];

      bool is_active = stack_ptr < my_required_stack_depth;
      if (is_active) {
        my_required_stack_depth = std::numeric_limits<int>::max();
      }

      bool need_coeffs =
          is_active && should_node_be_approximated(
                           my_query, current_node.getAABB(), beta_2, two_eps);
      uint32_t load_mask = __ballot_sync(0xFFFFFFFF, need_coeffs);
      if (load_mask > 0) {
        load_shared_cooperative<PayloadTaylorCoefficientsF16<D>>(
            &current_coeff_cache[warp_id], node_coefficients + current_node_idx,
            lane_id);
        __syncwarp();
      }

      if (need_coeffs) {
        const auto &coeffs = current_coeff_cache[warp_id];
        Vec3 tilde_0 = Vec3::zero();
        Mat3x3 tilde_1 = Mat3x3::zero();
        Tensor3_compressed tilde_2 = Tensor3_compressed::zero();

#pragma unroll
        for (uint32_t c = 0; c < COMPONENT_PER_THREAD; ++c) {
          if (c >= my_d_count)
            break;
          const float g = g_block[c];
          const uint32_t d = my_d0 + c;
          tilde_0 += g * Vec3::from_f16(coeffs.getZeroOrder(d));
          tilde_1 += g * Mat3x3::from_f16(coeffs.getFirstOrder(d));
          tilde_2 +=
              g * Tensor3_compressed::from_bf16(coeffs.getSecondOrder(d));
        }
        Vec3 approx_grad = compute_node_query_grad_approximation(
            my_query, current_node.getAABB().center_of_mass,
            Vec3_f16::from_float(tilde_0), Mat3x3_f16::from_float(tilde_1),
            Tensor3_bf16_compressed::from_float(tilde_2));
        if constexpr (UseKahan) {
          kahan_add(acc_grad, approx_grad, acc_grad_comp);
        } else {
          acc_grad += approx_grad;
        }
        my_required_stack_depth = stack_ptr;
      }

      bool is_still_active = my_required_stack_depth > stack_ptr;
      uint32_t interest_mask = __ballot_sync(0xFFFFFFFF, is_still_active);
      if (interest_mask == 0) {
        continue;
      }

      bool is_leaf = lane_id < 8
                         ? current_node.getChildMeta(lane_id) == ChildType::LEAF
                         : false;
      uint32_t leaf_mask = __ballot_sync(0xFFFFFFFF, is_leaf);
      if (leaf_mask > 0) {
        load_shared_cooperative(&shared_leaf_ptrs[warp_id],
                                &bvh8_leaf_pointers[current_node_idx], lane_id);
        __syncwarp();
      }

      uint32_t added_inner_node_counter = 0;
      for (uint32_t child_idx = 0; child_idx < 8; ++child_idx) {
        ChildType child_type = current_node.getChildMeta(child_idx);
        if (child_type == ChildType::EMPTY) {
          continue;
        }

        if (child_type == ChildType::LEAF) {
          const uint32_t leaf_idx =
              shared_leaf_ptrs[warp_id].indices[child_idx];
          const AABB child_aabb = leaf_aabbs[leaf_idx];

          bool need_leaf_coeffs =
              is_still_active && should_node_be_approximated(
                                     my_query, child_aabb, beta_2, two_eps);
          uint32_t load_mask_l = __ballot_sync(0xFFFFFFFF, need_leaf_coeffs);
          if (load_mask_l > 0) {
            load_shared_cooperative<PayloadTaylorCoefficientsF16<D>>(
                &current_coeff_cache[warp_id], leaf_coefficients + leaf_idx,
                lane_id);
            __syncwarp();
          }

          bool detail_needed = true;
          if (need_leaf_coeffs) {
            const auto &coeffs = current_coeff_cache[warp_id];
            Vec3 tilde_0 = Vec3::zero();
            Mat3x3 tilde_1 = Mat3x3::zero();
            Tensor3_compressed tilde_2 = Tensor3_compressed::zero();
#pragma unroll
            for (uint32_t c = 0; c < COMPONENT_PER_THREAD; ++c) {
              if (c >= my_d_count)
                break;
              const float g = g_block[c];
              const uint32_t d = my_d0 + c;
              tilde_0 += g * Vec3::from_f16(coeffs.getZeroOrder(d));
              tilde_1 += g * Mat3x3::from_f16(coeffs.getFirstOrder(d));
              tilde_2 +=
                  g * Tensor3_compressed::from_bf16(coeffs.getSecondOrder(d));
            }
            Vec3 approx_grad = compute_node_query_grad_approximation(
                my_query, child_aabb.center_of_mass,
                Vec3_f16::from_float(tilde_0), Mat3x3_f16::from_float(tilde_1),
                Tensor3_bf16_compressed::from_float(tilde_2));
            if constexpr (UseKahan)
              kahan_add(acc_grad, approx_grad, acc_grad_comp);
            else
              acc_grad += approx_grad;
            detail_needed = false;
          }

          uint32_t detail_mask =
              __ballot_sync(0xFFFFFFFF, detail_needed && is_still_active);
          if (detail_mask == 0) {
            continue;
          }

          const uint32_t my_geom_idx = leaf_idx * 32 + lane_id;
          const bool my_geom_in_bounds = my_geom_idx < geometry_count;
          Geometry my_geometry =
              Geometry::load(sorted_geometry, my_geom_idx, geometry_count);

          while (detail_mask > 0) {
            const int leader_lane = __ffs(detail_mask) - 1;
            const uint32_t leader_slot = leader_lane / THREADS_PER_QUERY;
            const uint32_t group_mask =
                ((THREADS_PER_QUERY >= 32U)
                     ? 0xFFFFFFFFU
                     : (((1U << THREADS_PER_QUERY) - 1U)
                        << (leader_slot * THREADS_PER_QUERY)));
            detail_mask &= ~group_mask;

            const Vec3 shared_query{
                .x = __shfl_sync(0xFFFFFFFF, my_query.x, leader_lane),
                .y = __shfl_sync(0xFFFFFFFF, my_query.y, leader_lane),
                .z = __shfl_sync(0xFFFFFFFF, my_query.z, leader_lane),
            };

            Vec3 my_grad = Vec3::zero();
            if (my_geom_in_bounds) {
              my_grad =
                  my_geometry.queryGradContribution(shared_query, 1.F, ctx);
            }
            grad_leaf_values[warp_id][lane_id] = my_grad;
            __syncwarp();

            if (my_query_slot == leader_slot) {
              for (uint32_t j = 0; j < LEAF_SIZE; ++j) {
                const uint32_t geom_j = leaf_idx * LEAF_SIZE + j;
                if (geom_j >= geometry_count)
                  break;
                const float *v_j =
                    sorted_payload + static_cast<size_t>(geom_j * D);
                float partial_dot = 0.F;
#pragma unroll
                for (uint32_t c = 0; c < COMPONENT_PER_THREAD; ++c) {
                  if (c >= my_d_count)
                    break;
                  partial_dot += g_block[c] * v_j[my_d0 + c];
                }
                const Vec3 w_grad = grad_leaf_values[warp_id][j];
                const Vec3 delta = w_grad * partial_dot;
                if constexpr (UseKahan)
                  kahan_add(acc_grad, delta, acc_grad_comp);
                else
                  acc_grad += delta;
              }
            }
            __syncwarp();
          }
        } else {
          const uint32_t child_node_idx =
              current_node.child_base + added_inner_node_counter;
          added_inner_node_counter++;
          if (lane_id == 0) {
            shared_stack[warp_id][stack_ptr] = child_node_idx;
          }
          stack_ptr++;
        }
      }
    }

    // Cross-dblock reduction (unchanged — output is 3D per query).
    if constexpr (THREADS_PER_QUERY > 1) {
      for (uint32_t offset = 1; offset < THREADS_PER_QUERY; offset *= 2) {
        acc_grad.x += __shfl_xor_sync(0xFFFFFFFF, acc_grad.x, offset);
        acc_grad.y += __shfl_xor_sync(0xFFFFFFFF, acc_grad.y, offset);
        acc_grad.z += __shfl_xor_sync(0xFFFFFFFF, acc_grad.z, offset);
      }
    }

    if (my_required_stack_depth >= 0 && my_dblock == 0) {
      const size_t out_idx = static_cast<size_t>(original_query_idx) * 3;
      out_query_gradients[out_idx] = acc_grad.x * norm.scale;
      out_query_gradients[out_idx + 1] = acc_grad.y * norm.scale;
      out_query_gradients[out_idx + 2] = acc_grad.z * norm.scale;
    }
  }
}

template <IsGeometry Geometry, uint32_t D, bool UseKahan>
__global__ void traverse_payload_query_grad_single_leaf_kernel(
    const Vec3 *__restrict__ queries,
    const uint32_t *__restrict__ sort_indirections,
    const SoAView<Geometry> sorted_geometry,
    const float *__restrict__ sorted_payload,
    const uint32_t payload_dim_user, // NEW
    const float *__restrict__ grad_output, const uint32_t query_count,
    const uint32_t geometry_count, float *__restrict__ out_query_gradients,
    const typename Geometry::Context ctx, const SceneNormalization norm) {

  constexpr uint32_t COMPONENTS_PER_THREAD = components_per_thread<D>;
  constexpr uint32_t THREADS_PER_QUERY = threads_per_query<D>;
  constexpr uint32_t QUERIES_PER_WARP = queries_per_warp<D>;

  const uint32_t warp_id = threadIdx.x / 32;
  const uint32_t lane_id = threadIdx.x % 32;
  const uint32_t my_query_slot = lane_id / THREADS_PER_QUERY;
  const uint32_t my_dblock = lane_id % THREADS_PER_QUERY;
  const uint32_t my_d0 = my_dblock * COMPONENTS_PER_THREAD;
  const uint32_t my_d_count =
      (my_d0 >= payload_dim_user)
          ? 0U
          : min(COMPONENTS_PER_THREAD, payload_dim_user - my_d0);

  __shared__ Geometry shared_geometry[32];
  __shared__ Vec3 grad_values[4][32];

  if (threadIdx.x < geometry_count) {
    shared_geometry[threadIdx.x] =
        Geometry::load(sorted_geometry, threadIdx.x, geometry_count);
  }
  __syncthreads();

  const uint32_t warps_per_block = blockDim.x / 32;
  const uint32_t warp_tile_base =
      blockIdx.x * warps_per_block * QUERIES_PER_WARP +
      warp_id * QUERIES_PER_WARP;
  const uint32_t my_query_idx = warp_tile_base + my_query_slot;

  const bool my_active = my_query_idx < query_count;
  Vec3 my_query{.x = 0.F, .y = 0.F, .z = 0.F};
  uint32_t original_idx = 0xFFFFFFFF;
  float g_block[COMPONENTS_PER_THREAD];
#pragma unroll
  for (uint32_t c = 0; c < COMPONENTS_PER_THREAD; ++c)
    g_block[c] = 0.F;

  if (my_active) {
    original_idx = sort_indirections[my_query_idx];
    my_query = norm.to_normalized(queries[original_idx]);
    const float *g_full = grad_output +
                          static_cast<size_t>(original_idx) * payload_dim_user +
                          my_d0;
#pragma unroll
    for (uint32_t c = 0; c < COMPONENTS_PER_THREAD; ++c) {
      if (c < my_d_count) {
        g_block[c] = g_full[c];
      }
    }
  }

  Vec3 acc_grad = Vec3::zero();
  Vec3 acc_grad_comp = Vec3::zero();

  for (uint32_t leader_slot = 0; leader_slot < QUERIES_PER_WARP;
       ++leader_slot) {
    const uint32_t leader_lane = leader_slot * THREADS_PER_QUERY;
    const bool is_leader_active = (warp_tile_base + leader_slot) < query_count;
    if (!is_leader_active)
      continue;

    const Vec3 shared_query{
        .x = __shfl_sync(0xFFFFFFFF, my_query.x, leader_lane),
        .y = __shfl_sync(0xFFFFFFFF, my_query.y, leader_lane),
        .z = __shfl_sync(0xFFFFFFFF, my_query.z, leader_lane),
    };

    for (uint32_t p_base = 0; p_base < geometry_count; p_base += LEAF_SIZE) {
      Vec3 my_grad = Vec3::zero();
      if (p_base + lane_id < geometry_count) {
        my_grad = shared_geometry[p_base + lane_id].queryGradContribution(
            shared_query, 1.F, ctx);
      }
      grad_values[warp_id][lane_id] = my_grad;
      __syncwarp();

      if (my_query_slot == leader_slot) {
        for (uint32_t j = 0; j < LEAF_SIZE; ++j) {
          const uint32_t geom_j = p_base + j;
          if (geom_j >= geometry_count)
            break;
          const float *v_j = sorted_payload + geom_j * D; // bucket stride
          float partial_dot = 0.F;
#pragma unroll
          for (uint32_t c = 0; c < COMPONENTS_PER_THREAD; ++c) {
            if (c >= my_d_count)
              break;
            partial_dot += g_block[c] * v_j[my_d0 + c];
          }
          const Vec3 w_grad = grad_values[warp_id][j];
          const Vec3 delta = w_grad * partial_dot;
          if constexpr (UseKahan)
            kahan_add(acc_grad, delta, acc_grad_comp);
          else
            acc_grad += delta;
        }
      }
      __syncwarp();
    }
  }

  if constexpr (THREADS_PER_QUERY > 1) {
    for (uint32_t offset = 1; offset < THREADS_PER_QUERY; offset *= 2) {
      acc_grad.x += __shfl_xor_sync(0xFFFFFFFF, acc_grad.x, offset);
      acc_grad.y += __shfl_xor_sync(0xFFFFFFFF, acc_grad.y, offset);
      acc_grad.z += __shfl_xor_sync(0xFFFFFFFF, acc_grad.z, offset);
    }
  }

  if (my_active && my_dblock == 0) {
    const size_t out_idx = static_cast<size_t>(original_idx) * 3;
    out_query_gradients[out_idx] = acc_grad.x * norm.scale;
    out_query_gradients[out_idx + 1] = acc_grad.y * norm.scale;
    out_query_gradients[out_idx + 2] = acc_grad.z * norm.scale;
  }
}

template <IsGeometry Geometry, uint32_t D>
void query_gradient_traversal_payload(
    const ComputePayloadQueryGradientParams<Geometry, D> &params, int device_id,
    cudaStream_t stream, bool use_kahan) {
  if (params.query_count == 0) {
    return;
  }

  const auto ctx = Geometry::Context::make(params.epsilon, params.scene_scale);

  if (params.geometry_count <= 32) {
    constexpr uint32_t threads = 256;
    constexpr uint32_t Q = queries_per_warp<D>;
    const uint32_t queries_per_block = (threads / 32) * Q;
    const uint32_t blocks =
        (params.query_count + queries_per_block - 1) / queries_per_block;
    if (use_kahan) {
      traverse_payload_query_grad_single_leaf_kernel<Geometry, D, true>
          <<<blocks, threads, 0, stream>>>(
              params.queries, params.sort_indirections, params.sorted_geometry,
              params.sorted_payload, params.payload_dim_user, params.grad_output, params.query_count,
              params.geometry_count, params.out_query_gradients, ctx,
              params.norm);
    } else {
      traverse_payload_query_grad_single_leaf_kernel<Geometry, D, false>
          <<<blocks, threads, 0, stream>>>(
              params.queries, params.sort_indirections, params.sorted_geometry,
              params.sorted_payload, params.payload_dim_user, params.grad_output, params.query_count,
              params.geometry_count, params.out_query_gradients, ctx,
              params.norm);
    }
    CUDA_CHECK(cudaGetLastError());
    return;
  }

  const float beta_2 = params.beta * params.beta;

  constexpr int threads = 128;

  uint32_t global_device_counter_reset = 0;
  CUDA_CHECK(cudaMemcpyAsync(params.global_device_counter,
                             &global_device_counter_reset, sizeof(uint32_t),
                             cudaMemcpyHostToDevice, stream));

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device_id);

  int blocks_per_sm = 0;
  if (use_kahan) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm,
        traverse_payload_query_gradient_kernel<Geometry, D, true>, threads, 0);
    int blocks = blocks_per_sm * deviceProp.multiProcessorCount;
    traverse_payload_query_gradient_kernel<Geometry, D, true>
        <<<blocks, threads, 0, stream>>>(
            params.queries, params.sort_indirections, params.bvh8_nodes,
            params.bvh8_leaf_pointers, params.node_coefficients,
            params.leaf_coefficients, params.leaf_aabbs, params.sorted_geometry,
            params.sorted_payload, params.payload_dim_user, params.grad_output, params.query_count,
            params.geometry_count, params.out_query_gradients,
            params.global_device_counter, beta_2, ctx, params.norm);
  } else {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm,
        traverse_payload_query_gradient_kernel<Geometry, D, false>, threads, 0);
    int blocks = blocks_per_sm * deviceProp.multiProcessorCount;
    traverse_payload_query_gradient_kernel<Geometry, D, false>
        <<<blocks, threads, 0, stream>>>(
            params.queries, params.sort_indirections, params.bvh8_nodes,
            params.bvh8_leaf_pointers, params.node_coefficients,
            params.leaf_coefficients, params.leaf_aabbs, params.sorted_geometry,
            params.sorted_payload, params.payload_dim_user, params.grad_output, params.query_count,
            params.geometry_count, params.out_query_gradients,
            params.global_device_counter, beta_2, ctx, params.norm);
  }
  CUDA_CHECK(cudaGetLastError());
}

#define WINDER_INSTANTIATE_PAYLOAD_TRAVERSAL(GEOM, DIM)                        \
  template void compute_payload_winding_numbers<GEOM, DIM>(                    \
      const ComputePayloadWindingNumbersParams<GEOM, DIM> &, int,              \
      cudaStream_t, bool);                                                     \
  template void query_gradient_traversal_payload<GEOM, DIM>(                   \
      const ComputePayloadQueryGradientParams<GEOM, DIM> &, int, cudaStream_t, \
      bool);

WINDER_INSTANTIATE_PAYLOAD_TRAVERSAL(PointNormal, 1)
WINDER_INSTANTIATE_PAYLOAD_TRAVERSAL(PointNormal, 2)
WINDER_INSTANTIATE_PAYLOAD_TRAVERSAL(PointNormal, 3)
WINDER_INSTANTIATE_PAYLOAD_TRAVERSAL(PointNormal, 4)
WINDER_INSTANTIATE_PAYLOAD_TRAVERSAL(PointNormal, 8)
WINDER_INSTANTIATE_PAYLOAD_TRAVERSAL(PointNormal, 16)
WINDER_INSTANTIATE_PAYLOAD_TRAVERSAL(PointNormal, 32)
WINDER_INSTANTIATE_PAYLOAD_TRAVERSAL(PointNormal, 64)
WINDER_INSTANTIATE_PAYLOAD_TRAVERSAL(PointNormal, 128)

WINDER_INSTANTIATE_PAYLOAD_TRAVERSAL(Triangle, 1)
WINDER_INSTANTIATE_PAYLOAD_TRAVERSAL(Triangle, 2)
WINDER_INSTANTIATE_PAYLOAD_TRAVERSAL(Triangle, 3)
WINDER_INSTANTIATE_PAYLOAD_TRAVERSAL(Triangle, 4)
WINDER_INSTANTIATE_PAYLOAD_TRAVERSAL(Triangle, 8)
WINDER_INSTANTIATE_PAYLOAD_TRAVERSAL(Triangle, 16)
WINDER_INSTANTIATE_PAYLOAD_TRAVERSAL(Triangle, 32)
WINDER_INSTANTIATE_PAYLOAD_TRAVERSAL(Triangle, 64)
WINDER_INSTANTIATE_PAYLOAD_TRAVERSAL(Triangle, 128)

#undef WINDER_INSTANTIATE_PAYLOAD_TRAVERSAL
