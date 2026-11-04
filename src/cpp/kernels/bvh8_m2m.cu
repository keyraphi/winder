#include "aabb.h"
#include "bvh8.h"
#include "common.cuh"
#include "mat3x3.h"
#include "taylor_coefficients.h"
#include "tensor3.h"
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
#include <vector_functions.h>
#include <vector_types.h>

__global__ void compute_internal_tailor_coefficients_m2m_kernel(
    BVH8Node *nodes, const uint32_t *internal_parent_map,
    const AABB *leaf_aabbs, const TaylorCoefficientsF16 *leaf_coefficients,
    const uint32_t *leaf_parents, const LeafPointers *leaf_pointers,
    TaylorCoefficientsF16 *node_tailor_coefficients,
    TaylorCoefficients *m2m_f32_coefficients, const uint32_t *nodes_child_count,
    const uint32_t leaf_count, uint32_t *atomic_counters) {

  uint32_t leaf_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (leaf_idx >= leaf_count) {
    return;
  }

  uint32_t current_node_idx = leaf_parents[leaf_idx];

  while (current_node_idx != 0xFFFFFFFF) {
    BVH8Node node = nodes[current_node_idx];
    // race to the top
    const uint32_t expected_children = nodes_child_count[current_node_idx];

    if (atomicAdd(&atomic_counters[current_node_idx], 1) <
        expected_children - 1) {
      return;
    }

    // only one thread for current node survives
    Vec3 parent_center = node._aabb_com;

    Vec3 zero_order = {0.F, 0.F, 0.F};
    Mat3x3 first_order = {0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F};
    Tensor3_compressed second_order = {0.F, 0.F, 0.F, 0.F, 0.F, 0.F,
                                       0.F, 0.F, 0.F, 0.F, 0.F, 0.F,
                                       0.F, 0.F, 0.F, 0.F, 0.F, 0.F};

    uint32_t internal_child_count = 0;
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
      if (node.getChildMeta(i) == ChildType::EMPTY) {
        continue;
      }
      Vec3 child_center;
      TaylorCoefficients child_coefficients;
      if (node.getChildMeta(i) == ChildType::LEAF) {
        uint32_t leaf_idx = leaf_pointers[current_node_idx].indices[i];
        const AABB &leaf_aabb = leaf_aabbs[leaf_idx];
        child_center = leaf_aabb.center_of_mass;
        child_coefficients =
            TaylorCoefficients::from_f16(leaf_coefficients[leaf_idx]);
      } else {
        uint32_t child_idx = node.child_base + internal_child_count;
        internal_child_count++;
        BVH8Node child_node = nodes[child_idx];
        child_center = child_node._aabb_com;

        // Load the float32 coefficients (no quantization happened here)
        child_coefficients = m2m_f32_coefficients[child_idx];
      }

      // Merge child coefficients into parent tailor by recentering child
      //
      Vec3 shift_vector = child_center - parent_center; // v in equations
      // zero order doesn't change
      // M_0' = M_0
      const Vec3 &zero_child = child_coefficients.zero_order;
      zero_order += zero_child;
      // first order
      // M_1' = M_1 + v \otimes M_0
      const Mat3x3 &child_first = child_coefficients.first_order;
      first_order += child_first + shift_vector.outer_product(zero_child);
      // second order
      // M_2' = M_2 + 1/2 * (v \times M_1 + perm(v \times M_1)) + 1/2 * (v
      // \times v \times M_0)
      // M2_jkl′ = M2_jkl + 1/2*(v_j * M1_kl + v_k * M1_jl) + 1/2 v_j * v_k *
      //           M0_l
      const Tensor3_compressed &child_second = child_coefficients.second_order;
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
    // store accumulated coefficients
    node_tailor_coefficients[current_node_idx].zero_order = zero_order;
    node_tailor_coefficients[current_node_idx].first_order = first_order;
    node_tailor_coefficients[current_node_idx].second_order = second_order;
    // store accumulated in m2m_f32_coefficients (full resolution)
    m2m_f32_coefficients[current_node_idx] =
        TaylorCoefficients{zero_order, first_order, second_order};

    __threadfence(); // ensure the tailor coefficients of childs are written
                     // before continuing with the next
    current_node_idx = __ldcg(internal_parent_map + current_node_idx);
  }
}

void compute_internal_tailor_coefficients_m2m(
    BVH8Node *nodes, const uint32_t *internal_parent_map,
    const AABB *leaf_aabbs, const TaylorCoefficientsF16 *leaf_coefficients,
    const uint32_t *leaf_parents, const LeafPointers *leaf_pointers,
    TaylorCoefficientsF16 *node_tailor_coefficients,
    TaylorCoefficients *m2m_f32_coefficients, const uint32_t *nodes_child_count,
    const uint32_t leaf_count, uint32_t *atomic_counters,
    const cudaStream_t &stream) {
  uint32_t threads = 256;
  uint32_t blocks = (leaf_count + threads - 1) / threads;
  compute_internal_tailor_coefficients_m2m_kernel<<<blocks, threads, 0,
                                                    stream>>>(
      nodes, internal_parent_map, leaf_aabbs, leaf_coefficients, leaf_parents,
      leaf_pointers, node_tailor_coefficients, m2m_f32_coefficients,
      nodes_child_count, leaf_count, atomic_counters);
  CUDA_CHECK(cudaGetLastError());
}

__global__ void compute_internal_tailor_coefficients_m2m_backward_kernel(
    BVH8Node *nodes, const uint32_t *internal_parent_map,
    const AABB *leaf_aabbs, const BackwardTaylorCoefficients *leaf_coefficients,
    const uint32_t *leaf_parents, const LeafPointers *leaf_pointers,
    BackwardTaylorCoefficients *node_tailor_coefficients,
    const uint32_t *nodes_child_count, const uint32_t leaf_count,
    uint32_t *atomic_counters) {

  uint32_t leaf_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (leaf_idx >= leaf_count) {
    return;
  }

  uint32_t current_node_idx = leaf_parents[leaf_idx];

  while (current_node_idx != 0xFFFFFFFF) {
    BVH8Node node = nodes[current_node_idx];
    // race to the top
    const uint32_t expected_children = nodes_child_count[current_node_idx];

    if (atomicAdd(&atomic_counters[current_node_idx], 1) <
        expected_children - 1) {
      return;
    }

    // only one thread for current node survives
    Vec3 parent_center = node._aabb_com;

    float zero_order = 0.F;
    Vec3 first_order = {0.F, 0.F, 0.F};
    SymMat3x3 second_order = {0.F, 0.F, 0.F, 0.F, 0.F, 0.F};

    uint32_t internal_child_count = 0;
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
      if (node.getChildMeta(i) == ChildType::EMPTY) {
        continue;
      }
      Vec3 child_center;
      BackwardTaylorCoefficients child_coefficients;
      if (node.getChildMeta(i) == ChildType::LEAF) {
        uint32_t leaf_idx = leaf_pointers[current_node_idx].indices[i];
        const AABB &leaf_aabb = leaf_aabbs[leaf_idx];
        child_center = leaf_aabb.center_of_mass;
        child_coefficients = leaf_coefficients[leaf_idx];
      } else {
        uint32_t child_idx = node.child_base + internal_child_count;
        internal_child_count++;
        BVH8Node child_node = nodes[child_idx];
        child_center = child_node._aabb_com;

        // Load the float32 coefficients (no quantization happened here)
        child_coefficients = node_tailor_coefficients[child_idx];
      }

      // Merge child coefficients into parent tailor by recentering child
      Vec3 shift_vector = child_center - parent_center; // v in equations
      // zero order doesn't change
      // M_0' = M_0
      const float zero_child = child_coefficients.zero_order;
      zero_order += zero_child;
      // first order
      // M_1' = M_1 + M_0 * v
      const Vec3 child_first = child_coefficients.first_order;
      first_order += child_first + zero_child * shift_vector;
      // second order
      // M_2' = M_2 + M_1 (x) v + v (x) M_1 + M_0 * (v (x) v)
      const SymMat3x3 &child_second = child_coefficients.second_order;
      const float *c2 = child_second.data;
      const float vx = shift_vector.x;
      const float vy = shift_vector.y;
      const float vz = shift_vector.z;
      const float g1x = child_first.x;
      const float g1y = child_first.y;
      const float g1z = child_first.z;
      const float g0 = zero_child;

      // We rely on nvccs CSE here
      second_order.data[0] += c2[0] + 2.0F * g1x * vx + g0 * vx * vx;     // xx
      second_order.data[1] += c2[1] + g1x * vy + vx * g1y + g0 * vx * vy; // xy
      second_order.data[2] += c2[2] + g1x * vz + vx * g1z + g0 * vx * vz; // xz
      // second_order.data[3] += c2[3] + g1y * vx + vy * g1x + g0 * vy * vx; //
      // yx == xy
      second_order.data[3] += c2[3] + 2.0F * g1y * vy + g0 * vy * vy;     // yy
      second_order.data[4] += c2[4] + g1y * vz + vy * g1z + g0 * vy * vz; // yz
      // second_order.data[6] += c2[6] + g1z * vx + vz * g1x + g0 * vz * vx; //
      // zx == xz second_order.data[7] += c2[7] + g1z * vy + vz * g1y + g0 * vz
      // * vy; // zy == yz
      second_order.data[5] += c2[5] + 2.0F * g1z * vz + g0 * vz * vz; // zz
    }
    // store accumulated coefficients (f32 resolution)
    node_tailor_coefficients[current_node_idx].zero_order = zero_order;
    node_tailor_coefficients[current_node_idx].first_order = first_order;
    node_tailor_coefficients[current_node_idx].second_order = second_order;

    __threadfence(); // ensure the tailor coefficients of childs are written
                     // before continuing with the next
    current_node_idx = __ldcg(internal_parent_map + current_node_idx);
  }
}

void compute_internal_tailor_coefficients_m2m_backward(
    BVH8Node *nodes, const uint32_t *internal_parent_map,
    const AABB *leaf_aabbs, const BackwardTaylorCoefficients *leaf_coefficients,
    const uint32_t *leaf_parents, const LeafPointers *leaf_pointers,
    BackwardTaylorCoefficients *node_tailor_coefficients,
    const uint32_t *nodes_child_count, const uint32_t leaf_count,
    uint32_t *atomic_counters, const cudaStream_t &stream) {
  uint32_t threads = 256;
  uint32_t blocks = (leaf_count + threads - 1) / threads;
  compute_internal_tailor_coefficients_m2m_backward_kernel<<<blocks, threads, 0,
                                                             stream>>>(
      nodes, internal_parent_map, leaf_aabbs, leaf_coefficients, leaf_parents,
      leaf_pointers, node_tailor_coefficients, nodes_child_count, leaf_count,
      atomic_counters);
  CUDA_CHECK(cudaGetLastError());
}

// Per-d-component scalar m2m shift. Identical formula to the scalar kernel,
// applied to one d-slice.
// The 18 compressed slots of the second-order tensor are expanded by hand so
// the compiler can keep the whole expression in one register window.
// ============================================================================
__device__ __forceinline__ void
m2m_shift_d(Vec3 &zero_acc, Mat3x3 &first_acc, Tensor3_compressed &second_acc,
            const Vec3 &child_zero, const Mat3x3 &child_first,
            const Tensor3_compressed &child_second, const Vec3 &shift) {

  zero_acc += child_zero;
  first_acc += child_first + shift.outer_product(child_zero);

  const float sx = shift.x;
  const float sy = shift.y;
  const float sz = shift.z;
  const float v0x = child_zero.x;
  const float v0y = child_zero.y;
  const float v0z = child_zero.z;
  const auto &c1 = child_first.data;
  const auto &c2 = child_second.data;
  auto &s2 = second_acc.data;

  // (x, x, ·)
  s2[0] += c2[0] + sx * c1[0] + 0.5F * sx * sx * v0x;
  s2[1] += c2[1] + sx * c1[1] + 0.5F * sx * sx * v0y;
  s2[2] += c2[2] + sx * c1[2] + 0.5F * sx * sx * v0z;
  // (x, y, ·) and (x, z, ·)  ; first two indices symmetric
  s2[3] += c2[3] + 0.5F * (sx * c1[3] + sy * c1[0]) + 0.5F * sx * sy * v0x;
  s2[4] += c2[4] + 0.5F * (sx * c1[4] + sy * c1[1]) + 0.5F * sx * sy * v0y;
  s2[5] += c2[5] + 0.5F * (sx * c1[5] + sy * c1[2]) + 0.5F * sx * sy * v0z;
  s2[6] += c2[6] + 0.5F * (sx * c1[6] + sz * c1[0]) + 0.5F * sx * sz * v0x;
  s2[7] += c2[7] + 0.5F * (sx * c1[7] + sz * c1[1]) + 0.5F * sx * sz * v0y;
  s2[8] += c2[8] + 0.5F * (sx * c1[8] + sz * c1[2]) + 0.5F * sx * sz * v0z;
  // (y, y, ·)
  s2[9] += c2[9] + sy * c1[3] + 0.5F * sy * sy * v0x;
  s2[10] += c2[10] + sy * c1[4] + 0.5F * sy * sy * v0y;
  s2[11] += c2[11] + sy * c1[5] + 0.5F * sy * sy * v0z;
  // (y, z, ·)
  s2[12] += c2[12] + 0.5F * (sy * c1[6] + sz * c1[3]) + 0.5F * sy * sz * v0x;
  s2[13] += c2[13] + 0.5F * (sy * c1[7] + sz * c1[4]) + 0.5F * sy * sz * v0y;
  s2[14] += c2[14] + 0.5F * (sy * c1[8] + sz * c1[5]) + 0.5F * sy * sz * v0z;
  // (z, z, ·)
  s2[15] += c2[15] + sz * c1[6] + 0.5F * sz * sz * v0x;
  s2[16] += c2[16] + sz * c1[7] + 0.5F * sz * sz * v0y;
  s2[17] += c2[17] + sz * c1[8] + 0.5F * sz * sz * v0z;
}

template <uint32_t D>
__device__ __forceinline__ void
load_d_f16(const PayloadTaylorCoefficientsF16<D> &src, uint32_t d, Vec3 &zero,
           Mat3x3 &first, Tensor3_compressed &second) {
  zero = Vec3::from_f16(src.getZeroOrder(d));
  first = Mat3x3::from_f16(src.getFirstOrder(d));
  second = Tensor3_compressed::from_bf16(src.getSecondOrder(d));
}

template <uint32_t D>
__device__ __forceinline__ void
load_d_f32(const PayloadTaylorCoefficients<D> &src, uint32_t d, Vec3 &zero,
           Mat3x3 &first, Tensor3_compressed &second) {
  zero = src.getZeroOrder(d);
  first = src.getFirstOrder(d);
  second = src.getSecondOrder(d);
}

template <uint32_t D>
__device__ __forceinline__ void
write_d_f16(PayloadTaylorCoefficientsF16<D> &dst, uint32_t d, const Vec3 &zero,
            const Mat3x3 &first, const Tensor3_compressed &second) {
  dst.set_zero_order(d, Vec3_f16::from_float(zero));
  dst.set_first_order(d, Mat3x3_f16::from_float(first));
  dst.set_second_order(d, Tensor3_bf16_compressed::from_float(second));
}

template <uint32_t D>
__device__ __forceinline__ void
write_d_f32(PayloadTaylorCoefficients<D> &dst, uint32_t d, const Vec3 &zero,
            const Mat3x3 &first, const Tensor3_compressed &second) {
  dst.set_zero_order(d, zero);
  dst.set_first_order(d, first);
  dst.set_second_order(d, second);
}

template <uint32_t D>
__global__ void compute_internal_tailor_coefficients_m2m_payload_kernel(
    BVH8Node *nodes, const uint32_t *internal_parent_map,
    const AABB *leaf_aabbs,
    const PayloadTaylorCoefficientsF16<D> *leaf_coefficients,
    const uint32_t *leaf_parents, const LeafPointers *leaf_pointers,
    PayloadTaylorCoefficientsF16<D> *node_tailor_coefficients,
    PayloadTaylorCoefficients<D> *m2m_f32_coefficients,
    const uint32_t *nodes_child_count, const uint32_t leaf_count,
    uint32_t *atomic_counters) {

  const uint32_t leaf_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (leaf_idx >= leaf_count) {
    return;
  }

  uint32_t current_node_idx = leaf_parents[leaf_idx];

  while (current_node_idx != 0xFFFFFFFF) {
    BVH8Node node = nodes[current_node_idx];
    const uint32_t expected_children = nodes_child_count[current_node_idx];

    // Only the last-arriving child thread merges. All others exit.
    if (atomicAdd(&atomic_counters[current_node_idx], 1) <
        expected_children - 1) {
      return;
    }

    const Vec3 parent_center = node._aabb_com;

    // Process one d-component at a time.
    for (uint32_t d = 0; d < D; ++d) {
      Vec3 zero_acc{.x = 0.F, .y = 0.F, .z = 0.F};
      Mat3x3 first_acc{};
      Tensor3_compressed second_acc{};

      uint32_t internal_child_count = 0;
#pragma unroll
      for (uint32_t i = 0; i < 8; ++i) {
        if (node.getChildMeta(i) == ChildType::EMPTY) {
          continue;
        }

        Vec3 child_center;
        Vec3 child_zero;
        Mat3x3 child_first;
        Tensor3_compressed child_second;

        if (node.getChildMeta(i) == ChildType::LEAF) {
          const uint32_t child_leaf_idx =
              leaf_pointers[current_node_idx].indices[i];
          child_center = leaf_aabbs[child_leaf_idx].center_of_mass;
          load_d_f16(leaf_coefficients[child_leaf_idx], d, child_zero,
                     child_first, child_second);
        } else {
          const uint32_t child_idx = node.child_base + internal_child_count;
          internal_child_count++;
          child_center = nodes[child_idx]._aabb_com;
          load_d_f32(m2m_f32_coefficients[child_idx], d, child_zero,
                     child_first, child_second);
        }

        const Vec3 shift = child_center - parent_center;
        m2m_shift_d(zero_acc, first_acc, second_acc, child_zero, child_first,
                    child_second, shift);
      }

      write_d_f16(node_tailor_coefficients[current_node_idx], d, zero_acc,
                  first_acc, second_acc);
      write_d_f32(m2m_f32_coefficients[current_node_idx], d, zero_acc,
                  first_acc,
                  second_acc); // TODO: Really needed for precision? Test
    }

    // Ensure this node's f32 coefficients are visible to the parent's
    // winning thread on the next tree level before we advance.
    __threadfence();
    current_node_idx = __ldcg(internal_parent_map + current_node_idx);
  }
}

template <uint32_t D>
void compute_internal_tailor_coefficients_m2m_payload(
    BVH8Node *nodes, const uint32_t *internal_parent_map,
    const AABB *leaf_aabbs,
    const PayloadTaylorCoefficientsF16<D> *leaf_coefficients,
    const uint32_t *leaf_parents, const LeafPointers *leaf_pointers,
    PayloadTaylorCoefficientsF16<D> *node_tailor_coefficients,
    PayloadTaylorCoefficients<D> *m2m_f32_coefficients,
    const uint32_t *nodes_child_count, const uint32_t leaf_count,
    uint32_t *atomic_counters, cudaStream_t stream) {
  const uint32_t threads = 256;
  const uint32_t blocks = (leaf_count + threads - 1) / threads;
  compute_internal_tailor_coefficients_m2m_payload_kernel<D>
      <<<blocks, threads, 0, stream>>>(
          nodes, internal_parent_map, leaf_aabbs, leaf_coefficients,
          leaf_parents, leaf_pointers, node_tailor_coefficients,
          m2m_f32_coefficients, nodes_child_count, leaf_count, atomic_counters);
  CUDA_CHECK(cudaGetLastError());
}

// Explicit instantiations — one per (D bucket).
template void compute_internal_tailor_coefficients_m2m_payload<1>(
    BVH8Node *, const uint32_t *, const AABB *,
    const PayloadTaylorCoefficientsF16<1> *, const uint32_t *,
    const LeafPointers *, PayloadTaylorCoefficientsF16<1> *,
    PayloadTaylorCoefficients<1> *, const uint32_t *, uint32_t, uint32_t *,
    cudaStream_t);
template void compute_internal_tailor_coefficients_m2m_payload<2>(
    BVH8Node *, const uint32_t *, const AABB *,
    const PayloadTaylorCoefficientsF16<2> *, const uint32_t *,
    const LeafPointers *, PayloadTaylorCoefficientsF16<2> *,
    PayloadTaylorCoefficients<2> *, const uint32_t *, uint32_t, uint32_t *,
    cudaStream_t);
template void compute_internal_tailor_coefficients_m2m_payload<3>(
    BVH8Node *, const uint32_t *, const AABB *,
    const PayloadTaylorCoefficientsF16<3> *, const uint32_t *,
    const LeafPointers *, PayloadTaylorCoefficientsF16<3> *,
    PayloadTaylorCoefficients<3> *, const uint32_t *, uint32_t, uint32_t *,
    cudaStream_t);
template void compute_internal_tailor_coefficients_m2m_payload<4>(
    BVH8Node *, const uint32_t *, const AABB *,
    const PayloadTaylorCoefficientsF16<4> *, const uint32_t *,
    const LeafPointers *, PayloadTaylorCoefficientsF16<4> *,
    PayloadTaylorCoefficients<4> *, const uint32_t *, uint32_t, uint32_t *,
    cudaStream_t);
template void compute_internal_tailor_coefficients_m2m_payload<8>(
    BVH8Node *, const uint32_t *, const AABB *,
    const PayloadTaylorCoefficientsF16<8> *, const uint32_t *,
    const LeafPointers *, PayloadTaylorCoefficientsF16<8> *,
    PayloadTaylorCoefficients<8> *, const uint32_t *, uint32_t, uint32_t *,
    cudaStream_t);
template void compute_internal_tailor_coefficients_m2m_payload<16>(
    BVH8Node *, const uint32_t *, const AABB *,
    const PayloadTaylorCoefficientsF16<16> *, const uint32_t *,
    const LeafPointers *, PayloadTaylorCoefficientsF16<16> *,
    PayloadTaylorCoefficients<16> *, const uint32_t *, uint32_t, uint32_t *,
    cudaStream_t);
template void compute_internal_tailor_coefficients_m2m_payload<32>(
    BVH8Node *, const uint32_t *, const AABB *,
    const PayloadTaylorCoefficientsF16<32> *, const uint32_t *,
    const LeafPointers *, PayloadTaylorCoefficientsF16<32> *,
    PayloadTaylorCoefficients<32> *, const uint32_t *, uint32_t, uint32_t *,
    cudaStream_t);
template void compute_internal_tailor_coefficients_m2m_payload<64>(
    BVH8Node *, const uint32_t *, const AABB *,
    const PayloadTaylorCoefficientsF16<64> *, const uint32_t *,
    const LeafPointers *, PayloadTaylorCoefficientsF16<64> *,
    PayloadTaylorCoefficients<64> *, const uint32_t *, uint32_t, uint32_t *,
    cudaStream_t);
template void compute_internal_tailor_coefficients_m2m_payload<128>(
    BVH8Node *, const uint32_t *, const AABB *,
    const PayloadTaylorCoefficientsF16<128> *, const uint32_t *,
    const LeafPointers *, PayloadTaylorCoefficientsF16<128> *,
    PayloadTaylorCoefficients<128> *, const uint32_t *, uint32_t, uint32_t *,
    cudaStream_t);
