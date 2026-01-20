#pragma once
#include "aabb.h"
#include "mat3x3.h"
#include "tensor3.h"
#include "vec3.h"
#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <vector_types.h>

// Info what each of the childs are
enum class ChildType : uint8_t { EMPTY = 0, INTERNAL = 1, LEAF = 2 };

// If a node is a leafe it points to a range of points&normals or triangles
struct LeafInfo {
  uint32_t range_start;
  uint32_t range_end;
};

// A Node or the BVH8 tree
// Aligned to 128 byte cache lines
struct BVH8Node {
  AABB parent_aabb;               // 24 bytes
  uint32_t child_base;            // 4 bytes
  ChildType child_meta[8];        // 8 bytes
  AABB8BitApprox child_approx[8]; // 48 bytes (6*8)
  // 84 bytes so far => 44 left for tailor coefficients

  // tailor coefficients quantized to 11 bit
  // with a shared 8 bit exponent
  // first 8 bytes are reused for parent index
  // cofficients: zero order (3), first order (9), second order compressed (18)
  // total: 30
  uint8_t tailor_data[44];
  // total 128 bytes

  // During construction we use the tailor_data memory for a parent pointer
  // which is used to spread the aabb and tailor coefficients to the inner
  // nodes.
  __host__ __device__ auto get_parent_idx() const -> uint32_t;

  // Set the parent idx. Reuses tailor coefficient data. Only call this before
  // the tailor coefficients and aabb are computed for the nodes.
  __host__ __device__ void set_parent_idx(uint32_t idx);

  __host__ __device__ inline auto get_shared_scale_factor() const -> float;

  __device__ inline auto
  set_tailor_coefficients(const Vec3 &zero_order, const Mat3x3 &first_order,
                          const Tensor3_compressed &second_order);

  __host__ __device__ inline auto
  get_tailor_zero_order(const float shared_scale_factor) const -> Vec3_bf16;

  __host__ __device__ inline auto
  get_tailor_first_order(const float shared_scale_factor) const -> Mat3x3_bf16;

  __host__ __device__ inline auto
  get_tailor_second_order(const float shared_scale_factor) const
      -> Tensor3_bf16_compressed;
};
