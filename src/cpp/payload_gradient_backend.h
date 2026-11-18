#pragma once

#include "aabb.h"
#include "bvh8.h"
#include "geometry.h"
#include "scene_normalization.h"
#include "soa.h"
#include "taylor_coefficients.h"
#include "vec3.h"
#include <cstddef>
#include <cstdint>
#include <driver_types.h>
#include <string>
#include <type_traits>

#define LEAF_SIZE 32


template <IsGeometry Geometry, uint32_t D> class PayloadGradientBackend {
public:
  PayloadGradientBackend(size_t query_count, uint32_t payload_dim_user,
                         int device_id, uint64_t stream = 0);
  ~PayloadGradientBackend();

  PayloadGradientBackend(const PayloadGradientBackend &) = delete;
  auto operator=(const PayloadGradientBackend &)
      -> PayloadGradientBackend & = delete;

  // Build the query BVH and per-node D-channel payload coefficients.
  //   `queries`:     (M, 3)     f32 CUDA
  //   `grad_output`: (M, D_user) f32 CUDA
  void init(const float *queries, const float *grad_output);

    auto compute(const Vec3 *points, const Vec3 *scaled_normals,
               const float *payload, size_t geometry_count, float *gradients,
               float *epsilon_gradient, float beta, float epsilon,
               uint64_t stream, bool use_kahan) const -> void
    requires std::is_same_v<Geometry, PointNormal>;

  auto compute(const Triangle *triangles, const float *payload,
               size_t geometry_count, float *gradients,
               float *epsilon_gradient, float beta, float epsilon,
               uint64_t stream, bool use_kahan) const -> void
    requires std::is_same_v<Geometry, Triangle>;

  auto compute(const float *vertices, const uint32_t *triangle_indices,
               const float *payload, size_t vertex_count,
               size_t triangle_count, float *vertex_gradients,
               float *epsilon_gradient, float beta, float epsilon,
               uint64_t stream, bool use_kahan) const -> void
      requires std::is_same_v<Geometry, Triangle>;

  [[nodiscard]] auto dump() const -> std::string;
  [[nodiscard]] auto dimension() const -> uint32_t { return D; }

private:
  int m_device;
  size_t m_query_count;
  uint32_t m_payload_dim_user;
  cudaStream_t m_build_stream;
  SceneNormalization m_norm;

  cudaEvent_t m_tree_construction_finished_event = nullptr;

  // Persistent query-side arrays
  uint32_t *m_to_internal = nullptr;
  float *m_sorted_queries = nullptr;
  float *m_sorted_grad_outputs_bucket = nullptr; // (M, D) bucket stride

  // BVH8
  AABB *m_binary_aabbs = nullptr;
  uint32_t *m_bvh8_node_count = nullptr;
  BVH8Node *m_bvh8_nodes = nullptr;
  LeafPointers *m_bvh8_leaf_pointers = nullptr;

  // Per-node D-channel backward Taylor coefficients
  BackwardPayloadTaylorCoefficients<D> *m_taylor_coefficients = nullptr;
};
