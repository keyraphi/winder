#pragma once
#include "aabb.h"
#include "bvh8.h"
#include "geometry.h"
#include "scene_normalization.h"
#include "taylor_coefficients.h"
#include "thrust/system/cuda/detail/execution_policy.h"
#include "utils.h"
#include "vec3.h"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <memory>
#include <string>
#include <thrust/execution_policy.h>

// #include "kernels/traversal_payload.cuh"
#define LEAF_SIZE 32

template <IsGeometry Geometry, uint32_t D> class PayloadWindingNumbersBackend {

public:
  ~PayloadWindingNumbersBackend();

  // --- Factories ---------------------------------------------------------
  // payload is (N, D) row-major float32, already padded to the bucket size.
  // D=1 is guaranteed bit-identical to WindingNumbersBackend<Geometry>.
  static auto CreateFromTriangles(const float *triangles, const float *payload,
                                  size_t triangle_count, uint32_t payload_user_dim, int device_id,
                                  uint64_t stream)
      -> std::unique_ptr<PayloadWindingNumbersBackend<Triangle, D>>
    requires std::is_same_v<Geometry, Triangle>;

  static auto CreateFromMesh(const float *vertices, size_t vertex_count,
                             const uint32_t *triangle_indices,
                             const float *payload, size_t triangle_count,
                             uint32_t payload_user_dim,
                             int device_id, uint64_t stream)
      -> std::unique_ptr<PayloadWindingNumbersBackend<Triangle, D>>
    requires std::is_same_v<Geometry, Triangle>;

  static auto CreateFromPoints(const float *points, const float *scaled_normals,
                               const float *payload, size_t point_count,
                               uint32_t payload_user_dim,
                               int device_id, uint64_t stream)
      -> std::unique_ptr<PayloadWindingNumbersBackend<PointNormal, D>>
    requires std::is_same_v<Geometry, PointNormal>;

  // ---- Accessors ----
  [[nodiscard]] auto payload_dim_user() const -> uint32_t {
    return m_payload_dim_user;
  }
  [[nodiscard]] auto payload_dim_bucket() const -> uint32_t { return D; }

  // --- Inference ---------------------------------------------------------
  auto compute_payload(const float *queries, size_t query_count,
                       float *out_payload, float beta = -1.F,
                       float epsilon = 0.004F, size_t stream = 0, bool use_kahan = false) const -> void;

  auto compute_payload_query_gradients(const float *grad_output,
                                       const float *queries, size_t query_count,
                                       float *out_query_gradients,
                                       float beta = -1.F,
                                       float epsilon = 0.004F,
                                       size_t stream = 0, bool use_kahan = false) const -> void;

  [[nodiscard]] auto point_count() const -> size_t { return m_count; }
  [[nodiscard]] auto device_id() const -> int { return m_device; }

  // Prints component d=0 only — full D-slice dumps would be huge for large D.
  [[nodiscard]] auto dump() const -> std::string;

  // Used in factories
  void initialize_triangle_data(const float *triangles, const float *payload)
    requires std::is_same_v<Geometry, Triangle>;
  void initialize_point_data(const float *points, const float *normals,
                             const float *payload)
    requires std::is_same_v<Geometry, PointNormal>;

private:
  int m_device;
  uint32_t m_payload_dim_user;
  cudaStream_t m_build_stream;
  cudaEvent_t m_start_tree_construction_event;
  cudaEvent_t m_tree_construction_finished_event;
  thrust::cuda_cub::execute_on_stream m_build_stream_policy;

  PayloadWindingNumbersBackend(size_t size, int device_id, uint64_t stream);

  SceneParams m_scene_params;
  SceneNormalization m_norm = SceneNormalization::identity();
  const size_t m_count;

  // --- Geometry + payload permutation --------------------------------
  uint32_t *m_to_internal;  // [N]
  float *m_sorted_geometry; // [N] interleaved
  float *m_sorted_payload;  // [N, D] — needed for leaf eval and query-grad

  // --- BVH8 structure ------------------------------------------------
  AABB *m_binary_aabbs;        // [2L-1]
  uint32_t *m_bvh8_node_count; // [1]
  BVH8Node *m_bvh8_nodes;      // [~0.2L]
  LeafPointers *m_bvh8_leaf_pointers;

  // --- Payload coefficient arrays ------------------------------------
  PayloadTaylorCoefficientsF16<D> *m_tailor_coefficients; // [~0.2L]
  PayloadTaylorCoefficientsF16<D> *m_leaf_coefficients;   // [L]
};

// Explicit instantiation declarations — one per (geometry, bucket) pair
extern template class PayloadWindingNumbersBackend<PointNormal, 1>;
extern template class PayloadWindingNumbersBackend<PointNormal, 2>;
extern template class PayloadWindingNumbersBackend<PointNormal, 3>;
extern template class PayloadWindingNumbersBackend<PointNormal, 4>;
extern template class PayloadWindingNumbersBackend<PointNormal, 8>;
extern template class PayloadWindingNumbersBackend<PointNormal, 16>;
extern template class PayloadWindingNumbersBackend<PointNormal, 32>;
extern template class PayloadWindingNumbersBackend<PointNormal, 64>;
extern template class PayloadWindingNumbersBackend<PointNormal, 128>;

extern template class PayloadWindingNumbersBackend<Triangle, 1>;
// ... same for {2, 3, 4, 8, 16, 32, 64, 128}
