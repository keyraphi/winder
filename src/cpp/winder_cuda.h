#pragma once
#include "aabb.h"
#include "binary_node.h"
#include "bvh8.h"
#include "geometry.h"
#include "tailor_coefficients.h"
#include "utils.h"
#include "vec3.h"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <memory>
#include <string>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/execution_policy.h>
#include <vector_types.h>

// one warp per leaf
#define LEAF_SIZE 32
#define L2_ALIGN 128


template <IsGeometry Geometry> class WindingNumbersBackend {

public:
  ~WindingNumbersBackend();

  static auto CreateFromTriangles(const float *triangles, size_t triangle_count,
                                  int device_id)
      -> std::unique_ptr<WindingNumbersBackend<Triangle>>;

  static auto CreateFromMesh(const float *vertices, size_t vertex_count,
                             const uint32_t *triangle_indices,
                             size_t triangle_count, int device_id)
      -> std::unique_ptr<WindingNumbersBackend<Triangle>>;

  static auto CreateFromPoints(const float *points, const float *scaled_normals,
                               size_t point_count, int device_id)
      -> std::unique_ptr<WindingNumbersBackend<PointNormal>>;

  auto compute(const float *queries, size_t query_count, float beta = -1.F,
               float epsilon = -1.F, size_t stream = 0) const
      -> CudaUniquePtr<float>;

  [[nodiscard]] auto point_count() const -> size_t { return m_count; }
  [[nodiscard]] auto device_id() const -> int { return m_device; }

  [[nodiscard]] auto dump() const -> std::string;

  // Used in factories
  void initialize_triangle_data(const float *triangles);
  void initialize_point_data(const float *points, const float *normals);

private:
  int m_device;
  cudaStream_t m_build_stream;
  cudaEvent_t m_start_tree_construction_event;
  cudaEvent_t m_tree_construction_finished_event;
  thrust::cuda_cub::execute_on_stream m_build_stream_policy;

  // Private constructor used in factories. Allocates vectors but doesn't fill
  // them yet
  WindingNumbersBackend(size_t size, int device_id);

public: // TODO DEBUG  make private!
  const size_t m_count;

  // --- Geometric Data & Permutation Maps ---
  uint32_t *m_to_internal;  // [N] Map: Original index -> Morton sorted index
  float *m_sorted_geometry; // [N] Interleaved P and N (or Triangles)

  AABB *m_binary_aabbs; // [2L-1] AABBs for all binary nodes/leaves
  uint32_t
      *m_bvh8_node_count; // [1] number of bvh8 nodes created during conversion

  // --- BVH8 Tree Structure (Final Output) ---
  BVH8Node *m_bvh8_nodes; // [~0.2L] The 8-way wide-tree nodes (Quantized AABBs
                          // + Topology)
  TailorCoefficientsF16
      *m_tailor_coefficients; // Tailor expansion terms for nodes
  TailorCoefficientsF16 *m_leaf_coefficients; // [L] Taylor expansion terms for
                                              // leaf clusters (half)

  // --- BVH8 Construction & M2M Support ---
  LeafPointers *m_bvh8_leaf_pointers; // [0.2L] Map: BVH8Node slot -> Leaf index
                                      // (for traversal)

private: // TODO DEBUG
  // private helpers
  template <IsPrimitiveGeometry PrimitiveGeometry>
  auto initializeMortonCodes(const PrimitiveGeometry *geometry,
                             uint64_t *geometry_morton_codes) -> void;
};
