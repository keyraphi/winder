#pragma once
#include "aabb.h"
#include "binary_node.h"
#include "bvh8.h"
#include "geometry.h"
#include "tailor_coefficients.h"
#include "vec3.h"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <memory>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/detail/par.h>
#include <vector_types.h>

// one warp per leaf
#define LEAF_SIZE 32
#define L2_ALIGN 128

// forward definitions

class ScopedCudaDevice {
private:
  int original_device_;

public:
  ScopedCudaDevice(int new_device) {
    cudaGetDevice(&original_device_);
    cudaSetDevice(new_device);
  }
  ~ScopedCudaDevice() { cudaSetDevice(original_device_); }
  // Disallow copying
  ScopedCudaDevice(const ScopedCudaDevice &) = delete;
  ScopedCudaDevice &operator=(const ScopedCudaDevice &) = delete;
};

struct CudaDeleter {
  void operator()(void *ptr) const;
};

template <typename T> using CudaUniquePtr = std::unique_ptr<T[], CudaDeleter>;

template <IsGeometry Geometry> class WinderBackend {

public:
  ~WinderBackend();

  static auto CreateFromMesh(const float *triangles, size_t triangle_count,
                             int device_id)
      -> std::unique_ptr<WinderBackend<Triangle>>;

  static auto CreateFromPoints(const float *points, const float *scaled_normals,
                               size_t point_count, int device_id)
      -> std::unique_ptr<WinderBackend<PointNormal>>;

  static auto CreateForSolver(const float *points, size_t point_count,
                              int device_id)
      -> std::unique_ptr<WinderBackend<PointNormal>>;

  auto compute(const float *queries, size_t query_count, float beta = -1,
               float epsilon = -1, size_t stream = 0) const
      -> CudaUniquePtr<float>;

  [[nodiscard]] auto get_normals() const -> CudaUniquePtr<float>;
  [[nodiscard]] auto grad_normals(const float *grad_output,
                                  size_t n_queries) const
      -> CudaUniquePtr<float>;
  [[nodiscard]] auto grad_points(const float *grad_output,
                                 size_t n_queries) const
      -> CudaUniquePtr<float>;

  [[nodiscard]] auto point_count() const -> size_t { return m_count; }
  [[nodiscard]] auto device_id() const -> int { return m_device; }

  void solve_for_normals(const float *extra_p, size_t extra_count,
                         const float *extra_wn, const float *pc_wn,
                         float alpha);

  // Used in factories
  void initialize_mesh_data(const float *triangles);
  void initialize_point_data(const float *points, const float *normals);

private:
  int m_device;
  cudaStream_t m_stream_0, m_stream_1;
  cudaEvent_t m_start_tree_construction_event;
  cudaEvent_t m_tree_construction_finished_event;
  thrust::cuda_cub::execute_on_stream m_stream_0_policy;
  thrust::cuda_cub::execute_on_stream m_stream_1_policy;

  // Private constructor used in factories. Allocates vectors but doesn't fill
  // them yet
  WinderBackend(size_t size, int device_id);

public: // TODO DEBUG
  size_t m_count;


  // --- Geometric Data & Permutation Maps ---
  uint32_t *m_to_internal;     // [N] Map: Original index -> Morton sorted index
  Geometry *m_sorted_geometry; // [N] Interleaved P and N (or Triangles)

  AABB *m_binary_aabbs;        // [2L-1] AABBs for all binary nodes/leaves
  uint32_t
      *m_bvh8_node_count; // [1] number of bvh8 nodes created during conversion

  // --- BVH8 Tree Structure (Final Output) ---
  BVH8Node *m_bvh8_nodes; // [~0.2L] The 8-way wide-tree nodes (Quantized AABBs
                          // + Topology)
  TailorCoefficientsBf16 *m_leaf_coefficients; // [L] Taylor expansion terms for
                                               // leaf clusters (Bfloat16)

  // --- BVH8 Construction & M2M Support ---
  LeafPointers *m_bvh8_leaf_pointers; // [0.2L] Map: BVH8Node slot -> Leaf index
                                      // (for traversal)

private: // TODO DEBUG
  // private helpers
  template <IsPrimitiveGeometry PrimitiveGeometry>
  auto initializeMortonCodes(const PrimitiveGeometry *geometry, uint32_t *geometry_morton_codes) -> void;
};
