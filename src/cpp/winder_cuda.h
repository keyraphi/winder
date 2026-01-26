#pragma once
#include "aabb.h"
#include "binary_node.h"
#include "bvh8.h"
#include "tailor_coefficients.h"
#include "geometry.h"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
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

enum class WinderMode : uint8_t { Point, Triangle };

struct BVH8View {
  const BVH8Node *nodes;
  const LeafPointers *leaf_pointers;
  const float *geometry;

  uint32_t node_count;
};

class WinderBackend {

public:
  static auto CreateFromMesh(const float *triangles, size_t triangle_count,
                             int device_id) -> std::unique_ptr<WinderBackend>;

  static auto CreateFromPoints(const float *points, const float *scaled_normals,
                               size_t point_count, int device_id)
      -> std::unique_ptr<WinderBackend>;

  static auto CreateForSolver(const float *points, size_t point_count,
                              int device_id) -> std::unique_ptr<WinderBackend>;

  [[nodiscard]] auto compute(const float *queries, size_t query_count) const
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

  void solve_for_normals(const float *extra_p, size_t n_extra,
                         const float *extra_wn, const float *pc_wn,
                         float alpha);

  // Used in factories
  void initialize_mesh_data(const float *triangles);
  void initialize_point_data(const float *points, const float *normals,
                             int device_id);

private:
  WinderMode m_mode;
  size_t m_count;
  int m_device;
  uint32_t m_bvh8_node_count;

  // Private constructor used in factories. Allocates vectors but doesn't fill
  // them yet
  WinderBackend(WinderMode mode, size_t size, int device_id);

  // provide a view on the BVH8 data for kernel launches
  auto get_bvh_view() -> BVH8View;

  // memory arena
  thrust::device_vector<uint8_t> m_memory_arena;

  /** * MEMORY ARENA POINTERS
   * All pointers below refer to segments within m_memory_arena.
   * N = Point Count, L = Leaf Count (N/32)
   */

  // --- Geometric Data & Permutation Maps ---
  uint32_t *m_to_internal;  // [N] Map: Original index -> Morton sorted index
  uint32_t *m_to_canonical; // [N] Map: Morton sorted index -> Original index
  float *m_sorted_geometry; // [N * floats_per_elem] Interleaved P and N (or
                            // Triangles)

  // --- Binary LBVH (Auxiliary Structure for BVH8 Construction) ---
  uint32_t *m_morton_codes; // [N] 30-bit Morton codes for individual points
  uint32_t
      *m_leaf_morton_codes; // [L] Morton code of the first point in each leaf
  BinaryNode
      *m_binary_nodes; // [L-1] Topology of the auxiliary binary radix tree
  uint32_t *m_binary_parents;  // [2L-1] Parent pointers for binary tree
                               // (bottom-up traversal)
  AABB *m_binary_aabbs;        // [2L-1] AABBs for all binary nodes/leaves
  uint32_t *m_atomic_counters; // [L-1] Counters for thread synchronization
                               // during AABB/M2M climb

  // --- BVH8 Tree Structure (Final Output) ---
  BVH8Node *m_bvh8_nodes; // [~0.2L] The 8-way wide-tree nodes (Quantized AABBs
                          // + Topology)
  TailorCoefficientsBf16 *m_leaf_coefficients; // [L] Taylor expansion terms for
                                               // leaf clusters (Bfloat16)

  // --- BVH8 Construction & M2M Support ---
  uint32_t *m_bvh8_leaf_parents; // [L] Map: Leaf index -> Parent BVH8Node index
  LeafPointers *m_bvh8_leaf_pointers; // [0.2L] Map: BVH8Node slot -> Leaf index
                                      // (for traversal)
  uint32_t *m_bvh8_internal_parent_map; // [0.2L] Map: BVH8Node -> Parent
                                        // BVH8Node index (for M2M climb)
  uint32_t *m_bvh8_work_queue_A; // [L-1] Double-buffer for level-by-level tree
                                 // conversion
  uint32_t *m_bvh8_work_queue_B; // [L-1] Double-buffer for level-by-level tree
                                 // conversion
  uint32_t *m_global_counter;    // [1] Atomic counter for work queue management
                                 // and node allocation

  // private helpers
  template <typename Geometry>
  auto initializeMortonCodes(const Geometry *geometry) -> void;
};
