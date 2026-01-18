#pragma once
#include "bvh8.h"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>
#include <type_traits>
#include <vector_types.h>

# define LEAF_MIN_SIZE 32;

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
  const LeafInfo *leaf_info;
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
  void initialize_point_data(const float *points, const float *normals);

private:
  WinderMode m_mode;
  size_t m_count;
  int m_device;

  // Private constructor used in factories. Allocates vectors but doesn't fill
  // them yet
  WinderBackend(WinderMode mode, size_t size, int device_id);

  // provide a view on the BVH8 data for kernel launches
  auto get_bvh_view() -> BVH8View;

  // 2. Permutation Maps for sorting and unsorting
  thrust::device_vector<uint32_t> m_to_internal;  // original -> sorted
  thrust::device_vector<uint32_t> m_to_canonical; // sorted -> original

  // sorted copy of the input data used by the bvh8
  thrust::device_vector<float> m_sorted_geometry;

  // 4. The Tree Structure
  thrust::device_vector<BVH8Node> m_nodes;
  thrust::device_vector<LeafInfo> m_leaf_info;

};
