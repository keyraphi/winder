#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "geometry.h"
#include "utils.h"
#include "winder_brute_force.h"
#include "winder_cuda.h"

namespace nb = nanobind;
using namespace nb::literals;

// Device-specific type aliases with exact constraints
using Vec3_t = nb::ndarray<nb::array_api, float, nb::shape<-1, 3>, nb::c_contig,
                           nb::device::cuda>;
using Triangle_t = nb::ndarray<nb::array_api, float, nb::shape<-1, 3, 3>,
                               nb::c_contig, nb::device::cuda>;
using TriangleIdx_t = nb::ndarray<nb::array_api, uint32_t, nb::shape<-1, 3>,
                                  nb::c_contig, nb::device::cuda>;
using Scalar_t = nb::ndarray<nb::array_api, float, nb::shape<-1>, nb::c_contig,
                             nb::device::cuda>;
using GradResult_t = nb::ndarray<nb::array_api, float, nb::shape<-1, -1, 3>,
                                 nb::c_contig, nb::device::cuda>;
namespace winder_cuda {

auto scalar_with_default(const std::optional<Scalar_t> &maybe_pc_wn,
                         size_t count, float default_value) -> Scalar_t {
  if (maybe_pc_wn.has_value()) {
    return maybe_pc_wn.value();
  }
  // Allocate raw memory via our decoupled utility
  auto *data = static_cast<float *>(cuda_allocate(count * sizeof(float)));
  thrust_fill_float(data, count, default_value);
  // Wrap the raw pointer in a Python capsule with a safe custom deleter
  nb::capsule owner(data, [](void *p) noexcept -> void { cuda_free(p); });

  return {data, {count}, owner};
}

} // namespace winder_cuda

struct AsyncCleanupGlue {
  float *ptr;
  CudaDeleter deleter;
};

auto brute_force_winding_numbers(const Vec3_t &points,
                                 const Vec3_t &scaled_normals,
                                 const Vec3_t &queries, float epsilon = -1,
                                 const uint64_t stream = 0) -> Scalar_t {
  if (points.device_id() != scaled_normals.device_id()) {
    throw std::runtime_error(
        "Points and Normals must be on the same CUDA device.");
  }
  if (points.shape(0) != scaled_normals.shape(0)) {
    throw std::runtime_error(
        "Shape of points must be equal to shape of normals.");
  }
  CudaUniquePtr<float> raw_ptr_unique = brute_force_point_normal_impl(
      points.data(), scaled_normals.data(), queries.data(), points.shape(0),
      queries.shape(0), epsilon, points.device_id(), stream);

  CudaDeleter deleter = raw_ptr_unique.get_deleter();
  float *raw_ptr = raw_ptr_unique.release();

  auto *glue = new AsyncCleanupGlue{raw_ptr, deleter};

  nb::capsule owner(glue, [](void *p) noexcept -> void {
    auto *g = static_cast<AsyncCleanupGlue *>(p);
    g->deleter(g->ptr);
    delete g;
  });
  return {raw_ptr, {queries.shape(0)}, owner};
}

auto brute_force_winding_numbers(const Vec3_t &vertices,
                                 const TriangleIdx_t &triangle_indices,
                                 const Vec3_t &queries,
                                 const uint64_t stream = 0) -> Scalar_t {
  if (vertices.device_id() != triangle_indices.device_id()) {
    throw std::runtime_error(
        "Vertices and triangle_indices must be on the same CUDA device.");
  }
  CudaUniquePtr<float> raw_ptr_unique = brute_force_mesh_impl(
      vertices.data(), triangle_indices.data(), queries.data(),
      triangle_indices.shape(0), vertices.shape(0), queries.shape(0),
      vertices.device_id(), stream);
  CudaDeleter deleter = raw_ptr_unique.get_deleter();
  float *raw_ptr = raw_ptr_unique.release();

  auto *glue = new AsyncCleanupGlue{raw_ptr, deleter};

  nb::capsule owner(glue, [](void *p) noexcept -> void {
    auto *g = static_cast<AsyncCleanupGlue *>(p);
    g->deleter(g->ptr);
    delete g;
  });
  return {raw_ptr, {queries.shape(0)}, owner};
}
auto brute_force_winding_numbers(const Triangle_t &triangles,
                                 const Vec3_t &queries,
                                 const uint64_t stream = 0) -> Scalar_t {
  CudaUniquePtr<float> raw_ptr_unique = brute_force_triangle_impl(
      triangles.data(), queries.data(), triangles.shape(0), queries.shape(0),
      triangles.device_id(), stream);
  CudaDeleter deleter = raw_ptr_unique.get_deleter();
  float *raw_ptr = raw_ptr_unique.release();

  auto *glue = new AsyncCleanupGlue{raw_ptr, deleter};

  nb::capsule owner(glue, [](void *p) noexcept -> void {
    auto *g = static_cast<AsyncCleanupGlue *>(p);
    g->deleter(g->ptr);
    delete g;
  });
  return {raw_ptr, {queries.shape(0)}, owner};
}

auto brute_force_gradients(const Scalar_t &grad_output, const Vec3_t &points,
                           const Vec3_t &scaled_normals, const Vec3_t &queries,
                           float epsilon, const uint64_t stream = 0)
    -> GradResult_t {
  if (points.device_id() != grad_output.device_id()) {
    throw std::runtime_error(
        "points and grad_output must be on the same CUDA device.");
  }
  if (points.device_id() != scaled_normals.device_id()) {
    throw std::runtime_error(
        "points and scaled_normals must be on the same CUDA device.");
  }
  if (points.device_id() != queries.device_id()) {
    throw std::runtime_error(
        "points and queries must be on the same CUDA device.");
  }
  if (points.shape(0) != scaled_normals.shape(0)) {
    throw std::runtime_error(
        "Shape of points must be equal to shape of normals.");
  }
  CudaUniquePtr<float> raw_ptr_unique = brute_force_point_normal_gradient_impl(
      grad_output.data(), points.data(), scaled_normals.data(), queries.data(),
      points.shape(0), queries.shape(0), epsilon, points.device_id(), stream);

  float *raw_ptr = raw_ptr_unique.release();
  nb::capsule owner(raw_ptr,
                    [](void *p) noexcept { winder_cuda::cuda_free(p); });

  return {raw_ptr, {points.shape(0), 2, 3}, owner};
}

auto brute_force_gradients(const Scalar_t &grad_output, const Vec3_t &vertices,
                           const TriangleIdx_t &triangle_indices,
                           const Vec3_t &queries, const uint64_t stream = 0)
    -> GradResult_t {
  if (vertices.device_id() != grad_output.device_id()) {
    throw std::runtime_error(
        "vertices and grad_output must be on the same CUDA device.");
  }
  if (vertices.device_id() != triangle_indices.device_id()) {
    throw std::runtime_error(
        "vertices and triangle_indices must be on the same CUDA device.");
  }
  if (vertices.device_id() != queries.device_id()) {
    throw std::runtime_error(
        "vertices and queries must be on the same CUDA device.");
  }
  CudaUniquePtr<float> raw_ptr_unique = brute_force_mesh_gradient_impl(
      grad_output.data(), vertices.data(), triangle_indices.data(),
      queries.data(), triangle_indices.shape(0), queries.shape(0),
      vertices.shape(0), vertices.device_id(), stream);

  float *raw_ptr = raw_ptr_unique.release();
  nb::capsule owner(raw_ptr,
                    [](void *p) noexcept { winder_cuda::cuda_free(p); });

  return {raw_ptr, {vertices.shape(0), 3, 3}, owner};
}

auto brute_force_gradients(const Scalar_t &grad_output,
                           const Triangle_t &triangles, const Vec3_t &queries,
                           const uint64_t stream = 0) -> GradResult_t {
  if (triangles.device_id() != grad_output.device_id()) {
    throw std::runtime_error(
        "triangles and grad_output must be on the same CUDA device.");
  }
  if (triangles.device_id() != queries.device_id()) {
    throw std::runtime_error(
        "triangles and queries must be on the same CUDA device.");
  }
  CudaUniquePtr<float> raw_ptr_unique = brute_force_triangle_gradient_impl(
      grad_output.data(), triangles.data(), queries.data(), triangles.shape(0),
      queries.shape(0), triangles.device_id(), stream);

  float *raw_ptr = raw_ptr_unique.release();
  nb::capsule owner(raw_ptr,
                    [](void *p) noexcept { winder_cuda::cuda_free(p); });

  return {raw_ptr, {triangles.shape(0), 3, 3}, owner};
}

class GradientEngine {
public:
  GradientEngine(const Vec3_t &queries) {
    throw std::runtime_error("GradientEngine not yet implemented");
  }

  auto compute(const Scalar_t &grad_output, const Vec3_t &points,
               const Vec3_t &scaled_normals, float beta = -1.F,
               float epsilon = -1.F, const uint64_t stream = 0) {
    throw std::runtime_error("GradientEngine not yet implemented");
  }
  auto compute(const Scalar_t &grad_output, const Vec3_t &vertices,
               const TriangleIdx_t &triangle_indices, float beta = -1.F,
               const uint64_t stream = 0) {
    throw std::runtime_error("GradientEngine not yet implemented");
  }
  auto compute(const Scalar_t &grad_output, const Triangle_t &triangles,
               float beta = -1.F, const uint64_t stream = 0) {
    throw std::runtime_error("GradientEngine not yet implemented");
  }
};

class WinderEngine {
public:
  // --- Triangle Mesh Constructor ---
  WinderEngine(const Triangle_t &triangles) {
    m_impl_tri = WinderBackend<Triangle>::CreateFromTriangles(
        triangles.data(), triangles.shape(0), triangles.device_id());
    is_backend_triangle = true;
  }

  WinderEngine(const Vec3_t &vertices, const TriangleIdx_t &triangle_indices) {
    if (vertices.device_id() != triangle_indices.device_id()) {
      throw std::runtime_error(
          "Vertices and triangle_indices must be on the same CUDA device.");
    }
    m_impl_tri = WinderBackend<Triangle>::CreateFromMesh(
        vertices.data(), vertices.shape(0), triangle_indices.data(),
        triangle_indices.shape(0), vertices.device_id());
    is_backend_triangle = true;
  }

  // --- Point Cloud Constructor ---
  WinderEngine(const Vec3_t &points, const Vec3_t &normals) {
    if (points.device_id() != normals.device_id()) {
      throw std::runtime_error(
          "Points and Normals must be on the same CUDA device.");
    }
    if (points.shape(0) != normals.shape(0)) {
      throw std::runtime_error(
          "Shape of points must be equal to shape of normals.");
    }

    m_impl_pn = WinderBackend<PointNormal>::CreateFromPoints(
        points.data(), normals.data(), points.shape(0), points.device_id());
    is_backend_triangle = false;
  }

  auto compute(const Vec3_t &queries, const float beta = -1.F,
               const float epsilon = -1.F, const size_t stream = 0)
      -> Scalar_t {
    size_t n = queries.shape(0);

    CudaUniquePtr<float> raw_ptr_unique;
    if (is_backend_triangle) {
      raw_ptr_unique =
          m_impl_tri->compute(queries.data(), n, beta, epsilon, stream);
    } else {
      raw_ptr_unique =
          m_impl_pn->compute(queries.data(), n, beta, epsilon, stream);
    }

    CudaDeleter deleter = raw_ptr_unique.get_deleter();
    float *raw_ptr = raw_ptr_unique.release();

    auto *glue = new AsyncCleanupGlue{raw_ptr, deleter};

    nb::capsule owner(glue, [](void *p) noexcept -> void {
      auto *g = static_cast<AsyncCleanupGlue *>(p);
      g->deleter(g->ptr);
      delete g;
    });
    return {raw_ptr, {n}, owner};
  }

  [[nodiscard]] auto dump() const -> std::string {
    std::string result;
    if (is_backend_triangle) {
      result += m_impl_tri->dump();
    } else {
      result += m_impl_pn->dump();
    }
    return result;
  }

private:
  // flag for used backend
  bool is_backend_triangle;
  std::unique_ptr<WinderBackend<PointNormal>> m_impl_pn;
  std::unique_ptr<WinderBackend<Triangle>> m_impl_tri;

  // Internal constructor used by factory methods
  explicit WinderEngine(std::unique_ptr<WinderBackend<PointNormal>> backend)
      : m_impl_pn(std::move(backend)) {}
  explicit WinderEngine(std::unique_ptr<WinderBackend<Triangle>> backend)
      : m_impl_tri(std::move(backend)) {}
};

NB_MODULE(winder_module, m) {
  m.doc() = R"doc(
        GPU-accelerated Differentiable Winding Number Field library.
        
        Compatible with any framework supporting DLPack / Array API 
        (PyTorch, JAX, CuPy, etc.). All inputs must reside on the same 
        CUDA device.
    )doc";

  m.def("brute_force_winding_numbers",
        nb::overload_cast<const Vec3_t &, const Vec3_t &, const Vec3_t &, float,
                          uint64_t>(&brute_force_winding_numbers),
        "points"_a, "scaled_normals"_a, "queries"_a, "epsilon"_a = -1.F,
        "stream"_a = 0,
        nb::sig("def brute_force_winding_numbers(points: Array[N, 3; float32, "
                "cuda], scaled_normals: Array[N, 3; float32, cuda], queries: "
                "Array[M, 3; float32, cuda], epsilon: float32 = -1, "
                "stream: uint64_t = 0) -> "
                "Array[M; float32, cuda]"),
        R"doc(
                Computes the winding number at the given query locations on GPU.

                NOTE: Brute Force implementation: exact, but in O(N*M)!

                Parameters
                ----------
                points : Array
                    A (N, 3) float32 CUDA array of point positions.
                scaled_normals : Array
                    A (N, 3) float32 CUDA array of scaled normals. 
                    The scaled normal direction is the orientation,
                    the scale the associated voronoi area.
                queries : Array
                    (M, 3) CUDA array of query points for which the winding number field is evaluated.
                epsilon : float, optional
                    Regularization scale (smoothing radius) used to prevent numerical 
                    singularities (NaN/infinity) when queries land near points in point clouds.
                    Only applies to point cloud backends.
                    - distance >= 2*epsilon: Acts as standard unregularized potential.
                    - distance < 2*epsilon: Smoothly dampens potential to a finite maximum.
                    Use any negative number to get the default value (1/250).
                stream  : int, optional
                    The raw 64-bit identifier (handle) of a CUDA stream. 
                    Allows enqueuing operations asynchronously within deep learning frameworks.
                    For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
                    Default is 0 (the default/null stream).

                Returns
                -------
                (M,) float32 CUDA array holding the winding numbers for the queries.
            )doc");
  m.def(
      "brute_force_winding_numbers",
      nb::overload_cast<const Vec3_t &, const TriangleIdx_t &, const Vec3_t &,
                        const uint64_t>(&brute_force_winding_numbers),
      "vertices"_a, "triangle_indices"_a, "queries"_a, "stream"_a = 0,
      nb::sig("def brute_force_winding_numbers(vertices: Array[K, 3; float32, "
              "cuda], triangle_indices: Array[N, 3; uint32, cuda], queries: "
              "Array[M, 3; float32, cuda], "
              "stream: uint64_t = 0) -> "
              "Array[M; float32, cuda]"),
      R"doc(
                Computes the winding number at the given query locations on GPU.

                NOTE: Brute Force implementation: exact, but in O(N*M)!

                Parameters
                ----------
                vertices : Array
                    A (K, 3) float32 CUDA array holding K vertices.
                triangle_indices: Array
                    A (N, 3) index array, where each row defines the three vertices of a triangle.
                    Note: Order the vertices counter-clockwise when seen from the front. 
                queries : Array
                    (M, 3) CUDA array of query points for which the winding number field is evaluated.
                stream  : int, optional
                    The raw 64-bit identifier (handle) of a CUDA stream. 
                    Allows enqueuing operations asynchronously within deep learning frameworks.
                    For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
                    Default is 0 (the default/null stream).

                Returns
                -------
                (M,) float32 CUDA array holding the winding numbers for the queries.
            )doc");
  m.def("brute_force_winding_numbers",
        nb::overload_cast<const Triangle_t &, const Vec3_t &, const uint64_t>(
            &brute_force_winding_numbers),
        "triangles"_a, "queries"_a, "stream"_a = 0,
        nb::sig("def brute_force_winding_numbers(triangles: Array[N, 3, 3; "
                "float32, cuda], queries: Array[M, 3; float32, cuda], stream: "
                "uint64_t = 0) -> Array[M; float32, cuda]"),
        R"doc(
                Computes the winding number at the given query locations on GPU.

                NOTE: Brute Force implementation: exact, but in O(N*M)!

                Parameters
                ----------
                triangles : Array
                    A (N, 3, 3) float32 CUDA array holding N triangles
                    which consists of 3 vertices. 
                    Note: Order the vertices counter-clockwise when seen from the front. 
                queries : Array
                    (M, 3) CUDA array of query points for which the winding number field is evaluated.
                stream  : int, optional
                    The raw 64-bit identifier (handle) of a CUDA stream. 
                    Allows enqueuing operations asynchronously within deep learning frameworks.
                    For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
                    Default is 0 (the default/null stream).

                Returns
                -------
                (M,) float32 CUDA array holding the winding numbers for the queries.
            )doc");

  m.def("brute_force_gradients",
        nb::overload_cast<const Scalar_t &, const Vec3_t &, const Vec3_t &,
                          const Vec3_t &, float, const uint64_t>(
            &brute_force_gradients),
        "grad_output"_a, "points"_a, "scaled_normals"_a, "queries"_a,
        "epsilon"_a = -1, "stream"_a = 0,
        nb::sig("def brute_force_gradients(grad_output: Array[N; float32, "
                "cuda], points: Array[N, 3; float32, cuda], scaled_normals: "
                "Array[N, 3; float32, cuda], queries: Array[M, 3; float32, "
                "cuda], epsilon: float, stream: uint64_t = 0) -> Array[N, "
                "2, 3; float32, cuda]"),
        R"doc(
                Compute the partial derivatives w.r.t. the given point
                positions and scaled normals.

                NOTE: Brute Force implementation: exact, but in O(N*M)!

                This method propagates the gradient of a scalar loss function with 
                respect to the computed winding numbers back to the geometry.


                Parameters
                ----------
                grad_output : Array
                    (M,) float32 CUDA array representing the gradient of the loss 
                    with respect to the winding numbers at the query location for which 
                    the GradientEngine was built (dL/dw).
                points  : Array
                    (N, 3) float32 CUDA array representing the point positions for which the
                    winding numbers and loss were computed.
                scaled_normals : Array
                    (N, 3) float32 CUDA array representing the area-scaled normals for which the
                    winding numbers and loss were computed.
                queries : Array
                    (M, 3) float32 CUDA array with the queries for which the loss was evaluated.
                    You have to make sure that the grad_output is aligned with those queries.
                epsilon : float, optional
                    Regularization scale (smoothing radius) used to prevent numerical 
                    singularities (NaN/infinity) when queries land near points in point clouds.
                    - distance >= 2*epsilon: Acts as standard unregularized potential.
                    - distance < 2*epsilon: Smoothly dampens potential to a finite maximum.
                    Use any negative number to get the default value (1/250).
                stream : int, optional
                    The raw 64-bit identifier (handle) of a CUDA stream. 
                    Allows enqueuing operations asynchronously within deep learning frameworks.
                    For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
                    Default is 0 (the default/null stream).

                Returns
                -------
                  Array (N, 2, 3) float32 CUDA array
                    output[i, 0] represents the gradient of the loss 
                      with respect to the input scaled normals at index i (dL/dn). 
                      Calculated as: dL/dn = (dL/dw)^T * (dw/dn).
                    output[i, 1] represents the gradient of the loss 
                      with respect to the source positions at index i (dL/dp)
                      Calculated as: dL/dp = (dL/dw)^T * (dw/dp).
            )doc");

  m.def(
      "brute_force_gradients",
      nb::overload_cast<const Scalar_t &, const Vec3_t &, const TriangleIdx_t &,
                        const Vec3_t &, const uint64_t>(&brute_force_gradients),
      "grad_output"_a, "vertices"_a, "triangle_indices"_a, "queries"_a,
      "stream"_a = 0,
      nb::sig("def brute_force_gradients(grad_output: Array[N; float32, "
              "cuda], vertices: Array[K, 3; float32, cuda], triangle_indices: "
              "Array[N, 3; uint32_t, cuda], queries: Array[M, 3; float32, "
              "cuda], epsilon: float, stream: uint64_t = 0) -> Array[K, "
              "2, 3; float32, cuda]"),
      R"doc(
                Compute the partial derivatives w.r.t. the given triangles vertex positions.

                NOTE: Brute Force implementation: exact, but in O(N*M)!

                This method propagates the gradient of a scalar loss function with 
                respect to the computed winding numbers back to the geometry.

                Parameters
                ----------
                grad_output : Array
                    (M,) float32 CUDA array representing the gradient of the loss 
                    with respect to the winding numbers at the query location for which 
                    the GradientEngine was built (dL/dw).
                    NOTE: You have to make sure that the queries used to compute L are the ones
                          used to build the GradientEngine!
                vertices  : Array
                    (K, 3) float32 CUDA array representing the shared vertices used in the triangles
                    for which the winding numbers and loss were computed.
                triangle_indices: Array
                    A (N, 3) index array, where each row defines the three vertices of a triangle.
                    Note: Order the vertices counter-clockwise when seen from the front. 
                queries : Array
                    (M, 3) float32 CUDA array with the queries for which the loss was evaluated.
                    You have to make sure that the grad_output is aligned with those queries.
                stream : int, optional
                    The raw 64-bit identifier (handle) of a CUDA stream. 
                    Allows enqueuing operations asynchronously within deep learning frameworks.
                    For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
                    Default is 0 (the default/null stream).

                Returns
                -------
                  Array (K, 3) float32 CUDA array
                    output[i] represents the gradient of the loss 
                      with respect to the vertex i (dL/dv_i)
                      Calculated as: dL/dv_i = (dL/dw)^T * (dw/dv_i).
            )doc");

  m.def("brute_force_gradients",
        nb::overload_cast<const Scalar_t &, const Triangle_t &, const Vec3_t &,
                          const uint64_t>(&brute_force_gradients),
        "grad_output"_a, "triangles"_a, "queries"_a, "stream"_a = 0,
        nb::sig("def brute_force_gradients(grad_output: Array[N; float32, "
                "cuda], triangles: Array[N, 3, 3; uint32_t, cuda], queries: "
                "Array[M, 3; float32, cuda], epsilon: float, stream: uint64_t "
                "= 0) -> Array[N, 2, 3; float32, cuda]"),
        R"doc(
                Compute the partial derivatives w.r.t. the given triangles vertex positions.

                NOTE: Brute Force implementation: exact, but in O(N*M)!

                This method propagates the gradient of a scalar loss function with 
                respect to the computed winding numbers back to the geometry.

                Parameters
                ----------
                grad_output : Array
                    (M,) float32 CUDA array representing the gradient of the loss 
                    with respect to the winding numbers at the query location for which 
                    the GradientEngine was built (dL/dw).
                    NOTE: You have to make sure that the queries used to compute L are the ones
                          used to build the GradientEngine!
                triangles  : Array
                    (N, 3, 3) float32 CUDA array representing the triangles used 
                    to evaluate the winding numbers for which the winding numbers and loss was computed.
                queries : Array
                    (M, 3) float32 CUDA array with the queries for which the loss was evaluated.
                    You have to make sure that the grad_output is aligned with those queries.
                stream : int, optional
                    The raw 64-bit identifier (handle) of a CUDA stream. 
                    Allows enqueuing operations asynchronously within deep learning frameworks.
                    For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
                    Default is 0 (the default/null stream).

                Returns
                -------
                  Array (N, 3, 3) float32 CUDA array
                    output[i, j] represents the gradient of the loss 
                      with respect to the vertex j of triangle i  (dL/dv_j)
                      Calculated as: dL/dv_j = (dL/dw)^T * (dw/dv_j).
            )doc");

  nb::class_<WinderEngine>(m, "WindingNumberEngine")
      // --- Triangle Mesh Constructor ---
      .def(nb::init<Triangle_t>(), "triangles"_a,
           nb::sig("def __init__(self, triangles: Array[N, 3, 3; float32, "
                   "cuda]) -> None"),
           R"doc(
                Initialize the engine for fast winding number calculation using a triangle soup.
                
                Parameters
                ----------
                triangles : Array
                    A (N, 3, 3) float32 CUDA array holding N triangles
                    which consists of 3 vertices. 
                    Note: Order the vertices counter-clockwise when seen from the front. 
            )doc")

      .def(nb::init<Vec3_t, TriangleIdx_t>(), "vertices"_a,
           "triangle_indices"_a,
           nb::sig("def __init__(self, vertices: Array[K, 3; float32, cuda], "
                   "triangle_indices: Array[N, 3; uint32, cuda]) -> None"),
           R"doc(
                Initialize the engine for fast winding number calculation using a triangle soup with N Triangles using K shared vertices.
                
                Parameters
                ----------
                vertices : Array
                    A (K, 3) float32 CUDA array holding K vertices.
                triangle_indices: Array
                    A (N, 3) index array, where each row defines the three vertices of a triangle.
                    Note: Order the vertices counter-clockwise when seen from the front. 
            )doc")

      // --- Point Cloud Constructor ---
      .def(nb::init<Vec3_t, Vec3_t>(), "points"_a, "scaled_normals"_a,
           nb::sig("def __init__(self, points: Array[N, 3; float32, cuda], "
                   "scaled_normals: Array[N, 3; float32, cuda]) -> None"),
           R"doc(
                Initialize the engine for fast winding number calculation using a point cloud with scaled normals.
                
                Parameters
                ----------
                points : Array
                    A (N, 3) float32 CUDA array of point positions.
                scaled_normals : Array
                    A (N, 3) float32 CUDA array of scaled normals. 
                    The scaled normal direction is the orientation,
                    the scale the associated voronoi area.
            )doc")

      // --- Inference ---
      .def("compute", &WinderEngine::compute, "queries"_a, "beta"_a = -1.F,
           "epsilon"_a = -1.F, "stream"_a = 0,
           nb::sig(
               "def compute(self, queries: Array[M, 3; float32, cuda], beta: "
               "float32 = -1, epsilon: float32 = -1, stream: uint64_t = 0) -> "
               "Array[M; float32, cuda]"),
           R"doc(
                Computes the winding number at the given query locations.

                Parameters
                ----------
                queries : Array
                    (M, 3) CUDA array of query points for which the winding number field is evaluated.
                beta    : float
                    Scalar that controls the degree of approximation.
                    Larger beta leads to more precise results, but also slower execution.
                    Use any negative number to get default values.
                    Default for point clouds is 2.0.
                    Default for triangles is 2.3
                epsilon : float, optional
                    Regularization scale (smoothing radius) used to prevent numerical 
                    singularities (NaN/infinity) when queries land near points in point clouds.
                    Only applies to point cloud fields and is ignored for triangle based fields.
                    - distance >= 2*epsilon: Acts as standard unregularized potential.
                    - distance < 2*epsilon: Smoothly dampens potential to a finite maximum.
                    Use any negative number to get the default value (1/250).
                stream  : int, optional
                    The raw 64-bit identifier (handle) of a CUDA stream. 
                    Allows enqueuing operations asynchronously within deep learning frameworks.
                    For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
                    Default is 0 (the default/null stream).

                Returns
                -------
                (M,) float32 CUDA array holding the winding numbers for the queries.
            )doc")
      // --- Utilities ---
      .def("dump", &WinderEngine::dump, nb::sig("def dump(self) -> str"),
           R"doc(
            Returns a detailed string representation of the internal BVH8 Tree.
          )doc");

  nb::class_<GradientEngine>(m, "GradientEngine")
      .def(nb::init<Vec3_t>(), "queries"_a,
           nb::sig("def __init__(self, queries: Array[M, 3; float32, "
                   "cuda]) -> None"),
           R"doc(
                Initialize the engine for fast gradient calculation from the given queries.
                
                Parameters
                ----------
                queries : Array
                    A (M, 3) float32 CUDA array holding M query positions (xyz).
                    The gradient is computed for a loss computed for the winding numbers
                    at those query locations.
            )doc")
      // --- Gradients ---
      .def("compute",
           nb::overload_cast<const Scalar_t &, const Vec3_t &,
                             const TriangleIdx_t &, float, const uint64_t>(
               &GradientEngine::compute),
           "grad_output"_a, "vertices"_a, "triangle_indices"_a, "beta"_a = -1.F,
           "stream"_a = 0,
           nb::sig("def compute(self, grad_output: Array[M; "
                   "float32, cuda], vertices: Array[K, 3; float32, cuda], "
                   "triangle_indices: Array[N, 3; uint32, cuda], "
                   "beta: float32 = -1, stream: uint64_t = 0) "
                   "-> Array[K, 3; float32, cuda]"),
           R"doc(
                Compute the partial derivatives w.r.t. the given triangles vertex positions.


                This method propagates the gradient of a scalar loss function with 
                respect to the computed winding numbers back to the geometry.

                Parameters
                ----------
                grad_output : Array
                    (M,) float32 CUDA array representing the gradient of the loss 
                    with respect to the winding numbers at the query location for which 
                    the GradientEngine was built (dL/dw).
                    NOTE: You have to make sure that the queries used to compute L are the ones
                          used to build the GradientEngine!
                vertices  : Array
                    (K, 3) float32 CUDA array representing the shared vertices used in the triangles
                    for which the winding numbers and loss were computed.
                triangle_indices: Array
                    A (N, 3) index array, where each row defines the three vertices of a triangle.
                    Note: Order the vertices counter-clockwise when seen from the front. 
                beta    : float
                    Scalar that controls the degree of approximation.
                    Larger beta leads to more precise results, but also slower execution.
                    Use any negative number to get default values.
                    Default for triangles is 2.3 TODO EXPERIMENT
                stream : int, optional
                    The raw 64-bit identifier (handle) of a CUDA stream. 
                    Allows enqueuing operations asynchronously within deep learning frameworks.
                    For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
                    Default is 0 (the default/null stream).

                Returns
                -------
                  Array (K, 3) float32 CUDA array
                    output[i] represents the gradient of the loss 
                      with respect to the vertex i (dL/dv_i)
                      Calculated as: dL/dv_i = (dL/dw)^T * (dw/dv_i).
            )doc")
      .def("compute",
           nb::overload_cast<const Scalar_t &, const Triangle_t &, float,
                             const uint64_t>(&GradientEngine::compute),
           "grad_output"_a, "triangles"_a, "beta"_a = -1.F, "stream"_a = 0,
           nb::sig("def compute(self, grad_output: Array[M; "
                   "float32, cuda], triangles: Array[N, 3, 3; float32, cuda], "
                   "beta: float32 = -1, stream: uint64_t = 0) "
                   "-> Array[N, 3, 3; float32, cuda]"),
           R"doc(
                Compute the partial derivatives w.r.t. the given triangles vertex positions.


                This method propagates the gradient of a scalar loss function with 
                respect to the computed winding numbers back to the geometry.

                Parameters
                ----------
                grad_output : Array
                    (M,) float32 CUDA array representing the gradient of the loss 
                    with respect to the winding numbers at the query location for which 
                    the GradientEngine was built (dL/dw).
                    NOTE: You have to make sure that the queries used to compute L are the ones
                          used to build the GradientEngine!
                triangles  : Array
                    (N, 3, 3) float32 CUDA array representing the triangles used 
                    to evaluate the winding numbers for which the winding numbers and loss was computed.
                beta    : float
                    Scalar that controls the degree of approximation.
                    Larger beta leads to more precise results, but also slower execution.
                    Use any negative number to get default values.
                    Default for triangles is 2.3 TODO EXPERIMENT
                stream : int, optional
                    The raw 64-bit identifier (handle) of a CUDA stream. 
                    Allows enqueuing operations asynchronously within deep learning frameworks.
                    For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
                    Default is 0 (the default/null stream).

                Returns
                -------
                  Array (N, 3, 3) float32 CUDA array
                    output[i, j] represents the gradient of the loss 
                      with respect to the vertex j of triangle i  (dL/dv_j)
                      Calculated as: dL/dv_j = (dL/dw)^T * (dw/dv_j).
            )doc")
      .def("compute",
           nb::overload_cast<const Scalar_t &, const Vec3_t &, const Vec3_t &,
                             float, float, const uint64_t>(
               &GradientEngine::compute),
           "grad_output"_a, "points"_a, "scaled_normals"_a, "beta"_a = -1.F,
           "epsilon"_a = -1.F, "stream"_a = 0,
           nb::sig("def compute(self, grad_output: Array[M; "
                   "float32, cuda], points: Array[N, 3; float32, cuda], "
                   "scaled_normals: Array[N, 3; float32, cuda], "
                   "beta: float32 = -1, epsilon: float32 = -1, stream: "
                   "uint64_t = 0) "
                   "-> Array[N, 2, 3; float32, cuda]"),
           R"doc(
                Compute the partial derivatives w.r.t. the given point positions and scaled normals.


                This method propagates the gradient of a scalar loss function with 
                respect to the computed winding numbers back to the geometry.


                Parameters
                ----------
                grad_output : Array
                    (M,) float32 CUDA array representing the gradient of the loss 
                    with respect to the winding numbers at the query location for which 
                    the GradientEngine was built (dL/dw).
                    NOTE: You have to make sure that the queries used to compute L are the ones
                          used to build the GradientEngine!
                points  : Array
                    (N, 3) float32 CUDA array representing the point positions for which the
                    winding numbers and loss were computed.
                scaled_normals : Array
                    (N, 3) float32 CUDA array representing the area-scaled normals for which the
                    winding numbers and loss were computed.
                beta    : float
                    Scalar that controls the degree of approximation.
                    Larger beta leads to more precise results, but also slower execution.
                    Use any negative number to get default values.
                    Default for point clouds is 2.0. TODO EXPERIMENT
                epsilon : float, optional
                    Regularization scale (smoothing radius) used to prevent numerical 
                    singularities (NaN/infinity) when queries land near points in point clouds.
                    - distance >= 2*epsilon: Acts as standard unregularized potential.
                    - distance < 2*epsilon: Smoothly dampens potential to a finite maximum.
                    Use any negative number to get the default value (1/250).
                stream : int, optional
                    The raw 64-bit identifier (handle) of a CUDA stream. 
                    Allows enqueuing operations asynchronously within deep learning frameworks.
                    For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
                    Default is 0 (the default/null stream).

                Returns
                -------
                  Array (N, 2, 3) float32 CUDA array
                    output[i, 0] represents the gradient of the loss 
                      with respect to the input scaled normals at index i (dL/dn). 
                      Calculated as: dL/dn = (dL/dw)^T * (dw/dn).
                    output[i, 1] represents the gradient of the loss 
                      with respect to the source positions at index i (dL/dp)
                      Calculated as: dL/dp = (dL/dw)^T * (dw/dp).
            )doc");
}
