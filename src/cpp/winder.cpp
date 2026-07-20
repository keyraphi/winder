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
               const float epsilon = -1.F, const bool is_brute_force = false,
               const size_t stream = 0) -> Scalar_t {
    size_t n = queries.shape(0);
    // todo ensure queries are on same device as m_impl

    CudaUniquePtr<float> raw_ptr_unique;
    if (is_brute_force) {
      if (is_backend_triangle) {
        raw_ptr_unique =
            m_impl_tri->brute_force(queries.data(), n, epsilon, stream);
      } else {
        raw_ptr_unique =
            m_impl_pn->brute_force(queries.data(), n, epsilon, stream);
      }
    } else {
      if (is_backend_triangle) {
        raw_ptr_unique =
            m_impl_tri->compute(queries.data(), n, beta, epsilon, stream);
      } else {
        raw_ptr_unique =
            m_impl_pn->compute(queries.data(), n, beta, epsilon, stream);
      }
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

  auto get_gradients(const Vec3_t &queries, const Scalar_t &grad_output,
                     const float beta = -1.F, const float epsilon = -1.F,
                     const bool is_brute_force = false, const size_t stream = 0)
      -> GradResult_t {
    // todo ensure queries are on same device as m_impl
    size_t n = queries.shape(0);

    CudaUniquePtr<float> raw_ptr_unique;
    if (is_brute_force) {
      if (is_backend_triangle) {
        raw_ptr_unique = m_impl_tri->grads_brute_force(
            queries.data(), grad_output.data(), n, epsilon, stream);
      } else {
        raw_ptr_unique = m_impl_pn->grads_brute_force(
            queries.data(), grad_output.data(), n, epsilon, stream);
      }
    } else {
      if (is_backend_triangle) {
        raw_ptr_unique = m_impl_tri->get_gradients(
            queries.data(), grad_output.data(), n, beta, epsilon, stream);
      } else {
        raw_ptr_unique = m_impl_pn->get_gradients(
            queries.data(), grad_output.data(), n, beta, epsilon, stream);
      }
    }

    float *raw_ptr = raw_ptr_unique.release();
    nb::capsule owner(raw_ptr,
                      [](void *p) noexcept { winder_cuda::cuda_free(p); });

    if (is_backend_triangle) {
      size_t n_triangles = m_impl_tri->point_count();
      return {raw_ptr, {n_triangles, 3, 3}, owner};
    }
    size_t n_points = m_impl_pn->point_count();
    return {raw_ptr, {n_points, 2, 3}, owner};
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

  m.def("brute_force_winding_numbers", brute_force_winding_numbers, "points"_a,
        "scaled_normals"_a, "queries"_a, "epsilon"_a = -1.F, "stream"_a = 0,
        nb::sig("def brute_force_winding_numbers(points: Array[N, 3; float32, "
                "cuda], scaled_normals: Array[N, 3; float32, cuda], queries: "
                "Array[M, 3; float32, cuda], epsilon: float32 = -1, "
                "stream: uint64_t = 0) -> "
                "Array[M; float32, cuda]"),
        R"doc(
                Computes the winding number at the given query locations on GPU
                with brute force in O(N*M).

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
      "brute_force_winding_numbers", brute_force_winding_numbers, "vertices"_a,
      "triangle_indices"_a, "queries"_a, "stream"_a = 0,
      nb::sig("def brute_force_winding_numbers(vertices: Array[K, 3; float32, "
              "cuda], triangle_indices: Array[N, 3; uint32, cuda], queries: "
              "Array[M, 3; float32, cuda], "
              "stream: uint64_t = 0) -> "
              "Array[M; float32, cuda]"),
      R"doc(
                Computes the winding number at the given query locations on GPU
                with brute force in O(N*M).

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
  m.def("brute_force_winding_numbers", brute_force_winding_numbers,
        "triangles"_a, "queries"_a, "stream"_a = 0,
        nb::sig("def brute_force_winding_numbers(triangles: Array[N, 3, 3; "
                "float32, cuda], queries: Array[M, 3; float32, cuda], stream: "
                "uint64_t = 0) -> Array[M; float32, cuda]"),
        R"doc(
                Computes the winding number at the given query locations on GPU
                with brute force in O(N*M).

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
      // --- Constructor ---
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
      .def("compute", &GradientEngine::compute, "grad_output"_a, "vertices"_a,
           "triangle_indices"_a, "beta"_a = -1.F, "stream"_a = 0,
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
      .def("compute", &GradientEngine::compute, "grad_output"_a, "triangles"_a,
           "beta"_a = -1.F, "stream"_a = 0,
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
      .def("compute", &GradientEngine::compute, "grad_output"_a, "points"_a,
           "scaled_normals"_a, "beta"_a = -1.F, "epsilon"_a = -1.F,
           "stream"_a = 0,
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
