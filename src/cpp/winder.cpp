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
#include "gradient_backend.h"
#include "utils.h"
#include "winder_brute_force.h"
#include "winding_numbers_backend.h"

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
using VertexGradResult_t = nb::ndarray<nb::array_api, float, nb::shape<-1, 3>,
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

auto brute_force_winding_numbers(const Vec3_t &points,
                                 const Vec3_t &scaled_normals,
                                 const Vec3_t &queries,
                                 Scalar_t &out_winding_numbers,
                                 float epsilon = 0.004,
                                 const uint64_t stream = 0) -> void {
  if (points.device_id() != scaled_normals.device_id()) {
    throw std::runtime_error(
        "Points and Normals must be on the same CUDA device.");
  }
  if (points.device_id() != out_winding_numbers.device_id()) {
    throw std::runtime_error(
        "Points and out_winding_numbers must be on the same CUDA device.");
  }
  if (points.shape(0) != scaled_normals.shape(0)) {
    throw std::runtime_error(
        "Shape of points must be equal to shape of normals.");
  }
  if (queries.shape(0) != out_winding_numbers.shape(0)) {
    throw std::runtime_error(
        "Query count must be equal to out_winding_numbers count (shape[0]).");
  }
  brute_force_point_normal_impl(points.data(), scaled_normals.data(),
                                queries.data(), points.shape(0),
                                queries.shape(0), out_winding_numbers.data(),
                                epsilon, points.device_id(), stream);
}

auto brute_force_winding_numbers(const Vec3_t &vertices,
                                 const TriangleIdx_t &triangle_indices,
                                 const Vec3_t &queries,
                                 Scalar_t &out_winding_numbers,
                                 float epsilon = 0.004F,
                                 const uint64_t stream = 0) -> void {
  if (vertices.device_id() != triangle_indices.device_id()) {
    throw std::runtime_error(
        "Vertices and triangle_indices must be on the same CUDA device.");
  }
  if (vertices.device_id() != out_winding_numbers.device_id()) {
    throw std::runtime_error(
        "Vertices and out_winding_numbers must be on the same CUDA device.");
  }
  if (queries.shape(0) != out_winding_numbers.shape(0)) {
    throw std::runtime_error(
        "Query count must be equal to out_winding_numbers count (shape[0]).");
  }
  brute_force_mesh_impl(
      vertices.data(), triangle_indices.data(), queries.data(),
      triangle_indices.shape(0), vertices.shape(0), queries.shape(0),
      out_winding_numbers.data(), epsilon, vertices.device_id(), stream);
}

auto brute_force_winding_numbers(const Triangle_t &triangles,
                                 const Vec3_t &queries,
                                 Scalar_t &out_winding_numbers,
                                 float epsilon = 0.004F,
                                 const uint64_t stream = 0) -> void {
  if (queries.shape(0) != out_winding_numbers.shape(0)) {
    throw std::runtime_error(
        "Query count must be equal to out_winding_numbers count (shape[0]).");
  }
  brute_force_triangle_impl(
      triangles.data(), queries.data(), triangles.shape(0), queries.shape(0),
      out_winding_numbers.data(), epsilon, triangles.device_id(), stream);
}

auto brute_force_gradients(const Scalar_t &grad_output, const Vec3_t &points,
                           const Vec3_t &scaled_normals, const Vec3_t &queries,
                           GradResult_t &output_gradients,
                           float epsilon = 0.004F, const uint64_t stream = 0)
    -> void {
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
  if (points.device_id() != queries.device_id()) {
    throw std::runtime_error(
        "points and output_gradients must be on the same CUDA device.");
  }
  if (points.shape(0) != scaled_normals.shape(0)) {
    throw std::runtime_error(
        "Shape of points must be equal to shape of normals.");
  }
  if (points.shape(0) != output_gradients.shape(0)) {
    throw std::runtime_error(
        "Shape[0] of points must be equal to shape[0] of output_gradients");
  }
  if (output_gradients.shape(1) != 2) {
    throw std::runtime_error("Shape[1] of output_gradients has to be 2");
  }
  if (output_gradients.shape(2) != 3) {
    throw std::runtime_error("Shape[2] of output_gradients has to be 3");
  }
  brute_force_point_normal_gradient_impl(
      grad_output.data(), points.data(), scaled_normals.data(), queries.data(),
      points.shape(0), queries.shape(0), output_gradients.data(), epsilon,
      points.device_id(), stream);
}

auto brute_force_gradients(const Scalar_t &grad_output, const Vec3_t &vertices,
                           const TriangleIdx_t &triangle_indices,
                           const Vec3_t &queries,
                           VertexGradResult_t &output_gradients,
                           float epsilon = 0.004F, const uint64_t stream = 0)
    -> void {
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
  if (vertices.device_id() != output_gradients.device_id()) {
    throw std::runtime_error(
        "vertices and queries must be on the same CUDA device.");
  }
  if (vertices.shape(0) != output_gradients.shape(0)) {
    throw std::runtime_error(
        "Shape[0] of vertices must be equal to shape[0] of output_gradients");
  }
  if (output_gradients.shape(1) != 3) {
    throw std::runtime_error("Shape[1] of output_gradients has to be 3");
  }
  brute_force_mesh_gradient_impl(grad_output.data(), vertices.data(),
                                 triangle_indices.data(), queries.data(),
                                 triangle_indices.shape(0), queries.shape(0),
                                 vertices.shape(0), output_gradients.data(),
                                 epsilon, vertices.device_id(), stream);
}

auto brute_force_gradients(const Scalar_t &grad_output,
                           const Triangle_t &triangles, const Vec3_t &queries,
                           GradResult_t &output_gradients,
                           float epsilon = 0.004F, const uint64_t stream = 0)
    -> void {
  if (triangles.device_id() != grad_output.device_id()) {
    throw std::runtime_error(
        "triangles and grad_output must be on the same CUDA device.");
  }
  if (triangles.device_id() != queries.device_id()) {
    throw std::runtime_error(
        "triangles and queries must be on the same CUDA device.");
  }
  if (triangles.device_id() != output_gradients.device_id()) {
    throw std::runtime_error(
        "triangles and output_gradients must be on the same CUDA device.");
  }
  if (triangles.shape(0) != output_gradients.shape(0)) {
    throw std::runtime_error(
        "Shape[0] of triangles must be equal to shape[0] of output_gradients");
  }
  if (output_gradients.shape(1) != 3) {
    throw std::runtime_error("Shape[1] of output_gradients has to be 3");
  }
  if (output_gradients.shape(2) != 3) {
    throw std::runtime_error("Shape[2] of output_gradients has to be 3");
  }
  brute_force_triangle_gradient_impl(grad_output.data(), triangles.data(),
                                     queries.data(), triangles.shape(0),
                                     queries.shape(0), output_gradients.data(),
                                     epsilon, triangles.device_id(), stream);
}

class GradientEngine {
public:
  GradientEngine(const Vec3_t &queries, const Scalar_t &grad_output,
                 const uint64_t stream = 0)
      : m_impl{GradientBackend(queries.shape(0), queries.device_id(), stream)} {
    // TODO initialize after the checks, not in the constructor!
    if (queries.shape(0) != grad_output.shape(0)) {
      throw std::runtime_error(
          "There have to be the same number of queries as grad_output!");
    }
    if (queries.device_id() != grad_output.device_id()) {
      throw std::runtime_error(
          "queries and grad_output has to be on the same device.");
    }
    m_impl.init(queries.data(), grad_output.data());
  }

  ~GradientEngine() = default;

  // PointNormal
  auto compute(const Vec3_t &points, const Vec3_t &scaled_normals,
               GradResult_t &output_gradients, float beta = -1.F,
               float epsilon = 0.004F, const uint64_t stream = 0) -> void {
    if (points.device_id() != scaled_normals.device_id()) {
      throw std::runtime_error(
          "points and scaled_normals must be on the same CUDA device.");
    }
    if (points.device_id() != output_gradients.device_id()) {
      throw std::runtime_error(
          "points and output_gradients must be on the same CUDA device.");
    }
    if (points.shape(0) != scaled_normals.shape(0)) {
      throw std::runtime_error(
          "Shape of points must be equal to shape of normals.");
    }
    if (points.shape(0) != output_gradients.shape(0)) {
      throw std::runtime_error(
          "Shape[0] of points must be equal to shape[0] of output_gradients");
    }
    m_impl.compute(points.data(), scaled_normals.data(), points.shape(0),
                   output_gradients.data(), beta, epsilon, stream);
  }

  // Mesh
  auto compute(const Vec3_t &vertices, const TriangleIdx_t &triangle_indices,
               VertexGradResult_t &output_gradients, float beta = -1.F,
               float epsilon = 0.004F, const uint64_t stream = 0) -> void {

    if (vertices.device_id() != triangle_indices.device_id()) {
      throw std::runtime_error(
          "vertices and triangle_indices must be on the same CUDA device.");
    }
    if (vertices.device_id() != output_gradients.device_id()) {
      throw std::runtime_error(
          "vertices and queries must be on the same CUDA device.");
    }
    if (vertices.shape(0) != output_gradients.shape(0)) {
      throw std::runtime_error(
          "Shape[0] of vertices must be equal to shape[0] of output_gradients");
    }
    m_impl.compute(vertices.data(), triangle_indices.data(), vertices.shape(0),
                   triangle_indices.shape(0), output_gradients.data(), beta,
                   epsilon, stream);
  }
  // Triangles
  auto compute(const Triangle_t &triangles, GradResult_t &output_gradients,
               float beta = -1.F, float epsilon = 0.004F,
               const uint64_t stream = 0) -> void {
    if (triangles.device_id() != output_gradients.device_id()) {
      throw std::runtime_error(
          "triangles and output_gradients must be on the same CUDA device.");
    }
    if (triangles.shape(0) != output_gradients.shape(0)) {
      throw std::runtime_error("Shape[0] of triangles must be equal to "
                               "shape[0] of output_gradients");
    }

    m_impl.compute(triangles.data(), triangles.shape(0),
                   output_gradients.data(), beta, epsilon, stream);
  }

  [[nodiscard]] auto dump() const -> std::string {
    std::string result = m_impl.dump();
    return result;
  }

private:
  GradientBackend m_impl;
};

class WindingNumbersEngine {
public:
  // --- Triangle Mesh Constructor ---
  WindingNumbersEngine(const Triangle_t &triangles, const uint64_t stream) {
    m_impl_tri = WindingNumbersBackend<Triangle>::CreateFromTriangles(
        triangles.data(), triangles.shape(0), triangles.device_id(), stream);
    is_backend_triangle = true;
  }

  WindingNumbersEngine(const Vec3_t &vertices,
                       const TriangleIdx_t &triangle_indices,
                       const uint64_t stream) {
    if (vertices.device_id() != triangle_indices.device_id()) {
      throw std::runtime_error(
          "Vertices and triangle_indices must be on the same CUDA device.");
    }
    m_impl_tri = WindingNumbersBackend<Triangle>::CreateFromMesh(
        vertices.data(), vertices.shape(0), triangle_indices.data(),
        triangle_indices.shape(0), vertices.device_id(), stream);
    is_backend_triangle = true;
  }

  // --- Point Cloud Constructor ---
  WindingNumbersEngine(const Vec3_t &points, const Vec3_t &normals,
                       const uint64_t stream) {
    if (points.device_id() != normals.device_id()) {
      throw std::runtime_error(
          "Points and Normals must be on the same CUDA device.");
    }
    if (points.shape(0) != normals.shape(0)) {
      throw std::runtime_error(
          "Shape of points must be equal to shape of normals.");
    }

    m_impl_pn = WindingNumbersBackend<PointNormal>::CreateFromPoints(
        points.data(), normals.data(), points.shape(0), points.device_id(),
        stream);
    is_backend_triangle = false;
  }

  auto compute(const Vec3_t &queries, Scalar_t &out_winding_numbers,
               const float beta = -1.F, const float epsilon = 0.004F,
               const size_t stream = 0) -> void {
    size_t n = queries.shape(0);

    if (is_backend_triangle) {
      m_impl_tri->compute(queries.data(), n, out_winding_numbers.data(), beta,
                          epsilon, stream);
    } else {
      m_impl_pn->compute(queries.data(), n, out_winding_numbers.data(), beta,
                         epsilon, stream);
    }
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
  std::unique_ptr<WindingNumbersBackend<PointNormal>> m_impl_pn;
  std::unique_ptr<WindingNumbersBackend<Triangle>> m_impl_tri;

  // Internal constructor used by factory methods
  explicit WindingNumbersEngine(
      std::unique_ptr<WindingNumbersBackend<PointNormal>> backend)
      : m_impl_pn(std::move(backend)) {}
  explicit WindingNumbersEngine(
      std::unique_ptr<WindingNumbersBackend<Triangle>> backend)
      : m_impl_tri(std::move(backend)) {}
};

NB_MODULE(winder_module, m) {
  m.doc() = R"doc(
        GPU-accelerated Differentiable Winding Number Field library.
        
        Compatible with any framework supporting DLPack / Array API 
        (PyTorch, JAX, CuPy, etc.). All inputs must reside on the same 
        CUDA device.
    )doc";

  // Reusable epsilon docstring (used by all functions that take an epsilon).
  constexpr const char *EPSILON_DOC = R"doc(
               `epsilon` : float, optional
                   Dimensionless regularization scale for the singular winding-number
                   kernel, expressed as a fraction of the scene's bounding box diagonal.
                   The input geometry is normalized internally, so epsilon is invariant
                   to the units and scale of the input, and is safe to optimize as a
                   trainable parameter across meshes of different sizes.

                   Regularization sets a softening length (epsilon * diag) below which
                   the unregularized kernel is replaced by a smooth falloff that is
                   finite at zero distance. This prevents NaN/Inf when a query lands
                   near a source primitive (a point for point clouds, a triangle edge
                   or vertex for triangle soups), and stabilizes both the forward
                   winding-number field and its gradient with respect to the geometry.

                   Behavior as a function of `t = d / (epsilon * diag)`, where `d` is
                   the distance from the query to the nearest source primitive:
                     - large t : unregularized field
                     - small t : smooth falloff, finite at t = 0

                   Negative values disable regularization (treated as 0). The default
                   is 1/250 = 0.004.
  )doc";

  // ===========================================================================
  // brute_force_winding_numbers_*
  // ===========================================================================

  // --- Point-normal ---------------------------------------------------------
  m.def("brute_force_winding_numbers_point_normal",
        nb::overload_cast<const Vec3_t &, const Vec3_t &, const Vec3_t &,
                          Scalar_t &, float, uint64_t>(
            &brute_force_winding_numbers),
        "points"_a, "scaled_normals"_a, "queries"_a, "out_winding_numbers"_a,
        "epsilon"_a = 0.004F, "stream"_a = 0,
        nb::sig("def brute_force_winding_numbers_point_normal(points: "
                "winder.types.Array[winder.types.Shape[winder.types.N, "
                "typing.Literal[3]], winder.types.float32, "
                "winder.types.cuda], scaled_normals: "
                "winder.types.Array[winder.types.Shape[winder.types.N, "
                "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
                "queries: "
                "winder.types.Array[winder.types.Shape[winder.types.M, "
                "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
                "out_winding_numbers: "
                "winder.types.Array[winder.types.Shape[winder.types.M], "
                "winder.types.float32, winder.types.cuda], "
                "epsilon: float = 0.004, "
                "stream: int = 0) -> "
                "None"),
        (std::string(R"doc(
                Computes the winding number at the given query locations on GPU
                for a point cloud with scaled normals.

                NOTE: Brute Force implementation: exact, but in `O(N*M)`!

                Parameters
                ----------
                `points` : Array
                    A (N, 3) float32 CUDA array of point positions.
                `scaled_normals` : Array
                    A (N, 3) float32 CUDA array of scaled normals. 
                    The scaled normal direction is the orientation,
                    the scale the associated voronoi area.
                `queries` : Array
                    (M, 3) CUDA array of query points for which the winding number field is evaluated.
                `out_winding_numbers`:
                    (M,) float32 CUDA array pre allocated. Will be filled with the winding numbers for the queries
            )doc") +
         EPSILON_DOC + R"doc(
                `stream`  : int, optional
                    The raw 64-bit identifier (handle) of a CUDA stream. 
                    Allows enqueuing operations asynchronously within deep learning frameworks.
                    For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
                    Default is 0 (the default/null stream).
            )doc")
            .c_str());

  // --- Mesh (shared vertices + index array) ---------------------------------
  m.def("brute_force_winding_numbers_mesh",
        nb::overload_cast<const Vec3_t &, const TriangleIdx_t &, const Vec3_t &,
                          Scalar_t &, float, const uint64_t>(
            &brute_force_winding_numbers),
        "vertices"_a, "triangle_indices"_a, "queries"_a,
        "out_winding_numbers"_a, "epsilon"_a = 0.004F, "stream"_a = 0,
        nb::sig("def brute_force_winding_numbers_mesh(vertices: "
                "winder.types.Array[winder.types.Shape[winder.types.K, "
                "typing.Literal[3]], winder.types.float32, "
                "winder.types.cuda], triangle_indices: "
                "winder.types.Array[winder.types.Shape[winder.types.N, "
                "typing.Literal[3]], winder.types.uint32, winder.types.cuda], "
                "queries: "
                "winder.types.Array[winder.types.Shape[winder.types.M, "
                "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
                "out_winding_numbers: "
                "winder.types.Array[winder.types.Shape[winder.types.M], "
                "winder.types.float32, winder.types.cuda], "
                "epsilon: float = 0.004, "
                "stream: int = 0) -> "
                "None"),
        (std::string(R"doc(
                Computes the winding number at the given query locations on GPU
                for a triangle mesh with shared vertices and an index array.

                NOTE: Brute Force implementation: exact, but in `O(N*M)`!

                Parameters
                ----------
                `vertices` : Array
                    A (K, 3) float32 CUDA array holding K vertices.
                `triangle_indices`: Array
                    A (N, 3) index array, where each row defines the three vertices of a triangle.
                    Note: Order the vertices counter-clockwise when seen from the front. 
                `queries` : Array
                    (M, 3) CUDA array of query points for which the winding number field is evaluated.
                `out_winding_numbers`:
                    (M,) float32 CUDA array pre allocated. Will be filled with the winding numbers for the queries
            )doc") +
         EPSILON_DOC + R"doc(
                `stream`  : int, optional
                    The raw 64-bit identifier (handle) of a CUDA stream. 
                    Allows enqueuing operations asynchronously within deep learning frameworks.
                    For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
                    Default is 0 (the default/null stream).
            )doc")
            .c_str());

  // --- Triangle soup --------------------------------------------------------
  m.def("brute_force_winding_numbers_triangle_soup",
        nb::overload_cast<const Triangle_t &, const Vec3_t &, Scalar_t &, float,
                          const uint64_t>(&brute_force_winding_numbers),
        "triangles"_a, "queries"_a, "out_winding_numbers"_a,
        "epsilon"_a = 0.004F, "stream"_a = 0,
        nb::sig("def brute_force_winding_numbers_triangle_soup(triangles: "
                "winder.types.Array[winder.types.Shape[winder.types.N, "
                "typing.Literal[3], typing.Literal[3]], "
                "winder.types.float32, winder.types.cuda], queries: "
                "winder.types.Array[winder.types.Shape[winder.types.M, "
                "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
                "out_winding_numbers: "
                "winder.types.Array[winder.types.Shape[winder.types.M], "
                "winder.types.float32, winder.types.cuda], "
                "epsilon: float = 0.004, "
                "stream: "
                "int = 0) -> None"),
        (std::string(R"doc(
                Computes the winding number at the given query locations on GPU
                for an explicit triangle soup.

                NOTE: Brute Force implementation: exact, but in `O(N*M)`!

                Parameters
                ----------
                `triangles` : Array
                    A (N, 3, 3) float32 CUDA array holding N triangles
                    which consists of 3 vertices. 
                    Note: Order the vertices counter-clockwise when seen from the front. 
                `queries` : Array
                    (M, 3) CUDA array of query points for which the winding number field is evaluated.
                `out_winding_numbers`:
                    (M,) float32 CUDA array pre allocated. Will be filled with the winding numbers for the queries
            )doc") +
         EPSILON_DOC + R"doc(
                `stream`  : int, optional
                    The raw 64-bit identifier (handle) of a CUDA stream. 
                    Allows enqueuing operations asynchronously within deep learning frameworks.
                    For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
                    Default is 0 (the default/null stream).
            )doc")
            .c_str());

  // ===========================================================================
  // brute_force_gradients_*
  // ===========================================================================

  // --- Point-normal ---------------------------------------------------------
  m.def(
      "brute_force_gradients_point_normal",
      nb::overload_cast<const Scalar_t &, const Vec3_t &, const Vec3_t &,
                        const Vec3_t &, GradResult_t &, float, const uint64_t>(
          &brute_force_gradients),
      "grad_output"_a, "points"_a, "scaled_normals"_a, "queries"_a,
      "out_gradients"_a, "epsilon"_a = 0.004F, "stream"_a = 0,
      nb::sig("def brute_force_gradients_point_normal(grad_output: "
              "winder.types.Array[winder.types.Shape[winder.types.M], "
              "winder.types.float32, "
              "winder.types.cuda], points: "
              "winder.types.Array[winder.types.Shape[winder.types.N, "
              "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
              "scaled_normals: "
              "winder.types.Array[winder.types.Shape[winder.types.N, "
              "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
              "queries: winder.types.Array[winder.types.Shape[winder.types.M, "
              "typing.Literal[3]], winder.types.float32, "
              "winder.types.cuda], "
              "out_gradients: "
              "winder.types.Array[winder.types.Shape[winder.types.N, "
              "typing.Literal[2], typing.Literal[3]], winder.types.float32, "
              "winder.types.cuda], epsilon: float = 0.004, "
              "stream: int = 0) -> None"),
      (std::string(R"doc(
                Compute the partial derivatives w.r.t. the given point
                positions and scaled normals.

                NOTE: Brute Force implementation: exact, but in `O(N*M)`!

                This method propagates the gradient of a scalar loss function with 
                respect to the computed winding numbers back to the geometry.


                Parameters
                ----------
                `grad_output` : Array
                    (M,) float32 CUDA array representing the gradient of the loss 
                    with respect to the winding numbers at the query location for which 
                    the GradientEngine was built (dL/dw).
                `points`  : Array
                    (N, 3) float32 CUDA array representing the point positions for which the
                    winding numbers and loss were computed.
                `scaled_normals` : Array
                    (N, 3) float32 CUDA array representing the area-scaled normals for which the
                    winding numbers and loss were computed.
                `queries` : Array
                    (M, 3) float32 CUDA array with the queries for which the loss was evaluated.
                    You have to make sure that the grad_output is aligned with those queries.
                `out_gradients`:
                    Array (N, 2, 3) float32 CUDA array, to which the results are written.
                    output[i, 0] represents the gradient of the loss 
                      with respect to the input scaled normals at index i (dL/dn). 
                      Calculated as: dL/dn = (dL/dw)^T * (dw/dn).
                    output[i, 1] represents the gradient of the loss 
                      with respect to the source positions at index i (dL/dp)
                      Calculated as: dL/dp = (dL/dw)^T * (dw/dp).
            )doc") +
       EPSILON_DOC + R"doc(
                `stream` : int, optional
                    The raw 64-bit identifier (handle) of a CUDA stream. 
                    Allows enqueuing operations asynchronously within deep learning frameworks.
                    For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
                    Default is 0 (the default/null stream).

            )doc")
          .c_str());

  // --- Mesh ----------------------------------------------------------------
  m.def(
      "brute_force_gradients_mesh",
      nb::overload_cast<const Scalar_t &, const Vec3_t &, const TriangleIdx_t &,
                        const Vec3_t &, VertexGradResult_t &, float,
                        const uint64_t>(&brute_force_gradients),
      "grad_output"_a, "vertices"_a, "triangle_indices"_a, "queries"_a,
      "out_gradients"_a, "epsilon"_a = 0.004F, "stream"_a = 0,
      nb::sig("def brute_force_gradients_mesh(grad_output: "
              "winder.types.Array[winder.types.Shape[winder.types.M], "
              "winder.types.float32, "
              "winder.types.cuda], vertices: "
              "winder.types.Array[winder.types.Shape[winder.types.K, "
              "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
              "triangle_indices: "
              "winder.types.Array[winder.types.Shape[winder.types.N, "
              "typing.Literal[3]], winder.types.uint32, winder.types.cuda], "
              "queries: winder.types.Array[winder.types.Shape[winder.types.M, "
              "typing.Literal[3]], winder.types.float32, "
              "winder.types.cuda], "
              "out_gradients: "
              "winder.types.Array[winder.types.Shape[winder.types.K, "
              "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
              "epsilon: float = 0.004, "
              "stream: int = 0) -> None"),
      (std::string(R"doc(
                Compute the partial derivatives w.r.t. the given triangle mesh vertex
                positions (shared vertices, indexed triangles).

                NOTE: Brute Force implementation: exact, but in `O(N*M)`!

                This method propagates the gradient of a scalar loss function with 
                respect to the computed winding numbers back to the geometry.

                Parameters
                ----------
                `grad_output` : Array
                    (M,) float32 CUDA array representing the gradient of the loss 
                    with respect to the winding numbers at the query location for which 
                    the GradientEngine was built (dL/dw).
                    NOTE: You have to make sure that the queries used to compute L are the ones
                          used to build the GradientEngine!
                `vertices`  : Array
                    (K, 3) float32 CUDA array representing the shared vertices used in the triangles
                    for which the winding numbers and loss were computed.
                `triangle_indices`: Array
                    A (N, 3) index array, where each row defines the three vertices of a triangle.
                    Note: Order the vertices counter-clockwise when seen from the front. 
                `queries` : Array
                    (M, 3) float32 CUDA array with the queries for which the loss was evaluated.
                    You have to make sure that the grad_output is aligned with those queries.
                `out_gradients`:
                    Array (K, 3) float32 CUDA array, to which the results are written.
                    output[i] represents the gradient of the loss 
                      with respect to the vertex i (dL/dv_i)
                      Calculated as: dL/dv_i = (dL/dw)^T * (dw/dv_i).
            )doc") +
       EPSILON_DOC + R"doc(
                `stream` : int, optional
                    The raw 64-bit identifier (handle) of a CUDA stream. 
                    Allows enqueuing operations asynchronously within deep learning frameworks.
                    For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
                    Default is 0 (the default/null stream).
            )doc")
          .c_str());

  // --- Triangle soup -------------------------------------------------------
  m.def("brute_force_gradients_triangle_soup",
        nb::overload_cast<const Scalar_t &, const Triangle_t &, const Vec3_t &,
                          GradResult_t &, float, const uint64_t>(
            &brute_force_gradients),
        "grad_output"_a, "triangles"_a, "queries"_a, "out_gradients"_a,
        "epsilon"_a = 0.004F, "stream"_a = 0,
        nb::sig("def brute_force_gradients_triangle_soup(grad_output: "
                "winder.types.Array[winder.types.Shape[winder.types.M], "
                "winder.types.float32, "
                "winder.types.cuda], triangles: "
                "winder.types.Array[winder.types.Shape[winder.types.N, "
                "typing.Literal[3], typing.Literal[3]], winder.types.uint32, "
                "winder.types.cuda], queries: "
                "winder.types.Array[winder.types.Shape[winder.types.M, "
                "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
                "out_gradients: "
                "winder.types.Array[winder.types.Shape[winder.types.N, "
                "typing.Literal[3], typing.Literal[3]], winder.types.float32, "
                "winder.types.cuda], "
                "epsilon: float = 0.004, "
                "stream: int "
                "= 0) -> None"),
        (std::string(R"doc(
                Compute the partial derivatives w.r.t. the vertices of an explicit
                triangle soup.

                NOTE: Brute Force implementation: exact, but in `O(N*M)`!

                This method propagates the gradient of a scalar loss function with 
                respect to the computed winding numbers back to the geometry.

                Parameters
                ----------
                `grad_output` : Array
                    (M,) float32 CUDA array representing the gradient of the loss 
                    with respect to the winding numbers at the query location for which 
                    the GradientEngine was built (dL/dw).
                    NOTE: You have to make sure that the queries used to compute L are the ones
                          used to build the GradientEngine!
                `triangles`  : Array
                    (N, 3, 3) float32 CUDA array representing the triangles used 
                    to evaluate the winding numbers for which the winding numbers and loss was computed.
                `queries` : Array
                    (M, 3) float32 CUDA array with the queries for which the loss was evaluated.
                    You have to make sure that the grad_output is aligned with those queries.
                `out_gradients` :
                    Array (N, 3, 3) float32 CUDA array, to which the results are written
                    output[i, j] represents the gradient of the loss 
                      with respect to the vertex j of triangle i  (dL/dv_j)
                      Calculated as: dL/dv_j = (dL/dw)^T * (dw/dv_j).
            )doc") +
         EPSILON_DOC + R"doc(
                `stream` : int, optional
                    The raw 64-bit identifier (handle) of a CUDA stream. 
                    Allows enqueuing operations asynchronously within deep learning frameworks.
                    For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
                    Default is 0 (the default/null stream).
            )doc")
            .c_str());

  // ===========================================================================
  // WindingNumberEngine
  // ===========================================================================
  nb::class_<WindingNumbersEngine>(m, "WindingNumberEngine")
      // --- Triangle Mesh Constructor ---
      .def(nb::init<const Triangle_t, const uint64_t>(), "triangles"_a,
           "stream"_a = 0,
           nb::sig(
               "def __init__(self, triangles: "
               "winder.types.Array[winder.types.Shape[winder.types.N, "
               "typing.Literal[3], typing.Literal[3]], winder.types.float32, "
               "winder.types.cuda, stream: int]) -> None"),
           R"doc(
                Initialize the engine for fast winding number calculation using a triangle soup.
                
                Parameters
                ----------
                `triangles` : Array
                    A (N, 3, 3) float32 CUDA array holding N triangles
                    which consists of 3 vertices. 
                    Note: Order the vertices counter-clockwise when seen from the front. 
                `stream`: int, optional
                    The raw 64-bit identifier (handle) of a CUDA stream. 
                    Allows enqueuing operations asynchronously within deep learning frameworks.
                    For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
                    Default is 0 (the default/null stream).
            )doc")

      .def(nb::init<const Vec3_t &, const TriangleIdx_t &, const uint64_t>(),
           "vertices"_a, "triangle_indices"_a, "stream"_a = 0,
           nb::sig(
               "def __init__(self, vertices: "
               "winder.types.Array[winder.types.Shape[winder.types.K, "
               "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
               "triangle_indices: "
               "winder.types.Array[winder.types.Shape[winder.types.N, "
               "typing.Literal[3]], winder.types.uint32, winder.types.cuda, "
               "stream: int]) -> None"),
           R"doc(
                Initialize the engine for fast winding number calculation using a triangle soup with N Triangles using K shared vertices.
                
                Parameters
                ----------
                `vertices` : Array
                    A (K, 3) float32 CUDA array holding K vertices.
                `triangle_indices`: Array
                    A (N, 3) index array, where each row defines the three vertices of a triangle.
                    Note: Order the vertices counter-clockwise when seen from the front. 
                `stream`: int, optional
                    The raw 64-bit identifier (handle) of a CUDA stream. 
                    Allows enqueuing operations asynchronously within deep learning frameworks.
                    For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
                    Default is 0 (the default/null stream).
            )doc")

      // --- Point Cloud Constructor ---
      .def(nb::init<const Vec3_t &, const Vec3_t &, const uint64_t>(),
           "points"_a, "scaled_normals"_a, "stream"_a = 0,
           nb::sig(
               "def __init__(self, points: "
               "winder.types.Array[winder.types.Shape[winder.types.N, "
               "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
               "scaled_normals: "
               "winder.types.Array[winder.types.Shape[winder.types.N, "
               "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
               "stream: int) -> None"),
           R"doc(
                Initialize the engine for fast winding number calculation using a point cloud with scaled normals.
                
                Parameters
                ----------
                `points` : Array
                    A (N, 3) float32 CUDA array of point positions.
                `scaled_normals` : Array
                    A (N, 3) float32 CUDA array of scaled normals. 
                    The scaled normal direction is the orientation,
                    the scale the associated voronoi area.
                `stream`: int, optional
                    The raw 64-bit identifier (handle) of a CUDA stream. 
                    Allows enqueuing operations asynchronously within deep learning frameworks.
                    For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
                    Default is 0 (the default/null stream).
            )doc")

      // --- Inference (single compute; the engine was built for one primitive)
      // ---
      .def("compute", &WindingNumbersEngine::compute, "queries"_a,
           "out_winding_numbers"_a, "beta"_a = -1.F, "epsilon"_a = 0.004F,
           "stream"_a = 0,
           nb::sig("def compute(self, queries: "
                   "winder.types.Array[winder.types.Shape[winder.types.M, "
                   "typing.Literal[3]], winder.types.float32, "
                   "winder.types.cuda], "
                   "out_winding_numbers: "
                   "winder.types.Array[winder.types.Shape[winder.types.M], "
                   "winder.types.float32, winder.types.cuda],"
                   "beta: float = -1, epsilon: float = 0.004, "
                   "stream: int = 0) -> "
                   "None"),
           (std::string(R"doc(
                Computes the winding number at the given query locations.

                Parameters
                ----------
                `queries` : Array
                    (M, 3) CUDA array of query points for which the winding number field is evaluated.
                `out_winding_numbers`:
                    (M,) float32 CUDA array pre allocated. Will be filled with the winding numbers for the queries
                `beta`    : float
                    Scalar that controls the degree of approximation.
                    Larger beta leads to more precise results, but also slower execution.
                    Use any negative number to get default values.
                    Default for point clouds is 2.3.
                    Default for triangles is 2.0
            )doc") +
            EPSILON_DOC + R"doc(
                `stream`  : int, optional
                    The raw 64-bit identifier (handle) of a CUDA stream. 
                    Allows enqueuing operations asynchronously within deep learning frameworks.
                    For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
                    Default is 0 (the default/null stream).
            )doc")
               .c_str())
      // --- Utilities ---
      .def("dump", &WindingNumbersEngine::dump,
           nb::sig("def dump(self) -> str"),
           R"doc(
            Returns a detailed .dot string representation of the internal BVH8 Tree.
          )doc");

  // ===========================================================================
  // GradientEngine
  // ===========================================================================
  nb::class_<GradientEngine>(m, "GradientEngine")
      .def(nb::init<const Vec3_t &, const Scalar_t &, uint64_t>(), "queries"_a,
           "grad_output"_a, "stream"_a = 0,
           nb::sig(
               "def __init__(self, queries: "
               "winder.types.Array[winder.types.Shape[winder.types.M, "
               "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
               "grad_output: "
               "winder.types.Array[winder.types.Shape[winder.types.M], "
               "winder.types.float32, winder.types.cuda], "
               "stream: int = 0) -> None"),
           R"doc(
                Initialize the engine for fast gradient calculation from the given queries.
                
                Parameters
                ----------
                `queries` : Array
                    A (M, 3) float32 CUDA array holding M query positions (xyz).
                    The gradient is computed for a loss computed for the winding numbers
                    at those query locations.
                `grad_output` : Array
                    (M,) float32 CUDA array representing the gradient of the loss 
                    with respect to the winding numbers at the queries.
                `stream`  : int, optional
                    The raw 64-bit identifier (handle) of a CUDA stream. 
                    Allows enqueuing operations asynchronously within deep learning frameworks.
                    For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
                    Default is 0 (the default/null stream).
                    )doc")

      // --- compute_mesh ----------------------------------------------------
      .def(
          "compute_mesh",
          nb::overload_cast<const Vec3_t &, const TriangleIdx_t &,
                            VertexGradResult_t &, float, float, const uint64_t>(
              &GradientEngine::compute),
          "vertices"_a, "triangle_indices"_a, "out_gradients"_a,
          "beta"_a = -1.F, "epsilon"_a = 0.004F, "stream"_a = 0,
          nb::sig(
              "def compute_mesh(self, vertices: "
              "winder.types.Array[winder.types.Shape[winder.types.K, "
              "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
              "triangle_indices: "
              "winder.types.Array[winder.types.Shape[winder.types.N, "
              "typing.Literal[3]], winder.types.uint32, winder.types.cuda], "
              "out_gradients: "
              "winder.types.Array[winder.types.Shape[winder.types.K, "
              "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
              "beta: float = -1, epsilon: float = 0.004, "
              "stream: int = 0) "
              "-> None"),
          (std::string(R"doc(
                Compute the partial derivatives w.r.t. the given triangle mesh vertex
                positions (shared vertices, indexed triangles).

                This method propagates the gradient of a scalar loss function with 
                respect to the computed winding numbers back to the geometry.

                Parameters
                ----------
                `vertices`  : Array
                    (K, 3) float32 CUDA array representing the shared vertices used in the triangles
                    for which the winding numbers and loss were computed.
                `triangle_indices`: Array
                    A (N, 3) index array, where each row defines the three vertices of a triangle.
                    Note: Order the vertices counter-clockwise when seen from the front. 
                `out_gradients`:
                    Array (K, 3) float32 CUDA array, to which the results are written.
                    output[i] represents the gradient of the loss 
                      with respect to the vertex i (dL/dv_i)
                      Calculated as: dL/dv_i = (dL/dw)^T * (dw/dv_i).
                `beta`    : float
                    Scalar that controls the degree of approximation.
                    Larger beta leads to more precise results, but also slower execution.
                    Use any negative number to get default values.
                    Default for triangles is 2.3 
            )doc") +
           EPSILON_DOC + R"doc(
                `stream` : int, optional
                    The raw 64-bit identifier (handle) of a CUDA stream. 
                    Allows enqueuing operations asynchronously within deep learning frameworks.
                    For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
                    Default is 0 (the default/null stream).
            )doc")
              .c_str())

      // --- compute_triangle_soup -------------------------------------------
      .def("compute_triangle_soup",
           nb::overload_cast<const Triangle_t &, GradResult_t &, float, float,
                             const uint64_t>(&GradientEngine::compute),
           "triangles"_a, "out_gradients"_a, "beta"_a = -1.F,
           "epsilon"_a = 0.004F, "stream"_a = 0,
           nb::sig("def compute_triangle_soup(self, triangles: "
                   "winder.types.Array[winder.types.Shape[winder.types.N, "
                   "typing.Literal[3], typing.Literal[3]], "
                   "winder.types.float32, winder.types.cuda], "
                   "out_gradients: "
                   "winder.types.Array[winder.types.Shape[winder.types.N, "
                   "typing.Literal[3], typing.Literal[3]], "
                   "winder.types.float32, winder.types.cuda], "
                   "beta: float = -1, "
                   "epsilon: float = 0.004, "
                   "stream: int = 0) "
                   "-> None"),
           (std::string(R"doc(
                Compute the partial derivatives w.r.t. the vertices of an explicit
                triangle soup.

                This method propagates the gradient of a scalar loss function with 
                respect to the computed winding numbers back to the geometry.

                Parameters
                ----------
                `triangles`  : Array
                    (N, 3, 3) float32 CUDA array representing the triangles used 
                    to evaluate the winding numbers for which the winding numbers and loss was computed.
                `out_gradients` :
                    Array (N, 3, 3) float32 CUDA array, to which the results are written
                    output[i, j] represents the gradient of the loss 
                      with respect to the vertex j of triangle i  (dL/dv_j)
                      Calculated as: dL/dv_j = (dL/dw)^T * (dw/dv_j).
                `beta`    : float
                    Scalar that controls the degree of approximation.
                    Larger beta leads to more precise results, but also slower execution.
                    Use any negative number to get default values.
                    Default for triangles is 2.3 
            )doc") +
            EPSILON_DOC + R"doc(
                `stream` : int, optional
                    The raw 64-bit identifier (handle) of a CUDA stream. 
                    Allows enqueuing operations asynchronously within deep learning frameworks.
                    For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
                    Default is 0 (the default/null stream).
            )doc")
               .c_str())

      // --- compute_point_normal --------------------------------------------
      .def("compute_point_normal",
           nb::overload_cast<const Vec3_t &, const Vec3_t &, GradResult_t &,
                             float, float, const uint64_t>(
               &GradientEngine::compute),
           "points"_a, "scaled_normals"_a, "out_gradients"_a, "beta"_a = -1.F,
           "epsilon"_a = 0.004F, "stream"_a = 0,
           nb::sig(
               "def compute_point_normal(self, points: "
               "winder.types.Array[winder.types.Shape[winder.types.N, "
               "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
               "scaled_normals: "
               "winder.types.Array[winder.types.Shape[winder.types.N, "
               "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
               "out_gradients: "
               "winder.types.Array[winder.types.Shape[winder.types.N, "
               "typing.Literal[2], typing.Literal[3]], winder.types.float32, "
               "winder.types.cuda], "
               "beta: float = -1, epsilon: float = 0.004, "
               "stream: int = 0) "
               "-> None"),
           (std::string(R"doc(
                Compute the partial derivatives w.r.t. the given point positions
                and scaled normals.

                This method propagates the gradient of a scalar loss function with 
                respect to the computed winding numbers back to the geometry.


                Parameters
                ----------
                `points`  : Array
                    (N, 3) float32 CUDA array representing the point positions for which the
                    winding numbers and loss were computed.
                `scaled_normals` : Array
                    (N, 3) float32 CUDA array representing the area-scaled normals for which the
                    winding numbers and loss were computed.
                `out_gradients`:
                    Array (N, 2, 3) float32 CUDA array, to which the results are written.
                    output[i, 0] represents the gradient of the loss 
                      with respect to the input scaled normals at index i (dL/dn). 
                      Calculated as: dL/dn = (dL/dw)^T * (dw/dn).
                    output[i, 1] represents the gradient of the loss 
                      with respect to the source positions at index i (dL/dp)
                      Calculated as: dL/dp = (dL/dw)^T * (dw/dp).
                `beta`    : float
                    Scalar that controls the degree of approximation.
                    Larger beta leads to more precise results, but also slower execution.
                    Use any negative number to get default values.
                    Default for point clouds is 2.3.
            )doc") +
            EPSILON_DOC + R"doc(
                `stream` : int, optional
                    The raw 64-bit identifier (handle) of a CUDA stream. 
                    Allows enqueuing operations asynchronously within deep learning frameworks.
                    For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
                    Default is 0 (the default/null stream).
            )doc")
               .c_str())

      .def("dump", &GradientEngine::dump, nb::sig("def dump(self) -> str"),
           R"doc(
            Returns a detailed .dot string representation of the internal BVH8 Tree.
          )doc");
}
