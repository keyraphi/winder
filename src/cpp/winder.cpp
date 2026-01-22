#include <cstddef>
#include <cstring>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "utils.h"
#include "winder_cuda.h"

namespace nb = nanobind;
using namespace nb::literals;

// Device-specific type aliases with exact constraints
using Vec3_t = nb::ndarray<nb::array_api, float, nb::shape<-1, 3>, nb::c_contig,
                           nb::device::cuda>;
using Triangle_t = nb::ndarray<nb::array_api, float, nb::shape<-1, 3, 3>,
                               nb::c_contig, nb::device::cuda>;
using Scalar_t = nb::ndarray<nb::array_api, float, nb::shape<-1>, nb::c_contig,
                             nb::device::cuda>;

class WinderEngine {
public:
  // --- Triangle Mesh Constructor ---
  WinderEngine(const Triangle_t &triangles) {
    m_impl = WinderBackend::CreateFromMesh(triangles.data(), triangles.shape(0), triangles.device_id());
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

    m_impl = WinderBackend::CreateFromPoints(points.data(), normals.data(), points.shape(0), points.device_id());
  }

  // --- Static Solver (Factory Method) ---
  static WinderEngine solve_from_constraints(
      const Vec3_t &points, const Vec3_t &extra_points,
      const Scalar_t &extra_winding_numbers,
      const std::optional<Scalar_t> &maybe_points_winding_nubers,
      float alpha_extra) {
    int device_id = points.device_id();
    if (extra_points.device_id() != device_id ||
        extra_winding_numbers.device_id() != device_id) {
      throw std::runtime_error(
          "All input arrays must reside on the same CUDA device.");
    }
    if (maybe_points_winding_nubers.has_value() &&
        maybe_points_winding_nubers->device_id() != device_id) {
      throw std::runtime_error(
          "pc_wn must reside on the same CUDA device as pc.");
    }
    if (maybe_points_winding_nubers.has_value() &&
        maybe_points_winding_nubers->shape(0) != points.shape(0)) {
      throw std::runtime_error("pc_wn must have one value per point.");
    }
    // by default the points are assumed to be on a surface. Ther winding number
    // is 0.5
    Scalar_t points_winding_numbers = winder_cuda::scalar_with_default(
        maybe_points_winding_nubers, points.shape(0), 0.5F);

    auto backend = WinderBackend::CreateForSolver(points.data(), points.shape(0), points.device_id());

    backend->solve_for_normals(extra_points.data(), extra_points.shape(0),
                               extra_winding_numbers.data(),
                               points_winding_numbers.data(), alpha_extra);

    return WinderEngine(std::move(backend));
  }

  Vec3_t get_scaled_normals() {
    auto raw_ptr_unique = m_impl->get_normals();

    // 3. Handle the Mesh case
    if (!raw_ptr_unique) {
      return {};
    }

    float *raw_ptr = raw_ptr_unique.release();
    nb::capsule owner(raw_ptr, [](void *p) noexcept { winder_cuda::cuda_free(p); });

    size_t n_points = m_impl->point_count();
    return {raw_ptr, {n_points, 3}, owner};
  }

  Scalar_t compute(const Vec3_t &queries) {
    size_t n = queries.shape(0);
    // todo ensure queries are on same device as m_impl

    auto raw_ptr_unique = m_impl->compute(queries.data(), n);
    float *raw_ptr = raw_ptr_unique.release();
    nb::capsule owner(raw_ptr, [](void *p) noexcept { winder_cuda::cuda_free(p); });

    return {raw_ptr, {n}, owner};
  }

  Vec3_t get_grad_normals(const Scalar_t &grad_output) {

    auto raw_ptr_unique =
        m_impl->grad_normals(grad_output.data(), grad_output.shape(0));

    float *raw_ptr = raw_ptr_unique.release();
    nb::capsule owner(raw_ptr, [](void *p) noexcept { winder_cuda::cuda_free(p); });

    size_t n_points = m_impl->point_count();
    return {raw_ptr, {n_points, 3}, owner};
  }

  Vec3_t get_grad_points(const Scalar_t &grad_output) {

    auto raw_ptr_unique =
        m_impl->grad_points(grad_output.data(), grad_output.shape(0));
    float *raw_ptr = raw_ptr_unique.release();
    nb::capsule owner(raw_ptr, [](void *p) noexcept { winder_cuda::cuda_free(p); });

    size_t n_points = m_impl->point_count();
    return {raw_ptr, {n_points, 3}, nb::handle()};
  }

private:
  std::unique_ptr<WinderBackend> m_impl;

  // Internal constructor used by factory methods
  explicit WinderEngine(std::unique_ptr<WinderBackend> backend)
      : m_impl(std::move(backend)) {}
};

NB_MODULE(winder_backend, m) {
  m.doc() = R"doc(
        GPU-accelerated Differentiable Winding Number Field library.
        
        Compatible with any framework supporting DLPack / Array API 
        (PyTorch, JAX, CuPy, etc.). All inputs must reside on the same 
        CUDA device.
    )doc";

  nb::class_<WinderEngine>(m, "WinderEngine")
      // --- Triangle Mesh Constructor ---
      .def(nb::init<Triangle_t>(), "triangles"_a,
           nb::sig("def __init__(self, triangles: Array) -> None"),
           R"doc(
                Initialize the engine using a triangle soup.
                
                Parameters
                ----------
                triangles : Array
                    A (N, 3, 3) float32 CUDA array holding N triangles
                    which consists of 3 points with (x,y,z) coordinates.
            )doc")

      // --- Point Cloud Constructor ---
      .def(nb::init<Vec3_t, Vec3_t>(), "points"_a, "normals"_a,
           nb::sig("def __init__(self, points: Array, normals: Array) -> None"),
           R"doc(
                Initialize the engine using a point cloud with scaled normals.
                
                Parameters
                ----------
                points : Array
                    A (N, 3) float32 CUDA array of point positions.
                normals : Array
                    A (N, 3) float32 CUDA array of scaled normals. 
                    The scaled normal direction is the orientation,
                    the scale the associated voronoi area.
            )doc")

      // --- Normal Solver ---
      .def_static("solve_normals", &WinderEngine::solve_from_constraints,
                  "pc"_a, "extra_points"_a, "extra_wn"_a,
                  "pc_wn"_a = nb::none(), "alpha_extra"_a = 0.2f,
                  nb::sig("def solve_normals(pc: Array, extra_points: Array, "
                          "extra_wn: Array, pc_wn: Optional[Array] = None, "
                          "alpha_extra: float = 0.2) -> WinderEngine"),
                  R"doc(
                Solves for optimal scaled normals for a point cloud.

                This solver finds the orientation and scale (area) of normals that 
                best satisfy the provided winding number constraints.

                Parameters
                ----------
                pc : Array
                    (N, 3) CUDA array of source points.
                extra_points : Array
                    (K, 3) CUDA array of query points with known winding values.
                extra_wn : Array
                    (K,) CUDA array of target winding values at extra_points.
                pc_wn : Array, optional
                    (N,) CUDA array of target winding values for the source points. 
                    If None, defaults to 0.5 for all points.
                alpha_extra : float, optional
                    Weighting factor for the 'extra_points' constraints relative to 
                    the point cloud constraints. A lower value prioritizes the 
                    pc_wn (0.5) manifold. Default is 0.2.

                Returns
                -------
                WinderEngine
                    A new engine instance initialized with the solved normals.
            )doc")
      // --- Data Access ---
      .def_prop_ro("scaled_normals", &WinderEngine::get_scaled_normals,
                   nb::sig("@property\ndef scaled_normals(self) -> Array"),
                   R"doc(
                Returns a copy of the scaled normals currently stored in the engine.

                Returns
                -------
                Array
                    (N, 3) float32 CUDA array.
                
                Note
                ----
                If the engine is in Triangle Mesh mode, this will return an empty array 
                or None, as meshes use face geometry instead of point normals.
            )doc")

      // --- Inference ---
      .def("compute", &WinderEngine::compute, "queries"_a,
           nb::sig("def compute(self, queries: Array) -> Array"),
           R"doc(
                Computes the winding number at the given query locations.

                Parameters
                ----------
                queries : Array
                    (N, 3) CUDA array of query points.

                Returns
                -------
                (N,) float32 CUDA array holding the winding numbers.
            )doc")

      // --- Gradients ---
      .def(
          "grad_scaled_normals", &WinderEngine::get_grad_normals,
          "grad_output"_a,
          nb::sig("def grad_scaled_normals(self, grad_output: Array) -> Array"),
          R"doc(
                Compute the Vector-Jacobian Product (VJP) for the scaled normals.

                This function propagates the gradient of a scalar loss function with 
                respect to the computed winding numbers back to the input normals.

                Parameters
                ----------
                grad_output : Array
                    (Q,) float32 CUDA array representing the gradient of the loss 
                    with respect to the winding numbers at each query location 
                    (dL/dw).

                Returns
                -------
                Array
                    (N, 3) float32 CUDA array representing the gradient of the loss 
                    with respect to the input scaled normals (dL/dn). 
                    Calculated as: dL/dn = (dL/dw)^T * (dw/dn).

                Note
                ----
                If the engine was initialized in Triangle Mesh mode, this returns 
                an array of zeros as normals are not primary inputs.
            )doc")

      .def("grad_points", &WinderEngine::get_grad_points, "grad_output"_a,
           nb::sig("def grad_points(self, grad_output: Array) -> Array"),
           R"doc(
                Compute the Vector-Jacobian Product (VJP) for the source point positions.

                Propagates the gradient of the loss from the query locations back to 
                the source geometry positions (vertices or point cloud centers).

                Parameters
                ----------
                grad_output : Array
                    (Q,) float32 CUDA array representing the gradient of the loss 
                    with respect to the winding numbers at each query location 
                    (dL/dw).

                Returns
                -------
                Array
                    (N, 3) float32 CUDA array representing the gradient of the loss 
                    with respect to the source positions (dL/dp).
                    
                Note
                ----
                For point clouds, this computes the gradient of the dipole potential.
                For meshes, this computes the gradient of the signed solid angle 
                with respect to triangle vertices.
            )doc");
}
