#include "geometry.h"
#include "gradient_backend.h"
#include "kernels/payload.cuh"
#include "payload_backend.h"
#include "winder_brute_force.h"
#include "winding_numbers_backend.h"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

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
using SingleScalar_t = nb::ndarray<nb::array_api, float, nb::shape<1>,
                                   nb::c_contig, nb::device::cuda>;
using Payload_t = nb::ndarray<nb::array_api, float, nb::shape<-1, -1>,
                              nb::c_contig, nb::device::cuda>;
using GradResult_t = nb::ndarray<nb::array_api, float, nb::shape<-1, -1, 3>,
                                 nb::c_contig, nb::device::cuda>;
using VertexGradResult_t = nb::ndarray<nb::array_api, float, nb::shape<-1, 3>,
                                       nb::c_contig, nb::device::cuda>;

auto brute_force_winding_numbers(
    const Vec3_t &points, const Vec3_t &scaled_normals, const Vec3_t &queries,
    Scalar_t &out_winding_numbers, float epsilon = 0.004,
    const uint64_t stream = 0, const bool use_kahan = true) -> void {
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
                                epsilon, points.device_id(), stream, use_kahan);
}

auto brute_force_winding_numbers(const Vec3_t &vertices,
                                 const TriangleIdx_t &triangle_indices,
                                 const Vec3_t &queries,
                                 Scalar_t &out_winding_numbers,
                                 float epsilon = 0.004F,
                                 const uint64_t stream = 0,
                                 const bool use_kahan = true) -> void {
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
  brute_force_mesh_impl(vertices.data(), triangle_indices.data(),
                        queries.data(), triangle_indices.shape(0),
                        vertices.shape(0), queries.shape(0),
                        out_winding_numbers.data(), epsilon,
                        vertices.device_id(), stream, use_kahan);
}

auto brute_force_winding_numbers(const Triangle_t &triangles,
                                 const Vec3_t &queries,
                                 Scalar_t &out_winding_numbers,
                                 float epsilon = 0.004F,
                                 const uint64_t stream = 0,
                                 const bool use_kahan = true) -> void {
  if (queries.shape(0) != out_winding_numbers.shape(0)) {
    throw std::runtime_error(
        "Query count must be equal to out_winding_numbers count (shape[0]).");
  }
  brute_force_triangle_impl(triangles.data(), queries.data(),
                            triangles.shape(0), queries.shape(0),
                            out_winding_numbers.data(), epsilon,
                            triangles.device_id(), stream, use_kahan);
}

auto brute_force_winding_payloads(const Vec3_t &points,
                                  const Vec3_t &scaled_normals,
                                  const Payload_t &payload,
                                  const Vec3_t &queries, Payload_t &out_payload,
                                  float epsilon = 0.004,
                                  const uint64_t stream = 0,
                                  const bool use_kahan = false) -> void {
  if (points.device_id() != scaled_normals.device_id()) {
    throw std::runtime_error(
        "Points and Normals must be on the same CUDA device.");
  }
  if (points.device_id() != payload.device_id()) {
    throw std::runtime_error(
        "Points and payload must be on the same CUDA device.");
  }
  if (points.device_id() != queries.device_id()) {
    throw std::runtime_error(
        "Points and queries must be on the same CUDA device.");
  }
  if (points.device_id() != out_payload.device_id()) {
    throw std::runtime_error(
        "Points and out_winding_numbers must be on the same CUDA device.");
  }
  if (points.shape(0) != scaled_normals.shape(0)) {
    throw std::runtime_error(
        "Shape of points must be equal to shape of normals.");
  }
  if (points.shape(0) != payload.shape(0)) {
    throw std::runtime_error(
        "Number of points must be equal to number of payload.");
  }
  if (payload.shape(1) != out_payload.shape(1)) {
    throw std::runtime_error("Dimensionality of payload must be equal to "
                             "dimensionality of out_payload.");
  }
  if (queries.shape(0) != out_payload.shape(0)) {
    throw std::runtime_error(
        "Query count must be equal to out_payload count (shape[0]).");
  }
  brute_force_point_normal_payload_impl(
      points.data(), scaled_normals.data(), payload.data(), queries.data(),
      points.shape(0), queries.shape(0), payload.shape(1), out_payload.data(),
      epsilon, points.device_id(), stream, use_kahan);
}

auto brute_force_winding_payloads(const Vec3_t &vertices,
                                  const TriangleIdx_t &triangle_indices,
                                  const Payload_t &payload,
                                  const Vec3_t &queries, Payload_t &out_payload,
                                  float epsilon = 0.004F,
                                  const uint64_t stream = 0,
                                  const bool use_kahan = false) -> void {
  if (vertices.device_id() != triangle_indices.device_id()) {
    throw std::runtime_error(
        "Vertices and triangle_indices must be on the same CUDA device.");
  }
  if (vertices.device_id() != payload.device_id()) {
    throw std::runtime_error(
        "Vertices and payload must be on the same CUDA device.");
  }
  if (vertices.device_id() != queries.device_id()) {
    throw std::runtime_error(
        "Vertices and queries must be on the same CUDA device.");
  }
  if (vertices.device_id() != out_payload.device_id()) {
    throw std::runtime_error(
        "Vertices and out_payload must be on the same CUDA device.");
  }
  if (payload.shape(1) != out_payload.shape(1)) {
    throw std::runtime_error("Dimensionality of payload must be equal to "
                             "dimensionality of out_payload.");
  }
  if (queries.shape(0) != out_payload.shape(0)) {
    throw std::runtime_error(
        "Query count must be equal to out_payload count (shape[0]).");
  }
  brute_force_mesh_payload_impl(
      vertices.data(), triangle_indices.data(), payload.data(), queries.data(),
      triangle_indices.shape(0), vertices.shape(0), queries.shape(0),
      payload.shape(1), out_payload.data(), epsilon, vertices.device_id(),
      stream, use_kahan);
}

auto brute_force_winding_payloads(const Triangle_t &triangles,
                                  const Payload_t &payload,
                                  const Vec3_t &queries, Payload_t &out_payload,
                                  float epsilon = 0.004F,
                                  const uint64_t stream = 0,
                                  const bool use_kahan = false) -> void {
  if (triangles.device_id() != payload.device_id()) {
    throw std::runtime_error(
        "Triangles and payload must be on the same CUDA device.");
  }
  if (triangles.device_id() != queries.device_id()) {
    throw std::runtime_error(
        "Triangles and queries must be on the same CUDA device.");
  }
  if (payload.shape(1) != out_payload.shape(1)) {
    throw std::runtime_error("Dimensionality of payload must be equal to "
                             "dimensionality of out_payload.");
  }
  if (queries.shape(0) != out_payload.shape(0)) {
    throw std::runtime_error(
        "Query count must be equal to out_payload count (shape[0]).");
  }
  brute_force_triangle_payload_impl(
      triangles.data(), payload.data(), queries.data(), triangles.shape(0),
      queries.shape(0), payload.shape(1), out_payload.data(), epsilon,
      triangles.device_id(), stream, use_kahan);
}

auto brute_force_gradients(const Scalar_t &grad_output, const Vec3_t &points,
                           const Vec3_t &scaled_normals, const Vec3_t &queries,
                           GradResult_t &output_gradients,
                           SingleScalar_t &epsilon_gradient,
                           float epsilon = 0.004F, const uint64_t stream = 0,
                           const bool use_kahan = false) -> void {
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
      points.shape(0), queries.shape(0), output_gradients.data(),
      epsilon_gradient.data(), epsilon, points.device_id(), stream, use_kahan);
}

auto brute_force_gradients(const Scalar_t &grad_output, const Vec3_t &vertices,
                           const TriangleIdx_t &triangle_indices,
                           const Vec3_t &queries,
                           VertexGradResult_t &output_gradients,
                           SingleScalar_t &epsilon_gradient,
                           float epsilon = 0.004F, const uint64_t stream = 0,
                           const bool use_kahan = false) -> void {
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
  brute_force_mesh_gradient_impl(
      grad_output.data(), vertices.data(), triangle_indices.data(),
      queries.data(), triangle_indices.shape(0), queries.shape(0),
      vertices.shape(0), output_gradients.data(), epsilon_gradient.data(),
      epsilon, vertices.device_id(), stream, use_kahan);
}

auto brute_force_gradients(const Scalar_t &grad_output,
                           const Triangle_t &triangles, const Vec3_t &queries,
                           GradResult_t &output_gradients,
                           SingleScalar_t &epsilon_gradient,
                           float epsilon = 0.004F, const uint64_t stream = 0,
                           const bool use_kahan = false) -> void {
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
  brute_force_triangle_gradient_impl(
      grad_output.data(), triangles.data(), queries.data(), triangles.shape(0),
      queries.shape(0), output_gradients.data(), epsilon_gradient.data(),
      epsilon, triangles.device_id(), stream, use_kahan);
}

auto brute_force_query_gradients(const Scalar_t &grad_output,
                                 const Vec3_t &points,
                                 const Vec3_t &scaled_normals,
                                 const Vec3_t &queries, Vec3_t &query_gradients,
                                 float epsilon = 0.004F,
                                 const uint64_t stream = 0,
                                 const bool use_kahan = true) -> void {
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
  if (points.device_id() != query_gradients.device_id()) {
    throw std::runtime_error(
        "points and query_gradients must be on the same CUDA device.");
  }
  if (points.shape(0) != scaled_normals.shape(0)) {
    throw std::runtime_error(
        "Shape of points must be equal to shape of normals.");
  }
  if (queries.shape(0) != grad_output.shape(0)) {
    throw std::runtime_error(
        "Shape[0] of queries must be equal to shape[0] of grad_output.");
  }
  if (queries.shape(0) != query_gradients.shape(0)) {
    throw std::runtime_error(
        "Shape[0] of queries must be equal to shape[0] of query_gradients.");
  }
  if (query_gradients.shape(1) != 3) {
    throw std::runtime_error("Shape[1] of query_gradients has to be 3");
  }
  brute_force_point_normal_query_gradient_impl(
      grad_output.data(), points.data(), scaled_normals.data(), queries.data(),
      points.shape(0), queries.shape(0), query_gradients.data(), epsilon,
      points.device_id(), stream, use_kahan);
}

auto brute_force_query_gradients(const Scalar_t &grad_output,
                                 const Triangle_t &triangles,
                                 const Vec3_t &queries, Vec3_t &query_gradients,
                                 float epsilon = 0.004F,
                                 const uint64_t stream = 0,
                                 const bool use_kahan = true) -> void {
  if (triangles.device_id() != grad_output.device_id()) {
    throw std::runtime_error(
        "triangles and grad_output must be on the same CUDA device.");
  }
  if (triangles.device_id() != queries.device_id()) {
    throw std::runtime_error(
        "triangles and queries must be on the same CUDA device.");
  }
  if (triangles.device_id() != query_gradients.device_id()) {
    throw std::runtime_error(
        "triangles and query_gradients must be on the same CUDA device.");
  }
  if (queries.shape(0) != grad_output.shape(0)) {
    throw std::runtime_error(
        "Shape[0] of queries must be equal to shape[0] of grad_output.");
  }
  if (queries.shape(0) != query_gradients.shape(0)) {
    throw std::runtime_error(
        "Shape[0] of queries must be equal to shape[0] of query_gradients.");
  }
  if (query_gradients.shape(1) != 3) {
    throw std::runtime_error("Shape[1] of query_gradients has to be 3");
  }
  brute_force_triangle_query_gradient_impl(
      grad_output.data(), triangles.data(), queries.data(), triangles.shape(0),
      queries.shape(0), query_gradients.data(), epsilon, triangles.device_id(),
      stream, use_kahan);
}

auto brute_force_query_gradients(const Scalar_t &grad_output,
                                 const Vec3_t &vertices,
                                 const TriangleIdx_t &triangle_indices,
                                 const Vec3_t &queries, Vec3_t &query_gradients,
                                 float epsilon = 0.004F,
                                 const uint64_t stream = 0,
                                 const bool use_kahan = false) -> void {
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
  if (vertices.device_id() != query_gradients.device_id()) {
    throw std::runtime_error(
        "vertices and queries must be on the same CUDA device.");
  }
  if (queries.shape(0) != query_gradients.shape(0)) {
    throw std::runtime_error(
        "Shape[0] of queries must be equal to shape[0] of query_gradients");
  }
  if (query_gradients.shape(1) != 3) {
    throw std::runtime_error("Shape[1] of query_gradients has to be 3");
  }
  brute_force_mesh_query_gradient_impl(
      grad_output.data(), vertices.data(), triangle_indices.data(),
      queries.data(), triangle_indices.shape(0), queries.shape(0),
      vertices.shape(0), query_gradients.data(), epsilon, vertices.device_id(),
      stream, use_kahan);
}

// PAYLOAD GRADIENTS
auto brute_force_payload_gradients(
    const Payload_t &grad_output, const Vec3_t &points,
    const Vec3_t &scaled_normals, const Vec3_t &queries,
    Payload_t &out_gradients, float epsilon = 0.004F, const uint64_t stream = 0,
    const bool use_kahan = false) -> void {
  if (points.device_id() != scaled_normals.device_id()) {
    throw std::runtime_error(
        "points and scaled_normals must be on the same CUDA device.");
  }
  if (points.device_id() != grad_output.device_id()) {
    throw std::runtime_error(
        "points and grad_output must be on the same CUDA device.");
  }
  if (points.device_id() != queries.device_id()) {
    throw std::runtime_error(
        "points and queries must be on the same CUDA device.");
  }
  if (points.device_id() != out_gradients.device_id()) {
    throw std::runtime_error(
        "points and out_gradients must be on the same CUDA device.");
  }
  if (points.shape(0) != scaled_normals.shape(0)) {
    throw std::runtime_error(
        "Shape of points must be equal to shape of normals.");
  }
  if (points.shape(0) != out_gradients.shape(0)) {
    throw std::runtime_error(
        "Shape[0] of points must be equal to shape[0] of out_gradients.");
  }
  if (queries.shape(0) != grad_output.shape(0)) {
    throw std::runtime_error(
        "Shape[0] of queries must be equal to shape[0] of grad_output.");
  }
  if (grad_output.shape(1) != out_gradients.shape(1)) {
    throw std::runtime_error(
        "Shape[1] of grad_output must be equal to shape[1] of out_gradients.");
  }
  brute_force_point_normal_payload_gradient_impl(
      grad_output.data(), points.data(), scaled_normals.data(), queries.data(),
      points.shape(0), queries.shape(0), grad_output.shape(1),
      out_gradients.data(), epsilon, points.device_id(), stream, use_kahan);
}

auto brute_force_payload_gradients(
    const Payload_t &grad_output, const Vec3_t &vertices,
    const TriangleIdx_t &triangle_indices, const Vec3_t &queries,
    Payload_t &out_gradients, float epsilon = 0.004F, const uint64_t stream = 0,
    const bool use_kahan = false) -> void {
  if (vertices.device_id() != triangle_indices.device_id()) {
    throw std::runtime_error(
        "vertices and triangle_indices must be on the same CUDA device.");
  }
  if (vertices.device_id() != grad_output.device_id()) {
    throw std::runtime_error(
        "vertices and grad_output must be on the same CUDA device.");
  }
  if (vertices.device_id() != queries.device_id()) {
    throw std::runtime_error(
        "vertices and queries must be on the same CUDA device.");
  }
  if (vertices.device_id() != out_gradients.device_id()) {
    throw std::runtime_error(
        "vertices and out_gradients must be on the same CUDA device.");
  }
  if (triangle_indices.shape(0) != out_gradients.shape(0)) {
    throw std::runtime_error(
        "Shape[0] of triangle_indices must equal shape[0] of out_gradients.");
  }
  if (queries.shape(0) != grad_output.shape(0)) {
    throw std::runtime_error(
        "Shape[0] of queries must equal shape[0] of grad_output.");
  }
  if (grad_output.shape(1) != out_gradients.shape(1)) {
    throw std::runtime_error(
        "Shape[1] of grad_output must equal shape[1] of out_gradients.");
  }
  brute_force_mesh_payload_gradient_impl(
      grad_output.data(), vertices.data(), triangle_indices.data(),
      queries.data(), triangle_indices.shape(0), vertices.shape(0),
      queries.shape(0), grad_output.shape(1), out_gradients.data(), epsilon,
      vertices.device_id(), stream, use_kahan);
}

auto brute_force_payload_gradients(
    const Payload_t &grad_output, const Triangle_t &triangles,
    const Vec3_t &queries, Payload_t &out_gradients, float epsilon = 0.004F,
    const uint64_t stream = 0, const bool use_kahan = false) -> void {
  if (triangles.device_id() != grad_output.device_id()) {
    throw std::runtime_error(
        "triangles and grad_output must be on the same CUDA device.");
  }
  if (triangles.device_id() != queries.device_id()) {
    throw std::runtime_error(
        "triangles and queries must be on the same CUDA device.");
  }
  if (triangles.device_id() != out_gradients.device_id()) {
    throw std::runtime_error(
        "triangles and out_gradients must be on the same CUDA device.");
  }
  if (triangles.shape(0) != out_gradients.shape(0)) {
    throw std::runtime_error(
        "Shape[0] of triangles must equal shape[0] of out_gradients.");
  }
  if (queries.shape(0) != grad_output.shape(0)) {
    throw std::runtime_error(
        "Shape[0] of queries must equal shape[0] of grad_output.");
  }
  if (grad_output.shape(1) != out_gradients.shape(1)) {
    throw std::runtime_error(
        "Shape[1] of grad_output must equal shape[1] of out_gradients.");
  }
  brute_force_triangle_payload_gradient_impl(
      grad_output.data(), triangles.data(), queries.data(), triangles.shape(0),
      queries.shape(0), grad_output.shape(1), out_gradients.data(), epsilon,
      triangles.device_id(), stream, use_kahan);
}

// PAYLOAD GRADIENTS WRT GEOMETRY
auto brute_force_gradients_with_payload(
    const Payload_t &grad_output, const Vec3_t &points,
    const Vec3_t &scaled_normals, const Payload_t &payload,
    const Vec3_t &queries, GradResult_t &output_gradients,
    SingleScalar_t &epsilon_gradient, float epsilon = 0.004F,
    const uint64_t stream = 0, const bool use_kahan = false) -> void {
  if (points.device_id() != scaled_normals.device_id()) {
    throw std::runtime_error(
        "points and scaled_normals must be on the same CUDA device.");
  }
  if (points.device_id() != payload.device_id()) {
    throw std::runtime_error(
        "points and payload must be on the same CUDA device.");
  }
  if (points.device_id() != grad_output.device_id()) {
    throw std::runtime_error(
        "points and grad_output must be on the same CUDA device.");
  }
  if (points.device_id() != queries.device_id()) {
    throw std::runtime_error(
        "points and queries must be on the same CUDA device.");
  }
  if (points.device_id() != output_gradients.device_id()) {
    throw std::runtime_error(
        "points and output_gradients must be on the same CUDA device.");
  }
  if (points.shape(0) != scaled_normals.shape(0)) {
    throw std::runtime_error(
        "Shape of points must equal shape of scaled_normals.");
  }
  if (points.shape(0) != payload.shape(0)) {
    throw std::runtime_error(
        "Shape[0] of points must equal shape[0] of payload.");
  }
  if (payload.shape(1) != grad_output.shape(1)) {
    throw std::runtime_error(
        "Shape[1] of payload must equal shape[1] of grad_output.");
  }
  if (queries.shape(0) != grad_output.shape(0)) {
    throw std::runtime_error(
        "Shape[0] of queries must equal shape[0] of grad_output.");
  }
  if (points.shape(0) != output_gradients.shape(0)) {
    throw std::runtime_error(
        "Shape[0] of points must equal shape[0] of output_gradients.");
  }
  if (output_gradients.shape(1) != 2 || output_gradients.shape(2) != 3) {
    throw std::runtime_error("Shape of output_gradients must be (N, 2, 3).");
  }
  brute_force_point_normal_geometry_gradient_with_payload_impl(
      grad_output.data(), points.data(), scaled_normals.data(), payload.data(),
      queries.data(), points.shape(0), queries.shape(0), payload.shape(1),
      output_gradients.data(), epsilon_gradient.data(), epsilon,
      points.device_id(), stream, use_kahan);
}

auto brute_force_gradients_with_payload(
    const Payload_t &grad_output, const Vec3_t &vertices,
    const TriangleIdx_t &triangle_indices, const Payload_t &payload,
    const Vec3_t &queries, VertexGradResult_t &output_gradients,
    SingleScalar_t &epsilon_gradient, float epsilon = 0.004F,
    const uint64_t stream = 0, const bool use_kahan = false) -> void {
  if (vertices.device_id() != triangle_indices.device_id()) {
    throw std::runtime_error(
        "vertices and triangle_indices must be on the same CUDA device.");
  }
  if (vertices.device_id() != payload.device_id()) {
    throw std::runtime_error(
        "vertices and payload must be on the same CUDA device.");
  }
  if (vertices.device_id() != grad_output.device_id()) {
    throw std::runtime_error(
        "vertices and grad_output must be on the same CUDA device.");
  }
  if (vertices.device_id() != queries.device_id()) {
    throw std::runtime_error(
        "vertices and queries must be on the same CUDA device.");
  }
  if (vertices.device_id() != output_gradients.device_id()) {
    throw std::runtime_error(
        "vertices and output_gradients must be on the same CUDA device.");
  }
  if (triangle_indices.shape(0) != payload.shape(0)) {
    throw std::runtime_error(
        "Shape[0] of triangle_indices must equal shape[0] of payload.");
  }
  if (payload.shape(1) != grad_output.shape(1)) {
    throw std::runtime_error(
        "Shape[1] of payload must equal shape[1] of grad_output.");
  }
  if (queries.shape(0) != grad_output.shape(0)) {
    throw std::runtime_error(
        "Shape[0] of queries must equal shape[0] of grad_output.");
  }
  if (vertices.shape(0) != output_gradients.shape(0)) {
    throw std::runtime_error(
        "Shape[0] of vertices must equal shape[0] of output_gradients.");
  }
  if (output_gradients.shape(1) != 3) {
    throw std::runtime_error("Shape[1] of output_gradients must be 3.");
  }
  brute_force_mesh_geometry_gradient_with_payload_impl(
      grad_output.data(), vertices.data(), triangle_indices.data(),
      payload.data(), queries.data(), triangle_indices.shape(0),
      queries.shape(0), vertices.shape(0), payload.shape(1),
      output_gradients.data(), epsilon_gradient.data(), epsilon,
      vertices.device_id(), stream, use_kahan);
}

auto brute_force_gradients_with_payload(
    const Payload_t &grad_output, const Triangle_t &triangles,
    const Payload_t &payload, const Vec3_t &queries,
    GradResult_t &output_gradients, SingleScalar_t &epsilon_gradient,
    float epsilon = 0.004F, const uint64_t stream = 0,
    const bool use_kahan = false) -> void {
  if (triangles.device_id() != payload.device_id()) {
    throw std::runtime_error(
        "triangles and payload must be on the same CUDA device.");
  }
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
  if (triangles.shape(0) != payload.shape(0)) {
    throw std::runtime_error(
        "Shape[0] of triangles must equal shape[0] of payload.");
  }
  if (payload.shape(1) != grad_output.shape(1)) {
    throw std::runtime_error(
        "Shape[1] of payload must equal shape[1] of grad_output.");
  }
  if (queries.shape(0) != grad_output.shape(0)) {
    throw std::runtime_error(
        "Shape[0] of queries must equal shape[0] of grad_output.");
  }
  if (triangles.shape(0) != output_gradients.shape(0)) {
    throw std::runtime_error(
        "Shape[0] of triangles must equal shape[0] of output_gradients.");
  }
  if (output_gradients.shape(1) != 3 || output_gradients.shape(2) != 3) {
    throw std::runtime_error("Shape of output_gradients must be (N, 3, 3).");
  }
  brute_force_triangle_geometry_gradient_with_payload_impl(
      grad_output.data(), triangles.data(), payload.data(), queries.data(),
      triangles.shape(0), queries.shape(0), payload.shape(1),
      output_gradients.data(), epsilon_gradient.data(), epsilon,
      triangles.device_id(), stream, use_kahan);
}

// PAYLOAD GRADIENTS WRT QUERIES
auto brute_force_query_gradients_with_payload(
    const Payload_t &grad_output, const Vec3_t &points,
    const Vec3_t &scaled_normals, const Payload_t &payload,
    const Vec3_t &queries, Vec3_t &query_gradients, float epsilon = 0.004F,
    const uint64_t stream = 0, const bool use_kahan = true) -> void {
  if (points.device_id() != scaled_normals.device_id()) {
    throw std::runtime_error(
        "points and scaled_normals must be on the same CUDA device.");
  }
  if (points.device_id() != payload.device_id()) {
    throw std::runtime_error(
        "points and payload must be on the same CUDA device.");
  }
  if (points.device_id() != grad_output.device_id()) {
    throw std::runtime_error(
        "points and grad_output must be on the same CUDA device.");
  }
  if (points.device_id() != queries.device_id()) {
    throw std::runtime_error(
        "points and queries must be on the same CUDA device.");
  }
  if (points.device_id() != query_gradients.device_id()) {
    throw std::runtime_error(
        "points and query_gradients must be on the same CUDA device.");
  }
  if (points.shape(0) != scaled_normals.shape(0)) {
    throw std::runtime_error(
        "Shape of points must equal shape of scaled_normals.");
  }
  if (points.shape(0) != payload.shape(0)) {
    throw std::runtime_error(
        "Shape[0] of points must equal shape[0] of payload.");
  }
  if (payload.shape(1) != grad_output.shape(1)) {
    throw std::runtime_error(
        "Shape[1] of payload must equal shape[1] of grad_output.");
  }
  if (queries.shape(0) != grad_output.shape(0)) {
    throw std::runtime_error(
        "Shape[0] of queries must equal shape[0] of grad_output.");
  }
  if (queries.shape(0) != query_gradients.shape(0)) {
    throw std::runtime_error(
        "Shape[0] of queries must equal shape[0] of query_gradients.");
  }
  if (query_gradients.shape(1) != 3) {
    throw std::runtime_error("Shape[1] of query_gradients must be 3.");
  }
  brute_force_point_normal_query_gradient_with_payload_impl(
      grad_output.data(), points.data(), scaled_normals.data(), payload.data(),
      queries.data(), points.shape(0), queries.shape(0), payload.shape(1),
      query_gradients.data(), epsilon, points.device_id(), stream, use_kahan);
}

auto brute_force_query_gradients_with_payload(
    const Payload_t &grad_output, const Vec3_t &vertices,
    const TriangleIdx_t &triangle_indices, const Payload_t &payload,
    const Vec3_t &queries, Vec3_t &query_gradients, float epsilon = 0.004F,
    const uint64_t stream = 0, const bool use_kahan = true) -> void {
  if (vertices.device_id() != triangle_indices.device_id()) {
    throw std::runtime_error(
        "vertices and triangle_indices must be on the same CUDA device.");
  }
  if (vertices.device_id() != payload.device_id()) {
    throw std::runtime_error(
        "vertices and payload must be on the same CUDA device.");
  }
  if (vertices.device_id() != grad_output.device_id()) {
    throw std::runtime_error(
        "vertices and grad_output must be on the same CUDA device.");
  }
  if (vertices.device_id() != queries.device_id()) {
    throw std::runtime_error(
        "vertices and queries must be on the same CUDA device.");
  }
  if (vertices.device_id() != query_gradients.device_id()) {
    throw std::runtime_error(
        "vertices and query_gradients must be on the same CUDA device.");
  }
  if (triangle_indices.shape(0) != payload.shape(0)) {
    throw std::runtime_error(
        "Shape[0] of triangle_indices must equal shape[0] of payload.");
  }
  if (payload.shape(1) != grad_output.shape(1)) {
    throw std::runtime_error(
        "Shape[1] of payload must equal shape[1] of grad_output.");
  }
  if (queries.shape(0) != grad_output.shape(0)) {
    throw std::runtime_error(
        "Shape[0] of queries must equal shape[0] of grad_output.");
  }
  if (queries.shape(0) != query_gradients.shape(0)) {
    throw std::runtime_error(
        "Shape[0] of queries must equal shape[0] of query_gradients.");
  }
  if (query_gradients.shape(1) != 3) {
    throw std::runtime_error("Shape[1] of query_gradients must be 3.");
  }
  brute_force_mesh_query_gradient_with_payload_impl(
      grad_output.data(), vertices.data(), triangle_indices.data(),
      payload.data(), queries.data(), triangle_indices.shape(0),
      queries.shape(0), vertices.shape(0), payload.shape(1),
      query_gradients.data(), epsilon, vertices.device_id(), stream, use_kahan);
}

auto brute_force_query_gradients_with_payload(
    const Payload_t &grad_output, const Triangle_t &triangles,
    const Payload_t &payload, const Vec3_t &queries, Vec3_t &query_gradients,
    float epsilon = 0.004F, const uint64_t stream = 0,
    const bool use_kahan = true) -> void {
  if (triangles.device_id() != payload.device_id()) {
    throw std::runtime_error(
        "triangles and payload must be on the same CUDA device.");
  }
  if (triangles.device_id() != grad_output.device_id()) {
    throw std::runtime_error(
        "triangles and grad_output must be on the same CUDA device.");
  }
  if (triangles.device_id() != queries.device_id()) {
    throw std::runtime_error(
        "triangles and queries must be on the same CUDA device.");
  }
  if (triangles.device_id() != query_gradients.device_id()) {
    throw std::runtime_error(
        "triangles and query_gradients must be on the same CUDA device.");
  }
  if (triangles.shape(0) != payload.shape(0)) {
    throw std::runtime_error(
        "Shape[0] of triangles must equal shape[0] of payload.");
  }
  if (payload.shape(1) != grad_output.shape(1)) {
    throw std::runtime_error(
        "Shape[1] of payload must equal shape[1] of grad_output.");
  }
  if (queries.shape(0) != grad_output.shape(0)) {
    throw std::runtime_error(
        "Shape[0] of queries must equal shape[0] of grad_output.");
  }
  if (queries.shape(0) != query_gradients.shape(0)) {
    throw std::runtime_error(
        "Shape[0] of queries must equal shape[0] of query_gradients.");
  }
  if (query_gradients.shape(1) != 3) {
    throw std::runtime_error("Shape[1] of query_gradients must be 3.");
  }
  brute_force_triangle_query_gradient_with_payload_impl(
      grad_output.data(), triangles.data(), payload.data(), queries.data(),
      triangles.shape(0), queries.shape(0), payload.shape(1),
      query_gradients.data(), epsilon, triangles.device_id(), stream,
      use_kahan);
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
               GradResult_t &output_gradients, SingleScalar_t &epsilon_gradient,
               float beta = -1.F, float epsilon = 0.004F,
               const uint64_t stream = 0) -> void {
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
                   output_gradients.data(), epsilon_gradient.data(), beta,
                   epsilon, stream);
  }

  // Mesh
  auto compute(const Vec3_t &vertices, const TriangleIdx_t &triangle_indices,
               VertexGradResult_t &output_gradients,
               SingleScalar_t &epsilon_gradient, float beta = -1.F,
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
                   triangle_indices.shape(0), output_gradients.data(),
                   epsilon_gradient.data(), beta, epsilon, stream);
  }
  // Triangles
  auto compute(const Triangle_t &triangles, GradResult_t &output_gradients,
               SingleScalar_t &epsilon_gradient, float beta = -1.F,
               float epsilon = 0.004F, const uint64_t stream = 0) -> void {
    if (triangles.device_id() != output_gradients.device_id()) {
      throw std::runtime_error(
          "triangles and output_gradients must be on the same CUDA device.");
    }
    if (triangles.shape(0) != output_gradients.shape(0)) {
      throw std::runtime_error("Shape[0] of triangles must be equal to "
                               "shape[0] of output_gradients");
    }

    m_impl.compute(triangles.data(), triangles.shape(0),
                   output_gradients.data(), epsilon_gradient.data(), beta,
                   epsilon, stream);
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

  auto compute_query_gradients(const Scalar_t &grad_output,
                               const Vec3_t &queries,
                               const Vec3_t &out_query_gradients,
                               const float beta = -1.F,
                               const float epsilon = 0.004F,
                               const size_t stream = 0) -> void {
    if (is_backend_triangle) {
      m_impl_tri->compute_query_gradients(
          grad_output.data(), queries.data(), queries.shape(0),
          out_query_gradients.data(), beta, epsilon, stream);
    } else {
      m_impl_pn->compute_query_gradients(
          grad_output.data(), queries.data(), queries.shape(0),
          out_query_gradients.data(), beta, epsilon, stream);
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

#include "payload_backend.h"
#include <format>
#include <memory>
#include <stdexcept>

// =========================================================================
// Type-erasure layer for PayloadWindingNumbersBackend.
//
// The backend is templated on both Geometry (PointNormal / Triangle) and D
// (one of 9 buckets). The Python-facing PayloadField must handle both at
// runtime, so we route every call through a small virtual interface.
//
// The virtual call overhead is irrelevant: compute() dispatches a ~1ms
// kernel launch, and compute_query_gradients the same. The interface pays
// for itself with zero measurable cost.
// =========================================================================

class IPayloadBackendBase {
public:
  virtual ~IPayloadBackendBase() = default;

  [[nodiscard]] virtual auto dimension() const -> uint32_t = 0;

  virtual auto compute(const float *queries, size_t query_count,
                       float *out_payload, float beta, float epsilon,
                       size_t stream, bool use_kahan = false) const -> void = 0;

  virtual auto compute_query_gradients(const float *grad_output,
                                       const float *queries, size_t query_count,
                                       float *out_query_gradients, float beta,
                                       float epsilon, size_t stream,
                                       bool use_kahan = false) const
      -> void = 0;

  [[nodiscard]] virtual auto dump() const -> std::string = 0;
};

template <IsGeometry Geometry, uint32_t D_bucket>
class PayloadBackendImpl final : public IPayloadBackendBase {
public:
  PayloadBackendImpl(
      std::unique_ptr<PayloadWindingNumbersBackend<Geometry, D_bucket>> impl,
      uint32_t d_user)
      : m_impl(std::move(impl)), m_dim_user(d_user) {}

  [[nodiscard]] auto dimension() const -> uint32_t override {
    return m_dim_user;
  }

  auto compute(const float *queries, size_t query_count, float *out_payload,
               float beta, float epsilon, size_t stream,
               bool use_kahan = false) const -> void override {
    m_impl->compute_payload(queries, query_count, out_payload, beta, epsilon,
                            stream, use_kahan);
  }

  auto compute_query_gradients(const float *grad_output, const float *queries,
                               size_t query_count, float *out_query_gradients,
                               float beta, float epsilon, size_t stream,
                               bool use_kahan = false) const -> void override {
    m_impl->compute_payload_query_gradients(grad_output, queries, query_count,
                                            out_query_gradients, beta, epsilon,
                                            stream, use_kahan);
  }

  [[nodiscard]] auto dump() const -> std::string override {
    return m_impl->dump();
  }

private:
  std::unique_ptr<PayloadWindingNumbersBackend<Geometry, D_bucket>> m_impl;
  uint32_t m_dim_user;
};

// Returns the smallest supported bucket >= d_user.
// Throws if d_user is outside [1, 128].
auto resolve_payload_bucket(uint64_t d_user) -> uint32_t {
  if (d_user < 1 || d_user > 128) {
    throw std::runtime_error(std::format(
        "Payload dimensionality D={} is out of range. Must be in [1, 128].",
        d_user));
  }
  return payload_d_bucket(static_cast<uint32_t>(d_user)); // existing helper
}

template <typename F>
auto dispatch_payload_dim(uint32_t d, F &&f)
    -> std::unique_ptr<IPayloadBackendBase> {
  switch (d) {
  case 1:
    return f.template operator()<1>();
  case 2:
    return f.template operator()<2>();
  case 3:
    return f.template operator()<3>();
  case 4:
    return f.template operator()<4>();
  case 8:
    return f.template operator()<8>();
  case 16:
    return f.template operator()<16>();
  case 32:
    return f.template operator()<32>();
  case 64:
    return f.template operator()<64>();
  case 128:
    return f.template operator()<128>();
  default:
    throw std::runtime_error("dispatch_payload_dim: unreachable");
  }
}

// =========================================================================
// PayloadField — user-facing engine
// =========================================================================

class PayloadField {
public:
  // --- Triangle soup ---
  PayloadField(const Triangle_t &triangles, const Payload_t &payload,
               const uint64_t stream = 0)
      : m_impl{make_triangle_backend(triangles, payload, stream)} {}

  // --- Shared-vertex mesh ---
  PayloadField(const Vec3_t &vertices, const TriangleIdx_t &triangle_indices,
               const Payload_t &payload, const uint64_t stream = 0)
      : m_impl{make_mesh_backend(vertices, triangle_indices, payload, stream)} {
  }

  // --- Point cloud ---
  PayloadField(const Vec3_t &points, const Vec3_t &scaled_normals,
               const Payload_t &payload, const uint64_t stream = 0)
      : m_impl{make_point_backend(points, scaled_normals, payload, stream)} {}

  // --- Forward ---
  auto compute(const Vec3_t &queries, Payload_t &out_payload, float beta = -1.F,
               float epsilon = 0.004F, const uint64_t stream = 0,
               const bool use_kahan = false) const -> void {
    if (out_payload.shape(0) != queries.shape(0)) {
      throw std::runtime_error(
          "Shape[0] of queries must equal shape[0] of out_payload.");
    }
    if (out_payload.shape(1) != m_impl->dimension()) {
      throw std::runtime_error(
          std::format("Shape[1] of out_payload ({}) must equal payload D ({}).",
                      out_payload.shape(1), m_impl->dimension()));
    }
    m_impl->compute(queries.data(), queries.shape(0), out_payload.data(), beta,
                    epsilon, stream, use_kahan);
  }

  // --- Query gradients ---
  auto compute_query_gradients(const Payload_t &grad_output,
                               const Vec3_t &queries,
                               Vec3_t &out_query_gradients, float beta = -1.F,
                               float epsilon = 0.004F,
                               const uint64_t stream = 0, bool use_kahan = false) const -> void {
    if (grad_output.shape(0) != queries.shape(0)) {
      throw std::runtime_error(
          "Shape[0] of grad_output must equal shape[0] of queries.");
    }
    if (grad_output.shape(1) != m_impl->dimension()) {
      throw std::runtime_error(
          std::format("Shape[1] of grad_output ({}) must equal payload D ({}).",
                      grad_output.shape(1), m_impl->dimension()));
    }
    if (out_query_gradients.shape(0) != queries.shape(0)) {
      throw std::runtime_error(
          "Shape[0] of out_query_gradients must equal shape[0] of queries.");
    }
    if (out_query_gradients.shape(1) != 3) {
      throw std::runtime_error("Shape[1] of out_query_gradients must be 3.");
    }
    m_impl->compute_query_gradients(
        grad_output.data(), queries.data(), queries.shape(0),
        out_query_gradients.data(), beta, epsilon, stream, use_kahan);
  }

  [[nodiscard]] auto dump() const -> std::string { return m_impl->dump(); }

  [[nodiscard]] auto dimension() const -> uint32_t {
    return m_impl->dimension();
  }

private:
  static auto make_triangle_backend(const Triangle_t &triangles,
                                    const Payload_t &payload, uint64_t stream)
      -> std::unique_ptr<IPayloadBackendBase> {
    const auto d_user = static_cast<uint32_t>(payload.shape(1));
    const uint32_t d_bucket = resolve_payload_bucket(d_user);
    return dispatch_payload_dim(
        d_bucket, [&]<uint32_t D>() -> std::unique_ptr<IPayloadBackendBase> {
          auto impl =
              PayloadWindingNumbersBackend<Triangle, D>::CreateFromTriangles(
                  triangles.data(), payload.data(), triangles.shape(0), d_user,
                  triangles.device_id(), stream);
          return std::make_unique<PayloadBackendImpl<Triangle, D>>(
              std::move(impl), d_user);
        });
  }

  static auto make_mesh_backend(const Vec3_t &vertices,
                                const TriangleIdx_t &triangle_indices,
                                const Payload_t &payload, uint64_t stream)
      -> std::unique_ptr<IPayloadBackendBase> {
    const auto d_user = static_cast<uint32_t>(payload.shape(1));
    const uint32_t d_bucket = resolve_payload_bucket(d_user);
    return dispatch_payload_dim(
        d_bucket, [&]<uint32_t D>() -> std::unique_ptr<IPayloadBackendBase> {
          auto impl = PayloadWindingNumbersBackend<Triangle, D>::CreateFromMesh(
              vertices.data(), vertices.shape(0), triangle_indices.data(),
              payload.data(), triangle_indices.shape(0), d_user,
              vertices.device_id(), stream);
          return std::make_unique<PayloadBackendImpl<Triangle, D>>(
              std::move(impl), d_user);
        });
  }

  static auto make_point_backend(const Vec3_t &points,
                                 const Vec3_t &scaled_normals,
                                 const Payload_t &payload, uint64_t stream)
      -> std::unique_ptr<IPayloadBackendBase> {
    const auto d_user = static_cast<uint32_t>(payload.shape(1));
    const uint32_t d_bucket = resolve_payload_bucket(d_user);
    return dispatch_payload_dim(
        d_bucket, [&]<uint32_t D>() -> std::unique_ptr<IPayloadBackendBase> {
          auto impl =
              PayloadWindingNumbersBackend<PointNormal, D>::CreateFromPoints(
                  points.data(), scaled_normals.data(), payload.data(),
                  points.shape(0), d_user, points.device_id(), stream);
          return std::make_unique<PayloadBackendImpl<PointNormal, D>>(
              std::move(impl), d_user);
        });
  }

  std::unique_ptr<IPayloadBackendBase> m_impl;
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
                   kernel, expressed as a fraction of the scene's bounding box max dimension.
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

  // Reusable docstring for the `epsilon_gradient` output.
  constexpr const char *EPSILON_GRADIENT_DOC = R"doc(
                `epsilon_gradient` : Array
                    A (1,) float32 CUDA array pre-allocated by the caller on the
                    same CUDA stream as the other outputs.
                    Will be filled with the gradient of the loss with respect to
                    the regularization scale `epsilon` (dL/depsilon).
                    The result is a single scalar because `epsilon` is a global
                    scalar shared by the entire winding-number field, so all
                    per-query contributions are accumulated into this one entry.
  )doc";

  // Reusable docstring for the `use_kahan` flag.
  constexpr const char *USE_KAHAN_DOC = R"doc(
               `use_kahan` : bool, optional
                   Use compensated (Kahan) summation for numerical stability.
                   Reduces floating-point roundoff noise at a modest performance
                   cost. The default is chosen per pass to be the best tradeoff.
  )doc";

  // Reusable docstring for the payload field output.
  constexpr const char *PAYLOAD_FIELD_DOC = R"doc(
               A *payload field* interpolates a per-primitive vector payload
               `v_i in R^D` into 3D space using the winding number of each
               primitive as its blending weight:

                   w_V(x) = sum_i  v_i * w(x | g_i, epsilon)

               For D = 1 and v_i = 1 for all i, this reduces exactly to the
               scalar winding number field. For general D and v_i, it is a
               smooth D-dimensional field that respects the geometry's
               inside/outside structure: far outside the field decays to 0,
               deep inside it converges to sum_i v_i.

               The field shares the winding number's robustness to broken
               geometry, its harmonicity away from the sources, and its
               invariance under the global regularization length scale.
  )doc";

  // Reusable docstring for the D-vector `grad_output` argument.
  constexpr const char *PAYLOAD_ADJOINT_DOC = R"doc(
                `grad_output` : Array
                    (M, D) float32 CUDA array representing the gradient of the
                    loss with respect to the payload field at each query
                    location (dL/dW). Row `j` is the D-vector adjoint for query
                    `j`, and must be aligned index-for-index with `queries`.
  )doc";

  // ===========================================================================
  // brute_force_winding_numbers_*
  // ===========================================================================

  // --- Point-normal ---------------------------------------------------------
  m.def("brute_force_winding_numbers_point_normal",
        nb::overload_cast<const Vec3_t &, const Vec3_t &, const Vec3_t &,
                          Scalar_t &, float, uint64_t, const bool>(
            &brute_force_winding_numbers),
        "points"_a, "scaled_normals"_a, "queries"_a, "out_winding_numbers"_a,
        "epsilon"_a = 0.004F, "stream"_a = 0, "use_kahan"_a = true,
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
                "stream: int = 0, use_kahan: bool = true) -> "
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
            )doc" +
         USE_KAHAN_DOC)
            .c_str());

  // --- Mesh (shared vertices + index array) ---------------------------------
  m.def("brute_force_winding_numbers_mesh",
        nb::overload_cast<const Vec3_t &, const TriangleIdx_t &, const Vec3_t &,
                          Scalar_t &, float, const uint64_t, const bool>(
            &brute_force_winding_numbers),
        "vertices"_a, "triangle_indices"_a, "queries"_a,
        "out_winding_numbers"_a, "epsilon"_a = 0.004F, "stream"_a = 0,
        "use_kahan"_a = true,
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
                "stream: int = 0, use_kahan: bool = true) -> "
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
            )doc" +
         USE_KAHAN_DOC)
            .c_str());

  // --- Triangle soup --------------------------------------------------------
  m.def("brute_force_winding_numbers_triangle_soup",
        nb::overload_cast<const Triangle_t &, const Vec3_t &, Scalar_t &, float,
                          const uint64_t, const bool>(
            &brute_force_winding_numbers),
        "triangles"_a, "queries"_a, "out_winding_numbers"_a,
        "epsilon"_a = 0.004F, "stream"_a = 0, "use_kahan"_a = true,
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
                "int = 0, use_kahan: bool = true) -> None"),
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
            )doc" +
         USE_KAHAN_DOC)
            .c_str());

  // ===========================================================================
  // brute_force_winding_payloads_*
  //
  // Vector-valued extension of the winding number field: the per-primitive
  // payloads are blended into space by the winding number of each primitive.
  // ===========================================================================

  // --- Point-normal ---------------------------------------------------------
  m.def("brute_force_winding_payloads_point_normal",
        nb::overload_cast<const Vec3_t &, const Vec3_t &, const Payload_t &,
                          const Vec3_t &, Payload_t &, float, const uint64_t,
                          const bool>(&brute_force_winding_payloads),
        "points"_a, "scaled_normals"_a, "payload"_a, "queries"_a,
        "out_payload"_a, "epsilon"_a = 0.004F, "stream"_a = 0,
        "use_kahan"_a = false,
        nb::sig("def brute_force_winding_payloads_point_normal(points: "
                "winder.types.Array[winder.types.Shape[winder.types.N, "
                "typing.Literal[3]], winder.types.float32, "
                "winder.types.cuda], scaled_normals: "
                "winder.types.Array[winder.types.Shape[winder.types.N, "
                "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
                "payload: "
                "winder.types.Array[winder.types.Shape[winder.types.N, "
                "winder.types.D], winder.types.float32, winder.types.cuda], "
                "queries: "
                "winder.types.Array[winder.types.Shape[winder.types.M, "
                "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
                "out_payload: "
                "winder.types.Array[winder.types.Shape[winder.types.M, "
                "winder.types.D], winder.types.float32, winder.types.cuda], "
                "epsilon: float = 0.004, "
                "stream: int = 0, use_kahan: bool = false) -> None"),
        (std::string(R"doc(
                Computes the payload field at the given query locations on GPU
                for a point cloud with scaled normals.

                NOTE: Brute Force implementation: exact, but in `O(N*M*D)`!

                Parameters
                ----------
                `points` : Array
                    A (N, 3) float32 CUDA array of point positions.
                `scaled_normals` : Array
                    A (N, 3) float32 CUDA array of scaled normals.
                    The scaled normal direction is the orientation,
                    the scale the associated voronoi area.
                `payload` : Array
                    A (N, D) float32 CUDA array of per-primitive payloads.
                    Row `i` is the D-vector blended by the winding contribution
                    of primitive `i`.
                `queries` : Array
                    (M, 3) float32 CUDA array of query points for which the
                    payload field is evaluated.
                `out_payload` : Array
                    (M, D) float32 CUDA array pre-allocated by the caller.
                    Will be filled with `w_V(q_j) = sum_i v_i * w(q_j | g_i)`.

            )doc") +
         PAYLOAD_FIELD_DOC + EPSILON_DOC + R"doc(
                `stream`  : int, optional
                    The raw 64-bit identifier (handle) of a CUDA stream.
                    Allows enqueuing operations asynchronously within deep learning frameworks.
                    For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
                    Default is 0 (the default/null stream).
            )doc" +
         USE_KAHAN_DOC)
            .c_str());

  // --- Mesh ----------------------------------------------------------------
  m.def("brute_force_winding_payloads_mesh",
        nb::overload_cast<const Vec3_t &, const TriangleIdx_t &,
                          const Payload_t &, const Vec3_t &, Payload_t &, float,
                          const uint64_t, const bool>(
            &brute_force_winding_payloads),
        "vertices"_a, "triangle_indices"_a, "payload"_a, "queries"_a,
        "out_payload"_a, "epsilon"_a = 0.004F, "stream"_a = 0,
        "use_kahan"_a = false,
        nb::sig("def brute_force_winding_payloads_mesh(vertices: "
                "winder.types.Array[winder.types.Shape[winder.types.K, "
                "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
                "triangle_indices: "
                "winder.types.Array[winder.types.Shape[winder.types.N, "
                "typing.Literal[3]], winder.types.uint32, winder.types.cuda], "
                "payload: "
                "winder.types.Array[winder.types.Shape[winder.types.N, "
                "winder.types.D], winder.types.float32, winder.types.cuda], "
                "queries: "
                "winder.types.Array[winder.types.Shape[winder.types.M, "
                "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
                "out_payload: "
                "winder.types.Array[winder.types.Shape[winder.types.M, "
                "winder.types.D], winder.types.float32, winder.types.cuda], "
                "epsilon: float = 0.004, "
                "stream: int = 0, use_kahan: bool = false) -> None"),
        (std::string(R"doc(
                Computes the payload field at the given query locations on GPU
                for a triangle mesh with shared vertices and an index array.

                NOTE: Brute Force implementation: exact, but in `O(N*M*D)`!

                Parameters
                ----------
                `vertices` : Array
                    A (K, 3) float32 CUDA array holding K vertices.
                `triangle_indices`: Array
                    A (N, 3) index array, where each row defines the three
                    vertices of a triangle.
                    Note: Order the vertices counter-clockwise when seen from the front.
                `payload` : Array
                    A (N, D) float32 CUDA array of per-triangle payloads.
                    Row `i` is the D-vector blended by the winding contribution
                    of triangle `i`.
                `queries` : Array
                    (M, 3) float32 CUDA array of query points for which the
                    payload field is evaluated.
                `out_payload` : Array
                    (M, D) float32 CUDA array pre-allocated by the caller.
                    Will be filled with `w_V(q_j) = sum_i v_i * w(q_j | g_i)`.

            )doc") +
         PAYLOAD_FIELD_DOC + EPSILON_DOC + R"doc(
                `stream`  : int, optional
                    The raw 64-bit identifier (handle) of a CUDA stream.
                    Allows enqueuing operations asynchronously within deep learning frameworks.
                    For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
                    Default is 0 (the default/null stream).
            )doc" +
         USE_KAHAN_DOC)
            .c_str());

  // --- Triangle soup --------------------------------------------------------
  m.def("brute_force_winding_payloads_triangle_soup",
        nb::overload_cast<const Triangle_t &, const Payload_t &, const Vec3_t &,
                          Payload_t &, float, const uint64_t, const bool>(
            &brute_force_winding_payloads),
        "triangles"_a, "payload"_a, "queries"_a, "out_payload"_a,
        "epsilon"_a = 0.004F, "stream"_a = 0, "use_kahan"_a = false,
        nb::sig("def brute_force_winding_payloads_triangle_soup(triangles: "
                "winder.types.Array[winder.types.Shape[winder.types.N, "
                "typing.Literal[3], typing.Literal[3]], "
                "winder.types.float32, winder.types.cuda], "
                "payload: "
                "winder.types.Array[winder.types.Shape[winder.types.N, "
                "winder.types.D], winder.types.float32, winder.types.cuda], "
                "queries: "
                "winder.types.Array[winder.types.Shape[winder.types.M, "
                "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
                "out_payload: "
                "winder.types.Array[winder.types.Shape[winder.types.M, "
                "winder.types.D], winder.types.float32, winder.types.cuda], "
                "epsilon: float = 0.004, "
                "stream: int = 0, use_kahan: bool = false) -> None"),
        (std::string(R"doc(
                Computes the payload field at the given query locations on GPU
                for an explicit triangle soup.

                NOTE: Brute Force implementation: exact, but in `O(N*M*D)`!

                Parameters
                ----------
                `triangles` : Array
                    A (N, 3, 3) float32 CUDA array holding N triangles
                    which consists of 3 vertices.
                    Note: Order the vertices counter-clockwise when seen from the front.
                `payload` : Array
                    A (N, D) float32 CUDA array of per-triangle payloads.
                    Row `i` is the D-vector blended by the winding contribution
                    of triangle `i`.
                `queries` : Array
                    (M, 3) float32 CUDA array of query points for which the
                    payload field is evaluated.
                `out_payload` : Array
                    (M, D) float32 CUDA array pre-allocated by the caller.
                    Will be filled with `w_V(q_j) = sum_i v_i * w(q_j | g_i)`.

            )doc") +
         PAYLOAD_FIELD_DOC + EPSILON_DOC + R"doc(
                `stream`  : int, optional
                    The raw 64-bit identifier (handle) of a CUDA stream.
                    Allows enqueuing operations asynchronously within deep learning frameworks.
                    For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
                    Default is 0 (the default/null stream).
            )doc" +
         USE_KAHAN_DOC)
            .c_str());

  // ===========================================================================
  // brute_force_gradients_*
  // ===========================================================================

  // --- Point-normal ---------------------------------------------------------
  m.def(
      "brute_force_gradients_point_normal",
      nb::overload_cast<const Scalar_t &, const Vec3_t &, const Vec3_t &,
                        const Vec3_t &, GradResult_t &, SingleScalar_t &, float,
                        const uint64_t, bool>(&brute_force_gradients),
      "grad_output"_a, "points"_a, "scaled_normals"_a, "queries"_a,
      "out_gradients"_a, "epsilon_gradient"_a, "epsilon"_a = 0.004F,
      "stream"_a = 0, "use_kahan"_a = false,
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
              "winder.types.cuda], "
              "epsilon_gradient: "
              "winder.types.Array[winder.types.Shape[typing.Literal[1]], "
              "winder.types.float32, winder.types.cuda], "
              "epsilon: float = 0.004, "
              "stream: int = 0, use_kahan: bool = false) -> None"),
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
       EPSILON_GRADIENT_DOC + EPSILON_DOC + R"doc(
                `stream` : int, optional
                    The raw 64-bit identifier (handle) of a CUDA stream. 
                    Allows enqueuing operations asynchronously within deep learning frameworks.
                    For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
                    Default is 0 (the default/null stream).

            )doc" +
       USE_KAHAN_DOC)
          .c_str());

  // --- Mesh ----------------------------------------------------------------
  m.def(
      "brute_force_gradients_mesh",
      nb::overload_cast<const Scalar_t &, const Vec3_t &, const TriangleIdx_t &,
                        const Vec3_t &, VertexGradResult_t &, SingleScalar_t &,
                        float, const uint64_t, const bool>(
          &brute_force_gradients),
      "grad_output"_a, "vertices"_a, "triangle_indices"_a, "queries"_a,
      "out_gradients"_a, "epsilon_gradient"_a, "epsilon"_a = 0.004F,
      "stream"_a = 0, "use_kahan"_a = false,
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
              "epsilon_gradient: "
              "winder.types.Array[winder.types.Shape[typing.Literal[1]], "
              "winder.types.float32, winder.types.cuda], "
              "epsilon: float = 0.004, "
              "stream: int = 0, use_kahan: bool = false) -> None"),
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
       EPSILON_GRADIENT_DOC + EPSILON_DOC + R"doc(
                `stream` : int, optional
                    The raw 64-bit identifier (handle) of a CUDA stream. 
                    Allows enqueuing operations asynchronously within deep learning frameworks.
                    For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
                    Default is 0 (the default/null stream).
            )doc" +
       USE_KAHAN_DOC)
          .c_str());

  // --- Triangle soup -------------------------------------------------------
  m.def("brute_force_gradients_triangle_soup",
        nb::overload_cast<const Scalar_t &, const Triangle_t &, const Vec3_t &,
                          GradResult_t &, SingleScalar_t &, float,
                          const uint64_t, const bool>(&brute_force_gradients),
        "grad_output"_a, "triangles"_a, "queries"_a, "out_gradients"_a,
        "epsilon_gradient"_a, "epsilon"_a = 0.004F, "stream"_a = 0,
        "use_kahan"_a = false,
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
                "epsilon_gradient: "
                "winder.types.Array[winder.types.Shape[typing.Literal[1]], "
                "winder.types.float32, winder.types.cuda], "
                "epsilon: float = 0.004, "
                "stream: int "
                "= 0, use_kahan: bool = false) -> None"),
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
         EPSILON_GRADIENT_DOC + EPSILON_DOC + R"doc(
                `stream` : int, optional
                    The raw 64-bit identifier (handle) of a CUDA stream. 
                    Allows enqueuing operations asynchronously within deep learning frameworks.
                    For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
                    Default is 0 (the default/null stream).
            )doc" +
         USE_KAHAN_DOC)
            .c_str());

  // ===========================================================================
  // brute_force_query_gradients_*
  // ===========================================================================

  // --- Point-normal ---------------------------------------------------------
  m.def("brute_force_query_gradients_point_normal",
        nb::overload_cast<const Scalar_t &, const Vec3_t &, const Vec3_t &,
                          const Vec3_t &, Vec3_t &, float, const uint64_t,
                          const bool>(&brute_force_query_gradients),
        "grad_output"_a, "points"_a, "scaled_normals"_a, "queries"_a,
        "out_query_gradients"_a, "epsilon"_a = 0.004F, "stream"_a = 0,
        "use_kahan"_a = true,
        nb::sig("def brute_force_query_gradients_point_normal(grad_output: "
                "winder.types.Array[winder.types.Shape[winder.types.M], "
                "winder.types.float32, winder.types.cuda], "
                "points: "
                "winder.types.Array[winder.types.Shape[winder.types.N, "
                "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
                "scaled_normals: "
                "winder.types.Array[winder.types.Shape[winder.types.N, "
                "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
                "queries: "
                "winder.types.Array[winder.types.Shape[winder.types.M, "
                "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
                "out_query_gradients: "
                "winder.types.Array[winder.types.Shape[winder.types.M, "
                "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
                "epsilon: float = 0.004, "
                "stream: int = 0, use_kahan: bool = true) -> None"),
        (std::string(R"doc(
                Computes the gradient of the loss with respect to the query
                positions for a point cloud with scaled normals.

                NOTE: Brute Force implementation: exact, but in `O(N*M)`!

                This is the query-side counterpart of
                `brute_force_gradients_point_normal`. For each query position
                `q_j`, it computes:

                    dL/dq_j = g_j * sum_i dw_i(q_j) / dq_j
                            = -g_j * sum_i dw_i(q_j) / dp_i

                The two gradients are transposes of each other at the loss
                level, not negatives, so they must both be computed
                independently. Callers who need both gradients should call
                both functions; they share no state.

                Parameters
                ----------
                `grad_output` : Array
                    (M,) float32 CUDA array representing the gradient of the
                    loss with respect to the winding numbers at the query
                    locations (dL/dw). Aligned with `queries`.
                `points` : Array
                    (N, 3) float32 CUDA array of point positions.
                `scaled_normals` : Array
                    (N, 3) float32 CUDA array of scaled normals.
                    The scaled normal direction is the orientation,
                    the scale the associated voronoi area.
                `queries` : Array
                    (M, 3) float32 CUDA array of query points for which the
                    gradient is evaluated.
                `out_query_gradients` : Array
                    (M, 3) float32 CUDA array pre-allocated by the caller.
                    Will be filled with dL/dq for each query.
            )doc") +
         EPSILON_DOC + R"doc(
                `stream` : int, optional
                    The raw 64-bit identifier (handle) of a CUDA stream.
                    Allows enqueuing operations asynchronously within deep learning frameworks.
                    For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
                    Default is 0 (the default/null stream).
            )doc" +
         USE_KAHAN_DOC)
            .c_str());

  // --- Triangle soup --------------------------------------------------------
  m.def("brute_force_query_gradients_triangle_soup",
        nb::overload_cast<const Scalar_t &, const Triangle_t &, const Vec3_t &,
                          Vec3_t &, float, const uint64_t, const bool>(
            &brute_force_query_gradients),
        "grad_output"_a, "triangles"_a, "queries"_a, "out_query_gradients"_a,
        "epsilon"_a = 0.004F, "stream"_a = 0, "use_kahan"_a = true,
        nb::sig("def brute_force_query_gradients_triangle_soup(grad_output: "
                "winder.types.Array[winder.types.Shape[winder.types.M], "
                "winder.types.float32, winder.types.cuda], "
                "triangles: "
                "winder.types.Array[winder.types.Shape[winder.types.N, "
                "typing.Literal[3], typing.Literal[3]], "
                "winder.types.float32, winder.types.cuda], "
                "queries: "
                "winder.types.Array[winder.types.Shape[winder.types.M, "
                "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
                "out_query_gradients: "
                "winder.types.Array[winder.types.Shape[winder.types.M, "
                "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
                "epsilon: float = 0.004, "
                "stream: int = 0, use_kahan: bool = true) -> None"),
        (std::string(R"doc(
                Computes the gradient of the loss with respect to the query
                positions for an explicit triangle soup.

                NOTE: Brute Force implementation: exact, but in `O(N*M)`!

                This is the query-side counterpart of
                `brute_force_gradients_triangle_soup`. For each query q_j:

                    dL/dq_j = -g_j * (dW_j/dv0 + dW_j/dv1 + dW_j/dv2)

                The three vertex gradients are computed exactly as in the
                primitive-gradient kernel, then summed and negated.

                Parameters
                ----------
                `grad_output` : Array
                    (M,) float32 CUDA array representing the gradient of the
                    loss with respect to the winding numbers at the query
                    locations (dL/dw).
                `triangles` : Array
                    (N, 3, 3) float32 CUDA array holding N triangles.
                    Note: Order the vertices counter-clockwise when seen
                    from the front.
                `queries` : Array
                    (M, 3) float32 CUDA array of query points for which the
                    gradient is evaluated.
                `out_query_gradients` : Array
                    (M, 3) float32 CUDA array pre-allocated by the caller.
                    Will be filled with dL/dq for each query.
            )doc") +
         EPSILON_DOC + R"doc(
                `stream` : int, optional
                    The raw 64-bit identifier (handle) of a CUDA stream.
                    Allows enqueuing operations asynchronously within deep learning frameworks.
                    For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
                    Default is 0 (the default/null stream).
            )doc" +
         USE_KAHAN_DOC)
            .c_str());

  // --- Mesh ----------------------------------------------------------------
  m.def(
      "brute_force_query_gradients_mesh",
      nb::overload_cast<const Scalar_t &, const Vec3_t &, const TriangleIdx_t &,
                        const Vec3_t &, Vec3_t &, float, const uint64_t,
                        const bool>(&brute_force_query_gradients),
      "grad_output"_a, "vertices"_a, "triangle_indices"_a, "queries"_a,
      "out_query_gradients"_a, "epsilon"_a = 0.004F, "stream"_a = 0,
      "use_kahan"_a = true,
      nb::sig("def brute_force_query_gradients_mesh(grad_output: "
              "winder.types.Array[winder.types.Shape[winder.types.M], "
              "winder.types.float32, winder.types.cuda], "
              "vertices: "
              "winder.types.Array[winder.types.Shape[winder.types.K, "
              "typing.Literal[3]], "
              "winder.types.float32, winder.types.cuda], "
              "triangle_indices: "
              "winder.types.Array[winder.types.Shape[winder.types.N, "
              "typing.Literal[3]], winder.types.uint32, winder.types.cuda], "
              "queries: "
              "winder.types.Array[winder.types.Shape[winder.types.M, "
              "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
              "out_query_gradients: "
              "winder.types.Array[winder.types.Shape[winder.types.M, "
              "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
              "epsilon: float = 0.004, "
              "stream: int = 0, use_kahan: bool = true) -> None"),
      (std::string(R"doc(
                Computes the gradient of the loss with respect to the query
                positions for a mesh.

                NOTE: Brute Force implementation: exact, but in `O(N*M)`!

                This is the query-side counterpart of
                `brute_force_gradients_mesh`. For each query q_j:

                    dL/dq_j = -g_j * (dW_j/dv0 + dW_j/dv1 + dW_j/dv2)

                The three vertex gradients are computed exactly as in the
                primitive-gradient kernel, then summed and negated.

                Parameters
                ----------
                `grad_output` : Array
                    (M,) float32 CUDA array representing the gradient of the
                    loss with respect to the winding numbers at the query
                    locations (dL/dw).
                `vertices`  : Array
                    (K, 3) float32 CUDA array representing the shared vertices used in the triangles
                    for which the winding numbers and loss were computed.
                `triangle_indices`: Array
                    A (N, 3) index array, where each row defines the three vertices of a triangle.
                    Note: Order the vertices counter-clockwise when seen from the front. 
                `queries` : Array
                    (M, 3) float32 CUDA array of query points for which the
                    gradient is evaluated.
                `out_query_gradients` : Array
                    (M, 3) float32 CUDA array pre-allocated by the caller.
                    Will be filled with dL/dq for each query.
            )doc") +
       EPSILON_DOC + R"doc(
                `stream` : int, optional
                    The raw 64-bit identifier (handle) of a CUDA stream.
                    Allows enqueuing operations asynchronously within deep learning frameworks.
                    For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
                    Default is 0 (the default/null stream).
            )doc" +
       USE_KAHAN_DOC)
          .c_str());

  // PAYLOAD
  // Gradient wrt payload
  m.def("brute_force_payload_gradients_point_normal",
        nb::overload_cast<const Payload_t &, const Vec3_t &, const Vec3_t &,
                          const Vec3_t &, Payload_t &, float, const uint64_t,
                          const bool>(&brute_force_payload_gradients),
        "grad_output"_a, "points"_a, "scaled_normals"_a, "queries"_a,
        "out_gradients"_a, "epsilon"_a = 0.004F, "stream"_a = 0,
        "use_kahan"_a = false,
        nb::sig("def brute_force_payload_gradients_point_normal(grad_output: "
                "winder.types.Array[winder.types.Shape[winder.types.M, "
                "winder.types.D], winder.types.float32, winder.types.cuda], "
                "points: "
                "winder.types.Array[winder.types.Shape[winder.types.N, "
                "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
                "scaled_normals: "
                "winder.types.Array[winder.types.Shape[winder.types.N, "
                "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
                "queries: "
                "winder.types.Array[winder.types.Shape[winder.types.M, "
                "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
                "out_gradients: "
                "winder.types.Array[winder.types.Shape[winder.types.N, "
                "winder.types.D], winder.types.float32, winder.types.cuda], "
                "epsilon: float = 0.004, "
                "stream: int = 0, use_kahan: bool = false) -> None"),
        (std::string(R"doc(
                Computes the gradient of the loss with respect to the
                per-primitive payloads of a point cloud.

                NOTE: Brute Force implementation: exact, but in `O(N*M*D)`!

                For each primitive `i`, the output row `i` contains the
                D-vector

                    dL/dv_i = sum_j w_i(q_j) · g_j

                where `w_i(q_j)` is the winding contribution of primitive `i`
                at query `j` and `g_j` is the incoming D-vector adjoint at
                query `j`.

                Parameters
                ----------
                `grad_output` : Array
                    (M, D) float32 CUDA array representing the gradient of the
                    loss with respect to the payload field at each query
                    location (dL/dW). Row `j` is the D-vector adjoint for query
                    `j`, and must be aligned index-for-index with `queries`.
                `points` : Array
                    A (N, 3) float32 CUDA array of point positions.
                `scaled_normals` : Array
                    A (N, 3) float32 CUDA array of scaled normals.
                    The scaled normal direction is the orientation,
                    the scale the associated voronoi area.
                `queries` : Array
                    (M, 3) float32 CUDA array of query points for which the
                    payload field was evaluated.
                `out_gradients` : Array
                    (N, D) float32 CUDA array pre-allocated by the caller.
                    Will be filled with `dL/dv_i` for each primitive.
            )doc") +
         EPSILON_DOC + R"doc(
                `stream` : int, optional
                    The raw 64-bit identifier (handle) of a CUDA stream.
                    Allows enqueuing operations asynchronously within deep learning frameworks.
                    For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
                    Default is 0 (the default/null stream).
            )doc" +
         USE_KAHAN_DOC)
            .c_str());

  m.def("brute_force_payload_gradients_mesh",
        nb::overload_cast<const Payload_t &, const Vec3_t &,
                          const TriangleIdx_t &, const Vec3_t &, Payload_t &,
                          float, const uint64_t, const bool>(
            &brute_force_payload_gradients),
        "grad_output"_a, "vertices"_a, "triangle_indices"_a, "queries"_a,
        "out_gradients"_a, "epsilon"_a = 0.004F, "stream"_a = 0,
        "use_kahan"_a = false,
        nb::sig("def brute_force_payload_gradients_mesh(grad_output: "
                "winder.types.Array[winder.types.Shape[winder.types.M, "
                "winder.types.D], winder.types.float32, winder.types.cuda], "
                "vertices: "
                "winder.types.Array[winder.types.Shape[winder.types.K, "
                "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
                "triangle_indices: "
                "winder.types.Array[winder.types.Shape[winder.types.N, "
                "typing.Literal[3]], winder.types.uint32, winder.types.cuda], "
                "queries: "
                "winder.types.Array[winder.types.Shape[winder.types.M, "
                "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
                "out_gradients: "
                "winder.types.Array[winder.types.Shape[winder.types.N, "
                "winder.types.D], winder.types.float32, winder.types.cuda], "
                "epsilon: float = 0.004, "
                "stream: int = 0, use_kahan: bool = false) -> None"),
        (std::string(R"doc(
                Computes the gradient of the loss with respect to the
                per-triangle payloads of a shared-vertex mesh.

                NOTE: Brute Force implementation: exact, but in `O(N*M*D)`!

                The payload is defined per-triangle, so the output is also
                per-triangle: row `i` of `out_gradients` corresponds to the
                i-th triangle in `triangle_indices`, not to a shared vertex.
                Gradients are not accumulated back to the vertices.

                For each triangle `i`:

                    dL/dv_i = sum_j w_i(q_j) · g_j

                where `w_i(q_j)` is the winding contribution of triangle `i`
                at query `j` and `g_j` is the incoming D-vector adjoint.

                Parameters
                ----------
                `grad_output` : Array
                    (M, D) float32 CUDA array representing the gradient of the
                    loss with respect to the payload field at each query
                    location (dL/dW). Row `j` is the D-vector adjoint for query
                    `j`, aligned with `queries`.
                `vertices` : Array
                    A (K, 3) float32 CUDA array holding the shared vertices.
                `triangle_indices` : Array
                    A (N, 3) uint32 index array, where each row defines the
                    three vertices of a triangle.
                    Note: Order the vertices counter-clockwise when seen from the front.
                `queries` : Array
                    (M, 3) float32 CUDA array of query points for which the
                    payload field was evaluated.
                `out_gradients` : Array
                    (N, D) float32 CUDA array pre-allocated by the caller.
                    Will be filled with `dL/dv_i` for each triangle.
            )doc") +
         EPSILON_DOC + R"doc(
                `stream` : int, optional
                    The raw 64-bit identifier (handle) of a CUDA stream.
                    Allows enqueuing operations asynchronously within deep learning frameworks.
                    For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
                    Default is 0 (the default/null stream).
            )doc" +
         USE_KAHAN_DOC)
            .c_str());

  m.def("brute_force_payload_gradients_triangle_soup",
        nb::overload_cast<const Payload_t &, const Triangle_t &, const Vec3_t &,
                          Payload_t &, float, const uint64_t, const bool>(
            &brute_force_payload_gradients),
        "grad_output"_a, "triangles"_a, "queries"_a, "out_gradients"_a,
        "epsilon"_a = 0.004F, "stream"_a = 0, "use_kahan"_a = false,
        nb::sig("def brute_force_payload_gradients_triangle_soup(grad_output: "
                "winder.types.Array[winder.types.Shape[winder.types.M, "
                "winder.types.D], winder.types.float32, winder.types.cuda], "
                "triangles: "
                "winder.types.Array[winder.types.Shape[winder.types.N, "
                "typing.Literal[3], typing.Literal[3]], "
                "winder.types.float32, winder.types.cuda], "
                "queries: "
                "winder.types.Array[winder.types.Shape[winder.types.M, "
                "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
                "out_gradients: "
                "winder.types.Array[winder.types.Shape[winder.types.N, "
                "winder.types.D], winder.types.float32, winder.types.cuda], "
                "epsilon: float = 0.004, "
                "stream: int = 0, use_kahan: bool = false) -> None"),
        (std::string(R"doc(
                Computes the gradient of the loss with respect to the
                per-triangle payloads of an explicit triangle soup.

                NOTE: Brute Force implementation: exact, but in `O(N*M*D)`!

                For each triangle `i`:

                    dL/dv_i = sum_j w_i(q_j) · g_j

                where `w_i(q_j)` is the winding contribution of triangle `i`
                at query `j` and `g_j` is the incoming D-vector adjoint.

                Parameters
                ----------
                `grad_output` : Array
                    (M, D) float32 CUDA array representing the gradient of the
                    loss with respect to the payload field at each query
                    location (dL/dW). Row `j` is the D-vector adjoint for query
                    `j`, aligned with `queries`.
                `triangles` : Array
                    A (N, 3, 3) float32 CUDA array holding N triangles.
                    Note: Order the vertices counter-clockwise when seen from the front.
                `queries` : Array
                    (M, 3) float32 CUDA array of query points for which the
                    payload field was evaluated.
                `out_gradients` : Array
                    (N, D) float32 CUDA array pre-allocated by the caller.
                    Will be filled with `dL/dv_i` for each triangle.
            )doc") +
         EPSILON_DOC + R"doc(
                `stream` : int, optional
                    The raw 64-bit identifier (handle) of a CUDA stream.
                    Allows enqueuing operations asynchronously within deep learning frameworks.
                    For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
                    Default is 0 (the default/null stream).
            )doc" +
         USE_KAHAN_DOC)
            .c_str());

  // Payload field Gradient wrt geometry
  m.def("brute_force_gradients_with_payload_point_normal",
        nb::overload_cast<const Payload_t &, const Vec3_t &, const Vec3_t &,
                          const Payload_t &, const Vec3_t &, GradResult_t &,
                          SingleScalar_t &, float, const uint64_t, const bool>(
            &brute_force_gradients_with_payload),
        "grad_output"_a, "points"_a, "scaled_normals"_a, "payload"_a,
        "queries"_a, "out_gradients"_a, "epsilon_gradient"_a,
        "epsilon"_a = 0.004F, "stream"_a = 0, "use_kahan"_a = false,
        nb::sig(
            "def brute_force_gradients_with_payload_point_normal(grad_output: "
            "winder.types.Array[winder.types.Shape[winder.types.M, "
            "winder.types.D], winder.types.float32, winder.types.cuda], "
            "points: "
            "winder.types.Array[winder.types.Shape[winder.types.N, "
            "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
            "scaled_normals: "
            "winder.types.Array[winder.types.Shape[winder.types.N, "
            "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
            "payload: "
            "winder.types.Array[winder.types.Shape[winder.types.N, "
            "winder.types.D], winder.types.float32, winder.types.cuda], "
            "queries: "
            "winder.types.Array[winder.types.Shape[winder.types.M, "
            "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
            "out_gradients: "
            "winder.types.Array[winder.types.Shape[winder.types.N, "
            "typing.Literal[2], typing.Literal[3]], winder.types.float32, "
            "winder.types.cuda], "
            "epsilon_gradient: "
            "winder.types.Array[winder.types.Shape[typing.Literal[1]], "
            "winder.types.float32, winder.types.cuda], "
            "epsilon: float = 0.004, "
            "stream: int = 0, use_kahan: bool = false) -> None"),
        (std::string(R"doc(
                Computes the partial derivatives of the payload field loss
                with respect to the source geometry (point positions and
                scaled normals).

                NOTE: Brute Force implementation: exact, but in `O(N*M*D)`!

                This is the payload-field analogue of
                `brute_force_gradients_point_normal`: it propagates the loss
                gradient from the D-vector payload field back to the geometry,
                using the same weighted contributions as the scalar case.
                For each primitive `i`, the output row `i` contains:

                    output[i, 0] = dL/dn_i   (gradient w.r.t. the scaled normal)
                    output[i, 1] = dL/dp_i   (gradient w.r.t. the point position)

                Parameters
                ----------
                `grad_output` : Array
                    (M, D) float32 CUDA array representing the gradient of the
                    loss with respect to the payload field at each query
                    location (dL/dW). Row `j` is the D-vector adjoint for query
                    `j`, aligned with `queries`.
                `points` : Array
                    A (N, 3) float32 CUDA array of point positions.
                `scaled_normals` : Array
                    A (N, 3) float32 CUDA array of scaled normals.
                    The scaled normal direction is the orientation,
                    the scale the associated voronoi area.
                `payload` : Array
                    A (N, D) float32 CUDA array of per-primitive payloads.
                    Row `i` is the D-vector blended by the winding contribution
                    of primitive `i`.
                `queries` : Array
                    (M, 3) float32 CUDA array of query points for which the
                    payload field was evaluated.
                `out_gradients` : Array
                    (N, 2, 3) float32 CUDA array pre-allocated by the caller.
                    Will be filled with dL/dn_i at index `[i, 0]` and dL/dp_i
                    at index `[i, 1]`.
            )doc") +
         EPSILON_GRADIENT_DOC + EPSILON_DOC + R"doc(
                `stream` : int, optional
                    The raw 64-bit identifier (handle) of a CUDA stream.
                    Allows enqueuing operations asynchronously within deep learning frameworks.
                    For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
                    Default is 0 (the default/null stream).
            )doc" +
         USE_KAHAN_DOC)
            .c_str());

  m.def("brute_force_gradients_with_payload_mesh",
        nb::overload_cast<const Payload_t &, const Vec3_t &,
                          const TriangleIdx_t &, const Payload_t &,
                          const Vec3_t &, VertexGradResult_t &,
                          SingleScalar_t &, float, const uint64_t, const bool>(
            &brute_force_gradients_with_payload),
        "grad_output"_a, "vertices"_a, "triangle_indices"_a, "payload"_a,
        "queries"_a, "out_gradients"_a, "epsilon_gradient"_a,
        "epsilon"_a = 0.004F, "stream"_a = 0, "use_kahan"_a = false,
        nb::sig("def brute_force_gradients_with_payload_mesh(grad_output: "
                "winder.types.Array[winder.types.Shape[winder.types.M, "
                "winder.types.D], winder.types.float32, winder.types.cuda], "
                "vertices: "
                "winder.types.Array[winder.types.Shape[winder.types.K, "
                "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
                "triangle_indices: "
                "winder.types.Array[winder.types.Shape[winder.types.N, "
                "typing.Literal[3]], winder.types.uint32, winder.types.cuda], "
                "payload: "
                "winder.types.Array[winder.types.Shape[winder.types.N, "
                "winder.types.D], winder.types.float32, winder.types.cuda], "
                "queries: "
                "winder.types.Array[winder.types.Shape[winder.types.M, "
                "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
                "out_gradients: "
                "winder.types.Array[winder.types.Shape[winder.types.K, "
                "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
                "epsilon_gradient: "
                "winder.types.Array[winder.types.Shape[typing.Literal[1]], "
                "winder.types.float32, winder.types.cuda], "
                "epsilon: float = 0.004, "
                "stream: int = 0, use_kahan: bool = false) -> None"),
        (std::string(R"doc(
                Computes the partial derivatives of the payload field loss
                with respect to the shared mesh vertices.

                NOTE: Brute Force implementation: exact, but in `O(N*M*D)`!

                This is the payload-field analogue of
                `brute_force_gradients_mesh`. The payload is per-triangle,
                so each triangle's contribution flows into the shared
                vertices through the usual accumulation step. Output has
                shape (K, 3), with row `i` holding dL/dv_i for vertex `i`.

                Parameters
                ----------
                `grad_output` : Array
                    (M, D) float32 CUDA array representing the gradient of the
                    loss with respect to the payload field at each query
                    location (dL/dW). Row `j` is the D-vector adjoint for query
                    `j`, aligned with `queries`.
                `vertices` : Array
                    A (K, 3) float32 CUDA array holding the shared vertices.
                `triangle_indices` : Array
                    A (N, 3) uint32 index array, where each row defines the
                    three vertices of a triangle.
                    Note: Order the vertices counter-clockwise when seen from the front.
                `payload` : Array
                    A (N, D) float32 CUDA array of per-triangle payloads.
                    Row `i` is the D-vector blended by the winding contribution
                    of triangle `i`.
                `queries` : Array
                    (M, 3) float32 CUDA array of query points for which the
                    payload field was evaluated.
                `out_gradients` : Array
                    (K, 3) float32 CUDA array pre-allocated by the caller.
                    Will be filled with dL/dv_i for each shared vertex.
            )doc") +
         EPSILON_GRADIENT_DOC + EPSILON_DOC + R"doc(
                `stream` : int, optional
                    The raw 64-bit identifier (handle) of a CUDA stream.
                    Allows enqueuing operations asynchronously within deep learning frameworks.
                    For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
                    Default is 0 (the default/null stream).
            )doc" +
         USE_KAHAN_DOC)
            .c_str());

  m.def("brute_force_gradients_with_payload_triangle_soup",
        nb::overload_cast<const Payload_t &, const Triangle_t &,
                          const Payload_t &, const Vec3_t &, GradResult_t &,
                          SingleScalar_t &, float, const uint64_t, const bool>(
            &brute_force_gradients_with_payload),
        "grad_output"_a, "triangles"_a, "payload"_a, "queries"_a,
        "out_gradients"_a, "epsilon_gradient"_a, "epsilon"_a = 0.004F,
        "stream"_a = 0, "use_kahan"_a = false,
        nb::sig(
            "def brute_force_gradients_with_payload_triangle_soup(grad_output: "
            "winder.types.Array[winder.types.Shape[winder.types.M, "
            "winder.types.D], winder.types.float32, winder.types.cuda], "
            "triangles: "
            "winder.types.Array[winder.types.Shape[winder.types.N, "
            "typing.Literal[3], typing.Literal[3]], winder.types.float32, "
            "winder.types.cuda], "
            "payload: "
            "winder.types.Array[winder.types.Shape[winder.types.N, "
            "winder.types.D], winder.types.float32, winder.types.cuda], "
            "queries: "
            "winder.types.Array[winder.types.Shape[winder.types.M, "
            "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
            "out_gradients: "
            "winder.types.Array[winder.types.Shape[winder.types.N, "
            "typing.Literal[3], typing.Literal[3]], winder.types.float32, "
            "winder.types.cuda], "
            "epsilon_gradient: "
            "winder.types.Array[winder.types.Shape[typing.Literal[1]], "
            "winder.types.float32, winder.types.cuda], "
            "epsilon: float = 0.004, "
            "stream: int = 0, use_kahan: bool = false) -> None"),
        (std::string(R"doc(
                Computes the partial derivatives of the payload field loss
                with respect to the vertices of an explicit triangle soup.

                NOTE: Brute Force implementation: exact, but in `O(N*M*D)`!

                This is the payload-field analogue of
                `brute_force_gradients_triangle_soup`. Output has shape
                (N, 3, 3), with `output[i, j]` holding the gradient of the
                loss with respect to vertex `j` of triangle `i`.

                Parameters
                ----------
                `grad_output` : Array
                    (M, D) float32 CUDA array representing the gradient of the
                    loss with respect to the payload field at each query
                    location (dL/dW). Row `j` is the D-vector adjoint for query
                    `j`, aligned with `queries`.
                `triangles` : Array
                    A (N, 3, 3) float32 CUDA array holding N triangles.
                    Note: Order the vertices counter-clockwise when seen from the front.
                `payload` : Array
                    A (N, D) float32 CUDA array of per-triangle payloads.
                    Row `i` is the D-vector blended by the winding contribution
                    of triangle `i`.
                `queries` : Array
                    (M, 3) float32 CUDA array of query points for which the
                    payload field was evaluated.
                `out_gradients` : Array
                    (N, 3, 3) float32 CUDA array pre-allocated by the caller.
                    `output[i, j]` is filled with dL/dv_j for vertex `j` of
                    triangle `i`.
            )doc") +
         EPSILON_GRADIENT_DOC + EPSILON_DOC + R"doc(
                `stream` : int, optional
                    The raw 64-bit identifier (handle) of a CUDA stream.
                    Allows enqueuing operations asynchronously within deep learning frameworks.
                    For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
                    Default is 0 (the default/null stream).
            )doc" +
         USE_KAHAN_DOC)
            .c_str());

  // payload field gradients wrt queries
  m.def(
      "brute_force_query_gradients_with_payload_point_normal",
      nb::overload_cast<const Payload_t &, const Vec3_t &, const Vec3_t &,
                        const Payload_t &, const Vec3_t &, Vec3_t &, float,
                        const uint64_t, const bool>(
          &brute_force_query_gradients_with_payload),
      "grad_output"_a, "points"_a, "scaled_normals"_a, "payload"_a, "queries"_a,
      "out_query_gradients"_a, "epsilon"_a = 0.004F, "stream"_a = 0,
      "use_kahan"_a = true,
      nb::sig(
          "def "
          "brute_force_query_gradients_with_payload_point_normal(grad_output: "
          "winder.types.Array[winder.types.Shape[winder.types.M, "
          "winder.types.D], winder.types.float32, winder.types.cuda], "
          "points: "
          "winder.types.Array[winder.types.Shape[winder.types.N, "
          "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
          "scaled_normals: "
          "winder.types.Array[winder.types.Shape[winder.types.N, "
          "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
          "payload: "
          "winder.types.Array[winder.types.Shape[winder.types.N, "
          "winder.types.D], winder.types.float32, winder.types.cuda], "
          "queries: "
          "winder.types.Array[winder.types.Shape[winder.types.M, "
          "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
          "out_query_gradients: "
          "winder.types.Array[winder.types.Shape[winder.types.M, "
          "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
          "epsilon: float = 0.004, "
          "stream: int = 0, use_kahan: bool = true) -> None"),
      (std::string(R"doc(
                Computes the gradient of the payload field loss with respect
                to the query positions for a point cloud.

                NOTE: Brute Force implementation: exact, but in `O(N*M*D)`!

                This is the query-side counterpart of
                `brute_force_payload_gradients_point_normal`. For each query
                `q_j`, it produces a 3-vector

                    dL/dq_j = sum_i ⟨g_j, v_i⟩ · ∂w_i(q_j)/∂q_j

                where `g_j` is the D-vector adjoint at query `j` and `v_i` is
                the payload of primitive `i`.

                Parameters
                ----------
                `grad_output` : Array
                    (M, D) float32 CUDA array representing the gradient of the
                    loss with respect to the payload field at each query
                    location (dL/dW). Row `j` is the D-vector adjoint for query
                    `j`, aligned with `queries`.
                `points` : Array
                    A (N, 3) float32 CUDA array of point positions.
                `scaled_normals` : Array
                    A (N, 3) float32 CUDA array of scaled normals.
                    The scaled normal direction is the orientation,
                    the scale the associated voronoi area.
                `payload` : Array
                    A (N, D) float32 CUDA array of per-primitive payloads.
                    Row `i` is the D-vector blended by the winding contribution
                    of primitive `i`.
                `queries` : Array
                    (M, 3) float32 CUDA array of query points for which the
                    payload field was evaluated.
                `out_query_gradients` : Array
                    (M, 3) float32 CUDA array pre-allocated by the caller.
                    Will be filled with dL/dq_j for each query.
            )doc") +
       EPSILON_DOC + R"doc(
                `stream` : int, optional
                    The raw 64-bit identifier (handle) of a CUDA stream.
                    Allows enqueuing operations asynchronously within deep learning frameworks.
                    For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
                    Default is 0 (the default/null stream).
            )doc" +
       USE_KAHAN_DOC)
          .c_str());

  m.def(
      "brute_force_query_gradients_with_payload_mesh",
      nb::overload_cast<const Payload_t &, const Vec3_t &,
                        const TriangleIdx_t &, const Payload_t &,
                        const Vec3_t &, Vec3_t &, float, const uint64_t,
                        const bool>(&brute_force_query_gradients_with_payload),
      "grad_output"_a, "vertices"_a, "triangle_indices"_a, "payload"_a,
      "queries"_a, "out_query_gradients"_a, "epsilon"_a = 0.004F,
      "stream"_a = 0, "use_kahan"_a = true,
      nb::sig("def brute_force_query_gradients_with_payload_mesh(grad_output: "
              "winder.types.Array[winder.types.Shape[winder.types.M, "
              "winder.types.D], winder.types.float32, winder.types.cuda], "
              "vertices: "
              "winder.types.Array[winder.types.Shape[winder.types.K, "
              "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
              "triangle_indices: "
              "winder.types.Array[winder.types.Shape[winder.types.N, "
              "typing.Literal[3]], winder.types.uint32, winder.types.cuda], "
              "payload: "
              "winder.types.Array[winder.types.Shape[winder.types.N, "
              "winder.types.D], winder.types.float32, winder.types.cuda], "
              "queries: "
              "winder.types.Array[winder.types.Shape[winder.types.M, "
              "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
              "out_query_gradients: "
              "winder.types.Array[winder.types.Shape[winder.types.M, "
              "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
              "epsilon: float = 0.004, "
              "stream: int = 0, use_kahan: bool = true) -> None"),
      (std::string(R"doc(
                Computes the gradient of the payload field loss with respect
                to the query positions for a shared-vertex mesh.

                NOTE: Brute Force implementation: exact, but in `O(N*M*D)`!

                This is the query-side counterpart of
                `brute_force_payload_gradients_mesh`. For each query `q_j`,
                it produces a 3-vector

                    dL/dq_j = sum_i ⟨g_j, v_i⟩ · ∂w_i(q_j)/∂q_j

                where `g_j` is the D-vector adjoint at query `j` and `v_i` is
                the payload of triangle `i`.

                Parameters
                ----------
                `grad_output` : Array
                    (M, D) float32 CUDA array representing the gradient of the
                    loss with respect to the payload field at each query
                    location (dL/dW). Row `j` is the D-vector adjoint for query
                    `j`, aligned with `queries`.
                `vertices` : Array
                    A (K, 3) float32 CUDA array holding the shared vertices.
                `triangle_indices` : Array
                    A (N, 3) uint32 index array, where each row defines the
                    three vertices of a triangle.
                    Note: Order the vertices counter-clockwise when seen from the front.
                `payload` : Array
                    A (N, D) float32 CUDA array of per-triangle payloads.
                    Row `i` is the D-vector blended by the winding contribution
                    of triangle `i`.
                `queries` : Array
                    (M, 3) float32 CUDA array of query points for which the
                    payload field was evaluated.
                `out_query_gradients` : Array
                    (M, 3) float32 CUDA array pre-allocated by the caller.
                    Will be filled with dL/dq_j for each query.
            )doc") +
       EPSILON_DOC + R"doc(
                `stream` : int, optional
                    The raw 64-bit identifier (handle) of a CUDA stream.
                    Allows enqueuing operations asynchronously within deep learning frameworks.
                    For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
                    Default is 0 (the default/null stream).
            )doc" +
       USE_KAHAN_DOC)
          .c_str());

  m.def(
      "brute_force_query_gradients_with_payload_triangle_soup",
      nb::overload_cast<const Payload_t &, const Triangle_t &,
                        const Payload_t &, const Vec3_t &, Vec3_t &, float,
                        const uint64_t, const bool>(
          &brute_force_query_gradients_with_payload),
      "grad_output"_a, "triangles"_a, "payload"_a, "queries"_a,
      "out_query_gradients"_a, "epsilon"_a = 0.004F, "stream"_a = 0,
      "use_kahan"_a = true,
      nb::sig(
          "def "
          "brute_force_query_gradients_with_payload_triangle_soup(grad_output: "
          "winder.types.Array[winder.types.Shape[winder.types.M, "
          "winder.types.D], winder.types.float32, winder.types.cuda], "
          "triangles: "
          "winder.types.Array[winder.types.Shape[winder.types.N, "
          "typing.Literal[3], typing.Literal[3]], winder.types.float32, "
          "winder.types.cuda], "
          "payload: "
          "winder.types.Array[winder.types.Shape[winder.types.N, "
          "winder.types.D], winder.types.float32, winder.types.cuda], "
          "queries: "
          "winder.types.Array[winder.types.Shape[winder.types.M, "
          "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
          "out_query_gradients: "
          "winder.types.Array[winder.types.Shape[winder.types.M, "
          "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
          "epsilon: float = 0.004, "
          "stream: int = 0, use_kahan: bool = true) -> None"),
      (std::string(R"doc(
                Computes the gradient of the payload field loss with respect
                to the query positions for an explicit triangle soup.

                NOTE: Brute Force implementation: exact, but in `O(N*M*D)`!

                This is the query-side counterpart of
                `brute_force_payload_gradients_triangle_soup`. For each query
                `q_j`, it produces a 3-vector

                    dL/dq_j = sum_i ⟨g_j, v_i⟩ · ∂w_i(q_j)/∂q_j

                where `g_j` is the D-vector adjoint at query `j` and `v_i` is
                the payload of triangle `i`.

                Parameters
                ----------
                `grad_output` : Array
                    (M, D) float32 CUDA array representing the gradient of the
                    loss with respect to the payload field at each query
                    location (dL/dW). Row `j` is the D-vector adjoint for query
                    `j`, aligned with `queries`.
                `triangles` : Array
                    A (N, 3, 3) float32 CUDA array holding N triangles.
                    Note: Order the vertices counter-clockwise when seen from the front.
                `payload` : Array
                    A (N, D) float32 CUDA array of per-triangle payloads.
                    Row `i` is the D-vector blended by the winding contribution
                    of triangle `i`.
                `queries` : Array
                    (M, 3) float32 CUDA array of query points for which the
                    payload field was evaluated.
                `out_query_gradients` : Array
                    (M, 3) float32 CUDA array pre-allocated by the caller.
                    Will be filled with dL/dq_j for each query.
            )doc") +
       EPSILON_DOC + R"doc(
                `stream` : int, optional
                    The raw 64-bit identifier (handle) of a CUDA stream.
                    Allows enqueuing operations asynchronously within deep learning frameworks.
                    For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
                    Default is 0 (the default/null stream).
            )doc" +
       USE_KAHAN_DOC)
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

      // --- Inference -------------------------------------------------------
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

      .def("compute_query_gradients",
           &WindingNumbersEngine::compute_query_gradients, "grad_output"_a,
           "queries"_a, "out_query_gradients"_a, "beta"_a = -1.F,
           "epsilon"_a = 0.004F, "stream"_a = 0,
           nb::sig("def compute_query_gradients(self, grad_output: "
                   "winder.types.Array[winder.types.Shape[winder.types.M], "
                   "winder.types.float32, winder.types.cuda], "
                   "queries: "
                   "winder.types.Array[winder.types.Shape[winder.types.M, "
                   "typing.Literal[3]], winder.types.float32, "
                   "winder.types.cuda], "
                   "out_query_gradients: "
                   "winder.types.Array[winder.types.Shape[winder.types.M, "
                   "typing.Literal[3]], winder.types.float32, "
                   "winder.types.cuda], "
                   "beta: float = -1, epsilon: float = 0.004, "
                   "stream: int = 0) -> None"),
           (std::string(R"doc(
                Computes the gradient of a scalar loss with respect to the query
                positions, using the same acceleration structure that this engine
                uses for `compute`.

                This is the query-side counterpart of `GradientEngine.compute`:
                `GradientEngine` propagates the adjoint to the source geometry,
                whereas this method propagates it to the query coordinates.

                The intended usage pattern is:

                    engine = WindingNumberEngine(triangles, stream)
                    wn = torch.empty(M, device="cuda")
                    engine.compute(queries, wn, beta=2.3, epsilon=0.004, stream=stream)

                    loss = some_loss(wn)
                    loss.backward()
                    dL_dw = wn.grad                       # or any (M,) tensor

                    grad_q = torch.empty(M, 3, device="cuda")
                    engine.compute_query_gradients(dL_dw, queries, grad_q,
                                                   beta=2.3, epsilon=0.004,
                                                   stream=stream)

                The BVH8 built at construction is reused; no re-build is triggered
                by this call. The traversal is independent of `compute` — the
                user is free to call it once, many times with different adjoints,
                or not at all.

                For each query index j the kernel computes:

                    dL/dq_j = g_j * sum_i dw_i(q_j) / dq_j

                where g_j = grad_output[j] is the adjoint supplied by the caller.
                The adjoint is applied once per query, not per (query, primitive)
                pair.

                Parameters
                ----------
                `grad_output` : Array
                    (M,) float32 CUDA array representing the gradient of the loss
                    with respect to the winding numbers at each query. Must be
                    aligned index-for-index with `queries`.
                `queries` : Array
                    (M, 3) float32 CUDA array of query points for which the
                    gradient is evaluated. The order is preserved in the output.
                `out_query_gradients` : Array
                    (M, 3) float32 CUDA array pre-allocated by the caller.
                    Will be filled with dL/dq for each query. 
                `beta` : float
                    Barnes-Hut opening criterion. Same semantics and defaults as
                    in `compute`. Larger beta leads to more precise results but
                    also slower execution. Use any negative number to get the
                    default (2.0 for triangles, 3.3 for point clouds).
            )doc") +
            EPSILON_DOC + R"doc(
                `stream` : int, optional
                    The raw 64-bit identifier (handle) of a CUDA stream.
                    Allows enqueuing operations asynchronously within deep learning frameworks.
                    For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
                    Default is 0 (the default/null stream).
            )doc")
               .c_str())

      // --- Utilities -------------------------------------------------------
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
      .def("compute_mesh",
           nb::overload_cast<const Vec3_t &, const TriangleIdx_t &,
                             VertexGradResult_t &, SingleScalar_t &, float,
                             float, const uint64_t>(&GradientEngine::compute),
           "vertices"_a, "triangle_indices"_a, "out_gradients"_a,
           "epsilon_gradient"_a, "beta"_a = -1.F, "epsilon"_a = 0.004F,
           "stream"_a = 0,
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
               "epsilon_gradient: "
               "winder.types.Array[winder.types.Shape[typing.Literal[1]], "
               "winder.types.float32, winder.types.cuda], "
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
            EPSILON_GRADIENT_DOC + EPSILON_DOC + R"doc(
                `stream` : int, optional
                    The raw 64-bit identifier (handle) of a CUDA stream. 
                    Allows enqueuing operations asynchronously within deep learning frameworks.
                    For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
                    Default is 0 (the default/null stream).
            )doc")
               .c_str())

      // --- compute_triangle_soup -------------------------------------------
      .def("compute_triangle_soup",
           nb::overload_cast<const Triangle_t &, GradResult_t &,
                             SingleScalar_t &, float, float, const uint64_t>(
               &GradientEngine::compute),
           "triangles"_a, "out_gradients"_a, "epsilon_gradient"_a,
           "beta"_a = -1.F, "epsilon"_a = 0.004F, "stream"_a = 0,
           nb::sig("def compute_triangle_soup(self, triangles: "
                   "winder.types.Array[winder.types.Shape[winder.types.N, "
                   "typing.Literal[3], typing.Literal[3]], "
                   "winder.types.float32, winder.types.cuda], "
                   "out_gradients: "
                   "winder.types.Array[winder.types.Shape[winder.types.N, "
                   "typing.Literal[3], typing.Literal[3]], "
                   "winder.types.float32, winder.types.cuda], "
                   "epsilon_gradient: "
                   "winder.types.Array[winder.types.Shape[typing.Literal[1]], "
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
            EPSILON_GRADIENT_DOC + EPSILON_DOC + R"doc(
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
                             SingleScalar_t &, float, float, const uint64_t>(
               &GradientEngine::compute),
           "points"_a, "scaled_normals"_a, "out_gradients"_a,
           "epsilon_gradient"_a, "beta"_a = -1.F, "epsilon"_a = 0.004F,
           "stream"_a = 0,
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
               "epsilon_gradient: "
               "winder.types.Array[winder.types.Shape[typing.Literal[1]], "
               "winder.types.float32, winder.types.cuda], "
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
            EPSILON_GRADIENT_DOC + EPSILON_DOC + R"doc(
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
  // ===========================================================================
  // PayloadField
  // ===========================================================================
  nb::class_<PayloadField>(m, "PayloadField")
      // --- Triangle soup ---
      .def(nb::init<const Triangle_t &, const Payload_t &, const uint64_t>(),
           "triangles"_a, "payload"_a, "stream"_a = 0,
           nb::sig(
               "def __init__(self, triangles: "
               "winder.types.Array[winder.types.Shape[winder.types.N, "
               "typing.Literal[3], typing.Literal[3]], winder.types.float32, "
               "winder.types.cuda], payload: "
               "winder.types.Array[winder.types.Shape[winder.types.N, "
               "winder.types.D], winder.types.float32, winder.types.cuda], "
               "stream: int = 0) -> None"),
           R"doc(
              Initialize the payload field engine from an explicit triangle soup.

              Parameters
              ----------
              `triangles` : Array
                  (N, 3, 3) float32 CUDA array holding N triangles.
                  Note: Order the vertices counter-clockwise when seen from the front.
              `payload` : Array
                  (N, D) float32 CUDA array. Row `i` is the D-vector blended by
                  triangle `i`'s winding contribution. `D` must be one of
                  {1, 2, 3, 4, 8, 16, 32, 64, 128}.
              `stream` : int, optional
                  Raw 64-bit CUDA stream handle. Default is 0.
          )doc")

      // --- Mesh ---
      .def(nb::init<const Vec3_t &, const TriangleIdx_t &, const Payload_t &,
                    const uint64_t>(),
           "vertices"_a, "triangle_indices"_a, "payload"_a, "stream"_a = 0,
           nb::sig(
               "def __init__(self, vertices: "
               "winder.types.Array[winder.types.Shape[winder.types.K, "
               "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
               "triangle_indices: "
               "winder.types.Array[winder.types.Shape[winder.types.N, "
               "typing.Literal[3]], winder.types.uint32, winder.types.cuda], "
               "payload: "
               "winder.types.Array[winder.types.Shape[winder.types.N, "
               "winder.types.D], winder.types.float32, winder.types.cuda], "
               "stream: int = 0) -> None"),
           R"doc(
              Initialize the payload field engine from a shared-vertex mesh.

              Parameters
              ----------
              `vertices` : Array
                  (K, 3) float32 CUDA array holding K vertices.
              `triangle_indices` : Array
                  (N, 3) uint32 CUDA array, each row indexes 3 vertices of a triangle.
                  Note: Order the vertices counter-clockwise when seen from the front.
              `payload` : Array
                  (N, D) float32 CUDA array, one D-vector per triangle (not
                  per vertex). `D` must be one of {1, 2, 3, 4, 8, 16, 32, 64, 128}.
              `stream` : int, optional
                  Raw 64-bit CUDA stream handle. Default is 0.
          )doc")

      // --- Point cloud ---
      .def(nb::init<const Vec3_t &, const Vec3_t &, const Payload_t &,
                    const uint64_t>(),
           "points"_a, "scaled_normals"_a, "payload"_a, "stream"_a = 0,
           nb::sig(
               "def __init__(self, points: "
               "winder.types.Array[winder.types.Shape[winder.types.N, "
               "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
               "scaled_normals: "
               "winder.types.Array[winder.types.Shape[winder.types.N, "
               "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
               "payload: "
               "winder.types.Array[winder.types.Shape[winder.types.N, "
               "winder.types.D], winder.types.float32, winder.types.cuda], "
               "stream: int = 0) -> None"),
           R"doc(
              Initialize the payload field engine from an oriented point cloud.

              Parameters
              ----------
              `points` : Array
                  (N, 3) float32 CUDA array of point positions.
              `scaled_normals` : Array
                  (N, 3) float32 CUDA array of area-scaled normals.
                  The scaled normal direction is the orientation,
                  the scale is the associated Voronoi area.
              `payload` : Array
                  (N, D) float32 CUDA array, one D-vector per point.
                  `D` must be one of {1, 2, 3, 4, 8, 16, 32, 64, 128}.
              `stream` : int, optional
                  Raw 64-bit CUDA stream handle. Default is 0.
          )doc")

      // --- Forward ---
      .def("compute", &PayloadField::compute, "queries"_a, "out_payload"_a,
           "beta"_a = -1.F, "epsilon"_a = 0.004F, "stream"_a = 0, "use_kahan"_a = false,
           nb::sig(
               "def compute(self, queries: "
               "winder.types.Array[winder.types.Shape[winder.types.M, "
               "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
               "out_payload: "
               "winder.types.Array[winder.types.Shape[winder.types.M, "
               "winder.types.D], winder.types.float32, winder.types.cuda], "
               "beta: float = -1, epsilon: float = 0.004, "
               "stream: int = 0, use_kahan: bool = false) -> None"),
           (std::string(R"doc(
              Computes the payload field at the given query locations.

              `out_payload` is filled with

                  w_V(q_j) = sum_i v_i * w(q_j | g_i)

              where `w(q_j | g_i)` is the winding contribution of primitive
              `i` at query `j`. For `D=1` and `v_i = 1` for all `i`, this
              reduces exactly to the scalar winding number field.

              Parameters
              ----------
              `queries` : Array
                  (M, 3) float32 CUDA array of query points.
              `out_payload` : Array
                  (M, D) float32 CUDA array pre-allocated by the caller.
              `beta` : float
                  Barnes-Hut opening criterion. Use any negative number to
                  get the default (2.3 for point clouds, 2.0 for triangles).
              `epsilon` : float
                  Dimensionless regularization scale. See the scalar engine's
                  docstring for details. Default is 0.004.
              `stream` : int, optional
                  Raw 64-bit CUDA stream handle. Default is 0.
          )doc") +
       USE_KAHAN_DOC).c_str())

      // --- Query gradients ---
      .def("compute_query_gradients", &PayloadField::compute_query_gradients,
           "grad_output"_a, "queries"_a, "out_query_gradients"_a,
           "beta"_a = -1.F, "epsilon"_a = 0.004F, "stream"_a = 0, "use_kahan"_a = false,
           nb::sig(
               "def compute_query_gradients(self, grad_output: "
               "winder.types.Array[winder.types.Shape[winder.types.M, "
               "winder.types.D], winder.types.float32, winder.types.cuda], "
               "queries: "
               "winder.types.Array[winder.types.Shape[winder.types.M, "
               "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
               "out_query_gradients: "
               "winder.types.Array[winder.types.Shape[winder.types.M, "
               "typing.Literal[3]], winder.types.float32, winder.types.cuda], "
               "beta: float = -1, epsilon: float = 0.004, "
               "stream: int = 0, use_kahan: bool = false) -> None"),
           (std::string(R"doc(
              Computes the gradient of the loss with respect to the query
              positions.

              Intended usage:

                  engine = PayloadField(triangles, payload, stream)
                  out = torch.empty(M, D, device="cuda")
                  engine.compute(queries, out, beta=2.3, epsilon=0.004,
                                 stream=stream)
                  loss = some_loss(out)
                  loss.backward()
                  dL_dW = out.grad
                  grad_q = torch.empty(M, 3, device="cuda")
                  engine.compute_query_gradients(dL_dW, queries, grad_q,
                                                 beta=2.3, epsilon=0.004,
                                                 stream=stream)

              The BVH8 built at construction is reused; no re-build is
              triggered by this call.

              Parameters
              ----------
              `grad_output` : Array
                  (M, D) float32 CUDA array — gradient of the loss w.r.t. the
                  payload field at each query. Row `j` is the D-vector adjoint
                  for query `j`, aligned index-for-index with `queries`.
              `queries` : Array
                  (M, 3) float32 CUDA array of query points.
              `out_query_gradients` : Array
                  (M, 3) float32 CUDA array pre-allocated by the caller.
              `beta`, `epsilon`, `stream` : as in `compute`.
          )doc")+
      USE_KAHAN_DOC).c_str())

      // --- Debug ---
      .def("dump", &PayloadField::dump, nb::sig("def dump(self) -> str"),
           R"doc(
              Returns a Graphviz .dot representation of the internal BVH8
              tree, including AABBs and Taylor coefficient tables.

              Coefficient tables are rendered for component `d=0`. For `D>1`
              each node additionally shows the L2 norm of the zero-order
              payload coefficients summed across all D components, so payload
              presence / overflow / NaN issues can be spotted at a glance.
          )doc")

      // --- Introspection ---
      .def_prop_ro(
          "dimension", &PayloadField::dimension,
          "The payload dimensionality D this engine was constructed with.");
}
