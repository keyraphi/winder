"""
GPU-accelerated Differentiable Winding Number Field library.

Compatible with any framework supporting DLPack / Array API 
(PyTorch, JAX, CuPy, etc.). All inputs must reside on the same 
CUDA device.
"""

from typing import Literal, overload

import winder


def brute_force_winding_numbers_point_normal(points: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3]], winder.types.float32, winder.types.cuda], scaled_normals: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3]], winder.types.float32, winder.types.cuda], queries: winder.types.Array[winder.types.Shape[winder.types.M, Literal[3]], winder.types.float32, winder.types.cuda], out_winding_numbers: winder.types.Array[winder.types.Shape[winder.types.M], winder.types.float32, winder.types.cuda], epsilon: float = 0.004, stream: int = 0, use_kahan: bool = true) -> None:
    """
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

     `stream`  : int, optional
         The raw 64-bit identifier (handle) of a CUDA stream. 
         Allows enqueuing operations asynchronously within deep learning frameworks.
         For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
         Default is 0 (the default/null stream).

    `use_kahan` : bool, optional
        Use compensated (Kahan) summation for numerical stability.
        Reduces floating-point roundoff noise at a modest performance
        cost. The default is chosen per pass to be the best tradeoff.
    """

def brute_force_winding_numbers_mesh(vertices: winder.types.Array[winder.types.Shape[winder.types.K, Literal[3]], winder.types.float32, winder.types.cuda], triangle_indices: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3]], winder.types.uint32, winder.types.cuda], queries: winder.types.Array[winder.types.Shape[winder.types.M, Literal[3]], winder.types.float32, winder.types.cuda], out_winding_numbers: winder.types.Array[winder.types.Shape[winder.types.M], winder.types.float32, winder.types.cuda], epsilon: float = 0.004, stream: int = 0, use_kahan: bool = true) -> None:
    """
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

     `stream`  : int, optional
         The raw 64-bit identifier (handle) of a CUDA stream. 
         Allows enqueuing operations asynchronously within deep learning frameworks.
         For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
         Default is 0 (the default/null stream).

    `use_kahan` : bool, optional
        Use compensated (Kahan) summation for numerical stability.
        Reduces floating-point roundoff noise at a modest performance
        cost. The default is chosen per pass to be the best tradeoff.
    """

def brute_force_winding_numbers_triangle_soup(triangles: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3], Literal[3]], winder.types.float32, winder.types.cuda], queries: winder.types.Array[winder.types.Shape[winder.types.M, Literal[3]], winder.types.float32, winder.types.cuda], out_winding_numbers: winder.types.Array[winder.types.Shape[winder.types.M], winder.types.float32, winder.types.cuda], epsilon: float = 0.004, stream: int = 0, use_kahan: bool = true) -> None:
    """
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

     `stream`  : int, optional
         The raw 64-bit identifier (handle) of a CUDA stream. 
         Allows enqueuing operations asynchronously within deep learning frameworks.
         For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
         Default is 0 (the default/null stream).

    `use_kahan` : bool, optional
        Use compensated (Kahan) summation for numerical stability.
        Reduces floating-point roundoff noise at a modest performance
        cost. The default is chosen per pass to be the best tradeoff.
    """

def brute_force_winding_payloads_point_normal(points: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3]], winder.types.float32, winder.types.cuda], scaled_normals: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3]], winder.types.float32, winder.types.cuda], payload: winder.types.Array[winder.types.Shape[winder.types.N, winder.types.D], winder.types.float32, winder.types.cuda], queries: winder.types.Array[winder.types.Shape[winder.types.M, Literal[3]], winder.types.float32, winder.types.cuda], out_payload: winder.types.Array[winder.types.Shape[winder.types.M, winder.types.D], winder.types.float32, winder.types.cuda], epsilon: float = 0.004, stream: int = 0, use_kahan: bool = false) -> None:
    """
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

     `stream`  : int, optional
         The raw 64-bit identifier (handle) of a CUDA stream.
         Allows enqueuing operations asynchronously within deep learning frameworks.
         For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
         Default is 0 (the default/null stream).

    `use_kahan` : bool, optional
        Use compensated (Kahan) summation for numerical stability.
        Reduces floating-point roundoff noise at a modest performance
        cost. The default is chosen per pass to be the best tradeoff.
    """

def brute_force_winding_payloads_mesh(vertices: winder.types.Array[winder.types.Shape[winder.types.K, Literal[3]], winder.types.float32, winder.types.cuda], triangle_indices: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3]], winder.types.uint32, winder.types.cuda], payload: winder.types.Array[winder.types.Shape[winder.types.N, winder.types.D], winder.types.float32, winder.types.cuda], queries: winder.types.Array[winder.types.Shape[winder.types.M, Literal[3]], winder.types.float32, winder.types.cuda], out_payload: winder.types.Array[winder.types.Shape[winder.types.M, winder.types.D], winder.types.float32, winder.types.cuda], epsilon: float = 0.004, stream: int = 0, use_kahan: bool = false) -> None:
    """
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

     `stream`  : int, optional
         The raw 64-bit identifier (handle) of a CUDA stream.
         Allows enqueuing operations asynchronously within deep learning frameworks.
         For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
         Default is 0 (the default/null stream).

    `use_kahan` : bool, optional
        Use compensated (Kahan) summation for numerical stability.
        Reduces floating-point roundoff noise at a modest performance
        cost. The default is chosen per pass to be the best tradeoff.
    """

def brute_force_winding_payloads_triangle_soup(triangles: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3], Literal[3]], winder.types.float32, winder.types.cuda], payload: winder.types.Array[winder.types.Shape[winder.types.N, winder.types.D], winder.types.float32, winder.types.cuda], queries: winder.types.Array[winder.types.Shape[winder.types.M, Literal[3]], winder.types.float32, winder.types.cuda], out_payload: winder.types.Array[winder.types.Shape[winder.types.M, winder.types.D], winder.types.float32, winder.types.cuda], epsilon: float = 0.004, stream: int = 0, use_kahan: bool = false) -> None:
    """
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

     `stream`  : int, optional
         The raw 64-bit identifier (handle) of a CUDA stream.
         Allows enqueuing operations asynchronously within deep learning frameworks.
         For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
         Default is 0 (the default/null stream).

    `use_kahan` : bool, optional
        Use compensated (Kahan) summation for numerical stability.
        Reduces floating-point roundoff noise at a modest performance
        cost. The default is chosen per pass to be the best tradeoff.
    """

def brute_force_gradients_point_normal(grad_output: winder.types.Array[winder.types.Shape[winder.types.M], winder.types.float32, winder.types.cuda], points: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3]], winder.types.float32, winder.types.cuda], scaled_normals: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3]], winder.types.float32, winder.types.cuda], queries: winder.types.Array[winder.types.Shape[winder.types.M, Literal[3]], winder.types.float32, winder.types.cuda], out_gradients: winder.types.Array[winder.types.Shape[winder.types.N, Literal[2], Literal[3]], winder.types.float32, winder.types.cuda], epsilon_gradient: winder.types.Array[winder.types.Shape[Literal[1]], winder.types.float32, winder.types.cuda], epsilon: float = 0.004, stream: int = 0, use_kahan: bool = false) -> None:
    """
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

     `epsilon_gradient` : Array
         A (1,) float32 CUDA array pre-allocated by the caller on the
         same CUDA stream as the other outputs.
         Will be filled with the gradient of the loss with respect to
         the regularization scale `epsilon` (dL/depsilon).
         The result is a single scalar because `epsilon` is a global
         scalar shared by the entire winding-number field, so all
         per-query contributions are accumulated into this one entry.

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

     `stream` : int, optional
         The raw 64-bit identifier (handle) of a CUDA stream. 
         Allows enqueuing operations asynchronously within deep learning frameworks.
         For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
         Default is 0 (the default/null stream).


    `use_kahan` : bool, optional
        Use compensated (Kahan) summation for numerical stability.
        Reduces floating-point roundoff noise at a modest performance
        cost. The default is chosen per pass to be the best tradeoff.
    """

def brute_force_gradients_mesh(grad_output: winder.types.Array[winder.types.Shape[winder.types.M], winder.types.float32, winder.types.cuda], vertices: winder.types.Array[winder.types.Shape[winder.types.K, Literal[3]], winder.types.float32, winder.types.cuda], triangle_indices: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3]], winder.types.uint32, winder.types.cuda], queries: winder.types.Array[winder.types.Shape[winder.types.M, Literal[3]], winder.types.float32, winder.types.cuda], out_gradients: winder.types.Array[winder.types.Shape[winder.types.K, Literal[3]], winder.types.float32, winder.types.cuda], epsilon_gradient: winder.types.Array[winder.types.Shape[Literal[1]], winder.types.float32, winder.types.cuda], epsilon: float = 0.004, stream: int = 0, use_kahan: bool = false) -> None:
    """
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

     `epsilon_gradient` : Array
         A (1,) float32 CUDA array pre-allocated by the caller on the
         same CUDA stream as the other outputs.
         Will be filled with the gradient of the loss with respect to
         the regularization scale `epsilon` (dL/depsilon).
         The result is a single scalar because `epsilon` is a global
         scalar shared by the entire winding-number field, so all
         per-query contributions are accumulated into this one entry.

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

     `stream` : int, optional
         The raw 64-bit identifier (handle) of a CUDA stream. 
         Allows enqueuing operations asynchronously within deep learning frameworks.
         For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
         Default is 0 (the default/null stream).

    `use_kahan` : bool, optional
        Use compensated (Kahan) summation for numerical stability.
        Reduces floating-point roundoff noise at a modest performance
        cost. The default is chosen per pass to be the best tradeoff.
    """

def brute_force_gradients_triangle_soup(grad_output: winder.types.Array[winder.types.Shape[winder.types.M], winder.types.float32, winder.types.cuda], triangles: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3], Literal[3]], winder.types.uint32, winder.types.cuda], queries: winder.types.Array[winder.types.Shape[winder.types.M, Literal[3]], winder.types.float32, winder.types.cuda], out_gradients: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3], Literal[3]], winder.types.float32, winder.types.cuda], epsilon_gradient: winder.types.Array[winder.types.Shape[Literal[1]], winder.types.float32, winder.types.cuda], epsilon: float = 0.004, stream: int = 0, use_kahan: bool = false) -> None:
    """
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

     `epsilon_gradient` : Array
         A (1,) float32 CUDA array pre-allocated by the caller on the
         same CUDA stream as the other outputs.
         Will be filled with the gradient of the loss with respect to
         the regularization scale `epsilon` (dL/depsilon).
         The result is a single scalar because `epsilon` is a global
         scalar shared by the entire winding-number field, so all
         per-query contributions are accumulated into this one entry.

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

     `stream` : int, optional
         The raw 64-bit identifier (handle) of a CUDA stream. 
         Allows enqueuing operations asynchronously within deep learning frameworks.
         For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
         Default is 0 (the default/null stream).

    `use_kahan` : bool, optional
        Use compensated (Kahan) summation for numerical stability.
        Reduces floating-point roundoff noise at a modest performance
        cost. The default is chosen per pass to be the best tradeoff.
    """

def brute_force_query_gradients_point_normal(grad_output: winder.types.Array[winder.types.Shape[winder.types.M], winder.types.float32, winder.types.cuda], points: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3]], winder.types.float32, winder.types.cuda], scaled_normals: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3]], winder.types.float32, winder.types.cuda], queries: winder.types.Array[winder.types.Shape[winder.types.M, Literal[3]], winder.types.float32, winder.types.cuda], out_query_gradients: winder.types.Array[winder.types.Shape[winder.types.M, Literal[3]], winder.types.float32, winder.types.cuda], epsilon: float = 0.004, stream: int = 0, use_kahan: bool = true) -> None:
    """
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

     `stream` : int, optional
         The raw 64-bit identifier (handle) of a CUDA stream.
         Allows enqueuing operations asynchronously within deep learning frameworks.
         For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
         Default is 0 (the default/null stream).

    `use_kahan` : bool, optional
        Use compensated (Kahan) summation for numerical stability.
        Reduces floating-point roundoff noise at a modest performance
        cost. The default is chosen per pass to be the best tradeoff.
    """

def brute_force_query_gradients_triangle_soup(grad_output: winder.types.Array[winder.types.Shape[winder.types.M], winder.types.float32, winder.types.cuda], triangles: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3], Literal[3]], winder.types.float32, winder.types.cuda], queries: winder.types.Array[winder.types.Shape[winder.types.M, Literal[3]], winder.types.float32, winder.types.cuda], out_query_gradients: winder.types.Array[winder.types.Shape[winder.types.M, Literal[3]], winder.types.float32, winder.types.cuda], epsilon: float = 0.004, stream: int = 0, use_kahan: bool = true) -> None:
    """
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

     `stream` : int, optional
         The raw 64-bit identifier (handle) of a CUDA stream.
         Allows enqueuing operations asynchronously within deep learning frameworks.
         For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
         Default is 0 (the default/null stream).

    `use_kahan` : bool, optional
        Use compensated (Kahan) summation for numerical stability.
        Reduces floating-point roundoff noise at a modest performance
        cost. The default is chosen per pass to be the best tradeoff.
    """

def brute_force_query_gradients_mesh(grad_output: winder.types.Array[winder.types.Shape[winder.types.M], winder.types.float32, winder.types.cuda], vertices: winder.types.Array[winder.types.Shape[winder.types.K, Literal[3]], winder.types.float32, winder.types.cuda], triangle_indices: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3]], winder.types.uint32, winder.types.cuda], queries: winder.types.Array[winder.types.Shape[winder.types.M, Literal[3]], winder.types.float32, winder.types.cuda], out_query_gradients: winder.types.Array[winder.types.Shape[winder.types.M, Literal[3]], winder.types.float32, winder.types.cuda], epsilon: float = 0.004, stream: int = 0, use_kahan: bool = true) -> None:
    """
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

     `stream` : int, optional
         The raw 64-bit identifier (handle) of a CUDA stream.
         Allows enqueuing operations asynchronously within deep learning frameworks.
         For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
         Default is 0 (the default/null stream).

    `use_kahan` : bool, optional
        Use compensated (Kahan) summation for numerical stability.
        Reduces floating-point roundoff noise at a modest performance
        cost. The default is chosen per pass to be the best tradeoff.
    """

def brute_force_payload_gradients_point_normal(grad_output: winder.types.Array[winder.types.Shape[winder.types.M, winder.types.D], winder.types.float32, winder.types.cuda], points: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3]], winder.types.float32, winder.types.cuda], scaled_normals: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3]], winder.types.float32, winder.types.cuda], queries: winder.types.Array[winder.types.Shape[winder.types.M, Literal[3]], winder.types.float32, winder.types.cuda], out_gradients: winder.types.Array[winder.types.Shape[winder.types.N, winder.types.D], winder.types.float32, winder.types.cuda], epsilon: float = 0.004, stream: int = 0, use_kahan: bool = false) -> None:
    """
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

     `stream` : int, optional
         The raw 64-bit identifier (handle) of a CUDA stream.
         Allows enqueuing operations asynchronously within deep learning frameworks.
         For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
         Default is 0 (the default/null stream).

    `use_kahan` : bool, optional
        Use compensated (Kahan) summation for numerical stability.
        Reduces floating-point roundoff noise at a modest performance
        cost. The default is chosen per pass to be the best tradeoff.
    """

def brute_force_payload_gradients_mesh(grad_output: winder.types.Array[winder.types.Shape[winder.types.M, winder.types.D], winder.types.float32, winder.types.cuda], vertices: winder.types.Array[winder.types.Shape[winder.types.K, Literal[3]], winder.types.float32, winder.types.cuda], triangle_indices: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3]], winder.types.uint32, winder.types.cuda], queries: winder.types.Array[winder.types.Shape[winder.types.M, Literal[3]], winder.types.float32, winder.types.cuda], out_gradients: winder.types.Array[winder.types.Shape[winder.types.N, winder.types.D], winder.types.float32, winder.types.cuda], epsilon: float = 0.004, stream: int = 0, use_kahan: bool = false) -> None:
    """
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

     `stream` : int, optional
         The raw 64-bit identifier (handle) of a CUDA stream.
         Allows enqueuing operations asynchronously within deep learning frameworks.
         For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
         Default is 0 (the default/null stream).

    `use_kahan` : bool, optional
        Use compensated (Kahan) summation for numerical stability.
        Reduces floating-point roundoff noise at a modest performance
        cost. The default is chosen per pass to be the best tradeoff.
    """

def brute_force_payload_gradients_triangle_soup(grad_output: winder.types.Array[winder.types.Shape[winder.types.M, winder.types.D], winder.types.float32, winder.types.cuda], triangles: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3], Literal[3]], winder.types.float32, winder.types.cuda], queries: winder.types.Array[winder.types.Shape[winder.types.M, Literal[3]], winder.types.float32, winder.types.cuda], out_gradients: winder.types.Array[winder.types.Shape[winder.types.N, winder.types.D], winder.types.float32, winder.types.cuda], epsilon: float = 0.004, stream: int = 0, use_kahan: bool = false) -> None:
    """
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

     `stream` : int, optional
         The raw 64-bit identifier (handle) of a CUDA stream.
         Allows enqueuing operations asynchronously within deep learning frameworks.
         For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
         Default is 0 (the default/null stream).

    `use_kahan` : bool, optional
        Use compensated (Kahan) summation for numerical stability.
        Reduces floating-point roundoff noise at a modest performance
        cost. The default is chosen per pass to be the best tradeoff.
    """

def brute_force_gradients_with_payload_point_normal(grad_output: winder.types.Array[winder.types.Shape[winder.types.M, winder.types.D], winder.types.float32, winder.types.cuda], points: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3]], winder.types.float32, winder.types.cuda], scaled_normals: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3]], winder.types.float32, winder.types.cuda], payload: winder.types.Array[winder.types.Shape[winder.types.N, winder.types.D], winder.types.float32, winder.types.cuda], queries: winder.types.Array[winder.types.Shape[winder.types.M, Literal[3]], winder.types.float32, winder.types.cuda], out_gradients: winder.types.Array[winder.types.Shape[winder.types.N, Literal[2], Literal[3]], winder.types.float32, winder.types.cuda], epsilon_gradient: winder.types.Array[winder.types.Shape[Literal[1]], winder.types.float32, winder.types.cuda], epsilon: float = 0.004, stream: int = 0, use_kahan: bool = false) -> None:
    """
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

     `epsilon_gradient` : Array
         A (1,) float32 CUDA array pre-allocated by the caller on the
         same CUDA stream as the other outputs.
         Will be filled with the gradient of the loss with respect to
         the regularization scale `epsilon` (dL/depsilon).
         The result is a single scalar because `epsilon` is a global
         scalar shared by the entire winding-number field, so all
         per-query contributions are accumulated into this one entry.

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

     `stream` : int, optional
         The raw 64-bit identifier (handle) of a CUDA stream.
         Allows enqueuing operations asynchronously within deep learning frameworks.
         For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
         Default is 0 (the default/null stream).

    `use_kahan` : bool, optional
        Use compensated (Kahan) summation for numerical stability.
        Reduces floating-point roundoff noise at a modest performance
        cost. The default is chosen per pass to be the best tradeoff.
    """

def brute_force_gradients_with_payload_mesh(grad_output: winder.types.Array[winder.types.Shape[winder.types.M, winder.types.D], winder.types.float32, winder.types.cuda], vertices: winder.types.Array[winder.types.Shape[winder.types.K, Literal[3]], winder.types.float32, winder.types.cuda], triangle_indices: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3]], winder.types.uint32, winder.types.cuda], payload: winder.types.Array[winder.types.Shape[winder.types.N, winder.types.D], winder.types.float32, winder.types.cuda], queries: winder.types.Array[winder.types.Shape[winder.types.M, Literal[3]], winder.types.float32, winder.types.cuda], out_gradients: winder.types.Array[winder.types.Shape[winder.types.K, Literal[3]], winder.types.float32, winder.types.cuda], epsilon_gradient: winder.types.Array[winder.types.Shape[Literal[1]], winder.types.float32, winder.types.cuda], epsilon: float = 0.004, stream: int = 0, use_kahan: bool = false) -> None:
    """
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

     `epsilon_gradient` : Array
         A (1,) float32 CUDA array pre-allocated by the caller on the
         same CUDA stream as the other outputs.
         Will be filled with the gradient of the loss with respect to
         the regularization scale `epsilon` (dL/depsilon).
         The result is a single scalar because `epsilon` is a global
         scalar shared by the entire winding-number field, so all
         per-query contributions are accumulated into this one entry.

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

     `stream` : int, optional
         The raw 64-bit identifier (handle) of a CUDA stream.
         Allows enqueuing operations asynchronously within deep learning frameworks.
         For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
         Default is 0 (the default/null stream).

    `use_kahan` : bool, optional
        Use compensated (Kahan) summation for numerical stability.
        Reduces floating-point roundoff noise at a modest performance
        cost. The default is chosen per pass to be the best tradeoff.
    """

def brute_force_gradients_with_payload_triangle_soup(grad_output: winder.types.Array[winder.types.Shape[winder.types.M, winder.types.D], winder.types.float32, winder.types.cuda], triangles: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3], Literal[3]], winder.types.float32, winder.types.cuda], payload: winder.types.Array[winder.types.Shape[winder.types.N, winder.types.D], winder.types.float32, winder.types.cuda], queries: winder.types.Array[winder.types.Shape[winder.types.M, Literal[3]], winder.types.float32, winder.types.cuda], out_gradients: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3], Literal[3]], winder.types.float32, winder.types.cuda], epsilon_gradient: winder.types.Array[winder.types.Shape[Literal[1]], winder.types.float32, winder.types.cuda], epsilon: float = 0.004, stream: int = 0, use_kahan: bool = false) -> None:
    """
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

     `epsilon_gradient` : Array
         A (1,) float32 CUDA array pre-allocated by the caller on the
         same CUDA stream as the other outputs.
         Will be filled with the gradient of the loss with respect to
         the regularization scale `epsilon` (dL/depsilon).
         The result is a single scalar because `epsilon` is a global
         scalar shared by the entire winding-number field, so all
         per-query contributions are accumulated into this one entry.

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

     `stream` : int, optional
         The raw 64-bit identifier (handle) of a CUDA stream.
         Allows enqueuing operations asynchronously within deep learning frameworks.
         For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
         Default is 0 (the default/null stream).

    `use_kahan` : bool, optional
        Use compensated (Kahan) summation for numerical stability.
        Reduces floating-point roundoff noise at a modest performance
        cost. The default is chosen per pass to be the best tradeoff.
    """

def brute_force_query_gradients_with_payload_point_normal(grad_output: winder.types.Array[winder.types.Shape[winder.types.M, winder.types.D], winder.types.float32, winder.types.cuda], points: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3]], winder.types.float32, winder.types.cuda], scaled_normals: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3]], winder.types.float32, winder.types.cuda], payload: winder.types.Array[winder.types.Shape[winder.types.N, winder.types.D], winder.types.float32, winder.types.cuda], queries: winder.types.Array[winder.types.Shape[winder.types.M, Literal[3]], winder.types.float32, winder.types.cuda], out_query_gradients: winder.types.Array[winder.types.Shape[winder.types.M, Literal[3]], winder.types.float32, winder.types.cuda], epsilon: float = 0.004, stream: int = 0, use_kahan: bool = true) -> None:
    """
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

     `stream` : int, optional
         The raw 64-bit identifier (handle) of a CUDA stream.
         Allows enqueuing operations asynchronously within deep learning frameworks.
         For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
         Default is 0 (the default/null stream).

    `use_kahan` : bool, optional
        Use compensated (Kahan) summation for numerical stability.
        Reduces floating-point roundoff noise at a modest performance
        cost. The default is chosen per pass to be the best tradeoff.
    """

def brute_force_query_gradients_with_payload_mesh(grad_output: winder.types.Array[winder.types.Shape[winder.types.M, winder.types.D], winder.types.float32, winder.types.cuda], vertices: winder.types.Array[winder.types.Shape[winder.types.K, Literal[3]], winder.types.float32, winder.types.cuda], triangle_indices: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3]], winder.types.uint32, winder.types.cuda], payload: winder.types.Array[winder.types.Shape[winder.types.N, winder.types.D], winder.types.float32, winder.types.cuda], queries: winder.types.Array[winder.types.Shape[winder.types.M, Literal[3]], winder.types.float32, winder.types.cuda], out_query_gradients: winder.types.Array[winder.types.Shape[winder.types.M, Literal[3]], winder.types.float32, winder.types.cuda], epsilon: float = 0.004, stream: int = 0, use_kahan: bool = true) -> None:
    """
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

     `stream` : int, optional
         The raw 64-bit identifier (handle) of a CUDA stream.
         Allows enqueuing operations asynchronously within deep learning frameworks.
         For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
         Default is 0 (the default/null stream).

    `use_kahan` : bool, optional
        Use compensated (Kahan) summation for numerical stability.
        Reduces floating-point roundoff noise at a modest performance
        cost. The default is chosen per pass to be the best tradeoff.
    """

def brute_force_query_gradients_with_payload_triangle_soup(grad_output: winder.types.Array[winder.types.Shape[winder.types.M, winder.types.D], winder.types.float32, winder.types.cuda], triangles: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3], Literal[3]], winder.types.float32, winder.types.cuda], payload: winder.types.Array[winder.types.Shape[winder.types.N, winder.types.D], winder.types.float32, winder.types.cuda], queries: winder.types.Array[winder.types.Shape[winder.types.M, Literal[3]], winder.types.float32, winder.types.cuda], out_query_gradients: winder.types.Array[winder.types.Shape[winder.types.M, Literal[3]], winder.types.float32, winder.types.cuda], epsilon: float = 0.004, stream: int = 0, use_kahan: bool = true) -> None:
    """
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

     `stream` : int, optional
         The raw 64-bit identifier (handle) of a CUDA stream.
         Allows enqueuing operations asynchronously within deep learning frameworks.
         For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
         Default is 0 (the default/null stream).

    `use_kahan` : bool, optional
        Use compensated (Kahan) summation for numerical stability.
        Reduces floating-point roundoff noise at a modest performance
        cost. The default is chosen per pass to be the best tradeoff.
    """

class WindingNumberEngine:
    @overload
    def __init__(self, triangles: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3], Literal[3]], winder.types.float32, winder.types.cuda, stream: int]) -> None:
        """
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
        """

    @overload
    def __init__(self, vertices: winder.types.Array[winder.types.Shape[winder.types.K, Literal[3]], winder.types.float32, winder.types.cuda], triangle_indices: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3]], winder.types.uint32, winder.types.cuda, stream: int]) -> None:
        """
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
        """

    @overload
    def __init__(self, points: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3]], winder.types.float32, winder.types.cuda], scaled_normals: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3]], winder.types.float32, winder.types.cuda], stream: int) -> None:
        """
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
        """

    def compute(self, queries: winder.types.Array[winder.types.Shape[winder.types.M, Literal[3]], winder.types.float32, winder.types.cuda], out_winding_numbers: winder.types.Array[winder.types.Shape[winder.types.M], winder.types.float32, winder.types.cuda],beta: float = -1, epsilon: float = 0.004, stream: int = 0) -> None:
        """
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

         `stream`  : int, optional
             The raw 64-bit identifier (handle) of a CUDA stream. 
             Allows enqueuing operations asynchronously within deep learning frameworks.
             For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
             Default is 0 (the default/null stream).
        """

    def compute_query_gradients(self, grad_output: winder.types.Array[winder.types.Shape[winder.types.M], winder.types.float32, winder.types.cuda], queries: winder.types.Array[winder.types.Shape[winder.types.M, Literal[3]], winder.types.float32, winder.types.cuda], out_query_gradients: winder.types.Array[winder.types.Shape[winder.types.M, Literal[3]], winder.types.float32, winder.types.cuda], beta: float = -1, epsilon: float = 0.004, stream: int = 0) -> None:
        """
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

         `stream` : int, optional
             The raw 64-bit identifier (handle) of a CUDA stream.
             Allows enqueuing operations asynchronously within deep learning frameworks.
             For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
             Default is 0 (the default/null stream).
        """

    def dump(self) -> str:
        """
        Returns a detailed .dot string representation of the internal BVH8 Tree.
        """

class GradientEngine:
    def __init__(self, queries: winder.types.Array[winder.types.Shape[winder.types.M, Literal[3]], winder.types.float32, winder.types.cuda], grad_output: winder.types.Array[winder.types.Shape[winder.types.M], winder.types.float32, winder.types.cuda], stream: int = 0) -> None:
        """
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
        """

    def compute_mesh(self, vertices: winder.types.Array[winder.types.Shape[winder.types.K, Literal[3]], winder.types.float32, winder.types.cuda], triangle_indices: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3]], winder.types.uint32, winder.types.cuda], out_gradients: winder.types.Array[winder.types.Shape[winder.types.K, Literal[3]], winder.types.float32, winder.types.cuda], epsilon_gradient: winder.types.Array[winder.types.Shape[Literal[1]], winder.types.float32, winder.types.cuda], beta: float = -1, epsilon: float = 0.004, stream: int = 0) -> None:
        """
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

         `epsilon_gradient` : Array
             A (1,) float32 CUDA array pre-allocated by the caller on the
             same CUDA stream as the other outputs.
             Will be filled with the gradient of the loss with respect to
             the regularization scale `epsilon` (dL/depsilon).
             The result is a single scalar because `epsilon` is a global
             scalar shared by the entire winding-number field, so all
             per-query contributions are accumulated into this one entry.

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

         `stream` : int, optional
             The raw 64-bit identifier (handle) of a CUDA stream. 
             Allows enqueuing operations asynchronously within deep learning frameworks.
             For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
             Default is 0 (the default/null stream).
        """

    def compute_triangle_soup(self, triangles: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3], Literal[3]], winder.types.float32, winder.types.cuda], out_gradients: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3], Literal[3]], winder.types.float32, winder.types.cuda], epsilon_gradient: winder.types.Array[winder.types.Shape[Literal[1]], winder.types.float32, winder.types.cuda], beta: float = -1, epsilon: float = 0.004, stream: int = 0) -> None:
        """
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

         `epsilon_gradient` : Array
             A (1,) float32 CUDA array pre-allocated by the caller on the
             same CUDA stream as the other outputs.
             Will be filled with the gradient of the loss with respect to
             the regularization scale `epsilon` (dL/depsilon).
             The result is a single scalar because `epsilon` is a global
             scalar shared by the entire winding-number field, so all
             per-query contributions are accumulated into this one entry.

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

         `stream` : int, optional
             The raw 64-bit identifier (handle) of a CUDA stream. 
             Allows enqueuing operations asynchronously within deep learning frameworks.
             For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
             Default is 0 (the default/null stream).
        """

    def compute_point_normal(self, points: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3]], winder.types.float32, winder.types.cuda], scaled_normals: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3]], winder.types.float32, winder.types.cuda], out_gradients: winder.types.Array[winder.types.Shape[winder.types.N, Literal[2], Literal[3]], winder.types.float32, winder.types.cuda], epsilon_gradient: winder.types.Array[winder.types.Shape[Literal[1]], winder.types.float32, winder.types.cuda], beta: float = -1, epsilon: float = 0.004, stream: int = 0) -> None:
        """
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

         `epsilon_gradient` : Array
             A (1,) float32 CUDA array pre-allocated by the caller on the
             same CUDA stream as the other outputs.
             Will be filled with the gradient of the loss with respect to
             the regularization scale `epsilon` (dL/depsilon).
             The result is a single scalar because `epsilon` is a global
             scalar shared by the entire winding-number field, so all
             per-query contributions are accumulated into this one entry.

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

         `stream` : int, optional
             The raw 64-bit identifier (handle) of a CUDA stream. 
             Allows enqueuing operations asynchronously within deep learning frameworks.
             For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
             Default is 0 (the default/null stream).
        """

    def dump(self) -> str:
        """
        Returns a detailed .dot string representation of the internal BVH8 Tree.
        """

class PayloadField:
    @overload
    def __init__(self, triangles: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3], Literal[3]], winder.types.float32, winder.types.cuda], payload: winder.types.Array[winder.types.Shape[winder.types.N, winder.types.D], winder.types.float32, winder.types.cuda], stream: int = 0) -> None:
        """
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
        """

    @overload
    def __init__(self, vertices: winder.types.Array[winder.types.Shape[winder.types.K, Literal[3]], winder.types.float32, winder.types.cuda], triangle_indices: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3]], winder.types.uint32, winder.types.cuda], payload: winder.types.Array[winder.types.Shape[winder.types.N, winder.types.D], winder.types.float32, winder.types.cuda], stream: int = 0) -> None:
        """
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
        """

    @overload
    def __init__(self, points: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3]], winder.types.float32, winder.types.cuda], scaled_normals: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3]], winder.types.float32, winder.types.cuda], payload: winder.types.Array[winder.types.Shape[winder.types.N, winder.types.D], winder.types.float32, winder.types.cuda], stream: int = 0) -> None:
        """
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
        """

    def compute(self, queries: winder.types.Array[winder.types.Shape[winder.types.M, Literal[3]], winder.types.float32, winder.types.cuda], out_payload: winder.types.Array[winder.types.Shape[winder.types.M, winder.types.D], winder.types.float32, winder.types.cuda], beta: float = -1, epsilon: float = 0.004, stream: int = 0, use_kahan: bool = false) -> None:
        """
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

         `use_kahan` : bool, optional
             Use compensated (Kahan) summation for numerical stability.
             Reduces floating-point roundoff noise at a modest performance
             cost. The default is chosen per pass to be the best tradeoff.
        """

    def compute_query_gradients(self, grad_output: winder.types.Array[winder.types.Shape[winder.types.M, winder.types.D], winder.types.float32, winder.types.cuda], queries: winder.types.Array[winder.types.Shape[winder.types.M, Literal[3]], winder.types.float32, winder.types.cuda], out_query_gradients: winder.types.Array[winder.types.Shape[winder.types.M, Literal[3]], winder.types.float32, winder.types.cuda], beta: float = -1, epsilon: float = 0.004, stream: int = 0, use_kahan: bool = false) -> None:
        """
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

         `use_kahan` : bool, optional
             Use compensated (Kahan) summation for numerical stability.
             Reduces floating-point roundoff noise at a modest performance
             cost. The default is chosen per pass to be the best tradeoff.
        """

    def dump(self) -> str:
        """
        Returns a Graphviz .dot representation of the internal BVH8
        tree, including AABBs and Taylor coefficient tables.

        Coefficient tables are rendered for component `d=0`. For `D>1`
        each node additionally shows the L2 norm of the zero-order
        payload coefficients summed across all D components, so payload
        presence / overflow / NaN issues can be spotted at a glance.
        """

    @property
    def dimension(self) -> int:
        """The payload dimensionality D this engine was constructed with."""
