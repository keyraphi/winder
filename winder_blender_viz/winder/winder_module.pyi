from typing import Literal, overload
import winder.types


@overload
def brute_force_winding_numbers(points: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3]], winder.types.float32, winder.types.cuda], scaled_normals: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3]], winder.types.float32, winder.types.cuda], queries: winder.types.Array[winder.types.Shape[winder.types.M, Literal[3]], winder.types.float32, winder.types.cuda], out_winding_numbers: winder.types.Array[winder.types.Shape[winder.types.M], winder.types.float32, winder.types.cuda], epsilon: float = -1, stream: int = 0) -> None:
    """
    Computes the winding number at the given query locations on GPU.

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
        Regularization scale (smoothing radius) used to prevent numerical 
        singularities (NaN/infinity) when queries land near points in point clouds.
        Only applies to point cloud backends.
        - distance >= 2*epsilon: Acts as standard unregularized potential.
        - distance < 2*epsilon: Smoothly dampens potential to a finite maximum.
        Use any negative number to get the default value (1/250).
    `stream`  : int, optional
        The raw 64-bit identifier (handle) of a CUDA stream. 
        Allows enqueuing operations asynchronously within deep learning frameworks.
        For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
        Default is 0 (the default/null stream).
    """

@overload
def brute_force_winding_numbers(vertices: winder.types.Array[winder.types.Shape[winder.types.K, Literal[3]], winder.types.float32, winder.types.cuda], triangle_indices: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3]], winder.types.uint32, winder.types.cuda], queries: winder.types.Array[winder.types.Shape[winder.types.M, Literal[3]], winder.types.float32, winder.types.cuda], out_winding_numbers: winder.types.Array[winder.types.Shape[winder.types.M], winder.types.float32, winder.types.cuda],stream: int = 0) -> None:
    """
    Computes the winding number at the given query locations on GPU.

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
    `stream`  : int, optional
        The raw 64-bit identifier (handle) of a CUDA stream. 
        Allows enqueuing operations asynchronously within deep learning frameworks.
        For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
        Default is 0 (the default/null stream).
    """

@overload
def brute_force_winding_numbers(triangles: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3], Literal[3]], winder.types.float32, winder.types.cuda], queries: winder.types.Array[winder.types.Shape[winder.types.M, Literal[3]], winder.types.float32, winder.types.cuda], out_winding_numbers: winder.types.Array[winder.types.Shape[winder.types.M], winder.types.float32, winder.types.cuda],stream: int = 0) -> None:
    """
    Computes the winding number at the given query locations on GPU.

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
    `stream`  : int, optional
        The raw 64-bit identifier (handle) of a CUDA stream. 
        Allows enqueuing operations asynchronously within deep learning frameworks.
        For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
        Default is 0 (the default/null stream).
    """

@overload
def brute_force_gradients(grad_output: winder.types.Array[winder.types.Shape[winder.types.N], winder.types.float32, winder.types.cuda], points: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3]], winder.types.float32, winder.types.cuda], scaled_normals: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3]], winder.types.float32, winder.types.cuda], queries: winder.types.Array[winder.types.Shape[winder.types.M, Literal[3]], winder.types.float32, winder.types.cuda], out_gradients: winder.types.Array[winder.types.Shape[winder.types.N, Literal[2], Literal[3]], winder.types.float32, winder.types.cuda], epsilon: float, stream: int = 0) -> None:
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
    `epsilon` : float, optional
        Regularization scale (smoothing radius) used to prevent numerical 
        singularities (NaN/infinity) when queries land near points in point clouds.
        - distance >= 2*epsilon: Acts as standard unregularized potential.
        - distance < 2*epsilon: Smoothly dampens potential to a finite maximum.
        Use any negative number to get the default value (1/250).
    `stream` : int, optional
        The raw 64-bit identifier (handle) of a CUDA stream. 
        Allows enqueuing operations asynchronously within deep learning frameworks.
        For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
        Default is 0 (the default/null stream).
    """

@overload
def brute_force_gradients(grad_output: winder.types.Array[winder.types.Shape[winder.types.N], winder.types.float32, winder.types.cuda], vertices: winder.types.Array[winder.types.Shape[winder.types.K, Literal[3]], winder.types.float32, winder.types.cuda], triangle_indices: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3]], winder.types.uint32, winder.types.cuda], queries: winder.types.Array[winder.types.Shape[winder.types.M, Literal[3]], winder.types.float32, winder.types.cuda], epsilon: float, out_gradients: winder.types.Array[winder.types.Shape[winder.types.K, Literal[3]], winder.types.float32, winder.types.cuda], stream: int = 0) -> None:
    """
    Compute the partial derivatives w.r.t. the given triangles vertex positions.

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
    `stream` : int, optional
        The raw 64-bit identifier (handle) of a CUDA stream. 
        Allows enqueuing operations asynchronously within deep learning frameworks.
        For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
        Default is 0 (the default/null stream).
    """

@overload
def brute_force_gradients(grad_output: winder.types.Array[winder.types.Shape[winder.types.N], winder.types.float32, winder.types.cuda], triangles: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3], Literal[3]], winder.types.uint32, winder.types.cuda], queries: winder.types.Array[winder.types.Shape[winder.types.M, Literal[3]], winder.types.float32, winder.types.cuda], out_gradients: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3], Literal[3]], winder.types.float32, winder.types.cuda], stream: int = 0) -> None:
    """
    Compute the partial derivatives w.r.t. the given triangles vertex positions.

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
    `stream` : int, optional
        The raw 64-bit identifier (handle) of a CUDA stream. 
        Allows enqueuing operations asynchronously within deep learning frameworks.
        For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
        Default is 0 (the default/null stream).
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

    @overload
    def compute(self, queries: winder.types.Array[winder.types.Shape[winder.types.M, Literal[3]], winder.types.float32, winder.types.cuda], out_winding_numbers: winder.types.Array[winder.types.Shape[winder.types.M], winder.types.float32, winder.types.cuda],beta: float = -1, epsilon: float = -1, stream: int = 0) -> None:
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
            Default for point clouds is 2.0.
            Default for triangles is 2.3
        `epsilon` : float, optional
            Regularization scale (smoothing radius) used to prevent numerical 
            singularities (NaN/infinity) when queries land near points in point clouds.
            Only applies to point cloud fields and is ignored for triangle based fields.
            - distance >= 2*epsilon: Acts as standard unregularized potential.
            - distance < 2*epsilon: Smoothly dampens potential to a finite maximum.
            Use any negative number to get the default value (1/250).
        `stream`  : int, optional
            The raw 64-bit identifier (handle) of a CUDA stream. 
            Allows enqueuing operations asynchronously within deep learning frameworks.
            For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
            Default is 0 (the default/null stream).
        """

    @overload
    def compute(self, queries: winder.types.Array[winder.types.Shape[winder.types.M, Literal[3]], winder.types.float32, winder.types.cuda], out_winding_numbers: winder.types.Array[winder.types.Shape[winder.types.M], winder.types.float32, winder.types.cuda],beta: float = -1, epsilon: float = -1, stream: int = 0) -> None:
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
            Default for point clouds is 2.0.
            Default for triangles is 2.3
        `epsilon` : float, optional
            Regularization scale (smoothing radius) used to prevent numerical 
            singularities (NaN/infinity) when queries land near points in point clouds.
            Only applies to point cloud fields and is ignored for triangle based fields.
            - distance >= 2*epsilon: Acts as standard unregularized potential.
            - distance < 2*epsilon: Smoothly dampens potential to a finite maximum.
            Use any negative number to get the default value (1/250).
        `stream`  : int, optional
            The raw 64-bit identifier (handle) of a CUDA stream. 
            Allows enqueuing operations asynchronously within deep learning frameworks.
            For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
            Default is 0 (the default/null stream).

        Returns
        -------
        (M,) float32 CUDA array holding the winding numbers for the queries.
        """

    @overload
    def dump(self) -> str:
        """
        Returns a detailed .dot string representation of the internal BVH8 Tree.
        """

    @overload
    def dump(self) -> str: ...

class GradientEngine:
    @overload
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

    @overload
    def __init__(self, queries: winder.types.Array[winder.types.Shape[winder.types.M, Literal[3]], winder.types.float32, winder.types.cuda], grad_output: winder.types.Array[winder.types.Shape[winder.types.M], winder.types.float32, winder.types.cuda], stream: int) -> None: ...

    @overload
    def compute(self, vertices: winder.types.Array[winder.types.Shape[winder.types.K, Literal[3]], winder.types.float32, winder.types.cuda], triangle_indices: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3]], winder.types.uint32, winder.types.cuda], out_gradients: winder.types.Array[winder.types.Shape[winder.types.K, Literal[3]], winder.types.float32, winder.types.cuda], beta: float = -1, stream: int = 0) -> None:
        """
        Compute the partial derivatives w.r.t. the given triangles vertex positions.


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
            Default for triangles is 2.3 TODO EXPERIMENT
        `stream` : int, optional
            The raw 64-bit identifier (handle) of a CUDA stream. 
            Allows enqueuing operations asynchronously within deep learning frameworks.
            For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
            Default is 0 (the default/null stream).
        """

    @overload
    def compute(self, triangles: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3], Literal[3]], winder.types.float32, winder.types.cuda], out_gradients: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3], Literal[3]], winder.types.float32, winder.types.cuda], beta: float = -1, stream: int = 0) -> None:
        """
        Compute the partial derivatives w.r.t. the given triangles vertex positions.


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
            Default for triangles is 2.3 TODO EXPERIMENT
        `stream` : int, optional
            The raw 64-bit identifier (handle) of a CUDA stream. 
            Allows enqueuing operations asynchronously within deep learning frameworks.
            For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
            Default is 0 (the default/null stream).
        """

    @overload
    def compute(self, points: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3]], winder.types.float32, winder.types.cuda], scaled_normals: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3]], winder.types.float32, winder.types.cuda], out_gradients: winder.types.Array[winder.types.Shape[winder.types.N, Literal[2], Literal[3]], winder.types.float32, winder.types.cuda], beta: float = -1, epsilon: float = -1, stream: int = 0) -> None:
        """
        Compute the partial derivatives w.r.t. the given point positions and scaled normals.


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
            Default for point clouds is 2.0. TODO EXPERIMENT
        `epsilon` : float, optional
            Regularization scale (smoothing radius) used to prevent numerical 
            singularities (NaN/infinity) when queries land near points in point clouds.
            - distance >= 2*epsilon: Acts as standard unregularized potential.
            - distance < 2*epsilon: Smoothly dampens potential to a finite maximum.
            Use any negative number to get the default value (1/250).
        `stream` : int, optional
            The raw 64-bit identifier (handle) of a CUDA stream. 
            Allows enqueuing operations asynchronously within deep learning frameworks.
            For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
            Default is 0 (the default/null stream).
        """

    @overload
    def compute(self, vertices: winder.types.Array[winder.types.Shape[winder.types.K, Literal[3]], winder.types.float32, winder.types.cuda], triangle_indices: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3]], winder.types.uint32, winder.types.cuda], out_gradients: winder.types.Array[winder.types.Shape[winder.types.K, Literal[3]], winder.types.float32, winder.types.cuda], beta: float = -1, stream: int = 0) -> None:
        """
        Compute the partial derivatives w.r.t. the given triangles vertex positions.


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
            Default for triangles is 2.3 TODO EXPERIMENT
        `stream` : int, optional
            The raw 64-bit identifier (handle) of a CUDA stream. 
            Allows enqueuing operations asynchronously within deep learning frameworks.
            For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
            Default is 0 (the default/null stream).
        """

    @overload
    def compute(self, triangles: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3], Literal[3]], winder.types.float32, winder.types.cuda], out_gradients: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3], Literal[3]], winder.types.float32, winder.types.cuda], beta: float = -1, stream: int = 0) -> None:
        """
        Compute the partial derivatives w.r.t. the given triangles vertex positions.


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
            Default for triangles is 2.3 TODO EXPERIMENT
        `stream` : int, optional
            The raw 64-bit identifier (handle) of a CUDA stream. 
            Allows enqueuing operations asynchronously within deep learning frameworks.
            For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
            Default is 0 (the default/null stream).
        """

    @overload
    def compute(self, points: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3]], winder.types.float32, winder.types.cuda], scaled_normals: winder.types.Array[winder.types.Shape[winder.types.N, Literal[3]], winder.types.float32, winder.types.cuda], out_gradients: winder.types.Array[winder.types.Shape[winder.types.N, Literal[2], Literal[3]], winder.types.float32, winder.types.cuda], beta: float = -1, epsilon: float = -1, stream: int = 0) -> None:
        """
        Compute the partial derivatives w.r.t. the given point positions and scaled normals.


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
            Default for point clouds is 2.0. TODO EXPERIMENT
        `epsilon` : float, optional
            Regularization scale (smoothing radius) used to prevent numerical 
            singularities (NaN/infinity) when queries land near points in point clouds.
            - distance >= 2*epsilon: Acts as standard unregularized potential.
            - distance < 2*epsilon: Smoothly dampens potential to a finite maximum.
            Use any negative number to get the default value (1/250).
        `stream` : int, optional
            The raw 64-bit identifier (handle) of a CUDA stream. 
            Allows enqueuing operations asynchronously within deep learning frameworks.
            For example, in PyTorch pass: `torch.cuda.current_stream().cuda_stream`.
            Default is 0 (the default/null stream).
        """

    @overload
    def dump(self) -> str:
        """
        Returns a detailed .dot string representation of the internal BVH8 Tree.
        """

    @overload
    def dump(self) -> str: ...
