"""PyTorch functional wrappers and Autograd functions for differentiable 3D winding numbers.

Provides differentiable winding number calculation across three geometry formats:
    1. Indexed Triangle Meshes (vertices + indices)
    2. Explicit Triangle Soups (triangles)
    3. Surfel Point Clouds (points + area-scaled normals)
"""

from __future__ import annotations

from typing import Literal, override
import torch
import winder

ModeType = Literal["fast", "brute_force"]


class _MeshWindingAutograd(torch.autograd.Function):
    """Autograd function evaluating winding numbers for indexed triangle meshes."""

    @staticmethod
    def forward(
        ctx,
        vertices: torch.Tensor,
        indices: torch.Tensor,
        queries: torch.Tensor,
        beta_forward: float,
        beta_backward: float,
        mode: ModeType,
    ) -> torch.Tensor:
        ctx.save_for_backward(vertices, indices, queries)
        ctx.beta_backward = beta_backward
        ctx.mode = mode

        stream = torch.cuda.current_stream().cuda_stream

        if mode == "fast":
            engine = winder.WindingNumberEngine(vertices, indices)
            winding_numbers = engine.compute(queries, beta=beta_forward, stream=stream)
        else:
            winding_numbers = winder.brute_force_winding_numbers(
                vertices, indices, queries, stream
            )

        return torch.from_dlpack(winding_numbers)

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None, None]:
        vertices, indices, queries = ctx.saved_tensors
        stream = torch.cuda.current_stream().cuda_stream

        if ctx.mode == "fast":
            grad_engine = winder.GradientEngine(queries, grad_output)
            grad_v = grad_engine.compute(
                vertices, indices, beta=ctx.beta_backward, stream=stream
            )
        else:
            grad_v = winder.brute_force_gradients(
                grad_output, vertices, indices, queries, stream
            )

        return (torch.from_dlpack(grad_v), None, None, None, None, None)


def winding_mesh(
    vertices: torch.Tensor,
    indices: torch.Tensor,
    queries: torch.Tensor,
    beta_forward: float = -1.0,
    beta_backward: float = -1.0,
    mode: ModeType = "fast",
) -> torch.Tensor:
    r"""Computes differentiable winding numbers for indexed triangle meshes.

    Parameters
    ----------
    vertices : torch.Tensor
        A ``(V, 3)`` float32 CUDA tensor of 3D mesh vertex coordinates.
    indices : torch.Tensor
        A ``(N, 3)`` uint32 CUDA tensor of triangle face vertex indices.
    queries : torch.Tensor
        A ``(M, 3)`` float32 CUDA tensor of 3D query positions.
    beta_forward : float, optional
        Accuracy trade-off parameter for BVH evaluation during the forward pass.
        Higher values increase precision at the expense of computational speed.
        A negative value defaults to the internal C++ default (e.g. 2.0).
        Default is -1.0.
    beta_backward : float, optional
        Accuracy trade-off parameter for BVH evaluation during gradient computation
        in the backward pass. Allows using a faster/coarser approximation for gradients.
        A negative value uses the C++ default. Default is -1.0.
    mode : ``{"fast", "brute_force"}``, optional
        Evaluation backend:

        * ``"fast"``: Uses BVH-accelerated multipole approximation :math:`O(M \log N)`.
        * ``"brute_force"``: Exact direct evaluation :math:`O(M \cdot N)`.

        Default is ``"fast"``.

    Returns
    -------
    torch.Tensor
        A ``(M,)`` float32 CUDA tensor containing winding numbers at each query location.

    Examples
    --------
    >>> import torch
    >>> from winder.functional import winding_mesh
    >>> verts = torch.rand((100, 3), device='cuda', requires_grad=True)
    >>> faces = torch.randint(0, 100, (200, 3), device='cuda', dtype=torch.uint32)
    >>> queries = torch.rand((1000, 3), device='cuda')
    >>> wn = winding_mesh(verts, faces, queries)
    >>> loss = wn.sum()
    >>> loss.backward()
    """
    return _MeshWindingAutograd.apply(
        vertices, indices, queries, beta_forward, beta_backward, mode
    )


class _TriangleWindingAutograd(torch.autograd.Function):
    """Autograd function evaluating winding numbers for explicit triangle soups."""

    @staticmethod
    def forward(
        ctx,
        triangles: torch.Tensor,
        queries: torch.Tensor,
        beta_forward: float,
        beta_backward: float,
        mode: ModeType,
    ) -> torch.Tensor:
        ctx.save_for_backward(triangles, queries)
        ctx.beta_backward = beta_backward
        ctx.mode = mode

        stream = torch.cuda.current_stream().cuda_stream

        if mode == "fast":
            engine = winder.WindingNumberEngine(triangles)
            winding_number = engine.compute(queries, beta=beta_forward, stream=stream)
        else:
            winding_number = winder.brute_force_winding_numbers(
                triangles, queries, stream=stream
            )

        return torch.from_dlpack(winding_number)

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None]:
        triangles, queries = ctx.saved_tensors
        stream = torch.cuda.current_stream().cuda_stream

        if ctx.mode == "fast":
            grad_engine = winder.GradientEngine(queries, grad_output)
            grad_t = grad_engine.compute(
                triangles, beta=ctx.beta_backward, stream=stream
            )
        else:
            grad_t = winder.brute_force_gradients(
                grad_output, triangles, queries, stream
            )

        return torch.from_dlpack(grad_t), None, None, None, None


def winding_triangles(
    triangles: torch.Tensor,
    queries: torch.Tensor,
    beta_forward: float = -1.0,
    beta_backward: float = -1.0,
    mode: ModeType = "fast",
) -> torch.Tensor:
    r"""Computes differentiable winding numbers for unindexed triangle soups.

    Parameters
    ----------
    triangles : torch.Tensor
        A ``(N, 3, 3)`` float32 CUDA tensor containing 3D vertex coordinates for each triangle.
    queries : torch.Tensor
        A ``(M, 3)`` float32 CUDA tensor of 3D evaluation coordinates.
    beta_forward : float, optional
        Accuracy parameter for forward pass BVH evaluation. Default is -1.0.
    beta_backward : float, optional
        Accuracy parameter for backward pass gradient computation. Default is -1.0.
    mode : {"fast", "brute_force"}, optional
        Computation backend (``"fast"`` for tree-accelerated or ``"brute_force"`` for exact).
        Default is ``"fast"``.

    Returns
    -------
    torch.Tensor
        A ``(M,)`` float32 CUDA tensor holding evaluated winding values.
    """
    return _TriangleWindingAutograd.apply(
        triangles, queries, beta_forward, beta_backward, mode
    )


class _PointNormalWindingAutograd(torch.autograd.Function):
    """Autograd function evaluating winding numbers for surfel point clouds."""

    @staticmethod
    def forward(
        ctx,
        points: torch.Tensor,
        scaled_normals: torch.Tensor,
        queries: torch.Tensor,
        beta_forward: float,
        beta_backward: float,
        epsilon: float,
        mode: ModeType,
    ) -> torch.Tensor:
        ctx.save_for_backward(points, scaled_normals, queries)
        ctx.beta_backward = beta_backward
        ctx.epsilon = epsilon
        ctx.mode = mode

        stream = torch.cuda.current_stream().cuda_stream

        if mode == "fast":
            engine = winder.WindingNumberEngine(points, scaled_normals)
            winding_numbers = engine.compute(
                queries, beta=beta_forward, epsilon=epsilon, stream=stream
            )
        else:
            winding_numbers = winder.brute_force_winding_numbers(
                points, scaled_normals, queries, epsilon, stream
            )

        return torch.from_dlpack(winding_numbers)

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None, None, None, None, None]:
        points, scaled_normals, queries = ctx.saved_tensors
        stream = torch.cuda.current_stream().cuda_stream

        if ctx.mode == "fast":
            grad_engine = winder.GradientEngine(queries, grad_output)
            # Fast engine compute returns shape (N, 2, 3) where:
            # grads[:, 0, :] -> dL / d(scaled_normals)
            # grads[:, 1, :] -> dL / d(points)
            grads = torch.from_dlpack(
                grad_engine.compute(
                    points,
                    scaled_normals,
                    beta=ctx.beta_backward,
                    epsilon=ctx.epsilon,
                    stream=stream,
                )
            )
        else:
            grads = torch.from_dlpack(
                winder.brute_force_gradients(
                    grad_output,
                    points,
                    scaled_normals,
                    queries,
                    ctx.epsilon,
                    stream,
                )
            )

        grad_normals = grads[:, 0, :]
        grad_points = grads[:, 1, :]

        return grad_points, grad_normals, None, None, None, None, None


def winding_point_normals(
    points: torch.Tensor,
    scaled_normals: torch.Tensor,
    queries: torch.Tensor,
    beta_forward: float = -1.0,
    beta_backward: float = -1.0,
    epsilon: float = -1.0,
    mode: ModeType = "fast",
) -> torch.Tensor:
    r"""Computes differentiable winding numbers for oriented point clouds (surfels).

    Parameters
    ----------
    points : torch.Tensor
        A ``(N, 3)`` float32 CUDA tensor of point positions.
    scaled_normals : torch.Tensor
        A ``(N, 3)`` float32 CUDA tensor of area-scaled normals (direction gives normal,
        magnitude gives local Voronoi area weight).
    queries : torch.Tensor
        A ``(M, 3)`` float32 CUDA tensor of evaluation locations.
    beta_forward : float, optional
        Accuracy parameter for forward pass BVH evaluation. Default is -1.0.
    beta_backward : float, optional
        Accuracy parameter for backward pass gradient computation. Default is -1.0.
    epsilon : float, optional
        Singularity regularization factor controlling kernel dampening near point locations.
        Negative values default to 1/250. Default is -1.0.
    mode : {"fast", "brute_force"}, optional
        Computation strategy (``"fast"`` for tree-accelerated or ``"brute_force"`` for exact).
        Default is ``"fast"``.

    Returns
    -------
    torch.Tensor
        A ``(M,)`` float32 CUDA tensor containing winding numbers for each query location.
    """
    return _PointNormalWindingAutograd.apply(
        points,
        scaled_normals,
        queries,
        beta_forward,
        beta_backward,
        epsilon,
        mode,
    )
