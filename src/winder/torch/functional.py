import torch
from typing import Literal
import winder


ModeType = Literal["fast", "brute_force"]


class _MeshWindingAutograd(torch.autograd.Function):
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
    beta_forward: float = -1,
    beta_backward: float = -1,
    mode: ModeType = "fast",
) -> torch.Tensor:
    return _MeshWindingAutograd.apply(
        vertices, indices, queries, beta_forward, beta_backward, mode
    )


class _TriangleWindingAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        triangles: torch.Tensor,
        queries: torch.Tensor,
        beta_forward: float,
        beta_backward: float,
        mode: ModeType,
    ):
        ctx.save_for_backward(triangles, queries)
        ctx.beta_backward = beta_backward
        ctx.mode = mode

        stream = torch.cuda.current_stream().cuda_stream

        if mode == "fast":
            engine = winder.WindingNumberEngine(triangles)
            winding_number = engine.compute(queries, beta=beta_forward, stream=stream)
        else:
            winding_number = winder.brute_force_winding_numbers(
                triangles, queries, stream
            )

        return torch.from_dlpack(winding_number)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
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
    return _TriangleWindingAutograd.apply(
        triangles, queries, beta_forward, beta_backward, mode
    )


class _PointNormalWindingAutograd(torch.autograd.Function):
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
    ):
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
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None, None, None, None, None]:
        points, scaled_normals, queries = ctx.saved_tensors
        stream = torch.cuda.current_stream().cuda_stream

        if ctx.mode == "fast":
            grad_engine = winder.GradientEngine(queries, grad_output)
            # Fast engine compute returns shape (N, 2, 3) where:
            # output[:, 0, :] -> dL/d(scaled_normals)
            # output[:, 1, :] -> dL/d(points)
            grads = torch.from_dlpack(
                grad_engine.compute(
                    points,
                    scaled_normals,
                    beta=ctx.beta_backward,
                    epsilon=ctx.epsilon,
                    stream=stream,
                )
            )
            grads = torch.from_dlpack(grads)
            grad_normals = grads[:, 0, :]
            grad_points = grads[:, 1, :]
        else:
            grads = winder.brute_force_gradients(
                grad_output,
                points,
                scaled_normals,
                queries,
                ctx.epsilon,
                stream,
            )
            grads = torch.from_dlpack(grads)
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
    return _PointNormalWindingAutograd.apply(
        points,
        scaled_normals,
        queries,
        beta_forward,
        beta_backward,
        epsilon,
        mode,
    )
