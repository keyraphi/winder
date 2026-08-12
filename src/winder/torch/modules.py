from typing import Literal
import torch
import torch.nn as nn
from . import functional as WF

ModeType = Literal["fast", "brute_force"]


class MeshWindingField(nn.Module):
    """Winding field for indexed triangle meshes."""

    def __init__(
        self,
        vertices: torch.Tensor,
        indices: torch.Tensor,
        beta_forward: float = -1.0,
        beta_backward: float = -1.0,
        mode: ModeType = "fast",
    ):
        super().__init__()
        if not isinstance(vertices, nn.Parameter):
            self.vertices: torch.Tensor = nn.Parameter(
                vertices.clone().detach().float()
            )
        else:
            self.vertices: torch.Tensor = vertices

        self.indices: torch.Tensor
        self.register_buffer("indices", indices.clone().detach().to(torch.uint32))
        self.beta_forward: float = beta_forward
        self.beta_backward: float = beta_backward
        self.mode: ModeType = mode

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        return WF.winding_mesh(
            self.vertices,
            self.indices,
            queries,
            beta_forward=self.beta_forward,
            beta_backward=self.beta_backward,
            mode=self.mode,
        )


class TriangleWindingField(nn.Module):
    """Winding field for explicit unindexed triangle soups."""

    def __init__(
        self,
        triangles: torch.Tensor,
        beta_forward: float = -1.0,
        beta_backward: float = -1.0,
        mode: ModeType = "fast",
    ):
        super().__init__()
        if not isinstance(triangles, nn.Parameter):
            self.triangles: torch.Tensor = nn.Parameter(
                triangles.clone().detach().float()
            )
        else:
            self.triangles: torch.Tensor = triangles

        self.beta_forward: float = beta_forward
        self.beta_backward: float = beta_backward
        self.mode: ModeType = mode

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        return WF.winding_triangles(
            self.triangles,
            queries,
            beta_forward=self.beta_forward,
            beta_backward=self.beta_backward,
            mode=self.mode,
        )


class PointNormalWindingField(nn.Module):
    """Winding field for surfel point clouds."""

    def __init__(
        self,
        points: torch.Tensor,
        scaled_normals: torch.Tensor,
        beta_forward: float = -1.0,
        beta_backward: float = -1.0,
        epsilon: float = -1.0,
        mode: ModeType = "fast",
    ):
        super().__init__()
        if not isinstance(points, nn.Parameter):
            self.points: torch.Tensor = nn.Parameter(points.clone().detach().float())
        else:
            self.points: torch.Tensor = points

        if not isinstance(scaled_normals, nn.Parameter):
            self.scaled_normals: torch.Tensor = nn.Parameter(
                scaled_normals.clone().detach().float()
            )
        else:
            self.scaled_normals: torch.Tensor = scaled_normals

        self.beta_forward: float = beta_forward
        self.beta_backward: float = beta_backward
        self.epsilon: float = epsilon
        self.mode: ModeType = mode

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        return WF.winding_point_normals(
            self.points,
            self.scaled_normals,
            queries,
            beta_forward=self.beta_forward,
            beta_backward=self.beta_backward,
            epsilon=self.epsilon,
            mode=self.mode,
        )


def WindingNumberField(*args, **kwargs) -> nn.Module:
    """Factory function creating the appropriate WindingNumberField subclass based on positional signature or keyword arguments.

    Signatures:
      - WindingNumberField(vertices, indices, ...)       -> MeshWindingField
      - WindingNumberField(triangles, ...)               -> TriangleWindingField
      - WindingNumberField(points, scaled_normals, ...)  -> PointNormalWindingField
    """
    if len(args) == 2:
        arg0, arg1 = args[0], args[1]
        if (
            isinstance(arg1, torch.Tensor)
            and not torch.is_floating_point(arg1)
            and arg1.ndim == 2
            and arg1.shape[-1] == 3
        ):
            return MeshWindingField(*args, **kwargs)
        elif (
            isinstance(arg0, torch.Tensor)
            and isinstance(arg1, torch.Tensor)
            and arg0.shape == arg1.shape
            and arg0.shape[-1] == 3
        ):
            return PointNormalWindingField(*args, **kwargs)

    elif len(args) == 1:
        arg0 = args[0]
        if (
            isinstance(arg0, torch.Tensor)
            and arg0.ndim == 3
            and arg0.shape[1:] == (3, 3)
        ):
            return TriangleWindingField(*args, **kwargs)

    if "vertices" in kwargs and "indices" in kwargs:
        return MeshWindingField(**kwargs)
    elif "triangles" in kwargs:
        return TriangleWindingField(**kwargs)
    elif "points" in kwargs and "scaled_normals" in kwargs:
        return PointNormalWindingField(**kwargs)

    raise ValueError(
        "Invalid signature for WindingNumberField. Provide either (vertices, indices), "
        "(triangles,), or (points, scaled_normals)."
    )
