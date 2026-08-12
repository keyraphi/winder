from typing import Literal, override
import torch
import torch.nn as nn
from .functional import winding_mesh, winding_triangles, winding_point_normals

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

    @override
    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        return winding_mesh(
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

    @override
    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        return winding_triangles(
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

    @override
    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        return winding_point_normals(
            self.points,
            self.scaled_normals,
            queries,
            beta_forward=self.beta_forward,
            beta_backward=self.beta_backward,
            epsilon=self.epsilon,
            mode=self.mode,
        )
