#!/usr/bin/env python3
import numpy as np
import time
import torch
import winder
import argparse
import imageio.v3 as iio
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


device = torch.device("cuda:0")


def positive_type(arg: str) -> int:
    x: int = int(arg)
    if x < 1:
        raise argparse.ArgumentTypeError("Minimum value is 1")
    return x


def create_mesh_sphere(
    target_triangle_count: int, sphere_id: int = 0, radius: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    # Estimate rings/sectors to match target_triangle_count
    # Total triangles approx = 2 * sectors * (rings - 1)
    # For a balanced sphere, sectors approx 2 * rings
    rings = int((target_triangle_count / 4) ** 0.5) + 1
    sectors = 2 * rings

    theta = torch.linspace(0, torch.pi, rings, device=device)
    phi = torch.linspace(0, 2 * torch.pi, sectors, device=device)

    grid_theta, grid_phi = torch.meshgrid(theta, phi, indexing="ij")

    x = radius * torch.sin(grid_theta) * torch.cos(grid_phi)
    y = radius * torch.sin(grid_theta) * torch.sin(grid_phi)
    z = radius * torch.cos(grid_theta)

    verts = torch.stack([x, y, z], dim=-1).view(-1, 3)

    if sphere_id > 0:
        direction = torch.randn([1, 3], device=device)
        direction = direction / torch.linalg.norm(direction, keepdim=True)
        offset = 2 * radius * direction
        verts = verts + offset

    indices = []
    for r in tqdm(range(rings - 1), desc="Creating Triangles"):
        for s in range(sectors - 1):
            i0 = r * sectors + s
            i1 = r * sectors + (s + 1)
            i2 = (r + 1) * sectors + s
            i3 = (r + 1) * sectors + (s + 1)

            # Triangle 1
            if r != 0:
                indices.append([i0, i2, i1])
            # Triangle 2
            if r != rings - 2:
                indices.append([i1, i2, i3])

    faces = torch.tensor(indices, dtype=torch.uint32, device=device)
    return verts, faces


def create_triangle_sphere(
    triangle_count: int, sphere_id: int = 0, radius: float = 1.0
) -> torch.Tensor:
    verts, faces = create_mesh_sphere(triangle_count, sphere_id, radius)
    # Convert indexed mesh to independent triangles [N, 3, 3]
    return verts[faces.long()]


def create_point_sphere(
    point_count: int, sphere_id: int = 0, radius: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Creates a sphere using a Fibonacci Lattice for near-uniform density."""
    indices = torch.arange(0, point_count, dtype=torch.float32, device=device)

    # Golden ratio increment
    phi = torch.acos(1 - 2 * (indices + 0.5) / point_count)
    theta = torch.pi * (1 + 5**0.5) * indices

    x = torch.cos(theta) * torch.sin(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(phi)

    points = torch.stack([x, y, z], dim=1) * radius

    # Each point represents exactly 1/N of the surface area
    area_per_point = (4 * torch.pi * radius**2) / point_count
    normals = (points / radius) * area_per_point

    if sphere_id > 0:
        direction = torch.randn([1, 3], device=device)
        direction = direction / torch.linalg.norm(direction, keepdim=True)
        offset = 2 * radius * direction
        points = points + offset

    return points, normals


def create_query_grid(resolution: int, extend: float = 5) -> torch.Tensor:
    xs = torch.linspace(-extend, extend, resolution, device=device)
    xss, yss, zss = torch.meshgrid([xs, xs, xs], indexing="ij")
    queries = torch.stack([xss, yss, zss], dim=-1)
    queries = queries.reshape([-1, 3])
    return queries


def main():
    parser = argparse.ArgumentParser()
    _ = parser.add_argument(
        "--geometry_count",
        type=positive_type,
        default=1000000,
        help="How many primitives (triangles or points) to use per sphere.",
    )
    _ = parser.add_argument(
        "--sphere_count",
        type=positive_type,
        default=1,
        help="Number of spheres to create",
    )
    _ = parser.add_argument(
        "--query_grid_resolution",
        type=positive_type,
        default=512,
        help="Resolution per dimension of query grid",
    )
    _ = parser.add_argument(
        "--grid_extend",
        type=float,
        default=5.0,
        help="Query grid extend. Sphere radius is 1.",
    )
    _ = parser.add_argument(
        "--beta",
        type=float,
        default=-1,
        help="Approximation factor. Larger value means more approximation. Default is around 2",
    )
    _ = parser.add_argument(
        "--geometry_type",
        choices=["triangles", "mesh", "points"],
        default="triangles",
        help="What kind of primitives to use. The triangles and mesh option result in the same construction.",
    )
    _ = parser.add_argument(
        "--evaluation_rounds",
        type=positive_type,
        default=1,
        help="Number of times to run the evaluation (for more robust timing)",
    )
    _ = parser.add_argument(
        "--figure_path", type=str, help="Optional path to safe figures."
    )
    args = parser.parse_args()

    torch.random.manual_seed(1)

    match args.geometry_type:
        case "triangles":
            triangles = []
            for i in range(args.sphere_count):
                tris = create_triangle_sphere(args.geometry_count, i)
                triangles.append(tris)
            triangles = torch.concatenate(triangles, dim=0)
            print(
                "DEBUG: triangles",
                triangles.shape,
                triangles.dtype,
                triangles.device,
            )
            engine = winder.WinderEngine(triangles)
        case "mesh":
            vertices = []
            indices = []
            for i in range(args.sphere_count):
                verts, inds = create_mesh_sphere(args.geometry_count, i)
                vertices.append(verts)
                indices.append(inds)
            vertices = torch.concatenate(vertices, dim=0)
            indices = torch.concatenate(indices, dim=0)
            print("DEBUG: vertices", vertices.shape, vertices.dtype, vertices.device)
            print("DEBUG: indices", indices.shape, indices.dtype, indices.device)
            engine = winder.WinderEngine(vertices, indices)
        case "points":
            points = []
            normals = []
            for i in range(args.sphere_count):
                pts, ns = create_point_sphere(args.geometry_count, i)
                points.append(pts)
                normals.append(ns)
            points = torch.concatenate(points, dim=0)
            normals = torch.concatenate(normals, dim=0)
            print("DEBUG: points", points.shape, points.dtype, points.device)
            print("DEBUG: normals", normals.shape, normals.dtype, normals.device)
            engine = winder.WinderEngine(points, normals)
        case _:
            raise RuntimeError("Unknown geometry_type")

    queries = create_query_grid(args.query_grid_resolution, args.grid_extend)

    for _ in tqdm(range(args.evaluation_rounds), desc="Eval. Repetitions"):
        winding_number_grid = engine.compute(queries, beta=args.beta, stream=torch.cuda.current_stream().cuda_stream)

    torch.cuda.synchronize()

if __name__ == "__main__":
    main()
