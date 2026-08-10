import argparse
import numpy as np
import torch
import winder

# Ensure deterministic execution
torch.manual_seed(42)
np.random.seed(42)


def mesh_to_point_surfels(
    vertices: np.ndarray, indices: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    """Converts a triangle mesh into point locations and scaled normals on GPU."""
    v0 = vertices[indices[:, 0]]
    v1 = vertices[indices[:, 1]]
    v2 = vertices[indices[:, 2]]

    points = (v0 + v1 + v2) / 3.0
    cross = np.cross(v1 - v0, v2 - v0)
    scaled_normals = cross / 2.0

    points_t = torch.from_numpy(points.astype(np.float32)).cuda()
    scaled_normals_t = torch.from_numpy(scaled_normals.astype(np.float32)).cuda()
    return points_t, scaled_normals_t


def generate_fixed_grid_queries(vertices: np.ndarray, num_queries: int) -> torch.Tensor:
    """Generates a repeatable 3D uniform grid around the mesh bounding box."""
    min_box = vertices.min(axis=0)
    max_box = vertices.max(axis=0)
    diag = np.linalg.norm(max_box - min_box)
    min_box -= diag * 0.5
    max_box += diag * 0.5

    side = int(np.ceil(num_queries ** (1.0 / 3.0)))
    x = np.linspace(min_box[0], max_box[0], side)
    y = np.linspace(min_box[1], max_box[1], side)
    z = np.linspace(min_box[2], max_box[2], side)

    gx, gy, gz = np.meshgrid(x, y, z, indexing="ij")
    queries = np.stack([gx.flatten(), gy.flatten(), gz.flatten()], axis=-1)[
        :num_queries
    ]
    return torch.from_numpy(queries.astype(np.float32)).cuda()


def profile_triangle(
    vertices: np.ndarray,
    indices: np.ndarray,
    queries: torch.Tensor,
    grad_output: torch.Tensor,
    beta: float | None,
    warmup: int,
    iters: int,
):
    triangles = vertices[indices]
    triangles_t = torch.from_numpy(triangles.astype(np.float32)).cuda()

    # Build Engine
    torch.cuda.nvtx.range_push("Engine Build")
    grad_engine = winder.GradientEngine(queries, grad_output)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    beta_val = -1.0 if beta is None else beta

    # Warmup
    for _ in range(warmup):
        _ = grad_engine.compute(triangles_t, beta=beta_val)
    torch.cuda.synchronize()

    # Profile Compute Loop
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.nvtx.range_push("Gradient Engine Compute")
    start_event.record()
    for _ in range(iters):
        _ = grad_engine.compute(triangles_t, beta=beta_val)
    end_event.record()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    avg_ms = start_event.elapsed_time(end_event) / iters
    print(
        f"[Triangle Mode] Average Execution Time: {avg_ms:.3f} ms over {iters} iterations"
    )


def profile_point_normal(
    vertices: np.ndarray,
    indices: np.ndarray,
    queries: torch.Tensor,
    grad_output: torch.Tensor,
    inv_epsilon: float,
    beta: float | None,
    warmup: int,
    iters: int,
):
    points_t, scaled_normals_t = mesh_to_point_surfels(vertices, indices)

    # Build Engine
    torch.cuda.nvtx.range_push("Engine Build")
    grad_engine = winder.GradientEngine(queries, grad_output)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    eps = 1.0 / inv_epsilon
    beta_val = -1.0 if beta is None else beta

    # Warmup
    for _ in range(warmup):
        _ = grad_engine.compute(
            points_t,
            scaled_normals_t,
            epsilon=eps,
            beta=beta_val,
            stream=torch.cuda.current_stream().cuda_stream,
        )
    torch.cuda.synchronize()

    # Profile Compute Loop
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.nvtx.range_push("Gradient Engine Compute")
    start_event.record()
    for _ in range(iters):
        _ = grad_engine.compute(
            points_t,
            scaled_normals_t,
            epsilon=eps,
            beta=beta_val,
            stream=torch.cuda.current_stream().cuda_stream,
        )
    end_event.record()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    avg_ms = start_event.elapsed_time(end_event) / iters
    print(
        f"[PointNormal Mode] Average Execution Time: {avg_ms:.3f} ms over {iters} iterations"
    )


if __name__ == "__main__":
    import igl

    parser = argparse.ArgumentParser(
        description="Deterministic Winder Gradient Profiler"
    )
    parser.add_argument("--obj_file", type=str, required=True, help="Path to mesh .obj")
    parser.add_argument(
        "--geometry_type",
        type=str,
        choices=["Triangle", "PointNormal"],
        default="Triangle",
        help="Primitive evaluation mode",
    )
    parser.add_argument(
        "--query_count", type=int, default=100000, help="Grid query points"
    )
    parser.add_argument(
        "--inv_epsilon", type=float, default=250.0, help="Inverse epsilon scale"
    )
    parser.add_argument(
        "--beta", type=float, default=None, help="BH accuracy trade-off parameter"
    )
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=20, help="Measured iterations")

    args = parser.parse_args()

    # Load Mesh
    vertices, _, _, indices, _, _ = igl.readOBJ(args.obj_file)
    print(f"Loaded {len(indices)} primitives | Grid queries: {args.query_count}")

    # Generate repeatable grid queries and gradient outputs
    queries = generate_fixed_grid_queries(vertices, args.query_count)
    grad_output = torch.randn(args.query_count, dtype=torch.float32, device="cuda:0")

    torch.cuda.profiler.start()
    if args.geometry_type == "Triangle":
        profile_triangle(
            vertices,
            indices,
            queries,
            grad_output,
            args.beta,
            args.warmup,
            args.iters,
        )
    else:
        profile_point_normal(
            vertices,
            indices,
            queries,
            grad_output,
            args.inv_epsilon,
            args.beta,
            args.warmup,
            args.iters,
        )
    torch.cuda.profiler.stop()
