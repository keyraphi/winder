import argparse
from time import time
import igl
import numpy as np
import torch
import winder


def mesh_to_point_surfels(
    vertices: np.ndarray, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converts a triangle mesh into a point-normal-area representation."""
    v0 = vertices[indices[:, 0]]
    v1 = vertices[indices[:, 1]]
    v2 = vertices[indices[:, 2]]

    points = (v0 + v1 + v2) / 3.0

    e1 = v1 - v0
    e2 = v2 - v0

    cross = np.cross(e1, e2)
    magnitudes = np.linalg.norm(cross, axis=-1, keepdims=True)
    areas = (magnitudes / 2.0).flatten()

    safe_magnitudes = np.where(magnitudes == 0, 1e-8, magnitudes)
    normals = cross / safe_magnitudes

    return (
        points.astype(np.float32),
        normals.astype(np.float32),
        areas.astype(np.float32),
    )


def generate_queries(
    vertices: np.ndarray, mode: str, num_queries: int = 100
) -> np.ndarray:
    """Generates query points either randomly or on a uniform grid around the mesh bounding box."""
    min_box = vertices.min(axis=0)
    max_box = vertices.max(axis=0)
    diag = np.linalg.norm(max_box - min_box)
    min_box -= diag * 0.5
    max_box += diag * 0.5
    print("DEBUG: queries created in:", min_box, max_box)

    if mode == "grid":
        side = int(np.ceil(num_queries ** (1.0 / 3.0)))
        x = np.linspace(min_box[0], max_box[0], side)
        y = np.linspace(min_box[1], max_box[1], side)
        z = np.linspace(min_box[2], max_box[2], side)
        gx, gy, gz = np.meshgrid(x, y, z)
        queries = np.stack([gx.flatten(), gy.flatten(), gz.flatten()], axis=-1)
        return queries[:num_queries].astype(np.float32)
    else:
        return np.random.uniform(min_box, max_box, size=(num_queries, 3)).astype(
            np.float32
        )


def validate_gradients(
    winder_grads: np.ndarray,
    brute_force_grads: np.ndarray,
    label: str,
    signal_threshold: float = 1e-5,
):
    """Validates winder gradients against brute force gradients."""
    winder_flat = winder_grads.reshape(-1)
    brute_force_flat = brute_force_grads.reshape(-1)

    # 1. Global Relative Norm Error: ||g_cuda - g_ref|| / ||g_ref||
    norm_diff = np.linalg.norm(winder_flat - brute_force_flat)
    norm_ref = np.linalg.norm(brute_force_flat)
    global_rel_norm_err = norm_diff / (norm_ref + 1e-12)

    # 2. Cosine Similarity (Direction Alignment)
    dot_prod = np.dot(winder_flat, brute_force_flat)
    norm_winder = np.linalg.norm(winder_flat)
    cosine_sim = dot_prod / (norm_winder * norm_ref + 1e-12)

    # 3. Masked Relative Error (Evaluate on non-zero gradient signals)
    mask = np.abs(brute_force_flat) > signal_threshold
    if np.any(mask):
        abs_err_masked = np.abs(winder_flat[mask] - brute_force_flat[mask])
        rel_err_masked = abs_err_masked / np.abs(brute_force_flat[mask])
        mean_masked_rel = np.mean(rel_err_masked)
        max_masked_rel = np.max(rel_err_masked)
    else:
        mean_masked_rel = 0.0
        max_masked_rel = 0.0

    print(f"\n=== Gradient Validation Results: {label} ===")
    print(f"  -> Global Relative Norm Error: {global_rel_norm_err:.6f}")
    print(
        f"  -> Vector Cosine Similarity:  {cosine_sim:.6f}  (1.000000 = Perfect alignment)"
    )
    print(
        f"  -> Signal-Masked Mean Rel Err: {mean_masked_rel:.6f}  (where |g| > {signal_threshold})"
    )
    print(f"  -> Signal-Masked Max Rel Err:  {max_masked_rel:.6f}")

    # Tightened validation threshold: Cosine Sim > 0.9999 and Rel Norm Error < 0.001 (0.1%)
    if cosine_sim > 0.9999 and global_rel_norm_err < 1e-3:
        print(
            f"\033[92m  ✓ SUCCESS: {label} winder gradients match brute force ground truth!\033[0m"
        )
    else:
        print(
            f"\033[91m  ✗ FAILURE: Significant directional or magnitude mismatch in {label}.\033[0m"
        )


def test_triangle_gradients(
    vertices: np.ndarray,
    indices: np.ndarray,
    query_mode: str,
    query_count: int,
    beta: None | float,
):
    triangles = vertices[indices]
    triangles_torch = torch.from_numpy(triangles).to(torch.float32).to("cuda:0")
    grad_output = torch.randn([query_count], dtype=torch.float32, device="cuda:0") * 100
    queries = generate_queries(vertices, query_mode, query_count)
    queries_torch = torch.from_numpy(queries).to(torch.float32).to("cuda:0")
    torch.cuda.synchronize()
    print("Forward:")
    t0 = time()
    wn = winder.brute_force_winding_numbers(triangles_torch, queries_torch)
    torch.cuda.synchronize()
    print(f"Brute force took {time() - t0} sec")
    t0 = time()
    engine = winder.WindingNumberEngine(triangles_torch)
    torch.cuda.synchronize()
    print(f"Building engine took {time() - t0} sec")
    t0 = time()
    wn = engine.compute(queries_torch)
    torch.cuda.synchronize()
    print(f"Computing winding numbers took {time() - t0} sec")

    print("Backward:")
    torch.cuda.synchronize()
    t0 = time()
    gt_grads = torch.from_dlpack(
        winder.brute_force_gradients(grad_output, triangles_torch, queries_torch)
    )
    torch.cuda.synchronize()
    print(f"Brute Force took {time() - t0} sec")

    t0 = time()
    grad_engine = winder.GradientEngine(queries_torch, grad_output)
    torch.cuda.synchronize()
    with open("/tmp/grad_engine_dump.dot", "w") as f:
        f.write(grad_engine.dump())
    print(f"Building Engine took {time() - t0} sec")
    t0 = time()
    grads = torch.from_dlpack(grad_engine.compute(triangles_torch, beta=-1 if beta is None else beta))
    torch.cuda.synchronize()
    print(f"Fast variant took {time() - t0} sec")

    validate_gradients(grads.cpu().numpy(), gt_grads.cpu().numpy(), "Triangle")


def test_point_normal_gradients(
    points: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    query_mode: str,
    query_count: int,
    inv_epsilon: float,
    beta: None | float,
):
    scaled_normals = normals * areas[:, None]
    points_torch = torch.from_numpy(points).to(torch.float32).to("cuda:0")
    scaled_normals_torch = (
        torch.from_numpy(scaled_normals).to(torch.float32).to("cuda:0")
    )
    grad_output = torch.randn([query_count], dtype=torch.float32, device="cuda:0")
    queries = generate_queries(points, query_mode, query_count)
    queries_torch = torch.from_numpy(queries).to(torch.float32).to("cuda:0")

    torch.cuda.synchronize()
    print("forward:")
    t0 = time()
    wn = winder.brute_force_winding_numbers(points_torch, scaled_normals_torch, queries_torch)
    torch.cuda.synchronize()
    print(f"Brute force took {time() - t0} sec")
    t0 = time()
    engine = winder.WindingNumberEngine(points_torch, scaled_normals_torch)
    torch.cuda.synchronize()
    print(f"Building engine took {time() - t0} sec")
    t0 = time()
    wn = engine.compute(queries_torch)
    torch.cuda.synchronize()
    print(f"Computing winding numbers took {time() - t0} sec")

    print("Backward:")
    t0 = time()
    gt_grads = torch.from_dlpack(
        winder.brute_force_gradients(
            grad_output,
            points_torch,
            scaled_normals_torch,
            queries_torch,
            epsilon=1 / inv_epsilon,
        )
    )
    torch.cuda.synchronize()
    print(f"Brute Force took {time() - t0} sec")

    t0 = time()
    grad_engine = winder.GradientEngine(queries_torch, grad_output)
    torch.cuda.synchronize()
    print(f"Building engine took {time() - t0} sec")

    t0 = time()
    grads = torch.from_dlpack(
        grad_engine.compute(
            points_torch, scaled_normals_torch, epsilon=1 / inv_epsilon, beta=-1 if beta is None else beta
        )
    )
    torch.cuda.synchronize()
    print(f"Fast Grads took {time() - t0} sec")

    validate_gradients(grads.cpu().numpy(), gt_grads.cpu().numpy(), "PointNormal")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify generalized winding number analytical gradients against PyTorch float64 autograd."
    )
    parser.add_argument(
        "--obj_file",
        type=str,
        required=True,
        help="Path to input wave-front .obj mesh file.",
    )
    parser.add_argument(
        "--geometry_type",
        type=str,
        choices=["PointNormal", "Triangle", "both"],
        default="both",
        help="Choose primitive type",
    )
    parser.add_argument(
        "--query_mode",
        type=str,
        choices=["random", "grid"],
        default="random",
        help="Distribution algorithm geometry configuration for query target fields.",
    )
    parser.add_argument(
        "--query_count",
        type=int,
        default=1000,
        help="Number of query points",
    )
    parser.add_argument(
        "--inv_epsilon",
        type=float,
        default=250.0,
        help="Inverse regularization scale for PointNormal surfels",
    )
    parser.add_argument(
        "--beta",
        type=float,
        help="How much to approximate",
    )

    args = parser.parse_args()

    print(f"Loading mesh structural data from: {args.obj_file}")
    vertices, _, _, indices, _, _ = igl.readOBJ(args.obj_file)

    print(f"INFO: object has {len(indices)} Triangles/PointNormals")

    if args.geometry_type in ["PointNormal", "both"]:
        points, normals, areas = mesh_to_point_surfels(vertices, indices)
        test_point_normal_gradients(
            points,
            normals,
            areas,
            args.query_mode,
            args.query_count,
            args.inv_epsilon,
            args.beta,
        )

    if args.geometry_type in ["Triangle", "both"]:
        test_triangle_gradients(
            vertices, indices, args.query_mode, args.query_count, args.beta
        )
