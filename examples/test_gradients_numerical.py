import argparse
import math
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


def pytorch_triangle_winding_grads_chunked_64(
    vertices: torch.Tensor,  # Shape: [M, 3, 3] float64
    queries: torch.Tensor,  # Shape: [Q, 3] float64
    grad_output: torch.Tensor,  # Shape: [Q] float64
    chunk_size: int = 50,
) -> torch.Tensor:
    """Computes exact float64 autograd gradients for Triangles matching CUDA formulation."""
    v = vertices.clone().detach().requires_grad_(True)
    num_queries = queries.shape[0]
    inv_two_pi = 1.0 / (2.0 * math.pi)

    for i in range(0, num_queries, chunk_size):
        q_chunk = queries[i : i + chunk_size]
        g_chunk = grad_output[i : i + chunk_size]

        v0 = v[:, 0, :][None, :, :]
        v1 = v[:, 1, :][None, :, :]
        v2 = v[:, 2, :][None, :, :]
        q = q_chunk[:, None, :]

        a = v0 - q
        b = v1 - q
        c = v2 - q

        a2 = torch.sum(a * a, dim=-1) + 1e-30
        b2 = torch.sum(b * b, dim=-1) + 1e-30
        c2 = torch.sum(c * c, dim=-1) + 1e-30

        inv_a = torch.rsqrt(a2)
        inv_b = torch.rsqrt(b2)
        inv_c = torch.rsqrt(c2)

        cos_ab = torch.sum(a * b, dim=-1) * inv_a * inv_b
        cos_ac = torch.sum(a * c, dim=-1) * inv_a * inv_c
        cos_bc = torch.sum(b * c, dim=-1) * inv_b * inv_c

        cross_bc = torch.cross(b, c, dim=-1)
        det_norm = torch.sum(a * cross_bc, dim=-1) * inv_a * inv_b * inv_c
        div_norm = 1.0 + cos_ab + cos_ac + cos_bc

        # Direct atan2 evaluation without artificial 1e-6 debug truncation
        sol_angle = torch.atan2(det_norm, div_norm) * inv_two_pi

        loss_chunk = torch.sum(sol_angle * g_chunk[:, None])
        loss_chunk.backward()

    return v.grad


def validate_gradients(
    cuda_grads: np.ndarray,
    ref_grads: np.ndarray,
    label: str,
    signal_threshold: float = 1e-5,
):
    """Validates CUDA gradients against reference float64 autograd."""
    cuda_flat = cuda_grads.reshape(-1)
    ref_flat = ref_grads.reshape(-1)

    # 1. Global Relative Norm Error: ||g_cuda - g_ref|| / ||g_ref||
    norm_diff = np.linalg.norm(cuda_flat - ref_flat)
    norm_ref = np.linalg.norm(ref_flat)
    global_rel_norm_err = norm_diff / (norm_ref + 1e-12)

    # 2. Cosine Similarity (Direction Alignment)
    dot_prod = np.dot(cuda_flat, ref_flat)
    norm_cuda = np.linalg.norm(cuda_flat)
    cosine_sim = dot_prod / (norm_cuda * norm_ref + 1e-12)

    # 3. Masked Relative Error (Evaluate on non-zero gradient signals)
    mask = np.abs(ref_flat) > signal_threshold
    if np.any(mask):
        abs_err_masked = np.abs(cuda_flat[mask] - ref_flat[mask])
        rel_err_masked = abs_err_masked / np.abs(ref_flat[mask])
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
            f"\033[92m  ✓ SUCCESS: {label} CUDA gradients match PyTorch autograd ground truth!\033[0m"
        )
    else:
        print(
            f"\033[91m  ✗ FAILURE: Significant directional or magnitude mismatch in {label}.\033[0m"
        )


def pytorch_point_normal_grads_chunked_64(
    points: torch.Tensor,  # Shape: [M, 3] float64
    normals: torch.Tensor,  # Shape: [M, 3] float64 (area-weighted)
    queries: torch.Tensor,  # Shape: [Q, 3] float64
    grad_output: torch.Tensor,  # Shape: [Q] float64
    inv_epsilon: float = 1.0,
    s_regularization_fn=None,
    chunk_size: int = 50,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes exact float64 autograd gradients for Point-Normal surfels matching CUDA."""
    p = points.clone().detach().requires_grad_(True)
    n = normals.clone().detach().requires_grad_(True)
    num_queries = queries.shape[0]

    four_over_3sqrt_pi = 4.0 / (3.0 * math.sqrt(math.pi))
    inv_four_pi = 1.0 / (4.0 * math.pi)
    near_limit_constant = four_over_3sqrt_pi * (inv_epsilon**3)

    for i in range(0, num_queries, chunk_size):
        q_chunk = queries[i : i + chunk_size]
        g_chunk = grad_output[i : i + chunk_size]

        p_bc = p[None, :, :]
        n_bc = n[None, :, :]
        q_bc = q_chunk[:, None, :]

        d = p_bc - q_bc
        dist2 = torch.sum(d * d, dim=-1)

        inv_distance = torch.rsqrt(dist2 + 1e-30)
        inv_dist2 = inv_distance * inv_distance
        inv_dist3 = inv_dist2 * inv_distance

        distance = dist2 * inv_distance
        t = distance * inv_epsilon

        if s_regularization_fn is not None:
            s_val = s_regularization_fn(t)
        else:
            s_val = torch.ones_like(t)

        s_reg_term = s_val * inv_dist3

        s_over_dist3 = torch.where(
            t < 0.1,
            near_limit_constant,
            torch.where(t < 2.0, s_reg_term, inv_dist3),
        )

        dot_n_d = torch.sum(n_bc * d, dim=-1)
        contributions = dot_n_d * inv_four_pi * s_over_dist3

        loss_chunk = torch.sum(contributions * g_chunk[:, None])
        loss_chunk.backward()

    return p.grad, n.grad


def test_triangle_gradients(
    vertices: np.ndarray,
    indices: np.ndarray,
    query_mode: str,
    query_count: int = 100,
):
    print("\n=== Testing Triangle Gradients against PyTorch float64 Autograd ===")

    flat_vertices = vertices[indices.flatten()]
    flat_indices = np.arange(len(flat_vertices)).reshape(-1, 3)
    num_triangles = len(flat_indices)

    queries = generate_queries(flat_vertices, query_mode, query_count)
    grad_output = np.random.normal(size=(query_count,)).astype(np.float32)

    v_tensor = torch.from_numpy(flat_vertices).cuda()
    i_tensor = torch.from_numpy(flat_indices).int().cuda()
    q_tensor = torch.from_numpy(queries).cuda()
    g_out_tensor = torch.from_numpy(grad_output).cuda()

    engine = winder.WinderEngine(v_tensor, i_tensor)
    cuda_grads = torch.from_dlpack(
        engine.gradients(
            queries=q_tensor, grad_output=g_out_tensor, is_brute_force=True
        )
    )

    print(
        f"Evaluating PyTorch float64 autograd across ALL {num_triangles} triangles..."
    )
    v_64 = v_tensor.reshape(-1, 3, 3).to(torch.float64)
    q_64 = q_tensor.to(torch.float64)
    g_out_64 = g_out_tensor.to(torch.float64)

    ref_grads_64 = pytorch_triangle_winding_grads_chunked_64(v_64, q_64, g_out_64)
    ref_grads_32 = ref_grads_64.to(torch.float32)

    validate_gradients(
        cuda_grads.cpu().numpy(), ref_grads_32.cpu().numpy(), "Triangles"
    )


def test_point_normal_gradients(
    vertices: np.ndarray,
    indices: np.ndarray,
    query_mode: str,
    query_count: int = 100,
    inv_epsilon: float = 1.0,
):
    print("\n=== Testing Point-Normal Gradients against PyTorch float64 Autograd ===")

    pts, normals, areas = mesh_to_point_surfels(vertices, indices)
    scaled_normals = normals * areas[..., None]
    num_points = len(pts)

    queries = generate_queries(vertices, query_mode, query_count)
    grad_output = np.random.normal(size=(query_count,)).astype(np.float32)

    p_tensor = torch.from_numpy(pts).cuda()
    n_tensor = torch.from_numpy(scaled_normals).cuda()
    q_tensor = torch.from_numpy(queries).cuda()
    g_out_tensor = torch.from_numpy(grad_output).cuda()

    engine = winder.WinderEngine(p_tensor, n_tensor)
    cuda_grads = torch.from_dlpack(
        engine.gradients(
            queries=q_tensor,
            grad_output=g_out_tensor,
            epsilon=1.0/inv_epsilon,
            is_brute_force=True,
        )
    )

    cuda_n_grads = cuda_grads[:, 0, :]
    cuda_p_grads = cuda_grads[:, 1, :]

    print(f"Evaluating PyTorch float64 autograd across ALL {num_points} surfels...")
    p_64 = p_tensor.to(torch.float64)
    n_64 = n_tensor.to(torch.float64)
    q_64 = q_tensor.to(torch.float64)
    g_out_64 = g_out_tensor.to(torch.float64)

    ref_p_grad_64, ref_n_grad_64 = pytorch_point_normal_grads_chunked_64(
        p_64, n_64, q_64, g_out_64, inv_epsilon=inv_epsilon
    )

    ref_p_grad_32 = ref_p_grad_64.to(torch.float32)
    ref_n_grad_32 = ref_n_grad_64.to(torch.float32)

    validate_gradients(
        cuda_n_grads.cpu().numpy(), ref_n_grad_32.cpu().numpy(), "Normal"
    )
    validate_gradients(
        cuda_p_grads.cpu().numpy(), ref_p_grad_32.cpu().numpy(), "Position"
    )


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
        default=100,
        help="Number of query points",
    )
    parser.add_argument(
        "--inv_epsilon",
        type=float,
        default=250.0,
        help="Inverse regularization scale for PointNormal surfels",
    )

    args = parser.parse_args()

    print(f"Loading mesh structural data from: {args.obj_file}")
    vertices, _, _, indices, _, _ = igl.readOBJ(args.obj_file)

    if args.geometry_type in ["Triangle", "both"]:
        test_triangle_gradients(
            vertices,
            indices,
            args.query_mode,
            args.query_count,
        )

    if args.geometry_type in ["PointNormal", "both"]:
        test_point_normal_gradients(
            vertices,
            indices,
            args.query_mode,
            args.query_count,
            args.inv_epsilon,
        )
