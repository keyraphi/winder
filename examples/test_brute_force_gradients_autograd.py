import argparse
import math
import igl
import numpy as np
import torch
import torch.nn.functional as F
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


def generate_queries(vertices, indices, mode, num_queries=100, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)

    min_box = vertices.min(axis=0)
    max_box = vertices.max(axis=0)
    diag = float(np.linalg.norm(max_box - min_box))
    pad_min = min_box - 0.5 * diag
    pad_max = max_box + 0.5 * diag

    if mode == "random":
        return rng.uniform(pad_min, pad_max, (num_queries, 3)).astype(np.float32)

    if mode == "grid":
        side = int(np.ceil(num_queries ** (1 / 3)))
        x = np.linspace(pad_min[0], pad_max[0], side)
        y = np.linspace(pad_min[1], pad_max[1], side)
        z = np.linspace(pad_min[2], pad_max[2], side)
        gx, gy, gz = np.meshgrid(x, y, z)
        return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], -1)[:num_queries].astype(
            np.float32
        )

    # --- surface-aware modes ---------------------------------------------
    def _sample_barycentric(tri_v, k, rng):
        u = rng.random((k, 1))
        v = rng.random((k, 1))
        flip = (u + v) > 1
        u = np.where(flip, 1 - u, u)
        v = np.where(flip, 1 - v, v)
        w = 1 - u - v
        return u * tri_v[:, 0, :] + v * tri_v[:, 1, :] + w * tri_v[:, 2, :]

    def _tri_normals(tri_v):
        e1 = tri_v[:, 1, :] - tri_v[:, 0, :]
        e2 = tri_v[:, 2, :] - tri_v[:, 0, :]
        n = np.cross(e1, e2)
        n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-30
        return n

    if mode == "near_surface":
        # log-uniform offset from the surface, magnitude swept across many decades
        tri = rng.integers(0, len(indices), size=num_queries)
        tri_v = vertices[indices[tri]]
        pts = _sample_barycentric(tri_v, num_queries, rng)
        n = _tri_normals(tri_v)
        mag = diag * (10.0 ** rng.uniform(-6.0, -1.0, size=(num_queries, 1)))
        sign = (2 * rng.integers(0, 2, size=(num_queries, 1)) - 1).astype(np.float64)
        return (pts + n * (sign * mag)).astype(np.float32)

    if mode == "adversarial":
        # split budget: on-vertex / on-edge / on-face / near-face / very-far
        k_v = num_queries // 5
        k_e = num_queries // 5
        k_f = num_queries // 5
        k_n = num_queries // 5
        k_r = num_queries - k_v - k_e - k_f - k_n

        # on-vertex
        vi = rng.integers(0, len(vertices), size=k_v)
        on_v = vertices[vi]

        # on-edge (random barycentric on an edge)
        ei = rng.integers(0, len(indices) * 3, size=k_e)
        tid, eid = ei // 3, ei % 3
        a = indices[tid, eid]
        b = indices[tid, (eid + 1) % 3]
        t = rng.random((k_e, 1))
        on_e = (1 - t) * vertices[a] + t * vertices[b]

        # on-face (centroids)
        fi = rng.integers(0, len(indices), size=k_f)
        tri_v = vertices[indices[fi]]
        on_f = tri_v.mean(axis=1)

        # near-face
        ni = rng.integers(0, len(indices), size=k_n)
        tri_v = vertices[indices[ni]]
        pts = _sample_barycentric(tri_v, k_n, rng)
        n = _tri_normals(tri_v)
        mag = diag * (10.0 ** rng.uniform(-8.0, -3.0, size=(k_n, 1)))
        sign = (2 * rng.integers(0, 2, size=(k_n, 1)) - 1).astype(np.float64)
        near = pts + n * (sign * mag)

        # far
        far = rng.uniform(3 * pad_min, 3 * pad_max, (k_r, 3))

        out = np.concatenate([on_v, on_e, on_f, near, far], axis=0)
        rng.shuffle(out, axis=0)
        return out[:num_queries].astype(np.float32)

    raise ValueError(f"Unknown mode: {mode}")


def pytorch_mesh_winding_grads_chunked_64(
    vertices: torch.Tensor,  # Shape: [K, 3] float64
    indices: torch.Tensor,  # Shape: [N, 3] int64
    queries: torch.Tensor,  # Shape: [Q, 3] float64
    grad_output: torch.Tensor,  # Shape: [Q] float64
    chunk_size: int = 50,
) -> torch.Tensor:
    """Computes exact float64 autograd gradients for shared Mesh vertices matching CUDA."""
    v = vertices.clone().detach().requires_grad_(True)
    num_queries = queries.shape[0]
    inv_two_pi = 1.0 / (2.0 * math.pi)

    for i in range(0, num_queries, chunk_size):
        q_chunk = queries[i : i + chunk_size]
        g_chunk = grad_output[i : i + chunk_size]

        # Index shared vertices into per-triangle corners: [N, 3, 3]
        tri_v = v[indices]

        v0 = tri_v[:, 0, :][None, :, :]
        v1 = tri_v[:, 1, :][None, :, :]
        v2 = tri_v[:, 2, :][None, :, :]
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

        # sol_angle = torch.atan2(det_norm, div_norm) * inv_two_pi
        div_small = div_norm.abs() < 1e-6
        sol_angle = torch.where(div_small,
                        torch.tensor(0.5, dtype=div_norm.dtype),
                        torch.atan2(det_norm, div_norm) * inv_two_pi)

        loss_chunk = torch.sum(sol_angle * g_chunk[:, None])
        loss_chunk.backward()

    return v.grad


def pytorch_triangle_winding_grads_chunked_64(
    vertices: torch.Tensor,  # Shape: [N, 3, 3] float64
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
    cuda_grads: torch.Tensor,
    ref_grads: torch.Tensor,
    label: str,
    signal_threshold: float = 1e-5,
    abs_tol: float = 1e-5,
    rel_norm_tol: float = 1e-4,
    cosine_tol: float = 0.9999,
    masked_mean_rel_tol: float = 1e-4,
    masked_p99_rel_tol: float = 1e-2,
    per_vec_cos_p01_tol: float = 0.99,
    hist_bins: int = 20,
) -> bool:
    """Rigorous validation of CUDA gradients vs float64 reference autograd.

    Passes only if ALL of the following hold (in the active-signal regime):
      * global cosine similarity (over flattened gradients) > cosine_tol
      * global relative norm error                  < rel_norm_tol
      * masked mean relative error                  < masked_mean_rel_tol
      * masked p99 relative error                   < masked_p99_rel_tol
      * p01 of per-vector cosine similarities       > per_vec_cos_p01_tol

    In the zero-signal regime (closed mesh vertices etc.), passes only if
    the maximum absolute error is below abs_tol.
    """
    cuda_grads = cuda_grads.detach().reshape([-1, 3]).float()
    ref_grads = ref_grads.detach().reshape([-1, 3]).float()
    cuda_flat = cuda_grads.reshape(-1)
    ref_flat = ref_grads.reshape(-1)

    abs_diff = (cuda_flat - ref_flat).abs()
    max_ref_signal = ref_flat.abs().max().item()
    max_cuda_signal = cuda_flat.abs().max().item()
    max_abs_err = abs_diff.max().item()
    mean_abs_err = abs_diff.mean().item()

    norm_diff = torch.linalg.norm(cuda_flat - ref_flat).item()
    norm_ref = torch.linalg.norm(ref_flat).item()
    norm_cuda = torch.linalg.norm(cuda_flat).item()

    has_active_signal = max_ref_signal > signal_threshold

    print(f"\n=== Gradient Validation Results: {label} ===")
    print(
        f"  Shape:                       {tuple(ref_grads.shape)} ({ref_flat.numel()} scalars)"
    )
    print(f"  Reference Signal Max |g|:    {max_ref_signal:.6e}")
    print(f"  CUDA Signal Max |g|:         {max_cuda_signal:.6e}")
    print(f"  Max Absolute Error:          {max_abs_err:.6e}")
    print(f"  Mean Absolute Error:         {mean_abs_err:.6e}")

    # ------------------------------------------------------------------
    # Regime 1: zero / cancelled signal (e.g. closed mesh vertices)
    # ------------------------------------------------------------------
    if not has_active_signal:
        print(
            f"  Signal Status:               [CANCELLED / NEAR ZERO]  "
            f"(max |g_ref| <= {signal_threshold:g})"
        )
        passed = max_abs_err < abs_tol
        if passed:
            print(
                f"\033[92m  ✓ SUCCESS: {label} gradients cancel to zero within "
                f"abs tol ({abs_tol:.1e}).\033[0m"
            )
        else:
            print(
                f"\033[91m  ✗ FAILURE: {label} gradients exceed zero-signal abs tol "
                f"({max_abs_err:.6e} > {abs_tol:.1e}).\033[0m"
            )
        return passed

    # ------------------------------------------------------------------
    # Global metrics
    # ------------------------------------------------------------------
    if norm_ref > 0.0 and norm_cuda > 0.0:
        global_cosine = torch.dot(cuda_flat, ref_flat).item() / (norm_ref * norm_cuda)
    else:
        global_cosine = float("nan")
    global_rel_norm_err = norm_diff / (norm_ref + 1e-30)

    print(
        f"  Global Relative Norm Error:  {global_rel_norm_err:.6e}   (tol {rel_norm_tol:.1e})"
    )
    print(f"  Global Cosine Similarity:    {global_cosine:.8f}   (tol {cosine_tol})")

    # ------------------------------------------------------------------
    # Masked scalar relative errors
    # ------------------------------------------------------------------
    mask = ref_flat.abs() > signal_threshold
    signal_coverage = mask.float().mean().item() if mask.numel() > 0 else 0.0
    if mask.any():
        rel_err_m = abs_diff[mask] / ref_flat[mask].abs()
        mean_masked_rel = rel_err_m.mean().item()
        p50 = torch.quantile(rel_err_m, 0.50).item()
        p95 = torch.quantile(rel_err_m, 0.95).item()
        p99 = torch.quantile(rel_err_m, 0.99).item()
        p999 = torch.quantile(rel_err_m, 0.999).item()
        max_masked_rel = rel_err_m.max().item()
    else:
        rel_err_m = None
        mean_masked_rel = p50 = p95 = p99 = p999 = max_masked_rel = float("nan")

    print(
        f"  Signal coverage:             {mask.sum().item()}/{mask.numel()} "
        f"({100.0 * signal_coverage:.2f}%)"
    )
    if rel_err_m is not None:
        print(
            f"  Masked Rel Err | mean:       {mean_masked_rel:.6e}   (tol {masked_mean_rel_tol:.1e})"
        )
        print(f"  Masked Rel Err | p50:        {p50:.6e}")
        print(f"  Masked Rel Err | p95:        {p95:.6e}")
        print(
            f"  Masked Rel Err | p99:        {p99:.6e}   (tol {masked_p99_rel_tol:.1e})"
        )
        print(f"  Masked Rel Err | p99.9:      {p999:.6e}")
        print(f"  Masked Rel Err | max:        {max_masked_rel:.6e}")

    # ------------------------------------------------------------------
    # Per-vector cosine, masked by magnitude on both sides
    # ------------------------------------------------------------------
    ref_vec_norm = torch.linalg.norm(ref_grads, dim=1)
    cuda_vec_norm = torch.linalg.norm(cuda_grads, dim=1)
    vec_mask = (ref_vec_norm > signal_threshold) & (cuda_vec_norm > signal_threshold)
    if vec_mask.any():
        cos_vec = F.cosine_similarity(cuda_grads[vec_mask], ref_grads[vec_mask], dim=1)
        cos_vec_mean = cos_vec.mean().item()
        cos_vec_p01 = torch.quantile(cos_vec, 0.01).item()
        cos_vec_p05 = torch.quantile(cos_vec, 0.05).item()
        cos_vec_min = cos_vec.min().item()
    else:
        cos_vec_mean = cos_vec_p01 = cos_vec_p05 = cos_vec_min = float("nan")

    print(f"  Per-vector cosine coverage:  {vec_mask.sum().item()}/{vec_mask.numel()}")
    print(f"  Per-vector cosine | mean:    {cos_vec_mean:.6f}")
    print(f"  Per-vector cosine | p05:     {cos_vec_p05:.6f}")
    print(
        f"  Per-vector cosine | p01:     {cos_vec_p01:.6f}   (tol {per_vec_cos_p01_tol})"
    )
    print(f"  Per-vector cosine | min:     {cos_vec_min:.6f}")

    # ------------------------------------------------------------------
    # Log10 relative-error histogram
    # ------------------------------------------------------------------
    if rel_err_m is not None and rel_err_m.numel() > 0 and hist_bins > 0:
        log_rel = torch.log10(rel_err_m + 1e-30)
        lo, hi = -9.0, 1.0
        hist = torch.histc(log_rel, bins=hist_bins, min=lo, max=hi)
        hist = hist / max(1, rel_err_m.numel())
        edges = torch.linspace(lo, hi, hist_bins + 1)
        print("\n  Log10 relative error distribution (masked):")
        for i in range(hist_bins):
            bar = "█" * int(round(hist[i].item() * 60))
            print(
                f"    10^[{edges[i].item():+5.2f}, {edges[i + 1].item():+5.2f}) : "
                f"{hist[i].item():7.4f}  {bar}"
            )

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    def _ok(v, cmp, thr):
        return True if math.isnan(v) else cmp(v, thr)

    checks = {
        "global_cosine": _ok(global_cosine, lambda a, b: a > b, cosine_tol),
        "rel_norm": _ok(global_rel_norm_err, lambda a, b: a < b, rel_norm_tol),
        "masked_mean_rel": _ok(
            mean_masked_rel, lambda a, b: a < b, masked_mean_rel_tol
        ),
        "masked_p99_rel": _ok(p99, lambda a, b: a < b, masked_p99_rel_tol),
        "per_vec_cos_p01": _ok(cos_vec_p01, lambda a, b: a > b, per_vec_cos_p01_tol),
    }
    passed = all(checks.values())

    if passed:
        print(
            f"\n\033[92m  ✓ SUCCESS: {label} CUDA gradients match PyTorch float64 "
            f"autograd within all tolerances.\033[0m"
        )
    else:
        failed = [k for k, v in checks.items() if not v]
        print(
            f"\n\033[91m  ✗ FAILURE: {label} — failed checks: "
            f"{', '.join(failed)}\033[0m"
        )

    return passed


TWO_OVER_SQRT_PI = 1.1283791671


def cuda_s_regularization(t: torch.Tensor) -> torch.Tensor:
    return torch.erf(t) - TWO_OVER_SQRT_PI * t * torch.exp(-t * t)


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

    triangles = vertices[indices]
    num_triangles = len(triangles)

    queries = generate_queries(vertices, indices, query_mode, query_count)
    grad_output = np.random.normal(size=(query_count,)).astype(np.float32)

    t_tensor = torch.from_numpy(triangles).to(torch.float32).cuda()
    q_tensor = torch.from_numpy(queries).cuda()
    g_out_tensor = torch.from_numpy(grad_output).cuda()
    cuda_grads = torch.empty_like(t_tensor, dtype=torch.float32).cuda()

    winder.brute_force_gradients(
        g_out_tensor,
        t_tensor,
        q_tensor,
        cuda_grads,
        stream=torch.cuda.current_stream().cuda_stream,
    )

    print(
        f"Evaluating PyTorch float64 autograd across ALL {num_triangles} triangles..."
    )
    t_64 = t_tensor.to(torch.float64)
    q_64 = q_tensor.to(torch.float64)
    g_out_64 = g_out_tensor.to(torch.float64)

    ref_grads_64 = pytorch_triangle_winding_grads_chunked_64(t_64, q_64, g_out_64)
    ref_grads_32 = ref_grads_64.to(torch.float32)

    validate_gradients(cuda_grads, ref_grads_32, "Triangles")


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

    # ------------------------------------------------------------------
    # Compute the geometry max extent, in the same way the engine does.
    # epsilon is interpreted as a fraction of this extent.
    # ------------------------------------------------------------------
    extent = pts.max(axis=0) - pts.min(axis=0)
    geom_max_dim = float(np.max(extent))
    if geom_max_dim < 1e-20 or not np.isfinite(geom_max_dim):
        geom_max_dim = 1.0

    print(f"Geometry max extent:     {geom_max_dim:.6e}")
    print(f"epsilon (fraction):      {1.0 / inv_epsilon:.6e}")
    print(f"epsilon (world units):   {(1.0 / inv_epsilon) * geom_max_dim:.6e}")
    print(f"inv_epsilon (world):     {inv_epsilon / geom_max_dim:.6e}")

    queries = generate_queries(vertices, indices, query_mode, query_count)
    grad_output = np.random.normal(size=(query_count,)).astype(np.float32)

    p_tensor = torch.from_numpy(pts).to(torch.float32).cuda()
    n_tensor = torch.from_numpy(scaled_normals).to(torch.float32).cuda()
    q_tensor = torch.from_numpy(queries).to(torch.float32).cuda()
    g_out_tensor = torch.from_numpy(grad_output).to(torch.float32).cuda()
    cuda_grads = torch.empty([p_tensor.shape[0], 2, 3], dtype=torch.float32).cuda()

    # Brute force receives the FRACTION; the C++ layer scales it internally.
    winder.brute_force_gradients(
        g_out_tensor,
        p_tensor,
        n_tensor,
        q_tensor,
        cuda_grads,
        epsilon=1.0 / inv_epsilon,
        stream=torch.cuda.current_stream().cuda_stream,
    )

    cuda_n_grads = cuda_grads[:, 0, :]
    cuda_p_grads = cuda_grads[:, 1, :]

    print(f"Evaluating PyTorch float64 autograd across ALL {num_points} surfels...")
    p_64 = p_tensor.to(torch.float64)
    n_64 = n_tensor.to(torch.float64)
    q_64 = q_tensor.to(torch.float64)
    g_out_64 = g_out_tensor.to(torch.float64)

    # Reference works in world coordinates, so it needs the world-space
    # inv_epsilon. Same convention as the C++: epsilon_frac * geom_max_dim.
    inv_epsilon_world = inv_epsilon / geom_max_dim

    ref_p_grad_64, ref_n_grad_64 = pytorch_point_normal_grads_chunked_64(
        p_64,
        n_64,
        q_64,
        g_out_64,
        inv_epsilon=inv_epsilon_world,
        s_regularization_fn=cuda_s_regularization,
    )

    ref_p_grad_32 = ref_p_grad_64.to(torch.float32)
    ref_n_grad_32 = ref_n_grad_64.to(torch.float32)

    validate_gradients(cuda_n_grads, ref_n_grad_32, "Normal")
    validate_gradients(cuda_p_grads, ref_p_grad_32, "Position")


def test_mesh_gradients(
    vertices: np.ndarray,
    indices: np.ndarray,
    query_mode: str,
    query_count: int = 100,
):
    print("\n=== Testing Mesh Gradients against PyTorch float64 Autograd ===")

    num_vertices = len(vertices)
    num_triangles = len(indices)

    queries = generate_queries(vertices, indices, query_mode, query_count)
    grad_output = np.random.normal(size=(query_count,)).astype(np.float32)

    v_tensor = torch.from_numpy(vertices).to(torch.float32).cuda()
    idx_tensor = torch.from_numpy(indices.astype(np.uint32)).cuda()
    q_tensor = torch.from_numpy(queries).cuda()
    g_out_tensor = torch.from_numpy(grad_output).cuda()
    cuda_grads = torch.empty_like(v_tensor)

    # Call C++ nanobind overload for shared mesh vertices
    winder.brute_force_gradients(
        g_out_tensor,
        v_tensor,
        idx_tensor,
        q_tensor,
        cuda_grads,
        stream=torch.cuda.current_stream().cuda_stream,
    )

    print(
        f"Evaluating PyTorch float64 autograd for {num_vertices} vertices across {num_triangles} triangles..."
    )
    v_64 = v_tensor.to(torch.float64)
    idx_64 = torch.from_numpy(indices.astype(np.int64)).cuda()
    q_64 = q_tensor.to(torch.float64)
    g_out_64 = g_out_tensor.to(torch.float64)

    ref_grads_64 = pytorch_mesh_winding_grads_chunked_64(v_64, idx_64, q_64, g_out_64)
    ref_grads_32 = ref_grads_64.to(torch.float32)

    validate_gradients(cuda_grads, ref_grads_32, "Mesh Vertices")


def slice_mesh_in_half(vertices: np.ndarray, indices: np.ndarray, dim: int = 2):
    """Slices a mesh by keeping faces whose centroids are above the mean position along `dim`."""
    # Compute face centroids [N, 3]
    face_verts = vertices[indices]  # [N, 3, 3]
    centroids = face_verts.mean(axis=1)  # [N, 3]

    # Slice at the median/mean along specified axis
    split_plane = np.median(centroids[:, dim])
    mask = centroids[:, dim] > split_plane

    sliced_indices = indices[mask]
    return vertices, sliced_indices


def drop_half_of_the_triangles(vertices: np.ndarray, indices: np.ndarray):
    """Slices a mesh by keeping faces whose centroids are above the mean position along `dim`."""
    choice = np.random.choice(np.arange(len(indices)), len(indices) // 2, replace=False)
    return vertices, indices[choice]


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
        choices=["PointNormal", "Triangle", "Mesh", "all"],
        default="all",
        help="Choose primitive type",
    )
    parser.add_argument(
        "--query_mode",
        type=str,
        choices=["random", "grid", "surface", "adversarial", "near_surface"],
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
    print("DEBUG", indices.shape)
    vertices, indices = drop_half_of_the_triangles(
        *slice_mesh_in_half(vertices, indices)
    )
    print("DEBUG", indices.shape)

    if args.geometry_type in ["Triangle", "all"]:
        test_triangle_gradients(
            vertices,
            indices,
            args.query_mode,
            args.query_count,
        )

    if args.geometry_type in ["Mesh", "all"]:
        test_mesh_gradients(
            vertices,
            indices,
            args.query_mode,
            args.query_count,
        )

    if args.geometry_type in ["PointNormal", "all"]:
        test_point_normal_gradients(
            vertices,
            indices,
            args.query_mode,
            args.query_count,
            args.inv_epsilon,
        )
