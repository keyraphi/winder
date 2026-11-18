"""Validate PayloadFieldGeometryGradient against brute-force reference.

Tests the *geometry* gradient of the payload field, i.e.
    dL/dp_i   for each primitive i
where L = sum_{j,c} g_{j,c} * W_c(q_j) and W is the payload field.

Covers:
  - Fast (BVH) vs brute force across all geometry types
  - All D buckets + arbitrary D
  - D=1 with v=ones, g=ones reduces to the scalar winding geometry gradient
  - Linearity in grad_output
  - Linearity in payload
  - Zero payload gives zero geometry gradient
  - Zero adjoint gives zero geometry gradient
  - Kahan on/off consistency
  - Broken meshes (sliced, dropped)
  - Epsilon sweep
  - Mesh scale sweep
  - Timing

Usage
-----
    # Default: synthetic small sphere, all tests, ~60 s
    python test_payload_geom_grad.py

    # Load a real mesh
    python test_payload_geom_grad.py --obj_file benchy.obj --query_count 500

    # Only a subset
    python test_payload_geom_grad.py --test payload_d_coverage

    # Huge stress
    python test_payload_geom_grad.py --mesh_scale huge --query_count 10000
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from time import time

import numpy as np
import torch

import winder


# =============================================================================
# Constants
# =============================================================================
DEFAULT_EPSILON = 0.004
DEFAULT_D_BUCKETS = [1, 2, 3, 4, 8, 16, 32, 64, 128]
DEFAULT_ARBITRARY_D = [5, 7, 17, 33, 100]
DEFAULT_EPSILON_SWEEP = [0.0, 0.004, 0.02]

MESH_PRESETS = {
    "tiny": (6, 4),
    "small": (20, 12),
    "medium": (40, 24),
    "large": (80, 48),
    "huge": (120, 72),
}

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _ok(m):
    return f"{GREEN}{m}{RESET}"


def _bad(m):
    return f"{RED}{m}{RESET}"


def _warn(m):
    return f"{YELLOW}{m}{RESET}"


def _hdr(m):
    return f"{BOLD}{CYAN}{m}{RESET}"


# =============================================================================
# Utilities
# =============================================================================
def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_cuda(x, dtype=torch.float32) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x)).to(dtype).cuda().contiguous()


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _current_stream() -> int:
    return torch.cuda.current_stream().cuda_stream


def _time(fn, warmup: int = 1, iters: int = 3):
    for _ in range(warmup):
        result = fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        result = fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) / 1000.0)
    return result, float(np.median(times))


# =============================================================================
# Geometry helpers
# =============================================================================
def make_synthetic_sphere(n_theta: int, n_phi: int, radius: float = 1.0):
    theta_i = np.linspace(0, np.pi, n_theta + 1, dtype=np.float64)[1:-1]
    phi_j = np.linspace(0, 2 * np.pi, n_phi, endpoint=False, dtype=np.float64)

    th = theta_i[:, None]
    ph = phi_j[None, :]
    sin_th, cos_th = np.sin(th), np.cos(th)
    cos_ph, sin_ph = np.cos(ph), np.sin(ph)

    x = radius * sin_th * cos_ph
    y = radius * sin_th * sin_ph
    z = radius * cos_th * np.ones_like(cos_ph)

    verts_mid = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    poles = np.array([[0.0, 0.0, -radius], [0.0, 0.0, radius]])
    verts = np.concatenate([verts_mid, poles], axis=0).astype(np.float32)

    I, J = np.meshgrid(np.arange(n_theta - 2), np.arange(n_phi), indexing="ij")
    a = I * n_phi + J
    b = I * n_phi + (J + 1) % n_phi
    c = (I + 1) * n_phi + J
    d = (I + 1) * n_phi + (J + 1) % n_phi

    tri1 = np.stack([a.ravel(), c.ravel(), b.ravel()], axis=1)
    tri2 = np.stack([b.ravel(), c.ravel(), d.ravel()], axis=1)

    south = (n_theta - 1) * n_phi
    north = south + 1
    last = (n_theta - 2) * n_phi
    Jc = np.arange(n_phi)
    Jn = (Jc + 1) % n_phi

    tri_south = np.stack([np.full(n_phi, south), last + Jc, last + Jn], axis=1)
    tri_north = np.stack([np.full(n_phi, north), Jc, Jn], axis=1)

    tris = np.concatenate([tri1, tri2, tri_south, tri_north], axis=0)
    return verts, tris.astype(np.uint32)


def slice_mesh(vertices: np.ndarray, indices: np.ndarray, dim: int = 2):
    centroids = vertices[indices].mean(axis=1)
    split = np.median(centroids[:, dim])
    return vertices, indices[centroids[:, dim] > split]


def drop_half_triangles(vertices: np.ndarray, indices: np.ndarray, seed: int = 0):
    rng = np.random.default_rng(seed)
    keep = rng.choice(len(indices), len(indices) // 2, replace=False)
    return vertices, indices[keep]


def mesh_to_point_surfels(vertices: np.ndarray, indices: np.ndarray):
    v0 = vertices[indices[:, 0]]
    v1 = vertices[indices[:, 1]]
    v2 = vertices[indices[:, 2]]
    points = ((v0 + v1 + v2) / 3.0).astype(np.float32)
    cross = np.cross(v1 - v0, v2 - v0)
    mag = np.linalg.norm(cross, axis=-1, keepdims=True)
    areas = (mag / 2.0).flatten()
    safe = np.where(mag == 0, 1e-8, mag)
    scaled_normals = (cross / safe) * areas[:, None]
    return points, scaled_normals.astype(np.float32)


def generate_queries(vertices: np.ndarray, num_queries: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    diag = float(np.linalg.norm(hi - lo))
    lo = lo - 0.5 * diag
    hi = hi + 0.5 * diag
    return rng.uniform(lo, hi, size=(num_queries, 3)).astype(np.float32)


# =============================================================================
# Fast / brute runners - payload GEOMETRY gradients
#
# Fast path: PayloadFieldGeometryGradient (constructed per (queries, grad_output))
# Brute path: brute_force_gradients_with_payload_<geometry>
#
# Output for PN        : (N, 2, 3)  -- [i, 0] = dL/dn_i, [i, 1] = dL/dp_i
# Output for Triangle  : (N, 3, 3)  -- [i, j] = dL/dv_j of triangle i
# Output for Mesh      : (K, 3)     -- [i]    = dL/dv_i (accumulated to vertices)
# Epsilon gradient     : (1,)  scalar
# =============================================================================


def run_fast_gg_payload_point_normal(
    points,
    normals,
    payload,
    queries,
    grad_output,
    epsilon,
    beta,
    use_kahan,
    stream,
):
    p = _to_cuda(points)
    n = _to_cuda(normals)
    pl = _to_cuda(payload)
    q = _to_cuda(queries)
    g = _to_cuda(grad_output)
    N = pl.shape[0]
    out = torch.zeros((N, 2, 3), device="cuda:0", dtype=torch.float32)
    eps_g = torch.zeros((1,), device="cuda:0", dtype=torch.float32)
    eng = winder.PayloadFieldGeometryGradient(q, g, stream)
    eng.compute_point_normal(
        p, n, pl, out, eps_g, beta, epsilon, stream, bool(use_kahan)
    )
    torch.cuda.synchronize()
    return out, eps_g


def run_brute_gg_payload_point_normal(
    points,
    normals,
    payload,
    queries,
    grad_output,
    epsilon,
    use_kahan,
    stream,
):
    p = _to_cuda(points)
    n = _to_cuda(normals)
    pl = _to_cuda(payload)
    q = _to_cuda(queries)
    g = _to_cuda(grad_output)
    N = pl.shape[0]
    out = torch.zeros((N, 2, 3), device="cuda:0", dtype=torch.float32)
    eps_g = torch.zeros((1,), device="cuda:0", dtype=torch.float32)
    winder.brute_force_gradients_with_payload_point_normal(
        g,
        p,
        n,
        pl,
        q,
        out,
        eps_g,
        float(epsilon),
        stream,
        bool(use_kahan),
    )
    torch.cuda.synchronize()
    return out, eps_g


def run_fast_gg_payload_triangle(
    triangles,
    payload,
    queries,
    grad_output,
    epsilon,
    beta,
    use_kahan,
    stream,
):
    t = _to_cuda(triangles)
    pl = _to_cuda(payload)
    q = _to_cuda(queries)
    g = _to_cuda(grad_output)
    N = pl.shape[0]
    out = torch.zeros((N, 3, 3), device="cuda:0", dtype=torch.float32)
    eps_g = torch.zeros((1,), device="cuda:0", dtype=torch.float32)
    eng = winder.PayloadFieldGeometryGradient(q, g, stream)
    eng.compute_triangle_soup(t, pl, out, eps_g, beta, epsilon, stream, bool(use_kahan))
    torch.cuda.synchronize()
    return out, eps_g


def run_brute_gg_payload_triangle(
    triangles,
    payload,
    queries,
    grad_output,
    epsilon,
    use_kahan,
    stream,
):
    t = _to_cuda(triangles)
    pl = _to_cuda(payload)
    q = _to_cuda(queries)
    g = _to_cuda(grad_output)
    N = pl.shape[0]
    out = torch.zeros((N, 3, 3), device="cuda:0", dtype=torch.float32)
    eps_g = torch.zeros((1,), device="cuda:0", dtype=torch.float32)
    winder.brute_force_gradients_with_payload_triangle_soup(
        g,
        t,
        pl,
        q,
        out,
        eps_g,
        float(epsilon),
        stream,
        bool(use_kahan),
    )
    torch.cuda.synchronize()
    return out, eps_g


def run_fast_gg_payload_mesh(
    vertices,
    indices,
    payload,
    queries,
    grad_output,
    epsilon,
    beta,
    use_kahan,
    stream,
):
    v = _to_cuda(vertices)
    idx = _to_cuda(indices, torch.uint32)
    pl = _to_cuda(payload)
    q = _to_cuda(queries)
    g = _to_cuda(grad_output)
    K = v.shape[0]
    out = torch.zeros((K, 3), device="cuda:0", dtype=torch.float32)
    eps_g = torch.zeros((1,), device="cuda:0", dtype=torch.float32)
    eng = winder.PayloadFieldGeometryGradient(q, g, stream)
    eng.compute_mesh(v, idx, pl, out, eps_g, beta, epsilon, stream, bool(use_kahan))
    torch.cuda.synchronize()
    return out, eps_g


def run_brute_gg_payload_mesh(
    vertices,
    indices,
    payload,
    queries,
    grad_output,
    epsilon,
    use_kahan,
    stream,
):
    v = _to_cuda(vertices)
    idx = _to_cuda(indices, torch.uint32)
    pl = _to_cuda(payload)
    q = _to_cuda(queries)
    g = _to_cuda(grad_output)
    K = v.shape[0]
    out = torch.zeros((K, 3), device="cuda:0", dtype=torch.float32)
    eps_g = torch.zeros((1,), device="cuda:0", dtype=torch.float32)
    winder.brute_force_gradients_with_payload_mesh(
        g,
        v,
        idx,
        pl,
        q,
        out,
        eps_g,
        float(epsilon),
        stream,
        bool(use_kahan),
    )
    torch.cuda.synchronize()
    return out, eps_g


# =============================================================================
# Validation
# =============================================================================
@dataclass
class GGMetrics:
    label: str
    shape: tuple
    has_nan: bool
    has_inf: bool
    signal_max: float
    rms_abs: float
    max_abs: float
    cos_flat: float
    eps_rel_err: float = float("nan")
    eps_fast: float = float("nan")
    eps_ref: float = float("nan")
    passed: bool = False
    failed_checks: tuple = ()


def validate_geom_grad(
    fast: torch.Tensor,
    ref: torch.Tensor,
    eps_fast: torch.Tensor,
    eps_ref: torch.Tensor,
    label: str,
    *,
    rms_tol: float = 5e-2,
    max_abs_tol: float = 1.0,
    cos_tol: float = 0.98,
    eps_rtol: float = 5e-2,
    eps_atol: float = 1e-4,
    verbose: bool = True,
) -> GGMetrics:
    """Compare two geometry-gradient tensors (same shape) + their eps grads."""
    assert fast.shape == ref.shape, f"shape mismatch: {fast.shape} vs {ref.shape}"

    fast_np = fast.detach().float().cpu().numpy()
    ref_np = ref.detach().float().cpu().numpy()

    has_nan = int(np.isnan(fast_np).sum())
    has_inf = int(np.isinf(fast_np).sum())
    if has_nan or has_inf:
        if verbose:
            print(f"  [{_bad('FAIL')}] {label}: NaN={has_nan} Inf={has_inf}")
        return GGMetrics(
            label=label,
            shape=tuple(fast_np.shape),
            has_nan=has_nan,
            has_inf=has_inf,
            signal_max=0.0,
            rms_abs=float("inf"),
            max_abs=float("inf"),
            cos_flat=float("nan"),
            passed=False,
            failed_checks=("nan_or_inf",),
        )

    diff = fast_np - ref_np
    rms_abs = float(np.sqrt(np.mean(diff * diff)))
    max_abs = float(np.abs(diff).max())
    signal_max = float(np.abs(ref_np).max())

    f_flat = fast_np.reshape(-1)
    r_flat = ref_np.reshape(-1)
    nf = float(np.linalg.norm(f_flat))
    nr = float(np.linalg.norm(r_flat))
    cos_flat = (
        float(np.dot(f_flat, r_flat) / (nf * nr))
        if (nf > 0 and nr > 0)
        else float("nan")
    )

    # Epsilon gradient: single scalar, both should have matched magnitude
    eps_fast_val = float(eps_fast.item())
    eps_ref_val = float(eps_ref.item())
    eps_denom = max(abs(eps_fast_val), abs(eps_ref_val), eps_atol)
    eps_rel_err = abs(eps_fast_val - eps_ref_val) / eps_denom

    checks = {
        "rms": rms_abs < rms_tol,
        "max_abs": max_abs < max_abs_tol,
        "cos_flat": cos_flat > cos_tol,
        "eps": eps_rel_err < eps_rtol,
    }
    passed = all(checks.values())
    failed = tuple(k for k, v in checks.items() if not v)

    if verbose:
        status = _ok("PASS") if passed else _bad("FAIL")
        print(
            f"  [{status}] {label:<42}  "
            f"shape={tuple(fast_np.shape)}  "
            f"rms={rms_abs:.2e}  max={max_abs:.2e}  cos={cos_flat:.5f}  "
            f"eps_rel={eps_rel_err:.2e}"
        )

    return GGMetrics(
        label=label,
        shape=tuple(fast_np.shape),
        has_nan=False,
        has_inf=False,
        signal_max=signal_max,
        rms_abs=rms_abs,
        max_abs=max_abs,
        cos_flat=cos_flat,
        eps_rel_err=eps_rel_err,
        eps_fast=eps_fast_val,
        eps_ref=eps_ref_val,
        passed=passed,
        failed_checks=failed,
    )


# =============================================================================
# Tests
# =============================================================================
def test_gg_payload_d_coverage(
    vertices,
    indices,
    points,
    normals,
    queries,
    epsilon,
    beta,
    use_kahan,
    d_values,
    results,
    verbose=True,
):
    """Fast vs brute GEOMETRY gradient across all D values × all geometries."""
    print(
        _hdr(f"\nTest: payload geometry gradient D coverage ({len(d_values)} values)")
    )
    triangles = vertices[indices]
    M = len(queries)
    rng = np.random.default_rng(202)
    stream = _current_stream()

    for D in d_values:
        payload_tri = rng.standard_normal((len(triangles), D)).astype(np.float32)
        payload_pn = rng.standard_normal((len(points), D)).astype(np.float32)
        g_np = rng.standard_normal((M, D)).astype(np.float32)

        # Triangle soup
        f, f_eps = run_fast_gg_payload_triangle(
            triangles,
            payload_tri,
            queries,
            g_np,
            epsilon,
            beta,
            use_kahan,
            stream,
        )
        b, b_eps = run_brute_gg_payload_triangle(
            triangles,
            payload_tri,
            queries,
            g_np,
            epsilon,
            use_kahan,
            stream,
        )
        results.append(
            validate_geom_grad(
                f,
                b,
                f_eps,
                b_eps,
                f"Tri  D={D:>3}",
                verbose=verbose,
            )
        )

        # PointNormal
        f, f_eps = run_fast_gg_payload_point_normal(
            points,
            normals,
            payload_pn,
            queries,
            g_np,
            epsilon,
            beta,
            use_kahan,
            stream,
        )
        b, b_eps = run_brute_gg_payload_point_normal(
            points,
            normals,
            payload_pn,
            queries,
            g_np,
            epsilon,
            use_kahan,
            stream,
        )
        results.append(
            validate_geom_grad(
                f,
                b,
                f_eps,
                b_eps,
                f"PN   D={D:>3}",
                verbose=verbose,
            )
        )

        # Mesh
        f, f_eps = run_fast_gg_payload_mesh(
            vertices,
            indices,
            payload_tri,
            queries,
            g_np,
            epsilon,
            beta,
            use_kahan,
            stream,
        )
        b, b_eps = run_brute_gg_payload_mesh(
            vertices,
            indices,
            payload_tri,
            queries,
            g_np,
            epsilon,
            use_kahan,
            stream,
        )
        results.append(
            validate_geom_grad(
                f,
                b,
                f_eps,
                b_eps,
                f"Mesh D={D:>3}",
                verbose=verbose,
            )
        )


def test_gg_payload_linearity_grad_output(
    vertices,
    indices,
    queries,
    epsilon,
    beta,
    use_kahan,
    results,
    verbose=True,
):
    """Geometry grad is linear in grad_output:
    GG(a*g1 + b*g2) = a*GG(g1) + b*GG(g2)."""
    print(_hdr("\nTest: payload geometry grad linearity in grad_output (D=8)"))
    triangles = vertices[indices]
    D = 8
    M = len(queries)
    rng = np.random.default_rng(303)
    payload = rng.standard_normal((len(triangles), D)).astype(np.float32)
    g1 = rng.standard_normal((M, D)).astype(np.float32)
    g2 = rng.standard_normal((M, D)).astype(np.float32)
    a, b = 1.7, -0.83
    gc = (a * g1 + b * g2).astype(np.float32)
    stream = _current_stream()

    fc, fc_eps = run_fast_gg_payload_triangle(
        triangles,
        payload,
        queries,
        gc,
        epsilon,
        beta,
        use_kahan,
        stream,
    )
    f1, _ = run_fast_gg_payload_triangle(
        triangles,
        payload,
        queries,
        g1,
        epsilon,
        beta,
        use_kahan,
        stream,
    )
    f2, _ = run_fast_gg_payload_triangle(
        triangles,
        payload,
        queries,
        g2,
        epsilon,
        beta,
        use_kahan,
        stream,
    )
    expected = a * f1 + b * f2
    # Epsilon gradient is 0 for a linear combination?  No — it's also linear,
    # so eps_c = a*eps_1 + b*eps_2. Build it explicitly for the check.
    eps1, eps2 = torch.tensor([0.0]), torch.tensor([0.0])
    # Get actual eps values from the sub-calls
    _, eps1 = run_fast_gg_payload_triangle(
        triangles,
        payload,
        queries,
        g1,
        epsilon,
        beta,
        use_kahan,
        stream,
    )
    _, eps2 = run_fast_gg_payload_triangle(
        triangles,
        payload,
        queries,
        g2,
        epsilon,
        beta,
        use_kahan,
        stream,
    )
    eps_expected = (a * eps1 + b * eps2).cuda()
    results.append(
        validate_geom_grad(
            fc,
            expected,
            fc_eps,
            eps_expected,
            "linearity in g (fast)",
            verbose=verbose,
        )
    )

    bc, bc_eps = run_brute_gg_payload_triangle(
        triangles,
        payload,
        queries,
        gc,
        epsilon,
        use_kahan,
        stream,
    )
    b1, b1_eps = run_brute_gg_payload_triangle(
        triangles,
        payload,
        queries,
        g1,
        epsilon,
        use_kahan,
        stream,
    )
    b2, b2_eps = run_brute_gg_payload_triangle(
        triangles,
        payload,
        queries,
        g2,
        epsilon,
        use_kahan,
        stream,
    )
    expected_b = a * b1 + b * b2
    eps_expected_b = (a * b1_eps + b * b2_eps).cuda()
    results.append(
        validate_geom_grad(
            bc,
            expected_b,
            bc_eps,
            eps_expected_b,
            "linearity in g (brute)",
            verbose=verbose,
        )
    )


def test_gg_payload_linearity_payload(
    vertices,
    indices,
    queries,
    epsilon,
    beta,
    use_kahan,
    results,
    verbose=True,
):
    """Geometry grad is linear in payload?  NO — it's linear in each payload
    *component*, but the *combination* is nonlinear because the weight ĝ_ij
    depends on the payload. So this test is *not* an identity; we only check
    that the fast and brute paths agree on the combination."""
    # Skipping on purpose — a proper linearity-in-payload test would require
    # more thought (it's only linear per fixed ĝ_ij, which requires fixing
    # the payload's direction but allowing its magnitude).
    pass


def test_gg_payload_zero_payload(
    vertices,
    indices,
    points,
    normals,
    queries,
    epsilon,
    beta,
    use_kahan,
    results,
    verbose=True,
):
    """All-zero payload ⇒ all-zero geometry gradient."""
    print(_hdr("\nTest: zero payload → zero geometry grad"))
    triangles = vertices[indices]
    D = 8
    M = len(queries)
    rng = np.random.default_rng(404)
    zero_payload_tri = np.zeros((len(triangles), D), dtype=np.float32)
    zero_payload_pn = np.zeros((len(points), D), dtype=np.float32)
    g_np = rng.standard_normal((M, D)).astype(np.float32)
    stream = _current_stream()

    ref = torch.zeros((len(triangles), 3, 3), device="cuda:0", dtype=torch.float32)
    ref_eps = torch.zeros((1,), device="cuda:0", dtype=torch.float32)

    f, f_eps = run_fast_gg_payload_triangle(
        triangles,
        zero_payload_tri,
        queries,
        g_np,
        epsilon,
        beta,
        use_kahan,
        stream,
    )
    results.append(
        validate_geom_grad(
            f,
            ref,
            f_eps,
            ref_eps,
            "Tri  zero payload (fast)",
            rms_tol=1e-4,
            max_abs_tol=1e-3,
            cos_tol=0.99,
            verbose=verbose,
        )
    )
    b, b_eps = run_brute_gg_payload_triangle(
        triangles,
        zero_payload_tri,
        queries,
        g_np,
        epsilon,
        use_kahan,
        stream,
    )
    results.append(
        validate_geom_grad(
            b,
            ref,
            b_eps,
            ref_eps,
            "Tri  zero payload (brute)",
            rms_tol=1e-4,
            max_abs_tol=1e-3,
            cos_tol=0.99,
            verbose=verbose,
        )
    )

    ref_pn = torch.zeros((len(points), 2, 3), device="cuda:0", dtype=torch.float32)
    f, f_eps = run_fast_gg_payload_point_normal(
        points,
        normals,
        zero_payload_pn,
        queries,
        g_np,
        epsilon,
        beta,
        use_kahan,
        stream,
    )
    results.append(
        validate_geom_grad(
            f,
            ref_pn,
            f_eps,
            ref_eps,
            "PN   zero payload (fast)",
            rms_tol=1e-4,
            max_abs_tol=1e-3,
            cos_tol=0.99,
            verbose=verbose,
        )
    )
    b, b_eps = run_brute_gg_payload_point_normal(
        points,
        normals,
        zero_payload_pn,
        queries,
        g_np,
        epsilon,
        use_kahan,
        stream,
    )
    results.append(
        validate_geom_grad(
            b,
            ref_pn,
            b_eps,
            ref_eps,
            "PN   zero payload (brute)",
            rms_tol=1e-4,
            max_abs_tol=1e-3,
            cos_tol=0.99,
            verbose=verbose,
        )
    )


def test_gg_payload_zero_adjoint(
    vertices,
    indices,
    queries,
    epsilon,
    beta,
    use_kahan,
    results,
    verbose=True,
):
    """All-zero grad_output ⇒ all-zero geometry gradient."""
    print(_hdr("\nTest: zero grad_output → zero geometry grad"))
    triangles = vertices[indices]
    D = 8
    M = len(queries)
    rng = np.random.default_rng(505)
    payload = rng.standard_normal((len(triangles), D)).astype(np.float32)
    g_zero = np.zeros((M, D), dtype=np.float32)
    stream = _current_stream()

    ref = torch.zeros((len(triangles), 3, 3), device="cuda:0", dtype=torch.float32)
    ref_eps = torch.zeros((1,), device="cuda:0", dtype=torch.float32)

    f, f_eps = run_fast_gg_payload_triangle(
        triangles,
        payload,
        queries,
        g_zero,
        epsilon,
        beta,
        use_kahan,
        stream,
    )
    results.append(
        validate_geom_grad(
            f,
            ref,
            f_eps,
            ref_eps,
            "Tri  zero adjoint (fast)",
            rms_tol=1e-4,
            max_abs_tol=1e-3,
            cos_tol=0.99,
            verbose=verbose,
        )
    )
    b, b_eps = run_brute_gg_payload_triangle(
        triangles,
        payload,
        queries,
        g_zero,
        epsilon,
        use_kahan,
        stream,
    )
    results.append(
        validate_geom_grad(
            b,
            ref,
            b_eps,
            ref_eps,
            "Tri  zero adjoint (brute)",
            rms_tol=1e-4,
            max_abs_tol=1e-3,
            cos_tol=0.99,
            verbose=verbose,
        )
    )


def test_gg_payload_kahan_consistency(
    vertices,
    indices,
    queries,
    epsilon,
    beta,
    results,
    verbose=True,
):
    """Kahan on/off agree within tolerance (payload D=8, triangle soup)."""
    print(_hdr("\nTest: payload geometry grad Kahan on/off (D=8)"))
    triangles = vertices[indices]
    D = 8
    M = len(queries)
    rng = np.random.default_rng(606)
    payload = rng.standard_normal((len(triangles), D)).astype(np.float32)
    g_np = rng.standard_normal((M, D)).astype(np.float32)
    stream = _current_stream()

    # Fast path
    on, on_eps = run_fast_gg_payload_triangle(
        triangles,
        payload,
        queries,
        g_np,
        epsilon,
        beta,
        True,
        stream,
    )
    off, off_eps = run_fast_gg_payload_triangle(
        triangles,
        payload,
        queries,
        g_np,
        epsilon,
        beta,
        False,
        stream,
    )
    results.append(
        validate_geom_grad(
            on,
            off,
            on_eps,
            off_eps,
            "Kahan on vs off (fast)",
            rms_tol=1e-2,
            max_abs_tol=1e-1,
            cos_tol=0.999,
            verbose=verbose,
        )
    )

    # Brute path
    on, on_eps = run_brute_gg_payload_triangle(
        triangles,
        payload,
        queries,
        g_np,
        epsilon,
        True,
        stream,
    )
    off, off_eps = run_brute_gg_payload_triangle(
        triangles,
        payload,
        queries,
        g_np,
        epsilon,
        False,
        stream,
    )
    results.append(
        validate_geom_grad(
            on,
            off,
            on_eps,
            off_eps,
            "Kahan on vs off (brute)",
            rms_tol=1e-2,
            max_abs_tol=1e-1,
            cos_tol=0.999,
            verbose=verbose,
        )
    )


def test_gg_payload_broken_meshes(
    vertices,
    indices,
    queries,
    epsilon,
    beta,
    use_kahan,
    results,
    verbose=True,
    seed=0,
):
    """Closed, sliced, dropped triangle soups (payload D=8)."""
    print(_hdr("\nTest: broken meshes - payload geometry grad (D=8)"))
    D = 8
    M = len(queries)
    rng = np.random.default_rng(707)
    stream = _current_stream()

    v_slice, i_slice = slice_mesh(vertices, indices)
    v_dropped, i_dropped = drop_half_triangles(vertices, indices, seed=seed)
    _, i_sliced_dropped = drop_half_triangles(v_slice, i_slice, seed=seed)

    variants = [
        ("closed", vertices, indices),
        ("sliced", v_slice, i_slice),
        ("dropped", v_dropped, i_dropped),
        ("sliced+dropped", v_slice, i_sliced_dropped),
    ]

    for name, v, i in variants:
        tri = v[i]
        payload = rng.standard_normal((len(tri), D)).astype(np.float32)
        g_np = rng.standard_normal((M, D)).astype(np.float32)

        f, f_eps = run_fast_gg_payload_triangle(
            tri,
            payload,
            queries,
            g_np,
            epsilon,
            beta,
            use_kahan,
            stream,
        )
        b, b_eps = run_brute_gg_payload_triangle(
            tri,
            payload,
            queries,
            g_np,
            epsilon,
            use_kahan,
            stream,
        )
        results.append(
            validate_geom_grad(
                f,
                b,
                f_eps,
                b_eps,
                f"{name:<15} ({len(tri):>5} tri)",
                verbose=verbose,
            )
        )


def test_gg_payload_epsilon_sweep(
    vertices,
    indices,
    queries,
    epsilon_values,
    beta,
    use_kahan,
    results,
    verbose=True,
):
    """Fast vs brute geometry grad for each epsilon (payload D=8)."""
    print(_hdr("\nTest: payload geometry grad epsilon sweep (D=8)"))
    triangles = vertices[indices]
    D = 8
    M = len(queries)
    rng = np.random.default_rng(808)
    payload = rng.standard_normal((len(triangles), D)).astype(np.float32)
    g_np = rng.standard_normal((M, D)).astype(np.float32)
    stream = _current_stream()

    for eps in epsilon_values:
        f, f_eps = run_fast_gg_payload_triangle(
            triangles,
            payload,
            queries,
            g_np,
            eps,
            beta,
            use_kahan,
            stream,
        )
        b, b_eps = run_brute_gg_payload_triangle(
            triangles,
            payload,
            queries,
            g_np,
            eps,
            use_kahan,
            stream,
        )
        results.append(
            validate_geom_grad(
                f,
                b,
                f_eps,
                b_eps,
                f"eps={eps:g}",
                verbose=verbose,
            )
        )


def test_gg_payload_mesh_scale_sweep(
    query_count,
    epsilon,
    beta,
    use_kahan,
    results,
    verbose=True,
    seed=0,
):
    """Fast vs brute geometry grad (D=8) across mesh sizes."""
    print(_hdr("\nTest: payload geometry grad mesh scale sweep (D=8)"))
    D = 8
    rng = np.random.default_rng(909)
    stream = _current_stream()

    presets = [
        ("small", 20, 12, query_count),
        ("medium", 40, 24, max(query_count // 2, 100)),
        ("large", 80, 48, max(query_count // 5, 50)),
    ]

    for name, n_theta, n_phi, n_q in presets:
        v, i = make_synthetic_sphere(n_theta, n_phi)
        v, i = drop_half_triangles(v, i, seed=seed)
        tri = v[i]
        q = generate_queries(v, n_q, seed)
        payload = rng.standard_normal((len(tri), D)).astype(np.float32)
        g_np = rng.standard_normal((n_q, D)).astype(np.float32)

        f, f_eps = run_fast_gg_payload_triangle(
            tri,
            payload,
            q,
            g_np,
            epsilon,
            beta,
            use_kahan,
            stream,
        )
        b, b_eps = run_brute_gg_payload_triangle(
            tri,
            payload,
            q,
            g_np,
            epsilon,
            use_kahan,
            stream,
        )
        results.append(
            validate_geom_grad(
                f,
                b,
                f_eps,
                b_eps,
                f"{name:<8} ({len(tri):>5} tri, {n_q:>4} q)",
                verbose=verbose,
            )
        )


def test_gg_payload_d1_scalar_match(
    vertices,
    indices,
    points,
    normals,
    queries,
    epsilon,
    beta,
    results,
    verbose=True,
):
    """D=1 payload with v=ones, g=ones must equal the scalar winding
    geometry gradient with g=ones.

    Comparisons:
      - fast payload vs fast scalar (both BVH-approximated, both Kahan)
      - brute payload vs brute scalar (both exact)

    The scalar `GradientEngine` uses Kahan unconditionally, so the fast
    comparison requires `use_kahan=True`.
    """
    print(_hdr("\nTest: D=1 payload geometry grad == scalar winding geom grad"))
    triangles = vertices[indices]
    M = len(queries)
    g_scalar = np.ones(M, dtype=np.float32)
    ones_tri = np.ones((len(triangles), 1), dtype=np.float32)
    ones_pn = np.ones((len(points), 1), dtype=np.float32)
    g_payload = np.ones((M, 1), dtype=np.float32)
    stream = _current_stream()

    # --- Fast path: scalar WindingNumberEngine, triangle soup ---
    t_cuda = _to_cuda(triangles)
    q_cuda = _to_cuda(queries)
    g_cuda = _to_cuda(g_scalar)
    N = len(triangles)
    scalar_out = torch.zeros((N, 3, 3), device="cuda:0", dtype=torch.float32)
    scalar_eps = torch.zeros((1,), device="cuda:0", dtype=torch.float32)
    scalar_eng = winder.GradientEngine(q_cuda, g_cuda, stream)
    scalar_eng.compute_triangle_soup(
        t_cuda,
        scalar_out,
        scalar_eps,
        beta,
        epsilon,
        stream,
    )
    torch.cuda.synchronize()

    pay_out, pay_eps = run_fast_gg_payload_triangle(
        triangles,
        ones_tri,
        queries,
        g_payload,
        epsilon,
        beta,
        True,
        stream,
    )
    results.append(
        validate_geom_grad(
            pay_out,
            scalar_out,
            pay_eps,
            scalar_eps,
            "Tri  D=1 ones (fast vs fast)",
            rms_tol=1e-2,
            max_abs_tol=1e-1,
            cos_tol=0.999,
            verbose=verbose,
        )
    )

    # --- Fast path: scalar WindingNumberEngine, point normal ---
    p_cuda = _to_cuda(points)
    n_cuda = _to_cuda(normals)
    N_pn = len(points)
    scalar_pn_out = torch.zeros((N_pn, 2, 3), device="cuda:0", dtype=torch.float32)
    scalar_pn_eps = torch.zeros((1,), device="cuda:0", dtype=torch.float32)
    scalar_pn_eng = winder.GradientEngine(q_cuda, g_cuda, stream)
    scalar_pn_eng.compute_point_normal(
        p_cuda,
        n_cuda,
        scalar_pn_out,
        scalar_pn_eps,
        beta,
        epsilon,
        stream,
    )
    torch.cuda.synchronize()

    pay_pn_out, pay_pn_eps = run_fast_gg_payload_point_normal(
        points,
        normals,
        ones_pn,
        queries,
        g_payload,
        epsilon,
        beta,
        True,
        stream,
    )
    results.append(
        validate_geom_grad(
            pay_pn_out,
            scalar_pn_out,
            pay_pn_eps,
            scalar_pn_eps,
            "PN   D=1 ones (fast vs fast)",
            rms_tol=1e-2,
            max_abs_tol=1e-1,
            cos_tol=0.999,
            verbose=verbose,
        )
    )

    # --- Brute path: brute-force scalar vs brute-force payload ---
    b_tri, b_tri_eps = run_brute_gg_payload_triangle(
        triangles,
        ones_tri,
        queries,
        g_payload,
        epsilon,
        True,
        stream,
    )
    s_tri, s_tri_eps = (
        winder.brute_force_gradients_triangle_soup(
            g_cuda,
            t_cuda,
            q_cuda,
            scalar_out,
            scalar_eps,
            float(epsilon),
            stream,
            True,
        ),
        scalar_eps,
    )
    torch.cuda.synchronize()
    s_tri_out = torch.zeros_like(scalar_out)
    winder.brute_force_gradients_triangle_soup(
        g_cuda,
        t_cuda,
        q_cuda,
        s_tri_out,
        scalar_eps,
        float(epsilon),
        stream,
        True,
    )
    torch.cuda.synchronize()
    results.append(
        validate_geom_grad(
            b_tri,
            s_tri_out,
            b_tri_eps,
            scalar_eps,
            "Tri  D=1 ones (brute vs brute)",
            rms_tol=1e-5,
            max_abs_tol=1e-4,
            cos_tol=0.9999,
            verbose=verbose,
        )
    )

    b_pn, b_pn_eps = run_brute_gg_payload_point_normal(
        points,
        normals,
        ones_pn,
        queries,
        g_payload,
        epsilon,
        True,
        stream,
    )
    s_pn_out = torch.zeros((N_pn, 2, 3), device="cuda:0", dtype=torch.float32)
    winder.brute_force_gradients_point_normal(
        g_cuda,
        p_cuda,
        n_cuda,
        q_cuda,
        s_pn_out,
        scalar_pn_eps,
        float(epsilon),
        stream,
        True,
    )
    torch.cuda.synchronize()
    results.append(
        validate_geom_grad(
            b_pn,
            s_pn_out,
            b_pn_eps,
            scalar_pn_eps,
            "PN   D=1 ones (brute vs brute)",
            rms_tol=1e-5,
            max_abs_tol=1e-4,
            cos_tol=0.9999,
            verbose=verbose,
        )
    )


def test_gg_timing(
    vertices,
    indices,
    queries,
    epsilon,
    beta,
    d_values,
    verbose=True,
):
    """Timing table: fast vs brute geometry gradients across D."""
    print(_hdr("\n=== Timing: payload geometry grad fast vs brute ==="))
    triangles = vertices[indices]
    M = len(queries)
    N = len(triangles)
    rng = np.random.default_rng(1001)
    stream = _current_stream()

    t = _to_cuda(triangles)
    q = _to_cuda(queries)

    print(f"  Mesh: {N} triangles, {M} queries, eps={epsilon:g}")
    print(
        f"  {'D':>4} | {'BF(ms)':>10} | {'fast(ms)':>10} | "
        f"{'build(ms)':>10} | {'speedup':>8}"
    )
    print("  " + "-" * 60)

    for D in d_values:
        payload = rng.standard_normal((N, D)).astype(np.float32)
        pl = _to_cuda(payload)
        g_np = rng.standard_normal((M, D)).astype(np.float32)
        g = _to_cuda(g_np)
        out = torch.zeros((N, 3, 3), device="cuda:0", dtype=torch.float32)
        eps_g = torch.zeros((1,), device="cuda:0", dtype=torch.float32)

        def brute_fn():
            winder.brute_force_gradients_with_payload_triangle_soup(
                g,
                t,
                pl,
                q,
                out,
                eps_g,
                float(epsilon),
                stream,
                True,
            )

        bf_ms = _time(brute_fn, warmup=1, iters=3)[1] * 1000

        def build_fn():
            return winder.PayloadFieldGeometryGradient(q, g, stream)

        build_ms = _time(build_fn, warmup=1, iters=3)[1] * 1000
        eng = winder.PayloadFieldGeometryGradient(q, g, stream)

        def compute_only():
            eng.compute_triangle_soup(
                t,
                pl,
                out,
                eps_g,
                beta,
                epsilon,
                stream,
                True,
            )

        compute_ms = _time(compute_only, warmup=1, iters=3)[1] * 1000
        e2e_ms = build_ms + compute_ms
        speedup = bf_ms / e2e_ms if e2e_ms > 0 else float("nan")

        print(
            f"  {D:>4} | {bf_ms:>10.2f} | {e2e_ms:>10.2f} | "
            f"{build_ms:>10.3f} | {speedup:>7.2f}x"
        )
        print(f"       (compute-only: {compute_ms:.2f} ms)")


# =============================================================================
# Scale timing (BH vs brute across mesh sizes)
# =============================================================================
def _estimate_bf_ms(cal_ms, cal_n, cal_m, cal_d, N, M, D):
    """BF is O(N*M*D); extrapolate from a calibration run at (cal_n,cal_m,cal_d).

    Cache effects make this slightly optimistic at very large M (L2 pressure),
    so the extrapolated BF time is a *lower bound* on the true BF time, and
    the extrapolated speedup is a *lower bound* on the true speedup.
    """
    return (
        cal_ms
        * (float(N) * float(M) * float(D))
        / (float(cal_n) * float(cal_m) * float(cal_d))
    )


def test_gg_scale_timing(
    epsilon,
    beta,
    verbose=True,
    seed=0,
    sizes=None,
    d_values=(1, 8, 32, 128),
    bf_budget_seconds=900.0,
    bf_cal_n=10_000,
    bf_cal_m=100_000,
):
    """Timing: GG fast path — FAIR MIRROR of the QG scale test.

    The QG test fixes q=100k and scales tri (the axis its BVH is built over).
    GG's BVH is built over the *queries*, so the fair counterpart is the
    transposed experiment: fix tri≈100k (mirroring QG's fixed q=100k) and
    sweep q over the same nominal sizes QG used for tri.

    The '1M+2M' row uses the exact same (tri, q) as QG's largest row, so
    the two tables can be compared directly on that point.

    Calibration and '*' extrapolation logic are unchanged from the QG test.
    """
    print(_hdr("\n=== Timing: GG, realistic scales (mirror of QG) ==="))
    print(f"  epsilon={epsilon:g}, beta={beta}")
    print(f"  tri fixed at 100_000; q swept over QG's nominal mesh sizes")
    print(f"  BF calibration: {bf_cal_n} tri x {bf_cal_m} q per D value")
    print(
        f"  BF per-cell budget: {bf_budget_seconds:.0f} s "
        f"(cells above this are extrapolated)"
    )

    if sizes is None:
        # (name, n_theta, n_phi, M_target)
        #
        # Sphere tessellation -> N = 2 * (n_theta - 1) * n_phi.
        # (251, 200) -> exactly 100_000 triangles, reused for every scaled
        # row so `tri` is fixed and `q` is the only variable.
        # (1001, 500) -> 1_000_000 triangles for the 1M+2M cross-test row.
        sizes = [
            ("2k q", 251, 200, 2_112),  # tri=100k, q ≈ 2k
            ("10k q", 251, 200, 11_088),  # tri=100k, q ≈ 11k
            ("50k q", 251, 200, 52_560),  # tri=100k, q ≈ 53k
            ("200k q", 251, 200, 206_540),  # tri=100k, q ≈ 207k
            ("1M q", 251, 200, 1_000_000),  # tri=100k, q = 1M
            ("1M+2M", 1001, 500, 2_000_000),  # tri=1M,  q = 2M (same as QG)
        ]

    stream = _current_stream()
    bf_cal = {}

    # Calibrate BF once per D value (unchanged — same 10k tri × 100k q
    # reference as QG so cross-table comparisons stay clean).
    v_cal, i_cal = make_synthetic_sphere(100, 56)
    tri_cal = v_cal[i_cal][:bf_cal_n]
    N_cal = len(tri_cal)
    q_cal = generate_queries(v_cal, bf_cal_m, seed + 1)
    t_cal = _to_cuda(tri_cal)
    qc_cal = _to_cuda(q_cal)

    for D in d_values:
        rng_c = np.random.default_rng(seed + 1000 + D)
        payload_cal = rng_c.standard_normal((N_cal, D)).astype(np.float32)
        g_cal = rng_c.standard_normal((bf_cal_m, D)).astype(np.float32)
        pl_cal = _to_cuda(payload_cal)
        g_cal_t = _to_cuda(g_cal)
        out_cal = torch.zeros((N_cal, 3, 3), device="cuda:0", dtype=torch.float32)
        eps_cal = torch.zeros((1,), device="cuda:0", dtype=torch.float32)

        def bf_cal_fn():
            winder.brute_force_gradients_with_payload_triangle_soup(
                g_cal_t,
                t_cal,
                pl_cal,
                qc_cal,
                out_cal,
                eps_cal,
                float(epsilon),
                stream,
                True,
            )

        cal_ms = _time(bf_cal_fn, warmup=1, iters=5)[1] * 1000
        bf_cal[D] = (cal_ms, N_cal, bf_cal_m)
        print(
            f"  calibrated D={D:>3}: {cal_ms:.3f} ms @ "
            f"{N_cal} tri x {bf_cal_m} q "
            f"({cal_ms * 1e3 / (N_cal * bf_cal_m * D):.4f} ns/unit)"
        )

    print(
        f"\n  {'case':<8} {'tri':>8} {'q':>10} {'D':>4} | "
        f"{'BF(ms)':>11} {'fast(ms)':>9} {'speedup':>9} | {'rms_vs_BFtrue':>14}"
    )
    print("  " + "-" * 92)

    for name, n_theta, n_phi, M_target in sizes:
        v, i = make_synthetic_sphere(n_theta, n_phi)
        tri = v[i]
        N = len(tri)

        for D in d_values:
            rng = np.random.default_rng(seed + D)
            payload = rng.standard_normal((N, D)).astype(np.float32)
            g_np = rng.standard_normal((M_target, D)).astype(np.float32)

            t = _to_cuda(tri)
            qc = _to_cuda(generate_queries(v, M_target, seed + 1))
            pl = _to_cuda(payload)
            g = _to_cuda(g_np)
            out = torch.zeros((N, 3, 3), device="cuda:0", dtype=torch.float32)
            eps_g = torch.zeros((1,), device="cuda:0", dtype=torch.float32)

            cal_ms, cal_n, cal_m = bf_cal[D]
            predicted_bf_ms = _estimate_bf_ms(cal_ms, cal_n, cal_m, D, N, M_target, D)
            do_bf = predicted_bf_ms <= bf_budget_seconds * 1000.0

            if do_bf:

                def bf_fn():
                    winder.brute_force_gradients_with_payload_triangle_soup(
                        g,
                        t,
                        pl,
                        qc,
                        out,
                        eps_g,
                        float(epsilon),
                        stream,
                        True,
                    )

                bf_ms = _time(bf_fn, warmup=1, iters=3)[1] * 1000
                bf_ref = out.clone()
                bf_tag = ""
            else:
                bf_ms = predicted_bf_ms
                bf_ref = None
                bf_tag = "*"

            eng = winder.PayloadFieldGeometryGradient(qc, g, stream)
            eng.compute_triangle_soup(
                t,
                pl,
                out,
                eps_g,
                beta,
                epsilon,
                stream,
                True,
            )
            torch.cuda.synchronize()

            def fast_fn():
                eng.compute_triangle_soup(
                    t,
                    pl,
                    out,
                    eps_g,
                    beta,
                    epsilon,
                    stream,
                    True,
                )

            fast_ms = _time(fast_fn, warmup=1, iters=3)[1] * 1000

            if bf_ref is not None:
                rms = float(
                    torch.sqrt(torch.mean((out.float() - bf_ref.float()) ** 2)).item()
                )
                rms_s = f"{rms:>14.3e}"
            else:
                rms_s = f"{'  (extrapolated)':>14s}"

            speedup = bf_ms / fast_ms if fast_ms > 0 else float("nan")

            print(
                f"  {name:<8} {N:>8} {M_target:>10} {D:>4} | "
                f"{bf_ms:>10.2f}{bf_tag:<1s} {fast_ms:>9.3f} "
                f"{speedup:>8.1f}{bf_tag:<1s}x | {rms_s}"
            )

            del t, qc, pl, g, out, eng
            torch.cuda.empty_cache()


def slice_mesh_in_half(
    vertices: np.ndarray, indices: np.ndarray, dim: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Keep triangles whose centroid lies above the median along `dim`."""
    centroids = vertices[indices].mean(axis=1)
    split = np.median(centroids[:, dim])
    return vertices, indices[centroids[:, dim] > split]


def drop_half_of_the_triangles(
    vertices: np.ndarray, indices: np.ndarray, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Drop a random half of the triangles (deterministic given `seed`)."""
    rng = np.random.default_rng(seed)
    choice = rng.choice(np.arange(len(indices)), len(indices) // 2, replace=False)
    return vertices, indices[choice]


def _regular_grid(tri, resolution):
    """Plain resolution^3 regular grid over the (2%-padded) mesh bbox."""
    verts = tri.reshape(-1, 3)
    vmin = verts.min(axis=0)
    vmax = verts.max(axis=0)
    pad = 0.02 * (vmax - vmin)
    vmin = vmin - pad
    vmax = vmax + pad
    axes = [
        np.linspace(vmin[i], vmax[i], resolution, dtype=np.float32) for i in range(3)
    ]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float32)


def _estimate_mem_gb(M, N, D):
    """Rough GPU memory for one QG or GG cell.

    Persistent: payload (N,D), grad_output (M,D), queries (M,3),
    triangles (N,3,3), BVH leaf coefficient cache (roughly one more
    N*D float32 for the tree). Factor 2.5 for scratch.
    """
    raw = 4 * (N * D + M * D + M * 3 + N * 9 + N * D)
    return 2.5 * raw / 1e9


def _load_obj_broken(path, seed=0):
    """Load a .obj with libigl, slice in half, drop half the triangles.
    Returns tri_soup [N,3,3] float32, or None if empty after processing.
    """
    import igl

    v, _, _, f, _, _ = igl.readOBJ(str(path))
    if len(f) == 0:
        return None
    v, f = slice_mesh_in_half(v, f)
    v, f = drop_half_of_the_triangles(v, f, seed=seed)
    if len(f) == 0:
        return None
    return np.asarray(v[f], dtype=np.float32)


def _load_obj_meshes(mesh_path, seed=0, max_meshes=None, max_triangles=None):
    """Load meshes from `mesh_path` (a .obj file OR a directory of .obj files).
    Every file goes through `_load_obj_broken`. Returns [(name, tri_soup)]
    sorted small -> large by post-broken triangle count.
    """
    from pathlib import Path

    p = Path(mesh_path)
    if p.is_dir():
        files = sorted(p.glob("*.obj"))
    elif p.suffix.lower() == ".obj" and p.is_file():
        files = [p]
    else:
        raise FileNotFoundError(
            f"{mesh_path} is neither a .obj file nor a directory of .obj files"
        )
    if not files:
        raise FileNotFoundError(f"No .obj files found at {mesh_path}")

    meshes = []
    for f in files:
        try:
            tri = _load_obj_broken(f, seed=seed)
        except Exception as e:
            print(f"  [warn] failed to load {f.name}: {e}")
            continue
        if tri is None:
            print(f"  [warn] {f.name}: empty after slice+drop, skipping")
            continue
        if max_triangles is not None and len(tri) > max_triangles:
            print(f"  [skip] {f.name}: {len(tri)} tri > {max_triangles}")
            continue
        meshes.append((f.stem, tri))

    meshes.sort(key=lambda x: len(x[1]))
    if max_meshes is not None:
        meshes = meshes[:max_meshes]
    return meshes


def test_gg_real_meshes(
    mesh_path,
    epsilon,
    beta,
    verbose=True,
    seed=0,
    grid_sizes=(64, 128, 256, 512),
    d_values=(1, 8, 32, 128),
    bf_budget_seconds=900.0,
    bf_cal_n=10_000,
    bf_cal_m=100_000,
    max_meshes=None,
    max_triangles_per_mesh=None,
    max_memory_gb=48.0,
):
    """Timing: GG on deliberately broken real meshes.

    Mirror of test_qg_real_meshes; drives the geometry-gradient path.
    Output is (N, 3, 3) per-triangle gradients plus the eps_g scalar buffer
    (same convention as the sphere GG scale test).
    """
    print(_hdr("\n=== Timing: GG, real broken meshes ==="))
    print(f"  epsilon={epsilon:g}, beta={beta}")
    print(f"  mesh_path={mesh_path}")
    print(f"  grids={grid_sizes}, D={d_values}")
    print(
        f"  BF budget: {bf_budget_seconds:.0f} s/cell, "
        f"cal={bf_cal_n} tri x {bf_cal_m} q"
    )

    meshes = _load_obj_meshes(
        mesh_path,
        seed=seed,
        max_meshes=max_meshes,
        max_triangles=max_triangles_per_mesh,
    )
    if not meshes:
        print("  [error] no usable meshes")
        return

    print(f"  loaded {len(meshes)} meshes (sorted by post-broken tri count):")
    for name, tri in meshes:
        print(f"    {name:<32} {len(tri):>10} tri")

    stream = _current_stream()
    v_cal, i_cal = make_synthetic_sphere(100, 56)
    tri_cal = v_cal[i_cal][:bf_cal_n]
    N_cal = len(tri_cal)
    q_cal = generate_queries(v_cal, bf_cal_m, seed + 1)
    t_cal = _to_cuda(tri_cal)
    qc_cal = _to_cuda(q_cal)

    bf_cal = {}
    for D in d_values:
        rng_c = np.random.default_rng(seed + 1000 + D)
        payload_cal = rng_c.standard_normal((N_cal, D)).astype(np.float32)
        g_cal = rng_c.standard_normal((bf_cal_m, D)).astype(np.float32)
        pl_cal = _to_cuda(payload_cal)
        g_cal_t = _to_cuda(g_cal)
        out_cal = torch.zeros((N_cal, 3, 3), device="cuda:0", dtype=torch.float32)
        eps_cal = torch.zeros((1,), device="cuda:0", dtype=torch.float32)

        def bf_cal_fn():
            winder.brute_force_gradients_with_payload_triangle_soup(
                g_cal_t,
                t_cal,
                pl_cal,
                qc_cal,
                out_cal,
                eps_cal,
                float(epsilon),
                stream,
                True,
            )

        cal_ms = _time(bf_cal_fn, warmup=1, iters=3)[1] * 1000
        bf_cal[D] = (cal_ms, N_cal, bf_cal_m)
        print(f"  calibrated D={D:>3}: {cal_ms:.3f} ms @ {N_cal} tri x {bf_cal_m} q")

    print(
        f"\n  {'mesh':<22} {'tri':>9} {'q':>10} {'res':>5} {'D':>4} | "
        f"{'BF(ms)':>11} {'fast(ms)':>9} {'speedup':>9} | {'rms':>12}"
    )
    print("  " + "-" * 105)

    for name, tri in meshes:
        N = len(tri)
        t = _to_cuda(tri)

        for res in grid_sizes:
            q_np = _regular_grid(tri, res)
            M = len(q_np)

            for D in d_values:
                est = _estimate_mem_gb(M, N, D)
                if est > max_memory_gb:
                    print(
                        f"  {name:<22} {N:>9} {M:>10} {res:>5} {D:>4} | "
                        f"  SKIP  (est {est:.1f} GB > {max_memory_gb:.0f})"
                    )
                    continue

                rng = np.random.default_rng(seed + D)
                payload = rng.standard_normal((N, D)).astype(np.float32)
                g_np = rng.standard_normal((M, D)).astype(np.float32)

                qc = _to_cuda(q_np)
                pl = _to_cuda(payload)
                g = _to_cuda(g_np)
                out = torch.zeros((N, 3, 3), device="cuda:0", dtype=torch.float32)
                eps_g = torch.zeros((1,), device="cuda:0", dtype=torch.float32)

                cal_ms, cal_n, cal_m = bf_cal[D]
                predicted = _estimate_bf_ms(cal_ms, cal_n, cal_m, D, N, M, D)
                do_bf = predicted <= bf_budget_seconds * 1000.0

                if do_bf:

                    def bf_fn():
                        winder.brute_force_gradients_with_payload_triangle_soup(
                            g,
                            t,
                            pl,
                            qc,
                            out,
                            eps_g,
                            float(epsilon),
                            stream,
                            True,
                        )

                    bf_ms = _time(bf_fn, warmup=1, iters=2)[1] * 1000
                    bf_ref = out.clone()
                    bf_tag = ""
                else:
                    bf_ms = predicted
                    bf_ref = None
                    bf_tag = "*"

                eng = winder.PayloadFieldGeometryGradient(qc, g, stream)
                eng.compute_triangle_soup(
                    t, pl, out, eps_g, beta, epsilon, stream, True
                )
                torch.cuda.synchronize()

                def fast_fn():
                    eng.compute_triangle_soup(
                        t, pl, out, eps_g, beta, epsilon, stream, True
                    )

                fast_ms = _time(fast_fn, warmup=1, iters=2)[1] * 1000

                if bf_ref is not None:
                    rms = float(
                        torch.sqrt(
                            torch.mean((out.float() - bf_ref.float()) ** 2)
                        ).item()
                    )
                    rms_s = f"{rms:>12.3e}"
                else:
                    rms_s = f"{'(extrap)':>12s}"

                speedup = bf_ms / fast_ms if fast_ms > 0 else float("nan")
                print(
                    f"  {name:<22} {N:>9} {M:>10} {res:>5} {D:>4} | "
                    f"{bf_ms:>10.2f}{bf_tag:<1s} {fast_ms:>9.3f} "
                    f"{speedup:>8.1f}{bf_tag:<1s}x | {rms_s}"
                )

                del qc, pl, g, out, eps_g, eng
                torch.cuda.empty_cache()

        del t
        torch.cuda.empty_cache()


# =============================================================================
# Main
# =============================================================================
def _summarize(results: list[GGMetrics]) -> bool:
    n_pass = sum(1 for r in results if r.passed)
    n_total = len(results)
    print("\n" + "=" * 88)
    print(_hdr(f"SUMMARY: {n_pass}/{n_total} tests passed"))
    print("=" * 88)

    failed = [r for r in results if not r.passed]
    if not failed:
        print(_ok("  ALL TESTS PASSED"))
        return True

    print(_bad(f"  FAILURES ({len(failed)}):"))
    for r in failed:
        checks = ", ".join(r.failed_checks) if r.failed_checks else "?"
        print(
            f"    - {r.label}  ({checks})  "
            f"rms={r.rms_abs:.2e}  max={r.max_abs:.2e}  "
            f"eps_rel={r.eps_rel_err:.2e}"
        )
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Validate payload geometry gradients against brute force."
    )
    parser.add_argument("--obj_file", type=str, default=None)
    parser.add_argument(
        "--mesh_scale",
        type=str,
        choices=list(MESH_PRESETS.keys()) + ["all"],
        default="small",
    )
    parser.add_argument("--query_count", type=int, default=100000)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument(
        "--epsilon_sweep",
        type=str,
        default=",".join(str(e) for e in DEFAULT_EPSILON_SWEEP),
    )
    parser.add_argument(
        "--d_values",
        type=str,
        default=",".join(str(d) for d in DEFAULT_D_BUCKETS + DEFAULT_ARBITRARY_D),
    )
    parser.add_argument("--beta", type=float, default=-1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--use_kahan", dest="use_kahan", action="store_true", default=True
    )
    parser.add_argument("--no_kahan", dest="use_kahan", action="store_false")
    parser.add_argument(
        "--test",
        type=str,
        default="all",
        help="'all' or comma-separated: payload_d_coverage, "
        "linearity_grad, zero_payload, zero_adjoint, kahan, broken, eps, "
        "mesh_scale, d1_scalar_match, timing",
    )
    parser.add_argument(
        "--scale-budget",
        type=float,
        default=900.0,
        help="BF per-cell budget in seconds; cells above this are extrapolated",
    )
    # --- Real-mesh benchmark ------------------------------------------------
    parser.add_argument(
        "--mesh-path",
        type=str,
        default=None,
        help="A .obj file OR a directory of .obj files. Setting this switches "
        "into the real-mesh benchmark; every file is loaded, sliced in "
        "half along z, then half the triangles are dropped. All cells use "
        "full regular grids over the mesh bbox.",
    )
    parser.add_argument(
        "--mesh-grid-sizes",
        type=str,
        default="64,128,256,512",
        help="Comma-separated grid resolutions; each is a full res^3 grid.",
    )
    parser.add_argument(
        "--mesh-max-meshes",
        type=int,
        default=None,
        help="Process at most N meshes (smallest-first by post-broken count).",
    )
    parser.add_argument(
        "--mesh-max-triangles",
        type=int,
        default=None,
        help="Skip meshes whose POST-BROKEN triangle count exceeds this. "
        "slice+drop reduces count ~4x, so a 1M source becomes ~250k.",
    )
    parser.add_argument(
        "--mesh-max-memory-gb",
        type=float,
        default=24.0,
        help="Skip cells estimated to need more GPU memory than this.",
    )

    # --- Drop-geometry flag for the existing tests --------------------------
    parser.add_argument(
        "--drop-geometry",
        action="store_true",
        default=False,
        help="In the existing validation and sphere scale tests, apply "
        "slice_mesh_in_half + drop_half_of_the_triangles to the loaded "
        "geometry (whether --obj_file or synthetic sphere). Useful for "
        "exercising non-manifold behaviour across the whole suite.",
    )
    args = parser.parse_args()
    set_seeds(args.seed)

    # ---- Real-mesh benchmark mode ----
    if args.mesh_path is not None:
        grid_sizes = tuple(int(s) for s in args.mesh_grid_sizes.split(","))
        d_values = tuple(int(x) for x in args.d_values.split(",") if x.strip())
        test_gg_real_meshes(
            mesh_path=args.mesh_path,
            epsilon=args.epsilon,
            beta=args.beta,
            seed=args.seed,
            grid_sizes=grid_sizes,
            d_values=d_values,
            bf_budget_seconds=args.scale_budget,
            max_meshes=args.mesh_max_meshes,
            max_triangles_per_mesh=args.mesh_max_triangles,
            max_memory_gb=args.mesh_max_memory_gb,
        )
        return

    # ---------------------------------------------------------------------
    # Load geometry
    # ---------------------------------------------------------------------
    if args.obj_file:
        import igl

        v, _, _, i, _, _ = igl.readOBJ(args.obj_file)
        vertices = np.ascontiguousarray(v, dtype=np.float32)
        indices = np.ascontiguousarray(i, dtype=np.uint32)
        print(
            f"Loaded {args.obj_file}: {len(indices)} triangles, {len(vertices)} verts"
        )
    else:
        n_theta, n_phi = MESH_PRESETS[
            "medium" if args.mesh_scale == "all" else args.mesh_scale
        ]
        vertices, indices = make_synthetic_sphere(n_theta, n_phi)
        print(f"Synthetic sphere: {len(indices)} triangles, {len(vertices)} verts")

    if args.drop_geometry:
        before = len(indices)
        vertices, indices = slice_mesh_in_half(vertices, indices)
        vertices, indices = drop_half_of_the_triangles(vertices, indices,
                                                       seed=args.seed)
        print(f"Drop-geometry: {before} -> {len(indices)} triangles "
              f"(sliced + dropped half)")

    triangles = vertices[indices]
    points, normals = mesh_to_point_surfels(vertices, indices)
    queries = generate_queries(vertices, args.query_count, args.seed)
    print(f"Point-normal surfels: {len(points)}")
    print(f"Queries: {len(queries)}")
    print(
        f"beta={args.beta}, epsilon={args.epsilon:g}, "
        f"Kahan={'on' if args.use_kahan else 'off'}"
    )

    d_values = [int(x) for x in args.d_values.split(",") if x.strip()]
    eps_values = [float(x) for x in args.epsilon_sweep.split(",") if x.strip()]
    print(f"D values ({len(d_values)}): {d_values}")
    print(f"eps values: {eps_values}")

    tests = (
        set(args.test.split(","))
        if args.test != "all"
        else {
            "payload_d_coverage",
            "linearity_grad",
            "zero_payload",
            "zero_adjoint",
            "kahan",
            "broken",
            "eps",
            "mesh_scale",
            "d1_scalar_match",
            "timing",
        }
    )

    results: list[GGMetrics] = []
    t0 = time()

    if "payload_d_coverage" in tests:
        test_gg_payload_d_coverage(
            vertices,
            indices,
            points,
            normals,
            queries,
            args.epsilon,
            args.beta,
            args.use_kahan,
            d_values,
            results,
        )

    if "linearity_grad" in tests:
        test_gg_payload_linearity_grad_output(
            vertices,
            indices,
            queries,
            args.epsilon,
            args.beta,
            args.use_kahan,
            results,
        )

    if "zero_payload" in tests:
        test_gg_payload_zero_payload(
            vertices,
            indices,
            points,
            normals,
            queries,
            args.epsilon,
            args.beta,
            args.use_kahan,
            results,
        )

    if "zero_adjoint" in tests:
        test_gg_payload_zero_adjoint(
            vertices,
            indices,
            queries,
            args.epsilon,
            args.beta,
            args.use_kahan,
            results,
        )

    if "kahan" in tests:
        test_gg_payload_kahan_consistency(
            vertices,
            indices,
            queries,
            args.epsilon,
            args.beta,
            results,
        )

    if "broken" in tests:
        test_gg_payload_broken_meshes(
            vertices,
            indices,
            queries,
            args.epsilon,
            args.beta,
            args.use_kahan,
            results,
            seed=args.seed,
        )

    if "eps" in tests:
        test_gg_payload_epsilon_sweep(
            vertices,
            indices,
            queries,
            eps_values,
            args.beta,
            args.use_kahan,
            results,
        )

    if "mesh_scale" in tests:
        test_gg_payload_mesh_scale_sweep(
            args.query_count,
            args.epsilon,
            args.beta,
            args.use_kahan,
            results,
            seed=args.seed,
        )

    if "d1_scalar_match" in tests:
        test_gg_payload_d1_scalar_match(
            vertices,
            indices,
            points,
            normals,
            queries,
            args.epsilon,
            args.beta,
            results,
        )

    if "timing" in tests:
        timing_d = [d for d in [1, 8, 32, 128] if d in d_values] or [1, 8]
        test_gg_timing(
            vertices,
            indices,
            queries,
            args.epsilon,
            args.beta,
            timing_d,
        )
    if "scale_timing" in tests:
        test_gg_scale_timing(
            args.epsilon,
            args.beta,
            seed=args.seed,
            bf_budget_seconds=args.scale_budget,
        )

    elapsed = time() - t0
    ok = _summarize(results)
    print(f"\nElapsed: {elapsed:.2f}s")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
