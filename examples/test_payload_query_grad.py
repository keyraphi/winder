"""Validate PayloadField and WindingNumberEngine query gradients.

Tests every kernel path:
  - Scalar winding query gradients
  - Payload query gradients (all D buckets + arbitrary D)
  - Fast (BVH) vs brute-force reference
  - D=1 payload reduces to scalar winding
  - Linearity in the adjoint (grad_output)
  - Kahan on/off
  - Closed / sliced / dropped meshes
  - Epsilon sweep
  - Mesh scale sweep (small .. 1M triangles)
  - Query scale sweep (1 .. 1M) - catches warp-tile-base bugs
  - Timing

Usage
-----
    # Default: synthetic small sphere, all tests, ~60 s
    python test_payload_query_grad.py

    # Load a real mesh
    python test_payload_query_grad.py --obj_file benchy.obj --query_count 500

    # Only a subset
    python test_payload_query_grad.py --test payload_d_coverage

    # Huge stress
    python test_payload_query_grad.py --mesh_scale huge --query_count 10000
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


def _ok(m): return f"{GREEN}{m}{RESET}"
def _bad(m): return f"{RED}{m}{RESET}"
def _warn(m): return f"{YELLOW}{m}{RESET}"
def _hdr(m): return f"{BOLD}{CYAN}{m}{RESET}"


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


def _rms(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.sqrt(np.mean(d * d)))


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
# Geometry helpers (shared with test_payload_forward.py)
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
# Fast / brute runners - scalar winding query gradients
# =============================================================================
def run_fast_qg_scalar_point_normal(points, normals, queries, grad_output,
                                    epsilon, beta, stream):
    p = _to_cuda(points)
    n = _to_cuda(normals)
    q = _to_cuda(queries)
    g = _to_cuda(grad_output)
    M = q.shape[0]
    out = torch.empty((M, 3), device="cuda:0", dtype=torch.float32)
    eng = winder.WindingNumberEngine(p, n, stream)
    eng.compute_query_gradients(g, q, out, beta, epsilon, stream)
    torch.cuda.synchronize()
    return out


def run_brute_qg_scalar_point_normal(points, normals, queries, grad_output,
                                     epsilon, use_kahan, stream):
    p = _to_cuda(points)
    n = _to_cuda(normals)
    q = _to_cuda(queries)
    g = _to_cuda(grad_output)
    M = q.shape[0]
    out = torch.empty((M, 3), device="cuda:0", dtype=torch.float32)
    winder.brute_force_query_gradients_point_normal(
        g, p, n, q, out, float(epsilon), stream, bool(use_kahan),
    )
    torch.cuda.synchronize()
    return out


def run_fast_qg_scalar_triangle(triangles, queries, grad_output,
                                epsilon, beta, stream):
    t = _to_cuda(triangles)
    q = _to_cuda(queries)
    g = _to_cuda(grad_output)
    M = q.shape[0]
    out = torch.empty((M, 3), device="cuda:0", dtype=torch.float32)
    eng = winder.WindingNumberEngine(t, stream)
    eng.compute_query_gradients(g, q, out, beta, epsilon, stream)
    torch.cuda.synchronize()
    return out


def run_brute_qg_scalar_triangle(triangles, queries, grad_output,
                                 epsilon, use_kahan, stream):
    t = _to_cuda(triangles)
    q = _to_cuda(queries)
    g = _to_cuda(grad_output)
    M = q.shape[0]
    out = torch.empty((M, 3), device="cuda:0", dtype=torch.float32)
    winder.brute_force_query_gradients_triangle_soup(
        g, t, q, out, float(epsilon), stream, bool(use_kahan),
    )
    torch.cuda.synchronize()
    return out


def run_fast_qg_scalar_mesh(vertices, indices, queries, grad_output,
                            epsilon, beta, stream):
    v = _to_cuda(vertices)
    idx = _to_cuda(indices, torch.uint32)
    q = _to_cuda(queries)
    g = _to_cuda(grad_output)
    M = q.shape[0]
    out = torch.empty((M, 3), device="cuda:0", dtype=torch.float32)
    eng = winder.WindingNumberEngine(v, idx, stream)
    eng.compute_query_gradients(g, q, out, beta, epsilon, stream)
    torch.cuda.synchronize()
    return out


def run_brute_qg_scalar_mesh(vertices, indices, queries, grad_output,
                             epsilon, use_kahan, stream):
    v = _to_cuda(vertices)
    idx = _to_cuda(indices, torch.uint32)
    q = _to_cuda(queries)
    g = _to_cuda(grad_output)
    M = q.shape[0]
    out = torch.empty((M, 3), device="cuda:0", dtype=torch.float32)
    winder.brute_force_query_gradients_mesh(
        g, v, idx, q, out, float(epsilon), stream, bool(use_kahan),
    )
    torch.cuda.synchronize()
    return out


# =============================================================================
# Fast / brute runners - payload query gradients
# =============================================================================
def run_fast_qg_payload_point_normal(points, normals, payload, queries,
                                     grad_output, epsilon, beta, use_kahan,
                                     stream):
    p = _to_cuda(points)
    n = _to_cuda(normals)
    pl = _to_cuda(payload)
    q = _to_cuda(queries)
    g = _to_cuda(grad_output)
    M = q.shape[0]
    out = torch.empty((M, 3), device="cuda:0", dtype=torch.float32)
    eng = winder.PayloadField(p, n, pl, stream)
    eng.compute_query_gradients(g, q, out, beta, epsilon, stream, bool(use_kahan))
    torch.cuda.synchronize()
    return out


def run_brute_qg_payload_point_normal(points, normals, payload, queries,
                                      grad_output, epsilon, use_kahan, stream):
    p = _to_cuda(points)
    n = _to_cuda(normals)
    pl = _to_cuda(payload)
    q = _to_cuda(queries)
    g = _to_cuda(grad_output)
    M = q.shape[0]
    out = torch.empty((M, 3), device="cuda:0", dtype=torch.float32)
    winder.brute_force_query_gradients_with_payload_point_normal(
        g, p, n, pl, q, out, float(epsilon), stream, bool(use_kahan),
    )
    torch.cuda.synchronize()
    return out


def run_fast_qg_payload_triangle(triangles, payload, queries, grad_output,
                                 epsilon, beta, use_kahan, stream):
    t = _to_cuda(triangles)
    pl = _to_cuda(payload)
    q = _to_cuda(queries)
    g = _to_cuda(grad_output)
    M = q.shape[0]
    out = torch.empty((M, 3), device="cuda:0", dtype=torch.float32)
    eng = winder.PayloadField(t, pl, stream)
    eng.compute_query_gradients(g, q, out, beta, epsilon, stream, bool(use_kahan))
    torch.cuda.synchronize()
    return out


def run_brute_qg_payload_triangle(triangles, payload, queries, grad_output,
                                  epsilon, use_kahan, stream):
    t = _to_cuda(triangles)
    pl = _to_cuda(payload)
    q = _to_cuda(queries)
    g = _to_cuda(grad_output)
    M = q.shape[0]
    out = torch.empty((M, 3), device="cuda:0", dtype=torch.float32)
    winder.brute_force_query_gradients_with_payload_triangle_soup(
        g, t, pl, q, out, float(epsilon), stream, bool(use_kahan),
    )
    torch.cuda.synchronize()
    return out


def run_fast_qg_payload_mesh(vertices, indices, payload, queries, grad_output,
                             epsilon, beta, use_kahan, stream):
    v = _to_cuda(vertices)
    idx = _to_cuda(indices, torch.uint32)
    pl = _to_cuda(payload)
    q = _to_cuda(queries)
    g = _to_cuda(grad_output)
    M = q.shape[0]
    out = torch.empty((M, 3), device="cuda:0", dtype=torch.float32)
    eng = winder.PayloadField(v, idx, pl, stream)
    eng.compute_query_gradients(g, q, out, beta, epsilon, stream, bool(use_kahan))
    torch.cuda.synchronize()
    return out


def run_brute_qg_payload_mesh(vertices, indices, payload, queries, grad_output,
                              epsilon, use_kahan, stream):
    v = _to_cuda(vertices)
    idx = _to_cuda(indices, torch.uint32)
    pl = _to_cuda(payload)
    q = _to_cuda(queries)
    g = _to_cuda(grad_output)
    M = q.shape[0]
    out = torch.empty((M, 3), device="cuda:0", dtype=torch.float32)
    winder.brute_force_query_gradients_with_payload_mesh(
        g, v, idx, pl, q, out, float(epsilon), stream, bool(use_kahan),
    )
    torch.cuda.synchronize()
    return out


# =============================================================================
# Validation
# =============================================================================
@dataclass
class QGMetrics:
    label: str
    shape: tuple
    has_nan: bool
    has_inf: bool
    signal_max: float
    rms_abs: float
    max_abs: float
    cos_flat: float
    passed: bool = False
    failed_checks: tuple = ()


def validate_query_grad(
    fast: torch.Tensor,
    ref: torch.Tensor,
    label: str,
    *,
    rms_tol: float = 5e-2,
    max_abs_tol: float = 1.0,
    cos_tol: float = 0.98,
    verbose: bool = True,
) -> QGMetrics:
    """Compare two (M, 3) query-gradient tensors."""
    assert fast.shape == ref.shape, f"shape mismatch: {fast.shape} vs {ref.shape}"

    fast_np = fast.detach().float().cpu().numpy()
    ref_np = ref.detach().float().cpu().numpy()

    has_nan = int(np.isnan(fast_np).sum())
    has_inf = int(np.isinf(fast_np).sum())
    if has_nan or has_inf:
        if verbose:
            print(f"  [{_bad('FAIL')}] {label}: NaN={has_nan} Inf={has_inf}")
        return QGMetrics(
            label=label, shape=tuple(fast_np.shape),
            has_nan=has_nan, has_inf=has_inf,
            signal_max=0.0, rms_abs=float("inf"), max_abs=float("inf"),
            cos_flat=float("nan"),
            passed=False, failed_checks=("nan_or_inf",),
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
        if (nf > 0 and nr > 0) else float("nan")
    )

    checks = {
        "rms": rms_abs < rms_tol,
        "max_abs": max_abs < max_abs_tol,
        "cos_flat": cos_flat > cos_tol,
    }
    passed = all(checks.values())
    failed = tuple(k for k, v in checks.items() if not v)

    if verbose:
        status = _ok("PASS") if passed else _bad("FAIL")
        print(
            f"  [{status}] {label:<42}  "
            f"shape={tuple(fast_np.shape)}  "
            f"rms={rms_abs:.2e}  max={max_abs:.2e}  cos={cos_flat:.5f}"
        )

    return QGMetrics(
        label=label, shape=tuple(fast_np.shape),
        has_nan=False, has_inf=False, signal_max=signal_max,
        rms_abs=rms_abs, max_abs=max_abs, cos_flat=cos_flat,
        passed=passed, failed_checks=failed,
    )



# =============================================================================
# Tests
# =============================================================================
def test_single_leaf_scalar_qg(
    epsilon, beta, results, verbose=True, seed=0, n_queries=100,
):
    """Exercise the scalar winding QG single-leaf kernel at the 32/33 boundary.

    `traverse_single_leaf_kernel<QueryGradient>` is used whenever
    `geometry_count <= 32`; `traverse_geometry_tree_kernel<QueryGradient>`
    takes over at 33. We test both sides of the boundary so any off-by-one
    in the dispatch condition shows up immediately.
    """
    print(_hdr("\nTest: single-leaf scalar winding QG (and 32/33 boundary)"))
    stream = _current_stream()

    v_base, i_base = make_synthetic_sphere(8, 6)  # ~84 triangles
    if len(i_base) < 33:
        raise RuntimeError(
            f"Base sphere has only {len(i_base)} triangles, need >= 33"
        )
    print(f"  Base sphere: {len(i_base)} triangles, {len(v_base)} verts")

    for n_tri in [1, 4, 16, 31, 32, 33]:
        indices = i_base[:n_tri]
        vertices = v_base
        triangles = vertices[indices]
        points, normals = mesh_to_point_surfels(vertices, indices)

        q_np = generate_queries(vertices, n_queries, seed)
        g_np = (
            np.random.default_rng(seed + 1)
            .standard_normal(n_queries)
            .astype(np.float32)
        )

        # Triangle soup
        f = run_fast_qg_scalar_triangle(
            triangles, q_np, g_np, epsilon, beta, stream,
        )
        b = run_brute_qg_scalar_triangle(
            triangles, q_np, g_np, epsilon, True, stream,
        )
        results.append(validate_query_grad(
            f, b, f"Tri  N={n_tri:>2}", verbose=verbose,
        ))

        # PointNormal
        f = run_fast_qg_scalar_point_normal(
            points, normals, q_np, g_np, epsilon, beta, stream,
        )
        b = run_brute_qg_scalar_point_normal(
            points, normals, q_np, g_np, epsilon, True, stream,
        )
        results.append(validate_query_grad(
            f, b, f"PN   N={n_tri:>2}", verbose=verbose,
        ))

        # Mesh
        f = run_fast_qg_scalar_mesh(
            vertices, indices, q_np, g_np, epsilon, beta, stream,
        )
        b = run_brute_qg_scalar_mesh(
            vertices, indices, q_np, g_np, epsilon, True, stream,
        )
        results.append(validate_query_grad(
            f, b, f"Mesh N={n_tri:>2}", verbose=verbose,
        ))


def test_single_leaf_payload_qg(
    epsilon, beta, use_kahan, results, verbose=True, seed=0, n_queries=100,
):
    """Exercise the payload QG single-leaf kernel (geometry_count <= 32).

    `traverse_payload_query_grad_single_leaf_kernel<Geometry, D, UseKahan>`
    is taken when `geometry_count <= 32`;
    `traverse_payload_query_gradient_kernel<...>` at > 32. We stress both,
    across all four COMPONENTS_PER_THREAD regimes and both Kahan settings.
    """
    print(_hdr("\nTest: single-leaf payload QG (and 32/33 boundary)"))
    stream = _current_stream()
    rng = np.random.default_rng(1001)

    v_base, i_base = make_synthetic_sphere(8, 6)  # ~84 triangles
    if len(i_base) < 33:
        raise RuntimeError(
            f"Base sphere has only {len(i_base)} triangles, need >= 33"
        )

    # D values covering all COMPONENTS_PER_THREAD regimes:
    #   D=1,2,3  ->  CPT = 1,2,3 (partial), THREADS_PER_QUERY = 1
    #   D=4      ->  CPT = 4 (full),       THREADS_PER_QUERY = 1
    #   D=8      ->  CPT = 4 (full),       THREADS_PER_QUERY = 2
    d_values = [1, 2, 3, 4, 8]

    for n_tri in [1, 4, 16, 31, 32, 33]:
        indices = i_base[:n_tri]
        vertices = v_base
        triangles = vertices[indices]
        points, normals = mesh_to_point_surfels(vertices, indices)

        q_np = generate_queries(vertices, n_queries, seed)

        for D in d_values:
            payload_tri = rng.standard_normal((n_tri, D)).astype(np.float32)
            payload_pn = rng.standard_normal((n_tri, D)).astype(np.float32)
            g_np = rng.standard_normal((n_queries, D)).astype(np.float32)

            # Triangle soup
            f = run_fast_qg_payload_triangle(
                triangles, payload_tri, q_np, g_np,
                epsilon, beta, use_kahan, stream,
            )
            b = run_brute_qg_payload_triangle(
                triangles, payload_tri, q_np, g_np,
                epsilon, use_kahan, stream,
            )
            results.append(validate_query_grad(
                f, b, f"Tri  N={n_tri:>2} D={D:>3}", verbose=verbose,
            ))

            # PointNormal
            f = run_fast_qg_payload_point_normal(
                points, normals, payload_pn, q_np, g_np,
                epsilon, beta, use_kahan, stream,
            )
            b = run_brute_qg_payload_point_normal(
                points, normals, payload_pn, q_np, g_np,
                epsilon, use_kahan, stream,
            )
            results.append(validate_query_grad(
                f, b, f"PN   N={n_tri:>2} D={D:>3}", verbose=verbose,
            ))

            # Mesh
            f = run_fast_qg_payload_mesh(
                vertices, indices, payload_tri, q_np, g_np,
                epsilon, beta, use_kahan, stream,
            )
            b = run_brute_qg_payload_mesh(
                vertices, indices, payload_tri, q_np, g_np,
                epsilon, use_kahan, stream,
            )
            results.append(validate_query_grad(
                f, b, f"Mesh N={n_tri:>2} D={D:>3}", verbose=verbose,
            ))


def test_single_leaf_d1_scalar_match(
    epsilon, beta, results, verbose=True, seed=0, n_queries=100,
):
    """In the single-leaf path, payload D=1 with v=1, g=1 must equal the
    scalar winding QG. This is the QG analogue of the forward-pass
    'D=1 with v=ones == scalar winding' test.

    If this passes but `test_payload_d_coverage` fails, the bug is in the
    BVH traversal. If both fail, the bug is in the shared coefficient
    handling (e.g. the `from_f16`/`from_float` round-trip).
    """
    print(_hdr("\nTest: single-leaf D=1 payload == scalar winding QG"))
    stream = _current_stream()

    v_base, i_base = make_synthetic_sphere(8, 6)

    for n_tri in [4, 16, 32]:
        indices = i_base[:n_tri]
        vertices = v_base
        triangles = vertices[indices]
        points, normals = mesh_to_point_surfels(vertices, indices)

        q_np = generate_queries(vertices, n_queries, seed)
        g_scalar = np.ones(n_queries, dtype=np.float32)
        ones_tri = np.ones((n_tri, 1), dtype=np.float32)
        ones_pn = np.ones((n_tri, 1), dtype=np.float32)
        g_payload = np.ones((n_queries, 1), dtype=np.float32)

        # --- Fast vs fast ---
        s_tri = run_fast_qg_scalar_triangle(
            triangles, q_np, g_scalar, epsilon, beta, stream,
        )
        p_tri = run_fast_qg_payload_triangle(
            triangles, ones_tri, q_np, g_payload,
            epsilon, beta, True, stream,
        )
        results.append(validate_query_grad(
            p_tri, s_tri, f"Tri  N={n_tri:>2} D=1 (fast)",
            rms_tol=1e-3, max_abs_tol=1e-2, cos_tol=0.9999, verbose=verbose,
        ))

        s_pn = run_fast_qg_scalar_point_normal(
            points, normals, q_np, g_scalar, epsilon, beta, stream,
        )
        p_pn = run_fast_qg_payload_point_normal(
            points, normals, ones_pn, q_np, g_payload,
            epsilon, beta, True, stream,
        )
        results.append(validate_query_grad(
            p_pn, s_pn, f"PN   N={n_tri:>2} D=1 (fast)",
            rms_tol=1e-3, max_abs_tol=1e-2, cos_tol=0.9999, verbose=verbose,
        ))

        # --- Brute vs brute (pure math identity, no BVH involved) ---
        b_tri = run_brute_qg_payload_triangle(
            triangles, ones_tri, q_np, g_payload, epsilon, True, stream,
        )
        s_tri_b = run_brute_qg_scalar_triangle(
            triangles, q_np, g_scalar, epsilon, True, stream,
        )
        results.append(validate_query_grad(
            b_tri, s_tri_b, f"Tri  N={n_tri:>2} D=1 (brute)",
            rms_tol=1e-3, max_abs_tol=1e-2, cos_tol=0.9999, verbose=verbose,
        ))
# =============================================================================
# Fast / brute runners - payload FORWARD pass
# =============================================================================
def run_fast_forward_payload_point_normal(points, normals, payload, queries,
                                          epsilon, beta, use_kahan, stream):
    p = _to_cuda(points)
    n = _to_cuda(normals)
    pl = _to_cuda(payload)
    q = _to_cuda(queries)
    M, D = q.shape[0], pl.shape[1]
    out = torch.empty((M, D), device="cuda:0", dtype=torch.float32)
    eng = winder.PayloadField(p, n, pl, stream)
    eng.compute(q, out, beta, epsilon, stream, bool(use_kahan))
    torch.cuda.synchronize()
    return out


def run_brute_forward_payload_point_normal(points, normals, payload, queries,
                                           epsilon, use_kahan, stream):
    p = _to_cuda(points)
    n = _to_cuda(normals)
    pl = _to_cuda(payload)
    q = _to_cuda(queries)
    M, D = q.shape[0], pl.shape[1]
    out = torch.empty((M, D), device="cuda:0", dtype=torch.float32)
    winder.brute_force_winding_payloads_point_normal(
        p, n, pl, q, out, float(epsilon), stream, bool(use_kahan),
    )
    torch.cuda.synchronize()
    return out


def run_fast_forward_payload_triangle(triangles, payload, queries,
                                      epsilon, beta, use_kahan, stream):
    t = _to_cuda(triangles)
    pl = _to_cuda(payload)
    q = _to_cuda(queries)
    M, D = q.shape[0], pl.shape[1]
    out = torch.empty((M, D), device="cuda:0", dtype=torch.float32)
    eng = winder.PayloadField(t, pl, stream)
    eng.compute(q, out, beta, epsilon, stream, bool(use_kahan))
    torch.cuda.synchronize()
    return out


def run_brute_forward_payload_triangle(triangles, payload, queries,
                                       epsilon, use_kahan, stream):
    t = _to_cuda(triangles)
    pl = _to_cuda(payload)
    q = _to_cuda(queries)
    M, D = q.shape[0], pl.shape[1]
    out = torch.empty((M, D), device="cuda:0", dtype=torch.float32)
    winder.brute_force_winding_payloads_triangle_soup(
        t, pl, q, out, float(epsilon), stream, bool(use_kahan),
    )
    torch.cuda.synchronize()
    return out


def run_fast_forward_payload_mesh(vertices, indices, payload, queries,
                                  epsilon, beta, use_kahan, stream):
    v = _to_cuda(vertices)
    idx = _to_cuda(indices, torch.uint32)
    pl = _to_cuda(payload)
    q = _to_cuda(queries)
    M, D = q.shape[0], pl.shape[1]
    out = torch.empty((M, D), device="cuda:0", dtype=torch.float32)
    eng = winder.PayloadField(v, idx, pl, stream)
    eng.compute(q, out, beta, epsilon, stream, bool(use_kahan))
    torch.cuda.synchronize()
    return out


def run_brute_forward_payload_mesh(vertices, indices, payload, queries,
                                   epsilon, use_kahan, stream):
    v = _to_cuda(vertices)
    idx = _to_cuda(indices, torch.uint32)
    pl = _to_cuda(payload)
    q = _to_cuda(queries)
    M, D = q.shape[0], pl.shape[1]
    out = torch.empty((M, D), device="cuda:0", dtype=torch.float32)
    winder.brute_force_winding_payloads_mesh(
        v, idx, pl, q, out, float(epsilon), stream, bool(use_kahan),
    )
    torch.cuda.synchronize()
    return out

def test_single_leaf_payload_forward(
    epsilon, beta, use_kahan, results, verbose=True, seed=0, n_queries=100,
):
    """Exercise the payload FORWARD single-leaf kernel (geometry_count <= 32).

    `traverse_payload_single_leaf_kernel<Geometry, D, UseKahan>` is used when
    `geometry_count <= 32`. This test compares against the brute-force
    reference across all D regimes and across the 32/33 boundary.

    This test catches the "w computed per-lane for that lane's own query"
    bug that was present in the forward single-leaf kernel for months: the
    winding contributions w_j were computed against different queries across
    lanes, producing a scrambled forward field. `test_payload_forward.py`
    never exercised this path because its smallest preset is ~48 triangles,
    which uses the BVH traversal.
    """
    print(_hdr("\nTest: single-leaf payload forward (and 32/33 boundary)"))
    stream = _current_stream()
    rng = np.random.default_rng(2001)

    v_base, i_base = make_synthetic_sphere(8, 6)  # ~84 triangles
    if len(i_base) < 33:
        raise RuntimeError(
            f"Base sphere has only {len(i_base)} triangles, need >= 33"
        )

    # Covers all COMPONENTS_PER_THREAD regimes:
    #   D=1,2,3  -> CPT=1,2,3 (partial), TPQ=1
    #   D=4      -> CPT=4 (full),        TPQ=1
    #   D=8      -> CPT=4 (full),        TPQ=2
    #   D=16     -> CPT=4 (full),        TPQ=4
    d_values = [1, 2, 3, 4, 8, 16]

    for n_tri in [1, 4, 16, 31, 32, 33]:
        indices = i_base[:n_tri]
        vertices = v_base
        triangles = vertices[indices]
        points, normals = mesh_to_point_surfels(vertices, indices)

        q_np = generate_queries(vertices, n_queries, seed)

        for D in d_values:
            payload_tri = rng.standard_normal((n_tri, D)).astype(np.float32)
            payload_pn = rng.standard_normal((n_tri, D)).astype(np.float32)

            f = run_fast_forward_payload_triangle(
                triangles, payload_tri, q_np, epsilon, beta, use_kahan, stream,
            )
            b = run_brute_forward_payload_triangle(
                triangles, payload_tri, q_np, epsilon, use_kahan, stream,
            )
            results.append(validate_query_grad(
                f, b, f"Tri  N={n_tri:>2} D={D:>3}",
                rms_tol=5e-3, max_abs_tol=2e-1, cos_tol=0.99, verbose=verbose,
            ))

            f = run_fast_forward_payload_point_normal(
                points, normals, payload_pn, q_np,
                epsilon, beta, use_kahan, stream,
            )
            b = run_brute_forward_payload_point_normal(
                points, normals, payload_pn, q_np,
                epsilon, use_kahan, stream,
            )
            results.append(validate_query_grad(
                f, b, f"PN   N={n_tri:>2} D={D:>3}",
                rms_tol=5e-3, max_abs_tol=2e-1, cos_tol=0.99, verbose=verbose,
            ))

            f = run_fast_forward_payload_mesh(
                vertices, indices, payload_tri, q_np,
                epsilon, beta, use_kahan, stream,
            )
            b = run_brute_forward_payload_mesh(
                vertices, indices, payload_tri, q_np,
                epsilon, use_kahan, stream,
            )
            results.append(validate_query_grad(
                f, b, f"Mesh N={n_tri:>2} D={D:>3}",
                rms_tol=5e-3, max_abs_tol=2e-1, cos_tol=0.99, verbose=verbose,
            ))

def test_scalar_coverage(
    vertices, indices, points, normals, queries, epsilon, beta,
    results, verbose=True,
):
    """Fast vs brute scalar winding QG across all geometry types."""
    print(_hdr("\nTest: scalar winding QG coverage"))
    triangles = vertices[indices]
    M = len(queries)
    rng = np.random.default_rng(101)
    g_np = rng.standard_normal(M).astype(np.float32)
    stream = _current_stream()

    for tag, fast_fn, brute_fn, args in [
        ("Tri", run_fast_qg_scalar_triangle, run_brute_qg_scalar_triangle,
         (triangles,)),
        ("PN", run_fast_qg_scalar_point_normal, run_brute_qg_scalar_point_normal,
         (points, normals)),
        ("Mesh", run_fast_qg_scalar_mesh, run_brute_qg_scalar_mesh,
         (vertices, indices)),
    ]:
        fast = fast_fn(*args, queries, g_np, epsilon, beta, stream)
        brute = brute_fn(*args, queries, g_np, epsilon, True, stream)
        results.append(validate_query_grad(fast, brute, tag, verbose=verbose))


def test_payload_d_coverage(
    vertices, indices, points, normals, queries, epsilon, beta, use_kahan,
    d_values, results, verbose=True,
):
    """Fast vs brute payload QG across all D values × all geometry types."""
    print(_hdr(f"\nTest: payload QG D coverage ({len(d_values)} values)"))
    triangles = vertices[indices]
    M = len(queries)
    rng = np.random.default_rng(202)
    stream = _current_stream()

    for D in d_values:
        payload_tri = rng.standard_normal((len(triangles), D)).astype(np.float32)
        payload_pn = rng.standard_normal((len(points), D)).astype(np.float32)
        g_np = rng.standard_normal((M, D)).astype(np.float32)

        # Triangle
        f = run_fast_qg_payload_triangle(
            triangles, payload_tri, queries, g_np, epsilon, beta, use_kahan, stream
        )
        b = run_brute_qg_payload_triangle(
            triangles, payload_tri, queries, g_np, epsilon, use_kahan, stream
        )
        results.append(validate_query_grad(f, b, f"Tri  D={D:>3}", verbose=verbose))

        # Mesh
        f = run_fast_qg_payload_mesh(
            vertices, indices, payload_tri, queries, g_np,
            epsilon, beta, use_kahan, stream,
        )
        b = run_brute_qg_payload_mesh(
            vertices, indices, payload_tri, queries, g_np,
            epsilon, use_kahan, stream,
        )
        results.append(validate_query_grad(f, b, f"Mesh D={D:>3}", verbose=verbose))

        # PointNormal
        f = run_fast_qg_payload_point_normal(
            points, normals, payload_pn, queries, g_np,
            epsilon, beta, use_kahan, stream,
        )
        b = run_brute_qg_payload_point_normal(
            points, normals, payload_pn, queries, g_np,
            epsilon, use_kahan, stream,
        )
        results.append(validate_query_grad(f, b, f"PN   D={D:>3}", verbose=verbose))


def test_payload_linearity(
    vertices, indices, queries, epsilon, beta, use_kahan,
    results, verbose=True,
):
    """QG is linear in grad_output: QG(a*g1 + b*g2) = a*QG(g1) + b*QG(g2)."""
    print(_hdr("\nTest: payload QG linearity in grad_output (D=8)"))
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

    fc = run_fast_qg_payload_triangle(
        triangles, payload, queries, gc, epsilon, beta, use_kahan, stream
    )
    f1 = run_fast_qg_payload_triangle(
        triangles, payload, queries, g1, epsilon, beta, use_kahan, stream
    )
    f2 = run_fast_qg_payload_triangle(
        triangles, payload, queries, g2, epsilon, beta, use_kahan, stream
    )
    expected = a * f1 + b * f2
    results.append(validate_query_grad(
        fc, expected, "linearity (fast)", verbose=verbose,
    ))

    bc = run_brute_qg_payload_triangle(
        triangles, payload, queries, gc, epsilon, use_kahan, stream
    )
    b1 = run_brute_qg_payload_triangle(
        triangles, payload, queries, g1, epsilon, use_kahan, stream
    )
    b2 = run_brute_qg_payload_triangle(
        triangles, payload, queries, g2, epsilon, use_kahan, stream
    )
    expected_b = a * b1 + b * b2
    results.append(validate_query_grad(
        bc, expected_b, "linearity (brute)", verbose=verbose,
    ))


def test_d1_scalar_match(
    vertices, indices, points, normals, queries, epsilon, beta,
    results, verbose=True,
):
    """D=1 payload with v=ones must equal the scalar winding QG with g=ones.

    We compare like-with-like:
      - fast payload vs fast scalar (both BVH-approximated)
      - brute payload vs brute scalar (both exact)
    Comparing across the fast/brute boundary would surface the ~1e-4
    approximation error of the fast path, which is expected and has
    nothing to do with payload correctness.
    """
    print(_hdr("\nTest: D=1 payload with v=ones == scalar winding QG"))
    triangles = vertices[indices]
    M = len(queries)
    g_scalar = np.ones(M, dtype=np.float32)
    stream = _current_stream()

    ones_tri = np.ones((len(triangles), 1), dtype=np.float32)
    ones_pn = np.ones((len(points), 1), dtype=np.float32)
    g_payload = np.ones((M, 1), dtype=np.float32)

    # ---- Fast path: payload vs scalar (both BVH) ----
    scalar_tri_fast = run_fast_qg_scalar_triangle(
        triangles, queries, g_scalar, epsilon, beta, stream
    )
    scalar_pn_fast = run_fast_qg_scalar_point_normal(
        points, normals, queries, g_scalar, epsilon, beta, stream
    )
    fast_tri = run_fast_qg_payload_triangle(
        triangles, ones_tri, queries, g_payload,
        epsilon, beta, True, stream,
    )
    fast_pn = run_fast_qg_payload_point_normal(
        points, normals, ones_pn, queries, g_payload,
        epsilon, beta, True, stream,
    )

    results.append(validate_query_grad(
        fast_tri, scalar_tri_fast, "Tri D=1 ones (fast vs fast)",
        rms_tol=1e-3, max_abs_tol=1e-2, cos_tol=0.9999, verbose=verbose,
    ))
    results.append(validate_query_grad(
        fast_pn, scalar_pn_fast, "PN D=1 ones (fast vs fast)",
        rms_tol=1e-3, max_abs_tol=1e-2, cos_tol=0.9999, verbose=verbose,
    ))

    # ---- Brute path: payload vs scalar (both exact) ----
    scalar_tri_brute = run_brute_qg_scalar_triangle(
        triangles, queries, g_scalar, epsilon, True, stream
    )
    scalar_pn_brute = run_brute_qg_scalar_point_normal(
        points, normals, queries, g_scalar, epsilon, True, stream
    )
    brute_tri = run_brute_qg_payload_triangle(
        triangles, ones_tri, queries, g_payload, epsilon, True, stream,
    )
    brute_pn = run_brute_qg_payload_point_normal(
        points, normals, ones_pn, queries, g_payload, epsilon, True, stream,
    )

    # With v=1, g=1 the two brute kernels do *exactly* the same arithmetic
    # per (query, primitive) pair, so we can use a very tight tolerance.
    results.append(validate_query_grad(
        brute_tri, scalar_tri_brute, "Tri D=1 ones (brute vs brute)",
        rms_tol=1e-6, max_abs_tol=1e-5, cos_tol=0.9999, verbose=verbose,
    ))
    results.append(validate_query_grad(
        brute_pn, scalar_pn_brute, "PN D=1 ones (brute vs brute)",
        rms_tol=1e-6, max_abs_tol=1e-5, cos_tol=0.9999, verbose=verbose,
    ))

    # ---- Cross-check (informational, not part of pass/fail) ----
    # Fast payload vs brute payload reflects the ~1e-4 BVH approximation
    # error. It should be within the same envelope as the fast-scalar-vs-
    # brute-scalar error already validated by `scalar_coverage`.
    results.append(validate_query_grad(
        fast_tri, brute_tri, "Tri D=1 ones (fast vs brute, info)",
        rms_tol=1e-3, max_abs_tol=1e-2, cos_tol=0.9999, verbose=verbose,
    ))
    results.append(validate_query_grad(
        fast_pn, brute_pn, "PN D=1 ones (fast vs brute, info)",
        rms_tol=1e-3, max_abs_tol=1e-2, cos_tol=0.9999, verbose=verbose,
    ))


def test_kahan_consistency(
    vertices, indices, queries, epsilon, beta, results, verbose=True,
):
    """Kahan on/off agree within tolerance (payload D=8)."""
    print(_hdr("\nTest: Kahan on/off consistency (payload D=8)"))
    triangles = vertices[indices]
    D = 8
    M = len(queries)
    rng = np.random.default_rng(404)
    payload = rng.standard_normal((len(triangles), D)).astype(np.float32)
    g_np = rng.standard_normal((M, D)).astype(np.float32)
    stream = _current_stream()

    for tag, runner, use_engine in [
        ("fast", run_fast_qg_payload_triangle, True),
        ("brute", run_brute_qg_payload_triangle, False),
    ]:
        if use_engine:
            on = runner(triangles, payload, queries, g_np, epsilon, beta, True, stream)
            off = runner(triangles, payload, queries, g_np, epsilon, beta, False, stream)
        else:
            on = runner(triangles, payload, queries, g_np, epsilon, True, stream)
            off = runner(triangles, payload, queries, g_np, epsilon, False, stream)
        results.append(validate_query_grad(
            on, off, f"Kahan on vs off ({tag})", verbose=verbose,
        ))


def test_broken_meshes(
    vertices, indices, queries, epsilon, beta, use_kahan,
    results, verbose=True, seed=0,
):
    """Closed, sliced, dropped variants (payload D=8)."""
    print(_hdr("\nTest: broken meshes - payload QG (D=8)"))
    D = 8
    M = len(queries)
    rng = np.random.default_rng(505)
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

        f = run_fast_qg_payload_triangle(
            tri, payload, queries, g_np, epsilon, beta, use_kahan, stream,
        )
        b = run_brute_qg_payload_triangle(
            tri, payload, queries, g_np, epsilon, use_kahan, stream,
        )
        results.append(validate_query_grad(
            f, b, f"{name:<15} ({len(tri):>5} tri)", verbose=verbose,
        ))


def test_epsilon_sweep(
    vertices, indices, queries, epsilon_values, beta, use_kahan,
    results, verbose=True,
):
    """Fast vs brute payload QG for each epsilon."""
    print(_hdr("\nTest: payload QG epsilon sweep (D=8)"))
    triangles = vertices[indices]
    D = 8
    M = len(queries)
    rng = np.random.default_rng(606)
    payload = rng.standard_normal((len(triangles), D)).astype(np.float32)
    g_np = rng.standard_normal((M, D)).astype(np.float32)
    stream = _current_stream()

    for eps in epsilon_values:
        f = run_fast_qg_payload_triangle(
            triangles, payload, queries, g_np, eps, beta, use_kahan, stream,
        )
        b = run_brute_qg_payload_triangle(
            triangles, payload, queries, g_np, eps, use_kahan, stream,
        )
        results.append(validate_query_grad(f, b, f"eps={eps:g}", verbose=verbose))


def test_mesh_scale_sweep(
    query_count, epsilon, beta, use_kahan, results, verbose=True, seed=0,
):
    """Payload QG (D=8) across mesh sizes."""
    print(_hdr("\nTest: payload QG mesh scale sweep (D=8)"))
    D = 8
    rng = np.random.default_rng(707)
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

        f = run_fast_qg_payload_triangle(
            tri, payload, q, g_np, epsilon, beta, use_kahan, stream,
        )
        b = run_brute_qg_payload_triangle(
            tri, payload, q, g_np, epsilon, use_kahan, stream,
        )
        results.append(validate_query_grad(
            f, b, f"{name:<8} ({len(tri):>5} tri, {n_q:>4} q)", verbose=verbose,
        ))


def test_query_scale_sweep(
    vertices, indices, epsilon, beta, use_kahan,
    results, verbose=True, seed=0,
):
    """Payload QG (D=3) across query counts, hitting warp/block boundaries.

    This is the test that catches warp-tile-base overflow, mis-sized grid
    launches, and edge cases at warp (32) / block (128) boundaries.
    """
    print(_hdr("\nTest: payload QG query scale sweep (D=3)"))
    triangles = vertices[indices]
    D = 3
    rng = np.random.default_rng(808)
    payload = rng.standard_normal((len(triangles), D)).astype(np.float32)
    stream = _current_stream()

    query_counts = [1, 31, 32, 33, 64, 127, 128, 129, 512, 1024, 4096, 16384, 65536]

    for M in query_counts:
        q = generate_queries(vertices, M, seed)
        g_np = rng.standard_normal((M, D)).astype(np.float32)

        f = run_fast_qg_payload_triangle(
            triangles, payload, q, g_np, epsilon, beta, use_kahan, stream,
        )
        b = run_brute_qg_payload_triangle(
            triangles, payload, q, g_np, epsilon, use_kahan, stream,
        )
        results.append(validate_query_grad(
            f, b, f"M={M:>6}", verbose=verbose,
        ))


def test_timing(
    vertices, indices, queries, epsilon, beta,
    d_values, verbose=True,
):
    """Timing table: fast vs brute QG across D values."""
    print(_hdr("\n=== Timing: payload QG fast vs brute ==="))
    triangles = vertices[indices]
    M = len(queries)
    rng = np.random.default_rng(909)
    stream = _current_stream()

    t = _to_cuda(triangles)
    q = _to_cuda(queries)

    print(f"  Mesh: {len(triangles)} triangles, {M} queries, eps={epsilon:g}")
    print(
        f"  {'D':>4} | {'BF(ms)':>10} | {'fast(ms)':>10} | "
        f"{'build(ms)':>10} | {'speedup':>8}"
    )
    print("  " + "-" * 60)

    for D in d_values:
        payload = rng.standard_normal((len(triangles), D)).astype(np.float32)
        pl = _to_cuda(payload)
        g_np = rng.standard_normal((M, D)).astype(np.float32)
        g = _to_cuda(g_np)
        out = torch.empty((M, 3), device="cuda:0", dtype=torch.float32)

        def brute_fn():
            winder.brute_force_query_gradients_with_payload_triangle_soup(
                g, t, pl, q, out, float(epsilon), stream, True,
            )

        bf_ms = _time(brute_fn, warmup=1, iters=3)[1] * 1000

        def build_fn():
            return winder.PayloadField(t, pl, stream=stream)

        build_ms = _time(build_fn, warmup=1, iters=3)[1] * 1000
        eng = winder.PayloadField(t, pl, stream=stream)

        def compute_only():
            eng.compute_query_gradients(g, q, out, beta, epsilon, stream, True)

        compute_ms = _time(compute_only, warmup=1, iters=3)[1] * 1000
        e2e_ms = build_ms + compute_ms
        speedup = bf_ms / e2e_ms if e2e_ms > 0 else float("nan")

        print(
            f"  {D:>4} | {bf_ms:>10.2f} | {e2e_ms:>10.2f} | "
            f"{build_ms:>10.3f} | {speedup:>7.2f}x"
        )
        print(f"       (compute-only: {compute_ms:.2f} ms)")


# =============================================================================
# Main
# =============================================================================
def _summarize(results: list[QGMetrics]) -> bool:
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
        print(f"    - {r.label}  ({checks})  rms={r.rms_abs:.2e}  max={r.max_abs:.2e}")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Validate payload & scalar winding query gradients."
    )
    parser.add_argument("--obj_file", type=str, default=None)
    parser.add_argument(
        "--mesh_scale", type=str,
        choices=list(MESH_PRESETS.keys()) + ["all"], default="small",
    )
    parser.add_argument("--query_count", type=int, default=100000)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument(
        "--epsilon_sweep", type=str,
        default=",".join(str(e) for e in DEFAULT_EPSILON_SWEEP),
    )
    parser.add_argument(
        "--d_values", type=str,
        default=",".join(str(d) for d in DEFAULT_D_BUCKETS + DEFAULT_ARBITRARY_D),
    )
    parser.add_argument("--beta", type=float, default=-1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--use_kahan", dest="use_kahan", action="store_true", default=True
    )
    parser.add_argument("--no_kahan", dest="use_kahan", action="store_false")
    parser.add_argument(
        "--test", type=str, default="all",
        help="'all' or comma-separated: scalar_coverage, payload_d_coverage, "
        "payload_linearity, d1_scalar_match, single_leaf, kahan, broken, eps, "
        "mesh_scale, query_scale, timing",
    )
    args = parser.parse_args()
    set_seeds(args.seed)

    # ---------------------------------------------------------------------
    # Load geometry
    # ---------------------------------------------------------------------
    if args.obj_file:
        import igl
        v, _, _, i, _, _ = igl.readOBJ(args.obj_file)
        vertices = np.ascontiguousarray(v, dtype=np.float32)
        indices = np.ascontiguousarray(i, dtype=np.uint32)
        print(f"Loaded {args.obj_file}: {len(indices)} triangles, {len(vertices)} verts")
    else:
        n_theta, n_phi = MESH_PRESETS["medium" if args.mesh_scale == "all" else args.mesh_scale]
        vertices, indices = make_synthetic_sphere(n_theta, n_phi)
        print(f"Synthetic sphere: {len(indices)} triangles, {len(vertices)} verts")

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
            "scalar_coverage",
            "payload_d_coverage",
            "payload_linearity",
            "d1_scalar_match",
            "single_leaf",
            "kahan",
            "broken",
            "eps",
            "mesh_scale",
            "query_scale",
            "timing",
        }
    )

    results: list[QGMetrics] = []
    t0 = time()

    if "scalar_coverage" in tests:
        test_scalar_coverage(
            vertices, indices, points, normals, queries,
            args.epsilon, args.beta, results,
        )

    if "payload_d_coverage" in tests:
        test_payload_d_coverage(
            vertices, indices, points, normals, queries,
            args.epsilon, args.beta, args.use_kahan,
            d_values, results,
        )

    if "payload_linearity" in tests:
        test_payload_linearity(
            vertices, indices, queries,
            args.epsilon, args.beta, args.use_kahan, results,
        )

    if "d1_scalar_match" in tests:
        test_d1_scalar_match(
            vertices, indices, points, normals, queries,
            args.epsilon, args.beta, results,
        )

    if "single_leaf" in tests:
        test_single_leaf_scalar_qg(
            args.epsilon, args.beta, results, seed=args.seed,
        )
        test_single_leaf_payload_qg(
            args.epsilon, args.beta, args.use_kahan, results, seed=args.seed,
        )
        test_single_leaf_payload_forward(
            args.epsilon, args.beta, args.use_kahan, results, seed=args.seed,
        )
        test_single_leaf_d1_scalar_match(
            args.epsilon, args.beta, results, seed=args.seed,
        )

    if "kahan" in tests:
        test_kahan_consistency(
            vertices, indices, queries, args.epsilon, args.beta, results,
        )

    if "broken" in tests:
        test_broken_meshes(
            vertices, indices, queries,
            args.epsilon, args.beta, args.use_kahan, results,
            seed=args.seed,
        )

    if "eps" in tests:
        test_epsilon_sweep(
            vertices, indices, queries, eps_values,
            args.beta, args.use_kahan, results,
        )

    if "mesh_scale" in tests:
        test_mesh_scale_sweep(
            args.query_count, args.epsilon, args.beta, args.use_kahan,
            results, seed=args.seed,
        )

    if "query_scale" in tests:
        test_query_scale_sweep(
            vertices, indices, args.epsilon, args.beta, args.use_kahan,
            results, seed=args.seed,
        )

    if "timing" in tests:
        timing_d = [d for d in [1, 8, 32, 128] if d in d_values] or [1, 8]
        test_timing(
            vertices, indices, queries,
            args.epsilon, args.beta, timing_d,
        )

    elapsed = time() - t0
    ok = _summarize(results)
    print(f"\nElapsed: {elapsed:.2f}s")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
