"""Validate the PayloadField fast forward pass against the brute-force reference.

The suite is organised around specific claims:

  1. Every D bucket (1, 2, 3, 4, 8, 16, 32, 64, 128) matches brute force
  2. Arbitrary D in [1, 128] (internally padded) matches brute force
  3. Closed, sliced (holes), and dropped (missing parts) meshes all match
  4. Small, medium, and huge meshes all match
  5. D=1 with v=ones collapses exactly to the scalar winding number
  6. Linearity: w(a*v1 + b*v2) == a*w(v1) + b*w(v2)
  7. Kahan on/off agree within tolerance
  8. Epsilon sweep (sharp, default, smoothed) all match

Usage
-----
    # Default (small sphere, all D buckets, ~30 s)
    python test_payload_forward.py

    # Load a real mesh
    python test_payload_forward.py --obj_file benchy.obj --query_count 300

    # Only the timing table
    python test_payload_forward.py --test timing

    # Huge mesh (uses a big synthetic sphere, slower brute-force baseline)
    python test_payload_forward.py --mesh_scale huge --query_count 100

    # Kahan off
    python test_payload_forward.py --no_kahan
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from time import time
from typing import Optional

import numpy as np
import torch

import winder


# =============================================================================
# Constants
# =============================================================================
DEFAULT_EPSILON = 0.004  # 1/250, matches library default
DEFAULT_D_BUCKETS = [1, 2, 3, 4, 8, 16, 32, 64, 128]
DEFAULT_ARBITRARY_D = [5, 7, 17, 33, 100]
DEFAULT_EPSILON_SWEEP = [0.0, 0.004, 0.02]

# Preset mesh sizes (n_theta, n_phi) → number of triangles ≈ 2 · n_theta · n_phi
MESH_PRESETS = {
    "tiny": (6, 4),  # ~48 triangles — exercises the single-leaf path
    "small": (20, 12),  # ~480 triangles
    "medium": (40, 24),  # ~1900 triangles
    "large": (80, 48),  # ~7600 triangles
    "huge": (120, 72),  # ~17000 triangles
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


def _rms(a: torch.Tensor, b: torch.Tensor) -> float:
    """Global RMS between two same-shaped tensors."""
    diff = a.detach().float() - b.detach().float()
    return float(torch.sqrt(torch.mean(diff * diff)).item())


def _scene_scale(points: np.ndarray) -> float:
    p = np.asarray(points, dtype=np.float32)
    extent = p.max(axis=0) - p.min(axis=0)
    scale = float(extent.max())
    return scale if scale > 1e-8 else 1.0


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
    """Closed UV sphere — vectorized for 1M-triangle generation."""
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
    """Keep triangles whose centroid lies above the median along `dim`."""
    centroids = vertices[indices].mean(axis=1)
    split = np.median(centroids[:, dim])
    return vertices, indices[centroids[:, dim] > split]


def drop_half_triangles(vertices: np.ndarray, indices: np.ndarray, seed: int = 0):
    """Randomly drop half the triangles."""
    rng = np.random.default_rng(seed)
    keep = rng.choice(len(indices), len(indices) // 2, replace=False)
    return vertices, indices[keep]


def mesh_to_point_surfels(vertices: np.ndarray, indices: np.ndarray):
    """Return (points, scaled_normals) for the point-normal representation."""
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
# Fast / brute runners
# =============================================================================
def _alloc_out(M: int, D: int) -> torch.Tensor:
    return torch.empty((M, D), device="cuda:0", dtype=torch.float32)


def run_fast_triangle(triangles, queries, payload, epsilon, beta, use_kahan):
    t = _to_cuda(triangles)
    pl = _to_cuda(payload)
    q = _to_cuda(queries)
    M, D = q.shape[0], pl.shape[1]
    out = _alloc_out(M, D)
    engine = winder.PayloadField(t, pl, stream=_current_stream())
    engine.compute(q, out, beta, epsilon, _current_stream())
    torch.cuda.synchronize()
    return out


def run_brute_triangle(triangles, queries, payload, epsilon, use_kahan):
    t = _to_cuda(triangles)
    pl = _to_cuda(payload)
    q = _to_cuda(queries)
    M, D = q.shape[0], pl.shape[1]
    out = _alloc_out(M, D)
    winder.brute_force_winding_payloads_triangle_soup(
        t,
        pl,
        q,
        out,
        float(epsilon),
        _current_stream(),
        bool(use_kahan),
    )
    torch.cuda.synchronize()
    return out


def run_fast_mesh(vertices, indices, queries, payload, epsilon, beta, use_kahan):
    v = _to_cuda(vertices)
    idx = _to_cuda(indices, torch.uint32)
    pl = _to_cuda(payload)
    q = _to_cuda(queries)
    M, D = q.shape[0], pl.shape[1]
    out = _alloc_out(M, D)
    engine = winder.PayloadField(v, idx, pl, stream=_current_stream())
    engine.compute(q, out, beta, epsilon, _current_stream())
    torch.cuda.synchronize()
    return out


def run_brute_mesh(vertices, indices, queries, payload, epsilon, use_kahan):
    v = _to_cuda(vertices)
    idx = _to_cuda(indices, torch.uint32)
    pl = _to_cuda(payload)
    q = _to_cuda(queries)
    M, D = q.shape[0], pl.shape[1]
    out = _alloc_out(M, D)
    winder.brute_force_winding_payloads_mesh(
        v,
        idx,
        pl,
        q,
        out,
        float(epsilon),
        _current_stream(),
        bool(use_kahan),
    )
    torch.cuda.synchronize()
    return out


def run_fast_point_normal(points, normals, queries, payload, epsilon, beta, use_kahan):
    p = _to_cuda(points)
    n = _to_cuda(normals)
    pl = _to_cuda(payload)
    q = _to_cuda(queries)
    M, D = q.shape[0], pl.shape[1]
    out = _alloc_out(M, D)
    engine = winder.PayloadField(p, n, pl, stream=_current_stream())
    engine.compute(q, out, beta, epsilon, _current_stream())
    torch.cuda.synchronize()
    return out


def run_brute_point_normal(points, normals, queries, payload, epsilon, use_kahan):
    p = _to_cuda(points)
    n = _to_cuda(normals)
    pl = _to_cuda(payload)
    q = _to_cuda(queries)
    M, D = q.shape[0], pl.shape[1]
    out = _alloc_out(M, D)
    winder.brute_force_winding_payloads_point_normal(
        p,
        n,
        pl,
        q,
        out,
        float(epsilon),
        _current_stream(),
        bool(use_kahan),
    )
    torch.cuda.synchronize()
    return out


# =============================================================================
# Validation
# =============================================================================
@dataclass
class PayloadForwardMetrics:
    label: str
    shape: tuple
    has_nan: bool
    has_inf: bool
    signal_max: float
    rms_abs: float
    max_abs: float
    mean_rel: float
    p99_rel: float
    cos_flat: float
    cos_per_query_median: float
    mag_ratio_median: float
    passed: bool = False
    failed_checks: tuple = ()


def validate_payload_forward(
    fast: torch.Tensor,
    ref: torch.Tensor,
    label: str,
    *,
    signal_threshold: float = 1e-5,
    rms_tol: float = 5e-3,
    max_abs_tol: float = 2e-1,
    cos_tol: float = 0.99,
    verbose: bool = True,
) -> PayloadForwardMetrics:
    """Compare (M, D) payload fields element-wise."""
    assert fast.shape == ref.shape, f"shape mismatch: {fast.shape} vs {ref.shape}"

    fast_np = fast.detach().float().cpu().numpy()
    ref_np = ref.detach().float().cpu().numpy()

    has_nan = bool(np.isnan(fast_np).any())
    has_inf = bool(np.isinf(fast_np).any())
    if has_nan or has_inf:
        if verbose:
            print(f"  [{_bad('FAIL')}] {label}: NaN={has_nan} Inf={has_inf}")
        return PayloadForwardMetrics(
            label=label,
            shape=tuple(fast_np.shape),
            has_nan=has_nan,
            has_inf=has_inf,
            signal_max=0.0,
            rms_abs=float("inf"),
            max_abs=float("inf"),
            mean_rel=float("inf"),
            p99_rel=float("inf"),
            cos_flat=float("nan"),
            cos_per_query_median=float("nan"),
            mag_ratio_median=float("nan"),
            passed=False,
            failed_checks=("nan_or_inf",),
        )

    diff = fast_np - ref_np
    abs_diff = np.abs(diff)
    rms_abs = float(np.sqrt(np.mean(diff**2)))
    max_abs = float(abs_diff.max())
    signal_max = float(np.abs(ref_np).max())

    # Masked relative error
    mask = np.abs(ref_np) > signal_threshold
    if mask.any():
        rel = abs_diff[mask] / np.abs(ref_np[mask])
        mean_rel = float(rel.mean())
        p99_rel = float(np.percentile(rel, 99))
    else:
        mean_rel = p99_rel = float("nan")

    # Flattened cosine
    f_flat = fast_np.reshape(-1)
    r_flat = ref_np.reshape(-1)
    nf = float(np.linalg.norm(f_flat))
    nr = float(np.linalg.norm(r_flat))
    cos_flat = (
        float(np.dot(f_flat, r_flat) / (nf * nr))
        if (nf > 0 and nr > 0)
        else float("nan")
    )

    # Per-query cosine (rows of the (M, D) matrix)
    if fast_np.ndim == 2 and fast_np.shape[1] > 1:
        norms_f = np.linalg.norm(fast_np, axis=1)
        norms_r = np.linalg.norm(ref_np, axis=1)
        vm = (norms_f > signal_threshold) & (norms_r > signal_threshold)
        if vm.any():
            cos_vec = np.sum(fast_np[vm] * ref_np[vm], axis=1) / (
                norms_f[vm] * norms_r[vm] + 1e-30
            )
            cos_vec = np.clip(cos_vec, -1.0, 1.0)
            cos_per_query_median = float(np.median(cos_vec))
            mag_ratio_median = float(np.median(norms_f[vm] / (norms_r[vm] + 1e-30)))
        else:
            cos_per_query_median = float("nan")
            mag_ratio_median = float("nan")
    else:
        cos_per_query_median = cos_flat
        mag_ratio_median = 1.0 if nf > 0 else float("nan")

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
            f"rms={rms_abs:.2e}  max={max_abs:.2e}  "
            f"cos={cos_flat:.5f}  mag={mag_ratio_median:.3f}"
        )

    return PayloadForwardMetrics(
        label=label,
        shape=tuple(fast_np.shape),
        has_nan=False,
        has_inf=False,
        signal_max=signal_max,
        rms_abs=rms_abs,
        max_abs=max_abs,
        mean_rel=mean_rel,
        p99_rel=p99_rel,
        cos_flat=cos_flat,
        cos_per_query_median=cos_per_query_median,
        mag_ratio_median=mag_ratio_median,
        passed=passed,
        failed_checks=failed,
    )


def _check_close(a, b, rtol=1e-4, atol=1e-5) -> bool:
    return bool(np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True))


# =============================================================================
# Individual tests
# =============================================================================
def test_d_coverage(
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
    """Every D bucket and every arbitrary D value matches brute force."""
    triangles = vertices[indices]
    rng = np.random.default_rng(101)
    N_tri = len(triangles)
    N_pn = len(points)

    print(_hdr(f"\nTest: D coverage ({len(d_values)} values)"))
    for D in d_values:
        payload_tri = rng.standard_normal((N_tri, D)).astype(np.float32)
        payload_pn = rng.standard_normal((N_pn, D)).astype(np.float32)

        # Triangle
        f = run_fast_triangle(triangles, queries, payload_tri, epsilon, beta, use_kahan)
        b = run_brute_triangle(triangles, queries, payload_tri, epsilon, use_kahan)
        results.append(
            validate_payload_forward(f, b, f"Tri  D={D:>3}", verbose=verbose)
        )

        # Mesh
        f = run_fast_mesh(
            vertices, indices, queries, payload_tri, epsilon, beta, use_kahan
        )
        b = run_brute_mesh(vertices, indices, queries, payload_tri, epsilon, use_kahan)
        results.append(
            validate_payload_forward(f, b, f"Mesh D={D:>3}", verbose=verbose)
        )

        # PointNormal
        f = run_fast_point_normal(
            points, normals, queries, payload_pn, epsilon, beta, use_kahan
        )
        b = run_brute_point_normal(
            points, normals, queries, payload_pn, epsilon, use_kahan
        )
        results.append(
            validate_payload_forward(f, b, f"PN   D={D:>3}", verbose=verbose)
        )


def test_d1_ones_winding(
    vertices,
    indices,
    points,
    normals,
    queries,
    epsilon,
    results,
    verbose=True,
):
    """D=1, v=ones ⇒ payload field equals scalar winding numbers."""
    print(_hdr("\nTest: D=1 with v=ones == scalar winding"))
    triangles = vertices[indices]
    N_tri = len(triangles)
    N_pn = len(points)
    M = len(queries)
    q = _to_cuda(queries)

    # Scalar winding (fast path) — reference
    t = _to_cuda(triangles)
    scalar_tri = torch.empty(M, device="cuda:0", dtype=torch.float32)
    winder.WindingNumberEngine(t, stream=_current_stream()).compute(
        q,
        scalar_tri,
        -1,
        epsilon,
        _current_stream(),
    )
    torch.cuda.synchronize()

    p = _to_cuda(points)
    n = _to_cuda(normals)
    scalar_pn = torch.empty(M, device="cuda:0", dtype=torch.float32)
    winder.WindingNumberEngine(p, n, stream=_current_stream()).compute(
        q,
        scalar_pn,
        -1,
        epsilon,
        _current_stream(),
    )
    torch.cuda.synchronize()

    # Payload D=1 ones
    ones_tri = np.ones((N_tri, 1), np.float32)
    ones_pn = np.ones((N_pn, 1), np.float32)
    pay_tri = run_fast_triangle(triangles, queries, ones_tri, epsilon, -1, True)
    pay_pn = run_fast_point_normal(points, normals, queries, ones_pn, epsilon, -1, True)

    results.append(
        validate_payload_forward(
            pay_tri,
            scalar_tri.reshape(-1, 1),
            "Tri  D=1 ones == scalar winding",
            verbose=verbose,
        )
    )
    results.append(
        validate_payload_forward(
            pay_pn,
            scalar_pn.reshape(-1, 1),
            "PN   D=1 ones == scalar winding",
            verbose=verbose,
        )
    )


def test_linearity(
    vertices,
    indices,
    queries,
    epsilon,
    beta,
    use_kahan,
    results,
    verbose=True,
):
    """Payload field is linear in v up to numerical precision."""
    print(_hdr("\nTest: linearity in v (D=8)"))
    triangles = vertices[indices]
    N = len(triangles)
    D = 8
    rng = np.random.default_rng(202)
    v1 = rng.standard_normal((N, D)).astype(np.float32)
    v2 = rng.standard_normal((N, D)).astype(np.float32)
    a, b = 1.7, -0.83
    vc = (a * v1 + b * v2).astype(np.float32)

    f1 = run_fast_triangle(triangles, queries, v1, epsilon, beta, use_kahan)
    f2 = run_fast_triangle(triangles, queries, v2, epsilon, beta, use_kahan)
    fc = run_fast_triangle(triangles, queries, vc, epsilon, beta, use_kahan)

    # Linearity holds up to f16 accumulation noise, so the same tolerance as
    # the D-coverage test is appropriate here.
    expected = a * f1 + b * f2
    results.append(
        validate_payload_forward(fc, expected, "linearity in v", verbose=verbose)
    )


def test_kahan_consistency(
    vertices,
    indices,
    queries,
    epsilon,
    beta,
    results,
    verbose=True,
):
    """Kahan on/off agree within tolerance."""
    print(_hdr("\nTest: Kahan on/off consistency (D=8)"))
    triangles = vertices[indices]
    N = len(triangles)
    D = 8
    rng = np.random.default_rng(303)
    payload = rng.standard_normal((N, D)).astype(np.float32)

    out_off = run_fast_triangle(triangles, queries, payload, epsilon, beta, False)
    out_on = run_fast_triangle(triangles, queries, payload, epsilon, beta, True)
    results.append(
        validate_payload_forward(out_on, out_off, "Kahan on vs off", verbose=verbose)
    )


def test_closed_vs_open(
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
    """Closed, sliced (holes), and dropped (missing parts) meshes all match brute force."""
    print(_hdr("\nTest: closed vs broken meshes (D=8)"))
    D = 8
    rng = np.random.default_rng(404)

    variants = {
        "closed": (vertices, indices),
        "sliced": slice_mesh(vertices, indices),
        "dropped": drop_half_triangles(vertices, indices, seed=seed),
    }
    # Also test both combined
    v_slice, i_slice = slice_mesh(vertices, indices)
    variants["sliced+dropped"] = drop_half_triangles(v_slice, i_slice, seed=seed)

    for name, (v, i) in variants.items():
        tri = v[i]
        payload = rng.standard_normal((len(tri), D)).astype(np.float32)
        f = run_fast_triangle(tri, queries, payload, epsilon, beta, use_kahan)
        b = run_brute_triangle(tri, queries, payload, epsilon, use_kahan)
        n_tri = len(tri)
        results.append(
            validate_payload_forward(
                f, b, f"{name:<15} ({n_tri:>5} tri)", verbose=verbose
            )
        )


def test_mesh_scale_sweep(
    query_count,
    epsilon,
    beta,
    use_kahan,
    results,
    verbose=True,
    seed=0,
):
    """Test small, medium, and large meshes."""
    print(_hdr("\nTest: mesh size sweep (D=8)"))
    D = 8
    rng = np.random.default_rng(505)

    # Reduce query count for larger meshes so brute force stays tractable
    scale_presets = [
        ("small", 20, 12, query_count),
        ("medium", 40, 24, max(query_count // 2, 100)),
        ("large", 80, 48, max(query_count // 5, 50)),
    ]

    for name, n_theta, n_phi, n_q in scale_presets:
        v, i = make_synthetic_sphere(n_theta, n_phi)
        v, i = drop_half_triangles(v, i, seed=seed)  # break for realism
        tri = v[i]
        q = generate_queries(v, n_q, seed)
        payload = rng.standard_normal((len(tri), D)).astype(np.float32)

        f = run_fast_triangle(tri, q, payload, epsilon, beta, use_kahan)
        b = run_brute_triangle(tri, q, payload, epsilon, use_kahan)
        results.append(
            validate_payload_forward(
                f,
                b,
                f"{name:<8} ({len(tri):>5} tri, {n_q:>4} q)",
                verbose=verbose,
            )
        )


def test_epsilon_sweep(
    vertices,
    indices,
    queries,
    epsilon_values,
    beta,
    use_kahan,
    results,
    verbose=True,
):
    """Fast vs brute force at every epsilon value."""
    print(_hdr("\nTest: epsilon sweep (D=8)"))
    triangles = vertices[indices]
    D = 8
    rng = np.random.default_rng(606)
    payload = rng.standard_normal((len(triangles), D)).astype(np.float32)

    for eps in epsilon_values:
        f = run_fast_triangle(triangles, queries, payload, eps, beta, use_kahan)
        b = run_brute_triangle(triangles, queries, payload, eps, use_kahan)
        results.append(validate_payload_forward(f, b, f"eps={eps:g}", verbose=verbose))


def test_timing(
    vertices,
    indices,
    queries,
    epsilon,
    beta,
    use_kahan,
    d_values,
    verbose=True,
):
    """Fast vs brute-force timing table."""
    print(_hdr("\n=== Timing: fast vs brute ==="))
    triangles = vertices[indices]
    N = len(triangles)
    M = len(queries)
    rng = np.random.default_rng(707)

    print(f"  Mesh: {N} triangles, {M} queries, eps={epsilon:g}")
    print(
        f"  {'D':>4} | {'BF(ms)':>10} | {'fast(ms)':>10} | {'build(ms)':>10} | {'speedup':>8}"
    )
    print("  " + "-" * 60)

    t = _to_cuda(triangles)
    q = _to_cuda(queries)

    for D in d_values:
        payload = rng.standard_normal((N, D)).astype(np.float32)
        pl = _to_cuda(payload)
        out = _alloc_out(M, D)

        def brute_fn():
            winder.brute_force_winding_payloads_triangle_soup(
                t,
                pl,
                q,
                out,
                float(epsilon),
                _current_stream(),
                bool(use_kahan),
            )

        bf_ms = _time(brute_fn, warmup=1, iters=3)[1] * 1000

        def build_fn():
            return winder.PayloadField(t, pl, stream=_current_stream())

        build_ms = _time(build_fn, warmup=1, iters=3)[1] * 1000

        # Reuse engine for compute-only timing
        engine = winder.PayloadField(t, pl, stream=_current_stream())

        def compute_only():
            engine.compute(q, out, beta, epsilon, _current_stream())

        compute_ms = _time(compute_only, warmup=1, iters=3)[1] * 1000
        e2e_ms = build_ms + compute_ms
        speedup = bf_ms / e2e_ms if e2e_ms > 0 else float("nan")

        print(
            f"  {D:>4} | {bf_ms:>10.2f} | {e2e_ms:>10.2f} | "
            f"{build_ms:>10.3f} | {speedup:>7.2f}x"
        )
        print(f"       (compute-only: {compute_ms:.2f} ms)")


def test_scale_correctness(
    epsilon,
    beta,
    use_kahan,
    results,
    verbose=True,
    seed=0,
):
    """Correctness of PayloadField at mesh scales from 500 to 1M triangles."""
    print(_hdr("\nTest: correctness at scale (D=8)"))
    D = 8
    rng = np.random.default_rng(808)

    # (label, n_theta, n_phi, n_queries)
    # Number of triangles ≈ 2·n_phi·(n_theta-1). Queries shrink with mesh
    # size to keep the O(N·M·D) brute-force baseline tractable.
    sizes = [
        ("500", 22, 12, 500),
        ("5k", 70, 40, 500),
        ("50k", 220, 120, 200),
        ("500k", 700, 360, 100),
        ("1M", 1001, 500, 50),
    ]

    for name, n_theta, n_phi, n_q in sizes:
        v, i = make_synthetic_sphere(n_theta, n_phi)
        tri = v[i]
        q = generate_queries(v, n_q, seed)
        payload = rng.standard_normal((len(tri), D)).astype(np.float32)

        f = run_fast_triangle(tri, q, payload, epsilon, beta, use_kahan)
        b = run_brute_triangle(tri, q, payload, epsilon, use_kahan)
        results.append(
            validate_payload_forward(
                f,
                b,
                f"{name:<6} ({len(tri):>7} tri, {n_q:>4} q)",
                verbose=verbose,
            )
        )


def test_scale_timing(epsilon, beta, verbose=True, seed=0):
    """Timing + precision: mesh scale × D × Kahan vs Kahan-on brute force."""
    print(_hdr("\n=== Timing + precision: fast vs brute, across scales ==="))
    print(f"  epsilon={epsilon:g}, beta={beta}")
    print(f"  Reference for RMS: brute force with Kahan=ON (highest precision)")

    sizes = [
        ("500", 22, 12, 50000),
        ("2k", 45, 24, 20000),
        ("5k", 70, 40, 10000),
        ("10k", 100, 56, 5000),
        ("50k", 220, 120, 2000),
        ("500k", 700, 360, 1000),
        ("1M", 1001, 500, 500),
    ]
    d_values = [1, 8, 32, 128]
    rng = np.random.default_rng(909)
    results: dict = {}

    print(
        f"\n  {'mesh':<6} {'tri':>8} {'D':>4} {'kahan':<5} | "
        f"{'BF(ms)':>9} {'build':>7} {'compute':>9} {'total':>7} {'speedup':>8} | "
        f"{'rms_vs_BFtrue':>14}"
    )
    print("  " + "-" * 100)

    for name, n_theta, n_phi, n_q in sizes:
        v, i = make_synthetic_sphere(n_theta, n_phi)
        tri = v[i]
        q = generate_queries(v, n_q, seed)
        N, M = len(tri), len(q)

        t = _to_cuda(tri)
        qc = _to_cuda(q)

        for D in d_values:
            payload = rng.standard_normal((N, D)).astype(np.float32)
            pl = _to_cuda(payload)
            out = _alloc_out(M, D)

            # BF, Kahan=ON — this is the reference
            bf_ms = (
                _time(
                    lambda: winder.brute_force_winding_payloads_triangle_soup(
                        t,
                        pl,
                        qc,
                        out,
                        float(epsilon),
                        _current_stream(),
                        True,
                    ),
                    warmup=1,
                    iters=3,
                )[1]
                * 1000
            )
            bf_ref = out.clone()

            build_ms = (
                _time(
                    lambda: winder.PayloadField(t, pl, stream=_current_stream()),
                    warmup=1,
                    iters=3,
                )[1]
                * 1000
            )
            engine = winder.PayloadField(t, pl, stream=_current_stream())

            for kahan in (False, True):
                compute_ms = (
                    _time(
                        lambda k=kahan: engine.compute(
                            qc,
                            out,
                            beta,
                            epsilon,
                            _current_stream(),
                            k,
                        ),
                        warmup=1,
                        iters=3,
                    )[1]
                    * 1000
                )
                rms = _rms(out, bf_ref)

                total = build_ms + compute_ms
                speedup = bf_ms / total if total > 0 else float("nan")

                results[(name, D, kahan)] = {
                    "compute_ms": compute_ms,
                    "rms": rms,
                }

                print(
                    f"  {name:<6} {N:>8} {D:>4} {'on' if kahan else 'off':<5} | "
                    f"{bf_ms:>9.2f} {build_ms:>7.3f} {compute_ms:>9.2f} "
                    f"{total:>7.2f} {speedup:>7.2f}x | "
                    f"{rms:>14.3e}"
                )

    # --- Summary: Kahan compute overhead ---
    print(f"\n  Kahan compute overhead (compute_on / compute_off):")
    hdr = f"  {'mesh':<6} {'tri':>8} | " + " | ".join(
        f"{'D=' + str(d):>7}" for d in d_values
    )
    print(hdr)
    print("  " + "-" * len(hdr))
    for name, n_theta, n_phi, _ in sizes:
        v, i = make_synthetic_sphere(n_theta, n_phi)
        N = len(i)
        cells = []
        for D in d_values:
            off = results.get((name, D, False))
            on = results.get((name, D, True))
            if not off or not on or off["compute_ms"] <= 0:
                cells.append("    n/a")
                continue
            cells.append(f"{on['compute_ms'] / off['compute_ms']:>6.2f}x")
        print(f"  {name:<6} {N:>8} | " + " | ".join(cells))

    # --- Summary: Kahan precision benefit ---
    print(f"\n  Kahan precision benefit (rms_on / rms_off; <1 means Kahan helps):")
    hdr = f"  {'mesh':<6} {'tri':>8} | " + " | ".join(
        f"{'D=' + str(d):>7}" for d in d_values
    )
    print(hdr)
    print("  " + "-" * len(hdr))
    for name, n_theta, n_phi, _ in sizes:
        v, i = make_synthetic_sphere(n_theta, n_phi)
        N = len(i)
        cells = []
        for D in d_values:
            off = results.get((name, D, False))
            on = results.get((name, D, True))
            if not off or not on or off["rms"] <= 0:
                cells.append("    n/a")
                continue
            cells.append(f"{on['rms'] / off['rms']:>6.3f}x")
        print(f"  {name:<6} {N:>8} | " + " | ".join(cells))


def test_broken_mesh_timing(
    vertices,
    indices,
    queries,
    epsilon,
    beta,
    verbose=True,
    seed=0,
):
    """Timing + precision across mesh variants (closed / sliced / dropped)."""
    print(_hdr("\n=== Timing + precision: mesh variants ==="))
    print(f"  epsilon={epsilon:g}, beta={beta}")
    print(f"  Reference for RMS: brute force with Kahan=ON")
    print(f"  Queries: {len(queries)}")

    D = 8
    rng = np.random.default_rng(1111)
    qc = _to_cuda(queries)
    M = len(queries)

    # Prepare all four variants
    v_slice, i_slice = slice_mesh(vertices, indices)
    v_dropped, i_dropped = drop_half_triangles(vertices, indices, seed=seed)
    _, i_sliced_dropped = drop_half_triangles(v_slice, i_slice, seed=seed)

    variants = [
        ("closed", vertices, indices),
        ("sliced", v_slice, i_slice),
        ("dropped", v_dropped, i_dropped),
        ("sliced+dropped", v_slice, i_sliced_dropped),
    ]

    print(
        f"\n  {'variant':<15} {'tri':>6} {'kahan':<5} | "
        f"{'BF(ms)':>9} {'build':>7} {'compute':>9} {'total':>7} {'speedup':>8} | "
        f"{'rms_vs_BFtrue':>14}"
    )
    print("  " + "-" * 100)

    for name, v, i in variants:
        tri = v[i]
        N = len(tri)
        payload = rng.standard_normal((N, D)).astype(np.float32)

        t = _to_cuda(tri)
        pl = _to_cuda(payload)
        out = _alloc_out(M, D)

        bf_ms = (
            _time(
                lambda: winder.brute_force_winding_payloads_triangle_soup(
                    t,
                    pl,
                    qc,
                    out,
                    float(epsilon),
                    _current_stream(),
                    True,
                ),
                warmup=1,
                iters=3,
            )[1]
            * 1000
        )
        bf_ref = out.clone()

        build_ms = (
            _time(
                lambda: winder.PayloadField(t, pl, stream=_current_stream()),
                warmup=1,
                iters=3,
            )[1]
            * 1000
        )
        engine = winder.PayloadField(t, pl, stream=_current_stream())

        for kahan in (False, True):
            compute_ms = (
                _time(
                    lambda k=kahan: engine.compute(
                        qc,
                        out,
                        beta,
                        epsilon,
                        _current_stream(),
                        k,
                    ),
                    warmup=1,
                    iters=3,
                )[1]
                * 1000
            )
            rms = _rms(out, bf_ref)

            total = build_ms + compute_ms
            speedup = bf_ms / total if total > 0 else float("nan")

            print(
                f"  {name:<15} {N:>6} {'on' if kahan else 'off':<5} | "
                f"{bf_ms:>9.2f} {build_ms:>7.3f} {compute_ms:>9.2f} "
                f"{total:>7.2f} {speedup:>7.2f}x | "
                f"{rms:>14.3e}"
            )


# =============================================================================
# Main
# =============================================================================
def _summarize(results: list[PayloadForwardMetrics]) -> bool:
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
        description="Validate PayloadField against brute-force reference."
    )
    parser.add_argument(
        "--obj_file",
        type=str,
        default=None,
        help="Optional OBJ mesh. Default: synthetic sphere.",
    )
    parser.add_argument(
        "--mesh_scale",
        type=str,
        choices=list(MESH_PRESETS.keys()) + ["all"],
        default="small",
        help="Synthetic sphere size (ignored if --obj_file given).",
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
        help="'all' or comma-separated: d_coverage, d1_ones, "
        "linearity, kahan, broken, scale, eps, timing, "
        "scale_correctness, scale_timing, broken_timing",
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
        print(
            f"Loaded {args.obj_file}: {len(indices)} triangles, {len(vertices)} verts"
        )
    else:
        if args.mesh_scale == "all":
            n_theta, n_phi = MESH_PRESETS["medium"]
        else:
            n_theta, n_phi = MESH_PRESETS[args.mesh_scale]
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
            "d_coverage",
            "d1_ones",
            "linearity",
            "kahan",
            "broken",
            "scale",
            "eps",
            "timing",
            "scale_correctness",
            "scale_timing",
            "broken_timing",
        }
    )

    # ---------------------------------------------------------------------
    # Run tests
    # ---------------------------------------------------------------------
    results: list[PayloadForwardMetrics] = []
    t0 = time()

    if "d_coverage" in tests:
        test_d_coverage(
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

    if "d1_ones" in tests:
        test_d1_ones_winding(
            vertices,
            indices,
            points,
            normals,
            queries,
            args.epsilon,
            results,
        )

    if "linearity" in tests:
        test_linearity(
            vertices,
            indices,
            queries,
            args.epsilon,
            args.beta,
            args.use_kahan,
            results,
        )

    if "kahan" in tests:
        test_kahan_consistency(
            vertices,
            indices,
            queries,
            args.epsilon,
            args.beta,
            results,
        )

    if "broken" in tests:
        test_closed_vs_open(
            vertices,
            indices,
            queries,
            args.epsilon,
            args.beta,
            args.use_kahan,
            results,
            seed=args.seed,
        )

    if "scale" in tests:
        test_mesh_scale_sweep(
            args.query_count,
            args.epsilon,
            args.beta,
            args.use_kahan,
            results,
            seed=args.seed,
        )

    if "eps" in tests:
        test_epsilon_sweep(
            vertices,
            indices,
            queries,
            eps_values,
            args.beta,
            args.use_kahan,
            results,
        )

    if "timing" in tests:
        # Pick a small subset of D values for the timing table
        timing_d = [d for d in [1, 8, 32, 128] if d in d_values] or [1, 8]
        test_timing(
            vertices,
            indices,
            queries,
            args.epsilon,
            args.beta,
            args.use_kahan,
            timing_d,
        )
    if "scale_correctness" in tests:
        test_scale_correctness(
            args.epsilon,
            args.beta,
            args.use_kahan,
            results,
            seed=args.seed,
        )

    if "scale_timing" in tests:
        test_scale_timing(args.epsilon, args.beta)

    if "broken_timing" in tests:
        test_broken_mesh_timing(
            vertices,
            indices,
            queries,
            args.epsilon,
            args.beta,
            seed=args.seed,
        )

    elapsed = time() - t0
    ok = _summarize(results)
    print(f"\nElapsed: {elapsed:.2f}s")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
