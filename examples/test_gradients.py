"""Validate the fast (Barnes-Hut) winding-number gradient engine against brute force.

Usage examples:
    # Basic validation, all geometry types
    python validate_gradients.py --obj_file mesh.obj

    # Reproducible multi-trial run with seed
    python validate_gradients.py --obj_file mesh.obj --seed 42 --num_trials 5

    # Beta sweep: verify monotone and bounded approximation error
    python validate_gradients.py --obj_file mesh.obj \
        --beta_sweep "-1,0.1,0.3,0.5,1.0,2.0" --num_trials 3

    # Only the beta sweep, no PointNormal/Triangle tests
    python validate_gradients.py --obj_file mesh.obj \
        --beta_sweep "-1,0.5,1.0" --beta_sweep_only
"""

import argparse
import csv
import math
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from time import time
from typing import Optional

import igl
import matplotlib.pyplot as plt
import numpy as np
import torch

import winder


def cuda_timer(fn, warmup: int = 3, iters: int = 10) -> float:
    """Run `fn` `warmup` times, then time `iters` runs with CUDA events.

    Returns the median wall-clock time in milliseconds as measured by the
    GPU. Uses median rather than mean so a single slow run (e.g. another
    process landing on the GPU) does not corrupt the result.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times_ms = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    return float(np.median(times_ms))


# =========================================================================
# Geometry helpers
# =========================================================================


def mesh_to_point_surfels(vertices: np.ndarray, indices: np.ndarray):
    """Convert a triangle mesh into (points, unit_normals, areas)."""
    v0 = vertices[indices[:, 0]]
    v1 = vertices[indices[:, 1]]
    v2 = vertices[indices[:, 2]]
    points = (v0 + v1 + v2) / 3.0
    e1 = v1 - v0
    e2 = v2 - v0
    cross = np.cross(e1, e2)
    magnitudes = np.linalg.norm(cross, axis=-1, keepdims=True)
    areas = (magnitudes / 2.0).flatten()
    safe = np.where(magnitudes == 0, 1e-8, magnitudes)
    normals = cross / safe
    return (
        points.astype(np.float32),
        normals.astype(np.float32),
        areas.astype(np.float32),
    )


def generate_queries(
    vertices: np.ndarray, mode: str, num_queries: int, seed: int
) -> np.ndarray:
    """Deterministic query generation."""
    rng = np.random.default_rng(seed)
    min_box = vertices.min(axis=0)
    max_box = vertices.max(axis=0)
    diag = float(np.linalg.norm(max_box - min_box))
    min_box = min_box - 0.5 * diag
    max_box = max_box + 0.5 * diag

    if mode == "grid":
        side = int(np.ceil(num_queries ** (1.0 / 3.0)))
        x = np.linspace(min_box[0], max_box[0], side)
        y = np.linspace(min_box[1], max_box[1], side)
        z = np.linspace(min_box[2], max_box[2], side)
        gx, gy, gz = np.meshgrid(x, y, z, indexing="ij")
        q = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)
        return q[:num_queries].astype(np.float32)

    return rng.uniform(min_box, max_box, size=(num_queries, 3)).astype(np.float32)


def make_grad_output(
    query_count: int, seed: int, scale: float = 1.0, device: str = "cuda:0"
) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(query_count, generator=g) * scale).to(device)


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def slice_mesh_in_half(vertices, indices, dim=2):
    centroids = vertices[indices].mean(axis=1)
    split = np.median(centroids[:, dim])
    return vertices, indices[centroids[:, dim] > split]


def drop_half_of_the_triangles(vertices, indices, seed: int = 0):
    rng = np.random.default_rng(seed)
    choice = rng.choice(np.arange(len(indices)), len(indices) // 2, replace=False)
    return vertices, indices[choice]


# =========================================================================
# Validator
# =========================================================================


@dataclass
class GradientMetrics:
    label: str
    n_scalars: int
    has_nan: bool
    has_inf: bool
    max_abs_err: float
    mean_abs_err: float
    global_rel_norm_err: float
    global_cosine_flat: float
    masked_mean_rel: float
    masked_p50_rel: float
    masked_p95_rel: float
    masked_p99_rel: float
    masked_max_rel: float
    per_vec_cos_mean: float
    per_vec_cos_p01: float
    per_vec_cos_min: float
    per_vec_ang_p99_deg: float
    per_vec_ang_max_deg: float
    signal_coverage: float
    passed: bool = False
    failed_checks: tuple = field(default_factory=tuple)


def _safe_quantile(x: np.ndarray, q: float) -> float:
    return float("nan") if x.size == 0 else float(np.quantile(x, q))


def print_ascii_histogram(
    data: np.ndarray, title: str, unit: str = "", n_bins: int = 12
) -> None:
    data = data[np.isfinite(data) & (data > 0)]
    if data.size == 0:
        print(f"\n  --- {title} --- (no data)")
        return

    lo = max(0, math.floor(math.log10(data.min())))
    hi = math.ceil(math.log10(data.max()))
    if hi - lo < 2:
        hi = lo + 2

    edges = np.logspace(lo, hi, n_bins + 1)
    edges[0] = 0.0
    counts, _ = np.histogram(data, bins=edges)
    total = counts.sum()
    max_count = max(1, counts.max())

    print(f"\n  --- {title} ---")
    for i in range(len(counts)):
        low, high = edges[i], edges[i + 1]
        count = int(counts[i])
        pct = 100.0 * count / total if total > 0 else 0.0
        bar = "█" * int(round(60 * count / max_count))
        label = f"[0, {high:.1e})" if low == 0.0 else f"[{low:.1e}, {high:.1e})"
        print(f"  {label:<22}{unit:<2} | {bar:<60} | {count:6d} ({pct:6.2f}%)")


def validate_gradients(
    fast_grads: torch.Tensor,
    ref_grads: torch.Tensor,
    label: str,
    *,
    signal_threshold: float = 1e-5,
    strict: bool = False,
    rel_norm_tol_strict: float = 1e-4,
    rel_norm_tol_loose: float = 1e-1,
    cosine_tol_strict: float = 0.99999,
    cosine_tol_loose: float = 0.99,
    masked_p99_rel_tol: float = 1e-2,
    per_vec_cos_p01_tol: float = 0.99,
    verbose: bool = True,
) -> GradientMetrics:
    """Compare two gradient tensors of identical shape and return a metric bundle.

    Uses strict tolerances when `strict=True` (should be used when the fast
    engine is expected to reproduce brute force exactly, i.e. beta=-1).
    """
    assert fast_grads.shape == ref_grads.shape, (
        f"shape mismatch: {fast_grads.shape} vs {ref_grads.shape}"
    )

    fast = fast_grads.detach().float().reshape(-1)
    ref = ref_grads.detach().float().reshape(-1)
    fast3 = fast_grads.detach().float().reshape(-1, 3)
    ref3 = ref_grads.detach().float().reshape(-1, 3)

    has_nan = bool(torch.isnan(fast).any().item())
    has_inf = bool(torch.isinf(fast).any().item())

    if has_nan or has_inf:
        if verbose:
            print(f"\n  [{label}] CRITICAL: NaN={has_nan} Inf={has_inf}")
        return GradientMetrics(
            label=label,
            n_scalars=fast.numel(),
            has_nan=has_nan,
            has_inf=has_inf,
            max_abs_err=float("inf"),
            mean_abs_err=float("inf"),
            global_rel_norm_err=float("inf"),
            global_cosine_flat=float("nan"),
            masked_mean_rel=float("inf"),
            masked_p50_rel=float("inf"),
            masked_p95_rel=float("inf"),
            masked_p99_rel=float("inf"),
            masked_max_rel=float("inf"),
            per_vec_cos_mean=float("nan"),
            per_vec_cos_p01=float("nan"),
            per_vec_cos_min=float("nan"),
            per_vec_ang_p99_deg=float("nan"),
            per_vec_ang_max_deg=float("nan"),
            signal_coverage=0.0,
            passed=False,
            failed_checks=("nan_or_inf",),
        )

    diff = fast - ref
    abs_diff = diff.abs()
    max_abs = float(abs_diff.max().item())
    mean_abs = float(abs_diff.mean().item())

    n_ref = float(torch.linalg.norm(ref).item())
    n_fast = float(torch.linalg.norm(fast).item())
    n_diff = float(torch.linalg.norm(diff).item())
    rel_norm = n_diff / (n_ref + 1e-30)

    cos_flat = (
        float(torch.dot(fast, ref).item() / (n_ref * n_fast))
        if (n_ref > 0 and n_fast > 0)
        else float("nan")
    )

    # Scalar relative errors, masked by reference magnitude.
    ref_np = ref.cpu().numpy()
    fast_np = fast.cpu().numpy()
    mask = np.abs(ref_np) > signal_threshold
    signal_coverage = float(mask.mean())
    if mask.any():
        rel_m = np.abs(fast_np[mask] - ref_np[mask]) / np.abs(ref_np[mask])
        mean_rel = float(rel_m.mean())
        p50 = _safe_quantile(rel_m, 0.50)
        p95 = _safe_quantile(rel_m, 0.95)
        p99 = _safe_quantile(rel_m, 0.99)
        mx = float(rel_m.max())
    else:
        rel_m = np.array([])
        mean_rel = p50 = p95 = p99 = mx = float("nan")

    # Per-vector cosine and angular error.
    ref_np3 = ref3.cpu().numpy()
    fast_np3 = fast3.cpu().numpy()
    rn = np.linalg.norm(ref_np3, axis=-1)
    fn = np.linalg.norm(fast_np3, axis=-1)
    vmask = (rn > signal_threshold) & (fn > signal_threshold)
    if vmask.any():
        a = fast_np3[vmask]
        b = ref_np3[vmask]
        cos = np.clip(
            (a * b).sum(-1)
            / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-30),
            -1.0,
            1.0,
        )
        ang = np.degrees(np.arccos(cos))
        cos_mean = float(cos.mean())
        cos_p01 = _safe_quantile(cos, 0.01)
        cos_min = float(cos.min())
        ang_p99 = _safe_quantile(ang, 0.99)
        ang_max = float(ang.max())
    else:
        cos_mean = cos_p01 = cos_min = float("nan")
        ang_p99 = ang_max = float("nan")

    # ---------------------------------------------------------------------
    # Pass criteria
    # ---------------------------------------------------------------------
    rel_norm_tol = rel_norm_tol_strict if strict else rel_norm_tol_loose
    cosine_tol = cosine_tol_strict if strict else cosine_tol_loose

    def _ok(v, cmp, thr):
        return True if (v is None or math.isnan(v)) else cmp(v, thr)

    checks = {
        "global_cosine": _ok(cos_flat, lambda a, b: a > b, cosine_tol),
        "rel_norm": _ok(rel_norm, lambda a, b: a < b, rel_norm_tol),
        "masked_p99_rel": _ok(p99, lambda a, b: a < b, masked_p99_rel_tol),
        "per_vec_cos_p01": _ok(cos_p01, lambda a, b: a > b, per_vec_cos_p01_tol),
    }
    failed = tuple(k for k, v in checks.items() if not v)
    passed = not failed

    if verbose:
        print(f"\n{'=' * 72}")
        print(f"  Gradient validation: {label}   [{fast.numel()} scalars]")
        print(f"{'=' * 72}")
        print(f"  Max abs error              : {max_abs:.4e}")
        print(f"  Mean abs error             : {mean_abs:.4e}")
        print(
            f"  Global rel-norm error      : {rel_norm:.6e}   (tol {rel_norm_tol:.1e})"
        )
        print(f"  Global cosine (flattened)  : {cos_flat:.8f}   (tol {cosine_tol})")
        print(f"  Signal coverage            : {100.0 * signal_coverage:.2f}%")
        print(f"  --- masked scalar rel err ---")
        print(
            f"  mean / p50 / p95 / p99     : "
            f"{mean_rel:.3e} / {p50:.3e} / {p95:.3e} / {p99:.3e}"
        )
        print(f"  max                        : {mx:.3e}")
        print(f"  --- per-vector ---")
        print(
            f"  cos mean / p01 / min       : "
            f"{cos_mean:.6f} / {cos_p01:.6f} / {cos_min:.6f}"
        )
        print(f"  angle p99 / max (deg)      : {ang_p99:.4f} / {ang_max:.4f}")

        if rel_m.size:
            print_ascii_histogram(rel_m, "Log10 relative error")
        if vmask.any():
            print_ascii_histogram(
                np.degrees(
                    np.arccos(
                        np.clip(
                            (fast_np3[vmask] * ref_np3[vmask]).sum(-1)
                            / (
                                np.linalg.norm(fast_np3[vmask], axis=-1)
                                * np.linalg.norm(ref_np3[vmask], axis=-1)
                                + 1e-30
                            ),
                            -1.0,
                            1.0,
                        )
                    )
                ),
                "Angular error (deg)",
            )

        if passed:
            print(f"\n\033[92m  ✓ PASS  [{label}]\033[0m")
        else:
            print(f"\n\033[91m  ✗ FAIL  [{label}]  failed: {', '.join(failed)}\033[0m")

    finite = np.isfinite(ref_np) & (np.abs(ref_np) > 1e-3)
    if finite.any():
        ratio = fast_np[finite] / ref_np[finite]
        print(
            f"  fast/brute ratio: median={np.median(ratio):.3e}  "
            f"p10={np.percentile(ratio, 10):.3e}  p90={np.percentile(ratio, 90):.3e}"
        )
        # Also: correlation between ratio and |ref|
        absref = np.abs(ref_np[finite])
        print(
            f"  corr(ratio, |ref|) = {np.corrcoef(np.log10(np.abs(ratio) + 1e-30), np.log10(absref))[0, 1]:.3f}"
        )

    return GradientMetrics(
        label=label,
        n_scalars=fast.numel(),
        has_nan=False,
        has_inf=False,
        max_abs_err=max_abs,
        mean_abs_err=mean_abs,
        global_rel_norm_err=rel_norm,
        global_cosine_flat=cos_flat,
        masked_mean_rel=mean_rel,
        masked_p50_rel=p50,
        masked_p95_rel=p95,
        masked_p99_rel=p99,
        masked_max_rel=mx,
        per_vec_cos_mean=cos_mean,
        per_vec_cos_p01=cos_p01,
        per_vec_cos_min=cos_min,
        per_vec_ang_p99_deg=ang_p99,
        per_vec_ang_max_deg=ang_max,
        signal_coverage=signal_coverage,
        passed=passed,
        failed_checks=failed,
    )


# =========================================================================
# Low-level runners
# =========================================================================


def _to_cuda(x, dtype=None):
    t = torch.from_numpy(np.ascontiguousarray(x))
    if dtype is not None:
        t = t.to(dtype)
    return t.to("cuda:0").contiguous()


def run_brute_force_triangle_gradients(triangles, queries, grad_output, out=None):
    t = _to_cuda(triangles, torch.float32)
    q = _to_cuda(queries, torch.float32)
    g = grad_output if grad_output.is_cuda else grad_output.to("cuda:0")
    g = g.contiguous()
    if out is None:
        out = torch.empty(
            [triangles.shape[0], 3, 3], device="cuda:0", dtype=torch.float32
        )
    winder.brute_force_gradients(
        g, t, q, out, stream=torch.cuda.current_stream().cuda_stream
    )
    torch.cuda.synchronize()
    return out


def run_fast_triangle_gradients(triangles, queries, grad_output, beta, out=None):
    t = _to_cuda(triangles, torch.float32)
    q = _to_cuda(queries, torch.float32)
    g = grad_output if grad_output.is_cuda else grad_output.to("cuda:0")
    g = g.contiguous()
    if out is None:
        out = torch.empty(
            [triangles.shape[0], 3, 3], device="cuda:0", dtype=torch.float32
        )
    engine = winder.GradientEngine(q, g, stream=torch.cuda.current_stream().cuda_stream)
    engine.compute(
        t,
        out,
        beta=-1 if beta is None else beta,
        stream=torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()
    return out


def run_brute_force_mesh_gradients(vertices, indices, queries, grad_output, out=None):
    v = _to_cuda(vertices, torch.float32)
    idx = _to_cuda(indices.astype(np.uint32))
    q = _to_cuda(queries, torch.float32)
    g = grad_output if grad_output.is_cuda else grad_output.to("cuda:0")
    g = g.contiguous()
    if out is None:
        out = torch.empty_like(v)
    winder.brute_force_gradients(
        g, v, idx, q, out, stream=torch.cuda.current_stream().cuda_stream
    )
    torch.cuda.synchronize()
    return out


def run_fast_mesh_gradients(vertices, indices, queries, grad_output, beta, out=None):
    v = _to_cuda(vertices, torch.float32)
    idx = _to_cuda(indices.astype(np.uint32))
    q = _to_cuda(queries, torch.float32)
    g = grad_output if grad_output.is_cuda else grad_output.to("cuda:0")
    g = g.contiguous()
    if out is None:
        out = torch.empty_like(v)
    engine = winder.GradientEngine(q, g, stream=torch.cuda.current_stream().cuda_stream)
    engine.compute(
        v,
        idx,
        out,
        beta=-1 if beta is None else beta,
        stream=torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()
    return out


def run_brute_force_point_normal_gradients(
    points, normals, queries, grad_output, epsilon, out=None
):
    p = _to_cuda(points, torch.float32)
    n = _to_cuda(normals, torch.float32)
    q = _to_cuda(queries, torch.float32)
    g = grad_output if grad_output.is_cuda else grad_output.to("cuda:0")
    g = g.contiguous()
    if out is None:
        out = torch.empty([points.shape[0], 2, 3], device="cuda:0", dtype=torch.float32)
    winder.brute_force_gradients(
        g,
        p,
        n,
        q,
        out,
        epsilon=epsilon,
        stream=torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()
    return out


def run_fast_point_normal_gradients(
    points, normals, queries, grad_output, epsilon, beta, out=None
):
    p = _to_cuda(points, torch.float32)
    n = _to_cuda(normals, torch.float32)
    q = _to_cuda(queries, torch.float32)
    g = grad_output if grad_output.is_cuda else grad_output.to("cuda:0")
    g = g.contiguous()
    if out is None:
        out = torch.empty([points.shape[0], 2, 3], device="cuda:0", dtype=torch.float32)
    engine = winder.GradientEngine(q, g, stream=torch.cuda.current_stream().cuda_stream)
    engine.compute(
        p,
        n,
        out,
        epsilon=epsilon,
        beta=-1 if beta is None else beta,
        stream=torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()
    return out


# =========================================================================
# Implementation-independent invariants
# =========================================================================


def _invariant_report(name: str, err: float, scale: float, tol_rel: float) -> bool:
    rel = err / (scale + 1e-30)
    ok = rel < tol_rel
    status = "\033[92mPASS\033[0m" if ok else "\033[91mFAIL\033[0m"
    print(f"  [{name}] max|Δg|/scale = {rel:.3e}  ->  {status}")
    return ok


def test_mesh_translation_invariance(vertices, indices, queries, g, beta) -> bool:
    """Winding gradients are invariant to simultaneous translation of mesh and queries."""
    diag = float(np.linalg.norm(vertices.max(0) - vertices.min(0)))
    t = np.array([0.37, -0.41, 0.19], dtype=np.float32) * diag

    a = run_fast_mesh_gradients(vertices, indices, queries, g, beta)
    b = run_fast_mesh_gradients(vertices + t, indices, queries + t, g, beta)
    err = float((a - b).abs().max().item())
    return _invariant_report(
        "translation invariance", err, float(a.abs().max().item()), 1e-4
    )


def test_mesh_query_permutation(vertices, indices, queries, g, beta, seed=0) -> bool:
    """Permuting queries (and their grad_output) must permute the accumulated
    primitive gradients by the same permutation (up to float summation order)."""
    n = queries.shape[0]
    perm = np.random.default_rng(seed).permutation(n)
    a = run_fast_mesh_gradients(vertices, indices, queries, g, beta)
    b = run_fast_mesh_gradients(vertices, indices, queries[perm], g[perm], beta)
    err = float((a - b).abs().max().item())
    return _invariant_report(
        "query permutation", err, float(a.abs().max().item()), 1e-5
    )


def test_mesh_grad_output_linearity(vertices, indices, queries, g, beta) -> bool:
    """The map grad_output -> vertex_gradients is linear."""
    a = run_fast_mesh_gradients(vertices, indices, queries, g, beta)
    b = run_fast_mesh_gradients(vertices, indices, queries, 3.0 * g, beta)
    err = float((b - 3.0 * a).abs().max().item())
    scale = float(a.abs().max().item()) * 3.0
    return _invariant_report("grad_output linearity", err, scale, 1e-5)


def test_mesh_determinism(vertices, indices, queries, g, beta) -> bool:
    a = run_fast_mesh_gradients(vertices, indices, queries, g, beta)
    b = run_fast_mesh_gradients(vertices, indices, queries, g, beta)
    err = float((a - b).abs().max().item())
    return _invariant_report("determinism", err, 1.0, 0.0)  # exact equality


def run_mesh_invariants(vertices, indices, query_mode, query_count, beta, seed) -> bool:
    print("\n=== Mesh gradient invariants (implementation-independent) ===")
    q = generate_queries(vertices, query_mode, query_count, seed)
    g = make_grad_output(query_count, seed + 1)
    results = [
        test_mesh_translation_invariance(vertices, indices, q, g, beta),
        test_mesh_query_permutation(vertices, indices, q, g, beta, seed),
        test_mesh_grad_output_linearity(vertices, indices, q, g, beta),
        test_mesh_determinism(vertices, indices, q, g, beta),
    ]
    return all(results)


# =========================================================================
# Main test drivers
# =========================================================================


def test_triangle_gradients(
    vertices,
    indices,
    query_mode,
    query_count,
    beta,
    num_trials,
    base_seed,
    verbose=True,
):
    triangles = vertices[indices]
    all_metrics = []
    for trial in range(num_trials):
        seed = base_seed + trial * 1009
        set_seeds(seed)
        queries = generate_queries(vertices, query_mode, query_count, seed)
        g = make_grad_output(query_count, seed + 1)

        gt = run_brute_force_triangle_gradients(triangles, queries, g)
        fast = run_fast_triangle_gradients(triangles, queries, g, beta)

        strict = beta == -1 or beta is None
        for idx, name in enumerate(["Triangle v0", "Triangle v1", "Triangle v2"]):
            m = validate_gradients(
                fast[:, idx],
                gt[:, idx],
                f"{name} [seed={seed}]",
                strict=strict,
                verbose=verbose,
            )
            all_metrics.append(m)
    # --- Timing ---
    set_seeds(base_seed)
    q = generate_queries(vertices, query_mode, query_count, base_seed)
    g = make_grad_output(query_count, base_seed + 1)

    bf_ms = cuda_timer(lambda: run_brute_force_triangle_gradients(triangles, q, g))
    fast_ms = cuda_timer(lambda: run_fast_triangle_gradients(triangles, q, g, beta))
    build_ms = cuda_timer(
        lambda: winder.GradientEngine(
            torch.from_numpy(q).cuda(),
            g,
            stream=torch.cuda.current_stream().cuda_stream,
        ),
        warmup=2,
        iters=5,
    )

    print(f"\n  Timing (median of 10, ms):")
    print(f"    brute force     : {bf_ms:8.3f}")
    print(f"    fast (build+go) : {fast_ms:8.3f}   speedup {bf_ms / fast_ms:.2f}x")
    print(
        f"    fast (build only): {build_ms:8.3f}   "
        f"({100 * build_ms / fast_ms:.1f}% of fast path)"
    )
    return all_metrics


def test_mesh_gradients(
    vertices,
    indices,
    query_mode,
    query_count,
    beta,
    num_trials,
    base_seed,
    verbose=True,
):
    all_metrics = []
    for trial in range(num_trials):
        seed = base_seed + trial * 1009
        set_seeds(seed)
        queries = generate_queries(vertices, query_mode, query_count, seed)
        g = make_grad_output(query_count, seed + 1)

        gt = run_brute_force_mesh_gradients(vertices, indices, queries, g)
        fast = run_fast_mesh_gradients(vertices, indices, queries, g, beta)

        strict = beta == -1 or beta is None
        m = validate_gradients(
            fast,
            gt,
            f"Mesh [seed={seed}]",
            strict=strict,
            verbose=verbose,
        )
        all_metrics.append(m)
    # --- Timing ---
    set_seeds(base_seed)
    q = generate_queries(vertices, query_mode, query_count, base_seed)
    g = make_grad_output(query_count, base_seed + 1)

    bf_ms = cuda_timer(lambda: run_brute_force_mesh_gradients(vertices, indices, q, g))
    fast_ms = cuda_timer(lambda: run_fast_mesh_gradients(vertices, indices, q, g, beta))
    build_ms = cuda_timer(
        lambda: winder.GradientEngine(
            torch.from_numpy(q).cuda(),
            g,
            stream=torch.cuda.current_stream().cuda_stream,
        ),
        warmup=2,
        iters=5,
    )

    print(f"\n  Timing (median of 10, ms):")
    print(f"    brute force     : {bf_ms:8.3f}")
    print(f"    fast (build+go) : {fast_ms:8.3f}   speedup {bf_ms / fast_ms:.2f}x")
    print(
        f"    fast (build only): {build_ms:8.3f}   "
        f"({100 * build_ms / fast_ms:.1f}% of fast path)"
    )
    return all_metrics


def test_point_normal_gradients(
    points,
    normals,
    areas,
    query_mode,
    query_count,
    inv_epsilon,
    beta,
    num_trials,
    base_seed,
    verbose=True,
):
    scaled_normals = normals * areas[:, None]
    epsilon = 1.0 / inv_epsilon
    all_metrics = []
    for trial in range(num_trials):
        seed = base_seed + trial * 1009
        set_seeds(seed)
        queries = generate_queries(points, query_mode, query_count, seed)
        g = make_grad_output(query_count, seed + 1)

        gt = run_brute_force_point_normal_gradients(
            points, scaled_normals, queries, g, epsilon
        )
        fast = run_fast_point_normal_gradients(
            points, scaled_normals, queries, g, epsilon, beta
        )

        strict = beta == -1 or beta is None
        for idx, name in enumerate(["PointNormal n", "PointNormal p"]):
            m = validate_gradients(
                fast[:, idx],
                gt[:, idx],
                f"{name} [seed={seed}]",
                strict=strict,
                verbose=verbose,
            )
            all_metrics.append(m)
    # --- Timing ---
    set_seeds(base_seed)
    q = generate_queries(points, query_mode, query_count, base_seed)
    g = make_grad_output(query_count, seed + 1)

    bf_ms = cuda_timer(
        lambda: run_brute_force_point_normal_gradients(
            points, scaled_normals, q, g, epsilon
        )
    )
    fast_ms = cuda_timer(
        lambda: run_fast_point_normal_gradients(
            points, scaled_normals, q, g, epsilon, beta
        )
    )
    build_ms = cuda_timer(
        lambda: winder.GradientEngine(
            torch.from_numpy(q).cuda(),
            g,
            stream=torch.cuda.current_stream().cuda_stream,
        ),
        warmup=2,
        iters=5,
    )

    print(f"\n  Timing (median of 10, ms):")
    print(f"    brute force     : {bf_ms:8.3f}")
    print(f"    fast (build+go) : {fast_ms:8.3f}   speedup {bf_ms / fast_ms:.2f}x")
    print(
        f"    fast (build only): {build_ms:8.3f}   "
        f"({100 * build_ms / fast_ms:.1f}% of fast path)"
    )

    return all_metrics


# =========================================================================
# Beta sweep
# =========================================================================


def run_beta_sweep(
    vertices,
    indices,
    points,
    normals,
    areas,
    query_mode,
    query_count,
    beta_values,
    num_trials,
    base_seed,
    inv_epsilon,
):
    print("\n" + "=" * 88)
    print("  BETA SWEEP — approximation error and compute time vs beta")
    print("=" * 88)

    table = {}  # beta -> {geom: [GradientMetrics]}
    timing = {}  # beta -> {geom: {bf_ms, build_ms, compute_ms}}

    # One representative query set for timing (timing is beta-independent
    # for brute force; for the fast engine it depends on beta only through
    # the tree traversal, not through the tree construction).
    set_seeds(base_seed)
    timing_queries = generate_queries(vertices, query_mode, query_count, base_seed)
    timing_g = make_grad_output(query_count, base_seed + 1)

    # -------------------------------------------------------------------
    # Time brute force once (it does not depend on beta).
    # -------------------------------------------------------------------
    tris = vertices[indices]
    scaled_normals = normals * areas[:, None]
    epsilon = 1.0 / inv_epsilon

    bf_mesh_ms = cuda_timer(
        lambda: run_brute_force_mesh_gradients(
            vertices, indices, timing_queries, timing_g
        )
    )
    bf_tri_ms = cuda_timer(
        lambda: run_brute_force_triangle_gradients(tris, timing_queries, timing_g)
    )
    bf_pn_ms = cuda_timer(
        lambda: run_brute_force_point_normal_gradients(
            points, scaled_normals, timing_queries, timing_g, epsilon
        )
    )
    print(f"\n  Brute-force timings (median of 10, ms):")
    print(f"    mesh          : {bf_mesh_ms:8.3f} ms")
    print(f"    triangle      : {bf_tri_ms:8.3f} ms")
    print(f"    point_normal  : {bf_pn_ms:8.3f} ms")

    # -------------------------------------------------------------------
    # Sweep
    # -------------------------------------------------------------------
    for beta in beta_values:
        table[beta] = {"mesh": [], "triangle": [], "point_normal": []}
        timing[beta] = {}

        # ------------------------------------------------------------
        # Correctness metrics (multi-trial)
        # ------------------------------------------------------------
        strict = False  # see note below re: exact_beta
        for trial in range(num_trials):
            seed = base_seed + trial * 1009
            set_seeds(seed)
            queries = generate_queries(vertices, query_mode, query_count, seed)
            g = make_grad_output(query_count, seed + 1)

            # Mesh
            gt = run_brute_force_mesh_gradients(vertices, indices, queries, g)
            fast = run_fast_mesh_gradients(vertices, indices, queries, g, beta)
            table[beta]["mesh"].append(
                validate_gradients(
                    fast,
                    gt,
                    f"mesh beta={beta} seed={seed}",
                    strict=strict,
                    verbose=False,
                )
            )

            # Triangle
            gt = run_brute_force_triangle_gradients(tris, queries, g)
            fast = run_fast_triangle_gradients(tris, queries, g, beta)
            for idx in range(3):
                table[beta]["triangle"].append(
                    validate_gradients(
                        fast[:, idx],
                        gt[:, idx],
                        f"tri{idx} beta={beta} seed={seed}",
                        strict=strict,
                        verbose=False,
                    )
                )

            # PointNormal
            gt = run_brute_force_point_normal_gradients(
                points, scaled_normals, queries, g, epsilon
            )
            fast = run_fast_point_normal_gradients(
                points, scaled_normals, queries, g, epsilon, beta
            )
            for idx in range(2):
                table[beta]["point_normal"].append(
                    validate_gradients(
                        fast[:, idx],
                        gt[:, idx],
                        f"pn{idx} beta={beta} seed={seed}",
                        strict=strict,
                        verbose=False,
                    )
                )

        # ------------------------------------------------------------
        # Timing at this beta
        # ------------------------------------------------------------
        # Fast engine build and compute are timed separately so you can
        # see whether build time (tree construction) or traversal time
        # dominates for this beta.
        def _build_mesh():
            return winder.GradientEngine(
                torch.from_numpy(timing_queries).cuda(),
                timing_g,
                stream=torch.cuda.current_stream().cuda_stream,
            )

        # Build-only timing.
        timing[beta]["mesh"] = {
            "build_ms": cuda_timer(_build_mesh, warmup=2, iters=5),
        }

        # Full path: build + compute, so the comparison against brute force
        # includes the acceleration structure setup.
        def _fast_mesh():
            run_fast_mesh_gradients(vertices, indices, timing_queries, timing_g, beta)

        def _fast_tri():
            run_fast_triangle_gradients(tris, timing_queries, timing_g, beta)

        def _fast_pn():
            run_fast_point_normal_gradients(
                points,
                scaled_normals,
                timing_queries,
                timing_g,
                epsilon,
                beta,
            )

        timing[beta]["mesh"]["total_ms"] = cuda_timer(_fast_mesh)
        timing[beta]["triangle"] = {"total_ms": cuda_timer(_fast_tri)}
        timing[beta]["point_normal"] = {"total_ms": cuda_timer(_fast_pn)}

    # -------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------
    print(
        f"\n{'beta':>6} | {'geometry':<14} | "
        f"{'rel_norm(med)':>13} | {'p99_rel(med)':>12} | "
        f"{'BF(ms)':>8} | {'fast(ms)':>9} | {'speedup':>8} | {'pass':>5}"
    )
    print("-" * 108)

    bf_ms_for = {
        "mesh": bf_mesh_ms,
        "triangle": bf_tri_ms,
        "point_normal": bf_pn_ms,
    }

    for beta in beta_values:
        for geom in ("mesh", "triangle", "point_normal"):
            ms = table[beta][geom]
            if not ms:
                continue
            rel_med = float(np.median([m.global_rel_norm_err for m in ms]))
            p99_med = float(np.median([m.masked_p99_rel for m in ms]))
            all_pass = all(m.passed for m in ms)

            bf_ms = bf_ms_for[geom]
            fast_ms = timing[beta][geom]["total_ms"]
            speedup = bf_ms / fast_ms if fast_ms > 0 else float("nan")

            print(
                f"{beta:>6.3f} | {geom:<14} | "
                f"{rel_med:>13.3e} | {p99_med:>12.3e} | "
                f"{bf_ms:>8.3f} | {fast_ms:>9.3f} | "
                f"{speedup:>7.2f}x | "
                f"{'YES' if all_pass else 'NO':>5}"
            )

    # -------------------------------------------------------------------
    # Monotonicity + plateau (unchanged)
    # -------------------------------------------------------------------
    print("-" * 108)
    non_monotone = []
    prev_med = {"mesh": None, "triangle": None, "point_normal": None}
    for beta in sorted(beta_values):
        for geom in ("mesh", "triangle", "point_normal"):
            ms = table[beta][geom]
            if not ms:
                continue
            rel_med = float(np.median([m.global_rel_norm_err for m in ms]))
            prev = prev_med[geom]
            if prev is not None and prev > 0 and rel_med > prev * 1.5:
                non_monotone.append((geom, beta, prev, rel_med))
            prev_med[geom] = rel_med if prev is None else min(prev, rel_med)

    if non_monotone:
        print(
            "\n\033[91mNon-monotone behaviour detected "
            "(error rose by >1.5x as beta increased):\033[0m"
        )
        for geom, beta, prev, cur in non_monotone:
            print(f"  {geom}: beta={beta:.3f}  rel_norm {prev:.3e} -> {cur:.3e}")
    else:
        print(
            "\n\033[92mMonotone non-increasing: error decreases "
            "(or plateaus) as beta grows.\033[0m"
        )

    print("\n  Plateau (min rel_norm over sweep):")
    for geom in ("mesh", "triangle", "point_normal"):
        mins = [
            float(np.median([m.global_rel_norm_err for m in table[b][geom]]))
            for b in beta_values
            if table[b][geom]
        ]
        if mins:
            print(f"    {geom:<14}: {min(mins):.3e}")

    return table, timing


# =========================================================================
# CSV + plotting
# =========================================================================

CSV_FIELDS = [
    "timestamp",
    "label",
    "unique_label",
    "geometry_type",
    "query_mode",
    "query_count",
    "beta",
    "inv_epsilon",
    "seed",
    "num_trials",
    "pn_rel_norm_err",
    "tri_rel_norm_err",
    "mesh_rel_norm_err",
    "pn_passed",
    "tri_passed",
    "mesh_passed",
]


def write_run_to_csv(
    csv_path,
    unique_label,
    raw_label,
    args,
    pn_metrics,
    tri_metrics,
    mesh_metrics,
):
    """Append a single summary row. Archives the CSV if its header differs."""
    file_exists = os.path.exists(csv_path)
    if file_exists:
        with open(csv_path, "r", newline="") as f:
            header = next(csv.reader(f), None)
        if header != CSV_FIELDS:
            backup = f"{csv_path}.bak.{datetime.now():%Y%m%d_%H%M%S}"
            shutil.move(csv_path, backup)
            print(f"[CSV] Header changed; archived previous file to {backup}")
            file_exists = False

    def _avg(ms, attr):
        if not ms:
            return ""
        vals = [getattr(m, attr) for m in ms if math.isfinite(getattr(m, attr))]
        return f"{float(np.mean(vals)):.6e}" if vals else ""

    def _pass(ms):
        return "" if not ms else str(all(m.passed for m in ms))

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "label": raw_label,
        "unique_label": unique_label,
        "geometry_type": args.geometry_type,
        "query_mode": args.query_mode,
        "query_count": args.query_count,
        "beta": args.beta if args.beta is not None else -1.0,
        "inv_epsilon": args.inv_epsilon,
        "seed": args.seed,
        "num_trials": args.num_trials,
        "pn_rel_norm_err": _avg(pn_metrics, "global_rel_norm_err"),
        "tri_rel_norm_err": _avg(tri_metrics, "global_rel_norm_err"),
        "mesh_rel_norm_err": _avg(mesh_metrics, "global_rel_norm_err"),
        "pn_passed": _pass(pn_metrics),
        "tri_passed": _pass(tri_metrics),
        "mesh_passed": _pass(mesh_metrics),
    }

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"[CSV] Appended 1 row to {csv_path}")


def plot_comparison(csv_path):
    if not os.path.exists(csv_path):
        print(f"CSV {csv_path} not found; nothing to plot.")
        return

    labels, pn_errs, tri_errs, mesh_errs = [], [], [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(row["unique_label"])

            def _f(v):
                try:
                    return float(v) if v else np.nan
                except ValueError:
                    return np.nan

            pn_errs.append(_f(row.get("pn_rel_norm_err", "")))
            tri_errs.append(_f(row.get("tri_rel_norm_err", "")))
            mesh_errs.append(_f(row.get("mesh_rel_norm_err", "")))

    if not labels:
        print("No runs in CSV to plot.")
        return

    x = np.arange(len(labels))
    w = 0.27
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.2), 5))
    ax.bar(x - w, pn_errs, w, label="PointNormal", color="#1f77b4", edgecolor="black")
    ax.bar(x, tri_errs, w, label="Triangle", color="#ff7f0e", edgecolor="black")
    ax.bar(x + w, mesh_errs, w, label="Mesh", color="#2ca02c", edgecolor="black")
    ax.set_yscale("log")
    ax.set_ylabel("Global relative norm error (log)")
    ax.set_title("Gradient error across runs")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.legend()
    ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


# =========================================================================
# CLI
# =========================================================================


def _summarize(metrics_list, name):
    if not metrics_list:
        print(f"\n[{name}] no metrics produced.")
        return None
    errs = [
        m.global_rel_norm_err
        for m in metrics_list
        if math.isfinite(m.global_rel_norm_err)
    ]
    any_failed = any(not m.passed for m in metrics_list)
    mean_err = float(np.mean(errs)) if errs else float("nan")
    status = "\033[91mFAIL\033[0m" if any_failed else "\033[92mPASS\033[0m"
    print(
        f"\n[{name}] mean rel_norm = {mean_err:.3e},  overall: {status}  "
        f"({sum(m.passed for m in metrics_list)}/{len(metrics_list)} sub-checks passed)"
    )
    return mean_err


def main():
    parser = argparse.ArgumentParser(
        description="Validate fast (Barnes-Hut) winding-number gradients against brute force."
    )
    parser.add_argument("--obj_file", type=str, required=True)
    parser.add_argument(
        "--geometry_type",
        type=str,
        choices=["PointNormal", "Triangle", "Mesh", "All"],
        default="All",
    )
    parser.add_argument(
        "--query_mode", type=str, choices=["random", "grid"], default="random"
    )
    parser.add_argument("--query_count", type=int, default=100000)
    parser.add_argument("--inv_epsilon", type=float, default=250.0)
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Barnes-Hut approximation parameter (-1 for exact).",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Base seed for reproducibility."
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=1,
        help="Number of independent trials (different seeds).",
    )
    parser.add_argument(
        "--beta_sweep",
        type=str,
        default="",
        help="Comma-separated beta values to sweep, e.g. '-1,0.5,1.0'.",
    )
    parser.add_argument(
        "--beta_sweep_only",
        action="store_true",
        help="Run only the beta sweep and exit.",
    )
    parser.add_argument(
        "--skip_invariants",
        action="store_true",
        help="Skip the implementation-independent invariant tests.",
    )
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--csv_path", type=str, default="benchmark_results.csv")

    args = parser.parse_args()

    set_seeds(args.seed)

    print(f"Loading mesh from {args.obj_file}")
    vertices, _, _, indices, _, _ = igl.readOBJ(args.obj_file)
    vertices, indices = slice_mesh_in_half(vertices, indices)
    vertices, indices = drop_half_of_the_triangles(vertices, indices, seed=args.seed)
    print(f"Mesh: {len(indices)} triangles, {len(vertices)} vertices")

    points, normals, areas = mesh_to_point_surfels(vertices, indices)

    # ---------------------------------------------------------------
    # Beta sweep
    # ---------------------------------------------------------------
    if args.beta_sweep:
        betas = [float(x) for x in args.beta_sweep.split(",")]
        run_beta_sweep(
            vertices,
            indices,
            points,
            normals,
            areas,
            args.query_mode,
            args.query_count,
            betas,
            args.num_trials,
            args.seed,
            args.inv_epsilon,
        )
        if args.beta_sweep_only:
            return

    # ---------------------------------------------------------------
    # Main geometry validation
    # ---------------------------------------------------------------
    pn_metrics = tri_metrics = mesh_metrics = None

    if args.geometry_type in ("Mesh", "All") and not args.skip_invariants:
        run_mesh_invariants(
            vertices,
            indices,
            args.query_mode,
            args.query_count,
            args.beta if args.beta is not None else -1,
            args.seed,
        )

    if args.geometry_type in ("PointNormal", "All"):
        print("\n" + "=" * 72)
        print("  PointNormal gradient validation")
        print("=" * 72)
        pn_metrics = test_point_normal_gradients(
            points,
            normals,
            areas,
            args.query_mode,
            args.query_count,
            args.inv_epsilon,
            args.beta,
            args.num_trials,
            args.seed,
        )
        _summarize(pn_metrics, "PointNormal")

    if args.geometry_type in ("Triangle", "All"):
        print("\n" + "=" * 72)
        print("  Triangle gradient validation")
        print("=" * 72)
        tri_metrics = test_triangle_gradients(
            vertices,
            indices,
            args.query_mode,
            args.query_count,
            args.beta,
            args.num_trials,
            args.seed,
        )
        _summarize(tri_metrics, "Triangle")

    if args.geometry_type in ("Mesh", "All"):
        print("\n" + "=" * 72)
        print("  Mesh gradient validation")
        print("=" * 72)
        mesh_metrics = test_mesh_gradients(
            vertices,
            indices,
            args.query_mode,
            args.query_count,
            args.beta,
            args.num_trials,
            args.seed,
        )
        _summarize(mesh_metrics, "Mesh")

    # ---------------------------------------------------------------
    # CSV bookkeeping
    # ---------------------------------------------------------------
    if args.label:
        unique = f"{datetime.now():%Y%m%d_%H%M%S}_{args.label}_seed{args.seed}_nt{args.num_trials}"
        write_run_to_csv(
            args.csv_path,
            unique,
            args.label,
            args,
            pn_metrics,
            tri_metrics,
            mesh_metrics,
        )
        plot_comparison(args.csv_path)


if __name__ == "__main__":
    main()
