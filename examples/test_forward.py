"""Compare fast WindingNumberEngine (BVH8) against brute-force winding numbers.

Usage:
    # Basic
    python test_forward.py --obj_file mesh.obj

    # Reproducible multi-trial
    python test_forward.py --obj_file mesh.obj --seed 42 --num_trials 5

    # Beta sweep
    python test_forward.py --obj_file mesh.obj \
        --beta_sweep "1.5,2.0,2.3,3.0,5.0,10.0" --beta_sweep_only

    # Epsilon sweep (regularization strength)
    python test_forward.py --obj_file mesh.obj \
        --epsilon_sweep "0,0.001,0.002,0.004,0.008,0.016" --epsilon_sweep_only
"""

import argparse
import csv
import math
import os
import shutil
import sys
import tarfile
import io
from dataclasses import dataclass, field
from datetime import datetime
from time import time
from typing import Optional

import igl
import numpy as np
import torch

import winder


# =============================================================================
# Utilities
# =============================================================================
DEFAULT_EPSILON = 0.004  # = 1/250, matches the library default


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate_queries(vertices, mode, num_queries, seed):
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


def mesh_to_point_surfels(vertices, indices):
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


def slice_mesh_in_half(vertices, indices, dim=2):
    centroids = vertices[indices].mean(axis=1)
    split = np.median(centroids[:, dim])
    return vertices, indices[centroids[:, dim] > split]


def drop_half_of_the_triangles(vertices, indices, seed=0):
    rng = np.random.default_rng(seed)
    choice = rng.choice(np.arange(len(indices)), len(indices) // 2, replace=False)
    return vertices, indices[choice]


# =============================================================================
# Forward metrics
# =============================================================================


def _print_histogram(data, title, unit="", n_bins=12):
    data = data[np.isfinite(data) & (data > 0)]
    if data.size == 0:
        print(f"\n  --- {title} --- (no data)")
        return

    lo_val = max(float(np.percentile(data, 1)), 1e-30)
    hi_val = float(np.percentile(data, 99.9))
    lo = math.floor(math.log10(lo_val))
    hi = math.ceil(math.log10(hi_val))
    if hi - lo < 3:
        mid = (hi + lo) / 2
        lo = math.floor(mid - 1.5)
        hi = math.ceil(mid + 1.5)

    edges = np.logspace(lo, hi, n_bins + 1)
    counts, _ = np.histogram(data, bins=edges)
    total = int(data.size)
    max_count = max(1, int(counts.max()))
    n_below = int((data < edges[0]).sum())
    n_above = int((data > edges[-1]).sum())

    print(f"\n  --- {title} ---")
    for i in range(len(counts)):
        low, high = edges[i], edges[i + 1]
        count = int(counts[i])
        pct = 100.0 * count / total
        bar = "█" * int(round(50 * count / max_count))
        print(
            f"  [{low:.1e}, {high:.1e}){unit:<2} | {bar:<50} | {count:7d} ({pct:5.2f}%)"
        )
    if n_below or n_above:
        print(f"  (out of range: {n_below} below, {n_above} above)")


def voxel_misclassification(
    fast: torch.Tensor,
    ref: torch.Tensor,
    *,
    threshold: float = 0.5,
    verbose: bool = True,
) -> float:
    fast_np = fast.detach().cpu().numpy().reshape(-1)
    ref_np = ref.detach().cpu().numpy().reshape(-1)

    ref_bin = (ref_np > threshold).astype(np.int32)
    fast_bin = (fast_np > threshold).astype(np.int32)
    mismatches = int((ref_bin != fast_bin).sum())
    total = int(ref_np.size)
    frac = mismatches / total

    if verbose:
        print(f"  Misclassifications: {mismatches}/{total} ({100 * frac:.4f}%)")
    return frac


@dataclass
class ForwardMetrics:
    label: str
    n_queries: int
    has_nan: bool
    has_inf: bool
    signal_max: float
    signal_coverage: float
    rms_abs: float
    mean_abs: float
    p50_abs: float
    p95_abs: float
    p99_abs: float
    p999_abs: float
    max_abs: float
    rms_rel_masked: float
    p99_rel_masked: float
    misclassified: int
    passed: bool = False
    failed_checks: tuple = field(default_factory=tuple)


def validate_forward(
    fast: torch.Tensor,
    ref: torch.Tensor,
    label: str,
    *,
    rms_tol: float = 1e-3,
    p99_abs_tol: float = 1e-2,
    max_abs_tol: float = 1e-1,
    signal_threshold: float = 1e-3,
    verbose: bool = True,
) -> ForwardMetrics:
    assert fast.shape == ref.shape, f"shape mismatch: {fast.shape} vs {ref.shape}"

    fast_np = fast.detach().float().cpu().numpy().reshape(-1)
    ref_np = ref.detach().float().cpu().numpy().reshape(-1)

    has_nan = bool(np.isnan(fast_np).any())
    has_inf = bool(np.isinf(fast_np).any())
    if has_nan or has_inf:
        if verbose:
            print(f"\n  [{label}] CRITICAL: NaN={has_nan} Inf={has_inf}")
        return ForwardMetrics(
            label=label,
            n_queries=fast_np.size,
            has_nan=has_nan,
            has_inf=has_inf,
            signal_max=0.0,
            signal_coverage=0.0,
            rms_abs=float("inf"),
            mean_abs=float("inf"),
            p50_abs=float("inf"),
            p95_abs=float("inf"),
            p99_abs=float("inf"),
            p999_abs=float("inf"),
            max_abs=float("inf"),
            rms_rel_masked=float("inf"),
            p99_rel_masked=float("inf"),
            misclassified=fast_np.size,
            passed=False,
            failed_checks=("nan_or_inf",),
        )

    missclassified = voxel_misclassification(fast, ref, verbose=False)

    diff = fast_np - ref_np
    abs_err = np.abs(diff)

    rms_abs = float(np.sqrt(np.mean(diff * diff)))
    mean_abs = float(abs_err.mean())
    p50 = float(np.percentile(abs_err, 50))
    p95 = float(np.percentile(abs_err, 95))
    p99 = float(np.percentile(abs_err, 99))
    p999 = float(np.percentile(abs_err, 99.9))
    mx = float(abs_err.max())

    signal_max = float(np.abs(ref_np).max())
    mask = np.abs(ref_np) > signal_threshold
    signal_coverage = float(mask.mean()) if mask.size else 0.0

    if mask.any():
        rel_m = abs_err[mask] / np.abs(ref_np[mask])
        rms_rel = float(np.sqrt(np.mean(rel_m * rel_m)))
        p99_rel = float(np.percentile(rel_m, 99))
    else:
        rms_rel = float("nan")
        p99_rel = float("nan")

    if verbose:
        print(f"\n{'=' * 72}")
        print(f"  Forward validation: {label}   [{fast_np.size} queries]")
        print(f"{'=' * 72}")
        print(f"  Signal max |Ω|             : {signal_max:.6e}")
        print(
            f"  Signal coverage (|Ω|>{signal_threshold:g}) : "
            f"{100.0 * signal_coverage:.2f}%"
        )
        print(f"  --- absolute error ---")
        print(f"  RMS                        : {rms_abs:.6e}   (tol {rms_tol:.1e})")
        print(f"  mean                       : {mean_abs:.6e}")
        print(f"  p50 / p95 / p99            : {p50:.3e} / {p95:.3e} / {p99:.3e}")
        print(
            f"  p99.9 / max                : {p999:.3e} / {mx:.3e}   "
            f"(p99 tol {p99_abs_tol:.1e}, max tol {max_abs_tol:.1e})"
        )
        if mask.any():
            print(f"  --- masked relative error ---")
            print(f"  RMS_rel / p99_rel          : {rms_rel:.3e} / {p99_rel:.3e}")
        print(f"  Voxel misclassification    : {missclassified}/{fast_np.size}")
        _print_histogram(abs_err, "Absolute error")

    checks = {
        "rms_abs": rms_abs < rms_tol,
        "p99_abs": p99 < p99_abs_tol,
        "max_abs": mx < max_abs_tol,
    }
    failed = tuple(k for k, v in checks.items() if not v)
    passed = not failed

    if verbose:
        if passed:
            print(f"\n\033[92m  ✓ PASS  [{label}]  RMS={rms_abs:.3e}\033[0m")
        else:
            print(f"\n\033[91m  ✗ FAIL  [{label}]  failed: {', '.join(failed)}\033[0m")

    return ForwardMetrics(
        label=label,
        n_queries=fast_np.size,
        has_nan=False,
        has_inf=False,
        signal_max=signal_max,
        signal_coverage=signal_coverage,
        rms_abs=rms_abs,
        mean_abs=mean_abs,
        p50_abs=p50,
        p95_abs=p95,
        p99_abs=p99,
        p999_abs=p999,
        max_abs=mx,
        rms_rel_masked=rms_rel,
        p99_rel_masked=p99_rel,
        misclassified=missclassified,
        passed=passed,
        failed_checks=failed,
    )


# =============================================================================
# Runners
# =============================================================================
def _to_cuda(x, dtype=torch.float32):
    return torch.from_numpy(np.ascontiguousarray(x)).to(dtype).cuda().contiguous()


def run_brute_force_triangle(triangles, queries, epsilon=DEFAULT_EPSILON, out=None):
    t = _to_cuda(triangles)
    q = _to_cuda(queries)
    if out is None:
        out = torch.empty([queries.shape[0]], device="cuda:0", dtype=torch.float32)
    winder.brute_force_winding_numbers_triangle_soup(
        t, q, out,
        epsilon,
        torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()
    return out


def run_fast_triangle(triangles, queries, beta, epsilon=DEFAULT_EPSILON, out=None):
    t = _to_cuda(triangles)
    q = _to_cuda(queries)
    if out is None:
        out = torch.empty([queries.shape[0]], device="cuda:0", dtype=torch.float32)
    engine = winder.WindingNumberEngine(
        t, stream=torch.cuda.current_stream().cuda_stream
    )
    engine.compute(
        q, out,
        -1.0 if beta is None else beta,
        epsilon,
        torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()
    return out


def run_brute_force_mesh(vertices, indices, queries, epsilon=DEFAULT_EPSILON, out=None):
    v = _to_cuda(vertices)
    idx = _to_cuda(indices, torch.uint32)
    q = _to_cuda(queries)
    if out is None:
        out = torch.empty([queries.shape[0]], device="cuda:0", dtype=torch.float32)
    winder.brute_force_winding_numbers_mesh(
        v, idx, q, out,
        epsilon,
        torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()
    return out


def run_fast_mesh(vertices, indices, queries, beta, epsilon=DEFAULT_EPSILON, out=None):
    v = _to_cuda(vertices)
    idx = _to_cuda(indices, dtype=torch.uint32)
    q = _to_cuda(queries)
    if out is None:
        out = torch.empty([queries.shape[0]], device="cuda:0", dtype=torch.float32)
    engine = winder.WindingNumberEngine(
        v, idx, stream=torch.cuda.current_stream().cuda_stream
    )
    engine.compute(
        q, out,
        -1.0 if beta is None else beta,
        epsilon,
        torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()
    return out


def run_brute_force_point_normal(points, scaled_normals, queries, epsilon, out=None):
    p = _to_cuda(points)
    n = _to_cuda(scaled_normals)
    q = _to_cuda(queries)
    if out is None:
        out = torch.empty([queries.shape[0]], device="cuda:0", dtype=torch.float32)
    winder.brute_force_winding_numbers_point_normal(
        p, n, q, out,
        epsilon,
        torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()
    return out


def run_fast_point_normal(points, scaled_normals, queries, beta, epsilon, out=None):
    p = _to_cuda(points)
    n = _to_cuda(scaled_normals)
    q = _to_cuda(queries)
    if out is None:
        out = torch.empty([queries.shape[0]], device="cuda:0", dtype=torch.float32)
    engine = winder.WindingNumberEngine(
        p, n, stream=torch.cuda.current_stream().cuda_stream
    )
    engine.compute(
        q, out,
        -1.0 if beta is None else beta,
        epsilon,
        torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()
    return out


def time_forward(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return float(np.median(times))


# =============================================================================
# igl reference implementation
# =============================================================================
def run_igl_triangle(vertices, indices, queries):
    """igl fast_winding_number reference. Ignores epsilon (always sharp)."""
    v = np.ascontiguousarray(vertices, dtype=np.float64)
    i = np.ascontiguousarray(indices, dtype=np.int32)
    q = np.ascontiguousarray(queries, dtype=np.float64)
    return igl.fast_winding_number(v, i, q)


# =============================================================================
# Main test drivers
# =============================================================================
def test_igl_agreement(
    vertices, indices, query_mode, query_count, beta,
    num_trials, base_seed, epsilon=DEFAULT_EPSILON, verbose=True,
):
    if epsilon > 0.01:
        if verbose:
            print(
                f"\n  [igl] skipping: igl is sharp, epsilon={epsilon:g} "
                f"is too large for a meaningful comparison."
            )
        return []

    all_metrics = []
    for trial in range(num_trials):
        seed = base_seed + trial * 1009
        set_seeds(seed)
        queries = generate_queries(vertices, query_mode, query_count, seed)

        igl_out = run_igl_triangle(vertices, indices, queries)
        igl_t = torch.from_numpy(igl_out.astype(np.float32)).cuda()

        fast = run_fast_mesh(vertices, indices, queries, beta, epsilon=epsilon)

        m_ours_vs_igl = validate_forward(
            fast, igl_t, f"ours-vs-igl[seed={seed}, beta={beta}, eps={epsilon:g}]",
            verbose=verbose,
        )
        all_metrics.append(m_ours_vs_igl)

        gt = run_brute_force_mesh(vertices, indices, queries, epsilon=epsilon)
        m_igl_vs_gt = validate_forward(
            igl_t, gt, f"igl-vs-brute[seed={seed}, eps={epsilon:g}]",
            verbose=verbose,
        )
        all_metrics.append(m_igl_vs_gt)

        if verbose:
            print(f"\n  Summary for seed={seed}:")
            print(f"    ours vs igl  : RMS={m_ours_vs_igl.rms_abs:.3e}")
            print(f"    igl vs brute : RMS={m_igl_vs_gt.rms_abs:.3e}")

    return all_metrics


def test_triangle_forward(
    vertices, indices, query_mode, query_count, beta,
    num_trials, base_seed, epsilon=DEFAULT_EPSILON,
    timing=False, verbose=True,
):
    triangles = vertices[indices]
    all_metrics = []
    for trial in range(num_trials):
        seed = base_seed + trial * 1009
        set_seeds(seed)
        queries = generate_queries(vertices, query_mode, query_count, seed)

        gt = run_brute_force_triangle(triangles, queries, epsilon=epsilon)
        fast = run_fast_triangle(triangles, queries, beta, epsilon=epsilon)

        m = validate_forward(
            fast, gt,
            f"Triangle[seed={seed}, beta={beta}, eps={epsilon:g}]",
            verbose=verbose,
        )
        all_metrics.append(m)

    if timing:
        set_seeds(base_seed)
        queries = generate_queries(vertices, query_mode, query_count, base_seed)
        bf_ms = time_forward(lambda: run_brute_force_triangle(triangles, queries, epsilon))
        fast_ms = time_forward(lambda: run_fast_triangle(triangles, queries, beta, epsilon))
        print(
            f"\n  Timing (median of 10 ms): BF={bf_ms:.3f}  fast={fast_ms:.3f}  "
            f"speedup={bf_ms / fast_ms:.2f}x"
        )
    return all_metrics


def test_mesh_forward(
    vertices, indices, query_mode, query_count, beta,
    num_trials, base_seed, epsilon=DEFAULT_EPSILON,
    timing=False, verbose=True,
):
    all_metrics = []
    for trial in range(num_trials):
        seed = base_seed + trial * 1009
        set_seeds(seed)
        queries = generate_queries(vertices, query_mode, query_count, seed)

        gt = run_brute_force_mesh(vertices, indices, queries, epsilon=epsilon)
        fast = run_fast_mesh(vertices, indices, queries, beta, epsilon=epsilon)

        m = validate_forward(
            fast, gt,
            f"Mesh[seed={seed}, beta={beta}, eps={epsilon:g}]",
            verbose=verbose,
        )
        all_metrics.append(m)

    if timing:
        set_seeds(base_seed)
        queries = generate_queries(vertices, query_mode, query_count, base_seed)
        bf_ms = time_forward(lambda: run_brute_force_mesh(vertices, indices, queries, epsilon))
        fast_ms = time_forward(lambda: run_fast_mesh(vertices, indices, queries, beta, epsilon))
        print(
            f"\n  Timing (median of 10 ms): BF={bf_ms:.3f}  fast={fast_ms:.3f}  "
            f"speedup={bf_ms / fast_ms:.2f}x"
        )
    return all_metrics


def test_point_normal_forward(
    points, scaled_normals, query_mode, query_count, beta, epsilon,
    num_trials, base_seed, timing=False, verbose=True,
):
    all_metrics = []
    for trial in range(num_trials):
        seed = base_seed + trial * 1009
        set_seeds(seed)
        queries = generate_queries(points, query_mode, query_count, seed)

        gt = run_brute_force_point_normal(points, scaled_normals, queries, epsilon)
        fast = run_fast_point_normal(points, scaled_normals, queries, beta, epsilon)
        m = validate_forward(
            fast, gt,
            f"PointNormal[seed={seed}, beta={beta}, eps={epsilon:g}]",
            verbose=verbose,
        )
        all_metrics.append(m)

    if timing:
        set_seeds(base_seed)
        queries = generate_queries(points, query_mode, query_count, base_seed)
        bf_ms = time_forward(
            lambda: run_brute_force_point_normal(points, scaled_normals, queries, epsilon)
        )
        fast_ms = time_forward(
            lambda: run_fast_point_normal(points, scaled_normals, queries, beta, epsilon)
        )
        print(
            f"\n  Timing (median of 10 ms): BF={bf_ms:.3f}  fast={fast_ms:.3f}  "
            f"speedup={bf_ms / fast_ms:.2f}x"
        )
    return all_metrics


# =============================================================================
# Beta sweep
# =============================================================================
def run_beta_sweep(
    vertices, indices, points, scaled_normals,
    query_mode, query_count, beta_values,
    num_trials, base_seed, epsilon,
    rms_tol=1e-2, p99_abs_tol=1e-2, max_abs_tol=1e-1,
    scale_tol_with_beta=False, reference_beta=2.3,
):
    print("\n" + "=" * 100)
    print(f"  BETA SWEEP — forward accuracy and speed vs beta  (epsilon = {epsilon:g})")
    print("=" * 100)

    triangles = vertices[indices]

    def _tol_for(beta):
        if not scale_tol_with_beta:
            return rms_tol, p99_abs_tol, max_abs_tol
        s = (reference_beta / max(beta, 1e-3)) ** 2
        return rms_tol * s, p99_abs_tol * s, max_abs_tol * s

    set_seeds(base_seed)
    timing_q = generate_queries(vertices, query_mode, query_count, base_seed)

    bf_tri_ms = time_forward(
        lambda: run_brute_force_triangle(triangles, timing_q, epsilon)
    )
    bf_mesh_ms = time_forward(
        lambda: run_brute_force_mesh(vertices, indices, timing_q, epsilon)
    )
    bf_pn_ms = time_forward(
        lambda: run_brute_force_point_normal(points, scaled_normals, timing_q, epsilon)
    )

    print(f"\n  Brute force (median of 10), eps = {epsilon:g}:")
    print(f"    triangle     = {bf_tri_ms:8.3f} ms")
    print(f"    mesh         = {bf_mesh_ms:8.3f} ms")
    print(f"    point_normal = {bf_pn_ms:8.3f} ms")

    table = {b: {"triangle": [], "mesh": [], "point_normal": []} for b in beta_values}

    for beta in beta_values:
        for trial in range(num_trials):
            seed = base_seed + trial * 1009
            set_seeds(seed)
            q = generate_queries(vertices, query_mode, query_count, seed)
            rms_t, p99_t, max_t = _tol_for(beta)

            gt = run_brute_force_triangle(triangles, q, epsilon)
            fast = run_fast_triangle(triangles, q, beta, epsilon)
            table[beta]["triangle"].append(validate_forward(
                fast, gt, f"tri b={beta} s={seed}",
                rms_tol=rms_t, p99_abs_tol=p99_t, max_abs_tol=max_t, verbose=False,
            ))

            gt = run_brute_force_mesh(vertices, indices, q, epsilon)
            fast = run_fast_mesh(vertices, indices, q, beta, epsilon)
            table[beta]["mesh"].append(validate_forward(
                fast, gt, f"mesh b={beta} s={seed}",
                rms_tol=rms_t, p99_abs_tol=p99_t, max_abs_tol=max_t, verbose=False,
            ))

            gt = run_brute_force_point_normal(points, scaled_normals, q, epsilon)
            fast = run_fast_point_normal(points, scaled_normals, q, beta, epsilon)
            table[beta]["point_normal"].append(validate_forward(
                fast, gt, f"pn b={beta} s={seed}",
                rms_tol=rms_t, p99_abs_tol=p99_t, max_abs_tol=max_t, verbose=False,
            ))

    print(
        f"\n{'beta':>6} | {'geometry':<14} | "
        f"{'RMS(med)':>11} | {'p99_abs(med)':>13} | {'max_abs(med)':>13} | "
        f"{'BF(ms)':>8} | {'fast(ms)':>9} | {'speedup':>8} | {'pass':>5}"
    )
    print("-" * 122)

    bf_ms_map = {"triangle": bf_tri_ms, "mesh": bf_mesh_ms, "point_normal": bf_pn_ms}

    for beta in beta_values:
        for geom in ("triangle", "mesh", "point_normal"):
            ms = table[beta][geom]
            if not ms:
                continue
            rms_med = float(np.median([m.rms_abs for m in ms]))
            p99_med = float(np.median([m.p99_abs for m in ms]))
            max_med = float(np.median([m.max_abs for m in ms]))
            all_pass = all(m.passed for m in ms)

            def _timed(g=geom, b=beta):
                if g == "triangle":
                    return run_fast_triangle(triangles, timing_q, b, epsilon)
                if g == "mesh":
                    return run_fast_mesh(vertices, indices, timing_q, b, epsilon)
                return run_fast_point_normal(points, scaled_normals, timing_q, b, epsilon)

            fast_ms = time_forward(_timed, warmup=2, iters=5)
            bf_ms = bf_ms_map[geom]
            speedup = bf_ms / fast_ms if fast_ms > 0 else float("nan")

            print(
                f"{beta:>6.3f} | {geom:<14} | "
                f"{rms_med:>11.3e} | {p99_med:>13.3e} | {max_med:>13.3e} | "
                f"{bf_ms:>8.3f} | {fast_ms:>9.3f} | {speedup:>7.2f}x | "
                f"{'YES' if all_pass else 'NO':>5}"
            )

    print("-" * 122)

    prev = {"triangle": None, "mesh": None, "point_normal": None}
    non_monotone = []
    for beta in sorted(beta_values):
        for geom in ("triangle", "mesh", "point_normal"):
            ms = table[beta][geom]
            if not ms:
                continue
            rms_med = float(np.median([m.rms_abs for m in ms]))
            p = prev[geom]
            if p is not None and p > 0 and rms_med > p * 1.5:
                non_monotone.append((geom, beta, p, rms_med))
            prev[geom] = rms_med if p is None else min(p, rms_med)

    if non_monotone:
        print("\n\033[91mNon-monotone RMS detected (>1.5x increase):\033[0m")
        for geom, beta, p, cur in non_monotone:
            print(f"  {geom}: beta={beta:.3f}  {p:.3e} -> {cur:.3e}")
    else:
        print("\n\033[92mMonotone non-increasing RMS with beta.\033[0m")

    print("\n  Minimum beta to achieve a given median RMS:")
    for target in (1e-2, 1e-3, 1e-4, 1e-5):
        best = None
        for beta in sorted(beta_values):
            for geom in ("triangle", "mesh", "point_normal"):
                ms = table[beta][geom]
                if not ms:
                    continue
                rms_med = float(np.median([m.rms_abs for m in ms]))
                if rms_med < target:
                    best = beta if best is None else min(best, beta)
                    break
        print(f"    RMS < {target:.0e}:  beta >= "
              f"{best if best is not None else 'not reached in sweep'}")

    return table


# =============================================================================
# Epsilon sweep
# =============================================================================
def run_epsilon_sweep(
    vertices, indices, points, scaled_normals,
    query_mode, query_count, epsilon_values,
    num_trials, base_seed, beta,
    rms_tol=1e-2, p99_abs_tol=1e-2, max_abs_tol=1e-1,
):
    """Sweep the regularization strength at a fixed beta.

    At each epsilon:
        - brute force uses the SAME epsilon (so it computes the regularized
          ground truth, matching what the fast engine approximates)
        - we report the approximation error, the median voxel misclassification,
          and the fast/brute timings
    """
    print("\n" + "=" * 100)
    print(f"  EPSILON SWEEP — regularization strength vs accuracy  (beta = {beta})")
    print("=" * 100)

    triangles = vertices[indices]
    set_seeds(base_seed)
    timing_q = generate_queries(vertices, query_mode, query_count, base_seed)

    table = {e: {"triangle": [], "mesh": [], "point_normal": []} for e in epsilon_values}

    for eps in epsilon_values:
        for trial in range(num_trials):
            seed = base_seed + trial * 1009
            set_seeds(seed)
            q = generate_queries(vertices, query_mode, query_count, seed)

            # Triangle
            gt = run_brute_force_triangle(triangles, q, eps)
            fast = run_fast_triangle(triangles, q, beta, eps)
            table[eps]["triangle"].append(validate_forward(
                fast, gt, f"tri eps={eps:g} s={seed}",
                rms_tol=rms_tol, p99_abs_tol=p99_abs_tol, max_abs_tol=max_abs_tol,
                verbose=False,
            ))

            # Mesh
            gt = run_brute_force_mesh(vertices, indices, q, eps)
            fast = run_fast_mesh(vertices, indices, q, beta, eps)
            table[eps]["mesh"].append(validate_forward(
                fast, gt, f"mesh eps={eps:g} s={seed}",
                rms_tol=rms_tol, p99_abs_tol=p99_abs_tol, max_abs_tol=max_abs_tol,
                verbose=False,
            ))

            # Point-normal
            gt = run_brute_force_point_normal(points, scaled_normals, q, eps)
            fast = run_fast_point_normal(points, scaled_normals, q, beta, eps)
            table[eps]["point_normal"].append(validate_forward(
                fast, gt, f"pn eps={eps:g} s={seed}",
                rms_tol=rms_tol, p99_abs_tol=p99_abs_tol, max_abs_tol=max_abs_tol,
                verbose=False,
            ))

    print(
        f"\n{'epsilon':>9} | {'geometry':<14} | "
        f"{'RMS(med)':>11} | {'p99_abs(med)':>13} | {'max_abs(med)':>13} | "
        f"{'misclass(med)':>14} | {'fast(ms)':>9} | {'pass':>5}"
    )
    print("-" * 128)

    for eps in epsilon_values:
        for geom in ("triangle", "mesh", "point_normal"):
            ms = table[eps][geom]
            if not ms:
                continue
            rms_med = float(np.median([m.rms_abs for m in ms]))
            p99_med = float(np.median([m.p99_abs for m in ms]))
            max_med = float(np.median([m.max_abs for m in ms]))
            mis_med = int(np.median([m.misclassified for m in ms]))
            all_pass = all(m.passed for m in ms)

            def _timed(g=geom, e=eps):
                if g == "triangle":
                    return run_fast_triangle(triangles, timing_q, beta, e)
                if g == "mesh":
                    return run_fast_mesh(vertices, indices, timing_q, beta, e)
                return run_fast_point_normal(points, scaled_normals, timing_q, beta, e)

            fast_ms = time_forward(_timed, warmup=2, iters=5)

            print(
                f"{eps:>9.5f} | {geom:<14} | "
                f"{rms_med:>11.3e} | {p99_med:>13.3e} | {max_med:>13.3e} | "
                f"{mis_med:>14d} | {fast_ms:>9.3f} | "
                f"{'YES' if all_pass else 'NO':>5}"
            )

    print("-" * 128)
    print("\n  Notes:")
    print("    - Ground truth is the SAME regularized field as what fast approximates.")
    print("    - misclass = voxel sign errors at threshold 0.5 (FWN paper metric).")
    print("    - Larger epsilon → smoother kernel → smaller approximation error,")
    print("      but the field itself deviates more from the sharp one.")
    return table


# =============================================================================
# CSV
# =============================================================================
CSV_FIELDS = [
    "timestamp",
    "label",
    "unique_label",
    "geometry_type",
    "query_mode",
    "query_count",
    "beta",
    "epsilon",
    "seed",
    "num_trials",
    "tri_rms_med",
    "tri_max_worst",
    "mesh_rms_med",
    "mesh_max_worst",
    "pn_rms_med",
    "pn_max_worst",
    "tri_passed",
    "mesh_passed",
    "pn_passed",
]


def _agg(metrics_list, attr, agg="median"):
    if not metrics_list:
        return ""
    vals = [getattr(m, attr) for m in metrics_list if math.isfinite(getattr(m, attr))]
    if not vals:
        return ""
    fn = np.median if agg == "median" else np.max
    return f"{float(fn(vals)):.6e}"


def _all_pass(metrics_list):
    if not metrics_list:
        return ""
    return str(all(m.passed for m in metrics_list))


def write_run_to_csv(
    csv_path, unique_label, raw_label, args, tri_metrics, mesh_metrics, pn_metrics
):
    file_exists = os.path.exists(csv_path)
    if file_exists:
        with open(csv_path, newline="") as f:
            header = next(csv.reader(f), None)
        if header != CSV_FIELDS:
            backup = f"{csv_path}.bak.{datetime.now():%Y%m%d_%H%M%S}"
            shutil.move(csv_path, backup)
            print(f"[CSV] Header changed; archived to {backup}")
            file_exists = False

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "label": raw_label,
        "unique_label": unique_label,
        "geometry_type": args.geometry_type,
        "query_mode": args.query_mode,
        "query_count": args.query_count,
        "beta": args.beta if args.beta is not None else -1.0,
        "epsilon": args.epsilon,
        "seed": args.seed,
        "num_trials": args.num_trials,
        "tri_rms_med": _agg(tri_metrics, "rms_abs", "median"),
        "tri_max_worst": _agg(tri_metrics, "max_abs", "max"),
        "mesh_rms_med": _agg(mesh_metrics, "rms_abs", "median"),
        "mesh_max_worst": _agg(mesh_metrics, "max_abs", "max"),
        "pn_rms_med": _agg(pn_metrics, "rms_abs", "median"),
        "pn_max_worst": _agg(pn_metrics, "max_abs", "max"),
        "tri_passed": _all_pass(tri_metrics),
        "mesh_passed": _all_pass(mesh_metrics),
        "pn_passed": _all_pass(pn_metrics),
    }
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            w.writeheader()
        w.writerow(row)
    print(f"[CSV] Appended to {csv_path}")


# =============================================================================
# CLI
# =============================================================================
def _summarize(metrics_list, name):
    if not metrics_list:
        return None
    rms_vals = [m.rms_abs for m in metrics_list if math.isfinite(m.rms_abs)]
    max_vals = [m.max_abs for m in metrics_list if math.isfinite(m.max_abs)]
    any_failed = any(not m.passed for m in metrics_list)

    mean_rms = float(np.mean(rms_vals)) if rms_vals else float("nan")
    worst_max = float(np.max(max_vals)) if max_vals else float("nan")
    status = "\033[91mFAIL\033[0m" if any_failed else "\033[92mPASS\033[0m"
    n_pass = sum(m.passed for m in metrics_list)

    print(
        f"\n[{name}] mean RMS = {mean_rms:.3e},  worst max_abs = {worst_max:.3e},  "
        f"overall: {status}  ({n_pass}/{len(metrics_list)} trials passed)"
    )
    return mean_rms


def main():
    p = argparse.ArgumentParser(
        description="Validate fast forward winding-number engine."
    )
    p.add_argument("--obj_file", type=str, required=True)
    p.add_argument(
        "--geometry_type",
        choices=["Triangle", "Mesh", "PointNormal", "All"],
        default="All",
    )
    p.add_argument("--query_mode", choices=["random", "grid"], default="random")
    p.add_argument("--query_count", type=int, default=10000)
    p.add_argument(
        "--epsilon",
        type=float,
        default=DEFAULT_EPSILON,
        help=f"Regularization strength (fraction of scene diagonal). "
             f"Default {DEFAULT_EPSILON} (= 1/250). Use 0 for the sharp kernel.",
    )
    p.add_argument("--beta", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_trials", type=int, default=1)
    p.add_argument("--beta_sweep", type=str, default="")
    p.add_argument("--beta_sweep_only", action="store_true")
    p.add_argument(
        "--epsilon_sweep",
        type=str,
        default="",
        help="Comma-separated list of epsilon values to sweep, e.g. "
             "'0,0.001,0.002,0.004,0.008,0.016'. Uses --beta for all runs.",
    )
    p.add_argument("--epsilon_sweep_only", action="store_true")
    p.add_argument("--timing", action="store_true")
    p.add_argument("--label", type=str, default=None)
    p.add_argument("--csv_path", type=str, default="forward_bench.csv")
    p.add_argument(
        "--test_igl",
        action="store_true",
        help="Compare against libigl FWN reference (requires igl python bindings). "
             "Only meaningful at epsilon <= 0.01, since igl is sharp.",
    )

    args = p.parse_args()
    set_seeds(args.seed)

    print(f"Loading mesh: {args.obj_file}")
    vertices, _, _, indices, _, _ = igl.readOBJ(args.obj_file)
    vertices, indices = slice_mesh_in_half(vertices, indices)
    vertices, indices = drop_half_of_the_triangles(vertices, indices, seed=args.seed)
    print(f"Mesh: {len(indices)} triangles, {len(vertices)} vertices")
    print(f"epsilon = {args.epsilon:g}")

    points, normals, areas = mesh_to_point_surfels(vertices, indices)
    scaled_normals = normals * areas[:, None]
    epsilon = args.epsilon
    beta = args.beta if args.beta is not None else 2.3  # library default

    if args.epsilon_sweep:
        eps_values = [
            float(x) for x in args.epsilon_sweep.replace(";", ",").split(",") if x.strip()
        ]
        run_epsilon_sweep(
            vertices, indices, points, scaled_normals,
            args.query_mode, args.query_count, eps_values,
            args.num_trials, args.seed, beta,
        )
        if args.epsilon_sweep_only:
            return

    if args.beta_sweep:
        betas = [
            float(x) for x in args.beta_sweep.replace(";", ",").split(",") if x.strip()
        ]
        run_beta_sweep(
            vertices, indices, points, scaled_normals,
            args.query_mode, args.query_count, betas,
            args.num_trials, args.seed, epsilon,
        )
        if args.beta_sweep_only:
            return

    tri_metrics = mesh_metrics = pn_metrics = None

    if args.geometry_type in ("Triangle", "All"):
        print("\n" + "=" * 72)
        print("  Triangle forward")
        print("=" * 72)
        tri_metrics = test_triangle_forward(
            vertices, indices,
            args.query_mode, args.query_count, args.beta,
            args.num_trials, args.seed,
            epsilon=epsilon, timing=args.timing,
        )
        _summarize(tri_metrics, "Triangle")

    if args.geometry_type in ("Mesh", "All"):
        print("\n" + "=" * 72)
        print("  Mesh forward")
        print("=" * 72)
        mesh_metrics = test_mesh_forward(
            vertices, indices,
            args.query_mode, args.query_count, args.beta,
            args.num_trials, args.seed,
            epsilon=epsilon, timing=args.timing,
        )
        _summarize(mesh_metrics, "Mesh")

    if args.geometry_type in ("PointNormal", "All"):
        print("\n" + "=" * 72)
        print("  PointNormal forward")
        print("=" * 72)
        pn_metrics = test_point_normal_forward(
            points, scaled_normals,
            args.query_mode, args.query_count, args.beta, epsilon,
            args.num_trials, args.seed,
            timing=args.timing,
        )
        _summarize(pn_metrics, "PointNormal")

    if args.test_igl:
        print("\n" + "=" * 72)
        print("  igl cross-validation")
        print("=" * 72)
        try:
            igl_metrics = test_igl_agreement(
                vertices, indices,
                args.query_mode, args.query_count, args.beta,
                args.num_trials, args.seed,
                epsilon=epsilon,
            )
            ours_vs_igl = [m for m in igl_metrics if "ours-vs-igl" in m.label]
            igl_vs_gt = [m for m in igl_metrics if "igl-vs-brute" in m.label]
            if ours_vs_igl:
                rms_o = float(np.median([m.rms_abs for m in ours_vs_igl]))
                print(f"\n  [ours vs igl]  median RMS = {rms_o:.3e}")
            if igl_vs_gt:
                rms_i = float(np.median([m.rms_abs for m in igl_vs_gt]))
                print(f"  [igl vs brute] median RMS = {rms_i:.3e}")
        except Exception as e:
            print(f"  igl comparison failed: {e}")

    if args.label:
        unique = f"{datetime.now():%Y%m%d_%H%M%S}_{args.label}_s{args.seed}_nt{args.num_trials}"
        write_run_to_csv(
            args.csv_path, unique, args.label, args,
            tri_metrics, mesh_metrics, pn_metrics,
        )


if __name__ == "__main__":
    main()
