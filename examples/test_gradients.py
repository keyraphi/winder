"""Validate the fast (Barnes-Hut) winding-number gradient engine against brute force.

Compares the analytic geometry gradients (and the scalar gradient with respect
to the regularization length epsilon) of the fast engine against the exact
brute-force reference.

Usage examples:
    # Basic validation, all geometry types
    python validate_gradients.py --obj_file mesh.obj

    # Reproducible multi-trial run with seed
    python validate_gradients.py --obj_file mesh.obj --seed 42 --num_trials 5

    # Beta sweep: verify monotone and bounded approximation error
    python validate_gradients.py --obj_file mesh.obj \
        --beta_sweep "-1,0.5,1.0,2.0,3.0" --num_trials 3

    # Epsilon sweep: regularization strength vs accuracy
    python validate_gradients.py --obj_file mesh.obj \
        --epsilon_sweep "0,0.001,0.002,0.004,0.008,0.016" --epsilon_sweep_only
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from time import time
from typing import Callable, Optional

import igl
import numpy as np
import torch

import winder


DEFAULT_EPSILON = 0.004  # = 1/250, matches the library default


# =========================================================================
# CUDA helpers
# =========================================================================


def cuda_timer(fn: Callable[[], None], warmup: int = 3, iters: int = 10) -> float:
    """Return the median wall time (ms) of `fn` measured with CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times_ms: list[float] = []
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


def mesh_to_point_surfels(
    vertices: np.ndarray, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a triangle mesh to per-face point surfels (points, normals, areas)."""
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
    """Sample queries either on a regular grid or uniformly inside the AABB."""
    rng = np.random.default_rng(seed)
    min_box = vertices.min(axis=0)
    max_box = vertices.max(axis=0)

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
    """Random scalar adjoint grad_output of length `query_count`."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(query_count, generator=g) * scale).to(device)


def set_seeds(seed: int) -> None:
    """Seed numpy, torch and CUDA RNGs."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


# =========================================================================
# Dataclasses
# =========================================================================


@dataclass
class GradientMetrics:
    """Metrics produced by `validate_gradients` for one vector-valued gradient."""

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
    mag_ratio_mean: float = float("nan")
    mag_ratio_min: float = float("nan")
    mag_ratio_p01: float = float("nan")
    mag_ratio_p50: float = float("nan")
    mag_ratio_p99: float = float("nan")
    mag_ratio_max: float = float("nan")
    worst_idxs: list[int] = field(default_factory=list)
    passed: bool = False
    failed_checks: tuple = field(default_factory=tuple)


@dataclass
class ScalarGradientMetrics:
    """Metrics produced by `validate_scalar_gradient` for the (1,) eps gradient."""

    label: str
    ref_val: float
    fast_val: float
    abs_err: float
    rel_err: float
    passed: bool
    failed_checks: tuple = field(default_factory=tuple)


# =========================================================================
# Vector gradient validator
# =========================================================================


def _safe_quantile(x: np.ndarray, q: float) -> float:
    return float("nan") if x.size == 0 else float(np.quantile(x, q))


def print_ascii_histogram(
    data: np.ndarray, title: str, unit: str = "", n_bins: int = 12
) -> None:
    """Print a log-scale ASCII histogram of strictly positive `data`."""
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
    rel_norm_tol_strict: float = 5e-4,
    rel_norm_tol_loose: float = 1e-2,
    cosine_tol_strict: float = 0.999,
    cosine_tol_loose: float = 0.99,
    masked_p99_rel_tol: float = 5e-1,
    per_vec_cos_p01_tol: float = 0.99,
    mag_ratio_p50_range_strict: tuple[float, float] = (0.99, 1.01),
    mag_ratio_p50_range_loose: tuple[float, float] = (0.90, 1.10),
    mag_ratio_p01_tol_strict: float = 0.95,
    mag_ratio_p01_tol_loose: float = 0.50,
    worst_vectors_n: int = 5,
    verbose: bool = True,
) -> GradientMetrics:
    """Compare two vector gradient fields and return a detailed metric bundle."""
    assert fast_grads.shape == ref_grads.shape

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

        mag_ratio = fn[vmask] / (rn[vmask] + 1e-30)
        mag_ratio_mean = float(mag_ratio.mean())
        mag_ratio_min = float(mag_ratio.min())
        mag_ratio_p01 = _safe_quantile(mag_ratio, 0.01)
        mag_ratio_p50 = _safe_quantile(mag_ratio, 0.50)
        mag_ratio_p99 = _safe_quantile(mag_ratio, 0.99)
        mag_ratio_max = float(mag_ratio.max())
    else:
        cos_mean = cos_p01 = cos_min = float("nan")
        ang_p99 = ang_max = float("nan")
        mag_ratio_mean = mag_ratio_min = float("nan")
        mag_ratio_p01 = mag_ratio_p50 = float("nan")
        mag_ratio_p99 = mag_ratio_max = float("nan")

    rel_norm_tol = rel_norm_tol_strict if strict else rel_norm_tol_loose
    cosine_tol = cosine_tol_strict if strict else cosine_tol_loose
    mag_range = mag_ratio_p50_range_strict if strict else mag_ratio_p50_range_loose
    mag_p01_tol = mag_ratio_p01_tol_strict if strict else mag_ratio_p01_tol_loose

    def _ok(v: float, cmp: Callable[[float, float], bool], thr: float) -> bool:
        return True if (v is None or math.isnan(v)) else cmp(v, thr)

    def _ok_range(v: float, lo: float, hi: float) -> bool:
        return True if (v is None or math.isnan(v)) else (lo <= v <= hi)

    checks = {
        "global_cosine": _ok(cos_flat, lambda a, b: a > b, cosine_tol),
        "rel_norm": _ok(rel_norm, lambda a, b: a < b, rel_norm_tol),
        "masked_p99_rel": _ok(p99, lambda a, b: a < b, masked_p99_rel_tol),
        "per_vec_cos_p01": _ok(cos_p01, lambda a, b: a > b, per_vec_cos_p01_tol),
        "mag_ratio_p50": _ok_range(mag_ratio_p50, mag_range[0], mag_range[1]),
        "mag_ratio_p01": _ok(mag_ratio_p01, lambda a, b: a > b, mag_p01_tol),
    }
    failed = tuple(k for k, v in checks.items() if not v)
    passed = not failed

    worst_idxs: list[int] = []
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
        print("  --- masked scalar rel err ---")
        print(
            f"  mean / p50 / p95 / p99     : "
            f"{mean_rel:.3e} / {p50:.3e} / {p95:.3e} / {p99:.3e}"
        )
        print(f"  max                        : {mx:.3e}")
        print("  --- per-vector ---")
        print(
            f"  cos mean / p01 / min       : "
            f"{cos_mean:.6f} / {cos_p01:.6f} / {cos_min:.6f}"
        )
        print(f"  angle p99 / max (deg)      : {ang_p99:.4f} / {ang_max:.4f}")
        print("  --- per-vector magnitude ratio ||fast||/||ref|| ---")
        print(
            f"  mean / p01 / p50 / p99     : "
            f"{mag_ratio_mean:.6f} / {mag_ratio_p01:.6f} / "
            f"{mag_ratio_p50:.6f} / {mag_ratio_p99:.6f}"
        )
        print(f"  min / max                  : {mag_ratio_min:.6f} / {mag_ratio_max:.6f}")
        print(f"  p50 window (tol)           : [{mag_range[0]:.3f}, {mag_range[1]:.3f}]")
        print(f"  p01 lower bound (tol)      : {mag_p01_tol:.3f}")

        if rel_m.size:
            print_ascii_histogram(rel_m, "Log10 relative error")
        if vmask.any():
            print_ascii_histogram(ang, "Angular error (deg)")
            print_ascii_histogram(mag_ratio, "Magnitude ratio ||fast||/||ref||")

        if failed and vmask.any() and worst_vectors_n > 0:
            vmask_idx = np.nonzero(vmask)[0]

            worst_by_cos_order = np.argsort(cos)[:worst_vectors_n]
            worst_by_mag_order = np.argsort(
                -np.abs(np.log(np.clip(mag_ratio, 1e-30, None)))
            )[:worst_vectors_n]

            print(f"\n  --- worst {worst_vectors_n} primitives by direction (1 - cos) ---")
            for k in worst_by_cos_order:
                prim_idx = int(vmask_idx[k])
                worst_idxs.append(prim_idx)
                print(
                    f"    [{prim_idx:6d}] "
                    f"cos={cos[k]:.6f}  ang={ang[k]:8.4f}°  "
                    f"||ref||={rn[vmask][k]:.4e}  ||fast||={fn[vmask][k]:.4e}  "
                    f"mag={mag_ratio[k]:.4f}"
                )
                print(f"             ref ={b[k]}")
                print(f"             fast={a[k]}")

            print(f"\n  --- worst {worst_vectors_n} primitives by magnitude ratio ---")
            for k in worst_by_mag_order:
                prim_idx = int(vmask_idx[k])
                print(
                    f"    [{prim_idx:6d}] "
                    f"mag={mag_ratio[k]:.6f}  "
                    f"||ref||={rn[vmask][k]:.4e}  ||fast||={fn[vmask][k]:.4e}  "
                    f"cos={cos[k]:.6f}"
                )
                print(f"             ref ={b[k]}")
                print(f"             fast={a[k]}")

        if passed:
            print(f"\n\033[92m  ✓ PASS  [{label}]\033[0m")
        else:
            print(f"\n\033[91m  ✗ FAIL  [{label}]  failed: {', '.join(failed)}\033[0m")

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
        mag_ratio_mean=mag_ratio_mean,
        mag_ratio_min=mag_ratio_min,
        mag_ratio_p01=mag_ratio_p01,
        mag_ratio_p50=mag_ratio_p50,
        mag_ratio_p99=mag_ratio_p99,
        mag_ratio_max=mag_ratio_max,
        worst_idxs=worst_idxs,
        passed=passed,
        failed_checks=failed,
    )


# =========================================================================
# Scalar (epsilon) gradient validator
# =========================================================================


def validate_scalar_gradient(
    fast_grad: torch.Tensor,
    ref_grad: torch.Tensor,
    label: str,
    *,
    abs_tol: float = 1e-3,
    rel_tol: float = 1e-2,
    abs_floor: float = 1e-2,
    verbose: bool = True,
) -> ScalarGradientMetrics:
    """Compare two scalar (shape (1,)) gradients with a signal-aware denominator.

    Passes if either the absolute error is small, or the relative error is small
    relative to `max(|ref|, abs_floor)`. The floor prevents a false pass when the
    reference value is near zero but the absolute difference is nevertheless
    significant for the given kernel.
    """
    fast_val = float(fast_grad.detach().float().reshape(-1)[0].item())
    ref_val = float(ref_grad.detach().float().reshape(-1)[0].item())

    if math.isnan(fast_val) or math.isnan(ref_val):
        if verbose:
            print(f"\n  [eps-grad {label}] CRITICAL: NaN (fast={fast_val}, ref={ref_val})")
        return ScalarGradientMetrics(
            label=label,
            ref_val=ref_val,
            fast_val=fast_val,
            abs_err=float("nan"),
            rel_err=float("nan"),
            passed=False,
            failed_checks=("nan",),
        )

    abs_err = abs(fast_val - ref_val)
    denom = max(abs(ref_val), abs_floor)
    rel_err = abs_err / denom
    passed = (abs_err < abs_tol) or (rel_err < rel_tol)
    failed = () if passed else ("abs_or_rel_tol",)

    if verbose:
        print(f"\n--- epsilon gradient: {label} ---")
        print(f"  Reference dL/d(eps)     : {ref_val:+.6e}")
        print(f"  Fast      dL/d(eps)     : {fast_val:+.6e}")
        print(f"  Absolute error          : {abs_err:.6e}   (tol {abs_tol:.1e})")
        print(f"  Relative error          : {rel_err:.6e}   (tol {rel_tol:.1e})")
        if passed:
            print(f"\033[92m  ✓ PASS  [{label}]\033[0m")
        else:
            print(f"\033[91m  ✗ FAIL  [{label}]  failed: {', '.join(failed)}\033[0m")

    return ScalarGradientMetrics(
        label=label,
        ref_val=ref_val,
        fast_val=fast_val,
        abs_err=abs_err,
        rel_err=rel_err,
        passed=passed,
        failed_checks=failed,
    )


# =========================================================================
# Low-level runners
# =========================================================================


def _to_cuda(x: np.ndarray, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(x))
    if dtype is not None:
        t = t.to(dtype)
    return t.to("cuda:0").contiguous()


def _stream() -> int:
    return torch.cuda.current_stream().cuda_stream


def _empty_eps_grad(device: str = "cuda:0") -> torch.Tensor:
    return torch.zeros(1, dtype=torch.float32, device=device)


def run_brute_force_triangle_gradients(
    triangles: np.ndarray,
    queries: np.ndarray,
    grad_output: torch.Tensor,
    epsilon: float = DEFAULT_EPSILON,
    out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (geometry_grads, epsilon_grad) from the brute-force triangle kernel."""
    t = _to_cuda(triangles, torch.float32)
    q = _to_cuda(queries, torch.float32)
    g = grad_output if grad_output.is_cuda else grad_output.to("cuda:0")
    g = g.contiguous()
    if out is None:
        out = torch.empty(
            [triangles.shape[0], 3, 3], device="cuda:0", dtype=torch.float32
        )
    eps_out = _empty_eps_grad()
    winder.brute_force_gradients_triangle_soup(
        g, t, q, out, eps_out, epsilon, _stream()
    )
    torch.cuda.synchronize()
    return out, eps_out


def run_fast_triangle_gradients(
    triangles: np.ndarray,
    queries: np.ndarray,
    grad_output: torch.Tensor,
    beta: Optional[float],
    epsilon: float = DEFAULT_EPSILON,
    out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (geometry_grads, epsilon_grad) from the fast triangle engine."""
    t = _to_cuda(triangles, torch.float32)
    q = _to_cuda(queries, torch.float32)
    g = grad_output if grad_output.is_cuda else grad_output.to("cuda:0")
    g = g.contiguous()
    if out is None:
        out = torch.empty(
            [triangles.shape[0], 3, 3], device="cuda:0", dtype=torch.float32
        )
    eps_out = _empty_eps_grad()
    engine = winder.GradientEngine(q, g, stream=_stream())
    engine.compute_triangle_soup(
        t, out, eps_out, -1.0 if beta is None else beta, epsilon, _stream()
    )
    torch.cuda.synchronize()
    return out, eps_out


def run_brute_force_mesh_gradients(
    vertices: np.ndarray,
    indices: np.ndarray,
    queries: np.ndarray,
    grad_output: torch.Tensor,
    epsilon: float = DEFAULT_EPSILON,
    out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (geometry_grads, epsilon_grad) from the brute-force mesh kernel."""
    v = _to_cuda(vertices, torch.float32)
    idx = _to_cuda(indices.astype(np.uint32))
    q = _to_cuda(queries, torch.float32)
    g = grad_output if grad_output.is_cuda else grad_output.to("cuda:0")
    g = g.contiguous()
    if out is None:
        out = torch.empty_like(v)
    eps_out = _empty_eps_grad()
    winder.brute_force_gradients_mesh(g, v, idx, q, out, eps_out, epsilon, _stream())
    torch.cuda.synchronize()
    return out, eps_out


def run_fast_mesh_gradients(
    vertices: np.ndarray,
    indices: np.ndarray,
    queries: np.ndarray,
    grad_output: torch.Tensor,
    beta: Optional[float],
    epsilon: float = DEFAULT_EPSILON,
    out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (geometry_grads, epsilon_grad) from the fast mesh engine."""
    v = _to_cuda(vertices, torch.float32)
    idx = _to_cuda(indices.astype(np.uint32))
    q = _to_cuda(queries, torch.float32)
    g = grad_output if grad_output.is_cuda else grad_output.to("cuda:0")
    g = g.contiguous()
    if out is None:
        out = torch.empty_like(v)
    eps_out = _empty_eps_grad()
    engine = winder.GradientEngine(q, g, stream=_stream())
    engine.compute_mesh(
        v, idx, out, eps_out, -1.0 if beta is None else beta, epsilon, _stream()
    )
    torch.cuda.synchronize()
    return out, eps_out


def run_brute_force_point_normal_gradients(
    points: np.ndarray,
    normals: np.ndarray,
    queries: np.ndarray,
    grad_output: torch.Tensor,
    epsilon: float,
    out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (geometry_grads, epsilon_grad) from the brute-force point-normal kernel."""
    p = _to_cuda(points, torch.float32)
    n = _to_cuda(normals, torch.float32)
    q = _to_cuda(queries, torch.float32)
    g = grad_output if grad_output.is_cuda else grad_output.to("cuda:0")
    g = g.contiguous()
    if out is None:
        out = torch.empty(
            [points.shape[0], 2, 3], device="cuda:0", dtype=torch.float32
        )
    eps_out = _empty_eps_grad()
    winder.brute_force_gradients_point_normal(
        g, p, n, q, out, eps_out, epsilon, _stream()
    )
    torch.cuda.synchronize()
    return out, eps_out


def run_fast_point_normal_gradients(
    points: np.ndarray,
    normals: np.ndarray,
    queries: np.ndarray,
    grad_output: torch.Tensor,
    epsilon: float,
    beta: Optional[float],
    out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (geometry_grads, epsilon_grad) from the fast point-normal engine."""
    p = _to_cuda(points, torch.float32)
    n = _to_cuda(normals, torch.float32)
    q = _to_cuda(queries, torch.float32)
    g = grad_output if grad_output.is_cuda else grad_output.to("cuda:0")
    g = g.contiguous()
    if out is None:
        out = torch.empty(
            [points.shape[0], 2, 3], device="cuda:0", dtype=torch.float32
        )
    eps_out = _empty_eps_grad()
    engine = winder.GradientEngine(q, g, stream=_stream())
    engine.compute_point_normal(
        p, n, out, eps_out, -1.0 if beta is None else beta, epsilon, _stream()
    )
    torch.cuda.synchronize()
    return out, eps_out


# =========================================================================
# Implementation-independent invariants
# =========================================================================


def _invariant_report(
    name: str, err: float, scale: float, tol_rel: float
) -> bool:
    """Print and return a pass/fail for an invariant with a relative tolerance."""
    rel = err / (scale + 1e-30)
    ok = rel < tol_rel
    status = "\033[92mPASS\033[0m" if ok else "\033[91mFAIL\033[0m"
    print(f"  [{name}] max|Δg|/scale = {rel:.3e}  ->  {status}")
    return ok


def test_mesh_translation_invariance(
    vertices: np.ndarray,
    indices: np.ndarray,
    queries: np.ndarray,
    g: torch.Tensor,
    beta: Optional[float],
    epsilon: float,
) -> bool:
    """Check that translating geometry and queries together leaves gradients unchanged."""
    diag = float(np.linalg.norm(vertices.max(0) - vertices.min(0)))
    t = np.array([0.37, -0.41, 0.19], dtype=np.float32) * diag
    a, _ = run_fast_mesh_gradients(vertices, indices, queries, g, beta, epsilon)
    b, _ = run_fast_mesh_gradients(vertices + t, indices, queries + t, g, beta, epsilon)
    err = float((a - b).abs().max().item())
    return _invariant_report(
        "translation invariance", err, float(a.abs().max().item()), 1e-4
    )


def test_mesh_query_permutation(
    vertices: np.ndarray,
    indices: np.ndarray,
    queries: np.ndarray,
    g: torch.Tensor,
    beta: Optional[float],
    epsilon: float,
    seed: int = 0,
) -> bool:
    """Check that permuting the query set permutes the gradients but not their values."""
    n = queries.shape[0]
    perm = np.random.default_rng(seed).permutation(n)
    a, _ = run_fast_mesh_gradients(vertices, indices, queries, g, beta, epsilon)
    b, _ = run_fast_mesh_gradients(
        vertices, indices, queries[perm], g[perm], beta, epsilon
    )
    err = float((a - b).abs().max().item())
    return _invariant_report(
        "query permutation", err, float(a.abs().max().item()), 1e-5
    )


def test_mesh_grad_output_linearity(
    vertices: np.ndarray,
    indices: np.ndarray,
    queries: np.ndarray,
    g: torch.Tensor,
    beta: Optional[float],
    epsilon: float,
) -> bool:
    """Check that scaling grad_output by 3 scales the geometry gradient by 3."""
    a, _ = run_fast_mesh_gradients(vertices, indices, queries, g, beta, epsilon)
    b, _ = run_fast_mesh_gradients(vertices, indices, queries, 3.0 * g, beta, epsilon)
    err = float((b - 3.0 * a).abs().max().item())
    scale = float(a.abs().max().item()) * 3.0
    return _invariant_report("grad_output linearity", err, scale, 1e-5)


def test_mesh_determinism(
    vertices: np.ndarray,
    indices: np.ndarray,
    queries: np.ndarray,
    g: torch.Tensor,
    beta: Optional[float],
    epsilon: float,
) -> bool:
    """Check that repeated invocations return identical results."""
    a, _ = run_fast_mesh_gradients(vertices, indices, queries, g, beta, epsilon)
    b, _ = run_fast_mesh_gradients(vertices, indices, queries, g, beta, epsilon)
    err = float((a - b).abs().max().item())
    return _invariant_report("determinism", err, 1.0, 0.0)


def run_mesh_invariants(
    vertices: np.ndarray,
    indices: np.ndarray,
    query_mode: str,
    query_count: int,
    beta: Optional[float],
    seed: int,
    epsilon: float,
) -> bool:
    """Run all mesh invariant checks and return True iff all pass."""
    print("\n=== Mesh gradient invariants (implementation-independent) ===")
    q = generate_queries(vertices, query_mode, query_count, seed)
    g = make_grad_output(query_count, seed + 1)
    results = [
        test_mesh_translation_invariance(vertices, indices, q, g, beta, epsilon),
        test_mesh_query_permutation(vertices, indices, q, g, beta, epsilon, seed),
        test_mesh_grad_output_linearity(vertices, indices, q, g, beta, epsilon),
        test_mesh_determinism(vertices, indices, q, g, beta, epsilon),
    ]
    return all(results)


# =========================================================================
# Main test drivers
# =========================================================================


def test_triangle_gradients(
    vertices: np.ndarray,
    indices: np.ndarray,
    query_mode: str,
    query_count: int,
    beta: Optional[float],
    num_trials: int,
    base_seed: int,
    epsilon: float,
    verbose: bool = True,
) -> tuple[list[GradientMetrics], list[ScalarGradientMetrics]]:
    """Validate fast triangle gradients (v0/v1/v2 and epsilon) vs brute force."""
    triangles = vertices[indices]
    all_metrics: list[GradientMetrics] = []
    eps_metrics: list[ScalarGradientMetrics] = []
    for trial in range(num_trials):
        seed = base_seed + trial * 1009
        set_seeds(seed)
        queries = generate_queries(vertices, query_mode, query_count, seed)
        g = make_grad_output(query_count, seed + 1)

        gt, gt_eps = run_brute_force_triangle_gradients(triangles, queries, g, epsilon)
        fast, fast_eps = run_fast_triangle_gradients(
            triangles, queries, g, beta, epsilon
        )

        strict = beta is not None and beta >= 10
        for idx, name in enumerate(["Triangle v0", "Triangle v1", "Triangle v2"]):
            all_metrics.append(
                validate_gradients(
                    fast[:, idx],
                    gt[:, idx],
                    f"{name} [seed={seed}, eps={epsilon:g}]",
                    strict=strict,
                    verbose=verbose,
                )
            )
        eps_metrics.append(
            validate_scalar_gradient(
                fast_eps,
                gt_eps,
                f"Triangle [seed={seed}, eps={epsilon:g}]",
                verbose=verbose,
            )
        )

    # --- Timing ---
    set_seeds(base_seed)
    q = generate_queries(vertices, query_mode, query_count, base_seed)
    g = make_grad_output(query_count, base_seed + 1)

    bf_ms = cuda_timer(
        lambda: run_brute_force_triangle_gradients(triangles, q, g, epsilon)
    )
    fast_ms = cuda_timer(
        lambda: run_fast_triangle_gradients(triangles, q, g, beta, epsilon)
    )
    build_ms = cuda_timer(
        lambda: winder.GradientEngine(torch.from_numpy(q).cuda(), g, stream=_stream()),
        warmup=2,
        iters=5,
    )
    print("\n  Timing (median of 10, ms):")
    print(f"    brute force     : {bf_ms:8.3f}")
    print(f"    fast (build+go) : {fast_ms:8.3f}   speedup {bf_ms / fast_ms:.2f}x")
    print(
        f"    fast (build only): {build_ms:8.3f}   "
        f"({100 * build_ms / fast_ms:.1f}% of fast path)"
    )
    return all_metrics, eps_metrics


def test_mesh_gradients(
    vertices: np.ndarray,
    indices: np.ndarray,
    query_mode: str,
    query_count: int,
    beta: Optional[float],
    num_trials: int,
    base_seed: int,
    epsilon: float,
    verbose: bool = True,
) -> tuple[list[GradientMetrics], list[ScalarGradientMetrics]]:
    """Validate fast mesh gradients (vertex + epsilon) vs brute force."""
    all_metrics: list[GradientMetrics] = []
    eps_metrics: list[ScalarGradientMetrics] = []
    for trial in range(num_trials):
        seed = base_seed + trial * 1009
        set_seeds(seed)
        queries = generate_queries(vertices, query_mode, query_count, seed)
        g = make_grad_output(query_count, seed + 1)

        gt, gt_eps = run_brute_force_mesh_gradients(
            vertices, indices, queries, g, epsilon
        )
        fast, fast_eps = run_fast_mesh_gradients(
            vertices, indices, queries, g, beta, epsilon
        )

        strict = beta is not None and beta >= 10
        all_metrics.append(
            validate_gradients(
                fast,
                gt,
                f"Mesh [seed={seed}, eps={epsilon:g}]",
                strict=strict,
                verbose=verbose,
            )
        )
        eps_metrics.append(
            validate_scalar_gradient(
                fast_eps,
                gt_eps,
                f"Mesh [seed={seed}, eps={epsilon:g}]",
                verbose=verbose,
            )
        )

    set_seeds(base_seed)
    q = generate_queries(vertices, query_mode, query_count, base_seed)
    g = make_grad_output(query_count, base_seed + 1)

    bf_ms = cuda_timer(
        lambda: run_brute_force_mesh_gradients(vertices, indices, q, g, epsilon)
    )
    fast_ms = cuda_timer(
        lambda: run_fast_mesh_gradients(vertices, indices, q, g, beta, epsilon)
    )
    build_ms = cuda_timer(
        lambda: winder.GradientEngine(torch.from_numpy(q).cuda(), g, stream=_stream()),
        warmup=2,
        iters=5,
    )
    print("\n  Timing (median of 10, ms):")
    print(f"    brute force     : {bf_ms:8.3f}")
    print(f"    fast (build+go) : {fast_ms:8.3f}   speedup {bf_ms / fast_ms:.2f}x")
    print(
        f"    fast (build only): {build_ms:8.3f}   "
        f"({100 * build_ms / fast_ms:.1f}% of fast path)"
    )
    return all_metrics, eps_metrics


def test_point_normal_gradients(
    points: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    query_mode: str,
    query_count: int,
    epsilon: float,
    beta: Optional[float],
    num_trials: int,
    base_seed: int,
    verbose: bool = True,
) -> tuple[list[GradientMetrics], list[ScalarGradientMetrics]]:
    """Validate fast point-normal gradients (n, p, epsilon) vs brute force."""
    scaled_normals = normals * areas[:, None]
    all_metrics: list[GradientMetrics] = []
    eps_metrics: list[ScalarGradientMetrics] = []
    for trial in range(num_trials):
        seed = base_seed + trial * 1009
        set_seeds(seed)
        queries = generate_queries(points, query_mode, query_count, seed)
        g = make_grad_output(query_count, seed + 1)

        gt, gt_eps = run_brute_force_point_normal_gradients(
            points, scaled_normals, queries, g, epsilon
        )
        fast, fast_eps = run_fast_point_normal_gradients(
            points, scaled_normals, queries, g, epsilon, beta
        )

        strict = beta is not None and beta >= 10
        for idx, name in enumerate(["PointNormal n", "PointNormal p"]):
            all_metrics.append(
                validate_gradients(
                    fast[:, idx],
                    gt[:, idx],
                    f"{name} [seed={seed}, eps={epsilon:g}]",
                    strict=strict,
                    verbose=verbose,
                )
            )
        eps_metrics.append(
            validate_scalar_gradient(
                fast_eps,
                gt_eps,
                f"PointNormal [seed={seed}, eps={epsilon:g}]",
                verbose=verbose,
            )
        )

    set_seeds(base_seed)
    q = generate_queries(points, query_mode, query_count, base_seed)
    g = make_grad_output(query_count, base_seed + 1)

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
        lambda: winder.GradientEngine(torch.from_numpy(q).cuda(), g, stream=_stream()),
        warmup=2,
        iters=5,
    )
    print("\n  Timing (median of 10, ms):")
    print(f"    brute force     : {bf_ms:8.3f}")
    print(f"    fast (build+go) : {fast_ms:8.3f}   speedup {bf_ms / fast_ms:.2f}x")
    print(
        f"    fast (build only): {build_ms:8.3f}   "
        f"({100 * build_ms / fast_ms:.1f}% of fast path)"
    )
    return all_metrics, eps_metrics


# =========================================================================
# Beta sweep
# =========================================================================


def run_beta_sweep(
    vertices: np.ndarray,
    indices: np.ndarray,
    points: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    query_mode: str,
    query_count: int,
    beta_values: list[float],
    num_trials: int,
    base_seed: int,
    epsilon: float,
) -> tuple[dict, dict]:
    """Sweep beta and report vector and scalar-gradient errors per geometry."""
    print("\n" + "=" * 88)
    print(f"  BETA SWEEP — approximation error vs beta  (epsilon = {epsilon:g})")
    print("=" * 88)

    table: dict[float, dict[str, list]] = {}
    eps_table: dict[float, dict[str, list]] = {}
    timing: dict[float, dict] = {}

    set_seeds(base_seed)
    timing_queries = generate_queries(vertices, query_mode, query_count, base_seed)
    timing_g = make_grad_output(query_count, base_seed + 1)

    tris = vertices[indices]
    scaled_normals = normals * areas[:, None]

    bf_mesh_ms = cuda_timer(
        lambda: run_brute_force_mesh_gradients(
            vertices, indices, timing_queries, timing_g, epsilon
        )
    )
    bf_tri_ms = cuda_timer(
        lambda: run_brute_force_triangle_gradients(
            tris, timing_queries, timing_g, epsilon
        )
    )
    bf_pn_ms = cuda_timer(
        lambda: run_brute_force_point_normal_gradients(
            points, scaled_normals, timing_queries, timing_g, epsilon
        )
    )
    print(f"\n  Brute-force timings (median of 10, ms), eps = {epsilon:g}:")
    print(f"    mesh          : {bf_mesh_ms:8.3f} ms")
    print(f"    triangle      : {bf_tri_ms:8.3f} ms")
    print(f"    point_normal  : {bf_pn_ms:8.3f} ms")

    for beta in beta_values:
        table[beta] = {"mesh": [], "triangle": [], "point_normal": []}
        eps_table[beta] = {"mesh": [], "triangle": [], "point_normal": []}
        timing[beta] = {}

        for trial in range(num_trials):
            seed = base_seed + trial * 1009
            set_seeds(seed)
            queries = generate_queries(vertices, query_mode, query_count, seed)
            g = make_grad_output(query_count, seed + 1)

            gt, gt_eps = run_brute_force_mesh_gradients(
                vertices, indices, queries, g, epsilon
            )
            fast, fast_eps = run_fast_mesh_gradients(
                vertices, indices, queries, g, beta, epsilon
            )
            table[beta]["mesh"].append(
                validate_gradients(
                    fast, gt, f"mesh beta={beta} seed={seed}", verbose=False
                )
            )
            eps_table[beta]["mesh"].append(
                validate_scalar_gradient(
                    fast_eps, gt_eps, f"mesh-eps beta={beta} seed={seed}", verbose=False
                )
            )

            gt, gt_eps = run_brute_force_triangle_gradients(tris, queries, g, epsilon)
            fast, fast_eps = run_fast_triangle_gradients(
                tris, queries, g, beta, epsilon
            )
            for idx in range(3):
                table[beta]["triangle"].append(
                    validate_gradients(
                        fast[:, idx],
                        gt[:, idx],
                        f"tri{idx} beta={beta} seed={seed}",
                        verbose=False,
                    )
                )
            eps_table[beta]["triangle"].append(
                validate_scalar_gradient(
                    fast_eps, gt_eps, f"tri-eps beta={beta} seed={seed}", verbose=False
                )
            )

            gt, gt_eps = run_brute_force_point_normal_gradients(
                points, scaled_normals, queries, g, epsilon
            )
            fast, fast_eps = run_fast_point_normal_gradients(
                points, scaled_normals, queries, g, epsilon, beta
            )
            for idx in range(2):
                table[beta]["point_normal"].append(
                    validate_gradients(
                        fast[:, idx],
                        gt[:, idx],
                        f"pn{idx} beta={beta} seed={seed}",
                        verbose=False,
                    )
                )
            eps_table[beta]["point_normal"].append(
                validate_scalar_gradient(
                    fast_eps, gt_eps, f"pn-eps beta={beta} seed={seed}", verbose=False
                )
            )

        def _build_mesh() -> winder.GradientEngine:
            return winder.GradientEngine(
                torch.from_numpy(timing_queries).cuda(), timing_g, stream=_stream()
            )

        timing[beta]["mesh"] = {"build_ms": cuda_timer(_build_mesh, warmup=2, iters=5)}

        def _fast_mesh() -> None:
            run_fast_mesh_gradients(
                vertices, indices, timing_queries, timing_g, beta, epsilon
            )

        def _fast_tri() -> None:
            run_fast_triangle_gradients(tris, timing_queries, timing_g, beta, epsilon)

        def _fast_pn() -> None:
            run_fast_point_normal_gradients(
                points, scaled_normals, timing_queries, timing_g, epsilon, beta
            )

        timing[beta]["mesh"]["total_ms"] = cuda_timer(_fast_mesh)
        timing[beta]["triangle"] = {"total_ms": cuda_timer(_fast_tri)}
        timing[beta]["point_normal"] = {"total_ms": cuda_timer(_fast_pn)}

    print(
        f"\n{'beta':>6} | {'geometry':<14} | "
        f"{'rel_norm(med)':>13} | {'p99_rel(med)':>12} | "
        f"{'eps_rel(med)':>13} | {'eps_pass':>8} | "
        f"{'BF(ms)':>8} | {'fast(ms)':>9} | {'speedup':>8}"
    )
    print("-" * 128)

    bf_ms_for = {"mesh": bf_mesh_ms, "triangle": bf_tri_ms, "point_normal": bf_pn_ms}

    for beta in beta_values:
        for geom in ("mesh", "triangle", "point_normal"):
            ms = table[beta][geom]
            if not ms:
                continue
            rel_med = float(np.median([m.global_rel_norm_err for m in ms]))
            p99_med = float(np.median([m.masked_p99_rel for m in ms]))

            ems = eps_table[beta][geom]
            eps_rel_med = (
                float(np.median([m.rel_err for m in ems]))
                if ems
                else float("nan")
            )
            eps_pass = all(m.passed for m in ems) if ems else True

            bf_ms = bf_ms_for[geom]
            fast_ms = timing[beta][geom]["total_ms"]
            speedup = bf_ms / fast_ms if fast_ms > 0 else float("nan")
            print(
                f"{beta:>6.3f} | {geom:<14} | "
                f"{rel_med:>13.3e} | {p99_med:>12.3e} | "
                f"{eps_rel_med:>13.3e} | "
                f"{'YES' if eps_pass else 'NO':>8} | "
                f"{bf_ms:>8.3f} | {fast_ms:>9.3f} | "
                f"{speedup:>7.2f}x"
            )

    print("-" * 128)

    non_monotone: list[tuple[str, float, float, float]] = []
    prev_med: dict[str, Optional[float]] = {
        "mesh": None,
        "triangle": None,
        "point_normal": None,
    }
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
        print("\n\033[91mNon-monotone behaviour detected:\033[0m")
        for geom, beta, prev, cur in non_monotone:
            print(f"  {geom}: beta={beta:.3f}  rel_norm {prev:.3e} -> {cur:.3e}")
    else:
        print("\n\033[92mMonotone non-increasing with beta.\033[0m")

    return table, timing


# =========================================================================
# Epsilon sweep
# =========================================================================


def run_epsilon_sweep(
    vertices: np.ndarray,
    indices: np.ndarray,
    points: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    query_mode: str,
    query_count: int,
    epsilon_values: list[float],
    num_trials: int,
    base_seed: int,
    beta: float,
) -> dict:
    """Sweep regularization strength at fixed beta and report errors per geometry."""
    print("\n" + "=" * 100)
    print(
        f"  EPSILON SWEEP — regularization strength vs fast-vs-brute error  (beta = {beta})"
    )
    print("=" * 100)

    tris = vertices[indices]
    scaled_normals = normals * areas[:, None]

    set_seeds(base_seed)
    timing_q = generate_queries(vertices, query_mode, query_count, base_seed)
    timing_g = make_grad_output(query_count, base_seed + 1)

    table: dict[float, dict[str, list[GradientMetrics]]] = {
        e: {"mesh": [], "triangle": [], "point_normal": []} for e in epsilon_values
    }
    eps_table: dict[float, dict[str, list[ScalarGradientMetrics]]] = {
        e: {"mesh": [], "triangle": [], "point_normal": []} for e in epsilon_values
    }

    for eps in epsilon_values:
        for trial in range(num_trials):
            seed = base_seed + trial * 1009
            set_seeds(seed)
            q = generate_queries(vertices, query_mode, query_count, seed)
            g = make_grad_output(query_count, seed + 1)

            # Mesh
            gt, gt_eps = run_brute_force_mesh_gradients(
                vertices, indices, q, g, eps
            )
            fast, fast_eps = run_fast_mesh_gradients(
                vertices, indices, q, g, beta, eps
            )
            table[eps]["mesh"].append(
                validate_gradients(fast, gt, f"mesh eps={eps:g} s={seed}", verbose=False)
            )
            eps_table[eps]["mesh"].append(
                validate_scalar_gradient(
                    fast_eps, gt_eps, f"mesh-eps eps={eps:g} s={seed}", verbose=False
                )
            )

            # Triangle
            gt, gt_eps = run_brute_force_triangle_gradients(tris, q, g, eps)
            fast, fast_eps = run_fast_triangle_gradients(tris, q, g, beta, eps)
            for idx in range(3):
                table[eps]["triangle"].append(
                    validate_gradients(
                        fast[:, idx],
                        gt[:, idx],
                        f"tri{idx} eps={eps:g} s={seed}",
                        verbose=False,
                    )
                )
            eps_table[eps]["triangle"].append(
                validate_scalar_gradient(
                    fast_eps, gt_eps, f"tri-eps eps={eps:g} s={seed}", verbose=False
                )
            )

            # PointNormal
            gt, gt_eps = run_brute_force_point_normal_gradients(
                points, scaled_normals, q, g, eps
            )
            fast, fast_eps = run_fast_point_normal_gradients(
                points, scaled_normals, q, g, eps, beta
            )
            for idx in range(2):
                table[eps]["point_normal"].append(
                    validate_gradients(
                        fast[:, idx],
                        gt[:, idx],
                        f"pn{idx} eps={eps:g} s={seed}",
                        verbose=False,
                    )
                )
            eps_table[eps]["point_normal"].append(
                validate_scalar_gradient(
                    fast_eps, gt_eps, f"pn-eps eps={eps:g} s={seed}", verbose=False
                )
            )

    print(
        f"\n{'epsilon':>9} | {'geometry':<14} | "
        f"{'rel_norm(med)':>13} | {'p99_rel(med)':>12} | "
        f"{'cos(med)':>10} | {'eps_rel(med)':>13} | {'eps_pass':>8} | "
        f"{'fast(ms)':>9}"
    )
    print("-" * 128)

    for eps in epsilon_values:
        for geom in ("mesh", "triangle", "point_normal"):
            ms = table[eps][geom]
            if not ms:
                continue
            rel_med = float(np.median([m.global_rel_norm_err for m in ms]))
            p99_med = float(np.median([m.masked_p99_rel for m in ms]))
            cos_med = float(np.median([m.global_cosine_flat for m in ms]))

            ems = eps_table[eps][geom]
            eps_rel_med = (
                float(np.median([m.rel_err for m in ems]))
                if ems
                else float("nan")
            )
            eps_pass = all(m.passed for m in ems) if ems else True

            def _timed(g: str = geom, e: float = eps) -> None:
                if g == "mesh":
                    run_fast_mesh_gradients(
                        vertices, indices, timing_q, timing_g, beta, e
                    )
                elif g == "triangle":
                    run_fast_triangle_gradients(tris, timing_q, timing_g, beta, e)
                else:
                    run_fast_point_normal_gradients(
                        points, scaled_normals, timing_q, timing_g, e, beta
                    )

            fast_ms = cuda_timer(_timed, warmup=2, iters=5)

            print(
                f"{eps:>9.5f} | {geom:<14} | "
                f"{rel_med:>13.3e} | {p99_med:>12.3e} | "
                f"{cos_med:>10.6f} | "
                f"{eps_rel_med:>13.3e} | "
                f"{'YES' if eps_pass else 'NO':>8} | "
                f"{fast_ms:>9.3f}"
            )

    print("-" * 128)
    print("\n  Notes:")
    print("    - Both brute force and fast use the SAME epsilon at each row.")
    print("    - Larger epsilon -> smoother kernel -> better fast approximation.")
    print("    - rel_norm measures fast-vs-brute (not vs sharp truth).")
    print("    - eps_rel is the fast-vs-brute relative error of dL/d(epsilon).")
    return table


# =========================================================================
# CSV + summary
# =========================================================================

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
    "pn_rel_norm_err",
    "tri_rel_norm_err",
    "mesh_rel_norm_err",
    "pn_eps_rel_err",
    "tri_eps_rel_err",
    "mesh_eps_rel_err",
    "pn_passed",
    "tri_passed",
    "mesh_passed",
    "pn_eps_passed",
    "tri_eps_passed",
    "mesh_eps_passed",
]


def write_run_to_csv(
    csv_path: str,
    unique_label: str,
    raw_label: str,
    args: argparse.Namespace,
    pn_metrics: Optional[list[GradientMetrics]],
    tri_metrics: Optional[list[GradientMetrics]],
    mesh_metrics: Optional[list[GradientMetrics]],
    pn_eps: Optional[list[ScalarGradientMetrics]] = None,
    tri_eps: Optional[list[ScalarGradientMetrics]] = None,
    mesh_eps: Optional[list[ScalarGradientMetrics]] = None,
) -> None:
    """Append one row of aggregated metrics to `csv_path` (archives on header change)."""
    file_exists = os.path.exists(csv_path)
    if file_exists:
        with open(csv_path, newline="") as f:
            header = next(csv.reader(f), None)
        if header != CSV_FIELDS:
            backup = f"{csv_path}.bak.{datetime.now():%Y%m%d_%H%M%S}"
            shutil.move(csv_path, backup)
            print(f"[CSV] Header changed; archived previous file to {backup}")
            file_exists = False

    def _avg(ms, attr: str) -> str:
        if not ms:
            return ""
        vals = [getattr(m, attr) for m in ms if math.isfinite(getattr(m, attr))]
        return f"{float(np.mean(vals)):.6e}" if vals else ""

    def _pass(ms) -> str:
        return "" if not ms else str(all(m.passed for m in ms))

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
        "pn_rel_norm_err": _avg(pn_metrics, "global_rel_norm_err"),
        "tri_rel_norm_err": _avg(tri_metrics, "global_rel_norm_err"),
        "mesh_rel_norm_err": _avg(mesh_metrics, "global_rel_norm_err"),
        "pn_eps_rel_err": _avg(pn_eps, "rel_err"),
        "tri_eps_rel_err": _avg(tri_eps, "rel_err"),
        "mesh_eps_rel_err": _avg(mesh_eps, "rel_err"),
        "pn_passed": _pass(pn_metrics),
        "tri_passed": _pass(tri_metrics),
        "mesh_passed": _pass(mesh_metrics),
        "pn_eps_passed": _pass(pn_eps),
        "tri_eps_passed": _pass(tri_eps),
        "mesh_eps_passed": _pass(mesh_eps),
    }

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"[CSV] Appended 1 row to {csv_path}")


def _summarize(metrics_list, name: str) -> Optional[float]:
    """Print a one-line summary of a list of GradientMetrics and return the mean error."""
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


def _summarize_scalar(metrics_list, name: str) -> Optional[float]:
    """Print a one-line summary of a list of ScalarGradientMetrics."""
    if not metrics_list:
        print(f"\n[{name} eps-grad] no metrics produced.")
        return None
    errs = [m.rel_err for m in metrics_list if math.isfinite(m.rel_err)]
    any_failed = any(not m.passed for m in metrics_list)
    mean_err = float(np.mean(errs)) if errs else float("nan")
    status = "\033[91mFAIL\033[0m" if any_failed else "\033[92mPASS\033[0m"
    print(
        f"\n[{name} eps-grad] mean rel_err = {mean_err:.3e},  overall: {status}  "
        f"({sum(m.passed for m in metrics_list)}/{len(metrics_list)} sub-checks passed)"
    )
    return mean_err


# =========================================================================
# CLI
# =========================================================================


def main() -> None:
    """Command-line entry point."""
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
    parser.add_argument("--query_count", type=int, default=1000000)
    parser.add_argument(
        "--epsilon",
        type=float,
        default=DEFAULT_EPSILON,
        help=f"Regularization fraction of scene scale. "
        f"Default {DEFAULT_EPSILON} (= 1/250). Use 0 for sharp.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Barnes-Hut approximation parameter (-1 for exact).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument(
        "--beta_sweep",
        type=str,
        default="",
        help="Comma-separated beta values to sweep.",
    )
    parser.add_argument("--beta_sweep_only", action="store_true")
    parser.add_argument(
        "--epsilon_sweep",
        type=str,
        default="",
        help="Comma-separated epsilon values to sweep.",
    )
    parser.add_argument("--epsilon_sweep_only", action="store_true")
    parser.add_argument("--skip_invariants", action="store_true")
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
    # Epsilon sweep
    # ---------------------------------------------------------------
    if args.epsilon_sweep:
        eps_values = [float(x) for x in args.epsilon_sweep.split(",") if x.strip()]
        beta = args.beta if args.beta is not None else 2.3
        run_epsilon_sweep(
            vertices,
            indices,
            points,
            normals,
            areas,
            args.query_mode,
            args.query_count,
            eps_values,
            args.num_trials,
            args.seed,
            beta,
        )
        if args.epsilon_sweep_only:
            return

    # ---------------------------------------------------------------
    # Beta sweep
    # ---------------------------------------------------------------
    if args.beta_sweep:
        betas = [float(x) for x in args.beta_sweep.split(",") if x.strip()]
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
            args.epsilon,
        )
        if args.beta_sweep_only:
            return

    # ---------------------------------------------------------------
    # Main geometry validation
    # ---------------------------------------------------------------
    pn_metrics: Optional[list[GradientMetrics]] = None
    tri_metrics: Optional[list[GradientMetrics]] = None
    mesh_metrics: Optional[list[GradientMetrics]] = None
    pn_eps: Optional[list[ScalarGradientMetrics]] = None
    tri_eps: Optional[list[ScalarGradientMetrics]] = None
    mesh_eps: Optional[list[ScalarGradientMetrics]] = None

    if args.geometry_type in ("Mesh", "All") and not args.skip_invariants:
        run_mesh_invariants(
            vertices,
            indices,
            args.query_mode,
            args.query_count,
            args.beta if args.beta is not None else -1,
            args.seed,
            args.epsilon,
        )

    if args.geometry_type in ("PointNormal", "All"):
        print("\n" + "=" * 72)
        print("  PointNormal gradient validation")
        print("=" * 72)
        pn_metrics, pn_eps = test_point_normal_gradients(
            points,
            normals,
            areas,
            args.query_mode,
            args.query_count,
            args.epsilon,
            args.beta,
            args.num_trials,
            args.seed,
        )
        _summarize(pn_metrics, "PointNormal")
        _summarize_scalar(pn_eps, "PointNormal")
        for metric in pn_metrics:
            for idx in metric.worst_idxs:
                p = points[idx]
                n = normals[idx]
                print(f"Bad PointNormal {idx}: p:{p}, n:{n}")

    if args.geometry_type in ("Triangle", "All"):
        print("\n" + "=" * 72)
        print("  Triangle gradient validation")
        print("=" * 72)
        tri_metrics, tri_eps = test_triangle_gradients(
            vertices,
            indices,
            args.query_mode,
            args.query_count,
            args.beta,
            args.num_trials,
            args.seed,
            args.epsilon,
        )
        _summarize(tri_metrics, "Triangle")
        _summarize_scalar(tri_eps, "Triangle")
        for metric in tri_metrics:
            for idx in metric.worst_idxs:
                tri_idxs = indices[idx]
                tri = vertices[tri_idxs]
                print(f"Bad Triangle {idx}: {tri}")

    if args.geometry_type in ("Mesh", "All"):
        print("\n" + "=" * 72)
        print("  Mesh gradient validation")
        print("=" * 72)
        mesh_metrics, mesh_eps = test_mesh_gradients(
            vertices,
            indices,
            args.query_mode,
            args.query_count,
            args.beta,
            args.num_trials,
            args.seed,
            args.epsilon,
        )
        _summarize(mesh_metrics, "Mesh")
        _summarize_scalar(mesh_eps, "Mesh")
        for metric in mesh_metrics:
            for idx in metric.worst_idxs:
                tri_rows = np.where((indices == idx).any(axis=1))[0]
                for row in tri_rows:
                    tri = vertices[indices[row]]
                    print(f"Bad Vertex {idx} in Triangle row {row}: {tri}")

    # ---------------------------------------------------------------
    # CSV
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
            pn_eps,
            tri_eps,
            mesh_eps,
        )


if __name__ == "__main__":
    main()
