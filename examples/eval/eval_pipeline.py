"""Shared evaluation pipeline for Winder dataset benchmarks.

Supports multiple evaluation modes:
  - forward_triangle        (winding numbers, triangle soup)
  - forward_point_normal    (winding numbers, point-normal soup)
  - backward_triangle       (gradients w.r.t. triangle vertices)
  - backward_mesh           (gradients w.r.t. shared mesh vertices)
  - backward_point_normal   (gradients w.r.t. surfel positions and normals)

Pipeline architecture (three-stage, decoupled CPU from GPU):

    reader ──► raw_queue ──► [preprocess pool × P] ──► input_queue
                                                           │
                                                           ▼
                                                    [GPU workers × W]
                                                           │
                                                           ▼
                                                    output_queue ──► writer

The producer reads/parses meshes. The preprocess pool does the CPU-heavy
work — mesh modification, query sampling, distance filtering (KD-tree),
point-normal conversion — on many threads. The GPU workers only upload,
run kernels, and download.

Each GPU worker runs its work on its own CUDA stream. This is required for
correct timing.

When --bucket-edges is set, backward modes are additionally evaluated once
per distance-to-surface bucket with a masked grad_output, giving a
per-bucket attribution of the gradient. Because the backward operator is
linear in grad_output, the result of a masked run is exactly the
contribution of that bucket's queries. Cost: (1 + number of buckets) times
the per-mode GPU work.

When --epsilon_sweep is set, each mode is evaluated at every epsilon in the
list. The mode key in the output JSONL gains a `@eps=<value>` suffix, e.g.
`backward_triangle@eps=0.004`. Each mode's metrics dict also carries an
explicit `"epsilon"` field. Set --epsilon for a single run.

Cost: sweeping over K epsilons multiplies the per-mesh GPU work by K (and
by (1 + n_buckets) if bucketing is enabled).

The one caveat that remains: workers still share the physical GPU. Per-item
timings taken at high worker counts reflect "shared GPU" time, not "exclusive
GPU" time. For a paper, run a small single-worker pass to get clean latencies.
"""

from __future__ import annotations

import argparse
import json
import igl
import math
import gc
import queue
import threading
import random
import time
import traceback
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Tuple

import numpy as np
import torch
import ctypes

import winder


# =============================================================================
# Per-worker stream
# =============================================================================


class WorkerStream:
    """A per-worker CUDA stream."""

    __slots__ = ("stream", "ev_start", "ev_stop")

    def __init__(self, device: Optional[str] = None):
        if device is None:
            device = f"cuda:{torch.cuda.current_device()}"
        self.stream = torch.cuda.Stream(device=device)
        self.ev_start = torch.cuda.Event(enable_timing=True)
        self.ev_stop = torch.cuda.Event(enable_timing=True)

    @property
    def handle(self) -> int:
        return self.stream.cuda_stream

    def __enter__(self):
        self._ctx = torch.cuda.stream(self.stream)
        self._ctx.__enter__()
        return self

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)


# =============================================================================
# Per-mesh shared GPU state
# =============================================================================


@dataclass
class MeshBatch:
    """GPU-resident state shared across all evaluation modes for one mesh."""

    tris_np: np.ndarray
    queries_np: np.ndarray
    grad_output_np: np.ndarray
    vertices_np: Optional[np.ndarray]
    indices_np: Optional[np.ndarray]

    tris_gpu: torch.Tensor
    queries_gpu: torch.Tensor
    grad_output_gpu: torch.Tensor
    vertices_gpu: Optional[torch.Tensor]
    indices_gpu: Optional[torch.Tensor]

    points_gpu: Optional[torch.Tensor] = None
    scaled_normals_gpu: Optional[torch.Tensor] = None

    @classmethod
    def build(cls, tris, queries, grad_output, vertices, indices):
        return cls(
            tris_np=tris,
            queries_np=queries,
            grad_output_np=grad_output,
            vertices_np=vertices,
            indices_np=indices,
            tris_gpu=_to_cuda(tris),
            queries_gpu=_to_cuda(queries),
            grad_output_gpu=_to_cuda(grad_output),
            vertices_gpu=(_to_cuda(vertices) if vertices is not None else None),
            indices_gpu=(
                _to_cuda(indices.astype(np.uint32)) if indices is not None else None
            ),
        )

    def get_point_normals(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.points_gpu is None:
            pts_np, sn_np = mesh_to_point_surfels(self.tris_np)
            self.points_gpu = _to_cuda(pts_np)
            self.scaled_normals_gpu = _to_cuda(sn_np)
        return self.points_gpu, self.scaled_normals_gpu

    def with_masked_grad_output(self, mask: np.ndarray) -> "MeshBatch":
        """Shallow copy with a new grad_output = mask * grad_output.

        Reuses queries_gpu, tris_gpu, vertices_gpu, indices_gpu and (if
        already built) points_gpu / scaled_normals_gpu. Only the gradient
        target is re-uploaded. By linearity of the backward operator, the
        result of evaluating with a masked grad_output is exactly the
        contribution of the selected queries.
        """
        import copy as _copy

        new = _copy.copy(self)
        new.grad_output_np = (mask * self.grad_output_np).astype(np.float32)
        new.grad_output_gpu = _to_cuda(new.grad_output_np)
        return new


# =============================================================================
# Numerical utilities
# =============================================================================


class Welford:
    __slots__ = ("n", "mean", "m2")

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)

    @property
    def variance(self) -> float:
        return self.m2 / (self.n - 1) if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)


class FatalNumericalError(RuntimeError):
    def __init__(self, msg: str, *, name: str, dump_path: Optional[Path]):
        super().__init__(msg)
        self.name = name
        self.dump_path = dump_path


# =============================================================================
# Query sampling
# =============================================================================


def sample_queries(
    tris: np.ndarray, n_queries: int, seed: int, mode: str = "grid"
) -> np.ndarray:
    if mode == "grid":
        return sample_queries_grid(tris, n_queries)
    flat = tris.reshape(-1, 3)
    mn = flat.min(axis=0)
    mx = flat.max(axis=0)
    diag = float(np.linalg.norm(mx - mn))
    pad = 0.3 * diag
    rng = np.random.default_rng(seed)
    return rng.uniform(mn - pad, mx + pad, size=(n_queries, 3)).astype(np.float32)


def sample_queries_grid(tris: np.ndarray, n_queries: int) -> np.ndarray:
    side = int(round(n_queries ** (1.0 / 3.0)))
    if side**3 != n_queries:
        raise ValueError(f"n_queries must be a perfect cube, got {n_queries}")
    flat = tris.reshape(-1, 3)
    mn = flat.min(axis=0)
    mx = flat.max(axis=0)
    axes = [mn[d] + (np.arange(side) + 0.5) * (mx[d] - mn[d]) / side for d in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)
    return pts.astype(np.float32)


def make_grad_output(n_queries: int, name: str) -> np.ndarray:
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    return rng.standard_normal(n_queries).astype(np.float32)


def compute_query_distances(queries: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Distance from each query to the nearest triangle surface point.

    Uses igl.point_mesh_squared_distance (AABB tree over triangles).
    ``tris`` is (N, 3, 3); igl wants (V, F) so we flatten.
    Returns float32 array of shape (M,).
    """
    V = tris.reshape(-1, 3).astype(np.float64)
    F = np.arange(V.shape[0], dtype=np.int64).reshape(-1, 3)
    sq_d, _, _ = igl.point_mesh_squared_distance(queries.astype(np.float64), V, F)
    return np.sqrt(sq_d).astype(np.float32)


def filter_queries_by_distance(queries, tris, min_distance):
    """Boolean mask: True for queries at least ``min_distance`` from surface.

    Kept for backwards compatibility; internally uses
    ``compute_query_distances``.
    """
    d = compute_query_distances(queries, tris)
    return d > min_distance


# =============================================================================
# Distance bucketing (backward modes only)
# =============================================================================


def _parse_bucket_edges(spec: str, diagonal: float):
    """'1e-3,1e-2,1e-1' -> [(0, 1e-3·diag), (1e-3·diag, 1e-2·diag), ...]."""
    if not spec or not spec.strip():
        return []
    fractions = [float(x) for x in spec.split(",") if x.strip()]
    edges = [0.0] + [f * diagonal for f in fractions] + [float("inf")]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def _bucket_label(lo: float, hi: float, diagonal: float) -> str:
    """Human-readable, sortable bucket label. Distances normalised by diagonal."""
    lo_f = lo / diagonal if lo > 0 else 0.0
    hi_f = hi / diagonal if hi != float("inf") else float("inf")
    if hi == float("inf"):
        return f"d_ge_{lo_f:.0e}"
    if lo == 0.0:
        return f"d_lt_{hi_f:.0e}"
    return f"d_{lo_f:.0e}_{hi_f:.0e}"


# =============================================================================
# Mesh-to-point-normal conversion
# =============================================================================


def mesh_to_point_surfels(tris: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    v0, v1, v2 = tris[:, 0, :], tris[:, 1, :], tris[:, 2, :]
    points = (v0 + v1 + v2) / 3.0
    cross = np.cross(v1 - v0, v2 - v0)
    magnitudes = np.linalg.norm(cross, axis=-1, keepdims=True)
    areas = magnitudes / 2.0
    safe = np.where(magnitudes == 0, 1e-8, magnitudes)
    unit_normals = cross / safe
    scaled_normals = unit_normals * areas
    return (points.astype(np.float32), scaled_normals.astype(np.float32))


def _triangles_diagonal(tris: np.ndarray) -> float:
    flat = tris.reshape(-1, 3)
    return float(np.linalg.norm(flat.max(0) - flat.min(0)))


def apply_mesh_modification(tris, mode, seed, noise_sigma_frac=1e-2):
    if mode == "none":
        return tris
    rng = np.random.default_rng(seed)
    out = tris
    for step in mode:
        step = step.strip()
        if step == "slice":
            z = out[:, :, 2].mean(axis=1)
            out = out[z > np.median(z)]
        elif step == "drop_half":
            keep = rng.choice(len(out), max(4, len(out) // 2), replace=False)
            out = out[keep]
        elif step == "drop_80pct":
            keep = rng.choice(len(out), max(4, len(out) // 5), replace=False)
            out = out[keep]
        elif step == "noise":
            sigma = noise_sigma_frac * _triangles_diagonal(out)
            out = out + rng.standard_normal(out.shape).astype(np.float32) * sigma
        elif step == "none":
            pass
        else:
            raise ValueError(f"unknown mesh modification step: {step}")
    return out


# =============================================================================
# Metric computation
# =============================================================================


def _forward_metrics(
    ref: np.ndarray, fast: np.ndarray, t_brute_ms: float, t_fast_ms: float
) -> dict:
    diff = fast - ref
    abs_err = np.abs(diff)
    mask = np.abs(ref) > 1e-3
    if mask.any():
        rel = abs_err[mask] / np.abs(ref[mask])
        rms_rel = float(np.sqrt(np.mean(rel * rel)))
        p99_rel = float(np.percentile(rel, 99))
    else:
        rms_rel = float("nan")
        p99_rel = float("nan")
    misclass = int(((ref > 0.5) != (fast > 0.5)).sum())
    return {
        "time_brute_ms": float(t_brute_ms),
        "time_fast_ms": float(t_fast_ms),
        "speedup": float(t_brute_ms / t_fast_ms) if t_fast_ms > 0 else float("nan"),
        "signal_max": float(np.abs(ref).max()) if ref.size else 0.0,
        "rms_abs": float(np.sqrt(np.mean(diff * diff))),
        "mean_abs": float(abs_err.mean()),
        "p50_abs": float(np.percentile(abs_err, 50)),
        "p95_abs": float(np.percentile(abs_err, 95)),
        "p99_abs": float(np.percentile(abs_err, 99)),
        "max_abs": float(abs_err.max()),
        "rms_rel_masked": rms_rel,
        "p99_rel_masked": p99_rel,
        "misclass_count": misclass,
        "misclass_frac": misclass / max(1, ref.size),
    }


def _backward_metrics(
    ref: np.ndarray,
    fast: np.ndarray,
    t_brute_ms: float,
    t_fast_ms: float,
    signal_threshold: float = 1e-6,
) -> dict:
    """Gradient metrics with full per-primitive distribution statistics."""
    ref_flat = ref.reshape(-1)
    fast_flat = fast.reshape(-1)
    diff = fast_flat - ref_flat

    norm_diff = float(np.linalg.norm(diff))
    norm_ref = float(np.linalg.norm(ref_flat))
    norm_fast = float(np.linalg.norm(fast_flat))
    rel_norm = norm_diff / (norm_ref + 1e-30)
    global_cosine = (
        float(np.dot(ref_flat, fast_flat) / (norm_ref * norm_fast + 1e-30))
        if (norm_ref > 0 and norm_fast > 0)
        else float("nan")
    )

    abs_err = np.abs(diff)
    mask = np.abs(ref_flat) > signal_threshold
    if mask.any():
        rel = abs_err[mask] / np.abs(ref_flat[mask])
        rms_rel = float(np.sqrt(np.mean(rel * rel)))
        p99_rel = float(np.percentile(rel, 99))
    else:
        rms_rel = float("nan")
        p99_rel = float("nan")

    ref3 = ref.reshape(-1, 3)
    fast3 = fast.reshape(-1, 3)
    ref_norms = np.linalg.norm(ref3, axis=1)
    fast_norms = np.linalg.norm(fast3, axis=1)

    vmask = ref_norms > signal_threshold
    percentiles = [10, 25, 50, 75, 90, 95, 99, 99.9]
    per_vec: dict[str, float | int] = {
        "n_vectors_masked": int(vmask.sum()),
        "n_vectors_total": int(ref3.shape[0]),
    }

    if vmask.any():
        a = ref3[vmask]
        b = fast3[vmask]
        cos = np.clip(
            (a * b).sum(-1)
            / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-30),
            -1.0,
            1.0,
        )
        ang = np.degrees(np.arccos(cos))
        mag_ratio = fast_norms[vmask] / (ref_norms[vmask] + 1e-30)

        per_vec["per_vec_cos_mean"] = float(cos.mean())
        per_vec["per_vec_cos_std"] = float(cos.std())
        per_vec["per_vec_cos_min"] = float(cos.min())
        per_vec["per_vec_ang_mean"] = float(ang.mean())
        per_vec["per_vec_ang_std"] = float(ang.std())
        per_vec["per_vec_ang_max"] = float(ang.max())
        per_vec["mag_ratio_mean"] = float(mag_ratio.mean())
        per_vec["mag_ratio_std"] = float(mag_ratio.std())

        for p in percentiles:
            key = f"per_vec_cos_p{str(p).replace('.', '_')}"
            per_vec[key] = float(np.percentile(cos, p / 100))
        for p in percentiles:
            key = f"per_vec_ang_p{str(p).replace('.', '_')}"
            per_vec[key] = float(np.percentile(ang, p / 100))
        for p in percentiles:
            key = f"mag_ratio_p{str(p).replace('.', '_')}"
            per_vec[key] = float(np.percentile(mag_ratio, p / 100))
    else:
        for k in (
            "per_vec_cos_mean",
            "per_vec_cos_std",
            "per_vec_cos_min",
            "per_vec_ang_mean",
            "per_vec_ang_std",
            "per_vec_ang_max",
            "mag_ratio_mean",
            "mag_ratio_std",
        ):
            per_vec[k] = float("nan")

    return {
        "time_brute_ms": float(t_brute_ms),
        "time_fast_ms": float(t_fast_ms),
        "speedup": float(t_brute_ms / t_fast_ms) if t_fast_ms > 0 else float("nan"),
        "signal_max": float(np.abs(ref_flat).max()) if ref_flat.size else 0.0,
        "rms_abs": float(np.sqrt(np.mean(diff * diff))),
        "mean_abs": float(abs_err.mean()),
        "p50_abs": float(np.percentile(abs_err, 50)),
        "p95_abs": float(np.percentile(abs_err, 95)),
        "p99_abs": float(np.percentile(abs_err, 99)),
        "max_abs": float(abs_err.max()),
        "rms_rel_masked": rms_rel,
        "p99_rel_masked": p99_rel,
        "global_rel_norm": float(rel_norm),
        "global_cosine": global_cosine,
        # Used by the bucket figure to compute per-bucket signal fractions.
        "ref_norm": float(norm_ref),
        "fast_norm": float(norm_fast),
        **per_vec,
        "misclass_count": 0,
        "misclass_frac": 0.0,
    }


# =============================================================================
# GPU helpers
# =============================================================================


def _to_cuda(x: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x)).to(dtype).cuda()


def _cuda_time(
    fn: Callable[[], None], ws: WorkerStream, warmup: int = 1, iters: int = 1
) -> float:
    """Median event time in ms on the worker's stream."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        ws.ev_start.record(ws.stream)
        fn()
        ws.ev_stop.record(ws.stream)
        torch.cuda.synchronize()
        times.append(ws.ev_start.elapsed_time(ws.ev_stop))
    return float(np.median(times))


# =============================================================================
# Forward runners
# =============================================================================

def _forward_triangle_engines(batch: MeshBatch, args, ws, epsilon: float):
    h = ws.handle
    t_tri = batch.tris_gpu
    t_q = batch.queries_gpu
    qcount = t_q.shape[0]
    wn_brute = torch.empty(qcount, device="cuda", dtype=torch.float32)
    wn_fast = torch.empty(qcount, device="cuda", dtype=torch.float32)

    t_brute = _cuda_time(
        lambda: winder.brute_force_winding_numbers_triangle_soup(
            t_tri, t_q, wn_brute, float(epsilon), h
        ),
        ws,
    )

    def _fast():
        eng = winder.WindingNumberEngine(t_tri, h)
        eng.compute(t_q, wn_fast, float(args.beta), float(epsilon), h)

    t_fast = _cuda_time(_fast, ws)
    return wn_brute.cpu().numpy(), wn_fast.cpu().numpy(), t_brute, t_fast


def _forward_point_normal_engines(batch: MeshBatch, args, ws, epsilon: float):
    h = ws.handle
    p, n = batch.get_point_normals()
    q = batch.queries_gpu
    qcount = q.shape[0]
    wn_brute = torch.empty(qcount, device="cuda", dtype=torch.float32)
    wn_fast = torch.empty(qcount, device="cuda", dtype=torch.float32)

    t_brute = _cuda_time(
        lambda: winder.brute_force_winding_numbers_point_normal(
            p, n, q, wn_brute, float(epsilon), h
        ),
        ws,
    )

    def _fast():
        eng = winder.WindingNumberEngine(p, n, h)
        eng.compute(q, wn_fast, float(args.beta), float(epsilon), h)

    t_fast = _cuda_time(_fast, ws)
    return wn_brute.cpu().numpy(), wn_fast.cpu().numpy(), t_brute, t_fast

# =============================================================================
# Backward runners
# =============================================================================

def _backward_triangle_engines(batch: MeshBatch, args, ws, epsilon: float):
    h = ws.handle
    t_tri = batch.tris_gpu
    t_q = batch.queries_gpu
    g = batch.grad_output_gpu
    N = t_tri.shape[0]
    out_brute = torch.empty([N, 3, 3], device="cuda", dtype=torch.float32)
    out_fast = torch.empty_like(out_brute)

    t_brute = _cuda_time(
        lambda: winder.brute_force_gradients_triangle_soup(
            g, t_tri, t_q, out_brute, float(epsilon), h
        ),
        ws,
    )

    def _fast():
        eng = winder.GradientEngine(t_q, g, h)
        eng.compute_triangle_soup(t_tri, out_fast, float(args.beta), float(epsilon), h)

    t_fast = _cuda_time(_fast, ws)
    return out_brute.cpu().numpy(), out_fast.cpu().numpy(), t_brute, t_fast


def _backward_mesh_engines(batch: MeshBatch, args, ws, epsilon: float):
    if batch.vertices_gpu is None or batch.indices_gpu is None:
        raise ValueError("backward_mesh requires vertices and indices")
    h = ws.handle
    v = batch.vertices_gpu
    idx = batch.indices_gpu
    q = batch.queries_gpu
    g = batch.grad_output_gpu
    K = v.shape[0]
    out_brute = torch.empty([K, 3], device="cuda", dtype=torch.float32)
    out_fast = torch.empty_like(out_brute)

    t_brute = _cuda_time(
        lambda: winder.brute_force_gradients_mesh(
            g, v, idx, q, out_brute, float(epsilon), h
        ),
        ws,
    )

    def _fast():
        eng = winder.GradientEngine(q, g, h)
        eng.compute_mesh(v, idx, out_fast, float(args.beta), float(epsilon), h)

    t_fast = _cuda_time(_fast, ws)
    return out_brute.cpu().numpy(), out_fast.cpu().numpy(), t_brute, t_fast


def _backward_point_normal_engines(batch: MeshBatch, args, ws, epsilon: float):
    h = ws.handle
    p, n = batch.get_point_normals()
    q = batch.queries_gpu
    g = batch.grad_output_gpu
    N = p.shape[0]
    out_brute = torch.empty([N, 2, 3], device="cuda", dtype=torch.float32)
    out_fast = torch.empty_like(out_brute)

    t_brute = _cuda_time(
        lambda: winder.brute_force_gradients_point_normal(
            g, p, n, q, out_brute, float(epsilon), h
        ),
        ws,
    )

    def _fast():
        eng = winder.GradientEngine(q, g, h)
        eng.compute_point_normal(
            p, n, out_fast, float(args.beta), float(epsilon), h
        )

    t_fast = _cuda_time(_fast, ws)
    return out_brute.cpu().numpy(), out_fast.cpu().numpy(), t_brute, t_fast

# =============================================================================
# PyTorch reference implementations (small-scale correctness + timing)
# =============================================================================


def _torch_reg_lengths(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
                       eps_world: float):
    """Plummer-regularized edge lengths matching the C++ kernel."""
    a2 = (a * a).sum(-1)
    b2 = (b * b).sum(-1)
    c2 = (c * c).sum(-1)
    if eps_world <= 0.0:
        return a2.sqrt(), b2.sqrt(), c2.sqrt()
    eps2 = eps_world * eps_world
    return (a2 + eps2).sqrt(), (b2 + eps2).sqrt(), (c2 + eps2).sqrt()


def _torch_ref_forward_winding_triangle(batch, n_q, eps_world, ws, chunk_size=256):
    with torch.cuda.stream(ws.stream):
        t_tri = batch.tris_gpu
        q = batch.queries_gpu[:n_q]
        Q = q.shape[0]
        out = torch.empty(Q, device="cuda", dtype=torch.float32)

        torch.cuda.synchronize()
        ws.ev_start.record(ws.stream)
        with torch.no_grad():
            v0 = t_tri[:, 0, :]
            v1 = t_tri[:, 1, :]
            v2 = t_tri[:, 2, :]
            inv_two_pi = 1.0 / (2.0 * math.pi)
            for i in range(0, Q, chunk_size):
                qc = q[i : i + chunk_size]
                a = v0[None, :, :] - qc[:, None, :]
                b = v1[None, :, :] - qc[:, None, :]
                c = v2[None, :, :] - qc[:, None, :]
                la, lb, lc = _torch_reg_lengths(a, b, c, eps_world)
                N = (a * torch.cross(b, c, dim=-1)).sum(-1)
                D = (
                    la * lb * lc
                    + (a * b).sum(-1) * lc
                    + (b * c).sum(-1) * la
                    + (c * a).sum(-1) * lb
                )
                out[i : i + chunk_size] = torch.atan2(N, D).sum(dim=1) * inv_two_pi
        ws.ev_stop.record(ws.stream)
        torch.cuda.synchronize()
        return out.cpu().numpy(), ws.ev_start.elapsed_time(ws.ev_stop)


def _torch_ref_backward_triangle(batch, n_q, eps_world, ws, chunk_size=64):
    with torch.cuda.stream(ws.stream):
        v = batch.tris_gpu.clone().requires_grad_(True)
        q = batch.queries_gpu[:n_q]
        g = batch.grad_output_gpu[:n_q]
        Q = q.shape[0]
        inv_two_pi = 1.0 / (2.0 * math.pi)

        torch.cuda.synchronize()
        ws.ev_start.record(ws.stream)
        for i in range(0, Q, chunk_size):
            qc = q[i : i + chunk_size]
            gc = g[i : i + chunk_size]
            a = v[None, :, 0, :] - qc[:, None, :]
            b = v[None, :, 1, :] - qc[:, None, :]
            c = v[None, :, 2, :] - qc[:, None, :]
            la, lb, lc = _torch_reg_lengths(a, b, c, eps_world)
            N = (a * torch.cross(b, c, dim=-1)).sum(-1)
            D = (
                la * lb * lc
                + (a * b).sum(-1) * lc
                + (b * c).sum(-1) * la
                + (c * a).sum(-1) * lb
            )
            loss = (torch.atan2(N, D) * inv_two_pi * gc[:, None]).sum()
            loss.backward()
        ws.ev_stop.record(ws.stream)
        torch.cuda.synchronize()
        return v.grad.detach().cpu().numpy(), ws.ev_start.elapsed_time(ws.ev_stop)

def _torch_ref_forward_winding_point_normal(
    batch: MeshBatch, n_q: int, epsilon: float, ws, chunk_size=256
):
    with torch.cuda.stream(ws.stream):
        p, n = batch.get_point_normals()
        q = batch.queries_gpu[:n_q]
        Q = q.shape[0]
        out = torch.empty(Q, device="cuda", dtype=torch.float32)

        inv_eps = 1.0 / epsilon
        inv_4pi = 1.0 / (4.0 * math.pi)
        four_over_3sqrt_pi = 4.0 / (3.0 * math.sqrt(math.pi))

        torch.cuda.synchronize()
        ws.ev_start.record(ws.stream)
        with torch.no_grad():
            for i in range(0, Q, chunk_size):
                qc = q[i : i + chunk_size]
                d = p[None, :, :] - qc[:, None, :]
                dist2 = (d * d).sum(-1)
                inv_dist = torch.rsqrt(dist2 + 1e-20)
                inv_dist3 = inv_dist**3
                t = dist2 * inv_dist * inv_eps
                near = four_over_3sqrt_pi * inv_eps**3
                mid_s = torch.erf(t) - (2.0 / math.sqrt(math.pi)) * t * torch.exp(
                    -t * t
                )
                mid = mid_s * inv_dist3
                far = inv_dist3
                s_over_d3 = torch.where(t < 0.1, near, torch.where(t < 2.0, mid, far))
                contrib = (n[None, :, :] * d).sum(-1) * inv_4pi * s_over_d3
                out[i : i + chunk_size] = contrib.sum(dim=1)
        ws.ev_stop.record(ws.stream)
        torch.cuda.synchronize()
        return out.cpu().numpy(), ws.ev_start.elapsed_time(ws.ev_stop)


def _torch_ref_backward_point_normal(
    batch: MeshBatch, n_q: int, epsilon: float, ws, chunk_size=64
):
    with torch.cuda.stream(ws.stream):
        p_base, n_base = batch.get_point_normals()
        p = p_base.clone().requires_grad_(True)
        n = n_base.clone().requires_grad_(True)
        q = batch.queries_gpu[:n_q]
        g = batch.grad_output_gpu[:n_q]
        Q = q.shape[0]

        inv_eps = 1.0 / epsilon
        inv_4pi = 1.0 / (4.0 * math.pi)
        four_over_3sqrt_pi = 4.0 / (3.0 * math.sqrt(math.pi))

        torch.cuda.synchronize()
        ws.ev_start.record(ws.stream)
        for i in range(0, Q, chunk_size):
            qc = q[i : i + chunk_size]
            gc = g[i : i + chunk_size]
            d = p[None, :, :] - qc[:, None, :]
            dist2 = (d * d).sum(-1)
            inv_dist = torch.rsqrt(dist2 + 1e-20)
            inv_dist3 = inv_dist**3
            t = dist2 * inv_dist * inv_eps
            near = four_over_3sqrt_pi * inv_eps**3
            mid_s = torch.erf(t) - (2.0 / math.sqrt(math.pi)) * t * torch.exp(-t * t)
            mid = mid_s * inv_dist3
            far = inv_dist3
            s_over_d3 = torch.where(t < 0.1, near, torch.where(t < 2.0, mid, far))
            contrib = (n[None, :, :] * d).sum(-1) * inv_4pi * s_over_d3
            loss = (contrib * gc[:, None]).sum()
            loss.backward()
        ws.ev_stop.record(ws.stream)
        torch.cuda.synchronize()
        return (
            np.stack(
                [p.grad.detach().cpu().numpy(), n.grad.detach().cpu().numpy()], axis=1
            ),
            ws.ev_start.elapsed_time(ws.ev_stop),
        )


# =============================================================================
# Mode runners
# =============================================================================

def run_forward_triangle(batch, args, ws, epsilon):
    ref, fast, tb, tf = _forward_triangle_engines(batch, args, ws, epsilon)
    m = _forward_metrics(ref, fast, tb, tf)
    m["_ref_full"] = ref
    m["_fast_full"] = fast
    return m


def run_forward_point_normal(batch, args, ws, epsilon):
    ref, fast, tb, tf = _forward_point_normal_engines(batch, args, ws, epsilon)
    m = _forward_metrics(ref, fast, tb, tf)
    m["_ref_full"] = ref
    m["_fast_full"] = fast
    return m


def run_backward_triangle(batch, args, ws, epsilon):
    ref, fast, tb, tf = _backward_triangle_engines(batch, args, ws, epsilon)
    m = _backward_metrics(ref, fast, tb, tf)
    m["_ref_full"] = ref
    m["_fast_full"] = fast
    return m


def run_backward_mesh(batch, args, ws, epsilon):
    ref, fast, tb, tf = _backward_mesh_engines(batch, args, ws, epsilon)
    m = _backward_metrics(ref, fast, tb, tf)
    m["_ref_full"] = ref
    m["_fast_full"] = fast
    return m


def run_backward_point_normal(batch, args, ws, epsilon):
    ref, fast, tb, tf = _backward_point_normal_engines(batch, args, ws, epsilon)
    m = _backward_metrics(ref, fast, tb, tf)
    m["_ref_full"] = ref
    m["_fast_full"] = fast
    return m



MODE_RUNNERS = {
    "forward_triangle": run_forward_triangle,
    "forward_point_normal": run_forward_point_normal,
    "backward_triangle": run_backward_triangle,
    "backward_mesh": run_backward_mesh,
    "backward_point_normal": run_backward_point_normal,
}

def _torch_world_epsilon(tris_np: np.ndarray, epsilon: float) -> float:
    """Match the C++ convention: epsilon is a fraction of the scene extent."""
    if epsilon <= 0.0:
        return 0.0
    flat = tris_np.reshape(-1, 3)
    extent = flat.max(0) - flat.min(0)
    return float(epsilon * np.max(extent))


def _run_torch_reference(mode, batch, args, ws, ref_full, fast_full, epsilon):
    n_q = min(args.torch_ref_query_count, batch.queries_np.shape[0])
    eps_world = _torch_world_epsilon(batch.tris_np, epsilon)

    if mode == "forward_triangle":
        vals, t_ms = _torch_ref_forward_winding_triangle(batch, n_q, eps_world, ws)
        ref_cuda = ref_full[:n_q]
    elif mode == "forward_point_normal":
        vals, t_ms = _torch_ref_forward_winding_point_normal(
            batch, n_q, eps_world, ws
        )
        ref_cuda = ref_full[:n_q]
    elif mode == "backward_triangle":
        vals, t_ms = _torch_ref_backward_triangle(batch, n_q, eps_world, ws)
        ref_cuda = ref_full
    elif mode == "backward_point_normal":
        vals, t_ms = _torch_ref_backward_point_normal(
            batch, n_q, eps_world, ws
        )
        ref_cuda = ref_full
    else:
        return {}

    diff = vals.reshape(-1) - ref_cuda.reshape(-1)
    rms = float(np.sqrt(np.mean(diff * diff)))
    nr = float(np.linalg.norm(ref_cuda.reshape(-1)))
    nv = float(np.linalg.norm(vals.reshape(-1)))
    cos = float(np.dot(vals.reshape(-1), ref_cuda.reshape(-1)) / (nr * nv + 1e-30))
    return {
        "time_ms": float(t_ms),
        "n_queries_used": int(n_q),
        "rms_vs_cuda_brute": rms,
        "cosine_vs_cuda_brute": cos,
        "epsilon_world": eps_world,
    }

# =============================================================================
# Per-bucket metrics (backward modes)
# =============================================================================


def _compute_bucket_metrics(
    mode_name, batch, args, ws, distances, bucket_edges, diagonal, epsilon,
):
    runner = MODE_RUNNERS[mode_name]
    out: dict = {}
    for lo, hi in bucket_edges:
        label = _bucket_label(lo, hi, diagonal)
        mask = (distances >= lo) & (distances < hi)
        n_in = int(mask.sum())
        if n_in == 0:
            out[label] = {"n_queries": 0, "empty": True}
            continue

        masked = batch.with_masked_grad_output(mask.astype(np.float32))
        try:
            m = runner(masked, args, ws, epsilon)
        except Exception as exc:
            out[label] = {
                "n_queries": n_in,
                "error": f"{type(exc).__name__}: {exc}",
            }
            continue

        m.pop("_ref_full", None)
        m.pop("_fast_full", None)
        m["n_queries"] = n_in
        out[label] = m
    return out


# =============================================================================
# Per-mesh evaluation — split into a CPU stage and a GPU stage.
# =============================================================================


def prepare_mesh(name, tris, args, vertices=None, indices=None) -> dict:
    """CPU-only per-mesh preparation.

    Computes mesh modification, query sampling, distance to surface (for
    filtering and bucketing), and gradient targets. Does not touch CUDA.
    """
    n_tri = len(tris)
    if n_tri < 4:
        raise ValueError(f"too few triangles: {n_tri}")
    if not np.isfinite(tris).all():
        raise FatalNumericalError(
            "input has non-finite vertices", name=name, dump_path=None
        )

    if args.max_triangles and n_tri > args.max_triangles:
        rng = np.random.default_rng(zlib.crc32(name.encode()))
        keep = rng.choice(n_tri, args.max_triangles, replace=False)
        tris = tris[keep]
        if indices is not None:
            indices = indices[keep]
        n_tri = args.max_triangles

    tris = apply_mesh_modification(
        tris,
        args.break_mesh,
        seed=zlib.crc32(name.encode()),
        noise_sigma_frac=args.noise_sigma_frac,
    )
    n_tri = len(tris)
    if n_tri < 4:
        raise ValueError(f"too few triangles after modification: {n_tri}")

    queries = sample_queries(
        tris, args.query_count, zlib.crc32(name.encode()) + 1, mode=args.query_mode
    )

    diagonal = _triangles_diagonal(tris)

    # Compute distance to surface once, if we need it for either filtering
    # or bucketing.
    distances = None
    if args.filter_query_distance > 0 or args.bucket_edges:
        distances = compute_query_distances(queries, tris)

    n_queries_pre_filter = len(queries)
    min_dist = 0.0
    if args.filter_query_distance > 0:
        min_dist = args.filter_query_distance * diagonal
        keep = distances > min_dist
        queries = queries[keep]
        distances = distances[keep]
        if len(queries) < 4:
            raise ValueError(
                f"query filter removed too many queries: "
                f"{n_queries_pre_filter} -> {len(queries)}"
            )

    if not np.isfinite(queries).all():
        raise FatalNumericalError("non-finite queries", name=name, dump_path=None)

    grad_output_np = make_grad_output(len(queries), name)

    return {
        "name": name,
        "tris": np.ascontiguousarray(tris),
        "queries": np.ascontiguousarray(queries),
        "grad_output": grad_output_np,
        "vertices": vertices,
        "indices": indices,
        "n_tri": n_tri,
        "n_queries": len(queries),
        "n_queries_pre_filter": n_queries_pre_filter,
        "min_dist": min_dist,
        "distances": distances,
        "diagonal": diagonal,
    }


def evaluate_mesh(prep: dict, args, ws: WorkerStream) -> dict:
    name = prep["name"]
    n_tri = prep["n_tri"]

    batch = MeshBatch.build(
        prep["tris"], prep["queries"], prep["grad_output"],
        prep["vertices"], prep["indices"],
    )

    record: dict[str, Any] = {
        "name": name,
        "n_tri": n_tri,
        "n_queries": prep["n_queries"],
        "n_queries_pre_filter": prep["n_queries_pre_filter"],
        "query_filter_min_distance": prep["min_dist"],
        "n_points": n_tri,
        "break_mesh": args.break_mesh,
        "modes": {},
    }

    bucket_edges = (
        _parse_bucket_edges(args.bucket_edges, prep["diagonal"])
        if args.bucket_edges and prep["distances"] is not None
        else []
    )

    # When multiple epsilons are swept, mode keys gain an "@eps=" suffix so
    # the resulting JSONL stays a flat dict. The suffix is omitted when only
    # one epsilon is being evaluated.
    eps_list = args.epsilon_list
    use_suffix = len(eps_list) > 1

    for eps in eps_list:
        for mode_name in args.modes:
            runner = MODE_RUNNERS[mode_name]
            try:
                metrics = runner(batch, args, ws, eps)
            except Exception as exc:
                key = f"{mode_name}@eps={eps}" if use_suffix else mode_name
                record["modes"][key] = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "epsilon": float(eps),
                }
                continue

            ref_full = metrics.pop("_ref_full", None)
            fast_full = metrics.pop("_fast_full", None)

            if args.with_torch_ref:
                try:
                    metrics["torch_ref"] = _run_torch_reference(
                        mode_name, batch, args, ws, ref_full, fast_full, eps
                    )
                except Exception as exc:
                    metrics["torch_ref"] = {
                        "error": f"{type(exc).__name__}: {exc}"
                    }

            if bucket_edges and mode_name.startswith("backward"):
                try:
                    metrics["buckets"] = _compute_bucket_metrics(
                        mode_name, batch, args, ws,
                        prep["distances"], bucket_edges, prep["diagonal"],
                        eps,
                    )
                except Exception as exc:
                    metrics["buckets"] = {
                        "error": f"{type(exc).__name__}: {exc}"
                    }

            metrics["epsilon"] = float(eps)
            key = f"{mode_name}@eps={eps}" if use_suffix else mode_name
            record["modes"][key] = metrics

    # Forward-triangle summarisation: promote the FIRST variant found
    # (or the eps=0 variant if sweeping) to the top-level record.
    # This keeps the aggregates in Aggregates.update() working.
    for k, v in record["modes"].items():
        base = k.split("@eps=")[0]
        if base == "forward_triangle" and "error" not in v:
            for field in (
                "signal_max", "rms_abs", "mean_abs", "p50_abs", "p95_abs",
                "p99_abs", "max_abs", "rms_rel_masked", "p99_rel_masked",
                "misclass_count", "misclass_frac",
                "time_brute_ms", "time_fast_ms", "speedup",
            ):
                if field in v:
                    record[field] = v[field]
            break

    if args.memory_diagnostics and random.random() < 0.1:
        alloc = torch.cuda.memory_allocated() / 1e9
        reserve = torch.cuda.memory_reserved() / 1e9
        libcudart = ctypes.CDLL("libcudart.so")
        libcudart.cudaDeviceGetDefaultMemPool.restype = ctypes.c_int
        pool = ctypes.c_void_p()
        libcudart.cudaDeviceGetDefaultMemPool(ctypes.byref(pool), 0)
        value = ctypes.c_size_t()
        libcudart.cudaMemPoolGetAttribute(pool, 3, ctypes.byref(value))
        free, total = torch.cuda.mem_get_info()
        print(
            f"[mem] torch allocated={alloc:.2f}GB reserved={reserve:.2f}GB "
            f"cuda_pool_reserved={value.value / 1e9:.2f}GB "
            f"device_free={free / 1e9:.2f}GB "
            f"GC: {len(gc.get_objects())}"
        )

    return record

# =============================================================================
# Source interface
# =============================================================================


class MeshSource(ABC):
    @abstractmethod
    def enumerate(self, name_filter: Optional[str]) -> Iterator[Tuple[str, Any]]: ...

    @abstractmethod
    def load(self, name: str, payload: Any) -> np.ndarray: ...

    @property
    def parse_in_producer(self) -> bool:
        return True

    def preprocess(self, name: str, tris: np.ndarray) -> np.ndarray:
        return tris

    def load_with_indices(
        self, name: str, payload: Any
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        return self.load(name, payload), None, None

    @property
    def dataset_name(self) -> str:
        return type(self).__name__.replace("Source", "")

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None: ...

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "MeshSource":
        return cls()


# =============================================================================
# CLI
# =============================================================================


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--query_count", type=int, default=1_000_000)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--max_triangles", type=int, default=200_000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--preprocess_workers", type=int, default=0)
    parser.add_argument("--queue_size", type=int, default=8)
    parser.add_argument("--name_filter", type=str, default=None)
    parser.add_argument("--max_meshes", type=int, default=None)
    parser.add_argument("--query_mode", choices=["grid", "random"], default="grid")
    parser.add_argument("--no_subsample", action="store_true")
    parser.add_argument("--dump_failures", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stop_on_fatal", action="store_true")
    parser.add_argument(
        "--modes",
        type=str,
        default="forward_triangle",
        help="Comma-separated list of evaluation modes.",
    )
    parser.add_argument(
        "--with-torch-ref",
        action="store_true",
        help="Also run a small PyTorch autograd reference per mode.",
    )
    parser.add_argument("--torch_ref_query_count", type=int, default=4096)
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.004,
        help="Regularization fraction of scene extent. Default 0.004 (= 1/250). "
             "Used when --epsilon_sweep is empty. Pass 0 for the sharp kernel.",
    )
    parser.add_argument(
        "--epsilon_sweep",
        type=str,
        default="",
        help="Comma-separated list of epsilon values, e.g. "
             "'0,0.001,0.002,0.004,0.008,0.016'. When set, each mode is "
             "evaluated once per epsilon; mode keys in the JSONL output get "
             "an '@eps=<value>' suffix. Ignored if empty.",
    )
    parser.add_argument(
        "--break-mesh",
        type=str,
        default="none",
        nargs="+",
        choices=["none", "slice", "drop_half", "drop_80pct", "noise"],
        help="Mesh transformation before evaluation. Comma-separated to chain.",
    )
    parser.add_argument("--noise-sigma-frac", type=float, default=1e-2)
    parser.add_argument(
        "--filter-query-distance",
        type=float,
        default=0.0,
        help="Remove queries closer than this fraction of the mesh diagonal "
             "to the triangle surface. 0 disables filtering.",
    )
    parser.add_argument(
        "--bucket-edges",
        type=str,
        default="",
        help="Comma-separated distances (fractions of the mesh diagonal) "
             "that define distance-to-surface buckets for BACKWARD modes. "
             "Empty disables bucketing.",
    )
    parser.add_argument("--memory_diagnostics", action="store_true")


def finalize_args(args: argparse.Namespace) -> None:
    if args.query_mode == "grid":
        side = int(round(args.query_count ** (1.0 / 3.0)))
        if side**3 != args.query_count:
            print(
                f"  Note: --query_count {args.query_count} is not a perfect "
                f"cube. Rounding to {side**3} ({side}³)."
            )
            args.query_count = side**3
    if args.no_subsample:
        args.max_triangles = 0
    if args.modes == "all":
        args.modes = list(MODE_RUNNERS.keys())
    else:
        args.modes = [m.strip() for m in args.modes.split(",") if m.strip()]
        for m in args.modes:
            if m not in MODE_RUNNERS:
                raise SystemExit(f"unknown mode: {m}")
    if args.preprocess_workers <= 0:
        args.preprocess_workers = args.workers

    if args.epsilon_sweep.strip():
        try:
            args.epsilon_list = [
                float(x) for x in args.epsilon_sweep.split(",") if x.strip()
            ]
        except ValueError as exc:
            raise SystemExit(f"invalid --epsilon_sweep: {exc}")
        if not args.epsilon_list:
            args.epsilon_list = [args.epsilon]
    else:
        args.epsilon_list = [args.epsilon]

    if args.bucket_edges:
        try:
            _ = [float(x) for x in args.bucket_edges.split(",") if x.strip()]
        except ValueError as exc:
            raise SystemExit(f"invalid --bucket-edges: {exc}")



# =============================================================================
# Pipeline threads
# =============================================================================


def _put_blocking(q: queue.Queue, item, stop_event: threading.Event) -> bool:
    while not stop_event.is_set():
        try:
            q.put(item, timeout=0.5)
            return True
        except queue.Full:
            continue
    return False


def _put_sentinel(q: queue.Queue, item, timeout: float = 2.0) -> None:
    try:
        q.put(item, timeout=timeout)
    except queue.Full:
        pass


def producer_loop(reader, raw_queue, done_set, args, stop_event, counter):
    produced = 0
    skipped = 0
    parse_errors = 0

    try:
        for name, payload in reader.enumerate(args.name_filter):
            if stop_event.is_set():
                break
            if args.max_meshes and produced >= args.max_meshes:
                break
            if name in done_set:
                skipped += 1
                continue
            if reader.parse_in_producer:
                try:
                    tris, verts, idxs = reader.load_with_indices(name, payload)
                except Exception as exc:
                    if not _put_blocking(
                        raw_queue,
                        (name, None, None, None, f"parse: {exc}"),
                        stop_event,
                    ):
                        break
                    parse_errors += 1
                    continue
                if not _put_blocking(
                    raw_queue, (name, tris, verts, idxs, None), stop_event
                ):
                    break
            else:
                if not _put_blocking(
                    raw_queue, (name, None, None, None, payload), stop_event
                ):
                    break
            produced += 1
    finally:
        for _ in range(args.preprocess_workers):
            if stop_event.is_set():
                break
            try:
                raw_queue.put(None, timeout=0.5)
            except queue.Full:
                _put_sentinel(raw_queue, None, timeout=2.0)
        with counter["lock"]:
            counter["produced"] = produced
            counter["skipped"] = skipped
            counter["parse_errors"] = parse_errors


def preprocess_loop(
    worker_id, reader, raw_queue, input_queue, args, stop_event, preprocess_done
):
    while True:
        try:
            item = raw_queue.get(timeout=0.5)
        except queue.Empty:
            if stop_event.is_set():
                break
            continue

        if item is None:
            break

        name, tris, verts, idxs, payload_or_err = item

        if isinstance(payload_or_err, str):
            _put_blocking(
                input_queue, (name, None, {"error": payload_or_err}), stop_event
            )
            continue

        try:
            if tris is None:
                tris, verts, idxs = reader.load_with_indices(name, payload_or_err)
            tris = reader.preprocess(name, tris)
            prep = prepare_mesh(name, tris, args, vertices=verts, indices=idxs)
            _put_blocking(input_queue, (name, prep, None), stop_event)

        except FatalNumericalError as exc:
            err_item = (
                name,
                None,
                {
                    "error": str(exc),
                    "fatal": True,
                    "dump": str(exc.dump_path) if exc.dump_path else None,
                },
            )
            _put_blocking(input_queue, err_item, stop_event)
            if args.stop_on_fatal:
                stop_event.set()
                break

        except Exception as exc:
            tb = traceback.format_exc()
            err_item = (
                name,
                None,
                {"error": f"{type(exc).__name__}: {exc}\n{tb}"},
            )
            _put_blocking(input_queue, err_item, stop_event)

    with preprocess_done["lock"]:
        preprocess_done["count"] += 1
        is_last = preprocess_done["count"] >= args.preprocess_workers
    if is_last:
        for _ in range(args.workers):
            _put_sentinel(input_queue, None, timeout=2.0)


def worker_loop(worker_id, input_queue, output_queue, args, stop_event):
    ws = WorkerStream()
    sent_done = False
    try:
        while True:
            try:
                item = input_queue.get(timeout=0.5)
            except queue.Empty:
                if stop_event.is_set():
                    return
                continue

            if item is None:
                output_queue.put(None)
                sent_done = True
                return

            name, prep, err = item

            if err is not None:
                output_queue.put((name, {"name": name, **err}))
                if err.get("fatal") and args.stop_on_fatal:
                    stop_event.set()
                    return
                continue

            try:
                with torch.cuda.stream(ws.stream):
                    record = evaluate_mesh(prep, args, ws)
                output_queue.put((name, record))

            except FatalNumericalError as exc:
                output_queue.put(
                    (
                        name,
                        {
                            "name": name,
                            "error": str(exc),
                            "fatal": True,
                            "dump": str(exc.dump_path) if exc.dump_path else None,
                        },
                    )
                )
                if args.stop_on_fatal:
                    stop_event.set()
                    return

            except Exception as exc:
                tb = traceback.format_exc()
                output_queue.put(
                    (
                        name,
                        {"name": name, "error": f"{type(exc).__name__}: {exc}\n{tb}"},
                    )
                )
    finally:
        if not sent_done:
            _put_sentinel(output_queue, None, timeout=2.0)


def writer_loop(results_path, output_queue, agg, stop_event, args, start_time):
    print_lock = threading.Lock()
    fp = open(results_path, "a", buffering=1)
    last_print = time.time()
    n_workers_done = 0
    try:
        while True:
            try:
                item = output_queue.get(timeout=0.5)
            except queue.Empty:
                if stop_event.is_set() and n_workers_done >= args.workers:
                    break
                continue
            if item is None:
                n_workers_done += 1
                if n_workers_done >= args.workers:
                    break
                continue
            _, rec = item
            fp.write(json.dumps(rec) + "\n")
            if "error" in rec:
                with print_lock:
                    print(
                        f"[ERROR] {rec['name']}: {rec['error'].splitlines()[0]}",
                        flush=True,
                    )
                continue
            agg.update(rec)
            now = time.time()
            if now - last_print > 2.0:
                elapsed = now - start_time
                rate = agg.n_meshes / max(elapsed, 1e-9)
                with print_lock:
                    modes = list(rec.get("modes", {}).keys())
                    print(
                        f"[{agg.n_meshes:5d}] {rate:5.2f} item/s  "
                        f"modes={','.join(modes)[:40]:<40}  "
                        f"last={rec['name'].split('/')[-1]}",
                        flush=True,
                    )
                last_print = now
    finally:
        fp.close()


# =============================================================================
# Aggregates
# =============================================================================


@dataclass
class Aggregates:
    n_meshes: int = 0
    n_errors: int = 0
    per_mode_speedup: dict = field(default_factory=dict)
    per_mode_rms: dict = field(default_factory=dict)
    per_mode_time_fast: dict = field(default_factory=dict)

    def update(self, rec: dict) -> None:
        if not isinstance(rec, dict) or rec.get("error"):
            self.n_errors += 1
            return
        self.n_meshes += 1
        for mode_name, metrics in rec.get("modes", {}).items():
            if not isinstance(metrics, dict) or "error" in metrics:
                continue
            for key, target in (
                ("speedup", self.per_mode_speedup),
                ("rms_abs", self.per_mode_rms),
                ("time_fast_ms", self.per_mode_time_fast),
            ):
                v = metrics.get(key)
                if v is None:
                    continue
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(v):
                    target.setdefault(mode_name, Welford()).update(v)

    def summary_lines(self) -> list:
        lines = [
            f"Items processed   : {self.n_meshes}",
            f"Items with errors : {self.n_errors}",
        ]
        for mode_name in sorted(self.per_mode_speedup.keys()):
            sp = self.per_mode_speedup[mode_name]
            rms = self.per_mode_rms.get(mode_name, Welford())
            tf = self.per_mode_time_fast.get(mode_name, Welford())
            lines.append("")
            lines.append(f"[{mode_name}]")
            lines.append(
                f"  speedup   : mean={sp.mean:.2f}x  std={sp.std:.2f}  n={sp.n}"
            )
            lines.append(f"  rms_abs   : mean={rms.mean:.4e}  std={rms.std:.4e}")
            lines.append(f"  fast_ms   : mean={tf.mean:.3f}  std={tf.std:.3f}")
        return lines


def load_existing(results_path: Path):
    done_set = set()
    agg = Aggregates()
    if not results_path.exists():
        return done_set, agg
    n_bad = 0
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                n_bad += 1
                continue
            done_set.add(rec["name"])
            agg.update(rec)
    if n_bad:
        print(f"  Note: {n_bad} unparsable lines skipped")
    return done_set, agg


# =============================================================================
# Summary + driver
# =============================================================================


def print_summary(agg: Aggregates, output_dir: Path, dataset_name: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {dataset_name} evaluation summary")
    print("=" * 72)
    for line in agg.summary_lines():
        print(f"  {line}")
    print("=" * 72)
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        for line in agg.summary_lines():
            f.write(line + "\n")
    print(f"  Summary written to {summary_path}")


def run_eval(reader: MeshSource, args: argparse.Namespace):
    dataset_name = reader.dataset_name
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"

    print(f"Dataset:        {dataset_name}")
    print(f"Output:         {output_dir}")
    print(f"Queries/mesh:   {args.query_count:,}")
    print(f"Beta:           {args.beta}")
    print(f"Modes:          {', '.join(args.modes)}")
    if len(args.epsilon_list) > 1:
        print(f"Epsilon sweep:  {args.epsilon_list}")
    else:
        print(f"Epsilon:        {args.epsilon_list[0]}")
    print(f"GPU workers:    {args.workers}  (each on its own CUDA stream)")
    print(f"Preprocessors:  {args.preprocess_workers}  (CPU-only threads)")
    if args.bucket_edges:
        print(f"Bucket edges:   {args.bucket_edges}  (backward modes only)")
    if args.with_torch_ref:
        print(f"Torch ref:      {args.torch_ref_query_count} queries")

    print(f"\nLoading existing results from {results_path}...")
    done_set, agg = load_existing(results_path)
    if args.force:
        print("  --force set: ignoring existing checkpoint")
        done_set = set()
    print(f"  Found {len(done_set)} completed items")

    raw_queue = queue.Queue(maxsize=args.queue_size)
    input_queue = queue.Queue(maxsize=args.queue_size)
    output_queue = queue.Queue()
    stop_event = threading.Event()
    counter = {"lock": threading.Lock(), "produced": 0, "skipped": 0, "parse_errors": 0}
    preprocess_done = {"lock": threading.Lock(), "count": 0}

    start_time = time.time()

    producer = threading.Thread(
        target=producer_loop,
        args=(reader, raw_queue, done_set, args, stop_event, counter),
        daemon=True,
        name="producer",
    )
    producer.start()

    preprocessors = []
    for i in range(args.preprocess_workers):
        t = threading.Thread(
            target=preprocess_loop,
            args=(i, reader, raw_queue, input_queue, args, stop_event, preprocess_done),
            daemon=True,
            name=f"preprocess-{i}",
        )
        t.start()
        preprocessors.append(t)

    workers = []
    for i in range(args.workers):
        t = threading.Thread(
            target=worker_loop,
            args=(i, input_queue, output_queue, args, stop_event),
            daemon=True,
            name=f"worker-{i}",
        )
        t.start()
        workers.append(t)

    writer = threading.Thread(
        target=writer_loop,
        args=(results_path, output_queue, agg, stop_event, args, start_time),
        daemon=True,
        name="writer",
    )
    writer.start()

    try:
        producer.join()
        for t in preprocessors:
            t.join()
        for t in workers:
            t.join()
        writer.join(timeout=5.0)
    except KeyboardInterrupt:
        print("\n\033[93mInterrupted by user\033[0m")
        stop_event.set()
        writer.join(timeout=3.0)
        producer.join(timeout=2.0)
        for t in preprocessors:
            t.join(timeout=2.0)
        for t in workers:
            t.join(timeout=2.0)

    print(
        f"\nProducer stats: produced={counter['produced']}, "
        f"skipped={counter['skipped']}, parse_errors={counter['parse_errors']}"
    )
    print_summary(agg, output_dir, dataset_name)
    return results_path, agg
