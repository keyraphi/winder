#!/usr/bin/env python3
"""Standalone benchmark: Winder's CUDA backward pass vs PyTorch autograd.

The full dataset sweep in eval_pipeline.py compares CUDA brute vs CUDA fast,
which measures the algorithmic acceleration independent of hardware. PyTorch
autograd is the natural external baseline for gradients, but it is orders of
magnitude slower at dataset scale and would dominate the runtime. This script
runs a small, representative comparison on a handful of meshes and produces a
single CSV/table that supports the paper's headline claim:

    "our CUDA backward implementation is Nx faster than PyTorch autograd"

Notes:
  * Timings are wall-clock with explicit torch.cuda.synchronize() at the
    boundaries so that CPU-side autograd overhead is included. CUDA-event
    timing would hide that overhead and unfairly favor CUDA.
  * A geometric mean is reported across all (mesh, query_count, mode) pairs.
  * Chunking of the PyTorch reference is automatic; the chunk size adapts to
    keep peak GPU memory under --torch-memory-budget-bytes.

Usage:
    # Sample 5 meshes from a Thingi10K archive
    python benchmark_backward_vs_torch.py \
        --archive /graphics/scratch3/datasets/thingi10K.tar.gz \
        --n-meshes 5 --query-counts 1000,10000 \
        --modes triangle,mesh,point_normal \
        --output bench_backward.csv

    # Explicit OBJ list
    python benchmark_backward_vs_torch.py \
        --obj-files bunny.obj,armadillo.obj \
        --query-counts 1000,10000,100000 \
        --modes triangle,point_normal
"""

import argparse
import csv
import math
import sys
import zlib
import tarfile
import tempfile
import time
from pathlib import Path

import igl
import numpy as np
import torch

import winder


DEFAULT_EPSILON = 0.004  # = 1/250, matches the library default


# ============================================================================
# Mesh loading
# ============================================================================


def load_obj(path: Path):
    v, _, _, f, _, _ = igl.readOBJ(str(path))
    return v.astype(np.float32), f.astype(np.uint32)


def load_thingi10k_sample(archive: str, n_meshes: int, seed: int = 0):
    """Stream the archive and take the first N meshes with a moderate triangle
    count (200 < n_tri < 20000) so that PyTorch autograd is feasible."""
    rng = np.random.default_rng(seed)
    tris_list = []
    print(f"  Sampling up to {n_meshes} meshes with 200 < n_tri < 20000 ...")
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".stl"):
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            fd, tmp = tempfile.mkstemp(suffix=".stl")
            try:
                with open(fd, "wb") as fh:
                    fh.write(f.read())
                v, fi = igl.read_triangle_mesh(tmp)
                if v.size == 0 or fi.size == 0 or fi.shape[1] != 3:
                    continue
                n_tri = len(fi)
                if 200 < n_tri < 20000:
                    tris_list.append(
                        (
                            member.name.split("/")[-1],
                            v.astype(np.float32),
                            fi.astype(np.uint32),
                        )
                    )
                    print(
                        f"    [{len(tris_list):2d}] {member.name.split('/')[-1]}  "
                        f"n_tri={n_tri}"
                    )
                    if len(tris_list) >= n_meshes:
                        break
            finally:
                try:
                    import os

                    os.unlink(tmp)
                except OSError:
                    pass
    return tris_list


# ============================================================================
# Torch reference implementations
# ============================================================================


def _auto_chunk_size(n_tri: int, user_chunk: int, budget_bytes: int) -> int:
    """Chunk size for PyTorch reference, capped so one chunk fits in memory."""
    # Roughly 60 bytes per (query, triangle) pair in intermediate tensors.
    per_pair = 60
    max_pairs = max(1024, budget_bytes // per_pair)
    fit = max(1, max_pairs // max(1, n_tri))
    return max(1, min(user_chunk, fit))


# Compact-profile constants (must match geometry.h).
_TWO_OVER_SQRT_PI = 1.1283791671
_QUARTIC_A2 = 0.4748737341529164
_QUARTIC_A4 = -0.0606601717798213
_CUBIC_V = math.sqrt(2.0)
_CUBIC_S = 1.0 / math.sqrt(2.0)
_CUBIC_C = -0.6568542494923802
_CUBIC_D = 0.5355339059327378


def _compact_g(t: torch.Tensor) -> torch.Tensor:
    tq = t.clamp(max=1.0)
    tc = t.clamp(min=1.0, max=2.0)
    ti = t.clamp(min=2.0)
    gq = 1.0 + _QUARTIC_A2 * tq * tq + _QUARTIC_A4 * tq**4
    u = tc - 1.0
    gc = _CUBIC_V + _CUBIC_S * u + _CUBIC_C * u * u + _CUBIC_D * u**3
    return torch.where(t <= 1.0, gq, torch.where(t <= 2.0, gc, ti))


def _torch_reg_inv_lengths(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    eps_world: torch.Tensor,
    reg_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return 1/r_a, 1/r_b, 1/r_c matching the CUDA edge regularizer."""
    a2 = (a * a).sum(-1) + 1e-20
    b2 = (b * b).sum(-1) + 1e-20
    c2 = (c * c).sum(-1) + 1e-20
    use_reg = float(eps_world.detach()) > 0.0 and reg_mode != "sharp"
    if not use_reg:
        return a2.rsqrt(), b2.rsqrt(), c2.rsqrt()
    if reg_mode == "plummer":
        eps2 = eps_world * eps_world
        return (
            (a2 + eps2).rsqrt(),
            (b2 + eps2).rsqrt(),
            (c2 + eps2).rsqrt(),
        )
    # compact
    inv_eps = 1.0 / eps_world
    ra = eps_world * _compact_g(a2.sqrt() * inv_eps)
    rb = eps_world * _compact_g(b2.sqrt() * inv_eps)
    rc = eps_world * _compact_g(c2.sqrt() * inv_eps)
    return 1.0 / ra, 1.0 / rb, 1.0 / rc

def _torch_reg_inv_lengths_reg(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    eps_world: torch.Tensor,
    reg_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Regularized inverse lengths; assumes eps_world > 0 and reg_mode != sharp.

    Kept branch-free so it can be traced by torch.compile without a
    Python-level graph break.
    """
    a2 = (a * a).sum(-1) + 1e-20
    b2 = (b * b).sum(-1) + 1e-20
    c2 = (c * c).sum(-1) + 1e-20
    if reg_mode == "plummer":
        eps2 = eps_world * eps_world
        return (a2 + eps2).rsqrt(), (b2 + eps2).rsqrt(), (c2 + eps2).rsqrt()
    # compact
    inv_eps = 1.0 / eps_world
    ra = eps_world * _compact_g(a2.sqrt() * inv_eps)
    rb = eps_world * _compact_g(b2.sqrt() * inv_eps)
    rc = eps_world * _compact_g(c2.sqrt() * inv_eps)
    return 1.0 / ra, 1.0 / rb, 1.0 / rc


def _triangle_grads_torch(
    triangles_np: np.ndarray,
    queries: np.ndarray,
    grad_output_np: np.ndarray,
    eps_world: float,
    reg_mode: str,
    chunk_size: int,
    try_compile: bool,
    need_eps_grad: bool,
) -> tuple[np.ndarray, float, float, float, str | None]:
    """Return (vertex_grads, eps_grad, geom_ms, eps_ms, compile_error)."""
    device = "cuda"
    tris = torch.from_numpy(triangles_np).to(device)
    q = torch.from_numpy(queries).to(device)
    g = torch.from_numpy(grad_output_np).to(device)

    N = tris.shape[0]
    Q = q.shape[0]
    chunk = _auto_chunk_size(N, chunk_size, 1_000_000_000)
    inv_two_pi = 1.0 / (2.0 * math.pi)

    # Branch selection outside the compiled region. Keeping this as a plain
    # Python bool means dynamo sees a single branch-free loss function and
    # does not emit a graph break on `float(eps_t.detach())`.
    use_reg = eps_world > 0.0 and reg_mode != "sharp"

    def _chunk_loss_reg(v, eps_t, qc, gc):
        a = v[None, :, 0, :] - qc[:, None, :]
        b = v[None, :, 1, :] - qc[:, None, :]
        c = v[None, :, 2, :] - qc[:, None, :]
        ia, ib, ic = _torch_reg_inv_lengths_reg(a, b, c, eps_t, reg_mode)
        P = (a * torch.cross(b, c, dim=-1)).sum(-1) * ia * ib * ic
        Qn = (
            1.0
            + (a * b).sum(-1) * ia * ib
            + (a * c).sum(-1) * ia * ic
            + (b * c).sum(-1) * ib * ic
        )
        return (torch.atan2(P, Qn) * inv_two_pi * gc[:, None]).sum()

    def _chunk_loss_sharp(v, eps_t, qc, gc):  # noqa: ARG001 (eps_t unused)
        a = v[None, :, 0, :] - qc[:, None, :]
        b = v[None, :, 1, :] - qc[:, None, :]
        c = v[None, :, 2, :] - qc[:, None, :]
        ia = (a * a).sum(-1).add(1e-20).rsqrt()
        ib = (b * b).sum(-1).add(1e-20).rsqrt()
        ic = (c * c).sum(-1).add(1e-20).rsqrt()
        P = (a * torch.cross(b, c, dim=-1)).sum(-1) * ia * ib * ic
        Qn = (
            1.0
            + (a * b).sum(-1) * ia * ib
            + (a * c).sum(-1) * ia * ic
            + (b * c).sum(-1) * ib * ic
        )
        return (torch.atan2(P, Qn) * inv_two_pi * gc[:, None]).sum()

    _chunk_loss = _chunk_loss_reg if use_reg else _chunk_loss_sharp

    compile_error = None
    if try_compile:
        try:
            _chunk_loss = torch.compile(_chunk_loss)
        except Exception as exc:
            compile_error = f"torch.compile failed: {exc}"

    # Warmup: pays compilation / kernel-caching cost before timing.
    v_warm = tris.clone().requires_grad_(True)
    eps_warm = torch.tensor(eps_world, device=device, dtype=torch.float32)
    _ = _chunk_loss(v_warm, eps_warm, q[: min(64, Q)], g[: min(64, Q)])
    torch.cuda.synchronize()
    del v_warm, eps_warm

    # ---- Geometry-only timing ----
    v = tris.clone().requires_grad_(True)
    eps_fixed = torch.tensor(eps_world, device=device, dtype=torch.float32)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(0, Q, chunk):
        loss = _chunk_loss(v, eps_fixed, q[i : i + chunk], g[i : i + chunk])
        loss.backward()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    v_grad = v.grad.detach().cpu().numpy()
    t_geom_ms = (t1 - t0) * 1e3

    # ---- Epsilon-gradient timing ----
    eps_grad_value = float("nan")
    t_eps_ms = float("nan")
    if need_eps_grad and use_reg:
        v = tris.clone().requires_grad_(True)
        eps_var = torch.tensor(
            eps_world, device=device, dtype=torch.float32, requires_grad=True
        )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(0, Q, chunk):
            # Multiplying by 1.0 keeps the autograd edge alive per chunk,
            # so eps_var.grad accumulates across the whole query set.
            eps_chunk = eps_var * 1.0
            loss = _chunk_loss(v, eps_chunk, q[i : i + chunk], g[i : i + chunk])
            loss.backward()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        eps_grad_value = float(eps_var.grad.item()) if eps_var.grad is not None else 0.0
        t_eps_ms = (t1 - t0) * 1e3

    return v_grad, eps_grad_value, t_geom_ms, t_eps_ms, compile_error


def _point_normal_grads_torch(
    triangles_np: np.ndarray,
    queries: np.ndarray,
    grad_output_np: np.ndarray,
    eps_world: float,
    chunk_size: int,
    try_compile: bool,
    need_eps_grad: bool,
) -> tuple[np.ndarray, float, float, float, str | None]:
    """Return (grads, eps_grad, geom_ms, eps_ms, compile_error).

    The point-normal kernel has a fixed regularization profile (the erf-based
    S(t) from geometry.h); there is no compact/plummer switch. Only eps_world
    is configurable.
    """
    # Convert to point-normal
    v0, v1, v2 = triangles_np[:, 0], triangles_np[:, 1], triangles_np[:, 2]
    points = ((v0 + v1 + v2) / 3.0).astype(np.float32)
    cross = np.cross(v1 - v0, v2 - v0)
    mag = np.linalg.norm(cross, axis=-1, keepdims=True)
    areas = mag / 2.0
    safe = np.where(mag == 0, 1e-8, mag)
    normals = (cross / safe) * areas
    normals = normals.astype(np.float32)

    device = "cuda"
    p = torch.from_numpy(points).to(device)
    n = torch.from_numpy(normals).to(device)
    q = torch.from_numpy(queries).to(device)
    g = torch.from_numpy(grad_output_np).to(device)

    N = p.shape[0]
    Q = q.shape[0]
    chunk = _auto_chunk_size(N, chunk_size, 1_000_000_000)
    inv_4pi = 1.0 / (4.0 * math.pi)
    four_over_3sqrt_pi = 4.0 / (3.0 * math.sqrt(math.pi))
    two_over_sqrt_pi = 2.0 / math.sqrt(math.pi)

    # Branch selection outside the compiled region (see triangle comment).
    use_reg = eps_world > 0.0

    def _chunk_loss_reg(p_, n_, eps_t, qc, gc):
        inv_eps = 1.0 / eps_t
        inv_eps3 = inv_eps**3
        d = p_[None, :, :] - qc[:, None, :]
        dist2 = (d * d).sum(-1)
        inv_dist = torch.rsqrt(dist2 + 1e-20)
        inv_dist3 = inv_dist**3
        t = dist2 * inv_dist * inv_eps
        near = four_over_3sqrt_pi * inv_eps3
        mid_s = torch.erf(t) - two_over_sqrt_pi * t * torch.exp(-t * t)
        mid = mid_s * inv_dist3
        s_over_d3 = torch.where(t < 0.1, near, torch.where(t < 2.0, mid, inv_dist3))
        contrib = (n_[None, :, :] * d).sum(-1) * inv_4pi * s_over_d3
        return (contrib * gc[:, None]).sum()

    def _chunk_loss_sharp(p_, n_, eps_t, qc, gc):  # noqa: ARG001 (eps_t unused)
        d = p_[None, :, :] - qc[:, None, :]
        dist2 = (d * d).sum(-1)
        inv_dist = torch.rsqrt(dist2 + 1e-20)
        inv_dist3 = inv_dist**3
        contrib = (n_[None, :, :] * d).sum(-1) * inv_4pi * inv_dist3
        return (contrib * gc[:, None]).sum()

    _chunk_loss = _chunk_loss_reg if use_reg else _chunk_loss_sharp

    compile_error = None
    if try_compile:
        try:
            _chunk_loss = torch.compile(_chunk_loss)
        except Exception as exc:
            compile_error = f"torch.compile failed: {exc}"

    # Warmup.
    p_warm = p.clone().requires_grad_(True)
    n_warm = n.clone().requires_grad_(True)
    eps_warm = torch.tensor(eps_world, device=device, dtype=torch.float32)
    _ = _chunk_loss(p_warm, n_warm, eps_warm, q[: min(64, Q)], g[: min(64, Q)])
    torch.cuda.synchronize()
    del p_warm, n_warm, eps_warm

    # ---- Geometry-only timing ----
    p_ = p.clone().requires_grad_(True)
    n_ = n.clone().requires_grad_(True)
    eps_fixed = torch.tensor(eps_world, device=device, dtype=torch.float32)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(0, Q, chunk):
        loss = _chunk_loss(p_, n_, eps_fixed, q[i : i + chunk], g[i : i + chunk])
        loss.backward()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    grads = np.stack(
        [p_.grad.detach().cpu().numpy(), n_.grad.detach().cpu().numpy()], axis=1
    )
    t_geom_ms = (t1 - t0) * 1e3

    # ---- Epsilon-gradient timing ----
    eps_grad_value = float("nan")
    t_eps_ms = float("nan")
    if need_eps_grad and use_reg:
        p_ = p.clone().requires_grad_(True)
        n_ = n.clone().requires_grad_(True)
        eps_var = torch.tensor(
            eps_world, device=device, dtype=torch.float32, requires_grad=True
        )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(0, Q, chunk):
            eps_chunk = eps_var * 1.0
            loss = _chunk_loss(p_, n_, eps_chunk, q[i : i + chunk], g[i : i + chunk])
            loss.backward()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        eps_grad_value = float(eps_var.grad.item()) if eps_var.grad is not None else 0.0
        t_eps_ms = (t1 - t0) * 1e3

    return grads, eps_grad_value, t_geom_ms, t_eps_ms, compile_error


def _mesh_grads_torch(
    vertices_np: np.ndarray,
    faces_np: np.ndarray,
    queries: np.ndarray,
    grad_output_np: np.ndarray,
    eps_world: float,
    reg_mode: str,
    chunk_size: int,
    try_compile: bool,
    need_eps_grad: bool,
) -> tuple[np.ndarray, float, float, float, str | None]:
    """Return (vertex_grads, eps_grad, geom_ms, eps_ms, compile_error)."""
    device = "cuda"
    v = torch.from_numpy(vertices_np).to(device)
    f = torch.from_numpy(faces_np.astype(np.int64)).to(device)
    q = torch.from_numpy(queries).to(device)
    g = torch.from_numpy(grad_output_np).to(device)

    N = f.shape[0]
    Q = q.shape[0]
    chunk = _auto_chunk_size(N, chunk_size, 1_000_000_000)
    inv_two_pi = 1.0 / (2.0 * math.pi)

    # Branch selection outside the compiled region (see triangle comment).
    use_reg = eps_world > 0.0 and reg_mode != "sharp"

    def _chunk_loss_reg(v_, eps_t, qc, gc):
        tri_v = v_[f]
        a = tri_v[None, :, 0, :] - qc[:, None, :]
        b = tri_v[None, :, 1, :] - qc[:, None, :]
        c = tri_v[None, :, 2, :] - qc[:, None, :]
        ia, ib, ic = _torch_reg_inv_lengths_reg(a, b, c, eps_t, reg_mode)
        P = (a * torch.cross(b, c, dim=-1)).sum(-1) * ia * ib * ic
        Qn = (
            1.0
            + (a * b).sum(-1) * ia * ib
            + (a * c).sum(-1) * ia * ic
            + (b * c).sum(-1) * ib * ic
        )
        return (torch.atan2(P, Qn) * inv_two_pi * gc[:, None]).sum()

    def _chunk_loss_sharp(v_, eps_t, qc, gc):  # noqa: ARG001 (eps_t unused)
        tri_v = v_[f]
        a = tri_v[None, :, 0, :] - qc[:, None, :]
        b = tri_v[None, :, 1, :] - qc[:, None, :]
        c = tri_v[None, :, 2, :] - qc[:, None, :]
        ia = (a * a).sum(-1).add(1e-20).rsqrt()
        ib = (b * b).sum(-1).add(1e-20).rsqrt()
        ic = (c * c).sum(-1).add(1e-20).rsqrt()
        P = (a * torch.cross(b, c, dim=-1)).sum(-1) * ia * ib * ic
        Qn = (
            1.0
            + (a * b).sum(-1) * ia * ib
            + (a * c).sum(-1) * ia * ic
            + (b * c).sum(-1) * ib * ic
        )
        return (torch.atan2(P, Qn) * inv_two_pi * gc[:, None]).sum()

    _chunk_loss = _chunk_loss_reg if use_reg else _chunk_loss_sharp

    compile_error = None
    if try_compile:
        try:
            _chunk_loss = torch.compile(_chunk_loss)
        except Exception as exc:
            compile_error = f"torch.compile failed: {exc}"

    # Warmup.
    v_warm = v.clone().requires_grad_(True)
    eps_warm = torch.tensor(eps_world, device=device, dtype=torch.float32)
    _ = _chunk_loss(v_warm, eps_warm, q[: min(64, Q)], g[: min(64, Q)])
    torch.cuda.synchronize()
    del v_warm, eps_warm

    # ---- Geometry-only timing ----
    v_ = v.clone().requires_grad_(True)
    eps_fixed = torch.tensor(eps_world, device=device, dtype=torch.float32)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(0, Q, chunk):
        loss = _chunk_loss(v_, eps_fixed, q[i : i + chunk], g[i : i + chunk])
        loss.backward()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    v_grad = v_.grad.detach().cpu().numpy()
    t_geom_ms = (t1 - t0) * 1e3

    # ---- Epsilon-gradient timing ----
    eps_grad_value = float("nan")
    t_eps_ms = float("nan")
    if need_eps_grad and use_reg:
        v_ = v.clone().requires_grad_(True)
        eps_var = torch.tensor(
            eps_world, device=device, dtype=torch.float32, requires_grad=True
        )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(0, Q, chunk):
            eps_chunk = eps_var * 1.0
            loss = _chunk_loss(v_, eps_chunk, q[i : i + chunk], g[i : i + chunk])
            loss.backward()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        eps_grad_value = float(eps_var.grad.item()) if eps_var.grad is not None else 0.0
        t_eps_ms = (t1 - t0) * 1e3

    return v_grad, eps_grad_value, t_geom_ms, t_eps_ms, compile_error


# ============================================================================
# CUDA timing helpers
# ============================================================================


def _stream():
    return torch.cuda.current_stream().cuda_stream


def _wall_time_cuda(fn, warmup: int = 1, iters: int = 3) -> float:
    """Wall-clock ms with CUDA synchronize at boundaries."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)
    return float(np.median(times))


def _cuda_backward_triangle(tris_np, queries_np, grad_output_np, beta, eps_fraction):
    device = "cuda"
    t = torch.from_numpy(tris_np).to(device)
    q = torch.from_numpy(queries_np).to(device)
    g = torch.from_numpy(grad_output_np).to(device)
    N = tris_np.shape[0]

    out_brute = torch.empty([N, 3, 3], device=device, dtype=torch.float32)
    out_fast = torch.empty_like(out_brute)
    eps_brute = torch.zeros(1, device=device, dtype=torch.float32)
    eps_fast = torch.zeros(1, device=device, dtype=torch.float32)

    def _brute():
        winder.brute_force_gradients_triangle_soup(
            g, t, q, out_brute, eps_brute, float(eps_fraction), _stream()
        )

    t_brute = _wall_time_cuda(_brute)

    def _fast():
        eng = winder.GradientEngine(q, g, _stream())
        eng.compute_triangle_soup(
            t, out_fast, eps_fast, float(beta), float(eps_fraction), _stream()
        )

    t_fast = _wall_time_cuda(_fast)

    return (
        out_brute.cpu().numpy(),
        out_fast.cpu().numpy(),
        float(eps_brute.item()),
        float(eps_fast.item()),
        t_brute,
        t_fast,
    )


def _cuda_backward_mesh(
    vertices_np, faces_np, queries_np, grad_output_np, beta, eps_fraction
):
    device = "cuda"
    v = torch.from_numpy(vertices_np).to(device)
    f = torch.from_numpy(faces_np.astype(np.uint32)).to(device)
    q = torch.from_numpy(queries_np).to(device)
    g = torch.from_numpy(grad_output_np).to(device)
    K = vertices_np.shape[0]

    out_brute = torch.empty([K, 3], device=device, dtype=torch.float32)
    out_fast = torch.empty_like(out_brute)
    eps_brute = torch.zeros(1, device=device, dtype=torch.float32)
    eps_fast = torch.zeros(1, device=device, dtype=torch.float32)

    def _brute():
        winder.brute_force_gradients_mesh(
            g, v, f, q, out_brute, eps_brute, float(eps_fraction), _stream()
        )

    t_brute = _wall_time_cuda(_brute)

    def _fast():
        eng = winder.GradientEngine(q, g, _stream())
        eng.compute_mesh(
            v, f, out_fast, eps_fast, float(beta), float(eps_fraction), _stream()
        )

    t_fast = _wall_time_cuda(_fast)

    return (
        out_brute.cpu().numpy(),
        out_fast.cpu().numpy(),
        float(eps_brute.item()),
        float(eps_fast.item()),
        t_brute,
        t_fast,
    )


def _cuda_backward_point_normal(
    tris_np, queries_np, grad_output_np, beta, eps_fraction
):
    v0, v1, v2 = tris_np[:, 0], tris_np[:, 1], tris_np[:, 2]
    points = ((v0 + v1 + v2) / 3.0).astype(np.float32)
    cross = np.cross(v1 - v0, v2 - v0)
    mag = np.linalg.norm(cross, axis=-1, keepdims=True)
    areas = mag / 2.0
    safe = np.where(mag == 0, 1e-8, mag)
    normals = ((cross / safe) * areas).astype(np.float32)

    device = "cuda"
    p = torch.from_numpy(points).to(device)
    n = torch.from_numpy(normals).to(device)
    q = torch.from_numpy(queries_np).to(device)
    g = torch.from_numpy(grad_output_np).to(device)
    N = points.shape[0]

    out_brute = torch.empty([N, 2, 3], device=device, dtype=torch.float32)
    out_fast = torch.empty_like(out_brute)
    eps_brute = torch.zeros(1, device=device, dtype=torch.float32)
    eps_fast = torch.zeros(1, device=device, dtype=torch.float32)

    def _brute():
        winder.brute_force_gradients_point_normal(
            g, p, n, q, out_brute, eps_brute, float(eps_fraction), _stream()
        )

    t_brute = _wall_time_cuda(_brute)

    def _fast():
        eng = winder.GradientEngine(q, g, _stream())
        eng.compute_point_normal(
            p, n, out_fast, eps_fast, float(beta), float(eps_fraction), _stream()
        )

    t_fast = _wall_time_cuda(_fast)

    return (
        out_brute.cpu().numpy(),
        out_fast.cpu().numpy(),
        float(eps_brute.item()),
        float(eps_fast.item()),
        t_brute,
        t_fast,
    )


# ============================================================================
# Query generation (matches eval_pipeline)
# ============================================================================


def _stable_seed(*parts) -> int:
    """Deterministic seed that is stable across processes."""
    return zlib.crc32("|".join(str(p) for p in parts).encode())


def _scene_epsilon_world(tris_or_vertices: np.ndarray, eps_fraction: float) -> float:
    """Match the C++ convention: eps_world = eps_fraction * max_extent."""
    if eps_fraction <= 0.0:
        return 0.0
    flat = tris_or_vertices.reshape(-1, 3)
    extent = flat.max(0) - flat.min(0)
    return float(eps_fraction * np.max(extent))


def _sample_queries(tris: np.ndarray, n_queries: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    flat = tris.reshape(-1, 3)
    mn = flat.min(0)
    mx = flat.max(0)
    diag = float(np.linalg.norm(mx - mn))
    return rng.uniform(mn - 0.3 * diag, mx + 0.3 * diag, (n_queries, 3)).astype(
        np.float32
    )


# ============================================================================
# Main
# ============================================================================


def main():
    p = argparse.ArgumentParser(
        description="Benchmark Winder backward pass vs PyTorch autograd."
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--archive", type=str, help="Thingi10K tar.gz to sample meshes from."
    )
    src.add_argument("--obj-files", type=str, help="Comma-separated list of OBJ files.")
    p.add_argument(
        "--n-meshes",
        type=int,
        default=5,
        help="Number of meshes to sample from the archive.",
    )
    p.add_argument(
        "--query-counts",
        type=str,
        default="1000,10000",
        help="Comma-separated list of query counts to test.",
    )
    p.add_argument(
        "--modes",
        type=str,
        default="triangle,mesh,point_normal",
        help="Comma-separated: triangle, mesh, point_normal.",
    )
    p.add_argument("--beta", type=float, default=2.3)
    p.add_argument(
        "--epsilon",
        type=float,
        default=DEFAULT_EPSILON,
        help=f"Regularization fraction of scene scale, applied to all modes. "
        f"Default {DEFAULT_EPSILON} (= 1/250). Use 0 for the sharp kernel.",
    )
    p.add_argument(
        "--torch-chunk-size",
        type=int,
        default=256,
        help="Maximum chunk size for the PyTorch reference. "
        "Reduced automatically if GPU memory would overflow.",
    )
    p.add_argument(
        "--try-torch-compile",
        action="store_true",
        help="Wrap the PyTorch reference loss in torch.compile.",
    )
    p.add_argument(
        "--cuda-iters", type=int, default=3, help="Timing iterations for CUDA kernels."
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str, default="bench_backward.csv")
    p.add_argument(
        "--reg-mode",
        choices=["sharp", "plummer", "compact"],
        default="compact",
        help="Must match the reg mode the CUDA library was compiled with "
        "(DefaultReg in geometry.h).",
    )
    p.add_argument(
        "--compute-eps-grad",
        action="store_true",
        help="Also benchmark dL/d(epsilon) on both PyTorch and CUDA.",
    )
    args = p.parse_args()

    np.random.seed(args.seed)

    # -----------------------------------------------------------------
    # Load meshes
    # -----------------------------------------------------------------
    meshes = []  # list of (name, vertices, faces)
    if args.obj_files:
        for path_str in args.obj_files.split(","):
            path = Path(path_str.strip())
            v, f = load_obj(path)
            meshes.append((path.stem, v, f))
            print(f"[loaded] {path.stem}  K={len(v)}  N={len(f)}")
    else:
        sample = load_thingi10k_sample(args.archive, args.n_meshes, args.seed)
        for name, v, f in sample:
            meshes.append((name, v, f))

    if not meshes:
        raise SystemExit("No meshes to benchmark.")

    query_counts = [int(x) for x in args.query_counts.split(",")]
    modes = [m.strip() for m in args.modes.split(",")]

    # -----------------------------------------------------------------
    # Benchmark loop
    # -----------------------------------------------------------------
    rows = []
    for mesh_name, vertices, faces in meshes:
        tris = vertices[faces]
        for mode in modes:
            for q_count in query_counts:
                seed = _stable_seed(mesh_name, mode, q_count, args.seed)
                queries = _sample_queries(tris, q_count, seed)
                grad_out = (
                    np.random.default_rng(seed + 1)
                    .standard_normal(q_count)
                    .astype(np.float32)
                )

                print(
                    f"\n[bench] {mesh_name}  N={len(faces)}  Q={q_count}  mode={mode}"
                )

                row = {
                    "mesh": mesh_name,
                    "n_tri": int(len(faces)),
                    "n_queries": int(q_count),
                    "mode": mode,
                }

                try:
                    if mode == "triangle":
                        eps_world = _scene_epsilon_world(tris, args.epsilon)
                        pt_grads, pt_eps, pt_ms, pt_eps_ms, c_err = (
                            _triangle_grads_torch(
                                tris,
                                queries,
                                grad_out,
                                eps_world,
                                args.reg_mode,
                                args.torch_chunk_size,
                                args.try_torch_compile,
                                args.compute_eps_grad,
                            )
                        )
                        row["pytorch_ms"] = pt_ms
                        row["pytorch_eps_ms"] = pt_eps_ms
                        row["torch_compile_error"] = c_err or ""
                        bf, ft, bf_eps, ft_eps, bf_ms, ft_ms = _cuda_backward_triangle(
                            tris, queries, grad_out, args.beta, args.epsilon
                        )
                        row["cuda_brute_ms"] = bf_ms
                        row["cuda_fast_ms"] = ft_ms
                        # CUDA eps grad is computed in the same kernel call as
                        # the geometry grad, so its cost is included in the
                        # geometry timings above.
                        row["cuda_brute_eps_ms"] = bf_ms
                        row["cuda_fast_eps_ms"] = ft_ms
                        row["pytorch_eps_grad"] = pt_eps
                        row["cuda_brute_eps_grad"] = bf_eps
                        row["cuda_fast_eps_grad"] = ft_eps
                        d = pt_grads.reshape(-1) - bf.reshape(-1)
                        row["pt_vs_cuda_brute_rel_norm"] = float(
                            np.linalg.norm(d) / (np.linalg.norm(bf.reshape(-1)) + 1e-30)
                        )
                        assert pt_grads.shape == bf.shape == ft.shape

                    elif mode == "mesh":
                        eps_world = _scene_epsilon_world(vertices, args.epsilon)
                        pt_grads, pt_eps, pt_ms, pt_eps_ms, c_err = _mesh_grads_torch(
                            vertices,
                            faces,
                            queries,
                            grad_out,
                            eps_world,
                            args.reg_mode,
                            args.torch_chunk_size,
                            args.try_torch_compile,
                            args.compute_eps_grad,
                        )
                        row["pytorch_ms"] = pt_ms
                        row["pytorch_eps_ms"] = pt_eps_ms
                        row["torch_compile_error"] = c_err or ""
                        bf, ft, bf_eps, ft_eps, bf_ms, ft_ms = _cuda_backward_mesh(
                            vertices,
                            faces,
                            queries,
                            grad_out,
                            args.beta,
                            args.epsilon,
                        )
                        row["cuda_brute_ms"] = bf_ms
                        row["cuda_fast_ms"] = ft_ms
                        row["cuda_brute_eps_ms"] = bf_ms
                        row["cuda_fast_eps_ms"] = ft_ms
                        row["pytorch_eps_grad"] = pt_eps
                        row["cuda_brute_eps_grad"] = bf_eps
                        row["cuda_fast_eps_grad"] = ft_eps
                        d = pt_grads.reshape(-1) - bf.reshape(-1)
                        row["pt_vs_cuda_brute_rel_norm"] = float(
                            np.linalg.norm(d) / (np.linalg.norm(bf.reshape(-1)) + 1e-30)
                        )
                        assert pt_grads.shape == bf.shape == ft.shape

                    elif mode == "point_normal":
                        eps_world = _scene_epsilon_world(tris, args.epsilon)
                        pt_grads, pt_eps, pt_ms, pt_eps_ms, c_err = (
                            _point_normal_grads_torch(
                                tris,
                                queries,
                                grad_out,
                                eps_world,
                                args.torch_chunk_size,
                                args.try_torch_compile,
                                args.compute_eps_grad,
                            )
                        )
                        row["pytorch_ms"] = pt_ms
                        row["pytorch_eps_ms"] = pt_eps_ms
                        row["torch_compile_error"] = c_err or ""
                        bf, ft, bf_eps, ft_eps, bf_ms, ft_ms = (
                            _cuda_backward_point_normal(
                                tris, queries, grad_out, args.beta, args.epsilon
                            )
                        )
                        row["cuda_brute_ms"] = bf_ms
                        row["cuda_fast_ms"] = ft_ms
                        row["cuda_brute_eps_ms"] = bf_ms
                        row["cuda_fast_eps_ms"] = ft_ms
                        row["pytorch_eps_grad"] = pt_eps
                        row["cuda_brute_eps_grad"] = bf_eps
                        row["cuda_fast_eps_grad"] = ft_eps
                        d = pt_grads.reshape(-1) - bf.reshape(-1)
                        row["pt_vs_cuda_brute_rel_norm"] = float(
                            np.linalg.norm(d) / (np.linalg.norm(bf.reshape(-1)) + 1e-30)
                        )
                        assert pt_grads.shape == bf.shape == ft.shape

                    else:
                        print(f"  skipping unknown mode {mode}")
                        continue

                except Exception as exc:
                    print(f"  ERROR: {type(exc).__name__}: {exc}")
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    rows.append(row)
                    continue

                row["speedup_pt_vs_fast"] = (
                    row["pytorch_ms"] / row["cuda_fast_ms"]
                    if row["cuda_fast_ms"] > 0
                    else float("nan")
                )
                row["speedup_pt_vs_brute"] = (
                    row["pytorch_ms"] / row["cuda_brute_ms"]
                    if row["cuda_brute_ms"] > 0
                    else float("nan")
                )
                row["speedup_brute_vs_fast"] = (
                    row["cuda_brute_ms"] / row["cuda_fast_ms"]
                    if row["cuda_fast_ms"] > 0
                    else float("nan")
                )
                rows.append(row)

                print(f"  pytorch : {row['pytorch_ms']:10.2f} ms")
                print(
                    f"  cuda BF : {row['cuda_brute_ms']:10.2f} ms  "
                    f"({row['speedup_pt_vs_brute']:7.2f}x vs pytorch)"
                )
                print(
                    f"  cuda FT : {row['cuda_fast_ms']:10.2f} ms  "
                    f"({row['speedup_pt_vs_fast']:7.2f}x vs pytorch, "
                    f"{row['speedup_brute_vs_fast']:5.2f}x vs BF)"
                )

    # -----------------------------------------------------------------
    # CSV
    # -----------------------------------------------------------------
    if rows:
        keys = sorted({k for r in rows for k in r.keys()})
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\n[csv] wrote {args.output}")

    # -----------------------------------------------------------------
    # Formatted table
    # -----------------------------------------------------------------
    valid = [r for r in rows if "error" not in r]
    if not valid:
        print("No successful benchmark rows.")
        return

    print("\n" + "=" * 152)
    print("  Backward pass: CUDA vs PyTorch autograd")
    print("=" * 152)
    hdr = (
        f"  {'mesh':<20} | {'N':>7} | {'Q':>7} | {'mode':<13} | "
        f"{'PyTorch':>9} | {'PT-eps':>8} | {'CUDA-BF':>8} | {'BF-eps':>8} | "
        f"{'CUDA-FT':>8} | {'FT-eps':>8} | {'PT/FT':>7} | {'BF/FT':>6}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in valid:
        name = r["mesh"][:20]
        print(
            f"  {name:<20} | {r['n_tri']:>7d} | {r['n_queries']:>7d} | "
            f"{r['mode']:<13} | {r['pytorch_ms']:>9.2f} | "
            f"{r.get('pytorch_eps_ms', float('nan')):>8.2f} | "
            f"{r['cuda_brute_ms']:>8.2f} | "
            f"{r.get('cuda_brute_eps_ms', float('nan')):>8.2f} | "
            f"{r['cuda_fast_ms']:>8.2f} | "
            f"{r.get('cuda_fast_eps_ms', float('nan')):>8.2f} | "
            f"{r['speedup_pt_vs_fast']:>6.2f}x | "
            f"{r['speedup_brute_vs_fast']:>5.2f}x"
        )
    print("  " + "-" * (len(hdr) - 2))

    # Aggregate by (mode, query_count) with geometric mean over meshes.
    print("\n  Summary (geometric mean across meshes):")
    groups: dict[tuple, list] = {}
    for r in valid:
        key = (r["mode"], r["n_queries"])
        groups.setdefault(key, []).append(r)
    for (mode, q), rs in sorted(groups.items()):
        pt = np.exp(np.mean(np.log([r["pytorch_ms"] for r in rs])))
        bf = np.exp(np.mean(np.log([r["cuda_brute_ms"] for r in rs])))
        ft = np.exp(np.mean(np.log([r["cuda_fast_ms"] for r in rs])))
        print(
            f"    {mode:<13} Q={q:>7d}  "
            f"pytorch={pt:>9.2f}ms  BF={bf:>8.2f}ms  FT={ft:>8.2f}ms  "
            f"PT/FT={pt / ft:>7.2f}x  BF/FT={bf / ft:>5.2f}x"
        )
    # Epsilon-gradient agreement summary.
    eps_rows = [
        r
        for r in valid
        if "pytorch_eps_grad" in r
        and math.isfinite(r.get("pytorch_eps_grad", float("nan")))
        and math.isfinite(r.get("cuda_fast_eps_grad", float("nan")))
    ]
    if eps_rows:
        print(
            "\n  Epsilon-gradient agreement (|fast − pytorch| / max(|pytorch|, 1e-2)):"
        )
        for r in eps_rows:
            ref = r["pytorch_eps_grad"]
            fast = r["cuda_fast_eps_grad"]
            denom = max(abs(ref), 1e-2)
            err = abs(fast - ref) / denom
            print(
                f"    {r['mesh'][:20]:<20} N={r['n_tri']:>7d} Q={r['n_queries']:>7d} "
                f"{r['mode']:<13}  pytorch={ref:+12.4e}  cuda={fast:+12.4e}  "
                f"rel={err:.3e}"
            )
    print("=" * 122)


if __name__ == "__main__":
    main()
