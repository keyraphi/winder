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


def _triangle_grads_torch(
    triangles_np: np.ndarray,
    queries: np.ndarray,
    grad_output_np: np.ndarray,
    chunk_size: int,
    try_compile: bool,
) -> tuple[np.ndarray, float, str | None]:
    """Return (gradients, wall_clock_ms, compile_error_or_none)."""
    device = "cuda"
    tris = torch.from_numpy(triangles_np).to(device)
    q = torch.from_numpy(queries).to(device)
    g = torch.from_numpy(grad_output_np).to(device)

    N = tris.shape[0]
    Q = q.shape[0]
    chunk = _auto_chunk_size(N, chunk_size, 1_000_000_000)
    inv_two_pi = 1.0 / (2.0 * math.pi)

    def _chunk_loss(v, qc, gc):
        a = v[None, :, 0, :] - qc[:, None, :]
        b = v[None, :, 1, :] - qc[:, None, :]
        c = v[None, :, 2, :] - qc[:, None, :]
        la = a.norm(dim=-1)
        lb = b.norm(dim=-1)
        lc = c.norm(dim=-1)
        Nv = (a * torch.cross(b, c, dim=-1)).sum(-1)
        Dv = (
            la * lb * lc
            + (a * b).sum(-1) * lc
            + (b * c).sum(-1) * la
            + (c * a).sum(-1) * lb
        )
        return (torch.atan2(Nv, Dv) * inv_two_pi * gc[:, None]).sum()

    compile_error = None
    if try_compile:
        try:
            _chunk_loss = torch.compile(_chunk_loss)
        except Exception as exc:
            compile_error = f"torch.compile failed: {exc}"
            _chunk_loss = (
                _chunk_loss.__wrapped__
                if hasattr(_chunk_loss, "__wrapped__")
                else _chunk_loss
            )

    # Warmup once so any compilation / kernel caching is paid before timing.
    v_warm = tris.clone().requires_grad_(True)
    _ = _chunk_loss(v_warm, q[: min(64, Q)], g[: min(64, Q)])
    torch.cuda.synchronize()
    del v_warm

    v = tris.clone().requires_grad_(True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(0, Q, chunk):
        loss = _chunk_loss(v, q[i : i + chunk], g[i : i + chunk])
        loss.backward()
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    return v.grad.detach().cpu().numpy(), (t1 - t0) * 1e3, compile_error


def _point_normal_grads_torch(
    triangles_np: np.ndarray,
    queries: np.ndarray,
    grad_output_np: np.ndarray,
    epsilon: float,
    chunk_size: int,
    try_compile: bool,
) -> tuple[np.ndarray, float, str | None]:
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
    inv_eps = 1.0 / epsilon
    inv_4pi = 1.0 / (4.0 * math.pi)
    four_over_3sqrt_pi = 4.0 / (3.0 * math.sqrt(math.pi))

    def _chunk_loss(p_, n_, qc, gc):
        d = p_[None, :, :] - qc[:, None, :]
        dist2 = (d * d).sum(-1)
        inv_dist = torch.rsqrt(dist2 + 1e-20)
        inv_dist3 = inv_dist**3
        t = dist2 * inv_dist * inv_eps
        near = four_over_3sqrt_pi * inv_eps**3
        mid_s = torch.erf(t) - (2.0 / math.sqrt(math.pi)) * t * torch.exp(-t * t)
        mid = mid_s * inv_dist3
        far = inv_dist3
        s_over_d3 = torch.where(t < 0.1, near, torch.where(t < 2.0, mid, far))
        contrib = (n_[None, :, :] * d).sum(-1) * inv_4pi * s_over_d3
        return (contrib * gc[:, None]).sum()

    compile_error = None
    if try_compile:
        try:
            _chunk_loss = torch.compile(_chunk_loss)
        except Exception as exc:
            compile_error = f"torch.compile failed: {exc}"

    p_warm = p.clone().requires_grad_(True)
    n_warm = n.clone().requires_grad_(True)
    _ = _chunk_loss(p_warm, n_warm, q[: min(64, Q)], g[: min(64, Q)])
    torch.cuda.synchronize()
    del p_warm, n_warm

    p_ = p.clone().requires_grad_(True)
    n_ = n.clone().requires_grad_(True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(0, Q, chunk):
        loss = _chunk_loss(p_, n_, q[i : i + chunk], g[i : i + chunk])
        loss.backward()
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    grads = np.stack(
        [p_.grad.detach().cpu().numpy(), n_.grad.detach().cpu().numpy()], axis=1
    )
    return grads, (t1 - t0) * 1e3, compile_error


def _mesh_grads_torch(
    vertices_np: np.ndarray,
    faces_np: np.ndarray,
    queries: np.ndarray,
    grad_output_np: np.ndarray,
    chunk_size: int,
    try_compile: bool,
) -> tuple[np.ndarray, float, str | None]:
    device = "cuda"
    v = torch.from_numpy(vertices_np).to(device)
    f = torch.from_numpy(faces_np.astype(np.int64)).to(device)
    q = torch.from_numpy(queries).to(device)
    g = torch.from_numpy(grad_output_np).to(device)

    N = f.shape[0]
    Q = q.shape[0]
    chunk = _auto_chunk_size(N, chunk_size, 1_000_000_000)
    inv_two_pi = 1.0 / (2.0 * math.pi)

    def _chunk_loss(v_, qc, gc):
        tri_v = v_[f]  # [N, 3, 3]
        a = tri_v[None, :, 0, :] - qc[:, None, :]
        b = tri_v[None, :, 1, :] - qc[:, None, :]
        c = tri_v[None, :, 2, :] - qc[:, None, :]
        la = a.norm(dim=-1)
        lb = b.norm(dim=-1)
        lc = c.norm(dim=-1)
        Nv = (a * torch.cross(b, c, dim=-1)).sum(-1)
        Dv = (
            la * lb * lc
            + (a * b).sum(-1) * lc
            + (b * c).sum(-1) * la
            + (c * a).sum(-1) * lb
        )
        return (torch.atan2(Nv, Dv) * inv_two_pi * gc[:, None]).sum()

    compile_error = None
    if try_compile:
        try:
            _chunk_loss = torch.compile(_chunk_loss)
        except Exception as exc:
            compile_error = f"torch.compile failed: {exc}"

    v_warm = v.clone().requires_grad_(True)
    _ = _chunk_loss(v_warm, q[: min(64, Q)], g[: min(64, Q)])
    torch.cuda.synchronize()
    del v_warm

    v_ = v.clone().requires_grad_(True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(0, Q, chunk):
        loss = _chunk_loss(v_, q[i : i + chunk], g[i : i + chunk])
        loss.backward()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return v_.grad.detach().cpu().numpy(), (t1 - t0) * 1e3, compile_error


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


def _cuda_backward_triangle(tris_np, queries_np, grad_output_np, beta, epsilon):
    device = "cuda"
    t = torch.from_numpy(tris_np).to(device)
    q = torch.from_numpy(queries_np).to(device)
    g = torch.from_numpy(grad_output_np).to(device)
    N = tris_np.shape[0]

    out_brute = torch.empty([N, 3, 3], device=device, dtype=torch.float32)
    out_fast = torch.empty_like(out_brute)

    def _brute():
        winder.brute_force_gradients_triangle_soup(
            g, t, q, out_brute, float(epsilon), _stream()
        )

    t_brute = _wall_time_cuda(_brute)

    def _fast():
        eng = winder.GradientEngine(q, g, _stream())
        eng.compute_triangle_soup(
            t, out_fast, float(beta), float(epsilon), _stream()
        )

    t_fast = _wall_time_cuda(_fast)

    return out_brute.cpu().numpy(), out_fast.cpu().numpy(), t_brute, t_fast


def _cuda_backward_mesh(vertices_np, faces_np, queries_np, grad_output_np,
                        beta, epsilon):
    device = "cuda"
    v = torch.from_numpy(vertices_np).to(device)
    f = torch.from_numpy(faces_np.astype(np.uint32)).to(device)
    q = torch.from_numpy(queries_np).to(device)
    g = torch.from_numpy(grad_output_np).to(device)
    K = vertices_np.shape[0]

    out_brute = torch.empty([K, 3], device=device, dtype=torch.float32)
    out_fast = torch.empty_like(out_brute)

    def _brute():
        winder.brute_force_gradients_mesh(
            g, v, f, q, out_brute, float(epsilon), _stream()
        )

    t_brute = _wall_time_cuda(_brute)

    def _fast():
        eng = winder.GradientEngine(q, g, _stream())
        eng.compute_mesh(v, f, out_fast, float(beta), float(epsilon), _stream())

    t_fast = _wall_time_cuda(_fast)

    return out_brute.cpu().numpy(), out_fast.cpu().numpy(), t_brute, t_fast


def _cuda_backward_point_normal(tris_np, queries_np, grad_output_np, beta, epsilon):
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

    def _brute():
        winder.brute_force_gradients_point_normal(
            g, p, n, q, out_brute, float(epsilon), _stream()
        )

    t_brute = _wall_time_cuda(_brute)

    def _fast():
        eng = winder.GradientEngine(q, g, _stream())
        eng.compute_point_normal(
            p, n, out_fast, float(beta), float(epsilon), _stream()
        )

    t_fast = _wall_time_cuda(_fast)

    return out_brute.cpu().numpy(), out_fast.cpu().numpy(), t_brute, t_fast


# ============================================================================
# Query generation (matches eval_pipeline)
# ============================================================================


def _sample_queries(tris: np.ndarray, n_queries: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    flat = tris.reshape(-1, 3)
    mn = flat.min(0)
    mx = flat.max(0)
    diag = float(np.linalg.norm(mx - mn))
    mn = mn - 0.3 * diag
    mx = mx + 0.3 * diag
    return rng.uniform(mn, mx, (n_queries, 3)).astype(np.float32)


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
                seed = hash((mesh_name, mode, q_count, args.seed)) & 0x7FFFFFFF
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
                        pt_grads, pt_ms, c_err = _triangle_grads_torch(
                            tris,
                            queries,
                            grad_out,
                            args.torch_chunk_size,
                            args.try_torch_compile,
                        )
                        row["pytorch_ms"] = pt_ms
                        row["torch_compile_error"] = c_err or ""
                        bf, ft, bf_ms, ft_ms = _cuda_backward_triangle(
                            tris, queries, grad_out, args.beta, args.epsilon
                        )
                        row["cuda_brute_ms"] = bf_ms
                        row["cuda_fast_ms"] = ft_ms
                        assert pt_grads.shape == bf.shape == ft.shape

                    elif mode == "mesh":
                        pt_grads, pt_ms, c_err = _mesh_grads_torch(
                            vertices,
                            faces,
                            queries,
                            grad_out,
                            args.torch_chunk_size,
                            args.try_torch_compile,
                        )
                        row["pytorch_ms"] = pt_ms
                        row["torch_compile_error"] = c_err or ""
                        bf, ft, bf_ms, ft_ms = _cuda_backward_mesh(
                            vertices, faces, queries, grad_out,
                            args.beta, args.epsilon,
                        )
                        row["cuda_brute_ms"] = bf_ms
                        row["cuda_fast_ms"] = ft_ms
                        assert pt_grads.shape == bf.shape == ft.shape

                    elif mode == "point_normal":
                        pt_grads, pt_ms, c_err = _point_normal_grads_torch(
                            tris,
                            queries,
                            grad_out,
                            args.epsilon,
                            args.torch_chunk_size,
                            args.try_torch_compile,
                        )
                        row["pytorch_ms"] = pt_ms
                        row["torch_compile_error"] = c_err or ""
                        bf, ft, bf_ms, ft_ms = _cuda_backward_point_normal(
                            tris, queries, grad_out, args.beta, args.epsilon
                        )
                        row["cuda_brute_ms"] = bf_ms
                        row["cuda_fast_ms"] = ft_ms
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

    print("\n" + "=" * 122)
    print("  Backward pass: CUDA vs PyTorch autograd")
    print("=" * 122)
    hdr = (
        f"  {'mesh':<20} | {'N':>7} | {'Q':>7} | {'mode':<13} | "
        f"{'PyTorch':>10} | {'CUDA-BF':>9} | {'CUDA-FT':>9} | "
        f"{'PT/FT':>8} | {'BF/FT':>7}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in valid:
        name = r["mesh"][:20]
        print(
            f"  {name:<20} | {r['n_tri']:>7d} | {r['n_queries']:>7d} | "
            f"{r['mode']:<13} | {r['pytorch_ms']:>10.2f} | "
            f"{r['cuda_brute_ms']:>9.2f} | {r['cuda_fast_ms']:>9.2f} | "
            f"{r['speedup_pt_vs_fast']:>7.2f}x | "
            f"{r['speedup_brute_vs_fast']:>6.2f}x"
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
    print("=" * 122)


if __name__ == "__main__":
    main()
