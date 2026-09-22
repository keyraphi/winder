"""Isolate a single mesh for debugging.

Caches the parsed mesh + queries to a .npz so subsequent runs skip the
slow tar stream.

Usage:
    # First time: streams from the archive and caches.
    CUDA_LAUNCH_BLOCKING=1 python examples/debug_one_mesh.py \
        --archive /graphics/scratch3/datasets/thingi10K.tar.gz \
        --mesh raw_meshes/1619332.stl \
        --query_count 1000000 \
        --cache_dir /tmp/thingi10k_cached

    # Subsequent runs: instant load.
    CUDA_LAUNCH_BLOCKING=1 python examples/debug_one_mesh.py \
        --archive /graphics/scratch3/datasets/thingi10K.tar.gz \
        --mesh raw_meshes/1619332.stl \
        --cache_dir /tmp/thingi10k_cached
"""

import argparse
import os
import tarfile
import tempfile
from pathlib import Path

import igl
import numpy as np
import torch
from tqdm.auto import tqdm

import winder


def cache_path_for(
    cache_dir: Path, mesh_name: str, n_queries: int, beta: float
) -> Path:
    """One cache file per (mesh, query count, beta)."""
    safe = mesh_name.replace("/", "_")
    return cache_dir / f"{safe}__q{n_queries}__b{beta}.npz"


def find_mesh(archive: str, mesh_name: str) -> bytes:
    with tarfile.open(archive, "r|gz") as tar:
        for member in tqdm(tar, desc=f"Searching for {mesh_name}"):
            if not member.isfile():
                continue
            if mesh_name not in member.name:
                continue
            f = tar.extractfile(member)
            if f is not None:
                print(f"[FOUND] {member.name}")
                return f.read()
    raise SystemExit(f"mesh {mesh_name} not found in {archive}")


def parse_stl_via_igl(data: bytes) -> np.ndarray:
    fd, path = tempfile.mkstemp(suffix=".stl")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        v, f_idx = igl.read_triangle_mesh(path)
        if v.size == 0 or f_idx.size == 0:
            raise ValueError("igl returned empty mesh")
        return v[f_idx].astype(np.float32)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


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


def load_or_build(archive, mesh_name, n_queries, beta, cache_path) -> tuple:
    """Return (tris, queries), loading from cache if it exists."""
    if cache_path.exists():
        print(f"[CACHE] loading {cache_path}")
        data = np.load(cache_path)
        tris = data["tris"]
        queries = data["queries"]
        print(f"        tris={tris.shape}  queries={queries.shape}")
        return tris, queries

    print(f"[CACHE] miss, streaming from archive")
    raw = find_mesh(archive, mesh_name)
    tris = parse_stl_via_igl(raw)
    queries = sample_queries_grid(tris, n_queries)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, tris=tris, queries=queries, beta=np.float32(beta))
    print(f"[CACHE] wrote {cache_path}")
    return tris, queries


def print_mesh_diagnostics(tris: np.ndarray) -> None:
    n_tri = len(tris)
    leaf_count = (n_tri + 31) // 32  # LEAF_SIZE = 32
    max_bvh8_nodes = max(0, leaf_count - 1)

    flat = tris.reshape(-1, 3)
    mn = flat.min(axis=0)
    mx = flat.max(axis=0)
    extent = mx - mn
    diag = float(np.linalg.norm(extent))

    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    areas = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    n_degenerate = int((areas < 1e-12).sum())
    n_tiny = int((areas < 1e-6).sum())

    print(f"[INFO] n_tri          = {n_tri}")
    print(f"[INFO] leaf_count     = {leaf_count}")
    print(f"[INFO] max_bvh8_nodes = {max_bvh8_nodes}")
    print(f"[INFO] bbox min       = {mn}")
    print(f"[INFO] bbox max       = {mx}")
    print(f"[INFO] extent         = {extent}")
    print(f"[INFO] diagonal       = {diag:.6e}")
    print(f"[INFO] finite         = {np.isfinite(tris).all()}")
    print(f"[INFO] degenerate     = {n_degenerate}  (area < 1e-12)")
    print(f"[INFO] tiny           = {n_tiny}      (area < 1e-6)")
    print(
        f"[INFO] area min/med/max = {areas.min():.3e} / "
        f"{np.median(areas):.3e} / {areas.max():.3e}"
    )

    # Vertex-duplication check: if many triangles share vertices, the tree
    # structure can behave differently than the raw count suggests.
    verts = flat.reshape(-1, 3)
    unique_verts = np.unique(np.round(verts, decimals=4), axis=0)
    print(f"[INFO] verts (raw)    = {len(verts)}")
    print(f"[INFO] verts (unique) = {len(unique_verts)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", required=True)
    p.add_argument("--mesh", required=True, help="Substring of the tar member name")
    p.add_argument("--query_count", type=int, default=1000000)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--cache_dir", type=str, default="/tmp/thingi10k_cached")
    p.add_argument(
        "--no_cache",
        action="store_true",
        help="Ignore the cache; always re-stream and re-write.",
    )
    p.add_argument(
        "--stage",
        choices=["all", "tensors", "brute", "engine", "compute"],
        default="all",
        help="Stop after the given stage (for isolating failures).",
    )
    args = p.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_path = cache_path_for(cache_dir, args.mesh, args.query_count, args.beta)
    if args.no_cache and cache_path.exists():
        cache_path.unlink()
        print(f"[CACHE] removed {cache_path}")

    tris, queries = load_or_build(
        args.archive, args.mesh, args.query_count, args.beta, cache_path
    )
    print_mesh_diagnostics(tris)
    print(f"[INFO] queries shape  = {queries.shape}")

    if args.stage == "all" or args.stage in ("tensors", "brute", "engine", "compute"):
        print("[STEP] moving tensors to GPU")
        t_tri = torch.from_numpy(np.ascontiguousarray(tris)).cuda()
        t_q = torch.from_numpy(np.ascontiguousarray(queries)).cuda()
        torch.cuda.synchronize()
        print("        OK")
        if args.stage == "tensors":
            return

    wn_brute = torch.empty(len(queries), device="cuda", dtype=torch.float32)
    wn_fast = torch.empty(len(queries), device="cuda", dtype=torch.float32)

    if args.stage == "all" or args.stage in ("brute", "engine", "compute"):
        print("[STEP] brute force")
        winder.brute_force_winding_numbers(t_tri, t_q, wn_brute, stream=0)
        torch.cuda.synchronize()
        print("        OK")
        if args.stage == "brute":
            return

    if args.stage == "all" or args.stage in ("engine", "compute"):
        print("[STEP] constructing WindingNumberEngine")
        engine = winder.WindingNumberEngine(t_tri, stream=0)
        torch.cuda.synchronize()
        print("        OK")
        if args.stage == "engine":
            return

    print("[STEP] computing winding numbers")
    engine.compute(t_q, wn_fast, beta=args.beta, stream=0)
    torch.cuda.synchronize()
    print("        OK")

    ref = wn_brute.cpu().numpy()
    fast = wn_fast.cpu().numpy()
    print(
        f"[RESULT] ref finite={np.isfinite(ref).all()}  "
        f"fast finite={np.isfinite(fast).all()}"
    )
    if np.isfinite(fast).all():
        diff = fast - ref
        print(
            f"[RESULT] max abs err = {np.abs(diff).max():.3e}  "
            f"rms = {np.sqrt(np.mean(diff * diff)):.3e}"
        )


if __name__ == "__main__":
    main()
