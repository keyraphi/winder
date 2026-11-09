"""Profiling driver for Winder's PayloadField kernels, designed for ncu.

This script exists to be driven by NVIDIA Nsight Compute. It:
  1. Loads (or synthesizes) a mesh
  2. For each (geometry, D, kahan) config: builds the engine, allocates
     the tensors, and launches `compute` + `compute_query_gradients`
     once per warmup iteration and once per profiled iteration.

The PyTorch-side work is deliberately minimal so ncu's kernel-replay
doesn't waste time re-running Python. Use `--no_profile` first to verify
timing and catch setup errors before invoking ncu.

Kernels of interest
-------------------
Forward:  traverse_payload_winding_numbers_kernel
          traverse_payload_single_leaf_kernel
Backward: traverse_payload_query_gradient_kernel
          traverse_payload_query_grad_single_leaf_kernel
Build:    convert_binary_tree_to_bvh8_kernel
          compute_internal_tailor_coefficients_m2m_payload_kernel

Typical ncu workflows
---------------------
    # 1. Fast sanity check (no profiling, ~30 s):
    python examples/profile_payload.py --no_profile

    # 2. Profile everything once per config (slow! ~20-40 min):
    ncu -o report --set default \\
        python examples/profile_payload.py --d_values 1,8,32,128

    # 3. Focus on ONE forward kernel launch of ONE config:
    ncu --kernel-name regex:traverse_payload_winding_numbers_kernel \\
        --launch-skip 2 --launch-count 1 \\
        python examples/profile_payload.py \\
          --config tri:32:kahan_on --iterations 2 --warmup 1

    # 4. Detailed roofline for one kernel, one config:
    ncu --kernel-name regex:traverse_payload_winding_numbers_kernel \\
        --launch-skip 2 --launch-count 1 --set detailed \\
        python examples/profile_payload.py \\
          --config tri:128:kahan_on --iterations 2
"""

import argparse
import sys
from dataclasses import dataclass
from time import time

import numpy as np
import torch

import winder


# =============================================================================
# Config
# =============================================================================
GEOMETRIES = ["tri", "mesh", "pn"]
D_BUCKETS = [1, 2, 3, 4, 8, 16, 32, 64, 128]
KAHAN_VALUES = [False, True]


@dataclass(frozen=True)
class ProfileConfig:
    geometry: str  # "tri" | "mesh" | "pn"
    D: int
    kahan: bool


def _parse_config_spec(spec: str) -> list[ProfileConfig]:
    configs = []
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid config {entry!r}; expected 'geom:D:kahan' "
                f"(e.g. 'tri:8:kahan_on')"
            )
        geom, d_str, kahan_str = parts
        if geom not in GEOMETRIES:
            raise ValueError(f"Unknown geometry {geom!r}; use {GEOMETRIES}")
        kahan = kahan_str.lower() in ("1", "true", "yes", "on", "kahan_on")
        configs.append(ProfileConfig(geom, int(d_str), kahan))
    return configs


def _default_configs(d_values: list[int], geoms: list[str]) -> list[ProfileConfig]:
    return [
        ProfileConfig(geom, d, kahan)
        for geom in geoms
        for d in d_values
        for kahan in KAHAN_VALUES
    ]


def _config_seed(cfg: ProfileConfig, base: int) -> int:
    g = {"tri": 1, "mesh": 2, "pn": 3}[cfg.geometry]
    return base + cfg.D * 100 + g * 10 + int(cfg.kahan)


# =============================================================================
# Geometry helpers
# =============================================================================
def make_synthetic_sphere(n_theta: int, n_phi: int, radius: float = 1.0):
    theta_i = np.linspace(0, np.pi, n_theta + 1, dtype=np.float64)[1:-1]
    phi_j = np.linspace(0, 2 * np.pi, n_phi, endpoint=False, dtype=np.float64)
    th = theta_i[:, None]
    ph = phi_j[None, :]
    sin_th = np.sin(th)
    cos_th = np.cos(th)
    cos_ph = np.cos(ph)
    sin_ph = np.sin(ph)
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
    last_row = (n_theta - 2) * n_phi
    Jc = np.arange(n_phi)
    Jn = (Jc + 1) % n_phi
    tri_south = np.stack([np.full(n_phi, south), last_row + Jc, last_row + Jn], axis=1)
    tri_north = np.stack([np.full(n_phi, north), Jc, Jn], axis=1)
    tris = np.concatenate([tri1, tri2, tri_south, tri_north], axis=0)
    return verts, tris.astype(np.uint32)


def mesh_to_point_surfels(vertices, indices):
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


def generate_queries(vertices, n, seed):
    rng = np.random.default_rng(seed)
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    diag = float(np.linalg.norm(hi - lo))
    lo = lo - 0.5 * diag
    hi = hi + 0.5 * diag
    return rng.uniform(lo, hi, size=(n, 3)).astype(np.float32)


def _to_cuda(x, dtype=torch.float32):
    return torch.from_numpy(np.ascontiguousarray(x)).to(dtype).cuda().contiguous()


def _stream() -> int:
    return torch.cuda.current_stream().cuda_stream


# =============================================================================
# Per-config setup
# =============================================================================
@dataclass
class Handles:
    engine: object
    queries: torch.Tensor
    grad_output: torch.Tensor
    out_payload: torch.Tensor
    out_query_grads: torch.Tensor


def _setup_config(
    cfg: ProfileConfig,
    vertices: np.ndarray,
    indices: np.ndarray,
    points: np.ndarray,
    normals: np.ndarray,
    queries_np: np.ndarray,
    seed: int,
) -> Handles:
    M = len(queries_np)
    D = cfg.D
    rng = np.random.default_rng(_config_seed(cfg, seed))

    q = _to_cuda(queries_np)
    grad_np = rng.standard_normal((M, D)).astype(np.float32)
    grad = _to_cuda(grad_np)

    if cfg.geometry == "tri":
        tri = vertices[indices]
        N = len(tri)
        payload = _to_cuda(rng.standard_normal((N, D)).astype(np.float32))
        engine = winder.PayloadField(_to_cuda(tri), payload, stream=_stream())
    elif cfg.geometry == "mesh":
        N = len(indices)
        payload = _to_cuda(rng.standard_normal((N, D)).astype(np.float32))
        engine = winder.PayloadField(
            _to_cuda(vertices),
            _to_cuda(indices, torch.uint32),
            payload,
            stream=_stream(),
        )
    elif cfg.geometry == "pn":
        N = len(points)
        payload = _to_cuda(rng.standard_normal((N, D)).astype(np.float32))
        engine = winder.PayloadField(
            _to_cuda(points),
            _to_cuda(normals),
            payload,
            stream=_stream(),
        )
    else:
        raise ValueError(cfg.geometry)

    out_payload = torch.empty((M, D), device="cuda:0", dtype=torch.float32)
    out_query_grads = torch.empty((M, 3), device="cuda:0", dtype=torch.float32)
    return Handles(engine, q, grad, out_payload, out_query_grads)


def _run_forward(h: Handles, cfg: ProfileConfig, beta: float, epsilon: float):
    h.engine.compute(h.queries, h.out_payload, beta, epsilon, _stream(), cfg.kahan)


def _run_backward(h: Handles, cfg: ProfileConfig, beta: float, epsilon: float):
    h.engine.compute_query_gradients(
        h.grad_output,
        h.queries,
        h.out_query_grads,
        beta,
        epsilon,
        _stream(),
        cfg.kahan,
    )


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Profiling driver for PayloadField kernels (ncu-friendly).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--obj_file",
        type=str,
        default=None,
        help="Optional OBJ mesh. Default: synthetic sphere.",
    )
    parser.add_argument(
        "--sphere_size",
        type=str,
        default="medium",
        choices=["tiny", "small", "medium", "large"],
    )
    parser.add_argument(
        "--query_count",
        type=int,
        default=20000,
        help="Number of queries (ncu overhead grows with M).",
    )
    parser.add_argument("--epsilon", type=float, default=0.004)
    parser.add_argument("--beta", type=float, default=-1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--iterations",
        type=int,
        default=2,
        help="Forward+backward iterations per config. "
        "ncu profiles every launch, so keep small.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup iterations per config (use --launch-skip to exclude from ncu).",
    )
    parser.add_argument(
        "--d_values",
        type=str,
        default="1,8,32,128",
        help="Comma-separated buckets ('1,8,32,128' or 'all').",
    )
    parser.add_argument(
        "--geometry",
        type=str,
        default="tri,mesh,pn",
        help="Comma-separated geometries ('tri,mesh,pn').",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Explicit config list; overrides --d_values and "
        "--geometry. E.g. 'tri:8:kahan_on,mesh:128:off'.",
    )
    parser.add_argument(
        "--no_profile",
        action="store_true",
        help="Just run + time; skip ncu-oriented printing.",
    )
    args = parser.parse_args()

    # ---- Geometry loading ----
    if args.obj_file:
        import igl

        v, _, _, i, _, _ = igl.readOBJ(args.obj_file)
        vertices = np.ascontiguousarray(v, dtype=np.float32)
        indices = np.ascontiguousarray(i, dtype=np.uint32)
        print(f"Loaded {args.obj_file}: {len(indices)} tri, {len(vertices)} verts")
    else:
        sizes = {
            "tiny": (6, 4),
            "small": (20, 12),
            "medium": (40, 24),
            "large": (80, 48),
        }
        n_theta, n_phi = sizes[args.sphere_size]
        vertices, indices = make_synthetic_sphere(n_theta, n_phi)
        print(f"Synthetic {args.sphere_size}: {len(indices)} tri")

    points, normals = mesh_to_point_surfels(vertices, indices)
    queries_np = generate_queries(vertices, args.query_count, args.seed)
    print(f"Queries: {len(queries_np)}")

    # ---- Config resolution ----
    if args.config:
        configs = _parse_config_spec(args.config)
    else:
        geoms = [g.strip() for g in args.geometry.split(",") if g.strip()]
        for g in geoms:
            if g not in GEOMETRIES:
                parser.error(f"Unknown geometry {g!r}; choose from {GEOMETRIES}")

        if args.d_values.strip().lower() == "all":
            d_values = list(D_BUCKETS)
        else:
            d_values = [int(x) for x in args.d_values.split(",") if x.strip()]

        configs = _default_configs(d_values, geoms)

    # Warn about non-bucket D values (they dispatch to the next bucket up)
    for c in configs:
        if c.D not in D_BUCKETS:
            bucket = next(b for b in D_BUCKETS if b >= c.D)
            print(f"  Note: D={c.D} dispatches to bucket D={bucket} kernel")

    n_configs = len(configs)
    launches_per_iter = 2  # fwd + bwd
    total_launches = n_configs * (args.warmup + args.iterations) * launches_per_iter

    print(f"\n{'-' * 80}")
    print(
        f"Profiling matrix: {n_configs} configs, "
        f"warmup={args.warmup}, iters={args.iterations}"
    )
    print(f"Total kernel launches: {total_launches}")
    print(f"{'-' * 80}")
    for cfg in configs:
        print(f"  {cfg.geometry:>4} D={cfg.D:>3} kahan={'on ' if cfg.kahan else 'off'}")
    print()

    # ---- Setup all engines ----
    print("Building engines...")
    t0 = time()
    handles_list = []
    for cfg in configs:
        handles_list.append(
            _setup_config(
                cfg,
                vertices,
                indices,
                points,
                normals,
                queries_np,
                args.seed,
            )
        )
    torch.cuda.synchronize()
    print(f"  {time() - t0:.2f}s for {n_configs} engines\n")

    # ---- Run ----
    if args.no_profile:
        print(f"  {'config':<24} {'fwd(ms)':>10} {'bwd(ms)':>10}")
        print(f"  {'-' * 50}")
    else:
        print(f"  Launching under ncu (no timing)\n")

    for idx, (cfg, h) in enumerate(zip(configs, handles_list)):
        label = f"{cfg.geometry:>4} D={cfg.D:>3} {'kahan' if cfg.kahan else 'no-kah'}"

        # Warmup
        for _ in range(args.warmup):
            _run_forward(h, cfg, args.beta, args.epsilon)
            _run_backward(h, cfg, args.beta, args.epsilon)
        torch.cuda.synchronize()

        if args.no_profile:
            f_times, b_times = [], []
            for _ in range(args.iterations):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                _run_forward(h, cfg, args.beta, args.epsilon)
                e.record()
                torch.cuda.synchronize()
                f_times.append(s.elapsed_time(e))

                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                _run_backward(h, cfg, args.beta, args.epsilon)
                e.record()
                torch.cuda.synchronize()
                b_times.append(s.elapsed_time(e))

            f_ms = float(np.median(f_times))
            b_ms = float(np.median(b_times))
            print(f"  {label:<24} {f_ms:>10.3f} {b_ms:>10.3f}")
        else:
            for _ in range(args.iterations):
                _run_forward(h, cfg, args.beta, args.epsilon)
                _run_backward(h, cfg, args.beta, args.epsilon)
            torch.cuda.synchronize()
            print(
                f"  [{idx:>2}/{n_configs}] {label} — "
                f"launched {args.iterations * launches_per_iter} kernels"
            )

    # ---- Final hint for ncu users ----
    if not args.no_profile:
        skip_first = args.warmup * launches_per_iter
        print(f"\n{'=' * 80}")
        print(f"Total launches executed: {total_launches}")
        print(f"\nTo profile just the FIRST post-warmup forward kernel:")
        print(f"  ncu --kernel-name regex:traverse_payload_winding_numbers_kernel \\")
        print(f"      --launch-skip {skip_first} --launch-count 1 \\")
        print(f"      --set detailed python {sys.argv[0]} <same args>")
        print(f"\nTo profile the first post-warmup pair (fwd + bwd):")
        print(f"  ncu --launch-skip {skip_first} --launch-count 2 \\")
        print(f"      --set detailed python {sys.argv[0]} <same args>")
        print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
