"""
Comprehensive validation of the brute-force payload field kernels against a
float64 PyTorch reference.

The tests are organised around specific claims:

  1. Sharp formula correctness (eps=0, no scene_scale needed)
  2. Regularized formula correctness (eps>0, needs scene_scale)
  3. Scene-scale discovery (diagnostic)
  4. Sanity: D=1 ones payload collapses to winding numbers
  5. Linearity: w(a*v1 + b*v2) = a*w(v1) + b*w(v2)
  6. Far-field limit: d >> 2*eps_length  =>  matches sharp kernel
  7. D-path coverage (all MaxAcc dispatch cases)
  8. Edge cases (empty, single, at-primitive, degenerate payloads)
  9. Triangle sharp formula
 10. Triangle regularized formula
 11-13. Same sanity/linearity/edge-case checks for triangles
 14. Mesh == Triangle soup
 15. Mesh ones == winding
 16. Kahan on/off consistency

Usage
-----
    # Synthetic geometry (fast, no file needed)
    python test_payload_brute_force.py

    # Real mesh
    python test_payload_brute_force.py --obj_file benchy.obj

    # Focus on specific D values (e.g., the chunked path)
    python test_payload_brute_force.py --d_values 32,33,64,100

    # Cross-check that D=1 with ones gives exactly the winding number
    python test_payload_brute_force.py --test ones_only
"""

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
# Constants and small utilities
# =============================================================================
DEFAULT_EPSILON = 0.004  # 1/250, matches the library default
DEFAULT_D_VALUES = [1, 2, 3, 4, 5, 7, 8, 12, 16, 17, 31, 32, 33, 48, 64, 100]
DEFAULT_EPSILON_SWEEP = [0.0, 0.001, 0.004, 0.02, 0.1]

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _ok(msg):
    return f"{GREEN}{msg}{RESET}"


def _bad(msg):
    return f"{RED}{msg}{RESET}"


def _warn(msg):
    return f"{YELLOW}{msg}{RESET}"


def _hdr(msg):
    return f"{BOLD}{CYAN}{msg}{RESET}"


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_cuda(x, dtype=torch.float32) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x)).to(dtype).cuda().contiguous()


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _bbox_diagonal(positions: np.ndarray) -> float:
    p = np.asarray(positions, dtype=np.float32)
    return float(np.linalg.norm(p.max(axis=0) - p.min(axis=0)))

def _scene_scale(points: np.ndarray) -> float:
    """Matches SceneNormalization::effective_extent(scene_bounds)."""
    p = np.asarray(points, dtype=np.float32)
    extent = p.max(axis=0) - p.min(axis=0)
    scale = float(extent.max())
    return scale if scale > 1e-8 else 1.0


def _compare_arrays(a: np.ndarray, b: np.ndarray, rtol: float, atol: float) -> dict:
    """Return diagnostic dict comparing two float arrays element-wise."""
    diff = np.abs(a - b)
    denom = np.maximum(np.abs(b), 1e-30)
    rel = diff / denom
    return {
        "rms_abs": float(np.sqrt(np.mean(diff**2))),
        "max_abs": float(diff.max()) if diff.size else 0.0,
        "max_rel": float(rel.max()) if rel.size else 0.0,
        "passed": bool(np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)),
    }


# =============================================================================
# Reference implementations (float64, torch, CPU)
# =============================================================================
# The references implement the same formulas as the CUDA kernels, but in
# float64 so they provide a ground truth. They are intentionally written in
# pure torch so no additional dependency (scipy, etc.) is required.

FOUR_OVER_3SQRT_PI = 4.0 / (3.0 * math.sqrt(math.pi))
TWO_OVER_SQRT_PI = 2.0 / math.sqrt(math.pi)
INV_FOUR_PI = 1.0 / (4.0 * math.pi)
INV_TWO_PI = 1.0 / (2.0 * math.pi)
INV_SQRT2 = 1.0 / math.sqrt(2.0)

# Compact-regularization constants (match geometry.h winder_reg_compact).
RC_A2 = 0.4748737341529164
RC_A4 = -0.0606601717798213
RC_V = math.sqrt(2.0)
RC_S = 1.0 / math.sqrt(2.0)
RC_C = -0.6568542494923802
RC_D = 0.5355339059327378


def _t(x) -> torch.Tensor:
    """Convert numpy array or python sequence to float64 CPU torch tensor."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().double()
    return torch.from_numpy(np.asarray(x, dtype=np.float64)).contiguous()


def ref_pn_winding_matrix_sharp(
    points: np.ndarray, normals: np.ndarray, queries: np.ndarray
) -> torch.Tensor:
    """(M, N) winding matrix for the sharp point-normal kernel (eps=0)."""
    p = _t(points)
    n = _t(normals)
    q = _t(queries)
    d = p.unsqueeze(0) - q.unsqueeze(1)  # (M, N, 3)
    dist2 = (d * d).sum(-1) + 1e-20  # (M, N)
    inv_dist3 = dist2.pow(-1.5)  # (M, N)
    ndotd = (n.unsqueeze(0) * d).sum(-1)  # (M, N)
    return ndotd * inv_dist3 * INV_FOUR_PI  # (M, N)


def ref_pn_winding_matrix(
    points: np.ndarray,
    normals: np.ndarray,
    queries: np.ndarray,
    epsilon: float,
    scene_scale: float,
) -> torch.Tensor:
    """(M, N) winding matrix for the regularized point-normal kernel."""
    p = _t(points)
    n = _t(normals)
    q = _t(queries)
    eps_length = float(epsilon) * float(scene_scale)

    d = p.unsqueeze(0) - q.unsqueeze(1)  # (M, N, 3)
    dist2 = (d * d).sum(-1) + 1e-20  # (M, N)
    dist = dist2.sqrt()
    inv_dist3 = dist2.pow(-1.5)
    ndotd = (n.unsqueeze(0) * d).sum(-1)  # (M, N)

    if eps_length <= 0.0:
        s_over_dist3 = inv_dist3
    else:
        t = dist / eps_length
        near = torch.full_like(t, FOUR_OVER_3SQRT_PI / (eps_length**3))
        s_t = torch.erf(t) - TWO_OVER_SQRT_PI * t * torch.exp(-t * t)
        mid = s_t / dist.pow(3)
        s_over_dist3 = torch.where(
            t < 0.1,
            near,
            torch.where(t < 2.0, mid, inv_dist3),
        )

    return ndotd * s_over_dist3 * INV_FOUR_PI


def _ref_compact_inv_r(a2: torch.Tensor, eps: float) -> torch.Tensor:
    """Regularized 1/|a| for the compact kernel."""
    if eps <= 0.0:
        return a2.pow(-0.5)

    inv_eps = 1.0 / eps
    inv_eps2 = 1.0 / (eps * eps)
    t2 = a2 * inv_eps2

    # Quartic [0, 1]
    t4 = t2 * t2
    g_q = 1.0 + RC_A2 * t2 + RC_A4 * t4
    inv_r_quartic = inv_eps / g_q

    # Cubic [1, 2]
    t = t2.sqrt()
    u = t - 1.0
    u2 = u * u
    u3 = u2 * u
    g_c = RC_V + RC_S * u + RC_C * u2 + RC_D * u3
    inv_r_cubic = inv_eps / g_c

    # Sharp (> 4)
    inv_r_sharp = a2.pow(-0.5)

    return torch.where(
        t2 <= 1.0,
        inv_r_quartic,
        torch.where(t2 <= 4.0, inv_r_cubic, inv_r_sharp),
    )


def _ref_angular_bump(u: torch.Tensor) -> torch.Tensor:
    t = torch.clamp((u - 0.25) * (4.0 / 3.0), 0.0, 1.0)
    return 1.0 - t * t * (3.0 - 2.0 * t)


def ref_tri_winding_matrix_sharp(
    triangles: np.ndarray, queries: np.ndarray
) -> torch.Tensor:
    """(M, N) winding matrix for the sharp triangle-soup kernel (eps=0)."""
    v = _t(triangles)  # (N, 3, 3)
    q = _t(queries)  # (M, 3)
    v0 = v[:, 0].unsqueeze(0)  # (1, N, 3)
    v1 = v[:, 1].unsqueeze(0)
    v2 = v[:, 2].unsqueeze(0)
    qb = q.unsqueeze(1)  # (M, 1, 3)

    a = v0 - qb
    b = v1 - qb
    c = v2 - qb

    a2 = (a * a).sum(-1) + 1e-20  # (M, N)
    b2 = (b * b).sum(-1) + 1e-20
    c2 = (c * c).sum(-1) + 1e-20
    inv_a = a2.pow(-0.5)
    inv_b = b2.pow(-0.5)
    inv_c = c2.pow(-0.5)

    cos_ab = (a * b).sum(-1) * inv_a * inv_b
    cos_ac = (a * c).sum(-1) * inv_a * inv_c
    cos_bc = (b * c).sum(-1) * inv_b * inv_c

    det = (a * torch.cross(b, c, dim=-1)).sum(-1) * inv_a * inv_b * inv_c
    div = 1.0 + cos_ab + cos_ac + cos_bc

    # Match geometry.h: RegCompact with eps=0 does NOT apply the |Q|<1e-6
    # -> 0.5 sharp check (that is RegSharp-only). Just use atan2.
    return torch.atan2(det, div) * INV_TWO_PI


def ref_tri_winding_matrix(
    triangles: np.ndarray,
    queries: np.ndarray,
    epsilon: float,
    scene_scale: float,
) -> torch.Tensor:
    """(M, N) winding matrix for the regularized compact triangle kernel."""
    v = _t(triangles)
    q = _t(queries)
    eps_length = float(epsilon) * float(scene_scale)
    eps_angle = float(epsilon)

    v0 = v[:, 0].unsqueeze(0)
    v1 = v[:, 1].unsqueeze(0)
    v2 = v[:, 2].unsqueeze(0)
    qb = q.unsqueeze(1)

    a = v0 - qb
    b = v1 - qb
    c = v2 - qb

    a2 = (a * a).sum(-1) + 1e-20
    b2 = (b * b).sum(-1) + 1e-20
    c2 = (c * c).sum(-1) + 1e-20

    inv_a = _ref_compact_inv_r(a2, eps_length)
    inv_b = _ref_compact_inv_r(b2, eps_length)
    inv_c = _ref_compact_inv_r(c2, eps_length)

    cos_ab = (a * b).sum(-1) * inv_a * inv_b
    cos_ac = (a * c).sum(-1) * inv_a * inv_c
    cos_bc = (b * c).sum(-1) * inv_b * inv_c

    det = (a * torch.cross(b, c, dim=-1)).sum(-1) * inv_a * inv_b * inv_c
    div = 1.0 + cos_ab + cos_ac + cos_bc
    P = det
    Q = div

    if eps_length > 0.0 and eps_angle > 0.0:
        inv_eps_angle2 = 1.0 / (eps_angle * eps_angle)
        u = (P * P + Q * Q) * inv_eps_angle2
        bump = _ref_angular_bump(u)
        c_shift = eps_angle * INV_SQRT2 * bump
        P = P + c_shift
        Q = Q + c_shift

    return torch.atan2(P, Q) * INV_TWO_PI


# =============================================================================
# Library call wrappers
# =============================================================================
def _run_pn_payload(
    points, normals, payload, queries, epsilon, use_kahan=False
) -> torch.Tensor:
    p = _to_cuda(points)
    n = _to_cuda(normals)
    pl = _to_cuda(payload)
    q = _to_cuda(queries)
    D = payload.shape[1]
    M = queries.shape[0]
    out = torch.empty((M, D), device="cuda:0", dtype=torch.float32)
    winder.brute_force_winding_payloads_point_normal(
        p,
        n,
        pl,
        q,
        out,
        float(epsilon),
        torch.cuda.current_stream().cuda_stream,
        bool(use_kahan),
    )
    torch.cuda.synchronize()
    return out


def _run_tri_payload(triangles, payload, queries, epsilon, use_kahan=False):
    t = _to_cuda(triangles)
    pl = _to_cuda(payload)
    q = _to_cuda(queries)
    D = payload.shape[1]
    M = queries.shape[0]
    out = torch.empty((M, D), device="cuda:0", dtype=torch.float32)
    winder.brute_force_winding_payloads_triangle_soup(
        t,
        pl,
        q,
        out,
        float(epsilon),
        torch.cuda.current_stream().cuda_stream,
        bool(use_kahan),
    )
    torch.cuda.synchronize()
    return out


def _run_mesh_payload(vertices, indices, payload, queries, epsilon, use_kahan=False):
    v = _to_cuda(vertices)
    idx = _to_cuda(indices, torch.uint32)
    pl = _to_cuda(payload)
    q = _to_cuda(queries)
    D = payload.shape[1]
    M = queries.shape[0]
    out = torch.empty((M, D), device="cuda:0", dtype=torch.float32)
    winder.brute_force_winding_payloads_mesh(
        v,
        idx,
        pl,
        q,
        out,
        float(epsilon),
        torch.cuda.current_stream().cuda_stream,
        bool(use_kahan),
    )
    torch.cuda.synchronize()
    return out


def _run_pn_winding(points, normals, queries, epsilon, use_kahan=True) -> np.ndarray:
    p = _to_cuda(points)
    n = _to_cuda(normals)
    q = _to_cuda(queries)
    M = queries.shape[0]
    out = torch.empty((M,), device="cuda:0", dtype=torch.float32)
    winder.brute_force_winding_numbers_point_normal(
        p,
        n,
        q,
        out,
        float(epsilon),
        torch.cuda.current_stream().cuda_stream,
        bool(use_kahan),
    )
    torch.cuda.synchronize()
    return _to_numpy(out)


def _run_tri_winding(triangles, queries, epsilon, use_kahan=True) -> np.ndarray:
    t = _to_cuda(triangles)
    q = _to_cuda(queries)
    M = queries.shape[0]
    out = torch.empty((M,), device="cuda:0", dtype=torch.float32)
    winder.brute_force_winding_numbers_triangle_soup(
        t,
        q,
        out,
        float(epsilon),
        torch.cuda.current_stream().cuda_stream,
        bool(use_kahan),
    )
    torch.cuda.synchronize()
    return _to_numpy(out)


def _run_mesh_winding(
    vertices, indices, queries, epsilon, use_kahan=True
) -> np.ndarray:
    v = _to_cuda(vertices)
    idx = _to_cuda(indices, torch.uint32)
    q = _to_cuda(queries)
    M = queries.shape[0]
    out = torch.empty((M,), device="cuda:0", dtype=torch.float32)
    winder.brute_force_winding_numbers_mesh(
        v,
        idx,
        q,
        out,
        float(epsilon),
        torch.cuda.current_stream().cuda_stream,
        bool(use_kahan),
    )
    torch.cuda.synchronize()
    return _to_numpy(out)


# =============================================================================
# Test result bookkeeping
# =============================================================================
@dataclass
class TestResult:
    name: str
    passed: bool
    details: dict = field(default_factory=dict)
    notes: str = ""


def _record(results: list, name: str, outcome: bool, **details) -> None:
    details.pop("passed", None)
    results.append(TestResult(name=name, passed=outcome, details=details))


def _print_subresult(name: str, passed: bool, summary: str = "") -> None:
    mark = _ok("PASS") if passed else _bad("FAIL")
    print(f"    [{mark}] {name}" + (f"  {summary}" if summary else ""))


# =============================================================================
# Individual tests
# =============================================================================
def test_pn_sharp_reference(points, normals, queries, D_values, results) -> None:
    """Point-normal payload matches the exact free-space formula at eps=0."""
    print(_hdr("\nTest: PN sharp reference (D sweep, eps=0)"))
    N = len(points)
    w_ref = ref_pn_winding_matrix_sharp(points, normals, queries)  # (M, N)

    for D in D_values:
        rng = np.random.default_rng(100 + D)
        payload = rng.standard_normal((N, D)).astype(np.float32)
        out = _run_pn_payload(points, normals, payload, queries, 0.0)
        ref = (w_ref @ _t(payload)).numpy()

        cmp = _compare_arrays(_to_numpy(out), ref, rtol=1e-4, atol=1e-5)
        ok = cmp["passed"] and out.shape == (len(queries), D)
        _print_subresult(
            f"D={D:>3}",
            ok,
            f"RMS={cmp['rms_abs']:.2e} max={cmp['max_abs']:.2e}",
        )
        _record(results, f"pn_sharp_D{D}", ok, **cmp)


def test_pn_regularized_reference(
    points, normals, queries, epsilon_values, D_values, scene_scale, results
) -> None:
    """Point-normal payload matches the regularized formula at eps>0."""
    print(_hdr(f"\nTest: PN regularized reference (scene_scale={scene_scale:.4g})"))
    N = len(points)

    for eps in epsilon_values:
        if eps <= 1e-12:
            continue  # covered by the sharp test
        w_ref = ref_pn_winding_matrix(points, normals, queries, eps, scene_scale)
        all_ok = True
        for D in D_values:
            rng = np.random.default_rng(300 + D + int(eps * 10000))
            payload = rng.standard_normal((N, D)).astype(np.float32)
            out = _run_pn_payload(points, normals, payload, queries, eps)
            ref = (w_ref @ _t(payload)).numpy()
            cmp = _compare_arrays(_to_numpy(out), ref, rtol=1e-3, atol=1e-4)
            if not cmp["passed"]:
                all_ok = False
                print(
                    f"      D={D:>3}: RMS={cmp['rms_abs']:.2e} max={cmp['max_abs']:.2e}"
                )
        _print_subresult(
            f"eps={eps:g}",
            all_ok,
            f"(checked {len(D_values)} D values)",
        )
        _record(results, f"pn_reg_eps{eps:g}", all_ok)


def test_pn_scene_scale_discovery(points, normals, queries, epsilon, results) -> None:
    """Diagnostic: report which scene_scale makes the reference match."""
    print(_hdr("\nTest: PN scene scale discovery (diagnostic only)"))
    N = len(points)
    wn_lib = _run_pn_winding(points, normals, queries, epsilon)

    candidates = {
        "bbox_diag": _bbox_diagonal(points),
        "bbox_max_dim": float((points.max(0) - points.min(0)).max()),
        "bbox_sum_dims": float((points.max(0) - points.min(0)).sum()),
        "bbox_rms_dims": float(np.sqrt(((points.max(0) - points.min(0)) ** 2).mean())),
    }
    base_diag = candidates["bbox_diag"]
    for mult in (0.5, 2.0, 4.0):
        candidates[f"bbox_diag_x{mult}"] = base_diag * mult

    best = None
    for name, s in candidates.items():
        if s <= 0:
            continue
        w_ref = ref_pn_winding_matrix(points, normals, queries, epsilon, s)
        wn_ref = w_ref.sum(1).numpy()
        rms = float(np.sqrt(np.mean((wn_lib - wn_ref) ** 2)))
        if best is None or rms < best[2]:
            best = (name, s, rms)
        print(f"    {name:<18} scene_scale={s:>10.4g}  RMS(vs lib)={rms:.3e}")

    if best is not None:
        print(
            f"    {_ok('BEST')}: {best[0]} (scene_scale={best[1]:.6g}, "
            f"RMS={best[2]:.3e})"
        )
    # This test is diagnostic, always passes
    _record(results, "pn_scene_scale_discovery", True, best=best)


def test_pn_ones_equals_winding(
    points, normals, queries, epsilon_values, D_values, results
) -> None:
    """D=1 ones payload = winding number; all D columns identical."""
    print(_hdr("\nTest: PN ones==winding and per-column equality"))
    N = len(points)

    for eps in epsilon_values:
        wn = _run_pn_winding(points, normals, queries, eps)  # (M,)
        for D in D_values:
            payload_ones = np.ones((N, D), np.float32)
            out = _run_pn_payload(
                points,
                normals,
                payload_ones,
                queries,
                eps,
                use_kahan=True,
            )
            col0 = _to_numpy(out)[:, 0]
            cmp = _compare_arrays(col0, wn, rtol=1e-3, atol=1e-4)

            all_cols_ok = True
            if D > 1:
                arr = _to_numpy(out)
                all_cols_ok = np.allclose(
                    arr,
                    arr[:, [0]] * np.ones((1, D), np.float32),
                    rtol=1e-6,
                    atol=1e-7,
                )

            ok = cmp["passed"] and all_cols_ok
            if not ok:
                print(
                    f"    eps={eps:g} D={D}: col0 RMS={cmp['rms_abs']:.2e} "
                    f"max={cmp['max_abs']:.2e} cols_identical={all_cols_ok}"
                )
            _record(
                results,
                f"pn_ones_eps{eps:g}_D{D}",
                ok,
                **cmp,
                cols_identical=all_cols_ok,
            )
        _print_subresult(
            f"eps={eps:g}",
            all(r.passed for r in results[-len(D_values) :]),
            f"(checked {len(D_values)} D values)",
        )


def test_pn_linearity(points, normals, queries, epsilon_values, results) -> None:
    """w(a*v1 + b*v2) == a*w(v1) + b*w(v2) at every eps."""
    print(_hdr("\nTest: PN linearity"))
    N = len(points)
    M = len(queries)
    D = 8
    rng = np.random.default_rng(999)
    v1 = rng.standard_normal((N, D)).astype(np.float32)
    v2 = rng.standard_normal((N, D)).astype(np.float32)
    a, b = 1.7, -0.83
    vc = (a * v1 + b * v2).astype(np.float32)

    for eps in epsilon_values:
        f1 = _to_numpy(_run_pn_payload(points, normals, v1, queries, eps))
        f2 = _to_numpy(_run_pn_payload(points, normals, v2, queries, eps))
        fc = _to_numpy(_run_pn_payload(points, normals, vc, queries, eps))
        cmp = _compare_arrays(fc, a * f1 + b * f2, rtol=1e-4, atol=1e-4)
        _print_subresult(
            f"eps={eps:g}",
            cmp["passed"],
            f"RMS={cmp['rms_abs']:.2e} max={cmp['max_abs']:.2e}",
        )
        _record(results, f"pn_linearity_eps{eps:g}", cmp["passed"], **cmp)


def test_pn_far_field_is_sharp(
    points, normals, queries, epsilon_values, results
) -> None:
    """For d >> 2*eps*scale, regularized == sharp."""
    print(_hdr("\nTest: PN far-field limit (regularized == sharp at large distance)"))
    # Push queries far away: 20x the original bbox
    diag = _bbox_diagonal(points)
    center = (points.max(0) + points.min(0)) * 0.5
    scale = diag * 20.0
    rng = np.random.default_rng(101)
    queries_far = (center + scale * rng.uniform(-1, 1, (len(queries), 3))).astype(
        np.float32
    )
    N = len(points)
    D = 4
    rng = np.random.default_rng(102)
    payload = rng.standard_normal((N, D)).astype(np.float32)

    w_sharp = _to_numpy(_run_pn_payload(points, normals, payload, queries_far, 0.0))

    for eps in epsilon_values:
        if eps <= 1e-12:
            continue
        w_reg = _to_numpy(_run_pn_payload(points, normals, payload, queries_far, eps))
        cmp = _compare_arrays(w_reg, w_sharp, rtol=1e-3, atol=1e-6)
        _print_subresult(
            f"eps={eps:g}",
            cmp["passed"],
            f"RMS={cmp['rms_abs']:.2e} max={cmp['max_abs']:.2e}",
        )
        _record(results, f"pn_farfield_eps{eps:g}", cmp["passed"], **cmp)


def test_pn_d_paths(points, normals, queries, results) -> None:
    """All D dispatch paths run and produce correct shape + finite output."""
    print(_hdr("\nTest: PN D-path coverage (all dispatch cases)"))
    N = len(points)
    M = len(queries)
    # Include the specialized bucket boundaries and the chunked ones
    path_ds = [1, 2, 3, 4, 5, 7, 8, 9, 12, 16, 17, 20, 31, 32, 33, 40, 48, 64, 100]
    all_ok = True
    for D in path_ds:
        rng = np.random.default_rng(200 + D)
        payload = rng.standard_normal((N, D)).astype(np.float32)
        try:
            out = _run_pn_payload(points, normals, payload, queries, 0.0)
            ok = out.shape == (M, D) and bool(torch.isfinite(out).all())
        except Exception as e:
            ok = False
            print(f"    {_bad('EXCEPTION')} D={D}: {e}")
        if not ok and out is not None:
            print(f"    {_bad('FAIL')} D={D} shape={tuple(out.shape)}")
        all_ok = all_ok and ok
    _print_subresult(f"all D in {path_ds}", all_ok)
    _record(results, "pn_d_paths", all_ok)


def test_pn_edge_cases(points, normals, queries, epsilon, results) -> None:
    """Zero/NaN/single/at-primitive/large payload."""
    print(_hdr("\nTest: PN edge cases"))
    N = len(points)
    M = len(queries)
    D = 4

    # 1. Zero payload -> zero output
    out = _run_pn_payload(
        points, normals, np.zeros((N, D), np.float32), queries, epsilon
    )
    ok = np.allclose(_to_numpy(out), 0.0, atol=1e-8)
    _print_subresult("zero payload -> zero output", ok)
    _record(results, "pn_edge_zero_payload", ok)

    # 2. NaN payload propagates
    pl = np.ones((N, D), np.float32)
    pl[0, 0] = np.nan
    out = _run_pn_payload(points, normals, pl, queries, epsilon)
    ok = bool(np.isnan(_to_numpy(out)[:, 0]).all())
    _print_subresult("NaN propagates through column", ok)
    _record(results, "pn_edge_nan_payload", ok)

    # 3. Single primitive, single query
    rng = np.random.default_rng(501)
    p1 = rng.standard_normal((1, 3)).astype(np.float32)
    n1 = rng.standard_normal((1, 3)).astype(np.float32)
    q1 = rng.standard_normal((1, 3)).astype(np.float32) * 2
    pl1 = rng.standard_normal((1, D)).astype(np.float32)
    out = _run_pn_payload(p1, n1, pl1, q1, epsilon)
    ref = (
        ref_pn_winding_matrix(p1, n1, q1, epsilon, _bbox_diagonal(p1)) @ _t(pl1)
    ).numpy()
    cmp = _compare_arrays(_to_numpy(out), ref, rtol=1e-3, atol=1e-4)
    _print_subresult("N=1, M=1", cmp["passed"])
    _record(results, "pn_edge_single", cmp["passed"], **cmp)

    # 4. Query exactly at primitive position (regularized -> finite, w=0)
    out = _run_pn_payload(
        points[:1],
        normals[:1],
        np.ones((1, D), np.float32),
        points[:1],
        epsilon,
    )
    arr = _to_numpy(out)
    ok = bool(np.isfinite(arr).all()) and np.allclose(arr, 0.0, atol=1e-6)
    _print_subresult("at-primitive (eps>0) -> finite, 0", ok)
    _record(results, "pn_edge_at_primitive", ok)

    # 5. Large payload values
    pl_big = np.full((N, D), 1e6, np.float32)
    out = _run_pn_payload(points, normals, pl_big, queries, epsilon)
    ok = bool(torch.isfinite(out).all())
    _print_subresult("large payload (1e6) -> finite", ok)
    _record(results, "pn_edge_large_payload", ok)


def test_tri_sharp_reference(triangles, queries, D_values, results) -> None:
    """Triangle payload matches exact Van Oosterom formula at eps=0."""
    print(_hdr("\nTest: Triangle sharp reference (D sweep, eps=0)"))
    N = len(triangles)
    w_ref = ref_tri_winding_matrix_sharp(triangles, queries)

    for D in D_values:
        rng = np.random.default_rng(600 + D)
        payload = rng.standard_normal((N, D)).astype(np.float32)
        out = _run_tri_payload(triangles, payload, queries, 0.0)
        ref = (w_ref @ _t(payload)).numpy()
        cmp = _compare_arrays(_to_numpy(out), ref, rtol=2e-3, atol=1e-4)
        _print_subresult(
            f"D={D:>3}",
            cmp["passed"],
            f"RMS={cmp['rms_abs']:.2e} max={cmp['max_abs']:.2e}",
        )
        _record(results, f"tri_sharp_D{D}", cmp["passed"], **cmp)


def test_tri_regularized_reference(
    triangles, queries, epsilon_values, D_values, scene_scale, results
) -> None:
    """Triangle payload matches the regularized compact kernel."""
    print(
        _hdr(f"\nTest: Triangle regularized reference (scene_scale={scene_scale:.4g})")
    )
    N = len(triangles)

    for eps in epsilon_values:
        if eps <= 1e-12:
            continue
        w_ref = ref_tri_winding_matrix(triangles, queries, eps, scene_scale)
        all_ok = True
        worst_rms = 0.0
        for D in D_values:
            rng = np.random.default_rng(700 + D + int(eps * 10000))
            payload = rng.standard_normal((N, D)).astype(np.float32)
            out = _run_tri_payload(triangles, payload, queries, eps)
            ref = (w_ref @ _t(payload)).numpy()
            cmp = _compare_arrays(_to_numpy(out), ref, rtol=1e-2, atol=1e-3)
            worst_rms = max(worst_rms, cmp["rms_abs"])
            if not cmp["passed"]:
                all_ok = False
                print(
                    f"      D={D:>3}: RMS={cmp['rms_abs']:.2e} max={cmp['max_abs']:.2e}"
                )
        _print_subresult(
            f"eps={eps:g}",
            all_ok,
            f"(worst RMS across D: {worst_rms:.2e})",
        )
        _record(results, f"tri_reg_eps{eps:g}", all_ok)


def test_tri_ones_equals_winding(
    triangles, queries, epsilon_values, D_values, results
) -> None:
    """Triangle D=1 ones = winding number; all D columns identical."""
    print(_hdr("\nTest: Triangle ones==winding and per-column equality"))
    N = len(triangles)

    for eps in epsilon_values:
        wn = _run_tri_winding(triangles, queries, eps)
        for D in D_values:
            payload_ones = np.ones((N, D), np.float32)
            out = _run_tri_payload(
                triangles, payload_ones, queries, eps, use_kahan=True
            )
            col0 = _to_numpy(out)[:, 0]
            cmp = _compare_arrays(col0, wn, rtol=3e-3, atol=3e-4)

            all_cols_ok = True
            if D > 1:
                arr = _to_numpy(out)
                all_cols_ok = np.allclose(
                    arr,
                    arr[:, [0]] * np.ones((1, D), np.float32),
                    rtol=1e-6,
                    atol=1e-7,
                )
            ok = cmp["passed"] and all_cols_ok
            if not ok:
                print(f"    eps={eps:g} D={D}: col0 RMS={cmp['rms_abs']:.2e}")
            _record(results, f"tri_ones_eps{eps:g}_D{D}", ok, **cmp)


def test_tri_linearity(triangles, queries, epsilon_values, results) -> None:
    """Same linearity property as PN."""
    print(_hdr("\nTest: Triangle linearity"))
    N = len(triangles)
    D = 8
    rng = np.random.default_rng(801)
    v1 = rng.standard_normal((N, D)).astype(np.float32)
    v2 = rng.standard_normal((N, D)).astype(np.float32)
    a, b = 1.7, -0.83
    vc = (a * v1 + b * v2).astype(np.float32)

    for eps in epsilon_values:
        f1 = _to_numpy(_run_tri_payload(triangles, v1, queries, eps))
        f2 = _to_numpy(_run_tri_payload(triangles, v2, queries, eps))
        fc = _to_numpy(_run_tri_payload(triangles, vc, queries, eps))
        cmp = _compare_arrays(fc, a * f1 + b * f2, rtol=1e-4, atol=1e-4)
        _print_subresult(
            f"eps={eps:g}",
            cmp["passed"],
            f"RMS={cmp['rms_abs']:.2e} max={cmp['max_abs']:.2e}",
        )
        _record(results, f"tri_linearity_eps{eps:g}", cmp["passed"], **cmp)


def test_tri_edge_cases(triangles, queries, epsilon, results) -> None:
    """Zero/NaN/single/at-primitive for triangles."""
    print(_hdr("\nTest: Triangle edge cases"))
    N = len(triangles)
    D = 4

    # Zero payload -> zero
    out = _run_tri_payload(triangles, np.zeros((N, D), np.float32), queries, epsilon)
    ok = np.allclose(_to_numpy(out), 0.0, atol=1e-8)
    _print_subresult("zero payload -> zero", ok)
    _record(results, "tri_edge_zero_payload", ok)

    # NaN propagation
    pl = np.ones((N, D), np.float32)
    pl[0, 0] = np.nan
    out = _run_tri_payload(triangles, pl, queries, epsilon)
    ok = bool(np.isnan(_to_numpy(out)[:, 0]).all())
    _print_subresult("NaN propagates", ok)
    _record(results, "tri_edge_nan_payload", ok)

    # Single triangle, single query (using triangle soup with N=1)
    tri1 = triangles[:1].copy()
    rng = np.random.default_rng(901)
    q1 = (tri1[0].mean(0) + rng.standard_normal(3) * 2.0).astype(np.float32)
    pl1 = rng.standard_normal((1, D)).astype(np.float32)
    out = _run_tri_payload(tri1, pl1, q1[None, :], epsilon)
    ref = (
        ref_tri_winding_matrix(
            tri1, q1[None, :], epsilon, _bbox_diagonal(tri1.reshape(-1, 3))
        )
        @ _t(pl1)
    ).numpy()
    cmp = _compare_arrays(_to_numpy(out), ref, rtol=1e-2, atol=1e-3)
    _print_subresult("N=1, M=1", cmp["passed"])
    _record(results, "tri_edge_single", cmp["passed"], **cmp)

    # Query at triangle centroid (regularized -> finite)
    centroid = triangles[0].mean(0).reshape(1, 3).astype(np.float32)
    out = _run_tri_payload(
        triangles[:1], np.ones((1, D), np.float32), centroid, epsilon
    )
    ok = bool(np.isfinite(_to_numpy(out)).all())
    _print_subresult("query at triangle centroid -> finite", ok)
    _record(results, "tri_edge_at_centroid", ok)


def test_mesh_vs_triangle_soup(
    vertices, indices, queries, epsilon_values, D_values, results
) -> None:
    """Mesh payload must be bit-identical to triangle soup on the same triangles."""
    print(_hdr("\nTest: Mesh == Triangle soup"))
    triangles = vertices[indices]
    N = len(triangles)

    for eps in epsilon_values:
        all_ok = True
        for D in D_values:
            rng = np.random.default_rng(1000 + D + int(eps * 10000))
            payload = rng.standard_normal((N, D)).astype(np.float32)
            out_mesh = _to_numpy(
                _run_mesh_payload(vertices, indices, payload, queries, eps)
            )
            out_soup = _to_numpy(_run_tri_payload(triangles, payload, queries, eps))
            cmp = _compare_arrays(out_mesh, out_soup, rtol=1e-7, atol=1e-7)
            if not cmp["passed"]:
                all_ok = False
                print(
                    f"      eps={eps:g} D={D}: RMS={cmp['rms_abs']:.2e} "
                    f"max={cmp['max_abs']:.2e}"
                )
        _print_subresult(f"eps={eps:g}", all_ok)
        _record(results, f"mesh_soup_eps{eps:g}", all_ok)


def test_mesh_ones_equals_winding(vertices, indices, queries, epsilon, results) -> None:
    """Mesh D=1 ones == winding through mesh path."""
    print(_hdr("\nTest: Mesh ones==winding"))
    N = len(indices)
    wn = _run_mesh_winding(vertices, indices, queries, epsilon)
    payload_ones = np.ones((N, 1), np.float32)
    out = _run_mesh_payload(
        vertices, indices, payload_ones, queries, epsilon, use_kahan=True
    )
    col0 = _to_numpy(out)[:, 0]
    cmp = _compare_arrays(col0, wn, rtol=3e-3, atol=3e-4)
    _print_subresult(
        f"eps={epsilon:g}",
        cmp["passed"],
        f"RMS={cmp['rms_abs']:.2e} max={cmp['max_abs']:.2e}",
    )
    _record(results, "mesh_ones", cmp["passed"], **cmp)


def test_kahan_consistency(points, normals, queries, epsilon, results) -> None:
    """Kahan on/off agree (both approximate the same field)."""
    print(_hdr("\nTest: Kahan on/off consistency"))
    N = len(points)
    D = 8
    rng = np.random.default_rng(1100)
    payload = rng.standard_normal((N, D)).astype(np.float32)
    out_off = _to_numpy(
        _run_pn_payload(points, normals, payload, queries, epsilon, use_kahan=False)
    )
    out_on = _to_numpy(
        _run_pn_payload(points, normals, payload, queries, epsilon, use_kahan=True)
    )
    cmp = _compare_arrays(out_on, out_off, rtol=1e-4, atol=1e-5)
    _print_subresult(
        "",
        cmp["passed"],
        f"RMS={cmp['rms_abs']:.2e} max={cmp['max_abs']:.2e}",
    )
    _record(results, "pn_kahan_consistency", cmp["passed"], **cmp)

def test_kahan_error_comparison(points, normals, queries, epsilon, D, results):
    """Report absolute error of kahan=on vs kahan=off against the fp64 reference.

    Uses the sharp kernel (eps=0) so the reference has no scene_scale dependency.
    """
    print(_hdr(f"\nTest: Kahan on vs off error  (D={D}, eps={epsilon:g})"))
    N = len(points)
    rng = np.random.default_rng(4242)
    payload = rng.standard_normal((N, D)).astype(np.float32)
    w_ref = (ref_pn_winding_matrix(points, normals, queries, epsilon,
                                   _bbox_diagonal(points))
             @ _t(payload)).numpy()

    out_off = _to_numpy(_run_pn_payload(points, normals, payload, queries,
                                         epsilon, use_kahan=False))
    out_on  = _to_numpy(_run_pn_payload(points, normals, payload, queries,
                                         epsilon, use_kahan=True))

    cmp_off = _compare_arrays(out_off, w_ref, rtol=1e-3, atol=1e-5)
    cmp_on  = _compare_arrays(out_on,  w_ref, rtol=1e-3, atol=1e-5)
    cmp_kt  = _compare_arrays(out_on, out_off, rtol=1e-4, atol=1e-6)

    ratio = cmp_off["rms_abs"] / max(cmp_on["rms_abs"], 1e-30)
    print(f"    kahan=off vs ref : RMS={cmp_off['rms_abs']:.3e}  max={cmp_off['max_abs']:.3e}")
    print(f"    kahan=on  vs ref : RMS={cmp_on ['rms_abs']:.3e}  max={cmp_on ['max_abs']:.3e}")
    print(f"    kahan=on  vs off : RMS={cmp_kt ['rms_abs']:.3e}  max={cmp_kt ['max_abs']:.3e}")
    print(f"    error ratio (off / on) = {ratio:.2f}x")

    # This test passes if both are within tolerance vs the reference;
    # the reported numbers show whether kahan actually helps.
    ok = cmp_off["passed"] and cmp_on["passed"]
    _print_subresult(f"D={D}", ok,
                     f"rms_off={cmp_off['rms_abs']:.2e}  rms_on={cmp_on['rms_abs']:.2e}")
    _record(results, f"pn_kahan_compare_D{D}", ok,
            rms_off=cmp_off["rms_abs"], rms_on=cmp_on["rms_abs"],
            ratio=ratio)

def test_eps_threshold(points, normals, queries, results):
    print(_hdr("\nTest: eps threshold sweep"))
    N = len(points)
    D = 4
    rng = np.random.default_rng(555)
    payload = rng.standard_normal((N, D)).astype(np.float32)
    scene_scale = _bbox_diagonal(points)
    for eps in [0.004, 0.006, 0.008, 0.010, 0.012, 0.015, 0.018, 0.020,
                0.025, 0.030, 0.040, 0.060, 0.080, 0.100]:
        w_ref = ref_pn_winding_matrix(points, normals, queries, eps, scene_scale)
        ref = (w_ref @ _t(payload)).numpy()
        out = _to_numpy(_run_pn_payload(points, normals, payload, queries, eps))
        diff = np.abs(out - ref)
        rms = float(np.sqrt((diff**2).mean()))
        mx = float(diff.max())
        # how many queries are > 1e-2 off
        n_bad = int(((diff.max(axis=1)) > 1e-2).sum())
        print(f"    eps={eps:>5.3f}  rms={rms:.3e}  max={mx:.3e}  n_bad_q={n_bad}/{len(queries)}")
    _record(results, "eps_threshold", True)

def test_single_prim_sweep(results):
    """Sweep a single primitive against a single query through every t range."""
    print(_hdr("\nTest: single-primitive contribution sweep (D=1)"))
    eps = 0.02
    scene_scale = 1.0
    eps_length = eps * scene_scale

    p = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    n = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    payload = np.array([[1.0]], dtype=np.float32)

    # t = dist / eps_length. Sweep through t=0.1 and t=2 boundaries.
    t_values = [
        1e-6, 1e-3, 0.01, 0.05, 0.09, 0.099, 0.0999,   # near -> near-boundary
        0.1001, 0.101, 0.11, 0.2, 0.5, 1.0, 1.5,       # mid
        1.9, 1.99, 1.999, 1.9999,                       # mid -> mid-boundary
        2.0001, 2.001, 2.01, 2.1, 3.0, 5.0, 10.0, 100.0  # far
    ]

    print(f"  {'t':>10} {'dist':>12} {'ref':>16} {'cuda':>16} {'rel_diff':>12} {'status':>8}")
    worst_rel = 0.0
    for t in t_values:
        dist = t * eps_length
        q = np.array([[dist, 0.0, 0.0]], dtype=np.float32)

        w_ref = ref_pn_winding_matrix(p, n, q, eps, scene_scale)
        ref_val = float(w_ref[0, 0])

        out = _to_numpy(_run_pn_payload(p, n, payload, q, eps))
        cuda_val = float(out[0, 0])

        diff = abs(cuda_val - ref_val)
        rel = diff / max(abs(ref_val), 1e-30)
        worst_rel = max(worst_rel, rel)
        status = "ok" if rel < 1e-4 else "MISMATCH"
        print(f"  {t:>10.4g} {dist:>12.4e} {ref_val:>16.8e} {cuda_val:>16.8e} "
              f"{rel:>12.2e} {status:>8}")

    print(f"\n  worst rel diff = {worst_rel:.3e}")
    _record(results, "single_prim_sweep", worst_rel < 1e-3)

def test_per_primitive_attribution(points, normals, queries, eps, results):
    """Find the (query, primitive) pair whose contribution CUDA and reference disagree on."""
    print(_hdr(f"\nTest: per-primitive attribution (eps={eps:g}, D=1)"))
    N = len(points)
    scene_scale = _bbox_diagonal(points)
    eps_length = max(eps * scene_scale, 1e-30)

    w_ref = ref_pn_winding_matrix(points, normals, queries, eps, scene_scale)
    ref_wn = w_ref.sum(1).numpy()

    payload_ones = np.ones((N, 1), np.float32)
    out = _to_numpy(_run_pn_payload(points, normals, payload_ones, queries, eps))[:, 0]
    err_per_q = np.abs(out - ref_wn)
    worst_q = int(err_per_q.argmax())

    print(f"  Worst query idx={worst_q}  err={err_per_q[worst_q]:.6e}")

    p = points.astype(np.float64)
    q_vec = queries[worst_q].astype(np.float64)
    d = p - q_vec[None, :]
    dist = np.linalg.norm(d, axis=1)
    t_vals = dist / eps_length
    contrib_ref = w_ref[worst_q].numpy()

    order = np.argsort(-np.abs(contrib_ref))

    print(f"\n  Top 20 reference contributions (by magnitude):")
    print(f"  {'idx':>5} {'t':>10} {'dist':>10} {'ref':>15} {'cuda_iso':>15} {'diff':>10} {'band':>6}")
    worst_pair = (-1, 0.0)
    for i in order[:20]:
        payload_i = np.zeros((N, 1), np.float32)
        payload_i[i, 0] = 1.0
        cuda_i = float(_to_numpy(
            _run_pn_payload(points, normals, payload_i, queries, eps)
        )[worst_q, 0])
        ref_i = float(contrib_ref[i])
        diff_i = abs(cuda_i - ref_i)
        band = "near" if t_vals[i] < 0.1 else ("mid" if t_vals[i] < 2.0 else "far")
        if diff_i > worst_pair[1]:
            worst_pair = (int(i), diff_i)
        print(f"  {i:>5} {t_vals[i]:>10.4e} {dist[i]:>10.4e} "
              f"{ref_i:>15.6e} {cuda_i:>15.6e} {diff_i:>10.3e} {band:>6}")

    print(f"\n  Worst isolated pair: primitive {worst_pair[0]}  diff={worst_pair[1]:.3e}")
    _record(results, f"per_prim_attrib_eps{eps:g}", True)
# =============================================================================
# Geometry setup
# =============================================================================
def make_synthetic_sphere(n_theta=20, n_phi=12, radius=1.0):
    """Closed UV sphere as (vertices, indices) — small enough to run fast."""
    theta = np.linspace(0, np.pi, n_theta + 1, dtype=np.float64)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False, dtype=np.float64)
    verts = []
    for i in range(1, n_theta):
        for j in range(n_phi):
            x = radius * np.sin(theta[i]) * np.cos(phi[j])
            y = radius * np.sin(theta[i]) * np.sin(phi[j])
            z = radius * np.cos(theta[i])
            verts.append([x, y, z])
    # Poles
    south = len(verts)
    verts.append([0, 0, -radius])
    north = len(verts)
    verts.append([0, 0, radius])
    verts = np.asarray(verts, dtype=np.float32)

    tris = []
    for i in range(n_theta - 2):
        for j in range(n_phi):
            a = i * n_phi + j
            b = i * n_phi + (j + 1) % n_phi
            c = (i + 1) * n_phi + j
            d = (i + 1) * n_phi + (j + 1) % n_phi
            tris.append([a, c, b])
            tris.append([b, c, d])
    for j in range(n_phi):
        tris.append(
            [south, (n_theta - 2) * n_phi + j, (n_theta - 2) * n_phi + (j + 1) % n_phi]
        )
        tris.append([north, j, (j + 1) % n_phi])
    indices = np.asarray(tris, dtype=np.uint32)
    return verts, indices


def mesh_to_point_surfels(vertices, indices):
    v0 = vertices[indices[:, 0]]
    v1 = vertices[indices[:, 1]]
    v2 = vertices[indices[:, 2]]
    centroids = (v0 + v1 + v2) / 3.0
    cross = np.cross(v1 - v0, v2 - v0)
    mag = np.linalg.norm(cross, axis=1, keepdims=True)
    areas = (mag / 2.0).flatten()
    safe = np.where(mag == 0, 1e-8, mag)
    unit_normals = cross / safe
    scaled_normals = (unit_normals * areas[:, None]).astype(np.float32)
    return centroids.astype(np.float32), scaled_normals


def generate_queries(vertices, num_queries, seed):
    rng = np.random.default_rng(seed)
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    diag = float(np.linalg.norm(hi - lo))
    lo = lo - 0.5 * diag
    hi = hi + 0.5 * diag
    return rng.uniform(lo, hi, size=(num_queries, 3)).astype(np.float32)


# =============================================================================
# Summary and main runner
# =============================================================================
def print_summary(results: list) -> bool:
    n_pass = sum(1 for r in results if r.passed)
    n_total = len(results)
    print("\n" + "=" * 100)
    print(_hdr(f"SUMMARY: {n_pass}/{n_total} tests passed"))
    print("=" * 100)

    failed = [r for r in results if not r.passed]
    if not failed:
        print(_ok("  ALL TESTS PASSED"))
        return True

    print(_bad(f"  FAILURES ({len(failed)}):"))
    for r in failed:
        print(f"    - {r.name}")
        if r.details:
            parts = []
            for k, v in r.details.items():
                if k == "passed":
                    continue
                if isinstance(v, float):
                    parts.append(f"{k}={v:.3e}")
                else:
                    parts.append(f"{k}={v}")
            print(f"        {', '.join(parts)}")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Validate brute-force payload kernels against PyTorch reference."
    )
    parser.add_argument(
        "--obj_file",
        type=str,
        default=None,
        help="Optional mesh file. If omitted, uses a synthetic sphere.",
    )
    parser.add_argument("--n_queries", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument(
        "--epsilon_values",
        type=str,
        default=",".join(str(e) for e in DEFAULT_EPSILON_SWEEP),
    )
    parser.add_argument(
        "--d_values", type=str, default=",".join(str(d) for d in DEFAULT_D_VALUES)
    )
    parser.add_argument(
        "--test",
        type=str,
        default="all",
        help="Which suite to run: 'all', 'pn', 'tri', 'mesh', "
        "'ones_only', 'sharp_only', 'discovery_only'",
    )
    args = parser.parse_args()

    set_seeds(args.seed)

    # ---- Geometry --------------------------------------------------------
    if args.obj_file:
        import igl

        v, _, _, f, _, _ = igl.readOBJ(args.obj_file)
        vertices = np.ascontiguousarray(v, dtype=np.float32)
        indices = np.ascontiguousarray(f, dtype=np.uint32)
        print(
            f"Loaded {args.obj_file}: {len(indices)} triangles, {len(vertices)} verts"
        )
    else:
        vertices, indices = make_synthetic_sphere()
        print(f"Synthetic sphere: {len(indices)} triangles, {len(vertices)} verts")

    triangles = vertices[indices]

    # Point-normal representation
    points, scaled_normals = mesh_to_point_surfels(vertices, indices)
    print(f"Point-normal surfels: {len(points)} points")
    print(f"Queries: {args.n_queries} uniform random")
    queries = generate_queries(vertices, args.n_queries, args.seed)

    # Scene scales (bbox diagonal of the underlying geometry)
    pn_scene_scale = _scene_scale(points)
    tri_scene_scale = _scene_scale(triangles.reshape(-1, 3))

    eps_values = [float(x) for x in args.epsilon_values.split(",") if x.strip()]
    D_values = [int(x) for x in args.d_values.split(",") if x.strip()]

    print(f"ε values: {eps_values}")
    print(f"D values: {D_values}")
    print(f"pn scene_scale={pn_scene_scale:.6g}, tri scene_scale={tri_scene_scale:.6g}")

    results: list = []
    t0 = time()

    # ---- Point-normal tests ---------------------------------------------
    if args.test in ("all", "pn", "sharp_only"):
        test_pn_sharp_reference(points, scaled_normals, queries, D_values, results)
    if args.test in ("all", "pn", "discovery_only"):
        test_pn_scene_scale_discovery(
            points, scaled_normals, queries, args.epsilon, results
        )
    if args.test in ("all", "pn") and not args.test == "sharp_only":
        test_pn_regularized_reference(
            points,
            scaled_normals,
            queries,
            eps_values,
            D_values,
            pn_scene_scale,
            results,
        )
    if args.test in ("all", "pn", "ones_only"):
        test_pn_ones_equals_winding(
            points, scaled_normals, queries, eps_values, D_values, results
        )
    if args.test in ("all", "pn"):
        test_pn_linearity(points, scaled_normals, queries, eps_values, results)
        test_pn_far_field_is_sharp(points, scaled_normals, queries, eps_values, results)
        test_pn_d_paths(points, scaled_normals, queries, results)
        test_pn_edge_cases(points, scaled_normals, queries, args.epsilon, results)
        test_kahan_consistency(points, scaled_normals, queries, args.epsilon, results)
        test_kahan_error_comparison(points, scaled_normals, queries, 0.0, 8, results)
        test_eps_threshold(points, scaled_normals, queries, results)
        test_single_prim_sweep(results)
        for  eps in eps_values:
            test_per_primitive_attribution(points, scaled_normals, queries, eps, results)

    # ---- Triangle tests --------------------------------------------------
    if args.test in ("all", "tri", "sharp_only"):
        test_tri_sharp_reference(triangles, queries, D_values, results)
    if args.test in ("all", "tri") and not args.test == "sharp_only":
        test_tri_regularized_reference(
            triangles, queries, eps_values, D_values, tri_scene_scale, results
        )
    if args.test in ("all", "tri", "ones_only"):
        test_tri_ones_equals_winding(triangles, queries, eps_values, D_values, results)
    if args.test in ("all", "tri"):
        test_tri_linearity(triangles, queries, eps_values, results)
        test_tri_edge_cases(triangles, queries, args.epsilon, results)

    # ---- Mesh tests ------------------------------------------------------
    if args.test in ("all", "mesh"):
        test_mesh_vs_triangle_soup(
            vertices, indices, queries, eps_values, D_values, results
        )
        test_mesh_ones_equals_winding(vertices, indices, queries, args.epsilon, results)

    elapsed = time() - t0
    ok = print_summary(results)
    print(f"\nElapsed: {elapsed:.2f}s")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
