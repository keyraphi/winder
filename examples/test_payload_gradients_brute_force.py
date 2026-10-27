"""
Comprehensive validation of the brute-force payload-gradient kernels against
a float64 PyTorch autograd reference.

Covers all three gradient families for the payload field:

  - ∂L/∂v_i         (payload gradient)
  - ∂L/∂geometry    (source geometry gradient, including ∂L/∂ε)
  - ∂L/∂x_j         (query gradient)

Reference: PyTorch float64 autograd computing the loss
    L = Σ_j <g_j, w_V(q_j)>   with   w_V(q_j) = Σ_i v_i · w_i(q_j)
end to end, then differentiating through the geometry / payload / queries /
epsilon that participate in the graph.

Full D-dispatch coverage: D ∈ {1, 2, 3, 4, 8, 16, 32, 33, 64}.
Sanity checks: D=1 with v=1 reproduces the scalar gradient; the payload
gradient sums to Σ_j g_j · W(q_j); linearity in both v and g.
Three geometry types (PointNormal / Triangle / Mesh) and three reg_modes
(sharp / plummer / compact).

The CUDA binary compiles with a specific `WINDER_DEFAULT_REG` compile-time
setting. To match, run with the corresponding `--reg_mode`. The default
of `--reg_mode compact` matches the default build.

Usage
-----
    # Synthetic sphere, full test matrix (fast, ~30 s)
    python test_payload_gradients_brute_force.py

    # Real mesh
    python test_payload_gradients_brute_force.py --obj_file benchy.obj

    # Focus on large D (chunked paths)
    python test_payload_gradients_brute_force.py --d_values 32,33,64,100

    # Only specific families
    python test_payload_gradients_brute_force.py --test payload
    python test_payload_gradients_brute_force.py --test sanity
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from time import time
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

import winder


# =============================================================================
# Constants
# =============================================================================
DEFAULT_EPSILON = 0.004  # 1/250, matches the library default
DEFAULT_D_VALUES = [1, 2, 3, 4, 8, 16, 32, 33, 64]
DEFAULT_EPSILON_SWEEP = [0.0, 0.004, 0.02]
REG_MODES = ["sharp", "plummer", "compact"]

_TWO_OVER_SQRT_PI = 1.1283791671
_FOUR_OVER_3SQRT_PI = 4.0 / (3.0 * math.sqrt(math.pi))
_QUARTIC_A2 = 0.4748737341529164
_QUARTIC_A4 = -0.0606601717798213
_CUBIC_V = math.sqrt(2.0)
_CUBIC_S = 1.0 / math.sqrt(2.0)
_CUBIC_C = -0.6568542494923802
_CUBIC_D = 0.5355339059327378

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
# Small utilities
# =============================================================================
def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_cuda(x, dtype=torch.float32) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x)).to(dtype).cuda().contiguous()


def _t(x) -> torch.Tensor:
    """numpy or torch → CPU float64 tensor."""
    if isinstance(x, torch.Tensor):
        return x.detach().double().cuda()
    return torch.from_numpy(np.asarray(x, dtype=np.float64)).contiguous().cuda()


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _scene_scale(positions: np.ndarray) -> float:
    """Matches SceneNormalization::effective_extent(scene_bounds)."""
    p = np.asarray(positions, dtype=np.float32)
    extent = p.max(axis=0) - p.min(axis=0)
    scale = float(extent.max())
    return scale if scale > 1e-8 else 1.0


def _as_eps_tensor(eps_grad: Optional[torch.Tensor]) -> torch.Tensor:
    if eps_grad is None:
        return torch.zeros(1, dtype=torch.float32, device="cuda")
    return eps_grad.detach().to(torch.float32).reshape(1)


# =============================================================================
# Regularization helpers (mirror geometry.h)
# =============================================================================
def torch_angular_bump(u: torch.Tensor) -> torch.Tensor:
    t = torch.clamp((u - 0.25) * (4.0 / 3.0), 0.0, 1.0)
    return 1.0 - t * t * (3.0 - 2.0 * t)


def torch_g_compact(t: torch.Tensor) -> torch.Tensor:
    tq = torch.clamp(t, max=1.0)
    tc = torch.clamp(t, min=1.0, max=2.0)
    ti = torch.clamp(t, min=2.0)
    g_q = 1.0 + _QUARTIC_A2 * tq * tq + _QUARTIC_A4 * tq**4
    u = tc - 1.0
    g_c = _CUBIC_V + _CUBIC_S * u + _CUBIC_C * u * u + _CUBIC_D * u**3
    g_i = ti
    return torch.where(t <= 1.0, g_q, torch.where(t <= 2.0, g_c, g_i))


def torch_regularized_edge_lengths(a, b, c, eps_world: torch.Tensor, reg_mode: str):
    """Return (r_a, r_b, r_c), the regularized edge lengths."""
    a2 = torch.sum(a * a, dim=-1) + 1e-20
    b2 = torch.sum(b * b, dim=-1) + 1e-20
    c2 = torch.sum(c * c, dim=-1) + 1e-20
    eps_value = float(eps_world.detach())

    if reg_mode == "sharp" or eps_value <= 0.0:
        return torch.sqrt(a2), torch.sqrt(b2), torch.sqrt(c2)
    if reg_mode == "plummer":
        eps2 = eps_world * eps_world
        return torch.sqrt(a2 + eps2), torch.sqrt(b2 + eps2), torch.sqrt(c2 + eps2)
    if reg_mode == "compact":
        inv_eps = 1.0 / eps_world
        return (
            eps_world * torch_g_compact(torch.sqrt(a2) * inv_eps),
            eps_world * torch_g_compact(torch.sqrt(b2) * inv_eps),
            eps_world * torch_g_compact(torch.sqrt(c2) * inv_eps),
        )
    raise ValueError(f"Unknown reg_mode: {reg_mode}")


def cuda_s_regularization(t: torch.Tensor) -> torch.Tensor:
    """S_regularization used by the CUDA point-normal kernel."""
    return torch.erf(t) - _TWO_OVER_SQRT_PI * t * torch.exp(-t * t)


# =============================================================================
# Shared triangular-soup contribution kernel (fp64 autograd)
# =============================================================================
def _triangle_sol_angle_chunk(
    triangles_var: torch.Tensor,
    queries_chunk: torch.Tensor,
    eps_world: torch.Tensor,
    eps_scalar: torch.Tensor,
    reg_mode: str,
) -> torch.Tensor:
    """Compute (chunk, N) sol_angle matrix, differentiable wrt everything."""
    v0 = triangles_var[:, 0, :][None, :, :]
    v1 = triangles_var[:, 1, :][None, :, :]
    v2 = triangles_var[:, 2, :][None, :, :]
    qbc = queries_chunk[:, None, :]

    a = v0 - qbc
    b = v1 - qbc
    c = v2 - qbc

    r_a, r_b, r_c = torch_regularized_edge_lengths(a, b, c, eps_world, reg_mode)
    inv_a = 1.0 / r_a
    inv_b = 1.0 / r_b
    inv_c = 1.0 / r_c

    cos_ab = torch.sum(a * b, dim=-1) * inv_a * inv_b
    cos_ac = torch.sum(a * c, dim=-1) * inv_a * inv_c
    cos_bc = torch.sum(b * c, dim=-1) * inv_b * inv_c

    cross_bc = torch.cross(b, c, dim=-1)
    det_norm = torch.sum(a * cross_bc, dim=-1) * inv_a * inv_b * inv_c
    div_norm = 1.0 + cos_ab + cos_ac + cos_bc

    if float(eps_scalar.detach()) > 0.0:
        u = (det_norm * det_norm + div_norm * div_norm) / (eps_scalar * eps_scalar)
        w = torch_angular_bump(u)
        shift = eps_scalar * (1.0 / math.sqrt(2.0)) * w
        det_norm = det_norm + shift
        div_norm = div_norm + shift

    return torch.atan2(det_norm, div_norm) * (1.0 / (2.0 * math.pi))


def _pn_contributions_chunk(
    points_bc: torch.Tensor,
    normals_bc: torch.Tensor,
    queries_chunk: torch.Tensor,
    eps_scalar: torch.Tensor,
    scale: float,
    s_reg_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Compute (chunk, N) point-normal contribution matrix."""
    qbc = queries_chunk[:, None, :]
    d = points_bc - qbc
    dist2 = torch.sum(d * d, dim=-1)

    inv_distance = torch.rsqrt(dist2 + 1e-30)
    inv_dist2 = inv_distance * inv_distance
    inv_dist3 = inv_dist2 * inv_distance

    dot_n_d = torch.sum(normals_bc * d, dim=-1)

    eps_value = float(eps_scalar.detach())
    if eps_value > 0.0:
        inv_epsilon_world = 1.0 / (eps_scalar * scale)
        near_limit_constant = _FOUR_OVER_3SQRT_PI * (inv_epsilon_world**3)
        distance = dist2 * inv_distance
        t = distance * inv_epsilon_world
        s_val = s_reg_fn(t)
        s_reg_term = s_val * inv_dist3
        s_over_dist3 = torch.where(
            t < 0.1,
            near_limit_constant,
            torch.where(t < 2.0, s_reg_term, inv_dist3),
        )
    else:
        s_over_dist3 = inv_dist3

    return dot_n_d * (1.0 / (4.0 * math.pi)) * s_over_dist3


# =============================================================================
# PyTorch autograd references — Payload gradient  (∂L/∂v_i)
# =============================================================================
def pytorch_triangle_payload_grads_chunked_64(
    triangles,
    queries,
    grad_output,
    payload,
    eps_tensor,
    scale,
    reg_mode,
    chunk_size=50,
):
    """Return ∂L/∂v_i for a triangle soup, shape (N, D)."""
    payload_var = payload.clone().detach().requires_grad_(True)
    num_queries = queries.shape[0]
    apply_bump = float(eps_tensor.detach()) > 0.0

    for i in range(0, num_queries, chunk_size):
        q_chunk = queries[i : i + chunk_size]
        g_chunk = grad_output[i : i + chunk_size]
        eps_world = eps_tensor * scale

        sol_angle = _triangle_sol_angle_chunk(
            triangles,
            q_chunk,
            eps_world,
            eps_tensor,
            reg_mode,
        )  # (chunk, N)
        w_V = sol_angle @ payload_var  # (chunk, D)
        loss = (w_V * g_chunk).sum()
        loss.backward()

    return payload_var.grad


def pytorch_mesh_payload_grads_chunked_64(
    vertices,
    indices,
    queries,
    grad_output,
    payload,
    eps_tensor,
    scale,
    reg_mode,
    chunk_size=50,
):
    """Return ∂L/∂v_i for a mesh (per-triangle output)."""
    payload_var = payload.clone().detach().requires_grad_(True)
    triangles = vertices[indices.long()]  # static gather
    num_queries = queries.shape[0]

    for i in range(0, num_queries, chunk_size):
        q_chunk = queries[i : i + chunk_size]
        g_chunk = grad_output[i : i + chunk_size]
        eps_world = eps_tensor * scale

        sol_angle = _triangle_sol_angle_chunk(
            triangles,
            q_chunk,
            eps_world,
            eps_tensor,
            reg_mode,
        )
        w_V = sol_angle @ payload_var
        loss = (w_V * g_chunk).sum()
        loss.backward()

    return payload_var.grad


def pytorch_pn_payload_grads_chunked_64(
    points,
    normals,
    queries,
    grad_output,
    payload,
    eps_tensor,
    scale,
    s_reg_fn,
    chunk_size=50,
):
    """Return ∂L/∂v_i for point-normal primitives, shape (N, D)."""
    payload_var = payload.clone().detach().requires_grad_(True)
    num_queries = queries.shape[0]
    p_bc = points[None, :, :]
    n_bc = normals[None, :, :]

    for i in range(0, num_queries, chunk_size):
        q_chunk = queries[i : i + chunk_size]
        g_chunk = grad_output[i : i + chunk_size]

        contributions = _pn_contributions_chunk(
            p_bc,
            n_bc,
            q_chunk,
            eps_tensor,
            scale,
            s_reg_fn,
        )  # (chunk, N)
        w_V = contributions @ payload_var  # (chunk, D)
        loss = (w_V * g_chunk).sum()
        loss.backward()

    return payload_var.grad


# =============================================================================
# PyTorch autograd references — Geometry gradient with payload
# =============================================================================
def pytorch_triangle_geometry_grads_with_payload_chunked_64(
    triangles,
    queries,
    grad_output,
    payload,
    eps_tensor,
    scale,
    reg_mode,
    chunk_size=50,
):
    """Return (∂L/∂triangles, ∂L/∂ε) for a triangle soup."""
    tri_var = triangles.clone().detach().requires_grad_(True)
    num_queries = queries.shape[0]

    for i in range(0, num_queries, chunk_size):
        q_chunk = queries[i : i + chunk_size]
        g_chunk = grad_output[i : i + chunk_size]
        eps_world = eps_tensor * scale

        sol_angle = _triangle_sol_angle_chunk(
            tri_var,
            q_chunk,
            eps_world,
            eps_tensor,
            reg_mode,
        )
        w_V = sol_angle @ payload
        loss = (w_V * g_chunk).sum()
        loss.backward()

    return tri_var.grad, eps_tensor.grad


def pytorch_mesh_geometry_grads_with_payload_chunked_64(
    vertices,
    indices,
    queries,
    grad_output,
    payload,
    eps_tensor,
    scale,
    reg_mode,
    chunk_size=50,
):
    """Return (∂L/∂vertices, ∂L/∂ε) for a shared-vertex mesh."""
    v_var = vertices.clone().detach().requires_grad_(True)
    num_queries = queries.shape[0]

    for i in range(0, num_queries, chunk_size):
        q_chunk = queries[i : i + chunk_size]
        g_chunk = grad_output[i : i + chunk_size]
        eps_world = eps_tensor * scale

        triangles_var = v_var[indices.long()]  # autograd flows through gather
        sol_angle = _triangle_sol_angle_chunk(
            triangles_var,
            q_chunk,
            eps_world,
            eps_tensor,
            reg_mode,
        )
        w_V = sol_angle @ payload
        loss = (w_V * g_chunk).sum()
        loss.backward()

    return v_var.grad, eps_tensor.grad


def pytorch_pn_geometry_grads_with_payload_chunked_64(
    points,
    normals,
    queries,
    grad_output,
    payload,
    eps_tensor,
    scale,
    s_reg_fn,
    chunk_size=50,
):
    """Return (∂L/∂points, ∂L/∂normals, ∂L/∂ε) for point-normal."""
    p_var = points.clone().detach().requires_grad_(True)
    n_var = normals.clone().detach().requires_grad_(True)
    num_queries = queries.shape[0]

    for i in range(0, num_queries, chunk_size):
        q_chunk = queries[i : i + chunk_size]
        g_chunk = grad_output[i : i + chunk_size]

        contributions = _pn_contributions_chunk(
            p_var[None, :, :],
            n_var[None, :, :],
            q_chunk,
            eps_tensor,
            scale,
            s_reg_fn,
        )
        w_V = contributions @ payload
        loss = (w_V * g_chunk).sum()
        loss.backward()

    return p_var.grad, n_var.grad, eps_tensor.grad


# =============================================================================
# PyTorch autograd references — Query gradient with payload
# =============================================================================
def pytorch_triangle_query_grads_with_payload_chunked_64(
    triangles,
    queries,
    grad_output,
    payload,
    eps_tensor,
    scale,
    reg_mode,
    chunk_size=50,
):
    """Return ∂L/∂queries for a triangle soup, shape (M, 3)."""
    q_var = queries.clone().detach().requires_grad_(True)
    num_queries = queries.shape[0]

    for i in range(0, num_queries, chunk_size):
        q_chunk = q_var[i : i + chunk_size]
        g_chunk = grad_output[i : i + chunk_size]
        eps_world = eps_tensor * scale

        sol_angle = _triangle_sol_angle_chunk(
            triangles,
            q_chunk,
            eps_world,
            eps_tensor,
            reg_mode,
        )
        w_V = sol_angle @ payload
        loss = (w_V * g_chunk).sum()
        loss.backward()

    return q_var.grad


def pytorch_mesh_query_grads_with_payload_chunked_64(
    vertices,
    indices,
    queries,
    grad_output,
    payload,
    eps_tensor,
    scale,
    reg_mode,
    chunk_size=50,
):
    """Return ∂L/∂queries for a shared-vertex mesh."""
    q_var = queries.clone().detach().requires_grad_(True)
    triangles = vertices[indices.long()]
    num_queries = queries.shape[0]

    for i in range(0, num_queries, chunk_size):
        q_chunk = q_var[i : i + chunk_size]
        g_chunk = grad_output[i : i + chunk_size]
        eps_world = eps_tensor * scale

        sol_angle = _triangle_sol_angle_chunk(
            triangles,
            q_chunk,
            eps_world,
            eps_tensor,
            reg_mode,
        )
        w_V = sol_angle @ payload
        loss = (w_V * g_chunk).sum()
        loss.backward()

    return q_var.grad


def pytorch_pn_query_grads_with_payload_chunked_64(
    points,
    normals,
    queries,
    grad_output,
    payload,
    eps_tensor,
    scale,
    s_reg_fn,
    chunk_size=50,
):
    """Return ∂L/∂queries for point-normal, shape (M, 3)."""
    q_var = queries.clone().detach().requires_grad_(True)
    p_bc = points[None, :, :]
    n_bc = normals[None, :, :]
    num_queries = queries.shape[0]

    for i in range(0, num_queries, chunk_size):
        q_chunk = q_var[i : i + chunk_size]
        g_chunk = grad_output[i : i + chunk_size]

        contributions = _pn_contributions_chunk(
            p_bc,
            n_bc,
            q_chunk,
            eps_tensor,
            scale,
            s_reg_fn,
        )
        w_V = contributions @ payload
        loss = (w_V * g_chunk).sum()
        loss.backward()

    return q_var.grad


# =============================================================================
# CUDA call wrappers
# =============================================================================
def _run_pn_payload_grad(points, normals, grad_output, queries, epsilon, use_kahan):
    p = _to_cuda(points)
    n = _to_cuda(normals)
    q = _to_cuda(queries)
    g = _to_cuda(grad_output)
    N = points.shape[0]
    D = grad_output.shape[1]
    out = torch.empty((N, D), device="cuda:0", dtype=torch.float32)
    winder.brute_force_payload_gradients_point_normal(
        g,
        p,
        n,
        q,
        out,
        float(epsilon),
        torch.cuda.current_stream().cuda_stream,
        use_kahan,
    )
    torch.cuda.synchronize()
    return out


def _run_tri_payload_grad(triangles, grad_output, queries, epsilon, use_kahan):
    t = _to_cuda(triangles)
    q = _to_cuda(queries)
    g = _to_cuda(grad_output)
    N = triangles.shape[0]
    D = grad_output.shape[1]
    out = torch.empty((N, D), device="cuda:0", dtype=torch.float32)
    winder.brute_force_payload_gradients_triangle_soup(
        g,
        t,
        q,
        out,
        float(epsilon),
        torch.cuda.current_stream().cuda_stream,
        use_kahan,
    )
    torch.cuda.synchronize()
    return out


def _run_mesh_payload_grad(vertices, indices, grad_output, queries, epsilon, use_kahan):
    v = _to_cuda(vertices)
    idx = _to_cuda(indices, torch.uint32)
    q = _to_cuda(queries)
    g = _to_cuda(grad_output)
    N = indices.shape[0]
    D = grad_output.shape[1]
    out = torch.empty((N, D), device="cuda:0", dtype=torch.float32)
    winder.brute_force_payload_gradients_mesh(
        g,
        v,
        idx,
        q,
        out,
        float(epsilon),
        torch.cuda.current_stream().cuda_stream,
        use_kahan,
    )
    torch.cuda.synchronize()
    return out


def _run_pn_geometry_grad_with_payload(
    points,
    normals,
    payload,
    grad_output,
    queries,
    epsilon,
    use_kahan,
):
    p = _to_cuda(points)
    n = _to_cuda(normals)
    pl = _to_cuda(payload)
    q = _to_cuda(queries)
    g = _to_cuda(grad_output)
    N = points.shape[0]
    out = torch.empty((N, 2, 3), device="cuda:0", dtype=torch.float32)
    eps_grad = torch.zeros(1, device="cuda:0", dtype=torch.float32)
    winder.brute_force_gradients_with_payload_point_normal(
        g,
        p,
        n,
        pl,
        q,
        out,
        eps_grad,
        float(epsilon),
        torch.cuda.current_stream().cuda_stream,
        use_kahan,
    )
    torch.cuda.synchronize()
    return out, eps_grad


def _run_tri_geometry_grad_with_payload(
    triangles,
    payload,
    grad_output,
    queries,
    epsilon,
    use_kahan,
):
    t = _to_cuda(triangles)
    pl = _to_cuda(payload)
    q = _to_cuda(queries)
    g = _to_cuda(grad_output)
    N = triangles.shape[0]
    out = torch.empty((N, 3, 3), device="cuda:0", dtype=torch.float32)
    eps_grad = torch.zeros(1, device="cuda:0", dtype=torch.float32)
    winder.brute_force_gradients_with_payload_triangle_soup(
        g,
        t,
        pl,
        q,
        out,
        eps_grad,
        float(epsilon),
        torch.cuda.current_stream().cuda_stream,
        use_kahan,
    )
    torch.cuda.synchronize()
    return out, eps_grad


def _run_mesh_geometry_grad_with_payload(
    vertices,
    indices,
    payload,
    grad_output,
    queries,
    epsilon,
    use_kahan,
):
    v = _to_cuda(vertices)
    idx = _to_cuda(indices, torch.uint32)
    pl = _to_cuda(payload)
    q = _to_cuda(queries)
    g = _to_cuda(grad_output)
    K = vertices.shape[0]
    out = torch.empty((K, 3), device="cuda:0", dtype=torch.float32)
    eps_grad = torch.zeros(1, device="cuda:0", dtype=torch.float32)
    winder.brute_force_gradients_with_payload_mesh(
        g,
        v,
        idx,
        pl,
        q,
        out,
        eps_grad,
        float(epsilon),
        torch.cuda.current_stream().cuda_stream,
        use_kahan,
    )
    torch.cuda.synchronize()
    return out, eps_grad


def _run_pn_query_grad_with_payload(
    points,
    normals,
    payload,
    grad_output,
    queries,
    epsilon,
    use_kahan,
):
    p = _to_cuda(points)
    n = _to_cuda(normals)
    pl = _to_cuda(payload)
    q = _to_cuda(queries)
    g = _to_cuda(grad_output)
    M = queries.shape[0]
    out = torch.empty((M, 3), device="cuda:0", dtype=torch.float32)
    winder.brute_force_query_gradients_with_payload_point_normal(
        g,
        p,
        n,
        pl,
        q,
        out,
        float(epsilon),
        torch.cuda.current_stream().cuda_stream,
        use_kahan,
    )
    torch.cuda.synchronize()
    return out


def _run_tri_query_grad_with_payload(
    triangles,
    payload,
    grad_output,
    queries,
    epsilon,
    use_kahan,
):
    t = _to_cuda(triangles)
    pl = _to_cuda(payload)
    q = _to_cuda(queries)
    g = _to_cuda(grad_output)
    M = queries.shape[0]
    out = torch.empty((M, 3), device="cuda:0", dtype=torch.float32)
    winder.brute_force_query_gradients_with_payload_triangle_soup(
        g,
        t,
        pl,
        q,
        out,
        float(epsilon),
        torch.cuda.current_stream().cuda_stream,
        use_kahan,
    )
    torch.cuda.synchronize()
    return out


def _run_mesh_query_grad_with_payload(
    vertices,
    indices,
    payload,
    grad_output,
    queries,
    epsilon,
    use_kahan,
):
    v = _to_cuda(vertices)
    idx = _to_cuda(indices, torch.uint32)
    pl = _to_cuda(payload)
    q = _to_cuda(queries)
    g = _to_cuda(grad_output)
    M = queries.shape[0]
    out = torch.empty((M, 3), device="cuda:0", dtype=torch.float32)
    winder.brute_force_query_gradients_with_payload_mesh(
        g,
        v,
        idx,
        pl,
        q,
        out,
        float(epsilon),
        torch.cuda.current_stream().cuda_stream,
        use_kahan,
    )
    torch.cuda.synchronize()
    return out


# Scalar (non-payload) call wrappers, used for the D=1 v=ones sanity check.
def _run_pn_geometry_grad_scalar(points, normals, grad_output, queries, epsilon):
    p = _to_cuda(points)
    n = _to_cuda(normals)
    q = _to_cuda(queries)
    g = _to_cuda(grad_output)
    N = points.shape[0]
    out = torch.empty((N, 2, 3), device="cuda:0", dtype=torch.float32)
    eps_grad = torch.zeros(1, device="cuda:0", dtype=torch.float32)
    winder.brute_force_gradients_point_normal(
        g,
        p,
        n,
        q,
        out,
        eps_grad,
        float(epsilon),
        torch.cuda.current_stream().cuda_stream,
        False,
    )
    torch.cuda.synchronize()
    return out, eps_grad


def _run_tri_geometry_grad_scalar(triangles, grad_output, queries, epsilon):
    t = _to_cuda(triangles)
    q = _to_cuda(queries)
    g = _to_cuda(grad_output)
    N = triangles.shape[0]
    out = torch.empty((N, 3, 3), device="cuda:0", dtype=torch.float32)
    eps_grad = torch.zeros(1, device="cuda:0", dtype=torch.float32)
    winder.brute_force_gradients_triangle_soup(
        g,
        t,
        q,
        out,
        eps_grad,
        float(epsilon),
        torch.cuda.current_stream().cuda_stream,
        False,
    )
    torch.cuda.synchronize()
    return out, eps_grad


def _run_mesh_geometry_grad_scalar(vertices, indices, grad_output, queries, epsilon):
    v = _to_cuda(vertices)
    idx = _to_cuda(indices, torch.uint32)
    q = _to_cuda(queries)
    g = _to_cuda(grad_output)
    K = vertices.shape[0]
    out = torch.empty((K, 3), device="cuda:0", dtype=torch.float32)
    eps_grad = torch.zeros(1, device="cuda:0", dtype=torch.float32)
    winder.brute_force_gradients_mesh(
        g,
        v,
        idx,
        q,
        out,
        eps_grad,
        float(epsilon),
        torch.cuda.current_stream().cuda_stream,
        False,
    )
    torch.cuda.synchronize()
    return out, eps_grad


def _run_pn_query_grad_scalar(points, normals, grad_output, queries, epsilon):
    p = _to_cuda(points)
    n = _to_cuda(normals)
    q = _to_cuda(queries)
    g = _to_cuda(grad_output)
    M = queries.shape[0]
    out = torch.empty((M, 3), device="cuda:0", dtype=torch.float32)
    winder.brute_force_query_gradients_point_normal(
        g,
        p,
        n,
        q,
        out,
        float(epsilon),
        torch.cuda.current_stream().cuda_stream,
        True,
    )
    torch.cuda.synchronize()
    return out


def _run_tri_query_grad_scalar(triangles, grad_output, queries, epsilon):
    t = _to_cuda(triangles)
    q = _to_cuda(queries)
    g = _to_cuda(grad_output)
    M = queries.shape[0]
    out = torch.empty((M, 3), device="cuda:0", dtype=torch.float32)
    winder.brute_force_query_gradients_triangle_soup(
        g,
        t,
        q,
        out,
        float(epsilon),
        torch.cuda.current_stream().cuda_stream,
        True,
    )
    torch.cuda.synchronize()
    return out


def _run_mesh_query_grad_scalar(vertices, indices, grad_output, queries, epsilon):
    v = _to_cuda(vertices)
    idx = _to_cuda(indices, torch.uint32)
    q = _to_cuda(queries)
    g = _to_cuda(grad_output)
    M = queries.shape[0]
    out = torch.empty((M, 3), device="cuda:0", dtype=torch.float32)
    winder.brute_force_query_gradients_mesh(
        g,
        v,
        idx,
        q,
        out,
        float(epsilon),
        torch.cuda.current_stream().cuda_stream,
        True,
    )
    torch.cuda.synchronize()
    return out


# =============================================================================
# Validators
# =============================================================================
def validate_payload_field(
    cuda: torch.Tensor,
    ref: torch.Tensor,
    label: str,
    rms_tol: float = 1e-3,
    cos_tol: float = 0.995,
) -> bool:
    """Compare payload gradient tensors of any last dim (N, D)."""
    cuda_flat = cuda.detach().reshape(-1).float().cpu()
    ref_flat = ref.detach().reshape(-1).float().cpu()
    diff = (cuda_flat - ref_flat).abs()

    rms = float(torch.sqrt((diff**2).mean()))
    mx = float(diff.max())
    ref_norm = float(torch.linalg.norm(ref_flat))
    cuda_norm = float(torch.linalg.norm(cuda_flat))
    if ref_norm > 1e-30 and cuda_norm > 1e-30:
        cos = float(torch.dot(cuda_flat, ref_flat)) / (ref_norm * cuda_norm)
    else:
        cos = 1.0
    rel_rms = rms / (ref_norm / math.sqrt(ref_flat.numel()) + 1e-30)

    passed = rel_rms < rms_tol and cos > cos_tol
    tag = _ok("PASS") if passed else _bad("FAIL")
    print(
        f"  [{tag}] {label:<55}  rms={rms:.2e}  max={mx:.2e}  "
        f"cos={cos:.6f}  rel_rms={rel_rms:.2e}"
    )
    return passed


def validate_vector_field(
    cuda: torch.Tensor,
    ref: torch.Tensor,
    label: str,
    rel_norm_tol: float = 1e-3,
    cos_tol: float = 0.995,
) -> bool:
    """Compare 3-vector fields (last dim = 3); used for geometry/query grads."""
    cuda_flat = cuda.detach().reshape(-1).float().cpu()
    ref_flat = ref.detach().reshape(-1).float().cpu()
    diff = (cuda_flat - ref_flat).abs()
    rms = float(torch.sqrt((diff**2).mean()))
    mx = float(diff.max())
    ref_norm = float(torch.linalg.norm(ref_flat))
    cuda_norm = float(torch.linalg.norm(cuda_flat))
    if ref_norm > 1e-30 and cuda_norm > 1e-30:
        cos = float(torch.dot(cuda_flat, ref_flat)) / (ref_norm * cuda_norm)
    else:
        cos = 1.0
    rel_norm = float(torch.linalg.norm(cuda_flat - ref_flat)) / (ref_norm + 1e-30)

    passed = rel_norm < rel_norm_tol and cos > cos_tol
    tag = _ok("PASS") if passed else _bad("FAIL")
    print(
        f"  [{tag}] {label:<55}  rel_norm={rel_norm:.2e}  cos={cos:.6f}  max={mx:.2e}"
    )
    return passed


def validate_scalar_gradient(
    cuda: torch.Tensor,
    ref: torch.Tensor,
    label: str,
    abs_tol: float = 1e-5,
    rel_tol: float = 1e-2,
) -> bool:
    c = float(cuda.detach().cpu().reshape(-1)[0])
    r = float(ref.detach().cpu().reshape(-1)[0])
    abs_err = abs(c - r)
    rel_err = abs_err / max(abs(r), 1e-6)
    passed = abs_err < abs_tol or rel_err < rel_tol
    tag = _ok("PASS") if passed else _bad("FAIL")
    print(
        f"  [{tag}] {label:<55}  cuda={c:+.4e}  ref={r:+.4e}  "
        f"abs={abs_err:.2e}  rel={rel_err:.2e}"
    )
    return passed


# =============================================================================
# Test result bookkeeping
# =============================================================================
@dataclass
class TestResult:
    name: str
    passed: bool


def _record(results: list, name: str, passed: bool) -> None:
    results.append(TestResult(name=name, passed=passed))


# =============================================================================
# Per-geometry gradient test
# =============================================================================
def test_payload_gradients_pn(
    points,
    normals,
    queries,
    epsilon,
    reg_mode,
    D,
    seed,
    results,
    use_kahan=False,
) -> bool:
    """Full payload, geometry, query gradient test for point-normal."""
    rng = np.random.default_rng(seed + D * 31 + int(epsilon * 1000))
    N = len(points)
    payload = rng.standard_normal((N, D)).astype(np.float32)
    grad_output = rng.standard_normal((queries.shape[0], D)).astype(np.float32)
    scale = _scene_scale(points)

    cuda_payload_grad = _run_pn_payload_grad(
        points,
        normals,
        grad_output,
        queries,
        epsilon,
        use_kahan,
    )
    cuda_geom_grad, cuda_eps_grad = _run_pn_geometry_grad_with_payload(
        points,
        normals,
        payload,
        grad_output,
        queries,
        epsilon,
        use_kahan,
    )
    cuda_query_grad = _run_pn_query_grad_with_payload(
        points,
        normals,
        payload,
        grad_output,
        queries,
        epsilon,
        use_kahan,
    )

    p64, n64, q64 = _t(points), _t(normals), _t(queries)
    pl64, g64 = _t(payload), _t(grad_output)

    eps_t1 = torch.tensor(epsilon, dtype=torch.float64, requires_grad=True, device="cuda")
    ref_payload = pytorch_pn_payload_grads_chunked_64(
        p64,
        n64,
        q64,
        g64,
        pl64,
        eps_t1,
        scale,
        cuda_s_regularization,
    )
    eps_t2 = torch.tensor(epsilon, dtype=torch.float64, requires_grad=True, device="cuda")
    ref_p, ref_n, ref_eps = pytorch_pn_geometry_grads_with_payload_chunked_64(
        p64,
        n64,
        q64,
        g64,
        pl64,
        eps_t2,
        scale,
        cuda_s_regularization,
    )
    eps_t3 = torch.tensor(epsilon, dtype=torch.float64, requires_grad=True, device="cuda")
    ref_query = pytorch_pn_query_grads_with_payload_chunked_64(
        p64,
        n64,
        q64,
        g64,
        pl64,
        eps_t3,
        scale,
        cuda_s_regularization,
    )

    lbl = f"PN  eps={epsilon:g} reg={reg_mode} D={D}"
    ok_payload = validate_payload_field(
        cuda_payload_grad,
        ref_payload.to(torch.float32),
        f"payload_grad[{lbl}]",
    )
    ok_n = validate_vector_field(
        cuda_geom_grad[:, 0, :],
        ref_n.to(torch.float32),
        f"grad_n[{lbl}]",
    )
    ok_p = validate_vector_field(
        cuda_geom_grad[:, 1, :],
        ref_p.to(torch.float32),
        f"grad_p[{lbl}]",
    )
    ok_eps = validate_scalar_gradient(
        cuda_eps_grad,
        _as_eps_tensor(ref_eps),
        f"grad_eps[{lbl}]",
    )
    ok_q = validate_vector_field(
        cuda_query_grad,
        ref_query.to(torch.float32),
        f"query_grad[{lbl}]",
    )

    passed = all([ok_payload, ok_n, ok_p, ok_eps, ok_q])
    _record(results, f"PN_full_{lbl}", passed)
    return passed


def test_payload_gradients_triangle(
    triangles,
    queries,
    epsilon,
    reg_mode,
    D,
    seed,
    results,
    use_kahan=False,
) -> bool:
    """Full payload, geometry, query gradient test for triangle soup."""
    rng = np.random.default_rng(seed + D * 31 + int(epsilon * 1000) + 1)
    N = len(triangles)
    payload = rng.standard_normal((N, D)).astype(np.float32)
    grad_output = rng.standard_normal((queries.shape[0], D)).astype(np.float32)
    scale = _scene_scale(triangles.reshape(-1, 3))

    cuda_payload_grad = _run_tri_payload_grad(
        triangles,
        grad_output,
        queries,
        epsilon,
        use_kahan,
    )
    cuda_geom_grad, cuda_eps_grad = _run_tri_geometry_grad_with_payload(
        triangles,
        payload,
        grad_output,
        queries,
        epsilon,
        use_kahan,
    )
    cuda_query_grad = _run_tri_query_grad_with_payload(
        triangles,
        payload,
        grad_output,
        queries,
        epsilon,
        use_kahan,
    )

    t64, q64 = _t(triangles), _t(queries)
    pl64, g64 = _t(payload), _t(grad_output)

    eps_t1 = torch.tensor(epsilon, dtype=torch.float64, requires_grad=True, device="cuda")
    ref_payload = pytorch_triangle_payload_grads_chunked_64(
        t64,
        q64,
        g64,
        pl64,
        eps_t1,
        scale,
        reg_mode,
    )
    eps_t2 = torch.tensor(epsilon, dtype=torch.float64, requires_grad=True, device="cuda")
    ref_vert, ref_eps = pytorch_triangle_geometry_grads_with_payload_chunked_64(
        t64,
        q64,
        g64,
        pl64,
        eps_t2,
        scale,
        reg_mode,
    )
    eps_t3 = torch.tensor(epsilon, dtype=torch.float64, requires_grad=True, device="cuda")
    ref_query = pytorch_triangle_query_grads_with_payload_chunked_64(
        t64,
        q64,
        g64,
        pl64,
        eps_t3,
        scale,
        reg_mode,
    )

    lbl = f"Tri eps={epsilon:g} reg={reg_mode} D={D}"
    ok_payload = validate_payload_field(
        cuda_payload_grad,
        ref_payload.to(torch.float32),
        f"payload_grad[{lbl}]",
    )
    ok_v = validate_vector_field(
        cuda_geom_grad.reshape(-1, 3),
        ref_vert.reshape(-1, 3).to(torch.float32),
        f"grad_vert[{lbl}]",
    )
    ok_eps = validate_scalar_gradient(
        cuda_eps_grad,
        _as_eps_tensor(ref_eps),
        f"grad_eps[{lbl}]",
    )
    ok_q = validate_vector_field(
        cuda_query_grad,
        ref_query.to(torch.float32),
        f"query_grad[{lbl}]",
    )
    passed = all([ok_payload, ok_v, ok_eps, ok_q])
    _record(results, f"Tri_full_{lbl}", passed)
    return passed


def test_payload_gradients_mesh(
    vertices,
    indices,
    queries,
    epsilon,
    reg_mode,
    D,
    seed,
    results,
    use_kahan=False,
) -> bool:
    """Full payload, geometry, query gradient test for mesh."""
    rng = np.random.default_rng(seed + D * 31 + int(epsilon * 1000) + 2)
    N = len(indices)
    payload = rng.standard_normal((N, D)).astype(np.float32)
    grad_output = rng.standard_normal((queries.shape[0], D)).astype(np.float32)
    scale = _scene_scale(vertices[indices].reshape(-1, 3))

    cuda_payload_grad = _run_mesh_payload_grad(
        vertices,
        indices,
        grad_output,
        queries,
        epsilon,
        use_kahan,
    )
    cuda_geom_grad, cuda_eps_grad = _run_mesh_geometry_grad_with_payload(
        vertices,
        indices,
        payload,
        grad_output,
        queries,
        epsilon,
        use_kahan,
    )
    cuda_query_grad = _run_mesh_query_grad_with_payload(
        vertices,
        indices,
        payload,
        grad_output,
        queries,
        epsilon,
        use_kahan,
    )

    v64, i64, q64 = _t(vertices), torch.from_numpy(indices.astype(np.int64)).cuda(), _t(queries)
    pl64, g64 = _t(payload), _t(grad_output)

    eps_t1 = torch.tensor(epsilon, dtype=torch.float64, requires_grad=True, device="cuda")
    ref_payload = pytorch_mesh_payload_grads_chunked_64(
        v64,
        i64,
        q64,
        g64,
        pl64,
        eps_t1,
        scale,
        reg_mode,
    )
    eps_t2 = torch.tensor(epsilon, dtype=torch.float64, requires_grad=True, device="cuda")
    ref_vert, ref_eps = pytorch_mesh_geometry_grads_with_payload_chunked_64(
        v64,
        i64,
        q64,
        g64,
        pl64,
        eps_t2,
        scale,
        reg_mode,
    )
    eps_t3 = torch.tensor(epsilon, dtype=torch.float64, requires_grad=True, device="cuda")
    ref_query = pytorch_mesh_query_grads_with_payload_chunked_64(
        v64,
        i64,
        q64,
        g64,
        pl64,
        eps_t3,
        scale,
        reg_mode,
    )

    lbl = f"Mesh eps={epsilon:g} reg={reg_mode} D={D}"
    ok_payload = validate_payload_field(
        cuda_payload_grad,
        ref_payload.to(torch.float32),
        f"payload_grad[{lbl}]",
    )
    ok_v = validate_vector_field(
        cuda_geom_grad,
        ref_vert.to(torch.float32),
        f"grad_vert[{lbl}]",
    )
    ok_eps = validate_scalar_gradient(
        cuda_eps_grad,
        _as_eps_tensor(ref_eps),
        f"grad_eps[{lbl}]",
    )
    ok_q = validate_vector_field(
        cuda_query_grad,
        ref_query.to(torch.float32),
        f"query_grad[{lbl}]",
    )
    passed = all([ok_payload, ok_v, ok_eps, ok_q])
    _record(results, f"Mesh_full_{lbl}", passed)
    return passed


# =============================================================================
# Sanity: D=1, v=ones reduces to the scalar gradients
# =============================================================================
def test_d1_ones_scalar_reduction(
    vertices,
    indices,
    queries,
    epsilon_values,
    results,
) -> None:
    print(_hdr("\nTest: D=1, v=ones reduces to scalar gradients"))

    points, normals, _ = mesh_to_point_surfels(vertices, indices)
    triangles = vertices[indices]
    rng = np.random.default_rng(11)
    N_pn = len(points)
    N_tri = len(triangles)
    M = len(queries)

    g_scalar = rng.standard_normal(M).astype(np.float32)
    g_payload = g_scalar[:, None].copy()
    payload_ones_pn = np.ones((N_pn, 1), np.float32)
    payload_ones_tri = np.ones((N_tri, 1), np.float32)

    for eps in epsilon_values:
        # PN geometry gradient
        cuda_pn_payload, cuda_pn_eps = _run_pn_geometry_grad_with_payload(
            points,
            normals,
            payload_ones_pn,
            g_payload,
            queries,
            eps,
            False,
        )
        ref_pn_scalar, ref_pn_eps_scalar = _run_pn_geometry_grad_scalar(
            points,
            normals,
            g_scalar,
            queries,
            eps,
        )
        ok = validate_vector_field(
            cuda_pn_payload.reshape(-1, 3),
            ref_pn_scalar.reshape(-1, 3),
            f"PN geom grad D=1 v=ones == scalar (eps={eps:g})",
            rel_norm_tol=1e-5,
            cos_tol=0.99999,
        )
        _record(results, f"sanity_pn_geom_eps{eps:g}", ok)

        cuda_pn_q = _run_pn_query_grad_with_payload(
            points,
            normals,
            payload_ones_pn,
            g_payload,
            queries,
            eps,
            True,
        )
        ref_pn_q = _run_pn_query_grad_scalar(
            points,
            normals,
            g_scalar,
            queries,
            eps,
        )
        ok = validate_vector_field(
            cuda_pn_q,
            ref_pn_q,
            f"PN query grad D=1 v=ones == scalar (eps={eps:g})",
            rel_norm_tol=1e-5,
            cos_tol=0.99999,
        )
        _record(results, f"sanity_pn_query_eps{eps:g}", ok)

        # Triangle soup
        cuda_tri_payload, _ = _run_tri_geometry_grad_with_payload(
            triangles,
            payload_ones_tri,
            g_payload,
            queries,
            eps,
            False,
        )
        ref_tri_scalar, _ = _run_tri_geometry_grad_scalar(
            triangles,
            g_scalar,
            queries,
            eps,
        )
        ok = validate_vector_field(
            cuda_tri_payload.reshape(-1, 3),
            ref_tri_scalar.reshape(-1, 3),
            f"Tri geom grad D=1 v=ones == scalar (eps={eps:g})",
            rel_norm_tol=1e-5,
            cos_tol=0.99999,
        )
        _record(results, f"sanity_tri_geom_eps{eps:g}", ok)

        # Mesh
        cuda_mesh_payload, _ = _run_mesh_geometry_grad_with_payload(
            vertices,
            indices,
            payload_ones_tri,
            g_payload,
            queries,
            eps,
            False,
        )
        ref_mesh_scalar, _ = _run_mesh_geometry_grad_scalar(
            vertices,
            indices,
            g_scalar,
            queries,
            eps,
        )
        ok = validate_vector_field(
            cuda_mesh_payload,
            ref_mesh_scalar,
            f"Mesh geom grad D=1 v=ones == scalar (eps={eps:g})",
            rel_norm_tol=1e-5,
            cos_tol=0.99999,
        )
        _record(results, f"sanity_mesh_geom_eps{eps:g}", ok)


# =============================================================================
# Sanity: Σ_i ∂L/∂v_i == Σ_j g_j · W(q_j)   (payload-grad sum identity)
# =============================================================================
def test_payload_grad_sum_identity(
    vertices,
    indices,
    queries,
    epsilon_values,
    results,
) -> None:
    print(_hdr("\nTest: Σ_i ∂L/∂v_i == Σ_j g_j · W(q_j)"))

    points, normals, _ = mesh_to_point_surfels(vertices, indices)
    triangles = vertices[indices]
    rng = np.random.default_rng(23)
    M = len(queries)
    N_tri = len(triangles)
    N_pn = len(points)

    g_scalar = rng.standard_normal(M).astype(np.float32)
    g_payload = g_scalar[:, None].copy()

    for eps in epsilon_values:
        # Compute W via forward pass (existing scalar API, well-tested)
        W_pn = _to_numpy(
            winder.brute_force_winding_numbers_point_normal(
                _to_cuda(points),
                _to_cuda(normals),
                _to_cuda(queries),
                torch.empty(M, device="cuda:0", dtype=torch.float32),
                float(eps),
                torch.cuda.current_stream().cuda_stream,
                True,
            )
            if False
            else torch.from_numpy(
                np.zeros(M, dtype=np.float32)
            ).cuda()  # placeholder; replaced below
        )
        # Simpler: do it directly
        p = _to_cuda(points)
        n = _to_cuda(normals)
        q = _to_cuda(queries)
        W_pn_buf = torch.empty(M, device="cuda:0", dtype=torch.float32)
        winder.brute_force_winding_numbers_point_normal(
            p,
            n,
            q,
            W_pn_buf,
            float(eps),
            torch.cuda.current_stream().cuda_stream,
            True,
        )
        torch.cuda.synchronize()
        W_pn_np = _to_numpy(W_pn_buf)
        expected_pn = float(np.sum(g_scalar * W_pn_np))

        # Payload gradient
        ones_pn = np.ones((N_pn, 1), np.float32)
        cuda_pn_grad = _run_pn_payload_grad(
            points,
            normals,
            g_payload,
            queries,
            eps,
            False,
        )
        actual_pn = float(_to_numpy(cuda_pn_grad).sum())
        ok = abs(actual_pn - expected_pn) < max(abs(expected_pn) * 1e-4, 1e-5)
        tag = _ok("PASS") if ok else _bad("FAIL")
        print(
            f"  [{tag}] PN  sum(grad_v) vs sum(g*W)  eps={eps:g}  "
            f"actual={actual_pn:+.4e}  expected={expected_pn:+.4e}"
        )
        _record(results, f"sum_identity_pn_eps{eps:g}", ok)

        # Triangle
        t = _to_cuda(triangles)
        W_tri_buf = torch.empty(M, device="cuda:0", dtype=torch.float32)
        winder.brute_force_winding_numbers_triangle_soup(
            t,
            q,
            W_tri_buf,
            float(eps),
            torch.cuda.current_stream().cuda_stream,
            True,
        )
        torch.cuda.synchronize()
        W_tri_np = _to_numpy(W_tri_buf)
        expected_tri = float(np.sum(g_scalar * W_tri_np))

        ones_tri = np.ones((N_tri, 1), np.float32)
        cuda_tri_grad = _run_tri_payload_grad(
            triangles,
            g_payload,
            queries,
            eps,
            False,
        )
        actual_tri = float(_to_numpy(cuda_tri_grad).sum())
        ok = abs(actual_tri - expected_tri) < max(abs(expected_tri) * 1e-4, 1e-5)
        tag = _ok("PASS") if ok else _bad("FAIL")
        print(
            f"  [{tag}] Tri sum(grad_v) vs sum(g*W)  eps={eps:g}  "
            f"actual={actual_tri:+.4e}  expected={expected_tri:+.4e}"
        )
        _record(results, f"sum_identity_tri_eps{eps:g}", ok)


# =============================================================================
# Linearity in payload  (∂L/∂ξ is linear in v)
# =============================================================================
def test_linearity_in_v(
    vertices,
    indices,
    queries,
    epsilon,
    results,
) -> None:
    """Geometry and query gradients are linear in v (for fixed g)."""
    print(_hdr("\nTest: geometry and query gradients are linear in v"))

    points, normals, _ = mesh_to_point_surfels(vertices, indices)
    triangles = vertices[indices]
    rng = np.random.default_rng(71)
    N = len(triangles)
    M = len(queries)
    D = 8

    v1 = rng.standard_normal((N, D)).astype(np.float32)
    v2 = rng.standard_normal((N, D)).astype(np.float32)
    g = rng.standard_normal((M, D)).astype(np.float32)
    a, b = 1.7, -0.83
    vc = (a * v1 + b * v2).astype(np.float32)

    # Triangle mesh
    c1, e1 = _run_tri_geometry_grad_with_payload(
        triangles,
        v1,
        g,
        queries,
        epsilon,
        False,
    )
    c2, e2 = _run_tri_geometry_grad_with_payload(
        triangles,
        v2,
        g,
        queries,
        epsilon,
        False,
    )
    cc, ec = _run_tri_geometry_grad_with_payload(
        triangles,
        vc,
        g,
        queries,
        epsilon,
        False,
    )
    linear = a * c1 + b * c2
    ok = validate_vector_field(
        cc.reshape(-1, 3),
        linear.reshape(-1, 3),
        "Tri geom_grad(vc) == a·grad(v1) + b·grad(v2)",
        rel_norm_tol=1e-4,
        cos_tol=0.9999,
    )
    _record(results, "linearity_tri_geom_v", ok)

    q1 = _run_tri_query_grad_with_payload(
        triangles,
        v1,
        g,
        queries,
        epsilon,
        True,
    )
    q2 = _run_tri_query_grad_with_payload(
        triangles,
        v2,
        g,
        queries,
        epsilon,
        True,
    )
    qc = _run_tri_query_grad_with_payload(
        triangles,
        vc,
        g,
        queries,
        epsilon,
        True,
    )
    ok = validate_vector_field(
        qc,
        a * q1 + b * q2,
        "Tri query_grad(vc) == a·q1 + b·q2",
        rel_norm_tol=1e-4,
        cos_tol=0.9999,
    )
    _record(results, "linearity_tri_query_v", ok)

    # PN
    Npn = len(points)
    pv1 = rng.standard_normal((Npn, D)).astype(np.float32)
    pv2 = rng.standard_normal((Npn, D)).astype(np.float32)
    pvc = (a * pv1 + b * pv2).astype(np.float32)
    cp1, _ = _run_pn_geometry_grad_with_payload(
        points,
        normals,
        pv1,
        g,
        queries,
        epsilon,
        False,
    )
    cp2, _ = _run_pn_geometry_grad_with_payload(
        points,
        normals,
        pv2,
        g,
        queries,
        epsilon,
        False,
    )
    cpc, _ = _run_pn_geometry_grad_with_payload(
        points,
        normals,
        pvc,
        g,
        queries,
        epsilon,
        False,
    )
    ok = validate_vector_field(
        cpc.reshape(-1, 3),
        (a * cp1 + b * cp2).reshape(-1, 3),
        "PN geom_grad(vc) == a·grad(v1) + b·grad(v2)",
        rel_norm_tol=1e-4,
        cos_tol=0.9999,
    )
    _record(results, "linearity_pn_geom_v", ok)


# =============================================================================
# Linearity in grad_output  (all gradients are linear in g)
# =============================================================================
def test_linearity_in_g(
    vertices,
    indices,
    queries,
    epsilon,
    results,
) -> None:
    print(_hdr("\nTest: all gradients are linear in grad_output"))

    triangles = vertices[indices]
    rng = np.random.default_rng(73)
    N = len(triangles)
    M = len(queries)
    D = 4

    v = rng.standard_normal((N, D)).astype(np.float32)
    g1 = rng.standard_normal((M, D)).astype(np.float32)
    g2 = rng.standard_normal((M, D)).astype(np.float32)
    a, b = 1.7, -0.83
    gc = (a * g1 + b * g2).astype(np.float32)

    # Payload gradient
    p1 = _run_tri_payload_grad(triangles, g1, queries, epsilon, False)
    p2 = _run_tri_payload_grad(triangles, g2, queries, epsilon, False)
    pc = _run_tri_payload_grad(triangles, gc, queries, epsilon, False)
    ok = validate_payload_field(
        pc,
        a * p1 + b * p2,
        "Tri payload_grad(gc) == a·p1 + b·p2",
    )
    _record(results, "linearity_tri_payload_g", ok)

    # Geometry gradient
    c1, _ = _run_tri_geometry_grad_with_payload(
        triangles,
        v,
        g1,
        queries,
        epsilon,
        False,
    )
    c2, _ = _run_tri_geometry_grad_with_payload(
        triangles,
        v,
        g2,
        queries,
        epsilon,
        False,
    )
    cc, _ = _run_tri_geometry_grad_with_payload(
        triangles,
        v,
        gc,
        queries,
        epsilon,
        False,
    )
    ok = validate_vector_field(
        cc.reshape(-1, 3),
        (a * c1 + b * c2).reshape(-1, 3),
        "Tri geom_grad(gc) == a·g1 + b·g2",
        rel_norm_tol=1e-4,
        cos_tol=0.9999,
    )
    _record(results, "linearity_tri_geom_g", ok)


# =============================================================================
# D-path coverage
# =============================================================================
def test_d_path_coverage(
    vertices,
    indices,
    queries,
    epsilon,
    results,
) -> None:
    print(_hdr("\nTest: D-path coverage (all dispatch cases)"))
    triangles = vertices[indices]
    N = len(triangles)
    rng = np.random.default_rng(999)
    g = rng.standard_normal((len(queries), 32)).astype(np.float32)

    path_ds = [1, 2, 3, 4, 5, 7, 8, 9, 12, 16, 17, 20, 31, 32, 33, 40, 48, 64, 100]

    for D in path_ds:
        g_d = rng.standard_normal((len(queries), D)).astype(np.float32)
        try:
            out = _to_numpy(
                _run_tri_payload_grad(triangles, g_d, queries, epsilon, False)
            )
            ok = out.shape == (N, D) and bool(np.isfinite(out).all())
        except Exception as e:
            ok = False
            print(f"    {_bad('EXCEPTION')} D={D}: {e}")
        _record(results, f"d_path_D{D}", ok)
    _print_subresult(
        "all D dispatch cases", all(r.passed for r in results[-len(path_ds) :])
    )


# =============================================================================
# Kahan consistency
# =============================================================================
def test_kahan_consistency(
    vertices,
    indices,
    queries,
    epsilon,
    results,
) -> None:
    print(_hdr("\nTest: Kahan on/off consistency"))
    triangles = vertices[indices]
    N = len(triangles)
    M = len(queries)
    D = 8
    rng = np.random.default_rng(77)
    v = rng.standard_normal((N, D)).astype(np.float32)
    g = rng.standard_normal((M, D)).astype(np.float32)

    for label, fn in [
        ("payload", lambda k: _run_tri_payload_grad(triangles, g, queries, epsilon, k)),
        (
            "geom",
            lambda k: _run_tri_geometry_grad_with_payload(
                triangles, v, g, queries, epsilon, k
            )[0],
        ),
        (
            "query",
            lambda k: _run_tri_query_grad_with_payload(
                triangles, v, g, queries, epsilon, k
            ),
        ),
    ]:
        off = _to_numpy(fn(False))
        on = _to_numpy(fn(True))
        if off.ndim > 2 or off.shape[-1] == D:
            ok = validate_payload_field(
                torch.from_numpy(on),
                torch.from_numpy(off),
                f"Kahan consistency [{label}]",
            )
        else:
            ok = validate_vector_field(
                torch.from_numpy(on.reshape(-1, 3)),
                torch.from_numpy(off.reshape(-1, 3)),
                f"Kahan consistency [{label}]",
                rel_norm_tol=1e-4,
                cos_tol=0.9999,
            )
        _record(results, f"kahan_{label}", ok)


# =============================================================================
# Edge cases
# =============================================================================
def test_edge_cases(vertices, indices, queries, epsilon, results) -> None:
    print(_hdr("\nTest: edge cases"))

    triangles = vertices[indices]
    N = len(triangles)
    M = len(queries)
    D = 4
    rng = np.random.default_rng(88)
    v = rng.standard_normal((N, D)).astype(np.float32)

    # Zero grad_output → zero everything
    g_zero = np.zeros((M, D), np.float32)
    p_grad = _to_numpy(
        _run_tri_payload_grad(triangles, g_zero, queries, epsilon, False)
    )
    ok = np.allclose(p_grad, 0.0, atol=1e-7)
    _print_subresult("zero grad_output → zero payload grad", ok)
    _record(results, "edge_zero_g_payload", ok)

    c_grad, c_eps = _run_tri_geometry_grad_with_payload(
        triangles,
        v,
        g_zero,
        queries,
        epsilon,
        False,
    )
    ok = np.allclose(_to_numpy(c_grad), 0.0, atol=1e-7) and abs(float(c_eps[0])) < 1e-7
    _print_subresult("zero grad_output → zero geometry grad + eps grad", ok)
    _record(results, "edge_zero_g_geom", ok)

    # Zero payload → zero geometry grad
    g_nz = rng.standard_normal((M, D)).astype(np.float32)
    v_zero = np.zeros((N, D), np.float32)
    c_grad, _ = _run_tri_geometry_grad_with_payload(
        triangles,
        v_zero,
        g_nz,
        queries,
        epsilon,
        False,
    )
    ok = np.allclose(_to_numpy(c_grad), 0.0, atol=1e-7)
    _print_subresult("zero payload → zero geometry grad", ok)
    _record(results, "edge_zero_v_geom", ok)

    # Single primitive, single query
    tri1 = triangles[:1].copy()
    q1 = (
        (tri1[0].mean(0) + rng.standard_normal(3) * 2.0)
        .reshape(1, 3)
        .astype(np.float32)
    )
    v1 = rng.standard_normal((1, D)).astype(np.float32)
    g1 = rng.standard_normal((1, D)).astype(np.float32)
    scale1 = _scene_scale(tri1.reshape(-1, 3))
    eps_t = torch.tensor(epsilon, dtype=torch.float64, requires_grad=True, device="cuda")
    ref = pytorch_triangle_payload_grads_chunked_64(
        _t(tri1),
        _t(q1),
        _t(g1),
        _t(v1),
        eps_t,
        scale1,
        "compact",
    )
    cuda = _run_tri_payload_grad(tri1, g1, q1, epsilon, False)
    ok = validate_payload_field(
        cuda,
        ref.to(torch.float32),
        "N=1, M=1 payload grad",
    )
    _record(results, "edge_single", ok)


# =============================================================================
# Geometry setup
# =============================================================================
def make_synthetic_sphere(n_theta=20, n_phi=12, radius=1.0):
    theta = np.linspace(0, np.pi, n_theta + 1, dtype=np.float64)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False, dtype=np.float64)
    verts = []
    for i in range(1, n_theta):
        for j in range(n_phi):
            verts.append(
                [
                    radius * np.sin(theta[i]) * np.cos(phi[j]),
                    radius * np.sin(theta[i]) * np.sin(phi[j]),
                    radius * np.cos(theta[i]),
                ]
            )
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
    return verts, np.asarray(tris, dtype=np.uint32)


def mesh_to_point_surfels(vertices, indices):
    v0 = vertices[indices[:, 0]]
    v1 = vertices[indices[:, 1]]
    v2 = vertices[indices[:, 2]]
    points = (v0 + v1 + v2) / 3.0
    cross = np.cross(v1 - v0, v2 - v0)
    mag = np.linalg.norm(cross, axis=1, keepdims=True)
    areas = (mag / 2.0).flatten()
    safe = np.where(mag == 0, 1e-8, mag)
    normals = cross / safe
    return (
        points.astype(np.float32),
        (normals * areas[:, None]).astype(np.float32),
        areas,
    )


def generate_queries(vertices, num_queries, seed):
    rng = np.random.default_rng(seed)
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    diag = float(np.linalg.norm(hi - lo))
    lo = lo - 0.5 * diag
    hi = hi + 0.5 * diag
    return rng.uniform(lo, hi, size=(num_queries, 3)).astype(np.float32)


def slice_mesh_in_half(
    vertices: np.ndarray, indices: np.ndarray, dim: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Keep triangles whose centroid is above the median along `dim`."""
    face_verts = vertices[indices]
    centroids = face_verts.mean(axis=1)
    split_plane = np.median(centroids[:, dim])
    mask = centroids[:, dim] > split_plane
    return vertices, indices[mask]


def drop_half_of_the_triangles(
    vertices: np.ndarray, indices: np.ndarray, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly drop half of the triangles."""
    rng = np.random.default_rng(seed)
    choice = rng.choice(len(indices), len(indices) // 2, replace=False)
    return vertices, indices[choice]


# =============================================================================
# Summary
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
    return False


def _print_subresult(name: str, passed: bool, summary: str = "") -> None:
    mark = _ok("PASS") if passed else _bad("FAIL")
    print(f"    [{mark}] {name}" + (f"  {summary}" if summary else ""))


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Validate payload-field gradients against PyTorch float64 autograd."
    )
    parser.add_argument("--obj_file", type=str, default=None)
    parser.add_argument("--n_queries", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
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
        "--reg_mode",
        type=str,
        choices=REG_MODES,
        default="compact",
        help="Regularization strategy. Must match the CUDA build's DEFAULT_REG.",
    )
    parser.add_argument(
        "--test",
        type=str,
        default="all",
        help="Which suite: 'all', 'full', 'sanity', 'linearity', 'paths', "
        "'kahan', 'edge', or a comma-separated list.",
    )
    parser.add_argument(
        "--open_mode",
        type=str,
        choices=["closed", "slice", "drop", "both"],
        default="both",
        help="How to break the synthetic sphere (default: both slice and drop, "
        "which maximizes mid-field coverage). 'closed' preserves the original "
        "watertight case for baseline comparison.",
    )
    args = parser.parse_args()

    set_seeds(args.seed)

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

    if args.open_mode != "closed":
        n_before = len(indices)
        if args.open_mode in ("slice", "both"):
            vertices, indices = slice_mesh_in_half(vertices, indices)
        if args.open_mode in ("drop", "both"):
            vertices, indices = drop_half_of_the_triangles(
                vertices, indices, seed=args.seed
            )
        print(f"Open mesh ({args.open_mode}): {n_before} → {len(indices)} triangles")

    triangles = vertices[indices]
    points, scaled_normals, _ = mesh_to_point_surfels(vertices, indices)
    print(f"Point-normal surfels: {len(points)} points")
    queries = generate_queries(vertices, args.n_queries, args.seed)
    print(f"Queries: {len(queries)}")
    print(f"Reg mode: {args.reg_mode}")

    eps_values = [float(x) for x in args.epsilon_values.split(",") if x.strip()]
    D_values = [int(x) for x in args.d_values.split(",") if x.strip()]
    print(f"eps values: {eps_values}")
    print(f"D values:   {D_values}")

    tests = (
        set(args.test.split(","))
        if args.test != "all"
        else {
            "full",
            "sanity",
            "linearity",
            "paths",
            "kahan",
            "edge",
        }
    )

    results: list = []
    t0 = time()

    if "full" in tests:
        print(_hdr("\n=== Full gradient matrix: geometry × eps × D ==="))
        # Keep the cross-product limited: representative D per eps/reg
        for eps in eps_values:
            for D in [1, 4, 20, 32, 42]:
                if D not in D_values:
                    continue
                test_payload_gradients_pn(
                    points,
                    scaled_normals,
                    queries,
                    eps,
                    args.reg_mode,
                    D,
                    seed=100 + int(eps * 1000),
                    results=results,
                )
                test_payload_gradients_triangle(
                    triangles,
                    queries,
                    eps,
                    args.reg_mode,
                    D,
                    seed=200 + int(eps * 1000),
                    results=results,
                )
                test_payload_gradients_mesh(
                    vertices,
                    indices,
                    queries,
                    eps,
                    args.reg_mode,
                    D,
                    seed=300 + int(eps * 1000),
                    results=results,
                )

    if "sanity" in tests:
        test_d1_ones_scalar_reduction(vertices, indices, queries, eps_values, results)
        test_payload_grad_sum_identity(vertices, indices, queries, eps_values, results)

    if "linearity" in tests:
        test_linearity_in_v(vertices, indices, queries, DEFAULT_EPSILON, results)
        test_linearity_in_g(vertices, indices, queries, DEFAULT_EPSILON, results)

    if "paths" in tests:
        test_d_path_coverage(vertices, indices, queries, DEFAULT_EPSILON, results)

    if "kahan" in tests:
        test_kahan_consistency(vertices, indices, queries, DEFAULT_EPSILON, results)

    if "edge" in tests:
        test_edge_cases(vertices, indices, queries, DEFAULT_EPSILON, results)

    elapsed = time() - t0
    ok = print_summary(results)
    print(f"\nElapsed: {elapsed:.2f}s")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
