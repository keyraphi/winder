import argparse
import math
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm
import winder
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    from skimage.measure import marching_cubes
except ImportError:
    raise ImportError(
        "scikit-image is required for iso-surface visualization. Install it via: pip install scikit-image"
    )

# -----------------------------------------------------------------------------
# Provided Basis Functions
# -----------------------------------------------------------------------------


def triangle_grads(
    vertices: torch.Tensor,  # Shape: [N, 3, 3] float64
    queries: torch.Tensor,  # Shape: [Q, 3] float64
    grad_output: torch.Tensor,  # Shape: [Q] float64
    backend: str = "winder_bf",
):  # result is [N, 3, 3]
    if backend == "winder_bf":
        result = torch.empty(
            [vertices.shape[0], 3, 3], dtype=torch.float32, device="cuda:0"
        )
        winder.brute_force_gradients(
            grad_output.contiguous().to("cuda:0"),
            vertices.contiguous().to("cuda:0"),
            queries.contiguous().to("cuda:0"),
            result,
        )
        return result.cpu().double()
    elif backend == "winder":
        result = torch.empty(
            [vertices.shape[0], 3, 3], dtype=torch.float32, device="cuda:0"
        )
        engine = winder.GradientEngine(
            queries.contiguous().float().to("cuda:0"),
            grad_output.contiguous().float().to("cuda:0"),
        )
        engine.compute(vertices, result)
        return result.cpu().double()
    return pytorch_triangle_winding_grads_chunked_64(vertices, queries, grad_output)


def s_regularization(t: torch.Tensor) -> torch.Tensor:
    """PyTorch translation of CUDA S_regularization(t).

    S(t) = (4 / (3 * sqrt(pi))) * t^3                   if t < 0.1
    S(t) = erf(t) - (2 / sqrt(pi)) * t * exp(-t^2)      otherwise
    """
    two_over_sqrt_pi = 2.0 / math.sqrt(math.pi)
    four_over_3sqrt_pi = 4.0 / (3.0 * math.sqrt(math.pi))

    s_small = four_over_3sqrt_pi * (t**3)
    s_general = torch.erf(t) - (two_over_sqrt_pi * t * torch.exp(-t * t))

    return torch.where(t < 0.1, s_small, s_general)


def pointnormal_grads(
    points: torch.Tensor,  # Shape: [M, 3] float64
    normals: torch.Tensor,  # Shape: [M, 3] float64
    queries: torch.Tensor,  # Shape: [Q, 3] float64
    grad_output: torch.Tensor,  # Shape: [Q] float64
    inv_epsilon: float = 1,
    backend: str = "winder_bf",
):  # result is tuple of point grad: [M, 3] and normal grad: [M, 3]
    if backend == "winder_bf":
        result = torch.empty(
            [points.shape[0], 2, 3], dtype=torch.float32, device="cuda:0"
        )
        winder.brute_force_gradients(
            grad_output.contiguous().to("cuda:0"),
            points.contiguous().to("cuda:0"),
            normals.contiguous().to("cuda:0"),
            queries.contiguous().to("cuda:0"),
            result,
            1 / inv_epsilon,
        )
        normal_grad, point_grad = result[:, 0], result[:, 1]
        return point_grad.cpu().double(), normal_grad.cpu().double()
    elif backend == "winder":
        result = torch.empty(
            [points.shape[0], 2, 3], dtype=torch.float32, device="cuda:0"
        )
        engine = winder.GradientEngine(
            queries.contiguous().float().to("cuda:0"),
            grad_output.contiguous().float().to("cuda:0"),
        )
        engine.compute(
            points.contiguous().float().to("cuda:0"),
            normals.contiguous().float().to("cuda:0"),
            result,
            epsilon=1 / inv_epsilon,
        )
        normal_grad, point_grad = result[:, 0], result[:, 1]
        return point_grad.cpu().double(), normal_grad.cpu().double()

    return pytorch_point_normal_grads_chunked_64(
        points, normals, queries, grad_output, inv_epsilon, s_regularization
    )


def pytorch_triangle_winding_grads_chunked_64(
    vertices: torch.Tensor,  # Shape: [N, 3, 3] float64
    queries: torch.Tensor,  # Shape: [Q, 3] float64
    grad_output: torch.Tensor,  # Shape: [Q] float64
    chunk_size: int = 50,
) -> torch.Tensor:
    v = vertices.clone().detach().requires_grad_(True)
    num_queries = queries.shape[0]
    inv_two_pi = 1.0 / (2.0 * math.pi)

    for i in range(0, num_queries, chunk_size):
        q_chunk = queries[i : i + chunk_size]
        g_chunk = grad_output[i : i + chunk_size]

        v0 = v[:, 0, :][None, :, :]
        v1 = v[:, 1, :][None, :, :]
        v2 = v[:, 2, :][None, :, :]
        q = q_chunk[:, None, :]

        a = v0 - q
        b = v1 - q
        c = v2 - q

        a2 = torch.sum(a * a, dim=-1) + 1e-30
        b2 = torch.sum(b * b, dim=-1) + 1e-30
        c2 = torch.sum(c * c, dim=-1) + 1e-30

        inv_a = torch.rsqrt(a2)
        inv_b = torch.rsqrt(b2)
        inv_c = torch.rsqrt(c2)

        cos_ab = torch.sum(a * b, dim=-1) * inv_a * inv_b
        cos_ac = torch.sum(a * c, dim=-1) * inv_a * inv_c
        cos_bc = torch.sum(b * c, dim=-1) * inv_b * inv_c

        cross_bc = torch.cross(b, c, dim=-1)
        det_norm = torch.sum(a * cross_bc, dim=-1) * inv_a * inv_b * inv_c
        div_norm = 1.0 + cos_ab + cos_ac + cos_bc

        sol_angle = torch.atan2(det_norm, div_norm) * inv_two_pi

        loss_chunk = torch.sum(sol_angle * g_chunk[:, None])
        loss_chunk.backward()

    return v.grad


def pytorch_point_normal_grads_chunked_64(
    points: torch.Tensor,  # Shape: [M, 3] float64
    normals: torch.Tensor,  # Shape: [M, 3] float64
    queries: torch.Tensor,  # Shape: [Q, 3] float64
    grad_output: torch.Tensor,  # Shape: [Q] float64
    inv_epsilon: float = 1.0,
    s_regularization_fn=None,
    chunk_size: int = 50,
) -> tuple[torch.Tensor, torch.Tensor]:
    p = points.clone().detach().requires_grad_(True)
    n = normals.clone().detach().requires_grad_(True)
    num_queries = queries.shape[0]

    four_over_3sqrt_pi = 4.0 / (3.0 * math.sqrt(math.pi))
    inv_four_pi = 1.0 / (4.0 * math.pi)
    near_limit_constant = four_over_3sqrt_pi * (inv_epsilon**3)

    for i in range(0, num_queries, chunk_size):
        q_chunk = queries[i : i + chunk_size]
        g_chunk = grad_output[i : i + chunk_size]

        p_bc = p[None, :, :]
        n_bc = n[None, :, :]
        q_bc = q_chunk[:, None, :]

        d = p_bc - q_bc
        dist2 = torch.sum(d * d, dim=-1)

        inv_distance = torch.rsqrt(dist2 + 1e-30)
        inv_dist2 = inv_distance * inv_distance
        inv_dist3 = inv_dist2 * inv_distance

        distance = dist2 * inv_distance
        t = distance * inv_epsilon

        if s_regularization_fn is not None:
            s_val = s_regularization_fn(t)
        else:
            s_val = torch.ones_like(t)

        s_reg_term = s_val * inv_dist3

        s_over_dist3 = torch.where(
            t < 0.1,
            near_limit_constant,
            torch.where(t < 2.0, s_reg_term, inv_dist3),
        )

        dot_n_d = torch.sum(n_bc * d, dim=-1)
        contributions = dot_n_d * inv_four_pi * s_over_dist3

        loss_chunk = torch.sum(contributions * g_chunk[:, None])
        loss_chunk.backward()

    return p.grad, n.grad


def generate_queries(
    vertices: torch.Tensor, mode: str, num_queries: int = 100
) -> torch.Tensor:
    min_box = vertices.min(axis=0).values
    max_box = vertices.max(axis=0).values
    diag = torch.linalg.norm(max_box - min_box)
    min_box -= diag * 0.5
    max_box += diag * 0.5

    if mode == "grid":
        side = int(np.ceil(num_queries ** (1.0 / 3.0)))
        x = torch.linspace(min_box[0], max_box[0], side)
        y = torch.linspace(min_box[1], max_box[1], side)
        z = torch.linspace(min_box[2], max_box[2], side)
        gx, gy, gz = torch.meshgrid(x, y, z, indexing="ij")
        queries = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
        return queries[:num_queries].to(torch.float64)
    else:
        result = torch.rand((num_queries, 3), dtype=torch.float64)
        result[:, 0] = result[:, 0] * (max_box[0] - min_box[0]) + min_box[0]
        result[:, 1] = result[:, 1] * (max_box[1] - min_box[1]) + min_box[1]
        result[:, 2] = result[:, 2] * (max_box[2] - min_box[2]) + min_box[2]
        return result


# -----------------------------------------------------------------------------
# Forward Evaluation Helpers
# -----------------------------------------------------------------------------


def evaluate_point_normal_field(
    p: torch.Tensor,
    n: torch.Tensor,
    q: torch.Tensor,
    inv_epsilon: float = 1.0,
    backend: str = "winder_bf",
) -> torch.Tensor:
    if backend == "winder_bf":
        result: torch.Tensor = torch.empty(
            [q.shape[0]], dtype=torch.float32, device="cuda:0"
        )
        winder.brute_force_winding_numbers(
            p.contiguous().to("cuda:0"),
            n.contiguous().to("cuda:0"),
            q.contiguous().to("cuda:0"),
            result,
            epsilon=1 / inv_epsilon,
        )
        return result.cpu()
    elif backend == "winder":
        result: torch.Tensor = torch.empty(
            [q.shape[0]], dtype=torch.float32, device="cuda:0"
        )
        engine = winder.WindingNumberEngine(
            p.contiguous().float().to("cuda:0"), n.contiguous().float().to("cuda:0")
        )
        engine.compute(q.contiguous().float().to("cuda:0"), result, epsilon=1/inv_epsilon)
        return result.cpu()

    four_over_3sqrt_pi = 4.0 / (3.0 * math.sqrt(math.pi))
    inv_four_pi = 1.0 / (4.0 * math.pi)
    near_limit_constant = four_over_3sqrt_pi * (inv_epsilon**3)

    d = p[None, :, :] - q[:, None, :]
    dist2 = torch.sum(d * d, dim=-1)
    inv_distance = torch.rsqrt(dist2 + 1e-30)
    inv_dist3 = inv_distance * inv_distance * inv_distance

    distance = dist2 * inv_epsilon
    t = distance * inv_epsilon

    s_over_dist3 = torch.where(
        t < 0.1,
        near_limit_constant,
        torch.where(t < 2.0, inv_dist3, inv_dist3),
    )
    dot_n_d = torch.sum(n[None, :, :] * d, dim=-1)
    result = torch.sum(dot_n_d * inv_four_pi * s_over_dist3, dim=-1)
    return result


def evaluate_triangle_field(
    v: torch.Tensor, q: torch.Tensor, backend: str = "winder_bf"
) -> torch.Tensor:
    if backend == "winder_bf":
        result: torch.Tensor = torch.empty(
            [q.shape[0]], dtype=torch.float32, device="cuda:0"
        )
        winder.brute_force_winding_numbers(
            v.contiguous().to("cuda:0"), q.contiguous().to("cuda:0"), result
        )
        return result.cpu()
    elif backend == "winder":
        result: torch.Tensor = torch.empty(
            [q.shape[0]], dtyp=torch.float32, device="cuda:0"
        )
        engine = winder.WindingNumberEngine(v.contiguous().float().to("cuda:0"))
        engine.compute(q.contiguous().float().to("cuda:0"), result)
        return result.cpu()

    inv_two_pi = 1.0 / (2.0 * math.pi)
    v0 = v[:, 0, :][None, :, :]
    v1 = v[:, 1, :][None, :, :]
    v2 = v[:, 2, :][None, :, :]
    q_bc = q[:, None, :]

    a = v0 - q_bc
    b = v1 - q_bc
    c = v2 - q_bc

    a2 = torch.sum(a * a, dim=-1) + 1e-30
    b2 = torch.sum(b * b, dim=-1) + 1e-30
    c2 = torch.sum(c * c, dim=-1) + 1e-30

    inv_a = torch.rsqrt(a2)
    inv_b = torch.rsqrt(b2)
    inv_c = torch.rsqrt(c2)

    cos_ab = torch.sum(a * b, dim=-1) * inv_a * inv_b
    cos_ac = torch.sum(a * c, dim=-1) * inv_a * inv_c
    cos_bc = torch.sum(b * c, dim=-1) * inv_b * inv_c

    cross_bc = torch.cross(b, c, dim=-1)
    det_norm = torch.sum(a * cross_bc, dim=-1) * inv_a * inv_b * inv_c
    div_norm = 1.0 + cos_ab + cos_ac + cos_bc

    sol_angle = torch.atan2(det_norm, div_norm) * inv_two_pi
    return torch.sum(sol_angle, dim=-1)


# -----------------------------------------------------------------------------
# Rendering Helpers
# -----------------------------------------------------------------------------


def render_query_gradient_quiver(
    queries: torch.Tensor,
    q_grads: np.ndarray,
    target_pos: np.ndarray = None,
    source_pos: np.ndarray = None,
    mode_name: str = "pn_same",
):
    """Renders a standalone figure displaying the per-query gradient contribution quiver field."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    q_np = queries.numpy()
    mags = np.linalg.norm(q_grads, axis=-1)

    # Direction vectors scaled for visualization
    max_mag = mags.max() if mags.max() > 1e-12 else 1.0
    scale = 0.15
    u_q = (q_grads[:, 0] / max_mag) * scale
    v_q = (q_grads[:, 1] / max_mag) * scale
    w_q = (q_grads[:, 2] / max_mag) * scale

    ax.quiver(
        q_np[:, 0],
        q_np[:, 1],
        q_np[:, 2],
        u_q,
        v_q,
        w_q,
        color="crimson",
        linewidth=0.9,
        alpha=0.7,
        arrow_length_ratio=0.3,
        label=r"Gradient Pull $\nabla_{p_{\mathrm{src}}} L_i$",
    )

    if target_pos is not None:
        ax.scatter(
            target_pos[0],
            target_pos[1],
            target_pos[2],
            color="green",
            s=120,
            label="Target Point ($p_0$)",
        )
    if source_pos is not None:
        ax.scatter(
            source_pos[0],
            source_pos[1],
            source_pos[2],
            color="royalblue",
            s=120,
            label="Source Point ($p_1$)",
        )

    ax.set_title(
        f"Per-Query Gradient Contributions Field ({mode_name.replace('_', ' ').title()})"
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="upper right", fontsize="small")
    plt.tight_layout()


def render_field_isosurfaces(
    ax,
    field_fn,
    bbox: list[float],
    iso_val: float,
    grid_res: int = 45,
    pos_color: str = "forestgreen",
    neg_color: str = "crimson",
    alpha: float = 0.15,
    label_prefix: str = "Target",
):
    x = np.linspace(bbox[0], bbox[1], grid_res)
    y = np.linspace(bbox[2], bbox[3], grid_res)
    z = np.linspace(bbox[4], bbox[5], grid_res)
    gx, gy, gz = np.meshgrid(x, y, z, indexing="ij")
    grid_pts = torch.tensor(
        np.stack([gx.flatten(), gy.flatten(), gz.flatten()], axis=-1),
        dtype=torch.float64,
    )

    with torch.no_grad():
        vals = field_fn(grid_pts).reshape(grid_res, grid_res, grid_res).numpy()

    spacing = (x[1] - x[0], y[1] - y[0], z[1] - z[0])
    origin = np.array([x[0], y[0], z[0]])

    if vals.max() >= iso_val:
        try:
            verts, faces, _, _ = marching_cubes(vals, level=iso_val, spacing=spacing)
            verts += origin
            mesh = Poly3DCollection(
                verts[faces], alpha=alpha, facecolor=pos_color, edgecolor="none"
            )
            mesh.set_label(f"{label_prefix} (+{iso_val:.2f})")
            ax.add_collection3d(mesh)
        except ValueError:
            pass

    if vals.min() <= -iso_val:
        try:
            verts, faces, _, _ = marching_cubes(vals, level=-iso_val, spacing=spacing)
            verts += origin
            mesh = Poly3DCollection(
                verts[faces], alpha=alpha, facecolor=neg_color, edgecolor="none"
            )
            mesh.set_label(f"{label_prefix} (-{iso_val:.2f})")
            ax.add_collection3d(mesh)
        except ValueError:
            pass


def render_position_loss_landscape_2d(
    field_fn_tgt,
    p_src: torch.Tensor,
    p_tgt: torch.Tensor,
    n_tgt: torch.Tensor,
    n_src: torch.Tensor,
    queries: torch.Tensor,
    inv_epsilon: float = 250.0,
    backend: str = "winder_bf",
    plane_extent: float = 1.4,
    grid_res: int = 60,
    quiver_res: int = 16,
    cmap_name: str = "magma",
    use_log_scale: bool = True,
    eps: float = 1e-6,
):
    """Renders 2D Spatial Position Loss Landscape with overlaid gradient steepest descent vectors."""
    fig, ax = plt.subplots(figsize=(9, 8))

    p0 = p_tgt[0].detach().cpu().numpy()
    p1 = p_src[0].detach().cpu().numpy()
    n_src_np = n_src[0].detach().cpu().numpy()

    # Define orthonormal plane basis (u, v) centered between p_tgt and p_src
    u0 = p1 - p0
    dist = np.linalg.norm(u0)
    u = u0 / dist if dist > 1e-6 else np.array([1.0, 0.0, 0.0])

    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(u, ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    v0 = ref - np.dot(ref, u) * u
    v = v0 / np.linalg.norm(v0)

    center = 0.5 * (p0 + p1)
    half_size = max(dist * plane_extent, 1.2)

    # -------------------------------------------------------------------------
    # 1. Scalar Loss Contour Grid over Candidate Positions p_src(s, t)
    # -------------------------------------------------------------------------
    s_vals = np.linspace(-half_size, half_size, grid_res)
    t_vals = np.linspace(-half_size, half_size, grid_res)
    S, T = np.meshgrid(s_vals, t_vals)

    with torch.no_grad():
        w_tgt = field_fn_tgt(queries)

    loss_grid = np.zeros((grid_res, grid_res))
    n_src_tensor = torch.tensor(n_src_np[None, :], dtype=torch.float64)

    for i in tqdm(range(grid_res), desc="Computing Position Loss Grid"):
        for j in range(grid_res):
            p_cand_3d = center + S[i, j] * u + T[i, j] * v
            p_cand_tensor = torch.tensor(p_cand_3d[None, :], dtype=torch.float64)

            with torch.no_grad():
                w_cand = evaluate_point_normal_field(
                    p_cand_tensor,
                    n_src_tensor,
                    queries,
                    inv_epsilon=inv_epsilon,
                    backend=backend
                )
                loss_grid[i, j] = torch.abs(w_cand - w_tgt).mean().item()

    loss_vals = np.log10(loss_grid + eps) if use_log_scale else loss_grid

    cmap = plt.get_cmap(cmap_name)
    contour = ax.contourf(S, T, loss_vals, levels=40, cmap=cmap)
    ax.contour(S, T, loss_vals, levels=20, colors="white", alpha=0.2, linewidths=0.5)

    # -------------------------------------------------------------------------
    # 2. Position Gradient Quiver Field (-grad_p)
    # -------------------------------------------------------------------------
    s_q = np.linspace(-half_size * 0.9, half_size * 0.9, quiver_res)
    t_q = np.linspace(-half_size * 0.9, half_size * 0.9, quiver_res)
    S_q, T_q = np.meshgrid(s_q, t_q)

    d_u = np.zeros((quiver_res, quiver_res))
    d_v = np.zeros((quiver_res, quiver_res))
    num_queries = queries.shape[0]

    for i in tqdm(range(quiver_res), desc="Position Loss quiver"):
        for j in range(quiver_res):
            p_cand_3d = center + S_q[i, j] * u + T_q[i, j] * v
            p_cand_tensor = torch.tensor(p_cand_3d[None, :], dtype=torch.float64)

            with torch.no_grad():
                w_cand = evaluate_point_normal_field(
                    p_cand_tensor,
                    n_src_tensor,
                    queries,
                    inv_epsilon=inv_epsilon,
                    backend=backend
                )
                dL_dw = torch.sign(w_cand.squeeze() - w_tgt.squeeze()) / num_queries

            grad_p_3d, _ = pointnormal_grads(
                p_cand_tensor,
                n_src_tensor,
                queries,
                dL_dw,
                inv_epsilon=inv_epsilon,
                backend=backend,
            )

            gp = grad_p_3d[0].detach().cpu().numpy()
            d_u[i, j] = -np.dot(gp, u)
            d_v[i, j] = -np.dot(gp, v)

    # Direction normalization for clean arrow visualization
    mag = np.hypot(d_u, d_v)
    valid_mask = mag > 1e-12
    d_u[valid_mask] /= mag[valid_mask]
    d_v[valid_mask] /= mag[valid_mask]

    ax.quiver(
        S_q,
        T_q,
        d_u,
        d_v,
        color="cyan",
        alpha=0.7,
        pivot="middle",
        scale=25,
        headwidth=3.5,
        headlength=4,
        label=r"Steepest Descent $-\nabla_{\vec{p}_{\mathrm{src}}} \mathcal{L}$",
    )

    # -------------------------------------------------------------------------
    # 3. Overlays & Annotations
    # -------------------------------------------------------------------------
    s_tgt, t_tgt = np.dot(p0 - center, u), np.dot(p0 - center, v)
    s_src, t_src = np.dot(p1 - center, u), np.dot(p1 - center, v)

    ax.scatter(
        s_tgt,
        t_tgt,
        color="lime",
        s=120,
        zorder=10,
        edgecolor="black",
        label=r"Target Position ($p_{\mathrm{tgt}}$)",
    )
    ax.scatter(
        s_src,
        t_src,
        color="orange",
        s=120,
        zorder=10,
        edgecolor="black",
        label=r"Source Init ($p_{\mathrm{src}}$)",
    )

    title_str = (
        r"Spatial Position Loss Landscape $\mathcal{L}(p_{\mathrm{src}})$ [Log10]"
        if use_log_scale
        else r"Spatial Position Loss Landscape $\mathcal{L}(p_{\mathrm{src}})$"
    )
    ax.set_title(title_str, pad=15)
    ax.set_xlabel(
        "Plane Axis S (along $p_{\\mathrm{tgt}} \\rightarrow p_{\\mathrm{src}}$)"
    )
    ax.set_ylabel("Plane Axis T (orthogonal)")
    ax.grid(True, linestyle=":", alpha=0.3)

    cbar = fig.colorbar(contour, ax=ax, pad=0.03, shrink=0.85)
    cbar.set_label(
        r"$\log_{10}$ Loss" if use_log_scale else "Loss",
        rotation=270,
        labelpad=15,
    )
    ax.legend(loc="upper right", fontsize="small")
    plt.tight_layout()
    return fig, ax


def render_loss_landscape_3d(
    p_src_init: torch.Tensor,
    n_src_init: torch.Tensor,
    p_tgt: torch.Tensor,
    n_tgt: torch.Tensor,
    queries: torch.Tensor,
    w_tgt: torch.Tensor,
    inv_epsilon: float = 250.0,
    plane_extent: float = 1.4,
    grid_res: int = 120,
    cmap_name: str = "magma",
    use_log_scale: bool = True,
    eps: float = 1e-6,
    backend: str = "winder_bf",
):
    """Evaluates integrated mean loss over fixed queries for candidate source positions p_src(s, t)."""
    p0 = p_tgt[0].detach().cpu().numpy()
    p1 = p_src_init[0].detach().cpu().numpy()
    n_tgt_np = n_tgt[0].detach().cpu().numpy()

    # Define orthonormal plane basis
    u0 = p1 - p0
    dist = np.linalg.norm(u0)
    u = u0 / dist if dist > 1e-6 else np.array([1.0, 0.0, 0.0])
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(u, ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    v0 = ref - np.dot(ref, u) * u
    v = v0 / np.linalg.norm(v0)

    center = 0.5 * (p0 + p1)
    half_size = max(dist * plane_extent, 1.2)

    s_vals = np.linspace(-half_size, half_size, grid_res)
    t_vals = np.linspace(-half_size, half_size, grid_res)
    S, T = np.meshgrid(s_vals, t_vals)

    loss_grid = np.zeros((grid_res, grid_res))
    n_src_tensor = n_src_init[0:1].to(dtype=torch.float64)

    # Sweep candidate source positions p_cand over (S, T) plane
    for i in tqdm(range(grid_res), desc="render_loss_landscape_3d"):
        for j in range(grid_res):
            p_cand_3d = center + S[i, j] * u + T[i, j] * v
            p_cand_tensor = torch.tensor(p_cand_3d[None, :], dtype=torch.float64)

            with torch.no_grad():
                w_cand = evaluate_point_normal_field(
                    p_cand_tensor,
                    n_src_tensor,
                    queries,
                    inv_epsilon=inv_epsilon,
                    backend=backend,
                )
                loss_grid[i, j] = torch.abs(w_cand - w_tgt).mean().item()

    loss_vals = np.log10(loss_grid + eps) if use_log_scale else loss_grid

    # 3D Visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap(cmap_name)

    surf = ax.plot_surface(
        S, T, loss_vals, cmap=cmap, linewidth=0, antialiased=True, alpha=0.85
    )
    z_min = loss_vals.min()
    ax.contour(
        S,
        T,
        loss_vals,
        zdir="z",
        offset=z_min,
        levels=40,
        cmap=cmap,
        linewidths=0.9,
    )

    # Project source and target points onto planar coordinates
    s_tgt, t_tgt = np.dot(p0 - center, u), np.dot(p0 - center, v)
    s_src, t_src = np.dot(p1 - center, u), np.dot(p1 - center, v)

    ax.scatter(
        s_tgt,
        t_tgt,
        z_min,
        color="lime",
        s=80,
        zorder=10,
        label=r"Target Position ($p_{\mathrm{tgt}}$)",
    )
    ax.scatter(
        s_src,
        t_src,
        z_min,
        color="orange",
        s=80,
        zorder=10,
        label=r"Source Init ($p_{\mathrm{src}}$)",
    )

    ax.set_title(r"Position Loss Landscape $\mathcal{L}(p_{\mathrm{src}})$")
    ax.set_xlabel("Plane Axis S")
    ax.set_ylabel("Plane Axis T")
    ax.set_zlabel(r"$\log_{10}$ Loss" if use_log_scale else "Loss")
    fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.1)
    ax.legend()
    plt.tight_layout()
    return fig, ax


# -----------------------------------------------------------------------------
# Per-Query Gradient Evaluation Helper
# -----------------------------------------------------------------------------


def compute_per_query_gradient_contributions(
    mode: str,
    p_src: torch.Tensor,
    n_src: torch.Tensor,
    v_src: torch.Tensor,
    queries: torch.Tensor,
    grad_out: torch.Tensor,
    inv_epsilon: float = 1.0,
    max_vis: int = 1000,
    backend="winder_bf",
) -> np.ndarray:
    """Evaluates the gradient function individually for every single query point."""
    step = max(1, len(queries) // max_vis)
    queries = queries[::step]
    num_queries = queries.shape[0]
    q_grads = []

    for i in range(num_queries):
        q_single = queries[i : i + 1]
        g_single = grad_out[i : i + 1]

        if mode in ["pn_same", "pn_random_normal", "pn_random_pos"]:
            p_grad_i, _ = pointnormal_grads(
                p_src,
                n_src,
                q_single,
                g_single,
                inv_epsilon=inv_epsilon,
                backend=backend,
            )
            q_grads.append(p_grad_i[0].numpy())
        elif mode == "triangle":
            v_grad_i = triangle_grads(v_src, q_single, g_single, backend=backend)
            q_grads.append(v_grad_i[0, 0].numpy())

    return queries, np.array(q_grads)


def get_plane_basis(p0: np.ndarray, p1: np.ndarray, n_tgt: np.ndarray = None):
    """Computes an orthonormal basis (u, v) for the 2D plane such that n_tgt

    lies completely within span(u, v).
    """
    # 1. Primary axis u: Direction between source and target points
    u0 = p1 - p0
    dist = np.linalg.norm(u0)
    if dist > 1e-6:
        u = u0 / dist
    elif n_tgt is not None and np.linalg.norm(n_tgt) > 1e-6:
        u = n_tgt / np.linalg.norm(n_tgt)
    else:
        u = np.array([1.0, 0.0, 0.0])

    # 2. Secondary axis v: Gram-Schmidt component of n_tgt orthogonal to u
    if n_tgt is not None and np.linalg.norm(n_tgt) > 1e-6:
        n_tgt_unit = n_tgt / np.linalg.norm(n_tgt)
        v0 = n_tgt_unit - np.dot(n_tgt_unit, u) * u
        v_norm = np.linalg.norm(v0)

        if v_norm > 1e-6:
            v = v0 / v_norm
        else:
            # n_tgt is collinear with u (already in plane along u axis)
            ref = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(u, ref)) > 0.9:
                ref = np.array([1.0, 0.0, 0.0])
            v0 = ref - np.dot(ref, u) * u
            v = v0 / np.linalg.norm(v0)
    else:
        # Fallback when n_tgt is not provided
        ref = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(u, ref)) > 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        v0 = ref - np.dot(ref, u) * u
        v = v0 / np.linalg.norm(v0)

    return u, v


def render_normal_loss_landscape_polar(
    field_fn_tgt,
    p_src: torch.Tensor,
    p_tgt: torch.Tensor,
    n_tgt: torch.Tensor,
    n_src_init: torch.Tensor,
    queries: torch.Tensor,
    grad_n_init: np.ndarray = None,
    inv_epsilon: float = 250.0,
    backend: str = "winder_bf",
    grid_r: int = 60,
    grid_theta: int = 120,
    quiver_r: int = 12,
    quiver_theta: int = 24,
    max_r: float = 2.0,
    cmap_name: str = "magma",
    use_log_scale: bool = True,
    eps: float = 1e-6,
):
    """Renders 2D Polar Loss Landscape using a single batched call for the quiver vector field."""
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="polar")

    p0 = p_tgt[0].numpy() if isinstance(p_tgt, torch.Tensor) else p_tgt
    p1 = p_src[0].numpy() if isinstance(p_src, torch.Tensor) else p_src
    n_tgt_np = (
        n_tgt[0].detach().numpy()
        if isinstance(n_tgt, torch.Tensor)
        else np.asarray(n_tgt)
    )
    n_src_np = (
        n_src_init[0].detach().numpy()
        if isinstance(n_src_init, torch.Tensor)
        else np.asarray(n_src_init)
    )

    u, v = get_plane_basis(p0, p1, n_tgt_np)

    # -------------------------------------------------------------------------
    # 1. Compute Scalar Loss Contour Grid
    # -------------------------------------------------------------------------
    r_vals = np.linspace(0.01, max_r, grid_r)
    theta_vals = np.linspace(0, 2 * np.pi, grid_theta)
    R, Theta = np.meshgrid(r_vals, theta_vals)

    Nx = R * np.cos(Theta)
    Ny = R * np.sin(Theta)

    with torch.no_grad():
        w_tgt = field_fn_tgt(queries)

    loss_grid = np.zeros((grid_theta, grid_r))

    for i in tqdm(range(grid_theta), desc="polar"):
        for j in range(grid_r):
            n_vec_3d = Nx[i, j] * u + Ny[i, j] * v
            n_tensor = torch.tensor(n_vec_3d[None, :], dtype=torch.float64)
            p_src_tensor = torch.tensor(p1[None, :], dtype=torch.float64)
            with torch.no_grad():
                w_src = evaluate_point_normal_field(
                    p_src_tensor, n_tensor, queries, inv_epsilon=inv_epsilon, backend=backend
                )
                loss_grid[i, j] = torch.abs(w_src - w_tgt).mean().item()

    loss_vals = np.log10(loss_grid + eps) if use_log_scale else loss_grid

    cmap = plt.get_cmap(cmap_name)
    contour = ax.contourf(Theta, R, loss_vals, levels=40, cmap=cmap)
    ax.contour(
        Theta, R, loss_vals, levels=20, colors="white", alpha=0.2, linewidths=0.5
    )

    # -------------------------------------------------------------------------
    # 2. Individual Quiver Gradient Evaluation (Single Primitive per Iteration)
    # -------------------------------------------------------------------------
    r_q = np.linspace(0.15, max_r * 0.95, quiver_r)
    theta_q = np.linspace(0, 2 * np.pi, quiver_theta, endpoint=False)
    R_q, Theta_q = np.meshgrid(r_q, theta_q)

    r_flat = R_q.ravel()
    theta_flat = Theta_q.ravel()
    num_quiver = len(r_flat)

    g_3d = np.zeros((num_quiver, 3))
    p_single = torch.tensor(p1[None, :], dtype=torch.float64)

    for k in tqdm(range(num_quiver), desc="polar quiver"):
        # 3D Normal for the current polar point
        nx = r_flat[k] * np.cos(theta_flat[k])
        ny = r_flat[k] * np.sin(theta_flat[k])
        n_vec_3d = nx * u + ny * v
        n_single = torch.tensor(n_vec_3d[None, :], dtype=torch.float64)

        # Forward evaluation of single primitive [1, 3] against queries [Q, 3]
        with torch.no_grad():
            w_src_single = evaluate_point_normal_field(
                p_single, n_single, queries, inv_epsilon=inv_epsilon, backend=backend
            )
            # L1 Loss gradient w.r.t. scalar field
            dL_dw_single = (
                torch.sign(w_src_single.squeeze() - w_tgt.squeeze()) / queries.shape[0]
            )

        # Gradient of single normal [1, 3]
        _, grad_n_single = pointnormal_grads(
            p_single,
            n_single,
            queries,
            dL_dw_single,
            inv_epsilon=inv_epsilon,
            backend=backend,
        )
        g_3d[k] = grad_n_single[0].detach().cpu().numpy()

    # -------------------------------------------------------------------------
    # 3. Direct Cartesian Plane Projection for Matplotlib Polar Quiver
    # -------------------------------------------------------------------------
    # Steepest descent direction in 3D: d = -g
    d_u = -(g_3d @ u)
    d_v = -(g_3d @ v)

    # Normalize directional lengths for uniform arrow visualization
    mag = np.hypot(d_u, d_v)
    valid_mask = mag > 1e-12
    d_u[valid_mask] /= mag[valid_mask]
    d_v[valid_mask] /= mag[valid_mask]

    # Reshape to quiver grid shape (quiver_theta, quiver_r)
    D_u_q = d_u.reshape(quiver_theta, quiver_r)
    D_v_q = d_v.reshape(quiver_theta, quiver_r)

    # Plot Quiver Field: U = D_u_q (horizontal screen delta), V = D_v_q (vertical screen delta)
    ax.quiver(
        Theta_q,
        R_q,
        D_u_q,
        D_v_q,
        color="cyan",
        alpha=0.6,
        pivot="middle",
        scale=25,
        headwidth=3.5,
        headlength=4,
    )
    # -------------------------------------------------------------------------
    # 4. Overlays & Annotations
    # -------------------------------------------------------------------------
    ax.plot(
        theta_vals,
        np.ones_like(theta_vals),
        color="white",
        linestyle="--",
        linewidth=1.8,
        label=r"Unit Sphere ($r=1.0$)",
    )

    r_tgt_proj = np.sqrt(np.dot(n_tgt_np, u) ** 2 + np.dot(n_tgt_np, v) ** 2)
    theta_tgt_proj = np.arctan2(np.dot(n_tgt_np, v), np.dot(n_tgt_np, u)) % (2 * np.pi)
    ax.scatter(
        theta_tgt_proj,
        r_tgt_proj,
        color="lime",
        s=120,
        zorder=10,
        edgecolor="black",
        label=r"$\vec{n}_{\mathrm{tgt}}$ Proj.",
    )

    r_src_proj = np.sqrt(np.dot(n_src_np, u) ** 2 + np.dot(n_src_np, v) ** 2)
    theta_src_proj = np.arctan2(np.dot(n_src_np, v), np.dot(n_src_np, u)) % (2 * np.pi)
    ax.scatter(
        theta_src_proj,
        r_src_proj,
        color="orange",
        s=120,
        zorder=10,
        edgecolor="black",
        label=r"$\vec{n}_{\mathrm{src}}$ Init Proj.",
    )

    title_str = (
        r"Polar Loss Landscape & Batched Normal Field $(r, \theta)$ [Log10]"
        if use_log_scale
        else r"Polar Loss Landscape & Batched Normal Field $(r, \theta)$"
    )
    ax.set_title(title_str, pad=20)
    cbar = fig.colorbar(contour, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label(
        r"$\log_{10}$ Loss" if use_log_scale else "Loss",
        rotation=270,
        labelpad=15,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize="small")
    plt.tight_layout()


def render_joint_loss_landscape_2d(
    p_src_init: torch.Tensor,
    n_src_init: torch.Tensor,
    p_tgt: torch.Tensor,
    n_tgt: torch.Tensor,
    queries: torch.Tensor,
    inv_epsilon: float = 250.0,
    plane_extent: float = 1.4,
    grid_res: int = 50,
    quiver_res: int = 14,
    backend: str = "winder_bf",
    cmap_name: str = "magma",
    use_log_scale: bool = True,
    eps: float = 1e-6,
):
    """Renders 2D spatial loss heightmap with overlaid spatial (-grad_p) and dipole (-grad_n) gradient quivers."""
    p0 = p_tgt[0].detach().cpu().numpy()
    p1 = p_src_init[0].detach().cpu().numpy()
    n_tgt_np = n_tgt[0].detach().cpu().numpy()
    n_src_np = n_src_init[0].detach().cpu().numpy()

    u0 = p1 - p0
    dist = np.linalg.norm(u0)
    u = u0 / dist if dist > 1e-6 else np.array([1.0, 0.0, 0.0])
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(u, ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    v0 = ref - np.dot(ref, u) * u
    v = v0 / np.linalg.norm(v0)

    center = 0.5 * (p0 + p1)
    half_size = max(dist * plane_extent, 1.2)

    with torch.no_grad():
        w_tgt = evaluate_point_normal_field(
            p_tgt.to(dtype=torch.float64),
            n_tgt.to(dtype=torch.float64),
            queries,
            inv_epsilon=inv_epsilon,
            backend=backend,
        )

    # 1. Scalar Position Loss Grid
    s_vals = np.linspace(-half_size, half_size, grid_res)
    t_vals = np.linspace(-half_size, half_size, grid_res)
    S, T = np.meshgrid(s_vals, t_vals)
    loss_grid = np.zeros((grid_res, grid_res))
    n_src_tensor = torch.tensor(n_src_np[None, :], dtype=torch.float64)

    for i in range(grid_res):
        for j in range(grid_res):
            p_cand_3d = center + S[i, j] * u + T[i, j] * v
            p_cand_tensor = torch.tensor(p_cand_3d[None, :], dtype=torch.float64)
            with torch.no_grad():
                w_cand = evaluate_point_normal_field(
                    p_cand_tensor,
                    n_src_tensor,
                    queries,
                    inv_epsilon=inv_epsilon,
                    backend=backend,
                )
                loss_grid[i, j] = torch.abs(w_cand - w_tgt).mean().item()

    loss_vals = np.log10(loss_grid + eps) if use_log_scale else loss_grid

    # 2. Sampled Joint Gradients
    s_q = np.linspace(-half_size * 0.9, half_size * 0.9, quiver_res)
    t_q = np.linspace(-half_size * 0.9, half_size * 0.9, quiver_res)
    S_q, T_q = np.meshgrid(s_q, t_q)

    dp_u = np.zeros((quiver_res, quiver_res))
    dp_v = np.zeros((quiver_res, quiver_res))
    dn_u = np.zeros((quiver_res, quiver_res))
    dn_v = np.zeros((quiver_res, quiver_res))

    num_q = queries.shape[0]

    for i in range(quiver_res):
        for j in range(quiver_res):
            p_cand_3d = center + S_q[i, j] * u + T_q[i, j] * v
            p_cand_tensor = torch.tensor(p_cand_3d[None, :], dtype=torch.float64)

            with torch.no_grad():
                w_cand = evaluate_point_normal_field(
                    p_cand_tensor,
                    n_src_tensor,
                    queries,
                    inv_epsilon=inv_epsilon,
                    backend=backend,
                )
                dL_dw = torch.sign(w_cand.squeeze() - w_tgt.squeeze()) / num_q

            grad_p_3d, grad_n_3d = pointnormal_grads(
                p_cand_tensor,
                n_src_tensor,
                queries,
                dL_dw,
                inv_epsilon=inv_epsilon,
                backend=backend,
            )

            gp = grad_p_3d[0].detach().cpu().numpy()
            gn = grad_n_3d[0].detach().cpu().numpy()

            # Steepest descent components projected onto (u, v) plane
            dp_u[i, j] = -np.dot(gp, u)
            dp_v[i, j] = -np.dot(gp, v)
            dn_u[i, j] = -np.dot(gn, u)
            dn_v[i, j] = -np.dot(gn, v)

    # Normalize vectors for visualization clarity
    mag_p = np.hypot(dp_u, dp_v)
    mask_p = mag_p > 1e-12
    dp_u[mask_p] /= mag_p[mask_p]
    dp_v[mask_p] /= mag_p[mask_p]

    mag_n = np.hypot(dn_u, dn_v)
    mask_n = mag_n > 1e-12
    dn_u[mask_n] /= mag_n[mask_n]
    dn_v[mask_n] /= mag_n[mask_n]

    # Plotting
    fig, ax = plt.subplots(figsize=(9, 8))
    contour = ax.contourf(S, T, loss_vals, levels=40, cmap=plt.get_cmap(cmap_name))
    ax.contour(S, T, loss_vals, levels=20, colors="white", alpha=0.2, linewidths=0.5)

    # Position descent vectors
    ax.quiver(
        S_q,
        T_q,
        dp_u,
        dp_v,
        color="cyan",
        alpha=0.7,
        scale=30,
        headwidth=3,
        label=r"Spatial Descent $-\nabla_{p_{\mathrm{src}}} \mathcal{L}$",
    )
    # Dipole moment descent vectors
    ax.quiver(
        S_q,
        T_q,
        dn_u,
        dn_v,
        color="springgreen",
        alpha=0.85,
        scale=30,
        headwidth=3,
        label=r"Dipole Descent $-\nabla_{n_{\mathrm{src}}} \mathcal{L}$",
    )

    # Mark Target and Initial Source Positions
    s_tgt, t_tgt = np.dot(p0 - center, u), np.dot(p0 - center, v)
    s_src, t_src = np.dot(p1 - center, u), np.dot(p1 - center, v)
    ax.scatter(
        s_tgt,
        t_tgt,
        color="lime",
        s=120,
        edgecolor="black",
        zorder=10,
        label=r"Target Position ($p_{\mathrm{tgt}}$)",
    )
    ax.scatter(
        s_src,
        t_src,
        color="orange",
        s=120,
        edgecolor="black",
        zorder=10,
        label=r"Source Init ($p_{\mathrm{src}}$)",
    )

    ax.set_title(r"Joint Position-Dipole Loss Landscape & Dual Gradient Flow Field")
    ax.set_xlabel("Plane Axis S")
    ax.set_ylabel("Plane Axis T")
    cbar = fig.colorbar(contour, ax=ax, shrink=0.8)
    cbar.set_label(r"$\log_{10}$ Loss" if use_log_scale else "Loss")
    ax.legend(loc="upper right", fontsize="small")
    plt.tight_layout()
    return fig, ax


# -----------------------------------------------------------------------------
# Main Visualization Loop
# -----------------------------------------------------------------------------


def run_visualization(
    mode: str,
    num_steps: int,
    lr_p: float,
    lr_n: float,
    num_queries: int,
    iso_val: float,
    num_lines: int,
    inv_epsilon=250.0,
    show_iterations: bool = True,
    show_isosurfaces: bool = True,
    show_loss_plane: bool = True,
    show_query_quiver: bool = True,
    use_log_scale: bool = True,
    backend: str = "winder_bf",
):
    torch.manual_seed(42)
    cmap = mcolors.LinearSegmentedColormap.from_list("RedToYellow", ["red", "yellow"])

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    loss_fn = torch.nn.L1Loss()

    if mode in ["pn_same", "pn_random_pos", "pn_random_normal"]:
        p_tgt = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        if mode == "pn_same":
            n_dir = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
            n_tgt = (n_dir / torch.norm(n_dir)).unsqueeze(0)
            n_src = n_tgt.clone().detach().requires_grad_(True)
            p_src = torch.tensor(
                [[0.6, 0.6, 0.5]], dtype=torch.float64, requires_grad=True
            )
        elif mode == "pn_random_pos":
            n_tgt = torch.randn((1, 3), dtype=torch.float64) * 0.8
            n_src = (torch.randn((1, 3), dtype=torch.float64) * 0.8).requires_grad_(
                True
            )
            p_src = torch.tensor(
                [[0.6, 0.6, 0.5]], dtype=torch.float64, requires_grad=True
            )
        elif mode == "pn_random_normal":
            n_tgt = torch.randn((1, 3), dtype=torch.float64) * 0.8
            n_src = (torch.randn((1, 3), dtype=torch.float64) * 0.8).requires_grad_(
                True
            )
            p_src = (
                p_tgt.clone() + torch.randn((1, 3), dtype=torch.float64) * 0.01
            ).requires_grad_(False)

        p_src_init = p_src.clone().detach()
        n_src_init = n_src.clone().detach()

        all_pts = torch.cat([p_tgt, p_src_init], dim=0)
        min_b = all_pts.min(dim=0).values.numpy() - 0.8
        max_b = all_pts.max(dim=0).values.numpy() + 0.8
        bbox = [min_b[0], max_b[0], min_b[1], max_b[1], min_b[2], max_b[2]]

        src_fn = lambda q: evaluate_point_normal_field(
            p_src_init, n_src_init, q, inv_epsilon=inv_epsilon, backend=backend
        )
        tgt_fn = lambda q: evaluate_point_normal_field(
            p_tgt, n_tgt, q, inv_epsilon=inv_epsilon, backend=backend
        )

        queries = generate_queries(all_pts, mode="random", num_queries=num_queries)
        w_tgt = tgt_fn(queries).detach()

        # Compute initial gradient at p1 for iteration 0
        p_temp = p_src_init.clone().detach().requires_grad_(True)
        n_temp = n_src_init.clone().detach().requires_grad_(True)
        w_temp = evaluate_point_normal_field(
            p_temp, n_temp, queries, inv_epsilon=inv_epsilon
        )
        w_temp_leaf = w_temp.detach().requires_grad_(True)
        l_temp = loss_fn(w_temp_leaf, w_tgt)
        l_temp.backward()
        p_grad_init, n_grad_init = pointnormal_grads(
            p_temp,
            n_temp,
            queries,
            w_temp_leaf.grad,
            inv_epsilon=inv_epsilon,
            backend=backend,
        )
        grad_p_init = p_grad_init[0].numpy()
        grad_n_init = n_grad_init[0].numpy()

        # Per-query gradient contributions for initial state
        if show_query_quiver:
            grad_queries, q_grads = compute_per_query_gradient_contributions(
                mode, p_temp, n_temp, None, queries, w_temp_leaf.grad, inv_epsilon
            )
            render_query_gradient_quiver(
                grad_queries,
                q_grads,
                target_pos=p_tgt[0].numpy(),
                source_pos=p_src_init[0].numpy(),
                mode_name=mode,
            )

        if show_isosurfaces:
            render_field_isosurfaces(
                ax,
                tgt_fn,
                bbox,
                iso_val=iso_val,
                pos_color="forestgreen",
                neg_color="crimson",
                alpha=0.15,
                label_prefix="Target",
            )
            render_field_isosurfaces(
                ax,
                src_fn,
                bbox,
                iso_val=iso_val,
                pos_color="royalblue",
                neg_color="coral",
                alpha=0.12,
                label_prefix="Source Init",
            )

        if show_loss_plane:
            p0 = p_tgt[0].numpy()
            p1 = p_src_init[0].numpy()
            if "pn" in mode:
                render_loss_landscape_3d(
                    p_src_init,
                    n_src_init,
                    p_tgt,
                    n_tgt,
                    queries,
                    w_tgt,
                    inv_epsilon,
                    backend=backend,
                )
                render_position_loss_landscape_2d(
                    field_fn_tgt=tgt_fn,
                    p_src=p_src_init,
                    p_tgt=p_tgt,
                    n_tgt=n_tgt,
                    n_src=n_src_init,
                    queries=queries,
                    inv_epsilon=inv_epsilon,
                    backend=backend,
                )

                # Render polar loss landscape for normal orientation
                render_normal_loss_landscape_polar(
                    field_fn_tgt=tgt_fn,
                    p_src=p_src_init,
                    p_tgt=p_tgt,
                    n_tgt=n_tgt,
                    n_src_init=n_src_init,
                    queries=queries,
                    grad_n_init=grad_n_init,
                    inv_epsilon=inv_epsilon,
                    backend=backend,
                    grid_r=60,
                    grid_theta=120,
                    quiver_r=12,
                    quiver_theta=24,
                    use_log_scale=use_log_scale,
                )

        p_history = [p_src.clone().detach().numpy()[0]]
        n_history = [n_src.clone().detach().numpy()[0]]

        # Decoupled Adam optimizer setup
        optimizer = torch.optim.Adam(
                [
                    {"params": [p_src], "lr": lr_p},
                    {"params": [n_src], "lr": lr_n},
                ]
            )

        for step in range(num_steps):
            optimizer.zero_grad()

            w_src = evaluate_point_normal_field(
                p_src, n_src, queries, inv_epsilon=inv_epsilon
            )
            w_src_leaf = w_src.detach().requires_grad_(True)

            loss = loss_fn(w_src_leaf, w_tgt)
            loss.backward()
            grad_out = w_src_leaf.grad

            p_grad, n_grad = pointnormal_grads(
                p_src, n_src, queries, grad_out, inv_epsilon=inv_epsilon
            )

            p_src.grad = p_grad
            n_src.grad = n_grad

            optimizer.step()

            p_history.append(p_src.clone().detach().numpy()[0])
            n_history.append(n_src.clone().detach().numpy()[0])

        p_hist = np.array(p_history)
        n_hist = np.array(n_history)

        p_t, n_t = p_tgt[0].numpy(), n_tgt[0].numpy()
        ax.scatter(p_t[0], p_t[1], p_t[2], color="green", s=100, label="Target Point")
        ax.quiver(
            p_t[0],
            p_t[1],
            p_t[2],
            n_t[0],
            n_t[1],
            n_t[2],
            color="green",
            linewidth=3,
            arrow_length_ratio=0.25,
        )

        if show_iterations:
            ax.plot(
                p_hist[:, 0],
                p_hist[:, 1],
                p_hist[:, 2],
                color="cyan",
                linewidth=2,
                linestyle="--",
                alpha=0.9,
                label="Trajectory Path",
            )

            for i in range(num_steps + 1):
                col = cmap(i / float(num_steps))
                p_curr, n_curr = p_hist[i], n_hist[i]
                ax.scatter(p_curr[0], p_curr[1], p_curr[2], color=col, s=30)
                if i % max(1, num_steps // 3) == 0:
                    ax.quiver(
                        p_curr[0],
                        p_curr[1],
                        p_curr[2],
                        n_curr[0],
                        n_curr[1],
                        n_curr[2],
                        color=col,
                        linewidth=1,
                        arrow_length_ratio=0.25,
                    )

        ax.set_title(
            f"Point-Normal Field Optimization ({mode.replace('_', ' ').title()})"
        )

    elif mode == "triangle":
        v_tgt = torch.tensor(
            [[[0.0, 0.5, 0.0], [-0.5, -0.5, 0.0], [0.5, -0.5, 0.0]]],
            dtype=torch.float64,
        )

        v_src = (
            v_tgt.clone() + torch.tensor([0.4, 0.4, 0.3], dtype=torch.float64)
        ).requires_grad_(True)
        v_src_init = v_src.clone().detach()

        all_pts = torch.cat([v_tgt.reshape(-1, 3), v_src_init.reshape(-1, 3)], dim=0)
        min_b = all_pts.min(dim=0).values.numpy() - 0.8
        max_b = all_pts.max(dim=0).values.numpy() + 0.8
        bbox = [min_b[0], max_b[0], min_b[1], max_b[1], min_b[2], max_b[2]]

        src_fn = lambda q: evaluate_triangle_field(v_src_init, q)
        tgt_fn = lambda q: evaluate_triangle_field(v_tgt, q)

        queries = generate_queries(all_pts, mode="grid", num_queries=num_queries)
        w_tgt = tgt_fn(queries).detach()

        # Compute initial gradient w.r.t vertex 0 (p1)
        v_temp = v_src_init.clone().detach().requires_grad_(True)
        w_temp = evaluate_triangle_field(v_temp, queries)
        w_temp_leaf = w_temp.detach().requires_grad_(True)
        l_temp = loss_fn(w_temp_leaf, w_tgt)
        l_temp.backward()
        v_grad_init = triangle_grads(v_temp, queries, w_temp_leaf.grad, backend=backend)
        grad_p_init = v_grad_init[0, 0].numpy()

        # Per-query gradient contributions for initial state
        if show_query_quiver:
            quiver_queries, q_grads = compute_per_query_gradient_contributions(
                mode, None, None, v_temp, queries, w_temp_leaf.grad
            )
            render_query_gradient_quiver(
                quiver_queries,
                q_grads,
                target_pos=v_tgt[0, 0].numpy(),
                source_pos=v_src_init[0, 0].numpy(),
                mode_name=mode,
            )

        if show_isosurfaces:
            render_field_isosurfaces(
                ax,
                tgt_fn,
                bbox,
                iso_val=iso_val,
                pos_color="forestgreen",
                neg_color="crimson",
                alpha=0.15,
                label_prefix="Target",
            )
            render_field_isosurfaces(
                ax,
                src_fn,
                bbox,
                iso_val=iso_val,
                pos_color="royalblue",
                neg_color="coral",
                alpha=0.12,
                label_prefix="Source Init",
            )

        v_history = [v_src.clone().detach().numpy()[0]]
        optimizer = torch.optim.Adam([v_src], lr=lr_p)

        for step in range(num_steps):
            optimizer.zero_grad()

            w_src = evaluate_triangle_field(v_src, queries)
            w_src_leaf = w_src.detach().requires_grad_(True)

            loss = loss_fn(w_src_leaf, w_tgt)
            loss.backward()
            grad_out = w_src_leaf.grad

            v_grad = triangle_grads(v_src, queries, grad_out)

            v_src.grad = v_grad
            optimizer.step()

            v_history.append(v_src.clone().detach().numpy()[0])

        v_hist = np.array(v_history)

        poly_tgt = Poly3DCollection(
            [v_tgt[0].numpy()],
            alpha=0.35,
            facecolors="green",
            edgecolors="darkgreen",
            linewidths=2,
            label="Target Triangle",
        )
        ax.add_collection3d(poly_tgt)

        if show_iterations:
            for k in range(3):
                ax.plot(
                    v_hist[:, k, 0],
                    v_hist[:, k, 1],
                    v_hist[:, k, 2],
                    color="cyan",
                    linewidth=1.5,
                    linestyle="--",
                    alpha=0.8,
                )

            for i in range(num_steps + 1):
                col = cmap(i / float(num_steps))
                tri = v_hist[i]
                ax.scatter(tri[:, 0], tri[:, 1], tri[:, 2], color=col, s=20)
                if i % max(1, num_steps // 8) == 0 or i == num_steps:
                    poly = Poly3DCollection(
                        [tri], alpha=0.1, facecolors=col, edgecolors=col, linewidths=1.5
                    )
                    ax.add_collection3d(poly)

        ax.set_title("Triangle Winding Field Optimization")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="upper right", fontsize="small")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Winding Field Optimization Visualizer"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["pn_same", "pn_random_pos", "pn_random_normal", "triangle"],
        default="pn_same",
        help="pn_same: Point-Normal same direction | pn_random: Point-Normal random directions/scales | pn_random_normal: same position random direction | triangle: Triangles",
    )
    parser.add_argument(
        "-n", "--steps", type=int, default=60, help="Number of gradient descent steps"
    )
    parser.add_argument(
        "--lr_p", type=float, default=0.02, help="Learning rate for position"
    )
    parser.add_argument(
        "--lr_n", type=float, default=0.2, help="Learning rate for normal"
    )
    parser.add_argument(
        "--queries", type=int, default=300, help="Number of integration query points"
    )
    parser.add_argument(
        "--inv_epsilon",
        type=float,
        default=250,
        help="Iso-surface threshold for Target and Source",
    )
    parser.add_argument(
        "--iso_val",
        type=float,
        default=0.05,
        help="Iso-surface threshold for Target and Source",
    )
    parser.add_argument(
        "--iso_lines",
        type=int,
        default=40,
        help="Number of 2D log-loss isolines projected on base",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["winder_bf", "winder", "torch"],
        default="winder_bf",
        help="What method to use to compute winding numbers and gradients?",
    )

    # Toggle Flags
    parser.add_argument(
        "--show-iterations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Toggle iteration trajectory visualization",
    )
    parser.add_argument(
        "--show-isosurfaces",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Toggle 3D iso-surface rendering",
    )
    parser.add_argument(
        "--show-loss-plane",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Toggle standalone 3D loss landscape figure",
    )
    parser.add_argument(
        "--show-query-quiver",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Toggle standalone 3D quiver plot of per-query gradient contributions",
    )
    parser.add_argument(
        "--use-log-scale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Toggle log10 scaling for loss landscape",
    )

    args = parser.parse_args()

    run_visualization(
        mode=args.mode,
        num_steps=args.steps,
        lr_p=args.lr_p,
        lr_n=args.lr_n,
        num_queries=args.queries,
        iso_val=args.iso_val,
        num_lines=args.iso_lines,
        inv_epsilon=args.inv_epsilon,
        show_iterations=args.show_iterations,
        show_isosurfaces=args.show_isosurfaces,
        show_loss_plane=args.show_loss_plane,
        show_query_quiver=args.show_query_quiver,
        use_log_scale=args.use_log_scale,
        backend=args.backend,
    )
