import numpy as np


def compute_exact_point_normal_ground_truth(
    geometry: np.ndarray,  # [2, 3]: geometry[0]=p, geometry[1]=n
    points: np.ndarray,  # [N, 3] query points q
    weights: np.ndarray,  # [N] query weights g
    epsilon: float = 1e-2,  # Regularization scale epsilon
) -> np.ndarray:  # [2, 3] output gradients [grad_p, grad_n]
    p = geometry[0].astype(np.float32)
    n = geometry[1].astype(np.float32)

    # Precomputed constants matching C++ definitions
    inv_epsilon = np.float32(1.0 / epsilon)
    inv_epsilon3 = inv_epsilon**3
    inv_pi_1_5 = np.float32(1.0 / (np.pi**1.5))
    reg_term_const = inv_epsilon3 * inv_pi_1_5
    near_field_g_denum = np.float32((1.0 / (3.0 * np.pi**1.5)) * inv_epsilon3)
    inv_4pi = np.float32(1.0 / (4.0 * np.pi))

    grad_p = np.zeros(3, dtype=np.float32)
    grad_n = np.zeros(3, dtype=np.float32)

    for q, g in zip(points, weights):
        d = p - q
        dist2 = np.dot(d, d)
        inv_dist = np.float32(1.0 / np.sqrt(dist2 + 1e-20))
        inv_dist2 = inv_dist * inv_dist
        inv_dist3 = inv_dist2 * inv_dist

        distance = dist2 * inv_dist
        t = distance * inv_epsilon

        if t < 0.1:
            scale_n = g * near_field_g_denum
            scale_d = 0.0
        else:
            if t < 2.0:
                # Replace with your actual S_regularization(t) implementation
                S_t = 1.0  # Placeholder for S_regularization(t)
                s_over_dist3 = S_t * inv_dist3
                exp_t2 = np.exp(-(t**2), dtype=np.float32)
                reg_term = exp_t2 * reg_term_const
            else:
                s_over_dist3 = inv_dist3
                reg_term = 0.0

            g_denum = inv_4pi * s_over_dist3
            dot_prod = np.dot(n, d)
            shared_factor = dot_prod * inv_dist2

            scale_n = g * g_denum
            scale_d = g * shared_factor * (reg_term - 3.0 * g_denum)

        grad_p += scale_n * n + scale_d * d
        grad_n += scale_n * d

    return np.stack([grad_p, grad_n], axis=0).astype(np.float32)


def compute_cluster_moments(points: np.ndarray, weights: np.ndarray):
    """
    Computes center of mass x_c and multi-order query moments:
    G0 = sum g_j
    G1 = sum g_j (x_j - x_c)
    G2 = sum g_j (x_j - x_c) x (x_j - x_c)
    """
    # Center of Mass (Geometric centroid of query points)
    x_c = np.mean(points, axis=0)

    # Displacements u_j = x_j - x_c
    u = points - x_c

    G0 = float(np.sum(weights))
    G1 = np.sum(weights[:, None] * u, axis=0)

    G2 = np.zeros((3, 3), dtype=np.float64)
    for u_j, g_j in zip(u, weights):
        G2 += g_j * np.outer(u_j, u_j)

    return x_c, G0, G1, G2


# =============================================================================
# 2. Float32 CUDA-Structured Taylor Gradient Function
# =============================================================================
def compute_node_gradient_approximation(
    geometry: np.ndarray,        # [2, 3]: geometry[0] = p, geometry[1] = m
    center_of_mass: np.ndarray,  # [3]: q_tilde (query cluster centroid)
    G0: float,                   # 0th-order scalar moment
    G1: np.ndarray,              # [3] 1st-order vector moment
    G2: np.ndarray,              # [3, 3] 2nd-order tensor moment
    order: int = 2,              # Expansion order (0, 1, or 2)
) -> np.ndarray:                 # [2, 3] gradients [grad_p, grad_m]
    p = geometry[0].astype(np.float32)
    m = geometry[1].astype(np.float32)
    q_c = center_of_mass.astype(np.float32)

    r_vec = q_c - p
    R = np.linalg.norm(r_vec)
    if R < 1e-10:
        return np.zeros((2, 3), dtype=np.float32)

    r_hat = r_vec / R
    m_r = np.dot(r_hat, m)

    # Global scale factor includes -1/4pi because r = q_c - p = -d
    inv_4pi = np.float32(-1.0 / (4.0 * np.pi))

    # Powers of distance decay
    inv_R2 = 1.0 / (R**2)
    inv_R3 = 1.0 / (R**3)
    inv_R4 = 1.0 / (R**4)
    inv_R5 = 1.0 / (R**5)

    # ------------------ Order 0 Contractions ------------------
    c0_n = G0 * r_hat
    c0_p = G0 * (3.0 * m_r * r_hat - m)

    grad_m = inv_R2 * c0_n
    grad_p = inv_R3 * c0_p

    # ------------------ Order 1 Contractions ------------------
    if order >= 1:
        G1 = G1.astype(np.float32)
        r_dot_G1 = np.dot(r_hat, G1)
        G1_dot_m = np.dot(G1, m)

        c1_n = G1 - 3.0 * r_dot_G1 * r_hat
        c1_p = (
            3.0 * m_r * G1
            + 3.0 * G1_dot_m * r_hat
            + 3.0 * r_dot_G1 * m
            - 15.0 * r_dot_G1 * m_r * r_hat
        )

        grad_m += inv_R3 * c1_n
        grad_p += inv_R4 * c1_p

    # ------------------ Order 2 Contractions ------------------
    if order >= 2:
        G2 = G2.astype(np.float32)
        G2_r = G2 @ r_hat
        G2_m = G2 @ m
        tr_G2 = np.trace(G2)
        q_rr = np.dot(r_hat, G2_r)
        q_rm = np.dot(r_hat, G2_m)

        # Normal contraction (Order 2)
        c_nr = 7.5 * q_rr - 1.5 * tr_G2
        c2_n = c_nr * r_hat - 3.0 * G2_r

        # Position contraction (Order 2) - CORRECTED SIGN
        c_rp = 15.0 * q_rm + (7.5 * tr_G2 - 52.5 * q_rr) * m_r
        c2_p = -15.0 * m_r * G2_r - c_rp * r_hat - c_nr * m + 3.0 * G2_m

        grad_m += inv_R4 * c2_n
        grad_p += inv_R5 * c2_p

    grad_m *= inv_4pi
    grad_p *= inv_4pi

    return np.stack([grad_p, grad_m], axis=0).astype(np.float32)

def generate_random_point_normal(scale: float = 1.0) -> np.ndarray:
    p = np.random.uniform(-scale, scale, size=(3,))
    m = np.random.normal(0.0, 1.0, size=(3,))
    m = m / np.linalg.norm(m)
    return np.stack([p, m], axis=0)


def generate_random_cluster(center: np.ndarray, radius: float, num_points: int = 50):
    offsets = np.random.normal(0.0, 1.0, size=(num_points, 3))
    offsets = offsets / np.linalg.norm(offsets, axis=1, keepdims=True)
    radii = radius * (np.random.uniform(0.0, 1.0, size=(num_points, 1)) ** (1.0 / 3.0))
    points = center + offsets * radii
    weights = np.random.normal(1.0, 0.5, size=num_points)
    return points, weights

import numpy as np


def run_point_normal_convergence_suite():
    print("=" * 88)
    print("RUNNING POINT-NORMAL DISTANCE SCALING & CONVERGENCE TEST SUITE")
    print("=" * 88)

    point_normal = np.array(
        [
            [0.2, -0.5, 0.8],  # Position p
            [0.0, 0.6, 0.8],   # Unit Normal m (normalized)
        ],
        dtype=np.float32,
    )
    point_normal[1] /= np.linalg.norm(point_normal[1])

    cluster_radius = 0.2
    direction = np.array([0.6, 0.8, 1.0])
    direction /= np.linalg.norm(direction)

    distances = [10.0, 5.0, 2.5, 1.25]

    components = [(0, "POSITION GRADIENT (p)"), (1, "NORMAL GRADIENT (m)")]

    for comp_idx, comp_name in components:
        print(f"\n--- {comp_name} ---")
        header = f"{'Distance':<9} | {'0th Rel Err':<12} {'(Halving)':<9} | {'1st Rel Err':<12} {'(Halving)':<9} | {'2nd Rel Err':<12} {'(Halving)':<9} | {'1st Gain':<9} | {'2nd Gain':<9}"
        print(header)
        print("-" * len(header))

        # Reset seed per component to guarantee evaluation on identical cluster geometries
        np.random.seed(42)
        prev_err0, prev_err1, prev_err2 = None, None, None

        for dist in distances:
            center = point_normal[0] + direction * dist
            points, weights = generate_random_cluster(
                center, cluster_radius, num_points=100
            )

            x_c, G0, G1, G2 = compute_cluster_moments(points, weights)

            grad_gt = compute_exact_point_normal_ground_truth(point_normal, points, weights)[comp_idx]
            grad_t0 = compute_node_gradient_approximation(point_normal, x_c, G0, G1, G2, order=0)[comp_idx]
            grad_t1 = compute_node_gradient_approximation(point_normal, x_c, G0, G1, G2, order=1)[comp_idx]
            grad_t2 = compute_node_gradient_approximation(point_normal, x_c, G0, G1, G2, order=2)[comp_idx]

            norm_gt = np.linalg.norm(grad_gt)
            err0 = np.linalg.norm(grad_gt - grad_t0) / norm_gt
            err1 = np.linalg.norm(grad_gt - grad_t1) / norm_gt
            err2 = np.linalg.norm(grad_gt - grad_t2) / norm_gt

            growth0 = f"{err0 / prev_err0:.2f}x" if prev_err0 is not None else "-"
            growth1 = f"{err1 / prev_err1:.2f}x" if prev_err1 is not None else "-"
            growth2 = f"{err2 / prev_err2:.2f}x" if prev_err2 is not None else "-"

            gain_1st = f"{err0 / err1:.1f}x" if err1 > 0 else "inf"
            gain_2nd = f"{err1 / err2:.1f}x" if err2 > 0 else "inf"

            print(
                f"{dist:<9.2f} | {err0:<12.3e} {growth0:<9} | {err1:<12.3e} {growth1:<9} | {err2:<12.3e} {growth2:<9} | {gain_1st:<9} | {gain_2nd:<9}"
            )

            prev_err0, prev_err1, prev_err2 = err0, err1, err2

    print(
        "\nExpected asymptotic growth when distance halves: 0th ~ 2.0x, 1st ~ 4.0x, 2nd ~ 8.0x\n"
    )

def run_point_normal_monte_carlo_suite(num_trials: int = 1000):
    print("=" * 88)
    print(f"RUNNING POINT-NORMAL MONTE CARLO RANDOM TRIALS (N={num_trials})")
    print("=" * 88)

    np.random.seed(1337)

    stats = {
        0: {
            "name": "POSITION GRADIENT (p)",
            "err0": [], "err1": [], "err2": [],
            "cos0": [], "cos1": [], "cos2": [],
            "g1": [], "g2": [], "gt": [],
            "mono": 0,
        },
        1: {
            "name": "NORMAL GRADIENT (m)",
            "err0": [], "err1": [], "err2": [],
            "cos0": [], "cos1": [], "cos2": [],
            "g1": [], "g2": [], "gt": [],
            "mono": 0,
        },
    }

    for trial in range(num_trials):
        point_normal = generate_random_point_normal(scale=2.0)

        dist = np.random.uniform(4.0, 10.0)
        dir_vec = np.random.normal(0.0, 1.0, size=3)
        dir_vec /= np.linalg.norm(dir_vec)

        cluster_center = point_normal[0] + dir_vec * dist
        cluster_radius = np.random.uniform(0.1, 0.4)

        points, weights = generate_random_cluster(
            cluster_center, cluster_radius, num_points=40
        )
        x_c, G0, G1, G2 = compute_cluster_moments(points, weights)

        grad_gt = compute_exact_point_normal_ground_truth(point_normal, points, weights)
        grad_t0 = compute_node_gradient_approximation(point_normal, x_c, G0, G1, G2, order=0)
        grad_t1 = compute_node_gradient_approximation(point_normal, x_c, G0, G1, G2, order=1)
        grad_t2 = compute_node_gradient_approximation(point_normal, x_c, G0, G1, G2, order=2)

        for comp_idx in (0, 1):
            gt_comp = grad_gt[comp_idx]
            norm_gt = np.linalg.norm(gt_comp)
            if norm_gt < 1e-12:
                continue

            t0_comp = grad_t0[comp_idx]
            t1_comp = grad_t1[comp_idx]
            t2_comp = grad_t2[comp_idx]

            norm_t0 = np.linalg.norm(t0_comp)
            norm_t1 = np.linalg.norm(t1_comp)
            norm_t2 = np.linalg.norm(t2_comp)

            # Relative Errors
            rel_err0 = np.linalg.norm(gt_comp - t0_comp) / norm_gt
            rel_err1 = np.linalg.norm(gt_comp - t1_comp) / norm_gt
            rel_err2 = np.linalg.norm(gt_comp - t2_comp) / norm_gt

            # Cosine Similarities
            cos0 = np.dot(gt_comp, t0_comp) / (norm_gt * norm_t0) if norm_t0 > 1e-12 else 0.0
            cos1 = np.dot(gt_comp, t1_comp) / (norm_gt * norm_t1) if norm_t1 > 1e-12 else 0.0
            cos2 = np.dot(gt_comp, t2_comp) / (norm_gt * norm_t2) if norm_t2 > 1e-12 else 0.0

            # Numerical stability clip [-1.0, 1.0]
            cos0 = float(np.clip(cos0, -1.0, 1.0))
            cos1 = float(np.clip(cos1, -1.0, 1.0))
            cos2 = float(np.clip(cos2, -1.0, 1.0))

            s = stats[comp_idx]
            s["err0"].append(rel_err0)
            s["err1"].append(rel_err1)
            s["err2"].append(rel_err2)

            s["cos0"].append(cos0)
            s["cos1"].append(cos1)
            s["cos2"].append(cos2)

            g1 = rel_err0 / max(rel_err1, 1e-15)
            g2 = rel_err1 / max(rel_err2, 1e-15)
            gt = rel_err0 / max(rel_err2, 1e-15)

            s["g1"].append(g1)
            s["g2"].append(g2)
            s["gt"].append(gt)

            if rel_err2 < rel_err1 < rel_err0:
                s["mono"] += 1

    for comp_idx in (0, 1):
        s = stats[comp_idx]
        total_valid = len(s["err0"])
        geom_g1 = np.exp(np.mean(np.log(s["g1"])))
        geom_g2 = np.exp(np.mean(np.log(s["g2"])))
        geom_gt = np.exp(np.mean(np.log(s["gt"])))

        print(f"\n--- {s['name']} ---")
        print("--- MEAN RELATIVE ERRORS ---")
        print(f"  0th-Order (Monopole):      {np.mean(s['err0']):.4e}")
        print(f"  1st-Order (+ Dipole):        {np.mean(s['err1']):.4e}")
        print(f"  2nd-Order (+ Quadrupole):   {np.mean(s['err2']):.4e}")

        print("\n--- COSINE SIMILARITY (DIRECTIONAL ACCURACY) ---")
        print(f"  0th-Order: Mean = {np.mean(s['cos0']):.6f}, Variance = {np.var(s['cos0']):.4e}")
        print(f"  1st-Order: Mean = {np.mean(s['cos1']):.6f}, Variance = {np.var(s['cos1']):.4e}")
        print(f"  2nd-Order: Mean = {np.mean(s['cos2']):.6f}, Variance = {np.var(s['cos2']):.4e}")

        print("\n--- ORDER-BY-ORDER ERROR REDUCTION GAINS (GEOMETRIC MEAN) ---")
        print(f"  0th -> 1st Order Gain:     {geom_g1:.2f}x error reduction")
        print(f"  1st -> 2nd Order Gain:     {geom_g2:.2f}x error reduction")
        print(f"  Total (0th -> 2nd) Gain:   {geom_gt:.2f}x total error reduction")

        print("\n--- VALIDATION METRICS ---")
        print(
            f"  Monotonicity (Err2 < Err1 < Err0): {s['mono']} / {total_valid} trials ({100.0 * s['mono'] / total_valid:.1f}%)"
        )
    print("=" * 88)

if __name__ == "__main__":
    run_point_normal_convergence_suite()
    run_point_normal_monte_carlo_suite(num_trials=1000)
