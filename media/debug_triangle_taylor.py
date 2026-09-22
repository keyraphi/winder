"""
Rigorous validation suite for the triangle Taylor approximation.

Design principles:
  * Every claim is asserted, not printed.
  * The exact field (oosterom_vertex_gradients) is itself FD-verified
    against the winding number forward before being used as truth.
  * Scaling laws are checked with multiple offsets and averaged, not
    inferred from a single sequence.
  * Rotation and scale equivariance are tested (very sensitive to sign
    and normalization bugs).
  * Statistics are reported, not just means.
"""

import numpy as np

# =============================================================================
# IMPORT THE IMPLEMENTATION UNDER TEST
# -----------------------------------------------------------------------------
# Save your implementation as `triangle_taylor_impl.py` and import from it.
# The names must match:
#     oosterom_vertex_gradients(v0, v1, v2, q)  -> [3, 3]
#     compute_exact_ground_truth(triangle, points, weights) -> [3, 3]
#     compute_cluster_moments(points, weights) -> (x_c, G0, G1, G2)
#     compute_node_gradient_approximation(tri, xc, G0, G1, G2, order) -> [3, 3]
# =============================================================================
from debug_triangle_backward_tailor import (
    oosterom_vertex_gradients,
    compute_exact_ground_truth,
    compute_cluster_moments,
    compute_node_gradient_approximation,
)


# =============================================================================
# UTILITIES
# =============================================================================
def rel_err(a: np.ndarray, b: np.ndarray) -> float:
    """Frobenius relative error ||a - b||_F / ||b||_F."""
    denom = np.linalg.norm(b.ravel())
    if denom < 1e-300:
        return float("nan")
    return float(np.linalg.norm((a - b).ravel()) / denom)


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniform random proper rotation matrix."""
    A = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(A)
    Q *= np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def make_small_triangle(scale: float = 0.1) -> np.ndarray:
    """A well-conditioned, non-degenerate, roughly-centroid-at-origin triangle."""
    verts = np.array(
        [
            [0.7, 0.2, -0.4],
            [-0.3, 0.9, 0.1],
            [-0.1, 0.2, 0.8],
        ],
        dtype=np.float64,
    )
    verts = verts - verts.mean(axis=0, keepdims=True)
    return verts * scale


def winding_number_forward(v0, v1, v2, q) -> float:
    """Ω_T(q) = (1 / 2π) atan2(N, D) — Oosterom-Strackee."""
    a, b, c = v0 - q, v1 - q, v2 - q
    da, db, dc = np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(c)
    N = np.dot(a, np.cross(b, c))
    D = da * db * dc + np.dot(a, b) * dc + np.dot(b, c) * da + np.dot(c, a) * db
    return np.arctan2(N, D) / (2.0 * np.pi)


def report_header(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


# =============================================================================
# TEST 0 — Verify the "exact" ground truth itself.
# -----------------------------------------------------------------------------
# If oosterom_vertex_gradients is wrong, every downstream test is meaningless.
# Verify it against finite differences of the winding number forward.
# =============================================================================
def test_0_ground_truth_fd(tol: float = 1e-5) -> None:
    report_header("TEST 0: oosterom_vertex_gradients vs FD of winding number")
    rng = np.random.default_rng(0)

    max_err = 0.0
    for trial in range(200):
        triangle = rng.uniform(-1.0, 1.0, (3, 3))
        # Avoid degenerate triangles.
        if (
            np.linalg.norm(
                np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            )
            < 1e-3
        ):
            continue
        q = rng.uniform(-3.0, 3.0, 3)

        # Analytic (÷4π since F_k = 4π · ∂Ω/∂v_k).
        F = oosterom_vertex_gradients(triangle[0], triangle[1], triangle[2], q) / (
            4.0 * np.pi
        )

        # Central-difference of Ω_T w.r.t. each vertex component.
        h = 1e-6
        FD = np.zeros((3, 3))
        for k in range(3):
            for d in range(3):
                tp, tm = triangle.copy(), triangle.copy()
                tp[k, d] += h
                tm[k, d] -= h
                FD[k, d] = (
                    winding_number_forward(*tp, q) - winding_number_forward(*tm, q)
                ) / (2.0 * h)

        err = rel_err(F, FD)
        max_err = max(max_err, err)

    print(f"  max relative error across 200 trials: {max_err:.3e}")
    assert max_err < tol, (
        f"Ground truth does NOT match FD of winding number. "
        f"max_err = {max_err:.3e} > tol = {tol:.1e}"
    )
    print("  PASS")


# =============================================================================
# TEST 1 — Pointwise Taylor convergence rate (fixed offset, varying distance).
# -----------------------------------------------------------------------------
# Averaged over many random offset directions. Asserts growth = 2 / 4 / 8.
# =============================================================================
def test_1_pointwise_convergence(
    offset_mag: float = 0.4,
    R_values=(15, 7.5, 3.75, 1.875),
    num_directions: int = 16,
    tol_rel: float = 0.35,
) -> None:
    report_header("TEST 1: pointwise Taylor convergence rate")

    rng = np.random.default_rng(1)
    triangle = make_small_triangle(scale=0.1)
    x_tri = triangle.mean(axis=0)

    # Average error over many random offset directions to remove
    # direction-specific cancellation effects.
    errors = {0: [], 1: [], 2: []}
    for R in R_values:
        move_dir = np.array([0.6, 0.8, 1.0])
        move_dir /= np.linalg.norm(move_dir)
        x_c = x_tri + R * move_dir

        for _ in range(num_directions):
            d = rng.standard_normal(3)
            d /= np.linalg.norm(d)
            offset = offset_mag * d
            x_q = x_c + offset

            F_exact = oosterom_vertex_gradients(
                triangle[0], triangle[1], triangle[2], x_q
            ) / (4.0 * np.pi)

            G0, G1, G2 = 1.0, offset, np.outer(offset, offset)

            F0 = compute_node_gradient_approximation(
                triangle, x_c, G0, G1, G2, order=0
            ).astype(np.float64)
            F1 = compute_node_gradient_approximation(
                triangle, x_c, G0, G1, G2, order=1
            ).astype(np.float64)
            F2 = compute_node_gradient_approximation(
                triangle, x_c, G0, G1, G2, order=2
            ).astype(np.float64)

            errors[0].append(rel_err(F0, F_exact))
            errors[1].append(rel_err(F1, F_exact))
            errors[2].append(rel_err(F2, F_exact))

    # Reshape into [order][R_index] arrays of (num_directions,) samples.
    e0 = np.array(errors[0]).reshape(len(R_values), num_directions).mean(axis=1)
    e1 = np.array(errors[1]).reshape(len(R_values), num_directions).mean(axis=1)
    e2 = np.array(errors[2]).reshape(len(R_values), num_directions).mean(axis=1)

    print(f"\n  R values: {list(R_values)}")
    print(f"  {'R':<8} | {'err(T0)':<12} | {'err(T1)':<12} | {'err(T2)':<12}")
    print("  " + "-" * 60)
    for i, R in enumerate(R_values):
        print(f"  {R:<8.2f} | {e0[i]:<12.3e} | {e1[i]:<12.3e} | {e2[i]:<12.3e}")

    # Growth on halving distance: err(R/2) / err(R).
    def growth(errs):
        return [errs[i + 1] / errs[i] for i in range(len(errs) - 1)]

    g0 = np.array(growth(e0))
    g1 = np.array(growth(e1))
    g2 = np.array(growth(e2))

    print(f"\n  Growth per halving of R:")
    print(f"    T0: {g0.round(3).tolist()}  (expected ~2)")
    print(f"    T1: {g1.round(3).tolist()}  (expected ~4)")
    print(f"    T2: {g2.round(3).tolist()}  (expected ~8)")

    for name, g, expected in [("T0", g0, 2.0), ("T1", g1, 4.0), ("T2", g2, 8.0)]:
        mean_g = float(np.mean(g))
        assert abs(mean_g - expected) < tol_rel * expected, (
            f"Pointwise convergence broken for {name}: "
            f"mean growth {mean_g:.3f}, expected {expected:.1f} ± {tol_rel * 100:.0f}%"
        )

    print("  PASS")


# =============================================================================
# TEST 2 — Frozen cluster: fixed geometry translated to varying distance.
# -----------------------------------------------------------------------------
# Isolates the distance effect from random cluster variation.
# =============================================================================
def test_2_frozen_cluster_scaling(
    R_values=(15, 7.5, 3.75, 1.875),
    num_clusters: int = 32,
    tol_rel: float = 0.35,
) -> None:
    report_header("TEST 2: frozen-cluster convergence rate")

    rng = np.random.default_rng(2)
    triangle = make_small_triangle(scale=0.1)
    x_tri = triangle.mean(axis=0)
    move_dir = np.array([0.6, 0.8, 1.0])
    move_dir /= np.linalg.norm(move_dir)

    # Per-cluster error curves.
    per_cluster_growths = {0: [], 1: [], 2: []}

    for cluster_trial in range(num_clusters):
        # Freeze a cluster at the ORIGIN (centroid = origin).
        n_pts = rng.integers(20, 60)
        raw = rng.standard_normal((n_pts, 3))
        raw /= np.linalg.norm(raw, axis=1, keepdims=True)
        # Uniform in a ball of radius r_cluster.
        r_cluster = 0.3
        radii = r_cluster * rng.uniform(0.0, 1.0, (n_pts, 1)) ** (1.0 / 3.0)
        offsets = raw * radii
        offsets -= offsets.mean(axis=0, keepdims=True)
        weights = rng.normal(1.0, 0.5, n_pts)

        errors = {0: [], 1: [], 2: []}
        for R in R_values:
            x_c = x_tri + R * move_dir
            points = x_c + offsets
            x_c_actual, G0, G1, G2 = compute_cluster_moments(points, weights)

            # The code will use x_c_actual (the real centroid after translation).
            exact = compute_exact_ground_truth(triangle, points, weights).astype(
                np.float64
            )

            A0 = compute_node_gradient_approximation(
                triangle, x_c_actual, G0, G1, G2, order=0
            ).astype(np.float64)
            A1 = compute_node_gradient_approximation(
                triangle, x_c_actual, G0, G1, G2, order=1
            ).astype(np.float64)
            A2 = compute_node_gradient_approximation(
                triangle, x_c_actual, G0, G1, G2, order=2
            ).astype(np.float64)

            errors[0].append(rel_err(A0, exact))
            errors[1].append(rel_err(A1, exact))
            errors[2].append(rel_err(A2, exact))

        for order in (0, 1, 2):
            e = errors[order]
            growths = [e[i + 1] / e[i] for i in range(len(e) - 1)]
            per_cluster_growths[order].append(np.mean(growths))

    for name, expected in [("T0", 2.0), ("T1", 4.0), ("T2", 8.0)]:
        order = {"T0": 0, "T1": 1, "T2": 2}[name]
        g = np.array(per_cluster_growths[order])
        med = float(np.median(g))
        q10, q90 = np.percentile(g, [10, 90])
        print(f"  {name}: median growth = {med:.3f}  [p10={q10:.3f}, p90={q90:.3f}]")
        assert abs(med - expected) < tol_rel * expected, (
            f"Frozen-cluster scaling for {name}: median {med:.3f} vs {expected:.1f}"
        )

    print("  PASS")


# =============================================================================
# TEST 3 — Sign convention: T0+T1 strictly better than T0.
# =============================================================================
def test_3_sign_convention() -> None:
    report_header("TEST 3: sign convention of first-order term")

    rng = np.random.default_rng(3)
    triangle = make_small_triangle(scale=0.1)
    x_tri = triangle.mean(axis=0)

    n_trials = 200
    wins = 0
    for _ in range(n_trials):
        R = rng.uniform(4.0, 20.0)
        direction = rng.standard_normal(3)
        direction /= np.linalg.norm(direction)
        x_c = x_tri + R * direction

        offset_mag = rng.uniform(0.05, 0.3)
        off_dir = rng.standard_normal(3)
        off_dir /= np.linalg.norm(off_dir)
        offset = offset_mag * off_dir
        x_q = x_c + offset

        exact = oosterom_vertex_gradients(
            triangle[0], triangle[1], triangle[2], x_q
        ) / (4.0 * np.pi)

        G0, G1, G2 = 1.0, offset, np.outer(offset, offset)
        A0 = compute_node_gradient_approximation(
            triangle, x_c, G0, G1, G2, order=0
        ).astype(np.float64)
        A1 = compute_node_gradient_approximation(
            triangle, x_c, G0, G1, G2, order=1
        ).astype(np.float64)

        e0 = rel_err(A0, exact)
        e1 = rel_err(A1, exact)
        if e1 < e0:
            wins += 1

    frac = wins / n_trials
    print(f"  T0+T1 beats T0 in {wins}/{n_trials} trials ({frac * 100:.1f}%)")
    assert frac > 0.95, (
        "Sign of first-order term is suspect — T0+T1 is rarely better than T0. "
        "Check `e = -G1_bar` in the implementation."
    )
    print("  PASS")


# =============================================================================
# TEST 4 — Rotation equivariance.
# =============================================================================
def test_4_rotation_equivariance(tol: float = 1e-5) -> None:
    report_header("TEST 4: rotation equivariance")

    rng = np.random.default_rng(4)
    triangle = make_small_triangle(scale=0.2)
    x_c = rng.uniform(-2.0, 2.0, 3)
    G0 = rng.normal(0.0, 1.0)
    G1 = rng.normal(0.0, 0.5, 3)
    G2 = rng.normal(0.0, 0.3, (3, 3))
    G2 = 0.5 * (G2 + G2.T)

    baseline = compute_node_gradient_approximation(
        triangle, x_c, G0, G1, G2, order=2
    ).astype(np.float64)

    max_err = 0.0
    for _ in range(50):
        R = random_rotation(rng)
        triangle_r = triangle @ R.T
        x_c_r = R @ x_c
        G1_r = R @ G1
        G2_r = R @ G2 @ R.T

        A_r = compute_node_gradient_approximation(
            triangle_r, x_c_r, G0, G1_r, G2_r, order=2
        ).astype(np.float64)

        # Expected: each vertex gradient rotates.
        expected = baseline @ R.T
        err = rel_err(A_r, expected)
        max_err = max(max_err, err)

    print(f"  max rotation-equivariance error over 50 trials: {max_err:.3e}")
    assert max_err < tol, f"Rotation equivariance violated: {max_err:.3e}"
    print("  PASS")


# =============================================================================
# TEST 5 — Scale equivariance.
# =============================================================================
def test_5_scale_equivariance(tol: float = 1e-5) -> None:
    report_header("TEST 5: scale equivariance")

    rng = np.random.default_rng(5)
    triangle = make_small_triangle(scale=0.2)
    x_c = rng.uniform(-2.0, 2.0, 3)
    G0 = rng.normal(0.0, 1.0)
    G1 = rng.normal(0.0, 0.5, 3)
    G2 = rng.normal(0.0, 0.3, (3, 3))
    G2 = 0.5 * (G2 + G2.T)

    baseline = compute_node_gradient_approximation(
        triangle, x_c, G0, G1, G2, order=2
    ).astype(np.float64)

    max_err = 0.0
    for alpha in [0.1, 0.5, 2.0, 10.0]:
        A_s = compute_node_gradient_approximation(
            alpha * triangle, alpha * x_c, G0, alpha * G1, alpha**2 * G2, order=2
        ).astype(np.float64)
        expected = baseline / alpha
        err = rel_err(A_s, expected)
        max_err = max(max_err, err)

    print(f"  max scale-equivariance error over 4 alphas: {max_err:.3e}")
    assert max_err < tol, f"Scale equivariance violated: {max_err:.3e}"
    print("  PASS")


# =============================================================================
# TEST 6 — Monte Carlo with full statistics.
# =============================================================================
def test_6_monte_carlo(num_trials: int = 500) -> None:
    report_header(f"TEST 6: Monte Carlo statistics ({num_trials} trials)")

    rng = np.random.default_rng(6)

    err0 = np.zeros(num_trials)
    err1 = np.zeros(num_trials)
    err2 = np.zeros(num_trials)
    valid = 0

    for i in range(num_trials):
        triangle = rng.uniform(-1.0, 1.0, (3, 3))
        if (
            np.linalg.norm(
                np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            )
            < 1e-2
        ):
            continue
        tri_centroid = triangle.mean(axis=0)

        R = rng.uniform(4.0, 15.0)
        direction = rng.standard_normal(3)
        direction /= np.linalg.norm(direction)
        x_c = tri_centroid + R * direction

        n_pts = rng.integers(20, 60)
        radius = rng.uniform(0.1, 0.5)
        raw = rng.standard_normal((n_pts, 3))
        raw /= np.linalg.norm(raw, axis=1, keepdims=True)
        offsets = raw * radius * rng.uniform(0.0, 1.0, (n_pts, 1)) ** (1 / 3)
        points = x_c + offsets
        weights = rng.normal(1.0, 0.5, n_pts)

        x_c_actual, G0, G1, G2 = compute_cluster_moments(points, weights)
        exact = compute_exact_ground_truth(triangle, points, weights).astype(np.float64)

        if np.linalg.norm(exact.ravel()) < 1e-12:
            continue

        A0 = compute_node_gradient_approximation(
            triangle, x_c_actual, G0, G1, G2, order=0
        ).astype(np.float64)
        A1 = compute_node_gradient_approximation(
            triangle, x_c_actual, G0, G1, G2, order=1
        ).astype(np.float64)
        A2 = compute_node_gradient_approximation(
            triangle, x_c_actual, G0, G1, G2, order=2
        ).astype(np.float64)

        err0[i] = rel_err(A0, exact)
        err1[i] = rel_err(A1, exact)
        err2[i] = rel_err(A2, exact)
        valid += 1

    err0, err1, err2 = err0[:valid], err1[:valid], err2[:valid]

    def stats(name, e):
        print(f"  {name}:")
        print(f"    mean   = {np.mean(e):.3e}")
        print(f"    median = {np.median(e):.3e}")
        print(f"    p10    = {np.percentile(e, 10):.3e}")
        print(f"    p90    = {np.percentile(e, 90):.3e}")
        print(f"    max    = {np.max(e):.3e}")

    print()
    stats("Order 0", err0)
    stats("Order 1", err1)
    stats("Order 2", err2)

    # Gains.
    g01 = np.median(err0 / np.maximum(err1, 1e-30))
    g12 = np.median(err1 / np.maximum(err2, 1e-30))
    g02 = np.median(err0 / np.maximum(err2, 1e-30))
    print(f"\n  Median gain 0->1: {g01:.2f}x")
    print(f"  Median gain 1->2: {g12:.2f}x")
    print(f"  Median gain 0->2: {g02:.2f}x")

    mono = np.sum((err2 < err1) & (err1 < err0)) / valid
    print(f"\n  Strict monotonicity (T2 < T1 < T0): {mono * 100:.1f}%")

    assert mono > 0.95, "Monotonicity degraded — likely a bug."
    assert g12 > 5.0, "Second-order term adds too little gain."
    assert g01 > 5.0, "First-order term adds too little gain."
    print("  PASS")


# =============================================================================
# TEST 7 — Float32 vs float64 agreement.
# -----------------------------------------------------------------------------
# The CUDA implementation uses float32. Verify it matches a float64 equivalent
# to within float32 rounding.
# =============================================================================
def test_7_float32_agreement(num_trials: int = 200, tol: float = 1e-4) -> None:
    report_header("TEST 7: float32 output vs float64 ground truth (residual floor)")

    rng = np.random.default_rng(7)
    triangle = make_small_triangle(scale=0.1)
    x_tri = triangle.mean(axis=0)

    # Measure the FLOOR: how small does the error get when the true Taylor
    # residual is negligible? This quantifies float32 noise.
    floors = []
    for _ in range(num_trials):
        R = rng.uniform(20.0, 60.0)
        direction = rng.standard_normal(3)
        direction /= np.linalg.norm(direction)
        x_c = x_tri + R * direction
        offset = rng.uniform(0.01, 0.1) * np.array([1.0, 0.0, 0.0])
        x_q = x_c + offset

        exact = oosterom_vertex_gradients(
            triangle[0], triangle[1], triangle[2], x_q
        ) / (4.0 * np.pi)

        G0, G1, G2 = 1.0, offset, np.outer(offset, offset)
        A2 = compute_node_gradient_approximation(
            triangle, x_c, G0, G1, G2, order=2
        ).astype(np.float64)

        floors.append(rel_err(A2, exact))

    floor_med = float(np.median(floors))
    floor_max = float(np.max(floors))
    print(f"  far-field residual floor (order 2):")
    print(f"    median = {floor_med:.3e}")
    print(f"    max    = {floor_max:.3e}")
    assert floor_max < tol, (
        f"Float32 noise floor too high: {floor_max:.3e} > {tol:.1e}. "
        f"Either the implementation is not converging or float32 is too lossy."
    )
    print("  PASS")


def test_8_degenerate_triangles() -> None:
    report_header("TEST 8: degenerate and thin triangles")
    rng = np.random.default_rng(8)
    x_c = np.array([0.0, 0.0, 5.0])

    for thinness in [0.5, 0.1, 0.01, 1e-3, 1e-4]:
        # Triangle with area ~ thinness^2 * base^2
        base = 1.0
        tri = np.array(
            [
                [0.0, 0.0, 0.0],
                [base, 0.0, 0.0],
                [base / 2, thinness, 0.0],
            ]
        )

        G0, G1, G2 = 1.0, np.array([0.1, 0.0, 0.0]), np.diag([0.01, 0.01, 0.01])

        # Exact
        pts = x_c + np.array([[0.1, 0, 0], [-0.1, 0, 0], [0, 0.1, 0]])
        weights = np.array([1.0, 1.0, 1.0])
        # (recompute moments from these points for consistency)
        x_c2, G0, G1, G2 = compute_cluster_moments(pts, weights)
        exact = compute_exact_ground_truth(tri, pts, weights)

        A2 = compute_node_gradient_approximation(tri, x_c2, G0, G1, G2, order=2).astype(
            np.float64
        )
        e = rel_err(A2, exact)
        print(f"  thinness {thinness:8.1e}  →  rel err {e:.3e}")
        assert e < 1e-3, f"Approximation broken for thin triangle (thinness={thinness})"


def test_9_offcenter_reference() -> None:
    report_header("TEST 9: reference point offset — third-order scaling")

    rng = np.random.default_rng(9)
    tri = make_small_triangle(scale=0.1)
    x_tri = tri.mean(axis=0)

    R = 8.0
    direction = np.array([0.6, 0.8, 1.0])
    direction /= np.linalg.norm(direction)
    true_centroid = x_tri + R * direction

    cluster_radius = 0.05
    n_pts = 40

    offsets_frac = [0.0, 0.3, 0.7, 1.0, 2.0, 4.0]
    shifts = []
    errs = []

    for offset_frac in offsets_frac:
        offsets = rng.standard_normal((n_pts, 3)) * cluster_radius
        pts = true_centroid + offsets
        weights = rng.normal(1.0, 0.5, n_pts)

        shift_mag = offset_frac * 0.3
        shift = direction * shift_mag
        x_c_used = true_centroid + shift

        # Moments around the shifted reference.
        u = pts - x_c_used
        G0 = float(weights.sum())
        G1 = (weights[:, None] * u).sum(axis=0)
        G2 = np.einsum("j,ji,jk->ik", weights, u, u)

        exact = compute_exact_ground_truth(tri, pts, weights).astype(np.float64)
        A2 = compute_node_gradient_approximation(
            tri, x_c_used, G0, G1, G2, order=2
        ).astype(np.float64)
        e = rel_err(A2, exact)

        # |u|² averaged over the cluster, plus the shift.
        u_mag = np.sqrt(np.mean(np.sum(u * u, axis=1)))
        shifts.append(u_mag)
        errs.append(e)
        print(f"  shift = {shift_mag:.3f}  |u| = {u_mag:.3f}  err = {e:.3e}")

    shifts = np.array(shifts)
    errs = np.array(errs)

    # Fit log(err) = a + 3 log(|u|) for the shifted cases.
    mask = shifts > 0.15  # avoid the far-field where the noise floor dominates
    coeffs = np.polyfit(np.log(shifts[mask]), np.log(errs[mask]), 1)
    slope = coeffs[0]

    print(f"\n  fitted slope d(log err)/d(log |u|) = {slope:.3f}")
    print(f"  expected: 3.0 (third-order truncation dominates)")
    assert 2.4 < slope < 3.6, (
        f"Expected third-order scaling of the truncation residual, "
        f"observed slope = {slope:.3f}. If it's much less than 3, "
        f"the second-order term itself may be wrong."
    )
    print("\n  PASS")


def test_10_criterion_boundary(
    r_over_R_values=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50),
    num_trials_per_ratio: int = 32,
    num_pts_per_cluster: int = 32,
    tol_order2_at_03: float = 5e-2,
) -> None:
    """
    Measure approximation quality at the r/R ratios where the BVH criterion
    actually fires.

    Background:
      Prior tests used r/R << 1 (asymptotic far field). The production engine
      fires the Taylor approximation when a node is far enough from the
      source, i.e. when r_node / R <= 1/beta. For beta=2.3 that boundary
      sits at r/R ~ 0.43; for beta=1.8 at r/R ~ 0.55. So the operational
      regime is r/R in [0.1, 0.5], NOT the far field.

      At r/R ~ 0.4, the third-order residual is (0.4)^3 ~ 6e-2 relative
      to the leading term — well above the second-order Taylor correction.
      That is exactly what limits the engine's accuracy at realistic betas.

    Design:
      * Fix |x_c - x_tri_centroid| = R = 10 so the only swept variable is
        the ratio r_cluster / R.
      * For each r/R, sample `num_trials_per_ratio` independent clusters of
        that radius, and measure the relative error of T0 / T1 / T2 against
        the exact field.
      * Report median and p90 of the error distribution; assert a loose
        bound so the test does not flake.

    Interpretation:
      * r/R <= 0.1  : far field; T2 error should be ~ float32 noise floor
      * r/R ~ 0.2-0.3 : aggressive beta regime
      * r/R ~ 0.4-0.5 : conservative beta regime

    Mapping to the criterion:
      The BVH criterion is expressed in terms of the AABB diagonal, not
      the cluster radius, but they differ only by a constant factor of
      order sqrt(3). So "r/R = 0.4" corresponds roughly to a node whose
      AABB diagonal is 0.4 * R.
    """
    report_header("TEST 10: accuracy at the BVH criterion boundary (r/R sweep)")

    rng = np.random.default_rng(10)

    R = 10.0
    tri = make_small_triangle(scale=0.3)
    x_tri = tri.mean(axis=0)

    move_dir = np.array([0.6, 0.8, 1.0])
    move_dir /= np.linalg.norm(move_dir)
    x_c_base = x_tri + R * move_dir

    print(f"\n  R = {R:.2f}, source triangle scale = 0.3")
    print(f"  Sampling {num_trials_per_ratio} clusters per r/R value")
    print(f"  Cluster points per trial: {num_pts_per_cluster}")
    print()

    # ---------------------------------------------------------------------
    # Header
    # ---------------------------------------------------------------------
    hdr = (
        f"  {'r/R':<6} | "
        f"{'err(T0) med':>12} {'p90':>11} | "
        f"{'err(T1) med':>12} {'p90':>11} | "
        f"{'err(T2) med':>12} {'p90':>11}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    results: dict[float, dict[int, np.ndarray]] = {}

    # ---------------------------------------------------------------------
    # Sweep
    # ---------------------------------------------------------------------
    for r_over_R in r_over_R_values:
        r_cluster = r_over_R * R

        errs0, errs1, errs2 = [], [], []

        for _ in range(num_trials_per_ratio):
            # Uniform-in-ball sampling of a cluster centered at x_c_base.
            raw = rng.standard_normal((num_pts_per_cluster, 3))
            raw /= np.linalg.norm(raw, axis=1, keepdims=True)
            radii = r_cluster * rng.uniform(0.0, 1.0, (num_pts_per_cluster, 1)) ** (
                1.0 / 3.0
            )
            offsets = raw * radii
            offsets -= offsets.mean(axis=0, keepdims=True)  # centroid at origin
            pts = x_c_base + offsets
            weights = rng.normal(1.0, 0.5, num_pts_per_cluster)

            x_c, G0, G1, G2 = compute_cluster_moments(pts, weights)
            exact = compute_exact_ground_truth(tri, pts, weights).astype(np.float64)
            if np.linalg.norm(exact.ravel()) < 1e-30:
                continue

            A0 = compute_node_gradient_approximation(
                tri, x_c, G0, G1, G2, order=0
            ).astype(np.float64)
            A1 = compute_node_gradient_approximation(
                tri, x_c, G0, G1, G2, order=1
            ).astype(np.float64)
            A2 = compute_node_gradient_approximation(
                tri, x_c, G0, G1, G2, order=2
            ).astype(np.float64)

            errs0.append(rel_err(A0, exact))
            errs1.append(rel_err(A1, exact))
            errs2.append(rel_err(A2, exact))

        errs0 = np.array(errs0)
        errs1 = np.array(errs1)
        errs2 = np.array(errs2)
        results[r_over_R] = {0: errs0, 1: errs1, 2: errs2}

        print(
            f"  {r_over_R:<6.2f} | "
            f"{np.median(errs0):>12.3e} {np.percentile(errs0, 90):>11.3e} | "
            f"{np.median(errs1):>12.3e} {np.percentile(errs1, 90):>11.3e} | "
            f"{np.median(errs2):>12.3e} {np.percentile(errs2, 90):>11.3e}"
        )

    # ---------------------------------------------------------------------
    # Diagnostic summary
    # ---------------------------------------------------------------------
    print("\n  Order-by-order threshold crossings (median error):")
    for thr in (1e-1, 1e-2, 1e-3, 1e-4):
        for order in (0, 1, 2):
            crossing = None
            for r_or in r_over_R_values:
                if np.median(results[r_or][order]) < thr:
                    crossing = r_or
                    break
            if crossing is not None:
                print(
                    f"    median err(T{order}) < {thr:.0e}  for  r/R <= {crossing:.2f}"
                )
            else:
                print(f"    median err(T{order}) < {thr:.0e}  never reached in sweep")

    # ---------------------------------------------------------------------
    # Beta mapping
    # ---------------------------------------------------------------------
    print("\n  Mapping to the BVH criterion (approximate):")
    print("    beta = 1 / (r/R).  So r/R = 0.43  ->  beta = 2.3")
    print("                       r/R = 0.55  ->  beta = 1.8")
    print("                       r/R = 0.30  ->  beta = 3.3")

    # ---------------------------------------------------------------------
    # Assertion (loose, to avoid flakiness)
    # ---------------------------------------------------------------------
    err_at_03_med = float(np.median(results[0.30][2]))
    err_at_04_med = float(np.median(results[0.40][2]))
    err_at_05_med = float(np.median(results[0.50][2]))

    print(f"\n  Median order-2 error at r/R = 0.3: {err_at_03_med:.3e}")
    print(f"  Median order-2 error at r/R = 0.4: {err_at_04_med:.3e}")
    print(f"  Median order-2 error at r/R = 0.5: {err_at_05_med:.3e}")

    assert err_at_03_med < tol_order2_at_03, (
        f"Order-2 accuracy at r/R = 0.30 ({err_at_03_med:.3e}) exceeds "
        f"tolerance {tol_order2_at_03:.1e}. The engine is inaccurate "
        f"at beta <= ~3.3 and the criterion is firing too aggressively."
    )
    print("\n  PASS")


def test_11_anisotropic_G2() -> None:
    report_header("TEST 11: anisotropic cluster moments")
    rng = np.random.default_rng(11)
    tri = make_small_triangle(scale=0.1)
    x_tri = tri.mean(axis=0)
    x_c = x_tri + 5.0 * np.array([0.6, 0.8, 1.0]) / np.linalg.norm([0.6, 0.8, 1.0])

    for shape in ["sphere", "slab", "line"]:
        if shape == "sphere":
            pts = x_c + rng.standard_normal((40, 3)) * 0.1
        elif shape == "slab":
            pts = x_c + np.column_stack(
                [
                    rng.standard_normal(40) * 0.2,
                    rng.standard_normal(40) * 0.2,
                    rng.standard_normal(40) * 1e-3,
                ]
            )
        elif shape == "line":
            t = rng.standard_normal(40) * 0.2
            dir = np.array([0.6, 0.8, 0.0])
            pts = x_c + t[:, None] * dir[None, :]

        weights = rng.normal(1.0, 0.5, 40)
        x_c_actual, G0, G1, G2 = compute_cluster_moments(pts, weights)
        exact = compute_exact_ground_truth(tri, pts, weights).astype(np.float64)
        A2 = compute_node_gradient_approximation(
            tri, x_c_actual, G0, G1, G2, order=2
        ).astype(np.float64)
        e = rel_err(A2, exact)
        print(f"  {shape:8s}  →  rel err {e:.3e}")
        assert e < 1e-3, f"Anisotropic cluster degrades approximation ({shape})"


def test_12_t2_absolute_magnitude():
    """
    Verify the absolute magnitude of the second-order term against a
    finite-difference of the exact field.
    """
    report_header("TEST 12: absolute magnitude of T2 via FD of exact field")

    rng = np.random.default_rng(12)
    triangle = make_small_triangle(scale=0.1)
    x_tri = triangle.mean(axis=0)

    max_err = 0.0
    for _ in range(50):
        R = rng.uniform(6.0, 12.0)
        direction = rng.standard_normal(3)
        direction /= np.linalg.norm(direction)
        x_c = x_tri + R * direction

        # Query configuration: two-point symmetric offset, so G1 = 0, G2 ≠ 0.
        h = 0.1
        u = rng.standard_normal(3)
        u /= np.linalg.norm(u)
        pts = np.array([x_c + h * u, x_c - h * u])
        weights = np.array([1.0, 1.0])

        x_c_actual, G0, G1, G2 = compute_cluster_moments(pts, weights)

        # Isolate T2 by evaluating the exact field at +h, 0, -h and taking
        # the second difference. T0 + T1 cancel out.
        def exact_at(q):
            return oosterom_vertex_gradients(
                triangle[0], triangle[1], triangle[2], q
            ) / (4.0 * np.pi)

        F_exact_centered = exact_at(x_c)
        F_exact_plus = exact_at(x_c + h * u)
        F_exact_minus = exact_at(x_c - h * u)
        T2_fd = F_exact_plus + F_exact_minus - 2.0 * F_exact_centered

        # Compute T2 from the Taylor approximation directly.
        A_full = compute_node_gradient_approximation(
            triangle, x_c_actual, G0, G1, G2, order=2
        ).astype(np.float64)
        A_order1 = compute_node_gradient_approximation(
            triangle, x_c_actual, G0, G1, G2, order=1
        ).astype(np.float64)
        A_order0 = compute_node_gradient_approximation(
            triangle, x_c_actual, G0, G1, G2, order=0
        ).astype(np.float64)

        # T0 + T1 should equal A_order1. So T2 = A_full - A_order1.
        T2_impl = A_full - A_order1

        err = rel_err(T2_impl, T2_fd)
        max_err = max(max_err, err)

    print(f"  max |T2_impl - T2_fd|/|T2_fd| over 50 trials: {max_err:.3e}")
    assert max_err < 1e-3, (
        f"T2 magnitude incorrect: {max_err:.3e}. "
        f"Likely a coefficient error in the second-order term."
    )
    print("  PASS")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    import sys

    tests = [
        ("0: ground-truth FD", test_0_ground_truth_fd),
        ("1: pointwise convergence", test_1_pointwise_convergence),
        ("2: frozen-cluster scaling", test_2_frozen_cluster_scaling),
        ("3: sign convention", test_3_sign_convention),
        ("4: rotation equivariance", test_4_rotation_equivariance),
        ("5: scale equivariance", test_5_scale_equivariance),
        ("6: Monte Carlo statistics", test_6_monte_carlo),
        ("7: float32 residual floor", test_7_float32_agreement),
        ("8: degenerate triangles", test_8_degenerate_triangles),
        ("9: offcenter center of mass", test_9_offcenter_reference),
        ("10: criterion boundary", test_10_criterion_boundary),
        ("11: rank deficient anisotropic", test_11_anisotropic_G2),
        ("12: T2 magnitude", test_12_t2_absolute_magnitude),
    ]

    failed = []
    for name, fn in tests:
        try:
            fn()
        except AssertionError as e:
            print(f"\n  *** FAIL  [{name}]: {e}")
            failed.append(name)

    print("\n" + "=" * 88)
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    print("ALL TESTS PASSED")
