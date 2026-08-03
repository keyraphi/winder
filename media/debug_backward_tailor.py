from dataclasses import dataclass
import numpy as np
import sympy as sp

# =============================================================================
# Precision Truncation Utilities
# =============================================================================


def to_bf16(val):
    """Simulates IEEE 754 bfloat16 bit truncation with round-to-nearest-even."""
    arr = np.asarray(val, dtype=np.float32)
    u32 = arr.view(np.uint32)
    lsb = (u32 >> 16) & 1
    u32_rounded = u32 + np.uint32(0x7FFF) + lsb
    bf16_bits = u32_rounded & np.uint32(0xFFFF0000)
    return bf16_bits.view(np.float32)


def to_fp16(val):
    """Simulates IEEE 754 float16 range limits and 10-bit mantissa truncation."""
    return np.asarray(val, dtype=np.float16).astype(np.float32)


def cast_prec(val, mode: str):
    """Casts input scalar or array to the designated simulated precision."""
    if mode == "fp32":
        return np.asarray(val, dtype=np.float32)
    elif mode == "fp16":
        return to_fp16(val)
    elif mode == "bf16":
        return to_bf16(val)
    else:
        raise ValueError(f"Unknown precision mode: {mode}")


# =============================================================================
# Granular Pipeline Configuration
# =============================================================================


@dataclass
class StagePrecisionConfig:
    primary_geometry: str = "fp32"  # r_i, d_i, r_hat, w, s_i
    directional_derivs: str = "fp32"  # d_i'', r_hat', r_hat''
    cross_products: str = "fp32"  # U_k, dU_k
    alpha_terms: str = "fp32"  # alpha_k, alpha_k', alpha_k''
    global_scalars: str = "fp32"  # N, D, S, S', S'', lambda, mu, c0
    vector_variations: str = "fp32"  # V_k, dV_k, dV_k'', W_k, dW_k, dW_k''
    horner_synthesis: str = "fp32"  # Horner weights & final sum G_k


# =============================================================================
# Configurable Modular Solver
# =============================================================================


def compute_node_gradient_variations_modular(
    v0, v1, v2, q, u, cfg: StagePrecisionConfig
):
    c = cast_prec

    # 1. Primary Geometry
    p_mode = cfg.primary_geometry
    w_np = c(-u, p_mode)
    w2 = c(np.dot(w_np, w_np), p_mode)

    r0_np = c(v0 - q, p_mode)
    r1_np = c(v1 - q, p_mode)
    r2_np = c(v2 - q, p_mode)

    d0 = c(np.linalg.norm(r0_np), p_mode)
    d1 = c(np.linalg.norm(r1_np), p_mode)
    d2 = c(np.linalg.norm(r2_np), p_mode)

    r0_hat = c(r0_np / d0, p_mode)
    r1_hat = c(r1_np / d1, p_mode)
    r2_hat = c(r2_np / d2, p_mode)

    s0 = c(np.dot(r0_np, w_np), p_mode)
    s1 = c(np.dot(r1_np, w_np), p_mode)
    s2 = c(np.dot(r2_np, w_np), p_mode)

    sh0 = c(s0 / d0, p_mode)
    sh1 = c(s1 / d1, p_mode)
    sh2 = c(s2 / d2, p_mode)

    # 2. Directional Derivatives (Lagrange)
    d_mode = cfg.directional_derivs
    wx0 = c(np.cross(w_np, r0_np), d_mode)
    wx1 = c(np.cross(w_np, r1_np), d_mode)
    wx2 = c(np.cross(w_np, r2_np), d_mode)

    d0_dprime = c(np.dot(wx0, wx0) / (d0**3), d_mode)
    d1_dprime = c(np.dot(wx1, wx1) / (d1**3), d_mode)
    d2_dprime = c(np.dot(wx2, wx2) / (d2**3), d_mode)

    r0_hat_prime = c((w_np - sh0 * r0_hat) / d0, d_mode)
    r1_hat_prime = c((w_np - sh1 * r1_hat) / d1, d_mode)
    r2_hat_prime = c((w_np - sh2 * r2_hat) / d2, d_mode)

    r0_hat_dprime = c(
        ((3.0 * sh0**2 - w2) * r0_hat - 2.0 * sh0 * w_np) / (d0**2), d_mode
    )
    r1_hat_dprime = c(
        ((3.0 * sh1**2 - w2) * r1_hat - 2.0 * sh1 * w_np) / (d1**2), d_mode
    )
    r2_hat_dprime = c(
        ((3.0 * sh2**2 - w2) * r2_hat - 2.0 * sh2 * w_np) / (d2**2), d_mode
    )

    # 3. Cross Products
    x_mode = cfg.cross_products
    U0 = c(np.cross(r1_np, r2_np), x_mode)
    U1 = c(np.cross(r2_np, r0_np), x_mode)
    U2 = c(np.cross(r0_np, r1_np), x_mode)

    dU0 = c(np.cross(w_np, r2_np - r1_np), x_mode)
    dU1 = c(np.cross(w_np, r0_np - r2_np), x_mode)
    dU2 = c(np.cross(w_np, r1_np - r0_np), x_mode)

    # 4. Alpha Terms
    a_mode = cfg.alpha_terms
    r0_dot_r1 = c(np.dot(r0_np, r1_np), a_mode)
    r1_dot_r2 = c(np.dot(r1_np, r2_np), a_mode)
    r2_dot_r0 = c(np.dot(r2_np, r0_np), a_mode)

    alpha0 = c(d1 * d2 + r1_dot_r2, a_mode)
    alpha1 = c(d2 * d0 + r2_dot_r0, a_mode)
    alpha2 = c(d0 * d1 + r0_dot_r1, a_mode)

    alpha0_prime = c(sh1 * d2 + d1 * sh2 + s1 + s2, a_mode)
    alpha1_prime = c(sh2 * d0 + d2 * sh0 + s2 + s0, a_mode)
    alpha2_prime = c(sh0 * d1 + d0 * sh1 + s0 + s1, a_mode)

    alpha0_dprime = c(
        d1_dprime * d2 + d1 * d2_dprime + 2.0 * sh1 * sh2 + 2.0 * w2, a_mode
    )
    alpha1_dprime = c(
        d2_dprime * d0 + d2 * d0_dprime + 2.0 * sh2 * sh0 + 2.0 * w2, a_mode
    )
    alpha2_dprime = c(
        d0_dprime * d1 + d0 * d1_dprime + 2.0 * sh0 * sh1 + 2.0 * w2, a_mode
    )

    # 5. Global Scalar Accumulators & Quotient Ratios
    g_mode = cfg.global_scalars
    N = c(np.dot(r0_np, U0), g_mode)
    S_U = c(np.dot(U0 + U1 + U2, w_np), g_mode)

    D = c(d0 * d1 * d2 + r0_dot_r1 * d2 + r1_dot_r2 * d0 + r2_dot_r0 * d1, g_mode)
    S_V = c(
        (sh0 * d1 * d2 + d0 * sh1 * d2 + d0 * d1 * sh2)
        + (s0 + s1) * d2
        + r0_dot_r1 * sh2
        + (s1 + s2) * d0
        + r1_dot_r2 * sh0
        + (s2 + s0) * d1
        + r2_dot_r0 * sh1,
        g_mode,
    )

    D_dprime = c(
        (d0_dprime * d1 * d2 + d0 * d1_dprime * d2 + d0 * d1 * d2_dprime)
        + 2.0 * (sh0 * sh1 * d2 + sh1 * sh2 * d0 + sh2 * sh0 * d1)
        + 2.0 * w2 * (d0 + d1 + d2)
        + 2.0 * (s0 + s1) * sh2
        + 2.0 * (s1 + s2) * sh0
        + 2.0 * (s2 + s0) * sh1
        + r0_dot_r1 * d2_dprime
        + r1_dot_r2 * d0_dprime
        + r2_dot_r0 * d1_dprime,
        g_mode,
    )

    S = c(N**2 + D**2, g_mode)
    S_prime = c(2.0 * (N * S_U + D * S_V), g_mode)
    S_dprime = c(2.0 * (S_U**2 + S_V**2 + D * D_dprime), g_mode)

    lam = c(S_prime / S, g_mode)
    mu = c(S_dprime / S, g_mode)
    c0 = c(2.0 / S, g_mode)

    # 6. Vector Variations
    v_mode = cfg.vector_variations
    V0 = c(r0_hat * alpha0 + r1_np * d2 + r2_np * d1, v_mode)
    V1 = c(r1_hat * alpha1 + r2_np * d0 + r0_np * d2, v_mode)
    V2 = c(r2_hat * alpha2 + r0_np * d1 + r1_np * d0, v_mode)

    dV0 = c(
        r0_hat_prime * alpha0
        + r0_hat * alpha0_prime
        + w_np * (d1 + d2)
        + r1_np * sh2
        + r2_np * sh1,
        v_mode,
    )
    dV1 = c(
        r1_hat_prime * alpha1
        + r1_hat * alpha1_prime
        + w_np * (d0 + d2)
        + r2_np * sh0
        + r0_np * sh2,
        v_mode,
    )
    dV2 = c(
        r2_hat_prime * alpha2
        + r2_hat * alpha2_prime
        + w_np * (d0 + d1)
        + r0_np * sh1
        + r1_np * sh0,
        v_mode,
    )

    dV0_dprime = c(
        r0_hat_dprime * alpha0
        + r0_hat_prime * (2.0 * alpha0_prime)
        + r0_hat * alpha0_dprime
        + w_np * (2.0 * (sh1 + sh2))
        + r1_np * d2_dprime
        + r2_np * d1_dprime,
        v_mode,
    )
    dV1_dprime = c(
        r1_hat_dprime * alpha1
        + r1_hat_prime * (2.0 * alpha1_prime)
        + r1_hat * alpha1_dprime
        + w_np * (2.0 * (sh2 + sh0))
        + r2_np * d0_dprime
        + r0_np * d2_dprime,
        v_mode,
    )
    dV2_dprime = c(
        r2_hat_dprime * alpha2
        + r2_hat_prime * (2.0 * alpha2_prime)
        + r2_hat * alpha2_dprime
        + w_np * (2.0 * (sh0 + sh1))
        + r0_np * d1_dprime
        + r1_np * d0_dprime,
        v_mode,
    )

    W = [
        c(U0 * D - V0 * N, v_mode),
        c(U1 * D - V1 * N, v_mode),
        c(U2 * D - V2 * N, v_mode),
    ]
    dW = [
        c(dU0 * D + U0 * S_V - dV0 * N - V0 * S_U, v_mode),
        c(dU1 * D + U1 * S_V - dV1 * N - V1 * S_U, v_mode),
        c(dU2 * D + U2 * S_V - dV2 * N - V2 * S_U, v_mode),
    ]
    dW_dprime = [
        c(
            dU0 * (2.0 * S_V) + U0 * D_dprime - dV0_dprime * N - dV0 * (2.0 * S_U),
            v_mode,
        ),
        c(
            dU1 * (2.0 * S_V) + U1 * D_dprime - dV1_dprime * N - dV1 * (2.0 * S_U),
            v_mode,
        ),
        c(
            dU2 * (2.0 * S_V) + U2 * D_dprime - dV2_dprime * N - dV2 * (2.0 * S_U),
            v_mode,
        ),
    ]

    # 7. Horner Combination Weights & Final Synthesis
    h_mode = cfg.horner_synthesis
    w_W = c(1.0 - lam + lam**2 - 0.5 * mu, h_mode)
    w_dW = c(1.0 - lam, h_mode)
    w_dW2 = c(0.5, h_mode)

    inv_4pi = c(1.0 / (4.0 * np.pi), h_mode)
    scale = c(c0 * inv_4pi, h_mode)

    grads_0th, grads_1st, grads_2nd = [], [], []

    for k in range(3):
        W_k = c(W[k], h_mode)
        dW_k = c(dW[k], h_mode)
        dW2_k = c(dW_dprime[k], h_mode)

        G0 = c(scale * W_k, h_mode)
        G1 = c(scale * (W_k + dW_k - lam * W_k), h_mode)
        G2 = c(scale * (w_W * W_k + w_dW * dW_k + w_dW2 * dW2_k), h_mode)

        grads_0th.append(G0)
        grads_1st.append(G1)
        grads_2nd.append(G2)

    return grads_0th, grads_1st, grads_2nd


# =============================================================================
# Benchmarking Framework
# =============================================================================


def build_exact_oosterom_grad_evaluator():
    ax, ay, az = sp.symbols("ax ay az")
    bx, by, bz = sp.symbols("bx by bz")
    cx, cy, cz = sp.symbols("cx cy cz")

    a, b, c = sp.Matrix([ax, ay, az]), sp.Matrix([bx, by, bz]), sp.Matrix([cx, cy, cz])
    Ra, Rb, Rc = sp.sqrt(a.dot(a)), sp.sqrt(b.dot(b)), sp.sqrt(c.dot(c))
    N = a.dot(b.cross(c))
    D = Ra * Rb * Rc + a.dot(b) * Rc + b.dot(c) * Ra + c.dot(a) * Rb
    denom = N**2 + D**2

    inv_4pi = 1.0 / (4.0 * np.pi)
    F1 = (
        inv_4pi
        * 2
        * (D * b.cross(c) - N * ((Rb * Rc + b.dot(c)) * (a / Ra) + Rc * b + Rb * c))
        / denom
    )
    F2 = (
        inv_4pi
        * 2
        * (D * c.cross(a) - N * ((Rc * Ra + c.dot(a)) * (b / Rb) + Ra * c + Rc * a))
        / denom
    )
    F3 = (
        inv_4pi
        * 2
        * (D * a.cross(b) - N * ((Ra * Rb + a.dot(b)) * (c / Rc) + Rb * a + Ra * b))
        / denom
    )

    f1_num = sp.lambdify([ax, ay, az, bx, by, bz, cx, cy, cz], F1, "numpy")
    f2_num = sp.lambdify([ax, ay, az, bx, by, bz, cx, cy, cz], F2, "numpy")
    f3_num = sp.lambdify([ax, ay, az, bx, by, bz, cx, cy, cz], F3, "numpy")

    def eval_all(v1, v2, v3, q):
        args = [*(v1 - q), *(v2 - q), *(v3 - q)]
        return (
            np.array(f1_num(*args)).flatten(),
            np.array(f2_num(*args)).flatten(),
            np.array(f3_num(*args)).flatten(),
        )

    return eval_all


def run_modular_stress_test(
    cfg: StagePrecisionConfig,
    eval_exact_grads,
    num_samples=1000,
    dtype=np.float64,
):
    drop_0th_list = []
    drop_1st_list = []
    drop_2nd_list = []
    failures_2nd = 0
    np.random.seed(42)

    for i in range(num_samples):
        v0 = np.random.uniform(-2, 2, 3).astype(dtype)
        v1 = np.random.uniform(-2, 2, 3).astype(dtype)
        v2 = np.random.uniform(-2, 2, 3).astype(dtype)

        if np.linalg.norm(np.cross(v1 - v0, v2 - v0)) < 1e-3:
            continue

        q = np.random.uniform(-3, 3, 3).astype(dtype)
        u = np.random.uniform(-0.1, 0.1, 3).astype(dtype)

        u1 = u * dtype(1.0)
        u2 = u * dtype(0.25)

        g0_s1, g1_s1, g2_s1 = compute_node_gradient_variations_modular(
            v0, v1, v2, q, u1, cfg
        )
        g0_s2, g1_s2, g2_s2 = compute_node_gradient_variations_modular(
            v0, v1, v2, q, u2, cfg
        )

        gt_s1 = eval_exact_grads(
            v0.astype(np.float64),
            v1.astype(np.float64),
            v2.astype(np.float64),
            (q + u1).astype(np.float64),
        )
        gt_s2 = eval_exact_grads(
            v0.astype(np.float64),
            v1.astype(np.float64),
            v2.astype(np.float64),
            (q + u2).astype(np.float64),
        )

        err0_s1 = np.linalg.norm(gt_s1[0] - g0_s1[0])
        err0_s2 = np.linalg.norm(gt_s2[0] - g0_s2[0])
        err1_s1 = np.linalg.norm(gt_s1[0] - g1_s1[0])
        err1_s2 = np.linalg.norm(gt_s2[0] - g1_s2[0])
        err2_s1 = np.linalg.norm(gt_s1[0] - g2_s1[0])
        err2_s2 = np.linalg.norm(gt_s2[0] - g2_s2[0])

        if err0_s2 > 0:
            drop_0th_list.append(err0_s1 / err0_s2)
        if err1_s2 > 0:
            drop_1st_list.append(err1_s1 / err1_s2)
        if err2_s2 > 0:
            drop2 = err2_s1 / err2_s2
            drop_2nd_list.append(drop2)
            if drop2 < 40.0:
                failures_2nd += 1

    print(
        f"  0th-Order Step Drop (Exp ~4x):  Mean: {np.mean(drop_0th_list):6.2f}x"
        f" | Min: {np.min(drop_0th_list):5.2f}x | Max:"
        f" {np.max(drop_0th_list):6.2f}x"
    )
    print(
        f"  1st-Order Step Drop (Exp ~16x): Mean: {np.mean(drop_1st_list):6.2f}x"
        f" | Min: {np.min(drop_1st_list):5.2f}x | Max:"
        f" {np.max(drop_1st_list):6.2f}x"
    )
    print(
        f"  2nd-Order Step Drop (Exp ~64x): Mean: {np.mean(drop_2nd_list):6.2f}x"
        f" | Min: {np.min(drop_2nd_list):5.2f}x | Max:"
        f" {np.max(drop_2nd_list):6.2f}x | Failures (<40x): {failures_2nd} /"
        f" {len(drop_2nd_list)}"
    )
    return failures_2nd == 0


# =============================================================================
# Automated Precision Hypothesis Sweep
# =============================================================================

if __name__ == "__main__":
    print("Initializing Exact Ground Truth Evaluator...")
    eval_exact_grads = build_exact_oosterom_grad_evaluator()

    test_configs = {
        "01_Full_FP32_Baseline": StagePrecisionConfig(),
        "11_Directional_Derivs_fp16": StagePrecisionConfig(
            primary_geometry="fp32",
            directional_derivs="fp16",
            cross_products="fp32",
            alpha_terms="fp32",
            global_scalars="fp32",
            vector_variations="fp32",
            horner_synthesis="fp32",
        ),
        "12_Directional_Derivs_bf16": StagePrecisionConfig(
            primary_geometry="fp32",
            directional_derivs="bf16",
            cross_products="fp32",
            alpha_terms="fp32",
            global_scalars="fp32",
            vector_variations="fp32",
            horner_synthesis="fp32",
        ),
    }

    print("\n=== RUNNING GRANULAR PRECISION STRESS SWEEP (1000 SAMPLES) ===")
    for name, cfg in test_configs.items():
        print(f"\nProfile: {name}")
        run_modular_stress_test(cfg, eval_exact_grads, num_samples=1000)
