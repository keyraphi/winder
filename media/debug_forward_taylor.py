"""Forward Taylor precision analysis.

Two questions, answered rigorously:

  Q1. Is the Taylor approximation formula correct?
      Test 1 answers this: fp64 Taylor must converge to the exact fp64 field
      at the theoretical asymptotic rate. Rate for order-2 expansion of a
      1/R^2 field is (r_cluster / R)^3, i.e. 8x per halving of R.

  Q2. How much precision is lost for each dtype choice?
      Tests 2, 3, 4 answer this: at configurations where the truncation
      residual is far below the fp32 noise floor, compare each precision
      config against the fp64 Taylor reference. Any difference is due
      to precision, not to the approximation.

Usage:
    python debug_forward_taylor.py
"""

import argparse
import math
import numpy as np


# =============================================================================
# bfloat16 emulation
# =============================================================================
def to_bf16(x):
    """Round-to-nearest-even emulation of bfloat16."""
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32).copy()
    lsb = (u >> 16) & np.uint32(1)
    u = u + np.uint32(0x7FFF) + lsb
    u = u & np.uint32(0xFFFF0000)
    return u.view(np.float32)


# =============================================================================
# Precision config
# =============================================================================
class PrecisionConfig:
    """Independent knobs for each precision decision."""

    def __init__(
        self,
        name: str,
        # Storage of the multipole coefficients
        coeff_dtype=np.float32,           # for zero- and first-order
        second_coeff_dtype=np.float32,    # for second-order (bf16 in CUDA)
        use_bf16_for_second_coeff=False,
        # Arithmetic during contraction
        r_hat_dtype=np.float32,
        math_dtype=np.float32,            # zero- and first-order
        second_math_dtype=np.float32,     # second-order
        accumulate_dtype=np.float32,
    ):
        self.name = name
        self.coeff_dtype = coeff_dtype
        self.second_coeff_dtype = second_coeff_dtype
        self.use_bf16_for_second_coeff = use_bf16_for_second_coeff
        self.r_hat_dtype = r_hat_dtype
        self.math_dtype = math_dtype
        self.second_math_dtype = second_math_dtype
        self.accumulate_dtype = accumulate_dtype

    def cast_coeff0(self, x): return np.asarray(x, dtype=self.coeff_dtype)
    def cast_coeff1(self, x): return np.asarray(x, dtype=self.coeff_dtype)
    def cast_coeff2(self, x):
        if self.use_bf16_for_second_coeff:
            return to_bf16(x)
        return np.asarray(x, dtype=self.second_coeff_dtype)
    def cast_r_hat(self, x): return np.asarray(x, dtype=self.r_hat_dtype)


FP64 = PrecisionConfig("fp64",
    coeff_dtype=np.float64, second_coeff_dtype=np.float64,
    r_hat_dtype=np.float64, math_dtype=np.float64,
    second_math_dtype=np.float64, accumulate_dtype=np.float64)

FP32_ALL = PrecisionConfig("fp32 all",
    coeff_dtype=np.float32, second_coeff_dtype=np.float32,
    r_hat_dtype=np.float32, math_dtype=np.float32,
    second_math_dtype=np.float32, accumulate_dtype=np.float32)

# Isolates STORAGE precision: coefficients in fp16, math in fp32.
FP16_COEFFS = PrecisionConfig("fp16 coeffs, fp32 math",
    coeff_dtype=np.float16, second_coeff_dtype=np.float16,
    r_hat_dtype=np.float32, math_dtype=np.float32,
    second_math_dtype=np.float32, accumulate_dtype=np.float32)

# Isolates ARITHMETIC precision: coefficients in fp32, math in fp16.
FP16_MATH = PrecisionConfig("fp32 coeffs, fp16 math",
    coeff_dtype=np.float32, second_coeff_dtype=np.float32,
    r_hat_dtype=np.float16, math_dtype=np.float16,
    second_math_dtype=np.float16, accumulate_dtype=np.float32)

# Isolates ARITHMETIC precision for the second-order term only.
FP16_MATH_2ND_ONLY = PrecisionConfig("fp32/fp32 + fp16 2nd-order math",
    coeff_dtype=np.float32, second_coeff_dtype=np.float32,
    r_hat_dtype=np.float32, math_dtype=np.float32,
    second_math_dtype=np.float16, accumulate_dtype=np.float32)

# Isolates ARITHMETIC precision for zero/first order only.
FP16_MATH_1ST_ONLY = PrecisionConfig("fp16 1st-order math, fp32 2nd",
    coeff_dtype=np.float32, second_coeff_dtype=np.float32,
    r_hat_dtype=np.float16, math_dtype=np.float16,
    second_math_dtype=np.float32, accumulate_dtype=np.float32)

# Mirrors the CUDA forward path.
CUDA_DEFAULT = PrecisionConfig("CUDA default",
    coeff_dtype=np.float16, second_coeff_dtype=np.float16,
    use_bf16_for_second_coeff=True,
    r_hat_dtype=np.float16, math_dtype=np.float16,
    second_math_dtype=np.float16, accumulate_dtype=np.float32)


# =============================================================================
# Multipole accumulation (returns fp64; the config only controls storage cast)
# =============================================================================
def triangle_scaled_normal(tri):
    return 0.5 * np.cross(tri[1] - tri[0], tri[2] - tri[0])


def accumulate_multipoles(triangles, p_center):
    """Compute fp64 multipole coefficients. No precision applied yet."""
    C0 = np.zeros(3, dtype=np.float64)
    C1 = np.zeros((3, 3), dtype=np.float64)
    C2 = np.zeros((3, 3, 3), dtype=np.float64)

    for tri in triangles:
        v0, v1, v2 = tri
        n = triangle_scaled_normal(tri)
        d = (v0 + v1 + v2) / 3.0 - p_center

        m_ij = (v0 + v1) / 2.0 - p_center
        m_jk = (v1 + v2) / 2.0 - p_center
        m_ki = (v2 + v0) / 2.0 - p_center

        Ct_sym = np.array([
            m_ij[0]*m_ij[0] + m_jk[0]*m_jk[0] + m_ki[0]*m_ki[0],
            m_ij[0]*m_ij[1] + m_jk[0]*m_jk[1] + m_ki[0]*m_ki[1],
            m_ij[0]*m_ij[2] + m_jk[0]*m_jk[2] + m_ki[0]*m_ki[2],
            m_ij[1]*m_ij[1] + m_jk[1]*m_jk[1] + m_ki[1]*m_ki[1],
            m_ij[1]*m_ij[2] + m_jk[1]*m_jk[2] + m_ki[1]*m_ki[2],
            m_ij[2]*m_ij[2] + m_jk[2]*m_jk[2] + m_ki[2]*m_ki[2],
        ]) / 3.0

        Ct = np.array([
            [Ct_sym[0], Ct_sym[1], Ct_sym[2]],
            [Ct_sym[1], Ct_sym[3], Ct_sym[4]],
            [Ct_sym[2], Ct_sym[4], Ct_sym[5]],
        ])

        C0 += n
        C1 += np.outer(d, n)
        for k in range(3):
            C2[:, :, k] += 0.5 * Ct * n[k]

    return C0, C1, C2


# =============================================================================
# Contraction (mirrors the CUDA forward exactly, parameterized by precision)
# =============================================================================
def _cast_pairwise_sum(a, b, dtype):
    """Force a two-input addition through `dtype` so intermediate rounds."""
    return np.asarray(np.asarray(a, dtype=dtype) + np.asarray(b, dtype=dtype),
                      dtype=dtype)


def contract_zero(C0, r_hat, cfg):
    md = cfg.math_dtype
    a = cfg.cast_coeff0(C0).astype(md)
    r = cfg.cast_r_hat(r_hat).astype(md)
    s = _cast_pairwise_sum(a[0] * r[0], a[1] * r[1], md)
    s = _cast_pairwise_sum(s, a[2] * r[2], md)
    return float(s)


def contract_first(C1, r_hat, cfg):
    md = cfg.math_dtype
    C = cfg.cast_coeff1(C1).astype(md)
    r = cfg.cast_r_hat(r_hat).astype(md)

    tr = _cast_pairwise_sum(C[0, 0], C[1, 1], md)
    tr = _cast_pairwise_sum(tr, C[2, 2], md)

    Cxy_yx = _cast_pairwise_sum(C[0, 1], C[1, 0], md)
    Cxz_zx = _cast_pairwise_sum(C[0, 2], C[2, 0], md)
    Cyz_zy = _cast_pairwise_sum(C[1, 2], C[2, 1], md)

    q = np.asarray(0.0, dtype=md)
    q = _cast_pairwise_sum(q, C[0, 0] * r[0] * r[0], md)
    q = _cast_pairwise_sum(q, C[1, 1] * r[1] * r[1], md)
    q = _cast_pairwise_sum(q, C[2, 2] * r[2] * r[2], md)
    q = _cast_pairwise_sum(q, Cxy_yx * r[0] * r[1], md)
    q = _cast_pairwise_sum(q, Cxz_zx * r[0] * r[2], md)
    q = _cast_pairwise_sum(q, Cyz_zy * r[1] * r[2], md)

    three = np.asarray(3.0, dtype=md)
    return float(np.asarray(tr - three * q, dtype=md))


def contract_second(C2, r_hat, cfg):
    md = cfg.second_math_dtype
    C = cfg.cast_coeff2(C2).astype(md)
    r = cfg.cast_r_hat(r_hat).astype(md)

    V1x = _cast_pairwise_sum(C[0,0,0], C[0,1,1], md)
    V1x = _cast_pairwise_sum(V1x, C[0,2,2], md)
    V1y = _cast_pairwise_sum(C[1,0,0], C[1,1,1], md)
    V1y = _cast_pairwise_sum(V1y, C[1,2,2], md)
    V1z = _cast_pairwise_sum(C[2,0,0], C[2,1,1], md)
    V1z = _cast_pairwise_sum(V1z, C[2,2,2], md)

    V2x = _cast_pairwise_sum(C[0,0,0], C[1,1,0], md)
    V2x = _cast_pairwise_sum(V2x, C[2,2,0], md)
    V2y = _cast_pairwise_sum(C[0,0,1], C[1,1,1], md)
    V2y = _cast_pairwise_sum(V2y, C[2,2,1], md)
    V2z = _cast_pairwise_sum(C[0,0,2], C[1,1,2], md)
    V2z = _cast_pairwise_sum(V2z, C[2,2,2], md)

    v1_dot_r = _cast_pairwise_sum(V1x * r[0], V1y * r[1], md)
    v1_dot_r = _cast_pairwise_sum(v1_dot_r, V1z * r[2], md)
    v2_dot_r = _cast_pairwise_sum(V2x * r[0], V2y * r[1], md)
    v2_dot_r = _cast_pairwise_sum(v2_dot_r, V2z * r[2], md)

    c = np.concatenate([C[0,0,:], C[0,1,:], C[0,2,:],
                        C[1,1,:], C[1,2,:], C[2,2,:]]).astype(md)
    rxx = (r[0] * r[0]).astype(md)
    ryy = (r[1] * r[1]).astype(md)
    rzz = (r[2] * r[2]).astype(md)
    rxy = (r[0] * r[1]).astype(md)
    rxz = (r[0] * r[2]).astype(md)
    ryz = (r[1] * r[2]).astype(md)

    two = np.asarray(2.0, dtype=md)
    tensor = np.asarray(0.0, dtype=md)
    terms = [
        c[0] * rxx * r[0], c[1] * rxx * r[1], c[2] * rxx * r[2],
        c[3] * rxy * r[0] * two, c[4] * rxy * r[1] * two, c[5] * rxy * r[2] * two,
        c[6] * rxz * r[0] * two, c[7] * ryz * r[0] * two, c[8] * rxz * r[2] * two,
        c[9] * ryy * r[0], c[10] * ryy * r[1], c[11] * ryy * r[2],
        c[12] * ryz * r[0] * two, c[13] * ryz * r[1] * two, c[14] * ryz * r[2] * two,
        c[15] * rzz * r[0], c[16] * rzz * r[1], c[17] * rzz * r[2],
    ]
    for t in terms:
        tensor = _cast_pairwise_sum(tensor, t, md)

    fifteen = np.asarray(15.0, dtype=md)
    six = np.asarray(6.0, dtype=md)
    three = np.asarray(3.0, dtype=md)
    v_part = _cast_pairwise_sum(six * v1_dot_r, three * v2_dot_r, md)
    result = np.asarray(fifteen * tensor - v_part, dtype=md)
    return float(result)


def compute_node_approximation(q, p_center, C0, C1, C2, cfg):
    r = p_center - q
    R = float(np.linalg.norm(r))
    if R < 1e-20:
        return 0.0
    r_hat = cfg.cast_r_hat(r / R)

    inv_4pi = 0.07957747154
    inv_R2 = 1.0 / (R * R)
    inv_R3 = inv_R2 / R
    inv_R4 = inv_R3 / R

    c0 = contract_zero(C0, r_hat, cfg) * inv_4pi * inv_R2
    c1 = contract_first(C1, r_hat, cfg) * inv_4pi * inv_R3
    c2 = contract_second(C2, r_hat, cfg) * inv_4pi * inv_R4

    ad = cfg.accumulate_dtype
    t = np.asarray(0.0, dtype=ad)
    t = _cast_pairwise_sum(t, c0, ad)
    t = _cast_pairwise_sum(t, c1, ad)
    t = _cast_pairwise_sum(t, c2, ad)
    return float(t)


# =============================================================================
# Exact ground truth
# =============================================================================
def exact_triangle_winding(triangles, q):
    """Sum of Oosterom-Strackee winding numbers over all triangles."""
    total = 0.0
    for tri in triangles:
        v0, v1, v2 = tri
        a, b, c = v0 - q, v1 - q, v2 - q
        la, lb, lc = np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(c)
        N = float(np.dot(a, np.cross(b, c)))
        D = la * lb * lc + float(np.dot(a, b)) * lc + float(np.dot(b, c)) * la + float(np.dot(c, a)) * lb
        total += math.atan2(N, D) / (2.0 * math.pi)
    return total


# =============================================================================
# Cluster generators
# =============================================================================
def make_sphere_cluster(rng, radius, n_tri):
    """Triangles uniformly distributed in a ball of the given radius."""
    tris = []
    for _ in range(n_tri):
        # Uniform in ball
        while True:
            c = rng.standard_normal(3)
            if np.linalg.norm(c) <= 1.0:
                break
        center = c * radius
        # Small triangle around that center
        perturb = rng.standard_normal((3, 3)) * radius * 0.05
        tris.append(center + perturb)
    return np.stack(tris)


def make_slab_cluster(rng, radius, n_tri, aspect=0.01):
    tris = []
    for _ in range(n_tri):
        center = np.array([
            rng.uniform(-radius, radius),
            rng.uniform(-radius, radius),
            rng.uniform(-radius, radius) * aspect,
        ])
        perturb = rng.standard_normal((3, 3)) * radius * 0.02
        perturb[:, 2] *= aspect
        tris.append(center + perturb)
    return np.stack(tris)


def make_line_cluster(rng, radius, n_tri):
    direction = np.array([0.6, 0.8, 0.0])
    direction /= np.linalg.norm(direction)
    tris = []
    for _ in range(n_tri):
        t = rng.uniform(-radius, radius)
        center = direction * t
        perturb = rng.standard_normal((3, 3)) * radius * 0.01
        tris.append(center + perturb)
    return np.stack(tris)


# =============================================================================
# TEST 1 — Formula correctness
# =============================================================================
def test_1_formula_correctness():
    """
    Verify that the fp64 Taylor approximation converges to the exact fp64
    field at the theoretical asymptotic rate (r/R)^3.

    Uses a fixed source cluster, varying distance R. The multipole expansion
    is around the cluster center. Since the cluster is well-localised and the
    query moves away, the residual must scale as 8x per halving of R.
    """
    print("\n" + "=" * 88)
    print("  TEST 1: Taylor formula correctness (fp64 vs exact)")
    print("=" * 88)

    rng = np.random.default_rng(1)
    cluster_radius = 0.2
    triangles = make_sphere_cluster(rng, cluster_radius, n_tri=20)
    p_center = np.zeros(3)

    # Precompute fp64 multipoles once; the cluster is fixed.
    C0, C1, C2 = accumulate_multipoles(triangles, p_center)

    direction = np.array([0.6, 0.8, 1.0]); direction /= np.linalg.norm(direction)
    R_values = [20.0, 10.0, 5.0, 2.5, 1.25]

    print(f"\n  Cluster radius r = {cluster_radius}")
    print(f"  Query at q = p_center + R * direction, on-axis")
    print()
    print(f"  {'R':<8} | {'r/R':<8} | {'|approx|':<12} | {'|exact|':<12} | "
          f"{'abs err':<12} | {'rel err':<12} | {'growth':<8}")
    print("  " + "-" * 92)

    prev_err = None
    for R in R_values:
        q = direction * R
        approx = compute_node_approximation(q, p_center, C0, C1, C2, FP64)
        exact = exact_triangle_winding(triangles, q)

        abs_err = abs(approx - exact)
        rel_err = abs_err / max(abs(exact), 1e-30)
        growth = f"{abs_err / prev_err:.2f}x" if prev_err else "-"

        print(f"  {R:<8.2f} | {cluster_radius/R:<8.3f} | "
              f"{abs(approx):<12.4e} | {abs(exact):<12.4e} | "
              f"{abs_err:<12.4e} | {rel_err:<12.4e} | {growth:<8}")
        prev_err = abs_err

    print("\n  Expected growth per halving of R: 8.00x (third-order residual)")
    print("  If this table shows 8x, the Taylor formula is correct.")


# =============================================================================
# TEST 2 — Precision impact at a well-separated configuration
# =============================================================================
def test_2_precision_well_separated(n_trials=300, cluster_radius=0.2, R=20.0):
    """
    Isolate precision impact. Use a configuration where the truncation
    residual is far below the fp16 noise floor.

    At R = 20 and r = 0.2, the residual is (r/R)^3 = 1e-6 — well below the
    fp16 arithmetic noise. Any difference between a precision config and the
    fp64 reference is therefore pure precision loss.
    """
    print("\n" + "=" * 88)
    print(f"  TEST 2: precision impact, r/R = {cluster_radius/R:.4f}")
    print(f"          {n_trials} trials, {cluster_radius=}, R={R}")
    print("=" * 88)

    configs = [
        FP64,
        FP32_ALL,
        FP16_COEFFS,
        FP16_MATH,
        FP16_MATH_1ST_ONLY,
        FP16_MATH_2ND_ONLY,
        CUDA_DEFAULT,
    ]

    errs = {cfg.name: [] for cfg in configs}
    rng = np.random.default_rng(2)

    for _ in range(n_trials):
        triangles = make_sphere_cluster(rng, cluster_radius, n_tri=20)
        p_center = np.zeros(3)

        # Reference: fp64 Taylor with the exact same formula.
        C0, C1, C2 = accumulate_multipoles(triangles, p_center)

        direction = rng.standard_normal(3); direction /= np.linalg.norm(direction)
        q = direction * R

        ref = compute_node_approximation(q, p_center, C0, C1, C2, FP64)

        for cfg in configs:
            v = compute_node_approximation(q, p_center, C0, C1, C2, cfg)
            errs[cfg.name].append(abs(v - ref))

    print(f"\n  {'config':<40} | {'RMS':<12} | {'p90':<12} | {'p99':<12} | {'max':<12}")
    print("  " + "-" * 100)
    for cfg in configs:
        e = np.array(errs[cfg.name])
        rms = float(np.sqrt(np.mean(e * e)))
        p90 = float(np.percentile(e, 90))
        p99 = float(np.percentile(e, 99))
        mx = float(e.max())
        print(f"  {cfg.name:<40} | {rms:<12.4e} | {p90:<12.4e} | {p99:<12.4e} | {mx:<12.4e}")

    # Derived quantities: precision loss attributable to each decision.
    ref_errs = np.array(errs[FP64.name])
    print("\n  Interpretation:")
    for cfg in configs[1:]:
        e = np.array(errs[cfg.name])
        ratio_vs_fp32 = np.median(e) / max(np.median(errs[FP32_ALL.name]), 1e-30)
        print(f"    {cfg.name:<40} median err / fp32-all median err = {ratio_vs_fp32:.2f}x")


# =============================================================================
# TEST 3 — Precision impact vs separation ratio
# =============================================================================
def test_3_precision_vs_separation(n_trials=100):
    """
    Sweep the r/R ratio. At each ratio, measure the precision error per
    config relative to fp64. This shows how precision loss grows as the
    approximation becomes harder (query moves closer to the cluster).
    """
    print("\n" + "=" * 88)
    print("  TEST 3: precision impact vs r/R ratio")
    print("=" * 88)

    rng = np.random.default_rng(3)
    cluster_radius = 0.2

    ratios = [0.01, 0.02, 0.05, 0.1, 0.2]
    configs = [FP32_ALL, FP16_COEFFS, FP16_MATH, CUDA_DEFAULT]

    print(f"\n  {'r/R':<8} | " + " | ".join(f"{c.name[:16]:<16}" for c in configs))
    print("  " + "-" * (8 + 3 + len(configs) * 19))

    for r_over_R in ratios:
        R = cluster_radius / r_over_R
        per_cfg = {c.name: [] for c in configs}

        for _ in range(n_trials):
            triangles = make_sphere_cluster(rng, cluster_radius, n_tri=20)
            p_center = np.zeros(3)
            C0, C1, C2 = accumulate_multipoles(triangles, p_center)
            direction = rng.standard_normal(3); direction /= np.linalg.norm(direction)
            q = direction * R
            ref = compute_node_approximation(q, p_center, C0, C1, C2, FP64)
            for cfg in configs:
                v = compute_node_approximation(q, p_center, C0, C1, C2, cfg)
                per_cfg[cfg.name].append(abs(v - ref))

        row = f"  {r_over_R:<8.3f} | "
        cells = []
        for cfg in configs:
            e = np.array(per_cfg[cfg.name])
            cells.append(f"{np.median(e):<16.4e}")
        print(row + " | ".join(cells))


# =============================================================================
# TEST 4 — Precision impact vs cluster anisotropy
# =============================================================================
def test_4_precision_vs_shape(n_trials=200, cluster_radius=0.2, R=20.0):
    """
    Sweep cluster shapes: sphere, slab (aspect 0.05), line. Anisotropic
    clusters produce ill-conditioned G2 in some directions and reveal
    precision loss that isotropic clusters hide.
    """
    print("\n" + "=" * 88)
    print("  TEST 4: precision impact vs cluster shape")
    print("=" * 88)

    rng = np.random.default_rng(4)
    configs = [FP32_ALL, FP16_COEFFS, FP16_MATH, CUDA_DEFAULT]
    shapes = {
        "sphere": lambda r: make_sphere_cluster(r, cluster_radius, 20),
        "slab (0.05)": lambda r: make_slab_cluster(r, cluster_radius, 20, aspect=0.05),
        "slab (0.01)": lambda r: make_slab_cluster(r, cluster_radius, 20, aspect=0.01),
        "line": lambda r: make_line_cluster(r, cluster_radius, 20),
    }

    print(f"\n  {'shape':<14} | " + " | ".join(f"{c.name[:16]:<16}" for c in configs))
    print("  " + "-" * (14 + 3 + len(configs) * 19))

    for shape_name, gen in shapes.items():
        per_cfg = {c.name: [] for c in configs}
        for _ in range(n_trials):
            triangles = gen(rng)
            p_center = np.zeros(3)
            C0, C1, C2 = accumulate_multipoles(triangles, p_center)
            direction = rng.standard_normal(3); direction /= np.linalg.norm(direction)
            q = direction * R
            ref = compute_node_approximation(q, p_center, C0, C1, C2, FP64)
            for cfg in configs:
                v = compute_node_approximation(q, p_center, C0, C1, C2, cfg)
                per_cfg[cfg.name].append(abs(v - ref))

        row = f"  {shape_name:<14} | "
        cells = []
        for cfg in configs:
            e = np.array(per_cfg[cfg.name])
            cells.append(f"{np.median(e):<16.4e}")
        print(row + " | ".join(cells))


# =============================================================================
# Main
# =============================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test", choices=["1", "2", "3", "4", "all"], default="all")
    args = p.parse_args()

    if args.test in ("1", "all"): test_1_formula_correctness()
    if args.test in ("2", "all"): test_2_precision_well_separated()
    if args.test in ("3", "all"): test_3_precision_vs_separation()
    if args.test in ("4", "all"): test_4_precision_vs_shape()


if __name__ == "__main__":
    main()
