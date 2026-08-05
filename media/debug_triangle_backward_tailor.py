import numpy as np
import math
import sympy as sp


# =============================================================================
# 1. Exact Oosterom-Strackee Ground Truth (NumPy Implementation)
# =============================================================================
def oosterom_vertex_gradients(
    v0: np.ndarray, v1: np.ndarray, v2: np.ndarray, q: np.ndarray
) -> np.ndarray:
    """
    Computes exact vertex response fields (F0, F1, F2) for a single query source q.
    Returns array of shape (3, 3) representing [F0, F1, F2].
    """
    a = v0 - q
    b = v1 - q
    c = v2 - q

    da, db, dc = np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(c)

    U0 = np.cross(b, c)
    U1 = np.cross(c, a)
    U2 = np.cross(a, b)

    N = np.dot(a, U0)
    D = da * db * dc + np.dot(a, b) * dc + np.dot(b, c) * da + np.dot(c, a) * db
    S = N**2 + D**2

    if S < 1e-12:
        return np.zeros((3, 3))

    alpha0 = db * dc + np.dot(b, c)
    alpha1 = dc * da + np.dot(c, a)
    alpha2 = da * db + np.dot(a, b)

    V0 = alpha0 * (a / da) + dc * b + db * c
    V1 = alpha1 * (b / db) + da * c + dc * a
    V2 = alpha2 * (c / dc) + db * a + da * b

    W0 = D * U0 - N * V0
    W1 = D * U1 - N * V1
    W2 = D * U2 - N * V2

    F0 = (2.0 / S) * W0
    F1 = (2.0 / S) * W1
    F2 = (2.0 / S) * W2

    return np.array([F0, F1, F2])


def compute_exact_ground_truth(
    triangle: np.ndarray, points: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """
    Sums exact Oosterom-Strackee responses over all point sources in the cluster.
    dL/dv = (1 / 4pi) * sum_j (g_j * F_k(v - x_j))
    """
    grad_exact = np.zeros((3, 3), dtype=np.float64)
    v0, v1, v2 = triangle[0], triangle[1], triangle[2]

    for x_j, g_j in zip(points, weights):
        F_k = oosterom_vertex_gradients(v0, v1, v2, x_j)
        grad_exact += g_j * F_k

    return (1.0 / (4.0 * np.pi)) * grad_exact


# =============================================================================
# 2. Moments Calculation from Query Points
# =============================================================================
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
# 1. Float32 CUDA-like Vec3 Primitive Struct
# =============================================================================
class Vec3:
    __slots__ = ("x", "y", "z")

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = np.float32(x)
        self.y = np.float32(y)
        self.z = np.float32(z)

    def __add__(self, o):
        return Vec3(self.x + o.x, self.y + o.y, self.z + o.z)

    def __sub__(self, o):
        return Vec3(self.x - o.x, self.y - o.y, self.z - o.z)

    def __neg__(self):
        return Vec3(-self.x, -self.y, -self.z)

    def __mul__(self, s):
        s_f = np.float32(s)
        return Vec3(self.x * s_f, self.y * s_f, self.z * s_f)

    def __rmul__(self, s):
        s_f = np.float32(s)
        return Vec3(self.x * s_f, self.y * s_f, self.z * s_f)

    def __truediv__(self, s):
        inv = np.float32(1.0) / np.float32(s)
        return Vec3(self.x * inv, self.y * inv, self.z * inv)

    def dot(self, o) -> np.float32:
        return np.float32(self.x * o.x + self.y * o.y + self.z * o.z)

    def cross(self, o):
        return Vec3(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )

    def norm(self) -> np.float32:
        return np.float32(np.sqrt(self.x * self.x + self.y * self.y + self.z * self.z))

    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float32)

    @staticmethod
    def from_numpy(a):
        return Vec3(a[0], a[1], a[2])


# =============================================================================
# 2. Float32 CUDA-Structured Taylor Gradient Function
# =============================================================================
def compute_node_gradient_approximation(
    triangle: np.ndarray,  # [3, 3] geometry
    center_of_mass: np.ndarray,  # [3]
    G0: float,  # Scalar half
    G1: np.ndarray,  # [3] Vec3_f16
    G2: np.ndarray,  # [3, 3] Mat3x3_bf16
    order: int = 2,  # 0, 1, or 2
) -> np.ndarray:
    """Simulates CUDA __device__ gradient computation in pure float32.
    """
    # -------------------------------------------------------------------------
    # 1. Inputs, Half-Precision Conversion & Non-Dimensionalization
    # -------------------------------------------------------------------------
    # Geometry converted to float32
    v = [Vec3.from_numpy(triangle[i].astype(np.float32)) for i in range(3)]
    com = Vec3.from_numpy(center_of_mass.astype(np.float32))

    # Convert G0, G1, G2 to half-precision (float16) first, then float32 registers
    G0_f32 = np.float32(G0)
    G1_vec = Vec3.from_numpy(G1.astype(np.float32))
    G2_f16 = G2.astype(np.float32)

    # G0_f32 = np.float32(np.float16(G0))
    # G1_f16 = G1.astype(np.float16)
    # G1_vec = Vec3.from_numpy(G1_f16.astype(np.float32))
    #
    # def _to_bf16(arr):
    #     u32 = arr.astype(np.float32).view(np.uint32)
    #     lsb = (u32 >> 16) & 1
    #     u32 += 0x7FFF + lsb
    #     return (u32 & 0xFFFF0000).view(np.float32)
    #
    # G2_f16 = _to_bf16(G2)

    r = [v[i] - com for i in range(3)]
    d_raw = [r[i].norm() for i in range(3)]

    eps = np.float32(1e-12)
    L_ref = max(max(d_raw[0], d_raw[1]), d_raw[2])
    if L_ref < eps:
        L_ref = eps

    inv_L = np.float32(1.0) / L_ref
    inv_L2 = inv_L * inv_L

    r_bar = [r[i] * inv_L for i in range(3)]
    d = [d_raw[i] * inv_L for i in range(3)]
    r_hat = [r_bar[i] / d[i] for i in range(3)]

    G0_bar = G0_f32
    G1_bar = G1_vec * inv_L

    g00 = np.float32(G2_f16[0, 0]) * inv_L2
    g11 = np.float32(G2_f16[1, 1]) * inv_L2
    g22 = np.float32(G2_f16[2, 2]) * inv_L2
    g01 = np.float32(G2_f16[0, 1]) * inv_L2
    g12 = np.float32(G2_f16[1, 2]) * inv_L2
    g20 = np.float32(G2_f16[2, 0]) * inv_L2

    # -------------------------------------------------------------------------
    # 2. Base 0th-Order Field Evaluation (T0)
    # -------------------------------------------------------------------------
    U = [Vec3()] * 3
    alpha = [np.float32(0.0)] * 3
    V = [Vec3()] * 3

    for p in range(3):
        j = (p + 1) % 3
        m = (p + 2) % 3

        U[p] = r_bar[j].cross(r_bar[m])
        dot_jm = r_bar[j].dot(r_bar[m])
        alpha[p] = d[j] * d[m] + dot_jm
        V[p] = alpha[p] * r_hat[p] + d[m] * r_bar[j] + d[j] * r_bar[m]

    N = r_bar[0].dot(U[0])

    dot12 = r_bar[1].dot(r_bar[2])
    dot20 = r_bar[2].dot(r_bar[0])
    dot01 = r_bar[0].dot(r_bar[1])
    D = d[0] * d[1] * d[2] + dot12 * d[0] + dot20 * d[1] + dot01 * d[2]

    S = N * N + D * D
    inv_S = np.float32(1.0) / S
    inv_S2 = inv_S * inv_S
    inv_S3 = inv_S2 * inv_S

    W = [D * U[p] - N * V[p] for p in range(3)]
    T0 = [G0_bar * (np.float32(2.0) * inv_S) * W[p] for p in range(3)]

    pi_f32 = np.float32(np.pi)
    scale = np.float32(1.0) / (np.float32(4.0) * pi_f32 * L_ref)

    if order == 0:
        return np.array(
            [(scale * T0[p]).to_numpy() for p in range(3)], dtype=np.float32
        )

    U_sum = U[0] + U[1] + U[2]
    V_sum = V[0] + V[1] + V[2]

    # -------------------------------------------------------------------------
    # 3. Exact 1st Directional Derivative Operator (T1)
    # -------------------------------------------------------------------------
    e = -G1_bar

    d_de = [r_hat[p].dot(e) for p in range(3)]
    r_hat_de = [(e - d_de[p] * r_hat[p]) / d[p] for p in range(3)]

    U_de = [Vec3()] * 3
    alpha_de = [np.float32(0.0)] * 3
    V_de = [Vec3()] * 3

    for p in range(3):
        j = (p + 1) % 3
        m = (p + 2) % 3

        U_de[p] = (r_bar[j] - r_bar[m]).cross(e)
        alpha_de[p] = d[m] * d_de[j] + d[j] * d_de[m] + (r_bar[j] + r_bar[m]).dot(e)
        V_de[p] = (
            alpha_de[p] * r_hat[p]
            + alpha[p] * r_hat_de[p]
            + d_de[m] * r_bar[j]
            + d_de[j] * r_bar[m]
            + (d[j] + d[m]) * e
        )

    N_de = U_sum.dot(e)
    D_de = V_sum.dot(e)
    S_de = np.float32(2.0) * N * N_de + np.float32(2.0) * D * D_de

    T1 = [Vec3()] * 3
    for p in range(3):
        W_de_p = (D_de * U[p] + D * U_de[p]) - (N_de * V[p] + N * V_de[p])
        T1[p] = (np.float32(2.0) * inv_S) * W_de_p - (
            np.float32(2.0) * S_de * inv_S2
        ) * W[p]

    if order == 1:
        return np.array(
            [(scale * (T0[p] + T1[p])).to_numpy() for p in range(3)], dtype=np.float32
        )

    # -------------------------------------------------------------------------
    # 4. Exact 2nd Directional Derivative Operator (T2 via Polarization)
    # -------------------------------------------------------------------------
    def eval_dir_deriv_2(u: Vec3):
        d_du = [r_hat[p].dot(u) for p in range(3)]
        d2_du2 = [(np.float32(1.0) - d_du[p] * d_du[p]) / d[p] for p in range(3)]

        r_hat_du = [(u - d_du[p] * r_hat[p]) / d[p] for p in range(3)]
        r_hat2_du2 = [
            (-np.float32(2.0) * d_du[p] * r_hat_du[p] - d2_du2[p] * r_hat[p]) / d[p]
            for p in range(3)
        ]

        U_du = [Vec3()] * 3
        alpha_du = [np.float32(0.0)] * 3
        alpha2_du2 = [np.float32(0.0)] * 3
        V_du = [Vec3()] * 3
        V2_du2 = [Vec3()] * 3

        for p in range(3):
            j = (p + 1) % 3
            m = (p + 2) % 3

            U_du[p] = (r_bar[j] - r_bar[m]).cross(u)
            alpha_du[p] = d[m] * d_du[j] + d[j] * d_du[m] + (r_bar[j] + r_bar[m]).dot(u)
            alpha2_du2[p] = (
                d2_du2[j] * d[m]
                + np.float32(2.0) * d_du[j] * d_du[m]
                + d[j] * d2_du2[m]
                + np.float32(2.0)
            )

            V_du[p] = (
                alpha_du[p] * r_hat[p]
                + alpha[p] * r_hat_du[p]
                + d_du[m] * r_bar[j]
                + d_du[j] * r_bar[m]
                + (d[j] + d[m]) * u
            )

            V2_du2[p] = (
                alpha2_du2[p] * r_hat[p]
                + np.float32(2.0) * alpha_du[p] * r_hat_du[p]
                + alpha[p] * r_hat2_du2[p]
                + d2_du2[m] * r_bar[j]
                + d2_du2[j] * r_bar[m]
                + np.float32(2.0) * d_du[m] * u
                + np.float32(2.0) * d_du[j] * u
            )

        N_du = U_sum.dot(u)
        D_du = V_sum.dot(u)
        D2_du2 = V_du[0].dot(u) + V_du[1].dot(u) + V_du[2].dot(u)

        S_du = np.float32(2.0) * N * N_du + np.float32(2.0) * D * D_du
        S2_du2 = (
            np.float32(2.0) * (N_du * N_du)
            + np.float32(2.0) * (D_du * D_du)
            + np.float32(2.0) * D * D2_du2
        )

        Q = [Vec3()] * 3
        for p in range(3):
            W_du_p = (D_du * U[p] + D * U_du[p]) - (N_du * V[p] + N * V_du[p])
            W2_du2_p = (D2_du2 * U[p] + np.float32(2.0) * D_du * U_du[p]) - (
                np.float32(2.0) * N_du * V_du[p] + N * V2_du2[p]
            )

            Q[p] = (
                (np.float32(2.0) * inv_S) * W2_du2_p
                - (np.float32(4.0) * S_du * inv_S2) * W_du_p
                - (np.float32(2.0) * S2_du2 * inv_S2) * W[p]
                + (np.float32(4.0) * S_du * S_du * inv_S3) * W[p]
            )

        return Q

    # Polarization weights
    w_e0 = np.float32(0.5) * (g00 - g01 - g20)
    w_e1 = np.float32(0.5) * (g11 - g01 - g12)
    w_e2 = np.float32(0.5) * (g22 - g12 - g20)
    w_e01 = np.float32(0.5) * g01
    w_e12 = np.float32(0.5) * g12
    w_e20 = np.float32(0.5) * g20

    T2 = [Vec3(), Vec3(), Vec3()]

    def accumulate_Q(u: Vec3, weight: np.float32):
        Q = eval_dir_deriv_2(u)
        for p in range(3):
            T2[p] = T2[p] + weight * Q[p]

    accumulate_Q(Vec3(1.0, 0.0, 0.0), w_e0)
    accumulate_Q(Vec3(0.0, 1.0, 0.0), w_e1)
    accumulate_Q(Vec3(0.0, 0.0, 1.0), w_e2)
    accumulate_Q(Vec3(1.0, 1.0, 0.0), w_e01)
    accumulate_Q(Vec3(0.0, 1.0, 1.0), w_e12)
    accumulate_Q(Vec3(1.0, 0.0, 1.0), w_e20)

    # -------------------------------------------------------------------------
    # 5. Final Assembly
    # -------------------------------------------------------------------------
    return np.array(
        [(scale * (T0[p] + T1[p] + T2[p])).to_numpy() for p in range(3)],
        dtype=np.float32,
    )


# =============================================================================
# 3. Taylor Approximation Function (Supports Order Selection 0, 1, 2)
# =============================================================================
def compute_taylor_gradient(
    triangle: np.ndarray,  # shape [3, 3] (vertices v_0, v_1, v_2)
    center_of_mass: np.ndarray,  # shape [3]    (center x_c)
    G0: float,  # 0th-order moment (scalar)
    G1: np.ndarray,  # 1st-order moment (shape [3])
    G2: np.ndarray,  # 2nd-order moment (shape [3, 3])
    order: int = 2,  # Approximation order (0, 1, or 2)
) -> np.ndarray:
    """Computes the exact analytical Taylor series loss gradient w.r.t. triangle

    vertex positions v_0, v_1, v_2 using pure directional differential
    calculus.

    All terms T0, T1, T2 are evaluated in dimensionless space and reassembled
    with the unified physical scale factor 1 / (4 * pi * L_ref).
    """
    # -------------------------------------------------------------------------
    # 1. Non-Dimensionalization
    # -------------------------------------------------------------------------
    r = triangle - center_of_mass  # shape [3, 3]
    r_norms = np.linalg.norm(r, axis=1)
    L_ref = np.max(r_norms)
    if L_ref < 1e-12:
        L_ref = 1e-12

    r_bar = r / L_ref
    G0_bar = G0
    G1_bar = G1 / L_ref
    G2_bar = G2 / (L_ref**2)

    # -------------------------------------------------------------------------
    # 2. Base 0th-Order Field Evaluation (T0)
    # -------------------------------------------------------------------------
    d = np.linalg.norm(r_bar, axis=1)  # shape [3]
    r_hat = r_bar / d[:, None]  # shape [3, 3]

    # Cyclic permutation mapping: k=0 -> (j=1, m=2), k=1 -> (j=2, m=0), k=2 -> (j=0, m=1)
    j_idx = np.array([1, 2, 0])
    m_idx = np.array([2, 0, 1])

    U = np.zeros((3, 3))
    dot_jm = np.zeros(3)
    for k in range(3):
        j, m = j_idx[k], m_idx[k]
        U[k] = np.cross(r_bar[j], r_bar[m])
        dot_jm[k] = np.dot(r_bar[j], r_bar[m])

    N = np.dot(r_bar[0], U[0])  # det([r_bar_0, r_bar_1, r_bar_2])
    D = d[0] * d[1] * d[2] + np.sum(dot_jm * d)
    S = N**2 + D**2

    alpha = np.zeros(3)
    V = np.zeros((3, 3))
    W = np.zeros((3, 3))
    F0 = np.zeros((3, 3))

    for k in range(3):
        j, m = j_idx[k], m_idx[k]
        alpha[k] = d[j] * d[m] + dot_jm[k]
        V[k] = alpha[k] * r_hat[k] + d[m] * r_bar[j] + d[j] * r_bar[m]
        W[k] = D * U[k] - N * V[k]
        F0[k] = (2.0 / S) * W[k]

    T0 = G0_bar * F0
    if order == 0:
        return (1.0 / (4.0 * np.pi * L_ref)) * T0

    # -------------------------------------------------------------------------
    # 3. Exact 1st Directional Derivative Operator (T1)
    # -------------------------------------------------------------------------
    def _dir_deriv_1(e: np.ndarray) -> np.ndarray:
        d_de = np.zeros(3)
        r_hat_de = np.zeros((3, 3))
        U_de = np.zeros((3, 3))
        alpha_de = np.zeros(3)
        V_de = np.zeros((3, 3))

        for p in range(3):
            d_de[p] = np.dot(r_hat[p], e)
            r_hat_de[p] = (1.0 / d[p]) * (e - d_de[p] * r_hat[p])

        for k in range(3):
            j, m = j_idx[k], m_idx[k]
            U_de[k] = np.cross(r_bar[j] - r_bar[m], e)
            alpha_de[k] = (
                d[m] * d_de[j] + d[j] * d_de[m] + np.dot(r_bar[j] + r_bar[m], e)
            )
            V_de[k] = (
                alpha_de[k] * r_hat[k]
                + alpha[k] * r_hat_de[k]
                + d_de[m] * r_bar[j]
                + d_de[j] * r_bar[m]
                + (d[j] + d[m]) * e
            )

        N_de = np.sum([np.dot(U[p], e) for p in range(3)])
        D_de = np.sum([np.dot(V[p], e) for p in range(3)])

        S_de = 2.0 * N * N_de + 2.0 * D * D_de
        F_de = np.zeros((3, 3))

        for k in range(3):
            W_de = (D_de * U[k] + D * U_de[k]) - (N_de * V[k] + N * V_de[k])
            F_de[k] = (2.0 / S) * W_de - (2.0 * S_de / (S**2)) * W[k]

        return F_de

    T1 = _dir_deriv_1(-G1_bar)
    if order == 1:
        return (1.0 / (4.0 * np.pi * L_ref)) * (T0 + T1)

    # -------------------------------------------------------------------------
    # 4. Exact 2nd-Order Contraction via Batched Polarization (T2)
    # -------------------------------------------------------------------------
    # Stack all 6 polarization basis vectors into a single (6, 3) matrix
    U_batch = np.array(
        [
            [1.0, 0.0, 0.0],  # e0
            [0.0, 1.0, 0.0],  # e1
            [0.0, 0.0, 1.0],  # e2
            [1.0, 1.0, 0.0],  # e0 + e1
            [0.0, 1.0, 1.0],  # e1 + e2
            [1.0, 0.0, 1.0],  # e2 + e0
        ]
    )

    # Contraction weights corresponding to: 2 * B(e_i, e_j) = Q(e_i+e_j) - Q(e_i) - Q(e_j)
    g00, g11, g22 = G2_bar[0, 0], G2_bar[1, 1], G2_bar[2, 2]
    g01, g12, g20 = G2_bar[0, 1], G2_bar[1, 2], G2_bar[2, 0]
    weights = 0.5 * np.array(
        [
            g00 - g01 - g20,
            g11 - g01 - g12,
            g22 - g12 - g20,
            g01,
            g12,
            g20,
        ]
    )

    # Vectorized directional evaluations across all 6 directions simultaneously
    d_du = np.zeros((3, 6))
    d2_du2 = np.zeros((3, 6))
    r_hat_du = np.zeros((3, 6, 3))
    r_hat2_du2 = np.zeros((3, 6, 3))

    for p in range(3):
        d_du[p] = U_batch @ r_hat[p]  # shape (6,)
        d2_du2[p] = (1.0 / d[p]) * (1.0 - d_du[p] ** 2)  # shape (6,)
        r_hat_du[p] = (1.0 / d[p]) * (
            U_batch - d_du[p, :, None] * r_hat[p]
        )  # shape (6, 3)
        r_hat2_du2[p] = (1.0 / d[p]) * (
            -2.0 * d_du[p, :, None] * r_hat_du[p] - d2_du2[p, :, None] * r_hat[p]
        )  # shape (6, 3)

    U_du = np.zeros((3, 6, 3))
    alpha_du = np.zeros((3, 6))
    alpha2_du2 = np.zeros((3, 6))
    V_du = np.zeros((3, 6, 3))
    V2_du2 = np.zeros((3, 6, 3))

    for k in range(3):
        j, m = j_idx[k], m_idx[k]
        U_du[k] = np.cross(r_bar[j] - r_bar[m], U_batch)  # shape (6, 3)
        alpha_du[k] = (
            d[m] * d_du[j] + d[j] * d_du[m] + U_batch @ (r_bar[j] + r_bar[m])
        )  # shape (6,)
        alpha2_du2[k] = (
            d2_du2[j] * d[m] + 2.0 * d_du[j] * d_du[m] + d[j] * d2_du2[m] + 2.0
        )  # shape (6,)

        V_du[k] = (
            alpha_du[k, :, None] * r_hat[k]
            + alpha[k] * r_hat_du[k]
            + d_du[m, :, None] * r_bar[j]
            + d_du[j, :, None] * r_bar[m]
            + (d[j] + d[m]) * U_batch
        )  # shape (6, 3)
        V2_du2[k] = (
            alpha2_du2[k, :, None] * r_hat[k]
            + 2.0 * alpha_du[k, :, None] * r_hat_du[k]
            + alpha[k] * r_hat2_du2[k]
            + d2_du2[m, :, None] * r_bar[j]
            + d2_du2[j, :, None] * r_bar[m]
            + 2.0 * (d_du[m, :, None] + d_du[j, :, None]) * U_batch
        )  # shape (6, 3)

    N_du = sum(U_batch @ U[p] for p in range(3))  # shape (6,)
    D_du = sum(U_batch @ V[p] for p in range(3))  # shape (6,)
    D2_du2 = sum(np.sum(V_du[p] * U_batch, axis=1) for p in range(3))  # shape (6,)

    S_du = 2.0 * N * N_du + 2.0 * D * D_du  # shape (6,)
    S2_du2 = 2.0 * (N_du**2) + 2.0 * (D_du**2) + 2.0 * D * D2_du2  # shape (6,)

    T2 = np.zeros((3, 3))
    for k in range(3):
        W_du = (D_du[:, None] * U[k] + D * U_du[k]) - (
            N_du[:, None] * V[k] + N * V_du[k]
        )  # shape (6, 3)
        W2_du2 = (D2_du2[:, None] * U[k] + 2.0 * D_du[:, None] * U_du[k]) - (
            2.0 * N_du[:, None] * V_du[k] + N * V2_du2[k]
        )  # shape (6, 3)

        F2_k = (
            (2.0 / S) * W2_du2
            - (4.0 * S_du[:, None] / (S**2)) * W_du
            - (2.0 * S2_du2[:, None] / (S**2)) * W[k]
            + (4.0 * (S_du[:, None] ** 2) / (S**3)) * W[k]
        )  # shape (6, 3)

        # Contract across the 6 polarization directions directly into row k
        T2[k] = weights @ F2_k  # shape (3,)

    # -------------------------------------------------------------------------
    # 5. Final Assembly
    # -------------------------------------------------------------------------
    return (1.0 / (4.0 * np.pi * L_ref)) * (T0 + T1 + T2)


def compute_taylor_gradient_v0(
    triangle: np.ndarray,  # shape [3, 3] (vertices v_0, v_1, v_2)
    center_of_mass: np.ndarray,  # shape [3]    (center x_c)
    G0: float,  # 0th-order moment (scalar)
    G1: np.ndarray,  # 1st-order moment (shape [3])
    G2: np.ndarray,  # 2nd-order moment (shape [3, 3])
    order: int = 2,  # Approximation order (0, 1, or 2)
) -> np.ndarray:
    """Computes the exact analytical Taylor series loss gradient w.r.t. triangle

    vertex positions v_0, v_1, v_2 using pure directional differential
    calculus.

    All terms T0, T1, T2 are evaluated in dimensionless space and reassembled
    with the unified physical scale factor 1 / (4 * pi * L_ref).
    """
    # -------------------------------------------------------------------------
    # 1. Non-Dimensionalization
    # -------------------------------------------------------------------------
    r = triangle - center_of_mass  # shape [3, 3]
    r_norms = np.linalg.norm(r, axis=1)
    L_ref = np.max(r_norms)
    if L_ref < 1e-12:
        L_ref = 1e-12

    r_bar = r / L_ref
    G0_bar = G0
    G1_bar = G1 / L_ref
    G2_bar = G2 / (L_ref**2)

    # -------------------------------------------------------------------------
    # 2. Base 0th-Order Field Evaluation (T0)
    # -------------------------------------------------------------------------
    d = np.linalg.norm(r_bar, axis=1)  # shape [3]
    r_hat = r_bar / d[:, None]  # shape [3, 3]

    # Cyclic permutation mapping: k=0 -> (j=1, m=2), k=1 -> (j=2, m=0), k=2 -> (j=0, m=1)
    j_idx = np.array([1, 2, 0])
    m_idx = np.array([2, 0, 1])

    U = np.zeros((3, 3))
    dot_jm = np.zeros(3)
    for k in range(3):
        j, m = j_idx[k], m_idx[k]
        U[k] = np.cross(r_bar[j], r_bar[m])
        dot_jm[k] = np.dot(r_bar[j], r_bar[m])

    N = np.dot(r_bar[0], U[0])  # det([r_bar_0, r_bar_1, r_bar_2])
    D = d[0] * d[1] * d[2] + np.sum(dot_jm * d)
    S = N**2 + D**2

    alpha = np.zeros(3)
    V = np.zeros((3, 3))
    W = np.zeros((3, 3))
    F0 = np.zeros((3, 3))

    for k in range(3):
        j, m = j_idx[k], m_idx[k]
        alpha[k] = d[j] * d[m] + dot_jm[k]
        V[k] = alpha[k] * r_hat[k] + d[m] * r_bar[j] + d[j] * r_bar[m]
        W[k] = D * U[k] - N * V[k]
        F0[k] = (2.0 / S) * W[k]

    T0 = G0_bar * F0
    if order == 0:
        return (1.0 / (4.0 * np.pi * L_ref)) * T0

    # -------------------------------------------------------------------------
    # 3. Exact 1st Directional Derivative Operator (T1)
    # -------------------------------------------------------------------------
    def _dir_deriv_1(e: np.ndarray) -> np.ndarray:
        d_de = np.zeros(3)
        r_hat_de = np.zeros((3, 3))
        U_de = np.zeros((3, 3))
        alpha_de = np.zeros(3)
        V_de = np.zeros((3, 3))

        for p in range(3):
            d_de[p] = np.dot(r_hat[p], e)
            r_hat_de[p] = (1.0 / d[p]) * (e - d_de[p] * r_hat[p])

        for k in range(3):
            j, m = j_idx[k], m_idx[k]
            U_de[k] = np.cross(r_bar[j] - r_bar[m], e)
            alpha_de[k] = (
                d[m] * d_de[j] + d[j] * d_de[m] + np.dot(r_bar[j] + r_bar[m], e)
            )
            V_de[k] = (
                alpha_de[k] * r_hat[k]
                + alpha[k] * r_hat_de[k]
                + d_de[m] * r_bar[j]
                + d_de[j] * r_bar[m]
                + (d[j] + d[m]) * e
            )

        N_de = np.sum([np.dot(U[p], e) for p in range(3)])
        D_de = np.sum([np.dot(V[p], e) for p in range(3)])

        S_de = 2.0 * N * N_de + 2.0 * D * D_de
        F_de = np.zeros((3, 3))

        for k in range(3):
            W_de = (D_de * U[k] + D * U_de[k]) - (N_de * V[k] + N * V_de[k])
            F_de[k] = (2.0 / S) * W_de - (2.0 * S_de / (S**2)) * W[k]

        return F_de

    T1 = _dir_deriv_1(-G1_bar)
    if order == 1:
        return (1.0 / (4.0 * np.pi * L_ref)) * (T0 + T1)

    # -------------------------------------------------------------------------
    # 4. Exact 2nd Directional Derivative Operator (T2)
    # -------------------------------------------------------------------------
    def _dir_deriv_2(u: np.ndarray) -> np.ndarray:
        d_du = np.zeros(3)
        d2_du2 = np.zeros(3)
        r_hat_du = np.zeros((3, 3))
        r_hat2_du2 = np.zeros((3, 3))

        for p in range(3):
            d_du[p] = np.dot(r_hat[p], u)
            d2_du2[p] = (1.0 / d[p]) * (1.0 - d_du[p] ** 2)
            r_hat_du[p] = (1.0 / d[p]) * (u - d_du[p] * r_hat[p])
            r_hat2_du2[p] = (1.0 / d[p]) * (
                -2.0 * d_du[p] * r_hat_du[p] - d2_du2[p] * r_hat[p]
            )

        U_du = np.zeros((3, 3))
        alpha_du = np.zeros(3)
        alpha2_du2 = np.zeros(3)
        V_du = np.zeros((3, 3))
        V2_du2 = np.zeros((3, 3))

        for k in range(3):
            j, m = j_idx[k], m_idx[k]
            U_du[k] = np.cross(r_bar[j] - r_bar[m], u)
            alpha_du[k] = (
                d[m] * d_du[j] + d[j] * d_du[m] + np.dot(r_bar[j] + r_bar[m], u)
            )
            alpha2_du2[k] = (
                d2_du2[j] * d[m] + 2.0 * d_du[j] * d_du[m] + d[j] * d2_du2[m] + 2.0
            )

            V_du[k] = (
                alpha_du[k] * r_hat[k]
                + alpha[k] * r_hat_du[k]
                + d_du[m] * r_bar[j]
                + d_du[j] * r_bar[m]
                + (d[j] + d[m]) * u
            )
            V2_du2[k] = (
                alpha2_du2[k] * r_hat[k]
                + 2.0 * alpha_du[k] * r_hat_du[k]
                + alpha[k] * r_hat2_du2[k]
                + d2_du2[m] * r_bar[j]
                + d2_du2[j] * r_bar[m]
                + 2.0 * d_du[m] * u
                + 2.0 * d_du[j] * u
            )

        N_du = np.sum([np.dot(U[p], u) for p in range(3)])
        D_du = np.sum([np.dot(V[p], u) for p in range(3)])
        D2_du2 = np.sum([np.dot(V_du[p], u) for p in range(3)])

        S_du = 2.0 * N * N_du + 2.0 * D * D_du
        S2_du2 = 2.0 * (N_du**2) + 2.0 * (D_du**2) + 2.0 * D * D2_du2

        F2_du2 = np.zeros((3, 3))
        for k in range(3):
            W_du = (D_du * U[k] + D * U_du[k]) - (N_du * V[k] + N * V_du[k])
            W2_du2 = (D2_du2 * U[k] + 2.0 * D_du * U_du[k]) - (
                2.0 * N_du * V_du[k] + N * V2_du2[k]
            )

            F2_du2[k] = (
                (2.0 / S) * W2_du2
                - (4.0 * S_du / (S**2)) * W_du
                - (2.0 * S2_du2 / (S**2)) * W[k]
                + (4.0 * (S_du**2) / (S**3)) * W[k]
            )

        return F2_du2

    # Direct contraction of symmetric bilinear form B(u,v) using polarization identity
    # over standard basis vectors without Eigenvalue Decomposition (EVD).
    e0 = np.array([1.0, 0.0, 0.0])
    e1 = np.array([0.0, 1.0, 0.0])
    e2 = np.array([0.0, 0.0, 1.0])

    Q_e0 = _dir_deriv_2(e0)
    Q_e1 = _dir_deriv_2(e1)
    Q_e2 = _dir_deriv_2(e2)

    Q_e01 = _dir_deriv_2(e0 + e1)
    Q_e12 = _dir_deriv_2(e1 + e2)
    Q_e20 = _dir_deriv_2(e2 + e0)

    # Polarization relation: 2 * B(e_i, e_j) = Q(e_i + e_j) - Q(e_i) - Q(e_j)
    # T2 = 0.5 * sum_{i,j} G2_bar_{i,j} * B(e_i, e_j)
    g00, g11, g22 = G2_bar[0, 0], G2_bar[1, 1], G2_bar[2, 2]
    g01, g12, g20 = G2_bar[0, 1], G2_bar[1, 2], G2_bar[2, 0]

    T2 = 0.5 * (
        (g00 - g01 - g20) * Q_e0
        + (g11 - g01 - g12) * Q_e1
        + (g22 - g12 - g20) * Q_e2
        + g01 * Q_e01
        + g12 * Q_e12
        + g20 * Q_e20
    )

    # -------------------------------------------------------------------------
    # 5. Final Assembly
    # -------------------------------------------------------------------------
    return (1.0 / (4.0 * np.pi * L_ref)) * (T0 + T1 + T2)


# # =============================================================================
# # 3. Taylor Approximation Function (Supports Order Selection 0, 1, 2)
# # =============================================================================
# def compute_taylor_gradient(
#     triangle: np.ndarray,  # shape [3, 3] (vertices v_0, v_1, v_2)
#     center_of_mass: np.ndarray,  # shape [3]   (center x_c)
#     G0: float,  # 0th-order moment (scalar)
#     G1: np.ndarray,  # 1st-order moment (shape [3])
#     G2: np.ndarray,  # 2nd-order moment (shape [3, 3])
#     order: int = 2,  # Approximation order (0, 1, or 2)
# ) -> np.ndarray:
#     """Computes the exact analytical Taylor series loss gradient w.r.t. triangle
#
#     vertex positions v_0, v_1, v_2 using pure directional differential
#     calculus.
#
#     All terms T0, T1, T2 are evaluated in dimensionless space and reassembled
#     with the unified physical scale factor 1 / (4 * pi * L_ref).
#     """
#     # -------------------------------------------------------------------------
#     # 1. Non-Dimensionalization
#     # -------------------------------------------------------------------------
#     r = triangle - center_of_mass  # shape [3, 3]
#     r_norms = np.linalg.norm(r, axis=1)
#     L_ref = np.max(r_norms)
#     if L_ref < 1e-12:
#         L_ref = 1e-12
#
#     r_bar = r / L_ref
#     G0_bar = G0
#     G1_bar = G1 / L_ref
#     G2_bar = G2 / (L_ref**2)
#
#     # -------------------------------------------------------------------------
#     # 2. Base 0th-Order Field Evaluation (T0)
#     # -------------------------------------------------------------------------
#     d = np.linalg.norm(r_bar, axis=1)  # shape [3]
#     r_hat = r_bar / d[:, None]  # shape [3, 3]
#
#     # Cyclic permutation mapping: k=0 -> (j=1, m=2), k=1 -> (j=2, m=0), k=2 -> (j=0, m=1)
#     j_idx = np.array([1, 2, 0])
#     m_idx = np.array([2, 0, 1])
#
#     U = np.zeros((3, 3))
#     dot_jm = np.zeros(3)
#     for k in range(3):
#         j, m = j_idx[k], m_idx[k]
#         U[k] = np.cross(r_bar[j], r_bar[m])
#         dot_jm[k] = np.dot(r_bar[j], r_bar[m])
#
#     N = np.dot(r_bar[0], U[0])  # det([r_bar_0, r_bar_1, r_bar_2])
#     D = d[0] * d[1] * d[2] + np.sum(dot_jm * d)
#     S = N**2 + D**2
#
#     alpha = np.zeros(3)
#     V = np.zeros((3, 3))
#     W = np.zeros((3, 3))
#     F0 = np.zeros((3, 3))
#
#     for k in range(3):
#         j, m = j_idx[k], m_idx[k]
#         alpha[k] = d[j] * d[m] + dot_jm[k]
#         V[k] = alpha[k] * r_hat[k] + d[m] * r_bar[j] + d[j] * r_bar[m]
#         W[k] = D * U[k] - N * V[k]
#         F0[k] = (2.0 / S) * W[k]
#
#     T0 = G0_bar * F0
#     if order == 0:
#         return (1.0 / (4.0 * np.pi * L_ref)) * T0
#
#     # -------------------------------------------------------------------------
#     # 3. Exact 1st Directional Derivative Operator (T1)
#     # -------------------------------------------------------------------------
#     def _dir_deriv_1(e: np.ndarray) -> np.ndarray:
#         d_de = np.zeros(3)
#         r_hat_de = np.zeros((3, 3))
#         U_de = np.zeros((3, 3))
#         alpha_de = np.zeros(3)
#         V_de = np.zeros((3, 3))
#
#         for p in range(3):
#             d_de[p] = np.dot(r_hat[p], e)
#             r_hat_de[p] = (1.0 / d[p]) * (e - d_de[p] * r_hat[p])
#
#         for k in range(3):
#             j, m = j_idx[k], m_idx[k]
#             U_de[k] = np.cross(r_bar[j] - r_bar[m], e)
#             alpha_de[k] = (
#                 d[m] * d_de[j] + d[j] * d_de[m] + np.dot(r_bar[j] + r_bar[m], e)
#             )
#             V_de[k] = (
#                 alpha_de[k] * r_hat[k]
#                 + alpha[k] * r_hat_de[k]
#                 + d_de[m] * r_bar[j]
#                 + d_de[j] * r_bar[m]
#                 + (d[j] + d[m]) * e
#             )
#
#         N_de = np.sum([np.dot(U[p], e) for p in range(3)])
#         D_de = np.sum([np.dot(V[p], e) for p in range(3)])
#
#         S_de = 2.0 * N * N_de + 2.0 * D * D_de
#         F_de = np.zeros((3, 3))
#
#         for k in range(3):
#             W_de = (D_de * U[k] + D * U_de[k]) - (N_de * V[k] + N * V_de[k])
#             F_de[k] = (2.0 / S) * W_de - (2.0 * S_de / (S**2)) * W[k]
#
#         return F_de
#
#     T1 = _dir_deriv_1(-G1_bar)
#     if order == 1:
#         return (1.0 / (4.0 * np.pi * L_ref)) * (T0 + T1)
#
#     # -------------------------------------------------------------------------
#     # 4. Exact 2nd Directional Derivative Operator (T2)
#     # -------------------------------------------------------------------------
#     def _dir_deriv_2(u: np.ndarray) -> np.ndarray:
#         d_du = np.zeros(3)
#         d2_du2 = np.zeros(3)
#         r_hat_du = np.zeros((3, 3))
#         r_hat2_du2 = np.zeros((3, 3))
#
#         for p in range(3):
#             d_du[p] = np.dot(r_hat[p], u)
#             d2_du2[p] = (1.0 / d[p]) * (1.0 - d_du[p] ** 2)
#             r_hat_du[p] = (1.0 / d[p]) * (u - d_du[p] * r_hat[p])
#             r_hat2_du2[p] = (1.0 / d[p]) * (
#                 -2.0 * d_du[p] * r_hat_du[p] - d2_du2[p] * r_hat[p]
#             )
#
#         U_du = np.zeros((3, 3))
#         alpha_du = np.zeros(3)
#         alpha2_du2 = np.zeros(3)
#         V_du = np.zeros((3, 3))
#         V2_du2 = np.zeros((3, 3))
#
#         for k in range(3):
#             j, m = j_idx[k], m_idx[k]
#             U_du[k] = np.cross(r_bar[j] - r_bar[m], u)
#             alpha_du[k] = (
#                 d[m] * d_du[j] + d[j] * d_du[m] + np.dot(r_bar[j] + r_bar[m], u)
#             )
#             alpha2_du2[k] = (
#                 d2_du2[j] * d[m]
#                 + 2.0 * d_du[j] * d_du[m]
#                 + d[j] * d2_du2[m]
#                 + 2.0
#             )
#
#             V_du[k] = (
#                 alpha_du[k] * r_hat[k]
#                 + alpha[k] * r_hat_du[k]
#                 + d_du[m] * r_bar[j]
#                 + d_du[j] * r_bar[m]
#                 + (d[j] + d[m]) * u
#             )
#             V2_du2[k] = (
#                 alpha2_du2[k] * r_hat[k]
#                 + 2.0 * alpha_du[k] * r_hat_du[k]
#                 + alpha[k] * r_hat2_du2[k]
#                 + d2_du2[m] * r_bar[j]
#                 + d2_du2[j] * r_bar[m]
#                 + 2.0 * d_du[m] * u
#                 + 2.0 * d_du[j] * u
#             )
#
#         N_du = np.sum([np.dot(U[p], u) for p in range(3)])
#         D_du = np.sum([np.dot(V[p], u) for p in range(3)])
#         D2_du2 = np.sum([np.dot(V_du[p], u) for p in range(3)])
#
#         S_du = 2.0 * N * N_du + 2.0 * D * D_du
#         S2_du2 = 2.0 * (N_du**2) + 2.0 * (D_du**2) + 2.0 * D * D2_du2
#
#         F2_du2 = np.zeros((3, 3))
#         for k in range(3):
#             W_du = (D_du * U[k] + D * U_du[k]) - (N_du * V[k] + N * V_du[k])
#             W2_du2 = (D2_du2 * U[k] + 2.0 * D_du * U_du[k]) - (
#                 2.0 * N_du * V_du[k] + N * V2_du2[k]
#             )
#
#             F2_du2[k] = (
#                 (2.0 / S) * W2_du2
#                 - (4.0 * S_du / (S**2)) * W_du
#                 - (2.0 * S2_du2 / (S**2)) * W[k]
#                 + (4.0 * (S_du**2) / (S**3)) * W[k]
#             )
#
#         return F2_du2
#
#     evals, evecs = np.linalg.eigh(G2_bar)
#     T2 = np.zeros((3, 3))
#     for m_mode in range(3):
#         lam = evals[m_mode]
#         if abs(lam) < 1e-15:
#             continue
#         u = evecs[:, m_mode]
#         T2 += 0.5 * lam * _dir_deriv_2(u)
#
#     # -------------------------------------------------------------------------
#     # 5. Final Assembly
#     # -------------------------------------------------------------------------
#     return (1.0 / (4.0 * np.pi * L_ref)) * (T0 + T1 + T2)


# =============================================================================
# 4. Randomized Test Suite & Convergence Suite
# =============================================================================
def generate_random_triangle(scale: float = 1.0) -> np.ndarray:
    return np.random.uniform(-scale, scale, size=(3, 3))


def generate_random_cluster(center: np.ndarray, radius: float, num_points: int = 50):
    # Uniform random distribution within a sphere
    offsets = np.random.normal(0.0, 1.0, size=(num_points, 3))
    offsets = offsets / np.linalg.norm(offsets, axis=1, keepdims=True)
    radii = radius * (np.random.uniform(0.0, 1.0, size=(num_points, 1)) ** (1.0 / 3.0))
    points = center + offsets * radii
    weights = np.random.normal(1.0, 0.5, size=num_points)
    return points, weights


def run_convergence_suite():
    print("=" * 88)
    print("RUNNING DISTANCE SCALING & CONVERGENCE TEST SUITE")
    print("=" * 88)

    np.random.seed(42)

    triangle = np.array([[1.2, 0.1, -0.2], [0.3, 1.4, 0.1], [-0.1, 0.2, 1.6]])
    tri_centroid = np.mean(triangle, axis=0)

    cluster_radius = 0.2
    direction = np.array([0.6, 0.8, 1.0])
    direction /= np.linalg.norm(direction)

    distances = [10.0, 5.0, 2.5, 1.25]

    prev_err0, prev_err1, prev_err2 = None, None, None

    header = f"{'Distance':<9} | {'0th Rel Err':<12} {'(Halving)':<9} | {'1st Rel Err':<12} {'(Halving)':<9} | {'2nd Rel Err':<12} {'(Halving)':<9} | {'1st Gain':<9} | {'2nd Gain':<9}"
    print(header)
    print("-" * len(header))

    for dist in distances:
        center = tri_centroid + direction * dist
        points, weights = generate_random_cluster(
            center, cluster_radius, num_points=100
        )

        x_c, G0, G1, G2 = compute_cluster_moments(points, weights)

        grad_gt = compute_exact_ground_truth(triangle, points, weights)
        grad_t0 = compute_node_gradient_approximation(
            triangle, x_c, G0, G1, G2, order=0
        )
        grad_t1 = compute_node_gradient_approximation(
            triangle, x_c, G0, G1, G2, order=1
        )
        grad_t2 = compute_node_gradient_approximation(
            triangle, x_c, G0, G1, G2, order=2
        )

        norm_gt = np.linalg.norm(grad_gt)
        err0 = np.linalg.norm(grad_gt - grad_t0) / norm_gt
        err1 = np.linalg.norm(grad_gt - grad_t1) / norm_gt
        err2 = np.linalg.norm(grad_gt - grad_t2) / norm_gt

        # Error growth factors when distance halves (Theoretical: ~2x for 0th, ~4x for 1st, ~8x for 2nd)
        growth0 = f"{err0 / prev_err0:.2f}x" if prev_err0 is not None else "-"
        growth1 = f"{err1 / prev_err1:.2f}x" if prev_err1 is not None else "-"
        growth2 = f"{err2 / prev_err2:.2f}x" if prev_err2 is not None else "-"

        # Order-by-order gain at the current distance
        gain_1st = f"{err0 / err1:.1f}x" if err1 > 0 else "inf"
        gain_2nd = f"{err1 / err2:.1f}x" if err2 > 0 else "inf"

        print(
            f"{dist:<9.2f} | {err0:<12.3e} {growth0:<9} | {err1:<12.3e} {growth1:<9} | {err2:<12.3e} {growth2:<9} | {gain_1st:<9} | {gain_2nd:<9}"
        )

        prev_err0, prev_err1, prev_err2 = err0, err1, err2

    print(
        "\nExpected asymptotic growth when distance halves: 0th ~ 2.0x, 1st ~ 4.0x, 2nd ~ 8.0x\n"
    )


def run_random_monte_carlo_suite(num_trials: int = 1000):
    print("=" * 88)
    print(f"RUNNING MONTE CARLO RANDOM TRIALS (N={num_trials})")
    print("=" * 88)

    np.random.seed(1337)

    errs_0th, errs_1st, errs_2nd = [], [], []
    cos_0th, cos_1st, cos_2nd = [], [], []
    gains_0_to_1, gains_1_to_2, gains_total = [], [], []
    monotonic_count = 0

    for trial in range(num_trials):
        triangle = generate_random_triangle(scale=2.0)
        tri_centroid = np.mean(triangle, axis=0)

        dist = np.random.uniform(4.0, 10.0)
        dir_vec = np.random.normal(0.0, 1.0, size=3)
        dir_vec /= np.linalg.norm(dir_vec)

        cluster_center = tri_centroid + dir_vec * dist
        cluster_radius = np.random.uniform(0.1, 0.4)

        points, weights = generate_random_cluster(
            cluster_center, cluster_radius, num_points=40
        )
        x_c, G0, G1, G2 = compute_cluster_moments(points, weights)

        grad_gt = compute_exact_ground_truth(triangle, points, weights)
        grad_t0 = compute_node_gradient_approximation(
            triangle, x_c, G0, G1, G2, order=0
        )
        grad_t1 = compute_node_gradient_approximation(
            triangle, x_c, G0, G1, G2, order=1
        )
        grad_t2 = compute_node_gradient_approximation(
            triangle, x_c, G0, G1, G2, order=2
        )

        norm_gt = np.linalg.norm(grad_gt)
        if norm_gt < 1e-12:
            continue

        norm_t0 = np.linalg.norm(grad_t0)
        norm_t1 = np.linalg.norm(grad_t1)
        norm_t2 = np.linalg.norm(grad_t2)

        # Relative Errors
        rel_err0 = np.linalg.norm(grad_gt - grad_t0) / norm_gt
        rel_err1 = np.linalg.norm(grad_gt - grad_t1) / norm_gt
        rel_err2 = np.linalg.norm(grad_gt - grad_t2) / norm_gt

        # Cosine Similarities (Frobenius inner product over full gradient matrix)
        cos0 = np.sum(grad_gt * grad_t0) / (norm_gt * norm_t0) if norm_t0 > 1e-12 else 0.0
        cos1 = np.sum(grad_gt * grad_t1) / (norm_gt * norm_t1) if norm_t1 > 1e-12 else 0.0
        cos2 = np.sum(grad_gt * grad_t2) / (norm_gt * norm_t2) if norm_t2 > 1e-12 else 0.0

        cos0 = float(np.clip(cos0, -1.0, 1.0))
        cos1 = float(np.clip(cos1, -1.0, 1.0))
        cos2 = float(np.clip(cos2, -1.0, 1.0))

        errs_0th.append(rel_err0)
        errs_1st.append(rel_err1)
        errs_2nd.append(rel_err2)

        cos_0th.append(cos0)
        cos_1st.append(cos1)
        cos_2nd.append(cos2)

        # Record gain ratios (clamped for floating point safety)
        g1 = rel_err0 / max(rel_err1, 1e-15)
        g2 = rel_err1 / max(rel_err2, 1e-15)
        gt = rel_err0 / max(rel_err2, 1e-15)

        gains_0_to_1.append(g1)
        gains_1_to_2.append(g2)
        gains_total.append(gt)

        # Strict order-by-order monotonicity test: 2nd < 1st < 0th
        if rel_err2 < rel_err1 < rel_err0:
            monotonic_count += 1

    total_valid = len(errs_0th)

    # Use Geometric Mean for ratio metrics to handle multiplicative variance safely
    geom_gain_0_to_1 = np.exp(np.mean(np.log(gains_0_to_1)))
    geom_gain_1_to_2 = np.exp(np.mean(np.log(gains_1_to_2)))
    geom_gain_total = np.exp(np.mean(np.log(gains_total)))

    print("--- MEAN RELATIVE ERRORS ---")
    print(f"  0th-Order (Monopole):      {np.mean(errs_0th):.4e}")
    print(f"  1st-Order (+ Dipole):        {np.mean(errs_1st):.4e}")
    print(f"  2nd-Order (+ Quadrupole):   {np.mean(errs_2nd):.4e}")

    print("\n--- COSINE SIMILARITY (DIRECTIONAL ACCURACY) ---")
    print(f"  0th-Order: Mean = {np.mean(cos_0th):.6f}, Variance = {np.var(cos_0th):.4e}")
    print(f"  1st-Order: Mean = {np.mean(cos_1st):.6f}, Variance = {np.var(cos_1st):.4e}")
    print(f"  2nd-Order: Mean = {np.mean(cos_2nd):.6f}, Variance = {np.var(cos_2nd):.4e}")

    print("\n--- ORDER-BY-ORDER ERROR REDUCTION GAINS (GEOMETRIC MEAN) ---")
    print(f"  0th -> 1st Order Gain:     {geom_gain_0_to_1:.2f}x error reduction")
    print(f"  1st -> 2nd Order Gain:     {geom_gain_1_to_2:.2f}x error reduction")
    print(f"  Total (0th -> 2nd) Gain:   {geom_gain_total:.2f}x total error reduction")

    print("\n--- VALIDATION METRICS ---")
    print(
        f"  Monotonicity (Err2 < Err1 < Err0): {monotonic_count} / {total_valid} trials ({100.0 * monotonic_count / total_valid:.1f}%)"
    )
    print("=" * 88)

if __name__ == "__main__":
    run_convergence_suite()
    run_random_monte_carlo_suite(num_trials=1000)
