import sympy as sp
import numpy as np

# ============================================================
# 1. Define Symbolic Variables & Symmetric G2
# ============================================================
print("Initializing symbolic trees...")
ax, ay, az = sp.symbols('ax ay az')
bx, by, bz = sp.symbols('bx by bz')
cx, cy, cz = sp.symbols('cx cy cz')

a = sp.Matrix([ax, ay, az])
b = sp.Matrix([bx, by, bz])
c = sp.Matrix([cx, cy, cz])

G2_00, G2_01, G2_02 = sp.symbols('G2_00 G2_01 G2_02')
G2_11, G2_12, G2_22 = sp.symbols('G2_11 G2_12 G2_22')
G2 = sp.Matrix([[G2_00, G2_01, G2_02],
                [G2_01, G2_11, G2_12],
                [G2_02, G2_12, G2_22]])

# ============================================================
# 2. Helper: Compute Edge Jacobians and Contractions
# ============================================================
def compute_edge_terms(p_start, p_end, G2):
    R_start = sp.sqrt(p_start.dot(p_start))
    R_end = sp.sqrt(p_end.dot(p_end))
    S = R_start * R_end + p_start.dot(p_end)
    L = 1/R_start + 1/R_end
    C = p_start.cross(p_end)
    
    hat_start = p_start / R_start
    hat_end = p_end / R_end
    
    w = -hat_start / (S * R_start**2) - (L / S**2) * (R_end * hat_start + p_end)
    z = -hat_end / (S * R_end**2) - (L / S**2) * (R_start * hat_end + p_start)
    
    def cross_matrix(v):
        return sp.Matrix([[    0, -v[2],  v[1]],
                          [ v[2],     0, -v[0]],
                          [-v[1],  v[0],     0]])
    
    J_a = C * w.T - (L / S) * cross_matrix(p_end)
    J_b = C * z.T + (L / S) * cross_matrix(p_start)
    
    # Second-order contraction loop
    def contract(J):
        V = sp.Matrix.zeros(3, 1)
        for m in range(3):
            val = 0
            for i in range(3):
                for j in range(3):
                    d_dstart = sp.diff(J[i, m], p_start[j])
                    d_dend = sp.diff(J[i, m], p_end[j])
                    spatial_deriv = -(d_dstart + d_dend)
                    val += spatial_deriv * G2[i, j]
            V[m] = val
        return V
        
    return J_a, J_b, contract(J_a), contract(J_b)

# ============================================================
# 3. Derive Complete Taylor Pieces for All Vertices (1, 2, and 3)
# ============================================================
print("Deriving Edge 1->2 contributions...")
J_a_12, J_b_12, V_a_12, V_b_12 = compute_edge_terms(a, b, G2)

print("Deriving Edge 2->3 contributions...")
J_a_23, J_b_23, V_a_23, V_b_23 = compute_edge_terms(b, c, G2)

print("Deriving Edge 3->1 contributions...")
J_a_31, J_b_31, V_a_31, V_b_31 = compute_edge_terms(c, a, G2)

# Common denominators and scalar terms (Invariant under cyclic permutation)
Ra, Rb, Rc = sp.sqrt(a.dot(a)), sp.sqrt(b.dot(b)), sp.sqrt(c.dot(c))
N = a.dot(b.cross(c))
D = Ra*Rb*Rc + a.dot(b)*Rc + b.dot(c)*Ra + c.dot(a)*Rb
denominator = N**2 + D**2

# Zero-Order Fields (Exact Oosterom-Strackee Vertex Gradient Formulas)
print("Formulating exact 0th-order Oosterom fields...")
F1_expr = 2 * (D * b.cross(c) - N * ((Rb*Rc + b.dot(c))*(a/Ra) + Rc*b + Rb*c)) / denominator
F2_expr = 2 * (D * c.cross(a) - N * ((Rc*Ra + c.dot(a))*(b/Rb) + Ra*c + Rc*a)) / denominator
F3_expr = 2 * (D * a.cross(b) - N * ((Ra*Rb + a.dot(b))*(c/Rc) + Rb*a + Ra*b)) / denominator

# First-Order Fields (Map edge Jacobians to Spatial Field Jacobians)
J_F1_expr = J_a_12.T + J_b_31.T
J_F2_expr = J_b_12.T + J_a_23.T
J_F3_expr = J_b_23.T + J_a_31.T

# Second-Order Fields (Combine edge contractions)
V_F1_expr = V_a_12 + V_b_31
V_F2_expr = V_b_12 + V_a_23
V_F3_expr = V_b_23 + V_a_31

# ============================================================
# 4. Compile Symbolic Trees to Executable Functions
# ============================================================
print("Compiling expressions to NumPy routines via lambdify...")
geom_syms = [ax, ay, az, bx, by, bz, cx, cy, cz]
all_syms = geom_syms + [G2_00, G2_01, G2_02, G2_11, G2_12, G2_22]

# Lambdify for all three vertices
F_funcs = [sp.lambdify(geom_syms, expr, 'numpy') for expr in [F1_expr, F2_expr, F3_expr]]
J_funcs = [sp.lambdify(geom_syms, expr, 'numpy') for expr in [J_F1_expr, J_F2_expr, J_F3_expr]]
V_funcs = [sp.lambdify(all_syms, expr, 'numpy') for expr in [V_F1_expr, V_F2_expr, V_F3_expr]]

# ============================================================
# 5. Numerical Test Bench Setup (Validating All Vertices)
# ============================================================
print("\n=== STARTING NUMERICAL CONVERGENCE TEST ===")

v1 = np.array([1.2, 0.1, -0.2])
v2 = np.array([0.3, 1.4, 0.1])
v3 = np.array([-0.1, 0.2, 1.6])

centroid = (v1 + v2 + v3) / 3.0
q = centroid + np.array([0.5, 0.8, 2.0]) 

a_q, b_q, c_q = v1 - q, v2 - q, v3 - q
args_q = [a_q[0], a_q[1], a_q[2], b_q[0], b_q[1], b_q[2], c_q[0], c_q[1], c_q[2]]

# Evaluate expansions anchored at 'q' for all vertices
F_q = [np.array(f(*args_q)).flatten() for f in F_funcs]
J_q = [np.array(j(*args_q)) for j in J_funcs]

base_displacement = np.array([0.08, -0.06, 0.09])
scales = [0.4, 0.1, 0.025] 

prev_errors = {0: None, 1: None, 2: None} # Tracks errors per vertex idx

for scale in scales:
    u = base_displacement * scale
    x = q + u  
    
    a_x, b_x, c_x = v1 - x, v2 - x, v3 - x
    args_x = [a_x[0], a_x[1], a_x[2], b_x[0], b_x[1], b_x[2], c_x[0], c_x[1], c_x[2]]
    
    g2_00, g2_01, g2_02 = u[0]*u[0], u[0]*u[1], u[0]*u[2]
    g2_11, g2_12, g2_22 = u[1]*u[1], u[1]*u[2], u[2]*u[2]
    args_v = args_q + [g2_00, g2_01, g2_02, g2_11, g2_12, g2_22]
    
    print(f"\nDisplacement Scale: {scale:.4f} (||u|| = {np.linalg.norm(u):.4f})")
    
    for v_idx in range(3):
        F_true = np.array(F_funcs[v_idx](*args_x)).flatten()
        approx_0 = F_q[v_idx]
        approx_1 = F_q[v_idx] + J_q[v_idx] @ u
        
        V_term = np.array(V_funcs[v_idx](*args_v)).flatten()
        approx_2 = approx_1 + 0.5 * V_term
        
        err_0 = np.linalg.norm(F_true - approx_0)
        err_1 = np.linalg.norm(F_true - approx_1)
        err_2 = np.linalg.norm(F_true - approx_2)
        
        print(f"  [Vertex {v_idx+1}] 0th Err: {err_0:.2e} | 1st Err: {err_1:.2e} | 2nd Err: {err_2:.2e}")
        
        if prev_errors[v_idx] is not None:
            p0, p1, p2 = prev_errors[v_idx]
            print(f"    -> Drops: 0th: {p0/err_0:.1f}x (~4) | 1st: {p1/err_1:.1f}x (~16) | 2nd: {p2/err_2:.1f}x (~64)")
            
        prev_errors[v_idx] = (err_0, err_1, err_2)

# ============================================================
# 6. Unified Auto-Generation of C++ Code for All Taylor Orders
# ============================================================
print("\nGenerating heavily optimized C++ code for 0th, 1st, and 2nd Order Terms...")

# Define the 18 compressed slots for the symmetric 3rd-order tensor
cpp_slots = [
    (0, 0, 0), (0, 0, 1), (0, 0, 2),  # (xxx, xxy, xxz)
    (0, 1, 0), (0, 1, 1), (0, 1, 2),  # (xyx, xyy, xyz)
    (0, 2, 0), (0, 2, 1), (0, 2, 2),  # (xzx, xzy, xzz)
    (1, 1, 0), (1, 1, 1), (1, 1, 2),  # (yyx, yyy, yyz)
    (1, 2, 0), (1, 2, 1), (1, 2, 2),  # (yzx, yzy, yzz)
    (2, 2, 0), (2, 2, 1), (2, 2, 2)   # (zzx, zzy, zzz)
]

geom_vectors = [a, b, c]
unified_taylor_exprs = []

# Pack all expressions into a single tracking array to optimize together
# Structural order per vertex: 3 (Zero) + 9 (First) + 18 (Second) = 30 elements per vertex
for v_idx, (F_expr, J_expr) in enumerate([(F1_expr, J_F1_expr), (F2_expr, J_F2_expr), (F3_expr, J_F3_expr)]):
    
    # A. 0th Order: Vec3 (3 expressions)
    unified_taylor_exprs.extend([F_expr[0], F_expr[1], F_expr[2]])
    
    # B. 1st Order: Mat3x3 Row-Major (9 expressions)
    # J_expr[c, j] is d(F_c)/d(x_j). We map row = field component c, col = spatial coord j
    for c_comp in range(3):
        for j in range(3):
            unified_taylor_exprs.append(J_expr[c_comp, j])
            
    # C. 2nd Order: Tensor3_compressed (18 expressions)
    for A, B, C in cpp_slots:
        i, j, c_comp = A, B, C
        expr = J_expr[c_comp, j]
        
        total_deriv = 0
        for vec in geom_vectors:
            total_deriv += sp.diff(expr, vec[i])
        unified_taylor_exprs.append(-total_deriv)

# Execute the massive 90-element CSE optimization pass
replacements, reduced_exprs = sp.cse(unified_taylor_exprs, symbols=sp.symbols('t0:4000'))

print("\n// " + "="*60)
print("// PASTE THIS DIRECTLY INTO YOUR get_all_taylor_terms() IMPLEMENTATION")
print("// " + "="*60)
print(f"// Generated using SymPy CSE optimization. Intermediates: {len(replacements)} variables.")

# 1. Print Intermediates
for var, expr in replacements:
    print(f"const float {var} = {sp.ccode(expr)};")

print("")

# 2. Map the 90 flattened expressions back to their respective C++ variables
expr_idx = 0
for v in range(1, 4):
    prefix = f"v{v}_"
    
    # Assign Zero Order (Vec3)
    print(f"{prefix}zero_order.x = {sp.ccode(reduced_exprs[expr_idx])};")
    print(f"{prefix}zero_order.y = {sp.ccode(reduced_exprs[expr_idx+1])};")
    print(f"{prefix}zero_order.z = {sp.ccode(reduced_exprs[expr_idx+2])};")
    expr_idx += 3
    
    # Assign First Order (Mat3x3 Row-Major)
    for m_idx in range(9):
        print(f"{prefix}first_order.data[{m_idx}] = {sp.ccode(reduced_exprs[expr_idx])};")
        expr_idx += 1
        
    # Assign Second Order (Tensor3_compressed)
    for t_idx in range(18):
        print(f"{prefix}second_order.data[{t_idx}] = {sp.ccode(reduced_exprs[expr_idx])};")
        expr_idx += 1
    print("")

# ============================================================
# 7. Auto-Generate Optimized C++ Code for Near-Field Exact Pass
# ============================================================
print("\nGenerating optimized C++ code for near-field exact derivatives...")

exact_near_field_exprs = [
    F1_expr[0], F1_expr[1], F1_expr[2],  # v1
    F2_expr[0], F2_expr[1], F2_expr[2],  # v2
    F3_expr[0], F3_expr[1], F3_expr[2]   # v3
]

replacements_nf, reduced_nf = sp.cse(exact_near_field_exprs, symbols=sp.symbols('s0:2000'))

print("\n// " + "="*60)
print("// PASTE THIS DIRECTLY INTO YOUR exactGradientsToQuery() IMPLEMENTATION")
print("// " + "="*60)
print(f"// Generated using SymPy CSE optimization. Elements: {len(replacements_nf)} intermediates.")

for var, expr in replacements_nf:
    print(f"const float {var} = {sp.ccode(expr)};")

print("")
print(f"grad_v1.x = {sp.ccode(reduced_nf[0])}; grad_v1.y = {sp.ccode(reduced_nf[1])}; grad_v1.z = {sp.ccode(reduced_nf[2])};")
print(f"grad_v2.x = {sp.ccode(reduced_nf[3])}; grad_v2.y = {sp.ccode(reduced_nf[4])}; grad_v2.z = {sp.ccode(reduced_nf[5])};")
print(f"grad_v3.x = {sp.ccode(reduced_nf[6])}; grad_v3.y = {sp.ccode(reduced_nf[7])}; grad_v3.z = {sp.ccode(reduced_nf[8])};")
