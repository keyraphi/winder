import argparse
import csv
from datetime import datetime
import os
from time import time
import igl
import matplotlib.pyplot as plt
import numpy as np
import torch
import winder


def mesh_to_point_surfels(
    vertices: np.ndarray, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converts a triangle mesh into a point-normal-area representation."""
    v0 = vertices[indices[:, 0]]
    v1 = vertices[indices[:, 1]]
    v2 = vertices[indices[:, 2]]

    points = (v0 + v1 + v2) / 3.0

    e1 = v1 - v0
    e2 = v2 - v0

    cross = np.cross(e1, e2)
    magnitudes = np.linalg.norm(cross, axis=-1, keepdims=True)
    areas = (magnitudes / 2.0).flatten()

    safe_magnitudes = np.where(magnitudes == 0, 1e-8, magnitudes)
    normals = cross / safe_magnitudes

    return (
        points.astype(np.float32),
        normals.astype(np.float32),
        areas.astype(np.float32),
    )


def generate_queries(
    vertices: np.ndarray, mode: str, num_queries: int = 100
) -> np.ndarray:
    """Generates query points either randomly or on a uniform grid around the mesh bounding box."""
    min_box = vertices.min(axis=0)
    max_box = vertices.max(axis=0)
    diag = np.linalg.norm(max_box - min_box)
    min_box -= diag * 0.5
    max_box += diag * 0.5
    print("DEBUG: queries created in:", min_box, max_box)

    if mode == "grid":
        side = int(np.ceil(num_queries ** (1.0 / 3.0)))
        x = np.linspace(min_box[0], max_box[0], side)
        y = np.linspace(min_box[1], max_box[1], side)
        z = np.linspace(min_box[2], max_box[2], side)
        gx, gy, gz = np.meshgrid(x, y, z, indexing="ij")
        queries = np.stack([gx.flatten(), gy.flatten(), gz.flatten()], axis=-1)
        return queries[:num_queries].astype(np.float32)
    else:
        return np.random.uniform(min_box, max_box, size=(num_queries, 3)).astype(
            np.float32
        )


def print_ascii_histogram(
    data: np.ndarray,
    bins: np.ndarray,
    title: str,
    unit: str = "",
    is_log_labels: bool = False,
):
    """Renders a clean ASCII histogram bar chart for array distributions."""
    counts, _ = np.histogram(data, bins=bins)
    total = len(data)
    if total == 0:
        return

    max_count = np.max(counts) if np.max(counts) > 0 else 1
    bar_max_width = 35

    print(f"\n  --- {title} ---")
    for i in range(len(counts)):
        low, high = bins[i], bins[i + 1]
        count = counts[i]
        pct = (count / total) * 100
        bar_len = int((count / max_count) * bar_max_width)
        bar = "█" * bar_len

        if is_log_labels:
            label = f"[{low:.1e}, {high:.1e}){unit}"
        else:
            label = f"[{low:6.2f}, {high:6.2f}){unit}"

        print(f"  {label:<24} | {bar:<35} | {count:7d} ({pct:6.2f}%)")


def validate_gradients(
    winder_grads: np.ndarray,
    brute_force_grads: np.ndarray,
    label: str,
    signal_threshold: float = 1e-5,
) -> float:
    """Validates winder gradients with detailed error distributions and returns global relative norm error."""
    assert winder_grads.shape == brute_force_grads.shape, (
        f"Shape mismatch: {winder_grads.shape} vs {brute_force_grads.shape}"
    )

    winder_flat = winder_grads.reshape(-1)
    brute_force_flat = brute_force_grads.reshape(-1)

    if np.any(np.isnan(winder_flat)) or np.any(np.isinf(winder_flat)):
        print(f"\033[91m  ✗ CRITICAL FAILURE: {label} output contains NaN or Inf values!\033[0m")
        return np.nan

    # Global Relative Norm Error & Cosine Similarity
    norm_diff = np.linalg.norm(winder_flat - brute_force_flat)
    norm_ref = np.linalg.norm(brute_force_flat)
    global_rel_norm_err = norm_diff / (norm_ref + 1e-12)

    dot_prod = np.dot(winder_flat, brute_force_flat)
    norm_winder = np.linalg.norm(winder_flat)
    global_cosine_sim = dot_prod / (norm_winder * norm_ref + 1e-12)

    # Signal-Masked Relative Error Metrics
    mask = np.abs(brute_force_flat) > signal_threshold
    if np.any(mask):
        abs_err_masked = np.abs(winder_flat[mask] - brute_force_flat[mask])
        rel_err_masked = abs_err_masked / np.abs(brute_force_flat[mask])

        mean_rel = np.mean(rel_err_masked)
        rms_rel = np.sqrt(np.mean(rel_err_masked**2))
        std_rel = np.std(rel_err_masked)
        p50_rel = np.percentile(rel_err_masked, 50)
        p95_rel = np.percentile(rel_err_masked, 95)
        p99_rel = np.percentile(rel_err_masked, 99)
        max_rel = np.max(rel_err_masked)
    else:
        rel_err_masked = np.array([0.0])
        mean_rel = rms_rel = std_rel = p50_rel = p95_rel = p99_rel = max_rel = 0.0

    # Per-Vector 3D Angular Error (Degrees)
    vec_winder = winder_grads.reshape(-1, 3)
    vec_ref = brute_force_grads.reshape(-1, 3)

    vec_ref_norms = np.linalg.norm(vec_ref, axis=-1, keepdims=True)
    vec_winder_norms = np.linalg.norm(vec_winder, axis=-1, keepdims=True)

    vec_mask = (vec_ref_norms > signal_threshold).flatten()
    if np.any(vec_mask):
        dot_vecs = np.sum(vec_winder[vec_mask] * vec_ref[vec_mask], axis=-1)
        denom = (vec_winder_norms[vec_mask] * vec_ref_norms[vec_mask]).flatten()
        cos_vecs = np.clip(dot_vecs / (denom + 1e-12), -1.0, 1.0)
        angular_err_deg = np.arccos(cos_vecs) * (180.0 / np.pi)

        mean_ang = np.mean(angular_err_deg)
        rms_ang = np.sqrt(np.mean(angular_err_deg**2))
        p95_ang = np.percentile(angular_err_deg, 95)
        p99_ang = np.percentile(angular_err_deg, 99)
        max_ang = np.max(angular_err_deg)
    else:
        angular_err_deg = np.array([0.0])
        mean_ang = rms_ang = p95_ang = p99_ang = max_ang = 0.0

    print(f"\n=================================================================")
    print(f"               Gradient Validation Results: {label}")
    print(f"=================================================================")
    print(f" Global Relative Norm Error : {global_rel_norm_err:.6e}")
    print(f" Global Cosine Similarity   : {global_cosine_sim:.8f}  (1.00000000 = exact)")
    print(f"-----------------------------------------------------------------")
    print(f" Relative Error (|g| > {signal_threshold:.1e}):")
    print(f"  -> Mean Relative Error    : {mean_rel:.6e}")
    print(f"  -> RMS Relative Error     : {rms_rel:.6e}")
    print(f"  -> Std Deviation (σ)      : {std_rel:.6e}")
    print(f"  -> Median (p50)            : {p50_rel:.6e}")
    print(f"  -> 95th Percentile (p95)  : {p95_rel:.6e}")
    print(f"  -> 99th Percentile (p99)  : {p99_rel:.6e}")
    print(f"  -> Max Relative Error     : {max_rel:.6e}")
    print(f"-----------------------------------------------------------------")
    print(f" Per-Vector Angular Error (Degrees):")
    print(f"  -> Mean Angular Error     : {mean_ang:.4f}°")
    print(f"  -> RMS Angular Error      : {rms_ang:.4f}°")
    print(f"  -> 95th Percentile (p95)  : {p95_ang:.4f}°")
    print(f"  -> 99th Percentile (p99)  : {p99_ang:.4f}°")
    print(f"  -> Max Angular Error      : {max_ang:.4f}°")

    if np.any(mask):
        rel_bins = np.array([0.0, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0, 10.0, np.inf])
        print_ascii_histogram(
            rel_err_masked,
            rel_bins,
            title="Relative Error Distribution",
            unit="",
            is_log_labels=True,
        )

    if np.any(vec_mask):
        ang_bins = np.array([0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 15.0, 45.0, 180.0])
        print_ascii_histogram(
            angular_err_deg,
            ang_bins,
            title="Per-Vector Angular Error Distribution",
            unit="°",
            is_log_labels=False,
        )

    print(f"=================================================================\n")
    return global_rel_norm_err


def test_triangle_gradients(
    vertices: np.ndarray,
    indices: np.ndarray,
    query_mode: str,
    query_count: int,
    beta: None | float,
) -> float:
    triangles = vertices[indices]
    triangles_torch = torch.from_numpy(triangles).to(torch.float32).to("cuda:0")
    grad_output = torch.randn([query_count], dtype=torch.float32, device="cuda:0") * 100
    queries = generate_queries(vertices, query_mode, query_count)
    queries_torch = torch.from_numpy(queries).to(torch.float32).to("cuda:0")

    torch.cuda.synchronize()
    print("Forward:")
    t0 = time()
    wn = winder.brute_force_winding_numbers(triangles_torch, queries_torch)
    torch.cuda.synchronize()
    print(f"Brute force took {time() - t0} sec")
    t0 = time()
    engine = winder.WindingNumberEngine(triangles_torch)
    torch.cuda.synchronize()
    print(f"Building engine took {time() - t0} sec")
    t0 = time()
    wn = engine.compute(queries_torch)
    torch.cuda.synchronize()
    print(f"Computing winding numbers took {time() - t0} sec")

    print("Backward:")
    torch.cuda.synchronize()
    t0 = time()
    gt_grads = torch.from_dlpack(
        winder.brute_force_gradients(grad_output, triangles_torch, queries_torch)
    )
    torch.cuda.synchronize()
    print(f"Brute Force took {time() - t0} sec")

    t0 = time()
    grad_engine = winder.GradientEngine(queries_torch, grad_output)
    torch.cuda.synchronize()
    print(f"Building Engine took {time() - t0} sec")
    with open("/tmp/grad_engine_dump.dot", "w") as f:
        f.write(grad_engine.dump())
    t0 = time()
    grads = torch.from_dlpack(grad_engine.compute(triangles_torch, beta=-1 if beta is None else beta))
    torch.cuda.synchronize()
    print(f"Fast variant took {time() - t0} sec")

    errs = []
    for idx, name in enumerate(["Triangle v0", "Triangle v1", "Triangle v2"]):
        err = validate_gradients(grads[idx].cpu().numpy(), gt_grads[idx].cpu().numpy(), name)
        errs.append(err)

    # Average global relative norm error across all 3 vertices
    return float(np.mean(errs))

def test_mesh_gradients(
    vertices: np.ndarray,
    indices: np.ndarray,
    query_mode: str,
    query_count: int,
    beta: None | float,
) -> float:
    vertices_torch = torch.from_numpy(vertices).to(torch.float32).to("cuda:0")
    indices_torch = torch.from_numpy(indices).to(torch.uint32).to("cuda:0")
    grad_output = torch.randn([query_count], dtype=torch.float32, device="cuda:0") * 100
    queries = generate_queries(vertices, query_mode, query_count)
    queries_torch = torch.from_numpy(queries).to(torch.float32).to("cuda:0")

    torch.cuda.synchronize()
    print("Forward:")
    t0 = time()
    wn = winder.brute_force_winding_numbers(vertices_torch, indices_torch, queries_torch)
    torch.cuda.synchronize()
    print(f"Brute force took {time() - t0} sec")
    t0 = time()
    engine = winder.WindingNumberEngine(vertices_torch, indices_torch)
    torch.cuda.synchronize()
    print(f"Building engine took {time() - t0} sec")
    t0 = time()
    wn = engine.compute(queries_torch)
    torch.cuda.synchronize()
    print(f"Computing winding numbers took {time() - t0} sec")

    print("Backward:")
    torch.cuda.synchronize()
    t0 = time()
    gt_grads = torch.from_dlpack(
        winder.brute_force_gradients(grad_output, vertices_torch, indices_torch, queries_torch)
    )
    torch.cuda.synchronize()
    print(f"Brute Force took {time() - t0} sec")

    t0 = time()
    grad_engine = winder.GradientEngine(queries_torch, grad_output)
    torch.cuda.synchronize()
    print(f"Building Engine took {time() - t0} sec")

    t0 = time()
    grads = torch.from_dlpack(grad_engine.compute(vertices_torch, indices_torch, beta=-1 if beta is None else beta))
    torch.cuda.synchronize()
    print(f"Fast variant took {time() - t0} sec")

    err = validate_gradients(grads.cpu().numpy(), gt_grads.cpu().numpy(), "Vertex")

    return float(err)

def test_point_normal_gradients(
    points: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    query_mode: str,
    query_count: int,
    inv_epsilon: float,
    beta: None | float,
) -> float:
    scaled_normals = normals * areas[:, None]
    points_torch = torch.from_numpy(points).to(torch.float32).to("cuda:0")
    scaled_normals_torch = (
        torch.from_numpy(scaled_normals).to(torch.float32).to("cuda:0")
    )
    grad_output = torch.randn([query_count], dtype=torch.float32, device="cuda:0")
    queries = generate_queries(points, query_mode, query_count)
    queries_torch = torch.from_numpy(queries).to(torch.float32).to("cuda:0")

    torch.cuda.synchronize()
    print("forward:")
    t0 = time()
    wn = winder.brute_force_winding_numbers(points_torch, scaled_normals_torch, queries_torch)
    torch.cuda.synchronize()
    print(f"Brute force took {time() - t0} sec")
    t0 = time()
    engine = winder.WindingNumberEngine(points_torch, scaled_normals_torch)
    torch.cuda.synchronize()
    print(f"Building engine took {time() - t0} sec")
    t0 = time()
    wn = engine.compute(queries_torch)
    torch.cuda.synchronize()
    print(f"Computing winding numbers took {time() - t0} sec")

    print("Backward:")
    t0 = time()
    gt_grads = torch.from_dlpack(
        winder.brute_force_gradients(
            grad_output,
            points_torch,
            scaled_normals_torch,
            queries_torch,
            epsilon=1 / inv_epsilon,
        )
    )
    torch.cuda.synchronize()
    print(f"Brute Force took {time() - t0} sec")

    t0 = time()
    grad_engine = winder.GradientEngine(queries_torch, grad_output)
    torch.cuda.synchronize()
    print(f"Building engine took {time() - t0} sec")

    t0 = time()
    grads = torch.from_dlpack(
        grad_engine.compute(
            points_torch, scaled_normals_torch, epsilon=1 / inv_epsilon, beta=-1 if beta is None else beta
        )
    )
    torch.cuda.synchronize()
    print(f"Fast Grads took {time() - t0} sec")

    errs = []
    for idx, name in enumerate(["PointNormal n", "PointNormal p"]):
        err = validate_gradients(grads[idx].cpu().numpy(), gt_grads[idx].cpu().numpy(), name)
        errs.append(err)

    # Average global relative norm error across normal n and position p
    return float(np.mean(errs))


def write_run_to_csv(
    csv_path: str,
    unique_label: str,
    raw_label: str,
    args: argparse.Namespace,
    pn_error: float | None,
    tri_error: float | None,
    mesh_error: float | None,
):
    """Appends a single summary row for the entire benchmark run into the CSV."""
    file_exists = os.path.exists(csv_path)

    fieldnames = [
        "timestamp",
        "label",
        "unique_label",
        "geometry_type",
        "query_mode",
        "query_count",
        "beta",
        "inv_epsilon",
        "pn_rel_norm_err",
        "tri_rel_norm_err",
        "mesh_rel_norm_err",
    ]

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "label": raw_label,
        "unique_label": unique_label,
        "geometry_type": args.geometry_type,
        "query_mode": args.query_mode,
        "query_count": args.query_count,
        "beta": args.beta if args.beta is not None else -1.0,
        "inv_epsilon": args.inv_epsilon,
        "pn_rel_norm_err": f"{pn_error:.6e}" if pn_error is not None else "",
        "tri_rel_norm_err": f"{tri_error:.6e}" if tri_error is not None else "",
        "mesh_rel_norm_err": f"{mesh_error:.6e}" if mesh_error is not None else "",
    }

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"[CSV Bookkeeping] Written 1 run entry to: {csv_path}")


def plot_comparison(csv_path: str):
    """Reads all historical runs from CSV and displays a grouped bar chart of PointNormal vs Triangle errors."""
    if not os.path.exists(csv_path):
        print(f"CSV file '{csv_path}' not found for plotting.")
        return

    labels = []
    pn_errs = []
    tri_errs = []

    with open(csv_path, mode="r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(row["unique_label"])
            pn_errs.append(float(row["pn_rel_norm_err"]) if row["pn_rel_norm_err"] else np.nan)
            tri_errs.append(float(row["tri_rel_norm_err"]) if row["tri_rel_norm_err"] else np.nan)

    if not labels:
        print("No valid runs found in CSV to plot.")
        return

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    
    rects1 = ax.bar(x - width/2, pn_errs, width, label="PointNormal (Avg)", color="#1f77b4", edgecolor="black")
    rects2 = ax.bar(x + width/2, tri_errs, width, label="Triangle (Avg Vertices)", color="#ff7f0e", edgecolor="black")

    ax.set_yscale("log")
    ax.set_ylabel("Global Relative Norm Error (Log Scale)", fontweight="bold")
    ax.set_title("Gradient Relative Error Comparison Across Runs", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.legend(loc="upper right")
    ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5, alpha=0.7)

    # Label value formatted in scientific notation above bars
    for rect in list(rects1) + list(rects2):
        height = rect.get_height()
        if not np.isnan(height):
            ax.annotate(
                f"{height:.2e}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=8, rotation=45
            )

    plt.tight_layout()
    plt.show()

def slice_mesh_in_half(vertices: np.ndarray, indices: np.ndarray, dim: int = 2):
    """Slices a mesh by keeping faces whose centroids are above the mean position along `dim`."""
    # Compute face centroids [N, 3]
    face_verts = vertices[indices]  # [N, 3, 3]
    centroids = face_verts.mean(axis=1)  # [N, 3]

    # Slice at the median/mean along specified axis
    split_plane = np.median(centroids[:, dim])
    mask = centroids[:, dim] > split_plane

    sliced_indices = indices[mask]
    return vertices, sliced_indices

def drop_half_of_the_triangles(vertices: np.ndarray, indices: np.ndarray):
    """Slices a mesh by keeping faces whose centroids are above the mean position along `dim`."""
    choice = np.random.choice(np.arange(len(indices)), len(indices) // 2, replace=False)
    return vertices, indices[choice]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify generalized winding number analytical gradients against PyTorch float64 autograd."
    )
    parser.add_argument(
        "--obj_file",
        type=str,
        required=True,
        help="Path to input wave-front .obj mesh file.",
    )
    parser.add_argument(
        "--geometry_type",
        type=str,
        choices=["PointNormal", "Triangle", "Mesh", "All"],
        default="All",
        help="Choose primitive type",
    )
    parser.add_argument(
        "--query_mode",
        type=str,
        choices=["random", "grid"],
        default="random",
        help="Distribution algorithm geometry configuration for query target fields.",
    )
    parser.add_argument(
        "--query_count",
        type=int,
        default=1000,
        help="Number of query points",
    )
    parser.add_argument(
        "--inv_epsilon",
        type=float,
        default=250.0,
        help="Inverse regularization scale for PointNormal surfels",
    )
    parser.add_argument(
        "--beta",
        type=float,
        help="How much to approximate",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Label for this benchmark run. If set, results are written to CSV and visualized.",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="benchmark_results.csv",
        help="Path to CSV file where metrics are recorded.",
    )

    args = parser.parse_args()

    print(f"Loading mesh structural data from: {args.obj_file}")
    vertices, _, _, indices, _, _ = igl.readOBJ(args.obj_file)
    vertices, indices = drop_half_of_the_triangles(*slice_mesh_in_half(vertices, indices))

    print(f"INFO: object has {len(indices)} Triangles/PointNormals")

    pn_error = None
    tri_error = None

    if args.geometry_type in ["PointNormal", "All"]:
        points, normals, areas = mesh_to_point_surfels(vertices, indices)
        pn_error = test_point_normal_gradients(
            points,
            normals,
            areas,
            args.query_mode,
            args.query_count,
            args.inv_epsilon,
            args.beta,
        )

    if args.geometry_type in ["Triangle", "All"]:
        tri_error = test_triangle_gradients(
            vertices, indices, args.query_mode, args.query_count, args.beta
        )

    if args.geometry_type in ["Mesh", "All"]:
        mesh_error = test_mesh_gradients(
            vertices, indices, args.query_mode, args.query_count, args.beta
        )

    if args.label:
        timestamp_label = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.label}"
        write_run_to_csv(args.csv_path, timestamp_label, args.label, args, pn_error, tri_error, mesh_error)
        plot_comparison(args.csv_path)
