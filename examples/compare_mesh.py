#!/usr/bin/env python3
import gc
from tqdm.auto import tqdm
from time import time
import argparse
import imageio.v3 as iio
import cv2
import igl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import numpy.typing as NPT
import open3d as o3d
import torch
import winder
import csv
import os

# Optional: Import winder depending on environment setup
# import winder


def positive_type(arg: str) -> int:
    x: int = int(arg)
    if x < 1:
        raise argparse.ArgumentTypeError("Minimum value is 1")
    return x


@torch.no_grad()
@torch.compile(mode="reduce-overhead")
def torch_winding_numbers(
    triangles: torch.Tensor, queries: torch.Tensor
) -> torch.Tensor:
    """Computes winding numbers using PyTorch with a solid angle formulation.

    Args:
        triangles: float32 shape [N, 3, 3] on CUDA
        queries: float32 shape [M, 3] on CUDA

    Returns:
        float32 shape [M] winding numbers at query positions.
    """
    num_queries = queries.shape[0]
    total_winding_numbers = torch.zeros(
        (num_queries,), dtype=torch.float32, device=queries.device
    )

    # Reshape queries for explicit broadcasting: [M, 1, 3]
    q = queries.unsqueeze(1)

    # Vectors from queries to triangle vertices: [M, N, 3]
    A = triangles[:, 0, :] - q
    B = triangles[:, 1, :] - q
    C = triangles[:, 2, :] - q

    # Magnitudes
    norm_A = torch.linalg.norm(A, dim=-1)
    norm_B = torch.linalg.norm(B, dim=-1)
    norm_C = torch.linalg.norm(C, dim=-1)

    # Determinant (Numerator): A . (B x C)
    cross_BC = torch.cross(B, C, dim=-1)
    num = torch.sum(A * cross_BC, dim=-1)

    # Denominator terms
    dot_AB = torch.sum(A * B, dim=-1)
    dot_BC = torch.sum(B * C, dim=-1)
    dot_CA = torch.sum(C * A, dim=-1)

    den = (
        (norm_A * norm_B * norm_C)
        + (dot_AB * norm_C)
        + (dot_BC * norm_A)
        + (dot_CA * norm_B)
    )

    # Solid angle calculation
    solid_angle = 2.0 * torch.atan2(num, den)
    total_winding_numbers += torch.sum(solid_angle, dim=-1)

    # Final division by 4 * PI to match standard winding definition
    return total_winding_numbers / (4.0 * np.pi)


def add_duration_text_to_frames(
    frames: NPT.ArrayLike, total_duration: float, duration_per_frame: float
) -> NPT.ArrayLike:
    """Draws metrics info texts onto the frames array safely using OpenCV."""
    text_total = f"total: {total_duration:.2f} s"
    text_frame = f"per frame: {duration_per_frame:.2e} s"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (255, 255, 255)  # White text
    thickness = 2
    line_type = cv2.LINE_AA

    for i in range(frames.shape[0]):
        # OpenCV requires continuous arrays and writes in place
        frame = np.ascontiguousarray(frames[i])
        cv2.putText(
            frame, text_total, (15, 30), font, font_scale, color, thickness, line_type
        )
        cv2.putText(
            frame, text_frame, (15, 55), font, font_scale, color, thickness, line_type
        )
        frames[i] = frame
    return frames


def add_geometry_text_to_frames(
    frames: NPT.ArrayLike, vertex_count: int, triangle_count: int
) -> NPT.ArrayLike:
    """Draws metrics info texts onto the frames array safely using OpenCV."""
    text_total = f"vertices: {vertex_count}"
    text_frame = f"triangles: {triangle_count}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (0, 0, 0)  # Black text
    thickness = 2
    line_type = cv2.LINE_AA

    for i in range(len(frames)):
        # OpenCV requires continuous arrays and writes in place
        frame = np.ascontiguousarray(frames[i])
        cv2.putText(
            frame, text_total, (15, 30), font, font_scale, color, thickness, line_type
        )
        cv2.putText(
            frame, text_frame, (15, 55), font, font_scale, color, thickness, line_type
        )
        frames[i] = frame
    return frames


def create_vis_mesh(
    vertices: NPT.ArrayLike,
    indices: NPT.ArrayLike,
    query_list: list[NPT.ArrayLike],
    resolution: int,
    video_path: str,
    mode: str,
    device: torch.device,
    add_duration_string: bool = True,
) -> tuple[dict, NPT.ArrayLike]:
    duration = 0.0
    duration_per_frame = 0.0
    winding_numbers = []

    metrics = {
        "mode": mode,
        "upload_time_sec": 0.0,
        "build_time_sec": 0.0,
        "compute_time_sec": 0.0,
        "download_time_sec": 0.0,
    }

    winding_numbers = []

    match mode:
        case "igl":
            print("Computing winding numbers with IGL...")
            start_time = time()
            winding_numbers = [
                igl.fast_winding_number(vertices, indices, q) for q in query_list
            ]
            end_time = time()
            duration = end_time - start_time
            duration_per_frame = duration / len(query_list)
            print(f"Done. Winding Number calculation took {duration:.4f} sec.")
            metrics["compute_time_sec"] = duration

        case "winder":
            print("Uploading vertices, indices and query_list...")
            start_upload_time = time()
            vertices_torch = torch.from_numpy(vertices).to(device)
            indices_torch = torch.from_numpy(indices).to(device)
            query_list_torch = [torch.from_numpy(q).to(device) for q in query_list]
            end_upload_time = time()
            print(f"Done. Upload took {end_upload_time - start_upload_time:.4f} sec.")

            print("Building winder engine...")
            start_build_time = time()
            engine = winder.WinderEngine(vertices_torch, indices_torch)
            torch.cuda.synchronize()
            end_build_time = time()
            print(
                f"Done. Building Engine took {end_build_time - start_build_time:.4f} sec."
            )

            print("Computing winding numbers with WINDER...")
            start_time = time()
            winding_numbers = [engine.compute(q) for q in query_list_torch]
            end_time = time()
            duration = end_time - start_time
            duration_per_frame = duration / len(query_list)
            print(f"Done. Winding Number calculation took {duration:.4f} sec.")

            print("Downloading winding Numbers...")
            start_download = time()
            winding_numbers = [
                torch.from_dlpack(w).cpu().numpy() for w in winding_numbers
            ]
            end_download = time()
            print("Done.")

            metrics["upload_time_sec"] = end_upload_time - start_upload_time
            metrics["build_time_sec"] = end_build_time - start_build_time
            metrics["compute_time_sec"] = duration
            metrics["download_time_sec"] = end_download - start_download

        case "brute_force":
            print("Uploading vertices, indices, and query_list...")
            start_upload_time = time()
            vertices_torch = torch.from_numpy(vertices).to(device)
            indices_torch = torch.from_numpy(indices).to(device)
            query_list_torch = [torch.from_numpy(q).to(device) for q in query_list]
            end_upload_time = time()
            print(f"Done. Upload took {end_upload_time - start_upload_time:.4f} sec.")

            print("Building winder engine...")
            start_build_time = time()
            engine = winder.WinderEngine(vertices_torch, indices_torch)
            torch.cuda.synchronize()
            end_build_time = time()
            print(
                f"Done. Building Engine took {end_build_time - start_build_time:.4f} sec."
            )

            print("Computing winding numbers with WINDER BRUTE FORCE...")
            start_time = time()
            winding_numbers = [engine.brute_force(q) for q in tqdm(query_list_torch)]
            end_time = time()
            duration = end_time - start_time
            duration_per_frame = duration / len(query_list)
            print(f"Done. Winding Number calculation took {duration:.4f} sec.")

            print("Downloading winding Numbers...")
            start_download = time()
            winding_numbers = [
                torch.from_dlpack(w).cpu().numpy() for w in winding_numbers
            ]
            end_download = time()
            print("Done.")

            metrics["upload_time_sec"] = end_upload_time - start_upload_time
            metrics["build_time_sec"] = end_build_time - start_build_time
            metrics["compute_time_sec"] = duration
            metrics["download_time_sec"] = end_download - start_download

        case "torch":
            print("Uploading vertices, indices and query list...")
            start_upload_time = time()
            vertices_torch = torch.from_numpy(vertices).to(device)
            indices_torch = torch.from_numpy(indices).to(device)
            query_list_torch = [torch.from_numpy(q).to(device) for q in query_list]
            end_upload_time = time()
            metrics["upload_time_sec"] = end_upload_time - start_upload_time
            metrics["build_time_sec"] = (
                0.0  # Torch doesn't have a distinct build step here
            )

            print("Computing winding numbers using torch implementation...")
            triangles = vertices_torch[indices_torch.long()]
            # compile torch function first
            print("Compiling...")
            _ = torch_winding_numbers(triangles, query_list_torch[0])
            print("Computing...")
            start_time = time()
            winding_numbers = []

            for q in tqdm(query_list_torch, desc="torch"):
                wn = torch_winding_numbers(triangles, q)
                winding_numbers.append(wn.clone())

            torch.cuda.synchronize()
            end_time = time()
            duration = end_time - start_time
            duration_per_frame = duration / len(query_list)
            metrics["compute_time_sec"] = duration
            print(f"Done. Winding Number calculation took {duration:.4f} sec.")

            print("Downloading winding Numbers...")
            start_download = time()
            winding_numbers = [w.cpu().numpy() for w in winding_numbers]
            metrics["download_time_sec"] = time() - start_download
            print("Done.")

        case _:
            raise ValueError("Unsupported mode.")

    print("Applying colormap...")
    winding_numbers = np.stack(winding_numbers)
    winding_numbers_vis = winding_numbers.reshape([len(query_list), resolution, resolution])
    winding_numbers_vis = np.clip(winding_numbers_vis, -2, 2)

    norm = mcolors.TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)
    cmap = plt.get_cmap("vanimo")
    winding_numbers_color = cmap(norm(winding_numbers_vis))[..., :3]
    winding_numbers_frames = (winding_numbers_color * 255).astype(np.uint8)

    if add_duration_string:
        print("Adding text overlays to frames...")
        winding_numbers_frames = add_duration_text_to_frames(
            winding_numbers_frames, duration, duration_per_frame
        )
        print("Done.")

    print(f"Writing video to {video_path}")
    iio.imwrite(
        video_path,
        winding_numbers_frames,
        extension=".mp4",
        fps=25,
        codec="libx264",
        is_batch=True,
    )
    print("Done.")

    return metrics, winding_numbers


def create_open3d_diagnostic_video(
    vertices: NPT.ArrayLike,
    indices: NPT.ArrayLike,
    query_frames: list[NPT.ArrayLike],
    resolution: int,
    video_path: str,
    add_geometry_detail_string: bool = True,
) -> None:
    """Generates a 3D diagnostic video using Open3D's native OffscreenRenderer."""
    print("Generating Open3D diagnostic tracking scene video natively headless...")

    # 1. Create geometries
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(indices.astype(np.int32))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.2, 0.4, 0.8])  # Soft blue

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(query_frames[0])
    pcd.paint_uniform_color([1.0, 0.1, 0.1])  # Bright red

    # 2. Use OffscreenRenderer instead of legacy Visualizer
    render = o3d.visualization.rendering.OffscreenRenderer(resolution, resolution)

    # Setup materials (required for modern OffscreenRenderer)
    mtl_mesh = o3d.visualization.rendering.MaterialRecord()
    mtl_mesh.base_color = [0.2, 0.4, 0.8, 1.0]
    mtl_mesh.shader = "defaultLit"

    mtl_pcd = o3d.visualization.rendering.MaterialRecord()
    mtl_pcd.base_color = [1.0, 0.1, 0.1, 1.0]
    mtl_pcd.point_size = 2.0
    mtl_pcd.shader = "defaultUnlit"

    render.scene.add_geometry("mesh", mesh, mtl_mesh)
    render.scene.add_geometry("pcd", pcd, mtl_pcd)

    # 3. Position the Camera Eye
    # center/lookat, eye/front, up
    render.scene.camera.look_at([0.0, 0.0, 0.0], [1.0, 0.8, 1.2], [0.0, 1.0, 0.0])

    frames = []
    for frame_queries in query_frames:
        # Update point positions
        pcd.points = o3d.utility.Vector3dVector(frame_queries)

        # In modern renderer, you remove and re-add or call post_redraw
        render.scene.remove_geometry("pcd")
        render.scene.add_geometry("pcd", pcd, mtl_pcd)

        # Render frame
        image = render.render_to_image()
        frame_buffer = np.asarray(image)
        frames.append(frame_buffer)

    if add_geometry_detail_string:
        print("Adding text overlays to frames...")
        frames = add_geometry_text_to_frames(
            frames, vertices.shape[0], indices.shape[0]
        )
        print("Done.")
    print(f"Writing video {video_path}")
    iio.imwrite(
        video_path,
        np.stack(frames),
        extension=".mp4",
        fps=25,
        codec="libx264",
        is_batch=True,
    )
    print("Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--obj_path", type=str, required=True, help="Mesh object (.obj) path."
    )
    parser.add_argument("--gpu", type=positive_type, default=0, help="GPU index.")
    parser.add_argument(
        "--resolution", type=positive_type, default=512, help="Plane pixel resolution."
    )
    parser.add_argument(
        "--frames_per_sequence",
        type=positive_type,
        default=200,
        help="Frames per shift sweep.",
    )
    parser.add_argument(
        "--video_prefix", type=str, required=True, help="Filename prefix string."
    )
    parser.add_argument("--methods", nargs="+", default=["igl", "winder"])
    parser.add_argument(
        "--no_quantitative_comparison",
        action="store_true",
        help="Skip calculating quantitative error values (MSE, MAE, RMSE) against brute force.",
    )
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    else:
        raise RuntimeError("I need a GPU! torch.cuda.is_available is false")
    vertices, _, _, indices, _, _ = igl.readOBJ(args.obj_path)

    # Normalize coordinates isotropically to stay within AABB [-1, 1]
    v_min = np.min(vertices, axis=0)
    v_max = np.max(vertices, axis=0)
    center = (v_min + v_max) / 2.0
    scale = 2.0 / np.max(v_max - v_min)
    normalized_vertices = (vertices - center) * scale

    vertices_np = normalized_vertices.astype(np.float32)
    indices_np = np.asarray(indices).astype(np.uint32)

    print(
        f"Mesh contains {vertices_np.shape[0]} vertices and {indices_np.shape[0]} triangles."
    )

    # Define underlying reference tracking grids
    line = np.linspace(-1, 1, args.resolution, dtype=np.float32)
    u, v = np.meshgrid(line, line, indexing="ij")
    zeros = np.zeros_like(u)

    base_xy = np.stack([u, v, zeros], axis=-1).reshape([-1, 3])
    base_xz = np.stack([u, zeros, v], axis=-1).reshape([-1, 3])
    base_yz = np.stack([zeros, u, v], axis=-1).reshape([-1, 3])

    query_frames_np: list[NPT.ArrayLike] = []
    F = args.frames_per_sequence
    steps = np.linspace(-1, 1, F, dtype=np.float32)
    angles = np.linspace(0, np.pi, F, dtype=np.float32)

    # 1. Sweep XY plane along Z-Axis
    for t in steps:
        frame = base_xy.copy()
        frame[:, 2] = t
        query_frames_np.append(frame)

    # 2. Sweep XZ plane along Y-Axis
    for t in steps:
        frame = base_xz.copy()
        frame[:, 1] = t
        query_frames_np.append(frame)

    # 3. Sweep YZ plane along X-Axis
    for t in steps:
        frame = base_yz.copy()
        frame[:, 0] = t
        query_frames_np.append(frame)

    # 4. Rotation of XY query plane around X-Axis
    for alpha in angles:
        cos_a, sin_a = np.cos(alpha), np.sin(alpha)
        R = np.array(
            [[1, 0, 0], [0, cos_a, -sin_a], [0, sin_a, cos_a]], dtype=np.float32
        )
        query_frames_np.append(base_xy @ R.T)

    # 5. Rotation of XY query plane around Y-Axis
    for alpha in angles:
        cos_a, sin_a = np.cos(alpha), np.sin(alpha)
        R = np.array(
            [[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]], dtype=np.float32
        )
        query_frames_np.append(base_xy @ R.T)

    # 6. Rotation of XZ query plane around Z-Axis
    for alpha in angles:
        cos_a, sin_a = np.cos(alpha), np.sin(alpha)
        R = np.array(
            [[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=np.float32
        )
        query_frames_np.append(base_xz @ R.T)

    directory = os.path.dirname(args.video_prefix)
    csv_filename = f"{directory}/benchmark_metrics.csv"
    file_exists = os.path.isfile(csv_filename)

    methods_to_execute = list(args.methods)
    if not args.no_quantitative_comparison:
        if "brute_force" in methods_to_execute:
            methods_to_execute.remove("brute_force")
        methods_to_execute.insert(0, "brute_force")

    gt_winding_numbers = None

    # Compute selected evaluations sequentially
    for method in methods_to_execute:
        output_video_path = f"{args.video_prefix}_{method}.mp4"

        run_metrics, raw_wn = create_vis_mesh(
            vertices_np,
            indices_np,
            query_frames_np,
            args.resolution,
            output_video_path,
            method,
            device,
        )

        run_metrics["mesh_name"] = os.path.basename(args.obj_path)
        run_metrics["vertices"] = vertices_np.shape[0]
        run_metrics["triangles"] = indices_np.shape[0]
        run_metrics["resolution"] = args.resolution
        run_metrics["frames"] = len(query_frames_np)

        # Handle quantitative math tracking against brute force baseline
        if method == "brute_force":
            gt_winding_numbers = raw_wn
            run_metrics["mse"] = 0.0
            run_metrics["mae"] = 0.0
            run_metrics["rmse"] = 0.0
        else:
            if gt_winding_numbers is not None and not args.no_quantitative_comparison:
                mse_val = float(np.mean((raw_wn - gt_winding_numbers) ** 2))
                mae_val = float(np.mean(np.abs(raw_wn - gt_winding_numbers)))
                rmse_val = float(np.sqrt(mse_val))

                run_metrics["mse"] = mse_val
                run_metrics["mae"] = mae_val
                run_metrics["rmse"] = rmse_val
            else:
                run_metrics["mse"] = "N/A"
                run_metrics["mae"] = "N/A"
                run_metrics["rmse"] = "N/A"

        # Append immediately to CSV so data is safe if a later mode crashes
        with open(csv_filename, mode="a", newline="") as csvfile:
            fieldnames = [
                "mesh_name",
                "vertices",
                "triangles",
                "resolution",
                "frames",
                "mode",
                "upload_time_sec",
                "build_time_sec",
                "compute_time_sec",
                "download_time_sec",
                "mse",
                "mae",
                "rmse",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()
                file_exists = True

            writer.writerow(run_metrics)

        torch.cuda.empty_cache()
        gc.collect()

    diagnostic_filename = f"{args.video_prefix}_3d_scene.mp4"
    create_open3d_diagnostic_video(
        vertices_np,
        indices_np,
        query_frames_np,
        args.resolution,
        diagnostic_filename,
    )


if __name__ == "__main__":
    main()
