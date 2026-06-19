#!/usr/bin/env python3
import os
import sys
import glob
import argparse
from matplotlib.scale import scale_factory
import numpy as np
import torch
import igl

# Ensure we can import winder if it's in the build directory or path
try:
    import winder
except ImportError:
    print(
        "Error: 'winder' module not found. Make sure it is installed or in your PYTHONPATH."
    )
    sys.exit(1)


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

    # Add a tiny epsilon to prevent division by zero on degenerate triangles
    safe_magnitudes = np.where(magnitudes == 0, 1e-8, magnitudes)
    normals = cross / safe_magnitudes

    return (
        points.astype(np.float32),
        normals.astype(np.float32),
        areas.astype(np.float32),
    )


def generate_query_grid(
    vertices: np.ndarray, grid_resolution: int = 64, scale_factor: float = 1.0
) -> np.ndarray:
    """Generates a uniform 3D query grid slightly padded around the mesh AABB."""
    v_min = np.min(vertices, axis=0) * scale_factor
    v_max = np.max(vertices, axis=0) * scale_factor

    # Pad by 10% to capture external, near-surface, and deep internal fields
    padding = (v_max - v_min) * 0.1
    v_min -= padding
    v_max += padding

    x = np.linspace(v_min[0], v_max[0], grid_resolution, dtype=np.float32)
    y = np.linspace(v_min[1], v_max[1], grid_resolution, dtype=np.float32)
    z = np.linspace(v_min[2], v_max[2], grid_resolution, dtype=np.float32)

    xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")
    return np.stack([xv.flatten(), yv.flatten(), zv.flatten()], axis=-1)


def run_mesh_validation(obj_path: str, device: torch.device, args) -> bool:
    print(f"\nProcessing: {os.path.basename(obj_path)}")
    print("-" * 60)

    try:
        vertices, _, _, indices, _, _ = igl.readOBJ(obj_path)
    except Exception as e:
        print(f"❌ Failed to parse OBJ file via libigl: {e}")
        return False

    if len(indices) == 0:
        print("❌ Mesh contains no valid face indices.")
        return False

    # Apply spatial mutations or normalizations depending on the test profile
    v_min = np.min(vertices, axis=0)
    v_max = np.max(vertices, axis=0)
    center = (v_min + v_max) / 2.0

    if args.normalize:
        # Isotropic scale mapping to fit comfortably inside a [-1, 1] AABB
        scale = 2.0 / np.max(v_max - v_min)
        vertices = (vertices - center) * scale
        print(" -> Applied isotropic normalization to [-1, 1] AABB.")

    if args.stress_scale != 1.0:
        # Intentionally blow up coordinates to test float16 overflow bounds
        vertices = vertices * args.stress_scale
        print(f" -> Applied stress scale factor multiplier: {args.stress_scale}")

    vertices_np = vertices.astype(np.float32)
    indices_np = np.asarray(indices).astype(np.uint32)

    # 1. Prepare shared query grid and move to device
    query_points = generate_query_grid(
        vertices_np, grid_resolution=args.resolution, scale_factor=args.scale_factor
    )
    query_torch = torch.from_numpy(query_points).to(device)
    total_queries = len(query_points)

    # Track overall validation pass status
    point_mode_passed = False
    mesh_mode_passed = False

    stream = torch.cuda.current_stream().cuda_stream

    # =========================================================================
    # PHASE 1: POINT MODE VALIDATION (SURFELS)
    # =========================================================================
    print(" -> Running POINT MODE evaluation...")
    points, normals, areas = mesh_to_point_surfels(vertices_np, indices_np)

    points_torch = torch.from_numpy(points).to(device)
    normals_torch = torch.from_numpy(normals).to(device)
    areas_torch = torch.from_numpy(areas).to(device)
    normals_torch = normals_torch * areas_torch[..., None]

    try:
        engine_point = winder.WinderEngine(points_torch, normals_torch)
        torch.cuda.synchronize()

        output_dlpack_pt = engine_point.compute(query_torch, stream=stream)
        torch.cuda.synchronize()
        wn_point = torch.from_dlpack(output_dlpack_pt).cpu().numpy()

        pt_nans = np.isnan(wn_point).sum()
        pt_infs = np.isinf(wn_point).sum()

        if pt_nans == 0 and pt_infs == 0:
            print(f"   ✅ POINT MODE PASS ({total_queries} queries clean)")
            point_mode_passed = True
        else:
            print(f"   💥 POINT MODE FAIL! NaNs: {pt_nans}, Infs: {pt_infs}")

    except Exception as e:
        print(f"   ❌ POINT MODE CRASHED: {e}")

    # =========================================================================
    # PHASE 2: MESH MODE VALIDATION (VERTICES & INDICES)
    # =========================================================================
    print(" -> Running MESH MODE evaluation...")
    vertices_torch = torch.from_numpy(vertices_np).to(device)
    indices_torch = torch.from_numpy(indices_np).to(device)

    try:
        engine_mesh = winder.WinderEngine(vertices_torch, indices_torch)
        torch.cuda.synchronize()

        output_dlpack_mesh = engine_mesh.compute(query_torch, stream=stream)
        torch.cuda.synchronize()
        wn_mesh = torch.from_dlpack(output_dlpack_mesh).cpu().numpy()

        mesh_nans = np.isnan(wn_mesh).sum()
        mesh_infs = np.isinf(wn_mesh).sum()

        if mesh_nans == 0 and mesh_infs == 0:
            print(f"   ✅ MESH MODE PASS ({total_queries} queries clean)")
            mesh_mode_passed = True
        else:
            print(f"   💥 MESH MODE FAIL! NaNs: {mesh_nans}, Infs: {mesh_infs}")

    except Exception as e:
        print(f"   ❌ MESH MODE CRASHED: {e}")

    # =========================================================================
    # SUMMARY REPORT FOR ASSET
    # =========================================================================
    if point_mode_passed and mesh_mode_passed:
        print(
            f"✅ PASS: {os.path.basename(obj_path)} is fully stable in both execution modes."
        )
        return True
    else:
        print(f"❌ FAIL: Stability conditions unmet for {os.path.basename(obj_path)}.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Winder float16 Numerical Expansion Stability Test Suite"
    )
    parser.add_argument(
        "--gpu", type=int, default=0, help="Target GPU index platform runner"
    )
    parser.add_argument(
        "--resolution", type=int, default=64, help="Grid dimension size (N x N x N)"
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        default=1.0,
        help="Scale of the query grid. 1 means AABB of mesh.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Force isotropic unit scaling prior to parsing",
    )
    parser.add_argument(
        "--stress-scale",
        type=float,
        default=1.0,
        help="Multiply coordinates by a scale factor to stress test float16 boundaries",
    )
    parser.add_argument(
        "--meshes-directory",
        type=str,
        default="examples/meshes",
        help="Directory holding .obj files to test.",
    )
    args = parser.parse_args()

    # Determine execution platform context
    if not torch.cuda.is_available():
        print("CRITICAL: CUDA execution runtime environment missing.")
        sys.exit(1)

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    mesh_pattern = os.path.abspath(os.path.join(args.meshes_directory, "**", "*.obj"))
    mesh_paths = glob.glob(mesh_pattern, recursive=True)

    if len(mesh_paths) == 0:
        print(f"Error: No target asset files found matching pattern: {mesh_pattern}")
        sys.exit(1)

    print("============================================================")
    print("          WINDER DATASET STABILITY EVALUATION PASS          ")
    print("============================================================")
    print(f"Found {len(mesh_paths)} testing configurations targets.")
    print(
        f"Evaluating with grid resolution: {args.resolution}^3 ({args.resolution**3} queries per asset)"
    )

    failed_meshes = []

    for path in mesh_paths:
        passed = run_mesh_validation(path, device, args)
        if not passed:
            failed_meshes.append(os.path.basename(path))

    print("\n" + "=" * 60)
    print("                     FINAL RESULTS REPORT                   ")
    print("=" * 60)
    if not failed_meshes:
        print("🎉 ALL MESHES PASSED. Taylor series expansion is numerically stable.")
        sys.exit(0)
    else:
        print(
            f"💥 FAILURE: {len(failed_meshes)} assets produced unstable output fields:"
        )
        for name in failed_meshes:
            print(f"  - {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
