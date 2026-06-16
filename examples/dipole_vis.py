#!/usr/bin/env python3
import argparse
import openvdb
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as NPT
import torch
from tqdm.auto import tqdm
import winder


def positive_type(arg: str) -> int:
    x: int = int(arg)
    if x < 1:
        raise argparse.ArgumentTypeError("Minimum value is 1")
    return x


def apply_colormap_gpu(
    winding_numbers, resolution: int, cmap_name: str = "vanimo", vmin=-2.0, vmax=2.0
) -> NPT.ArrayLike:
    """Colorizes winding number frames entirely on the GPU using a Look-Up Table."""
    device = (
        winding_numbers[0].device
        if isinstance(winding_numbers, list) and torch.is_tensor(winding_numbers[0])
        else "cuda:0"
    )

    # Generate the Colormap Look-Up Table (LUT) once on CPU, then send to GPU
    cmap = plt.get_cmap(cmap_name)
    lut_np = cmap(np.linspace(0, 1, 256))[..., :3]  # Extract RGB [256, 3]
    lut = torch.tensor(lut_np, dtype=torch.float32, device=device)

    if isinstance(winding_numbers, list):
        if torch.is_tensor(winding_numbers[0]):
            wn_tensor = torch.stack(winding_numbers)
        else:
            wn_tensor = torch.tensor(np.array(winding_numbers), device=device)
    elif isinstance(winding_numbers, np.ndarray):
        wn_tensor = torch.from_numpy(winding_numbers).to(device)
    else:
        wn_tensor = winding_numbers.to(device)

    wn_tensor = wn_tensor.view(-1, resolution, resolution)

    # Maps [vmin, vcenter] -> [0, 0.5] and [vcenter, vmax] -> [0.5, 1.0]
    vcenter = 0
    wn_clip = torch.clamp(wn_tensor, vmin, vmax)

    # Piecewise linear normalization matching Matplotlib's TwoSlopeNorm behavior
    normed = torch.where(
        wn_clip < vcenter,
        0.5 * (wn_clip - vmin) / (vcenter - vmin),
        0.5 + 0.5 * (wn_clip - vcenter) / (vmax - vcenter),
    )

    indices = (normed * 255).long()
    color_tensor = lut[indices]  # Shape: [Frames, Res, Res, 3]

    return (color_tensor * 255).to(torch.uint8).cpu().numpy()


def write_video(
    video_path: str, frames_numpy_array: NPT.ArrayLike, fps=25, is_lossless: bool = True
):
    with iio.imopen(video_path, "w", plugin="pyav") as file:
        # FIX: Changed "libx264rgb" to "libx264" to properly support mobile-friendly yuv420p
        file.init_video_stream("libx264", fps=fps, pixel_format="yuv420p")

        if is_lossless:
            # high profile, visually lossless
            file._video_stream.options = {"crf": "0", "preset": "slow"}
        else:
            file._video_stream.options = {"crf": "18", "preset": "slow"}

        for frame in tqdm(frames_numpy_array, desc="writing video"):
            file.write_frame(frame)


def export_to_vdb(field: torch.Tensor, filename: str):
    winding_number_data = field.detach().cpu().numpy().astype(np.float64)

    wind_ct = openvdb.FloatGrid()
    wind_ct.copyFromArray(winding_number_data)
    wind_ct.name = "winding_field"
    resolution = winding_number_data.shape[0]
    voxel_size = 2.0 / resolution  # Spans 2.0 total units (-1 to 1)
    wind_ct.transform = openvdb.createLinearTransform(voxelSize=voxel_size)

    openvdb.write(filename, [wind_ct])
    print(f"Exported scaled VDB: {filename}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resolution", type=positive_type, default=512, help="Grid resolution"
    )
    parser.add_argument(
        "--prefix", type=str, required=True, help="Filename prefix string."
    )
    parser.add_argument("--type", choices=["dipole", "triangle"], default="dipole")
    args = parser.parse_args()

    print(
        f"Generating {args.resolution}^3 grid queries from [-1, -1, -1] to [1, 1, 1]..."
    )

    lin = torch.linspace(-1.0, 1.0, args.resolution, device="cuda:0")
    # Using 'ij' indexing guarantees that x varies first, acting perfectly as our time/frame axis
    grid_x, grid_y, grid_z = torch.meshgrid(lin, lin, lin, indexing="ij")
    queries = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)

    if args.type == "dipole":
        points = torch.tensor(
            [[0, 0, 0]], dtype=torch.float32, device="cuda:0"
        ).reshape([1, 3])
        normals = torch.tensor(
            [[0, 0, 1]], dtype=torch.float32, device="cuda:0"
        ).reshape([1, 3])
        engine = winder.WinderEngine(points, normals)
    else:
        triangles = torch.tensor(
            [[[-0.7, -0.7, 0], [0.7, -0.7, 0], [0, 0.7, 0]]],
            dtype=torch.float32,
            device="cuda:0",
        ).reshape([1, 3, 3])
        engine = winder.WinderEngine(triangles)

    print("Evaluating winding numbers on GPU...")
    raw_field = torch.from_dlpack(engine.brute_force(queries))

    # Reshape the flat output back into a structured 3D volume [X, Y, Z]
    winding_number_field = raw_field.view(
        args.resolution, args.resolution, args.resolution
    )

    # Generate the sweeping video slices along the X-axis
    video_path = f"{args.prefix}_slices.mp4"
    print(f"Colorizing and saving video slices to {video_path}...")
    if args.type == "dipole":
        winding_number_color = apply_colormap_gpu(winding_number_field.permute(0,2,1), args.resolution, vmin=-2.0, vmax=2.0)
    else:
        winding_number_color = apply_colormap_gpu(winding_number_field.permute(0,2,1), args.resolution, vmin=-0.5, vmax=0.5)

    write_video(video_path, winding_number_color, is_lossless=False)

    # Export the raw 3D volume as OpenVDB for Blender
    vdb_path = f"{args.prefix}.vdb"
    export_to_vdb(winding_number_field, vdb_path)


if __name__ == "__main__":
    main()
