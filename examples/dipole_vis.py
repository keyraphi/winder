#!/usr/bin/env python3
import gc
from tqdm.auto import tqdm
import argparse
import imageio.v3 as iio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import numpy.typing as NPT
import torch
import winder
import csv
import os


def positive_type(arg: str) -> int:
    x: int = int(arg)
    if x < 1:
        raise argparse.ArgumentTypeError("Minimum value is 1")
    return x

def apply_colormap_gpu(
    winding_numbers, resolution: int, cmap_name: str = "vanimo"
) -> NPT.ArrayLike:
    """Colorizes winding number frames entirely on the GPU using a Look-Up

    Table.
    """
    device = (
        winding_numbers[0].device
        if isinstance(winding_numbers, list) and torch.is_tensor(winding_numbers[0])
        else "cuda"
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
    vmin, vcenter, vmax = -2.0, 0.0, 2.0
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
        file.init_video_stream("libx264rgb", fps=fps, pixel_format="yuv420p")

        if is_lossless:
            # lossless, best compression
            file._video_stream.options = {"crf": "0", "preset": "slow"}
        else:
            file._video_stream.options = {"crf": "18", "preset": "slow"}

        for frame in tqdm(frames_numpy_array, desc="writing video"):
            file.write_frame(frame)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resolution", type=positive_type, default=512, help="Grid resolution"
    )
    parser.add_argument(
        "--prefix", type=str, required=True, help="Filename prefix string."
    )
    parser.add_argument("--type", choices=["dipole", "triangle"])
    args = parser.parse_args()


    # Define underlying reference tracking grids


if __name__ == "__main__":
    main()
