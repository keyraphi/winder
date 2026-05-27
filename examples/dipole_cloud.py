#!/usr/bin/env python3
import numpy as np
import time
import torch
import winder
import argparse
import imageio.v3 as iio
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


device = torch.device("cuda:0")


def positive_type(arg: str) -> int:
    x: int = int(arg)
    if x < 1:
        raise argparse.ArgumentTypeError("Minimum value is 1")
    return x


def create_query_grid(resolution: int, extend: float = 5) -> torch.Tensor:
    xs = torch.linspace(-extend, extend, resolution, device=device)
    xss, yss, zss = torch.meshgrid([xs, xs, xs], indexing="ij")
    queries = torch.stack([xss, yss, zss], dim=-1)
    queries = queries.reshape([-1, 3])
    return queries


def visualize_result(
    winding_number_grid: torch.Tensor,
    winding_number_grid_brute_force: torch.Tensor,
    grid_resolution: int,
    video_path: str | None,
):
    if video_path is None:
        return

    gt_grid = winding_number_grid_brute_force.view(
        grid_resolution, grid_resolution, grid_resolution
    )
    approx_grid = winding_number_grid.view(
        grid_resolution, grid_resolution, grid_resolution
    )
    diff_grid = torch.abs(gt_grid - approx_grid)

    gt_grid = torch.clamp(gt_grid, -2, 2)
    approx_grid = torch.clamp(approx_grid, -2, 2)
    diff_grid = torch.clamp(diff_grid, 0, 1)#diff_threshold_quantile.item())
    # normalize difference to [0,1]

    gt_cpu = gt_grid.cpu().numpy()
    approx_cpu = approx_grid.cpu().numpy()
    diff_cpu = diff_grid.cpu().numpy()

    norm = mcolors.TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)

    cmap = plt.get_cmap("vanimo")
    cmap_diff = plt.get_cmap("magma")

    color_gt = cmap(norm(gt_cpu))[..., :3]
    color_approx = cmap(norm(approx_cpu))[..., :3]
    color_diff = cmap_diff(diff_cpu)[..., :3]

    # Scale to uint8
    color_gt = (color_gt * 255).astype(np.uint8)
    color_approx = (color_approx * 255).astype(np.uint8)
    color_diff = (color_diff * 255).astype(np.uint8)

    print("DEBUG: color_diff", color_diff.min(), color_diff.max())

    divider = np.zeros((grid_resolution, grid_resolution, 4, 3), dtype=np.uint8)

    full_video_stack = np.concatenate([color_gt, divider, color_approx, divider, color_diff], axis=2)

    print(f"Writing batch of shape {full_video_stack.shape} to {video_path}...")
    iio.imwrite(
        video_path,
        full_video_stack,
        extension=".mp4",
        fps=25,
        codec="libx264",
        is_batch=True,
    )


def main():
    parser = argparse.ArgumentParser()
    _ = parser.add_argument(
        "--geometry_count",
        type=positive_type,
        default=32,
        help="How many dipoles to generate",
    )
    _ = parser.add_argument(
        "--query_grid_resolution",
        type=positive_type,
        default=512,
        help="Resolution per dimension of query grid",
    )
    _ = parser.add_argument(
        "--grid_extend",
        type=float,
        default=5.0,
        help="Query grid extend. Sphere radius is 1.",
    )
    _ = parser.add_argument(
        "--beta",
        type=float,
        default=-1,
        help="Approximation factor. Larger value means more approximation. Default is around 2",
    )
    _ = parser.add_argument(
        "--figure_path", type=str, help="Optional path to safe figures."
    )
    args = parser.parse_args()

    torch.random.manual_seed(1)

    points = torch.randn([args.geometry_count, 3], device=device)
    normals = 15 * torch.randn([args.geometry_count, 3], device=device)
    # points = torch.tensor([[0, 0, 0]], dtype=torch.float32, device=device)
    # normals = torch.tensor([[0, 1, 0]], dtype=torch.float32, device=device)
    engine = winder.WinderEngine(points, normals)

    queries = create_query_grid(args.query_grid_resolution, args.grid_extend)

    t0 = time.time()
    N_RUNS = 1000
    for _ in tqdm(range(1000), desc=f"Running {N_RUNS} runs"):
        winding_number_grid = engine.compute(queries, beta=args.beta)
    t1 = time.time()
    winding_number_grid_brute_force = engine.brute_force(queries)
    t2 = time.time()
    print(f"Approx took {(t1-t0)/N_RUNS} sec. Bruteforce took {t2-t1} sec.")

    visualize_result(
        torch.from_dlpack(winding_number_grid),
        torch.from_dlpack(winding_number_grid_brute_force),
        args.query_grid_resolution,
        args.figure_path,
    )

    print("Dumping engine")
    # print(engine.dump())


if __name__ == "__main__":
    main()
