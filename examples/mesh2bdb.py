#!/usr/bin/env python3
import argparse
from time import time
import openvdb
import numpy as np
import torch
import igl
import winder


def positive_type(arg: str) -> int:
    x: int = int(arg)
    if x < 1:
        raise argparse.ArgumentTypeError("Minimum value is 1")
    return x


def export_to_vdb(
    field: torch.Tensor, filename: str, voxel_size: float, origin: np.ndarray
):
    winding_number_data = field.detach().cpu().numpy().astype(np.float64)

    wind_ct = openvdb.FloatGrid()
    wind_ct.copyFromArray(winding_number_data)
    wind_ct.name = "winding_field"

    # Linear transform matrix matching the sequential translation logic OpenVDB expects
    matrix = [
        [float(voxel_size), 0.0, 0.0, 0.0],
        [0.0, float(voxel_size), 0.0, 0.0],
        [0.0, 0.0, float(voxel_size), 0.0],
        [-1, -1, -1, 1],
    ]

    wind_ct.transform = openvdb.createLinearTransform(matrix=matrix)

    openvdb.write(filename, [wind_ct])
    print(f"Exported aligned world-space VDB: {filename}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_path", type=str, required=True, help="Obj file")
    parser.add_argument(
        "--resolution", type=positive_type, default=256, help="Grid resolution"
    )
    parser.add_argument("--prefix", type=str, required=True, help="Output file prefix.")
    args = parser.parse_args()

    # Read the raw source mesh coordinates
    vertices, _, _, indices, _, _ = igl.readOBJ(args.obj_path)
    print(f"Info: mesh has {len(vertices)} vertices and {len(indices)} triangles.")

    print("DEBUG:", vertices.min(axis=0), vertices.max(axis=0))
    v_min = np.min(vertices, axis=0)
    v_max = np.max(vertices, axis=0)

    # Calculate uniform normalization scaling factor matching a max-bound box dimension of 2.0
    center = (v_min + v_max) / 2.0
    scale = 2.0 / np.max(v_max - v_min)

    # Scale down raw vertices for processing
    normalized_vertices = (vertices - center) * scale

    print("DEBUG:", normalized_vertices.min(axis=0), vertices.max(axis=0))
    vertices_np = normalized_vertices.astype(np.float32)
    indices_np = np.asarray(indices).astype(np.uint32)
    print("DEBUG:", vertices_np.min(axis=0), vertices_np.max(axis=0))

    verts = torch.from_numpy(vertices_np).to("cuda:0")
    idxs = torch.from_numpy(indices_np).to("cuda:0")


    print(
        f"Generating {args.resolution}^3 grid queries from [-1, -1, -1] to [1, 1, 1]..."
    )
    lin = torch.linspace(-1.0, 1.0, args.resolution, device="cuda:0")
    grid_x, grid_y, grid_z = torch.meshgrid(lin, lin, lin, indexing="ij")
    queries = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)

    start = time()
    engine = winder.WinderEngine(verts, idxs)
    print("Evaluating winding numbers on GPU...")
    raw_field = torch.from_dlpack(engine.compute(queries, stream=torch.cuda.current_stream().cuda_stream))
    torch.cuda.synchronize()
    end = time()
    print(f"INFO: Total compute time: {end - start} sec.")
                                  

    # Reshape the flat output back into structured 3D volume [X, Y, Z]
    winding_number_field = raw_field.view(
        args.resolution, args.resolution, args.resolution
    )

    # Scale down the world-space offset coordinates to match the small Blender footprint
    scaled_v_min = (v_min - center) * scale
    voxel_size = 2.0 / args.resolution

    # Export the VDB using matched parameters
    vdb_path = f"{args.prefix}.vdb"
    export_to_vdb(winding_number_field, vdb_path, voxel_size, scaled_v_min)


if __name__ == "__main__":
    main()
