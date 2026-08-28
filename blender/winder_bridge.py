import numpy as np
import torch
import winder


def extract_blender_mesh_data(obj, mode="POINT_NORMAL"):
    """Extracts raw mesh geometry into GPU tensors based on active mode."""
    mesh = obj.to_mesh()
    mesh.calc_loop_triangles()

    vertices = np.empty((len(mesh.vertices), 3), dtype=np.float32)
    mesh.vertices.foreach_get("co", vertices.ravel())

    # Apply Object World Matrix
    matrix = np.array(obj.matrix_world, dtype=np.float32)
    R = matrix[:3, :3]
    T = matrix[:3, 3]
    vertices_world = vertices @ R.T + T

    tri_indices = np.empty((len(mesh.loop_triangles), 3), dtype=np.uint32)
    mesh.loop_triangles.foreach_get("vertices", tri_indices.ravel())

    obj.to_mesh_clear()

    v_tensor = torch.tensor(vertices_world, device="cuda:0", dtype=torch.float32)
    idx_tensor = torch.tensor(tri_indices, device="cuda:0", dtype=torch.uint32)

    if mode == "MESH":
        return {"vertices": v_tensor, "triangle_indices": idx_tensor}

    v0 = v_tensor[idx_tensor[:, 0]]
    v1 = v_tensor[idx_tensor[:, 1]]
    v2 = v_tensor[idx_tensor[:, 2]]

    if mode == "TRIANGLE":
        triangles = torch.stack([v0, v1, v2], dim=1)
        return {"triangles": triangles}

    elif mode == "POINT_NORMAL":
        points = (v0 + v1 + v2) / 3.0
        raw_normals = torch.linalg.cross(v1 - v0, v2 - v0)
        scaled_normals = 0.5 * raw_normals
        return {"points": points, "scaled_normals": scaled_normals}


def compute_winding_fields(geo_data, queries, mode="POINT_NORMAL", epsilon=-1.0):
    """Computes winding numbers using Engine for N > 1000 or Brute Force for N <= 1000."""
    M = queries.shape[0]
    out_winding = torch.empty((M,), dtype=torch.float32, device="cuda:0")

    if mode == "POINT_NORMAL":
        N = geo_data["points"].shape[0]
        if N > 1000:
            engine = winder.WindingNumberEngine(
                geo_data["points"], geo_data["scaled_normals"], stream=0
            )
            engine.compute(queries, out_winding, epsilon=epsilon, stream=0)
        else:
            winder.brute_force_winding_numbers(
                geo_data["points"],
                geo_data["scaled_normals"],
                queries,
                out_winding,
                epsilon=epsilon,
                stream=0,
            )

    elif mode == "TRIANGLE":
        N = geo_data["triangles"].shape[0]
        if N > 1000:
            engine = winder.WindingNumberEngine(geo_data["triangles"], stream=0)
            engine.compute(queries, out_winding, stream=0)
        else:
            winder.brute_force_winding_numbers(
                geo_data["triangles"], queries, out_winding, stream=0
            )

    elif mode == "MESH":
        N = geo_data["triangle_indices"].shape[0]
        if N > 1000:
            engine = winder.WindingNumberEngine(
                geo_data["vertices"], geo_data["triangle_indices"], stream=0
            )
            engine.compute(queries, out_winding, stream=0)
        else:
            winder.brute_force_winding_numbers(
                geo_data["vertices"],
                geo_data["triangle_indices"],
                queries,
                out_winding,
                stream=0,
            )

    return out_winding


def compute_geometry_gradients(
    grad_output, geo_data, queries, mode="POINT_NORMAL", epsilon=-1.0
):
    """Computes reverse-mode gradients on input geometry primitives."""
    if mode == "POINT_NORMAL":
        N = geo_data["points"].shape[0]
        out_grads = torch.empty((N, 2, 3), dtype=torch.float32, device="cuda:0")
        if N > 1000:
            engine = winder.GradientEngine(queries, grad_output, stream=0)
            engine.compute(
                geo_data["points"],
                geo_data["scaled_normals"],
                out_grads,
                epsilon=epsilon,
                stream=0,
            )
        else:
            winder.brute_force_gradients(
                grad_output,
                geo_data["points"],
                geo_data["scaled_normals"],
                queries,
                out_grads,
                epsilon=epsilon,
                stream=0,
            )
        return out_grads

    elif mode == "TRIANGLE":
        N = geo_data["triangles"].shape[0]
        out_grads = torch.empty((N, 3, 3), dtype=torch.float32, device="cuda:0")
        if N > 1000:
            engine = winder.GradientEngine(queries, grad_output, stream=0)
            engine.compute(geo_data["triangles"], out_grads, stream=0)
        else:
            winder.brute_force_gradients(
                grad_output,
                geo_data["triangles"],
                queries,
                out_grads,
                stream=0,
            )
        return out_grads

    elif mode == "MESH":
        K = geo_data["vertices"].shape[0]
        out_grads = torch.empty((K, 3), dtype=torch.float32, device="cuda:0")
        if K > 1000:
            engine = winder.GradientEngine(queries, grad_output, stream=0)
            engine.compute(
                geo_data["vertices"],
                geo_data["triangle_indices"],
                out_grads,
                stream=0,
            )
        else:
            winder.brute_force_gradients(
                grad_output,
                geo_data["vertices"],
                geo_data["triangle_indices"],
                queries,
                epsilon=epsilon,
                out_gradients=out_grads,
                stream=0,
            )
        return out_grads
