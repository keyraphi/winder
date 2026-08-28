import numpy as np
import bpy
import winder
from .dlpack_bridge import CudaBuffer


def extract_geometry_data(obj: bpy.types.Object, mode: str):
    """Extracts GPU array buffers based on selected geometry mode."""
    mesh = obj.to_mesh()
    mesh.calc_loop_triangles()

    verts = np.empty((len(mesh.vertices), 3), dtype=np.float32)
    mesh.vertices.foreach_get("co", verts.ravel())

    tri_indices = np.empty((len(mesh.loop_triangles), 3), dtype=np.uint32)
    mesh.loop_triangles.foreach_get("vertices", tri_indices.ravel())

    obj.to_mesh_clear()

    if mode == "PointNormal":
        # Compute triangle centers and scaled normals (area * normal)
        v0 = verts[tri_indices[:, 0]]
        v1 = verts[tri_indices[:, 1]]
        v2 = verts[tri_indices[:, 2]]

        centers = (v0 + v1 + v2) / 3.0
        cross_prod = np.cross(v1 - v0, v2 - v0)
        scaled_normals = 0.5 * cross_prod  # Length equals triangle Voronoi area

        buf_pts = CudaBuffer(centers.shape, dtype=np.float32)
        buf_normals = CudaBuffer(scaled_normals.shape, dtype=np.float32)
        buf_pts.copy_from_numpy_async(centers)
        buf_normals.copy_from_numpy_async(scaled_normals)
        buf_pts.synchronize()
        return {
            "mode": "PointNormal",
            "points": buf_pts,
            "scaled_normals": buf_normals,
            "count": len(centers),
        }

    elif mode == "Triangle":
        v0 = verts[tri_indices[:, 0]]
        v1 = verts[tri_indices[:, 1]]
        v2 = verts[tri_indices[:, 2]]
        triangles = np.stack([v0, v1, v2], axis=1).astype(np.float32)

        buf_tris = CudaBuffer(triangles.shape, dtype=np.float32)
        buf_tris.copy_from_numpy_async(triangles)
        buf_tris.synchronize()
        return {"mode": "Triangle", "triangles": buf_tris, "count": len(triangles)}

    elif mode == "Mesh":
        buf_verts = CudaBuffer(verts.shape, dtype=np.float32)
        buf_tris = CudaBuffer(tri_indices.shape, dtype=np.uint32)
        buf_verts.copy_from_numpy_async(verts)
        buf_tris.copy_from_numpy_async(tri_indices)
        buf_verts.synchronize()
        return {
            "mode": "Mesh",
            "vertices": buf_verts,
            "triangle_indices": buf_tris,
            "count": len(tri_indices),
        }


def compute_winding_field(
    geom_data: dict,
    queries_buf: CudaBuffer,
    out_buf: CudaBuffer,
    epsilon: float = -1.0,
    beta: float = -1.0,
    stream: int = 0,
):
    """Evaluates winding numbers using Engine acceleration (N > 1000) or Brute-Force."""
    mode = geom_data["mode"]
    use_engine = geom_data["count"] > 1000

    if mode == "PointNormal":
        if use_engine:
            engine = winder.WindingNumberEngine(
                geom_data["points"], geom_data["scaled_normals"], stream
            )
            engine.compute(
                queries_buf, out_buf, beta=beta, epsilon=epsilon, stream=stream
            )
        else:
            print("DEBUG WINDER", dir(winder))
            help(winder)
            winder.brute_force_winding_numbers(
                geom_data["points"],
                geom_data["scaled_normals"],
                queries_buf,
                out_buf,
                epsilon=epsilon,
                stream=stream,
            )

    elif mode == "Triangle":
        if use_engine:
            engine = winder.WindingNumberEngine(geom_data["triangles"], stream)
            engine.compute(queries_buf, out_buf, beta=beta, stream=stream)
        else:
            winder.brute_force_winding_numbers(
                geom_data["triangles"], queries_buf, out_buf, stream=stream
            )

    elif mode == "Mesh":
        if use_engine:
            engine = winder.WindingNumberEngine(
                geom_data["vertices"], geom_data["triangle_indices"], stream
            )
            engine.compute(queries_buf, out_buf, beta=beta, stream=stream)
        else:
            winder.brute_force_winding_numbers(
                geom_data["vertices"],
                geom_data["triangle_indices"],
                queries_buf,
                out_buf,
                stream=stream,
            )


def compute_geometry_gradients(
    geom_data: dict,
    queries_buf: CudaBuffer,
    dL_dw_buf: CudaBuffer,
    out_grad_buf: CudaBuffer,
    epsilon: float = -1.0,
    beta: float = -1.0,
    stream: int = 0,
):
    """Computes field gradients w.r.t geometry primitives."""
    mode = geom_data["mode"]
    use_engine = geom_data["count"] > 1000

    if mode == "PointNormal":
        if use_engine:
            engine = winder.GradientEngine(queries_buf, dL_dw_buf, stream)
            engine.compute(
                geom_data["points"],
                geom_data["scaled_normals"],
                out_grad_buf,
                beta=beta,
                epsilon=epsilon,
                stream=stream,
            )
        else:
            winder.brute_force_gradients(
                dL_dw_buf,
                geom_data["points"],
                geom_data["scaled_normals"],
                queries_buf,
                out_grad_buf,
                epsilon=epsilon,
                stream=stream,
            )

    elif mode == "Triangle":
        if use_engine:
            engine = winder.GradientEngine(queries_buf, dL_dw_buf, stream)
            engine.compute(
                geom_data["triangles"], out_grad_buf, beta=beta, stream=stream
            )
        else:
            winder.brute_force_gradients(
                dL_dw_buf,
                geom_data["triangles"],
                queries_buf,
                out_grad_buf,
                stream=stream,
            )

    elif mode == "Mesh":
        if use_engine:
            engine = winder.GradientEngine(queries_buf, dL_dw_buf, stream)
            engine.compute(
                geom_data["vertices"],
                geom_data["triangle_indices"],
                out_grad_buf,
                beta=beta,
                stream=stream,
            )
        else:
            winder.brute_force_gradients(
                dL_dw_buf,
                geom_data["vertices"],
                geom_data["triangle_indices"],
                queries_buf,
                out_grad_buf,
                stream=stream,
            )
