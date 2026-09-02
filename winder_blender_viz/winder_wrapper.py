import bmesh
import numpy as np
import winder
from .dlpack_bridge import CudaBuffer, CudaStream


def _get_stream_handle(stream) -> int:
    print("DEBUG: _get_stream_handle")
    if stream is None:
        return 0
    if isinstance(stream, CudaStream):
        return stream.handle
    return int(stream)


def extract_bmesh_positions(obj):
    """Safely extracts vertex positions from Edit Mode BMesh or Object Mesh."""
    print("DEBUG: extract_bmesh_positions")
    if obj.mode == "EDIT":
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        verts = np.empty((len(bm.verts), 3), dtype=np.float32)
        for i, v in enumerate(bm.verts):
            verts[i] = v.co
        return verts
    else:
        num_verts = len(obj.data.vertices)
        verts = np.empty((num_verts, 3), dtype=np.float32)
        obj.data.vertices.foreach_get("co", verts.ravel())
        return verts


def extract_geometry_data(obj, mode="Mesh", stream=None):
    """Extracts vertex/face geometry and uploads directly to CudaBuffer objects.

    Retains host NumPy references in host_refs to prevent Python GC from unmapping
    host memory during asynchronous CUDA stream copies.
    """
    print("DEBUG: extract_geometry_data")
    world_mat = np.array(obj.matrix_world, dtype=np.float32).T
    raw_verts = extract_bmesh_positions(obj)
    homo_verts = np.hstack([raw_verts, np.ones((len(raw_verts), 1), dtype=np.float32)])
    world_verts = (homo_verts @ world_mat)[:, :3].astype(np.float32)

    host_refs = [raw_verts, homo_verts, world_verts]

    if obj.mode == "EDIT":
        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        tris = []
        for f in bm.faces:
            if len(f.verts) == 3:
                tris.append([v.index for v in f.verts])
            elif len(f.verts) > 3:
                for i in range(1, len(f.verts) - 1):
                    tris.append(
                        [f.verts[0].index, f.verts[i].index, f.verts[i + 1].index]
                    )
        tri_indices = np.array(tris, dtype=np.uint32)
    else:
        obj.data.calc_loop_triangles()
        num_tris = len(obj.data.loop_triangles)
        tri_indices = np.empty((num_tris, 3), dtype=np.uint32)
        obj.data.loop_triangles.foreach_get("vertices", tri_indices.ravel())

    host_refs.append(tri_indices)

    if mode == "Mesh":
        v_buf = CudaBuffer(world_verts.shape, dtype=np.float32)
        i_buf = CudaBuffer(tri_indices.shape, dtype=np.uint32)

        v_buf.copy_from_numpy_async(world_verts, stream=stream)
        i_buf.copy_from_numpy_async(tri_indices, stream=stream)

        print("DEBUG: extract_geometry_data MESH: v_buf.shape", v_buf.shape)
        print("DEBUG: extract_geometry_data MESH: i_buf.shape", i_buf.shape)

        return {
            "mode": "Mesh",
            "vertices": v_buf,
            "triangle_indices": i_buf,
            "indices": i_buf,
            "num_verts": len(world_verts),
            "num_tris": len(tri_indices),
            "count": len(tri_indices),
            "_host_refs": host_refs,
        }

    elif mode == "PointNormal":
        world_tris = world_verts[tri_indices]
        v0, v1, v2 = world_tris[:, 0, :], world_tris[:, 1, :], world_tris[:, 2, :]

        centroids = ((v0 + v1 + v2) / 3.0).astype(np.float32)
        cross_prod = np.cross(v1 - v0, v2 - v0)
        scaled_normals = (0.5 * cross_prod).astype(np.float32)

        host_refs.extend([world_tris, centroids, scaled_normals])

        pts_buf = CudaBuffer(centroids.shape, dtype=np.float32)
        norm_buf = CudaBuffer(scaled_normals.shape, dtype=np.float32)

        pts_buf.copy_from_numpy_async(centroids, stream=stream)
        norm_buf.copy_from_numpy_async(scaled_normals, stream=stream)

        print("DEBUG: extract_geometry_data PointNormal: pts_buf.shape", pts_buf.shape)
        print(
            "DEBUG: extract_geometry_data PointNormal: norm_buf.shape", norm_buf.shape
        )
        return {
            "mode": "PointNormal",
            "points": pts_buf,
            "normals": norm_buf,
            "scaled_normals": norm_buf,
            "count": len(tri_indices),
            "_host_refs": host_refs,
        }

    elif mode == "Triangle":
        world_tris = world_verts[tri_indices].astype(np.float32)
        host_refs.append(world_tris)

        tri_buf = CudaBuffer(world_tris.shape, dtype=np.float32)
        tri_buf.copy_from_numpy_async(world_tris, stream=stream)

        print("DEBUG: extract_geometry_data Triangle: tri_buf.shape", tri_buf.shape)
        return {
            "mode": "Triangle",
            "triangles": tri_buf,
            "count": len(world_tris),
            "_host_refs": host_refs,
        }

    else:
        raise ValueError(f"Unknown geometry mode: {mode}")


def compute_winding_field(
    geom_data: dict,
    queries_buf: CudaBuffer,
    out_buf: CudaBuffer,
    epsilon: float = -1.0,
    beta: float = -1.0,
    stream=None,
):
    print("DEBUG: compute_winding_field")
    mode = geom_data["mode"]
    use_engine = geom_data["count"] > 1000
    stream_handle = _get_stream_handle(stream)
    host_refs = [geom_data["_host_refs"]]

    if mode == "PointNormal":
        if use_engine:
            print(
                "DEBUG: compute_winding_field -> PointNormal creating engine",
                stream_handle,
            )
            engine = winder.WindingNumberEngine(
                geom_data["points"], geom_data["scaled_normals"], stream=stream_handle
            )
            print(
                "DEBUG: compute_winding_field -> PointNormalCreating compute()",
                stream_handle,
            )
            engine.compute(
                queries_buf, out_buf, float(beta), float(epsilon), stream_handle
            )
            host_refs.append(engine)
        else:
            print(
                "DEBUG: compute_winding_field -> brute_force()",
                "points:",
                geom_data["points"].shape,
                geom_data["points"].dtype,
                geom_data["points"].device_id,
                "scaled_normals:",
                geom_data["scaled_normals"].shape,
                geom_data["scaled_normals"].dtype,
                geom_data["scaled_normals"].device_id,
                "queries_buf:",
                queries_buf.shape,
                queries_buf.dtype,
                queries_buf.device_id,
                "out_buf:",
                out_buf.shape,
                out_buf.dtype,
                out_buf.device_id,
                epsilon,
                stream_handle,
            )
            winder.brute_force_winding_numbers(
                geom_data["points"],
                geom_data["scaled_normals"],
                queries_buf,
                out_buf,
                float(epsilon),
                stream_handle,
            )

    elif mode == "Triangle":
        if use_engine:
            engine = winder.WindingNumberEngine(
                geom_data["triangles"], stream=stream_handle
            )
            engine.compute(queries_buf, out_buf, float(beta), stream_handle)
            host_refs.append(engine)
        else:
            winder.brute_force_winding_numbers(
                geom_data["triangles"], queries_buf, out_buf, stream_handle
            )

    elif mode == "Mesh":
        if use_engine:
            engine = winder.WindingNumberEngine(
                geom_data["vertices"],
                geom_data["triangle_indices"],
                stream=stream_handle,
            )
            engine.compute(queries_buf, out_buf, float(beta), stream_handle)
            host_refs.append(engine)
        else:
            winder.brute_force_winding_numbers(
                geom_data["vertices"],
                geom_data["triangle_indices"],
                queries_buf,
                out_buf,
                stream_handle,
            )
    return host_refs


def compute_geometry_gradients(
    geom_data: dict,
    queries_buf: CudaBuffer,
    dL_dw_buf: CudaBuffer,
    out_grad_buf: CudaBuffer,
    epsilon: float = -1.0,
    beta: float = -1.0,
    stream=None,
):
    print("DEBUG: compute_geometry_gradients mode:", geom_data["mode"])
    mode = geom_data["mode"]
    use_engine = geom_data["count"] > 1000
    stream_handle = _get_stream_handle(stream)
    host_refs = [geom_data["_host_refs"]]

    if mode == "PointNormal":
        if use_engine:
            print("DEBUG: running GradientEngine for PointNormals ")
            engine = winder.GradientEngine(queries_buf, dL_dw_buf, stream_handle)
            engine.compute(
                geom_data["points"],
                geom_data["scaled_normals"],
                out_grad_buf,
                float(beta),
                float(epsilon),
                stream_handle,
            )
            host_refs.append(engine)
        else:
            print("DEBUG: running Brute Force Gradients for PointNormals ")
            winder.brute_force_gradients(
                dL_dw_buf,
                geom_data["points"],
                geom_data["scaled_normals"],
                queries_buf,
                out_grad_buf,
                float(epsilon),
                stream_handle,
            )

    elif mode == "Triangle":
        if use_engine:
            print("DEBUG: running GradientEngine for Triangles ")
            engine = winder.GradientEngine(queries_buf, dL_dw_buf, stream_handle)
            engine.compute(
                geom_data["triangles"], out_grad_buf, float(beta), stream_handle
            )
            host_refs.append(engine)
        else:
            print("DEBUG: running Brute Force Gradients for Triangles ")
            winder.brute_force_gradients(
                dL_dw_buf,
                geom_data["triangles"],
                queries_buf,
                out_grad_buf,
                stream_handle,
            )

    elif mode == "Mesh":
        print("DEBUG GRADIENT INPUTS")
        print("DEBUG dL_dw_buf:", dL_dw_buf.shape, dL_dw_buf.dtype)
        print(
            "DEBUG: vertices:", geom_data["vertices"].shape, geom_data["vertices"].dtype
        )
        print(
            "DEBUG: triangle_indices:",
            geom_data["triangle_indices"].shape,
            geom_data["triangle_indices"].dtype,
        )
        print("DEBUG: queries_buf:", queries_buf.shape, queries_buf.dtype)
        print("DEBUG: out_grad_buf:", out_grad_buf.shape, out_grad_buf.dtype)
        if use_engine:
            print("DEBUG: running GradientEngine for Mesh ")
            engine = winder.GradientEngine(queries_buf, dL_dw_buf, stream_handle)
            engine.compute(
                geom_data["vertices"],
                geom_data["triangle_indices"],
                out_grad_buf,
                float(beta),
                stream_handle,
            )
            host_refs.append(engine)
        else:
            print("DEBUG: running Brute Force Gradients for Mesh ")
            winder.brute_force_gradients(
                dL_dw_buf,
                geom_data["vertices"],
                geom_data["triangle_indices"],
                queries_buf,
                out_grad_buf,
                stream_handle,
            )
    return host_refs
