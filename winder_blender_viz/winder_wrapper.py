import bmesh
import bpy
import numpy as np
import winder
from .dlpack_bridge import CudaBuffer, CudaStream


# Default regularization strength (fraction of scene diagonal).
# Matches the library default of 1/250. Pass 0.0 (or any negative value) to
# disable regularization and use the sharp kernel.
DEFAULT_EPSILON = 0.004


def _get_stream_handle(stream) -> int:
    if stream is None:
        return 0
    if isinstance(stream, CudaStream):
        return stream.handle
    return int(stream)


def extract_bmesh_positions(obj):
    """Safely extracts vertex positions from Edit Mode BMesh or Object Mesh."""
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
    """Extracts evaluated vertex/face geometry (post-modifiers & Geometry Nodes)
    and uploads directly to CudaBuffer objects.
    """

    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    mesh_eval = obj_eval.to_mesh()

    try:
        num_verts = len(mesh_eval.vertices)
        raw_verts = np.empty((num_verts, 3), dtype=np.float32)
        mesh_eval.vertices.foreach_get("co", raw_verts.ravel())

        world_mat = np.array(obj_eval.matrix_world, dtype=np.float32).T
        homo_verts = np.hstack([raw_verts, np.ones((num_verts, 1), dtype=np.float32)])
        world_verts = (homo_verts @ world_mat)[:, :3].astype(np.float32)

        mesh_eval.calc_loop_triangles()
        num_tris = len(mesh_eval.loop_triangles)
        tri_indices = np.empty((num_tris, 3), dtype=np.uint32)
        mesh_eval.loop_triangles.foreach_get("vertices", tri_indices.ravel())

    finally:
        obj_eval.to_mesh_clear()

    host_refs = [raw_verts, homo_verts, world_verts, tri_indices]

    if mode == "Mesh":
        v_buf = CudaBuffer(world_verts.shape, dtype=np.float32)
        i_buf = CudaBuffer(tri_indices.shape, dtype=np.uint32)

        v_buf.copy_from_numpy_async(world_verts, stream=stream)
        i_buf.copy_from_numpy_async(tri_indices, stream=stream)

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
        v0, v1, v2 = (
            world_tris[:, 0, :],
            world_tris[:, 1, :],
            world_tris[:, 2, :],
        )

        centroids = ((v0 + v1 + v2) / 3.0).astype(np.float32)
        cross_prod = np.cross(v1 - v0, v2 - v0)
        scaled_normals = (0.5 * cross_prod).astype(np.float32)

        host_refs.extend([world_tris, centroids, scaled_normals])

        pts_buf = CudaBuffer(centroids.shape, dtype=np.float32)
        norm_buf = CudaBuffer(scaled_normals.shape, dtype=np.float32)

        pts_buf.copy_from_numpy_async(centroids, stream=stream)
        norm_buf.copy_from_numpy_async(scaled_normals, stream=stream)

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

        return {
            "mode": "Triangle",
            "triangles": tri_buf,
            "count": len(world_tris),
            "_host_refs": host_refs,
        }

    else:
        raise ValueError(f"Unknown geometry mode: {mode}")


def extract_combined_geometry_data(objs, mode="Mesh", stream=None):
    """Combines geometry from multiple objects into unified GPU CudaBuffers."""
    if len(objs) == 1:
        return extract_geometry_data(objs[0], mode=mode, stream=stream)

    host_refs = []

    if mode == "Mesh":
        all_verts = []
        all_indices = []
        vert_offset = 0

        for obj in objs:
            world_mat = np.array(obj.matrix_world, dtype=np.float32).T
            raw_verts = extract_bmesh_positions(obj)
            homo = np.hstack(
                [raw_verts, np.ones((len(raw_verts), 1), dtype=np.float32)]
            )
            w_verts = (homo @ world_mat)[:, :3].astype(np.float32)

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
                                [
                                    f.verts[0].index,
                                    f.verts[i].index,
                                    f.verts[i + 1].index,
                                ]
                            )
                t_indices = np.array(tris, dtype=np.uint32)
            else:
                obj.data.calc_loop_triangles()
                t_indices = np.empty((len(obj.data.loop_triangles), 3), dtype=np.uint32)
                obj.data.loop_triangles.foreach_get("vertices", t_indices.ravel())

            all_verts.append(w_verts)
            all_indices.append(t_indices + vert_offset)
            vert_offset += len(w_verts)

        cat_verts = np.vstack(all_verts).astype(np.float32)
        cat_indices = np.vstack(all_indices).astype(np.uint32)
        host_refs.extend([all_verts, all_indices, cat_verts, cat_indices])

        v_buf = CudaBuffer(cat_verts.shape, dtype=np.float32)
        i_buf = CudaBuffer(cat_indices.shape, dtype=np.uint32)

        v_buf.copy_from_numpy_async(cat_verts, stream=stream)
        i_buf.copy_from_numpy_async(cat_indices, stream=stream)

        return {
            "mode": "Mesh",
            "vertices": v_buf,
            "triangle_indices": i_buf,
            "indices": i_buf,
            "num_verts": len(cat_verts),
            "num_tris": len(cat_indices),
            "count": len(cat_indices),
            "_host_refs": host_refs,
        }

    elif mode == "PointNormal":
        all_pts = []
        all_norms = []

        for obj in objs:
            world_mat = np.array(obj.matrix_world, dtype=np.float32).T
            raw_verts = extract_bmesh_positions(obj)
            homo = np.hstack(
                [raw_verts, np.ones((len(raw_verts), 1), dtype=np.float32)]
            )
            w_verts = (homo @ world_mat)[:, :3].astype(np.float32)

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
                                [
                                    f.verts[0].index,
                                    f.verts[i].index,
                                    f.verts[i + 1].index,
                                ]
                            )
                t_indices = np.array(tris, dtype=np.uint32)
            else:
                obj.data.calc_loop_triangles()
                t_indices = np.empty((len(obj.data.loop_triangles), 3), dtype=np.uint32)
                obj.data.loop_triangles.foreach_get("vertices", t_indices.ravel())

            world_tris = w_verts[t_indices]
            v0, v1, v2 = world_tris[:, 0, :], world_tris[:, 1, :], world_tris[:, 2, :]
            centroids = ((v0 + v1 + v2) / 3.0).astype(np.float32)
            scaled_normals = (0.5 * np.cross(v1 - v0, v2 - v0)).astype(np.float32)

            all_pts.append(centroids)
            all_norms.append(scaled_normals)

        cat_pts = np.vstack(all_pts).astype(np.float32)
        cat_norms = np.vstack(all_norms).astype(np.float32)
        host_refs.extend([cat_pts, cat_norms])

        pts_buf = CudaBuffer(cat_pts.shape, dtype=np.float32)
        norm_buf = CudaBuffer(cat_norms.shape, dtype=np.float32)

        pts_buf.copy_from_numpy_async(cat_pts, stream=stream)
        norm_buf.copy_from_numpy_async(cat_norms, stream=stream)

        return {
            "mode": "PointNormal",
            "points": pts_buf,
            "normals": norm_buf,
            "scaled_normals": norm_buf,
            "count": len(cat_pts),
            "_host_refs": host_refs,
        }

    elif mode == "Triangle":
        all_tris = []

        for obj in objs:
            world_mat = np.array(obj.matrix_world, dtype=np.float32).T
            raw_verts = extract_bmesh_positions(obj)
            homo = np.hstack(
                [raw_verts, np.ones((len(raw_verts), 1), dtype=np.float32)]
            )
            w_verts = (homo @ world_mat)[:, :3].astype(np.float32)

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
                                [
                                    f.verts[0].index,
                                    f.verts[i].index,
                                    f.verts[i + 1].index,
                                ]
                            )
                t_indices = np.array(tris, dtype=np.uint32)
            else:
                obj.data.calc_loop_triangles()
                t_indices = np.empty((len(obj.data.loop_triangles), 3), dtype=np.uint32)
                obj.data.loop_triangles.foreach_get("vertices", t_indices.ravel())

            all_tris.append(w_verts[t_indices].astype(np.float32))

        cat_tris = np.vstack(all_tris).astype(np.float32)
        host_refs.append(cat_tris)

        tri_buf = CudaBuffer(cat_tris.shape, dtype=np.float32)
        tri_buf.copy_from_numpy_async(cat_tris, stream=stream)

        return {
            "mode": "Triangle",
            "triangles": tri_buf,
            "count": len(cat_tris),
            "_host_refs": host_refs,
        }


def compute_winding_field(
    geom_data: dict,
    queries_buf: CudaBuffer,
    out_buf: CudaBuffer,
    epsilon: float = DEFAULT_EPSILON,
    beta: float = -1.0,
    stream=None,
):
    """Forward winding number evaluation.

    `epsilon` is a fraction of the scene diagonal. 0.0 (or any negative value)
    disables regularization and returns the sharp field. The library default
    is DEFAULT_EPSILON = 1/250 = 0.004.
    """
    mode = geom_data["mode"]
    use_engine = geom_data["count"] > 1000
    stream_handle = _get_stream_handle(stream)
    host_refs = [geom_data["_host_refs"]]

    if mode == "PointNormal":
        if use_engine:
            engine = winder.WindingNumberEngine(
                geom_data["points"], geom_data["scaled_normals"], stream_handle
            )
            engine.compute(
                queries_buf, out_buf, float(beta), float(epsilon), stream_handle
            )
            host_refs.append(engine)
        else:
            winder.brute_force_winding_numbers_point_normal(
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
                geom_data["triangles"], stream_handle
            )
            engine.compute(
                queries_buf, out_buf, float(beta), float(epsilon), stream_handle
            )
            host_refs.append(engine)
        else:
            winder.brute_force_winding_numbers_triangle_soup(
                geom_data["triangles"],
                queries_buf,
                out_buf,
                float(epsilon),
                stream_handle,
            )

    elif mode == "Mesh":
        if use_engine:
            engine = winder.WindingNumberEngine(
                geom_data["vertices"],
                geom_data["triangle_indices"],
                stream_handle,
            )
            engine.compute(
                queries_buf, out_buf, float(beta), float(epsilon), stream_handle
            )
            host_refs.append(engine)
        else:
            winder.brute_force_winding_numbers_mesh(
                geom_data["vertices"],
                geom_data["triangle_indices"],
                queries_buf,
                out_buf,
                float(epsilon),
                stream_handle,
            )
    return host_refs


def compute_geometry_gradients(
    geom_data: dict,
    queries_buf: CudaBuffer,
    dL_dw_buf: CudaBuffer,
    out_grad_buf: CudaBuffer,
    epsilon: float = DEFAULT_EPSILON,
    beta: float = -1.0,
    stream=None,
):
    """Backward gradient of the winding number field w.r.t. geometry.

    `epsilon` is a fraction of the scene diagonal. 0.0 (or negative) disables
    regularization and returns the sharp-kernel gradient. Default 1/250.
    """
    mode = geom_data["mode"]
    use_engine = geom_data["count"] > 1000
    stream_handle = _get_stream_handle(stream)
    host_refs = [geom_data["_host_refs"]]

    if mode == "PointNormal":
        if use_engine:
            engine = winder.GradientEngine(queries_buf, dL_dw_buf, stream_handle)
            engine.compute_point_normal(
                geom_data["points"],
                geom_data["scaled_normals"],
                out_grad_buf,
                float(beta),
                float(epsilon),
                stream_handle,
            )
            host_refs.append(engine)
        else:
            winder.brute_force_gradients_point_normal(
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
            engine = winder.GradientEngine(queries_buf, dL_dw_buf, stream_handle)
            engine.compute_triangle_soup(
                geom_data["triangles"],
                out_grad_buf,
                float(beta),
                float(epsilon),
                stream_handle,
            )
            host_refs.append(engine)
        else:
            winder.brute_force_gradients_triangle_soup(
                dL_dw_buf,
                geom_data["triangles"],
                queries_buf,
                out_grad_buf,
                float(epsilon),
                stream_handle,
            )

    elif mode == "Mesh":
        if use_engine:
            engine = winder.GradientEngine(queries_buf, dL_dw_buf, stream_handle)
            engine.compute_mesh(
                geom_data["vertices"],
                geom_data["triangle_indices"],
                out_grad_buf,
                float(beta),
                float(epsilon),
                stream_handle,
            )
            host_refs.append(engine)
        else:
            winder.brute_force_gradients_mesh(
                dL_dw_buf,
                geom_data["vertices"],
                geom_data["triangle_indices"],
                queries_buf,
                out_grad_buf,
                float(epsilon),
                stream_handle,
            )
    return host_refs


def dump_forward_bvh(geom_data: dict, stream=None) -> str:
    """Instantiates WindingNumberEngine for geom_data and returns its BVH8 .dot string."""
    mode = geom_data["mode"]
    stream_handle = _get_stream_handle(stream)

    if mode == "PointNormal":
        engine = winder.WindingNumberEngine(
            geom_data["points"], geom_data["scaled_normals"], stream=stream_handle
        )
    elif mode == "Triangle":
        engine = winder.WindingNumberEngine(
            geom_data["triangles"], stream=stream_handle
        )
    elif mode == "Mesh":
        engine = winder.WindingNumberEngine(
            geom_data["vertices"],
            geom_data["triangle_indices"],
            stream=stream_handle,
        )
    else:
        raise ValueError(f"Unsupported geometry mode: {mode}")

    return engine.dump()


def dump_backward_bvh(
    queries_buf: CudaBuffer, dL_dw_buf: CudaBuffer, stream=None
) -> str:
    """Instantiates GradientEngine for query/loss buffers and returns its BVH8 .dot string."""
    stream_handle = _get_stream_handle(stream)
    engine = winder.GradientEngine(queries_buf, dL_dw_buf, stream_handle)
    return engine.dump()
