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
            engine = winder.WindingNumberEngine(geom_data["triangles"], stream_handle)
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

    epsilon_grad = CudaBuffer([1], np.float32, queries_buf.device_id)

    if mode == "PointNormal":
        if use_engine:
            engine = winder.GradientEngine(queries_buf, dL_dw_buf, stream_handle)
            engine.compute_point_normal(
                geom_data["points"],
                geom_data["scaled_normals"],
                out_grad_buf,
                epsilon_grad,
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
                epsilon_grad,
                float(epsilon),
                stream_handle,
            )

    elif mode == "Triangle":
        if use_engine:
            engine = winder.GradientEngine(queries_buf, dL_dw_buf, stream_handle)
            engine.compute_triangle_soup(
                geom_data["triangles"],
                out_grad_buf,
                epsilon_grad,
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
                epsilon_grad,
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
                epsilon_grad,
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
                epsilon_grad,
                float(epsilon),
                stream_handle,
            )
    return host_refs


def compute_query_gradients(
    geom_data: dict,
    queries_buf: CudaBuffer,
    dL_dw_buf: CudaBuffer,
    out_grad_buf: CudaBuffer,
    epsilon: float = DEFAULT_EPSILON,
    beta: float = -1.0,
    stream=None,
):
    """Gradient of a loss w.r.t. the query positions, dL/dx_j.

    Pass `dL_dw_buf = 1` to obtain the raw field Jacobian ∇_x W(x).
    `out_grad_buf` must be a (M, 3) float32 CudaBuffer.
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
            engine.compute_query_gradients(
                dL_dw_buf,
                queries_buf,
                out_grad_buf,
                float(beta),
                float(epsilon),
                stream_handle,
            )
            host_refs.append(engine)
        else:
            winder.brute_force_query_gradients_point_normal(
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
            engine = winder.WindingNumberEngine(geom_data["triangles"], stream_handle)
            engine.compute_query_gradients(
                dL_dw_buf,
                queries_buf,
                out_grad_buf,
                float(beta),
                float(epsilon),
                stream_handle,
            )
            host_refs.append(engine)
        else:
            winder.brute_force_query_gradients_triangle_soup(
                dL_dw_buf,
                geom_data["triangles"],
                queries_buf,
                out_grad_buf,
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
            engine.compute_query_gradients(
                dL_dw_buf,
                queries_buf,
                out_grad_buf,
                float(beta),
                float(epsilon),
                stream_handle,
            )
            host_refs.append(engine)
        else:
            winder.brute_force_query_gradients_mesh(
                dL_dw_buf,
                geom_data["vertices"],
                geom_data["triangle_indices"],
                queries_buf,
                out_grad_buf,
                float(epsilon),
                stream_handle,
            )
    return host_refs

# ---------------------------------------------------------------------------
# Streamline / field-line tracing through ∇_x W(x)
# ---------------------------------------------------------------------------
def _seed_on_point_normal(points, n_seeds, rng, jitter_scale):
    """Sample seeds near the point cloud positions.

    Sampling is WITH replacement so `n_seeds` is honoured regardless of the
    cloud size. Gaussian jitter of scale `jitter_scale` (world units) breaks
    the exact coincidence of duplicated or near-duplicated seeds.
    """
    if len(points) == 0 or n_seeds <= 0:
        return np.zeros((0, 3), np.float32)

    idx = rng.choice(len(points), size=n_seeds, replace=True)
    seeds = points[idx].astype(np.float32)

    if jitter_scale > 0.0:
        jitter = rng.normal(0.0, jitter_scale, size=(n_seeds, 3)).astype(np.float32)
        seeds = seeds + jitter
    return seeds


def _seed_on_triangles(tris, n_seeds, rng, jitter_scale):
    """Area-weighted barycentric sampling on a triangle set, with optional jitter."""
    if len(tris) == 0 or n_seeds <= 0:
        return np.zeros((0, 3), np.float32)

    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    areas = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    total = areas.sum()

    if total <= 0.0:
        seeds = tris[np.arange(n_seeds) % len(tris), 0].astype(np.float32)
    else:
        probs = areas / total
        tri_ids = rng.choice(len(tris), size=n_seeds, p=probs)
        u = rng.random(n_seeds)
        v = rng.random(n_seeds)
        flip = (u + v) > 1.0
        u[flip] = 1.0 - u[flip]
        v[flip] = 1.0 - v[flip]
        w = 1.0 - u - v
        tri = tris[tri_ids]
        seeds = (
            tri[:, 0] * w[:, None]
            + tri[:, 1] * u[:, None]
            + tri[:, 2] * v[:, None]
        ).astype(np.float32)

    if jitter_scale > 0.0:
        jitter = rng.normal(0.0, jitter_scale, size=(n_seeds, 3)).astype(np.float32)
        seeds = seeds + jitter
    return seeds


def trace_query_streamlines(
    geom_data, seeds, step_size, max_steps, max_length,
    bounds_min, bounds_max, min_grad_mag,
    epsilon, beta, bidirectional, stream,
):
    """Integrate streamlines of ∇_x W(x) with batched RK2.

    Returns:
        line_points:   (n_lines, max_steps + 1, 3) float32, NaN-padded
        line_mags:     (n_lines, max_steps + 1) float32, gradient magnitudes
        line_counts:   (n_lines,) int32, number of valid points per line
    """
    n_seeds = len(seeds)
    n_lines = 2 * n_seeds if bidirectional else n_seeds

    if bidirectional:
        init_pos = np.concatenate([seeds, seeds], axis=0).astype(np.float32)
        dir_signs = np.concatenate(
            [np.ones(n_seeds), -np.ones(n_seeds)]
        ).astype(np.float32)
    else:
        init_pos = seeds.astype(np.float32).copy()
        dir_signs = np.ones(n_seeds, dtype=np.float32)

    line_points = np.full((n_lines, max_steps + 1, 3), np.nan, dtype=np.float32)
    line_points[:, 0] = init_pos
    line_mags = np.zeros((n_lines, max_steps + 1), dtype=np.float32)
    line_counts = np.ones(n_lines, dtype=np.int32)

    positions = init_pos.copy()
    line_lengths = np.zeros(n_lines, dtype=np.float32)
    active = np.ones(n_lines, dtype=bool)

    # ---- Pre-allocate max-size GPU buffers (reused via slice_rows) ----
    q_buf = CudaBuffer((n_lines, 3), np.float32)
    ones_buf = CudaBuffer((n_lines,), np.float32)
    ones_buf.copy_from_numpy_async(np.ones(n_lines, np.float32), stream=stream)
    g_buf = CudaBuffer((n_lines, 3), np.float32)

    # ---- Build the engine ONCE (not per step) ----
    mode = geom_data["mode"]
    use_engine = geom_data["count"] > 1000
    stream_handle = stream.handle if hasattr(stream, "handle") else int(stream)

    if use_engine:
        if mode == "PointNormal":
            engine = winder.WindingNumberEngine(
                geom_data["points"], geom_data["scaled_normals"], stream_handle
            )
        elif mode == "Triangle":
            engine = winder.WindingNumberEngine(geom_data["triangles"], stream_handle)
        else:
            engine = winder.WindingNumberEngine(
                geom_data["vertices"], geom_data["triangle_indices"], stream_handle
            )
        stream.keep_alive(engine)

    def _gradient_batch(pos_arr):
        """Return (N, 3) float32 ∇W on host for a batch of positions."""
        n = len(pos_arr)
        q_view = q_buf.slice_rows(n)
        q_view.copy_from_numpy_async(pos_arr, stream=stream)
        g_view = g_buf.slice_rows(n)
        ones_view = ones_buf.slice_rows(n)

        if use_engine:
            engine.compute_query_gradients(
                ones_view, q_view, g_view,
                float(beta), float(epsilon), stream_handle,
            )
        else:
            if mode == "PointNormal":
                winder.brute_force_query_gradients_point_normal(
                    ones_view, geom_data["points"], geom_data["scaled_normals"],
                    q_view, g_view, float(epsilon), stream_handle,
                )
            elif mode == "Triangle":
                winder.brute_force_query_gradients_triangle_soup(
                    ones_view, geom_data["triangles"],
                    q_view, g_view, float(epsilon), stream_handle,
                )
            else:
                winder.brute_force_query_gradients_mesh(
                    ones_view, geom_data["vertices"], geom_data["triangle_indices"],
                    q_view, g_view, float(epsilon), stream_handle,
                )

        g_host = np.empty((n, 3), np.float32)
        g_view.copy_to_numpy_async(g_host, stream=stream)
        stream.synchronize()
        return g_host

    # ---- Main integration loop ----
    for step in range(max_steps):
        active_idx = np.where(active)[0]
        n_a = len(active_idx)
        if n_a == 0:
            break

        pos_a = positions[active_idx].copy()
        sign = dir_signs[active_idx, None]

        # RK2 stage 1 (forward Euler)
        g1 = _gradient_batch(pos_a)
        n1 = np.linalg.norm(g1, axis=1, keepdims=True)
        k1 = g1 / np.maximum(n1, 1e-12) * sign

        # RK2 stage 2 (midpoint)
        pos_mid = pos_a + 0.5 * step_size * k1
        g2 = _gradient_batch(pos_mid)
        n2 = np.linalg.norm(g2, axis=1, keepdims=True)
        k2 = g2 / np.maximum(n2, 1e-12) * sign

        # Final update
        new_pos = pos_a + step_size * k2
        new_len = line_lengths[active_idx] + step_size

        # Termination checks
        out_of_bounds = (
            (new_pos[:, 0] < bounds_min[0]) | (new_pos[:, 0] > bounds_max[0]) |
            (new_pos[:, 1] < bounds_min[1]) | (new_pos[:, 1] > bounds_max[1]) |
            (new_pos[:, 2] < bounds_min[2]) | (new_pos[:, 2] > bounds_max[2])
        )
        too_long = new_len > max_length
        weak = n2.squeeze(1) < min_grad_mag
        still = ~(out_of_bounds | weak | too_long)

        # Commit
        positions[active_idx] = new_pos
        line_lengths[active_idx] = new_len

        for j, idx in enumerate(active_idx):
            if still[j]:
                k = line_counts[idx]
                line_points[idx, k] = new_pos[j]
                line_mags[idx, k] = float(n2[j, 0])
                line_counts[idx] = k + 1

        active[active_idx[~still]] = False

    return line_points, line_mags, line_counts

def update_mesh_from_lines(mesh, line_points, line_mags, line_counts):
    """Rewrite an existing mesh datablock with new streamline geometry.

    Does not touch the object, modifier stack, or materials. Safe to call
    from property update callbacks and depsgraph timers.
    """
    mesh.clear_geometry()

    verts_list = []
    edges_list = []
    attr_mag = []
    attr_line = []
    attr_idx = []

    offset = 0
    for line_id in range(line_points.shape[0]):
        n = int(line_counts[line_id])
        if n < 2:
            continue
        pts = line_points[line_id, :n]
        mags = line_mags[line_id, :n]

        verts_list.append(pts.astype(np.float32))
        attr_mag.append(mags.astype(np.float32))
        attr_line.append(np.full(n, line_id, dtype=np.int32))
        attr_idx.append(np.arange(n, dtype=np.int32))

        e = np.empty((n - 1, 2), dtype=np.int32)
        e[:, 0] = offset + np.arange(n - 1)
        e[:, 1] = offset + np.arange(n - 1) + 1
        edges_list.append(e)

        offset += n

    if not verts_list:
        # All lines were too short: leave the mesh empty but valid.
        for name in ("grad_mag", "line_id", "point_idx"):
            if name in mesh.attributes:
                mesh.attributes.remove(mesh.attributes[name])
        mesh.update()
        return

    verts = np.concatenate(verts_list, axis=0)
    edges = np.concatenate(edges_list, axis=0)
    attr_mag = np.concatenate(attr_mag)
    attr_line = np.concatenate(attr_line)
    attr_idx = np.concatenate(attr_idx)

    mesh.vertices.add(count=len(verts))
    mesh.vertices.foreach_set("co", verts.ravel())
    mesh.edges.add(count=len(edges))
    mesh.edges.foreach_set("vertices", edges.ravel())

    def _set_point_attr(name, dtype, data):
        if name in mesh.attributes:
            attr = mesh.attributes[name]
            if len(attr.data) != len(verts):
                mesh.attributes.remove(attr)
                attr = mesh.attributes.new(name=name, type=dtype, domain="POINT")
        else:
            attr = mesh.attributes.new(name=name, type=dtype, domain="POINT")
        attr.data.foreach_set("value", data)

    _set_point_attr("grad_mag", "FLOAT", attr_mag)
    _set_point_attr("line_id", "INT", attr_line)
    _set_point_attr("point_idx", "INT", attr_idx)

    mesh.update()


def build_streamline_mesh(line_points, line_mags, line_counts, name):
    """Create a fresh mesh datablock and populate it."""
    mesh = bpy.data.meshes.new(name)
    update_mesh_from_lines(mesh, line_points, line_mags, line_counts)
    return mesh


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
