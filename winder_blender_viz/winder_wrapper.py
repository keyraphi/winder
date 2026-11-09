import bmesh
import bpy
import numpy as np
import winder
from .dlpack_bridge import CudaBuffer, CudaStream


DEFAULT_EPSILON = 0.004

# ---------------------------------------------------------------------------
# High-contrast palette for payload color patterns
# ---------------------------------------------------------------------------
# Indexed by (i%2)*4 + (j%2)*2 + (k%2) for the 3D checkerboard. All entries
# are high-saturation, so triangles stay visually distinct (no gray averaging).
CHECKERBOARD_PALETTE = np.array([
    [1.00, 0.10, 0.10],  # 000: red
    [0.10, 0.90, 0.10],  # 001: green
    [0.10, 0.30, 1.00],  # 010: blue
    [1.00, 1.00, 0.10],  # 011: yellow
    [1.00, 0.10, 1.00],  # 100: magenta
    [0.10, 1.00, 1.00],  # 101: cyan
    [1.00, 0.50, 0.10],  # 110: orange
    [0.60, 0.10, 0.90],  # 111: purple
], dtype=np.float32)

# Cells per axis for the checkerboard. 12 is a good default: each cell holds
# ~1000 triangles on a benchy-sized mesh, so the pattern reads as big visible
# blocks rather than noise.
CHECKERBOARD_CELLS_DEFAULT = 12


def _get_stream_handle(stream) -> int:
    if stream is None:
        return 0
    if isinstance(stream, CudaStream):
        return stream.handle
    return int(stream)


def _resolve_use_engine(geom_data: dict, use_fast_engine) -> bool:
    if use_fast_engine is None:
        return geom_data["count"] > 1000
    return bool(use_fast_engine)


# ---------------------------------------------------------------------------
# Geometry extraction (unchanged)
# ---------------------------------------------------------------------------
def extract_bmesh_positions(obj):
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
            "mode": "Mesh", "vertices": v_buf, "triangle_indices": i_buf,
            "indices": i_buf, "num_verts": len(world_verts),
            "num_tris": len(tri_indices), "count": len(tri_indices),
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
        return {
            "mode": "PointNormal", "points": pts_buf, "normals": norm_buf,
            "scaled_normals": norm_buf, "count": len(tri_indices),
            "_host_refs": host_refs,
        }
    elif mode == "Triangle":
        world_tris = world_verts[tri_indices].astype(np.float32)
        host_refs.append(world_tris)
        tri_buf = CudaBuffer(world_tris.shape, dtype=np.float32)
        tri_buf.copy_from_numpy_async(world_tris, stream=stream)
        return {
            "mode": "Triangle", "triangles": tri_buf,
            "count": len(world_tris), "_host_refs": host_refs,
        }
    else:
        raise ValueError(f"Unknown geometry mode: {mode}")


def extract_combined_geometry_data(objs, mode="Mesh", stream=None):
    if len(objs) == 1:
        return extract_geometry_data(objs[0], mode=mode, stream=stream)

    host_refs = []
    if mode == "Mesh":
        all_verts, all_indices, vert_offset = [], [], 0
        for obj in objs:
            world_mat = np.array(obj.matrix_world, dtype=np.float32).T
            raw_verts = extract_bmesh_positions(obj)
            homo = np.hstack([raw_verts, np.ones((len(raw_verts), 1), dtype=np.float32)])
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
                            tris.append([f.verts[0].index, f.verts[i].index, f.verts[i + 1].index])
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
            "mode": "Mesh", "vertices": v_buf, "triangle_indices": i_buf,
            "indices": i_buf, "num_verts": len(cat_verts),
            "num_tris": len(cat_indices), "count": len(cat_indices),
            "_host_refs": host_refs,
        }
    elif mode == "PointNormal":
        all_pts, all_norms = [], []
        for obj in objs:
            world_mat = np.array(obj.matrix_world, dtype=np.float32).T
            raw_verts = extract_bmesh_positions(obj)
            homo = np.hstack([raw_verts, np.ones((len(raw_verts), 1), dtype=np.float32)])
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
                            tris.append([f.verts[0].index, f.verts[i].index, f.verts[i + 1].index])
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
            "mode": "PointNormal", "points": pts_buf, "normals": norm_buf,
            "scaled_normals": norm_buf, "count": len(cat_pts),
            "_host_refs": host_refs,
        }
    elif mode == "Triangle":
        all_tris = []
        for obj in objs:
            world_mat = np.array(obj.matrix_world, dtype=np.float32).T
            raw_verts = extract_bmesh_positions(obj)
            homo = np.hstack([raw_verts, np.ones((len(raw_verts), 1), dtype=np.float32)])
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
                            tris.append([f.verts[0].index, f.verts[i].index, f.verts[i + 1].index])
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
            "mode": "Triangle", "triangles": tri_buf,
            "count": len(cat_tris), "_host_refs": host_refs,
        }


# ---------------------------------------------------------------------------
# Winding / query-gradient / geometry-gradient (unchanged, just uses use_fast_engine)
# ---------------------------------------------------------------------------
def compute_winding_field(
    geom_data, queries_buf, out_buf,
    epsilon=DEFAULT_EPSILON, beta=-1.0, stream=None,
    use_fast_engine=None,
):
    mode = geom_data["mode"]
    use_engine = _resolve_use_engine(geom_data, use_fast_engine)
    stream_handle = _get_stream_handle(stream)
    host_refs = [geom_data["_host_refs"]]

    if mode == "PointNormal":
        if use_engine:
            engine = winder.WindingNumberEngine(
                geom_data["points"], geom_data["scaled_normals"], stream_handle
            )
            engine.compute(queries_buf, out_buf, float(beta), float(epsilon), stream_handle)
            host_refs.append(engine)
        else:
            winder.brute_force_winding_numbers_point_normal(
                geom_data["points"], geom_data["scaled_normals"],
                queries_buf, out_buf, float(epsilon), stream_handle,
            )
    elif mode == "Triangle":
        if use_engine:
            engine = winder.WindingNumberEngine(geom_data["triangles"], stream_handle)
            engine.compute(queries_buf, out_buf, float(beta), float(epsilon), stream_handle)
            host_refs.append(engine)
        else:
            winder.brute_force_winding_numbers_triangle_soup(
                geom_data["triangles"], queries_buf, out_buf,
                float(epsilon), stream_handle,
            )
    elif mode == "Mesh":
        if use_engine:
            engine = winder.WindingNumberEngine(
                geom_data["vertices"], geom_data["triangle_indices"], stream_handle
            )
            engine.compute(queries_buf, out_buf, float(beta), float(epsilon), stream_handle)
            host_refs.append(engine)
        else:
            winder.brute_force_winding_numbers_mesh(
                geom_data["vertices"], geom_data["triangle_indices"],
                queries_buf, out_buf, float(epsilon), stream_handle,
            )
    return host_refs


def compute_geometry_gradients(
    geom_data, queries_buf, dL_dw_buf, out_grad_buf,
    epsilon=DEFAULT_EPSILON, beta=-1.0, stream=None,
    use_fast_engine=None,
):
    mode = geom_data["mode"]
    use_engine = _resolve_use_engine(geom_data, use_fast_engine)
    stream_handle = _get_stream_handle(stream)
    host_refs = [geom_data["_host_refs"]]
    epsilon_grad = CudaBuffer([1], np.float32, queries_buf.device_id)

    if mode == "PointNormal":
        if use_engine:
            engine = winder.GradientEngine(queries_buf, dL_dw_buf, stream_handle)
            engine.compute_point_normal(
                geom_data["points"], geom_data["scaled_normals"],
                out_grad_buf, epsilon_grad, float(beta), float(epsilon), stream_handle,
            )
            host_refs.append(engine)
        else:
            winder.brute_force_gradients_point_normal(
                dL_dw_buf, geom_data["points"], geom_data["scaled_normals"],
                queries_buf, out_grad_buf, epsilon_grad, float(epsilon), stream_handle,
            )
    elif mode == "Triangle":
        if use_engine:
            engine = winder.GradientEngine(queries_buf, dL_dw_buf, stream_handle)
            engine.compute_triangle_soup(
                geom_data["triangles"], out_grad_buf, epsilon_grad,
                float(beta), float(epsilon), stream_handle,
            )
            host_refs.append(engine)
        else:
            winder.brute_force_gradients_triangle_soup(
                dL_dw_buf, geom_data["triangles"], queries_buf,
                out_grad_buf, epsilon_grad, float(epsilon), stream_handle,
            )
    elif mode == "Mesh":
        if use_engine:
            engine = winder.GradientEngine(queries_buf, dL_dw_buf, stream_handle)
            engine.compute_mesh(
                geom_data["vertices"], geom_data["triangle_indices"],
                out_grad_buf, epsilon_grad, float(beta), float(epsilon), stream_handle,
            )
            host_refs.append(engine)
        else:
            winder.brute_force_gradients_mesh(
                dL_dw_buf, geom_data["vertices"], geom_data["triangle_indices"],
                queries_buf, out_grad_buf, epsilon_grad, float(epsilon), stream_handle,
            )
    return host_refs


def compute_query_gradients(
    geom_data, queries_buf, dL_dw_buf, out_grad_buf,
    epsilon=DEFAULT_EPSILON, beta=-1.0, stream=None,
    use_fast_engine=None,
):
    mode = geom_data["mode"]
    use_engine = _resolve_use_engine(geom_data, use_fast_engine)
    stream_handle = _get_stream_handle(stream)
    host_refs = [geom_data["_host_refs"]]

    if mode == "PointNormal":
        if use_engine:
            engine = winder.WindingNumberEngine(
                geom_data["points"], geom_data["scaled_normals"], stream_handle
            )
            engine.compute_query_gradients(
                dL_dw_buf, queries_buf, out_grad_buf,
                float(beta), float(epsilon), stream_handle,
            )
            host_refs.append(engine)
        else:
            winder.brute_force_query_gradients_point_normal(
                dL_dw_buf, geom_data["points"], geom_data["scaled_normals"],
                queries_buf, out_grad_buf, float(epsilon), stream_handle,
            )
    elif mode == "Triangle":
        if use_engine:
            engine = winder.WindingNumberEngine(geom_data["triangles"], stream_handle)
            engine.compute_query_gradients(
                dL_dw_buf, queries_buf, out_grad_buf,
                float(beta), float(epsilon), stream_handle,
            )
            host_refs.append(engine)
        else:
            winder.brute_force_query_gradients_triangle_soup(
                dL_dw_buf, geom_data["triangles"], queries_buf,
                out_grad_buf, float(epsilon), stream_handle,
            )
    elif mode == "Mesh":
        if use_engine:
            engine = winder.WindingNumberEngine(
                geom_data["vertices"], geom_data["triangle_indices"], stream_handle
            )
            engine.compute_query_gradients(
                dL_dw_buf, queries_buf, out_grad_buf,
                float(beta), float(epsilon), stream_handle,
            )
            host_refs.append(engine)
        else:
            winder.brute_force_query_gradients_mesh(
                dL_dw_buf, geom_data["vertices"], geom_data["triangle_indices"],
                queries_buf, out_grad_buf, float(epsilon), stream_handle,
            )
    return host_refs


# ---------------------------------------------------------------------------
# NEW: vertex-color helpers
# ---------------------------------------------------------------------------
def _expand_poly_colors_to_corners(mesh, poly_colors):
    """Expand a (num_polys, 3) RGB array to per-corner RGBA in CORNER domain."""
    num_polys = len(mesh.polygons)
    poly_loop_total = np.empty(num_polys, dtype=np.int32)
    mesh.polygons.foreach_get("loop_total", poly_loop_total)
    loop_to_poly = np.repeat(np.arange(num_polys), poly_loop_total)

    num_loops = len(mesh.loops)
    corner_colors = np.zeros((num_loops, 4), dtype=np.float32)
    corner_colors[:, 3] = 1.0
    corner_colors[:, :3] = poly_colors[loop_to_poly]
    return corner_colors


def _compute_checkerboard_colors(mesh, num_cells_per_axis):
    num_polys = len(mesh.polygons)
    if num_polys == 0:
        return None

    # Blender provides per-polygon center directly - no need to iterate verts.
    centers = np.empty((num_polys, 3), dtype=np.float32)
    mesh.polygons.foreach_get("center", centers.ravel())

    bbox_min = centers.min(axis=0)
    bbox_max = centers.max(axis=0)
    extent = bbox_max - bbox_min
    safe_extent = np.maximum(extent, 1e-6)

    u = np.clip((centers - bbox_min) / safe_extent, 0.0, 1.0 - 1e-6)
    cell_idx = (u * float(num_cells_per_axis)).astype(np.int32)

    # 3D checkerboard: 8 colors from the 3 axis parities.
    parity = cell_idx % 2
    color_idx = parity[:, 0] * 4 + parity[:, 1] * 2 + parity[:, 2]
    poly_colors = CHECKERBOARD_PALETTE[color_idx]

    return _expand_poly_colors_to_corners(mesh, poly_colors)


def _compute_random_polygon_colors(mesh):
    num_polys = len(mesh.polygons)
    if num_polys == 0:
        return None

    rng = np.random.default_rng()
    palette_idx = rng.integers(0, len(CHECKERBOARD_PALETTE), size=num_polys)
    poly_colors = CHECKERBOARD_PALETTE[palette_idx]

    return _expand_poly_colors_to_corners(mesh, poly_colors)


def _compute_xyz_gradient_colors(mesh):
    num_polys = len(mesh.polygons)
    if num_polys == 0:
        return None

    centers = np.empty((num_polys, 3), dtype=np.float32)
    mesh.polygons.foreach_get("center", centers.ravel())

    bbox_min = centers.min(axis=0)
    bbox_max = centers.max(axis=0)
    extent = bbox_max - bbox_min
    safe_extent = np.maximum(extent, 1e-6)

    poly_colors = np.clip((centers - bbox_min) / safe_extent, 0.0, 1.0)

    return _expand_poly_colors_to_corners(mesh, poly_colors)


def ensure_vertex_colors(obj, pattern="checkerboard", num_cells=None):
    """Ensure the mesh has an active color attribute.

    If none exists, creates a per-corner FLOAT_COLOR attribute with a
    high-contrast *per-polygon* pattern (not per-vertex - per-vertex random
    averages to gray per triangle). Returns True if a new attribute was
    created.

    Parameters
    ----------
    obj : bpy.types.Object
    pattern : str
        "checkerboard"    - 3D spatial checkerboard, 8 saturated colors
        "random_polygon"  - random saturated color per polygon
        "xyz_gradient"    - smooth RGB gradient along world XYZ
    num_cells : int, optional
        Cells per axis for checkerboard mode (default: 12).
    """
    mesh = obj.data
    if mesh.color_attributes:
        if mesh.color_attributes.active_color is None:
            mesh.color_attributes.active_color = mesh.color_attributes[0]
        return False

    mode_was_edit = (obj.mode == "EDIT")
    if mode_was_edit:
        bpy.ops.object.mode_set(mode="OBJECT")

    try:
        if len(mesh.loops) == 0:
            raise ValueError(
                f"Mesh '{obj.name}' has no faces to color. "
                "Add faces or use a triangle mesh."
            )

        if pattern == "checkerboard":
            cells = CHECKERBOARD_CELLS_DEFAULT if num_cells is None else int(num_cells)
            colors = _compute_checkerboard_colors(mesh, cells)
        elif pattern == "random_polygon":
            colors = _compute_random_polygon_colors(mesh)
        elif pattern == "xyz_gradient":
            colors = _compute_xyz_gradient_colors(mesh)
        else:
            raise ValueError(f"Unknown payload color pattern: {pattern}")

        if colors is None:
            raise ValueError(f"Mesh '{obj.name}' has no polygons to color.")

        attr = mesh.color_attributes.new(
            name="PayloadColor",
            type="FLOAT_COLOR",
            domain="CORNER",
        )
        attr.data.foreach_set("color", colors.ravel())
        mesh.color_attributes.active_color = attr
        mesh.update()
    finally:
        if mode_was_edit:
            bpy.ops.object.mode_set(mode="EDIT")

    return True


# ---------------------------------------------------------------------------
# Payload field extraction (unchanged)
# ---------------------------------------------------------------------------
def extract_payload_from_vertex_colors(obj, mode="Mesh"):
    """Read the active color attribute from `obj` and derive per-triangle RGB."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    mesh_eval = obj_eval.to_mesh()

    try:
        if not mesh_eval.color_attributes:
            raise ValueError(
                f"Mesh '{obj.name}' has no color attributes. "
                "Paint vertex colors or use 'Create Payload Field' "
                "to generate random ones."
            )
        color_attr = mesh_eval.color_attributes.active_color
        if color_attr is None:
            color_attr = mesh_eval.color_attributes[0]

        mesh_eval.calc_loop_triangles()
        num_tris = len(mesh_eval.loop_triangles)
        tri_indices = np.empty((num_tris, 3), dtype=np.uint32)
        mesh_eval.loop_triangles.foreach_get("vertices", tri_indices.ravel())

        if color_attr.domain == "POINT":
            colors = np.empty((len(mesh_eval.vertices), 4), np.float32)
            color_attr.data.foreach_get("color", colors.ravel())
            tri_colors = colors[tri_indices].mean(axis=1)
        elif color_attr.domain == "CORNER":
            colors = np.empty((len(mesh_eval.loops), 4), np.float32)
            color_attr.data.foreach_get("color", colors.ravel())
            tri_loops = np.empty((num_tris, 3), dtype=np.int32)
            mesh_eval.loop_triangles.foreach_get("loops", tri_loops.ravel())
            tri_colors = colors[tri_loops].mean(axis=1)
        else:
            raise ValueError(f"Unsupported color attribute domain: {color_attr.domain}")
    finally:
        obj_eval.to_mesh_clear()

    return tri_colors[:, :3].astype(np.float32)


# ---------------------------------------------------------------------------
# Payload forward / geometry gradient / query gradient
# ---------------------------------------------------------------------------
def compute_payload_field(
    geom_data, payload_buf, queries_buf, out_buf,
    epsilon=DEFAULT_EPSILON, beta=-1.0, stream=None,
    use_fast_engine=None, use_kahan=True,
):
    mode = geom_data["mode"]
    use_engine = _resolve_use_engine(geom_data, use_fast_engine)
    stream_handle = _get_stream_handle(stream)
    host_refs = [geom_data["_host_refs"]]

    if mode == "PointNormal":
        if use_engine:
            engine = winder.PayloadField(
                geom_data["points"], geom_data["scaled_normals"],
                payload_buf, stream_handle,
            )
            engine.compute(queries_buf, out_buf, float(beta), float(epsilon),
                           stream_handle, bool(use_kahan))
            host_refs.append(engine)
        else:
            winder.brute_force_winding_payloads_point_normal(
                geom_data["points"], geom_data["scaled_normals"], payload_buf,
                queries_buf, out_buf, float(epsilon), stream_handle,
                bool(use_kahan),
            )
    elif mode == "Triangle":
        if use_engine:
            engine = winder.PayloadField(
                geom_data["triangles"], payload_buf, stream_handle,
            )
            engine.compute(queries_buf, out_buf, float(beta), float(epsilon),
                           stream_handle, bool(use_kahan))
            host_refs.append(engine)
        else:
            winder.brute_force_winding_payloads_triangle_soup(
                geom_data["triangles"], payload_buf, queries_buf, out_buf,
                float(epsilon), stream_handle, bool(use_kahan),
            )
    elif mode == "Mesh":
        if use_engine:
            engine = winder.PayloadField(
                geom_data["vertices"], geom_data["triangle_indices"],
                payload_buf, stream_handle,
            )
            engine.compute(queries_buf, out_buf, float(beta), float(epsilon),
                           stream_handle, bool(use_kahan))
            host_refs.append(engine)
        else:
            winder.brute_force_winding_payloads_mesh(
                geom_data["vertices"], geom_data["triangle_indices"],
                payload_buf, queries_buf, out_buf,
                float(epsilon), stream_handle, bool(use_kahan),
            )
    return host_refs


def compute_payload_geometry_gradients(
    geom_data, payload_buf, queries_buf, dL_dW_buf, out_grad_buf,
    epsilon=DEFAULT_EPSILON, beta=-1.0, stream=None,
    use_fast_engine=None,
):
    """Payload field loss gradient w.r.t. geometry.

    NOTE: only the brute-force path is exposed in the current bindings
    (`brute_force_gradients_with_payload_*`). The BVH-accelerated
    geometry-gradient kernel for payloads is not yet wrapped.
    """
    mode = geom_data["mode"]
    use_engine = _resolve_use_engine(geom_data, use_fast_engine)
    stream_handle = _get_stream_handle(stream)
    host_refs = [geom_data["_host_refs"]]
    epsilon_grad = CudaBuffer([1], np.float32, queries_buf.device_id)

    if use_engine:
        raise NotImplementedError(
            "Fast (BVH) payload geometry gradients are not yet exposed in "
            "the winder binding. Set 'Use Fast Engine' off to use the "
            "brute-force path."
        )

    if mode == "PointNormal":
        winder.brute_force_gradients_with_payload_point_normal(
            dL_dW_buf, geom_data["points"], geom_data["scaled_normals"],
            payload_buf, queries_buf, out_grad_buf, epsilon_grad,
            float(epsilon), stream_handle, False,
        )
    elif mode == "Triangle":
        winder.brute_force_gradients_with_payload_triangle_soup(
            dL_dW_buf, geom_data["triangles"], payload_buf, queries_buf,
            out_grad_buf, epsilon_grad, float(epsilon), stream_handle, False,
        )
    elif mode == "Mesh":
        winder.brute_force_gradients_with_payload_mesh(
            dL_dW_buf, geom_data["vertices"], geom_data["triangle_indices"],
            payload_buf, queries_buf, out_grad_buf, epsilon_grad,
            float(epsilon), stream_handle, False,
        )
    return host_refs


def compute_payload_query_gradients_rgb(
    geom_data, payload_buf, queries_buf, out_rgb_buf,
    epsilon=DEFAULT_EPSILON, beta=-1.0, stream=None,
    use_fast_engine=None, use_kahan=True,
):
    """Per-channel query-gradient magnitudes of the payload field.

    For each query m and channel c ∈ {0,1,2}:
        out_rgb_buf[m, c] = |∂ W_c(x_m) / ∂ x|
    """
    mode = geom_data["mode"]
    use_engine = _resolve_use_engine(geom_data, use_fast_engine)
    stream_handle = _get_stream_handle(stream)
    host_refs = [geom_data["_host_refs"]]

    num_q = queries_buf.shape[0]
    D = payload_buf.shape[1]
    if D < 3:
        raise ValueError(
            f"RGB visualization requires D >= 3, got D={D}"
        )

    grad_output_np = np.zeros((num_q, D), dtype=np.float32)
    per_channel_grad = np.empty((num_q, 3, 3), dtype=np.float32)

    go_buf = CudaBuffer((num_q, D), np.float32)
    tmp_out = CudaBuffer((num_q, 3), np.float32)
    host_refs.extend([go_buf, tmp_out])

    # Build the engine / fallback once
    engine = None
    if use_engine:
        if mode == "PointNormal":
            engine = winder.PayloadField(
                geom_data["points"], geom_data["scaled_normals"],
                payload_buf, stream_handle,
            )
        elif mode == "Triangle":
            engine = winder.PayloadField(
                geom_data["triangles"], payload_buf, stream_handle,
            )
        elif mode == "Mesh":
            engine = winder.PayloadField(
                geom_data["vertices"], geom_data["triangle_indices"],
                payload_buf, stream_handle,
            )
        else:
            raise ValueError(f"Unknown geometry mode: {mode}")
        host_refs.append(engine)

        if not hasattr(engine, "compute_query_gradients"):
            raise NotImplementedError(
                "PayloadField.compute_query_gradients is missing from the "
                "winder binding. Rebuild the extension, or disable "
                "'Use Fast Engine' to use the brute-force path."
            )

    for c in range(3):
        grad_output_np.fill(0.0)
        grad_output_np[:, c] = 1.0
        go_buf.copy_from_numpy_async(grad_output_np, stream=stream)

        if use_engine:
            engine.compute_query_gradients(
                go_buf, queries_buf, tmp_out,
                float(beta), float(epsilon), stream_handle, bool(use_kahan),
            )
        elif mode == "PointNormal":
            winder.brute_force_query_gradients_with_payload_point_normal(
                go_buf, geom_data["points"], geom_data["scaled_normals"],
                payload_buf, queries_buf, tmp_out,
                float(epsilon), stream_handle, bool(use_kahan),
            )
        elif mode == "Triangle":
            winder.brute_force_query_gradients_with_payload_triangle_soup(
                go_buf, geom_data["triangles"], payload_buf, queries_buf,
                tmp_out, float(epsilon), stream_handle, bool(use_kahan),
            )
        elif mode == "Mesh":
            winder.brute_force_query_gradients_with_payload_mesh(
                go_buf, geom_data["vertices"], geom_data["triangle_indices"],
                payload_buf, queries_buf, tmp_out,
                float(epsilon), stream_handle, bool(use_kahan),
            )

        tmp_host = np.empty((num_q, 3), dtype=np.float32)
        tmp_out.copy_to_numpy_async(tmp_host, stream=stream)
        stream.synchronize()
        per_channel_grad[:, c, :] = tmp_host

    mags = np.linalg.norm(per_channel_grad, axis=2)  # (M, 3)
    out_rgb_buf.copy_from_numpy_async(mags.astype(np.float32), stream=stream)

    return host_refs


# ---------------------------------------------------------------------------
# Field-line tracing
# ---------------------------------------------------------------------------
def _seed_on_point_normal(points, n_seeds, rng, jitter_scale):
    if len(points) == 0 or n_seeds <= 0:
        return np.zeros((0, 3), np.float32)
    idx = rng.choice(len(points), size=n_seeds, replace=True)
    seeds = points[idx].astype(np.float32)
    if jitter_scale > 0.0:
        jitter = rng.normal(0.0, jitter_scale, size=(n_seeds, 3)).astype(np.float32)
        seeds = seeds + jitter
    return seeds


def _seed_on_triangles(tris, n_seeds, rng, jitter_scale):
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
        seeds = (tri[:, 0] * w[:, None] + tri[:, 1] * u[:, None] + tri[:, 2] * v[:, None]).astype(np.float32)
    if jitter_scale > 0.0:
        jitter = rng.normal(0.0, jitter_scale, size=(n_seeds, 3)).astype(np.float32)
        seeds = seeds + jitter
    return seeds


def _trace_lines_impl(
    n_lines, init_pos, dir_signs,
    step_size, max_steps, max_length,
    bounds_min, bounds_max, min_grad_mag,
    gradient_batch_fn,
):
    """Shared RK2 integration loop for field-line tracing.

    `gradient_batch_fn` maps an (N, 3) position array to an (N, 3) gradient
    array. It is expected to be synchronous from Python's point of view
    (i.e. it must `stream.synchronize()` internally after the copy back).
    """
    line_points = np.full((n_lines, max_steps + 1, 3), np.nan, dtype=np.float32)
    line_points[:, 0] = init_pos
    line_mags = np.zeros((n_lines, max_steps + 1), dtype=np.float32)
    line_counts = np.ones(n_lines, dtype=np.int32)

    positions = init_pos.copy()
    line_lengths = np.zeros(n_lines, dtype=np.float32)
    active = np.ones(n_lines, dtype=bool)

    for step in range(max_steps):
        active_idx = np.where(active)[0]
        n_a = len(active_idx)
        if n_a == 0:
            break

        pos_a = positions[active_idx].copy()
        sign = dir_signs[active_idx, None]

        g1 = gradient_batch_fn(pos_a)
        n1 = np.linalg.norm(g1, axis=1, keepdims=True)
        k1 = g1 / np.maximum(n1, 1e-12) * sign

        pos_mid = pos_a + 0.5 * step_size * k1
        g2 = gradient_batch_fn(pos_mid)
        n2 = np.linalg.norm(g2, axis=1, keepdims=True)
        k2 = g2 / np.maximum(n2, 1e-12) * sign

        new_pos = pos_a + step_size * k2
        new_len = line_lengths[active_idx] + step_size

        out_of_bounds = (
            (new_pos[:, 0] < bounds_min[0]) | (new_pos[:, 0] > bounds_max[0])
            | (new_pos[:, 1] < bounds_min[1]) | (new_pos[:, 1] > bounds_max[1])
            | (new_pos[:, 2] < bounds_min[2]) | (new_pos[:, 2] > bounds_max[2])
        )
        too_long = new_len > max_length
        weak = n2.squeeze(1) < min_grad_mag
        still = ~(out_of_bounds | weak | too_long)

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


def _prepare_seeds(seeds, bidirectional):
    n_seeds = len(seeds)
    if bidirectional:
        init_pos = np.concatenate([seeds, seeds], axis=0).astype(np.float32)
        dir_signs = np.concatenate(
            [np.ones(n_seeds), -np.ones(n_seeds)]
        ).astype(np.float32)
        n_lines = 2 * n_seeds
    else:
        init_pos = seeds.astype(np.float32).copy()
        dir_signs = np.ones(n_seeds, dtype=np.float32)
        n_lines = n_seeds
    return n_lines, init_pos, dir_signs


def trace_query_streamlines(
    geom_data, seeds, step_size, max_steps, max_length,
    bounds_min, bounds_max, min_grad_mag,
    epsilon, beta, bidirectional, stream,
    use_fast_engine=None,
):
    """Winding-field streamlines of ∇W (unchanged behaviour)."""
    n_lines, init_pos, dir_signs = _prepare_seeds(seeds, bidirectional)

    q_buf = CudaBuffer((n_lines, 3), np.float32)
    ones_buf = CudaBuffer((n_lines,), np.float32)
    ones_buf.copy_from_numpy_async(np.ones(n_lines, np.float32), stream=stream)
    g_buf = CudaBuffer((n_lines, 3), np.float32)

    mode = geom_data["mode"]
    use_engine = _resolve_use_engine(geom_data, use_fast_engine)
    stream_handle = stream.handle if hasattr(stream, "handle") else int(stream)

    engine = None
    if use_engine:
        if mode == "PointNormal":
            engine = winder.WindingNumberEngine(
                geom_data["points"], geom_data["scaled_normals"], stream_handle,
            )
        elif mode == "Triangle":
            engine = winder.WindingNumberEngine(geom_data["triangles"], stream_handle)
        else:
            engine = winder.WindingNumberEngine(
                geom_data["vertices"], geom_data["triangle_indices"], stream_handle,
            )
        stream.keep_alive(engine)

    def _gradient_batch(pos_arr):
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
                    ones_view, geom_data["triangles"], q_view, g_view,
                    float(epsilon), stream_handle,
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

    return _trace_lines_impl(
        n_lines, init_pos, dir_signs,
        step_size, max_steps, max_length,
        bounds_min, bounds_max, min_grad_mag,
        _gradient_batch,
    )


def trace_payload_query_streamlines(
    geom_data, payload_buf, seeds, step_size, max_steps, max_length,
    bounds_min, bounds_max, min_grad_mag,
    epsilon, beta, bidirectional, stream,
    use_fast_engine=None, use_kahan=True,
):
    """Payload-field streamlines of the summed query-gradient Σ_c ∂W_c/∂x.

    For D=1, this reduces to the scalar winding field's ∇W. For D=3 with a
    mixed-color payload, lines follow the "net color increase" direction,
    which is a reasonable aggregate visualization.
    """
    n_lines, init_pos, dir_signs = _prepare_seeds(seeds, bidirectional)

    D = payload_buf.shape[1]

    q_buf = CudaBuffer((n_lines, 3), np.float32)
    go_buf = CudaBuffer((n_lines, D), np.float32)
    ones_full = np.ones((n_lines, D), dtype=np.float32)
    go_buf.copy_from_numpy_async(ones_full, stream=stream)
    g_buf = CudaBuffer((n_lines, 3), np.float32)

    mode = geom_data["mode"]
    use_engine = _resolve_use_engine(geom_data, use_fast_engine)
    stream_handle = stream.handle if hasattr(stream, "handle") else int(stream)

    engine = None
    if use_engine:
        if mode == "PointNormal":
            engine = winder.PayloadField(
                geom_data["points"], geom_data["scaled_normals"],
                payload_buf, stream_handle,
            )
        elif mode == "Triangle":
            engine = winder.PayloadField(
                geom_data["triangles"], payload_buf, stream_handle,
            )
        else:
            engine = winder.PayloadField(
                geom_data["vertices"], geom_data["triangle_indices"],
                payload_buf, stream_handle,
            )
        if not hasattr(engine, "compute_query_gradients"):
            raise NotImplementedError(
                "PayloadField.compute_query_gradients is missing from the "
                "winder binding. Rebuild the extension, or disable "
                "'Use Fast Engine' to use the brute-force path."
            )
        stream.keep_alive(engine)

    def _gradient_batch(pos_arr):
        n = len(pos_arr)
        q_view = q_buf.slice_rows(n)
        q_view.copy_from_numpy_async(pos_arr, stream=stream)
        g_view = g_buf.slice_rows(n)
        go_view = go_buf.slice_rows(n)

        if use_engine:
            engine.compute_query_gradients(
                go_view, q_view, g_view,
                float(beta), float(epsilon), stream_handle, bool(use_kahan),
            )
        elif mode == "PointNormal":
            winder.brute_force_query_gradients_with_payload_point_normal(
                go_view, geom_data["points"], geom_data["scaled_normals"],
                payload_buf, q_view, g_view,
                float(epsilon), stream_handle, bool(use_kahan),
            )
        elif mode == "Triangle":
            winder.brute_force_query_gradients_with_payload_triangle_soup(
                go_view, geom_data["triangles"], payload_buf, q_view, g_view,
                float(epsilon), stream_handle, bool(use_kahan),
            )
        elif mode == "Mesh":
            winder.brute_force_query_gradients_with_payload_mesh(
                go_view, geom_data["vertices"], geom_data["triangle_indices"],
                payload_buf, q_view, g_view,
                float(epsilon), stream_handle, bool(use_kahan),
            )

        g_host = np.empty((n, 3), np.float32)
        g_view.copy_to_numpy_async(g_host, stream=stream)
        stream.synchronize()
        return g_host

    return _trace_lines_impl(
        n_lines, init_pos, dir_signs,
        step_size, max_steps, max_length,
        bounds_min, bounds_max, min_grad_mag,
        _gradient_batch,
    )


# ---------------------------------------------------------------------------
# Mesh update helpers (unchanged)
# ---------------------------------------------------------------------------
def update_mesh_from_lines(mesh, line_points, line_mags, line_counts):
    mesh.clear_geometry()

    verts_list, edges_list = [], []
    attr_mag, attr_line, attr_idx = [], [], []

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
    mesh = bpy.data.meshes.new(name)
    update_mesh_from_lines(mesh, line_points, line_mags, line_counts)
    return mesh


def dump_forward_bvh(geom_data, stream=None):
    mode = geom_data["mode"]
    stream_handle = _get_stream_handle(stream)
    if mode == "PointNormal":
        engine = winder.WindingNumberEngine(
            geom_data["points"], geom_data["scaled_normals"], stream=stream_handle,
        )
    elif mode == "Triangle":
        engine = winder.WindingNumberEngine(geom_data["triangles"], stream=stream_handle)
    elif mode == "Mesh":
        engine = winder.WindingNumberEngine(
            geom_data["vertices"], geom_data["triangle_indices"], stream=stream_handle,
        )
    else:
        raise ValueError(f"Unsupported geometry mode: {mode}")
    return engine.dump()


def dump_backward_bvh(queries_buf, dL_dw_buf, stream=None):
    stream_handle = _get_stream_handle(stream)
    engine = winder.GradientEngine(queries_buf, dL_dw_buf, stream_handle)
    return engine.dump()
