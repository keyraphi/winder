import os
import uuid
import tempfile
import openvdb
import numpy as np
import mathutils
import bpy

from .bvh_parser import build_bvh_wireframe_mesh, parse_bvh_dot

from .dlpack_bridge import CudaBuffer, CudaStream
from .nodes_builder import (
    build_quiver_geometry_nodes,
    build_marching_cubes_contour_nodes,
    build_query_grad_geometry_nodes,
    build_streamline_geometry_nodes,
    get_or_create_quiver_material,
    get_or_create_query_grad_material,
    get_or_create_streamline_material,
    get_or_create_payload_isosurface_material,
    _build_iso_gn_tree,
    get_or_create_volume_material,
    get_or_create_payload_volume_material,
)
from .winder_wrapper import (
    extract_geometry_data,
    extract_combined_geometry_data,
    extract_payload_from_vertex_colors,
    ensure_vertex_colors,
    compute_winding_field,
    compute_geometry_gradients,
    compute_query_gradients,
    compute_payload_field,
    compute_payload_geometry_gradients,
    compute_payload_query_gradients_rgb,
    trace_query_streamlines,
    trace_payload_query_streamlines,
    update_mesh_from_lines,
    build_streamline_mesh,
    dump_forward_bvh,
    dump_backward_bvh,
    _seed_on_point_normal,
    _seed_on_triangles,
)


# ===========================================================================
# Scene / grid helpers
# ===========================================================================
def get_or_create_winding_objects(context, obj):
    vol_name = f"WindingVol_{obj.name}"
    quiver_name = f"WindingQuiver_{obj.name}"

    vol_obj = bpy.data.objects.get(vol_name)
    if not vol_obj:
        vol_data = bpy.data.volumes.new(f"WindingField_{obj.name}")
        vol_mat = get_or_create_volume_material()
        vol_data.materials.append(vol_mat)
        vol_obj = bpy.data.objects.new(vol_name, vol_data)
        context.collection.objects.link(vol_obj)

    quiver_obj = bpy.data.objects.get(quiver_name)
    if not quiver_obj:
        quiver_mesh = bpy.data.meshes.new(f"Quivers_{obj.name}")
        mat = get_or_create_quiver_material()
        quiver_mesh.materials.append(mat)
        quiver_obj = bpy.data.objects.new(quiver_name, quiver_mesh)
        quiver_obj.matrix_world = mathutils.Matrix.Identity(4)
        context.collection.objects.link(quiver_obj)

        gn_tree = build_quiver_geometry_nodes(default_scale=0.15)
        mod = quiver_obj.modifiers.new(name="GNQuiver", type="NODES")
        mod.node_group = gn_tree

    return vol_obj, quiver_obj


def get_grid_queries(obj, res=64, padding=0.2, min_aspect_ratio=0.33):
    if (
        obj.get("winder_is_grid_frozen", False)
        and "winder_frozen_min" in obj
        and "winder_frozen_max" in obj
    ):
        min_b = np.array(obj["winder_frozen_min"], dtype=np.float32)
        max_b = np.array(obj["winder_frozen_max"], dtype=np.float32)
    else:
        bbox = np.array([obj.matrix_world @ mathutils.Vector(b) for b in obj.bound_box])
        min_raw, max_raw = bbox.min(axis=0), bbox.max(axis=0)

        center = (min_raw + max_raw) * 0.5
        extent = max_raw - min_raw

        max_dim = np.max(extent)
        if max_dim < 1e-6:
            max_dim = 1.0

        min_extent = max_dim * min_aspect_ratio
        adj_extent = np.maximum(extent, min_extent)

        padded_half_extent = (adj_extent * 0.5) * (1.0 + 2.0 * padding)

        min_b = center - padded_half_extent
        max_b = center + padded_half_extent

    x = np.linspace(min_b[0], max_b[0], res, dtype=np.float32)
    y = np.linspace(min_b[1], max_b[1], res, dtype=np.float32)
    z = np.linspace(min_b[2], max_b[2], res, dtype=np.float32)

    gx, gy, gz = np.meshgrid(x, y, z, indexing="ij")
    queries = np.stack([gx.flatten(), gy.flatten(), gz.flatten()], axis=-1)
    return queries, (res, res, res), (min_b, max_b)


def get_multi_object_grid_queries(objs, res=64, padding=0.1):
    min_b = np.array([float("inf"), float("inf"), float("inf")], dtype=np.float32)
    max_b = np.array([-float("inf"), -float("inf"), -float("inf")], dtype=np.float32)

    for obj in objs:
        bbox = np.array(
            [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box],
            dtype=np.float32,
        )
        min_b = np.minimum(min_b, bbox.min(axis=0))
        max_b = np.maximum(max_b, bbox.max(axis=0))

    diag = np.linalg.norm(max_b - min_b)
    pad_val = diag * padding
    min_b -= pad_val
    max_b += pad_val

    if isinstance(res, int):
        dims = np.array([res, res, res], dtype=np.int32)
    else:
        dims = np.array(res, dtype=np.int32)

    xs = np.linspace(min_b[0], max_b[0], dims[0], dtype=np.float32)
    ys = np.linspace(min_b[1], max_b[1], dims[1], dtype=np.float32)
    zs = np.linspace(min_b[2], max_b[2], dims[2], dtype=np.float32)

    grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing="ij")
    queries = np.stack([grid_x, grid_y, grid_z], axis=-1)

    return queries, tuple(dims), (min_b, max_b)


# ===========================================================================
# Freeze-grid operator
# ===========================================================================
class WM_OT_toggle_freeze_grid(bpy.types.Operator):
    bl_idname = "winder.toggle_freeze_grid"
    bl_label = "Toggle Freeze Grid Bounds"
    bl_description = "Lock current grid bounds so object transformations don't alter the volume query box"

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != "MESH":
            self.report({"ERROR"}, "Select a Mesh Object")
            return {"CANCELLED"}

        props = context.scene.winder_props
        is_frozen = obj.get("winder_is_grid_frozen", False)

        if is_frozen:
            obj["winder_is_grid_frozen"] = False
            if "winder_frozen_min" in obj:
                del obj["winder_frozen_min"]
            if "winder_frozen_max" in obj:
                del obj["winder_frozen_max"]
            self.report({"INFO"}, f"Unfrozen grid bounds for {obj.name}")
        else:
            obj["winder_is_grid_frozen"] = False
            _, _, (min_b, max_b) = get_grid_queries(
                obj, res=props.query_res, padding=props.grid_padding
            )
            obj["winder_frozen_min"] = min_b.tolist()
            obj["winder_frozen_max"] = max_b.tolist()
            obj["winder_is_grid_frozen"] = True
            self.report({"INFO"}, f"Frozen grid bounds for {obj.name}")

        if context.scene.get("winder_active_target") == obj.name:
            recompute_winding_field(context, obj)

        return {"FINISHED"}


# ===========================================================================
# Winding-field recompute
# ===========================================================================
def recompute_winding_field(context, obj):
    props = context.scene.winder_props
    vol_obj, quiver_obj = get_or_create_winding_objects(context, obj)
    obj.display_type = "WIRE"

    res = props.query_res
    padding = props.grid_padding
    show_quivers = props.is_quiver_creation_active
    use_fast_engine = props.use_fast_engine

    with CudaStream() as stream:
        queries, shape, (min_b, max_b) = get_grid_queries(obj, res=res, padding=padding)
        queries = queries.reshape([-1, 3])
        num_q = len(queries)

        q_buf = CudaBuffer(queries.shape, dtype=np.float32)
        out_w_buf = CudaBuffer((num_q,), dtype=np.float32)
        q_buf.copy_from_numpy_async(queries, stream=stream)

        geom_data = extract_geometry_data(obj, props.geometry_mode, stream=stream)
        stream.keep_alive(geom_data)
        stream.keep_alive(
            compute_winding_field(
                geom_data,
                q_buf,
                out_w_buf,
                props.epsilon,
                stream=stream,
                use_fast_engine=use_fast_engine,
            )
        )

        host_w = np.empty(num_q, dtype=np.float32)
        out_w_buf.copy_to_numpy_async(host_w, stream=stream)

        mode = props.geometry_mode
        if show_quivers:
            ones_arr = np.ones(num_q, dtype=np.float32)
            ones_buf = CudaBuffer((num_q,), dtype=np.float32)
            ones_buf.copy_from_numpy_async(ones_arr, stream=stream)

            if mode == "PointNormal":
                grad_shape = (geom_data["count"], 2, 3)
            elif mode == "Triangle":
                grad_shape = (geom_data["count"], 3, 3)
            elif mode == "Mesh":
                grad_shape = (geom_data["num_verts"], 3)

            out_g_buf = CudaBuffer(grad_shape, dtype=np.float32)
            stream.keep_alive(
                compute_geometry_gradients(
                    geom_data,
                    q_buf,
                    ones_buf,
                    out_g_buf,
                    epsilon=props.epsilon,
                    stream=stream,
                    use_fast_engine=use_fast_engine,
                )
            )

            grad_data = np.empty(grad_shape, dtype=np.float32)
            out_g_buf.copy_to_numpy_async(grad_data, stream=stream)

    voxels = host_w.reshape(shape)
    res_arr = np.array(shape, dtype=np.float64)
    dx = (max_b - min_b) / np.maximum(res_arr - 1.0, 1.0)
    transform = openvdb.createLinearTransform(
        matrix=[
            [dx[0], 0.0, 0.0, 0.0],
            [0.0, dx[1], 0.0, 0.0],
            [0.0, 0.0, dx[2], 0.0],
            [min_b[0], min_b[1], min_b[2], 1.0],
        ]
    )

    grid_density = openvdb.FloatGrid()
    grid_density.copyFromArray(np.abs(voxels).astype(np.float64))
    grid_density.name = "density"
    grid_density.transform = transform

    grid_winding = openvdb.FloatGrid()
    grid_winding.copyFromArray(voxels.astype(np.float64))
    grid_winding.name = "winding"
    grid_winding.transform = transform

    old_vdb = vol_obj.data.filepath
    temp_vdb = os.path.join(
        tempfile.gettempdir(), f"winding_{obj.name}_{uuid.uuid4().hex[:6]}.vdb"
    )
    openvdb.write(temp_vdb, grids=[grid_density, grid_winding])
    vol_obj.data.filepath = temp_vdb
    vol_obj.data.update_tag()
    vol_obj.update_tag()

    if old_vdb and os.path.exists(old_vdb) and old_vdb != temp_vdb:
        try:
            os.remove(old_vdb)
        except OSError:
            pass

    if show_quivers:
        if mode == "PointNormal":
            pts = np.empty((geom_data["count"], 3), dtype=np.float32)
            geom_data["points"].copy_to_numpy_async(pts, stream=stream)
            stream.synchronize()
            vecs = grad_data[:, 1, :]
        elif mode == "Triangle":
            tris = np.empty((geom_data["count"], 3, 3), dtype=np.float32)
            geom_data["triangles"].copy_to_numpy_async(tris, stream=stream)
            stream.synchronize()
            pts = tris.reshape(-1, 3)
            vecs = grad_data.reshape(-1, 3)
        elif mode == "Mesh":
            pts = np.empty((geom_data["num_verts"], 3), dtype=np.float32)
            geom_data["vertices"].copy_to_numpy_async(pts, stream=stream)
            stream.synchronize()
            vecs = grad_data

        mags = np.linalg.norm(vecs, axis=1)
        safe_mags = np.maximum(mags[:, None], 1e-8)
        dirs = vecs / safe_mags

        quiver_mesh = quiver_obj.data
        quiver_mesh.clear_geometry()
        quiver_mesh.from_pydata(pts.tolist(), [], [])

        if "gradient_dir" in quiver_mesh.attributes:
            attr_dir = quiver_mesh.attributes["gradient_dir"]
        else:
            attr_dir = quiver_mesh.attributes.new(
                name="gradient_dir", type="FLOAT_VECTOR", domain="POINT"
            )
        attr_dir.data.foreach_set("vector", dirs.ravel())

        if "gradient_mag" in quiver_mesh.attributes:
            attr_mag = quiver_mesh.attributes["gradient_mag"]
        else:
            attr_mag = quiver_mesh.attributes.new(
                name="gradient_mag", type="FLOAT", domain="POINT"
            )
        attr_mag.data.foreach_set("value", mags.ravel())

        quiver_mesh.update()
    else:
        quiver_name = f"WindingQuiver_{obj.name}"
        existing_quiver = bpy.data.objects.get(quiver_name)
        if existing_quiver:
            mesh_data = existing_quiver.data
            bpy.data.objects.remove(existing_quiver, do_unlink=True)
            if mesh_data and mesh_data.users == 0:
                bpy.data.meshes.remove(mesh_data)

    cdf_percentiles = np.linspace(0.0, 100.0, 1001)
    cdf_values = np.percentile(host_w, cdf_percentiles).astype(float).tolist()

    vol_obj["winder_quantile_cdf"] = cdf_values
    vol_obj["winder_source_object"] = obj.name

    contour_obj_name = f"IsoContours_{vol_obj.name}"
    cut_empty_name = f"CutPlane_{vol_obj.name}"
    contour_obj = bpy.data.objects.get(contour_obj_name)
    cut_empty = bpy.data.objects.get(cut_empty_name)

    if contour_obj and cut_empty:
        thresholds = get_contour_thresholds(
            vol_obj,
            num_shells=props.contour_count,
            is_winding_field=props.is_winding_field,
        )
        if thresholds:
            build_marching_cubes_contour_nodes(
                vol_obj=vol_obj,
                cut_empty=cut_empty,
                thresholds=thresholds,
                is_winding_field=props.is_winding_field,
            )

    if hasattr(context, "screen") and context.screen:
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


# ===========================================================================
# Winding operator
# ===========================================================================
class WM_OT_create_winding_field(bpy.types.Operator):
    bl_idname = "winder.create_winding_field"
    bl_label = "Create Winding Number Field"

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != "MESH":
            self.report({"ERROR"}, "Select a Mesh Object")
            return {"CANCELLED"}

        context.scene["winder_active_target"] = obj.name
        recompute_winding_field(context, obj)
        self.report(
            {"INFO"}, f"Generated Winding Field for {obj.name} (Live Sync Active)"
        )
        return {"FINISHED"}


# ===========================================================================
# Loss landscape stub
# ===========================================================================
class WM_OT_create_loss_landscape(bpy.types.Operator):
    bl_idname = "winder.create_loss_landscape"
    bl_label = "Create Loss Landscape"

    def execute(self, context):
        self.report({"INFO"}, "Evaluated 3D Position Loss Landscape")
        return {"FINISHED"}


# ===========================================================================
# 3D contours
# ===========================================================================
def get_contour_thresholds(
    vol_obj, num_shells, is_winding_field=True, p_min=2.0, p_max=98.0
):
    if is_winding_field:
        if num_shells == 1:
            return [0.5]
        return np.linspace(-0.5, 1.0, num_shells).tolist()
    else:
        cdf = vol_obj.get("winder_quantile_cdf")
        if not cdf:
            return None

        if num_shells == 1:
            ps = np.array([50.0])
        else:
            ps = np.linspace(p_min, p_max, num_shells)

        cdf_percentiles = np.linspace(0.0, 100.0, len(cdf))
        thresholds = np.interp(ps, cdf_percentiles, cdf)
        return thresholds.tolist()


class WM_OT_create_3d_contours(bpy.types.Operator):
    bl_idname = "winder.create_3d_contours"
    bl_label = "Create 3D Contours"

    def execute(self, context):
        vol_obj = context.active_object
        if not vol_obj or vol_obj.type != "VOLUME":
            self.report({"ERROR"}, "Select a Volume Object")
            return {"CANCELLED"}

        props = context.scene.winder_props

        thresholds = get_contour_thresholds(
            vol_obj,
            num_shells=props.contour_count,
            is_winding_field=props.is_winding_field,
        )
        if not thresholds:
            self.report(
                {"ERROR"},
                f"No field data found for '{vol_obj.name}'. Run 'Create Winding Number Field' first.",
            )
            return {"CANCELLED"}

        contour_obj_name = f"IsoContours_{vol_obj.name}"
        cut_empty_name = f"CutPlane_{vol_obj.name}"

        existing_contour_obj = bpy.data.objects.get(contour_obj_name)
        if existing_contour_obj:
            self.report(
                {"INFO"},
                f"Iso-Contours already exist for '{vol_obj.name}'. Reusing object.",
            )
            bpy.ops.object.select_all(action="DESELECT")
            existing_contour_obj.select_set(True)
            context.view_layer.objects.active = existing_contour_obj
            return {"FINISHED"}

        cut_empty = bpy.data.objects.get(cut_empty_name)
        if not cut_empty:
            cut_empty = bpy.data.objects.new(cut_empty_name, None)
            cut_empty.empty_display_type = "SINGLE_ARROW"
            context.collection.objects.link(cut_empty)

        contour_mesh = bpy.data.meshes.new(contour_obj_name)
        contour_obj = bpy.data.objects.new(contour_obj_name, contour_mesh)
        context.collection.objects.link(contour_obj)

        gn_tree = build_marching_cubes_contour_nodes(
            vol_obj=vol_obj,
            cut_empty=cut_empty,
            thresholds=thresholds,
            is_winding_field=props.is_winding_field,
        )

        mod = contour_obj.modifiers.new(name="GNContours", type="NODES")
        mod.node_group = gn_tree

        bpy.ops.object.select_all(action="DESELECT")
        contour_obj.select_set(True)
        context.view_layer.objects.active = contour_obj

        self.report({"INFO"}, f"Created Quantile 3D Iso-Contours for {vol_obj.name}")
        return {"FINISHED"}


def update_contour_shells(self, context):
    shell_count = self.contour_count
    is_winding = self.is_winding_field

    for obj in bpy.data.objects:
        if obj.name.startswith("IsoContours_"):
            vol_name = obj.name.replace("IsoContours_", "")
            vol_obj = bpy.data.objects.get(vol_name)
            cut_empty = bpy.data.objects.get(f"CutPlane_{vol_name}")

            if vol_obj and cut_empty:
                thresholds = get_contour_thresholds(
                    vol_obj, num_shells=shell_count, is_winding_field=is_winding
                )
                if thresholds:
                    build_marching_cubes_contour_nodes(
                        vol_obj=vol_obj,
                        cut_empty=cut_empty,
                        thresholds=thresholds,
                        is_winding_field=is_winding,
                    )

    if hasattr(context, "screen") and context.screen:
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


# ===========================================================================
# Bulk recompute callback
# ===========================================================================
def update_all_winding_fields(self, context):
    """Refresh every existing Winder visualization when a property changes.

    Never creates new volumes — only updates what the user already has.
    """
    target_objs = set()

    if hasattr(context, "scene") and "winder_active_target" in context.scene:
        active_name = context.scene["winder_active_target"]
        active_obj = bpy.data.objects.get(active_name)
        if active_obj and active_obj.type == "MESH":
            target_objs.add(active_obj)

    for obj in bpy.data.objects:
        if obj.type == "VOLUME":
            source_name = obj.get("winder_source_object") or obj.get(
                "winder_payload_source"
            )
            if source_name and source_name in bpy.data.objects:
                source_obj = bpy.data.objects[source_name]
                if source_obj.type == "MESH":
                    target_objs.add(source_obj)
                    continue

            if "winder_quantile_cdf" in obj:
                for prefix in ("WindingVol_", "Winding_", "Vol_"):
                    if obj.name.startswith(prefix):
                        mesh_name = obj.name[len(prefix) :]
                        mesh_obj = bpy.data.objects.get(mesh_name)
                        if mesh_obj and mesh_obj.type == "MESH":
                            target_objs.add(mesh_obj)
                            break

    for mesh_obj in target_objs:
        name = mesh_obj.name
        if bpy.data.objects.get(f"WindingVol_{name}"):
            try:
                recompute_winding_field(context, mesh_obj)
            except Exception as e:
                print(f"[Winder] Error updating winding field for '{name}': {e}")

        if bpy.data.objects.get(f"PayloadVol_{name}"):
            try:
                recompute_payload_field(context, mesh_obj)
            except Exception as e:
                print(f"[Winder] Error updating payload field for '{name}': {e}")

        if bpy.data.objects.get(f"PayloadGradVol_{name}"):
            try:
                recompute_payload_query_gradients(context, mesh_obj)
            except Exception as e:
                print(f"[Winder] Error updating payload grad volume for '{name}': {e}")

        if bpy.data.objects.get(f"PayloadStreamlines_{name}"):
            try:
                recompute_payload_query_streamlines(context, mesh_obj)
            except Exception as e:
                print(f"[Winder] Error updating payload field lines for '{name}': {e}")

        if bpy.data.objects.get(f"PayloadGeomQuiver_{name}"):
            try:
                _recompute_payload_geometry_gradients(context, mesh_obj)
            except NotImplementedError:
                pass  # user has fast engine on — leave stale
            except Exception as e:
                print(
                    f"[Winder] Error updating payload geom quiver for "
                    f"'{name}': {e}"
                )

    update_all_query_gradients(self, context)
    update_all_streamlines(self, context)

    if hasattr(context, "screen") and context.screen:
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


# ===========================================================================
# Optimization (loss minimization)
# ===========================================================================
def get_or_create_optimization_objects(context, source_obj):
    vol_name = f"LossVolume_{source_obj.name}"
    quiver_name = f"LossQuiver_{source_obj.name}"

    vol_obj = bpy.data.objects.get(vol_name)
    if not vol_obj:
        vol_data = bpy.data.volumes.new(vol_name)
        vol_obj = bpy.data.objects.new(vol_name, vol_data)
        context.collection.objects.link(vol_obj)

    quiver_obj = bpy.data.objects.get(quiver_name)
    if not quiver_obj:
        mesh_data = bpy.data.meshes.new(quiver_name)
        quiver_obj = bpy.data.objects.new(quiver_name, mesh_data)
        context.collection.objects.link(quiver_obj)

    return vol_obj, quiver_obj


class WM_OT_create_optimization(bpy.types.Operator):
    bl_idname = "winder.create_optimization"
    bl_label = "Create Field Optimization"
    bl_description = (
        "Computes source vs. target winding field loss and visualizes loss "
        "gradients on the source geometry"
    )

    def execute(self, context):
        source_obj = context.active_object
        selected_objs = [o for o in context.selected_objects if o.type == "MESH"]

        if not source_obj or source_obj.type != "MESH":
            self.report({"ERROR"}, "Active selection must be a Mesh object (Source).")
            return {"CANCELLED"}

        target_objs = [o for o in selected_objs if o != source_obj]
        if not target_objs:
            self.report(
                {"ERROR"},
                "Select at least two mesh objects (Active = Source, Others = Target).",
            )
            return {"CANCELLED"}

        props = context.scene.winder_props
        res = props.query_res
        padding = props.grid_padding
        mode = props.geometry_mode
        use_fast_engine = props.use_fast_engine

        vol_obj, quiver_obj = get_or_create_optimization_objects(context, source_obj)
        source_obj.display_type = "WIRE"

        host_loss = None
        quiver_pts = None
        quiver_vecs = None

        with CudaStream() as stream:
            all_objs = [source_obj] + target_objs
            queries, shape, (min_b, max_b) = get_multi_object_grid_queries(
                all_objs, res=res, padding=padding
            )
            queries = queries.reshape([-1, 3])
            num_q = len(queries)

            q_buf = CudaBuffer(queries.shape, dtype=np.float32)
            w_src_buf = CudaBuffer((num_q,), dtype=np.float32)
            w_tgt_buf = CudaBuffer((num_q,), dtype=np.float32)

            stream.keep_alive(q_buf, w_src_buf, w_tgt_buf)
            q_buf.copy_from_numpy_async(queries, stream=stream)

            src_geom = extract_geometry_data(source_obj, mode=mode, stream=stream)
            stream.keep_alive(
                compute_winding_field(
                    src_geom,
                    q_buf,
                    w_src_buf,
                    props.epsilon,
                    stream=stream,
                    use_fast_engine=use_fast_engine,
                )
            )

            tgt_geom = extract_combined_geometry_data(
                target_objs, mode=mode, stream=stream
            )
            stream.keep_alive(
                compute_winding_field(
                    tgt_geom,
                    q_buf,
                    w_tgt_buf,
                    props.epsilon,
                    stream=stream,
                    use_fast_engine=use_fast_engine,
                )
            )

            host_w_src = np.empty(num_q, dtype=np.float32)
            host_w_tgt = np.empty(num_q, dtype=np.float32)

            w_src_buf.copy_to_numpy_async(host_w_src, stream=stream)
            w_tgt_buf.copy_to_numpy_async(host_w_tgt, stream=stream)
            stream.synchronize()

            diff = host_w_src - host_w_tgt
            if self.loss_type == "L1":
                host_loss = np.abs(diff)
                dL_dw = np.sign(diff).astype(np.float32)
            else:
                host_loss = np.square(diff)
                dL_dw = (2.0 * diff).astype(np.float32)

            dL_dw_buf = CudaBuffer((num_q,), dtype=np.float32)
            dL_dw_buf.copy_from_numpy_async(dL_dw, stream=stream)

            if mode == "PointNormal":
                grad_shape = (src_geom["count"], 2, 3)
            elif mode == "Triangle":
                grad_shape = (src_geom["count"], 3, 3)
            elif mode == "Mesh":
                grad_shape = (src_geom["num_verts"], 3)

            out_grad_buf = CudaBuffer(grad_shape, dtype=np.float32)
            grad_data = np.empty(grad_shape, dtype=np.float32)

            stream.keep_alive(dL_dw_buf, out_grad_buf)

            stream.keep_alive(
                compute_geometry_gradients(
                    src_geom,
                    q_buf,
                    dL_dw_buf,
                    out_grad_buf,
                    epsilon=props.epsilon,
                    stream=stream,
                    use_fast_engine=use_fast_engine,
                )
            )

            out_grad_buf.copy_to_numpy_async(grad_data, stream=stream)

            if mode == "PointNormal":
                pts = np.empty((src_geom["count"], 3), dtype=np.float32)
                src_geom["points"].copy_to_numpy_async(pts, stream=stream)
                quiver_pts = pts
                quiver_vecs = grad_data[:, 1, :]
            elif mode == "Triangle":
                tris = np.empty((src_geom["count"], 3, 3), dtype=np.float32)
                src_geom["triangles"].copy_to_numpy_async(tris, stream=stream)
                quiver_pts = tris.reshape(-1, 3)
                quiver_vecs = grad_data.reshape(-1, 3)
            elif mode == "Mesh":
                pts = np.empty((src_geom["num_verts"], 3), dtype=np.float32)
                src_geom["vertices"].copy_to_numpy_async(pts, stream=stream)
                quiver_pts = pts
                quiver_vecs = grad_data

        voxels = host_loss.reshape(shape)
        res_arr = np.array(shape, dtype=np.float64)
        dx = (max_b - min_b) / np.maximum(res_arr - 1.0, 1.0)
        transform = openvdb.createLinearTransform(
            matrix=[
                [dx[0], 0.0, 0.0, 0.0],
                [0.0, dx[1], 0.0, 0.0],
                [0.0, 0.0, dx[2], 0.0],
                [min_b[0], min_b[1], min_b[2], 1.0],
            ]
        )

        grid_loss = openvdb.FloatGrid()
        grid_loss.copyFromArray(voxels.astype(np.float64))
        grid_loss.name = "loss"
        grid_loss.transform = transform

        old_vdb = vol_obj.data.filepath
        temp_vdb = os.path.join(
            tempfile.gettempdir(),
            f"loss_{source_obj.name}_{uuid.uuid4().hex[:6]}.vdb",
        )
        openvdb.write(temp_vdb, grids=[grid_loss])
        vol_obj.data.filepath = temp_vdb

        if old_vdb and os.path.exists(old_vdb) and old_vdb != temp_vdb:
            try:
                os.remove(old_vdb)
            except OSError:
                pass

        if quiver_pts is not None and quiver_vecs is not None:
            mags = np.linalg.norm(quiver_vecs, axis=1)
            safe_mags = np.maximum(mags[:, None], 1e-8)
            dirs = quiver_vecs / safe_mags

            quiver_mesh = quiver_obj.data
            quiver_mesh.clear_geometry()
            quiver_mesh.from_pydata(quiver_pts.tolist(), [], [])

            if "gradient_dir" in quiver_mesh.attributes:
                attr_dir = quiver_mesh.attributes["gradient_dir"]
            else:
                attr_dir = quiver_mesh.attributes.new(
                    name="gradient_dir", type="FLOAT_VECTOR", domain="POINT"
                )
            attr_dir.data.foreach_set("vector", dirs.ravel())

            if "gradient_mag" in quiver_mesh.attributes:
                attr_mag = quiver_mesh.attributes["gradient_mag"]
            else:
                attr_mag = quiver_mesh.attributes.new(
                    name="gradient_mag", type="FLOAT", domain="POINT"
                )
            attr_mag.data.foreach_set("value", mags.ravel())

            quiver_mesh.update()

        cdf_percentiles = np.linspace(0.0, 100.0, 1001)
        cdf_values = np.percentile(host_loss, cdf_percentiles).astype(float).tolist()
        vol_obj["winder_loss_quantile_cdf"] = cdf_values
        vol_obj["winder_source_object"] = source_obj.name
        vol_obj["winder_target_objects"] = [o.name for o in target_objs]

        if hasattr(context, "screen") and context.screen:
            for area in context.screen.areas:
                if area.type == "VIEW_3D":
                    area.tag_redraw()

        self.report(
            {"INFO"},
            f"Created Field Optimization Loss ({self.loss_type}) for {source_obj.name}",
        )
        return {"FINISHED"}


# ===========================================================================
# BVH debug operators
# ===========================================================================
class WM_OT_dump_forward_bvh(bpy.types.Operator):
    bl_idname = "winder.dump_forward_bvh"
    bl_label = "Visualize Forward BVH8"
    bl_description = (
        "Parse WindingNumberEngine BVH8 .dot representation and create a "
        "colorized 3D wireframe mesh"
    )

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != "MESH":
            self.report({"ERROR"}, "Select a Mesh Object")
            return {"CANCELLED"}

        props = context.scene.winder_props
        with CudaStream() as stream:
            geom_data = extract_geometry_data(obj, props.geometry_mode, stream=stream)
            dot_str = dump_forward_bvh(geom_data, stream=stream)

        nodes, adj, root_id = parse_bvh_dot(dot_str)

        if not nodes:
            self.report({"WARNING"}, "BVH8 string is empty or contains no geometry")
            return {"CANCELLED"}

        bvh_obj = build_bvh_wireframe_mesh(
            nodes, adj, root_id, obj_name=f"BVH8_Forward_{obj.name}"
        )

        bpy.ops.object.select_all(action="DESELECT")
        bvh_obj.select_set(True)
        context.view_layer.objects.active = bvh_obj

        self.report(
            {"INFO"},
            f"Created BVH8 Mesh for {obj.name} ({len(nodes)} total AABB nodes)",
        )
        return {"FINISHED"}


class WM_OT_dump_backward_bvh(bpy.types.Operator):
    bl_idname = "winder.dump_backward_bvh"
    bl_label = "Visualize Backward BVH8"
    bl_description = (
        "Parse GradientBackend BVH8 .dot representation and create a "
        "colorized 3D wireframe mesh"
    )

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != "MESH":
            self.report({"ERROR"}, "Select a Mesh Object")
            return {"CANCELLED"}

        props = context.scene.winder_props

        with CudaStream() as stream:
            queries, shape, (min_b, max_b) = get_grid_queries(
                obj, res=props.query_res, padding=props.grid_padding
            )
            queries = queries.reshape([-1, 3])
            num_q = len(queries)

            q_buf = CudaBuffer(queries.shape, dtype=np.float32)
            q_buf.copy_from_numpy_async(queries, stream=stream)

            ones_arr = np.ones(num_q, dtype=np.float32)
            ones_buf = CudaBuffer((num_q,), dtype=np.float32)
            ones_buf.copy_from_numpy_async(ones_arr, stream=stream)

            dot_str = dump_backward_bvh(q_buf, ones_buf, stream=stream)

        nodes, adj, root_id = parse_bvh_dot(dot_str)

        if not nodes:
            self.report(
                {"WARNING"}, "Backward BVH8 string is empty or contains no queries"
            )
            return {"CANCELLED"}

        bvh_obj = build_bvh_wireframe_mesh(
            nodes,
            adj,
            root_id,
            obj_name=f"BVH8_Backward_{obj.name}",
            pass_type="Backward_Jet",
        )

        bpy.ops.object.select_all(action="DESELECT")
        bvh_obj.select_set(True)
        context.view_layer.objects.active = bvh_obj

        self.report(
            {"INFO"},
            f"Created Backward BVH8 Mesh for {obj.name} ({len(nodes)} total AABB nodes)",
        )
        return {"FINISHED"}


# ===========================================================================
# Query gradients (∇_x W)
# ===========================================================================
def get_or_create_query_grad_quiver(context, obj, props):
    quiver_name = f"QueryGradQuiver_{obj.name}"
    quiver_obj = bpy.data.objects.get(quiver_name)

    if quiver_obj is None:
        quiver_mesh = bpy.data.meshes.new(f"QueryGradQuivers_{obj.name}")
        quiver_mesh.materials.append(get_or_create_query_grad_material())
        quiver_obj = bpy.data.objects.new(quiver_name, quiver_mesh)
        quiver_obj.matrix_world = mathutils.Matrix.Identity(4)
        context.collection.objects.link(quiver_obj)

        gn_tree = build_query_grad_geometry_nodes(
            default_scale=float(props.query_grad_scale)
        )
        mod = quiver_obj.modifiers.new(name="GNQueryGradQuiver", type="NODES")
        mod.node_group = gn_tree

    return quiver_obj


def get_or_create_payload_geom_quiver(context, obj, props):
    """Retrieves or creates the payload geometry-gradient quiver object."""
    quiver_name = f"PayloadGeomQuiver_{obj.name}"
    quiver_obj = bpy.data.objects.get(quiver_name)

    if quiver_obj is None:
        quiver_mesh = bpy.data.meshes.new(f"PayloadGeomQuivers_{obj.name}")
        quiver_mesh.materials.append(get_or_create_query_grad_material())
        quiver_obj = bpy.data.objects.new(quiver_name, quiver_mesh)
        quiver_obj.matrix_world = mathutils.Matrix.Identity(4)
        context.collection.objects.link(quiver_obj)

        # Reuse the query-gradient GN cone tree: it reads `gradient_dir` and
        # `gradient_mag`, exactly what we set below.
        gn_tree = build_query_grad_geometry_nodes(
            default_scale=float(props.query_grad_scale)
        )
        mod = quiver_obj.modifiers.new(name="GNPayloadGeomQuiver", type="NODES")
        mod.node_group = gn_tree

    return quiver_obj


def _recompute_query_gradients(context, obj) -> int:
    props = context.scene.winder_props
    res = props.query_res
    padding = props.grid_padding
    stride = max(1, int(props.query_grad_stride))
    use_fast_engine = props.use_fast_engine

    quiver_obj = get_or_create_query_grad_quiver(context, obj, props)

    with CudaStream() as stream:
        queries_full, shape_full, _ = get_grid_queries(obj, res=res, padding=padding)
        q3d = queries_full.reshape(shape_full + (3,))
        sub = q3d[::stride, ::stride, ::stride]
        queries = sub.reshape(-1, 3).astype(np.float32)
        num_q = len(queries)

        if num_q == 0:
            return 0

        q_buf = CudaBuffer(queries.shape, dtype=np.float32)
        q_buf.copy_from_numpy_async(queries, stream=stream)

        ones = np.ones(num_q, dtype=np.float32)
        ones_buf = CudaBuffer((num_q,), dtype=np.float32)
        ones_buf.copy_from_numpy_async(ones, stream=stream)

        out_g_buf = CudaBuffer((num_q, 3), dtype=np.float32)

        geom_data = extract_geometry_data(obj, props.geometry_mode, stream=stream)
        stream.keep_alive(geom_data)
        stream.keep_alive(
            compute_query_gradients(
                geom_data,
                q_buf,
                ones_buf,
                out_g_buf,
                epsilon=props.epsilon,
                stream=stream,
                use_fast_engine=use_fast_engine,
            )
        )

        grad_data = np.empty((num_q, 3), dtype=np.float32)
        out_g_buf.copy_to_numpy_async(grad_data, stream=stream)

    mags = np.linalg.norm(grad_data, axis=1)
    safe_mags = np.maximum(mags[:, None], 1e-12)
    dirs = (grad_data / safe_mags).astype(np.float32)

    quiver_mesh = quiver_obj.data
    quiver_mesh.clear_geometry()
    quiver_mesh.from_pydata(queries.tolist(), [], [])

    if "gradient_dir" in quiver_mesh.attributes:
        attr_dir = quiver_mesh.attributes["gradient_dir"]
    else:
        attr_dir = quiver_mesh.attributes.new(
            name="gradient_dir", type="FLOAT_VECTOR", domain="POINT"
        )
    attr_dir.data.foreach_set("vector", dirs.ravel())

    if "gradient_mag" in quiver_mesh.attributes:
        attr_mag = quiver_mesh.attributes["gradient_mag"]
    else:
        attr_mag = quiver_mesh.attributes.new(
            name="gradient_mag", type="FLOAT", domain="POINT"
        )
    attr_mag.data.foreach_set("value", mags.ravel())

    quiver_mesh.update()

    _apply_query_grad_scale_to_object(quiver_obj, props.query_grad_scale)
    return num_q


def _recompute_payload_geometry_gradients(context, obj) -> int:
    """Geometry gradients of the payload field, visualized as quivers at the
    primitive positions.

    Uses `grad_output = ones(M, D)` so the visualized loss is
    `L = Σ_j Σ_c W_c(q_j)` — i.e. how moving each primitive changes the total
    payload field summed over all queries and channels.

    Only the brute-force binding is implemented; if `use_fast_engine` resolves
    to True the wrapper raises `NotImplementedError` before any work is done.
    """
    props = context.scene.winder_props
    res = props.query_res
    padding = props.grid_padding
    stride = max(1, int(props.query_grad_stride))
    use_fast_engine = props.use_fast_engine

    if bpy.data.objects.get(f"PayloadVol_{obj.name}") is None:
        raise RuntimeError(
            f"No payload field for {obj.name}. Create a payload field first."
        )

    payload = _extract_padded_payload(obj, props.geometry_mode)
    if len(payload) == 0:
        raise ValueError(f"No triangles in '{obj.name}'")
    num_geom = len(payload)

    quiver_obj = get_or_create_payload_geom_quiver(context, obj, props)

    with CudaStream() as stream:
        # Subsample the query grid with `query_grad_stride` — same knob as
        # the query-gradient quivers, and the brute-force cost is O(N·M·D)
        # so the stride is what keeps this tractable on large meshes.
        queries_full, shape_full, _ = get_grid_queries(obj, res=res, padding=padding)
        q3d = queries_full.reshape(shape_full + (3,))
        sub = q3d[::stride, ::stride, ::stride]
        queries = sub.reshape(-1, 3).astype(np.float32)
        num_q = len(queries)
        if num_q == 0:
            return 0

        q_buf = CudaBuffer(queries.shape, dtype=np.float32)
        q_buf.copy_from_numpy_async(queries, stream=stream)

        pl_buf = CudaBuffer(payload.shape, dtype=np.float32)
        pl_buf.copy_from_numpy_async(payload, stream=stream)

        # Adjoint: ones(M, D) — sum of all channels across all queries.
        adjoint = np.ones((num_q, payload.shape[1]), dtype=np.float32)
        adjoint_buf = CudaBuffer((num_q, payload.shape[1]), dtype=np.float32)
        adjoint_buf.copy_from_numpy_async(adjoint, stream=stream)

        geom_data = extract_geometry_data(obj, props.geometry_mode, stream=stream)
        stream.keep_alive(geom_data)
        if geom_data["count"] != num_geom:
            raise RuntimeError(
                f"Payload/geometry mismatch: {num_geom} vs "
                f"{geom_data['count']} elements."
            )

        mode = props.geometry_mode
        if mode == "PointNormal":
            grad_shape = (num_geom, 2, 3)
        elif mode == "Triangle":
            grad_shape = (num_geom, 3, 3)
        else:  # Mesh
            grad_shape = (geom_data["num_verts"], 3)

        out_grad_buf = CudaBuffer(grad_shape, dtype=np.float32)
        stream.keep_alive(out_grad_buf)

        stream.keep_alive(
            compute_payload_geometry_gradients(
                geom_data,
                pl_buf,
                q_buf,
                adjoint_buf,
                out_grad_buf,
                epsilon=props.epsilon,
                stream=stream,
                use_fast_engine=use_fast_engine,
            )
        )

        grad_data = np.empty(grad_shape, dtype=np.float32)
        out_grad_buf.copy_to_numpy_async(grad_data, stream=stream)

        # Mirror the winding-quiver extraction: positions of the primitives
        # and the *position* gradient (index 1 for PointNormal).
        if mode == "PointNormal":
            pts = np.empty((num_geom, 3), dtype=np.float32)
            geom_data["points"].copy_to_numpy_async(pts, stream=stream)
            stream.synchronize()
            vecs = grad_data[:, 1, :]
        elif mode == "Triangle":
            tris = np.empty((num_geom, 3, 3), dtype=np.float32)
            geom_data["triangles"].copy_to_numpy_async(tris, stream=stream)
            stream.synchronize()
            pts = tris.reshape(-1, 3)
            vecs = grad_data.reshape(-1, 3)
        else:  # Mesh
            pts = np.empty((geom_data["num_verts"], 3), dtype=np.float32)
            geom_data["vertices"].copy_to_numpy_async(pts, stream=stream)
            stream.synchronize()
            vecs = grad_data

    mags = np.linalg.norm(vecs, axis=1)
    safe_mags = np.maximum(mags[:, None], 1e-12)
    dirs = (vecs / safe_mags).astype(np.float32)

    quiver_mesh = quiver_obj.data
    quiver_mesh.clear_geometry()
    quiver_mesh.from_pydata(pts.tolist(), [], [])

    if "gradient_dir" in quiver_mesh.attributes:
        attr_dir = quiver_mesh.attributes["gradient_dir"]
    else:
        attr_dir = quiver_mesh.attributes.new(
            name="gradient_dir", type="FLOAT_VECTOR", domain="POINT"
        )
    attr_dir.data.foreach_set("vector", dirs.ravel())

    if "gradient_mag" in quiver_mesh.attributes:
        attr_mag = quiver_mesh.attributes["gradient_mag"]
    else:
        attr_mag = quiver_mesh.attributes.new(
            name="gradient_mag", type="FLOAT", domain="POINT"
        )
    attr_mag.data.foreach_set("value", mags.ravel())

    quiver_mesh.update()

    _apply_query_grad_scale_to_object(quiver_obj, props.query_grad_scale)
    return len(pts)


def _apply_query_grad_scale_to_object(quiver_obj, new_scale: float):
    for mod in quiver_obj.modifiers:
        if mod.type != "NODES" or not mod.node_group:
            continue
        for item in mod.node_group.interface.items_tree:
            if (
                item.item_type == "SOCKET"
                and getattr(item, "in_out", "") == "INPUT"
                and item.name == "Arrow Scale"
            ):
                try:
                    mod[item.identifier] = float(new_scale)
                except Exception:
                    pass
                return


def update_all_query_gradients(self, context):
    target_objs = []
    for obj in bpy.data.objects:
        if obj.type == "MESH" and bpy.data.objects.get(f"QueryGradQuiver_{obj.name}"):
            target_objs.append(obj)

    for obj in target_objs:
        try:
            _recompute_query_gradients(context, obj)
        except Exception as e:
            print(f"[Winder] Query gradient update error for '{obj.name}': {e}")

    if hasattr(context, "screen") and context.screen:
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def update_query_grad_scale(self, context):
    new_scale = float(self.query_grad_scale)
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        q_obj = bpy.data.objects.get(f"QueryGradQuiver_{obj.name}")
        if not q_obj:
            continue
        _apply_query_grad_scale_to_object(q_obj, new_scale)

    if hasattr(context, "screen") and context.screen:
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


class WM_OT_create_query_gradients(bpy.types.Operator):
    bl_idname = "winder.create_query_gradients"
    bl_label = "Visualize Query Gradients"
    bl_description = (
        "Compute ∇_x W(x) (spatial gradient of the winding field) at "
        "subsampled query grid points and visualize with instanced arrows"
    )

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != "MESH":
            self.report({"ERROR"}, "Select a Mesh Object")
            return {"CANCELLED"}

        props = context.scene.winder_props
        try:
            num_q = _recompute_query_gradients(context, obj)
        except Exception as e:
            self.report({"ERROR"}, f"Query gradient failed: {e}")
            return {"CANCELLED"}

        if num_q == 0:
            self.report({"WARNING"}, "No query points after subsampling")
            return {"CANCELLED"}

        quiver_obj = bpy.data.objects.get(f"QueryGradQuiver_{obj.name}")
        if quiver_obj:
            bpy.ops.object.select_all(action="DESELECT")
            quiver_obj.select_set(True)
            context.view_layer.objects.active = quiver_obj

        self.report(
            {"INFO"},
            f"Query gradients: {num_q} arrows "
            f"(res={props.query_res}, stride={props.query_grad_stride})",
        )
        return {"FINISHED"}


# ===========================================================================
# Streamlines (winding-field field lines)
# ===========================================================================
def _ensure_streamline_modifier(streamline_obj, props, diag):
    """Ensure the streamline object has exactly one GN modifier with a
    per-object node group.

    Using a per-object node group (name suffix `_{streamline_obj.name}`)
    means each modifier owns its own `Radius` default, so multiple objects
    with different `diag` values can each have the correct radius without
    any per-modifier override.
    """
    ng_name = f"GN_Streamlines_{streamline_obj.name}"
    gn = build_streamline_geometry_nodes(
        radius=props.streamline_radius * diag,
        node_group_name=ng_name,
    )

    # Migrate an existing NODES modifier to the per-object node group
    # (this also handles objects that were created with the old shared
    # `GN_Streamlines` name).
    for mod in streamline_obj.modifiers:
        if mod.type == "NODES":
            if mod.node_group != gn:
                mod.node_group = gn
            return

    mod = streamline_obj.modifiers.new(name="GNStreamlines", type="NODES")
    mod.node_group = gn


def _apply_streamline_radius(streamline_obj, radius):
    """Set the Radius socket default on this object's node group.

    Each streamline object owns its own node group (see
    `_ensure_streamline_modifier`), so updating `item.default_value` here
    changes only that modifier.
    """
    radius = float(radius)
    for mod in streamline_obj.modifiers:
        if mod.type != "NODES" or not mod.node_group:
            continue

        for item in mod.node_group.interface.items_tree:
            if (
                item.item_type == "SOCKET"
                and getattr(item, "in_out", "") == "INPUT"
                and item.name == "Radius"
            ):
                item.default_value = radius
                streamline_obj.update_tag()
                return

        print(f"[Winder] No 'Radius' socket on {streamline_obj.name}'s GN group")
        return


def _recompute_streamlines(context, obj):
    props = context.scene.winder_props
    use_fast_engine = props.use_fast_engine

    bbox = np.array(
        [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box],
        dtype=np.float32,
    )
    center = (bbox.min(axis=0) + bbox.max(axis=0)) * 0.5
    diag = float(np.linalg.norm(bbox.max(axis=0) - bbox.min(axis=0)))
    if diag < 1e-6:
        diag = 1.0

    pad = (props.grid_padding + 0.3) * diag
    half = (bbox.max(axis=0) - bbox.min(axis=0)) * 0.5 + pad
    bounds_min = (center - half).astype(np.float32)
    bounds_max = (center + half).astype(np.float32)

    step_size = float(props.streamline_step) * diag
    max_length = float(props.streamline_max_length) * diag

    with CudaStream() as stream:
        geom_data = extract_geometry_data(obj, props.geometry_mode, stream=stream)
        stream.keep_alive(geom_data)

        rng = np.random.default_rng(42)
        n_seeds = int(props.streamline_count)
        mode = props.geometry_mode
        jitter_scale = float(props.streamline_seed_jitter) * diag

        if mode == "PointNormal":
            pts = np.empty((geom_data["count"], 3), np.float32)
            geom_data["points"].copy_to_numpy_async(pts, stream=stream)
            stream.synchronize()
            seeds = _seed_on_point_normal(pts, n_seeds, rng, jitter_scale)
        elif mode == "Triangle":
            tris = np.empty((geom_data["count"], 3, 3), np.float32)
            geom_data["triangles"].copy_to_numpy_async(tris, stream=stream)
            stream.synchronize()
            seeds = _seed_on_triangles(tris, n_seeds, rng, jitter_scale)
        else:
            verts = np.empty((geom_data["num_verts"], 3), np.float32)
            idxs = np.empty((geom_data["num_tris"], 3), np.uint32)
            geom_data["vertices"].copy_to_numpy_async(verts, stream=stream)
            geom_data["triangle_indices"].copy_to_numpy_async(idxs, stream=stream)
            stream.synchronize()
            seeds = _seed_on_triangles(verts[idxs], n_seeds, rng, jitter_scale)

        line_points, line_mags, line_counts = trace_query_streamlines(
            geom_data,
            seeds,
            step_size,
            int(props.streamline_max_steps),
            max_length,
            bounds_min,
            bounds_max,
            float(props.streamline_min_grad),
            float(props.epsilon),
            -1.0,
            bool(props.streamline_bidirectional),
            stream,
            use_fast_engine=use_fast_engine,
        )

    num_lines = int(np.sum(line_counts >= 2))

    obj_name = f"Streamlines_{obj.name}"
    streamline_obj = bpy.data.objects.get(obj_name)

    if streamline_obj is None:
        mesh = build_streamline_mesh(
            line_points, line_mags, line_counts, f"{obj_name}_Mesh"
        )
        streamline_obj = bpy.data.objects.new(obj_name, mesh)
        context.collection.objects.link(streamline_obj)
        streamline_obj.matrix_world = mathutils.Matrix.Identity(4)
        _ensure_streamline_modifier(streamline_obj, props, diag)
    else:
        if streamline_obj.data is None:
            streamline_obj.data = bpy.data.meshes.new(f"{obj_name}_Mesh")
        update_mesh_from_lines(streamline_obj.data, line_points, line_mags, line_counts)
        _ensure_streamline_modifier(streamline_obj, props, diag)

    return num_lines


def update_all_streamlines(self, context):
    target_objs = []
    for obj in bpy.data.objects:
        if obj.type == "MESH" and bpy.data.objects.get(f"Streamlines_{obj.name}"):
            target_objs.append(obj)

    for obj in target_objs:
        try:
            _recompute_streamlines(context, obj)
        except Exception as e:
            print(f"[Winder] Streamline update error for '{obj.name}': {e}")

    if hasattr(context, "screen") and context.screen:
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def update_streamline_radius(self, context):
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        sl = bpy.data.objects.get(f"Streamlines_{obj.name}")
        if not sl:
            continue
        bbox = np.array(
            [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box],
            dtype=np.float32,
        )
        diag = float(np.linalg.norm(bbox.max(axis=0) - bbox.min(axis=0)))
        if diag < 1e-6:
            diag = 1.0
        _apply_streamline_radius(sl, self.streamline_radius * diag)

    if hasattr(context, "screen") and context.screen:
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


class WM_OT_create_query_streamlines(bpy.types.Operator):
    bl_idname = "winder.create_query_streamlines"
    bl_label = "Create Field Lines"
    bl_description = (
        "Trace streamlines of the winding-field gradient ∇_x W(x) from seeds "
        "sampled on the surface, integrate both directions, and render as tubes"
    )

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != "MESH":
            self.report({"ERROR"}, "Select a Mesh Object")
            return {"CANCELLED"}

        try:
            n_lines = _recompute_streamlines(context, obj)
        except Exception as e:
            self.report({"ERROR"}, f"Streamline tracing failed: {e}")
            return {"CANCELLED"}

        if n_lines == 0:
            self.report(
                {"WARNING"}, "No streamlines generated — try more seeds or larger step"
            )
            return {"CANCELLED"}

        self.report({"INFO"}, f"Traced {n_lines} field lines from {obj.name}")
        return {"FINISHED"}


# ===========================================================================
# Payload Field (D=3 RGB) visualization
# ===========================================================================
def _write_rgb_payload_vdb(voxels, shape, min_b, max_b, filepath):
    """Write an RGB payload volume to `filepath` as a VDB.

    voxels: (Nx, Ny, Nz, 3) float32 in [0,1]. Also writes a `density` grid
    equal to the vector magnitude (used for opacity in the shader).
    """
    res_arr = np.array(shape, dtype=np.float64)
    dx = (max_b - min_b) / np.maximum(res_arr - 1.0, 1.0)
    transform = openvdb.createLinearTransform(
        matrix=[
            [dx[0], 0.0, 0.0, 0.0],
            [0.0, dx[1], 0.0, 0.0],
            [0.0, 0.0, dx[2], 0.0],
            [min_b[0], min_b[1], min_b[2], 1.0],
        ]
    )

    clamped = np.clip(voxels, 0.0, 1.0)
    magnitude = np.linalg.norm(clamped, axis=-1)

    grids = []
    for c, name in enumerate(("payload_r", "payload_g", "payload_b")):
        g = openvdb.FloatGrid()
        g.copyFromArray(clamped[..., c].astype(np.float64))
        g.name = name
        g.transform = transform
        grids.append(g)

    gd = openvdb.FloatGrid()
    gd.copyFromArray(magnitude.astype(np.float64))
    gd.name = "density"
    gd.transform = transform
    grids.append(gd)

    openvdb.write(filepath, grids=grids)


def _get_or_create_payload_vol(context, obj):
    name = f"PayloadVol_{obj.name}"
    vol_obj = bpy.data.objects.get(name)
    if vol_obj is None:
        vol_data = bpy.data.volumes.new(f"PayloadField_{obj.name}")
        vol_data.materials.append(get_or_create_payload_volume_material())
        vol_obj = bpy.data.objects.new(name, vol_data)
        context.collection.objects.link(vol_obj)
    else:
        if not vol_obj.data.materials:
            vol_obj.data.materials.append(get_or_create_payload_volume_material())
    return vol_obj


def _get_or_create_payload_grad_vol(context, obj):
    name = f"PayloadGradVol_{obj.name}"
    vol_obj = bpy.data.objects.get(name)
    if vol_obj is None:
        vol_data = bpy.data.volumes.new(f"PayloadGradField_{obj.name}")
        vol_data.materials.append(get_or_create_payload_volume_material())
        vol_obj = bpy.data.objects.new(name, vol_data)
        context.collection.objects.link(vol_obj)
    else:
        if not vol_obj.data.materials:
            vol_obj.data.materials.append(get_or_create_payload_volume_material())
    return vol_obj


def _swap_vdb(vol_obj, new_filepath):
    old = vol_obj.data.filepath
    vol_obj.data.filepath = new_filepath
    vol_obj.data.update_tag()
    vol_obj.update_tag()
    if old and old != new_filepath and os.path.exists(old):
        try:
            os.remove(old)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Payload vertex-color material (EEVEE viewport)
# ---------------------------------------------------------------------------
def _get_or_create_payload_color_material(color_attr_name):
    mat_name = "WinderPayloadColors"
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()

        out = nodes.new("ShaderNodeOutputMaterial")
        out.location = (300, 0)
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.location = (0, 0)
        bsdf.inputs["Roughness"].default_value = 0.6

        attr = nodes.new("ShaderNodeAttribute")
        attr.location = (-300, 0)
        attr.attribute_type = "GEOMETRY"
        attr.attribute_name = color_attr_name

        links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
        links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    else:
        for node in mat.node_tree.nodes:
            if node.type == "ATTRIBUTE" and node.attribute_type == "GEOMETRY":
                node.attribute_name = color_attr_name
                break

    return mat


def apply_payload_color_material(obj):
    """Apply a vertex-color shader so payload colors are visible in EEVEE.

    Non-destructive: the previous material in slot 0 is stored on the object
    and restored by `remove_payload_color_material`.
    """
    mesh = obj.data
    if not mesh.color_attributes:
        return False

    active_color = mesh.color_attributes.active_color
    if active_color is None:
        return False

    mat = _get_or_create_payload_color_material(active_color.name)

    if not mesh.materials:
        mesh.materials.append(mat)
    else:
        if mesh.materials[0] != mat:
            if not obj.get("_winder_payload_prev_material"):
                obj["_winder_payload_prev_material"] = (
                    mesh.materials[0].name if mesh.materials[0] else ""
                )
            mesh.materials[0] = mat

    obj["_winder_payload_colors_active"] = True
    return True


def remove_payload_color_material(obj):
    """Restore the material that was in slot 0 before payload colors were shown."""
    mesh = obj.data
    if not obj.get("_winder_payload_colors_active"):
        return False

    our_mat = bpy.data.materials.get("WinderPayloadColors")

    if mesh.materials and our_mat and mesh.materials[0] == our_mat:
        prev_name = obj.get("_winder_payload_prev_material", "")
        prev_mat = bpy.data.materials.get(prev_name) if prev_name else None

        if prev_mat is not None:
            mesh.materials[0] = prev_mat
        else:
            mesh.materials.pop(index=0)

    obj["_winder_payload_colors_active"] = False
    if "_winder_payload_prev_material" in obj:
        del obj["_winder_payload_prev_material"]

    return True


def _extract_padded_payload(obj, mode, pattern="checkerboard", num_cells=None):
    """Read vertex colors as per-triangle RGB, auto-creating them if missing.

    Pads to 3 channels if the underlying attribute has fewer.
    """
    ensure_vertex_colors(obj, pattern=pattern, num_cells=num_cells)
    payload = extract_payload_from_vertex_colors(obj, mode=mode)
    if len(payload) == 0:
        raise ValueError(f"No triangles in '{obj.name}'")

    if payload.shape[1] < 3:
        padded = np.zeros((len(payload), 3), dtype=np.float32)
        padded[:, : payload.shape[1]] = payload
        payload = padded

    return payload


def recompute_payload_field(context, obj):
    """Forward payload RGB volume: W(x) = Σ_i w_i(x) * v_i, v_i from vertex colors."""
    props = context.scene.winder_props
    res = props.query_res
    padding = props.grid_padding
    use_fast_engine = props.use_fast_engine

    payload = _extract_padded_payload(
        obj,
        props.geometry_mode,
        pattern=props.payload_color_pattern,
        num_cells=props.payload_color_cells,
    )
    num_geom = len(payload)

    vol_obj = _get_or_create_payload_vol(context, obj)

    with CudaStream() as stream:
        queries, shape, (min_b, max_b) = get_grid_queries(obj, res=res, padding=padding)
        queries = queries.reshape([-1, 3])
        num_q = len(queries)

        q_buf = CudaBuffer(queries.shape, dtype=np.float32)
        pl_buf = CudaBuffer((num_geom, 3), dtype=np.float32)
        out_buf = CudaBuffer((num_q, 3), dtype=np.float32)

        q_buf.copy_from_numpy_async(queries, stream=stream)
        pl_buf.copy_from_numpy_async(payload, stream=stream)

        geom_data = extract_geometry_data(obj, props.geometry_mode, stream=stream)
        stream.keep_alive(geom_data)

        if geom_data["count"] != num_geom:
            raise RuntimeError(
                f"Payload/geometry mismatch: {num_geom} payload entries vs "
                f"{geom_data['count']} geometry elements. "
                "Re-extract vertex colors or switch geometry mode."
            )

        stream.keep_alive(
            compute_payload_field(
                geom_data,
                pl_buf,
                q_buf,
                out_buf,
                epsilon=props.epsilon,
                stream=stream,
                use_fast_engine=use_fast_engine,
                use_kahan=True,
            )
        )

        host_w = np.empty((num_q, 3), dtype=np.float32)
        out_buf.copy_to_numpy_async(host_w, stream=stream)

    voxels = host_w.reshape(shape + (3,))
    tmp_vdb = os.path.join(
        tempfile.gettempdir(),
        f"payload_{obj.name}_{uuid.uuid4().hex[:6]}.vdb",
    )
    _write_rgb_payload_vdb(voxels, shape, min_b, max_b, tmp_vdb)
    _swap_vdb(vol_obj, tmp_vdb)

    vol_obj["winder_payload_source"] = obj.name

    if hasattr(context, "screen") and context.screen:
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def recompute_payload_query_gradients(context, obj):
    """Per-channel query-gradient magnitudes as an RGB volume.

    Red = |∂W_R/∂x|, Green = |∂W_G/∂x|, Blue = |∂W_B/∂x|.
    """
    props = context.scene.winder_props
    res = props.query_res
    padding = props.grid_padding
    use_fast_engine = props.use_fast_engine

    payload = _extract_padded_payload(
        obj,
        props.geometry_mode,
        pattern=props.payload_color_pattern,
        num_cells=props.payload_color_cells,
    )
    num_geom = len(payload)

    vol_obj = _get_or_create_payload_grad_vol(context, obj)

    with CudaStream() as stream:
        queries, shape, (min_b, max_b) = get_grid_queries(obj, res=res, padding=padding)
        queries = queries.reshape([-1, 3])
        num_q = len(queries)

        q_buf = CudaBuffer(queries.shape, dtype=np.float32)
        pl_buf = CudaBuffer((num_geom, 3), dtype=np.float32)
        out_buf = CudaBuffer((num_q, 3), dtype=np.float32)

        q_buf.copy_from_numpy_async(queries, stream=stream)
        pl_buf.copy_from_numpy_async(payload, stream=stream)

        geom_data = extract_geometry_data(obj, props.geometry_mode, stream=stream)
        stream.keep_alive(geom_data)

        if geom_data["count"] != num_geom:
            raise RuntimeError(
                f"Payload/geometry mismatch: {num_geom} payload entries vs "
                f"{geom_data['count']} geometry elements."
            )

        stream.keep_alive(
            compute_payload_query_gradients_rgb(
                geom_data,
                pl_buf,
                q_buf,
                out_buf,
                epsilon=props.epsilon,
                stream=stream,
                use_fast_engine=use_fast_engine,
                use_kahan=True,
            )
        )

        host_grad = np.empty((num_q, 3), dtype=np.float32)
        out_buf.copy_to_numpy_async(host_grad, stream=stream)

    voxels = host_grad.reshape(shape + (3,))
    # Log1p scaling: raw query-gradient magnitudes span many orders of
    # magnitude (roughly 1e-8 to 1e+0 for typical scenes), so naive
    # per-channel normalization leaves almost every voxel visually
    # empty. Log1p maps that range into a more evenly distributed space,
    # so low-magnitude regions become visible. Apply *before* normalizing
    # so the log has full range to work with.
    for c in range(3):
        voxels[..., c] = np.log1p(voxels[..., c])
        vmax = float(voxels[..., c].max())
        if vmax > 1e-12:
            voxels[..., c] /= vmax

    tmp_vdb = os.path.join(
        tempfile.gettempdir(),
        f"payload_grad_{obj.name}_{uuid.uuid4().hex[:6]}.vdb",
    )
    _write_rgb_payload_vdb(voxels, shape, min_b, max_b, tmp_vdb)
    _swap_vdb(vol_obj, tmp_vdb)

    vol_obj["winder_payload_source"] = obj.name

    if hasattr(context, "screen") and context.screen:
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def recompute_payload_query_streamlines(context, obj):
    """Field lines of the summed payload query-gradient Σ_c ∂W_c/∂x."""
    props = context.scene.winder_props
    use_fast_engine = props.use_fast_engine

    payload = _extract_padded_payload(
        obj,
        props.geometry_mode,
        pattern=props.payload_color_pattern,
        num_cells=props.payload_color_cells,
    )

    bbox = np.array(
        [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box],
        dtype=np.float32,
    )
    center = (bbox.min(axis=0) + bbox.max(axis=0)) * 0.5
    diag = float(np.linalg.norm(bbox.max(axis=0) - bbox.min(axis=0)))
    if diag < 1e-6:
        diag = 1.0

    pad = (props.grid_padding + 0.3) * diag
    half = (bbox.max(axis=0) - bbox.min(axis=0)) * 0.5 + pad
    bounds_min = (center - half).astype(np.float32)
    bounds_max = (center + half).astype(np.float32)

    step_size = float(props.streamline_step) * diag
    max_length = float(props.streamline_max_length) * diag

    with CudaStream() as stream:
        geom_data = extract_geometry_data(obj, props.geometry_mode, stream=stream)
        stream.keep_alive(geom_data)

        if geom_data["count"] != len(payload):
            raise RuntimeError(
                f"Payload/geometry mismatch: {len(payload)} payload entries vs "
                f"{geom_data['count']} geometry elements."
            )

        payload_buf = CudaBuffer(payload.shape, dtype=np.float32)
        payload_buf.copy_from_numpy_async(payload, stream=stream)

        rng = np.random.default_rng(42)
        n_seeds = int(props.streamline_count)
        mode = props.geometry_mode
        jitter_scale = float(props.streamline_seed_jitter) * diag

        if mode == "PointNormal":
            pts = np.empty((geom_data["count"], 3), np.float32)
            geom_data["points"].copy_to_numpy_async(pts, stream=stream)
            stream.synchronize()
            seeds = _seed_on_point_normal(pts, n_seeds, rng, jitter_scale)
        elif mode == "Triangle":
            tris = np.empty((geom_data["count"], 3, 3), np.float32)
            geom_data["triangles"].copy_to_numpy_async(tris, stream=stream)
            stream.synchronize()
            seeds = _seed_on_triangles(tris, n_seeds, rng, jitter_scale)
        else:
            verts = np.empty((geom_data["num_verts"], 3), np.float32)
            idxs = np.empty((geom_data["num_tris"], 3), np.uint32)
            geom_data["vertices"].copy_to_numpy_async(verts, stream=stream)
            geom_data["triangle_indices"].copy_to_numpy_async(idxs, stream=stream)
            stream.synchronize()
            seeds = _seed_on_triangles(verts[idxs], n_seeds, rng, jitter_scale)

        line_points, line_mags, line_counts = trace_payload_query_streamlines(
            geom_data,
            payload_buf,
            seeds,
            step_size,
            int(props.streamline_max_steps),
            max_length,
            bounds_min,
            bounds_max,
            float(props.streamline_min_grad),
            float(props.epsilon),
            -1.0,
            bool(props.streamline_bidirectional),
            stream,
            use_fast_engine=use_fast_engine,
            use_kahan=True,
        )

    num_lines = int(np.sum(line_counts >= 2))

    obj_name = f"PayloadStreamlines_{obj.name}"
    streamline_obj = bpy.data.objects.get(obj_name)

    if streamline_obj is None:
        mesh = build_streamline_mesh(
            line_points, line_mags, line_counts, f"{obj_name}_Mesh"
        )
        streamline_obj = bpy.data.objects.new(obj_name, mesh)
        context.collection.objects.link(streamline_obj)
        streamline_obj.matrix_world = mathutils.Matrix.Identity(4)
        _ensure_streamline_modifier(streamline_obj, props, diag)
    else:
        if streamline_obj.data is None:
            streamline_obj.data = bpy.data.meshes.new(f"{obj_name}_Mesh")
        update_mesh_from_lines(streamline_obj.data, line_points, line_mags, line_counts)
        _ensure_streamline_modifier(streamline_obj, props, diag)

    return num_lines


# ---------------------------------------------------------------------------
# Payload operators
# ---------------------------------------------------------------------------
class WM_OT_create_payload_field(bpy.types.Operator):
    bl_idname = "winder.create_payload_field"
    bl_label = "Create Payload Field (RGB)"
    bl_description = (
        "Read the active color attribute from the mesh as per-triangle payload, "
        "compute the forward payload field, and visualize as an RGB volume. "
        "Auto-generates random vertex colors if the mesh has none"
    )

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != "MESH":
            self.report({"ERROR"}, "Select a Mesh Object")
            return {"CANCELLED"}

        try:
            recompute_payload_field(context, obj)
        except Exception as e:
            self.report({"ERROR"}, f"Payload field failed: {e}")
            return {"CANCELLED"}

        # Auto-show the payload colors on the source mesh so they are visible
        # while painting / iterating
        try:
            apply_payload_color_material(obj)
        except Exception as e:
            print(f"[Winder] Could not apply payload color material: {e}")

        vol_obj = bpy.data.objects.get(f"PayloadVol_{obj.name}")
        if vol_obj:
            bpy.ops.object.select_all(action="DESELECT")
            vol_obj.select_set(True)
            context.view_layer.objects.active = vol_obj

        self.report({"INFO"}, f"Created RGB Payload Field for {obj.name}")
        return {"FINISHED"}


class WM_OT_create_payload_query_streamlines(bpy.types.Operator):
    bl_idname = "winder.create_payload_query_streamlines"
    bl_label = "Create Payload Field Lines"
    bl_description = (
        "Trace field lines of the summed payload query-gradient Σ_c ∂W_c/∂x "
        "from seeds sampled on the surface. This is the field-line analogue "
        "of the winding-field streamlines, extended to the D=3 payload case. "
        "Auto-generates random vertex colors if the mesh has none"
    )

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != "MESH":
            self.report({"ERROR"}, "Select a Mesh Object")
            return {"CANCELLED"}

        try:
            n_lines = recompute_payload_query_streamlines(context, obj)
        except NotImplementedError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        except Exception as e:
            self.report({"ERROR"}, f"Payload field-line tracing failed: {e}")
            return {"CANCELLED"}

        if n_lines == 0:
            self.report(
                {"WARNING"},
                "No payload field lines generated — try more seeds or a "
                "larger step size",
            )
            return {"CANCELLED"}

        streamline_obj = bpy.data.objects.get(f"PayloadStreamlines_{obj.name}")
        if streamline_obj:
            bpy.ops.object.select_all(action="DESELECT")
            streamline_obj.select_set(True)
            context.view_layer.objects.active = streamline_obj

        self.report({"INFO"}, f"Traced {n_lines} payload field lines from {obj.name}")
        return {"FINISHED"}


class WM_OT_create_payload_query_gradients(bpy.types.Operator):
    bl_idname = "winder.create_payload_query_gradients"
    bl_label = "Create Payload Query Gradients (RGB)"
    bl_description = (
        "Compute per-channel query-gradient magnitudes of the payload field "
        "and visualize as an RGB volume. Auto-generates random vertex colors "
        "if the mesh has none"
    )

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != "MESH":
            self.report({"ERROR"}, "Select a Mesh Object")
            return {"CANCELLED"}

        try:
            recompute_payload_query_gradients(context, obj)
        except NotImplementedError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        except Exception as e:
            self.report({"ERROR"}, f"Payload query gradients failed: {e}")
            return {"CANCELLED"}

        vol_obj = bpy.data.objects.get(f"PayloadGradVol_{obj.name}")
        if vol_obj:
            bpy.ops.object.select_all(action="DESELECT")
            vol_obj.select_set(True)
            context.view_layer.objects.active = vol_obj

        self.report(
            {"INFO"},
            f"Created Payload Query-Gradient RGB volume for {obj.name}",
        )
        return {"FINISHED"}


class WM_OT_toggle_payload_colors(bpy.types.Operator):
    bl_idname = "winder.toggle_payload_colors"
    bl_label = "Toggle Payload Vertex Colors"
    bl_description = (
        "Show or hide the per-vertex payload colors on the active mesh in the "
        "viewport (EEVEE). Auto-creates random colors if the mesh has none"
    )

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != "MESH":
            self.report({"ERROR"}, "Select a Mesh Object")
            return {"CANCELLED"}

        if obj.get("_winder_payload_colors_active", False):
            remove_payload_color_material(obj)
            self.report({"INFO"}, f"Removed payload color material from {obj.name}")
        else:
            ensure_vertex_colors(obj)
            if not apply_payload_color_material(obj):
                self.report({"ERROR"}, "Failed to apply payload color material")
                return {"CANCELLED"}
            self.report({"INFO"}, f"Applied payload color material to {obj.name}")

        if hasattr(context, "screen") and context.screen:
            for area in context.screen.areas:
                if area.type == "VIEW_3D":
                    area.tag_redraw()

        return {"FINISHED"}


def update_payload_color_pattern(self, context):
    """Regenerate colors on meshes with an active payload visualization."""
    pattern = self.payload_color_pattern
    cells = int(self.payload_color_cells)

    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue

        name = obj.name
        is_active = bool(obj.get("_winder_payload_colors_active", False))
        has_vol = bpy.data.objects.get(f"PayloadVol_{name}") is not None
        has_grad_vol = bpy.data.objects.get(f"PayloadGradVol_{name}") is not None
        has_stream = bpy.data.objects.get(f"PayloadStreamlines_{name}") is not None

        if not (is_active or has_vol or has_grad_vol or has_stream):
            continue

        mesh = obj.data
        if "PayloadColor" in mesh.color_attributes:
            mesh.color_attributes.remove(mesh.color_attributes["PayloadColor"])

        try:
            ensure_vertex_colors(obj, pattern=pattern, num_cells=cells)
        except Exception as e:
            print(f"[Winder] Color regen failed for {name}: {e}")
            continue

        if has_vol:
            try:
                recompute_payload_field(context, obj)
            except Exception as e:
                print(f"[Winder] Payload field regen failed for {name}: {e}")
        if has_grad_vol:
            try:
                recompute_payload_query_gradients(context, obj)
            except Exception as e:
                print(f"[Winder] Payload grad regen failed for {name}: {e}")
        if has_stream:
            try:
                recompute_payload_query_streamlines(context, obj)
            except Exception as e:
                print(f"[Winder] Payload field lines regen failed for {name}: {e}")

    if hasattr(context, "screen") and context.screen:
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


class WM_OT_regenerate_payload_colors(bpy.types.Operator):
    bl_idname = "winder.regenerate_payload_colors"
    bl_label = "Regenerate Payload Colors"
    bl_description = (
        "Replace the existing 'PayloadColor' attribute on the active mesh with "
        "the currently selected pattern. Existing payload volumes are "
        "recomputed to reflect the new colors"
    )

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != "MESH":
            self.report({"ERROR"}, "Select a Mesh Object")
            return {"CANCELLED"}

        props = context.scene.winder_props
        mesh = obj.data
        if "PayloadColor" in mesh.color_attributes:
            mesh.color_attributes.remove(mesh.color_attributes["PayloadColor"])

        try:
            ensure_vertex_colors(
                obj,
                pattern=props.payload_color_pattern,
                num_cells=props.payload_color_cells,
            )
        except Exception as e:
            self.report({"ERROR"}, f"Failed to generate colors: {e}")
            return {"CANCELLED"}

        # Refresh any existing payload visualizations for this object
        name = obj.name
        for existing, fn in [
            (f"PayloadVol_{name}", recompute_payload_field),
            (f"PayloadGradVol_{name}", recompute_payload_query_gradients),
            (f"PayloadStreamlines_{name}", recompute_payload_query_streamlines),
        ]:
            if bpy.data.objects.get(existing):
                try:
                    fn(context, obj)
                except Exception as e:
                    print(f"[Winder] Recompute after color regen failed: {e}")

        self.report(
            {"INFO"},
            f"Regenerated '{props.payload_color_pattern}' colors on {obj.name}",
        )
        return {"FINISHED"}


_ISO_PENDING: dict[str, float] = {}
_ISO_TIMER_SCHEDULED = False


def _bake_iso_mesh(context, src_obj, vol_obj, threshold):
    """Evaluate the isosurface at `threshold` and return a colored mesh.

    Returns None if no geometry was produced (threshold outside the field).
    """
    props = context.scene.winder_props
    scratch_mesh = bpy.data.meshes.new(f"_scratch_iso_{src_obj.name}")
    scratch_obj = bpy.data.objects.new(f"_scratch_iso_{src_obj.name}", scratch_mesh)
    context.scene.collection.objects.link(scratch_obj)

    try:
        gn = _build_iso_gn_tree(vol_obj)

        # Set the threshold on the interface BEFORE adding the modifier.
        # The modifier reads the interface default at creation time, so this
        # order matters — setting it afterwards can be ignored depending on
        # the node group cache.
        for item in gn.interface.items_tree:
            if (
                item.item_type == "SOCKET"
                and getattr(item, "in_out", "") == "INPUT"
                and item.name == "Threshold"
            ):
                item.default_value = float(threshold)
                item.min_value = 0.0
                item.max_value = 1.0
                break

        mod = scratch_obj.modifiers.new("GNIso", "NODES")
        mod.node_group = gn

        depsgraph = context.evaluated_depsgraph_get()
        depsgraph.update()
        scratch_eval = scratch_obj.evaluated_get(depsgraph)
        baked_mesh = bpy.data.meshes.new_from_object(
            scratch_eval, preserve_all_data_layers=True, depsgraph=depsgraph
        )
    finally:
        bpy.data.objects.remove(scratch_obj, do_unlink=True)
        if scratch_mesh.users == 0:
            bpy.data.meshes.remove(scratch_mesh)

    n_verts = len(baked_mesh.vertices)
    print(f"[Winder] Iso bake: threshold={threshold:.4f} -> {n_verts} verts")
    if n_verts == 0:
        bpy.data.meshes.remove(baked_mesh)
        return None

    # ---- Query winder.PayloadField at the iso vertices ----
    local_verts = np.empty((n_verts, 3), dtype=np.float32)
    baked_mesh.vertices.foreach_get("co", local_verts.ravel())

    with CudaStream() as stream:
        payload = _extract_padded_payload(src_obj, props.geometry_mode)
        if len(payload) == 0:
            raise RuntimeError("Empty payload")
        geom_data = extract_geometry_data(src_obj, props.geometry_mode, stream=stream)
        stream.keep_alive(geom_data)

        pl_buf = CudaBuffer(payload.shape, dtype=np.float32)
        q_buf = CudaBuffer(local_verts.shape, dtype=np.float32)
        out_buf = CudaBuffer((n_verts, 3), dtype=np.float32)

        pl_buf.copy_from_numpy_async(payload, stream=stream)
        q_buf.copy_from_numpy_async(local_verts, stream=stream)

        stream.keep_alive(
            compute_payload_field(
                geom_data,
                pl_buf,
                q_buf,
                out_buf,
                epsilon=props.epsilon,
                stream=stream,
                use_fast_engine=props.use_fast_engine,
                use_kahan=True,
            )
        )

        colors = np.empty((n_verts, 3), dtype=np.float32)
        out_buf.copy_to_numpy_async(colors, stream=stream)

    # ---- Store as PayloadColor POINT attribute ----
    if "PayloadColor" in baked_mesh.attributes:
        baked_mesh.attributes.remove(baked_mesh.attributes["PayloadColor"])
    attr = baked_mesh.attributes.new(
        name="PayloadColor", type="FLOAT_COLOR", domain="POINT"
    )
    rgba = np.hstack([colors, np.ones((n_verts, 1), dtype=np.float32)])
    attr.data.foreach_set("color", rgba.ravel())

    baked_mesh.update()
    return baked_mesh


def _refresh_iso(context, src_obj, threshold, force_recreate=False):
    """Rebake the isosurface object for `src_obj` at `threshold`."""
    vol_obj = bpy.data.objects.get(f"PayloadVol_{src_obj.name}")
    if vol_obj is None:
        return None

    baked = _bake_iso_mesh(context, src_obj, vol_obj, threshold)
    if baked is None:
        return None

    iso_name = f"PayloadIsosurface_{src_obj.name}"
    iso_obj = bpy.data.objects.get(iso_name)

    if iso_obj is None or force_recreate:
        if iso_obj is not None:
            old_mesh = iso_obj.data
            bpy.data.objects.remove(iso_obj, do_unlink=True)
            if old_mesh and old_mesh.users == 0:
                bpy.data.meshes.remove(old_mesh)
        iso_obj = bpy.data.objects.new(iso_name, baked)
        context.scene.collection.objects.link(iso_obj)
        iso_obj.matrix_world = mathutils.Matrix.Identity(4)
    else:
        old_mesh = iso_obj.data
        iso_obj.data = baked
        if old_mesh and old_mesh.users == 0:
            bpy.data.meshes.remove(old_mesh)

    # IMPORTANT: new_from_object copies empty material slots from the scratch
    # object. Clear them and force our material into slot 0 so the shader
    # actually reads the PayloadColor attribute.
    mat = get_or_create_payload_isosurface_material()
    baked.materials.clear()
    baked.materials.append(mat)

    iso_obj["_winder_payload_iso_threshold"] = float(threshold)
    iso_obj.update_tag()
    if hasattr(context, "view_layer"):
        context.view_layer.update()
    return iso_obj


# ---- Slider callback: synchronous bake, cooldown to avoid dragging storms ----

_ISO_LAST_BAKE_TIME = [0.0]
_ISO_PENDING_BAKE = [None]  # (context, src_name, thresh) or None


def _flush_iso_pending():
    """Timer: run any pending bake once the cooldown has elapsed."""
    import time

    if _ISO_PENDING_BAKE[0] is None:
        return None

    now = time.time()
    if now - _ISO_LAST_BAKE_TIME[0] < 0.12:
        return 0.05  # re-check in 50 ms

    ctx, src_name, thresh = _ISO_PENDING_BAKE[0]
    _ISO_PENDING_BAKE[0] = None
    _ISO_LAST_BAKE_TIME[0] = now

    src = bpy.data.objects.get(src_name)
    if src is None:
        return None
    try:
        _refresh_iso(ctx, src, thresh)
        if hasattr(ctx, "screen") and ctx.screen:
            for area in ctx.screen.areas:
                if area.type == "VIEW_3D":
                    area.tag_redraw()
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"[Winder] Iso refresh failed for {src_name}: {e}")
    return None


def update_payload_isosurface(self, context):
    """Slider callback — debounced rebake of every iso object."""
    thresh = float(self.payload_isosurface_threshold)

    # Find the first source with an existing isosurface
    target = None
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if bpy.data.objects.get(f"PayloadIsosurface_{obj.name}"):
            target = obj.name
            break

    if target is None:
        return

    _ISO_PENDING_BAKE[0] = (context, target, thresh)
    if not bpy.app.timers.is_registered(_flush_iso_pending):
        bpy.app.timers.register(_flush_iso_pending, first_interval=0.12)


class WM_OT_create_payload_isosurface(bpy.types.Operator):
    bl_idname = "winder.create_payload_isosurface"
    bl_label = "Create Payload Isosurface"
    bl_description = (
        "Extract the isosurface of the payload volume at the given threshold "
        "and color each vertex by querying winder.PayloadField at that vertex"
    )

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != "MESH":
            self.report({"ERROR"}, "Select a Mesh Object")
            return {"CANCELLED"}
        if bpy.data.objects.get(f"PayloadVol_{obj.name}") is None:
            self.report(
                {"ERROR"},
                f"No payload volume for {obj.name}. Create one first.",
            )
            return {"CANCELLED"}

        props = context.scene.winder_props
        thresh = float(props.payload_isosurface_threshold)
        try:
            iso_obj = _refresh_iso(context, obj, thresh, force_recreate=True)
        except Exception as e:
            import traceback

            traceback.print_exc()
            self.report({"ERROR"}, f"Isosurface failed: {e}")
            return {"CANCELLED"}

        if iso_obj is None:
            self.report({"ERROR"}, "Isosurface produced no geometry")
            return {"CANCELLED"}

        bpy.ops.object.select_all(action="DESELECT")
        iso_obj.select_set(True)
        context.view_layer.objects.active = iso_obj
        self.report(
            {"INFO"},
            f"Isosurface for {obj.name} (threshold={thresh:.3f})",
        )
        return {"FINISHED"}


class WM_OT_create_payload_geometry_gradients(bpy.types.Operator):
    bl_idname = "winder.create_payload_geometry_gradients"
    bl_label = "Create Payload Geometry Gradient Quivers"
    bl_description = (
        "Visualize the gradient of sum(W) (total payload field, summed over "
        "all queries and channels) with respect to the source geometry as "
        "arrows at the primitive positions. Same layout as the winding-"
        "number geometry-gradient quivers. Requires the brute-force engine; "
        "the fast (BVH) path is not yet implemented."
    )

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != "MESH":
            self.report({"ERROR"}, "Select a Mesh Object")
            return {"CANCELLED"}

        try:
            n = _recompute_payload_geometry_gradients(context, obj)
        except NotImplementedError:
            self.report(
                {"ERROR"},
                "Fast (BVH) payload geometry gradients are not yet "
                "implemented. Uncheck 'Use Fast Engine' in the Winder panel "
                "to use the brute-force path. For large meshes, reduce "
                "Query Res and Query Grad Stride to keep it tractable.",
            )
            return {"CANCELLED"}
        except Exception as e:
            self.report({"ERROR"}, f"Payload geometry gradients failed: {e}")
            return {"CANCELLED"}

        if n == 0:
            self.report({"WARNING"}, "No arrows generated")
            return {"CANCELLED"}

        quiver_obj = bpy.data.objects.get(f"PayloadGeomQuiver_{obj.name}")
        if quiver_obj:
            bpy.ops.object.select_all(action="DESELECT")
            quiver_obj.select_set(True)
            context.view_layer.objects.active = quiver_obj

        props = context.scene.winder_props
        self.report(
            {"INFO"},
            f"Payload geometry gradients: {n} arrows "
            f"(res={props.query_res}, stride={props.query_grad_stride})",
        )
        return {"FINISHED"}
