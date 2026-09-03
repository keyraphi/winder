import os
import uuid
import tempfile
import openvdb
import numpy as np
import mathutils
import bpy

from .dlpack_bridge import CudaBuffer, CudaStream
from .winder_wrapper import (
    extract_geometry_data,
    compute_winding_field,
    compute_geometry_gradients,
)
from .nodes_builder import (
    build_quiver_geometry_nodes,
    build_marching_cubes_contour_nodes,
    get_or_create_quiver_material,
    get_or_create_volume_material,
)
from contextlib import contextmanager


@contextmanager
def cuda_execution_scope(device_id: int = 0):
    stream = CudaStream(device_id)
    try:
        yield stream
    finally:
        # Guarantee all GPU operations on this stream complete
        stream.synchronize()


def get_or_create_winding_objects(context, obj):
    """Retrieves or instantiates persistent Volume and Quiver objects in place."""
    print("DEBUG: get_or_create_winding_objects")
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
    """Generates uniform 3D grid queries expanded proportionally by bounding box extent.

    Automatically handles 2D / flat objects (e.g. single triangles or planes)
    by enforcing a minimum thickness relative to the object's maximum dimension.
    """
    print("DEBUG: get_grid_queries")
    bbox = np.array([obj.matrix_world @ mathutils.Vector(b) for b in obj.bound_box])
    min_raw, max_raw = bbox.min(axis=0), bbox.max(axis=0)

    center = (min_raw + max_raw) * 0.5
    extent = max_raw - min_raw

    # Characteristic size of the object
    max_dim = np.max(extent)
    if max_dim < 1e-6:
        max_dim = 1.0

    # Clamp thin/flat dimensions so the grid captures 3D field falloff above & below
    min_extent = max_dim * min_aspect_ratio
    adj_extent = np.maximum(extent, min_extent)

    # Expand bounds symmetrically around center with padding
    padded_half_extent = (adj_extent * 0.5) * (1.0 + 2.0 * padding)

    min_b = center - padded_half_extent
    max_b = center + padded_half_extent

    x = np.linspace(min_b[0], max_b[0], res, dtype=np.float32)
    y = np.linspace(min_b[1], max_b[1], res, dtype=np.float32)
    z = np.linspace(min_b[2], max_b[2], res, dtype=np.float32)

    gx, gy, gz = np.meshgrid(x, y, z, indexing="ij")
    queries = np.stack([gx.flatten(), gy.flatten(), gz.flatten()], axis=-1)
    return queries, (res, res, res), (min_b, max_b)


def recompute_winding_field(context, obj):
    """Evaluates Volume + optional Quivers using in-place data block updates."""
    print("DEBUG: recompute_winding_field")
    props = context.scene.winder_props

    # Fetch or create Volume object (Quiver handling is conditional below)
    vol_obj, quiver_obj = get_or_create_winding_objects(context, obj)
    obj.display_type = "WIRE"

    res = props.query_res
    padding = props.grid_padding
    show_quivers = props.is_quiver_creation_active

    with cuda_execution_scope() as stream:

        # 1. Grid Queries & Winding Field Evaluation
        queries, shape, (min_b, max_b) = get_grid_queries(obj, res=res, padding=padding)
        queries = queries.reshape([-1, 3])
        num_q = len(queries)

        q_buf = CudaBuffer(queries.shape, dtype=np.float32)
        out_w_buf = CudaBuffer((num_q,), dtype=np.float32)

        q_buf.copy_from_numpy_async(queries, stream=stream)

        geom_data = extract_geometry_data(obj, props.geometry_mode, stream=stream)
        host_refs = []
        refs = compute_winding_field(geom_data, q_buf, out_w_buf, stream=stream)
        host_refs.extend(refs)

        host_w = np.empty(num_q, dtype=np.float32)
        out_w_buf.copy_to_numpy_async(host_w, stream=stream)

        # 2. Conditional Quiver / Gradient Computation
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
            refs = compute_geometry_gradients(
                geom_data, q_buf, ones_buf, out_g_buf, stream=stream
            )
            host_refs.extend(refs)

            grad_data = np.empty(grad_shape, dtype=np.float32)
            out_g_buf.copy_to_numpy_async(grad_data, stream=stream)

        # Stream sync guarantees host reads are safe and geom_data host_refs stay alive
        stream.synchronize()
        host_refs.clear()

    # 3. Update OpenVDB Volume File
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

    if old_vdb and os.path.exists(old_vdb) and old_vdb != temp_vdb:
        try:
            os.remove(old_vdb)
        except OSError:
            pass

    # 4. Update or Remove Quiver Geometry
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
        # If quivers are disabled, remove existing quiver object and mesh data
        quiver_name = f"WindingQuiver_{obj.name}"
        existing_quiver = bpy.data.objects.get(quiver_name)
        if existing_quiver:
            mesh_data = existing_quiver.data
            bpy.data.objects.remove(existing_quiver, do_unlink=True)
            if mesh_data and mesh_data.users == 0:
                bpy.data.meshes.remove(mesh_data)

    # 5. Store Metadata
    cdf_percentiles = np.linspace(0.0, 100.0, 1001)
    cdf_values = np.percentile(host_w, cdf_percentiles).astype(float).tolist()

    vol_obj["winder_quantile_cdf"] = cdf_values
    vol_obj["winder_source_object"] = obj.name

    # 6. Live-Sync IsoContours
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


class WM_OT_create_winding_field(bpy.types.Operator):
    bl_idname = "winder.create_winding_field"
    bl_label = "Create Winding Number Field"

    def execute(self, context):
        print("DEBUG: WM_OT_create_winding_field.execute")
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


class WM_OT_create_optimization(bpy.types.Operator):
    bl_idname = "winder.create_optimization"
    bl_label = "Create Field Optimization"

    def execute(self, context):
        selected = context.selected_objects
        if len(selected) != 2:
            self.report(
                {"ERROR"}, "Select exactly 2 objects (Target primary, Source secondary)"
            )
            return {"CANCELLED"}

        target = context.active_object
        source = [o for o in selected if o != target][0]

        context.scene.winder_opt_target = target
        context.scene.winder_opt_source = source
        context.scene.winder_is_optimizing = True

        self.report(
            {"INFO"}, f"Optimization paired: Target={target.name}, Source={source.name}"
        )
        return {"FINISHED"}


class WM_OT_create_loss_landscape(bpy.types.Operator):
    bl_idname = "winder.create_loss_landscape"
    bl_label = "Create Loss Landscape"

    def execute(self, context):
        self.report({"INFO"}, "Evaluated 3D Position Loss Landscape")
        return {"FINISHED"}


def get_contour_thresholds(
    vol_obj, num_shells, is_winding_field=True, p_min=2.0, p_max=98.0
):
    """Calculates iso-surface thresholds based on field mode:

    - Winding Field Mode: Linear spacing across [0.0, 1.0] (50% centered for 1 shell).
    - General Field Mode: Quantile percentiles derived from stored CDF metadata.
    """
    if is_winding_field:
        if num_shells == 1:
            return [0.5]  # Surface boundary
        # Linear spacing from 0.0 to 1.0 across requested shell count
        return np.linspace(0.0, 1.0, num_shells).tolist()
    else:
        cdf = vol_obj.get("winder_quantile_cdf")
        if not cdf:
            return None

        if num_shells == 1:
            ps = np.array([50.0])  # Median
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

        # Verify CDF metadata exists on the Volume object
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

        # Check if contour object already exists for this volume
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

        # Get or create Cut Plane Empty
        cut_empty = bpy.data.objects.get(cut_empty_name)
        if not cut_empty:
            cut_empty = bpy.data.objects.new(cut_empty_name, None)
            cut_empty.empty_display_type = "SINGLE_ARROW"
            context.collection.objects.link(cut_empty)

        # Create new IsoContours mesh object
        contour_mesh = bpy.data.meshes.new(contour_obj_name)
        contour_obj = bpy.data.objects.new(contour_obj_name, contour_mesh)
        context.collection.objects.link(contour_obj)

        # Build GN node tree using quantile thresholds
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
    """Callback triggered whenever contour_count or is_winding_field property changes."""
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


def update_all_winding_fields(self, context):
    """Callback triggered when geometry_mode, query_res, or grid_padding changes.

    Recomputes winding fields for all source mesh objects in the scene.
    """
    target_objs = set()

    # 1. Include scene's active target object if set
    if hasattr(context, "scene") and "winder_active_target" in context.scene:
        active_name = context.scene["winder_active_target"]
        active_obj = bpy.data.objects.get(active_name)
        if active_obj and active_obj.type == "MESH":
            target_objs.add(active_obj)

    # 2. Scan scene for all Volume objects generated by Winder
    for obj in bpy.data.objects:
        if obj.type == "VOLUME":
            # Check explicit source object reference first
            source_name = obj.get("winder_source_object")
            if source_name and source_name in bpy.data.objects:
                source_obj = bpy.data.objects[source_name]
                if source_obj.type == "MESH":
                    target_objs.add(source_obj)
                    continue

            # Fallback to standard naming conventions
            if "winder_quantile_cdf" in obj:
                for prefix in ("WindingVol_", "Winding_", "Vol_"):
                    if obj.name.startswith(prefix):
                        mesh_name = obj.name[len(prefix) :]
                        mesh_obj = bpy.data.objects.get(mesh_name)
                        if mesh_obj and mesh_obj.type == "MESH":
                            target_objs.add(mesh_obj)
                            break

    # 3. Trigger live recomputation for all matched meshes
    for mesh_obj in target_objs:
        try:
            recompute_winding_field(context, mesh_obj)
        except Exception as e:
            print(f"[Winder] Error updating winding field for '{mesh_obj.name}': {e}")

    if hasattr(context, "screen") and context.screen:
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()
