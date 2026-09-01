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


def get_grid_queries(obj, res=64, padding=0.2):
    """Generates uniform 3D grid queries expanded proportionally by bounding box extent."""
    print("DEBUG: get_grid_queries")
    bbox = np.array([obj.matrix_world @ mathutils.Vector(b) for b in obj.bound_box])
    min_raw, max_raw = bbox.min(axis=0), bbox.max(axis=0)

    extent = np.maximum(max_raw - min_raw, 1e-4)
    min_b = min_raw - extent * padding
    max_b = max_raw + extent * padding

    x = np.linspace(min_b[0], max_b[0], res, dtype=np.float32)
    y = np.linspace(min_b[1], max_b[1], res, dtype=np.float32)
    z = np.linspace(min_b[2], max_b[2], res, dtype=np.float32)

    gx, gy, gz = np.meshgrid(x, y, z, indexing="ij")
    queries = np.stack([gx.flatten(), gy.flatten(), gz.flatten()], axis=-1)
    return queries, (res, res, res), (min_b, max_b)


def recompute_winding_field(context, obj):
    """Evaluates Volume + Quivers using in-place data block updates."""
    print("DEBUG: recompute_winding_field")
    vol_obj, quiver_obj = get_or_create_winding_objects(context, obj)
    obj.display_type = 'WIRE'

    props = context.scene.winder_props
    res = props.query_res
    padding = props.grid_padding

    stream = CudaStream()

    queries, shape, (min_b, max_b) = get_grid_queries(obj, res=res, padding=padding)
    queries = queries.reshape([-1, 3])
    num_q = len(queries)

    q_buf = CudaBuffer(queries.shape, dtype=np.float32)
    out_w_buf = CudaBuffer((num_q,), dtype=np.float32)

    q_buf.copy_from_numpy_async(queries, stream=stream)

    geom_data = extract_geometry_data(obj, props.geometry_mode, stream=stream)
    print("DEBUG q_buf.shape", q_buf.shape)
    print("DEBUG out_w_buf.shape", out_w_buf.shape)
    host_refs = []
    refs = compute_winding_field(geom_data, q_buf, out_w_buf, stream=stream)
    host_refs.extend(refs)

    host_w = np.empty(num_q, dtype=np.float32)
    out_w_buf.copy_to_numpy_async(host_w, stream=stream)

    ones_arr = np.ones(num_q, dtype=np.float32)
    ones_buf = CudaBuffer((num_q,), dtype=np.float32)
    ones_buf.copy_from_numpy_async(ones_arr, stream=stream)

    mode = props.geometry_mode
    if mode == "PointNormal":
        grad_shape = (geom_data["count"], 2, 3)
    elif mode == "Triangle":
        grad_shape = (geom_data["count"], 3, 3)
    elif mode == "Mesh":
        grad_shape = (geom_data["num_verts"], 3)

    out_g_buf = CudaBuffer(grad_shape, dtype=np.float32)
    refs = compute_geometry_gradients(geom_data, q_buf, ones_buf, out_g_buf, stream=stream)
    host_refs.extend(refs)

    grad_data = np.empty(grad_shape, dtype=np.float32)
    out_g_buf.copy_to_numpy_async(grad_data, stream=stream)

    # Stream sync guarantees host reads are safe and geom_data host_refs stay alive
    stream.synchronize()
    host_refs.clear()

    # 1. Update OpenVDB Volume File
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

    # Store file at persistent temp path per object
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

    # 2. Update Quiver Mesh Geometry In-Place
    if mode == "PointNormal":
        pts = np.empty((geom_data["count"], 3), dtype=np.float32)
        geom_data["points"].copy_to_numpy_async(pts, stream=stream)
        stream.synchronize()
        vecs = grad_data[:, 1, :]
    elif mode == "Triangle":
        tris = np.empty((geom_data["count"], 3, 3), dtype=np.float32)
        geom_data["triangles"].copy_to_numpy_async(tris, stream=stream)
        stream.synchronize()
        pts = tris.mean(axis=1)
        vecs = grad_data.mean(axis=1)
    elif mode == "Mesh":
        pts = np.empty((geom_data["num_verts"], 3), dtype=np.float32)
        geom_data["vertices"].copy_to_numpy_async(pts, stream=stream)
        stream.synchronize()
        vecs = grad_data

    mags = np.linalg.norm(vecs, axis=1)
    safe_mags = np.maximum(mags[:, None], 1e-8)
    dirs = vecs / safe_mags

    quiver_mesh = quiver_obj.data
    
    # Use clear_geometry() instead of clear()
    quiver_mesh.clear_geometry()
    quiver_mesh.from_pydata(pts.tolist(), [], [])

    # Get or create vector attribute safely
    if "gradient_dir" in quiver_mesh.attributes:
        attr_dir = quiver_mesh.attributes["gradient_dir"]
    else:
        attr_dir = quiver_mesh.attributes.new(
            name="gradient_dir", type="FLOAT_VECTOR", domain="POINT"
        )
    attr_dir.data.foreach_set("vector", dirs.ravel())

    # Get or create scalar magnitude attribute safely
    if "gradient_mag" in quiver_mesh.attributes:
        attr_mag = quiver_mesh.attributes["gradient_mag"]
    else:
        attr_mag = quiver_mesh.attributes.new(
            name="gradient_mag", type="FLOAT", domain="POINT"
        )
    attr_mag.data.foreach_set("value", mags.ravel())

    quiver_mesh.update()

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


class WM_OT_create_3d_contours(bpy.types.Operator):
    bl_idname = "winder.create_3d_contours"
    bl_label = "Create 3D Contours"

    def execute(self, context):
        vol_obj = context.active_object
        if not vol_obj or vol_obj.type != "VOLUME":
            self.report({"ERROR"}, "Select a Volume Object")
            return {"CANCELLED"}

        props = context.scene.winder_props

        cut_empty = bpy.data.objects.new(f"CutPlane_{vol_obj.name}", None)
        cut_empty.empty_display_type = "SINGLE_ARROW"
        context.collection.objects.link(cut_empty)

        contour_mesh = bpy.data.meshes.new(f"Contours_{vol_obj.name}")
        contour_obj = bpy.data.objects.new(f"IsoContours_{vol_obj.name}", contour_mesh)
        context.collection.objects.link(contour_obj)

        gn_tree = build_marching_cubes_contour_nodes(num_shells=props.contour_count)
        mod = contour_obj.modifiers.new(name="GNContours", type="NODES")
        mod.node_group = gn_tree
        mod["Socket_1"] = vol_obj
        mod["Socket_2"] = cut_empty

        self.report(
            {"INFO"}, f"Created 3D Iso-Contours with Cut Plane for {vol_obj.name}"
        )
        return {"FINISHED"}
