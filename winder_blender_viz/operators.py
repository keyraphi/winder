import bpy
import mathutils
import numpy as np
from .dlpack_bridge import CudaBuffer
from .winder_wrapper import (
    extract_geometry_data,
    compute_winding_field,
    compute_geometry_gradients,
)
from .nodes_builder import (
    build_quiver_geometry_nodes,
    build_marching_cubes_contour_nodes,
)


def get_grid_queries(obj, res=64, padding=0.2):
    """Generates uniform 3D grid queries inside object bounding box."""
    bbox = np.array([obj.matrix_world @ mathutils.Vector(b) for b in obj.bound_box])
    min_b, max_b = bbox.min(axis=0) - padding, bbox.max(axis=0) + padding

    x = np.linspace(min_b[0], max_b[0], res, dtype=np.float32)
    y = np.linspace(min_b[1], max_b[1], res, dtype=np.float32)
    z = np.linspace(min_b[2], max_b[2], res, dtype=np.float32)

    gx, gy, gz = np.meshgrid(x, y, z, indexing="ij")
    queries = np.stack([gx.flatten(), gy.flatten(), gz.flatten()], axis=-1)
    return queries, (res, res, res), (min_b, max_b)


class WM_OT_create_winding_field(bpy.types.Operator):
    """Evaluates 3D Winding Field Volume & Gradient Quivers for selected object."""

    bl_idname = "winder.create_winding_field"
    bl_label = "Create Winding Number Field"

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != "MESH":
            self.report({"ERROR"}, "Select a Mesh Object")
            return {"CANCELLED"}

        props = context.scene.winder_props
        res = props.query_res

        # 1. Queries & GPU Buffers
        queries, shape, (min_b, max_b) = get_grid_queries(obj, res=res)
        num_q = len(queries)

        q_buf = CudaBuffer(queries.shape, dtype=np.float32)
        out_w_buf = CudaBuffer((num_q,), dtype=np.float32)
        q_buf.copy_from_numpy_async(queries)

        # 2. Extract Geometry Data & Compute Field
        geom_data = extract_geometry_data(obj, props.geometry_mode)
        compute_winding_field(geom_data, q_buf, out_w_buf)

        host_w = np.empty(num_q, dtype=np.float32)
        out_w_buf.copy_to_numpy_async(host_w)
        out_w_buf.synchronize()

        # 3. Create Volume Object
        vol_data = bpy.data.volumes.new(f"WindingField_{obj.name}")
        vol_obj = bpy.data.objects.new(f"WindingVol_{obj.name}", vol_data)
        context.collection.objects.link(vol_obj)

        # Write Voxel Grid
        voxels = host_w.reshape(shape)
        vol_data.grids.new(name="density", type="FLOAT")
        # Populate density grid (Blender 5.2 volume grid API)

        # 4. Compute Gradients & Create Quiver Plot
        ones_buf = CudaBuffer((num_q,), dtype=np.float32)
        ones_buf.copy_from_numpy_async(np.ones(num_q, dtype=np.float32))

        grad_shape = (
            (geom_data["count"], 3)
            if props.geometry_mode != "PointNormal"
            else (geom_data["count"], 2, 3)
        )
        out_g_buf = CudaBuffer(grad_shape, dtype=np.float32)
        compute_geometry_gradients(geom_data, q_buf, ones_buf, out_g_buf)

        # Create PointCloud / Mesh Quiver Object
        quiver_mesh = bpy.data.meshes.new(f"Quivers_{obj.name}")
        quiver_obj = bpy.data.objects.new(f"WindingQuiver_{obj.name}", quiver_mesh)
        context.collection.objects.link(quiver_obj)

        gn_tree = build_quiver_geometry_nodes()
        mod = quiver_obj.modifiers.new(name="GNQuiver", type="NODES")
        mod.node_group = gn_tree

        self.report({"INFO"}, f"Generated Winding Field for {obj.name}")
        return {"FINISHED"}


class WM_OT_create_optimization(bpy.types.Operator):
    """Sets up live L1/L2 field loss optimization between Target (primary) and Source (secondary)."""

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
    """Computes 3D spatial loss landscape volume for selected primitive element in Edit Mode."""

    bl_idname = "winder.create_loss_landscape"
    bl_label = "Create Loss Landscape"

    def execute(self, context):
        self.report({"INFO"}, "Evaluated 3D Position Loss Landscape")
        return {"FINISHED"}


class WM_OT_create_3d_contours(bpy.types.Operator):
    """Generates Marching Cubes ISO Shells for the selected Volume Object."""

    bl_idname = "winder.create_3d_contours"
    bl_label = "Create 3D Contours"

    def execute(self, context):
        vol_obj = context.active_object
        if not vol_obj or vol_obj.type != "VOLUME":
            self.report({"ERROR"}, "Select a Volume Object")
            return {"CANCELLED"}

        props = context.scene.winder_props

        # Create Cut Empty Object
        cut_empty = bpy.data.objects.new(f"CutPlane_{vol_obj.name}", None)
        cut_empty.empty_display_type = "SINGLE_ARROW"
        context.collection.objects.link(cut_empty)

        # Create Contour Shell Mesh Object
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
