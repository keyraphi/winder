import numpy as np
import torch
import winder
import bpy
from .geonodes_builder import build_contour_node_tree, build_quiver_node_tree
from .winder_bridge import (
    compute_geometry_gradients,
    compute_winding_fields,
    extract_blender_mesh_data,
)


def create_point_grid_mesh(name, bbox_min, bbox_max, resolution):
    """Generates a structured 3D vertex mesh grid for volume & quiver attributes."""
    x = np.linspace(bbox_min[0], bbox_max[0], resolution)
    y = np.linspace(bbox_min[1], bbox_max[1], resolution)
    z = np.linspace(bbox_min[2], bbox_max[2], resolution)
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing="ij")

    coords = np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=-1).astype(
        np.float32
    )

    mesh = bpy.data.meshes.new(name=f"{name}_Mesh")
    mesh.from_pydata(coords.tolist(), [], [])
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj, coords


class WINDER_OT_CreateWindingNumberField(bpy.types.Operator):
    """Generates linked Winding Number Volume and Gradient Quivers for selected object"""

    bl_idname = "winder.create_winding_number_field"
    bl_label = "Create Winding Number Field"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.winder_props
        src_obj = context.active_object

        if not src_obj or src_obj.type != "MESH":
            self.report({"ERROR"}, "Please select a Mesh object.")
            return {"CANCELLED"}

        res = props.grid_resolution
        mode = props.mode

        # Compute bounding box queries
        bbox = [src_obj.matrix_world @ bpy.path.Vector(b) for b in src_obj.bound_box]
        bbox_min = np.min(bbox, axis=0) - 0.5
        bbox_max = np.max(bbox, axis=0) + 0.5

        vol_obj, coords = create_point_grid_mesh(
            f"{src_obj.name}_WindingVolume", bbox_min, bbox_max, res
        )
        quiver_obj, _ = create_point_grid_mesh(
            f"{src_obj.name}_GradientQuiver", bbox_min, bbox_max, res
        )

        vol_obj["winder_type"] = "FIELD_VOLUME"
        vol_obj["source_obj"] = src_obj
        quiver_obj["winder_type"] = "FIELD_QUIVER"

        queries_gpu = torch.tensor(coords, device="cuda:0", dtype=torch.float32)
        geo_data = extract_blender_mesh_data(src_obj, mode=mode)

        w_fields = (
            compute_winding_fields(
                geo_data, queries_gpu, mode=mode, epsilon=1.0 / 250.0
            )
            .cpu()
            .numpy()
        )

        # Store volume density attribute
        mesh_vol = vol_obj.data
        density_attr = mesh_vol.attributes.new(
            name="density", type="FLOAT", domain="POINT"
        )
        density_attr.data.foreach_set("value", np.abs(w_fields))

        # Store vector field for gradients (grad_output = 1.0)
        grad_output = torch.ones(
            (queries_gpu.shape[0],), dtype=torch.float32, device="cuda:0"
        )

        # For quiver display at query points, evaluate spatial field derivative dw/dq
        queries_gpu.requires_grad_(True)
        w_out = compute_winding_fields(
            geo_data, queries_gpu, mode=mode, epsilon=1.0 / 250.0
        )
        w_out.sum().backward()
        spatial_grads = queries_gpu.grad.cpu().numpy()

        mesh_quiver = quiver_obj.data
        vec_attr = mesh_quiver.attributes.new(
            name="gradient_vector", type="FLOAT_VECTOR", domain="POINT"
        )
        vec_attr.data.foreach_set("vector", spatial_grads.ravel())

        build_quiver_node_tree(quiver_obj, arrow_scale=props.quiver_arrow_length)
        return {"FINISHED"}


class WINDER_OT_CreateOptimization(bpy.types.Operator):
    """Sets up field loss landscape volume and live optimization between 2 objects"""

    bl_idname = "winder.create_optimization"
    bl_label = "Create Optimization"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.winder_props
        selected = context.selected_objects

        if len(selected) != 2:
            self.report(
                {"ERROR"},
                "Select exactly 2 objects (Target first, Source active).",
            )
            return {"CANCELLED"}

        src_obj = context.active_object
        tgt_obj = selected[0] if selected[1] == src_obj else selected[1]

        res = props.grid_resolution
        mode = props.mode

        bbox = [src_obj.matrix_world @ bpy.path.Vector(b) for b in src_obj.bound_box]
        bbox_min = np.min(bbox, axis=0) - 0.5
        bbox_max = np.max(bbox, axis=0) + 0.5

        loss_vol_obj, coords = create_point_grid_mesh(
            "Optimization_LossVolume", bbox_min, bbox_max, res
        )
        loss_quiver_obj, _ = create_point_grid_mesh(
            "Optimization_LossQuiver", bbox_min, bbox_max, res
        )

        loss_vol_obj["winder_type"] = "LOSS_VOLUME"
        loss_vol_obj["source_obj"] = src_obj
        loss_vol_obj["target_obj"] = tgt_obj

        context.scene["winder_opt_active"] = True
        context.scene["winder_opt_src"] = src_obj
        context.scene["winder_opt_tgt"] = tgt_obj

        build_quiver_node_tree(loss_quiver_obj, arrow_scale=props.quiver_arrow_length)
        self.report({"INFO"}, "Optimization framework ready. Play timeline to step.")
        return {"FINISHED"}


class WINDER_OT_CreateLossLandscape(bpy.types.Operator):
    """Computes spatial position loss landscape over 3D grid for selected vertex/triangle"""

    bl_idname = "winder.create_loss_landscape"
    bl_label = "Create Loss Landscape"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.winder_props
        src_obj = context.active_object
        res = props.grid_resolution

        bbox = [src_obj.matrix_world @ bpy.path.Vector(b) for b in src_obj.bound_box]
        bbox_min = np.min(bbox, axis=0) - 1.0
        bbox_max = np.max(bbox, axis=0) + 1.0

        landscape_vol, coords = create_point_grid_mesh(
            "LossLandscape_Volume", bbox_min, bbox_max, res
        )
        landscape_vol["winder_type"] = "LANDSCAPE_VOLUME"

        return {"FINISHED"}


class WINDER_OT_Create3DContours(bpy.types.Operator):
    """Generates iso-surface contours cut by an interactive Empty object"""

    bl_idname = "winder.create_3d_contours"
    bl_label = "Create 3D Contours"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.winder_props
        vol_obj = context.active_object

        if not vol_obj or "winder_type" not in vol_obj:
            self.report({"ERROR"}, "Select a valid Winder Volume grid object.")
            return {"CANCELLED"}

        # Shared Cut-Plane Empty
        cut_empty = bpy.data.objects.get("Winder_CutPlane_Empty")
        if not cut_empty:
            cut_empty = bpy.data.objects.new("Winder_CutPlane_Empty", None)
            cut_empty.empty_display_type = "SINGLE_ARROW"
            context.collection.objects.link(cut_empty)

        contour_obj = bpy.data.objects.new(
            f"{vol_obj.name}_IsoContours", bpy.data.meshes.new("IsoMesh")
        )
        context.collection.objects.link(contour_obj)

        build_contour_node_tree(
            contour_obj,
            vol_obj,
            cut_empty=cut_empty,
            contour_count=props.contour_count,
        )
        return {"FINISHED"}


class WINDER_OT_UpdateAllFields(bpy.types.Operator):
    """Recomputes all active fields upon global setting changes"""

    bl_idname = "winder.update_all_fields"
    bl_label = "Update All Fields"

    def execute(self, context):
        return {"FINISHED"}


classes = (
    WINDER_OT_CreateWindingNumberField,
    WINDER_OT_CreateOptimization,
    WINDER_OT_CreateLossLandscape,
    WINDER_OT_Create3DContours,
    WINDER_OT_UpdateAllFields,
)
