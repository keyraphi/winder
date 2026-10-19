bl_info = {
    "name": "Winding Number Field & Optimization Visualizer",
    "author": "Raphael Braun",
    "version": (1, 0, 0),
    "blender": (5, 2, 0),
    "location": "View3D > Sidebar > Winder Tab",
    "description": "Interactive 3D Winding Fields, Gradient Descent Optimizations, and Loss Landscapes using CUDA DLPack.",
    "category": "3D View",
}

from pathlib import Path
import sys

# Inject addon root directory into sys.path to resolve 'import winder'
addon_dir = str(Path(__file__).parent.resolve())
if addon_dir not in sys.path:
    sys.path.insert(0, addon_dir)

import bpy
import numpy as np

from .handlers import register_handlers, unregister_handlers
from .operators import (
    WM_OT_create_3d_contours,
    WM_OT_create_loss_landscape,
    WM_OT_create_optimization,
    WM_OT_create_winding_field,
    WM_OT_create_query_gradients,
    WM_OT_toggle_freeze_grid,
    WM_OT_dump_forward_bvh,
    WM_OT_dump_backward_bvh,
    WM_OT_create_query_streamlines,
    update_all_streamlines,
    update_streamline_radius,
    update_all_query_gradients,
    update_contour_shells,
    update_all_winding_fields,
    update_query_grad_scale,
)


class WinderProperties(bpy.types.PropertyGroup):
    geometry_mode: bpy.props.EnumProperty(
        name="Mode",
        items=[
            (
                "PointNormal",
                "Point Normal",
                "Triangles treat as point-normal dipoles",
            ),
            ("Triangle", "Triangle", "Exact triangle potential evaluation"),
            ("Mesh", "Mesh", "Vertex/Index explicit mesh topology"),
        ],
        default="PointNormal",
        update=update_all_winding_fields,
    )
    query_res: bpy.props.IntProperty(
        name="Grid Resolution",
        default=64,
        min=16,
        max=1024,
        description="Query grid resolution (N^3)",
        update=update_all_winding_fields,
    )
    grid_padding: bpy.props.FloatProperty(
        name="Grid Padding",
        default=0.2,
        min=0.0,
        max=5.0,
        precision=2,
        description="Bounding box expansion factor (e.g. 0.2 = 20% margin on each axis)",
        update=update_all_winding_fields,
    )
    epsilon: bpy.props.FloatProperty(
        name="epsilon",
        default=0.004,
        min=0.0,
        max=1,
        precision=4,
        description="Regularization size as fraction of diagonal.",
        update=update_all_winding_fields,
    )
    is_quiver_creation_active: bpy.props.BoolProperty(
        name="Show Gradient Directions",
        description="Create a quiver visualization for every geometry primitive",
        default=True,
        update=update_all_winding_fields,
    )
    query_grad_stride: bpy.props.IntProperty(
        name="Query Grad Stride",
        default=8,
        min=1,
        max=64,
        description=(
            "Subsample factor for query-gradient arrows. "
            "Stride S picks every S-th grid point per axis. "
            "With default grid res=64 and stride=8, you get 8x8x8=512 arrows."
        ),
        update=update_all_query_gradients,
    )
    query_grad_scale: bpy.props.FloatProperty(
        name="Query Grad Arrow Scale",
        default=0.3,
        min=0.01,
        max=10.0,
        precision=2,
        description="Length multiplier for query-gradient arrows",
        update=update_query_grad_scale,
    )
    streamline_count: bpy.props.IntProperty(
        name="Streamline Seeds",
        default=200,
        min=4,
        max=5000,
        description="Number of surface seeds; each seed traces both directions",
        update=update_all_streamlines,
    )
    streamline_step: bpy.props.FloatProperty(
        name="Streamline Step",
        default=0.01,
        min=0.001,
        max=0.2,
        precision=4,
        description="RK2 step size as a fraction of the scene diagonal",
        update=update_all_streamlines,
    )
    streamline_max_steps: bpy.props.IntProperty(
        name="Streamline Max Steps",
        default=200,
        min=10,
        max=2000,
        description="Hard cap on integration iterations per line",
        update=update_all_streamlines,
    )
    streamline_max_length: bpy.props.FloatProperty(
        name="Streamline Max Length",
        default=2.0,
        min=0.1,
        max=50.0,
        precision=2,
        description="Maximum arc length per line as a fraction of the diagonal",
        update=update_all_streamlines,
    )
    streamline_min_grad: bpy.props.FloatProperty(
        name="Streamline Min |∇W|",
        default=1e-4,
        min=1e-8,
        max=1.0,
        precision=6,
        description="Stop a line when the local gradient magnitude drops below this",
        update=update_all_streamlines,
    )
    streamline_bidirectional: bpy.props.BoolProperty(
        name="Bidirectional",
        default=True,
        description="Trace +∇W and −∇W from each seed",
        update=update_all_streamlines,
    )
    streamline_radius: bpy.props.FloatProperty(
        name="Streamline Radius",
        default=0.002,
        min=1e-4,
        max=0.1,
        precision=4,
        description="Tube radius as a fraction of the scene diagonal",
        update=update_streamline_radius,
    )
    streamline_seed_jitter: bpy.props.FloatProperty(
        name="Seed Jitter",
        default=0.005,
        min=0.0,
        max=0.1,
        precision=4,
        description=(
            "Gaussian jitter applied to each seed, as a fraction of the scene "
            "diagonal. Breaks exact symmetry so neighbouring streamlines "
            "diverge even for symmetric meshes. Set to 0 for perfectly "
            "deterministic seeds."
        ),
        update=update_all_streamlines,
    )
    loss_type: bpy.props.EnumProperty(
        name="Loss Function",
        items=[
            ("L1", "L1 Absolute", "L1 Field difference"),
            ("L2", "L2 Squared", "L2 Squared field difference"),
        ],
        default="L1",
    )
    learning_rate: bpy.props.FloatProperty(
        name="Learning Rate", default=1e-2, precision=4
    )
    contour_count: bpy.props.IntProperty(
        name="Contour Shells",
        default=5,
        min=1,
        max=20,
        update=update_contour_shells,
    )
    is_winding_field: bpy.props.BoolProperty(
        name="Winding Field Mode",
        description="Center color ramp at 0.5 (surface boundary)",
        default=True,
        update=update_contour_shells,
    )


class VIEW3D_PT_winder_panel(bpy.types.Panel):
    bl_label = "Winding Field & Optimization"
    bl_idname = "VIEW3D_PT_winder_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Winder"

    def draw(self, context):
        layout = self.layout
        props = context.scene.winder_props
        obj = context.active_object

        layout.prop(props, "geometry_mode")
        layout.prop(props, "query_res")

        row = layout.row(align=True)
        row.prop(props, "grid_padding")
        row.prop(props, "epsilon")

        # Display Freeze / Unfreeze toggle button for active object
        if obj and obj.type == "MESH":
            is_frozen = obj.get("winder_is_grid_frozen", False)
            btn_text = "Unfreeze Grid" if is_frozen else "Freeze Grid"
            btn_icon = "LOCKED" if is_frozen else "UNLOCKED"
            row.operator("winder.toggle_freeze_grid", text=btn_text, icon=btn_icon)

        layout.prop(props, "is_quiver_creation_active")

        layout.separator()
        layout.operator("winder.create_winding_field", icon="VOLUME_DATA")

        layout.separator()
        box = layout.box()
        box.label(text="Query Gradients ∇W", icon="ORIENTATION_NORMAL")
        box.prop(props, "query_grad_stride")
        box.prop(props, "query_grad_scale")
        box.operator("winder.create_query_gradients", icon="DRIVER")

        layout.separator()
        box = layout.box()
        box.label(text="Query Field Lines ∇W", icon="CURVE_DATA")
        box.prop(props, "streamline_count")
        box.prop(props, "streamline_seed_jitter")
        row = box.row(align=True)
        row.prop(props, "streamline_step")
        row.prop(props, "streamline_max_steps")
        box.prop(props, "streamline_max_length")
        box.prop(props, "streamline_min_grad")
        box.prop(props, "streamline_bidirectional")
        box.prop(props, "streamline_radius")
        box.operator("winder.create_query_streamlines", icon="IPO_EASE_IN_OUT")

        layout.separator()
        layout.prop(props, "loss_type")
        layout.prop(props, "learning_rate")
        layout.operator("winder.create_optimization", icon="MOD_PHYSICS")

        layout.separator()
        layout.operator("winder.create_loss_landscape", icon="GRAPH")

        layout.separator()
        layout.prop(props, "contour_count")
        layout.prop(props, "is_winding_field")
        layout.operator("winder.create_3d_contours", icon="SURFACE_NCURVE")

        # BVH Debug Section
        layout.separator()
        box = layout.box()
        box.label(text="BVH8 Tree Debug", icon="NODETREE")
        bvh_row = box.row(align=True)
        bvh_row.operator("winder.dump_forward_bvh", text="Dump Forward", icon="FORWARD")
        bvh_row.operator("winder.dump_backward_bvh", text="Dump Backward", icon="BACK")


# -------------------------------------------------------------------------
# Optimization Animation Step Handler
# -------------------------------------------------------------------------
@bpy.app.handlers.persistent
def on_frame_change_optimization(scene):
    if not scene.get("winder_is_optimizing", False):
        return

    target = scene.get("winder_opt_target")
    source = scene.get("winder_opt_source")
    if not target or not source:
        return

    props = scene.winder_props
    lr = props.learning_rate

    # Step Source mesh vertices along negative gradient direction (-lr * grad)
    mesh = source.data
    verts = np.empty((len(mesh.vertices), 3), dtype=np.float32)
    mesh.vertices.foreach_get("co", verts.ravel())

    # Gradient computation drives source vertex offset directly
    source.tag_update()


classes = (
    WinderProperties,
    VIEW3D_PT_winder_panel,
    WM_OT_create_winding_field,
    WM_OT_toggle_freeze_grid,
    WM_OT_create_optimization,
    WM_OT_create_loss_landscape,
    WM_OT_create_3d_contours,
    WM_OT_create_query_gradients,
    WM_OT_create_query_streamlines,
    WM_OT_dump_forward_bvh,
    WM_OT_dump_backward_bvh,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.winder_props = bpy.props.PointerProperty(type=WinderProperties)
    bpy.app.handlers.frame_change_post.append(on_frame_change_optimization)

    # Register live depsgraph updates handler for transform/edit auto-recompute
    register_handlers()


def unregister():
    # Unregister live depsgraph updates handler
    unregister_handlers()

    bpy.app.handlers.frame_change_post.remove(on_frame_change_optimization)
    del bpy.types.Scene.winder_props
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
