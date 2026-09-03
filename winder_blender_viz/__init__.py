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

from .dlpack_bridge import CudaBuffer
from .handlers import register_handlers, unregister_handlers
from .operators import (
    WM_OT_create_3d_contours,
    WM_OT_create_loss_landscape,
    WM_OT_create_optimization,
    WM_OT_create_winding_field,
    update_contour_shells,
    update_all_winding_fields,
)
from .winder_wrapper import (
    compute_geometry_gradients,
    compute_winding_field,
    extract_geometry_data,
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
    is_quiver_creation_active: bpy.props.BoolProperty(
        name="Show Gradient Directions",
        description="Create a quiver visualization for every geometry primitive",
        default=True,
        update=update_all_winding_fields,
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

        layout.prop(props, "geometry_mode")
        layout.prop(props, "query_res")
        layout.prop(props, "grid_padding")
        layout.prop(props, "is_quiver_creation_active")

        layout.separator()
        layout.operator("winder.create_winding_field", icon="VOLUME_DATA")

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
    WM_OT_create_optimization,
    WM_OT_create_loss_landscape,
    WM_OT_create_3d_contours,
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
