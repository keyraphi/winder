import bpy


class WINDER_PT_MainPanel(bpy.types.Panel):
    """Creates an N-Panel tab in the 3D Viewport"""

    bl_label = "Winder GPU Field Viz"
    bl_idname = "WINDER_PT_main_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Winder Viz"

    def draw(self, context):
        layout = self.layout
        props = context.scene.winder_props

        box = layout.box()
        box.label(text="Global Configuration", icon="PREFERENCES")
        box.prop(props, "mode")
        box.prop(props, "grid_resolution")

        box = layout.box()
        box.label(text="Field & Optimization", icon="PHYSICS")
        box.operator(
            "winder.create_winding_number_field", icon="FIELD_VOLUMETRIC"
        )

        box.separator()
        box.prop(props, "loss_type")
        box.prop(props, "learning_rate")
        box.operator("winder.create_optimization", icon="MOD_PHYSICS")

        box = layout.box()
        box.label(text="Analysis Tools", icon="VIEW_ZOOM")
        box.operator("winder.create_loss_landscape", icon="SURFACE_NCURVE")
        box.prop(props, "contour_count")
        box.operator("winder.create_3d_contours", icon="MOD_FLUIDSIM")


classes = (WINDER_PT_MainPanel,)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
