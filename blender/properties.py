import bpy


def update_grid_resolution(self, context):
    """Triggers recalculation for all active volume objects when grid resolution changes."""
    new_res = context.scene.winder_props.grid_resolution
    for obj in bpy.data.objects:
        if obj.get("winder_type") in {
            "FIELD_VOLUME",
            "LOSS_VOLUME",
            "LANDSCAPE_VOLUME",
        }:
            obj["grid_res"] = new_res
            bpy.ops.winder.update_all_fields()


class WinderSceneProperties(bpy.types.PropertyGroup):
    mode: bpy.props.EnumProperty(
        name="Geometry Mode",
        description="Representation strategy for Winder computations",
        items=[
            (
                "POINT_NORMAL",
                "Point Normal",
                "Centroids + Scaled Voronoi Normals",
            ),
            ("TRIANGLE", "Triangle", "Disjoint 3x3 Triangle Arrays"),
            ("MESH", "Mesh", "Shared Vertices + Triangle Indices"),
        ],
        default="POINT_NORMAL",
    )

    grid_resolution: bpy.props.IntProperty(
        name="Grid Resolution",
        description="Resolution of 3D volumetric query grids (N x N x N)",
        default=64,
        min=8,
        max=256,
        update=update_grid_resolution,
    )

    loss_type: bpy.props.EnumProperty(
        name="Loss Function",
        items=[
            ("L1", "L1 Loss", "Absolute Field Difference |w_src - w_tgt|"),
            ("L2", "L2 Loss", "Squared Field Difference (w_src - w_tgt)^2"),
        ],
        default="L1",
    )

    learning_rate: bpy.props.FloatProperty(
        name="Learning Rate",
        description="Step size for live geometry gradient descent optimization",
        default=0.01,
        min=1e-6,
        max=1.0,
        precision=4,
    )

    quiver_arrow_length: bpy.props.FloatProperty(
        name="Arrow Length Scale",
        description="Constant baseline scale for gradient quiver vectors",
        default=0.1,
        min=0.001,
        max=10.0,
    )

    contour_count: bpy.props.IntProperty(
        name="Contour Shells",
        description="Number of iso-surface shells extracted in 3D Contours mode",
        default=5,
        min=1,
        max=20,
    )


classes = (WinderSceneProperties,)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.winder_props = bpy.props.PointerProperty(type=WinderSceneProperties)


def unregister():
    del bpy.types.Scene.winder_props
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
