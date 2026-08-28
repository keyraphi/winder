import bpy


def get_or_create_material(name, is_volume=False):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodes.clear()

        if is_volume:
            vol_node = nodes.new("ShaderNodeVolumePrincipled")
            output_node = nodes.new("ShaderNodeOutputMaterial")
            mat.node_tree.links.new(
                vol_node.outputs["Volume"], output_node.inputs["Volume"]
            )

            attr_node = nodes.new("ShaderNodeAttribute")
            attr_node.attribute_name = "density"
            mat.node_tree.links.new(
                attr_node.outputs["Color"], vol_node.inputs["Density"]
            )
    return mat


def build_quiver_node_tree(obj, arrow_scale=0.1):
    """Constructs a Geometry Nodes modifier for displaying 3D Magma gradient quivers."""
    mod = obj.modifiers.get("WinderQuiverNodes")
    if not mod:
        mod = obj.modifiers.new(name="WinderQuiverNodes", type="NODES")

    node_tree = bpy.data.node_groups.new(
        name=f"WinderQuiver_{obj.name}", type="GeometryNodeTree"
    )
    mod.node_group = node_tree

    nodes = node_tree.nodes
    links = node_tree.links
    nodes.clear()

    in_node = nodes.new("NodeGroupInput")
    out_node = nodes.new("NodeGroupOutput")
    node_tree.interface.new_socket(
        name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
    )
    node_tree.interface.new_socket(
        name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
    )

    # Primitive Arrow
    cone = nodes.new("GeometryNodeMeshCone")
    cone.inputs["Radius Bottom"].default_value = 0.02 * arrow_scale
    cone.inputs["Depth"].default_value = 0.1 * arrow_scale

    # Instance vector field
    named_vec = nodes.new("GeometryNodeInputNamedAttribute")
    named_vec.data_type = "FLOAT_VECTOR"
    named_vec.inputs["Name"].default_value = "gradient_vector"

    align_rot = nodes.new("FunctionNodeAlignRotationToVector")
    align_rot.axis = "Z"
    links.new(named_vec.outputs["Attribute"], align_rot.inputs["Vector"])

    instance_node = nodes.new("GeometryNodeInstanceOnPoints")
    links.new(in_node.outputs["Geometry"], instance_node.inputs["Points"])
    links.new(cone.outputs["Mesh"], instance_node.inputs["Instance"])
    links.new(align_rot.outputs["Rotation"], instance_node.inputs["Rotation"])

    links.new(instance_node.outputs["Instances"], out_node.inputs["Geometry"])


def build_contour_node_tree(obj, volume_obj, cut_empty=None, contour_count=5):
    """Builds a Volume-to-Mesh contour extraction setup with an interactive Cut-Plane."""
    mod = obj.modifiers.get("WinderContourNodes")
    if not mod:
        mod = obj.modifiers.new(name="WinderContourNodes", type="NODES")

    node_tree = bpy.data.node_groups.new(
        name=f"WinderContour_{obj.name}", type="GeometryNodeTree"
    )
    mod.node_group = node_tree
    nodes = node_tree.nodes
    links = node_tree.links
    nodes.clear()

    out_node = nodes.new("NodeGroupOutput")
    node_tree.interface.new_socket(
        name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
    )

    vol_info = nodes.new("GeometryNodeObjectInfo")
    vol_info.inputs["Object"].default_value = volume_obj

    v2m = nodes.new("GeometryNodeVolumeToMesh")
    links.new(vol_info.outputs["Geometry"], v2m.inputs["Volume"])

    current_geo = v2m.outputs["Mesh"]

    if cut_empty:
        empty_info = nodes.new("GeometryNodeObjectInfo")
        empty_info.inputs["Object"].default_value = cut_empty

        pos = nodes.new("GeometryNodeInputPosition")
        trans = nodes.new("GeometryNodeTransformPoint")
        links.new(pos.outputs["Position"], trans.inputs["Vector"])
        links.new(
            empty_info.outputs["Transform"], trans.inputs["Transform"]
        )  # Inverse matrix local coordinates

        separate = nodes.new("ShaderNodeSeparateXYZ")
        links.new(trans.outputs["Vector"], separate.inputs["Vector"])

        compare = nodes.new("FunctionNodeCompare")
        compare.operation = "GREATER_THAN"
        compare.inputs[1].default_value = 0.0  # Cut positive Y axis
        links.new(separate.outputs["Y"], compare.inputs[0])

        delete_geo = nodes.new("GeometryNodeDeleteGeometry")
        links.new(current_geo, delete_geo.inputs["Geometry"])
        links.new(compare.outputs["Result"], delete_geo.inputs["Selection"])
        current_geo = delete_geo.outputs["Geometry"]

    links.new(current_geo, out_node.inputs["Geometry"])
