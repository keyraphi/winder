import bpy


def build_quiver_geometry_nodes(node_group_name="GN_QuiverPlot"):
    """Constructs Geometry Nodes tree to instance arrows colored by vector magnitude."""
    ng = bpy.data.node_groups.get(node_group_name)
    if ng:
        bpy.data.node_groups.remove(ng)

    ng = bpy.data.node_groups.new(name=node_group_name, type="GeometryNodeTree")

    # Interface
    ng.interface.new_socket(
        name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
    )
    ng.interface.new_socket(
        name="Arrow Scale", in_out="INPUT", socket_type="NodeSocketFloat"
    ).default_value = 0.1
    ng.interface.new_socket(
        name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
    )

    nodes = ng.nodes
    links = ng.links

    in_node = nodes.new("NodeGroupInput")
    out_node = nodes.new("NodeGroupOutput")

    # Primitive Cone for Arrow Head
    cone = nodes.new("GeometryNodeMeshCone")
    cone.inputs["Radius Bottom"].default_value = 0.05
    cone.inputs["Depth"].default_value = 0.2

    # Instance on Points
    iop = nodes.new("GeometryNodeInstanceOnPoints")
    iop.inputs["Instance"].link(cone.outputs["Mesh"])

    # Vector Attribute Align
    named_vector = nodes.new("GeometryNodeInputNamedAttribute")
    named_vector.data_type = "FLOAT_VECTOR"
    named_vector.inputs["Name"].default_value = "gradient_dir"

    align_rot = nodes.new("FunctionNodeAlignRotationToVector")
    align_rot.axis = "Z"
    align_rot.inputs["Vector"].link(named_vector.outputs["Attribute"])
    iop.inputs["Rotation"].link(align_rot.outputs["Rotation"])

    # Scaling
    scale_node = nodes.new("ShaderNodeVectorMath")
    scale_node.operation = "SCALE"
    scale_node.inputs["Vector"].link(named_vector.outputs["Attribute"])
    scale_node.inputs["Scale"].link(in_node.outputs["Arrow Scale"])
    iop.inputs["Scale"].link(scale_node.outputs["Vector"])

    links.new(in_node.outputs["Geometry"], iop.inputs["Points"])
    links.new(iop.outputs["Instances"], out_node.outputs["Geometry"])
    return ng


def build_marching_cubes_contour_nodes(node_group_name="GN_3DContours", num_shells=5):
    """Generates Marching Cubes Iso-Surfaces sliced by an Empty XY Cut Plane."""
    ng = bpy.data.node_groups.get(node_group_name)
    if ng:
        bpy.data.node_groups.remove(ng)

    ng = bpy.data.node_groups.new(name=node_group_name, type="GeometryNodeTree")

    ng.interface.new_socket(
        name="Volume", in_out="INPUT", socket_type="NodeSocketGeometry"
    )
    ng.interface.new_socket(
        name="Cut Empty", in_out="INPUT", socket_type="NodeSocketObject"
    )
    ng.interface.new_socket(
        name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
    )

    nodes = ng.nodes
    links = ng.links

    in_node = nodes.new("NodeGroupInput")
    out_node = nodes.new("NodeGroupOutput")

    join_geo = nodes.new("GeometryNodeJoinGeometry")

    # Generate ISO Shells using Volume to Mesh
    for i in range(num_shells):
        v2m = nodes.new("GeometryNodeVolumeToMesh")
        v2m.inputs["Threshold"].default_value = 0.1 + (i * 0.15)
        v2m.inputs["Adaptivity"].default_value = 0.1
        links.new(in_node.outputs["Volume"], v2m.inputs["Volume"])
        links.new(v2m.outputs["Mesh"], join_geo.inputs["Geometry"])

    # Plane Slicing based on Empty's XY plane
    obj_info = nodes.new("GeometryNodeObjectInfo")
    links.new(in_node.outputs["Cut Empty"], obj_info.inputs["Object"])

    transform_pos = nodes.new("GeometryNodeVectorTransform")
    transform_pos.transform_type = "HANDLED"
    transform_pos.convert_from = "WORLD"
    transform_pos.convert_to = "LOCAL"

    pos = nodes.new("GeometryNodeInputPosition")
    links.new(pos.outputs["Position"], transform_pos.inputs["Vector"])

    separate_xyz = nodes.new("ShaderNodeSeparateXYZ")
    links.new(transform_pos.outputs["Vector"], separate_xyz.inputs["Vector"])

    # Delete geometry where local Y > 0
    compare_y = nodes.new("FunctionNodeCompare")
    compare_y.data_type = "FLOAT"
    compare_y.operation = "GREATER_THAN"
    compare_y.inputs[1].default_value = 0.0
    links.new(separate_xyz.outputs["Y"], compare_y.inputs[0])

    delete_geo = nodes.new("GeometryNodeDeleteGeometry")
    links.new(join_geo.outputs["Geometry"], delete_geo.inputs["Geometry"])
    links.new(compare_y.outputs["Result"], delete_geo.inputs["Selection"])

    links.new(delete_geo.outputs["Geometry"], out_node.outputs["Geometry"])
    return ng
