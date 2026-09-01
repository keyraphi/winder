import bpy

import bpy


def get_or_create_volume_material(mat_name="Winder_Volume_Material"):
    """Volume shader reading 'density' (|w|) for opacity and 'winding' (signed w) for color."""
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()

        out_node = nodes.new("ShaderNodeOutputMaterial")
        princ_vol = nodes.new("ShaderNodeVolumePrincipled")

        # 1. Density Attribute (|w|) -> Volume Opacity
        attr_density = nodes.new("ShaderNodeAttribute")
        attr_density.attribute_type = "GEOMETRY"
        attr_density.attribute_name = "density"

        dens_scale = nodes.new("ShaderNodeMath")
        dens_scale.operation = "MULTIPLY"
        dens_scale.inputs[1].default_value = 2.0
        links.new(attr_density.outputs["Fac"], dens_scale.inputs[0])
        links.new(dens_scale.outputs["Value"], princ_vol.inputs["Density"])

        # 2. Signed Winding Attribute (w) -> Color Ramp (-1.0 to 1.0)
        attr_winding = nodes.new("ShaderNodeAttribute")
        attr_winding.attribute_type = "GEOMETRY"
        attr_winding.attribute_name = "winding"

        map_range = nodes.new("ShaderNodeMapRange")
        map_range.clamp = True
        map_range.inputs["From Min"].default_value = -1.0
        map_range.inputs["From Max"].default_value = 1.0
        map_range.inputs["To Min"].default_value = 0.0
        map_range.inputs["To Max"].default_value = 1.0
        links.new(attr_winding.outputs["Fac"], map_range.inputs["Value"])

        color_ramp = nodes.new("ShaderNodeValToRGB")

        # Position 0.0: Red (Negative Winding)
        color_ramp.color_ramp.elements[0].color = (0.9, 0.02, 0.02, 1.0)
        color_ramp.color_ramp.elements[0].position = 0.0

        # Position 0.5: Dark Center (Zero Winding)
        elem_zero = color_ramp.color_ramp.elements.new(0.5)
        elem_zero.color = (0.02, 0.02, 0.02, 1.0)

        # Position 1.0: Green (Positive Winding)
        color_ramp.color_ramp.elements[1].color = (0.02, 0.9, 0.02, 1.0)
        color_ramp.color_ramp.elements[1].position = 1.0

        links.new(map_range.outputs["Result"], color_ramp.inputs["Fac"])
        links.new(color_ramp.outputs["Color"], princ_vol.inputs["Color"])

        # Connect Output
        links.new(princ_vol.outputs["Volume"], out_node.inputs["Volume"])

    return mat

def get_or_create_quiver_material(mat_name="Winder_Quiver_Material"):
    """Creates a shader material mapping log-scaled gradient magnitude to a color ramp."""
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()

        out_node = nodes.new("ShaderNodeOutputMaterial")
        emission = nodes.new("ShaderNodeEmission")

        # Color Ramp (Cool Blue -> Yellow -> Hot Red)
        color_ramp = nodes.new("ShaderNodeValToRGB")
        color_ramp.color_ramp.elements[0].color = (0.05, 0.2, 0.9, 1.0)
        color_ramp.color_ramp.elements.new(0.5).color = (0.9, 0.8, 0.1, 1.0)
        color_ramp.color_ramp.elements[1].color = (0.9, 0.1, 0.05, 1.0)

        # Map log-magnitude range to [0.0, 1.0] for color ramp
        map_range = nodes.new("ShaderNodeMapRange")
        map_range.clamp = True
        map_range.inputs["From Min"].default_value = 0.0
        map_range.inputs["From Max"].default_value = 10.0  # log(30000) ~ 10.3

        log_math = nodes.new("ShaderNodeMath")
        log_math.operation = "LOGARITHM"
        log_math.inputs[1].default_value = 2.7182818  # Natural log

        attr_node = nodes.new("ShaderNodeAttribute")
        attr_node.attribute_type = "GEOMETRY"
        attr_node.attribute_name = "gradient_mag"

        links.new(attr_node.outputs["Fac"], log_math.inputs[0])
        links.new(log_math.outputs["Value"], map_range.inputs["Value"])
        links.new(map_range.outputs["Result"], color_ramp.inputs["Fac"])
        links.new(color_ramp.outputs["Color"], emission.inputs["Color"])
        links.new(emission.outputs["Emission"], out_node.inputs["Surface"])

    return mat


def build_quiver_geometry_nodes(node_group_name="GN_QuiverPlot", default_scale=0.15):
    """Constructs Geometry Nodes tree to instance 3-vert arrows log-scaled & colored by vector magnitude."""
    ng = bpy.data.node_groups.get(node_group_name)
    if ng:
        bpy.data.node_groups.remove(ng)

    ng = bpy.data.node_groups.new(name=node_group_name, type="GeometryNodeTree")

    # Interface Sockets (Setting default_value directly on socket avoids modifier IDProperty errors)
    ng.interface.new_socket(
        name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
    )
    scale_socket = ng.interface.new_socket(
        name="Arrow Scale", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    scale_socket.default_value = default_scale

    ng.interface.new_socket(
        name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
    )

    nodes = ng.nodes
    links = ng.links

    in_node = nodes.new("NodeGroupInput")
    out_node = nodes.new("NodeGroupOutput")

    # 3-Vertex Primitive Cone for lightweight arrows
    cone = nodes.new("GeometryNodeMeshCone")
    cone.inputs["Vertices"].default_value = 3
    cone.inputs["Radius Bottom"].default_value = 0.05
    cone.inputs["Depth"].default_value = 0.2

    iop = nodes.new("GeometryNodeInstanceOnPoints")
    links.new(cone.outputs["Mesh"], iop.inputs["Instance"])

    # Read Normalized Direction Attribute
    named_dir = nodes.new("GeometryNodeInputNamedAttribute")
    named_dir.data_type = "FLOAT_VECTOR"
    named_dir.inputs["Name"].default_value = "gradient_dir"

    # Align Arrow Rotation along Z-axis
    align_rot = nodes.new("FunctionNodeAlignRotationToVector")
    align_rot.axis = "Z"
    links.new(named_dir.outputs["Attribute"], align_rot.inputs["Vector"])
    links.new(align_rot.outputs["Rotation"], iop.inputs["Rotation"])

    # Read Magnitude Attribute
    named_mag = nodes.new("GeometryNodeInputNamedAttribute")
    named_mag.data_type = "FLOAT"
    named_mag.inputs["Name"].default_value = "gradient_mag"

    # Logarithmic scaling: log1p(mag) prevents visual explosion of large gradients
    log_node = nodes.new("ShaderNodeMath")
    log_node.operation = "LOGARITHM"
    log_node.inputs[1].default_value = 2.7182818
    links.new(named_mag.outputs["Attribute"], log_node.inputs[0])

    scale_math = nodes.new("ShaderNodeMath")
    scale_math.operation = "MULTIPLY"
    links.new(log_node.outputs["Value"], scale_math.inputs[0])
    links.new(in_node.outputs["Arrow Scale"], scale_math.inputs[1])

    # Pass uniform float scale to Instances on Points
    links.new(scale_math.outputs["Value"], iop.inputs["Scale"])
    links.new(in_node.outputs["Geometry"], iop.inputs["Points"])

    # Realize Instances to evaluate mesh attributes in shader
    realize = nodes.new("GeometryNodeRealizeInstances")
    links.new(iop.outputs["Instances"], realize.inputs["Geometry"])

    # Set Material
    mat = get_or_create_quiver_material()
    set_mat = nodes.new("GeometryNodeSetMaterial")
    set_mat.inputs["Material"].default_value = mat
    links.new(realize.outputs["Geometry"], set_mat.inputs["Geometry"])

    links.new(set_mat.outputs["Geometry"], out_node.inputs["Geometry"])
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

    # NodeGroupOutput uses inputs, not outputs
    links.new(delete_geo.outputs["Geometry"], out_node.inputs["Geometry"])
    return ng
