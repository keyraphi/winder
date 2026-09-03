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
        dens_scale.inputs[1].default_value = 5.0
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

        # Position 1.0: Green (Positive Winding)
        color_ramp.color_ramp.elements[1].color = (0.02, 0.9, 0.02, 1.0)
        color_ramp.color_ramp.elements[1].position = 1.0

        # Position 0.5: Dark Center (Zero Winding)
        elem_zero = color_ramp.color_ramp.elements.new(0.5)
        elem_zero.color = (0.02, 0.02, 0.02, 1.0)


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


def set_color_ramp_stops(color_ramp_node, stops):
    """Safely updates a ShaderNodeValToRGB node with a list of (position, RGBA) tuples."""
    ramp = color_ramp_node.color_ramp
    stops = sorted(stops, key=lambda s: s[0])

    # Ensure the exact number of elements exist
    while len(ramp.elements) < len(stops):
        ramp.elements.new(0.5)
    while len(ramp.elements) > len(stops):
        ramp.elements.remove(ramp.elements[-1])

    # Assign positions and colors in ascending order
    for elem, (pos, color) in zip(ramp.elements, stops):
        elem.position = pos
        elem.color = color


def get_or_create_contour_material(
    vol_obj_name, min_val, max_val, is_winding_field=True
):
    """Creates or updates the contour shader material with dynamic range mapping and custom color ramps."""
    mat_name = f"Mat_IsoContours_{vol_obj_name}"
    mat = bpy.data.materials.get(mat_name)

    if not mat:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True

    node_tree = mat.node_tree
    nodes = node_tree.nodes
    links = node_tree.links

    nodes.clear()

    # 1. Attribute Node
    attr_node = nodes.new("ShaderNodeAttribute")
    attr_node.attribute_name = "winding"
    attr_node.location = (-600, 0)

    # 2. Map Range Node
    map_range = nodes.new("ShaderNodeMapRange")
    map_range.clamp = True
    map_range.location = (-400, 0)

    if is_winding_field:
        # Center 0.5 in the middle of the range [0.5 - D, 0.5 + D]
        max_dev = max(abs(0.5 - min_val), abs(max_val - 0.5))
        if max_dev < 1e-6:
            max_dev = 0.5
        from_min = 0.5 - max_dev
        from_max = 0.5 + max_dev
    else:
        # Unconstrained range mapping [min_val, max_val] -> [0.0, 1.0]
        from_min = min_val
        from_max = max_val
        if abs(from_max - from_min) < 1e-6:
            from_max = from_min + 1.0

    map_range.inputs["From Min"].default_value = float(from_min)
    map_range.inputs["From Max"].default_value = float(from_max)
    map_range.inputs["To Min"].default_value = 0.0
    map_range.inputs["To Max"].default_value = 1.0

    # 3. Color Ramp Node
    color_ramp = nodes.new("ShaderNodeValToRGB")
    color_ramp.location = (-150, 0)

    if is_winding_field:
        # Winding Field Mode:
        # Outside (<0.5): Deep Red -> Orange
        # Center (=0.5): Sharp Black Peak
        # Inside (>0.5): Orange -> Bright Green
        stops = [
            (0.00, (0.50, 0.00, 0.00, 1.0)),  # Deep Red (Far Outside)
            (0.48, (1.00, 0.40, 0.00, 1.0)),  # Orange (Near Outside)
            (0.50, (0.00, 0.00, 0.00, 1.0)),  # Sharp Black Peak @ 0.5 Boundary
            (0.52, (1.00, 0.40, 0.00, 1.0)),  # Orange (Near Inside)
            (1.00, (0.00, 0.90, 0.20, 1.0)),  # Bright Green (Far Inside)
        ]
    else:
        # General Scalar Mode:
        # Standard Viridis scientific colormap (Dark Purple -> Teal -> Bright Yellow)
        stops = [
            (0.00, (0.267, 0.004, 0.329, 1.0)),  # Dark Purple
            (0.25, (0.228, 0.322, 0.545, 1.0)),  # Blue
            (0.50, (0.127, 0.567, 0.550, 1.0)),  # Teal
            (0.75, (0.369, 0.788, 0.383, 1.0)),  # Green
            (1.00, (0.993, 0.906, 0.144, 1.0)),  # Yellow
        ]

    set_color_ramp_stops(color_ramp, stops)

    # 4. Principled BSDF
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (150, 0)

    if "Emission Strength" in bsdf.inputs:
        bsdf.inputs["Emission Strength"].default_value = 1.0

    # 5. Output Node
    out_node = nodes.new("ShaderNodeOutputMaterial")
    out_node.location = (450, 0)

    # ---- Connections ----
    links.new(attr_node.outputs["Fac"], map_range.inputs["Value"])
    links.new(map_range.outputs["Result"], color_ramp.inputs["Fac"])

    # Base Color
    links.new(color_ramp.outputs["Color"], bsdf.inputs["Base Color"])

    # Emission Color
    emission_socket = bsdf.inputs.get("Emission Color") or bsdf.inputs.get(
        "Emission"
    )
    if emission_socket:
        links.new(color_ramp.outputs["Color"], emission_socket)

    links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])

    return mat

def build_marching_cubes_contour_nodes(
    vol_obj,
    cut_empty,
    thresholds,
    is_winding_field=True,
    node_group_prefix="GN_3DContours",
    grid_name="winding",
):
    """Generates Marching Cubes Iso-Surfaces using explicit quantile threshold values."""
    node_group_name = f"{node_group_prefix}_{vol_obj.name}"

    ng = bpy.data.node_groups.get(node_group_name)
    if not ng:
        ng = bpy.data.node_groups.new(name=node_group_name, type="GeometryNodeTree")
    else:
        ng.nodes.clear()
        ng.interface.clear()

    ng.interface.new_socket(
        name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
    )

    nodes = ng.nodes
    links = ng.links

    in_node = nodes.new("NodeGroupInput")
    out_node = nodes.new("NodeGroupOutput")
    join_geo = nodes.new("GeometryNodeJoinGeometry")

    vol_info = nodes.new("GeometryNodeObjectInfo")
    vol_info.inputs["Object"].default_value = vol_obj

    pos = nodes.new("GeometryNodeInputPosition")

    # Generate ISO Shells for each quantile threshold
    for thresh_val in thresholds:
        v2m = nodes.new("GeometryNodeVolumeToMesh")
        if hasattr(v2m, "grid_name"):
            v2m.grid_name = grid_name
        elif "Grid Name" in v2m.inputs:
            v2m.inputs["Grid Name"].default_value = grid_name

        v2m.inputs["Threshold"].default_value = float(thresh_val)
        v2m.inputs["Adaptivity"].default_value = 0.0  # no adaptivity for clean cuts
        links.new(vol_info.outputs["Geometry"], v2m.inputs["Volume"])

        # Store threshold as named attribute for shader consumption
        store_attr = nodes.new("GeometryNodeStoreNamedAttribute")
        store_attr.domain = "POINT"
        store_attr.data_type = "FLOAT"
        store_attr.inputs["Name"].default_value = grid_name
        store_attr.inputs["Value"].default_value = float(thresh_val)

        links.new(v2m.outputs["Mesh"], store_attr.inputs["Geometry"])
        links.new(store_attr.outputs["Geometry"], join_geo.inputs["Geometry"])

    # Create & Assign Dynamic Material
    min_val = min(thresholds)
    max_val = max(thresholds)
    mat = get_or_create_contour_material(
        vol_obj_name=vol_obj.name,
        min_val=min_val,
        max_val=max_val,
        is_winding_field=is_winding_field,
    )

    set_mat = nodes.new("GeometryNodeSetMaterial")
    set_mat.inputs["Material"].default_value = mat
    links.new(join_geo.outputs["Geometry"], set_mat.inputs["Geometry"])

    # Fetch Cut Empty Object Info
    obj_info = nodes.new("GeometryNodeObjectInfo")
    obj_info.inputs["Object"].default_value = cut_empty

    # Transform World Position -> Local Space of Cut Empty
    invert_mat = nodes.new("FunctionNodeInvertMatrix")
    links.new(obj_info.outputs["Transform"], invert_mat.inputs[0])

    transform_pos = nodes.new("FunctionNodeTransformPoint")
    links.new(pos.outputs["Position"], transform_pos.inputs[0])
    links.new(invert_mat.outputs[0], transform_pos.inputs[1])

    separate_xyz = nodes.new("ShaderNodeSeparateXYZ")
    links.new(transform_pos.outputs[0], separate_xyz.inputs["Vector"])

    # Delete geometry where local Y > 0
    compare_y = nodes.new("FunctionNodeCompare")
    compare_y.data_type = "FLOAT"
    compare_y.operation = "GREATER_THAN"
    compare_y.inputs[1].default_value = 0.0
    links.new(separate_xyz.outputs["Y"], compare_y.inputs[0])

    delete_geo = nodes.new("GeometryNodeDeleteGeometry")
    links.new(set_mat.outputs["Geometry"], delete_geo.inputs["Geometry"])
    links.new(compare_y.outputs["Result"], delete_geo.inputs["Selection"])

    # Smooth Shading Pass
    set_smooth = nodes.new("GeometryNodeSetShadeSmooth")
    set_smooth.inputs["Shade Smooth"].default_value = True
    links.new(delete_geo.outputs["Geometry"], set_smooth.inputs["Geometry"])

    links.new(set_smooth.outputs["Geometry"], out_node.inputs["Geometry"])
    return ng


