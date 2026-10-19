import bpy


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
        dens_scale.inputs[1].default_value = 10.0
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
        # Standard Jet colormap linear color stops
        stops = [
            (0.000, (0.0, 0.0, 0.5, 1.0)),  # Dark Blue
            (0.125, (0.0, 0.0, 1.0, 1.0)),  # Bright Blue
            (0.375, (0.0, 1.0, 1.0, 1.0)),  # Cyan
            (0.625, (1.0, 1.0, 0.0, 1.0)),  # Yellow
            (0.875, (1.0, 0.0, 0.0, 1.0)),  # Red
            (1.000, (0.5, 0.0, 0.0, 1.0)),  # Dark Red
        ]

        set_color_ramp_stops(color_ramp, stops)

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
    cone.inputs["Radius Bottom"].default_value = 0.02
    cone.inputs["Depth"].default_value = 0.3

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
        # Dont touch the range
        from_min = 0.0
        from_max = 1.1
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

    stops = [
        (0.000, (0.0, 0.0, 0.5, 1.0)),  # Dark Blue
        (0.125, (0.0, 0.0, 1.0, 1.0)),  # Bright Blue
        (0.375, (0.0, 1.0, 1.0, 1.0)),  # Cyan
        (0.625, (1.0, 1.0, 0.0, 1.0)),  # Yellow
        (0.875, (1.0, 0.0, 0.0, 1.0)),  # Red
        (1.000, (0.5, 0.0, 0.0, 1.0)),  # Dark Red
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
    emission_socket = bsdf.inputs.get("Emission Color") or bsdf.inputs.get("Emission")
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


def get_or_create_query_grad_material(mat_name="Winder_QueryGrad_Material"):
    """Query-gradient material with log1p magnitude coloring.

    log1p(mag) is always non-negative and grows slowly, so both
    near-surface gradients (~1e3) and far-field gradients (~1e-4) remain
    visible and correctly oriented.
    """
    mat = bpy.data.materials.get(mat_name)
    if mat is not None:
        return mat

    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out_node = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")

    color_ramp = nodes.new("ShaderNodeValToRGB")
    stops = [
        (0.00, (0.02, 0.05, 0.55, 1.0)),  # deep blue (tiny)
        (0.25, (0.00, 0.60, 0.90, 1.0)),  # cyan-blue
        (0.50, (0.10, 0.85, 0.30, 1.0)),  # green
        (0.75, (1.00, 0.75, 0.05, 1.0)),  # yellow
        (1.00, (0.90, 0.05, 0.05, 1.0)),  # red (huge)
    ]
    set_color_ramp_stops(color_ramp, stops)

    # log1p(mag) over the typical query-gradient range [~1e-4, ~1e3]
    # maps to roughly [0, 6.9]. Compress to a focused range [0, 5].
    log_math = nodes.new("ShaderNodeMath")
    log_math.operation = "LOGARITHM"
    log_math.inputs[1].default_value = 2.718281828459045

    add_node = nodes.new("ShaderNodeMath")
    add_node.operation = "ADD"
    add_node.inputs[1].default_value = 1.0

    map_range = nodes.new("ShaderNodeMapRange")
    map_range.clamp = True
    map_range.inputs["From Min"].default_value = 0.0
    map_range.inputs["From Max"].default_value = 5.0

    attr_node = nodes.new("ShaderNodeAttribute")
    attr_node.attribute_type = "GEOMETRY"
    attr_node.attribute_name = "gradient_mag"

    links.new(attr_node.outputs["Fac"], add_node.inputs[0])
    links.new(add_node.outputs["Value"], log_math.inputs[0])
    links.new(log_math.outputs["Value"], map_range.inputs["Value"])
    links.new(map_range.outputs["Result"], color_ramp.inputs["Fac"])
    links.new(color_ramp.outputs["Color"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], out_node.inputs["Surface"])

    return mat


def build_query_grad_geometry_nodes(
    node_group_name="GN_QueryGradQuiver", default_scale=0.3
):
    """GN tree for query-gradient arrows using log1p magnitude scaling.

    Reuses the same `gradient_dir` / `gradient_mag` attribute convention as
    the geometry-gradient quiver, so it is drop-in compatible with the
    existing quiver-mesh update code path.
    """
    ng = bpy.data.node_groups.get(node_group_name)
    if ng:
        return ng

    ng = bpy.data.node_groups.new(name=node_group_name, type="GeometryNodeTree")

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

    cone = nodes.new("GeometryNodeMeshCone")
    cone.inputs["Vertices"].default_value = 3
    cone.inputs["Radius Bottom"].default_value = 0.02
    cone.inputs["Depth"].default_value = 0.3

    iop = nodes.new("GeometryNodeInstanceOnPoints")
    links.new(cone.outputs["Mesh"], iop.inputs["Instance"])

    named_dir = nodes.new("GeometryNodeInputNamedAttribute")
    named_dir.data_type = "FLOAT_VECTOR"
    named_dir.inputs["Name"].default_value = "gradient_dir"

    align_rot = nodes.new("FunctionNodeAlignRotationToVector")
    align_rot.axis = "Z"
    links.new(named_dir.outputs["Attribute"], align_rot.inputs["Vector"])
    links.new(align_rot.outputs["Rotation"], iop.inputs["Rotation"])

    named_mag = nodes.new("GeometryNodeInputNamedAttribute")
    named_mag.data_type = "FLOAT"
    named_mag.inputs["Name"].default_value = "gradient_mag"

    add_node = nodes.new("ShaderNodeMath")
    add_node.operation = "ADD"
    add_node.inputs[1].default_value = 1.0
    links.new(named_mag.outputs["Attribute"], add_node.inputs[0])

    log_node = nodes.new("ShaderNodeMath")
    log_node.operation = "LOGARITHM"
    log_node.inputs[1].default_value = 2.718281828459045
    links.new(add_node.outputs["Value"], log_node.inputs[0])

    scale_math = nodes.new("ShaderNodeMath")
    scale_math.operation = "MULTIPLY"
    links.new(log_node.outputs["Value"], scale_math.inputs[0])
    links.new(in_node.outputs["Arrow Scale"], scale_math.inputs[1])

    links.new(scale_math.outputs["Value"], iop.inputs["Scale"])
    links.new(in_node.outputs["Geometry"], iop.inputs["Points"])

    realize = nodes.new("GeometryNodeRealizeInstances")
    links.new(iop.outputs["Instances"], realize.inputs["Geometry"])

    mat = get_or_create_query_grad_material()
    set_mat = nodes.new("GeometryNodeSetMaterial")
    set_mat.inputs["Material"].default_value = mat
    links.new(realize.outputs["Geometry"], set_mat.inputs["Geometry"])

    links.new(set_mat.outputs["Geometry"], out_node.inputs["Geometry"])
    return ng


def get_or_create_streamline_material(mat_name="Winder_Streamline_Material"):
    """Emissive material colored by log1p(gradient magnitude)."""
    mat = bpy.data.materials.get(mat_name)
    if mat:
        return mat

    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out_node = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 2.0

    attr = nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "grad_mag"

    add = nodes.new("ShaderNodeMath")
    add.operation = "ADD"
    add.inputs[1].default_value = 1.0
    links.new(attr.outputs["Fac"], add.inputs[0])

    logn = nodes.new("ShaderNodeMath")
    logn.operation = "LOGARITHM"
    logn.inputs[1].default_value = 2.718281828459045
    links.new(add.outputs["Value"], logn.inputs[0])

    mr = nodes.new("ShaderNodeMapRange")
    mr.clamp = True
    mr.inputs["From Min"].default_value = 0.0
    mr.inputs["From Max"].default_value = 5.0
    links.new(logn.outputs["Value"], mr.inputs["Value"])

    ramp = nodes.new("ShaderNodeValToRGB")
    stops = [
        (0.00, (0.02, 0.05, 0.55, 1.0)),
        (0.30, (0.00, 0.60, 0.95, 1.0)),
        (0.55, (0.10, 0.90, 0.35, 1.0)),
        (0.80, (1.00, 0.80, 0.05, 1.0)),
        (1.00, (0.95, 0.10, 0.05, 1.0)),
    ]
    set_color_ramp_stops(ramp, stops)
    links.new(mr.outputs["Result"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], out_node.inputs["Surface"])
    return mat


def build_streamline_geometry_nodes(
    radius, node_group_name="GN_Streamlines", resample_count=16, profile_res=5
):
    """GN tree: polylines → smooth tubes.

    On every call, the Radius socket default is refreshed so existing
    modifier instances see the new value even without an explicit override.
    """
    ng = bpy.data.node_groups.get(node_group_name)

    if ng is not None:
        # Update existing node group's Radius default in place
        for item in ng.interface.items_tree:
            if (
                item.item_type == "SOCKET"
                and getattr(item, "in_out", "") == "INPUT"
                and item.name == "Radius"
            ):
                item.default_value = float(radius)
                break
        return ng

    # ---- Fresh node group creation (radius baked into default) ----
    ng = bpy.data.node_groups.new(name=node_group_name, type="GeometryNodeTree")
    ng.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    rs = ng.interface.new_socket(name="Radius", in_out="INPUT", socket_type="NodeSocketFloat")
    rs.default_value = float(radius)
    ng.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    nodes = ng.nodes
    links = ng.links
    in_node = nodes.new("NodeGroupInput")
    out_node = nodes.new("NodeGroupOutput")

    to_curve = nodes.new("GeometryNodeMeshToCurve")
    links.new(in_node.outputs["Geometry"], to_curve.inputs["Mesh"])

    profile = nodes.new("GeometryNodeCurvePrimitiveCircle")
    profile.inputs["Resolution"].default_value = int(profile_res)
    links.new(in_node.outputs["Radius"], profile.inputs["Radius"])

    c2m = nodes.new("GeometryNodeCurveToMesh")
    links.new(to_curve.outputs["Curve"], c2m.inputs["Curve"])
    links.new(profile.outputs["Curve"], c2m.inputs["Profile Curve"])

    mat = get_or_create_streamline_material()
    set_mat = nodes.new("GeometryNodeSetMaterial")
    set_mat.inputs["Material"].default_value = mat
    links.new(c2m.outputs["Mesh"], set_mat.inputs["Geometry"])

    smooth = nodes.new("GeometryNodeSetShadeSmooth")
    links.new(set_mat.outputs["Geometry"], smooth.inputs["Geometry"])
    links.new(smooth.outputs["Geometry"], out_node.inputs["Geometry"])
    return ng
