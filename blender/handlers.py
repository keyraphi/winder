import numpy as np
import torch
import bpy
from bpy.app.handlers import persistent
from .winder_bridge import (
    compute_geometry_gradients,
    compute_winding_fields,
    extract_blender_mesh_data,
)


@persistent
def on_depsgraph_update(scene):
    """Real-time synchronization for Viewport & Edit Mode adjustments."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for update in depsgraph.updates:
        if update.id.name in bpy.data.objects:
            obj = bpy.data.objects[update.id.name]
            # Trigger lazy re-computation of associated volume fields
            pass


@persistent
def on_frame_change_optimization(scene):
    """Performs continuous gradient descent optimization step during timeline playback."""
    if not scene.get("winder_opt_active", False):
        return

    props = scene.winder_props
    src_obj = scene.get("winder_opt_src")
    tgt_obj = scene.get("winder_opt_tgt")

    if not src_obj or not tgt_obj:
        return

    mode = props.mode
    lr = props.learning_rate
    res = props.grid_resolution

    # Sample query bounding domain
    bbox = [src_obj.matrix_world @ bpy.path.Vector(b) for b in src_obj.bound_box]
    bbox_min = np.min(bbox, axis=0) - 0.5
    bbox_max = np.max(bbox, axis=0) + 0.5

    x = np.linspace(bbox_min[0], bbox_max[0], res)
    y = np.linspace(bbox_min[1], bbox_max[1], res)
    z = np.linspace(bbox_min[2], bbox_max[2], res)
    gx, gy, gz = np.meshgrid(x, y, z, indexing="ij")
    coords = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1).astype(np.float32)

    queries = torch.tensor(coords, device="cuda:0", dtype=torch.float32)

    src_geo = extract_blender_mesh_data(src_obj, mode=mode)
    tgt_geo = extract_blender_mesh_data(tgt_obj, mode=mode)

    w_src = compute_winding_fields(src_geo, queries, mode=mode)
    w_tgt = compute_winding_fields(tgt_geo, queries, mode=mode)

    diff = w_src - w_tgt
    M = queries.shape[0]

    # Compute dL/dw
    if props.loss_type == "L1":
        dL_dw = torch.sign(diff) / M
    else:
        dL_dw = (2.0 * diff) / M

    # Backpropagate to geometry
    grads = compute_geometry_gradients(dL_dw, src_geo, queries, mode=mode)
    grads_np = grads.cpu().numpy()

    # Step Vertices
    mesh = src_obj.data
    if mode == "MESH":
        v_coords = np.empty((len(mesh.vertices), 3), dtype=np.float32)
        mesh.vertices.foreach_get("co", v_coords.ravel())
        v_coords -= lr * grads_np
        mesh.vertices.foreach_set("co", v_coords.ravel())
        mesh.update()
    elif mode == "POINT_NORMAL":
        # Displace vertices of each triangle using point/scaled normal gradients
        pass


def register_handlers():
    bpy.app.handlers.depsgraph_update_post.append(on_depsgraph_update)
    bpy.app.handlers.frame_change_pre.append(on_frame_change_optimization)


def unregister_handlers():
    if on_depsgraph_update in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(on_depsgraph_update)
    if on_frame_change_optimization in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.remove(on_frame_change_optimization)
