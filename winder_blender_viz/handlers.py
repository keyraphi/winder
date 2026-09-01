import time
import bmesh
import numpy as np
import bpy

from .operators import recompute_winding_field

_RECOMPUTE_TIMER_SCHEDULED = False
_TARGET_OBJ_NAME = None
_LAST_UPDATE_TIME = 0.0
_LAST_GEO_FINGERPRINT = None
_DEBOUNCE_DELAY = 0.05  # 50ms debounce window


def _get_geometry_fingerprint(obj):
    """Computes a lightweight fingerprint of transform and vertex positions

    to prevent recomputing when only selection state or UI mode changes.
    """
    matrix_sum = float(np.sum(obj.matrix_world))

    if obj.mode == "EDIT":
        bm = bmesh.from_edit_mesh(obj.data)
        if not bm.verts:
            return (matrix_sum, 0, 0.0)
        # Fast sum of first, middle, and last vertex coordinates + vertex count
        v_count = len(bm.verts)
        co_sum = float(
            bm.verts[0].co.x + bm.verts[v_count // 2].co.y + bm.verts[-1].co.z
        )
        return (matrix_sum, v_count, round(co_sum, 5))
    else:
        v_count = len(obj.data.vertices)
        if v_count == 0:
            return (matrix_sum, 0, 0.0)
        # Use first and last vertex coords from mesh data
        v0 = obj.data.vertices[0].co
        v1 = obj.data.vertices[-1].co
        co_sum = float(v0.x + v0.y + v1.z)
        return (matrix_sum, v_count, round(co_sum, 5))


def _deferred_recompute_callback():
    """Executes on Blender's main thread after the debounce delay elapses."""
    global \
        _RECOMPUTE_TIMER_SCHEDULED, \
        _TARGET_OBJ_NAME, \
        _LAST_UPDATE_TIME, \
        _LAST_GEO_FINGERPRINT

    now = time.time()
    elapsed = now - _LAST_UPDATE_TIME

    # If updates arrived recently, delay execution to trail the continuous edit
    if elapsed < _DEBOUNCE_DELAY:
        return _DEBOUNCE_DELAY - elapsed

    _RECOMPUTE_TIMER_SCHEDULED = False

    if not _TARGET_OBJ_NAME:
        return None

    obj = bpy.data.objects.get(_TARGET_OBJ_NAME)
    if obj and obj.type == "MESH":
        # Validate that actual geometry/transform coordinates changed
        current_fingerprint = _get_geometry_fingerprint(obj)
        if current_fingerprint != _LAST_GEO_FINGERPRINT:
            _LAST_GEO_FINGERPRINT = current_fingerprint
            try:
                recompute_winding_field(bpy.context, obj)
            except Exception as e:
                print(f"[Winder] Live recompute error: {e}")

    return None  # Unregisters timer


@bpy.app.handlers.persistent
def winder_depsgraph_update_handler(scene, depsgraph):
    """Monitors Depsgraph updates for target object matrix or geometry edits."""
    global _RECOMPUTE_TIMER_SCHEDULED, _TARGET_OBJ_NAME, _LAST_UPDATE_TIME

    active_name = scene.get("winder_active_target")
    if not active_name:
        return

    obj = scene.objects.get(active_name)
    if not obj or obj.type != "MESH":
        return

    # Cache pointers for robust comparison against dynamic Blender RNA wrappers
    target_obj_ptr = obj.as_pointer()
    target_data_ptr = obj.data.as_pointer()

    target_updated = False
    for update in depsgraph.updates:
        up_id = update.id
        up_orig = getattr(up_id, "original", up_id)
        up_ptr = up_orig.as_pointer()

        if up_ptr == target_obj_ptr or up_ptr == target_data_ptr:
            if update.is_updated_transform or update.is_updated_geometry:
                target_updated = True
                break

    if target_updated:
        _TARGET_OBJ_NAME = obj.name
        _LAST_UPDATE_TIME = time.time()

        if not _RECOMPUTE_TIMER_SCHEDULED:
            _RECOMPUTE_TIMER_SCHEDULED = True
            bpy.app.timers.register(
                _deferred_recompute_callback, first_interval=_DEBOUNCE_DELAY
            )


def register_handlers():
    if winder_depsgraph_update_handler not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(winder_depsgraph_update_handler)


def unregister_handlers():
    global _RECOMPUTE_TIMER_SCHEDULED, _TARGET_OBJ_NAME, _LAST_GEO_FINGERPRINT
    _RECOMPUTE_TIMER_SCHEDULED = False
    _TARGET_OBJ_NAME = None
    _LAST_GEO_FINGERPRINT = None

    if winder_depsgraph_update_handler in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(winder_depsgraph_update_handler)
