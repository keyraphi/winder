import time
import numpy as np
import bpy

from .operators import recompute_winding_field

_RECOMPUTE_TIMER_SCHEDULED = False
_TARGET_OBJ_NAME = None
_LAST_UPDATE_TIME = 0.0
_LAST_GEO_FINGERPRINT = None
_LAST_FRAME = None
_DEBOUNCE_DELAY = 0.05  # 50ms debounce window for interactive edits


def _get_geometry_fingerprint(obj):
    """Computes a lightweight fingerprint of transform and evaluated (post-modifier) vertex coordinates."""
    matrix_sum = float(np.sum(obj.matrix_world))

    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    mesh_eval = obj_eval.to_mesh()

    try:
        v_count = len(mesh_eval.vertices)
        if v_count == 0:
            return (matrix_sum, 0, 0.0)

        v0 = mesh_eval.vertices[0].co
        vm = mesh_eval.vertices[v_count // 2].co
        v1 = mesh_eval.vertices[-1].co
        co_sum = float(v0.x + v0.y + vm.z + v1.x + v1.y + v1.z)

        return (matrix_sum, v_count, round(co_sum, 5))
    finally:
        obj_eval.to_mesh_clear()


def _deferred_recompute_callback():
    """Executes on Blender's main thread after debounce or immediately on frame change."""
    global \
        _RECOMPUTE_TIMER_SCHEDULED, \
        _TARGET_OBJ_NAME, \
        _LAST_UPDATE_TIME, \
        _LAST_GEO_FINGERPRINT, \
        _LAST_FRAME

    scene = bpy.context.scene
    current_frame = scene.frame_current
    now = time.time()
    elapsed = now - _LAST_UPDATE_TIME

    # If remaining on the same frame, apply the debounce window for interactive edits
    is_frame_change = current_frame != _LAST_FRAME
    if not is_frame_change and elapsed < _DEBOUNCE_DELAY:
        return _DEBOUNCE_DELAY - elapsed

    _RECOMPUTE_TIMER_SCHEDULED = False

    if not _TARGET_OBJ_NAME:
        return None

    obj = bpy.data.objects.get(_TARGET_OBJ_NAME)
    if obj and obj.type == "MESH":
        current_fingerprint = _get_geometry_fingerprint(obj)

        if is_frame_change or current_fingerprint != _LAST_GEO_FINGERPRINT:
            _LAST_FRAME = current_frame
            _LAST_GEO_FINGERPRINT = current_fingerprint
            try:
                recompute_winding_field(bpy.context, obj)
            except Exception as e:
                print(f"[Winder] Live recompute error: {e}")

    return None  # Unregisters timer


def _schedule_recompute(obj_name, immediate=False):
    """Schedules the deferred recompute timer."""
    global _RECOMPUTE_TIMER_SCHEDULED, _TARGET_OBJ_NAME, _LAST_UPDATE_TIME

    _TARGET_OBJ_NAME = obj_name
    _LAST_UPDATE_TIME = time.time()

    if not _RECOMPUTE_TIMER_SCHEDULED:
        _RECOMPUTE_TIMER_SCHEDULED = True
        delay = 0.0 if immediate else _DEBOUNCE_DELAY
        bpy.app.timers.register(_deferred_recompute_callback, first_interval=delay)


@bpy.app.handlers.persistent
def winder_depsgraph_update_handler(scene, depsgraph):
    """Monitors Depsgraph updates for target object transforms, mesh edits, or modifier tweaks."""
    active_name = scene.get("winder_active_target")
    if not active_name:
        return

    obj = scene.objects.get(active_name)
    if not obj or obj.type != "MESH":
        return

    target_obj_ptr = obj.as_pointer()
    target_data_ptr = obj.data.as_pointer()

    for update in depsgraph.updates:
        up_id = update.id
        up_orig = getattr(up_id, "original", up_id)
        up_ptr = up_orig.as_pointer()

        if up_ptr == target_obj_ptr or up_ptr == target_data_ptr:
            _schedule_recompute(obj.name, immediate=False)
            break


@bpy.app.handlers.persistent
def winder_frame_change_handler(scene):
    """Monitors timeline frame changes (animation playback and timeline scrubbing)."""
    active_name = scene.get("winder_active_target")
    if not active_name:
        return

    obj = scene.objects.get(active_name)
    if not obj or obj.type != "MESH":
        return

    _schedule_recompute(obj.name, immediate=True)


def register_handlers():
    if winder_depsgraph_update_handler not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(winder_depsgraph_update_handler)

    if winder_frame_change_handler not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(winder_frame_change_handler)


def unregister_handlers():
    global \
        _RECOMPUTE_TIMER_SCHEDULED, \
        _TARGET_OBJ_NAME, \
        _LAST_GEO_FINGERPRINT, \
        _LAST_FRAME
    _RECOMPUTE_TIMER_SCHEDULED = False
    _TARGET_OBJ_NAME = None
    _LAST_GEO_FINGERPRINT = None
    _LAST_FRAME = None

    if winder_depsgraph_update_handler in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(winder_depsgraph_update_handler)

    if winder_frame_change_handler in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(winder_frame_change_handler)
