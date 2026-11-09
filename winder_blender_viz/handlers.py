import time
import numpy as np
import bpy

from .operators import (
    recompute_winding_field,
    recompute_payload_field,
    recompute_payload_query_gradients,
    recompute_payload_query_streamlines,
    _recompute_streamlines,
    _recompute_query_gradients,
    _refresh_iso,
)

# Per-object tracking (was a single global name before)
_PENDING_OBJS: set[str] = set()
_TIMER_SCHEDULED = False
_LAST_UPDATE_TIME = 0.0
_LAST_FINGERPRINTS: dict[str, tuple] = {}
_LAST_FRAME = None
_DEBOUNCE_DELAY = 0.05

_VIS_PREFIXES = (
    "WindingVol_",
    "WindingQuiver_",
    "PayloadVol_",
    "PayloadGradVol_",
    "Streamlines_",
    "PayloadStreamlines_",
    "QueryGradQuiver_",
)


def _get_geometry_fingerprint(obj):
    """Fingerprint of transform, evaluated vertices, and color attribute.

    Includes the color hash so vertex-painting triggers a payload recompute,
    and uses the *evaluated* mesh so modifier / geometry-node changes are
    picked up.
    """
    matrix_sum = float(np.sum(obj.matrix_world))

    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    mesh_eval = obj_eval.to_mesh()

    try:
        v_count = len(mesh_eval.vertices)
        if v_count == 0:
            return (matrix_sum, 0, 0.0, 0.0)

        v0 = mesh_eval.vertices[0].co
        vm = mesh_eval.vertices[v_count // 2].co
        v1 = mesh_eval.vertices[-1].co
        co_sum = float(v0.x + v0.y + vm.z + v1.x + v1.y + v1.z)

        color_hash = 0.0
        if mesh_eval.color_attributes:
            attr = mesh_eval.color_attributes.active_color
            if attr and len(attr.data) > 0:
                n = len(attr.data)
                stride = max(1, n // 64)
                for i in range(0, n, stride):
                    c = attr.data[i].color
                    color_hash += c[0] + c[1] * 2.0 + c[2] * 3.0
                color_hash = round(color_hash, 4)

        return (matrix_sum, v_count, round(co_sum, 5), color_hash)
    finally:
        obj_eval.to_mesh_clear()


def _has_winder_visualization(name: str) -> bool:
    """True if `name` has any Winder-created helper object."""
    for prefix in _VIS_PREFIXES:
        if bpy.data.objects.get(f"{prefix}{name}"):
            return True
    return False


def _recompute_all_for_object(context, obj):
    """Update every existing Winder visualization for `obj`.

    Never creates new objects — only refreshes what already exists.
    """
    name = obj.name

    if bpy.data.objects.get(f"WindingVol_{name}"):
        try:
            recompute_winding_field(context, obj)
        except Exception as e:
            print(f"[Winder] Live winding recompute error for '{name}': {e}")

    if bpy.data.objects.get(f"PayloadVol_{name}"):
        try:
            recompute_payload_field(context, obj)
        except Exception as e:
            print(f"[Winder] Live payload recompute error for '{name}': {e}")

    if bpy.data.objects.get(f"PayloadGradVol_{name}"):
        try:
            recompute_payload_query_gradients(context, obj)
        except Exception as e:
            print(
                f"[Winder] Live payload grad recompute error for "
                f"'{name}': {e}"
            )

    if bpy.data.objects.get(f"Streamlines_{name}"):
        try:
            _recompute_streamlines(context, obj)
        except Exception as e:
            print(f"[Winder] Live streamline recompute error for '{name}': {e}")

    if bpy.data.objects.get(f"PayloadStreamlines_{name}"):
        try:
            recompute_payload_query_streamlines(context, obj)
        except Exception as e:
            print(
                f"[Winder] Live payload streamline recompute error for "
                f"'{name}': {e}"
            )
    if bpy.data.objects.get(f"PayloadIsosurface_{name}"):
        try:
            thresh = float(
                bpy.data.objects[f"PayloadIsosurface_{name}"].get(
                    "_winder_payload_iso_threshold",
                    bpy.context.scene.winder_props.payload_isosurface_threshold,
                )
            )
            _refresh_iso(context, obj, thresh)
        except Exception as e:
            print(f"[Winder] Live iso refresh error for '{name}': {e}")

    if bpy.data.objects.get(f"QueryGradQuiver_{name}"):
        try:
            _recompute_query_gradients(context, obj)
        except Exception as e:
            print(f"[Winder] Live query grad recompute error for '{name}': {e}")


def _deferred_recompute_callback():
    global _TIMER_SCHEDULED, _PENDING_OBJS, _LAST_UPDATE_TIME, _LAST_FRAME

    scene = bpy.context.scene
    current_frame = scene.frame_current
    now = time.time()
    elapsed = now - _LAST_UPDATE_TIME

    is_frame_change = current_frame != _LAST_FRAME
    if not is_frame_change and elapsed < _DEBOUNCE_DELAY:
        return _DEBOUNCE_DELAY - elapsed

    _TIMER_SCHEDULED = False

    if not _PENDING_OBJS:
        return None

    pending = list(_PENDING_OBJS)
    _PENDING_OBJS.clear()

    for obj_name in pending:
        obj = bpy.data.objects.get(obj_name)
        if not obj or obj.type != "MESH":
            continue

        current_fingerprint = _get_geometry_fingerprint(obj)
        last_fp = _LAST_FINGERPRINTS.get(obj_name)

        if is_frame_change or current_fingerprint != last_fp:
            _LAST_FINGERPRINTS[obj_name] = current_fingerprint
            _recompute_all_for_object(bpy.context, obj)

    _LAST_FRAME = current_frame
    return None


def _schedule_recompute(obj_name, immediate=False):
    global _TIMER_SCHEDULED, _PENDING_OBJS, _LAST_UPDATE_TIME

    _PENDING_OBJS.add(obj_name)
    _LAST_UPDATE_TIME = time.time()

    if not _TIMER_SCHEDULED:
        _TIMER_SCHEDULED = True
        delay = 0.0 if immediate else _DEBOUNCE_DELAY
        bpy.app.timers.register(_deferred_recompute_callback, first_interval=delay)


@bpy.app.handlers.persistent
def winder_depsgraph_update_handler(scene, depsgraph):
    # Fast lookup of every object that currently has a Winder
    # visualization attached. No dependency on `winder_active_target`.
    winder_sources: dict[int, str] = {}
    winder_data: dict[int, str] = {}
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if _has_winder_visualization(obj.name):
            winder_sources[obj.as_pointer()] = obj.name
            winder_data[obj.data.as_pointer()] = obj.name

    if not winder_sources and not winder_data:
        return

    scheduled: set[str] = set()
    for update in depsgraph.updates:
        up_id = update.id
        up_orig = getattr(up_id, "original", up_id)
        up_ptr = up_orig.as_pointer()

        name = winder_sources.get(up_ptr) or winder_data.get(up_ptr)
        if name and name not in scheduled:
            scheduled.add(name)
            _schedule_recompute(name, immediate=False)


@bpy.app.handlers.persistent
def winder_frame_change_handler(scene):
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if _has_winder_visualization(obj.name):
            _schedule_recompute(obj.name, immediate=True)


def register_handlers():
    if winder_depsgraph_update_handler not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(winder_depsgraph_update_handler)

    if winder_frame_change_handler not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(winder_frame_change_handler)


def unregister_handlers():
    global _TIMER_SCHEDULED, _PENDING_OBJS, _LAST_FINGERPRINTS, _LAST_FRAME
    _TIMER_SCHEDULED = False
    _PENDING_OBJS.clear()
    _LAST_FINGERPRINTS.clear()
    _LAST_FRAME = None

    if winder_depsgraph_update_handler in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(winder_depsgraph_update_handler)

    if winder_frame_change_handler in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(winder_frame_change_handler)
