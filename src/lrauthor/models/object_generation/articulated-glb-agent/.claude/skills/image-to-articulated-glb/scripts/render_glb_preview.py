"""Render-check a GLB headlessly: auto-framed camera, closed + open frames,
plus a world-bbox printout per mesh (catches scale/offset bugs the eye misses).

Usage:
  blender -b --factory-startup --python render_glb_preview.py -- \
      <model.glb> <out_dir> [--frames 1:closed,62:open] [--res 1100x720] \
      [--samples 48] [--view iso|front|side]

Defaults: frames = "1:closed" plus an auto mid-animation "open" frame,
iso view, Cycles 48 samples (headless EEVEE is unreliable on macOS).
"""

import math
import os
import sys

import bpy
import mathutils


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(argv) < 2:
        print(__doc__)
        sys.exit(1)
    opts = {
        "glb": argv[0],
        "out": argv[1],
        "frames": None,
        "res": (1100, 720),
        "samples": 48,
        "view": "iso",
    }
    i = 2
    while i < len(argv):
        if argv[i] == "--frames":
            opts["frames"] = [
                (int(p.split(":")[0]), p.split(":")[1]) for p in argv[i + 1].split(",")
            ]
            i += 2
        elif argv[i] == "--res":
            w, h = argv[i + 1].split("x")
            opts["res"] = (int(w), int(h))
            i += 2
        elif argv[i] == "--samples":
            opts["samples"] = int(argv[i + 1])
            i += 2
        elif argv[i] == "--view":
            opts["view"] = argv[i + 1]
            i += 2
        else:
            i += 1
    return opts


VIEW_DIRS = {  # camera offset direction (Blender Z-up, front = -Y)
    "iso": (-0.85, -1.0, 0.55),
    "front": (0.0, -1.0, 0.25),
    "side": (-1.0, -0.15, 0.3),
}


def scene_bbox():
    mn = mathutils.Vector((1e9, 1e9, 1e9))
    mx = -mn.copy()
    for ob in bpy.data.objects:
        if ob.type != "MESH":
            continue
        for c in ob.bound_box:
            p = ob.matrix_world @ mathutils.Vector(c)
            mn = mathutils.Vector(map(min, mn, p))
            mx = mathutils.Vector(map(max, mx, p))
    return mn, mx


def print_bboxes():
    print("--- world bounding boxes ---")
    for ob in sorted(bpy.data.objects, key=lambda o: o.name):
        if ob.type != "MESH":
            continue
        pts = [ob.matrix_world @ mathutils.Vector(c) for c in ob.bound_box]
        mn = [round(min(p[i] for p in pts), 3) for i in range(3)]
        mx = [round(max(p[i] for p in pts), 3) for i in range(3)]
        print(f"BBOX {ob.name:24s} min={mn} max={mx}")


def main():
    opts = parse_args()
    os.makedirs(opts["out"], exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)  # no default cube!
    bpy.ops.import_scene.gltf(filepath=opts["glb"])
    scene = bpy.context.scene

    # The importer only activates the FIRST animation clip; assign every
    # action to its object by name prefix so all parts move in the open
    # render (Blender 5.x slotted actions).
    for act in bpy.data.actions:
        cands = [o for o in bpy.data.objects if act.name.startswith(o.name)]
        if not cands:
            continue
        ob = max(cands, key=lambda o: len(o.name))
        if ob.animation_data is None:
            ob.animation_data_create()
        # Re-assigning an already-active action clears its slot binding on
        # Blender 4.5 (object freezes at rest pose) — leave it untouched.
        if ob.animation_data.action is act:
            continue
        # glTF imports rotations as quaternion fcurves; the object must be in
        # QUATERNION mode or the keyframes silently don't drive it (renders the
        # door "closed" at every frame).
        ob.rotation_mode = "QUATERNION"
        ob.animation_data.action = act
        try:
            ob.animation_data.action_slot = act.slots[0]
        except Exception:
            pass

    print_bboxes()

    # default frames: closed + auto open frame at ~40% of the longest action
    frames = opts["frames"]
    if frames is None:
        max_f = max((a.frame_range[1] for a in bpy.data.actions), default=0)
        frames = [(1, "closed")]
        if max_f > 2:
            frames.append((int(1 + 0.4 * (max_f - 1)), "open"))

    # auto-framed camera: track an empty at the bbox center
    mn, mx = scene_bbox()
    center = (mn + mx) / 2
    radius = max((mx - mn).length / 2, 0.1)
    direction = mathutils.Vector(VIEW_DIRS.get(opts["view"], VIEW_DIRS["iso"]))
    direction.normalize()

    cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam"))
    scene.collection.objects.link(cam)
    cam.location = center + direction * radius * 2.6
    target = bpy.data.objects.new("Target", None)
    target.location = center
    scene.collection.objects.link(target)
    cam.constraints.new("TRACK_TO").target = target
    scene.camera = cam

    sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
    sun.data.energy = 4.0
    sun.rotation_euler = (math.radians(45), math.radians(-15), math.radians(-35))
    scene.collection.objects.link(sun)
    world = bpy.data.worlds.new("W")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.9, 0.9, 0.95, 1)
    bg.inputs[1].default_value = 0.7

    scene.render.engine = "CYCLES"
    scene.cycles.samples = opts["samples"]
    scene.render.resolution_x, scene.render.resolution_y = opts["res"]

    for frame, name in frames:
        scene.frame_set(frame)
        scene.render.filepath = os.path.join(opts["out"], f"preview_{name}.png")
        bpy.ops.render.render(write_still=True)
        print("RENDERED:", scene.render.filepath)
    print("RENDER DONE")


main()
