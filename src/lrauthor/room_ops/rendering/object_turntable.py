"""objview.py — Blender headless: import a GLB, render N turntable views to PNGs.
Run:  blender -b --python objview.py -- <in.glb> <out_dir> <n_views> [open_frac]

open_frac (default 0.0) chooses the articulation pose to render:
  0.0 = closed (frame 1);  1.0 = the most-open pose;  in between = partly open.
The most-open frame is found by scanning the baked swing/slide animation for the frame whose
articulated parts are displaced furthest from the closed pose — robust to any open-hold-close
schedule. Lets the refinement loop see (and check) that a door/window actually opens.
"""
import math
import os
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
glb, out_dir = argv[0], argv[1]
n = int(argv[2]) if len(argv) > 2 else 4
open_frac = float(argv[3]) if len(argv) > 3 else 0.0
os.makedirs(out_dir, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)
scn = bpy.context.scene
meshes = [o for o in bpy.data.objects if o.type == "MESH"]


def world_bbox_pts():
    """World-space bound-box corners of every mesh AT THE CURRENT FRAME, read from the evaluated
    depsgraph so baked articulation (swing/slide) is reflected — not the rest pose."""
    dg = bpy.context.evaluated_depsgraph_get()
    pts = []
    for o in meshes:
        ev = o.evaluated_get(dg)
        pts.extend(ev.matrix_world @ Vector(c) for c in ev.bound_box)
    return pts


# pose the articulation: closed by default, or the most-open frame when open_frac > 0
if open_frac > 0:
    fs, fe = scn.frame_start, scn.frame_end
    if fe > fs:
        scn.frame_set(fs)
        base = world_bbox_pts()
        best_f, best_d = fs, -1.0
        for f in range(fs, fe + 1):
            scn.frame_set(f)
            cur = world_bbox_pts()
            d = sum((cur[i] - base[i]).length for i in range(min(len(cur), len(base))))
            if d > best_d:
                best_d, best_f = d, f
        scn.frame_set(int(round(fs + open_frac * (best_f - fs))))

# bounds of all mesh objects AT THE CURRENT POSE (so an open door stays framed)
mins = Vector((1e9, 1e9, 1e9))
maxs = Vector((-1e9, -1e9, -1e9))
for w in world_bbox_pts():
    mins = Vector((min(mins[i], w[i]) for i in range(3)))
    maxs = Vector((max(maxs[i], w[i]) for i in range(3)))
center = (mins + maxs) / 2
radius = max((maxs - mins).length / 2, 0.3)

# neutral studio: world light + a sun
world = bpy.data.worlds.new("w")
bpy.context.scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[1].default_value = 1.2
sun = bpy.data.lights.new("sun", "SUN")
sun.energy = 3.0
so = bpy.data.objects.new("sun", sun)
bpy.context.collection.objects.link(so)
so.rotation_euler = (math.radians(55), 0, math.radians(30))

cam_data = bpy.data.cameras.new("cam")
cam = bpy.data.objects.new("cam", cam_data)
bpy.context.collection.objects.link(cam)
bpy.context.scene.camera = cam
engines = bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items.keys()
scn.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "CYCLES"
if scn.render.engine == "CYCLES":
    scn.cycles.samples = 32
scn.render.resolution_x = scn.render.resolution_y = 512
scn.render.film_transparent = False

dist = radius * 3.2
elev = math.radians(20)
for i in range(n):
    az = math.radians(35 + i * (360.0 / n))  # front-ish first, then around
    x = center.x + dist * math.cos(elev) * math.cos(az)
    y = center.y + dist * math.cos(elev) * math.sin(az)
    z = center.z + dist * math.sin(elev)
    cam.location = (x, y, z)
    d = center - cam.location
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    scn.render.filepath = os.path.join(out_dir, f"view_{i}.png")
    bpy.ops.render.render(write_still=True)
print(f"[objview] {n} views (open_frac={open_frac}) -> {out_dir}")
