"""extract_shell.py — turn room.usdz into a compact, editable semantic SHELL (replacing the binary usdz).

(en) Extract `room.usdz` into a compact, editable semantic SHELL (replaces the binary usdz).
Instead of dumping 4x4 matrices, emit a floorplan-style schema an agent can read + edit:
  walls (start/end/thickness), openings (wall/type/offset/width/height/sill),
  objects (category/center/size/yaw), floor polygon, floor_z/ceiling_z, root_matrix (ARKit→Blender).

Rather than dumping 4x4 matrices, this writes floor-plan semantics an agent can read and edit:
  walls:    per wall  = {start:[x,y], end:[x,y], thickness}   (height = ceiling_z - floor_z)
  openings: per door/window = {wall, type, offset (distance along the wall from its start),
            width, height, sill (height above the floor)}
  objects:  per object box = {category, center:[x,y,z], size:[w,d,h], yaw (degrees)}
  floor:    the floor polygon (world vertices/faces, kept faithful)
  + floor_z / ceiling_z / root_matrix (the ARKit->Blender calibration, used by the cameras)

  blender -b --python extract_shell.py -- <room.usdz> <out.json>
"""

import json
import math
import re
import sys
import zipfile

import bpy
from mathutils import Vector

_argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
USDZ, OUT = _argv[0], _argv[1]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.usd_import(filepath=USDZ)


def box_frame(o, prefer_long=True):
    cols = [o.matrix_world.to_3x3().col[i] for i in range(3)]
    L = [c.length for c in cols]
    vert = max(range(3), key=lambda i: abs(cols[i].normalized().z))
    horis = [i for i in range(3) if i != vert]
    if prefer_long:
        # walls: the long horizontal axis IS the wall direction — pick it.
        wi, di = horis if L[horis[0]] >= L[horis[1]] else horis[::-1]
    else:
        # objects: keep the object's OWN local axes (local X = wi). Picking by length
        # rotates objects whose local-Y is longer by 90° — that is the chair-orientation
        # bug. The local frame already carries the RoomPlan rotation; don't re-derive it.
        wi, di = horis[0], horis[1]
    up = cols[vert].normalized()
    if up.z < 0:
        up = -up
    wdir = Vector((cols[wi].x, cols[wi].y, 0.0))
    if wdir.length < 1e-6:
        wdir = Vector((1.0, 0.0, 0.0))
    wdir.normalize()
    return o.matrix_world.translation.copy(), up, wdir, up.cross(wdir), L[wi], L[di], L[vert]


walls, objects, openings_raw = {}, {}, {}
floor_z = ceiling_z = None
floor_data = None

for o in bpy.data.objects:
    if o.type != "MESH":
        continue
    name = o.name
    c, up, wdir, ddir, W, D, H = box_frame(o)
    if re.match(r"Wall\d+$", name):
        half = wdir * (W / 2)
        walls[name] = {
            "start": [round(c.x - half.x, 4), round(c.y - half.y, 4)],
            "end": [round(c.x + half.x, 4), round(c.y + half.y, 4)],
            "thickness": round(D, 4),
        }
        zb, zt = c.z - H / 2, c.z + H / 2
        floor_z = zb if floor_z is None else min(floor_z, zb)
        ceiling_z = zt if ceiling_z is None else max(ceiling_z, zt)
    elif re.match(r"(Door|Window|Opening)\d+$", name):
        openings_raw[name] = {"c": c.copy(), "W": W, "H": H, "zb": c.z - H / 2}
    elif re.match(r"Floor\d+$", name):
        floor_data = {
            "verts": [[round(v, 4) for v in (o.matrix_world @ vt.co)] for vt in o.data.vertices],
            "faces": [[int(i) for i in p.vertices] for p in o.data.polygons],
        }
    else:
        # objects: read the object's OWN local frame (prefer_long=False) so yaw is the
        # true RoomPlan rotation, not the longer-axis heuristic (which 90°-rotates pieces
        # whose local-Y is longer). The usdz matrix already carries the full orientation.
        c, up, wdir, ddir, W, D, H = box_frame(o, prefer_long=False)
        objects[name] = {
            "category": re.sub(r"\d+$", "", name).lower(),
            "center": [round(c.x, 4), round(c.y, 4), round(c.z, 4)],
            "size": [round(W, 4), round(D, 4), round(H, 4)],
            "yaw": round(math.degrees(math.atan2(wdir.y, wdir.x)), 2),
        }

# Openings -> the wall they belong to, plus along-the-wall semantics
z = zipfile.ZipFile(USDZ)
op2wall = {}
for nm in z.namelist():
    g = re.search(r"Walls/(Wall\d+)/([A-Za-z0-9]+)\.usda$", nm)
    if g and g.group(2) != g.group(1):
        op2wall[g.group(2)] = g.group(1)

openings = {}
for name, d in openings_raw.items():
    wall = op2wall.get(name)
    offset = 0.0
    if wall and wall in walls:
        ws = Vector(walls[wall]["start"] + [0.0])
        we = Vector(walls[wall]["end"] + [0.0])
        wd = we - ws
        wd.normalize()
        offset = round((Vector((d["c"].x, d["c"].y, 0.0)) - ws).dot(wd), 4)
    typ = (
        "door" if name.startswith("Door") else "window" if name.startswith("Window") else "opening"
    )
    openings[name] = {
        "wall": wall,
        "type": typ,
        "offset": offset,
        "width": round(d["W"], 4),
        "height": round(d["H"], 4),
        "sill": round(d["zb"] - floor_z, 4),
    }

root = bpy.data.objects.get("room")
root_matrix = (
    [round(root.matrix_world[r][cc], 6) for r in range(4) for cc in range(4)] if root else None
)

SHELL = {
    "floor_z": round(floor_z, 4),
    "ceiling_z": round(ceiling_z, 4),
    "walls": walls,
    "openings": openings,
    "objects": objects,
    "floor": floor_data,
    "root_matrix": root_matrix,
}
with open(OUT, "w") as fh:
    json.dump(SHELL, fh, indent=2)
print(f"wrote {OUT}: {len(walls)} walls, {len(openings)} openings, {len(objects)} objects")
