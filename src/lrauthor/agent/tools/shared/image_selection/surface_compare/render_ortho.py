import json
import os
import re
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

# _REPO is the CHECKOUT (data); the sibling tool below is resolved from __file__ instead.
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
# render_room_cameras is package code under room_ops. This used to join "../../room_render",
# a sibling relationship that stopped holding when image_selection moved under agent/tools.
_PKG = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(_PKG) != "lrauthor" and os.path.dirname(_PKG) != _PKG:
    _PKG = os.path.dirname(_PKG)
sys.path.insert(0, os.path.join(_PKG, "room_ops", "rendering", "room_render"))
import render_room_cameras as R

argv = sys.argv[sys.argv.index("--") + 1 :]
scan_dir, assets, out_dir = argv[0], argv[1], argv[2]
os.makedirs(out_dir, exist_ok=True)
scene = R.build_scene(scan_dir, assets, out_dir)
R.setup_render(engine="EEVEE", light=True)
sc = bpy.context.scene
sc.view_settings.exposure = -1.0

FURN_CATS = {
    "chair",
    "table",
    "sofa",
    "bed",
    "stool",
    "bench",
    "desk",
    "television",
    "monitor",
    "lamp",
    "plant",
    "rug",
    "appliance",
}
FURN_NAME = (
    "chair",
    "table",
    "sofa",
    "bed",
    "stool",
    "bench",
    "desk",
    "television",
    "monitor",
    "lamp",
    "plant",
    "chaircluster",
)
catmap = {}
for rid, rec in getattr(scene, "objects", {}).items():
    h = bpy.data.objects.get(rec.get("handle"))
    if not h:
        continue
    stk = [h]
    while stk:
        x = stk.pop()
        stk += list(x.children)
        catmap[x.name] = rec.get("category")
is_wallpart = lambda o: bool(
    re.match(r"Wall\d+(_|$)", o.name)
)  # solid wall OR glazing/frame component
is_floor = lambda o: bool(re.match(r"Floor\d+$", o.name))
is_ceil = lambda o: bool(
    re.match(r"Ceiling\d+$", o.name)
)  # the surface only; Ceiling_Vent0 etc. are fixtures
def _lineage_colls(o):
    """Collection names of the object AND all its ancestors (imported GLB children are meshes
    named geometry_* whose only reliable classification is the Furniture/Openings collection)."""
    names = set()
    a = o
    while a is not None:
        try:
            names |= {c.name for c in a.users_collection}
        except Exception:
            pass
        a = a.parent
    return names


def is_furn(o):
    cols = _lineage_colls(o)
    if "Furniture" in cols:
        return True  # chairs/tables/etc — NEVER drawn on a wall ortho
    if "Openings" in cols:
        return False  # doors/windows stay visible on their wall
    return (catmap.get(o.name) in FURN_CATS) or any(k in o.name.lower() for k in FURN_NAME)


def own_verts(o):
    if o.type != "MESH" or not o.data.vertices:
        return None
    mw = o.matrix_world
    return np.array([list(mw @ v.co) for v in o.data.vertices])


def face_normal(o):
    me = o.data
    if not me.polygons:
        return None
    p = max(me.polygons, key=lambda f: f.area)
    n = np.array(o.matrix_world.to_3x3() @ p.normal, float)
    return n / (np.linalg.norm(n) + 1e-9)


def axes(V):
    c = V.mean(0)
    cc = V - c
    _, _, vh = np.linalg.svd(cc, full_matrices=False)
    return c, vh, np.array([np.ptp(cc @ a) for a in vh])


def center_of(o):
    V = own_verts(o)
    return V.mean(0) if V is not None else np.array(o.matrix_world.translation)


def group_verts(group):
    Vs = [own_verts(o) for o in group]
    Vs = [v for v in Vs if v is not None]
    return np.vstack(Vs) if Vs else None


def group_normal(group):  # wall normal = thinnest PCA axis of the
    V = group_verts(group)  # whole group (robust for thin/glazed walls
    if V is None or len(V) < 3:  # where the largest face is a mullion side)
        return np.array([0.0, 0.0, 1.0])
    _, ax, ext = axes(V)
    return ax[int(np.argmin(ext))]


meshes = [o for o in bpy.data.objects if o.type == "MESH"]
# a wall = the GROUP of all Wall<N>* meshes (solid wall + glazing/mullions/jambs/openings)
wall_idx = sorted({int(re.match(r"Wall(\d+)", o.name).group(1)) for o in meshes if is_wallpart(o)})
wall_groups = {N: [o for o in meshes if re.match(rf"Wall{N}(_|$)", o.name)] for N in wall_idx}
floors = [o for o in meshes if is_floor(o)]
ceils = [o for o in meshes if is_ceil(o)]
surfaces = (
    [(f"Wall{N}", wall_groups[N], "wall") for N in wall_idx]
    + [(o.name, [o], "floor") for o in floors]
    + [(o.name, [o], "ceiling") for o in ceils]
)
fixtures = [
    o
    for o in meshes
    if not is_wallpart(o) and not is_floor(o) and not is_ceil(o) and not is_furn(o)
]

allc = np.mean([center_of(o) for o in meshes], axis=0)
fl = floors[0] if floors else None
up = face_normal(fl) if fl is not None else np.array([0.0, 0.0, 1.0])
if up is None:
    up = np.array([0.0, 0.0, 1.0])
if fl is not None and np.dot(allc - center_of(fl), up) < 0:
    up = -up


def floor_u_axis(mesh, n):
    """Match room_pipeline: u = LONGEST horizontal boundary edge of the floor outline
    (follows a wall), NOT a PCA axis (which picks an ambiguous diagonal on square rooms)."""
    me = mesh.data
    mw = mesh.matrix_world
    from collections import Counter

    ec = Counter()
    for p in me.polygons:
        for ek in p.edge_keys:
            ec[ek] += 1
    bnd = [ek for ek, cc in ec.items() if cc == 1] or list(ec.keys())
    best, u = 0.0, None
    for a, b in bnd:
        d = np.array(mw @ me.vertices[a].co) - np.array(mw @ me.vertices[b].co)
        d = d - np.dot(d, n) * n  # horizontal component (in plane)
        L = np.linalg.norm(d)
        if L > best:
            best, u = L, d / L
    if u is None:
        u = np.array([1.0, 0.0, 0.0])
    if abs(u[0]) >= abs(u[1]):  # room_pipeline sign rule (Blender axes)
        if u[0] < 0:
            u = -u
    elif u[1] > 0:
        u = -u
    return u


# floor u/v: prefer room_pipeline's EXACT axes (USD Y-up) -> Blender, so the render orientation
# is IDENTICAL to the stitch. R:(x,y,z)_usd -> (x,-z,y)_blender (proper rotation, det=1).
def usd_to_blender(d):
    return np.array([d[0], -d[2], d[1]], float)


FLOOR_UV = None
_uv_path = os.environ.get("SB_SURF_UV")
if _uv_path and os.path.exists(_uv_path):
    j = json.load(open(_uv_path))
    hor = usd_to_blender(j["u"])
    hor /= np.linalg.norm(hor) + 1e-9
    vert = usd_to_blender(j["v"])
    vert /= np.linalg.norm(vert) + 1e-9
    FLOOR_UV = (hor, vert)
else:
    for name, group, k in surfaces:  # fallback: longest-edge in Blender
        if k == "floor":
            nf = group_normal(group)
            if np.dot(allc - group_verts(group).mean(0), nf) < 0:
                nf = -nf
            u = floor_u_axis(group[0], nf)
            v = np.cross(nf, u)
            v /= np.linalg.norm(v) + 1e-9
            FLOOR_UV = (u, v)
            break


# interior reference for WALL normals: mean of the wall-group centres — the same
# convention Room.py's _WallFrame uses to place fixtures. The all-object centroid
# can land just behind a short nook wall (e.g. a kitchen pillar face), flipping
# the ortho camera to the back of the wall and hiding every fixture on it.
_wall_ctrs = [group_verts(g).mean(0) for _n, g, _k in surfaces if _k == "wall" and group_verts(g) is not None]
wallc = np.mean(_wall_ctrs, axis=0) if _wall_ctrs else allc


def plane_of(group, kind):
    V = group_verts(group)
    c = V.mean(0)
    n = group_normal(group)
    ref = wallc if kind == "wall" else allc
    if np.dot(ref - c, n) < 0:  # normal -> into the room
        n = -n
    if kind == "wall":
        vert = up - np.dot(up, n) * n  # image-up = world-up in the plane
        vert = vert / (np.linalg.norm(vert) + 1e-9)
        hor = np.cross(vert, n)
        hor /= np.linalg.norm(hor) + 1e-9
    else:  # floor/ceiling: room_pipeline u/v
        if FLOOR_UV is not None:
            hor, vert = FLOOR_UV
        else:
            hor = floor_u_axis(group[0], n)
            vert = np.cross(n, hor)
            vert /= np.linalg.norm(vert) + 1e-9
    cc = V - c
    return c, n, vert, hor, float(np.ptp(cc @ hor)), float(np.ptp(cc @ vert))


planes = {name: plane_of(group, k) for name, group, k in surfaces}

# associate each fixture to the surface it sits on
on_surf = {name: [] for name, _, _ in surfaces}
for fx in fixtures:
    fc = center_of(fx)
    best = None
    for name, group, _ in surfaces:
        c, n, vert, hor, w, h = planes[name]
        d = np.dot(fc - c, n)
        u = abs(np.dot(fc - c, hor))
        v = abs(np.dot(fc - c, vert))
        # generous window: fixtures can be slightly embedded in the wall (negative d) or stand
        # off it (radiators/shelves) — missing one on the ortho makes the before/after useless.
        if -0.45 < d < 0.9 and u < w / 2 + 0.35 and v < h / 2 + 0.35:
            if best is None or d < best[1]:
                best = (name, d)
    if best:
        on_surf[best[0]].append(fx.name)

fxmap = {f.name: f for f in fixtures}
cam_d = bpy.data.cameras.new("ortho")
cam_d.type = "ORTHO"
cam_d.clip_start = 0.05
cam_d.clip_end = 20
cam = bpy.data.objects.new("ortho", cam_d)
sc.collection.objects.link(cam)
sc.camera = cam
DIST = 3.0
flips = {}
for name, group, kind in surfaces:
    V = group_verts(group)
    if V is None:
        continue
    for m in meshes:
        m.hide_render = True
    for g in group:
        g.hide_render = False
    keep = set(on_surf[name])
    for fn in keep:
        fxmap[fn].hide_render = False
    c, n, vert, hor, w, h = planes[name]
    if kind == "wall" and (w < 0.35 or h < 0.35):  # degenerate/sliver wall from stage-2 edits
        print(f"skip {name} [wall] degenerate {w:.2f}x{h:.2f}")
        continue
    P = [V] + [own_verts(fxmap[fn]) for fn in keep]
    P = np.vstack([p for p in P if p is not None])
    u = (P - c) @ hor
    vv = (P - c) @ vert
    umid, vmid = (u.min() + u.max()) / 2, (vv.min() + vv.max()) / 2
    W, H = max(u.max() - u.min(), 0.1), max(vv.max() - vv.min(), 0.1)
    cam_center = Vector(c + umid * hor + vmid * vert)
    Z = Vector(n).normalized()
    Y = Vector(vert).normalized()
    X = Y.cross(Z).normalized()
    if kind == "wall":
        flips[name] = {"sflip": bool(X.x < 0), "rflip": False}
    elif kind == "floor":
        flips[name] = {"sflip": False, "rflip": False}
    else:  # ceiling: viewed from below -> render mirrored vs stitch u-layout
        flips[name] = {"sflip": False, "rflip": True}
    rot = Matrix(((X.x, Y.x, Z.x), (X.y, Y.y, Z.y), (X.z, Y.z, Z.z)))
    cam.matrix_world = Matrix.Translation(cam_center + Z * DIST) @ rot.to_4x4()
    cam_d.ortho_scale = max(W, H) * 1.02
    asp = W / max(H, 1e-3)
    base = 760
    sc.render.resolution_x = int(base * asp) if asp >= 1 else base
    sc.render.resolution_y = base if asp >= 1 else int(base / asp)
    sc.render.filepath = os.path.join(out_dir, f"{name}_ortho.png")
    bpy.ops.render.render(write_still=True)
    print(
        f"rendered {name} [{kind}]  {W:.2f}x{H:.2f}  parts={len(group)} fixtures={len(keep)}  flip={flips[name]}"
    )
open(os.path.join(out_dir, "orient.json"), "w").write(json.dumps(flips))
print("ORTHO DONE")
