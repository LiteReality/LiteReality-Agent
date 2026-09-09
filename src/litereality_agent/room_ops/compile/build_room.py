"""
build_room.py — deterministic blank-sheet room assembly
=======================================================
Take a RoomPlan scan (`room.usdz` + capture frames) plus a set of generated
per-object GLBs, and assemble them into ONE blank room scene. This is the
"layer" between asset generation and the harness: pure, deterministic geometry
placement on a neutral baseline. Same inputs -> same `room.glb`, no agents,
no randomness, no materials, no fixtures — those are the harness's job.

What it does (and ONLY this):
  1. import the RoomPlan shell (walls + floor)            -> "Room_Shell"
  2. thicken the thin walls outward (clean inside corners)
  3. boolean-cut the door/window openings through the walls
  4. drop every generated GLB into its RoomPlan box        -> "Furniture" / "Openings"
     (each asset becomes a NAMED empty handle: Table0, Chair0, Door0, ...)
  5. rebuild the ARKit capture cameras (exact pose + intrinsics) -> "ARKit_Cameras"
  6. paint the shell a single neutral grey (the BLANK baseline — no ceiling,
     no textures, no wall fixtures) and floor-ground the furniture
  7. export:  room.glb              the assembled blank scene (named nodes)
              room_layout.json      machine-readable objects + cameras

Inputs
------
  <scan_dir>   a folder with room.usdz + frame_*.{jpg,json}  (RoomPlan capture)
  <assets_dir> a folder with glb/*.glb + manifest.json       (see pack_assets.py)
               (no manifest -> glob glb/*.glb, map each by prim name)
Output
------
  <out.glb>    and room_layout.json next to it

CLI
---
  blender -b --python build_room.py -- <scan_dir> <assets_dir> <out.glb>

Python API (inside Blender)
---------------------------
  scene = RoomScene(scan_dir, assets_dir, out_glb).build()
  scene.summary(); scene.tables(); scene.top_surface("Table0")
  scene.place_on("Table0", "/path/lamp.glb", offset=(0.2, -0.1))
  scene.render_from(30, "/tmp/v.png")
"""

import json
import math
import os
import re
import sys
import zipfile

import bpy
from mathutils import Matrix, Vector

# ============================================================================
# CONFIG
# ============================================================================
try:
    SELF_PATH = os.path.abspath(__file__)
    HERE = os.path.dirname(SELF_PATH)
except NameError:
    SELF_PATH = None
    HERE = os.getcwd()
PROJECT = os.path.dirname(HERE)  # room_ops package root


def _default_scan():
    inp = os.path.join(PROJECT, "input")
    if os.path.isdir(inp):
        for n in sorted(os.listdir(inp)):
            if os.path.exists(os.path.join(inp, n, "room.usdz")):
                return os.path.join(inp, n)
    return os.path.join(inp, "scan")


_argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
SCAN_DIR = _argv[0] if len(_argv) > 0 else _default_scan()
ASSETS_DIR = _argv[1] if len(_argv) > 1 else os.path.join(PROJECT, "input", "assets")
OUT_GLB = _argv[2] if len(_argv) > 2 else os.path.join(PROJECT, "run", "room.glb")

BUILD_CAMERAS = True  # rebuild the ARKit capture cameras + record their info

# assembly tuning
WALL_THICK = 0.10  # wall thickness, extruded OUTWARD only (m)
OPENING_FILL = 1.10  # door/window sized vs the hole -> fills + overlaps rim
CUT_DEPTH = 1.0  # opening cutter depth along wall normal (> wall thickness)
CUT_MARGIN = 1.04  # hole enlarged slightly vs the opening box
# Emissive card behind windows: reads as daylight in Cycles/Eevee, but glTF has no emission-
# strength concept to carry it, so in Room.glb / the Three.js viewer it exports as a flat opaque
# WHITE rectangle plugging the opening. Off by default; turn on only for Blender renders.
DAYLIGHT_CARD = False
CHAIR_YAW = 0.0  # no blanket chair flip: orientation now comes from the box yaw
# (set non-zero only if a real, uniform front-convention offset is found)
SENSOR_W = 36.0  # reference camera sensor width (mm) for lens math

# the blank baseline: one neutral grey on every wall + the floor, nothing else.
# Materials, the ceiling and wall fixtures are deliberately NOT built here —
# they are what the harness produces, so every scene starts identical + blank.
BLANK_SHELL_SRGB = (0.62, 0.62, 0.62)

# scene-graph collections
COLL_SHELL = "Room_Shell"  # walls + floor
COLL_FURN = "Furniture"  # tables, chairs, storage, ...
COLL_OPEN = "Openings"  # doors, windows
COLL_CAM = "ARKit_Cameras"  # capture cameras
COLL_ADDED = "Added"  # anything placed after build (e.g. items on tables)

OPENING_PREFIXES = ("Door", "Window", "Opening")
PLACEABLE_CATS = {"table", "desk", "storage", "counter", "cabinet", "shelf"}

# RoomPlan prim prefix -> semantic category
_CATEGORY_PREFIXES = [
    ("Wall", "wall"),
    ("Floor", "floor"),
    ("Ceiling", "ceiling"),
    ("Door", "door"),
    ("Window", "window"),
    ("Opening", "opening"),
    ("Table", "table"),
    ("Chair", "chair"),
    ("Storage", "storage"),
    ("Sofa", "sofa"),
    ("Bed", "bed"),
    ("Television", "television"),
    ("Refrigerator", "refrigerator"),
    ("Oven", "oven"),
    ("Stove", "stove"),
    ("Sink", "sink"),
    ("Toilet", "toilet"),
    ("Bathtub", "bathtub"),
    ("Stairs", "stairs"),
    ("Fireplace", "fireplace"),
]


def category_of(name):
    base = re.sub(r"\.\d+$", "", name)
    for prefix, cat in _CATEGORY_PREFIXES:
        if base.startswith(prefix):
            return cat
    return "object"


# ============================================================================
# Low-level Blender helpers
# ============================================================================
def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for coll in (bpy.data.meshes, bpy.data.cameras, bpy.data.lights, bpy.data.images):
        for db in list(coll):
            if db.users == 0:
                coll.remove(db)
    for c in list(bpy.data.collections):
        if not c.objects and not c.children:
            bpy.data.collections.remove(c)


def find_usdz(scan_dir):
    for p in (os.path.join(scan_dir, "room.usdz"), os.path.join(scan_dir, "roomplan", "room.usdz")):
        if os.path.exists(p):
            return p
    raise SystemExit(f"room.usdz not found in {scan_dir}")


def import_usd(path):
    before = set(bpy.data.objects)
    bpy.ops.wm.usd_import(filepath=path)
    return [o for o in bpy.data.objects if o not in before]


def import_glb(path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    imported = [o for o in bpy.data.objects if o not in before]
    # Articulated assets (doors/windows) ship a closed->open animation. The assembled room must
    # hold the CLOSED/rest pose for bbox/placement (else a window swung open reads as ~4.8 m), so
    # evaluate frame 0 (closed). We KEEP the animation (do NOT clear it) so the clip survives into
    # the assembled Room.glb (export_animations=True) — the leaf stays closed at rest and plays on
    # demand in a viewer. Placement transforms the ROOT matrix only, so the leaf's local clip rides
    # along untouched.
    bpy.context.scene.frame_set(0)
    return imported


def get_or_make_collection(name):
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    return col


def move_to_collection(obj, coll):
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    coll.objects.link(obj)


def group_fixture(name, category, parts, parent_coll="Fixtures", *,
                  rests_on=None, attached_to=None):
    """Bundle a wall/ceiling fixture's part-objects (frame bars, radiator fins, pen tray, faceplate,
    louvre slats …) into ONE named, hide-as-a-unit group — the fixture analogue of `_wrap_handle`
    for furniture. It creates an empty handle `name` (carrying room_id/category custom props),
    parents every part under it (keeping world pose), and moves the whole set into a per-fixture
    collection nested under `parent_coll`. In the viewer / Blender outliner / exported glTF the
    fixture then selects, hides and toggles as a SINGLE object — e.g. `Whiteboard0`, `Radiator0`,
    `Shelf0` — instead of a loose pile of boxes you'd have to hide one at a time.

    `parts` = the list of objects the box helpers returned (drop the returns of each `_wall_box(...)`
    / `_uv_box(...)` call for this one fixture into a list and pass it in). Returns the empty handle.
    """
    parts = [p for p in parts if p is not None]
    grp = bpy.data.objects.new(name, None)
    grp.empty_display_size = 0.05
    grp["room_id"] = name
    grp["category"] = category
    grp["fixture"] = True
    # SIMULATION READINESS. A physics engine needs to know what holds what up: a mug rests on a
    # desk that rests on the floor, a whiteboard is bolted to a wall. Geometry alone cannot say
    # which — two boxes touching is not the same as one supporting the other, and a prop authored
    # a millimetre proud of its surface is indistinguishable from one floating. So the author
    # states it, and it travels with the object into `room_layout.json` where a simulator (or a
    # QC pass) can read it back. `rests_on` = the surface it stands on; `attached_to` = what it is
    # fixed to when it is not resting on anything (a wall, a ceiling).
    if rests_on:
        grp["rests_on"] = rests_on
    if attached_to:
        grp["attached_to"] = attached_to
    coll = get_or_make_collection(name)
    # nest this fixture's own collection under the shared Fixtures parent, so the outliner shows
    # Whiteboard0 / Radiator0 / … as tidy sub-groups rather than every part loose in one bucket.
    if parent_coll:
        try:
            parent = get_or_make_collection(parent_coll)
            if coll.name not in {c.name for c in parent.children}:
                try:
                    bpy.context.scene.collection.children.unlink(coll)
                except Exception:  # not linked at the scene root (already nested) — fine
                    pass
                parent.children.link(coll)
        except Exception:  # nesting is cosmetic; never let it break the build
            pass
    coll.objects.link(grp)
    for p in parts:
        move_to_collection(p, coll)
        p.parent = grp
        p.matrix_parent_inverse = grp.matrix_world.inverted()
    return grp


def mesh_descendants(obj):
    out, stack = [], [obj]
    while stack:
        o = stack.pop()
        if o.type == "MESH":
            out.append(o)
        stack.extend(o.children)
    return out


def world_bbox(meshes):
    mn = Vector((1e9, 1e9, 1e9))
    mx = Vector((-1e9, -1e9, -1e9))
    for o in meshes:
        if o.type != "MESH":
            continue
        for c in o.bound_box:
            w = o.matrix_world @ Vector(c)
            for i in range(3):
                mn[i] = min(mn[i], w[i])
                mx[i] = max(mx[i], w[i])
    return mn, mx


# ============================================================================
# Assembly primitives (walls / openings / box-fit) — proven geometry
# ============================================================================
def _floor_triangles(floors):
    """World-space XY triangles of the floor plate, for deciding which side of a wall is inside."""
    tris = []
    for f in floors:
        me = f.data
        for poly in me.polygons:
            vs = [(f.matrix_world @ me.vertices[i].co).xy for i in poly.vertices]
            for k in range(1, len(vs) - 1):
                tris.append((vs[0], vs[k], vs[k + 1]))
    return tris


def _on_floor(pt, tris):
    for a, b, c in tris:
        den = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y)
        if abs(den) < 1e-12:
            continue
        u = ((b.y - c.y) * (pt.x - c.x) + (c.x - b.x) * (pt.y - c.y)) / den
        v = ((c.y - a.y) * (pt.x - c.x) + (a.x - c.x) * (pt.y - c.y)) / den
        if u >= -1e-6 and v >= -1e-6 and u + v <= 1.0 + 1e-6:
            return True
    return False


def thicken_walls(walls, room_center, target=WALL_THICK, floor_tris=None):
    """Extrude each wall's OUTER face outward only, keeping the interior face
    where RoomPlan put it. Interior footprint is unchanged -> inside corners stay
    clean; the added thickness/overlap goes to the outside, out of view.

    Which face is "outer" is decided by the FLOOR PLATE, not by the room centroid. The centroid
    test is right for a convex room and wrong exactly where it matters: a wall in the concave notch
    of an L-shaped plan sits on the far side of the centroid from its own interior, so the extrusion
    goes inward and the wall grows 10 cm into the room. On tea_room that is Wall7 and Wall8, and it
    swallows whatever is installed against them — the dishwasher the layout pass had just fitted
    into that corner ends up 2 cm inside the wall, with the pass reporting the room clean because
    the box it settled IS clear of the wall line it was given.

    `floor_tris` is optional: without it this falls back to the centroid test, so a caller that has
    no floor polygon behaves as before rather than failing.
    """
    n = 0
    for o in walls:
        vs = o.data.vertices
        if not vs:
            continue
        lo = [min(v.co[i] for v in vs) for i in range(3)]
        hi = [max(v.co[i] for v in vs) for i in range(3)]
        M3 = o.matrix_world.to_3x3()
        col = [M3.col[i].length for i in range(3)]
        world_ext = [(hi[i] - lo[i]) * col[i] for i in range(3)]
        t = min(range(3), key=lambda i: world_ext[i])  # wall-normal (thinnest)
        cur = world_ext[t]
        if cur < 1e-6 or cur >= target:
            continue
        add_local = (target - cur) / col[t]
        mid = (lo[t] + hi[t]) / 2.0
        axis_local = Vector((0.0, 0.0, 0.0))
        axis_local[t] = 1.0
        n_world = (M3 @ axis_local).normalized()
        centre = o.matrix_world.translation
        outward_plus = n_world.dot(centre - room_center) >= 0
        if floor_tris:
            probe = max(target, 0.05) * 2.0
            plus_in = _on_floor((centre + n_world * probe).xy, floor_tris)
            minus_in = _on_floor((centre - n_world * probe).xy, floor_tris)
            if plus_in != minus_in:          # one side is the room; grow away from it
                outward_plus = minus_in
        for v in vs:
            if outward_plus and v.co[t] > mid:
                v.co[t] += add_local
            elif (not outward_plus) and v.co[t] < mid:
                v.co[t] -= add_local
        o.data.update()
        n += 1
    return n


def opening_to_wall(usdz):
    """{opening_prim: wall_prim} from the usdz directory structure."""
    z = zipfile.ZipFile(usdz)
    mp = {}
    for nm in z.namelist():
        g = re.search(r"Walls/(Wall\d+)/([A-Za-z0-9]+)\.usda$", nm)
        if g and g.group(2) != g.group(1):
            mp[g.group(2)] = g.group(1)
    return mp


def _unit_cube_mesh(name):
    verts = [
        (-0.5, -0.5, -0.5),
        (0.5, -0.5, -0.5),
        (0.5, 0.5, -0.5),
        (-0.5, 0.5, -0.5),
        (-0.5, -0.5, 0.5),
        (0.5, -0.5, 0.5),
        (0.5, 0.5, 0.5),
        (-0.5, 0.5, 0.5),
    ]
    faces = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    return me


def cut_opening(opening_name, wall_name):
    op = bpy.data.objects.get(opening_name)
    wall = bpy.data.objects.get(wall_name)
    if op is None or wall is None:
        return False
    M = op.matrix_world.copy()
    cols = [M.to_3x3().col[i] for i in range(3)]
    thin = min(range(3), key=lambda i: cols[i].length)
    nc = [cols[i].normalized() * CUT_DEPTH if i == thin else cols[i] * CUT_MARGIN for i in range(3)]
    L4 = Matrix(
        (
            (nc[0].x, nc[1].x, nc[2].x, 0.0),
            (nc[0].y, nc[1].y, nc[2].y, 0.0),
            (nc[0].z, nc[1].z, nc[2].z, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    cutter = bpy.data.objects.new(
        "cutter_" + opening_name, _unit_cube_mesh("cutter_" + opening_name)
    )
    bpy.context.scene.collection.objects.link(cutter)
    cutter.matrix_world = Matrix.Translation(M.translation) @ L4
    mod = wall.modifiers.new("cut_" + opening_name, "BOOLEAN")
    mod.operation = "DIFFERENCE"
    mod.object = cutter
    # Blender 4.x offers ("FAST", "EXACT"); 5.x replaced FAST with ("FLOAT", "EXACT", "MANIFOLD").
    # Pick the cheap solver available on this build rather than hardcoding a name that raises
    # TypeError on the other major version.
    _solvers = {i.identifier for i in mod.bl_rna.properties["solver"].enum_items}
    mod.solver = next((s for s in ("FAST", "FLOAT", "EXACT") if s in _solvers), "EXACT")
    bpy.context.view_layer.objects.active = wall
    wall.hide_viewport = False
    ok = True
    # Back up the wall mesh + its footprint so a misbehaving EXACT boolean (which occasionally shreds
    # a wall down to a thin strip -> the wall "goes missing") can be rolled back to the solid wall.
    saved = wall.data.copy()
    bpy.context.view_layer.update()
    dim0 = max(wall.dimensions)
    try:
        bpy.ops.object.modifier_apply(modifier=mod.name)
        bpy.context.view_layer.update()
        # if the cut collapsed the wall (largest span < 60% of the original), the boolean misfired —
        # keep the solid wall instead (the door/window GLB still sits in the opening visually).
        if dim0 > 1e-6 and max(wall.dimensions) < 0.6 * dim0:
            print(f"  ! cut {opening_name} collapsed {wall_name} "
                  f"({max(wall.dimensions):.2f}<{0.6*dim0:.2f}m) — keeping solid wall")
            wall.data = saved
            ok = False
        else:
            bpy.data.meshes.remove(saved)
    except Exception as e:
        print(f"  ! boolean failed {opening_name}->{wall_name}: {e}")
        wall.modifiers.remove(mod)
        wall.data = saved
        ok = False
    bpy.data.objects.remove(cutter, do_unlink=True)
    return ok


def box_frame(box):
    """(center, up, wdir, ddir, W, D, H) of a box's world OBB."""
    cols = [box.matrix_world.to_3x3().col[i] for i in range(3)]
    L = [c.length for c in cols]
    vert = max(range(3), key=lambda i: abs(cols[i].normalized().z))
    horis = [i for i in range(3) if i != vert]
    # box-LOCAL X = width, Y = depth — keep that order (the box already carries the
    # RoomPlan yaw). Do NOT reorder by length: doing so flips the orientation of any
    # object whose depth > width. Orientation must come from the box, not aspect ratio.
    wi, di = horis[0], horis[1]
    up = cols[vert].normalized()
    if up.z < 0:
        up = -up
    wdir = Vector((cols[wi].x, cols[wi].y, 0.0))
    if wdir.length < 1e-6:
        wdir = Vector((1.0, 0.0, 0.0))
    wdir.normalize()
    return box.matrix_world.translation.copy(), up, wdir, up.cross(wdir), L[wi], L[di], L[vert]


# ============================================================================
# Camera math (ARKit capture cameras)
# ============================================================================
def pose_matrix(flat16):
    """cameraPoseARFrame: row-major 4x4 camera->world (translation last column)."""
    return Matrix((flat16[0:4], flat16[4:8], flat16[8:12], flat16[12:16]))


def measure_C(imported):
    """Recover the ARKit(Y-up) -> Blender(Z-up) transform the USD importer used."""
    root = bpy.data.objects.get("room")
    if root is not None and root.type == "EMPTY":
        return root.matrix_world.copy()
    return Matrix.Rotation(math.radians(90), 4, "X")


def make_camera(name, M_blender, intr, render_w, render_h):
    fx, _, cx, _, fy, cy, _, _, _ = intr
    cd = bpy.data.cameras.new(name)
    cd.sensor_fit = "HORIZONTAL"
    cd.sensor_width = SENSOR_W
    cd.lens = fx * SENSOR_W / render_w
    cd.shift_x = (render_w * 0.5 - cx) / render_w
    cd.shift_y = (cy - render_h * 0.5) / render_w
    cd.clip_start = 0.01
    cd.clip_end = 100.0
    cd.display_size = 0.15
    obj = bpy.data.objects.new(name, cd)
    obj.matrix_world = M_blender
    return obj


def detect_resolution(scan_dir, sample_intrinsics):
    jpgs = sorted(f for f in os.listdir(scan_dir) if f.startswith("frame_") and f.endswith(".jpg"))
    if jpgs:
        img = bpy.data.images.load(os.path.join(scan_dir, jpgs[0]), check_existing=True)
        w, h = int(img.size[0]), int(img.size[1])
        bpy.data.images.remove(img)
        if w > 0 and h > 0:
            return w, h
    fx, _, cx, _, fy, cy, _, _, _ = sample_intrinsics
    return int(round(cx * 2)), int(round(cy * 2))


def srgb_to_linear(c):
    return tuple(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in c)


# ============================================================================
# The scene
# ============================================================================
class RoomScene:
    GROUND_CATS = ("chair", "table", "sofa", "desk", "stool", "bench", "bed")

    def __init__(
        self, scan_dir=SCAN_DIR, assets_dir=ASSETS_DIR, out_glb=OUT_GLB, build_cameras=BUILD_CAMERAS
    ):
        self.scan_dir = scan_dir
        self.assets_dir = assets_dir
        self.out_glb = out_glb
        self.build_cameras_flag = build_cameras
        self.objects = {}  # id -> record (see _register)
        self.cameras = []  # list of camera records
        self.render_w = self.render_h = None
        self.C = None
        self._added_counts = {}

    # ---------------------------------------------------------------- build
    def build(self):
        print("=== RoomScene.build (blank) ===")
        clear_scene()
        imported, mapping = self._load_shell()

        shell = get_or_make_collection(COLL_SHELL)
        walls = [o for o in imported if o.type == "MESH" and re.match(r"Wall\d+$", o.name)]
        floors = [o for o in imported if o.type == "MESH" and category_of(o.name) == "floor"]
        # the per-object placement boxes (everything that isn't a wall/floor/opening) —
        # any of these still a MESH after placement means "no GLB was generated for it".
        object_box_names = [
            o.name
            for o in imported
            if o.type == "MESH" and o not in walls and o not in floors and o.name not in mapping
        ]
        for o in walls + floors:
            move_to_collection(o, shell)

        room_center = sum((o.matrix_world.translation for o in walls), Vector()) / max(
            len(walls), 1
        )
        n_thick = thicken_walls(walls, room_center, floor_tris=_floor_triangles(floors))
        print(f"  thickened {n_thick} walls outward to ~{WALL_THICK} m")

        n_cut = sum(cut_opening(op, wall) for op, wall in mapping.items())
        print(f"  cut {n_cut}/{len(mapping)} openings {mapping}")

        placed = self._place_all_assets()
        print(f"  placed {placed} asset instances")

        # Drop any object box that never received a GLB. A filled box was removed and
        # replaced by an EMPTY handle of the same name; an unfilled one is still a MESH.
        # We DON'T keep it as a placeholder: RoomPlan over-detects, and an empty slot
        # (nothing generated, or a spurious detection) should simply be absent.
        dropped = []
        for nm in object_box_names:
            ob = bpy.data.objects.get(nm)
            if ob is not None and ob.type == "MESH":
                bpy.data.objects.remove(ob, do_unlink=True)
                dropped.append(nm)
        if dropped:
            print(f"  dropped {len(dropped)} empty object box(es) (no GLB generated): {dropped}")

        if self.build_cameras_flag:
            n_cam = self._build_cameras()
            print(f"  built {n_cam} capture cameras")

        self._flatten_shell()
        self._apply_blank_shell()  # neutral grey shell, no ceiling/materials/fixtures
        self.index()
        print(f"  indexed {len(self.objects)} objects, {len(self.cameras)} cameras")

        # Snap floor-standing furniture's visible mesh to the floor (vertical only;
        # keeps canonical XY). Fixes assets whose mesh bottom sits off the box floor
        # (e.g. a TRELLIS chair whose legs dip below the box origin).
        ng = self._ground_visual_furniture()
        self.index()
        print(f"  floor-grounded {ng} visible piece(s)")

        # bake shell transforms LAST so local coords == world metres (downstream)
        self._apply_shell_transforms()
        return self

    def _load_shell(self):
        """Rebuild the shell; returns (imported objects, opening->wall map). In priority order:
        1. the SHELL embedded in this file (what the agent edits), 2. room_shell.json,
        3. room.usdz as a fallback."""
        g = globals().get("SHELL")
        if isinstance(g, dict) and g.get("walls"):
            return self._build_shell_from_compact(g)
        for d in (self.assets_dir, self.scan_dir):
            sj = os.path.join(d, "room_shell.json")
            if os.path.exists(sj):
                return self._build_shell_from_compact(json.load(open(sj)))
        usdz = find_usdz(self.scan_dir)
        return import_usd(usdz), opening_to_wall(usdz)

    def _box_obj(self, name, center, xd, xl, yd, yl, zl):
        """One box: a unit cube, axis-aligned in local space, scaled and rotated into world by a
        matrix. xd/yd are the horizontal direction vectors; xl/yl/zl the lengths along x, y and
        vertical. box_frame can recover the OBB from these."""
        me = _unit_cube_mesh(name)
        ob = bpy.data.objects.new(name, me)
        ob.matrix_world = Matrix(
            (
                (xd.x * xl, yd.x * yl, 0.0, center[0]),
                (xd.y * xl, yd.y * yl, 0.0, center[1]),
                (0.0, 0.0, zl, center[2]),
                (0.0, 0.0, 0.0, 1.0),
            )
        )
        bpy.context.scene.collection.objects.link(ob)
        return ob

    def _wall_free_spans(self, wall_name):
        """The runs along this wall NOT taken by a door or window: [(s,e),...] in metres, measured
        from the wall's start. Horizontal fixtures (skirting, trunking, radiators) should only be
        laid within these, so they break at each opening instead of smearing across a Door or
        Window."""
        S = globals().get("SHELL") or {}
        w = (S.get("walls") or {}).get(wall_name)
        if not w:
            return []
        sx, sy = w["start"]
        ex, ey = w["end"]
        length = math.hypot(ex - sx, ey - sy)
        blocks = sorted(
            (op["offset"] - op["width"] / 2.0, op["offset"] + op["width"] / 2.0)
            for op in (S.get("openings") or {}).values()
            if op.get("wall") == wall_name
        )
        spans, cur = [], 0.0
        for b0, b1 in blocks:
            if b0 > cur + 1e-3:
                spans.append((cur, b0))
            cur = max(cur, b1)
        if cur < length - 1e-3:
            spans.append((cur, length))
        return spans

    def _build_shell_from_compact(self, S):
        """Rebuild the geometry from the semantic SHELL: walls as start/end points, openings as
        wall + offset + width/height + sill, objects as centre + size + yaw."""
        fz, cz = S["floor_z"], S["ceiling_z"]
        H = cz - fz
        imported = []

        def wall_dirs(w):
            s = Vector((w["start"][0], w["start"][1], 0.0))
            e = Vector((w["end"][0], w["end"][1], 0.0))
            d = e - s
            ln = d.length
            d = d.normalized() if ln > 1e-9 else Vector((1.0, 0.0, 0.0))
            return s, e, d, Vector((-d.y, d.x, 0.0)), ln

        for name, w in S["walls"].items():  # walls: a thin box, thickened afterwards
            s, e, d, n, ln = wall_dirs(w)
            mid = (s + e) / 2
            th = max(w.get("thickness", 0.001), 0.001)
            imported.append(self._box_obj(name, (mid.x, mid.y, (fz + cz) / 2), d, ln, n, th, H))

        for name, od in S["objects"].items():  # object boxes: centre / size / yaw
            r = math.radians(od["yaw"])
            xd = Vector((math.cos(r), math.sin(r), 0.0))
            yd = Vector((-math.sin(r), math.cos(r), 0.0))
            sz = od["size"]
            imported.append(self._box_obj(name, od["center"], xd, sz[0], yd, sz[1], sz[2]))

        mapping = {}
        for name, op in S["openings"].items():  # openings: offset along wall + size + sill -> cutter box
            wall = op["wall"]
            if wall not in S["walls"]:
                continue
            s, e, d, n, ln = wall_dirs(S["walls"][wall])
            pos = s + d * op["offset"]
            zc = fz + op["sill"] + op["height"] / 2.0
            imported.append(
                self._box_obj(name, (pos.x, pos.y, zc), d, op["width"], n, 0.02, op["height"])
            )
            mapping[name] = wall

        if S.get("floor"):  # the floor polygon, kept faithful
            f = S["floor"]
            me = bpy.data.meshes.new("Floor0")
            me.from_pydata([tuple(v) for v in f["verts"]], [], [tuple(x) for x in f["faces"]])
            me.update()
            ob = bpy.data.objects.new("Floor0", me)
            bpy.context.scene.collection.objects.link(ob)
            imported.append(ob)

        rm = S.get("root_matrix")
        if rm:  # lets measure_C recover the camera calibration
            root = bpy.data.objects.new("room", None)
            root.matrix_world = Matrix((rm[0:4], rm[4:8], rm[8:12], rm[12:16]))
            bpy.context.scene.collection.objects.link(root)

        print(
            f"  shell <- SHELL(compact): {len(S['walls'])} walls, {len(S['openings'])} "
            f"openings, {len(S['objects'])} objects (no usdz)"
        )
        return imported, mapping

    def _asset_plan(self):
        """-> [(glb_abspath, [box_name, ...])] from manifest.json, else glob."""
        manifest = os.path.join(self.assets_dir, "manifest.json")
        plan = []
        if os.path.exists(manifest):
            data = json.load(open(manifest))
            for a in data.get("assets", []):
                glb = os.path.join(self.assets_dir, a["glb"])
                targets = a.get("represents_prims") or [a.get("maps_to_prim")]
                plan.append((glb, [t for t in targets if t]))
        else:
            import glob

            d = os.path.join(self.assets_dir, "glb")
            d = d if os.path.isdir(d) else self.assets_dir
            for f in sorted(glob.glob(os.path.join(d, "*.glb"))):
                base = os.path.splitext(os.path.basename(f))[0].split("__")[-1]
                m = re.match(r"Wall\d+_(Door|Window|Opening)_(\d+)$", base) or re.match(
                    r"ChairCluster(\d+)$", base
                )
                tgt = (
                    f"{m.group(1)}{m.group(2)}"
                    if m and m.lastindex == 2
                    else f"Chair{m.group(1)}"
                    if m
                    else base
                )
                plan.append((f, [tgt]))
        return plan

    def _place_all_assets(self):
        placed = 0
        for glb, targets in self._asset_plan():
            if not os.path.exists(glb):
                print(f"  MISSING {glb}")
                continue
            for box_name in targets:
                if self._place_in_box(glb, box_name):
                    placed += 1
        return placed

    def _place_in_box(self, glb_path, box_name):
        """Fit a GLB into a RoomPlan box, replace the box with a named empty
        handle (Table0, Chair2, Door0, ...) holding the asset meshes."""
        box = bpy.data.objects.get(box_name)
        if box is None:
            return False
        cat = category_of(box_name)
        is_open = box_name.startswith(OPENING_PREFIXES)
        imported = import_glb(glb_path)
        meshes = [o for o in imported if o.type == "MESH"]
        if not meshes:
            return False
        roots = [o for o in imported if o.parent not in imported]

        amn, amx = world_bbox(meshes)
        asize = amx - amn
        acenter = (amn + amx) / 2.0
        aw, ad, ah = max(asize.x, 1e-6), max(asize.y, 1e-6), max(asize.z, 1e-6)
        center, up, wdir, ddir, W, D, H = box_frame(box)
        height_len = (H * OPENING_FILL) / ah if is_open else H / ah

        # Map the asset's OWN axes straight onto the box axes — asset X(width)->box width,
        # asset Y(depth)->box depth — and let the box's yaw set the facing. No aspect-ratio
        # swap: the asset is already built in its canonical frame, so swapping here is what
        # rotated correctly-built objects 90° / backwards.
        dirX, dirY = wdir, ddir
        if is_open:
            sxlen = (W * OPENING_FILL) / aw
            sylen = max(sxlen, (WALL_THICK + 0.05) / ad)
        else:
            sxlen, sylen = W / aw, D / ad

        colX, colY, colZ = dirX * sxlen, dirY * sylen, up * height_len
        L4 = Matrix(
            (
                (colX.x, colY.x, colZ.x, 0.0),
                (colX.y, colY.y, colZ.y, 0.0),
                (colX.z, colY.z, colZ.z, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            )
        )
        Mw = Matrix.Translation(center) @ L4 @ Matrix.Translation(-acenter)
        # A door STANDS on the floor, so its bottom is a hard constraint. OPENING_FILL scales
        # about the box CENTRE, which buries the bottom rail by half the fill (10 cm on a 2.06 m
        # door at 1.10) and lifts the top by the same. Push it back up so the bottom lands exactly
        # on the box floor and the whole overlap goes to the header, where it is wanted — it
        # covers the cut rim. Windows are centred in their opening, so they keep the symmetric fit.
        if cat == "door":
            Mw = Matrix.Translation(up * (H * (OPENING_FILL - 1.0) / 2.0)) @ Mw
        for r in roots:
            r.matrix_world = Mw @ r.matrix_world

        if cat == "chair" and CHAIR_YAW:
            Rz = Matrix.Rotation(math.radians(CHAIR_YAW), 4, up)
            Yw = Matrix.Translation(center) @ Rz @ Matrix.Translation(-center)
            for r in roots:
                r.matrix_world = Yw @ r.matrix_world

        # WINDOWS: the recon glass is dark/tinted and there is nothing bright behind the
        # opening, so it renders as a black void instead of a daylit window. Add a bright
        # emissive "daylight" card BEHIND the glass (away from the room) so it reads as a lit
        # window. This does NOT touch the object_init window GLB — it's a separate card.
        if cat == "window" and DAYLIGHT_CARD:
            self._add_daylight_card(center, wdir, up, ddir, W, H)

        # the placeholder box is consumed; free its name for the asset handle
        bpy.data.objects.remove(box, do_unlink=True)
        coll = get_or_make_collection(COLL_OPEN if is_open else COLL_FURN)
        handle = self._wrap_handle(
            box_name, cat, roots, imported, coll, source=os.path.basename(glb_path)
        )
        return handle is not None

    def _room_centroid(self):
        """Approx room centre = centroid of wall-mesh object locations (cached)."""
        if getattr(self, "_room_c", None) is not None:
            return self._room_c
        pts = [
            o.matrix_world.translation
            for o in bpy.data.objects
            if o.type == "MESH" and category_of(o.name) == "wall"
        ]
        self._room_c = (sum(pts, Vector()) / len(pts)) if pts else Vector()
        return self._room_c

    def _add_daylight_card(self, center, wdir, up, ddir, W, H):
        """A bright emissive plane just BEHIND a window opening (side away from the room),
        sized a touch larger than the hole, so tinted recon glass reads as a lit window."""
        n = ddir.normalized()
        if n.dot(center - self._room_centroid()) < 0:  # face AWAY from the room
            n = -n
        loc = center + n * 0.06
        bpy.ops.mesh.primitive_plane_add(size=1, location=loc)
        card = bpy.context.active_object
        card.name = "daylight_card"
        R = Matrix(
            (
                (wdir.x, up.x, n.x, 0.0),
                (wdir.y, up.y, n.y, 0.0),
                (wdir.z, up.z, n.z, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            )
        )
        card.matrix_world = Matrix.Translation(loc) @ R
        card.scale = (W * 1.15, H * 1.15, 1.0)
        m = bpy.data.materials.new("daylight")
        m.use_nodes = True
        nt = m.node_tree
        for node in list(nt.nodes):
            nt.nodes.remove(node)
        emis = nt.nodes.new("ShaderNodeEmission")
        emis.inputs["Color"].default_value = (0.92, 0.95, 1.0, 1.0)
        emis.inputs["Strength"].default_value = 4.0
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(emis.outputs["Emission"], out.inputs["Surface"])
        card.data.materials.append(m)
        move_to_collection(card, get_or_make_collection(COLL_OPEN))
        return card

    def _wrap_handle(self, name, cat, roots, all_objs, coll, source=None):
        """Create a named empty parent for an asset and move it into a collection."""
        handle = bpy.data.objects.new(name, None)
        handle.empty_display_size = 0.1
        handle["room_id"] = name
        handle["category"] = cat
        if source:
            handle["source_glb"] = source
        coll.objects.link(handle)
        for r in roots:
            r.parent = handle
            r.matrix_parent_inverse = handle.matrix_world.inverted()
        for o in all_objs:
            move_to_collection(o, coll)
        return handle

    def _build_cameras(self):
        frame_jsons = sorted(
            f for f in os.listdir(self.scan_dir) if re.match(r"frame_\d+\.json$", f)
        )
        if not frame_jsons:
            return 0
        first = json.load(open(os.path.join(self.scan_dir, frame_jsons[0])))
        self.render_w, self.render_h = detect_resolution(self.scan_dir, first["intrinsics"])
        self.C = measure_C(list(bpy.data.objects))
        coll = get_or_make_collection(COLL_CAM)
        self.cameras = []
        for fname in frame_jsons:
            data = json.load(open(os.path.join(self.scan_dir, fname)))
            idx = data.get("frame_index", int(re.search(r"\d+", fname).group()))
            cam = make_camera(
                f"cam_{idx:05d}",
                self.C @ pose_matrix(data["cameraPoseARFrame"]),
                data["intrinsics"],
                self.render_w,
                self.render_h,
            )
            coll.objects.link(cam)
            mw = cam.matrix_world
            fx, _, cx, _, fy, cy, _, _, _ = data["intrinsics"]
            self.cameras.append(
                {
                    "name": cam.name,
                    "frame_index": idx,
                    "image": f"frame_{idx:05d}.jpg",
                    "position": [round(v, 4) for v in mw.translation],
                    "forward": [
                        round(v, 4) for v in (mw.to_3x3() @ Vector((0, 0, -1))).normalized()
                    ],
                    "up": [round(v, 4) for v in (mw.to_3x3() @ Vector((0, 1, 0))).normalized()],
                    "intrinsics": {
                        "fx": round(fx, 3),
                        "fy": round(fy, 3),
                        "cx": round(cx, 3),
                        "cy": round(cy, 3),
                    },
                    "lens_mm": round(fx * SENSOR_W / self.render_w, 4),
                    "resolution": [self.render_w, self.render_h],
                }
            )
        bpy.context.scene.camera = bpy.data.objects.get("cam_00000")
        return len(self.cameras)

    def _flatten_shell(self):
        """Bake the USD import hierarchy away: unparent the shell meshes (keeping
        world transform) and delete the leftover RoomPlan group empties, leaving a
        flat, readable graph — shell meshes + named asset handles + cameras."""
        shell = get_or_make_collection(COLL_SHELL)
        for o in [
            o
            for o in bpy.data.objects
            if o.type == "MESH" and re.match(r"(Wall|Floor)\d+$", o.name)
        ]:
            mw = o.matrix_world.copy()
            o.parent = None
            o.matrix_world = mw
            move_to_collection(o, shell)

        def managed(o):  # asset-internal empties have a handle ancestor
            p = o
            while p:
                if p.get("room_id"):
                    return True
                p = p.parent
            return False

        for o in list(bpy.data.objects):
            if o.type == "EMPTY" and not managed(o):
                bpy.data.objects.remove(o, do_unlink=True)
        for c in list(bpy.data.collections):
            if not c.objects and not c.children:
                bpy.data.collections.remove(c)

    def _apply_shell_transforms(self):
        """Bake each shell mesh's transform so its local coords equal world metres."""
        shell = [
            o
            for o in bpy.data.objects
            if o.type == "MESH" and re.match(r"(Wall|Floor|Ceiling)\d+$", o.name)
        ]
        if not shell:
            return
        bpy.ops.object.select_all(action="DESELECT")
        for o in shell:
            o.hide_set(False)
            o.select_set(True)
        bpy.context.view_layer.objects.active = shell[0]
        try:
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        except Exception as e:
            print(f"  ! transform_apply on shell failed: {e}")
        bpy.ops.object.select_all(action="DESELECT")

    def _apply_blank_shell(self):
        """The blank baseline: a single neutral-grey material on every Wall*/Floor0,
        no ceiling, no textures. Wall/floor/ceiling materials are the harness's job,
        so this deterministic baseline is deliberately blank + identical across scenes."""
        mat = self._solid_mat("Shell_Blank", BLANK_SHELL_SRGB, rough=0.9)
        n = 0
        for o in bpy.data.objects:
            if o.type == "MESH" and (re.match(r"Wall\d+$", o.name) or o.name == "Floor0"):
                o.data.materials.clear()
                o.data.materials.append(mat)
                n += 1
        return n

    def _solid_mat(self, name, srgb, rough=0.5, metal=0.0):
        mat = bpy.data.materials.get(name)
        if mat is not None:
            return mat
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        b = nt.nodes.new("ShaderNodeBsdfPrincipled")
        nt.links.new(b.outputs["BSDF"], out.inputs["Surface"])
        lin = srgb_to_linear(srgb)
        b.inputs["Base Color"].default_value = (*lin, 1.0)
        b.inputs["Roughness"].default_value = rough
        if "Metallic" in b.inputs:
            b.inputs["Metallic"].default_value = metal
        return mat

    # ---------------------------------------------------------------- index
    def _register(self, rid, cat, meshes, source=None, handle=None,
                  rests_on=None, attached_to=None):
        mn, mx = world_bbox(meshes)
        self.objects[rid] = {
            "id": rid,
            "category": cat,
            "handle": handle or rid,
            "center": [round((mn[i] + mx[i]) / 2, 4) for i in range(3)],
            "bbox_min": [round(mn[i], 4) for i in range(3)],
            "bbox_max": [round(mx[i], 4) for i in range(3)],
            "size": [round(mx[i] - mn[i], 4) for i in range(3)],
            "top_z": round(mx.z, 4),
            "placeable_surface": cat in PLACEABLE_CATS,
            "rests_on": rests_on,
            "attached_to": attached_to,
            "source_glb": source,
        }

    def index(self):
        """(Re)build the object registry from the live scene graph."""
        self.objects = {}
        for o in bpy.data.objects:  # shell meshes
            if o.type == "MESH" and re.match(r"(Wall|Floor|Ceiling)\d+$", o.name):
                self._register(o.name, category_of(o.name), [o])
        for o in bpy.data.objects:  # asset handles
            if o.type == "EMPTY" and o.get("room_id"):
                self._register(
                    o["room_id"],
                    o.get("category", category_of(o.name)),
                    mesh_descendants(o),
                    source=o.get("source_glb"),
                    handle=o.name,
                    rests_on=o.get("rests_on"),
                    attached_to=o.get("attached_to"),
                )
        return self.objects

    def _ground_visual_furniture(self):
        """Snap each FLOOR-STANDING piece's VISIBLE mesh bottom to the floor.
        Vertical-only, so tucked-under chairs stay tucked; wall items keep height."""
        floor = self.objects.get("Floor0")
        if floor is None:
            return 0
        bpy.context.view_layer.update()  # critical: refresh the depsgraph, or we read stale
        # world coordinates from before the parenting
        floor_z = floor["top_z"]
        n = 0
        for rec in self.objects.values():
            if rec.get("category") not in self.GROUND_CATS:
                continue
            h = bpy.data.objects.get(rec["handle"])
            if h is None:
                continue
            meshes = [o for o in h.children_recursive if o.type == "MESH"]
            if not meshes:
                continue
            mn, _ = world_bbox(meshes)
            dz = floor_z - mn.z
            if abs(dz) > 1e-3:
                h.location.z += dz
                n += 1
        bpy.context.view_layer.update()
        return n

    # ---------------------------------------------------------------- queries
    def by_category(self, cat):
        return sorted(r["id"] for r in self.objects.values() if r["category"] == cat)

    def tables(self):
        return self.by_category("table")

    def chairs(self):
        return self.by_category("chair")

    def walls(self):
        return self.by_category("wall")

    def openings(self):
        return sorted(
            r["id"] for r in self.objects.values() if r["category"] in ("door", "window", "opening")
        )

    def placeable_surfaces(self):
        return sorted(r["id"] for r in self.objects.values() if r["placeable_surface"])

    def get(self, rid):
        return self.objects.get(rid)

    def bbox(self, rid):
        r = self.objects[rid]
        return Vector(r["bbox_min"]), Vector(r["bbox_max"])

    def top_surface(self, rid):
        """Top surface of an object, for placing things on it."""
        r = self.objects[rid]
        cx, cy, _ = r["center"]
        return {
            "z": r["top_z"],
            "center": [cx, cy, r["top_z"]],
            "size": [r["size"][0], r["size"][1]],
        }

    # ---------------------------------------------------------------- edits
    def place_on(
        self, target_id, glb_path, offset=(0.0, 0.0), yaw_deg=0.0, max_footprint_frac=0.5, name=None
    ):
        """Place a GLB on top of target_id's surface (bottom rests on the top face),
        centered + offset in the surface plane. Returns the new object id."""
        if target_id not in self.objects:
            raise ValueError(f"unknown object '{target_id}'")
        top = self.top_surface(target_id)
        tx, ty = top["center"][0] + offset[0], top["center"][1] + offset[1]
        tz = top["z"]

        imported = import_glb(glb_path)
        meshes = [o for o in imported if o.type == "MESH"]
        roots = [o for o in imported if o.parent not in imported]
        mn, mx = world_bbox(meshes)
        size = mx - mn
        ctr = (mn + mx) / 2.0

        scale = 1.0
        if max_footprint_frac:
            sw = (top["size"][0] * max_footprint_frac) / max(size.x, 1e-6)
            sd = (top["size"][1] * max_footprint_frac) / max(size.y, 1e-6)
            scale = min(1.0, sw, sd)

        M = (
            Matrix.Translation((tx, ty, tz))
            @ Matrix.Rotation(math.radians(yaw_deg), 4, "Z")
            @ Matrix.Scale(scale, 4)
            @ Matrix.Translation((-ctr.x, -ctr.y, -mn.z))
        )
        for r in roots:
            r.matrix_world = M @ r.matrix_world

        k = self._added_counts.get(target_id, 0)
        self._added_counts[target_id] = k + 1
        rid = name or f"{target_id}_item{k}"
        cat = category_of(os.path.basename(glb_path))
        self._wrap_handle(
            rid,
            cat,
            roots,
            imported,
            get_or_make_collection(COLL_ADDED),
            source=os.path.basename(glb_path),
        )
        bpy.context.view_layer.update()  # settle parenting before reading bbox
        self._register(
            rid,
            cat,
            mesh_descendants(bpy.data.objects[rid]),
            source=os.path.basename(glb_path),
            handle=rid,
        )
        print(f"  placed '{rid}' on '{target_id}' (scale {scale:.2f}) at z={tz:.3f}")
        return rid

    def render_from(self, frame_index, out_path, res_div=1):
        """Render the assembled scene from capture camera <frame_index>."""
        cam = bpy.data.objects.get(f"cam_{frame_index:05d}")
        if cam is None:
            raise SystemExit(f"camera cam_{frame_index:05d} not built")
        sc = bpy.context.scene
        sc.camera = cam
        if self.render_w:
            sc.render.resolution_x, sc.render.resolution_y = self.render_w, self.render_h
        sc.render.resolution_percentage = max(1, int(100 / res_div))
        sc.render.filepath = out_path
        bpy.ops.render.render(write_still=True)
        sc.render.resolution_percentage = 100
        print(f"  rendered cam_{frame_index:05d} -> {out_path}")

    # ---------------------------------------------------------------- output
    def summary(self):
        print("=== ROOM SCENE (blank) ===")
        cats = {}
        for r in self.objects.values():
            cats.setdefault(r["category"], []).append(r["id"])
        for cat in sorted(cats):
            print(f"  {cat:9s}: {', '.join(sorted(cats[cat]))}")
        if self.placeable_surfaces():
            print("  placeable surfaces (top z):")
            for rid in self.placeable_surfaces():
                print(
                    f"    {rid:9s} z={self.objects[rid]['top_z']:.3f} "
                    f"center={self.objects[rid]['center']}"
                )
        print(
            f"  cameras: {len(self.cameras)} (render res {self.render_w}x{self.render_h})"
            if self.cameras
            else "  cameras: none"
        )

    def export_glb(self):
        os.makedirs(os.path.dirname(self.out_glb), exist_ok=True)
        bpy.ops.object.select_all(action="DESELECT")
        for o in bpy.data.objects:
            if o.type in {"MESH", "EMPTY"} and not o.hide_get():
                o.select_set(True)
        bpy.ops.export_scene.gltf(
            filepath=self.out_glb,
            export_format="GLB",
            use_selection=True,
            export_apply=True,
            export_animations=True,
            export_cameras=False,
            export_lights=False,
        )
        print(f"  EXPORTED -> {self.out_glb} ({os.path.getsize(self.out_glb) // 1024} KB)")

    def export_layout(self, path=None):
        path = path or os.path.join(os.path.dirname(self.out_glb), "room_layout.json")
        mn, mx = world_bbox([o for o in bpy.data.objects if o.type == "MESH"])
        doc = {
            "scene": os.path.basename(self.scan_dir),
            "coordinate_frame": {
                "up": "Z",
                "units": "meters",
                "note": "Blender Z-up; ARKit Y-up mapped by C=Rx(+90).",
            },
            "bounds": {
                "min": [round(v, 4) for v in mn],
                "max": [round(v, 4) for v in mx],
                "size": [round(mx[i] - mn[i], 4) for i in range(3)],
            },
            "collections": {
                "Room_Shell": "walls + floor (blank neutral grey)",
                "Furniture": "tables, chairs, ...",
                "Openings": "doors, windows",
                "ARKit_Cameras": "capture cameras",
                "Added": "objects placed after build (e.g. items on tables)",
            },
            "note": "Blank-sheet assembly: geometry placed on a neutral shell. "
            "Materials, ceiling and wall fixtures are added by the harness.",
            "objects": [self.objects[k] for k in sorted(self.objects)],
            "cameras": self.cameras,
        }
        with open(path, "w") as fh:
            json.dump(doc, fh, indent=2)
        print(f"  LAYOUT   -> {path} ({len(self.objects)} objects, {len(self.cameras)} cameras)")
        return path

    def export_blend(self, path=None):
        """Save the fully assembled scene as a .blend project, so it opens directly for
        inspection or editing."""
        path = path or os.path.join(os.path.dirname(self.out_glb), "Room.blend")
        try:
            bpy.ops.wm.save_as_mainfile(filepath=path)
            print(f"  BLEND    -> {path} ({os.path.getsize(path) // 1024} KB)")
        except Exception as e:
            print(f"  ! failed to save .blend: {e}")
        return path

    def copy_self(self):
        if SELF_PATH and os.path.exists(SELF_PATH):
            import shutil

            dst = os.path.join(os.path.dirname(self.out_glb), "build_room.py")
            if os.path.abspath(dst) != SELF_PATH:
                shutil.copyfile(SELF_PATH, dst)
                print(f"  COPIED   -> {dst}")


# ============================================================================
# main
# ============================================================================
def main():
    scene = RoomScene().build()
    scene.summary()
    scene.export_glb()
    scene.export_layout()
    scene.export_blend()
    scene.copy_self()
    print("DONE")
    return scene


if __name__ in ("__main__", "room_py_exec"):  # room_py_exec: render_room_cameras execs us
    ROOM = main()
