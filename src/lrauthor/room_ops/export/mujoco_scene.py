#!/usr/bin/env python3
"""mujoco_scene.py — the authored room as an MJCF a physics engine can actually step.

    python -m lrauthor.room_ops.export.mujoco_scene --room <room dir> [--out <dir>]

`Room.glb` is a picture of a room: one file, every object baked into it, no notion of what is
furniture and what is wall, no mass, no joints. MuJoCo needs the opposite — a scene of BODIES, each
with its own collision geometry, its own mass, and a joint (or deliberately none) saying how it may
move. This turns one into the other, and the information to do it already exists across three
files that nothing had yet read together:

    Room.py `SHELL`          walls, openings, floor/ceiling heights — exact metric structure
    room_layout.json         every object's pose, category, and `rests_on` / `attached_to`
    Room.glb                 the geometry, per named handle
    Object/<name>.glb        per-object `extras` — the articulation the room glb drops

Four kinds of body, decided from that data rather than by guessing at names:

``structure``   walls, floor and ceiling become static BOXES built from the SHELL numbers, not from
                the mesh. A box is exact here (a wall IS a box), collides perfectly, and costs one
                geom instead of a few thousand triangles. Openings are subtracted, so a wall with a
                door in it becomes jamb / jamb / lintel and the door has somewhere to swing.
``articulated`` a door or window: the frame is static, and each part carrying `articulation_type`
                in its glTF extras becomes its own body on a hinge or slide, with the axis and
                limits the object's build recipe recorded.
``free``        anything that `rests_on` something — furniture and every authored prop. A free
                joint, mesh geometry, and mass from its own volume. These are the things that can
                be pushed, knocked over, and picked up.
``attached``    anything `attached_to` a wall or ceiling — sockets, trunking, skirting, shelves,
                the radiator. Static geometry. A socket is not a rigid body that can fall off.

COLLISION IS NOT THE VISUAL MESH. MuJoCo collides a mesh as its CONVEX HULL, and the hull of a
table is a solid block — chairs tucked under it would be launched on the first step, and a shelf
would be a filled cupboard. So every concave body is decomposed into convex parts (CoACD) and the
parts are the colliders, while the original mesh is kept as a visual-only geom. Convex bodies skip
the decomposition and use the mesh directly.

The support relations the authoring pass recorded are what make the result start at REST rather
than settling: an object whose `rests_on` surface is already under it does not fall, and one that
was authored floating would. That is checked and reported rather than silently corrected — a scene
that needs 200 steps to stop twitching is telling you something true about the room.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

from . import sim_assets

STRUCTURE = {"wall", "floor", "ceiling"}
# Things HUNG on a wall rather than built into it. A socket is part of the building; a picture is
# an object someone put there on a nail, and in a room that is being shaken hard enough to move the
# furniture, the pictures are among the first things to come down. Welding them permanently makes
# the room less true, not more.
# Only what hangs on a HOOK. A whiteboard, a pinboard and a sign are screwed through into the
# wall and a room shaking hard enough to drop them has bigger problems; a socket is part of the
# building's wiring. Those stay fixed. A picture, a mirror, a clock hangs on one nail.
HANGING = {"framed_photo", "picture", "painting", "artwork", "photo", "poster", "mirror",
           "clock", "certificate", "frame"}
# Newtons of pull a hanging survives before its fixing lets go. A picture hook is rated around
# 5 kg (~50 N) and the ones in a scanned room are rarely new, so this is deliberately modest.
HANGING_HOLD_N = 22.0

# Mounted on the CEILING. `is_wall_hung` asks about walls and nothing else, so in a room with no
# declarations at all every downlight and smoke detector became a free body and rained onto the
# floor — twelve of them in one kitchen, a 2.96 m drop each. A luminaire at ceiling height is
# fixed to the ceiling for the same reason a socket is fixed to its wall.
CEILING_MOUNTED = {"ceiling_light", "downlight", "spotlight", "light", "luminaire", "pendant",
                   "chandelier", "smoke_detector", "detector", "alarm", "sprinkler", "vent",
                   "extractor", "fan", "speaker", "projector", "ceiling_rose", "diffuser"}
CEILING_GAP = 0.30       # m below the ceiling and still counted as mounted on it

# BUILDING FABRIC. These are not objects in a room, they are part of its surfaces: a skirting board
# is nailed to the wall it runs along and a length of trunking is screwed to it. Left free, a 3 m
# skirting board slides across the floor like a plank.
FABRIC = {"skirting", "trunking", "conduit", "rail", "cornice", "coving", "dado", "architrave",
          "socket", "switch", "data_socket", "power_socket", "radiator", "pipe", "boxing",
          "shelf_standard", "curtain_rail", "picture_rail", "beam"}
# FURNITURE THAT STANDS ON THE FLOOR AND STAYS THERE. A room's tables, worktops, counters and
# cabinets are scenery: nobody pushes a desk across an office, and a simulator that lets them is
# not more realistic, it is less. Left free they are also the multiplier on every other error in
# the room — a cabinet ejected by a 5 mm authored overlap takes everything standing on it with it.
#
# The trigger is a real gap rather than a preference. `rests_on` is written by `group_fixture`, so
# only the fixtures and props the AUTHOR added ever carry it; the furniture that came from the scan
# is placed from the manifest and has neither `rests_on` nor `attached_to`. Office-Elliott's Table0
# is 27.8 kg on a free joint for exactly that reason, while Table1 — the same kind of desk — is
# static only by accident, because it happens to have a lift top and anything with a moving part
# was already being pinned.
#
# CHAIRS AND STOOLS ARE DELIBERATELY NOT HERE. A chair is the one piece of furniture a room really
# does move, it is what a robot has to get around, and freezing them would take the most useful
# obstacle in the scene out of the physics.
SCENERY = {"table", "desk", "worktop", "countertop", "counter", "bench", "storage", "cabinet",
           "cupboard", "wardrobe", "sideboard", "dresser", "shelving", "bookcase", "sink",
           "oven", "stove", "hob", "dishwasher", "refrigerator", "freezer", "washer", "dryer",
           "bed", "sofa", "settee", "couch", "piano", "reception_desk"}
# ...and only when it is actually standing on the floor. A "table" whose underside is a metre up is
# a wall-mounted shelf that happens to be named one, and that is a different thing.
SCENERY_FLOOR_GAP = 0.08

# TRIM IS PART OF THE BUILDING, NOT AN OBJECT IN IT. Skirting, trunking and coving are nailed to
# the fabric and run the length of a room, geometrically INSIDE the walls they trim. They can never
# be free bodies, and unlike SCENERY that is true at any height — trunking sits at 0.95 m.
#
# This is a set rather than an inference because the inference cannot reach them. `hung` is only
# consulted when an object has neither `attached_to` nor `rests_on`, so a single authored
# `rests_on: Floor0` is enough to make one free with no further test — and skirting genuinely does
# sit on the floor, so the claim is not even wrong. Office-Elliott authored under a short step
# budget produced exactly that: `Skirting0`, 86 kg of trim ringing the room, emitted as a free body
# 40 mm inside the door lining. MuJoCo read the overlap as stored energy and threw it 158 mm, and
# the scene failed the stability gate on that one body alone. The fully authored room escaped it
# only because the author got far enough to write `attached_to: Room_Shell` — which is to say the
# room was one interrupted session away from being unusable, with nothing to warn anyone.
TRIM = {"skirting", "baseboard", "trunking", "coving", "cornice", "architrave", "dado",
        "beading", "moulding", "molding", "threshold", "picture_rail", "chair_rail"}

# Materials whose NAME says they are see-through. glTF carries the pane as an ordinary opaque
# texture — Blender's transmission does not survive the export — so a window arrives as a solid
# painted panel and the room has no daylight in it. The name is the only surviving evidence that
# it was glass, and the authoring pass names its materials deliberately (`Glazing`, `Glass_Centre`).
# `pane` also matches `panel`, and a recessed LED ceiling PANEL is not glazing: keying glassiness
# on the node name turned all four of Elliott's luminaires 22% transparent and the ceiling started
# showing the void above it. The lookahead is the whole fix — every other token is unambiguous.
GLASSY_RE = re.compile(r"glaz|glass|pane(?!l)|window_gl|sash", re.I)
GLASS_ALPHA = 0.22
OPENING = {"door", "window", "opening"}

# kg per cubic metre of the object's BOUNDING BOX — not of its mesh. A real object is mostly air:
# a chair is four thin legs and a thin seat, so its mesh volume is a few litres and material
# density gave it 1.5 kg. Everything then behaved like cardboard, and the contact solver threw the
# lightest bodies across the room — Kitchen's chair reached 175 m. Occupancy density is what makes
# a chair weigh what a chair weighs, and mass is the single biggest lever on whether a scene is
# stable, because a heavy body absorbs the contact noise that launches a light one.
# Calibrated against what the things actually weigh: a stacking chair is ~5 kg in a 0.4 m^3 box,
# a meeting table ~50 kg in 2.7 m^3. The first pass used 45 and 70 and produced 23 kg chairs and
# 186 kg tables — furniture that a 2 m/s^2 shake cannot move, which is exactly what it looked like.
DENSITY = {
    "default": 45.0,
    "chair": 14.0, "stool": 14.0, "table": 20.0, "desk": 22.0,
    "storage": 55.0, "cabinet": 55.0, "wardrobe": 50.0, "shelf": 45.0,
    "bed": 35.0, "sofa": 30.0, "refrigerator": 110.0, "radiator": 160.0,
    "television": 90.0, "monitor": 110.0, "laptop": 300.0, "keyboard": 250.0,
    "mug": 250.0, "lidded_mug": 250.0, "travel_cup": 200.0, "vacuum_flask": 300.0,
    "glass_bottle": 450.0, "coffee_press": 300.0, "rug": 60.0,
    "papers": 400.0, "notebook": 400.0, "cardboard_box": 60.0, "storage_box": 60.0,
    "backpack": 90.0, "jacket": 40.0, "cable": 300.0, "keys": 800.0,
}
MIN_BODY_MASS = 0.15     # kg — below this MuJoCo throws anything it touches

# A hull this close to its own mesh is already convex enough to collide as itself.
CONVEX_TOL = 0.06        # 6% volume difference
DECOMP_MAX_PARTS = 24
DECOMP_TIMEOUT = 45.0    # seconds per body — see `_decompose`
# Below this bounding-box diagonal an object is collided as its convex hull, however concave it is.
# Decomposition exists so a chair can go UNDER a table and a shelf is not a solid cupboard — it is
# about the space an object encloses, and a trinket encloses nothing anyone will put anything in.
# Against that it costs stability: CoACD split a 52 g photo frame into 24 convex parts, several of
# them 4 mm slivers, and two dozen overlapping thin hulls on a near-weightless body give the solver
# a stack of simultaneous contacts to satisfy at once. That frame was the object launching itself
# across the room — and worse at LOW shake amplitudes than high ones, which is the signature of a
# numerical artefact rather than a push.
MIN_DECOMP_DIAGONAL = 0.30
# How many parts a body may be collided piece-by-piece before it goes back to one decomposition of
# the whole carcass. See the note where it is used.
MAX_PARTWISE_COLLIDERS = 16
# ...and a cap on how finely ANY body is cut. Twenty-four thin hulls on a 1.5 kg chair is a stack of
# simultaneous contacts for the solver to satisfy at once, and it resolves them by launching it:
# Kitchen's Chair0 travelled 175 m. Decomposition buys the ability to put a chair UNDER a table —
# a handful of parts delivers that; two dozen buys nothing and costs stability. The budget scales
# with mass, because a heavy body absorbs contact noise that throws a light one across the room.
def _part_budget(mesh, density: float) -> int:
    try:
        # OCCUPANCY VOLUME, NOT MESH VOLUME. `DENSITY` is kg per cubic metre of the BOUNDING BOX —
        # it says so at its definition — and multiplying it by the mesh volume instead makes every
        # hollow thing weightless. A 2.5 m shelving unit is a few thin boards, so it came out under
        # 3 kg and was given four convex hulls for the whole carcass; one of them spanned 2.4 m and
        # swallowed the shelves, and every prop standing on one started 200 mm inside it and was
        # ejected three metres on the first step.
        extents = np.asarray(mesh.extents, dtype=float)
        mass = max(float(np.prod(np.maximum(extents, 0.02))) * density, 0.05)
    except Exception:                                   # noqa: BLE001
        mass = 1.0
    if mass < 3.0:
        return 4
    if mass < 15.0:
        return 8
    return DECOMP_MAX_PARTS
# RoomPlan measures a wall as a surface, not a solid, and this capture records every one of them as
# 0.1 mm thick. That is a plane, and it fails twice over: from any raised angle the room renders as
# an open box with no walls at all, and a light object under a hard impulse can cross 0.1 mm in one
# 2 ms step and tunnel straight through it. Real partition walls are 75-100 mm, so a wall thinner
# than this is a measurement artefact rather than a thin wall, and it is given a believable body.
MIN_WALL_THICKNESS = 0.09


def _coacd_worker(vertices, faces, mode, queue, budget=DECOMP_MAX_PARTS):
    """Run CoACD in a throwaway process, so a hang can be killed rather than waited out."""
    try:
        import coacd

        parts = coacd.run_coacd(coacd.Mesh(vertices, faces),
                                max_convex_hull=budget, preprocess_mode=mode)
        queue.put([(np.asarray(v).tolist(), np.asarray(f).tolist()) for v, f in parts])
    except Exception:                                   # noqa: BLE001 — report nothing, not a crash
        queue.put([])


def _decompose(mesh, timeout: float = DECOMP_TIMEOUT, budget: int = DECOMP_MAX_PARTS):
    """Convex parts for one mesh, or []. Never hangs.

    CoACD does not merely fail on awkward input, it SPINS. `Pinboard0` — a watertight 60-face flat
    board — ran its search for two hours and twenty minutes with preprocessing off before anyone
    noticed the export had stopped moving. A near-planar mesh offers no good split and the tree
    search keeps hunting for one, so there is no exception to catch: only a clock helps. Each body
    therefore gets its own process and a deadline, and one that misses it falls back to voxelised
    decomposition and then to its own convex hull. A slightly coarse collider on one board is a
    fair price for an export that always finishes.
    """
    import multiprocessing as mp

    for mode in ("off", "auto"):
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        proc = ctx.Process(target=_coacd_worker,
                           args=(np.asarray(mesh.vertices), np.asarray(mesh.faces), mode,
                                 queue, budget))
        proc.start()
        try:
            parts = queue.get(timeout=timeout)
        except Exception:                               # noqa: BLE001 — Empty means it hung
            parts = []
        proc.terminate()
        proc.join(5)
        if parts:
            return parts
    return []


def _kelvin_rgb(kelvin: float) -> tuple[float, float, float]:
    """Colour temperature -> a normalised RGB. 2700 K is orange, 6500 K is white.

    The authoring pass records the temperature it measured from the photographs, and throwing that
    away makes every room the same neutral white — which is exactly the thing that reads as CG.
    """
    t = max(1000.0, min(12000.0, float(kelvin))) / 100.0
    if t <= 66:
        r, g = 255.0, 99.4708025861 * math.log(t) - 161.1195681661
    else:
        r = 329.698727446 * (t - 60) ** -0.1332047592
        g = 288.1221695283 * (t - 60) ** -0.0755148492
    b = 255.0 if t >= 66 else (0.0 if t <= 19 else
                               138.5177312231 * math.log(t - 10) - 305.0447927307)
    return tuple(max(0.0, min(1.0, v / 255.0)) for v in (r, g, b))


def _in_category(category: str, table) -> bool:
    """Does this category — a merged run's compound name included — belong to `table`?

    Deliberately a local copy of `scene_init.layout.graph.in_category` rather than an import.
    `room_ops` is the inner layer: the pipeline may call it, it may not call back, and the
    architecture test enforces that. Six lines of set membership is a smaller cost than a layer
    violation, and the two callers want the same thing for the same reason — a box merge names a
    fused unit after its members (`oven_storage_stove`), so exact-string category tests miss it.
    """
    table = set(table)
    return category in table or bool(
        {token for token in (category or "").split("_") if token} & table)


def _supported_from_below(rec: dict, records: dict, tol: float = 0.06) -> bool:
    """Is there something directly under this object that could be holding it up?

    Footprints must overlap and the supporter's top must meet this object's underside. Anything
    else — a shelf two metres away, a table beside it — is not support, and an object with no
    support and no floor under it is hanging from the structure whether or not anyone said so.
    """
    lo = np.asarray(rec["bbox_min"], dtype=float)
    hi = np.asarray(rec["bbox_max"], dtype=float)
    for other in records.values():
        if other is rec:
            continue
        # A ceiling is not underneath anything. Its slab's top surface sits a few centimetres above
        # a downlight's underside, which passed the height test and made the ceiling "support" every
        # light screwed into it — so the lights stayed free and fell 2.65 m. Walls likewise.
        if other.get("category") in {"wall", "ceiling"}:
            continue
        o_lo = np.asarray(other["bbox_min"], dtype=float)
        o_hi = np.asarray(other["bbox_max"], dtype=float)
        if o_hi[2] > float(rec["center"][2]):
            continue                                     # a supporter is below what it holds up
        if o_hi[2] < lo[2] - tol or o_hi[2] > lo[2] + tol:
            continue                                     # its top is not at our underside
        if o_hi[0] <= lo[0] or o_lo[0] >= hi[0] or o_hi[1] <= lo[1] or o_lo[1] >= hi[1]:
            continue                                     # footprints do not overlap
        return True
    return False


def _material_spec(mesh) -> tuple:
    """A hashable description of one mesh's appearance: ('tex', name) or ('rgba', r, g, b, a).

    glTF's `baseColorFactor` arrives here on 0-255 while MuJoCo wants 0-1, and a factor is present
    even on meshes that also carry an image — the texture wins when there is one.
    """
    visual = getattr(mesh, "visual", None)
    material = getattr(visual, "material", None)
    if material is None:
        return ("rgba", 0.7, 0.7, 0.7, 1.0, "")
    image = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
    uv = getattr(visual, "uv", None)
    name = getattr(material, "name", None) or ""
    if image is not None and uv is not None and len(uv):
        return ("tex", name or f"tex{id(image):x}", image)
    factor = getattr(material, "baseColorFactor", None)
    if factor is None:
        return ("rgba", 0.7, 0.7, 0.7, 1.0, name)
    values = [float(v) for v in np.asarray(factor, dtype=float).ravel()[:4]]
    while len(values) < 4:
        values.append(255.0)
    scale = 255.0 if max(values) > 1.001 else 1.0
    return ("rgba", *[round(v / scale, 4) for v in values], name)


def _material_groups(meshes_in, prefix: str, out_dir: Path, reuse: bool):
    """Split a body's parts by appearance and write one OBJ per group.

    Returns [(mesh name, group, spec)]. Grouping by material rather than by source node keeps the
    geom count down: a chair with 40 parts in two materials becomes two visual geoms, not forty.
    """
    from collections import defaultdict

    import trimesh

    # Key on the WHOLE spec, not its first two fields. `spec[:2]` for a colour is ("rgba", red),
    # so a beige wall and a red book with the same red channel were being merged into one geom and
    # painted with whichever won — a quiet way for materials to come out wrong with nothing to see
    # in the logs.
    buckets: dict[tuple, list] = defaultdict(list)
    for mesh in meshes_in:
        spec = _material_spec(mesh)
        buckets[spec if spec[0] == "rgba" else spec[:2]].append((spec, mesh))

    out = []
    for index, (_key, items) in enumerate(sorted(buckets.items(), key=lambda kv: str(kv[0]))):
        spec = items[0][0]
        group = trimesh.util.concatenate([m for _s, m in items])
        name = f"{prefix}_v{index}"
        path = out_dir / f"{name}.obj"
        if not (reuse and path.is_file()):
            try:
                group.export(path, include_texture=(spec[0] == "tex"))
            except Exception:                           # noqa: BLE001 — UV export is best-effort
                group.export(path)
        out.append((name, group, spec))
    return out


def _fill_unobserved(image, threshold: int = 26):
    """Replace a stitched texture's unseen regions with the colour of the parts that WERE seen.

    A wall texture is a stitch of the capture, and a hand-held scan never sees the top of a wall:
    `tex_Wall0_Paint.png` is 96% near-black across its upper quarter. Blender hides that because
    the authored material only samples the observed part, but MuJoCo maps the image straight onto
    the box and the missing data renders as a black band across every wall — which looks exactly
    like the walls being cut off half way up, and was diagnosed twice as a lighting fault before
    anyone looked at the texture. Painting the void the median observed colour is what the room
    would have been had the scan reached it.
    """
    import numpy as np
    from PIL import Image

    rgb = image.convert("RGB")
    a = np.asarray(rgb).astype(np.int16)
    void = a.max(axis=2) < threshold
    if not void.any() or void.all():
        return rgb
    seen = a[~void]
    if not len(seen):
        return rgb
    a[void] = np.median(seen, axis=0).astype(np.int16)
    return Image.fromarray(a.astype("uint8"), "RGB")


def _material_name(asset, spec: tuple, fallback: str, out_dir: Path, reuse: bool,
                   structural: bool = False, node: str = "") -> str:
    """Declare (once) the MJCF material for a spec and return its name."""
    registry = _material_name.__dict__.setdefault("seen", {})
    # GLAZING IS NAMED BY THE PART, NOT ALWAYS BY THE MATERIAL. Deciding transparency from the
    # material name alone works only when the author named it: these rooms carry materials called
    # `Material_0_007` or `tex7f8677eaecb0`, and a window's opening leaves came through as flat
    # opaque grey panes because nothing in "tex7f8677eaecb0" says glass. The NODE knows —
    # `Window0_sash_right` is a glazed leaf whatever its material is called. The flag joins the
    # cache key so the same grey used elsewhere as a solid stays solid.
    glass_by_node = bool(GLASSY_RE.search(node))
    key = (id(asset), spec if spec[0] == "rgba" else spec[:2], glass_by_node)
    if key in registry:
        return registry[key]
    if spec[0] == "tex":
        image = spec[2]
        safe = "".join(c if c.isalnum() else "_" for c in str(spec[1]))[:40] or "tex"
        png = out_dir / f"tex_{safe}.png"
        # A STRUCTURAL texture is always rewritten. Its void-filling is a function of code that has
        # changed under it: the floors kept 41-61% black caches from before the fill existed, and
        # `--reuse-meshes` faithfully preserved them. Reuse is only safe for bytes that are a pure
        # function of the source asset, which a repaired texture is not.
        if not (reuse and png.is_file() and not structural):
            try:
                # Only for walls/floor/ceiling: on an OBJECT a black region is usually a black
                # object — a monitor screen, a television — and repainting it would be a lie.
                (_fill_unobserved(image) if structural else image.convert("RGB")).save(png)
            except Exception:                           # noqa: BLE001 — unreadable image
                registry[key] = fallback
                return fallback
        glassy = glass_by_node or bool(GLASSY_RE.search(str(spec[1])))
        if glassy and glass_by_node:
            safe = f"{safe}_glass"[:46]
        if f"m_{safe}" in {m.get("name") for m in asset.findall("material")}:
            registry[key] = f"m_{safe}"
            return registry[key]
        ET.SubElement(asset, "texture", name=f"t_{safe}", type="2d", file=png.name)
        ET.SubElement(asset, "material", name=f"m_{safe}", texture=f"t_{safe}",
                      texuniform="false",
                      specular="0.6" if glassy else "0.2",
                      shininess="0.85" if glassy else "0.3",
                      reflectance="0.25" if glassy else "0",
                      rgba=f"1 1 1 {GLASS_ALPHA}" if glassy else "1 1 1 1")
        registry[key] = f"m_{safe}"
        return registry[key]
    values = list(spec[1:5])
    if glass_by_node or bool(GLASSY_RE.search(str(spec[5] if len(spec) > 5 else ""))):
        values[3] = GLASS_ALPHA
    rgba = " ".join(f"{v:.4f}" for v in values)
    safe = "c" + "_".join(f"{int(round(v * 255))}" for v in values)
    if safe not in {m.get("name") for m in asset.findall("material")}:
        ET.SubElement(asset, "material", name=safe, rgba=rgba, specular="0.15", shininess="0.25")
    registry[key] = safe
    return safe


def _load_shell(room: Path) -> dict:
    src = (room / "Room.py").read_text(encoding="utf-8")
    i = src.rindex("SHELL = ") + len("SHELL = ")
    return json.JSONDecoder().raw_decode(src[i:])[0]


def _glb_to_shell(mesh):
    """glTF is Y-up, the SHELL and MuJoCo are Z-up: (x, y, z)_glb -> (x, -z, y)."""
    m = mesh.copy()
    t = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float)
    m.apply_transform(t)
    return m


# glTF is Y-up, the SHELL and MuJoCo are Z-up. `A` maps a vector from one to the other and `A_INV`
# back again; both are needed because an articulation axis is recorded in the object's OWN frame.
_A = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
_A_INV = _A.T
_A4 = np.eye(4)
_A4[:3, :3] = _A


def _axis_to_world(axis, transform) -> list[float]:
    """An articulation axis from the object's local frame into the world the scene is built in.

    The recipe records the axis where the object was MODELLED — a drawer slides along its own
    -Y, a door hinges about its own Z — and the room then places that object with a rotation.
    Using the recorded axis verbatim as a world axis is only correct for objects that happen to sit
    square to the world, which in a scanned room is almost none of them: every drawer in a unit
    against an angled wall slid sideways out of its own carcass, and doors swung about the wrong
    line. `room_layout.json` does not record yaw, so the object's true orientation has to come from
    the node's own transform in the room glb.

        world = A . R . A_inv . local        (A: glTF -> SHELL, R: the node's world rotation)
    """
    rotation = np.asarray(transform, dtype=float)[:3, :3]
    world = _A @ rotation @ _A_INV @ np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(world))
    if norm < 1e-9:
        return [0.0, 0.0, 1.0]
    return [round(float(v), 6) for v in world / norm]


def _bodies_from_glb(glb: Path, handles: set[str]) -> tuple[dict[str, Any], Any]:
    """Every named handle in the room glb -> its world mesh, in the SHELL frame.

    The loaded SCENE comes back too. It is what `sim_assets.placement_transform` needs to work out
    where the room put each object, and re-loading a 200 MB glb a second time to get it is minutes
    of the export spent reading a file that is already open.
    """
    from collections import defaultdict

    import networkx as nx
    import trimesh

    scene = trimesh.load(str(glb))
    graph = scene.graph.to_networkx()
    preds = nx.predecessor(graph, scene.graph.base_frame)

    def owner(node: str) -> str | None:
        cur, seen = node, 0
        while cur is not None and seen < 64:
            if cur in handles:
                return cur
            parent = preds.get(cur) or []
            cur = parent[0] if parent else None
            seen += 1
        return None

    parts: dict[str, list] = defaultdict(list)
    for node in scene.graph.nodes_geometry:
        h = owner(node)
        if h is None:
            continue
        transform, name = scene.graph[node]
        m = scene.geometry[name].copy()
        m.apply_transform(transform)
        # The world mesh is what gets exported; the TRANSFORM is what says which way the object is
        # actually facing, and an articulation axis is meaningless without it.
        parts[h].append((node, m, np.asarray(transform, dtype=float)))
    return parts, scene


def _articulation(object_glb: Path) -> dict[str, dict]:
    """part name -> {type, axis, limit_min, limit_max}, read from the object's glTF extras.

    The per-object exporter runs with `export_extras=True`, so the joint the build recipe declared
    survives in the file. The ROOM exporter does not, which is why this is read from the object's
    own glb rather than from `Room.glb` — the room keeps the node NAMES but drops what they mean.
    """
    import struct

    if not object_glb.is_file():
        return {}
    raw = object_glb.read_bytes()
    length, _ = struct.unpack_from("<II", raw, 12)
    gltf = json.loads(raw[20:20 + length].decode("utf-8"))
    out = {}
    for node in gltf.get("nodes", []):
        extras = node.get("extras") or {}
        if extras.get("articulation_type"):
            out[node.get("name", "")] = {
                "type": extras["articulation_type"],
                "axis": [float(v) for v in extras.get("articulation_axis", (0, 0, 1))],
                "min": float(extras.get("limit_min", 0.0)),
                "max": float(extras.get("limit_max", 0.0)),
            }
    return out


def _convex_parts(mesh, name: str, out_dir: Path, decompose: bool,
                  reuse: bool = False, density: float = 300.0) -> list[Path]:
    """Colliders for one body. A convex mesh is its own collider; a concave one is decomposed.

    This is the difference between a table you can put a chair under and a table that is a solid
    block of wood from floor to top — MuJoCo collides a mesh geom as its convex hull, always.
    """
    if reuse:
        have = sorted(out_dir.glob(f"{name}_col*.obj"))
        if have:
            return have
    written: list[Path] = []
    try:
        hull = mesh.convex_hull
        convex = hull.volume <= 0 or abs(hull.volume - mesh.volume) / max(hull.volume, 1e-9) < CONVEX_TOL
    except Exception:                                   # noqa: BLE001 — degenerate mesh
        convex = True
    try:
        diagonal = float(np.linalg.norm(mesh.extents))
    except Exception:                                   # noqa: BLE001
        diagonal = 1.0
    if convex or not decompose or diagonal < MIN_DECOMP_DIAGONAL:
        path = out_dir / f"{name}_col0.obj"
        mesh.convex_hull.export(path)
        return [path]
    # `preprocess_mode="off"` is not an optimisation, it is the difference between a collider and
    # a slightly larger object. CoACD's default voxel preprocessing remeshes at resolution 50 and
    # inflates the result by ~32% in volume, lifting a table's top surface 7.4 mm above where the
    # mesh actually ends. Every object authored to sit EXACTLY on that surface then starts 7 mm
    # inside it, and MuJoCo resolves the penetration by ejecting it: measured, that threw a chair
    # 66 cm and knocked a photo frame off a shelf onto the floor. With preprocessing off the
    # inflation is 1.7% and the top surface is exact — and it runs 3.5x faster. Voxelising is only
    # needed for meshes CoACD cannot handle directly, so it stays as the fallback.
    parts = _decompose(mesh, budget=_part_budget(mesh, density))
    if not parts:
        path = out_dir / f"{name}_col0.obj"
        mesh.convex_hull.export(path)
        return [path]
    import trimesh
    for i, (verts, faces) in enumerate(parts):
        piece = trimesh.Trimesh(np.asarray(verts), np.asarray(faces))
        if piece.volume <= 1e-9:
            continue
        path = out_dir / f"{name}_col{i}.obj"
        piece.export(path)
        written.append(path)
    return written or [out_dir / f"{name}_col0.obj"]


def _object_physics(room: Path, rec: dict, room_scene, node_names, placed_mesh, yaw, report: dict):
    """The physics the OBJECT states about itself, placed where the room put it — or None.

    This is the difference between a room that is sim-ready and one that merely renders. Without
    it the exporter has to invent every physical number at the last moment from a category table
    and a bounding box: one friction for the whole building, a mass shared out by nothing, a
    hinge pivot guessed from the shape of the leaf. The generated object already knows all three,
    was gated by a solver before it was placed, and carries the answer in a sidecar beside it.

    Returns None — and says why in `report` — whenever the sidecar is missing, unreadable, or
    cannot be matched to the placed geometry. Each of those is a real condition and none of them
    should stop the export: the caller falls back to deriving physics from the placed mesh, which
    is what every room built before this existed already did.
    """
    import trimesh

    source = rec.get("source_glb")
    if not source:
        # Authored straight into `Room.py` — a mug, a cable, a length of trunking — so there is no
        # object package and never was one. That is expected, but it is NOT free: everything
        # physical about this body is about to be invented from a category table and a CoACD run.
        # It used to return here without recording anything, which made the difference invisible:
        # an authored room reported an empty `no_sidecar` while more than half its colliders had
        # been derived at export time, and the stage's warning is keyed on that list.
        report.setdefault("authored_no_package", []).append(rec.get("handle") or "?")
        return None
    name = Path(source).stem
    sim_dir = sim_assets.sidecar_dir(room, name)
    if sim_dir is None:
        report.setdefault("no_sidecar", []).append(name)
        return None
    try:
        model = sim_assets.load(sim_dir / f"{name}.physics.json")
    except Exception as exc:                            # noqa: BLE001 — a bad sidecar is not fatal
        report.setdefault("unreadable_sidecar", []).append(f"{name}: {exc}")
        return None

    object_glb = room.parent / "room_preview" / "Object" / f"{name}.glb"
    if not object_glb.is_file():
        report.setdefault("no_object_glb", []).append(name)
        return None
    try:
        transform = sim_assets.placement_transform(trimesh.load(str(object_glb)), room_scene,
                                                   node_names)
    except Exception as exc:                            # noqa: BLE001
        report.setdefault("unplaceable", []).append(f"{name}: {exc}")
        return None
    if transform is None:
        # NO NODE NAME SURVIVED. An object placed more than once — four chairs from one cluster —
        # has every node renamed for every instance after the first, and a cluster whose own parts
        # already repeat has nothing unique inside its own file either. The placement still has a
        # known shape, so it is recovered from the geometry and the SHELL's yaw instead. Rejecting
        # here would have left exactly the objects a room has several of collided by guesswork.
        transform = sim_assets.placement_from_bbox(model, placed_mesh, yaw)
        if transform is None:
            report.setdefault("unplaceable", []).append(f"{name}: no node matched the room glb")
            return None
        report.setdefault("placed_by_geometry", []).append(name)

    placed = sim_assets.place(model, transform)
    if not placed.links:
        report.setdefault("no_colliders", []).append(name)
        return None
    # PROVE THE PLACEMENT before trusting it. The transform is derived from ONE node (or from two
    # bounding boxes), so a room that had moved the object's parts relative to each other would be
    # silently mis-collided. The colliders are a decomposition of the same geometry the room is
    # holding, so their union has to land on THE PLACED MESH — not on `room_layout.json`'s bbox for
    # the handle, which covers everything grouped under it and is 20 cm larger on a table the
    # author hung nothing on. Comparing against the wrong box rejected a correct placement.
    world = trimesh.util.concatenate([piece for link in placed.links.values()
                                      for piece in link.colliders])
    reference = np.asarray(placed_mesh.bounds, dtype=float)
    span = float(np.max(reference[1] - reference[0])) or 1.0
    error = float(np.abs(np.asarray(world.bounds, dtype=float) - reference).max())
    if error > max(0.05, 0.08 * span):
        report.setdefault("placement_rejected", []).append(
            f"{name}: colliders land {error:.3f} m from the placed mesh")
        return None
    report.setdefault("from_sidecar", []).append(name)
    return placed


def _wall_segments(wall: dict, openings: list[dict], floor_z: float,
                   room_centre: np.ndarray | None = None,
                   room_height: float | None = None) -> list[tuple]:
    """One wall minus its openings -> the boxes that are left.

    A wall carrying a door is not a box: it is two jambs and a lintel. Emitting it whole gives the
    room a door-shaped hole in the render and a solid wall in the physics, so the door cannot open
    and nobody can walk through — the single most obvious way for a "simulation-ready" room to be
    nothing of the kind.
    """
    start = np.asarray(wall["start"], dtype=float)
    end = np.asarray(wall["end"], dtype=float)
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length < 1e-6:
        return []
    along = delta / length
    yaw = math.atan2(along[1], along[0])
    measured = float(wall.get("thickness", 0.1))
    thickness = max(measured, MIN_WALL_THICKNESS)
    # Thicken AWAY from the room. Growing a 0.1 mm plane symmetrically pushes 4.5 cm of new wall
    # into the room, swallowing every skirting board, socket and shelf that was flush against it —
    # measured, that threw the settle drift from 0.9 mm to 2.9 m. The inner face is where the scan
    # put it and must not move; only the outside is invented.
    # 2 mm of setback, always. The authored room mounts its wall panels, dados and splashbacks
    # FLUSH against the wall plane, and a box whose face is exactly on that plane is coplanar with
    # them — two surfaces at the same depth, which the renderer resolves per-pixel and draws as a
    # dithered band across the wall. It reads as dirty plaster and survives any amount of shadow
    # map or anti-aliasing, because it is not a shadow or an edge. Retreating the wall by less than
    # its own render precision puts every panel unambiguously in front of it.
    setback = 0.002
    outward = np.zeros(2)
    if room_centre is not None:
        normal = np.array([along[1], -along[0]])
        mid_wall = (np.asarray(wall["start"], float) + np.asarray(wall["end"], float)) / 2.0
        sign = 1.0 if float(np.dot(normal, mid_wall - room_centre[:2])) > 0 else -1.0
        grow = (thickness - measured) / 2.0 if thickness > measured else 0.0
        outward = normal * sign * (grow + setback)
    # This capture records no wall height at all, and the old fallback of a flat 2.5 m left every
    # wall 20 cm short of a 2.707 m ceiling — a room with a gap running round the top of it, which
    # reads from above as a doll's house with low walls. The room's own floor-to-ceiling distance
    # is the right default: it is measured, and it is what a wall in that room actually is.
    height = float(wall.get("height") or 0.0) or float(room_height or 2.5)

    spans = []
    for o in openings:
        half = o["width"] / 2.0
        t0, t1 = o["offset"] - half, o["offset"] + half
        sill = float(o.get("sill", 0.0))
        spans.append((max(0.0, t0), min(length, t1), sill, sill + float(o["height"])))
    spans.sort()

    boxes = []

    def box(t0, t1, z0, z1):
        if t1 - t0 < 0.02 or z1 - z0 < 0.02:
            return
        mid = start + along * ((t0 + t1) / 2.0) + outward
        boxes.append(((float(mid[0]), float(mid[1]), floor_z + (z0 + z1) / 2.0),
                      ((t1 - t0) / 2.0, thickness / 2.0, (z1 - z0) / 2.0), yaw))

    cursor = 0.0
    for t0, t1, z0, z1 in spans:
        box(cursor, t0, 0.0, height)          # full-height pier before the opening
        box(t0, t1, 0.0, z0)                  # under the sill (a window's spandrel)
        box(t0, t1, z1, height)               # the lintel above it
        cursor = max(cursor, t1)
    box(cursor, length, 0.0, height)
    return boxes


def _quat_z(yaw: float) -> str:
    return f"{math.cos(yaw / 2):.6f} 0 0 {math.sin(yaw / 2):.6f}"


def export(room: Path, out: Path | None = None, *, decompose: bool = True,
           reuse_meshes: bool = False) -> Path:
    """Write `<out>/scene.xml` + `meshes/` from an authored room. Returns the xml path."""
    import trimesh

    room = Path(room).resolve()
    preview = room.parent / "room_preview"
    layout_path = preview / "room_layout.json"
    glb = preview / "Room.glb"
    for p in (room / "Room.py", layout_path, glb):
        if not p.is_file():
            raise SystemExit(f"missing {p} — build the room first (compile_room)")

    out = Path(out) if out else (room.parent / "mujoco")
    meshes = out / "meshes"
    # Decomposing 130 bodies costs ~25 minutes, and every XML-level fix would otherwise pay it
    # again. The meshes depend only on the room's geometry, so they are reusable across any number
    # of iterations on the scene structure.
    if out.exists() and not reuse_meshes:
        shutil.rmtree(out)
    meshes.mkdir(parents=True, exist_ok=True)

    _material_name.__dict__.pop("seen", None)   # a fresh registry per export
    shell = _load_shell(room)
    floor_z = float(shell.get("floor_z", 0.0))
    ceiling_z = float(shell.get("ceiling_z", floor_z + 2.5))
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    _bounds = layout.get("bounds") or {}
    lo = np.asarray(_bounds.get("min") or [-3, -3, floor_z], dtype=float)
    hi = np.asarray(_bounds.get("max") or [3, 3, ceiling_z], dtype=float)
    mid = (lo + hi) / 2.0
    records = {o["handle"]: o for o in layout["objects"]}
    parts, room_scene = _bodies_from_glb(glb, set(records))

    mujoco = ET.Element("mujoco", model=layout.get("scene", "room"))
    # `texturedir` as well as `meshdir`: MuJoCo resolves texture files against their OWN search
    # path, which defaults to the model's directory, so meshes and their textures living together
    # in one folder is not enough to make them both resolve.
    ET.SubElement(mujoco, "compiler", angle="radian", meshdir="meshes", texturedir="meshes",
                  autolimits="true")
    ET.SubElement(mujoco, "option", timestep="0.002", integrator="implicitfast")
    # One user slot per actuator, so an authored joint can carry the velocity limit its recipe
    # stated. MuJoCo rejects a `user` attribute outright unless the space for it is declared here.
    ET.SubElement(mujoco, "size", nuser_actuator="1")
    # MuJoCo's offscreen framebuffer defaults to 640x480, and a Renderer larger than it raises
    # rather than downscaling. A room is worth looking at at more than VGA.
    visual = ET.SubElement(mujoco, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="960")
    # MuJoCo has no bounced light, and a cutaway removes the walls that would have bounced it. The
    # authored lamps then light only what they point at and everything else is black. The headlight
    # is the stand-in for the light a real room gets from its own surfaces; it is deliberately
    # strong on ambient and weak on specular so it fills without flattening the authored lamps.
    # Dimmer than the first attempt: 0.45 ambient washed every authored colour towards white, so
    # the beige walls and the blue carpet both drifted grey. It is a fill for the light MuJoCo
    # cannot bounce, not a substitute for the lamps.
    # Lower than before so the lamps and daylight above actually shape the room. A high ambient
    # flattens everything to the same brightness, which is the other way a render says "synthetic".
    ET.SubElement(visual, "headlight", ambient="0.25 0.25 0.263", diffuse="0.18 0.18 0.18",
                  specular="0.03 0.03 0.03")
    ET.SubElement(visual, "map", shadowclip="6", shadowscale="0.6")
    # QUALITY. MuJoCo defaults to a 1024 px shadow map and 4x multisampling, and at 1024 the shadow
    # map cannot resolve a room: the depth comparison lands between texels and scatters dark
    # speckles over every wall — the "dirty" mottling on the plaster in every frame so far was
    # shadow acne, not a texture. 4096 gives roughly 4 cm of shadow resolution across a 5 m room,
    # and 8x samples clean up the geometry edges that make a render look like a render.
    ET.SubElement(visual, "quality", shadowsize="4096", offsamples="8",
                  numslices="28", numstacks="16")
    default = ET.SubElement(mujoco, "default")
    # `solref` time constant of 0.004 s is exactly 2x the 0.002 s timestep — the hard floor of what
    # MuJoCo can integrate, and at that stiffness contacts buzz. The buzz pumps energy in, and
    # furniture that should not move at all creeps: a chest of drawers slid 62 cm under a 3 m/s^2
    # shake when breaking its friction needs nearly 9. Five timesteps of compliance costs a
    # millimetre of penetration and stops the scene inventing motion that no force produced.
    # 0.9 is a rubber-on-concrete coefficient. Furniture on carpet or vinyl is nearer 0.5, and the
    # difference decides the whole character of a shake: at 0.9 a 5 kg chair needs 8.8 m/s^2 before
    # it will move at all, so any shake gentle enough to leave the room standing left every chair
    # welded to the floor. At 0.55 it starts to slide around 5.4 — a strong tremor, which is what a
    # shake that visibly moves furniture should be.
    ET.SubElement(default, "geom", condim="4", friction="0.55 0.03 0.005",
                  solref="0.01 1", solimp="0.9 0.95 0.001")
    ET.SubElement(ET.SubElement(default, "default", {"class": "visual"}),
                  "geom", group="2", contype="0", conaffinity="0", density="0")
    ET.SubElement(ET.SubElement(default, "default", {"class": "collision"}), "geom", group="3")

    asset = ET.SubElement(mujoco, "asset")
    world = ET.SubElement(mujoco, "worldbody")
    # THE ROOM ITSELF IS A BODY. Oscillating gravity shakes the contents while the walls stand
    # perfectly still, which is not an earthquake — it is a room where physics changed direction.
    # A mocap body is MuJoCo's kinematically-driven rigid body: infinitely heavy, contacts resolved
    # normally, position set from outside each step. Driving it moves the walls, the floor and
    # everything bolted to them together, and the free objects then slide because the FLOOR moved
    # under them — which is also what finally makes a hung door swing, since its frame is what is
    # being shaken.
    # NOT a mocap body. A mocap body is teleported each step, so its children are dragged along
    # kinematically and never feel the acceleration: the free objects still slide (the floor pushes
    # them through contact) but a hung door simply travels with its frame and never swings, because
    # no torque ever reaches the hinge. Three slide joints driven by stiff position servos put the
    # room's motion INSIDE the dynamics instead, so the frame really accelerates and the leaf
    # really swings. The mass is set far above the room's contents so nothing inside can shove it.
    # `gravcomp="1"`: the room is held by position servos, and a servo sags by mg/kp under load —
    # 50 t on kp=2e7 is 2.45 cm, and the whole room with everything in it sank by exactly that,
    # reporting a 2.3 cm "settle drift" for every object in the building. Compensating gravity on
    # the room alone leaves its contents falling normally.
    room_body = ET.SubElement(world, "body", name="room", pos="0 0 0", gravcomp="1")
    ET.SubElement(room_body, "inertial", pos="0 0 0", mass="50000", diaginertia="50000 50000 50000")
    for axis, name in (("1 0 0", "room_x"), ("0 1 0", "room_y"), ("0 0 1", "room_z")):
        ET.SubElement(room_body, "joint", name=name, type="slide", axis=axis,
                      limited="false", damping="200")
    # AND A TWIST. Three sliding axes move every object in the room by the same vector at the same
    # instant, so the whole contents translate together and only differences in friction separate
    # them. A real building also rotates about its vertical, and that is what makes the far corner
    # of a room move further than the middle — the motion objects actually feel is different
    # depending on where they stand.
    ET.SubElement(room_body, "joint", name="room_yaw", type="hinge", axis="0 0 1",
                  limited="false", damping="400")

    synthesised_lights = 0
    lights = layout.get("lights") or []
    if not lights:
        # NO AUTHORED LIGHTING. Only a room whose authoring pass was asked for it records its
        # luminaires; every earlier room has none, and a single fill lamp lights a patch of floor
        # and leaves the walls in hard-edged darkness — which reads as holes in the walls rather
        # than as an unlit room. A plain grid of neutral downlights is not the room's real lighting
        # and does not pretend to be; it is enough to SEE the room, which is the point of a viewer.
        span_x, span_y = float(hi[0] - lo[0]), float(hi[1] - lo[1])
        for ix in (0.3, 0.7):
            for iy in (0.3, 0.7):
                lights.append({
                    "name": f"auto_{ix}_{iy}".replace(".", ""),
                    "position": [lo[0] + ix * span_x, lo[1] + iy * span_y, ceiling_z - 0.12],
                    "power_watts": 26.0, "color_temperature_kelvin": 4000})
        synthesised_lights = len(lights)

    for lamp in lights:
        pos = lamp.get("position") or [0, 0, 0]
        rgb = _kelvin_rgb(lamp.get("color_temperature_kelvin") or 4000)
        # MuJoCo has no photometry: watts only order the lamps by strength. 60 W is taken as full.
        gain = max(0.25, min(1.0, float(lamp.get("power_watts") or 30.0) / 60.0))
        # AIM THE CONE SO IT LANDS ON THE FLOOR AND NOT ON A WALL. MuJoCo fits ONE shadow map across
        # a spot's whole cone, and where that cone meets a surface at a shallow angle the map's depth
        # comparison starts failing against the very surface it measured — a dithered band up the
        # wall. It is not fixed by a bigger shadow map (4096, 8192 and 16384 measure identically),
        # nor by cleaner textures, nor by more anti-aliasing, because none of them touch the cause.
        # Widening the cone only feeds it: at the 160 degrees this once used, every light grazed
        # every wall in the room.
        #
        # A cone whose footprint stops short of the walls cannot graze one. Its floor radius is
        # h*tan(theta/2), so asking for a radius equal to the light's own distance from the nearest
        # wall gives theta = 2*atan(d/h) — which is self-tuning: a lamp in the middle of a small room
        # opens up, and one close to a wall in a big room closes down, both landing on the floor. The
        # clamp keeps it a plausible fitting either way, and what the narrower cone stops reaching is
        # picked up by the directional fills below, which write no shadow map and so cannot band.
        height = max(0.5, float(pos[2]) - floor_z)
        to_wall = min(abs(float(pos[0]) - lo[0]), abs(hi[0] - float(pos[0])),
                      abs(float(pos[1]) - lo[1]), abs(hi[1] - float(pos[1])))
        # 0.8, not 1.0: a footprint that reaches exactly to the wall still grazes its foot, which is
        # where the shadow map fails. Stopping the cone short of the skirting is what actually keeps
        # the wall out of it.
        cutoff_deg = max(45.0, min(75.0,
                                   2.0 * math.degrees(math.atan2(0.8 * max(to_wall, 0.2), height))))
        ET.SubElement(world, "light", name=lamp.get("name", "lamp").replace(" ", "_"),
                      pos=" ".join(f"{v:.4f}" for v in pos), dir="0 0 -1",
                      diffuse=" ".join(f"{c * gain:.3f}" for c in rgb),
                      specular="0.15 0.15 0.15", directional="false",
                      castshadow="true",
                      # A MuJoCo light is a SPOT with a 45 degree cone by default, so a ceiling
                      # panel aimed down lit the floor and left every wall black — the room read as
                      # furniture floating in a void. A real luminaire in a real room throws light
                      # at the walls too, and the walls are most of what you look at.
                      # exponent 0: `exponent=1` tapers towards the cone edge and drew a hard
                      # diagonal line across every wall. A room lamp does not have an edge, and
                      # quadratic falloff over 4 m of room only darkens the far wall.
                      cutoff=f"{cutoff_deg:.1f}", exponent="0", attenuation="1 0 0")
    # A dim fill INSIDE the room, not above its roof: the first export put a single lamp at z = 3
    # on a room whose ceiling is at z = 1.1, so the only thing lighting the interior was MuJoCo's
    # default headlight. It is a fill, not the lighting — the authored lamps above are that.
    # DIRECTIONAL fill, because a spot cannot light a wall. MuJoCo clamps a spotlight's cutoff to
    # 90 degrees, so a ceiling lamp aimed down covers a downward hemisphere and every wall top sits
    # exactly at the cone edge — which rendered as a hard black band across the upper half of every
    # wall and read as holes in the room. A directional light has no cone and no position: it lights
    # whatever faces it, walls included. Two of them, from opposite quarters, so no wall is missed.
    # FOUR fills, one per quarter of the compass. Two left the walls facing away from both in a
    # black band: a directional light illuminates only surfaces turned towards it, and a room has
    # walls pointing every way. Four at 90 degrees apart means every wall faces one of them, and
    # each can then be dim enough not to wash the room flat.
    for i in range(4):
        azimuth = math.radians(45 + 90 * i)
        ET.SubElement(world, "light", name=f"fill_{i}", directional="true",
                      dir=f"{math.cos(azimuth):.4f} {math.sin(azimuth):.4f} -0.55",
                      diffuse="0.40 0.40 0.412", specular="0.01 0.01 0.01", castshadow="false")
    # UPWARD fill. Every other light in the room points down, and a ceiling faces down — so nothing
    # lit it and it rendered black in every frame. In a real room the ceiling is lit almost entirely
    # by light bounced up off the floor, which MuJoCo cannot do, so it is supplied directly.
    ET.SubElement(world, "light", name="bounce", directional="true", dir="0.1 0.1 0.99",
                  diffuse="0.20 0.20 0.19", specular="0 0 0", castshadow="false")

    # DAYLIGHT, aimed the way the room's own windows face. In every scanned room the windows are
    # the dominant source and the reason one side of it is bright — a uniform interior fill makes
    # any room look like a showroom. The direction comes from the wall each window is cut into,
    # pointing INTO the room, so the light arrives through the glass rather than from nowhere.
    inward = np.zeros(2)
    for opening in (shell.get("openings") or {}).values():
        if opening.get("type") != "window":
            continue
        wall = (shell.get("walls") or {}).get(opening.get("wall") or "")
        if not wall:
            continue
        start = np.asarray(wall["start"], dtype=float)
        end = np.asarray(wall["end"], dtype=float)
        along = end - start
        norm = float(np.linalg.norm(along))
        if norm < 1e-6:
            continue
        along /= norm
        normal = np.array([along[1], -along[0]])
        centre_wall = (start + end) / 2.0
        if float(np.dot(normal, mid[:2] - centre_wall)) < 0:
            normal = -normal                    # always the side the room is on
        inward += normal
    if np.linalg.norm(inward) > 1e-6:
        inward /= np.linalg.norm(inward)
        ET.SubElement(world, "light", name="daylight", directional="true",
                      dir=f"{inward[0]:.4f} {inward[1]:.4f} -0.45",
                      diffuse=" ".join(f"{c:.3f}" for c in
                                       (v * 0.55 for v in _kelvin_rgb(6500))),
                      # NO SHADOW from this one. It is a directional light standing in for the sun,
                      # which means it arrives from outside and above — and the ceiling slab is in
                      # the way, so it cast a hard horizontal shadow across the top of every wall
                      # that looked exactly like the walls being cut off half way up. Real daylight
                      # comes through the window, under the ceiling; a shadowless fill from the
                      # window's direction is the honest approximation. The lamps below the ceiling
                      # still cast, so objects keep their shadows.
                      specular="0.06 0.06 0.06", castshadow="false")

    # Colour by KIND, not by appearance. The exported meshes carry no materials, so an untinted
    # scene renders as one grey mass in which nothing can be seen to move. Tinting the bodies that
    # can move against the structure that cannot turns the shake from a grey blur into a readable
    # result: if anything grey moves, the export is wrong.
    ET.SubElement(asset, "material", name="mat_free", rgba="0.85 0.55 0.25 1")
    ET.SubElement(asset, "material", name="mat_fixed", rgba="0.55 0.60 0.68 1")
    ET.SubElement(asset, "material", name="mat_moving", rgba="0.30 0.65 0.45 1")

    stats = {"structure": 0, "free": 0, "attached": 0, "articulated": 0, "colliders": 0,
             "skipped": [], "synthesised_lights": synthesised_lights}
    # A moving part overlaps the thing it moves within — that is what "fits" means, not a defect.
    excludes: list[tuple[str, str]] = []
    actuated: list[tuple[str, float, float]] = []      # (joint, effort, velocity) from a sidecar
    placed: list[tuple[str, "np.ndarray", "np.ndarray", bool]] = []
    hangings: list[tuple[str, float]] = []

    # ── structure: exact boxes from the SHELL, with the openings taken out ────
    by_wall: dict[str, list] = {}
    for oid, o in (shell.get("openings") or {}).items():
        by_wall.setdefault(o.get("wall", ""), []).append(o)
    def structural_material(handle: str, fallback: str) -> str:
        """The material the room was AUTHORED with, not one invented here.

        Walls and the floor are rebuilt as boxes from the SHELL, and the first version painted them
        a hardcoded grey and blue. They are the two largest surfaces in any frame, so that one
        shortcut threw away more of the room's appearance than every prop put together — including
        the floor's fetched 1024x1024 carpet PBR.
        """
        meshes_for = [m for _n, m, _t in (parts.get(handle) or [])]
        if not meshes_for:
            return fallback
        return _material_name(asset, _material_spec(meshes_for[0]), fallback, meshes,
                              reuse_meshes, structural=True)

    wall_points = [p for w in (shell.get("walls") or {}).values() for p in (w["start"], w["end"])]
    room_centre = (np.asarray(wall_points, dtype=float).mean(axis=0) if wall_points
                   else np.zeros(2))
    for wid, wall in (shell.get("walls") or {}).items():
        wall_mat = structural_material(wid, "mat_fixed")
        for i, (pos, size, yaw) in enumerate(_wall_segments(wall, by_wall.get(wid, []), floor_z,
                                                            room_centre,
                                                            ceiling_z - floor_z)):
            ET.SubElement(room_body, "geom", name=f"{wid}_{i}", type="box",
                          pos=" ".join(f"{v:.4f}" for v in pos),
                          size=" ".join(f"{v:.4f}" for v in size),
                          quat=_quat_z(yaw), material=wall_mat, group="1")
            stats["structure"] += 1

    verts = np.asarray((shell.get("floor") or {}).get("verts") or [], dtype=float)
    if len(verts):
        lo, hi = verts[:, :2].min(axis=0), verts[:, :2].max(axis=0)
        centre = (lo + hi) / 2.0
        half = np.maximum((hi - lo) / 2.0, 0.05)
        # The slab sits BELOW `floor_z`, not straddling it. Objects were authored with their
        # undersides exactly on that plane, so a slab centred on it puts every one of them 5 cm
        # inside the floor — and MuJoCo resolves that penetration by launching them. The first
        # export did exactly that: Bag0 travelled 2.09 m before the shake even started.
        for name, handle, z in (("floor", "Floor0", floor_z - 0.05),
                                ("ceiling", "Ceiling0", ceiling_z + 0.05)):
            # Walls in group 1 and the ceiling in group 0, so a renderer can drop either and look
            # into the room. They stay in the model and keep colliding — a cutaway is a VIEW, and a
            # room whose walls stop existing when you look at it is not the room being simulated.
            # NOT group 5: MuJoCo's default `geomgroup` is [1,1,1,1,1,0], so group 5 is invisible
            # unless a viewer turns it on. Every render came back with the walls missing and no
            # error to explain it — they were being hidden by a default, not by any flag.
            # COLLISION ONLY. The floor is a 54-vertex polygon and the room is rotated, so the
            # axis-aligned box that contains it spills well outside the walls — the carpet ran out
            # past the building. A box is still the right CONTACT surface (flat, exact, cheap), so
            # it stays as an invisible collider and the visible floor is the real polygon below.
            ET.SubElement(room_body, "geom", name=name, type="box",
                          pos=f"{centre[0]:.4f} {centre[1]:.4f} {z:.4f}",
                          size=f"{half[0]:.4f} {half[1]:.4f} 0.05",
                          group="3", rgba="0.5 0.5 0.5 0")
            # FACE THE ROOM. A ceiling slab exported from Blender keeps the normals it was built
            # with, which point up and out of the building; seen from inside, every face is a back
            # face and shades black no matter what light is in the room. The floor has the same
            # problem upside down. Flipping any surface whose average normal points away from the
            # room makes it a surface the room can see.
            surface = []
            for _n, mesh, _t in (parts.get(handle) or []):
                converted = _glb_to_shell(mesh)
                want_up = name == "floor"
                try:
                    mean_z = float(np.asarray(converted.face_normals)[:, 2].mean())
                    if (mean_z < 0) if want_up else (mean_z > 0):
                        converted.invert()
                except Exception:                       # noqa: BLE001 — no normals to judge by
                    pass
                surface.append(converted)
            for suffix, _grp, spec in _material_groups(
                    surface, f"{name}_surface", meshes, reuse_meshes):
                ET.SubElement(asset, "mesh", name=suffix, file=f"{suffix}.obj", inertia="shell")
                ET.SubElement(room_body, "geom", {
                    "class": "visual", "type": "mesh", "mesh": suffix,
                    # `structural=True`: the floor and ceiling SURFACES come through here, not
                    # through `structural_material` which the walls use, so they were the one part
                    # of the shell never getting their unobserved regions filled. A scan sees very
                    # little of a floor it is standing on — 41% of Airbnb's and 61% of fallside's
                    # carpet textures were empty — and that void renders as black flooring.
                    "material": _material_name(asset, spec, "mat_fixed", meshes, reuse_meshes,
                                               structural=True),
                    # Group 0, NOT 4. MuJoCo's default geomgroup is [1,1,1,0,0,0]: groups 3, 4 and
                    # 5 are all off unless a viewer turns them on. The walls were invisible in group
                    # 5 for the same reason and were moved to 1; the ceiling was left in 4 and went
                    # on rendering as a black void above the walls through three separate attempts
                    # to fix it as a lighting and then a normals problem. Only 0, 1 and 2 are drawn.
                    "group": "0" if name == "ceiling" else "2"})
            stats["structure"] += 1

    # ── objects ──────────────────────────────────────────────────────────────
    obj_dir = preview / "Object"
    # The yaw `build_room` fitted each asset with. `room_layout.json` records a pose but not this,
    # and without it an object whose node names did not survive the room build cannot be placed.
    shell_objects = shell.get("objects") or {}
    for handle, rec in sorted(records.items()):
        category = rec.get("category", "")
        if category in STRUCTURE:
            continue                                    # already emitted as exact boxes
        pieces = parts.get(handle) or []
        if not pieces:
            stats["skipped"].append(handle)
            continue

        source = rec.get("source_glb")
        placed_mesh = trimesh.util.concatenate([_glb_to_shell(m) for _n, m, _t in pieces])
        physics = _object_physics(room, rec, room_scene, [n for n, _m, _t in pieces], placed_mesh,
                                  math.radians(float(shell_objects.get(handle, {}).get("yaw", 0.0))),
                                  stats)
        joints = {}
        if physics is None and source:
            # No sidecar to read, so fall back to the raw articulation extras: the four numbers a
            # build recipe wrote onto the node. They say which parts move and about what axis, and
            # nothing about mass, friction or where the pivot is.
            joints = _articulation(obj_dir / Path(source).name)

        centre = np.asarray(rec["center"], dtype=float)
        # Match a node to its articulated part by PREFIX, not equality. The room builder appends a
        # hex suffix to any node name it has seen before — the door leaf arrives as
        # `door_leaf_20cf2a` and `door_leaf_f03ab9`, the window's sashes likewise. Only `Lift_Top`
        # happened to be unique, so exact matching articulated the desk and silently left the door
        # and both window sashes welded shut: the joints existed in the source and reached nothing.
        # One part can also own SEVERAL nodes, so they are grouped rather than overwriting.
        moving: dict[str, list] = {}
        fixed = []
        if physics is not None:
            by_node = {n: (m, t) for n, m, t in pieces}
            for link, owned in sim_assets.assign_nodes(list(by_node), physics.model).items():
                # A link the placement could not give colliders to is welded into the carcass
                # rather than emitted as a body that would fall through everything it touches.
                if link == physics.model.root or link not in physics.links:
                    fixed.extend((n, by_node[n][0]) for n in owned)
                else:
                    moving[link] = [by_node[n] for n in owned]
        else:
            for node_name, mesh, node_t in pieces:
                key = None
                for candidate in joints:
                    if node_name == candidate or node_name.startswith(candidate + "_"):
                        if key is None or len(candidate) > len(key):
                            key = candidate
                if key is not None:
                    moving.setdefault(key, []).append((mesh, node_t))
                else:
                    fixed.append((node_name, mesh))

        # INFER what the authoring pass did not declare. `attached_to` only exists in rooms
        # authored to declare it; a room that was not has none, and without it a
        # wall-mounted television is a free rigid body that falls off the wall on the first step —
        # measured, MIL-Meeting's `Television0` dropped 95 cm before the shake even started. The
        # test is the one `scene_init/layout` already uses for the same question: a category that
        # CAN hang, a bottom clearly off the floor, and a wall actually within reach. All three,
        # so a cabinet standing in the middle of a room is still a cabinet standing on the floor.
        hung = None
        if not rec.get("attached_to") and not rec.get("rests_on"):
            # NOTHING UNDER IT MEANS SOMETHING HOLDS IT. Enumerating mountable categories was a
            # losing game — after downlights came alarms, fire blankets and exit signs, and the
            # next room would bring something else. The general fact is simpler: an object whose
            # underside is well clear of the floor and has nothing beneath it is not standing, it
            # is FIXED, whatever it happens to be called. Anything with a surface under it is left
            # free to stand, fall or be knocked over as normal.
            bottom = float(rec["center"][2]) - float(rec["size"][2]) / 2.0
            if _in_category(category, FABRIC):
                hung = "fabric"                          # part of the building, not an object in it
            elif bottom - floor_z > 0.30 and not _supported_from_below(rec, records):
                top = float(rec["center"][2]) + float(rec["size"][2]) / 2.0
                hung = "ceiling" if ceiling_z - top < CEILING_GAP else "wall"
            if hung:
                stats.setdefault("inferred_attached", []).append(f"{handle}->{hung}")

        # A hanging is NOT static: it is a free body held by a weld that the shake can break.
        hanging = (category in HANGING and (rec.get("attached_to") or hung)
                   and not moving and category not in OPENING)
        # Scenery: floor-standing furniture, pinned because that is what it is, not because it
        # happened to be given a door. See `SCENERY`.
        scenery = (not hanging and _in_category(category, SCENERY)
                   and float(rec["center"][2]) - float(rec["size"][2]) / 2.0 - floor_z
                   < SCENERY_FLOOR_GAP)
        if scenery:
            stats.setdefault("scenery", []).append(handle)
        # Trim is static whatever the layout says about it — including an authored `rests_on`,
        # which is the one claim that otherwise skips every other test above.
        trim = not hanging and _in_category(category, TRIM)
        if trim and not (rec.get("attached_to") or hung):
            stats.setdefault("trim_pinned", []).append(handle)
        static = (not hanging and (bool(rec.get("attached_to")) or bool(hung)
                                   or category in OPENING or bool(moving) or scenery or trim))
        body_attrs = {"name": handle, "pos": " ".join(f"{v:.4f}" for v in centre)}
        # A free object hangs off the world; anything fixed to the structure hangs off the ROOM, so
        # it travels with the walls when they move instead of being left behind in mid-air.
        body = ET.SubElement(world if not static else room_body, "body", **body_attrs)
        if not static:
            ET.SubElement(body, "freejoint", name=f"{handle}_free")
            stats["free"] += 1
            placed.append((handle, np.asarray(rec["center"], float),
                           np.asarray(rec["size"], float), bool(hanging)))
            if hanging:
                # THE NAIL, not the whole back of the frame. The body origin is the object's own
                # centre, so its top edge is half its height above that; pulled 10 mm down so the
                # anchor sits in the frame rather than on its rim. `rec["size"]` is the object's
                # own extent — NOT the `lo`/`hi` in this scope, which are the room's 2D bounds.
                nail = np.array([0.0, 0.0, max(float(rec["size"][2]) / 2.0 - 0.01, 0.0)])
                hangings.append((handle, DENSITY.get(category, DENSITY["default"]), nail))
        else:
            placed.append((handle, np.asarray(rec["center"], float),
                           np.asarray(rec["size"], float), False))
            if rec.get("attached_to") or hung:
                stats["attached"] += 1

        tint = "mat_fixed" if static else "mat_free"

        def emit(target, meshes_in, name_prefix, density, material=None, link=None,
                 origin=None):
            origin = centre if origin is None else origin
            shifted = [_glb_to_shell(m) for m in meshes_in]
            for m in shifted:
                m.apply_translation(-origin)

            # VISUAL: one geom per MATERIAL, not one per body. MuJoCo binds a single material to a
            # geom, so merging a chair's fabric and its chrome legs into one mesh forces one colour
            # onto both — which is why the first export rendered the whole room in flat grey while
            # `Room.glb` carried a distinct material on all 1206 of its geometries. Splitting costs
            # extra visual geoms and nothing in physics: they are contype=0.
            for suffix, group, spec in _material_groups(shifted, name_prefix, meshes, reuse_meshes):
                mat_name = material or _material_name(asset, spec, tint, meshes, reuse_meshes, node=suffix)
                # `inertia="shell"` because a VISUAL mesh may legitimately be flat — a monitor's
                # screen, a poster, a sheet of paper. MuJoCo computes an inertia for every mesh at
                # compile time, even one that can never collide, and refuses a zero-volume solid.
                ET.SubElement(asset, "mesh", name=suffix, file=f"{suffix}.obj", inertia="shell")
                ET.SubElement(target, "geom", {"class": "visual", "type": "mesh",
                                               "mesh": suffix, "material": mat_name})

            # COLLISION. The object's OWN colliders when it has any: already convex, already
            # decomposed, already run past a solver on their own before this room existed, and
            # carrying the friction of the material the recipe chose rather than one number for the
            # whole building. Appearance splits the visual geoms above; physics does not follow it.
            if link is not None:
                for i, piece in enumerate(link.colliders):
                    body_local = piece.copy()
                    body_local.apply_translation(-origin)
                    path = meshes / f"{name_prefix}_c{i}.obj"
                    if not (reuse_meshes and path.is_file()):
                        body_local.export(path)
                    ET.SubElement(asset, "mesh", name=f"{name_prefix}_c{i}", file=path.name)
                    ET.SubElement(target, "geom", {
                        "class": "collision", "type": "mesh", "mesh": f"{name_prefix}_c{i}",
                        # No `density`: the mass is stated on the body from the sidecar, and a
                        # density here would have MuJoCo compute a second, different one.
                        "friction": f"{link.friction:.4g} 0.03 0.005"})
                    stats["colliders"] += 1
                stats["sidecar_colliders"] = stats.get("sidecar_colliders", 0) + len(link.colliders)
                return

            # COLLIDE THE PARTS, NOT THEIR UNION. An authored fixture is built out of the shapes it
            # is actually made of — a shelving unit is five boards, a desk is a top and four legs —
            # and each of those is already convex or nearly so. Concatenating them first throws that
            # away and hands CoACD a single carcass to guess at, which it fills: Office-Elliott's
            # Shelving1 came back as one hull 2.4 m tall with the shelves solid inside it. Colliding
            # each part on its own gives exact shelves for no decomposition at all.
            #
            # Only up to a point. A generated chair arrives as forty small parts, and forty geoms on
            # one body is a stack of simultaneous contacts for the solver rather than a better
            # shape, so a body with more parts than this keeps the single-carcass decomposition.
            pieces = shifted if len(shifted) <= MAX_PARTWISE_COLLIDERS else [
                trimesh.util.concatenate(shifted)]
            index = 0
            for part_no, piece in enumerate(pieces):
                stem = name_prefix if len(pieces) == 1 else f"{name_prefix}_p{part_no}"
                for col in _convex_parts(piece, stem, meshes, decompose, reuse_meshes, density):
                    ET.SubElement(asset, "mesh", name=f"{name_prefix}_c{index}", file=col.name)
                    ET.SubElement(target, "geom", {"class": "collision", "type": "mesh",
                                                   "mesh": f"{name_prefix}_c{index}",
                                                   "density": f"{density:.1f}"})
                    index += 1
                    stats["colliders"] += 1

        # MASS STATED EXPLICITLY. The object's own figure when it has one — occupancy density over
        # its bounding box, or a mass the recipe read off a label, split across its links by volume
        # and re-scaled by however much the room stretched it into its RoomPlan box. Otherwise the
        # category table, which is what every room built before the sidecars existed still uses.
        # Leaving MuJoCo to integrate it from the mesh is what produced 1.5 kg chairs and 0.3 kg
        # signs, and a solver resolves a contact on a body that light by throwing it.
        root_link = physics.links.get(physics.model.root) if physics is not None else None
        # STATED WHETHER OR NOT THE BODY CAN MOVE. The mass used to be written only for free
        # bodies, because everything else got it from a `density` on its collision geoms — and the
        # sidecar path deliberately writes no density, since the mass is already known. Left
        # unstated, MuJoCo falls back to its own 1000 kg/m^3 default and a uPVC window frame
        # compiles at 305 kg. Nothing in the scene is dynamically wrong (a fixed body is welded to
        # a 50 t room) but every mass anyone reads out of the model is a fiction.
        if root_link is not None:
            ET.SubElement(body, "inertial",
                          pos=" ".join(f"{v:.6f}" for v in (root_link.com - centre)),
                          mass=f"{max(root_link.mass, MIN_BODY_MASS):.4f}",
                          fullinertia=" ".join(f"{v:.8g}" for v in root_link.inertia))
        elif not static:
            box = np.asarray(rec["size"], dtype=float)
            mass = max(float(np.prod(box)) * DENSITY.get(category, DENSITY["default"]),
                       MIN_BODY_MASS)
            extents = np.maximum(box, 0.02)
            inertia = mass / 12.0 * np.array([extents[1] ** 2 + extents[2] ** 2,
                                              extents[0] ** 2 + extents[2] ** 2,
                                              extents[0] ** 2 + extents[1] ** 2])
            ET.SubElement(body, "inertial", pos="0 0 0", mass=f"{mass:.4f}",
                          diaginertia=" ".join(f"{v:.6f}" for v in inertia))

        density = DENSITY.get(category, DENSITY["default"])
        if fixed:
            emit(body, [m for _, m in fixed], handle, density, link=root_link)

        # each articulated part becomes its own body, hinged where it meets what holds it
        for part_name, part_meshes in moving.items():
            leaf = trimesh.util.concatenate([_glb_to_shell(m) for m, _t in part_meshes])
            part_link = physics.links.get(part_name) if physics is not None else None
            joint = physics.joint_for(part_name) if physics is not None else None
            if joint is not None:
                # THE PIVOT IS AUTHORED. `object.py` placed this hinge and wrote down where it put
                # it; the alternative below has to infer it from the shape of the leaf, which is
                # right for a door with one obvious free edge and wrong for a sash, a lid, a
                # drawer, or any leaf whose hinged edge is not its nearest one. Reading it is the
                # single biggest reason to prefer the sidecar over the extras.
                pivot = physics.pivots[part_name]
                spec = {"axis": physics.axes[part_name], "type": joint.type,
                        "min": joint.limit_lower, "max": joint.limit_upper,
                        "damping": joint.damping, "friction": joint.friction,
                        "effort": joint.effort, "velocity": joint.velocity}
            else:
                raw = joints[part_name]
                local_axis = list(raw["axis"])
                spec = dict(raw, axis=_axis_to_world(local_axis, part_meshes[0][1]),
                            damping=0.05, friction=0.02)
                pivot = _hinge_point(leaf, [_glb_to_shell(m) for _, m in fixed], spec,
                                     local_axis, part_meshes[0][1])
            child = ET.SubElement(body, "body", name=f"{handle}_{part_name}",
                                  pos=" ".join(f"{v:.4f}" for v in (pivot - centre)))
            kind = "slide" if str(spec["type"]).startswith("pris") else "hinge"
            ET.SubElement(child, "joint", name=f"{handle}_{part_name}", type=kind,
                          axis=" ".join(f"{v:.4f}" for v in spec["axis"]),
                          range=f"{spec['min']:.4f} {spec['max']:.4f}",
                          damping=f"{spec['damping']:.4g}",
                          frictionloss=f"{spec['friction']:.4g}", armature="0.002")
            # A JOINT NOTHING CAN DRIVE IS SCENERY. The sidecar states an effort limit — 45 N·m to
            # swing this door, 800 N to raise that desk — compiled from the part the recipe built.
            # It was read for nothing: the export emitted the joint and dropped the number, so the
            # only way to open a door was to push it with another body. An actuator per authored
            # joint is what makes the difference between a room you can look at and one a policy
            # can act in. Recorded only when the sidecar SAID the effort; a joint recovered from
            # the raw extras carries no such statement and inventing one would be a guess.
            if "effort" in spec:
                actuated.append((f"{handle}_{part_name}", float(spec["effort"]),
                                 float(spec.get("velocity") or 0.0)))
            if part_link is not None:
                ET.SubElement(child, "inertial",
                              pos=" ".join(f"{v:.6f}" for v in (part_link.com - pivot)),
                              mass=f"{max(part_link.mass, MIN_BODY_MASS):.4f}",
                              fullinertia=" ".join(f"{v:.8g}" for v in part_link.inertia))
            # The cached mesh was written relative to whatever pivot was computed AT THE TIME.
            # Change how the pivot is derived and `--reuse-meshes` keeps the old geometry while the
            # body origin moves to the new one — the two no longer cancel and the part is drawn
            # metres from where it belongs. Kitchen doors were 1.95 m out and it looked like a
            # physics fault. An articulated part's mesh is a function of its pivot, so it is never
            # reused: the cost is one small export, the alternative is silent misplacement.
            leaf.apply_translation(-pivot)
            for suffix, _grp, vspec in _material_groups([leaf], f"{handle}_{part_name}",
                                                        meshes, reuse=False):
                ET.SubElement(asset, "mesh", name=suffix, file=f"{suffix}.obj",
                              inertia="shell")
                ET.SubElement(child, "geom", {
                    "class": "visual", "type": "mesh", "mesh": suffix,
                    "material": _material_name(asset, vspec, "mat_moving", meshes, reuse_meshes,
                                               node=suffix)})
            if part_link is not None:
                for i, piece in enumerate(part_link.colliders):
                    local = piece.copy()
                    local.apply_translation(-pivot)
                    local.export(meshes / f"{handle}_{part_name}_c{i}.obj")
                    ET.SubElement(asset, "mesh", name=f"{handle}_{part_name}_c{i}",
                                  file=f"{handle}_{part_name}_c{i}.obj")
                    ET.SubElement(child, "geom", {
                        "class": "collision", "type": "mesh",
                        "mesh": f"{handle}_{part_name}_c{i}",
                        "friction": f"{part_link.friction:.4g} 0.03 0.005"})
                    stats["colliders"] += 1
                stats["sidecar_colliders"] = (stats.get("sidecar_colliders", 0)
                                              + len(part_link.colliders))
            else:
                for i, col in enumerate(_convex_parts(leaf, f"{handle}_{part_name}", meshes,
                                                      decompose, False, density)):
                    ET.SubElement(asset, "mesh", name=f"{handle}_{part_name}_c{i}", file=col.name)
                    ET.SubElement(child, "geom", {"class": "collision", "type": "mesh",
                                                  "mesh": f"{handle}_{part_name}_c{i}",
                                                  "density": f"{density:.1f}"})
                    stats["colliders"] += 1
            excludes.append((handle, f"{handle}_{part_name}"))
            stats["articulated"] += 1

    if hangings:
        equality = ET.SubElement(mujoco, "equality")
        for handle, _density, nail in hangings:
            # A NAIL IS A POINT, NOT A CLAMP. This was a `weld`, which fixes all six degrees of
            # freedom: the picture could not tilt, could not swing, and simply rode the wall as if
            # glued to it until the hold broke and it dropped. That is not how anything hangs. A
            # `connect` constrains POSITION at one point and leaves rotation free, which is exactly
            # a nail through the frame's top rail — the picture swings and tilts against the wall it
            # rests on, and the wall contact is what keeps it flat rather than the constraint.
            # `solimp`'s low upper bound keeps the hold soft, so it rattles on its nail before it
            # goes rather than being rigid right up to the instant it is not.
            ET.SubElement(equality, "connect", name=f"hang_{handle}", body1=handle, body2="room",
                          anchor=" ".join(f"{v:.4f}" for v in nail),
                          solref="0.02 1", solimp="0.85 0.92 0.001")
        stats["hangings"] = [h for h, _d, _n in hangings]

    actuator = ET.SubElement(mujoco, "actuator")
    # `room_yaw` is driven on the same terms as the three slides. It was emitted as a joint but
    # never given a servo, and `mujoco_shake` asks for `drive_room_yaw` by name — so the twist the
    # joint exists for silently never happened, and the room was additionally left free to rotate
    # about its vertical under whatever contact torque its contents applied. The gains carry over
    # unchanged because the room's `diaginertia` about z is 50000, numerically equal to its mass,
    # so kp=5e7 puts the rotational resonance at the same 5.0 Hz and kv damps it just as critically.
    for name in ("room_x", "room_y", "room_z", "room_yaw"):
        # CRITICALLY DAMPED, AND STIFF ENOUGH TO STAY OUT OF THE WAY. The room is a 50 t body on a
        # position servo, which is a mass-spring: kp=2e7 put its natural frequency at 3.18 Hz and
        # kv=2e5 left it at a damping ratio of 0.10. The vertical drive runs at 1.93x the base
        # frequency — 3.09 Hz — a ratio of 0.97 to that resonance, so the room bounced about five
        # times harder than commanded. Everything in it was repeatedly unweighted until friction
        # could not hold, then slammed down: chests slid 62 cm and chairs toppled under a shake far
        # too gentle to do either. kp=5e7 moves the resonance to 5.0 Hz — clear of the 3.09 Hz
        # drive without being so stiff that the integrator cannot follow it, which at 2e8 threw a
        # kitchen 172 m. kv = 2*sqrt(kp*m) damps it critically so it tracks without ringing.
        ET.SubElement(actuator, "position", name=f"drive_{name}", joint=name,
                      kp="5e7", kv="3.16e6")

    # ONE MOTOR PER AUTHORED JOINT, AT THE EFFORT THE ASSET STATED.
    #
    # A `motor` rather than a `position` servo, deliberately. A servo holds a setpoint, so emitting
    # one would clamp every door shut and every drawer closed at ctrl=0 — the scene would stop
    # behaving the way it does today and a door would no longer swing when the room is shaken. A
    # motor applies exactly `ctrl` and nothing at rest, so the passive dynamics are bit-for-bit
    # what they were before this existed, and the only change is that the joint can now be driven.
    #
    # `ctrlrange` is the effort the recipe compiled, symmetric because these joints open and close.
    # A door leaf that its own build says needs 45 N·m cannot be driven at 450 by a policy that
    # discovers doing so is cheaper than opening it properly.
    for joint_name, effort, velocity in actuated:
        motor = ET.SubElement(actuator, "motor", name=f"act_{joint_name}", joint=joint_name,
                              gear="1", ctrllimited="true",
                              ctrlrange=f"{-abs(effort):.4g} {abs(effort):.4g}")
        if velocity:
            # MuJoCo has no per-actuator velocity limit, so this cannot be enforced here. It is the
            # asset's own statement about how fast the part may move and the only lossless place to
            # keep it is on the element itself, where a controller can read it back.
            motor.set("user", f"{velocity:.6g}")
    stats["actuated_joints"] = [j for j, _e, _v in actuated]

    # A PICTURE DOES NOT FIGHT THE RAIL IT HANGS BESIDE. These frames are attached to WALLS, but
    # they are authored overlapping separate `PictureRail` fixtures at the same height — Picture1
    # starts 23.6 mm inside PictureRail4. The rail shoves it out, the nail holds it back, and the
    # hook carries 69x the frame's own weight before the room has moved at all, which tears it off a
    # wall that is standing still. The overlap cannot be simulated away, and nudging the frame clear
    # only violates the `connect` anchor — which is fixed at compile time — so MuJoCo drags it back.
    # The pair simply does not collide. Contact with the ROOM is deliberately kept: the wall is what
    # a picture rests flat against, and losing that would let it swing through the plaster.
    for _handle, _centre, _size, _is_hanging in placed:
        if not _is_hanging:
            continue
        lo_h, hi_h = _centre - _size / 2.0, _centre + _size / 2.0
        for _other, _c2, _s2, _other_hanging in placed:
            if _other == _handle or _other_hanging:
                continue
            lo_o, hi_o = _c2 - _s2 / 2.0, _c2 + _s2 / 2.0
            if bool(np.all(hi_h >= lo_o - 0.005)) and bool(np.all(hi_o >= lo_h - 0.005)):
                excludes.append((_handle, _other))

    if excludes:
        contact = ET.SubElement(mujoco, "contact")
        for a, b in excludes:
            ET.SubElement(contact, "exclude", body1=a, body2=b)

    # The capture cameras, as MuJoCo cameras. Rendering the simulation from the SAME viewpoint the
    # room was photographed from is what makes a physics result comparable to the scan rather than
    # merely pretty — and MuJoCo's default free camera, aimed at the origin of a room whose floor
    # sits at z = -1.6, renders black.
    # STRAIGHT DOWN, from just under the ceiling. Above the ceiling would be simpler and would
    # see nothing but the ceiling slab, which is solid geometry like any other. The vertical field
    # of view is computed from the room rather than guessed: at 2.7 m the default 45 degrees covers
    # 2.2 m of a 4.9 m room, so a fixed value either crops the room or is wrong for the next one.
    span = float(max(hi[0] - lo[0], hi[1] - lo[1]))
    height = max(float(ceiling_z - floor_z) - 0.15, 0.5)
    fovy = min(150.0, 2.0 * math.degrees(math.atan((span / 2.0) / height)) + 8.0)
    ET.SubElement(world, "camera", name="topdown",
                  pos=f"{mid[0]:.4f} {mid[1]:.4f} {ceiling_z - 0.15:.4f}",
                  xyaxes="1 0 0 0 1 0", fovy=f"{fovy:.1f}")

    # DOLLHOUSE: outside the room and well above it, looking down into it at roughly 50 degrees.
    # Useless on its own — it sees the backs of the walls — and the intended view once groups 4 and
    # 5 are hidden. Top-down gives the best coverage of the FLOOR; this gives the best coverage of
    # the room, because objects have height and a plan view foreshortens all of it away.
    diag = float(np.linalg.norm(hi[:2] - lo[:2]))
    eye = np.array([mid[0] + 0.55 * diag, mid[1] - 0.55 * diag, ceiling_z + 0.85 * diag])
    fwd = np.array([mid[0], mid[1], floor_z + 0.35 * (ceiling_z - floor_z)]) - eye
    fwd /= max(float(np.linalg.norm(fwd)), 1e-9)
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    right /= max(float(np.linalg.norm(right)), 1e-9)
    ET.SubElement(world, "camera", name="dollhouse",
                  pos=" ".join(f"{v:.4f}" for v in eye),
                  xyaxes=" ".join(f"{v:.4f}" for v in list(right) + list(np.cross(right, fwd))),
                  fovy=f"{min(120.0, 2.0 * math.degrees(math.atan((diag * 0.52) / float(np.linalg.norm(eye - mid)))) ):.1f}")

    # EYE LEVEL, standing in a corner. A room photographed from 1.6 m reads as a room: the walls
    # rise past the frame, the floor recedes, and nothing has to be hidden to see in. The raised
    # views above are for watching everything at once; this is for judging whether it looks right.
    # Placed from the FLOOR POLYGON, not the bounding box. A bbox corner is only inside the room
    # when the room is a rectangle; on MIL-Meeting's angled plan the corner of the box sits inside a
    # wall, and the camera rendered a solid grey slab. Standing back from the centroid towards the
    # most distant floor vertex keeps the viewpoint on the floor that actually exists.
    plate = np.asarray((shell.get("floor") or {}).get("verts") or [], dtype=float)
    if len(plate) >= 3:
        centroid = plate[:, :2].mean(axis=0)
        # AND STANDING SOMEWHERE EMPTY. Retreating to the furthest floor corner put the camera
        # 3.8 cm from the bed in Airbnb-Cam — effectively standing on it — and the view filled with
        # a duvet seen from within, which reads as a bed flying through the air. The simulation was
        # correct the whole time: nothing in that room ever rose more than 70 cm. So every candidate
        # is scored against the furniture it would be standing in, and the clearest wins.
        standing = [(np.asarray(o["bbox_min"], float)[:2], np.asarray(o["bbox_max"], float)[:2])
                    for o in records.values()
                    if o.get("category") not in STRUCTURE
                    and float(o["bbox_max"][2]) - floor_z > 0.35]
        segments = [(np.asarray(w["start"], float), np.asarray(w["end"], float))
                    for w in (shell.get("walls") or {}).values()]

        def _room_clearance(point):
            """Distance to the nearest thing the camera could be standing in — WALLS INCLUDED.

            Scoring against furniture alone put the camera 5 cm from a wall in the kitchen, with a
            cabinet slab filling the frame: it was 0.63 m from the nearest object and inside the
            wall behind it. A viewpoint has to be clear of the room as well as of its contents.
            """
            gaps = [float(np.linalg.norm(point - np.clip(point, f_lo, f_hi)))
                    for f_lo, f_hi in standing]
            for a, b in segments:
                span = b - a
                length2 = float(span @ span)
                t = 0.0 if length2 < 1e-9 else float(np.clip((point - a) @ span / length2, 0.0, 1.0))
                gaps.append(float(np.linalg.norm(point - (a + t * span))))
            return min(gaps) if gaps else 9.9

        def _view_depth(point):
            """How far the camera can SEE towards the middle before something blocks it.

            Clearance alone asks only how much room the camera has to stand in, and answers the
            wrong question: a spot with a metre of space in every direction still scores well with a
            2.5 m wall exactly a metre in front of it, and the render is then a grey slab with a
            room somewhere behind it. What matters is the distance along the direction the camera
            actually looks.
            """
            ray = centroid - point
            reach = float(np.linalg.norm(ray))
            if reach < 1e-6:
                return 0.0
            ray = ray / reach
            nearest = reach
            for f_lo, f_hi in standing:
                # slab test against the footprint, in the plane
                t0, t1 = 0.0, nearest
                for axis in (0, 1):
                    if abs(ray[axis]) < 1e-9:
                        if not (f_lo[axis] <= point[axis] <= f_hi[axis]):
                            t0 = nearest + 1.0
                        continue
                    ta = (f_lo[axis] - point[axis]) / ray[axis]
                    tb = (f_hi[axis] - point[axis]) / ray[axis]
                    t0, t1 = max(t0, min(ta, tb)), min(t1, max(ta, tb))
                if t0 <= t1 and t0 > 0.05:
                    nearest = min(nearest, t0)
            return nearest

        scored = []
        for vertex in plate[:, :2]:
            for pull in (0.30, 0.45, 0.60, 0.72, 0.84):
                candidate = centroid + pull * (vertex - centroid)
                gap = _room_clearance(candidate)
                # Prefer standing back towards the edge, but never at the cost of clearance: a
                # metre of space is worth more than another half metre of setback — and never at
                # the cost of being able to see, which the view depth below is what enforces.
                scored.append((min(gap, 1.2) + 0.15 * pull + 0.45 * min(_view_depth(candidate), 3.0),
                               candidate))
        scored.sort(key=lambda sc: -sc[0])
        eye_xy = scored[0][1] if scored else centroid
    else:
        centroid = mid[:2]
        eye_xy = np.array([lo[0] + 0.12 * (hi[0] - lo[0]), lo[1] + 0.12 * (hi[1] - lo[1])])
    stand = np.array([eye_xy[0], eye_xy[1], floor_z + 1.68])
    look = np.array([centroid[0], centroid[1], floor_z + 1.05])
    fwd = look - stand
    fwd /= max(float(np.linalg.norm(fwd)), 1e-9)
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    right /= max(float(np.linalg.norm(right)), 1e-9)
    ET.SubElement(world, "camera", name="eye",
                  pos=" ".join(f"{v:.4f}" for v in stand),
                  xyaxes=" ".join(f"{v:.4f}" for v in list(right) + list(np.cross(right, fwd))),
                  fovy="72")

    def _look_from(name, eye_pt, look_at, fovy):
        forward = look_at - eye_pt
        forward /= max(float(np.linalg.norm(forward)), 1e-9)
        side = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        side /= max(float(np.linalg.norm(side)), 1e-9)
        ET.SubElement(world, "camera", name=name,
                      pos=" ".join(f"{v:.4f}" for v in eye_pt),
                      xyaxes=" ".join(f"{v:.4f}" for v in list(side) + list(np.cross(side, forward))),
                      fovy=f"{fovy}")

    # A SECOND CORNER, FAR FROM THE FIRST. One viewpoint answers "does this look right"; it cannot
    # answer "is that chair inside the table", because whatever the first corner happens to hide
    # stays hidden however long the video runs. The next-best scoring candidate is usually a hand's
    # width from the best one and shows the same half of the room, so this takes the best candidate
    # that stands at least a third of the room's diagonal away — a different corner by construction,
    # still chosen for clearance rather than picked by hand.
    diag_xy = float(np.linalg.norm(hi[:2] - lo[:2]))
    alt_xy = None
    for _, candidate in (scored if len(plate) >= 3 else []):
        if float(np.linalg.norm(candidate - eye_xy)) >= 0.34 * diag_xy:
            alt_xy = candidate
            break
    if alt_xy is not None:
        alt_stand = np.array([alt_xy[0], alt_xy[1], floor_z + 1.68])
        _look_from("eye_alt", alt_stand, np.array([centroid[0], centroid[1], floor_z + 1.05]), 72)

    # OVERVIEW, from the same clearance-scored spot but just under the ceiling. It used to be placed
    # at a fraction of the BOUNDING BOX, which is only inside the room when the room is a rectangle:
    # on MIL-Meeting's angled plan that corner sits inside a wall and the camera rendered a flat grey
    # slab. The floor polygon is the only description of where the room actually is.
    high_xy = alt_xy if alt_xy is not None else eye_xy
    _look_from("overview", np.array([high_xy[0], high_xy[1], ceiling_z - 0.25]),
               np.array([centroid[0], centroid[1], floor_z + 0.7]), 75)

    for cam in (layout.get("cameras") or []):
        pos = cam.get("position") or [0, 0, 0]
        fwd = np.asarray(cam.get("forward") or [0, 1, 0], dtype=float)
        up = np.asarray(cam.get("up") or [0, 0, 1], dtype=float)
        # MuJoCo's xyaxes wants the camera's own +X (right) and +Y (up); it looks down -Z.
        right = np.cross(fwd, up)
        if np.linalg.norm(right) < 1e-6:
            continue
        right /= np.linalg.norm(right)
        true_up = np.cross(right, fwd)
        ET.SubElement(world, "camera", name=cam.get("name", "cam"),
                      pos=" ".join(f"{v:.4f}" for v in pos),
                      xyaxes=" ".join(f"{v:.4f}" for v in list(right) + list(true_up)))

    xml = out / "scene.xml"
    ET.indent(mujoco, space="  ")
    xml.write_text(ET.tostring(mujoco, encoding="unicode"), encoding="utf-8")

    # WHAT FRACTION OF THIS SCENE'S PHYSICS THE ASSETS ACTUALLY STATED. Every other number in the
    # report counts what was emitted; this one is the only one that says how much of it was
    # invented here. It has to be computed rather than inferred by a reader subtracting two fields,
    # because the honest answer on an authored room is well under half and nothing else says so.
    # DOES THE SCENE WE JUST WROTE ACTUALLY LOAD CLEAN? Everything above reasons about the room
    # from the layout; this is the only step that asks MuJoCo. A body emitted free that starts
    # inside the structure is stored energy — the solver reads the overlap as a compressed spring
    # and ejects it — and until now the first thing to notice was the stability gate, long after
    # the export had reported success. Compiling the file here costs about a second and turns that
    # into a number in the report. Deliberately non-fatal and best-effort: a scene that cannot be
    # loaded here is still written out, because a file you can inspect beats no file at all.
    try:
        import mujoco  # noqa: PLC0415  — optional, and only at load-check time

        _m = mujoco.MjModel.from_xml_path(str(xml))
        _d = mujoco.MjData(_m)
        mujoco.mj_forward(_m, _d)
        _bn = lambda g: mujoco.mj_id2name(                # noqa: E731
            _m, mujoco.mjtObj.mjOBJ_BODY, _m.geom_bodyid[g]) or "?"
        overlaps = {}
        for _c in range(_d.ncon):
            con = _d.contact[_c]
            if con.dist < -0.005:
                pair = " <-> ".join(sorted((_bn(con.geom1), _bn(con.geom2))))
                overlaps[pair] = min(overlaps.get(pair, 0.0), float(con.dist))
        stats["loads_clean"] = not overlaps
        stats["initial_overlaps"] = [{"bodies": k, "mm": round(v * 1000, 1)}
                                     for k, v in sorted(overlaps.items(), key=lambda kv: kv[1])]
    except Exception as exc:                              # noqa: BLE001 — a check is not the export
        stats["loads_clean"] = None
        stats["load_check_error"] = f"{type(exc).__name__}: {exc}"

    derived = int(stats["colliders"]) - int(stats.get("sidecar_colliders", 0))
    stats["derived_colliders"] = max(0, derived)
    stats["sidecar_coverage"] = (round(stats.get("sidecar_colliders", 0) / stats["colliders"], 3)
                                 if stats["colliders"] else None)

    (out / "export_report.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return xml


def _hinge_point(leaf, frame_meshes, spec, local_axis=None, transform=None) -> np.ndarray:
    """Where the joint sits. For a hinge, the leaf edge CLOSEST to what it hangs from.

    The extras record the axis and the limits but not the pivot, and the pivot is what decides
    whether a door swings on its hinges or pirouettes about its middle. Rather than trusting a
    naming convention ("hinges are on +X"), which holds for the door this was written against and
    for nothing else, it is measured: of the two candidate edges perpendicular to the axis, the
    hinge is the one nearer the frame the leaf is hung in.
    """
    if spec["type"].startswith("pris"):
        return (leaf.bounds[0] + leaf.bounds[1]) / 2.0

    # IN THE LEAF'S OWN FRAME. A world axis-aligned bounding box has its corners on the box, not on
    # the door: for a door in a wall running at 40 degrees, the AABB corner sits well off the leaf
    # and the hinge line with it, so the door swept through the wall instead of along it. The leaf
    # was modelled square, so its own frame is where its hinge edge actually is.
    if local_axis is not None and transform is not None:
        composite = _A4 @ np.asarray(transform, dtype=float)
        try:
            local = leaf.copy()
            inverse = np.linalg.inv(composite)
            local.apply_transform(inverse)
            # IN THE SAME FRAME AS THE BOUNDS. `local_axis` is written in the shell's Z-up
            # convention, but `local` above is the leaf in its own glTF Y-up frame, and comparing
            # one against the other silently mixes up which dimension is which. The axis then
            # matched the door's 39 mm THICKNESS instead of its height, `across` fell through to
            # the height, and the pivot landed halfway along the door's width — a cupboard door
            # that pirouettes about its own middle instead of swinging on a stile. It only showed
            # on doors TALLER than they are wide: a squat door's width still won the comparison, so
            # two units in the same kitchen were right and two were wrong. Rotating the world axis
            # back through the same inverse keeps axis and bounds in one frame.
            world_ref = np.asarray(spec["axis"], dtype=float)
            axis = inverse[:3, :3] @ world_ref
            if float(np.linalg.norm(axis)) < 1e-9:
                axis = np.asarray(local_axis, dtype=float)
            axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
            lo, hi = local.bounds
            centre = (lo + hi) / 2.0
            extent = hi - lo
            across = int(np.argmax(np.where(np.abs(axis) > 0.5, -1.0, extent)))
            along = int(np.argmax(np.abs(axis)))
            picks = []
            for edge in (lo[across], hi[across]):
                point = centre.copy()
                point[across] = edge
                point[along] = lo[along]
                picks.append((composite @ np.append(point, 1.0))[:3])
            # A HORIZONTAL hinge axis means a drop-down door — a dishwasher, an oven, a washing
            # machine — and those are hinged along their BOTTOM edge. Choosing by proximity to the
            # frame picked the top edge for the dishwasher, so opening it swung the door up into
            # the worktop instead of down into the room. With a vertical axis both candidates are
            # at the same height and the frame test is still the right discriminator.
            world_axis = (composite[:3, :3] @ axis)
            if abs(world_axis[2]) < 0.5 and len(picks) == 2:
                return picks[int(np.argmin([p[2] for p in picks]))]
            if not frame_meshes:
                return picks[1]
            import trimesh as _tm
            frame = _tm.util.concatenate(frame_meshes)
            gaps = [float(np.min(np.linalg.norm(frame.vertices - p, axis=1))) for p in picks]
            return picks[int(np.argmin(gaps))]
        except Exception:                               # noqa: BLE001 — fall back to the AABB
            pass

    axis = np.asarray(spec["axis"], dtype=float)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
    lo, hi = leaf.bounds
    centre = (lo + hi) / 2.0

    # the in-plane direction of greatest extent is the leaf's width; its two ends are the candidates
    extent = hi - lo
    horizontal = np.argmax(np.where(np.abs(axis) > 0.5, -1.0, extent))
    candidates = []
    for edge in (lo[horizontal], hi[horizontal]):
        p = centre.copy()
        p[horizontal] = edge
        p[int(np.argmax(np.abs(axis)))] = lo[int(np.argmax(np.abs(axis)))]
        candidates.append(p)
    if not frame_meshes:
        return candidates[1]
    import trimesh
    frame = trimesh.util.concatenate(frame_meshes)
    dists = [float(np.min(np.linalg.norm(frame.vertices - c, axis=1))) for c in candidates]
    return candidates[int(np.argmin(dists))]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--room", required=True, type=Path, help="room dir holding Room.py")
    ap.add_argument("--out", type=Path, default=None, help="output dir (default: <room>/../mujoco)")
    ap.add_argument("--reuse-meshes", action="store_true",
                    help="keep the meshes already in <out>/meshes and only rebuild scene.xml")
    ap.add_argument("--no-decompose", action="store_true",
                    help="skip convex decomposition (faster, but concave bodies collide as hulls)")
    a = ap.parse_args(argv)
    xml = export(a.room, a.out, decompose=not a.no_decompose, reuse_meshes=a.reuse_meshes)
    print(f"MJCF -> {xml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
