"""scene_collision.py — robotics-style TRUE-MESH collision model of an authored room.

The box/OBB check (tools/check_collisions) is fast but coarse: a bounding box fills a chair's or
table's concavity, so a chair correctly tucked under a desk reads as a clash. This builds the real
collision model from the compiled `Room.glb` and uses FCL (the same engine MoveIt uses) so the
actual triangles decide contact — a tucked chair whose legs interleave the table legs reads CLEAR.

The scene is decomposed the way a robot cell is:

  * STRUCTURE  — one static body: walls + ceiling + every opening FRAME (the fixed wall structure).
  * FLOOR      — the ground plane (furniture rests ON it, so contact with it is expected, not a clash).
  * FURNITURE  — one body per object (Table0, Chair1, …), the things that must not interpenetrate.
  * LEAVES     — each door leaf / window sash / blind is its own articulated link (revolute door &
                 window, prismatic blind). Checked CLOSED (nothing may sit inside a closed leaf) and
                 by OPEN swing-clearance (furniture must not block it from opening).

Checks (what the caller asked for — no object↔object, no object↔wall):
  object_clash    two furniture bodies' meshes interpenetrate
  wall_clash      a furniture body penetrates the structure (wall / ceiling / opening frame)
  opening_blocked a furniture body sits in a closed leaf, or in a door/window's open swing arc

An Allowed-Collision-Matrix ignores expected contact: furniture↔floor (resting), leaf↔its own
frame, parts of one object. Everything is computed in the `Room.glb` frame; fix vectors are mapped
back to the `SHELL` frame the model edits via  SHELL(x, y) = glb(x, -z).

Requires `python-fcl` (via trimesh.collision) and a compiled `Room.glb`.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path

FURNITURE_PREFIXES = ("Table", "Chair", "Sofa", "Storage", "Desk", "Cabinet", "Shelf", "Bed")
_STRUCTURE_RE = re.compile(r"^(Wall|Ceiling)\d+$")
_FLOOR_RE = re.compile(r"^Floor\d+$")
_OPENING_RE = re.compile(r"^(Door|Window)\d+$")
_FURN_RE = re.compile(r"^(%s)\d+$" % "|".join(FURNITURE_PREFIXES))
# a moving leaf part vs the fixed frame, by GLB part-node name
_FRAME_MARK = ("frame",)
_MOVING_MARK = ("leaf", "sash", "blind", "curtain", "glass", "pane")

WALL_PEN = 0.03   # metres of mesh interpenetration below which a touch is not worth reporting
OBJ_PEN = 0.03


def glb_to_shell_xy(vx: float, vz: float) -> tuple[float, float]:
    """Horizontal glb vector → the SHELL floor-plane vector the model edits: SHELL(x,y)=glb(x,-z)."""
    return vx, -vz


def _handle_of(node, preds, handles):
    """Climb the scene-graph parent chain from a geometry node to its named object handle."""
    cur, seen = node, 0
    while cur is not None and seen < 500:
        if cur in handles:
            return cur
        p = preds.get(cur)
        cur = p[0] if p else None
        seen += 1
    return None


def build_bodies(glb_path: str | Path, shell: dict) -> dict:
    """Group `Room.glb` into collision bodies (all meshes in the glb world frame)."""
    import networkx as nx
    import trimesh

    scene = trimesh.load(str(glb_path))
    G = scene.graph.to_networkx()
    base = scene.graph.base_frame
    preds = nx.predecessor(G, base)  # one BFS: node → [parent]

    furn_cat = {i: o.get("category") for i, o in (shell.get("objects") or {}).items()}
    structure_h, floor_h, furn_h, opening_h = set(), set(), set(), set()
    for n in G.nodes:
        if _STRUCTURE_RE.match(n):
            structure_h.add(n)
        elif _FLOOR_RE.match(n):
            floor_h.add(n)
        elif _OPENING_RE.match(n):
            opening_h.add(n)
        elif furn_cat.get(n) in _FURNITURE_CATS or _FURN_RE.match(n):
            furn_h.add(n)
    handles = structure_h | floor_h | furn_h | opening_h

    parts: dict[str, list] = defaultdict(list)  # handle → [(part_node_name, world mesh)]
    for node in scene.graph.nodes_geometry:
        h = _handle_of(node, preds, handles)
        if h is None:
            continue
        T, gname = scene.graph[node]
        m = scene.geometry[gname].copy()
        m.apply_transform(T)
        parts[h].append((node, m))

    def _concat(meshes):
        meshes = [m for m in meshes if m is not None]
        return trimesh.util.concatenate(meshes) if meshes else None

    structure_meshes = [m for h in structure_h for _, m in parts.get(h, [])]
    leaves: dict[str, "object"] = {}
    for op in opening_h:
        frame = [m for nn, m in parts.get(op, []) if any(k in nn.lower() for k in _FRAME_MARK)]
        moving = [m for nn, m in parts.get(op, []) if not any(k in nn.lower() for k in _FRAME_MARK)]
        structure_meshes += frame  # the frame is fixed wall structure
        lm = _concat(moving)
        if lm is not None:
            leaves[op] = lm

    furniture = {h: _concat([m for _, m in parts.get(h, [])]) for h in furn_h if parts.get(h)}
    furniture = {k: v for k, v in furniture.items() if v is not None}
    return {
        "structure": _concat(structure_meshes),
        "floor": _concat([m for h in floor_h for _, m in parts.get(h, [])]),
        "furniture": furniture,
        "leaves": leaves,
    }


# categories that count as furniture (imported lazily to avoid a heavy qc_room import at module load)
_FURNITURE_CATS = {
    "table", "chair", "sofa", "storage", "television", "bed", "desk", "cabinet",
    "refrigerator", "stove", "oven", "sink", "toilet", "bathtub",
}
# floor-standing subset — the rest (shelf, wall units) are wall-mounted and legitimately float
_FLOOR_STANDING = {
    "table", "chair", "sofa", "storage", "desk", "cabinet", "bed",
    "refrigerator", "stove", "sink", "toilet", "bathtub",
}


def mesh_contacts(bodies: dict) -> dict:
    """True-mesh contact BOOLEANS over the bodies (FCL). We deliberately return contact facts, not
    FCL penetration depth: the furniture meshes are non-watertight, on which FCL's depth is garbage
    (a desk flush against a wall reported 1.16 m). Depth/direction for the FIX comes from the
    reliable SHELL oriented box in the caller; mesh is the accurate yes/no on whether shapes touch —
    which is exactly what boxes get wrong (a chair tucked under a desk).

    Returns {'object_pairs': {frozenset{a,b}, …}, 'leaf_hits': {(obj_id, opening_id), …}}.
    Furniture↔structure is intentionally NOT returned — a desk against a wall is correct contact;
    the caller keeps the box center-in-slab test for genuine "buried in the wall"."""
    from trimesh.collision import CollisionManager

    cm = CollisionManager()
    for oid, m in bodies["furniture"].items():
        cm.add_object(f"obj::{oid}", m)
    for op, m in (bodies.get("leaves") or {}).items():
        cm.add_object(f"leaf::{op}", m)

    hit, names = cm.in_collision_internal(return_names=True, return_data=False)
    object_pairs, leaf_hits = set(), set()
    if hit:
        for a, b in names:
            ao, bo = a.startswith("obj::"), b.startswith("obj::")
            if ao and bo:
                object_pairs.add(frozenset((a[5:], b[5:])))
            elif ao ^ bo:  # object vs a closed leaf
                obj = (a if ao else b)[5:]
                leaf = (b if ao else a)[6:]
                leaf_hits.add((obj, leaf))
    return {"object_pairs": object_pairs, "leaf_hits": leaf_hits}


def _point_in_room(px: float, pz: float, walls: dict) -> bool:
    """Even-odd ray cast of a glb-plan point (x, z) against the wall loop (SHELL(x,y)→glb(x,-y))."""
    inside = False
    edges = [((w["start"][0], -w["start"][1]), (w["end"][0], -w["end"][1]))
             for w in walls.values() if w.get("start") and w.get("end")]
    for (x1, z1), (x2, z2) in edges:
        if (z1 > pz) != (z2 > pz):
            xin = x1 + (pz - z1) * (x2 - x1) / (z2 - z1)
            if px < xin:
                inside = not inside
    return inside


def check_all(bodies: dict, shell: dict) -> list[dict]:
    """Every furniture check, from the REAL meshes — no bounding boxes. Object↔object and
    object↔leaf come from FCL contact; wall penetration, containment and grounding from the actual
    vertices. Fixes are in the SHELL plane the model edits (SHELL(x,y)=glb(x,-z), height=glb y)."""
    fur = bodies["furniture"]
    contacts = mesh_contacts(bodies)
    findings: list[dict] = []

    # object ↔ object — true mesh contact; separation from the meshes' own AABB overlap (not FCL depth)
    for pair in contacts["object_pairs"]:
        a, b = sorted(pair)
        amin, amax = fur[a].bounds
        bmin, bmax = fur[b].bounds
        ox = min(amax[0], bmax[0]) - max(amin[0], bmin[0])
        oz = min(amax[2], bmax[2]) - max(amin[2], bmin[2])
        if ox <= oz:  # push b off a along the shallower-overlap axis
            sign = 1.0 if (bmin[0] + bmax[0]) >= (amin[0] + amax[0]) else -1.0
            gx, gz, depth = sign * ox, 0.0, ox
        else:
            sign = 1.0 if (bmin[2] + bmax[2]) >= (amin[2] + amax[2]) else -1.0
            gx, gz, depth = 0.0, sign * oz, oz
        sdx, sdy = glb_to_shell_xy(gx, gz)
        findings.append({
            "id": b, "kind": "object_clash", "with": a, "overlap_m": round(float(depth), 3),
            "fix": {"move": {"id": b, "world_dxdy": [round(sdx, 3), round(sdy, 3)],
                             "dist_m": round(float(depth), 3)},
                    "or_move": {"id": a, "world_dxdy": [round(-sdx, 3), round(-sdy, 3)]},
                    "or_resize": {"shrink_m": round(float(depth), 3),
                                  "note": f"trim {a} or {b} so their meshes clear"}},
        })

    findings.extend(wall_penetrations(bodies, shell))

    for oid, opening in sorted(contacts["leaf_hits"]):
        findings.append({"id": oid, "kind": "opening_blocked", "opening": opening,
                         "detail": f"{oid} intersects {opening}'s closed leaf — move it clear of the opening"})

    # grounding + containment, from the mesh (glb y = height = SHELL z). Floating/sunk apply only to
    # FLOOR-STANDING furniture — a wall-mounted shelf legitimately sits well above the floor.
    floor_z = shell.get("floor_z")
    ceil_z = shell.get("ceiling_z")
    walls = shell.get("walls") or {}
    cats = {i: o.get("category") for i, o in (shell.get("objects") or {}).items()}
    for oid, m in fur.items():
        lo, hi = m.bounds
        if floor_z is not None and cats.get(oid) in _FLOOR_STANDING:
            gap = lo[1] - floor_z
            if gap > 0.08:
                findings.append({"id": oid, "kind": "floating", "detail": f"{gap:.2f} m above floor"})
            elif gap < -0.05:
                findings.append({"id": oid, "kind": "sunk", "detail": f"{-gap:.2f} m into floor"})
        if ceil_z is not None and hi[1] > ceil_z + 0.05:
            findings.append({"id": oid, "kind": "above_ceiling",
                             "detail": f"top {hi[1]:.2f} > ceiling {ceil_z:.2f}"})
        c = m.centroid
        if walls and not _point_in_room(c[0], c[2], walls):
            findings.append({"id": oid, "kind": "outside_room",
                             "detail": f"{oid}'s centre is outside the room outline"})
    return findings


def wall_penetrations(bodies: dict, shell: dict, *, tol: float = 0.04) -> list[dict]:
    """Furniture whose MESH EXTENT crosses a wall plane to the exterior (a table poking THROUGH a
    wall). The box checks miss this: they test the object CENTRE, and these walls are ~0 mm thick, so
    a piece whose centre is inside but whose body pokes out is never flagged. This scans the object's
    real vertices against each wall plane and reports how far the deepest one crosses to the outside,
    with a SHELL-frame snap-INWARD fix. A piece merely flush against a wall (crossing < tol) is fine.

    Walls come from the SHELL (plan frame); map to the glb plan via SHELL(x, y) → glb(x, -y)."""
    walls = shell.get("walls") or {}
    pts = [(p[0], -p[1]) for w in walls.values() for p in (w.get("start"), w.get("end")) if p]
    if not pts:
        return []
    fcx = sum(p[0] for p in pts) / len(pts)
    fcz = sum(p[1] for p in pts) / len(pts)

    findings = []
    for oid, m in bodies["furniture"].items():
        V = m.vertices  # glb (x, y, z); plan = (x, z)
        best = None
        for wid, w in walls.items():
            s, e = w.get("start"), w.get("end")
            if not s or not e:
                continue
            sx, sz, ex, ez = s[0], -s[1], e[0], -e[1]
            dx, dz = ex - sx, ez - sz
            L = math.hypot(dx, dz)
            if L < 1e-6:
                continue
            ux, uz = dx / L, dz / L
            nx, nz = -uz, ux
            if (fcx - sx) * nx + (fcz - sz) * nz < 0:  # orient the normal toward the room interior
                nx, nz = -nx, -nz
            rx, rz = V[:, 0] - sx, V[:, 2] - sz
            d = rx * nx + rz * nz          # >0 interior of this wall, <0 poked through to exterior
            t = (rx * ux + rz * uz) / L    # position along the wall span
            span = (t >= -0.05) & (t <= 1.05)
            if not span.any():
                continue
            pen = float((-d)[span].max())  # deepest exterior crossing within the wall's span
            if pen > tol and (best is None or pen > best[1]):
                best = (wid, pen, (nx, nz))
        if best:
            wid, pen, (nx, nz) = best
            mv = pen + 0.02
            sdx, sdy = glb_to_shell_xy(nx * mv, nz * mv)  # snap interior, in the SHELL plane
            findings.append({
                "id": oid,
                "kind": "wall_clash",
                "wall": wid,
                "penetration_m": round(pen, 3),
                "fix": {
                    "snap_to_wall": {"id": oid, "world_dxdy": [round(sdx, 3), round(sdy, 3)],
                                     "dist_m": round(mv, 3)},
                    "or_resize": {"shrink_m": round(pen, 3),
                                  "note": f"trim {oid} depth so it clears {wid}"},
                },
            })
    return findings


def open_clearance(bodies: dict, shell: dict, *, swing_steps: int = 6) -> list[dict]:
    """Best-effort OPEN-swing clearance: sweep each leaf from closed toward open about a vertical
    hinge at its outermost vertical edge and flag any furniture the swept volume hits. Approximate
    (hinge/side inferred from geometry), so reported as a soft warning, not a hard clash."""
    import trimesh
    from trimesh.collision import CollisionManager

    warns = []
    for op, leaf in (bodies.get("leaves") or {}).items():
        # hinge: the vertical edge of the leaf AABB. Sweep 0→90° about BOTH vertical edges and keep
        # whichever swing stays inside the room (toward the furniture centroid cloud).
        b = leaf.bounds
        cy = (b[0][1] + b[1][1]) / 2  # glb up-axis is Y
        # candidate hinge lines (vertical) at the two x/z-extreme corners of the footprint
        corners = [(b[0][0], b[0][2]), (b[1][0], b[1][2]), (b[0][0], b[1][2]), (b[1][0], b[0][2])]
        # pick the hinge nearest the structure by using the leaf's short axis as swing radius
        width = max(b[1][0] - b[0][0], b[1][2] - b[0][2])
        hinge = corners[0]
        swept = [leaf]
        for k in range(1, swing_steps + 1):
            ang = math.radians(90.0 * k / swing_steps)
            R = trimesh.transformations.rotation_matrix(ang, [0, 1, 0], [hinge[0], cy, hinge[1]])
            swept.append(leaf.copy().apply_transform(R))
        vol = trimesh.util.concatenate(swept)
        cm = CollisionManager()
        cm.add_object("swing", vol)
        for oid, m in bodies["furniture"].items():
            if cm.in_collision_single(m):
                warns.append({"opening": op, "id": oid, "kind": "open_swing_blocked",
                              "detail": f"{oid} may block {op} from opening (swept ~{width:.2f} m arc)"})
    return warns
