"""fix.py — DETERMINISTIC collision resolution for a built room. No model.

`checks.py` reports interpenetrating furniture; this nudges it apart. Both use the same SAT
collision math (`geometry._obb_mtv_2d`) and the same allow-list (`geometry.EXPECTED_CONTAINMENT`),
so the fixer can never "resolve" something the linter considers correct — pushing an undermount
sink out of its counter would be a regression, not a fix.

Method: iterative relaxation. Each colliding pair is separated along its minimum translation
vector — by construction the shortest move that resolves it — split between the two by MOBILITY:

  * an object up against a wall is ANCHORED (mobility 0). A counter run belongs against its wall;
    the sink in front of it is what should move. Anchored-vs-anchored is reported, never forced.
  * movement is XY only. Vertical placement is floor-grounding's job (`Room.py::_ground_furniture`),
    and a 'floating' violation means a missing support object, not a misplaced one.

Every move is then validated — displacement cap, still inside the floor, not driven into a wall —
and any object failing validation is reverted and reported as unresolved rather than left somewhere
worse than it started. Writes back only the `center` arrays it changed, so the Room.py diff is
reviewable line-by-line.

    python -m litereality_agent.pipeline.compile.qc.fix --room <room dir>            # dry run, prints the plan
    python -m litereality_agent.pipeline.compile.qc.fix --room <room dir> --apply    # edit Room.py in place
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

from litereality_agent.agent.tools.check_collisions.source.geometry import (
    FURNITURE,
    OBJ_PEN,
    _expected,
    _extract_shell,
    _obb_mtv_2d,
    _wall_frame,
)

MAX_NUDGE = 0.30      # m — a bigger correction than this is a reconstruction error, not a nudge
ANCHOR_TOL = 0.12     # m — gap to a wall under which an object counts as placed against it
SEPARATION = 0.005    # m — extra clearance so a resolved pair is separated, not merely touching
MAX_ITERS = 60
MAX_TRIM_FRAC = 0.35   # shortening more than this is not a trim, it is a different object

# Things people slide around: standing against a wall is where they happen to be, not where they
# belong, so they stay mobile. Everything else (counters, cabinets, baths, appliances) is INSTALLED
# — once it is against its wall, that position is load-bearing information from the scan and the
# other object in the pair is the one that should give way.
FREE_STANDING = {"chair", "table", "desk", "sofa", "television"}


def _obb(oid, so, pos=None):
    """(cx, cy, hx, hy, yaw_rad, cz, hz) for a SHELL object, optionally at an overridden XY."""
    cx, cy, cz = so["center"]
    if pos and oid in pos:
        cx, cy = pos[oid]
    w, d, h = so["size"]
    return (cx, cy, w / 2, d / 2, math.radians(so.get("yaw", 0)), cz, h / 2)


def _collisions(items, pos):
    """Every genuine interpenetration among `items` at positions `pos` → (a, b, depth, axis)."""
    out = []
    boxes = [(oid, so, _obb(oid, so, pos)) for oid, so in items]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            (ia, sa, ba), (ib, sb, bb) = boxes[i], boxes[j]
            if _expected(sa.get("category", ""), sb.get("category", "")):
                continue
            zov = min(ba[5] + ba[6], bb[5] + bb[6]) - max(ba[5] - ba[6], bb[5] - bb[6])
            if zov <= 0.05:
                continue                                    # no shared height band → can't collide
            mtv = _obb_mtv_2d(ba[:5], bb[:5])
            if mtv and mtv[0] > OBJ_PEN:
                out.append((ia, ib, mtv[0], mtv[1]))
    return out


def _support(box, n):
    """Half-extent of an oriented box along unit direction n — how far it reaches that way."""
    _, _, hx, hy, r = box[:5]
    c, s = math.cos(r), math.sin(r)
    return abs(hx * (c * n[0] + s * n[1])) + abs(hy * (-s * n[0] + c * n[1]))


def _anchored(box, walls):
    """Is this object placed against a wall? Then it should stay there and the other one moves."""
    cx, cy = box[0], box[1]
    for w in (walls or {}).values():
        fr = _wall_frame(w)
        if not fr:
            continue
        sx, sy, ux, uy, L = fr
        rx, ry = cx - sx, cy - sy
        t = rx * ux + ry * uy
        if not (0.0 <= t <= L):
            continue
        perp = abs(rx * uy - ry * ux)
        n = (uy, -ux)                                       # wall normal
        if perp - _support(box, n) < ANCHOR_TOL + w.get("thickness", 0.1) / 2:
            return True
    return False


def _in_wall(box, walls):
    """Would this object's centre sit inside a wall slab? Mirrors qc_room's wall_clash test."""
    cx, cy = box[0], box[1]
    for w in (walls or {}).values():
        fr = _wall_frame(w)
        if not fr:
            continue
        sx, sy, ux, uy, L = fr
        rx, ry = cx - sx, cy - sy
        t = rx * ux + ry * uy
        if 0.0 <= t <= L and abs(rx * uy - ry * ux) < w.get("thickness", 0.1) / 2 + 0.02:
            return True
    return False


def _floor_bounds(SHELL):
    verts = (SHELL.get("floor") or {}).get("verts") or []
    if not verts:
        return None
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    return min(xs), min(ys), max(xs), max(ys)


def plan_from_mesh(shell_path, mesh, max_nudge=MAX_NUDGE):
    """Turn measured mesh clashes into SHELL nudges.

    Each clash arrives with the smallest translation that actually clears it, measured against the
    real triangles — so nothing here estimates a penetration depth. This only decides WHICH of the
    pair absorbs the move (mobility, as in `plan`) and whether the result is allowed.

    Only objects with a SHELL box can be moved. Wall fixtures are built by code in Room.py and have
    no placement entry, so their clashes come back under `unmovable` with the exact vector needed —
    which is the actionable instruction for a follow-up authoring pass, not a silent drop.
    """
    shell_path = Path(shell_path)
    src = shell_path.read_text()
    SHELL = _extract_shell(src)
    walls = SHELL.get("walls", {})
    shobjs = SHELL.get("objects") or {}

    pos = {oid: [so["center"][0], so["center"][1]] for oid, so in shobjs.items()}
    start = {oid: list(p) for oid, p in pos.items()}
    anchored = {oid: (so.get("category") not in FREE_STANDING and _anchored(_obb(oid, so), walls))
                for oid, so in shobjs.items()}

    unmovable, unresolvable = [], []
    trims: dict[str, tuple] = {}    # oid -> (axis_index, new_size)
    req: dict[str, list] = {}       # oid -> [(dx, dy, amount), ...] requirements to satisfy
    for c in mesh.get("clashes", []):
        a, b, fix = c["a"], c["b"], c.get("fix")
        ina, inb = a in pos, b in pos
        if not fix:
            unresolvable.append((a, b, "no escape within the search budget"))
            continue
        if not ina and not inb:
            unmovable.append((a, b, fix, "neither has a SHELL box to move (walls / code-built)"))
            continue
        # `fix` moves b away from a; a moves the opposite way.
        dx, dy, dist = fix["dir"][0], fix["dir"][1], fix["dist"]

        # THE SHELL vs AN OBJECT — handled BEFORE the mobility test, because the objects that most
        # need this are precisely the anchored ones: a counter buried in its wall is "against a
        # wall" and would otherwise be written off as unmovable. Trimming does not take it off its
        # wall, it shortens it back to where the wall is, so anchoring is irrelevant here.
        shell_side = ("b" if c.get("cat_b") == "shell"
                      else "a" if c.get("cat_a") == "shell" else None)
        if shell_side is not None:
            oid = a if shell_side == "b" else b
            if oid not in pos:
                unmovable.append((a, b, fix, "shell clash, but the object has no SHELL box"))
                continue
            # Solve the box against its wall attachments first: faces already touching a wall are
            # HELD, buried faces snap back to the wall plane. This keeps a corner-fitted appliance
            # in its corner, where shifting the whole box would drag it into its neighbour.
            fit = fit_to_walls(shobjs[oid], walls)
            if fit:
                size, cxy = fit
                trims[oid] = (0, size)
                pos[oid] = cxy
                continue
            # The box is clear of every wall CENTRELINE, yet the mesh still hits the shell. Cause:
            # SHELL records walls as ~0.0001 m planes, but `Room.py::thicken_walls` extrudes the
            # built slab to WALL_THICK (0.10 m) on one side — so "flush to the centreline" can be
            # ~0.1 m inside the real wall. Until wall_faces models the slab rather than the
            # centreline, translating here would only drag the object off its wall attachments and
            # into its neighbour, so report instead of guessing.
            faces = wall_faces(shobjs[oid], walls)
            if faces and all(g >= 0 for _, g in faces.values()):
                unmovable.append((a, b, fix,
                                  "box clear of wall centrelines but mesh hits the slab — "
                                  "wall_faces models a plane, the build extrudes 0.1 m"))
                continue
            edx, edy = (-dx, -dy) if shell_side == "b" else (dx, dy)
            t = trim_against_shell(shobjs[oid], edx, edy, dist)
            if t:
                k, size, cxy = t
                trims[oid] = (k, size)
                pos[oid] = cxy
            else:
                req.setdefault(oid, []).append((edx, edy, dist))   # not fittable — translate out
            continue

        wa = 0.0 if (not ina or anchored.get(a)) else 1.0
        wb = 0.0 if (not inb or anchored.get(b)) else 1.0
        if wa + wb == 0:
            why = ("neither has a SHELL box" if not (ina or inb) else "both are against walls")
            unmovable.append((a, b, fix, why))
            continue
        if ina and wa:
            req.setdefault(a, []).append((-dx, -dy, dist * wa / (wa + wb)))
        if inb and wb:
            req.setdefault(b, []).append((dx, dy, dist * wb / (wa + wb)))

    # Requirements pointing the SAME way are satisfied together — take the largest, don't add them.
    # A table wedged into a corner is reported once per wall, both pushing the same direction; summing
    # them moved it twice as far as anything asked for.
    for oid, rs in req.items():
        merged: dict = {}
        for dx, dy, amt in rs:
            k = (round(dx, 3), round(dy, 3))
            merged[k] = max(merged.get(k, 0.0), amt)
        for (dx, dy), amt in merged.items():
            pos[oid][0] += dx * amt
            pos[oid][1] += dy * amt

    moves, rejected = _validate(pos, start, shobjs, SHELL, walls, max_nudge)
    trims = {k: v for k, v in trims.items() if k not in {r[0] for r in rejected}}
    return {"shell_path": shell_path, "src": src, "objects": shobjs, "pos": pos,
            "before": mesh.get("clashes", []), "after": [], "moves": moves, "trims": trims,
            "rejected": rejected, "unresolvable": unresolvable, "unmovable": unmovable}


def _validate(pos, start, byid, SHELL, walls, max_nudge):
    """Drop any proposed move that trades a collision for something worse."""
    fb = _floor_bounds(SHELL)
    rejected = []
    for oid in list(pos):
        dx, dy = pos[oid][0] - start[oid][0], pos[oid][1] - start[oid][1]
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            continue
        why = None
        if dist > max_nudge:
            why = f"needs {dist:.2f} m (cap {max_nudge:.2f})"
        elif _in_wall(_obb(oid, byid[oid], pos), walls):
            why = "would land inside a wall"
        elif fb and not (fb[0] - 0.10 <= pos[oid][0] <= fb[2] + 0.10
                         and fb[1] - 0.10 <= pos[oid][1] <= fb[3] + 0.10):
            why = "would land outside the room"
        if why:
            rejected.append((oid, dist, why))
            pos[oid] = list(start[oid])
    moves = [(oid, start[oid], pos[oid], math.dist(pos[oid], start[oid]))
             for oid in pos if math.dist(pos[oid], start[oid]) > 1e-6]
    moves.sort(key=lambda m: -m[3])
    return moves, rejected


def plan(room_dir=None, shell_path=None, max_nudge=MAX_NUDGE):
    """Compute the nudges from BOX geometry. Kept as the Blender-free path; `plan_from_mesh` is
    the accurate one when a built room is available."""
    shell_path = Path(shell_path or Path(room_dir) / "Room.py")
    src = shell_path.read_text()
    SHELL = _extract_shell(src)
    walls = SHELL.get("walls", {})
    items = [(oid, so) for oid, so in (SHELL.get("objects") or {}).items()
             if so.get("category") in FURNITURE]

    pos = {oid: [so["center"][0], so["center"][1]] for oid, so in items}
    start = {oid: list(p) for oid, p in pos.items()}
    before = _collisions(items, pos)

    # Mobility is judged ONCE, from the original placement — otherwise an object nudged off its
    # wall would become "mobile" mid-solve and drift across the room.
    anchored = {oid: (so.get("category") not in FREE_STANDING and _anchored(_obb(oid, so), walls))
                for oid, so in items}

    unresolvable = []
    for a, b, depth, _ in before:
        if anchored[a] and anchored[b]:
            unresolvable.append((a, b, depth))

    for _ in range(MAX_ITERS):
        cols = _collisions(items, pos)
        if not cols:
            break
        moved = False
        for a, b, depth, axis in cols:
            wa = 0.0 if anchored[a] else 1.0
            wb = 0.0 if anchored[b] else 1.0
            if wa + wb == 0:
                continue                                    # both pinned to walls — report instead
            push = depth + SEPARATION
            pos[a][0] -= axis[0] * push * (wa / (wa + wb))
            pos[a][1] -= axis[1] * push * (wa / (wa + wb))
            pos[b][0] += axis[0] * push * (wb / (wa + wb))
            pos[b][1] += axis[1] * push * (wb / (wa + wb))
            moved = True
        if not moved:
            break

    byid = dict(items)
    moves, rejected = _validate(pos, start, byid, SHELL, walls, max_nudge)
    return {"shell_path": shell_path, "src": src, "objects": byid, "pos": pos,
            "before": before, "after": _collisions(items, pos), "moves": moves,
            "rejected": rejected, "unresolvable": unresolvable, "unmovable": []}


ATTACH_TOL = 0.03      # m — a face this close to a wall plane is ATTACHED and must not leave it
MAX_REFIT = 0.50       # m — a face further through a wall than this is not a fit problem
REFIT_SCALE = (0.5, 1.5)   # a refit may not halve or 1.5x a dimension; that is a different object

# A face landed EXACTLY on its wall plane is coincident with it, and coincident triangles register as
# an intersection — so snapping flush manufactures a fresh clash, which the next round then "fixes"
# by translating the object off the wall, undoing the fit. Seat constrained faces this far inside the
# wall instead: invisible at 2 mm, but strictly non-intersecting, so the loop converges.
FLUSH_EPS = 0.002


def wall_faces(so, walls):
    """Where each of the object's four side faces sits relative to the wall it faces.

    Pure 2D plan geometry — object box (centre/size/yaw) against the SHELL wall segments. Returns
    {(axis, sign): (wall_id, gap)} where gap is measured along the face's outward normal:
    positive = clear of the wall, ~0 = ATTACHED, negative = driven through it.
    """
    cx, cy = so["center"][0], so["center"][1]
    r = math.radians(so.get("yaw", 0))
    axes = [(math.cos(r), math.sin(r)), (-math.sin(r), math.cos(r))]
    out = {}
    for k, ek in enumerate(axes):
        tc = cx * ek[0] + cy * ek[1]
        half = so["size"][k] / 2.0
        for sgn in (-1, 1):
            face = tc + sgn * half
            best = None
            for wid, w in (walls or {}).items():
                fr = _wall_frame(w)
                if not fr:
                    continue
                wsx, wsy, ux, uy, L = fr
                # the wall must face this axis, and the object must lie along its span
                if abs(uy * ek[0] - ux * ek[1]) < 0.85:
                    continue
                t = (cx - wsx) * ux + (cy - wsy) * uy
                if not (-0.35 <= t <= L + 0.35):
                    continue
                tw = wsx * ek[0] + wsy * ek[1]          # wall plane in this axis' coordinate
                if (tw - tc) * sgn <= 0:
                    continue                            # wall is on the other side of the object
                gap = (tw - face) * sgn
                if best is None or abs(gap) < abs(best[1]):
                    best = (wid, gap)
            if best:
                out[(k, sgn)] = best
    return out


def fit_to_walls(so, walls):
    """Re-fit an object's box to the walls it is attached to.

    The unit of truth is the object PLUS its wall attachments. A dishwasher flush to a back wall and
    a side wall has only one free face; its size is set by the walls, not by the scan box. So rather
    than shifting the whole box (which drags the attached faces off their walls and into the
    neighbouring cabinet), each face is solved independently:

        attached (|gap| <= tol) -> HELD exactly where it is
        through the wall        -> SNAPPED back to the wall plane
        free                    -> left alone; it absorbs the change

    Size and centre both fall out of the resulting face positions. Returns (new_size, new_centre_xy)
    or None when nothing is driven through a wall, or the refit would rescale beyond reason.
    """
    faces = wall_faces(so, walls)
    # Trigger on any face that is not STRICTLY clear of its wall — buried faces obviously, but also
    # already-flush ones, since those are the coincident-triangle case that keeps re-reporting.
    # The trigger sits BELOW the seat distance so a fitted object does not re-trigger on its own
    # 2 mm gap (rounding alone would flip it) — that would make the harness loop churn forever.
    if not any(g < FLUSH_EPS * 0.5 for _, g in faces.values()):
        return None
    cx, cy = so["center"][0], so["center"][1]
    r = math.radians(so.get("yaw", 0))
    axes = [(math.cos(r), math.sin(r)), (-math.sin(r), math.cos(r))]
    size = list(so["size"])
    dc = [0.0, 0.0]
    for k, ek in enumerate(axes):
        tc = cx * ek[0] + cy * ek[1]
        half = size[k] / 2.0
        span = {-1: tc - half, 1: tc + half}
        for sgn in (-1, 1):
            hit = faces.get((k, sgn))
            if not hit:
                continue
            _wid, gap = hit
            # Constrained = touching or moderately buried. Either way the face is seated FLUSH_EPS
            # inside its wall plane: a buried face comes out, an already-flush one is nudged off
            # coincidence. Faces that are genuinely clear (gap >= ATTACH_TOL) are left alone.
            if -MAX_REFIT <= gap < ATTACH_TOL:
                span[sgn] = span[sgn] + sgn * (gap - FLUSH_EPS)
        new_size = span[1] - span[-1]
        lo, hi = REFIT_SCALE
        if not (size[k] * lo <= new_size <= size[k] * hi) or new_size < 0.05:
            return None
        new_tc = (span[1] + span[-1]) / 2.0
        dc[0] += (new_tc - tc) * ek[0]
        dc[1] += (new_tc - tc) * ek[1]
        size[k] = round(new_size, 4)
    return size, [round(cx + dc[0], 4), round(cy + dc[1], 4)]


def trim_against_shell(so, dirx, diry, dist):
    """Pull an object OUT of the wall by shortening it, not by shoving it into the room.

    A counter whose scanned box overshoots into the wall should keep its visible front face exactly
    where the scan put it — that face is the evidence. Translating the whole box moves the front
    face too, and against the shell as one body the required translation is large (half a metre for
    fallside's Storage0) because escaping one wall drives it toward another.

    So: shrink along the object's own axis best aligned with the escape, and shift the centre by
    half that amount in the same direction. The buried face comes out by `dist`; the front face does
    not move at all. Returns (axis_index, new_size, new_center_xy) or None when the escape is not
    aligned with an object axis, where trimming would distort rather than shorten.
    """
    r = math.radians(so.get("yaw", 0))
    axes = [(math.cos(r), math.sin(r)), (-math.sin(r), math.cos(r))]   # local x, y in world
    dots = [dirx * ax + diry * ay for ax, ay in axes]
    k = 0 if abs(dots[0]) >= abs(dots[1]) else 1
    if abs(dots[k]) < 0.85:
        return None                      # escape is diagonal to the box — shortening would skew it
    size = list(so["size"])
    if dist > size[k] * MAX_TRIM_FRAC:
        return None                      # this is not a trim, it is a different-sized object
    ux, uy = axes[k]
    s = 1.0 if dots[k] > 0 else -1.0     # local axis may point opposite the escape
    size[k] -= dist
    c = list(so["center"])
    c[0] += ux * s * dist / 2.0
    c[1] += uy * s * dist / 2.0
    return k, size, [c[0], c[1]]


def _replace_vec3(src: str, oid: str, key: str, xy) -> str:
    """Rewrite the first two components of `key`'s array inside this object's SHELL entry."""
    m = re.search(r'"objects"\s*:\s*\{', src)
    if not m:
        raise ValueError("no SHELL objects block")
    depth, obj_end = 0, len(src)
    for j in range(src.index("{", m.start()), len(src)):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                obj_end = j
                break
    k = re.search(rf'"{re.escape(oid)}"\s*:\s*\{{', src[:obj_end])
    if not k:
        raise ValueError(f"{oid} not found in SHELL objects")
    c = re.compile(rf'("{key}"\s*:\s*\[\s*)(-?[\d.eE+]+)(\s*,\s*)(-?[\d.eE+]+)').search(src, k.end())
    if not c or c.start() > obj_end:
        raise ValueError(f"{oid} has no {key} array")
    return (src[:c.start()]
            + f"{c.group(1)}{round(xy[0], 4)}{c.group(3)}{round(xy[1], 4)}"
            + src[c.end():])


def _replace_center(src: str, oid: str, xy) -> str:
    """Rewrite just this object's `center` X and Y, leaving Z and everything else byte-identical —
    a two-line diff per moved object instead of a reserialised SHELL."""
    return _replace_vec3(src, oid, "center", xy)


def apply(p: dict) -> str:
    src = p["src"]
    for oid, _, new, _ in p["moves"]:
        src = _replace_center(src, oid, new)
    for oid, (_k, size) in p.get("trims", {}).items():
        src = _replace_vec3(src, oid, "size", size)   # x,y only — height is never touched
        src = _replace_center(src, oid, p["pos"][oid])
    return src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", help="room dir containing Room.py")
    ap.add_argument("--shell", help="Room.py path (overrides --room)")
    ap.add_argument("--apply", action="store_true", help="write the nudges into Room.py")
    ap.add_argument("--out", help="write to this path instead of in place")
    ap.add_argument("--max-nudge", type=float, default=MAX_NUDGE,
                    help=f"m; refuse corrections larger than this (default {MAX_NUDGE})")
    a = ap.parse_args()
    if not (a.room or a.shell):
        ap.error("need --room or --shell")
    p = plan(a.room, a.shell, max_nudge=a.max_nudge)

    print(f"COLLISION FIX {p['shell_path']}")
    print(f"  clashes: {len(p['before'])} → {len(p['after'])}   moves: {len(p['moves'])}")
    for oid, o, n, d in p["moves"]:
        print(f"  ↔ {oid:16} {d:.3f} m   ({o[0]:.3f}, {o[1]:.3f}) → ({n[0]:.3f}, {n[1]:.3f})")
    for x, y, depth in p["unresolvable"]:
        print(f"  ! {x:16} vs {y}: both against walls, {depth:.2f} m — needs a human")
    for oid, dist, why in p["rejected"]:
        print(f"  ! {oid:16} reverted: {why}")
    for x, y, depth, _ in p["after"]:
        print(f"  ✗ {x:16} still clashes {y} ~{depth:.2f} m")
    if not p["before"]:
        print("  ✓ no furniture collisions to fix")
        return 0

    if a.apply or a.out:
        dst = Path(a.out) if a.out else p["shell_path"]
        dst.write_text(apply(p))
        print(f"  wrote {dst}  — rebuild the room, then re-run qc_room.py to verify")
    elif p["moves"]:
        print("  (dry run — pass --apply to write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
