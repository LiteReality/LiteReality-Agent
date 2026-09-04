"""repair.py — repair as a search over legal ACTIONS, where no action can make the room worse.

Every solver up to v10 computed a displacement field and hoped the result was better. That is why
the results were sometimes worse than the input: a push that separates one pair drives an object
into a third, and nothing in the loop is obliged to notice. Here the shape is inverted. Each
violation proposes a small number of concrete, human-recognisable actions; each action is applied
to a COPY and scored; and one is kept only if the score strictly improves. A move that creates a
new clash scores worse and is discarded, so a regression cannot be produced — not "is unlikely",
cannot.

The actions are ordered the way a person would try them, least invasive first:

1. ``drop``        one of two boxes that were already sitting on top of each other in the INPUT is
                   a duplicate detection, not a collision. Deleting the extra is free; moving it
                   is nonsense.
2. ``snap``        an installed unit reads as inside its wall because its depth was measured long.
                   Putting it flush against that wall is a restoration, not a correction.
3. ``dewrap``      two units on the SAME wall overlap along it. That is a one-dimensional problem
                   — slide them apart along the wall — and solving it in 2D is what walks a
                   dishwasher off its cabinet line.
4. ``slide``       an installed unit slides along its own wall. It never leaves it.
5. ``separate``    ordinary pair separation along the minimum translation vector, carrying the
                   object's group with it so a table takes its chairs.
6. ``shrink``      a merged box recorded deeper than the thing it bounds, trimmed and re-seated
                   against its wall. Bounded, and only when it beats every cheaper action.

Scoring is lexicographic: violations first, then wall anchors broken, then total displacement. So
an action that removes no violation but moves something is rejected outright, and between two that
both work, the one that disturbs the room less wins.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from . import adjust
from .adjust import (ANCHOR_TOL, FLOOR_OUT_TOL, SEPARATION, Move, check,
                     _floor_triangles, _inside_floor, _obb,
                     _support, ground, reseat_openings)
from .graph import build_graph, obb_mtv, wall_distance
from .shell import object_footprint, wall_frame

__all__ = ["duplicates", "score", "repair", "solve_v11"]

REVIEW_SHIFT = 0.22  # past this, a move is a claim about the room rather than a correction
DUP_SAME = 0.50      # overlap / smaller volume, for two boxes of the same category
DUP_CROSS = 0.75     # more is demanded across categories: a tucked chair reaches ~0.5 honestly
SHRINK_MAX = 0.35    # a box may not lose more than this fraction of a dimension
# Only these get trimmed. RoomPlan merges ADJACENT INSTALLED units — a counter with the wall behind
# it, an oven into its cabinet run — and records the result deeper than the thing it bounds. It
# does not do that to a standalone table, so trimming a table 30% to clear a wall is inventing a
# different table. Airbnb-Cam's Table0 lost a third of its width that way, which is why this list
# exists rather than a blanket permission.
SHRINKABLE = {"storage", "cabinet", "counter", "oven", "stove", "sink", "dishwasher",
              "washer", "refrigerator", "shelf", "bathtub"}
JOINT_SHRINK_MAX = 0.12   # a JOINT trim takes a little off each, never a lot off one
ROOM_BUFFER = 0.10   # the floor plate traces the inner wall face; a flush unit overhangs it


# ── the room, as facts ───────────────────────────────────────────────────────
def attachments(obj, walls, tol: float = ANCHOR_TOL) -> dict[str, float]:
    """Walls this object is installed against → {wall_id: signed gap}."""
    out = {}
    for wall_id, wall in (walls or {}).items():
        measured = wall_distance(obj, wall)
        if measured is None:
            continue
        gap, along_t = measured
        length = wall_frame(wall)[3]
        if abs(gap) <= tol and -0.10 <= along_t <= length + 0.10:
            out[wall_id] = gap
    return out


def _volume(obj) -> float:
    return max(1e-9, obj["size"][0] * obj["size"][1] * obj["size"][2])


def _overlap_fraction(a, b) -> float:
    """Intersection volume over the SMALLER box's volume.

    Min-normalised rather than IoU on purpose: a small duplicate annotation sitting entirely inside
    a larger one scores ~1.0 here, where IoU would stay low and miss it.
    """
    mtv = obb_mtv(_obb(a), _obb(b))
    if not mtv:
        return 0.0
    az0, az1 = a["center"][2] - a["size"][2] / 2, a["center"][2] + a["size"][2] / 2
    bz0, bz1 = b["center"][2] - b["size"][2] / 2, b["center"][2] + b["size"][2] / 2
    z = min(az1, bz1) - max(az0, bz0)
    if z <= 0:
        return 0.0
    # footprint intersection area, approximated from the separating-axis depth against the
    # shorter shared edge — enough to rank duplicates, and it never over-reports
    small = min(a["size"][0] * a["size"][1], b["size"][0] * b["size"][1])
    inter = min(mtv[0] * min(max(a["size"][0], a["size"][1]),
                             max(b["size"][0], b["size"][1])), small)
    return (inter * z) / min(_volume(a), _volume(b))


def duplicates(shell: dict[str, Any]) -> set[tuple[str, str]]:
    """Pairs that were already heavily overlapping in the INPUT — duplicate detections.

    Decided from the input alone, before any solver runs, so it cannot be gamed: a solver has no
    way to manufacture the initial overlap that would license a deletion.
    """
    objects = shell.get("objects") or {}
    ids = list(objects)
    found = set()
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = objects[ids[i]], objects[ids[j]]
            same = a.get("category") == b.get("category")
            if _overlap_fraction(a, b) >= (DUP_SAME if same else DUP_CROSS):
                found.add((ids[i], ids[j]))
    return found


def groups(shell: dict[str, Any]) -> dict[str, set[str]]:
    """parent -> the objects that move with it, so a table carries its chairs."""
    graph = build_graph(shell, name="repair")
    out: dict[str, set[str]] = {}
    for edge in graph.relation("belongs_to"):
        out.setdefault(edge.target, set()).add(edge.source)
    return out


# ── the score every action must beat ─────────────────────────────────────────
def score(shell, baseline, held, exempt: str | None = None) -> tuple[int, int, float, float]:
    """(violations, anchors broken, total penetration, displacement). Lexicographic, lower better.

    Penetration sits third for a reason. Without it a repair that needs two steps can never take
    the first: trimming an over-deep counter clears the wall behind it and reveals a 7 cm clip on
    the wall beside it, so the violation COUNT is unchanged and a count-only gate discards the
    trim — the room is measurably closer to correct and the solver cannot tell. Ranking depth
    below violations and anchors keeps the guarantee (neither can rise) while letting genuine
    progress through, and since every accepted action strictly reduces a quantity bounded at zero,
    the loop still terminates.
    """
    errors = [v for v in check(shell) if v.severity == "error"]
    now = {oid: set(attachments(o, shell.get("walls") or {}))
           for oid, o in (shell.get("objects") or {}).items()}
    # `exempt` is the object an action is resizing. A box that is legitimately smaller than it was
    # recorded cannot still reach both walls of the corner it was merged across, and counting that
    # as damage rejected the only correct repair for an over-deep counter run. Every OTHER object's
    # anchors stay protected — the thing that must never happen is one edit quietly displacing
    # something else.
    broken = sum(len(walls - now.get(oid, set()))
                 for oid, walls in held.items() if oid != exempt)
    moved = 0.0
    for oid, obj in (shell.get("objects") or {}).items():
        was = (baseline.get("objects") or {}).get(oid)
        if was:
            moved += math.dist(obj["center"][:2], was["center"][:2])
    # A furniture item that travelled half a metre from where it was scanned is a defect in its own
    # right, not a tiebreak. Counting it only as "displacement", ranked below penetration, let the
    # search buy one fewer overlap with a 54 cm slide across the room — legal, cheaper on paper,
    # and obviously wrong to anyone looking at the result. Wall-mounted things are exempt: sliding
    # a fixture along its own wall is how a fixture is repaired.
    implausible = 0
    for oid, obj in (shell.get("objects") or {}).items():
        was = (baseline.get("objects") or {}).get(oid)
        if not was or oid == exempt:
            continue
        if math.dist(obj["center"][:2], was["center"][:2]) > REVIEW_SHIFT and oid in held:
            implausible += 1
    penetration = sum(float(v.magnitude or 0.0) for v in errors)
    # A collision and a unit knocked off its wall are counted as the SAME kind of defect, in one
    # term. Ranking violations strictly above anchors let the search trade a dishwasher's wall for
    # one fewer clash, which is a worse room by eye and a better score on paper — Airbnb-Cam's
    # Table0 travelled 54 cm off its wall to remove a single overlap. Equal weight makes the trade
    # explicit instead of automatic, and penetration then breaks the tie.
    return (len(errors) + broken + implausible, round(penetration, 4), round(moved, 4))


# ── the actions ──────────────────────────────────────────────────────────────
def _shift(shell, object_id, delta, family):
    """Move an object and everything that belongs to it, so a group stays a group."""
    for oid in (object_id, *sorted(family.get(object_id, ()))):
        obj = (shell.get("objects") or {}).get(oid)
        if obj:
            obj["center"][0] = round(obj["center"][0] + float(delta[0]), 4)
            obj["center"][1] = round(obj["center"][1] + float(delta[1]), 4)


def _flush_to(obj, wall, inward) -> float:
    """The perpendicular offset an object installed against this wall should have: touching its
    inner face, with nothing of it inside the wall."""
    return _support(obj, inward) + wall.get("thickness", 0.0) / 2.0


def _snap_delta(obj, wall):
    """The move that puts this object's back face on its wall's line, on the side it is already on."""
    start, along, normal, length = wall_frame(wall)
    inward = normal
    centre = np.array(obj["center"][:2], float)
    perp = float(np.dot(centre - start, inward))
    # Far enough that the box is out of the wall AND that `check` agrees. For a 3 cm wardrobe
    # panel on a 0.1 mm wall those differ: its back face clears the wall 1.6 cm out while its
    # centre is still inside the tolerance band, so aiming at the geometry alone leaves the
    # violation exactly where it was.
    clear = max(_support(obj, inward) + wall.get("thickness", 0.0) / 2.0,
                adjust._wall_slab(wall) + SEPARATION)
    want = math.copysign(clear, perp if perp else 1.0)
    return inward * (want - perp)


def _dewrap_delta(a, b, wall):
    """Separate two units on the SAME wall by sliding ALONG it — a 1-D problem, solved in 1-D."""
    start, along, _normal, _length = wall_frame(wall)
    ta = float(np.dot(np.array(a["center"][:2], float) - start, along))
    tb = float(np.dot(np.array(b["center"][:2], float) - start, along))
    half_a, half_b = _support(a, along), _support(b, along)
    gap = abs(tb - ta) - (half_a + half_b)
    if gap >= 0:
        return None
    push = (-gap + SEPARATION) / 2.0
    direction = 1.0 if tb >= ta else -1.0
    return along * (-push * direction), along * (push * direction)


def _floor_correction(obj, triangles, position) -> np.ndarray | None:
    """Shortest vector that brings a footprint hanging off the plate back onto it.

    Each outside corner is pulled to its nearest point on the plate's boundary; the corner that is
    furthest out sets the move. Using the mesh rather than a bounding box matters here: an L-shaped
    or multi-room plate has interior boundary that an AABB cannot see, and pulling an object toward
    the middle of a bounding box can push it straight through the notch.
    """
    if not triangles:
        return None
    corners = object_footprint(obj) + (np.asarray(position, float)
                                       - np.asarray(obj["center"][:2], float))
    worst, best_vec = 0.0, None
    for corner in corners:
        if _inside_floor(corner, triangles):
            continue
        near, dist = None, float("inf")
        for a, b, c in triangles:
            for i, j in ((a, b), (b, c), (c, a)):
                edge = j - i
                length_sq = float(np.dot(edge, edge)) or 1e-12
                t = max(0.0, min(1.0, float(np.dot(corner - i, edge)) / length_sq))
                point = i + t * edge
                d = float(np.linalg.norm(corner - point))
                if d < dist:
                    near, dist = point, d
        if near is not None and dist > worst:
            worst, best_vec = dist, near - corner
    if best_vec is None or worst <= FLOOR_OUT_TOL:
        return None
    norm = float(np.linalg.norm(best_vec)) or 1.0
    return best_vec / norm * (worst + SEPARATION)


def _corner_fit(obj, wall_a, wall_b):
    """Trim just enough, and translate, so a unit touches BOTH walls of its corner.

    The case a human reads instantly and the generic solver cannot: a wardrobe stands in a corner,
    is recorded slightly too deep, and reads as inside one wall while sitting 11 cm off the other.
    Pushing it out of the first wall answers the violation and leaves it floating away from the
    second — which is worse, not better, and is exactly what the plain shrink did here. The unit
    belongs against both, so the repair is to make it fit the corner.
    """
    deltas, size = [], list(obj["size"])
    for wall in (wall_a, wall_b):
        start, along, normal, _length = wall_frame(wall)
        centre = np.array(obj["center"][:2], float)
        perp = float(np.dot(centre - start, normal))
        want = math.copysign(_support(obj, normal) + wall.get("thickness", 0.0) / 2.0,
                             perp if perp else 1.0)
        gap = abs(perp) - abs(want)          # + means it stands off the wall, - means it is in it
        deltas.append((normal, perp, want, gap))
    # trim the axis facing whichever wall it is INSIDE, by that much, then re-seat against both
    yaw = math.radians(obj.get("yaw", 0.0))
    axes = (np.array([math.cos(yaw), math.sin(yaw)]), np.array([-math.sin(yaw), math.cos(yaw)]))
    for normal, _perp, _want, gap in deltas:
        if gap >= -1e-4:
            continue
        which = 0 if abs(float(np.dot(axes[0], normal))) > abs(float(np.dot(axes[1], normal))) else 1
        trimmed = size[which] + gap - SEPARATION
        if trimmed < size[which] * (1.0 - SHRINK_MAX) or trimmed < 0.12:
            return None
        size[which] = round(trimmed, 4)
    return size


def _axis_facing(obj, direction) -> int:
    """Which of the box's own two horizontal axes points most along ``direction``."""
    yaw = math.radians(obj.get("yaw", 0.0))
    axis_x = np.array([math.cos(yaw), math.sin(yaw)])
    axis_y = np.array([-math.sin(yaw), math.cos(yaw)])
    return 0 if abs(float(np.dot(axis_x, direction))) > abs(float(np.dot(axis_y, direction))) else 1


def _joint_shrink(a, b, axis, depth):
    """Take HALF the overlap off each of two boxes, along the axis they overlap on.

    The case this exists for is a room that is simply over-full once everything is where it belongs.
    Both units are correctly against their walls, neither may move without leaving them, and the
    only remaining freedom is size — so the right answer is a few centimetres off each rather than
    a large trim on one or a large slide on either. Capped hard: a joint trim is a small correction
    to two slightly generous boxes, and if it needs to be more than that the problem is not size.
    """
    out = []
    for obj in (a, b):
        which = _axis_facing(obj, np.asarray(axis, float))
        size = list(obj["size"])
        take = depth / 2.0 + SEPARATION
        if take > size[which] * JOINT_SHRINK_MAX:
            return None                      # too much to call it a trim
        size[which] = round(size[which] - take, 4)
        if size[which] < 0.12:
            return None
        out.append(size)
    return out


def _shrink_to_clear(obj, wall, depth):
    """Trim the axis that faces this wall by what is buried, then re-seat against it."""
    start, along, normal, _length = wall_frame(wall)
    yaw = math.radians(obj.get("yaw", 0.0))
    axis_x = np.array([math.cos(yaw), math.sin(yaw)])
    axis_y = np.array([-math.sin(yaw), math.cos(yaw)])
    which = 0 if abs(float(np.dot(axis_x, normal))) > abs(float(np.dot(axis_y, normal))) else 1
    size = list(obj["size"])
    # Trim by what is buried, ONCE. Doubling it would be right for a shrink about the centre, but
    # the box is re-seated against the wall immediately afterwards, so the far face does not move.
    # With the doubling, every real case exceeded SHRINK_MAX and no shrink was ever offered.
    trimmed = size[which] - (depth + SEPARATION)
    if trimmed < size[which] * (1.0 - SHRINK_MAX) or trimmed < 0.12:
        return None
    size[which] = round(trimmed, 4)
    return size, which


@dataclass
class Candidate:
    """One legal repair, ready to be tried and kept only if it earns its place."""

    kind: str
    detail: str
    apply: Callable[[dict], None]
    exempt: str | None = None        # an object whose own anchors this action may legitimately change


def _candidates(shell, violation, held, family, dups) -> list[Candidate]:
    """Everything worth trying for one violation, cheapest and least invasive first."""
    objects = shell.get("objects") or {}
    walls = shell.get("walls") or {}
    obj = objects.get(violation.object)
    if obj is None:
        return []
    out: list[Candidate] = []

    # 1. a duplicate detection is deleted, never rearranged
    for a_id, b_id in dups:
        if violation.object not in (a_id, b_id):
            continue
        other = b_id if a_id == violation.object else a_id
        for victim in (violation.object, other):
            if victim in family:            # never delete something that hosts a group
                continue
            out.append(Candidate("drop", f"{victim} duplicates {a_id if victim == b_id else b_id}",
                                 (lambda v: lambda s: s["objects"].pop(v, None))(victim)))

    if violation.kind in ("wall_clash", "outside_room"):
        wall_id = violation.other if violation.other in walls else None
        # ONLY walls this object is genuinely installed against. The old 0.75 m sweep offered any
        # wall within three quarters of a metre as somewhere the object "belongs", and Airbnb-Cam's
        # Table0 was duly snapped onto a wall 50 cm away — abandoning the wall it was actually
        # touching. Half a metre from a wall is not against it.
        near = attachments(obj, walls, tol=ANCHOR_TOL)

        # 2a. the minimal repair: push straight out of the wall it is inside, by exactly the depth
        if wall_id:
            wall = walls[wall_id]
            depth = adjust._wall_crossing(obj, wall)
            if depth > 0:
                _start, _along, normal, _length = wall_frame(wall)
                centre = np.array(obj["center"][:2], float)
                side = math.copysign(1.0, float(np.dot(centre - _start, normal)) or 1.0)
                push = normal * side * (depth + SEPARATION)
                out.append(Candidate("push", f"{violation.object} out of {wall_id}",
                                     (lambda d, o: lambda s: _shift(s, o, d, family))(
                                         push, violation.object)))

        # 2b. keep the wall it IS against and slide ALONG it until clear of the one it crosses.
        # This is what a person does, and until now nothing in the action set could express it:
        # the only wall repair was "seat it against wall X", so clearing wall B meant adopting
        # some other wall as home.
        if wall_id:
            crossed = walls[wall_id]
            for home_id in sorted(near):
                if home_id == wall_id:
                    continue
                _s, along, _n, _l = wall_frame(walls[home_id])
                for direction in (1.0, -1.0):
                    probe = copy.deepcopy(obj)
                    step = 0.0
                    for _ in range(60):                 # 2 cm steps, up to 1.2 m
                        step += 0.02
                        probe["center"][0] = obj["center"][0] + along[0] * direction * step
                        probe["center"][1] = obj["center"][1] + along[1] * direction * step
                        if adjust._wall_crossing(probe, crossed) <= 0.0:
                            break
                    else:
                        continue
                    delta = along * direction * (step + SEPARATION)
                    out.append(Candidate(
                        "slide", f"{violation.object} along {home_id}, clear of {wall_id}",
                        (lambda d, o: lambda s: _shift(s, o, d, family))(delta, violation.object)))

        for candidate_wall in ([wall_id] if wall_id else []) + sorted(near):
            wall = walls.get(candidate_wall or "")
            if not wall:
                continue
            # 2. put it back flush against its wall
            delta = _snap_delta(obj, wall)
            if float(np.linalg.norm(delta)) > 1e-4:
                out.append(Candidate("snap", f"{violation.object} flush to {candidate_wall}",
                                     (lambda d, o: lambda s: _shift(s, o, d, family))(delta,
                                                                                      violation.object)))
            # 6. a box measured deeper than the thing it bounds, trimmed and re-seated
            depth = adjust._wall_crossing(obj, wall)
            may_shrink = obj.get("category", "") in SHRINKABLE and bool(near)
            trimmed = _shrink_to_clear(obj, wall, depth) if (depth > 0 and may_shrink) else None
            if trimmed:
                size, _which = trimmed

                def _do(o=violation.object, sz=size, w=candidate_wall):
                    def run(s):
                        s["objects"][o]["size"] = sz
                        s["objects"][o]["center"][0] += float(_snap_delta(s["objects"][o],
                                                                         s["walls"][w])[0])
                        s["objects"][o]["center"][1] += float(_snap_delta(s["objects"][o],
                                                                          s["walls"][w])[1])
                    return run
                out.append(Candidate("shrink",
                                     f"{violation.object} trimmed to clear {candidate_wall}",
                                     _do(), exempt=violation.object))

    if violation.kind == "outside_room":
        # Bring it back onto the plate. Measured from the nearest point INSIDE the room, so an
        # object whose scan pose is outside can always be pulled back regardless of how far out it
        # started — without this there was no action at all for a box off the floor.
        triangles = _floor_triangles(shell)
        back = _floor_correction(obj, triangles, tuple(obj["center"][:2]))
        if back is not None:
            out.append(Candidate("return", f"{violation.object} back onto the floor plate",
                                 (lambda d, o: lambda s: _shift(s, o, d, family))(back,
                                                                                  violation.object)))

    if violation.kind in ("wall_clash", "outside_room") and obj.get("category", "") in SHRINKABLE:
        near2 = sorted(attachments(obj, walls, tol=0.45))
        for i in range(len(near2)):
            for j in range(i + 1, len(near2)):
                wall_a, wall_b = walls[near2[i]], walls[near2[j]]
                head_a = math.degrees(math.atan2(*wall_frame(wall_a)[1][::-1])) % 180
                head_b = math.degrees(math.atan2(*wall_frame(wall_b)[1][::-1])) % 180
                if min(abs(head_a - head_b), 180 - abs(head_a - head_b)) < 25.0:
                    continue                      # same direction: not a corner
                fitted = _corner_fit(obj, wall_a, wall_b)
                if not fitted:
                    continue

                def _fit(o=violation.object, sz=fitted, wa=near2[i], wb=near2[j]):
                    def run(s):
                        target = s["objects"][o]
                        target["size"] = sz
                        for wall_id in (wa, wb):
                            d = _snap_delta(target, s["walls"][wall_id])
                            target["center"][0] += float(d[0])
                            target["center"][1] += float(d[1])
                    return run
                out.append(Candidate("corner_fit",
                                     f"{violation.object} fitted into {near2[i]}/{near2[j]}",
                                     _fit(), exempt=violation.object))

    if violation.kind == "fixture_over_opening":
        opening = (shell.get("openings") or {}).get(violation.other or "")
        wall = walls.get((opening or {}).get("wall") or "") if opening else None
        if wall:
            slid = adjust._fixture_escape(obj, wall, opening, tuple(obj["center"][:2]))
            if slid is not None:
                delta = np.asarray(slid, float) - np.asarray(obj["center"][:2], float)
                out.append(Candidate("slide", f"{violation.object} clear of {violation.other}",
                                     (lambda d, o: lambda s: _shift(s, o, d, family))(
                                         delta, violation.object)))

    if violation.kind == "object_clash" and violation.other in objects:
        other = objects[violation.other]
        shared = set(attachments(obj, walls)) & set(attachments(other, walls))
        # 3. two units on the same wall separate ALONG it
        for wall_id in sorted(shared):
            pair = _dewrap_delta(obj, other, walls[wall_id])
            if pair:
                da, db = pair
                out.append(Candidate("dewrap", f"{violation.object}/{violation.other} along {wall_id}",
                                     (lambda a, b, x, y: lambda s: (_shift(s, a, x, family),
                                                                    _shift(s, b, y, family)))(
                                         violation.object, violation.other, da, db)))
        # 5b. both correctly placed and still overlapping: take a little off each
        mtv_joint = obb_mtv(_obb(obj), _obb(other))
        if mtv_joint:
            pair = _joint_shrink(obj, other, mtv_joint[1], mtv_joint[0])
            if pair:
                def _joint(a_id=violation.object, b_id=violation.other, sizes=pair):
                    def run(s):
                        for oid, size in zip((a_id, b_id), sizes):
                            s["objects"][oid]["size"] = size
                            for wall_id in sorted(attachments(s["objects"][oid],
                                                              s.get("walls") or {})):
                                d = _snap_delta(s["objects"][oid], s["walls"][wall_id])
                                s["objects"][oid]["center"][0] += float(d[0])
                                s["objects"][oid]["center"][1] += float(d[1])
                    return run
                out.append(Candidate(
                    "joint_shrink",
                    f"{violation.object}+{violation.other} each trimmed {mtv_joint[0] / 2 * 100:.0f}cm",
                    _joint(), exempt=violation.object))

        # 4/5. otherwise separate along the MTV, moving whichever side is free
        mtv = obb_mtv(_obb(obj), _obb(other))
        if mtv:
            push = np.array(mtv[1], float) * (mtv[0] + SEPARATION)
            a_fixed = len(attachments(obj, walls)) > 1
            b_fixed = len(attachments(other, walls)) > 1
            options = []
            if not b_fixed:
                options.append((violation.other, push))
            if not a_fixed:
                options.append((violation.object, -push))
            if not options:
                options = [(violation.other, push * 0.5), (violation.object, -push * 0.5)]
            for who, delta in options:
                held_walls = attachments(objects[who], walls)
                if len(held_walls) == 1:      # installed: project the push onto its wall
                    wall = walls[next(iter(held_walls))]
                    along = wall_frame(wall)[1]
                    delta = along * float(np.dot(delta, along))
                if float(np.linalg.norm(delta)) < 1e-4:
                    continue
                out.append(Candidate("separate", f"{who} away from its neighbour",
                                     (lambda w, d: lambda s: _shift(s, w, d, family))(who, delta)))
    return out


def repair(shell: dict[str, Any], *, max_rounds: int = 14, beam: int = 3,
           on_stuck: Callable[[dict, list], dict] | None = None) -> tuple[dict, list[dict]]:
    """Search legal actions for the best room reachable, and return the best one SEEN.

    Strictly greedy was not enough, and Airbnb-Cam is the proof: from its best greedy state every
    single available action scores worse — 2 violations become 3, 4, or 5 — while the room is
    plainly not finished. Two objects need to move together and no single move can start the pair.

    So the search keeps a small beam and is allowed to step through a worse state to reach a better
    one. The guarantee is unchanged, because it is enforced at the END rather than at every step:
    the state returned is the best state ever visited, and the input is in that set. A room can
    therefore never come back worse than it went in — the search simply has room to manoeuvre on
    the way.
    """
    baseline = copy.deepcopy(shell)
    work, moves = ground(copy.deepcopy(shell))
    work, opening_moves = reseat_openings(work)
    held = {oid: set(attachments(o, work.get("walls") or {}))
            for oid, o in (work.get("objects") or {}).items()}
    held = {k: v for k, v in held.items() if v}
    dups = duplicates(baseline)
    family = groups(work)
    log: list[dict] = []

    start_score = score(work, baseline, held)
    frontier = [(start_score, work, [])]
    best_score, best_state, best_log = start_score, work, []

    for _round in range(max_rounds):
        expanded = []
        for state_score, state, trail in frontier:
            errors = [v for v in check(state) if v.severity == "error"]
            if not errors:
                continue
            for violation in errors:
                for candidate in _candidates(state, violation, held, family, dups):
                    trial = copy.deepcopy(state)
                    try:
                        candidate.apply(trial)
                    except Exception:                   # noqa: BLE001 — a bad action is just skipped
                        continue
                    trial_score = score(trial, baseline, held, exempt=candidate.exempt)
                    step = trail + [{"action": candidate.kind, "detail": candidate.detail,
                                     "defects": trial_score[0]}]
                    expanded.append((trial_score, trial, step))
                    if trial_score < best_score:
                        best_score, best_state, best_log = trial_score, trial, step
        if not expanded or best_score[0] == 0:
            break
        expanded.sort(key=lambda item: item[0])
        frontier = expanded[:beam]

    work, log = best_state, best_log
    current = best_score

    if on_stuck is not None:
        errors = [v for v in check(work) if v.severity == "error"]
        if errors:
            proposed = on_stuck(work, errors)
            if proposed is not None:
                trial_score = score(proposed, baseline, held)
                if trial_score < current:
                    work, current = proposed, trial_score
                    log.append({"action": "photo", "detail": "reference image consulted",
                        "defects": current[0]})

    for oid, obj in (work.get("objects") or {}).items():
        was = (baseline.get("objects") or {}).get(oid)
        if not was:
            continue
        travelled = math.dist(obj["center"][:2], was["center"][:2])
        resized = [round(v, 3) for v in obj["size"]] != [round(v, 3) for v in was["size"]]
        if travelled >= 1e-4 or resized:
            moves.append(Move(oid, [was["center"][0], was["center"][1]],
                              [obj["center"][0], obj["center"][1]], travelled,
                              "trimmed to its measured depth" if resized
                              else "repaired within the room's own structure"))
    return work, moves + opening_moves, log


def solve_v11(shell: dict[str, Any], *, detail: bool = False):
    """Action-based repair with a strict monotone gate. Cannot return a worse room than it got."""
    out, moves, log = repair(shell)
    return (out, moves, log) if detail else out


def solve_v12(shell: dict[str, Any], *, detail: bool = False):
    """v11, with the reference photograph as the last resort rather than the first.

    The action set is deliberately exhausted before any model is asked. By the time ``on_stuck``
    fires, every cheap, checkable, deterministic repair has already been tried and rejected, so the
    question put to the model is narrow and the evidence for it — the crop with the box drawn on
    the photo — is the only thing left that geometry does not have. Its answer goes through the
    same gate as everything else: applied to a copy, kept only if the room improves.
    """
    from .agent import apply_proposals, propose
    from pathlib import Path

    root = Path((shell.get("meta") or {}).get("batch_dir") or ".")

    def ask(state, errors):
        if not root.is_dir() or not (root / "references").is_dir():
            return None
        proposals = [p for v in errors if (p := propose(state, v, root))]
        if not proposals:
            return None
        fixed, _log = apply_proposals(state, proposals)
        return fixed

    out, moves, log = repair(shell, on_stuck=ask)
    return (out, moves, log) if detail else out


def solve_v13(shell: dict[str, Any], *, detail: bool = False):
    """Deterministic first; where its answer is legal but implausible, UNDO it and ask.

    v11 clears Airbnb-Cam — by sliding a table half a metre across the room. Nothing in the
    violation count objects, and to a person the result is obviously wrong: that table did not move
    50 cm, and the wardrobe beside it plainly belongs in its corner rather than 25 cm out from it.
    Scoring cannot see this, because the room is genuinely collision-free either way.

    So a move that is large enough to be a claim rather than a correction is treated as a question,
    not an answer. Those objects are put BACK where the scan measured them — the model is asked
    about the real measurement, not about a guess it would have to unpick — and it answers with
    "attach it to these walls", "it is this size", or "leave it". Whatever it says goes through the
    same gate, and if nothing it proposes beats the deterministic result, that result stands. The
    ceiling is never lower than v11's; only the reasoning is better.
    """
    from pathlib import Path

    from .agent import apply_proposals, propose

    baseline = copy.deepcopy(shell)
    solved, moves, log = repair(shell)
    root = Path((shell.get("meta") or {}).get("batch_dir") or ".")

    walls = solved.get("walls") or {}
    suspects: dict[str, str] = {}
    for oid, obj in (solved.get("objects") or {}).items():
        was = (baseline.get("objects") or {}).get(oid)
        if not was:
            continue
        travelled = math.dist(obj["center"][:2], was["center"][:2])
        if travelled > REVIEW_SHIFT:
            suspects[oid] = f"sliding it {travelled * 100:.0f} cm from where the scan put it"
    for violation in (v for v in check(solved) if v.severity == "error"):
        suspects.setdefault(violation.object, "failing to resolve it at all")

    if not suspects or not root.is_dir() or not (root / "references").is_dir():
        return (solved, moves, log) if detail else solved

    # put the suspects back where the scan measured them, and ask about THAT
    reverted = copy.deepcopy(solved)
    for oid, _why in suspects.items():
        was = (baseline.get("objects") or {}).get(oid)
        if was:
            reverted["objects"][oid]["center"] = list(was["center"])
            reverted["objects"][oid]["size"] = list(was["size"])

    held = {oid: set(attachments(o, walls)) for oid, o in (baseline.get("objects") or {}).items()}
    held = {k: v for k, v in held.items() if v}
    proposals = []
    for violation in check(reverted):
        if violation.severity != "error" or violation.object not in suspects:
            continue
        proposal = propose(reverted, violation, root, suspicion=suspects[violation.object])
        if proposal:
            proposals.append(proposal)
    if not proposals:
        return (solved, moves, log) if detail else solved

    asked, gate = apply_proposals(reverted, proposals)
    # let the deterministic actions clean up whatever the model's edit left behind
    finished, more_moves, more_log = repair(asked)

    if score(finished, baseline, held) < score(solved, baseline, held):
        log = log + [{"action": "rollback+ask", "detail": ", ".join(
            f"{g.get('object_id')}:{g.get('action')}:{'kept' if g.get('accepted') else 'dropped'}"
            for g in gate)}] + more_log
        return (finished, more_moves, log) if detail else finished
    return (solved, moves, log) if detail else solved
