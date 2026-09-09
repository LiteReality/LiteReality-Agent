"""adjust.py — STAGE 3: make the layout physically consistent, deterministically.

A RoomPlan capture is a *measurement*, so its boxes overlap. Chairs interpenetrate, a table runs
into a wall, a shelf floats 4 cm off the floor. None of that can be simulated: a physics engine
handed two boxes that already overlap either explodes them apart on the first step or wedges them.
This stage produces a layout in which nothing overlaps that should not, so the room is
**simulation-ready** before any agent or solver is involved.

Two halves, in the same shape as ``pipeline/compile/quality_check`` upstream:

``check(shell)``
    read-only. Every violation the geometry admits — ``below_floor``, ``above_ceiling``,
    ``floating``, ``sunk``, ``outside_room``, ``wall_clash``, ``object_clash``,
    ``fixture_over_opening`` — measured, never guessed.

``resolve(shell)``
    repair. Three faults are CONTAINMENT rather than collision — a box buried in a wall, a box off
    the floor plate, a box through the ceiling — and the pair relaxation below cannot see any of
    them: a wall is not in ``objects``, the floor plate is not a box, and a lifted object clashes
    with nothing. They are put back first, each firing only on what ``check`` actually reports, so
    a correctly placed unit is never "corrected". A buried object goes back to touching the face of
    its own wall, which is both the shortest move that clears the slab and the one that keeps it
    installed against that wall — shoving it into the room would trade a collision for a counter
    that is no longer a counter.

    Then the iterative relaxation. Each clashing pair is separated along its **minimum translation
    vector** — by construction the shortest move that fixes it — and the move is split between the
    two by MOBILITY:

    * an object standing against a wall is ANCHORED and does not move. A counter run belongs
      against its wall; the thing in front of it is what should give way. Anchored-vs-anchored is
      reported, never forced.
    * ``FREE_STANDING`` categories (chairs, tables, desks…) stay mobile even against a wall —
      that is where they happen to be, not where they belong.
    * motion is XY only. Vertical placement is grounding's job, and a ``floating`` violation means
      a missing support object, not a misplaced one — so it is reported and grounded separately.

Every candidate move is then validated (displacement cap, still inside the floor polygon, not
driven into a wall) and reverted if it fails, so an object is never left somewhere worse than it
started. What survives is reported as ``unresolved``: honest output beats a silent bad fix.

The allow-list is shared with :mod:`physical_engine.graph`, so the resolver can never "fix"
something the graph considers correct — pushing an undermount sink out of its counter would be a
regression, not a repair.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .graph import (
    CLASH_TOL,
    FLOOR_STANDING,
    FURNITURE,
    OPEN_FRAME,
    PASSTHROUGH,
    WALL_MOUNTED,
    Z_BAND_TOL,
    expected_pair,
    is_wall_hung,
    obb_mtv,
    wall_distance,
)
from .shell import object_footprint, opening_span, wall_frame

__all__ = ["Violation", "Move", "AdjustResult", "check", "resolve", "ground", "reseat_openings"]

# tolerances (metres)
Z_TOL = 0.05           # poke through the floor or ceiling
FLOAT_TOL = 0.08       # gap under a floor-standing object
SUNK_TOL = 0.05
FIXTURE_PEN = 0.08     # wall-fixture overlap, measured in the wall's own plane
MAX_NUDGE = 0.30       # a bigger correction than this is a reconstruction error, not a nudge
ANCHOR_TOL = 0.12      # gap to a wall under which an object counts as installed against it
OUTSIDE_TOL = 0.10     # slack on the floor plate before a centre counts as off it
OPENING_OVERHANG_TOL = 0.10   # how far an opening may run past its wall before it is misplaced
WALL_SLAB_TOL = 0.02   # how far past a wall's own half-thickness a centre still counts as inside it
WALL_CROSS_TOL = 0.05  # how far a FOOTPRINT may cross a wall line before it is inside the wall
FLOOR_OUT_TOL = 0.05   # how far a footprint corner may sit off the floor plate
SEPARATION = 0.005     # extra clearance, so a resolved pair is separated and not merely touching
MAX_ITERS = 60

# Things people slide around: standing against a wall is where they happen to be, not where they
# belong. Everything else (counters, cabinets, baths, appliances) is INSTALLED — once against its
# wall, that position is load-bearing information from the scan.
FREE_STANDING = {"chair", "stool", "table", "desk", "sofa", "television"}


@dataclass
class Violation:
    """One finding. ``severity`` separates a DEFECT from an observation worth recording.

    ``info`` findings are things the geometry cannot distinguish from a fault but which are almost
    always correct — a wall cabinet is not touching the floor because it is bolted to a wall. They
    are reported, never "fixed", and never counted as failures. Dropping them silently would hide
    the one case that IS a fault (a cabinet floating nowhere near a wall); calling them errors
    buries the real findings, which is what upstream does: 21 of its 39 findings across these
    scans are wall-hung kitchen units.
    """

    object: str
    kind: str
    detail: str
    magnitude: float = 0.0
    other: str | None = None
    severity: str = "error"          # error | info

    def __str__(self) -> str:
        return f"{self.object:16} {self.kind:22} {self.detail}"


@dataclass
class Move:
    object: str
    before: list[float]
    after: list[float]
    distance: float
    reason: str

    def __str__(self) -> str:
        return (f"{self.object:16} {self.distance * 100:5.1f} cm  "
                f"({self.before[0]:.3f},{self.before[1]:.3f}) -> "
                f"({self.after[0]:.3f},{self.after[1]:.3f})   {self.reason}")


@dataclass
class AdjustResult:
    shell: dict[str, Any]
    moves: list[Move] = field(default_factory=list)
    before: list[Violation] = field(default_factory=list)
    after: list[Violation] = field(default_factory=list)
    unresolved: list[Violation] = field(default_factory=list)
    iterations: int = 0

    @staticmethod
    def _errors(items: list[Violation]) -> list[Violation]:
        return [v for v in items if v.severity == "error"]

    @property
    def errors_before(self) -> list[Violation]:
        return self._errors(self.before)

    @property
    def errors_after(self) -> list[Violation]:
        return self._errors(self.after)

    @property
    def notes(self) -> list[Violation]:
        return [v for v in self.after if v.severity == "info"]

    @property
    def resolved(self) -> int:
        return len(self.errors_before) - len(self.errors_after)

    def report(self) -> str:
        lines = [f"layout adjustment: {len(self.errors_before)} violations in, "
                 f"{len(self.errors_after)} out ({self.iterations} iterations, "
                 f"{len(self.moves)} objects moved)"]
        if self.errors_before:
            lines.append("  before:")
            lines += [f"    ✗ {v}" for v in self.errors_before]
        if self.moves:
            lines.append("  moves:")
            lines += [f"    → {m}" for m in self.moves]
        lines.append("  after:")
        lines += ([f"    ✗ {v}" for v in self.errors_after] if self.errors_after
                  else ["    ✓ clean — no geometric violations"])
        if self.notes:
            lines.append("  noted (correct, not defects):")
            lines += [f"    · {v}" for v in self.notes]
        return "\n".join(lines)


# ── shared geometry ──────────────────────────────────────────────────────────
def _obb(obj: dict[str, Any], position: tuple[float, float] | None = None):
    cx, cy = position if position else (obj["center"][0], obj["center"][1])
    return (cx, cy, obj["size"][0] / 2.0, obj["size"][1] / 2.0, math.radians(obj.get("yaw", 0.0)))


def _support(obj: dict[str, Any], direction) -> float:
    hw, hd = obj["size"][0] / 2.0, obj["size"][1] / 2.0
    yaw = math.radians(obj.get("yaw", 0.0))
    c, s = math.cos(yaw), math.sin(yaw)
    return (abs(hw * (c * direction[0] + s * direction[1]))
            + abs(hd * (-s * direction[0] + c * direction[1])))


def _floor_bounds(shell: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    verts = np.asarray((shell.get("floor") or {}).get("verts") or [], dtype=float)
    if len(verts) == 0:
        walls = shell.get("walls") or {}
        pts = [p for w in walls.values() for p in (w["start"], w["end"])]
        if not pts:
            return None
        verts = np.asarray(pts, dtype=float)
    return verts[:, 0:2].min(axis=0), verts[:, 0:2].max(axis=0)


def _wall_slab(wall: dict[str, Any]) -> float:
    """Half-width of the band a centre must be outside of to be clear of this wall.

    RoomPlan reports walls as planes — thicknesses of 0.1 mm are common — so the wall's own
    geometry is too thin to test against, and :data:`WALL_SLAB_TOL` is what actually decides.
    ``_in_wall`` and ``_wall_escape`` both measure against this, so the band a repair aims to clear
    is by construction the band the check complains about.
    """
    return wall.get("thickness", 0.1) / 2.0 + WALL_SLAB_TOL


def _in_wall(obj: dict[str, Any], walls: dict[str, Any], position=None) -> str | None:
    """Is this object's CENTRE inside a wall slab? Furniture merely resting against a wall has its
    centre about half its depth away, well outside a 1 mm RoomPlan slab, so this does not fire on it."""
    cx, cy = position if position else (obj["center"][0], obj["center"][1])
    centre = np.array([cx, cy])
    for wall_id, wall in (walls or {}).items():
        start, along, normal, length = wall_frame(wall)
        rel = centre - start
        t = float(np.dot(rel, along))
        if 0.0 <= t <= length and abs(float(np.dot(rel, normal))) < _wall_slab(wall):
            return wall_id
    return None


def _floor_triangles(shell: dict[str, Any]) -> list[tuple] | None:
    """The floor plate as XY triangles, or None when the scan gave no usable mesh.

    ``floor`` is a TRIANGLE MESH, not a boundary ring: MIL-Meeting's is 30 vertices and 16 faces,
    with every vertex duplicated, and consecutive entries a median of 2.8 m apart. Ray-casting that
    vertex list as if it were an ordered polygon returns an arbitrary answer — it only looks right
    on the synthetic rooms, whose plate happens to be a 4-vertex quad. Going through ``faces`` is
    the only reading that works on both, and it handles an L-shaped or multi-room plate for free,
    since "inside" is just "inside any triangle".
    """
    floor = shell.get("floor") or {}
    verts = np.asarray(floor.get("verts") or [], dtype=float)
    faces = floor.get("faces")
    if len(verts) < 3 or not faces:
        return None
    return [(verts[a][:2], verts[b][:2], verts[c][:2]) for a, b, c in faces
            if max(a, b, c) < len(verts)]


def _inside_floor(point, triangles: list[tuple]) -> bool:
    """Is this XY point on the floor plate? Inside any one of its triangles."""
    def side(p, q, r):
        return (p[0] - r[0]) * (q[1] - r[1]) - (q[0] - r[0]) * (p[1] - r[1])

    for a, b, c in triangles:
        d1, d2, d3 = side(point, a, b), side(point, b, c), side(point, c, a)
        if not (((d1 < 0) or (d2 < 0) or (d3 < 0)) and ((d1 > 0) or (d2 > 0) or (d3 > 0))):
            return True
    return False


def _outside(position, bounds) -> bool:
    """Is this centre off the floor plate? The single definition — ``check`` and the move
    validator both call it, so the tolerance cannot drift between detecting and repairing."""
    low, high = bounds
    return not (low[0] - OUTSIDE_TOL <= position[0] <= high[0] + OUTSIDE_TOL
                and low[1] - OUTSIDE_TOL <= position[1] <= high[1] + OUTSIDE_TOL)


def _interior_normal(wall: dict[str, Any], polygon, centroid) -> np.ndarray:
    """Which way out of this wall is INTO the room.

    A wall's normal has no inherent sign, and pushing a buried object out of the wrong face puts it
    in the corridor next door. The floor plate settles it: probe just off each face and keep the
    side that is inside it. An interior partition has room on both sides and no probe can decide —
    there the room centroid is the tie-break, which is the best available guess and is why this
    returns a direction rather than a certainty.
    """
    start, along, normal, length = wall_frame(wall)
    mid = start + along * (length / 2.0)
    if polygon is not None:
        probe = wall.get("thickness", 0.1) / 2.0 + 0.10
        plus, minus = (_inside_floor(mid + normal * probe, polygon),
                       _inside_floor(mid - normal * probe, polygon))
        if plus != minus:
            return normal if plus else -normal
    if centroid is not None:
        return normal if float(np.dot(np.asarray(centroid, float) - mid, normal)) >= 0 else -normal
    return normal


def _wall_escape(obj: dict[str, Any], wall: dict[str, Any], inward: np.ndarray,
                 position) -> tuple[float, float]:
    """Where a box buried in a wall belongs: its back face on that wall's own line.

    Not "just far enough that ``check`` stops complaining" — clearing the centre out of a 4 cm slab
    takes 4 cm and leaves the object 90% inside the wall, which passes the check and fails the
    room. The target is where a unit standing at a wall actually sits, which is measured rather
    than assumed: across 324 synthetic and 139 captured wall-standing objects the median distance
    from centre to wall line is exactly the box's own support in that direction — back face on the
    line, not on the face of the slab. RoomPlan reports walls as planes and reconstructs the unit
    up to that plane, so adding half a thickness on top of it overshoots every repair by ~2 cm and
    pushes borderline ones past the nudge cap for no gain.
    """
    start, along, normal, length = wall_frame(wall)
    centre = np.asarray(position, dtype=float)
    # Far enough that the box is out of the wall, AND far enough that the check agrees it is. The
    # second is not implied by the first: a 3 cm wardrobe panel on a 0.1 mm wall is clear of the
    # slab 1.6 cm out and still inside the tolerance band, so aiming at the geometry alone produces
    # a move that is immediately reverted for landing back in the wall it just left.
    clear = max(_support(obj, inward), _wall_slab(wall) + SEPARATION)
    perp = float(np.dot(centre - start, inward))
    if perp >= clear:
        return float(centre[0]), float(centre[1])
    target = centre + inward * (clear - perp)
    return float(target[0]), float(target[1])


def _wall_crossing(obj: dict[str, Any], wall: dict[str, Any], position=None) -> float:
    """How far this object's FOOTPRINT crosses the wall's line, within that wall's own segment.

    The test that matters, and the one the centre cannot express. RoomPlan reports walls as planes
    — thicknesses of 0.1 mm are normal — so "is the centre inside the slab" asks whether the centre
    is within about 4 cm of the line, and a 1.4 m table can cross a wall by half a metre with its
    centre comfortably in the room. Across the 21 captures that is 18 objects through walls, of
    which the centre test finds none.

    Two conditions, both necessary. The footprint must have corners on BOTH sides of the line, so
    an object standing in the next room is not reported as being a wall's depth "inside" it; and
    those corners must project within the wall's own extent, so a wall does not act as an infinite
    half-space across the whole floor plan. The depth returned is the shallower side — by
    construction the shortest move that would take the box off the line.
    """
    start, along, normal, length = wall_frame(wall)
    corners = object_footprint(obj)
    if position is not None:
        corners = corners + (np.asarray(position, dtype=float)
                             - np.asarray(obj["center"][:2], dtype=float))
    along_t = [float(np.dot(c - start, along)) for c in corners]
    if max(along_t) < -0.05 or min(along_t) > length + 0.05:
        return 0.0
    perp = [float(np.dot(c - start, normal)) for c in corners]
    if min(perp) >= 0.0 or max(perp) <= 0.0:
        return 0.0                       # entirely on one side: not a crossing
    return min(-min(perp), max(perp))


def _worst_wall_crossing(obj, walls, position=None) -> tuple[str, float] | None:
    worst: tuple[str, float] | None = None
    for wall_id, wall in (walls or {}).items():
        depth = _wall_crossing(obj, wall, position)
        if depth > WALL_CROSS_TOL and (worst is None or depth > worst[1]):
            worst = (wall_id, depth)
    return worst


def _off_floor(obj: dict[str, Any], triangles, position=None) -> float:
    """How far the footprint's worst corner lies off the floor plate. 0 when it is all on it."""
    if not triangles:
        return 0.0
    corners = object_footprint(obj)
    if position is not None:
        corners = corners + (np.asarray(position, dtype=float)
                             - np.asarray(obj["center"][:2], dtype=float))
    outside = [c for c in corners if not _inside_floor(c, triangles)]
    if not outside:
        return 0.0

    def edge_distance(point, a, b) -> float:
        ab = b - a
        length_sq = float(np.dot(ab, ab)) or 1e-12
        t = max(0.0, min(1.0, float(np.dot(point - a, ab)) / length_sq))
        return float(np.linalg.norm(point - (a + t * ab)))

    return max(min(min(edge_distance(c, t[i], t[(i + 1) % 3]) for i in range(3))
                   for t in triangles) for c in outside)


def _fixture_escape(obj: dict[str, Any], wall: dict[str, Any], opening: dict[str, Any],
                    position) -> tuple[float, float] | None:
    """Slide a wall fixture along its wall until it is off an opening. None if it cannot be.

    Along the wall, never off it: a television reads as mounted over a window because one of the
    two is a few centimetres out, and the fix is to put the fixture beside the window on the wall
    it is actually on. Moving it away from the wall would unmount it, and moving it up or down is
    the one degree of freedom a scan cannot justify. Whichever side is nearer wins; if neither side
    has room left on the wall, this returns None and the violation stands.
    """
    start, along, _normal, length = wall_frame(wall)
    centre = np.asarray(position, dtype=float)
    t = float(np.dot(centre - start, along))
    half = _support(obj, along)
    o0, o1 = opening_span(opening)
    options = [target for target in (o0 - half - SEPARATION, o1 + half + SEPARATION)
               if half <= target <= length - half]
    if not options:
        return None
    moved = centre + along * (min(options, key=lambda target: abs(target - t)) - t)
    return float(moved[0]), float(moved[1])


def _return_inside(obj: dict[str, Any], bounds, position) -> tuple[float, float]:
    """Slide a box that is off the floor plate back onto it — footprint and all, not just centre."""
    low, high = bounds
    out = [float(position[0]), float(position[1])]
    for axis, direction in ((0, np.array([1.0, 0.0])), (1, np.array([0.0, 1.0]))):
        support = _support(obj, direction)
        lo, hi = low[axis] + support, high[axis] - support
        if lo > hi:                       # the plate is narrower than the object — centre it
            lo = hi = (low[axis] + high[axis]) / 2.0
        out[axis] = min(max(out[axis], lo), hi)
    return out[0], out[1]


def _anchored(obj: dict[str, Any], walls: dict[str, Any]) -> bool:
    """Installed against a wall → it stays put and the other side of the pair moves."""
    if obj.get("category", "") in FREE_STANDING:
        return False
    for wall in (walls or {}).values():
        measured = wall_distance(obj, wall)
        if measured is not None and measured[0] < ANCHOR_TOL:
            return True
    return False


# ── DETECT ───────────────────────────────────────────────────────────────────
def check(shell: dict[str, Any]) -> list[Violation]:
    """Every geometric violation in a SHELL. Pure arithmetic — no Blender, no compiled glb."""
    walls = shell.get("walls") or {}
    objects = shell.get("objects") or {}
    openings = shell.get("openings") or {}
    floor_z = float(shell.get("floor_z", 0.0))
    ceiling_z = float(shell.get("ceiling_z", floor_z + 2.5))
    bounds = _floor_bounds(shell)
    triangles = _floor_triangles(shell)

    out: list[Violation] = []
    for object_id, obj in objects.items():
        category = obj.get("category", "")
        bottom = obj["center"][2] - obj["size"][2] / 2.0
        top = obj["center"][2] + obj["size"][2] / 2.0
        if bottom < floor_z - Z_TOL:
            out.append(Violation(object_id, "below_floor",
                                 f"bottom {bottom:.2f} < floor {floor_z:.2f}", floor_z - bottom))
        if top > ceiling_z + Z_TOL:
            out.append(Violation(object_id, "above_ceiling",
                                 f"top {top:.2f} > ceiling {ceiling_z:.2f}", top - ceiling_z))
        if category in FLOOR_STANDING:
            gap = bottom - floor_z
            hung = is_wall_hung(obj, walls, floor_z)
            if gap > FLOAT_TOL and hung:
                out.append(Violation(object_id, "wall_hung",
                                     f"{gap:.2f} m above the floor, mounted on {hung}",
                                     gap, hung, severity="info"))
            elif gap > FLOAT_TOL:
                out.append(Violation(object_id, "floating", f"{gap:.2f} m above the floor", gap))
            elif gap < -SUNK_TOL:
                out.append(Violation(object_id, "sunk", f"{-gap:.2f} m into the floor", -gap))
        if triangles:
            off = _off_floor(obj, triangles)
            if off > FLOOR_OUT_TOL:
                out.append(Violation(object_id, "outside_room",
                                     f"footprint {off:.2f} m off the floor plate", off))
        elif bounds is not None and _outside(obj["center"][:2], bounds):
            cx, cy = obj["center"][0], obj["center"][1]
            out.append(Violation(object_id, "outside_room",
                                 f"centre ({cx:.2f},{cy:.2f}) outside the footprint"))
        if category in FURNITURE:
            crossed = _worst_wall_crossing(obj, walls)
            if crossed:
                out.append(Violation(object_id, "wall_clash",
                                     f"footprint {crossed[1]:.2f} m into {crossed[0]}",
                                     crossed[1], crossed[0]))
            elif _in_wall(obj, walls):
                hit = _in_wall(obj, walls)
                out.append(Violation(object_id, "wall_clash", f"centre inside {hit}", other=hit))

    ids = list(objects)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a_id, b_id = ids[i], ids[j]
            a, b = objects[a_id], objects[b_id]
            cat_a, cat_b = a.get("category", ""), b.get("category", "")
            if expected_pair(cat_a, cat_b):
                continue
            if {cat_a, cat_b} & (PASSTHROUGH | OPEN_FRAME):
                continue
            a_bottom, a_top = a["center"][2] - a["size"][2] / 2, a["center"][2] + a["size"][2] / 2
            b_bottom, b_top = b["center"][2] - b["size"][2] / 2, b["center"][2] + b["size"][2] / 2
            if min(a_top, b_top) - max(a_bottom, b_bottom) <= Z_BAND_TOL:
                continue
            mtv = obb_mtv(_obb(a), _obb(b))
            if mtv and mtv[0] > CLASH_TOL:
                out.append(Violation(a_id, "object_clash",
                                     f"footprint into {b_id} ~{mtv[0]:.2f} m", mtv[0], b_id))

    # an opening that runs off the end of the wall that hosts it. RoomPlan puts a door in a wall's
    # folder even when the geometry straddles a corner, so `offset + width` can exceed the wall
    # length — MIL-Meeting's Door1 overhangs Wall1 by 0.55 m. Nothing downstream notices, and the
    # opening is then cut in the wrong place (or off the end, i.e. not at all).
    for opening_id, opening in openings.items():
        wall = walls.get(opening.get("wall") or "")
        if not wall:
            out.append(Violation(opening_id, "orphan_opening", "not hosted by any wall"))
            continue
        length = wall_frame(wall)[3]
        t0, t1 = opening_span(opening)
        overhang = max(-t0, t1 - length)
        if overhang > OPENING_OVERHANG_TOL:
            out.append(Violation(opening_id, "opening_off_wall",
                                 f"runs {overhang:.2f} m past {opening['wall']} "
                                 f"(len {length:.2f} m)", overhang, opening.get("wall")))

    # a wall fixture sitting over a door or window, measured in the wall's own plane
    for opening_id, opening in openings.items():
        wall = walls.get(opening.get("wall") or "")
        if not wall:
            continue
        start, along, normal, length = wall_frame(wall)
        o0, o1 = opening_span(opening)
        sill = floor_z + opening.get("sill", 0.0)
        z0, z1 = sill, sill + opening["height"]
        for object_id, obj in objects.items():
            if obj.get("category", "") not in WALL_MOUNTED:
                continue
            measured = wall_distance(obj, wall)
            if measured is None or measured[0] > 0.75:
                continue
            t = measured[1]
            half = _support(obj, along)
            bottom = obj["center"][2] - obj["size"][2] / 2.0
            top = obj["center"][2] + obj["size"][2] / 2.0
            overlap_t = min(t + half, o1) - max(t - half, o0)
            overlap_z = min(top, z1) - max(bottom, z0)
            if overlap_t > FIXTURE_PEN and overlap_z > FIXTURE_PEN:
                out.append(Violation(object_id, "fixture_over_opening",
                                     f"sits over {opening_id} on {opening['wall']} "
                                     f"~{min(overlap_t, overlap_z):.2f} m",
                                     min(overlap_t, overlap_z), opening_id))
    return out


# ── REPAIR ───────────────────────────────────────────────────────────────────
def ground(shell: dict[str, Any]) -> tuple[dict[str, Any], list[Move]]:
    """Vertical placement: drop what is floating onto the floor, pull what pokes through the
    ceiling back under it. Z only, so it commutes with the XY solver.

    The two passes are ordered, not alternatives. An object lifted through the ceiling is usually
    *also* floating, and dropping it to the floor is the repair that puts it back where it belongs;
    the ceiling clamp is what is left for the ones grounding cannot claim — a wall-hung cabinet has
    no floor contact to restore, so the most that can be said about it is that its top is not
    outside the room.
    """
    shell = copy.deepcopy(shell)
    floor_z = float(shell.get("floor_z", 0.0))
    ceiling_z = float(shell.get("ceiling_z", floor_z + 2.5))
    moves: list[Move] = []
    for object_id, obj in (shell.get("objects") or {}).items():
        if obj.get("category", "") not in FLOOR_STANDING:
            continue
        if is_wall_hung(obj, shell.get("walls") or {}, floor_z):
            continue                       # bolted to a wall — dropping it to the floor is wrong
        bottom = obj["center"][2] - obj["size"][2] / 2.0
        delta = floor_z - bottom
        if abs(delta) <= max(FLOAT_TOL, SUNK_TOL):
            continue
        if abs(delta) > MAX_NUDGE * 2:
            continue                       # a metre off the floor is a missing support, not a slip
        before = list(obj["center"])
        obj["center"][2] = round(obj["center"][2] + delta, 4)
        moves.append(Move(object_id, before, list(obj["center"]), abs(delta), "grounded to floor_z"))

    settled = {move.object for move in moves}
    for object_id, obj in (shell.get("objects") or {}).items():
        if object_id in settled:
            continue
        height = obj["size"][2]
        overshoot = obj["center"][2] + height / 2.0 - ceiling_z
        if overshoot <= Z_TOL:
            continue
        if height > ceiling_z - floor_z:
            continue                       # taller than the room: a size error, not a placement one
        # With the box no taller than the room, clearing the ceiling can never drive it through the
        # floor: bottom - drop = floor_z + (ceiling_z - floor_z - height) - 0.001 >= floor_z.
        drop = overshoot + 0.001
        if drop > MAX_NUDGE * 2:
            continue                       # a metre through the ceiling is not a slip either
        before = list(obj["center"])
        obj["center"][2] = round(obj["center"][2] - drop, 4)
        moves.append(Move(object_id, before, list(obj["center"]), drop, "lowered under ceiling_z"))
    return shell, moves


def reseat_openings(shell: dict[str, Any]) -> tuple[dict[str, Any], list[Move]]:
    """Slide an opening that runs off the end of its wall back onto it.

    RoomPlan files a door under one wall while placing its geometry across the corner into the
    next, so ``offset + width / 2`` can exceed the wall it is hosted by — MIL-Meeting's Door1
    overhangs Wall1 by 0.55 m in the real capture. An opening that is not on its wall is not cut
    where the door is, so the compiled room has a doorway in a solid wall and a hole in the wrong
    one. Clamping the span into the wall is the minimal repair and the one that needs no guess.
    The deeper fix — re-hosting the opening to the wall its geometry actually crosses — needs the
    neighbouring wall's frame to agree, so it is left to whoever has that evidence.
    """
    shell = copy.deepcopy(shell)
    walls = shell.get("walls") or {}
    moves: list[Move] = []
    for opening_id, opening in (shell.get("openings") or {}).items():
        wall = walls.get(opening.get("wall") or "")
        if not wall:
            continue                       # orphaned: no wall to slide along. Reported, not guessed.
        start, along, _normal, length = wall_frame(wall)
        t0, t1 = opening_span(opening)
        if max(-t0, t1 - length) <= OPENING_OVERHANG_TOL:
            continue
        half = opening["width"] / 2.0
        if opening["width"] > length:
            continue                       # wider than the wall it is on — not a placement fault
        offset = min(max(opening["offset"], half), length - half)
        before, after = start + along * opening["offset"], start + along * offset
        distance = abs(offset - opening["offset"])
        opening["offset"] = round(float(offset), 4)
        moves.append(Move(opening_id, [float(before[0]), float(before[1])],
                          [float(after[0]), float(after[1])], distance,
                          f"slid back onto {opening['wall']}"))
    return shell, moves


def _clashing_pairs(objects: dict[str, Any], positions: dict[str, tuple[float, float]]):
    """Every genuine interpenetration at the given positions → (a, b, depth, axis)."""
    out = []
    ids = list(objects)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a_id, b_id = ids[i], ids[j]
            a, b = objects[a_id], objects[b_id]
            cat_a, cat_b = a.get("category", ""), b.get("category", "")
            if expected_pair(cat_a, cat_b) or ({cat_a, cat_b} & (PASSTHROUGH | OPEN_FRAME)):
                continue
            a_bottom, a_top = a["center"][2] - a["size"][2] / 2, a["center"][2] + a["size"][2] / 2
            b_bottom, b_top = b["center"][2] - b["size"][2] / 2, b["center"][2] + b["size"][2] / 2
            if min(a_top, b_top) - max(a_bottom, b_bottom) <= Z_BAND_TOL:
                continue
            mtv = obb_mtv(_obb(a, positions[a_id]), _obb(b, positions[b_id]))
            if mtv and mtv[0] > CLASH_TOL:
                out.append((a_id, b_id, mtv[0], mtv[1]))
    return out


def resolve(shell: dict[str, Any], *, max_nudge: float = MAX_NUDGE,
            max_iters: int = MAX_ITERS, do_ground: bool = True,
            do_openings: bool = True) -> AdjustResult:
    """Put the room back inside itself, then nudge what interpenetrates apart."""
    original = copy.deepcopy(shell)
    before = check(original)

    working = copy.deepcopy(shell)
    moves: list[Move] = []
    if do_ground:
        working, moves = ground(working)
    if do_openings:
        working, opening_moves = reseat_openings(working)
        moves += opening_moves

    objects = working.get("objects") or {}
    walls = working.get("walls") or {}
    start = {oid: (o["center"][0], o["center"][1]) for oid, o in objects.items()}
    positions = dict(start)
    anchored = {oid: _anchored(o, walls) for oid, o in objects.items()}
    reasons: dict[str, str] = {}

    # --- containment, before anything is relaxed ----------------------------
    # A wall is not in `objects` and the floor plate is not a box, so neither of these faults is
    # visible to the pair loop below; and an object buried in a wall has no clash worth separating
    # until it is out of it. Both fire only where `check` reports, so nothing undamaged is touched,
    # and both run first so that whatever the correction pushes them into is resolved as a clash.
    polygon = _floor_triangles(working)
    bounds = _floor_bounds(working)
    centroid = (bounds[0] + bounds[1]) / 2.0 if bounds is not None else None
    openings = working.get("openings") or {}
    # Driven off `check` itself rather than re-derived, so a repair cannot fire anywhere the
    # detector is silent — that is what keeps a correctly placed unit from being "corrected" — and
    # so the two cannot drift apart later. Each takes the running position, so an object with two
    # findings against it is repaired for both.
    for violation in check(working):
        obj = objects.get(violation.object)
        if obj is None:
            continue
        if violation.kind == "wall_clash" and violation.other in walls:
            wall = walls[violation.other]
            positions[violation.object] = _wall_escape(
                obj, wall, _interior_normal(wall, polygon, centroid), positions[violation.object])
            reasons[violation.object] = f"pushed out of {violation.other}, back against its face"
        elif violation.kind == "outside_room" and bounds is not None:
            positions[violation.object] = _return_inside(obj, bounds, positions[violation.object])
            reasons[violation.object] = "returned onto the floor plate"
        elif violation.kind == "fixture_over_opening":
            opening = openings.get(violation.other or "")
            wall = walls.get((opening or {}).get("wall") or "") if opening else None
            slid = _fixture_escape(obj, wall, opening, positions[violation.object]) if wall else None
            if slid is not None:
                positions[violation.object] = slid
                reasons[violation.object] = f"slid clear of {violation.other}"

    iterations = 0
    for iterations in range(1, max_iters + 1):
        pairs = _clashing_pairs(objects, positions)
        if not pairs:
            break
        for a_id, b_id, depth, axis in pairs:
            push = depth + SEPARATION
            a_fixed, b_fixed = anchored[a_id], anchored[b_id]
            if a_fixed and b_fixed:
                continue                          # both installed — report it, never force it
            share_a, share_b = (0.0, 1.0) if a_fixed else (1.0, 0.0) if b_fixed else (0.5, 0.5)
            ax, ay = positions[a_id]
            bx, by = positions[b_id]
            positions[a_id] = (ax - axis[0] * push * share_a, ay - axis[1] * push * share_a)
            positions[b_id] = (bx + axis[0] * push * share_b, by + axis[1] * push * share_b)
    else:
        pairs = _clashing_pairs(objects, positions)

    # --- validate every move, revert the ones that made things worse ---------
    for object_id, obj in objects.items():
        new = positions[object_id]
        old = start[object_id]
        distance = math.dist(new, old)
        if distance < 1e-4:
            positions[object_id] = old
            continue
        reason = None
        if distance > max_nudge:
            reason = f"needs {distance * 100:.0f} cm, over the {max_nudge * 100:.0f} cm cap"
        elif _in_wall(obj, walls, new):
            reason = f"would push the centre into {_in_wall(obj, walls, new)}"
        elif bounds is not None and _outside(new, bounds):
            reason = "would leave the floor footprint"
        if reason:
            positions[object_id] = old
            obj["_reverted"] = reason
        else:
            obj["center"][0], obj["center"][1] = round(new[0], 4), round(new[1], 4)
            moves.append(Move(object_id, [old[0], old[1]], [new[0], new[1]], distance,
                              reasons.get(object_id,
                                          "separated along the minimum translation vector")))

    unresolved: list[Violation] = []
    for object_id, obj in objects.items():
        note = obj.pop("_reverted", None)
        if note:
            unresolved.append(Violation(object_id, "unresolved", note))
    for a_id, b_id, depth, _axis in _clashing_pairs(objects, positions):
        if anchored[a_id] and anchored[b_id]:
            unresolved.append(Violation(a_id, "unresolved",
                                        f"both {a_id} and {b_id} are installed against a wall "
                                        f"(~{depth:.2f} m overlap)", depth, b_id))

    return AdjustResult(shell=working, moves=moves, before=before, after=check(working),
                        unresolved=unresolved, iterations=iterations)
