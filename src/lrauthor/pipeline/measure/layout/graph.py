"""graph.py — SHELL → a scene graph, by rule. No model, no learning, no randomness.

STAGE 2. The SHELL is a flat bag of boxes. The graph states how they are *related*, and two kinds
of relation are what a simulator actually needs:

**1. Support — what holds what up.** A simulator has to know that a laptop rests on a desk that
rests on the floor, and that a wall cabinet is bolted to a wall rather than floating. So support is
a **tree, not a set of hints**: every object gets exactly ONE parent, resolved in priority order,
and the parent is always something that can bear it — a wall, the floor, the ceiling, or a lower
object::

    Room0
     ├── Floor0 ◀── Table0 ◀── Laptop0          rests on / stacked on
     │          ◀── Chair0
     ├── Wall2  ◀── Storage5                    hung on
     └── Ceiling0 ◀── Vent0

    Every object resolves to exactly one parent, or is reported `unsupported` — which is a defect,
    not a silence. `graph.support_roots()` and `graph.unsupported()` expose both.

**2. Grouping — what belongs to what.** The chairs around a table are not merely near it; they are
part of that table's setting. That is not support (a chair rests on the floor, not on the table)
and not collision (a tucked chair overlaps the table on purpose), so it needs its own relation:
``belongs_to``. It is what lets a whole dining set be moved, reasoned about, or kept together.

The remaining relations are structural or diagnostic:

===============  =========================================================================
``contains``     the room holds this element (structural, always true)
``hosts``        an opening is cut into this wall (from the archive path — see roomplan.py)
``corner``       two walls share an endpoint within :data:`CORNER_TOL`
``supported_by`` THE SUPPORT TREE. ``kind`` says how: floor / object / wall / ceiling
``belongs_to``   a chair is part of its table's setting
``against``      a floor-standing object's box reaches a wall — contact, but NOT support
``tucked_under`` an expected containment (chair under table, sink in counter) — NOT a collision
``clashes``      a genuine interpenetration; this is the list stage 3 has to resolve
===============  =========================================================================

Support is decided by geometry, not by category: a laptop is supported by the desk because its
underside meets the desk's top surface. Only the *exceptions* are category-driven — which
categories can hang on a wall, which pairs may legitimately interpenetrate, which children belong
to which parents — and those lists are carried over from ``pipeline/compile/quality_check`` so a graph built here
and a room checked upstream agree about what counts as wrong.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .shell import facing_yaw, object_footprint, opening_span, wall_frame

__all__ = [
    "Node", "Edge", "SceneGraph", "build_graph", "is_wall_hung",
    "FURNITURE", "FLOOR_STANDING", "WALL_MOUNTED", "WALL_HUNG_CAPABLE", "WALL_HUNG_MIN_Z",
    "PASSTHROUGH", "OPEN_FRAME", "EXPECTED_CONTAINMENT", "GROUP_PARENTS", "expected_pair",
    "category_tokens", "in_category",
]

# ── category knowledge (carried over from pipeline/compile/quality_check so both agree) ────
FURNITURE = {"table", "chair", "sofa", "storage", "television", "bed", "desk", "cabinet",
             "refrigerator", "stove", "oven", "sink", "toilet", "bathtub", "washer", "dishwasher",
             "fireplace", "stairs"}
FLOOR_STANDING = {"table", "chair", "sofa", "storage", "desk", "cabinet", "bed",
                  "refrigerator", "stove", "sink", "toilet", "bathtub", "washer", "dishwasher"}
WALL_MOUNTED = {"television", "whiteboard", "noticeboard", "board", "socket", "switch", "radiator",
                "shelf", "panel", "sign", "poster", "sensor", "dispenser", "access_panel", "ac",
                "heater", "thermostat", "vent", "mirror", "clock", "hook"}
# Categories that are routinely WALL-HUNG. A kitchen wall cabinet, a wall oven, a mounted TV and a
# bracket shelf all sit a metre or more off the floor by design, so "not touching the floor" is
# correct for them, not a defect. RoomPlan records no support information at all, which is why this
# has to be stated as knowledge rather than measured.
WALL_HUNG_CAPABLE = {"storage", "cabinet", "shelf", "shelving", "bookcase", "sink", "stove", "oven",
                     "hob", "cooktop", "television", "microwave", "radiator", "heater", "mirror",
                     "whiteboard", "noticeboard", "board", "panel", "dispenser", "ac"}
WALL_HUNG_MIN_Z = 0.30   # m above the floor before "off the floor" is deliberate rather than noise

# Long runs and trims that legitimately pass behind or through everything else.
PASSTHROUGH = {"trunking", "conduit", "skirting", "ceiling_grid", "cornice", "rail"}
# Open structures whose bounding box is mostly void — holding things inside it is the point.
OPEN_FRAME = {"shelf", "shelving", "rack", "bookcase", "bookshelf"}
# Interpenetration that is CORRECT: an undermount sink in its counter, a chair tucked under a table.
EXPECTED_CONTAINMENT = {
    ("chair", "table"), ("chair", "desk"), ("stool", "table"), ("stool", "desk"),
    ("sink", "storage"), ("sink", "cabinet"), ("sink", "table"),
    ("oven", "storage"), ("oven", "cabinet"), ("stove", "storage"), ("stove", "cabinet"),
    ("dishwasher", "storage"), ("dishwasher", "cabinet"),
    ("television", "storage"), ("television", "cabinet"),
}

# ── rule tolerances, in metres. Every edge below cites one of these. ─────────
CORNER_TOL = 0.20      # two wall endpoints this close are the same corner
SUPPORT_TOL = 0.08     # underside-to-surface gap that still counts as resting on it
SUPPORT_OVERLAP = 0.25 # fraction of the smaller footprint that must overlap to call it support
AGAINST_TOL = 0.15     # gap from an object's box to a wall plane that counts as "against" it
FIXTURE_PERP = 0.75    # how far off a wall a fixture may sit and still be mounted to it
CLASH_TOL = 0.001      # zero, to within float noise: two boxes that overlap at all, collide
# 0.08 was too generous to be useful. On the 21 captures it hides 12 of the 13 genuine
# object-object overlaps -- a storage unit 6.4 cm into the table beside it reads as clean. The
# floor it has to clear is the drift on a box that is correctly placed, and on the synthetic
# control (severe noise, no injected defect) that tops out at 6.8 cm only because the generator
# leaves a 6 cm clearance and then jitters both boxes 4 cm; a pair whose footprints genuinely
# overlap by 3 cm is touching, and a physics engine will treat it as contact either way.
Z_BAND_TOL = 0.05      # boxes must share at least this much height to be able to collide
CEILING_TOL = 0.15     # top this close to the ceiling counts as ceiling-mounted
GROUP_GAP = 0.80       # m — how far a chair may sit from its table and still belong to it
GROUP_FACING_BONUS = 60.0   # deg — a chair facing its table wins over a marginally closer one

# Which child categories belong to which parents. Grouping is a FUNCTIONAL relation, so unlike
# support it cannot be derived from geometry alone — proximity says a chair is near a table, not
# that it is part of that table's setting.
GROUP_PARENTS: dict[str, set[str]] = {
    "chair": {"table", "desk"},
    "stool": {"table", "desk", "counter"},
}


def is_wall_hung(obj: dict[str, Any], walls: dict[str, Any], floor_z: float) -> str | None:
    """Is this object mounted on a wall rather than standing on the floor? → the wall id, or None.

    Three conditions, all necessary: a category that CAN hang (:data:`WALL_HUNG_CAPABLE`), a
    bottom clearly off the floor (:data:`WALL_HUNG_MIN_Z`), and a wall actually within reach. A
    cabinet floating in the middle of a room satisfies the first two and is still a defect.
    """
    if obj.get("category", "") not in WALL_HUNG_CAPABLE:
        return None
    if obj["center"][2] - obj["size"][2] / 2 - floor_z < WALL_HUNG_MIN_Z:
        return None
    best: tuple[float, str] | None = None
    for wall_id, wall in (walls or {}).items():
        measured = wall_distance(obj, wall)
        if measured is None:
            continue
        if measured[0] <= FIXTURE_PERP and (best is None or measured[0] < best[0]):
            best = (measured[0], wall_id)
    return best[1] if best else None


def category_tokens(category: str) -> set[str]:
    """The categories a name stands for. ``oven_storage_stove`` is an oven AND a storage AND a stove.

    The box merge fuses a counter run into one object and names it after its members, so a scan
    that has been through it carries categories no table in this package was written against.
    Nothing errors — the lookups simply stop matching, silently, and every category rule the merged
    unit should have been subject to switches off for it.
    """
    return {token for token in (category or "").split("_") if token}


def in_category(category: str, table: Iterable[str]) -> bool:
    """Does this category — compound names included — belong to ``table``?"""
    table = set(table)
    return category in table or bool(category_tokens(category) & table)


def expected_pair(a: str, b: str, table: Iterable[tuple[str, str]] = EXPECTED_CONTAINMENT) -> bool:
    """Is this category pair an overlap we EXPECT? Order-independent.

    A merged run is expanded to its members, which is the whole point: Kitchen's ``Sink1`` sits
    inside ``Sink_Storage0``, and that is ``("sink", "storage")`` — already in the table, already
    meant to be exempt, and missed only because the merge renamed one side. Read literally it was
    reported as a collision, and the repair then tried to resolve it by deleting the counter run.

    Only ONE side is expanded. Two compound names are matched exactly, because expanding both lets
    any two merged counter runs find some member pair in the table and exempt a genuine
    interpenetration between them.
    """
    table = set(table)
    if (a, b) in table or (b, a) in table:
        return True
    ta, tb = category_tokens(a), category_tokens(b)
    if len(ta) > 1 and len(tb) > 1:
        return False
    return any((x, y) in table or (y, x) in table for x in ta for y in tb)


# ── graph types ──────────────────────────────────────────────────────────────
@dataclass
class Node:
    id: str
    kind: str                                  # room | wall | opening | floor | ceiling | object
    category: str = ""
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    source: str
    target: str
    relation: str
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneGraph:
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def node(self, node_id: str) -> Node | None:
        return next((n for n in self.nodes if n.id == node_id), None)

    def of_kind(self, kind: str) -> list[Node]:
        return [n for n in self.nodes if n.kind == kind]

    def relation(self, relation: str) -> list[Edge]:
        return [e for e in self.edges if e.relation == relation]

    def support_parent(self, node_id: str) -> str | None:
        """What holds this object up — exactly one node, or None if nothing does."""
        return next((e.target for e in self.edges
                     if e.source == node_id and e.relation == "supported_by"), None)

    def support_kind(self, node_id: str) -> str | None:
        edge = next((e for e in self.edges
                     if e.source == node_id and e.relation == "supported_by"), None)
        return edge.attrs.get("kind") if edge else None

    def support_children(self, node_id: str) -> list[str]:
        return [e.source for e in self.edges
                if e.target == node_id and e.relation == "supported_by"]

    def support_roots(self) -> list[str]:
        """The things everything ultimately rests on: the floor, the walls, the ceiling."""
        targets = {e.target for e in self.edges if e.relation == "supported_by"}
        return sorted(t for t in targets if self.support_parent(t) is None)

    def support_chain(self, node_id: str) -> list[str]:
        """This object's path down to its root — ``[Laptop0, Table0, Floor0]``."""
        chain, seen = [node_id], {node_id}
        while (parent := self.support_parent(chain[-1])) and parent not in seen:
            chain.append(parent)
            seen.add(parent)
        return chain

    def unsupported(self) -> list[str]:
        """Objects nothing holds up — a defect, reported rather than silently grounded."""
        return sorted(n.id for n in self.nodes
                      if n.kind == "object" and n.attrs.get("unsupported"))

    def group_members(self, parent_id: str) -> list[str]:
        return sorted(e.source for e in self.edges
                      if e.target == parent_id and e.relation == "belongs_to")

    def groups(self) -> dict[str, list[str]]:
        """Every functional group: parent id -> its members (a table and its chairs)."""
        out: dict[str, list[str]] = {}
        for edge in self.edges:
            if edge.relation == "belongs_to":
                out.setdefault(edge.target, []).append(edge.source)
        return {k: sorted(v) for k, v in sorted(out.items())}

    def neighbours(self, node_id: str, relation: str | None = None) -> list[str]:
        return [e.target for e in self.edges
                if e.source == node_id and (relation is None or e.relation == relation)]

    def to_dict(self) -> dict[str, Any]:
        return {"meta": self.meta,
                "nodes": [asdict(n) for n in self.nodes],
                "edges": [asdict(e) for e in self.edges]}

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    def to_networkx(self):
        """A ``networkx.MultiDiGraph`` view, for layout and analysis."""
        import networkx as nx

        g = nx.MultiDiGraph()
        for n in self.nodes:
            g.add_node(n.id, kind=n.kind, category=n.category, **n.attrs)
        for e in self.edges:
            g.add_edge(e.source, e.target, key=e.relation, relation=e.relation, **e.attrs)
        return g

    def describe(self) -> str:
        """A compact human-readable rendering of the whole graph."""
        lines = [f"scene graph: {len(self.nodes)} nodes, {len(self.edges)} edges"]
        by_relation: dict[str, list[Edge]] = {}
        for e in self.edges:
            by_relation.setdefault(e.relation, []).append(e)
        for relation in ("contains", "hosts", "corner", "supported_by", "belongs_to",
                         "against", "tucked_under", "clashes"):
            group = by_relation.get(relation)
            if not group:
                continue
            lines.append(f"  {relation} ({len(group)})")
            for e in group[:40]:
                note = " ".join(f"{k}={v}" for k, v in e.attrs.items())
                lines.append(f"    {e.source} -> {e.target}" + (f"   [{note}]" if note else ""))
            if len(group) > 40:
                lines.append(f"    … {len(group) - 40} more")
        return "\n".join(lines)


# ── geometry the rules run on ────────────────────────────────────────────────
def _obb(obj: dict[str, Any]) -> tuple[float, float, float, float, float]:
    """(cx, cy, half_w, half_d, yaw_rad) — the 2-D oriented footprint box."""
    return (obj["center"][0], obj["center"][1],
            obj["size"][0] / 2.0, obj["size"][1] / 2.0, math.radians(obj.get("yaw", 0.0)))


def obb_mtv(a, b) -> tuple[float, tuple[float, float]] | None:
    """Separating-axis overlap of two oriented rectangles → (depth, axis), or None if apart.

    The axis is the **minimum translation vector**, oriented a→b, so moving ``b`` by
    ``axis * depth`` is by construction the shortest move that separates them. Stage 3's whole
    resolver is built on that guarantee; an AABB test cannot provide it and false-clashes every
    rotated chair besides.
    """
    def geometry(o):
        cx, cy, hx, hy, r = o
        c, s = math.cos(r), math.sin(r)
        ux, uy = (c, s), (-s, c)
        corners = [(cx + sx * hx * ux[0] + sy * hy * uy[0],
                    cy + sx * hx * ux[1] + sy * hy * uy[1])
                   for sx in (-1, 1) for sy in (-1, 1)]
        return corners, (ux, uy)

    ca, axes_a = geometry(a)
    cb, axes_b = geometry(b)
    best, best_axis = float("inf"), None
    for axis in (*axes_a, *axes_b):
        length = math.hypot(*axis) or 1.0
        u = (axis[0] / length, axis[1] / length)
        pa = [p[0] * u[0] + p[1] * u[1] for p in ca]
        pb = [p[0] * u[0] + p[1] * u[1] for p in cb]
        overlap = min(max(pa), max(pb)) - max(min(pa), min(pb))
        if overlap <= 0:
            return None                                    # a separating axis exists → no contact
        if overlap < best:
            centre_delta = (sum(p[0] * u[0] + p[1] * u[1] for p in cb) / 4
                            - sum(p[0] * u[0] + p[1] * u[1] for p in ca) / 4)
            best, best_axis = overlap, (u if centre_delta >= 0 else (-u[0], -u[1]))
    return best, best_axis


def _polygon_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _clip_polygon(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland–Hodgman clip of one convex polygon by another → the intersection polygon."""
    output = list(subject)
    for i in range(len(clip)):
        a, b = clip[i], clip[(i + 1) % len(clip)]
        edge = b - a
        inside = lambda p: edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0]) >= -1e-12  # noqa: E731
        current, output = output, []
        for j in range(len(current)):
            p, q = current[j], current[(j + 1) % len(current)]
            if inside(p):
                output.append(p)
                if not inside(q):
                    output.append(_line_cross(p, q, a, b))
            elif inside(q):
                output.append(_line_cross(p, q, a, b))
        if not output:
            return np.zeros((0, 2))
    return np.asarray(output)


def _line_cross(p, q, a, b) -> np.ndarray:
    d1, d2 = q - p, b - a
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-12:
        return p
    t = ((a[0] - p[0]) * d2[1] - (a[1] - p[1]) * d2[0]) / denom
    return p + t * d1


def footprint_overlap(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Shared footprint area, as a fraction of the SMALLER of the two footprints."""
    pa, pb = object_footprint(a), object_footprint(b)
    area_a, area_b = _polygon_area(pa), _polygon_area(pb)
    if min(area_a, area_b) < 1e-9:
        return 0.0
    shared = _clip_polygon(pa, pb)
    if len(shared) < 3:
        return 0.0
    return _polygon_area(shared) / min(area_a, area_b)


def _footprint_gap(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Shortest distance between two oriented footprints. 0 when they touch or overlap.

    Centre-to-centre distance would make a chair beside a 3.5 m table look further away than one
    at the far end of a small table, which is the wrong ordering for grouping.
    """
    pa, pb = object_footprint(a), object_footprint(b)
    if len(_clip_polygon(pa, pb)) >= 3:
        return 0.0
    best = float("inf")
    for poly, other in ((pa, pb), (pb, pa)):
        for point in poly:
            for k in range(len(other)):
                p, q = other[k], other[(k + 1) % len(other)]
                seg = q - p
                length2 = float(seg @ seg)
                t = 0.0 if length2 < 1e-12 else max(0.0, min(1.0, float((point - p) @ seg) / length2))
                best = min(best, float(np.linalg.norm(point - (p + t * seg))))
    return best


def _support_along(obj: dict[str, Any], direction: np.ndarray) -> float:
    """How far an object's footprint reaches along a unit direction from its centre."""
    hw, hd = obj["size"][0] / 2.0, obj["size"][1] / 2.0
    yaw = math.radians(obj.get("yaw", 0.0))
    c, s = math.cos(yaw), math.sin(yaw)
    return abs(hw * (c * direction[0] + s * direction[1])) + abs(hd * (-s * direction[0] + c * direction[1]))


def wall_distance(obj: dict[str, Any], wall: dict[str, Any]) -> tuple[float, float] | None:
    """(gap from the object's box to the wall plane, position along the wall) or None if off its ends.

    The gap is measured from the box's *support* in the wall-normal direction, not from its centre,
    so a deep counter standing flush against a wall reads 0 and not half its depth. Walls are
    frequently diagonal, which is exactly why a world-AABB proxy is useless here.
    """
    start, along, normal, length = wall_frame(wall)
    centre = np.asarray(obj["center"][:2], dtype=float)
    rel = centre - start
    t = float(np.dot(rel, along))
    if not (-0.25 <= t <= length + 0.25):
        return None
    perp = abs(float(np.dot(rel, normal)))
    return perp - _support_along(obj, normal) - wall.get("thickness", 0.0) / 2.0, t


# ── the builder ──────────────────────────────────────────────────────────────
def build_graph(shell: dict[str, Any], *, name: str = "room", meta: dict[str, Any] | None = None) -> SceneGraph:
    """Apply every rule above to a SHELL and return the resulting :class:`SceneGraph`."""
    walls = shell.get("walls") or {}
    openings = shell.get("openings") or {}
    objects = shell.get("objects") or {}
    floor_z = float(shell.get("floor_z", 0.0))
    ceiling_z = float(shell.get("ceiling_z", floor_z + 2.5))
    verts = np.asarray((shell.get("floor") or {}).get("verts") or [], dtype=float)

    graph = SceneGraph(meta={"name": name, "floor_z": floor_z, "ceiling_z": ceiling_z,
                             **(meta or {})})
    room_id = "Room0"

    footprint_area = 0.0
    if len(verts):
        hull = verts[:, :2]
        footprint_area = _polygon_area(hull[_convex_hull_indices(hull)]) if len(hull) >= 3 else 0.0
    graph.nodes.append(Node(room_id, "room", name, {
        "height": round(ceiling_z - floor_z, 4),
        "floor_area": round(footprint_area, 3),
        "n_walls": len(walls), "n_openings": len(openings), "n_objects": len(objects),
    }))

    # --- structural nodes ---------------------------------------------------
    for wall_id, wall in walls.items():
        start, along, normal, length = wall_frame(wall)
        graph.nodes.append(Node(wall_id, "wall", "wall", {
            "start": [round(v, 4) for v in start],
            "end": [round(v, 4) for v in np.asarray(wall["end"], float)],
            "length": round(length, 4),
            "thickness": round(float(wall.get("thickness", 0.0)), 4),
            "height": round(ceiling_z - floor_z, 4),
            "normal": [round(v, 4) for v in normal],
            "sliver": bool(length < 0.35),      # RoomPlan corner artifact, kept but flagged
        }))
        graph.edges.append(Edge(room_id, wall_id, "contains"))

    for surface, z in (("Floor0", floor_z), ("Ceiling0", ceiling_z)):
        graph.nodes.append(Node(surface, surface[:-1].lower(), surface[:-1].lower(),
                                {"z": round(z, 4), "area": round(footprint_area, 3)}))
        graph.edges.append(Edge(room_id, surface, "contains"))

    for opening_id, opening in openings.items():
        graph.nodes.append(Node(opening_id, "opening", opening.get("type", "opening"), {
            "wall": opening.get("wall"),
            "offset": opening.get("offset"),   # CENTRE along the wall — see shell.opening_span
            "span": [round(v, 3) for v in opening_span(opening)],
            "width": opening.get("width"),
            "height": opening.get("height"), "sill": opening.get("sill"),
        }))
        graph.edges.append(Edge(room_id, opening_id, "contains"))
        wall_id = opening.get("wall")
        if wall_id in walls:
            t0, t1 = opening_span(opening)
            graph.edges.append(Edge(wall_id, opening_id, "hosts",
                                    {"span": [round(t0, 3), round(t1, 3)],
                                     "wall_length": round(wall_frame(wall)[3], 3)}))

    for object_id, obj in objects.items():
        bottom = obj["center"][2] - obj["size"][2] / 2.0
        top = obj["center"][2] + obj["size"][2] / 2.0
        graph.nodes.append(Node(object_id, "object", obj.get("category", ""), {
            "center": [round(v, 4) for v in obj["center"]],
            "size": [round(v, 4) for v in obj["size"]],
            "yaw": obj.get("yaw", 0.0),        # heading of the box's WIDTH axis
            "facing": round(facing_yaw(obj), 2),  # where the object points; see shell.facing_yaw
            "bottom_z": round(bottom, 4), "top_z": round(top, 4),
            "footprint_area": round(obj["size"][0] * obj["size"][1], 3),
            **({"members": obj["members"]} if obj.get("members") else {}),
        }))
        graph.edges.append(Edge(room_id, object_id, "contains"))

    # --- rule: walls that share a corner ------------------------------------
    wall_ids = list(walls)
    for i in range(len(wall_ids)):
        for j in range(i + 1, len(wall_ids)):
            a, b = walls[wall_ids[i]], walls[wall_ids[j]]
            for ka in ("start", "end"):
                for kb in ("start", "end"):
                    gap = float(np.linalg.norm(np.asarray(a[ka], float) - np.asarray(b[kb], float)))
                    if gap <= CORNER_TOL:
                        graph.edges.append(Edge(wall_ids[i], wall_ids[j], "corner",
                                                {"at": ka + "/" + kb, "gap": round(gap, 4)}))
                        break
                else:
                    continue
                break

    # --- rule: THE SUPPORT TREE — exactly one parent per object -------------
    # Resolved lowest-first so a supporter is already placed before anything can rest on it, and
    # a supporter must sit strictly lower than its child, which makes cycles impossible.
    ordered = sorted(objects.items(), key=lambda kv: kv[1]["center"][2] - kv[1]["size"][2] / 2.0)
    for object_id, obj in ordered:
        bottom = obj["center"][2] - obj["size"][2] / 2.0
        top = obj["center"][2] + obj["size"][2] / 2.0

        # 1. hung on a wall — a wall cabinet or a mounted TV is held by the wall, not the floor
        hung = is_wall_hung(obj, walls, floor_z)
        if hung is not None:
            graph.edges.append(Edge(object_id, hung, "supported_by",
                                    {"kind": "wall", "height": round(bottom - floor_z, 3)}))
            continue

        # 2. stacked on a lower object whose top surface it meets
        best: tuple[float, str, float] | None = None
        for other_id, other in objects.items():
            if other_id == object_id:
                continue
            other_top = other["center"][2] + other["size"][2] / 2.0
            other_bottom = other["center"][2] - other["size"][2] / 2.0
            if other_bottom >= bottom - 1e-6:
                continue                                   # not below us — cannot hold us up
            if not (-SUPPORT_TOL <= bottom - other_top <= SUPPORT_TOL):
                continue
            if footprint_overlap(obj, other) < SUPPORT_OVERLAP:
                continue
            if best is None or other_top > best[2]:
                best = (bottom - other_top, other_id, other_top)
        if best is not None:
            graph.edges.append(Edge(object_id, best[1], "supported_by",
                                    {"kind": "object", "gap": round(best[0], 4)}))
            continue

        # 3. standing on the floor
        if abs(bottom - floor_z) <= SUPPORT_TOL:
            graph.edges.append(Edge(object_id, "Floor0", "supported_by",
                                    {"kind": "floor", "gap": round(bottom - floor_z, 4)}))
            continue

        # 4. fixed to the ceiling — lights, vents, ceiling grids
        if abs(ceiling_z - top) <= CEILING_TOL:
            graph.edges.append(Edge(object_id, "Ceiling0", "supported_by",
                                    {"kind": "ceiling", "gap": round(ceiling_z - top, 4)}))
            continue

        # 5. nothing holds it up. Reported, never invented: an unsupported object is a defect in
        #    the scan, and silently attaching it to the floor would hide exactly that.
        graph.nodes[[n.id for n in graph.nodes].index(object_id)].attrs["unsupported"] = True

    # --- rule: which wall an object is merely AGAINST (contact, not support) --
    for object_id, obj in objects.items():
        if graph.support_parent(object_id) is not None and \
                graph.support_kind(object_id) == "wall":
            continue                                       # already held by a wall
        best_against: tuple[float, str] | None = None
        for wall_id, wall in walls.items():
            measured = wall_distance(obj, wall)
            if measured is None:
                continue
            if measured[0] <= AGAINST_TOL and (best_against is None or measured[0] < best_against[0]):
                best_against = (measured[0], wall_id)
        if best_against is not None:
            graph.edges.append(Edge(object_id, best_against[1], "against",
                                    {"gap": round(best_against[0], 4)}))

    # --- rule: GROUPING — which parent object a child belongs to -------------
    # Functional, not physical: a chair rests on the FLOOR but belongs to its TABLE. Proximity
    # alone is not enough in a room of eight tables, so a chair that FACES a table is preferred
    # over one that is marginally closer — that is what distinguishes "at this table" from
    # "standing next to it on the way past".
    for object_id, obj in objects.items():
        parents = GROUP_PARENTS.get(obj.get("category", ""))
        if not parents:
            continue
        centre = np.asarray(obj["center"][:2], dtype=float)
        facing = math.radians(facing_yaw(obj))
        best: tuple[float, str, float, float] | None = None
        for other_id, other in objects.items():
            if other_id == object_id or other.get("category", "") not in parents:
                continue
            gap = _footprint_gap(obj, other)
            if gap > GROUP_GAP:
                continue
            to_parent = np.asarray(other["center"][:2], dtype=float) - centre
            heading = math.degrees(math.atan2(to_parent[1], to_parent[0]))
            off = abs((math.degrees(facing) - heading + 180.0) % 360.0 - 180.0)
            # A chair facing its table is worth up to GROUP_FACING_BONUS of "closeness".
            cost = gap + (off / 180.0) * (GROUP_FACING_BONUS / 180.0)
            if best is None or cost < best[0]:
                best = (cost, other_id, gap, off)
        if best is not None:
            graph.edges.append(Edge(object_id, best[1], "belongs_to",
                                    {"gap": round(best[2], 3), "facing_off_deg": round(best[3], 1)}))

    # --- rule: containment vs. genuine clash --------------------------------
    ids = list(objects)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a_id, b_id = ids[i], ids[j]
            a, b = objects[a_id], objects[b_id]
            a_bottom, a_top = a["center"][2] - a["size"][2] / 2, a["center"][2] + a["size"][2] / 2
            b_bottom, b_top = b["center"][2] - b["size"][2] / 2, b["center"][2] + b["size"][2] / 2
            if min(a_top, b_top) - max(a_bottom, b_bottom) <= Z_BAND_TOL:
                continue                                    # no shared height band → cannot collide
            mtv = obb_mtv(_obb(a), _obb(b))
            if not mtv or mtv[0] <= CLASH_TOL:
                continue
            depth, axis = mtv
            cat_a, cat_b = a.get("category", ""), b.get("category", "")
            if expected_pair(cat_a, cat_b):
                # EXPECTED_CONTAINMENT is written (contained, container), so the pair itself says
                # which way the edge points: the chair goes under the table, never the reverse.
                if len(category_tokens(cat_a)) > 1 or len(category_tokens(cat_b)) > 1:
                    # A merged run names no direction: ``storage`` inside ``oven_storage_stove``
                    # matches ``("oven", "storage")`` in BOTH orders, and reading the table would
                    # put the counter run inside the cabinet it swallowed. Volume is not ambiguous.
                    volumes = {i: o["size"][0] * o["size"][1] * o["size"][2]
                               for i, o in ((a_id, a), (b_id, b))}
                    inner = min(volumes, key=volumes.get)
                else:
                    inner = a_id if (cat_a, cat_b) in EXPECTED_CONTAINMENT else b_id
                outer = b_id if inner == a_id else a_id
                graph.edges.append(Edge(inner, outer, "tucked_under", {"depth": round(depth, 4)}))
            elif cat_a in PASSTHROUGH or cat_b in PASSTHROUGH \
                    or cat_a in OPEN_FRAME or cat_b in OPEN_FRAME:
                continue                                    # open frames and runs are meant to overlap
            else:
                graph.edges.append(Edge(a_id, b_id, "clashes", {
                    "depth": round(depth, 4),
                    "axis": [round(axis[0], 4), round(axis[1], 4)],
                }))

    graph.meta["n_clashes"] = len(graph.relation("clashes"))
    graph.meta["support_roots"] = graph.support_roots()
    graph.meta["unsupported"] = graph.unsupported()
    graph.meta["groups"] = graph.groups()
    return graph


def _convex_hull_indices(points: np.ndarray) -> list[int]:
    """Monotone-chain hull — used only to give the room a stable floor area."""
    order = sorted(range(len(points)), key=lambda i: (points[i][0], points[i][1]))
    def half(seq):
        out: list[int] = []
        for i in seq:
            while len(out) >= 2:
                o, a, p = points[out[-2]], points[out[-1]], points[i]
                if (a[0] - o[0]) * (p[1] - o[1]) - (a[1] - o[1]) * (p[0] - o[0]) > 1e-12:
                    break
                out.pop()
            out.append(i)
        return out
    lower, upper = half(order), half(reversed(order))
    return lower[:-1] + upper[:-1]
