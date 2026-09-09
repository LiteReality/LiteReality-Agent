"""sim_assets.py — the physics a generated object carries, read back and placed in a room.

Every object the reconstruct stage builds now leaves a sidecar beside its GLB: a
`<name>.physics.json` holding the links, joints, masses, inertias, frictions and CONVEX COLLIDERS
that `object_generation.sim` compiled from the mesh, plus the OBJ files those colliders live in.
It is the asset's own statement of what it is physically, gated by a solver before it was ever put
in a room.

This module is how the room-level MuJoCo export reads that statement instead of re-inventing it.
The difference is not cosmetic:

  * the PIVOT of a hinge is authored — `object.py` placed the hinge and recorded where. The room
    exporter's alternative is to guess it from the leaf's geometry, which is right for a door with
    an obvious free edge and wrong for a sash, a lid, or a drawer.
  * the MASS is split across the links by volume, so a door leaf weighs what a door leaf weighs
    rather than a share of the whole object's bounding box.
  * FRICTION comes from the material the recipe deliberately chose. Glass grips at 0.30, carpet at
    0.85, and a room where everything is 0.55 slides wrong in a way no amount of tuning fixes.
  * the COLLIDERS are already decomposed, so a 25-minute CoACD pass over the whole room turns into
    reading a few dozen OBJ files.

`room_ops` may not import the pipeline or the models layer (the architecture test enforces it), so
nothing here imports the compiler. The sidecar is a FILE FORMAT, and this reads it as one — the
dataclasses below mirror `object_generation.sim.properties` and are checked against
`schema_version`.

FRAMES. The sidecar is Z-up metres in the object's own build frame, with each link's geometry
expressed relative to that link's own frame origin (which sits on its joint). A room places the
object with a rotation and a NON-UNIFORM scale — RoomPlan measured a box and the asset was fitted
into it — so everything has to travel through one affine on the way in. `placement_transform`
recovers that affine exactly from the two glTF files, and `place` applies it.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1

# glTF is Y-up, the SHELL and MuJoCo are Z-up. Kept local rather than imported from mujoco_scene so
# this module can be read (and tested) on its own.
_A = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
_A4 = np.eye(4)
_A4[:3, :3] = _A
_A4_INV = np.linalg.inv(_A4)

# trimesh appends `_<6 hex>` to a node name it has already seen, and it picks a DIFFERENT suffix
# each time a file is loaded. So `DoorLeaf_5a8bb9` in the object's own glb and `DoorLeaf_22e63d` in
# the room glb are the same node, and no amount of string equality will say so. Only names that are
# unique within their own file can be matched across the two — which is enough, because one matched
# node determines the whole placement.
_DEDUP_SUFFIX = re.compile(r"_[0-9a-f]{6}$")

MIN_LINK_MASS = 0.15


def base_name(node: str) -> str:
    """The node name with trimesh's de-duplication suffix removed."""
    return _DEDUP_SUFFIX.sub("", node)


@dataclass
class Collider:
    file: str
    volume: float


@dataclass
class Link:
    name: str
    mass: float
    com: list
    inertia: list
    friction: float
    restitution: float
    material: str = ""
    visual: str = ""
    colliders: list = field(default_factory=list)
    mass_source: str = "derived"
    inertia_source: str = "geometry"


@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    origin: list
    axis: list
    limit_lower: float
    limit_upper: float
    damping: float = 0.05
    friction: float = 0.02
    effort: float = 20.0
    velocity: float = 3.0
    origin_source: str = "node_transform"


@dataclass
class SimModel:
    name: str
    category: str
    root: str
    links: list = field(default_factory=list)
    joints: list = field(default_factory=list)
    bbox: list = field(default_factory=list)
    total_mass: float = 0.0
    notes: list = field(default_factory=list)
    directory: Path = field(default=Path("."))

    def link(self, name: str) -> Link | None:
        return next((link for link in self.links if link.name == name), None)

    def frames(self) -> dict[str, np.ndarray]:
        """Each link's frame origin in the OBJECT's own Z-up frame.

        A joint records its pivot in the PARENT's frame, so the absolute origin is the chain of
        them accumulated from the root. Written iteratively rather than recursively because a
        malformed sidecar could otherwise describe a cycle and blow the stack.
        """
        origins = {self.root: np.zeros(3)}
        pending = list(self.joints)
        for _ in range(len(pending) + 1):
            if not pending:
                break
            rest = []
            for joint in pending:
                if joint.parent in origins:
                    origins[joint.child] = origins[joint.parent] + np.asarray(joint.origin, float)
                else:
                    rest.append(joint)
            if len(rest) == len(pending):
                break                      # an orphan or a cycle: leave the rest at the root
            pending = rest
        for link in self.links:
            origins.setdefault(link.name, np.zeros(3))
        return origins


def sidecar_dir(room: Path, object_name: str) -> Path | None:
    """The `sim/` directory an object's physics was compiled into, inside the Room package.

    `export_room` carries it in beside the object's own source, so a Room directory is a complete
    statement of the room INCLUDING its physics. Both object kinds are searched because a
    generative mesh (a TRELLIS chair) is as much a rigid body as a recipe-built one.
    """
    for kind in ("Procedural", "Static"):
        candidate = Path(room) / "Objects" / kind / object_name / "sim"
        if (candidate / f"{object_name}.physics.json").is_file():
            return candidate
    return None


def load(path: Path) -> SimModel:
    """One `<name>.physics.json` -> the model. Raises on a schema this code does not know."""
    path = Path(path)
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    version = raw.pop("schema_version", 0)
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"{path} is physics schema v{version}, this reads v{SCHEMA_VERSION}. Reading it "
            f"anyway would put a wrong mass or a wrong hinge into a scene and report success."
        )
    links = [Link(**{**row, "colliders": [Collider(**c) for c in row.get("colliders", [])]})
             for row in raw.get("links", [])]
    joints = [Joint(**row) for row in raw.get("joints", [])]
    return SimModel(name=raw["name"], category=raw.get("category", ""),
                    root=raw.get("root", "base_link"), links=links, joints=joints,
                    bbox=raw.get("bbox") or [], total_mass=float(raw.get("total_mass") or 0.0),
                    notes=list(raw.get("notes") or []), directory=path.parent)


# ------------------------------------------------------------------ placement


def placement_transform(object_scene, room_scene, room_nodes) -> np.ndarray | None:
    """The affine that carries the object's own glb frame onto where the room put it (Y-up).

    `build_room` fits an asset into the RoomPlan box it belongs to, which is a rotation about the
    vertical and a per-axis SCALE — the box was measured by the scan and the asset was built to its
    own proportions, so the two rarely agree. Any single node present in both files pins the whole
    thing down:

        T = T_room(node) . T_object(node)^-1

    and every other node then maps through it exactly. Returns None when no node can be matched,
    which is the caller's signal to fall back to deriving physics from the placed mesh instead of
    quietly using an unplaced one.
    """
    from collections import Counter

    def unique(nodes) -> dict[str, str]:
        counts = Counter(base_name(n) for n in nodes)
        return {base_name(n): n for n in nodes if counts[base_name(n)] == 1}

    in_object = unique(list(object_scene.graph.nodes_geometry))
    in_room = unique(list(room_nodes))
    shared = sorted(set(in_object) & set(in_room))
    if not shared:
        return None
    # Prefer the node with the most geometry: a big part pins the transform with the least relative
    # error if the two files ever disagree, and it is the one most likely to be the carcass rather
    # than a fitting that a later edit moved.
    def area(name: str) -> float:
        try:
            return float(object_scene.geometry[object_scene.graph[in_object[name]][1]].area)
        except Exception:                                    # noqa: BLE001 — degenerate geometry
            return 0.0

    best = max(shared, key=area)
    to_object = np.asarray(object_scene.graph[in_object[best]][0], dtype=float)
    to_room = np.asarray(room_scene.graph[in_room[best]][0], dtype=float)
    try:
        return to_room @ np.linalg.inv(to_object)
    except np.linalg.LinAlgError:
        return None


def placement_from_bbox(model: SimModel, placed_mesh, yaw: float) -> np.ndarray | None:
    """The same affine, recovered from GEOMETRY when no node name survived the room build.

    An object placed more than once — four chairs from one cluster, a repeated cabinet — has every
    node renamed for every instance after the first, and a cluster whose own parts already repeat
    has nothing unique inside its own file either. Name matching then reports nothing for objects
    that are perfectly well placed.

    But the placement has a known SHAPE. `build_room` fits an asset into its box as

        world = T(centre) . Rz(yaw) . diag(s) . (p - asset centre)

    with `yaw` stated in the SHELL, so rotating the placed mesh back by that yaw makes the fit
    axis-aligned and the two bounding boxes give `s` and `centre` directly. It is exact for exactly
    the transform the room applies, which is why the caller still checks the result.

    Returns the Y-up matrix, so it feeds `place` identically to `placement_transform`.
    """
    if not model.bbox or len(model.bbox) != 2:
        return None
    lo = np.asarray(model.bbox[0], dtype=float)
    hi = np.asarray(model.bbox[1], dtype=float)
    asset_size = hi - lo
    if np.any(asset_size < 1e-6):
        return None

    c, s_ = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
    aligned = np.asarray(placed_mesh.vertices, dtype=float) @ rotation     # == Rz(-yaw) . v
    placed_lo, placed_hi = aligned.min(axis=0), aligned.max(axis=0)
    scale = (placed_hi - placed_lo) / asset_size
    if np.any(~np.isfinite(scale)) or np.any(np.abs(scale) < 1e-9):
        return None

    linear = rotation @ np.diag(scale)
    centre = rotation @ ((placed_lo + placed_hi) / 2.0)
    transform = np.eye(4)
    transform[:3, :3] = linear
    transform[:3, 3] = centre - linear @ ((lo + hi) / 2.0)
    return _A4_INV @ transform @ _A4


@dataclass
class PlacedLink:
    name: str
    mass: float
    friction: float
    restitution: float
    inertia: np.ndarray              # [ixx, iyy, izz, ixy, ixz, iyz] about the com
    com: np.ndarray                  # world Z-up, metres
    colliders: list                  # trimesh meshes, world Z-up
    mass_source: str
    inertia_source: str


@dataclass
class Placed:
    """A SimModel expressed in the room's world frame, ready to be written as MJCF bodies."""
    model: SimModel
    transform: np.ndarray            # 4x4, world Z-up <- object Z-up
    volume_ratio: float
    links: dict                      # name -> PlacedLink
    pivots: dict                     # link name -> world Z-up pivot
    axes: dict                       # link name -> world Z-up unit axis
    total_mass: float

    def joint_for(self, link: str) -> Joint | None:
        return next((j for j in self.model.joints if j.child == link), None)


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def _inertia_about_com(mesh, mass: float):
    """Inertia about the centre of mass, from the placed geometry, at the sidecar's mass.

    The sidecar's own tensor describes the object at the size it was BUILT, and the room scales it
    — a door stretched 10% taller has a different tensor, not a scaled mass with the old one. It is
    cheaper and more honest to recompute from the geometry that is actually going into the scene.
    The ladder mesh -> convex hull -> bounding box is the one `object_generation.sim` uses, and it
    exists because a flat visual part (a pane, a poster) has no volume for trimesh to integrate and
    MuJoCo refuses a non-positive-definite inertia at compile time.
    """
    for candidate, source in ((mesh, "geometry"), (mesh.convex_hull, "hull")):
        try:
            if candidate.volume > 1e-9:
                scaled = candidate.copy()
                scaled.density = mass / float(candidate.volume)
                tensor = np.asarray(scaled.moment_inertia, dtype=float)
                if np.all(np.linalg.eigvalsh(tensor) > 0):
                    return tensor, np.asarray(scaled.center_mass, dtype=float), source
        except Exception:                                    # noqa: BLE001 — degenerate mesh
            continue
    extents = np.maximum(np.asarray(mesh.extents, dtype=float), 0.01)
    diagonal = mass / 12.0 * np.array([extents[1] ** 2 + extents[2] ** 2,
                                       extents[0] ** 2 + extents[2] ** 2,
                                       extents[0] ** 2 + extents[1] ** 2])
    return np.diag(diagonal), np.asarray(mesh.bounds, dtype=float).mean(axis=0), "bounding_box"


def place(model: SimModel, transform_yup: np.ndarray) -> Placed:
    """Carry a model's colliders, masses, pivots and axes into the room's world frame.

    `transform_yup` comes from `placement_transform`; it is converted here so that every number
    this returns is Z-up and nothing downstream has to think about frames again.
    """
    import trimesh

    transform = _A4 @ np.asarray(transform_yup, dtype=float) @ _A4_INV
    linear = transform[:3, :3]
    # How much bigger the room made this object. A mass the sidecar DERIVED came from occupancy
    # density over the object's own bounding box, so at a different size it is a different mass;
    # a mass the recipe AUTHORED is a fact about the real object and survives the fit unchanged.
    volume_ratio = abs(float(np.linalg.det(linear))) or 1.0
    frames = model.frames()

    links: dict[str, PlacedLink] = {}
    pivots: dict[str, np.ndarray] = {}
    axes: dict[str, np.ndarray] = {}
    for link in model.links:
        frame = frames.get(link.name, np.zeros(3))
        pivots[link.name] = _transform_points(frame.reshape(1, 3), transform)[0]

        pieces = []
        for collider in link.colliders:
            path = model.directory / collider.file
            if not path.is_file():
                continue
            try:
                piece = trimesh.load(str(path), process=False, force="mesh")
            except Exception:                                # noqa: BLE001 — unreadable collider
                continue
            # The collider was written relative to the link's own frame; put it back in the
            # object's frame before placing it, or every moving part lands on the carcass.
            piece.vertices = _transform_points(np.asarray(piece.vertices, float) + frame, transform)
            if piece.volume > 1e-9 or len(piece.vertices) >= 4:
                pieces.append(piece)
        if not pieces:
            continue

        mass = float(link.mass)
        if link.mass_source != "authored":
            mass *= volume_ratio
        mass = max(mass, MIN_LINK_MASS)
        merged = trimesh.util.concatenate(pieces)
        inertia, com, source = _inertia_about_com(merged, mass)
        links[link.name] = PlacedLink(
            name=link.name, mass=mass, friction=float(link.friction),
            restitution=float(link.restitution),
            inertia=np.array([inertia[0, 0], inertia[1, 1], inertia[2, 2],
                              inertia[0, 1], inertia[0, 2], inertia[1, 2]], dtype=float),
            com=np.asarray(com, dtype=float), colliders=pieces,
            mass_source=link.mass_source, inertia_source=source)

    for joint in model.joints:
        # A joint AXIS is a direction in the object's frame and travels through the linear part —
        # the same map `mujoco_scene._axis_to_world` applies, and for the rotation-plus-diagonal
        # scale a room fit produces it is the correct one. It is renormalised because the scale is
        # not uniform and an unnormalised MuJoCo axis silently changes a slide joint's units.
        world = linear @ np.asarray(joint.axis, dtype=float)
        norm = float(np.linalg.norm(world))
        axes[joint.child] = world / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])

    return Placed(model=model, transform=transform, volume_ratio=volume_ratio, links=links,
                  pivots=pivots, axes=axes,
                  total_mass=sum(link.mass for link in links.values()))


def assign_nodes(node_names, model: SimModel) -> dict[str, list[str]]:
    """Room-glb node names -> the link each belongs to.

    A link owns the node it was named for and everything welded under it, and the room appends a
    de-duplication suffix to any name it has seen before. Longest match wins so that `door_leaf`
    does not swallow a `door_leaf_handle` that has its own joint. Anything unclaimed belongs to the
    root, which is what "welded into the carcass" means.
    """
    names = [link.name for link in model.links if link.name != model.root]
    out: dict[str, list[str]] = {}
    for node in node_names:
        stem = base_name(node)
        owner = None
        for candidate in names:
            if stem == candidate or stem.startswith(candidate + "_"):
                if owner is None or len(candidate) > len(owner):
                    owner = candidate
        out.setdefault(owner or model.root, []).append(node)
    return out
