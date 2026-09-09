"""The physical properties an articulated object needs before a simulator will accept it.

The build agent authors GEOMETRY and, now, whatever physics it actually knows: a mass it can
read off a label, the friction of a material it deliberately chose. Everything it does not know
is DERIVED here from the mesh it did build. Authored always wins; derivation only fills gaps.

That split is the whole point. Before this, an object left `obj_stage` carrying four numbers per
moving part (`articulation_type`, `articulation_axis`, `limit_min`, `limit_max`) and nothing else
— no mass, no inertia, no collider, no friction. The room-level MuJoCo export invented all of it
from a category table at the last moment, which works for one consumer and leaves the asset itself
unusable anywhere. Here the asset carries its own physics, and every exporter (URDF, MJCF) is a
pure function of this model.

Frame: a GLB from `blender_lib.export_glb` is Y-up, and its node translations are converted ONCE
on the way in to the Z-up metre frame URDF and MuJoCo both use. The articulation EXTRAS are the
exception and are already Z-up: `set_articulation` writes the axis raw, in the Blender build frame,
so a single file carries geometry in one frame and joint axes in another. Everything below is
Z-up; no exporter has to think about it again.
"""

from __future__ import annotations

import json
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# glTF (Y-up) -> URDF/MuJoCo (Z-up).  (x, y, z) -> (x, -z, y)
YUP_TO_ZUP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

# kg per cubic metre of the object's BOUNDING BOX, not of its mesh. A real object is mostly air:
# a chair is four thin legs and a thin seat, so material density over its mesh volume makes it
# weigh 1.5 kg and behave like cardboard. These are the numbers the room-level MuJoCo export
# arrived at by watching furniture get thrown across rooms, so they stay calibrated the same way;
# what is new is that the total is then split across the object's links by mesh volume, which is
# what gives a drawer front a believable mass of its own.
OCCUPANCY_DENSITY = {
    "default": 45.0,
    # 14.0 put a task chair at 7.2 kg and, after the room's fit rescaled it, 4.9 kg on the floor —
    # against 12-15 kg for the real thing. Everything downstream inherits that: a body at a third
    # of its weight takes a third of the push to send it across the room, which reads as the
    # simulation being unstable when it is the mass that is wrong. Measured against real furniture
    # over its own bounding box, a chair is 23-27 kg/m^3 whatever kind it is —
    #   task chair, castors + gas strut   0.49 m^3   13 kg -> 26.8
    #   cantilever meeting chair          0.23 m^3  5.5 kg -> 23.5
    #   wooden dining chair               0.21 m^3  5.0 kg -> 23.4
    # so one number serves them all, which is just as well: the category that reaches here is a
    # bare "chair" with no statement of which sort it is. A stool is the exception and needs its
    # own, because it is small enough that its fixed frame dominates its enclosed volume (0.12 m^3,
    # 5 kg -> 41.7).
    "chair": 25.0, "stool": 40.0, "table": 20.0, "desk": 22.0,
    "storage": 55.0, "cabinet": 55.0, "wardrobe": 50.0, "shelf": 45.0,
    "bed": 35.0, "sofa": 30.0, "refrigerator": 110.0, "dishwasher": 120.0,
    "oven": 120.0, "washer": 150.0, "radiator": 160.0,
    "television": 90.0, "monitor": 110.0, "sink": 60.0,
    "door": 90.0, "window": 45.0, "opening": 45.0,
}

# Sliding friction keyed on the material NAME the recipe chose, because that name is the only
# surviving statement of what the surface is meant to be. Everything else falls back to 0.55 —
# the value the room export uses globally, so a scene built from these assets does not change
# behaviour on the day it starts reading them.
MATERIAL_FRICTION = {
    "glass": 0.30, "glaz": 0.30, "mirror": 0.30,
    "metal": 0.42, "steel": 0.42, "chrome": 0.35, "alu": 0.42, "brass": 0.40,
    "wood": 0.55, "oak": 0.55, "beech": 0.55, "pine": 0.55, "ply": 0.55, "mdf": 0.58,
    "laminate": 0.45, "melamine": 0.45, "gloss": 0.38, "paint": 0.50,
    "fabric": 0.75, "cloth": 0.75, "felt": 0.80, "carpet": 0.85, "leather": 0.62,
    "rubber": 0.95, "plastic": 0.48, "abs": 0.48, "stone": 0.60, "granite": 0.60,
    "tile": 0.52, "ceramic": 0.52, "porcelain": 0.52,
}
DEFAULT_FRICTION = 0.55

# Bounce. MuJoCo has no restitution parameter at all and Isaac/Bullet do, so this is carried for
# their benefit and simply ignored by the MJCF exporter — the same choice Articraft makes.
MATERIAL_RESTITUTION = {
    "glass": 0.30, "metal": 0.25, "steel": 0.25, "chrome": 0.25,
    "rubber": 0.60, "plastic": 0.35, "wood": 0.20, "fabric": 0.05, "carpet": 0.05,
}
DEFAULT_RESTITUTION = 0.15

# Joint dynamics. A cabinet door is not frictionless and it is not a robot actuator either; these
# are the values the room export runs with, promoted from constants in that script to defaults
# that an authored object is free to override.
JOINT_DEFAULTS = {
    "revolute":  {"damping": 0.05, "friction": 0.02, "effort": 20.0, "velocity": 3.0},
    "prismatic": {"damping": 0.05, "friction": 0.02, "effort": 40.0, "velocity": 1.0},
    "continuous": {"damping": 0.05, "friction": 0.02, "effort": 20.0, "velocity": 3.0},
    "fixed": {},
}

MIN_LINK_MASS = 0.15          # kg — below this MuJoCo throws whatever the body touches
CONVEX_TOL = 0.06             # hull within 6% of the mesh volume is already its own collider
MIN_DECOMP_DIAGONAL = 0.30    # m — smaller than this, a convex hull is a fine collider
DECOMP_TIMEOUT = 45.0         # s per link


# ------------------------------------------------------------------ the model


@dataclass
class Collider:
    """One CONVEX piece. Every engine here collides a mesh as its hull, so concavity that
    matters — the space under a table, the inside of a cupboard — has to be spelled out as
    several convex pieces or it is silently filled in."""
    file: str
    volume: float


@dataclass
class Link:
    name: str
    mass: float
    com: list                      # centre of mass, link frame, metres
    inertia: list                  # [ixx, iyy, izz, ixy, ixz, iyz] about the com
    friction: float
    restitution: float
    material: str = ""
    visual: str = ""               # mesh file, link frame
    colliders: list = field(default_factory=list)
    mass_source: str = "derived"   # "authored" | "derived"
    inertia_source: str = "geometry"


@dataclass
class Joint:
    name: str
    type: str                      # revolute | prismatic | continuous | fixed
    parent: str
    child: str
    origin: list                   # the pivot, in the PARENT link frame — explicit, not implied
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

    def to_json(self) -> dict:
        d = asdict(self)
        d["schema_version"] = 1
        return d


# ------------------------------------------------------------------ glTF reading


def _gltf_json(path: Path) -> dict:
    raw = path.read_bytes()
    if raw[:4] != b"glTF":
        raise ValueError(f"not a GLB: {path}")
    length, _ = struct.unpack_from("<II", raw, 12)
    return json.loads(raw[20:20 + length].decode("utf-8"))


def _node_material(gltf: dict, node: dict) -> str:
    """The glTF material name on a node's first primitive — our only evidence of what it is made
    of, and therefore of how much it grips."""
    mi = node.get("mesh")
    if mi is None:
        return ""
    prims = gltf.get("meshes", [])[mi].get("primitives", [])
    for p in prims:
        m = p.get("material")
        if m is not None:
            return gltf.get("materials", [])[m].get("name", "") or ""
    return ""


def _lookup(table: dict, key: str, default):
    k = (key or "").lower()
    for token, value in table.items():
        if token in k:
            return value
    return default


# ------------------------------------------------------------------ derivation


def _inertia_from_geometry(mesh, mass: float):
    """Inertia tensor about the centre of mass, from the mesh the agent actually built.

    A visual mesh is not always a solid: a door leaf modelled as a single quad has zero volume and
    trimesh returns a zero (or negative) tensor, which MuJoCo rejects at compile time. The ladder
    is mesh -> convex hull -> the mesh's own bounding box as a uniform solid, which is never wrong
    by more than the shape's own concavity and is always positive definite.
    """
    for candidate, source in ((mesh, "geometry"), (mesh.convex_hull, "hull")):
        try:
            if candidate.volume > 1e-9:
                m = candidate.copy()
                m.density = mass / float(candidate.volume)
                it = np.asarray(m.moment_inertia, dtype=float)
                if np.all(np.linalg.eigvalsh(it) > 0):
                    return it, np.asarray(m.center_mass, dtype=float), source
        except Exception:                                    # noqa: BLE001 — degenerate mesh
            continue
    e = np.maximum(np.asarray(mesh.extents, dtype=float), 0.01)
    diag = mass / 12.0 * np.array([e[1] ** 2 + e[2] ** 2, e[0] ** 2 + e[2] ** 2,
                                   e[0] ** 2 + e[1] ** 2])
    centre = np.asarray(mesh.bounds, dtype=float).mean(axis=0)
    return np.diag(diag), centre, "bounding_box"


def _coacd_worker(vertices, faces, budget, mode, queue):
    # The IMPORT is reported separately from the run. "coacd cannot be loaded" and "coacd could not
    # split this mesh" both end in an empty result, and only the second is a fact about the mesh;
    # collapsing them is what let a whole scene fall back to convex hulls and still pass.
    try:
        import coacd
    except Exception as exc:                                 # noqa: BLE001 — broken wheel included
        queue.put({"import_error": f"{type(exc).__name__}: {exc}"})
        return
    try:
        coacd.set_log_level("error")          # it logs a progress bar per split, to stdout
        parts = coacd.run_coacd(coacd.Mesh(vertices, faces),
                                max_convex_hull=budget, preprocess_mode=mode)
        queue.put([(np.asarray(v).tolist(), np.asarray(f).tolist()) for v, f in parts])
    except Exception:                                        # noqa: BLE001
        queue.put([])


def require(module: str, what: str) -> None:
    """A declared dependency that is missing is an INSTALL fault, and it has to say so.

    Both of this module's native dependencies fail quietly if you let them. Without `coacd` every
    concave link falls back to its own convex hull and the gate still reports a clean pass — a
    cupboard becomes a solid block and nothing in the report says why. Without `mujoco` the physics
    json and URDF are still written, so the output looks finished and was never checked. Neither is
    a condition to tolerate now that both are declared in pyproject: raise, and let the caller
    record it as the error it is.
    """
    from importlib.util import find_spec

    if find_spec(module) is None:
        raise RuntimeError(
            f"{module} is not installed, so {what}. It is a declared dependency — "
            f"run `uv sync` rather than working around this: without it the output looks "
            f"complete and is quietly wrong."
        )


def _decompose(mesh, budget: int):
    """Convex parts, or []. Never hangs, and never dies with the caller.

    TWO MODES, IN THIS ORDER, and the second one is not optional. `preprocess_mode="off"` is the
    one worth having: the default voxel remesh inflates a part by about a third, which lifts a
    shelf above where its own mesh ends and ejects whatever was resting on it. But CoACD does not
    merely fail on a mesh it cannot handle with preprocessing off — it SEGFAULTS, and a cupboard
    carcass is exactly the kind of mesh that does it. Without the voxelised fallback those links
    fall all the way through to a single convex hull, which turns a cupboard into a solid block and
    a drawer into a brick: the precise failure this function exists to prevent. A slightly inflated
    decomposition is a far better collider than no decomposition at all.

    Each attempt runs in its own process because neither failure mode is catchable in-process: a
    segfault takes the interpreter with it, and a near-planar mesh makes CoACD spin rather than
    raise — one flat board once ran for over two hours. The loop polls instead of blocking on the
    timeout so a crash costs milliseconds rather than the full deadline, and reads the queue before
    joining, because joining a process that has put a large object on a queue deadlocks.
    """
    import multiprocessing as mp
    import queue as queue_mod
    import time

    require("coacd", "concave links would silently collapse to a single convex hull")
    for mode in ("off", "auto"):
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        proc = ctx.Process(target=_coacd_worker,
                           args=(np.asarray(mesh.vertices), np.asarray(mesh.faces), budget,
                                 mode, q))
        proc.start()
        parts, deadline = [], time.time() + DECOMP_TIMEOUT
        while time.time() < deadline:
            try:
                parts = q.get(timeout=0.5)
                break
            except queue_mod.Empty:
                if not proc.is_alive():           # segfaulted without putting anything
                    break
        proc.terminate()
        proc.join(5)
        if isinstance(parts, dict):               # could not load coacd at all
            raise RuntimeError(
                f"coacd could not be imported ({parts['import_error']}), so concave links would "
                "silently collapse to a single convex hull — a cupboard as a solid block. It is a "
                "declared dependency: run `uv sync` rather than working around this."
            )
        if parts:
            return parts
    return []


def _part_budget(mass: float) -> int:
    """Two dozen thin hulls on a light body is a stack of simultaneous contacts for the solver to
    satisfy at once, and it resolves them by launching it. The budget scales with mass because a
    heavy body absorbs the contact noise that throws a light one."""
    return 4 if mass < 3.0 else (8 if mass < 15.0 else 24)


def _colliders(mesh, name: str, out_dir: Path, mass: float, decompose: bool):
    """`name` must be unique across every object that shares `out_dir` — see `build_model`."""
    import trimesh

    written = []
    try:
        hull = mesh.convex_hull
        near_convex = (hull.volume <= 0
                       or abs(hull.volume - mesh.volume) / max(hull.volume, 1e-9) < CONVEX_TOL)
        diagonal = float(np.linalg.norm(mesh.extents))
    except Exception:                                        # noqa: BLE001
        near_convex, diagonal = True, 1.0

    if near_convex or not decompose or diagonal < MIN_DECOMP_DIAGONAL:
        path = out_dir / f"{name}_col0.obj"
        mesh.convex_hull.export(path)
        return [Collider(path.name, float(max(mesh.convex_hull.volume, 0.0)))]

    for i, (verts, faces) in enumerate(_decompose(mesh, _part_budget(mass))):
        piece = trimesh.Trimesh(np.asarray(verts), np.asarray(faces))
        if piece.volume <= 1e-9:
            continue
        path = out_dir / f"{name}_col{i}.obj"
        piece.export(path)
        written.append(Collider(path.name, float(piece.volume)))
    if written:
        return written
    path = out_dir / f"{name}_col0.obj"
    mesh.convex_hull.export(path)
    return [Collider(path.name, float(max(mesh.convex_hull.volume, 0.0)))]


# ------------------------------------------------------------------ the compiler


def build_model(glb: Path, out_dir: Path, *, category: str = "", decompose: bool = True) -> SimModel:
    """A built GLB -> the full physical model, with meshes written into `out_dir`."""
    import trimesh

    glb, out_dir = Path(glb), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gltf = _gltf_json(glb)
    nodes = gltf.get("nodes", [])
    by_name = {n.get("name", f"node{i}"): i for i, n in enumerate(nodes)}
    parent_of = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}

    scene = trimesh.load(str(glb))
    if not isinstance(scene, trimesh.Scene):
        scene = trimesh.Scene(scene)

    # Every geometry node, as a mesh in the Z-up frame the exporters want.
    meshes: dict[str, object] = {}
    for node in scene.graph.nodes_geometry:
        transform, geom = scene.graph[node]
        m = scene.geometry[geom].copy()
        m.apply_transform(np.asarray(transform, dtype=float))
        m.vertices = m.vertices @ YUP_TO_ZUP.T
        meshes[node] = trimesh.util.concatenate([meshes[node], m]) if node in meshes else m

    articulated = {n.get("name"): (n.get("extras") or {}) for n in nodes
                   if (n.get("extras") or {}).get("articulation_type")}

    name = glb.stem
    category = category or _category_of(name)
    notes: list[str] = []

    # TOTAL MASS FIRST, then split. Occupancy density is calibrated against whole objects — what a
    # chair weighs, what a table weighs — so it is applied to the whole object's bounding box and
    # the result is shared out by mesh volume. Deriving each link independently would make a
    # cabinet weigh whatever the sum of its panels happened to come to.
    all_mesh = trimesh.util.concatenate(list(meshes.values())) if meshes else None
    if all_mesh is None:
        raise ValueError(f"{glb} has no geometry")
    bbox = np.asarray(all_mesh.bounds, dtype=float)
    box_volume = float(np.prod(np.maximum(bbox[1] - bbox[0], 1e-3)))
    authored_total = _authored_total_mass(nodes)
    if authored_total is not None:
        total_mass, mass_source = authored_total, "authored"
    else:
        total_mass = box_volume * _lookup(OCCUPANCY_DENSITY, category, OCCUPANCY_DENSITY["default"])
        mass_source = "derived"

    # Group nodes into LINKS: one per articulated part, plus everything else welded into the base.
    link_nodes: dict[str, list[str]] = {"base_link": []}
    for node in meshes:
        owner = next((a for a in articulated if node == a or node.startswith(a + "_")), None)
        link_nodes.setdefault(owner or "base_link", []).append(node)
    if not link_nodes["base_link"]:
        notes.append("no static geometry — base_link is a massless frame")

    volumes = {}
    for link, group in link_nodes.items():
        merged = trimesh.util.concatenate([meshes[n] for n in group]) if group else None
        volumes[link] = float(max(merged.convex_hull.volume, 1e-6)) if merged is not None else 1e-6
    volume_total = sum(volumes.values())

    links, joints = [], []
    for link, group in link_nodes.items():
        if not group:
            links.append(Link(name=link, mass=MIN_LINK_MASS, com=[0, 0, 0],
                              inertia=[1e-4, 1e-4, 1e-4, 0, 0, 0], friction=DEFAULT_FRICTION,
                              restitution=DEFAULT_RESTITUTION, inertia_source="placeholder"))
            continue
        merged = trimesh.util.concatenate([meshes[n] for n in group])
        extras = articulated.get(link, {})
        node_json = nodes[by_name[link]] if link in by_name else {}
        material = _node_material(gltf, node_json) if node_json else ""
        if not material:
            material = _node_material(gltf, nodes[by_name[group[0]]]) if group[0] in by_name else ""

        # The link frame. An articulated part's frame sits ON ITS JOINT so the pivot is the origin
        # — that is what makes the exported origin exact rather than re-derived from geometry.
        if link == "base_link":
            frame = np.zeros(3)
        else:
            frame = _joint_origin(nodes, by_name, link, extras)
        local = merged.copy()
        local.apply_translation(-frame)

        mass = extras.get("mass")
        if mass is None:
            density = extras.get("density")
            mass = (float(density) * float(max(local.volume, 1e-6)) if density is not None
                    else total_mass * volumes[link] / volume_total)
            link_mass_source = "authored_density" if density is not None else "derived"
        else:
            mass, link_mass_source = float(mass), "authored"
        mass = max(float(mass), MIN_LINK_MASS)

        inertia, com, isource = _inertia_from_geometry(local, mass)
        # NAMED FOR THE OBJECT, NOT JUST THE LINK. A recipe-built object gets its own directory, but
        # a generative one is a bare glb at the top of `reconstruct/` and `sim/` is then SHARED by
        # every chair in the room. Both chairs have a `base_link`, so `base_link_col0.obj` was
        # written twice and the second one won: `ChairCluster0.physics.json` recorded eight
        # colliders whose volumes were its own and whose files held ChairCluster1's geometry. There
        # is nothing to see in the report — both objects pass their own gate — and the room gets one
        # chair collided as another.
        stem = f"{name}_{link}"
        visual = out_dir / f"{stem}_vis.obj"
        local.export(visual)
        colliders = _colliders(local, stem, out_dir, mass, decompose)

        links.append(Link(
            name=link, mass=round(mass, 4), com=[round(float(v), 6) for v in com],
            inertia=[round(float(inertia[0, 0]), 8), round(float(inertia[1, 1]), 8),
                     round(float(inertia[2, 2]), 8), round(float(inertia[0, 1]), 8),
                     round(float(inertia[0, 2]), 8), round(float(inertia[1, 2]), 8)],
            friction=float(extras.get("friction", _lookup(MATERIAL_FRICTION, material,
                                                          DEFAULT_FRICTION))),
            restitution=float(extras.get("restitution",
                                         _lookup(MATERIAL_RESTITUTION, material,
                                                 DEFAULT_RESTITUTION))),
            material=material, visual=visual.name, colliders=colliders,
            mass_source=link_mass_source, inertia_source=isource))

        if link == "base_link":
            continue

        jtype = str(extras["articulation_type"])
        jtype = "revolute" if jtype.startswith("rev") else ("prismatic" if jtype.startswith("pris")
                                                            else jtype)
        # NOT converted. The mesh is exported Y-up, but `set_articulation` writes the axis
        # RAW, in the Blender Z-up build frame it was authored in — so the two live in different
        # frames inside the same file. Transforming it here is the mistake that lays every door
        # hinge on its side: a vertical [0,0,-1] becomes a horizontal [0,1,0] and the door
        # cartwheels instead of swinging.
        axis = np.asarray(extras.get("articulation_axis", [0, 0, 1]), dtype=float)
        n = float(np.linalg.norm(axis)) or 1.0
        defaults = JOINT_DEFAULTS.get(jtype, JOINT_DEFAULTS["revolute"])
        parent_idx = parent_of.get(by_name.get(link, -1))
        parent_link = nodes[parent_idx].get("name") if parent_idx is not None else "base_link"
        if parent_link not in link_nodes:
            parent_link = "base_link"
        parent_frame = (np.zeros(3) if parent_link == "base_link"
                        else _joint_origin(nodes, by_name, parent_link,
                                           articulated.get(parent_link, {})))
        joints.append(Joint(
            name=f"{link}_joint", type=jtype, parent=parent_link, child=link,
            origin=[round(float(v), 6) for v in (frame - parent_frame)],
            axis=[round(float(v), 6) for v in (axis / n)],
            limit_lower=float(extras.get("limit_min", 0.0)),
            limit_upper=float(extras.get("limit_max", 0.0)),
            damping=float(extras.get("joint_damping", defaults["damping"])),
            friction=float(extras.get("joint_friction", defaults["friction"])),
            effort=float(extras.get("joint_effort", defaults["effort"])),
            velocity=float(extras.get("joint_velocity", defaults["velocity"])),
            origin_source="authored" if "joint_origin" in extras else "node_transform"))

    if mass_source == "derived":
        notes.append(f"total mass {total_mass:.2f} kg derived from {category or 'default'} "
                     f"occupancy density over a {box_volume:.3f} m^3 bounding box")
    model = SimModel(name=name, category=category, root="base_link", links=links, joints=joints,
                     bbox=[[round(float(v), 4) for v in bbox[0]],
                           [round(float(v), 4) for v in bbox[1]]],
                     total_mass=round(sum(k.mass for k in links), 4), notes=notes)
    return model


def _authored_total_mass(nodes) -> float | None:
    """An `object_mass` on any node states the whole object's mass and stops derivation."""
    for n in nodes:
        e = n.get("extras") or {}
        if e.get("object_mass") is not None:
            return float(e["object_mass"])
    return None


def _joint_origin(nodes, by_name, link: str, extras: dict) -> np.ndarray:
    """The pivot, in the object's own frame, Z-up.

    Preferred source is an authored `joint_origin`: the recipe placed the hinge and knows exactly
    where it is. Failing that it comes from the node's accumulated translation, which is the same
    number — the build puts a moving part's origin ON its hinge line so the baked clip rotates
    about it — but recorded implicitly rather than stated.
    """
    if extras.get("joint_origin") is not None:
        # Authored in the build frame, which is already Z-up — see the axis note in build_model().
        return np.asarray(extras["joint_origin"], dtype=float)
    idx = by_name.get(link)
    acc = np.zeros(3)
    parent_of = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
    seen = 0
    while idx is not None and seen < 64:
        acc = acc + np.asarray(nodes[idx].get("translation", [0.0, 0.0, 0.0]), dtype=float)
        idx = parent_of.get(idx)
        seen += 1
    return YUP_TO_ZUP @ acc


_CATEGORY_TOKENS = ("refrigerator", "dishwasher", "oven", "washer", "wardrobe", "cabinet",
                    "storage", "shelf", "television", "monitor", "table", "desk", "chair",
                    "stool", "sofa", "bed", "sink", "radiator", "door", "window")


def _category_of(name: str) -> str:
    low = name.lower()
    return next((t for t in _CATEGORY_TOKENS if t in low), "")


def write_model(model: SimModel, out_dir: Path) -> Path:
    path = Path(out_dir) / f"{model.name}.physics.json"
    path.write_text(json.dumps(model.to_json(), indent=2) + "\n")
    return path


def load_model(path: Path) -> SimModel:
    d = json.loads(Path(path).read_text())
    d.pop("schema_version", None)
    d["links"] = [Link(**{**row, "colliders": [Collider(**c) for c in row["colliders"]]})
                  for row in d["links"]]
    d["joints"] = [Joint(**j) for j in d["joints"]]
    return SimModel(**d)
