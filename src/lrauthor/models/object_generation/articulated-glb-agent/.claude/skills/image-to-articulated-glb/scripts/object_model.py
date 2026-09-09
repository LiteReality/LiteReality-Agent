"""Declarative articulation model — define an articulated object as DATA (parts + joints),
then compile it to the SAME blender_lib build + GLB from ONE source of truth.

Why this exists (Step 2): the ad-hoc recipe sets the articulation *extras* (`set_articulation`)
and the *animation* (`animate_*`) in two separate places that can silently disagree, and the
"is it built backwards?" question is only answerable after rendering. Here a `Joint` is defined
ONCE; `build()` emits consistent extras + animation from it, and `validate()` checks the
definition (part refs, joint sanity, and front-facing) BEFORE anything is built — so a
reversed design is caught at author time, not by eye.

Frame: the BUILD frame is Blender Z-up with the canonical FRONT = -Y (see category_specs
CANONICAL_ORIENTATION). `validate()` reasons in this frame (parts must open toward -Y);
`build()` exports the GLB Y-up via blender_lib.export_glb (front becomes +Z there — the frame
`probe_glb.py` checks). Importing this module does NOT import bpy; only `build()` does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Material:
    name: str
    color: tuple = (0.8, 0.8, 0.8)  # linear rgb 0-1
    roughness: float = 0.6
    metallic: float = 0.0
    # optional real PBR set (paths relative to the build's texture dir); color used if unset
    diffuse: str | None = None
    rough: str | None = None
    normal: str | None = None
    # Contact properties. A material is where these belong: chrome legs and a fabric seat grip
    # differently, and the material is the only place that difference is already recorded.
    # Leave them None and the sim compiler keys a default off the material NAME.
    friction: float | None = None      # sliding coefficient
    restitution: float | None = None   # bounce; used by Isaac/Bullet, ignored by MuJoCo


@dataclass
class Part:
    name: str
    shape: str  # "box" | "rounded_box" | "cylinder"
    size: tuple  # box/rounded_box: (dx, dy, dz) metres · cylinder: (radius, depth)
    at: tuple = (0.0, 0.0, 0.0)  # centre location, metres, build frame (X=width, Y=depth, Z=up)
    material: str = ""
    rot: tuple = (0.0, 0.0, 0.0)  # euler, for a cylinder whose axis isn't +Z
    # What it weighs. Give `mass` (kg) when the reference tells you, `density` (kg/m^3 of this
    # part's own volume) when the material is obvious but the weight is not, and neither when you
    # would be guessing — the compiler then shares the object's occupancy mass out by volume.
    mass: float | None = None
    density: float | None = None


@dataclass
class Joint:
    part: str  # the part that MOVES
    type: str  # "revolute" | "prismatic" | "fixed"
    axis: tuple = (0.0, 0.0, 1.0)  # hinge axis (revolute) / slide direction (prismatic), build frame
    origin: tuple = (0.0, 0.0, 0.0)  # hinge point (revolute only)
    limit_min: float = 0.0
    limit_max: float = 0.0  # radians (revolute) / metres (prismatic)
    # How it moves. None means "use the default for this joint type"; set them for a joint that is
    # genuinely unusual — a soft-close drawer damps hard, a fire door needs real effort to hold.
    damping: float | None = None
    friction: float | None = None
    effort: float | None = None     # N or Nm, for URDF consumers
    velocity: float | None = None   # m/s or rad/s


@dataclass
class ArticulatedModel:
    name: str
    parts: list = field(default_factory=list)
    materials: list = field(default_factory=list)
    joints: list = field(default_factory=list)


# --------------------------------- geometry (pure) ---------------------------------- #
def _part_aabb(p: Part):
    cx, cy, cz = p.at
    if p.shape == "cylinder":
        r, depth = p.size[0], p.size[1]
        hx = hy = r
        hz = depth / 2
    else:  # box / rounded_box
        hx, hy, hz = p.size[0] / 2, p.size[1] / 2, p.size[2] / 2
    return [[cx - hx, cy - hy, cz - hz], [cx + hx, cy + hy, cz + hz]]


def _rodrigues(p, origin, axis, angle):
    ax, ay, az = axis
    n = math.sqrt(ax * ax + ay * ay + az * az) or 1.0
    ax, ay, az = ax / n, ay / n, az / n
    vx, vy, vz = p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]
    c, s = math.cos(angle), math.sin(angle)
    dot = ax * vx + ay * vy + az * vz
    cx, cy, cz = ay * vz - az * vy, az * vx - ax * vz, ax * vy - ay * vx
    return [origin[i] + [vx, vy, vz][i] * c + [cx, cy, cz][i] * s + [ax, ay, az][i] * dot * (1 - c)
            for i in range(3)]


def _open_travel(p: Part, j: Joint):
    """Centroid displacement from closed→open, in the build frame (front = -Y)."""
    c0 = [(p.at[i]) for i in range(3)]
    if j.type == "prismatic":
        n = math.sqrt(sum(a * a for a in j.axis)) or 1.0
        return [j.axis[i] / n * j.limit_max for i in range(3)]
    if j.type == "revolute":
        c1 = _rodrigues(c0, j.origin, j.axis, j.limit_max)
        return [c1[i] - c0[i] for i in range(3)]
    return [0.0, 0.0, 0.0]


# --------------------------------- validate (pure) ---------------------------------- #
FRONT_Y_TOL = 0.02  # metres of centroid travel toward +Y (into body) before we flag it


def validate(model: ArticulatedModel) -> dict:
    """Schema-level QC — same report shape as probe_glb.py, but on the DEFINITION (no Blender)."""
    parts = {p.name: p for p in model.parts}
    mats = {m.name for m in model.materials}
    violations = []

    def add(sev, check, part, detail):
        violations.append({"severity": sev, "check": check, "part": part, "detail": detail})

    if len(parts) != len(model.parts):
        add("hard", "duplicate_part", "-", "two parts share a name")
    for p in model.parts:
        if p.material and p.material not in mats:
            add("hard", "material_ref", p.name, f"references unknown material {p.material!r}")
        if p.shape not in ("box", "rounded_box", "cylinder"):
            add("hard", "shape", p.name, f"unknown shape {p.shape!r}")
        if p.mass is not None and p.density is not None:
            add("hard", "physics", p.name, "sets both mass and density — they will disagree")
        if p.mass is not None and p.mass <= 0:
            add("hard", "physics", p.name, f"mass must be positive, got {p.mass}")
        if p.density is not None and p.density <= 0:
            add("hard", "physics", p.name, f"density must be positive, got {p.density}")

    for m in model.materials:
        if m.friction is not None and not 0.0 <= m.friction <= 2.0:
            add("hard", "physics", m.name, f"friction {m.friction} is outside 0..2")
        if m.restitution is not None and not 0.0 <= m.restitution <= 1.0:
            add("hard", "physics", m.name, f"restitution {m.restitution} is outside 0..1")

    for j in model.joints:
        if j.part not in parts:
            add("hard", "joint_ref", j.part, "joint targets a part that does not exist")
            continue
        if j.type not in ("revolute", "prismatic", "fixed"):
            add("hard", "joint_type", j.part, f"unknown joint type {j.type!r}")
            continue
        if j.type == "fixed":
            continue
        if math.sqrt(sum(a * a for a in j.axis)) < 1e-6:
            add("hard", "articulation_sanity", j.part, "joint axis is zero-length")
        if abs(j.limit_max - j.limit_min) < 1e-6:
            add("hard", "articulation_sanity", j.part,
                f"degenerate limits (min={j.limit_min}, max={j.limit_max}) — cannot move")
        if j.type == "revolute" and j.origin == (0.0, 0.0, 0.0):
            add("soft", "joint_origin", j.part,
                "revolute joint leaves origin at (0,0,0) — state the hinge point, or every "
                "downstream consumer has to re-derive it from the mesh")
        travel = _open_travel(parts[j.part], j)
        if travel[1] > FRONT_Y_TOL:  # build frame: front is -Y
            add("hard", "orientation", j.part,
                f"opens toward +Y (into the body) by {travel[1]:.3f} m — front must face -Y "
                "(the object is defined 180deg about the up-axis)")

    # SOFT: an isolated part (touches nothing)
    boxes = {p.name: _part_aabb(p) for p in model.parts}
    for name, a in boxes.items():
        if len(boxes) > 1 and not any(
            other != name and all(a[1][i] >= b[0][i] - 1e-3 and b[1][i] >= a[0][i] - 1e-3
                                   for i in range(3))
            for other, b in boxes.items()
        ):
            add("soft", "isolated_part", name, "part touches no other part — likely detached")

    return {"model": model.name, "pass": not any(v["severity"] == "hard" for v in violations),
            "n_parts": len(model.parts), "n_joints": len(model.joints), "violations": violations}


# --------------------------------- build (needs bpy) -------------------------------- #
def _principal(axis):
    i = max(range(3), key=lambda k: abs(axis[k]))
    return i, (1 if axis[i] >= 0 else -1)


def build(model: ArticulatedModel, out_glb: str, texdir: str | None = None):
    """Compile the model to a GLB via blender_lib. ONE joint → consistent extras + animation.
    Raises ValueError if validate() finds a HARD issue (never build a broken/backwards model)."""
    report = validate(model)
    hard = [v for v in report["violations"] if v["severity"] == "hard"]
    if hard:
        raise ValueError("model failed validate(): "
                         + "; ".join(f"{v['check']}/{v['part']}: {v['detail']}" for v in hard))

    import bpy  # noqa: F401  (Blender only)

    import blender_lib as bl

    bl.reset_scene()
    mats = {}
    for m in model.materials:
        if m.diffuse:
            import os
            td = texdir or "."
            mats[m.name] = bl.make_pbr_material(
                m.name, os.path.join(td, m.diffuse),
                os.path.join(td, m.rough) if m.rough else None,
                os.path.join(td, m.normal) if m.normal else None,
            )
        else:
            mats[m.name] = bl.make_plain_material(m.name, m.color, m.metallic, m.roughness)
    contact = {m.name: (m.friction, m.restitution) for m in model.materials}

    obs = {}
    for p in model.parts:
        mat = mats.get(p.material)
        if p.shape == "cylinder":
            obs[p.name] = bl.add_cylinder(p.name, p.size[0], p.size[1], p.at, mat, rot=p.rot)
        elif p.shape == "rounded_box":
            obs[p.name] = bl.add_rounded_box(p.name, p.size, p.at, mat)
        else:
            obs[p.name] = bl.add_box(p.name, p.size, p.at, mat)
        friction, restitution = contact.get(p.material, (None, None))
        bl.set_physics(obs[p.name], mass=p.mass, density=p.density,
                       friction=friction, restitution=restitution)

    scene = bpy.context.scene
    revolute, prismatic = [], []
    for j in model.joints:
        if j.type not in ("revolute", "prismatic"):
            continue
        ob = obs[j.part]
        if j.type == "revolute":  # put the origin on the hinge line so the swing pivots correctly
            ob = bl.join_parts([ob], j.part, j.origin)
            obs[j.part] = ob
        # The pivot is STATED, not left to be recovered from the node transform downstream.
        bl.set_articulation(ob, j.type, tuple(j.axis), j.limit_min, j.limit_max,
                            origin=tuple(j.origin) if j.type == "revolute" else None,
                            damping=j.damping, friction=j.friction,
                            effort=j.effort, velocity=j.velocity)
        ai, sign = _principal(j.axis)
        if j.type == "revolute":
            revolute.append((ob, sign * j.limit_max, ai))
        else:
            prismatic.append((ob, j.limit_max, ai, sign))

    # animate_* group by axis_index; here we drive one part at a time to honour each joint's axis
    for ob, ang, ai in revolute:
        bl.animate_revolute(scene, [(ob, ang)], axis_index=ai)
    for ob, ext, ai, sign in prismatic:
        bl.animate_prismatic(scene, [(ob, ext)], axis_index=ai, direction=sign)

    bl.export_glb(out_glb)
    return report
