"""The physics gate for ONE object, run before it is ever placed in a room.

Every existing gate on a generated object is geometric or visual — `probe_glb.py` asks whether the
parts are connected, `validate_glb.py` whether the file has animations, the VLM whether it looks
like the photograph. None of them touches a solver. The first thing that does is the room-level
shake, which needs a whole assembled scene and reports "something moved 175 m" without saying
which asset was at fault.

So: three scenarios on the object alone, in seconds, at the point where the recipe can still be
fixed.

``drop``     let it go just above the floor and watch. Catches parts that were never attached,
             colliders that start interpenetrating, and inertias a solver refuses to integrate.
``release``  hold the body still, start every joint mid-travel and let go. A joint whose axis or
             origin is wrong drives its part into the carcass, and the part leaves at speed.
``tilt``     lean gravity over until it slides. The only test that checks the friction the asset
             CLAIMS is the friction it behaves with.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DROP_HEIGHT = 0.05        # m — enough to prove it falls, small enough not to mask a bad inertia
DROP_SECONDS = 3.0
REST_SPEED = 0.05         # m/s below which we call it at rest
MAX_OVERLAP_AT_REST = 0.005   # m of penetration that is solver slack rather than a modelling bug
MAX_DRIFT = 0.02          # m a pair of bodies may separate by before the object is coming apart
TILT_MAX_DEG = 60.0
TILT_STEP_DEG = 2.0
TILT_SLIDE = 0.02         # m of travel that counts as sliding


@dataclass
class Report:
    name: str
    scenarios: dict = field(default_factory=dict)
    findings: list = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(f["severity"] == "hard" for f in self.findings)

    def add(self, severity, check, detail):
        self.findings.append({"severity": severity, "check": check, "detail": detail})

    def to_json(self) -> dict:
        return {"object": self.name, "pass": self.passed,
                "scenarios": self.scenarios, "findings": self.findings}


def _load(xml: Path):
    import mujoco
    return mujoco.MjModel.from_xml_path(str(xml))


def _body_origins(model, data) -> np.ndarray:
    return np.asarray(data.xpos[1:], dtype=float).copy()      # body 0 is the world


def _max_overlap(data) -> float:
    """Deepest contact penetration right now. MuJoCo reports a NEGATIVE distance for penetration."""
    if data.ncon == 0:
        return 0.0
    return float(max(0.0, -min(data.contact[i].dist for i in range(data.ncon))))


def _pairwise_spread(origins: np.ndarray) -> np.ndarray:
    d = origins[:, None, :] - origins[None, :, :]
    return np.linalg.norm(d, axis=-1)


# ------------------------------------------------------------------ scenarios


def drop_test(model_xml: Path, report: Report) -> dict:
    import mujoco

    model = _load(model_xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    start = _body_origins(model, data)
    start_low = float(start[:, 2].min())
    spread0 = _pairwise_spread(start)
    peak_overlap = _max_overlap(data)

    steps = int(DROP_SECONDS / model.opt.timestep)
    for _ in range(steps):
        mujoco.mj_step(model, data)
        peak_overlap = max(peak_overlap, _max_overlap(data))

    end = _body_origins(model, data)
    end_low = float(end[:, 2].min())
    rest_overlap = _max_overlap(data)
    speed = float(np.abs(data.cvel[1:, 3:]).max()) if model.nbody > 1 else 0.0
    drift = float(np.abs(_pairwise_spread(end) - spread0).max()) if len(end) > 1 else 0.0

    out = {"initial_low_z": round(start_low, 4), "final_low_z": round(end_low, 4),
           "fell_m": round(start_low - end_low, 4), "contacts_at_rest": int(data.ncon),
           "max_overlap_on_impact_m": round(peak_overlap, 5),
           "max_overlap_at_rest_m": round(rest_overlap, 5),
           "max_body_drift_m": round(drift, 4), "final_speed_m_s": round(speed, 4),
           "at_rest": speed < REST_SPEED, "intact": drift < MAX_DRIFT}

    if not math.isfinite(speed) or speed > 50.0:
        report.add("hard", "drop_unstable",
                   f"body speed {speed:.1f} m/s after {DROP_SECONDS}s — the solver is ejecting it")
    elif not out["at_rest"]:
        report.add("soft", "drop_not_at_rest",
                   f"still moving at {speed:.3f} m/s after {DROP_SECONDS}s")
    if not out["intact"]:
        report.add("hard", "drop_came_apart",
                   f"two bodies moved {drift:.3f} m relative to each other — a part is unattached")
    if rest_overlap > MAX_OVERLAP_AT_REST:
        report.add("hard", "resting_penetration",
                   f"{rest_overlap * 1000:.1f} mm of penetration at rest — colliders overlap")
    if data.ncon == 0:
        report.add("soft", "no_contacts", "nothing is touching the floor at the end of the drop")
    return out


def release_test(model_xml: Path, report: Report) -> dict:
    """Joints at mid-travel, body held. A wrong axis or origin shows up as penetration or speed.

    ONE JOINT AT A TIME. Opening everything at once poses a dishwasher with its door half down and
    its rack half out — a pose the real mechanism cannot reach — and the resulting clash says
    nothing about either joint. Isolated, each joint answers only for itself.

    A part that penetrates its OWN carcass is a hard fault: the axis or the origin is wrong and
    nothing about the object will move correctly. A part that penetrates a SIBLING is a different
    animal — a rack sliding through a closed door is what the real appliance would do too if the
    door were shut, and neither URDF nor MJCF can express "this joint requires that one open".
    That is worth reporting, and it is not a reason to reject the asset.
    """
    import mujoco

    model = _load(model_xml)
    data = mujoco.MjData(model)
    named = [(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j), j)
             for j in range(model.njnt)]
    # int() on both sides: jnt_type is a numpy int32 and mjtJoint is not an IntEnum, so the
    # obvious membership test is quietly False for every joint and the scenario reports "none".
    hinge_slide = {int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)}
    movable = [(n, j) for n, j in named if int(model.jnt_type[j]) in hinge_slide]
    if not movable:
        return {"joints": 0, "note": "no articulated joints"}

    def ancestors(body: int) -> set:
        out, cur = set(), int(model.body_parentid[body])
        while cur > 0:
            out.add(cur)
            cur = int(model.body_parentid[cur])
        return out

    def name_of(body: int) -> str:
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or f"body{body}"

    per_joint = {}
    for jname, j in movable:
        mujoco.mj_resetData(model, data)
        lo, hi = model.jnt_range[j]
        data.qpos[model.jnt_qposadr[j]] = (lo + hi) / 2.0
        mujoco.mj_forward(model, data)

        child = int(model.jnt_bodyid[j])
        chain = ancestors(child)
        own, sibling = 0.0, (0.0, "")
        for i in range(data.ncon):
            c = data.contact[i]
            b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
            if child not in (b1, b2):
                continue
            other = b2 if b1 == child else b1
            depth = max(0.0, -float(c.dist))
            if other in chain:
                own = max(own, depth)
            elif depth > sibling[0]:
                sibling = (depth, name_of(other))
        per_joint[jname] = {"overlap_with_carcass_m": round(own, 5),
                            "overlap_with_sibling_m": round(sibling[0], 5),
                            "sibling": sibling[1]}
        if own > 0.01:
            report.add("hard", "joint_self_collision",
                       f"{jname}: {own * 1000:.1f} mm into its own carcass at mid-travel — axis "
                       "or origin is wrong")
        elif sibling[0] > 0.01:
            report.add("soft", "sibling_sweep",
                       f"{jname}: {sibling[0] * 1000:.1f} mm into {sibling[1]} at mid-travel — "
                       "these two parts cannot both be open; no joint format records that")

    # ...and one dynamic pass with everything released, to catch a joint the solver ejects.
    mujoco.mj_resetData(model, data)
    for _, j in movable:
        lo, hi = model.jnt_range[j]
        data.qpos[model.jnt_qposadr[j]] = (lo + hi) / 2.0
    mujoco.mj_forward(model, data)
    peak = {n: 0.0 for n, _ in movable}
    for _ in range(int(1.5 / model.opt.timestep)):
        mujoco.mj_step(model, data)
        for n, j in movable:
            peak[n] = max(peak[n], abs(float(data.qvel[model.jnt_dofadr[j]])))

    worst = max(peak.values())
    if worst > 25.0:
        report.add("hard", "joint_ejected",
                   f"joint speed reached {worst:.1f} — the solver is throwing a part")
    return {"joints": len(movable), "max_joint_speed": round(worst, 4),
            "per_joint": {k: {**per_joint[k], "peak_speed": round(peak[k], 4)}
                          for k in sorted(per_joint)}}


def tilt_test(model_xml_for_angle, report: Report, claimed_friction: float) -> dict:
    """Lean gravity over in steps until it slides, and compare with what the asset claims.

    `model_xml_for_angle` is a callable so each angle gets a clean model: MuJoCo bakes gravity at
    compile time only if you ask it to, but re-running from rest is what makes the result a
    property of the asset rather than of the previous angle's momentum.
    """
    import mujoco

    slid_at = None
    for deg in np.arange(TILT_STEP_DEG, TILT_MAX_DEG + 1e-9, TILT_STEP_DEG):
        model = _load(model_xml_for_angle)
        data = mujoco.MjData(model)
        rad = math.radians(float(deg))
        model.opt.gravity[:] = [9.81 * math.sin(rad), 0.0, -9.81 * math.cos(rad)]
        mujoco.mj_forward(model, data)
        x0 = float(data.xpos[1][0])
        for _ in range(int(1.0 / model.opt.timestep)):
            mujoco.mj_step(model, data)
        if abs(float(data.xpos[1][0]) - x0) > TILT_SLIDE:
            slid_at = float(deg)
            break

    expected = math.degrees(math.atan(claimed_friction))
    out = {"claimed_friction": round(claimed_friction, 3),
           "expected_slide_deg": round(expected, 1),
           "measured_slide_deg": slid_at,
           "note": "MuJoCo has one sliding friction and no restitution; restitution is exported "
                   "for other engines and not tested here"}
    if slid_at is None:
        report.add("soft", "never_slid",
                   f"did not slide by {TILT_MAX_DEG:.0f}deg — expected around {expected:.0f}deg")
    elif abs(slid_at - expected) > 12.0:
        report.add("soft", "friction_mismatch",
                   f"slid at {slid_at:.0f}deg, friction {claimed_friction:.2f} predicts "
                   f"{expected:.0f}deg")
    return out


# ------------------------------------------------------------------ entry point


def check(model, out_dir: Path, *, scenarios=("drop", "release", "tilt")) -> Report:
    """Run the gate on a compiled SimModel. Writes the scenario MJCFs next to the meshes."""
    from .mjcf import to_mjcf

    out_dir = Path(out_dir)
    report = Report(name=model.name)

    if "drop" in scenarios:
        xml = to_mjcf(model, out_dir, drop_height=DROP_HEIGHT, suffix="_drop")
        report.scenarios["drop"] = drop_test(xml, report)
    if "release" in scenarios:
        held = to_mjcf(model, out_dir, drop_height=0.0, free=False, suffix="_release")
        report.scenarios["release"] = release_test(held, report)
    if "tilt" in scenarios:
        root = next((l for l in model.links if l.name == model.root), model.links[0])
        xml = to_mjcf(model, out_dir, drop_height=0.0, suffix="_tilt")
        report.scenarios["tilt"] = tilt_test(xml, report, root.friction)
    return report


def write_report(report: Report, out_dir: Path) -> Path:
    path = Path(out_dir) / f"{report.name}.sim_check.json"
    path.write_text(json.dumps(report.to_json(), indent=2) + "\n")
    return path
