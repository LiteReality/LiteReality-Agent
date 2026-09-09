#!/usr/bin/env python3
"""stability.py — does this scene blow up? The question a drift number does not answer.

    <python> stability.py <scene.xml> [<scene.xml> ...] [--seconds 30] [--resets 8]

"Total motion over 5 s" says the room settles. It says nothing about whether the SOLVER is healthy,
and a scene can post a small drift while quietly doing something that will destroy a training run
the first time an episode resets somewhere slightly different. So this asks the questions a
simulator actually cares about at initialisation:

``warnings``     MuJoCo's own counters. `BADQACC`, `BADQPOS`, `BADQVEL` are the engine saying it
                 produced a number it does not believe; `CONTACTFULL` and `CNSTRFULL` mean it
                 silently DROPPED contacts, so the scene you are stepping is not the scene you
                 wrote. Any of them non-zero is a failure however good the room looks.
``qacc``         peak |acceleration| over the run. A room at rest sits near g. Anything in the
                 thousands is a contact the solver cannot satisfy, and it is the leading indicator
                 of an ejection — it spikes before anything visibly moves.
``penetration``  deepest overlap at t=0 and the worst reached while stepping. Overlap at t=0 is
                 stored energy: the solver reads it as a compressed spring.
``escape``       anything that leaves the room's own bounding box. This is the "blow up" everyone
                 means — a body reaching 175 m — and it is worth naming separately because it does
                 not always show up in a mean.
``resets``       the same scene re-initialised with each free body nudged, the way an episode reset
                 would. A scene that is stable only from its one authored pose is not sim-ready;
                 an RL run will find the poses it is not stable from within an hour.

Exit status is 0 only if every scene passes every check.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

# A body at rest experiences g. Contact resolution spikes above that legitimately, but a scene
# that is quietly fighting itself sits orders of magnitude higher — the threshold is deliberately
# generous so that tripping it means something.
QACC_LIMIT = 5_000.0
PENETRATION_LIMIT = 0.010        # m — deeper than this at t=0 is stored energy, not solver slack
ESCAPE_MARGIN = 1.0              # m outside the room's own bounds counts as having left it
# Read off the enum rather than written out: the set differs between MuJoCo versions (3.13 has no
# VGEOMFULL), and a hardcoded list either misses a warning or indexes past the end of the array.
WARNINGS = [name for _i, name in sorted(
    (int(getattr(mujoco.mjtWarning, n)), n.removeprefix("mjWARN_"))
    for n in dir(mujoco.mjtWarning) if n.startswith("mjWARN_"))]


def _free_bodies(model):
    out = {}
    for b in range(1, model.nbody):
        for j in range(int(model.body_jntadr[b]), int(model.body_jntadr[b]) + int(model.body_jntnum[b])):
            if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
                out[b] = int(model.jnt_qposadr[j])
    return out


def _pin_room(model, data):
    for name in ("room_x", "room_y", "room_z", "room_yaw"):
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j >= 0:
            data.qpos[model.jnt_qposadr[j]] = 0.0
            data.qvel[model.jnt_dofadr[j]] = 0.0


def _deepest(data) -> float:
    return min((float(data.contact[c].dist) for c in range(data.ncon)), default=0.0)


def _warnings(model, data) -> dict:
    return {WARNINGS[i]: int(data.warning[i].number)
            for i in range(min(len(WARNINGS), len(data.warning)))
            if data.warning[i].number}


def _run(model, data, steps: int, bounds) -> dict:
    peak_qacc, worst_pen, escaped = 0.0, 0.0, []
    lo, hi = bounds
    for _ in range(steps):
        _pin_room(model, data)
        mujoco.mj_step(model, data)
        peak_qacc = max(peak_qacc, float(np.abs(data.qacc).max()))
        worst_pen = min(worst_pen, _deepest(data))
        if not np.isfinite(data.qpos).all():
            escaped.append("NON-FINITE qpos")
            break
    for b in range(1, model.nbody):
        p = data.xpos[b]
        if np.any(p < lo - ESCAPE_MARGIN) or np.any(p > hi + ESCAPE_MARGIN):
            escaped.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}")
    # Read BEFORE the next `mj_resetData`, which clears the counters — otherwise only the last
    # trial's warnings survive and a scene that blew up on trial 2 reports clean.
    return {"peak_qacc": peak_qacc, "worst_penetration_mm": worst_pen * 1000, "escaped": escaped,
            "warnings": _warnings(model, data)}


def check(scene: Path, *, seconds: float = 30.0, resets: int = 8, jitter: float = 0.02) -> dict:
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # The room's own extent, so "escaped" means left the building rather than moved a lot.
    finite = data.xpos[np.isfinite(data.xpos).all(axis=1)]
    bounds = (finite.min(axis=0), finite.max(axis=0))

    row: dict = {
        "scene": str(scene), "bodies": int(model.nbody), "geoms": int(model.ngeom),
        "contacts_at_rest": int(data.ncon),
        "initial_penetration_mm": round(_deepest(data) * 1000, 2),
        "initial_qacc": round(float(np.abs(data.qacc).max()), 1),
        "nan_at_init": not (np.isfinite(data.qpos).all() and np.isfinite(data.qacc).all()),
    }

    steps = int(seconds / model.opt.timestep)
    free = _free_bodies(model)
    runs = []
    rng = np.random.default_rng(0)
    for trial in range(max(1, resets)):
        mujoco.mj_resetData(model, data)
        if trial:                       # trial 0 is the authored pose; the rest are episode resets
            for _b, adr in free.items():
                data.qpos[adr:adr + 3] += rng.uniform(-jitter, jitter, 3)
        mujoco.mj_forward(model, data)
        runs.append(_run(model, data, steps, bounds))

    row["peak_qacc"] = round(max(r["peak_qacc"] for r in runs), 1)
    row["worst_penetration_mm"] = round(min(r["worst_penetration_mm"] for r in runs), 2)
    row["escaped"] = sorted({n for r in runs for n in r["escaped"]})
    merged: dict = {}
    for r in runs:
        for name, count in r["warnings"].items():
            merged[name] = merged.get(name, 0) + count
    row["warnings"] = merged
    row["resets"] = len(runs)

    row["failures"] = failures = []
    if row["nan_at_init"]:
        failures.append("non-finite state at initialisation")
    if row["warnings"]:
        failures.append(f"solver warnings: {', '.join(row['warnings'])}")
    if row["peak_qacc"] > QACC_LIMIT:
        failures.append(f"peak |qacc| {row['peak_qacc']:.0f} over {QACC_LIMIT:.0f}")
    if -row["initial_penetration_mm"] / 1000 > PENETRATION_LIMIT:
        failures.append(f"starts {-row['initial_penetration_mm']:.0f} mm interpenetrating")
    if row["escaped"]:
        failures.append(f"left the room: {', '.join(row['escaped'][:4])}")
    row["pass"] = not failures
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("scenes", nargs="+", type=Path)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--resets", type=int, default=8)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)

    rows = []
    for scene in a.scenes:
        row = check(scene, seconds=a.seconds, resets=a.resets)
        rows.append(row)
        mark = "PASS" if row["pass"] else "FAIL"
        print(f"{mark}  {scene.parent.parent.name if scene.parent.name.startswith('mujoco') else scene}")
        print(f"      init: {row['contacts_at_rest']} contacts, deepest "
              f"{row['initial_penetration_mm']:.1f} mm, |qacc| {row['initial_qacc']:.1f}")
        print(f"      {a.seconds:.0f}s x{row['resets']}: peak |qacc| {row['peak_qacc']:.0f}, "
              f"worst penetration {row['worst_penetration_mm']:.1f} mm, "
              f"warnings {row['warnings'] or 'none'}, escaped {row['escaped'] or 'none'}")
        for f in row["failures"]:
            print(f"      ! {f}")
    if a.json:
        a.json.write_text(json.dumps(rows, indent=2))
    return 0 if all(r["pass"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
