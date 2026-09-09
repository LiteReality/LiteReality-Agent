#!/usr/bin/env python3
"""survey.py — render every exported room and measure whether it holds still.

    <python> survey.py <scan> [<scan> ...] --root run-simready --out <dir>

One row per room: what the scene is made of, where its physics came from, and what happens over
5 s of gravity with nothing touching it. The last of those is the only number that says whether a
room is usable — a scene that renders beautifully and then throws its contents across the floor is
a picture, not an environment.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import render_scene  # noqa: E402  — sibling


def _pin(model, data):
    for name in ("room_x", "room_y", "room_z", "room_yaw"):
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j >= 0:
            data.qpos[model.jnt_qposadr[j]] = 0.0
            data.qvel[model.jnt_dofadr[j]] = 0.0


def measure(scene: Path, seconds: float = 5.0) -> dict:
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    name = lambda t, i: mujoco.mj_id2name(model, t, i) or "?"       # noqa: E731

    free = [b for b in range(1, model.nbody)
            if any(model.jnt_type[j] == 0
                   for j in range(model.body_jntadr[b], model.body_jntadr[b] + model.body_jntnum[b]))]
    hinges = [j for j in range(model.njnt)
              if model.jnt_type[j] in (2, 3)
              and not (name(mujoco.mjtObj.mjOBJ_JOINT, j) or "").startswith("room")]
    deep = sum(1 for c in range(data.ncon) if data.contact[c].dist < -0.005)

    start = {b: data.xpos[b].copy() for b in free}
    for _ in range(int(seconds / model.opt.timestep)):
        _pin(model, data)
        mujoco.mj_step(model, data)
    moved = sorted(((float(np.linalg.norm(data.xpos[b] - start[b])),
                     name(mujoco.mjtObj.mjOBJ_BODY, b)) for b in free), reverse=True)

    return {
        "bodies": int(model.nbody), "geoms": int(model.ngeom),
        "free": len(free), "articulated": len(hinges),
        "joints": [{"name": name(mujoco.mjtObj.mjOBJ_JOINT, j),
                    "type": "slide" if model.jnt_type[j] == 2 else "hinge",
                    "range_deg": [round(float(np.degrees(v)), 1) for v in model.jnt_range[j]]}
                   for j in hinges],
        "mass_kg": round(float(sum(model.body_mass)) - 50000.0, 1),   # less the room's own ballast
        "deep_overlaps": deep,
        "moved_over_10mm": sum(1 for v, _ in moved if v > 0.010),
        "total_motion_mm": round(sum(v for v, _ in moved) * 1000, 1),
        "worst": [{"body": n, "mm": round(v * 1000, 1)} for v, n in moved[:5]],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("scans", nargs="+")
    ap.add_argument("--root", default="run-simready")
    ap.add_argument("--scene-dir", default="realism_authoring/mujoco",
                    help="where the scene.xml sits under <root>/<scan> "
                         "(use mujoco_seed for the un-authored room)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--settle", type=float, default=2.0)
    a = ap.parse_args(argv)

    a.out.mkdir(parents=True, exist_ok=True)
    rows = {}
    for scan in a.scans:
        scene = Path(a.root) / scan / a.scene_dir / "scene.xml"
        if not scene.is_file():
            print(f"  ! {scan}: no scene.xml")
            continue
        print(f"── {scan} ──")
        row = measure(scene)
        report = scene.parent / "export_report.json"
        if report.is_file():
            row["export"] = json.loads(report.read_text())
        render_scene.render(scene, a.out / scan, settle=a.settle)
        row["views"] = [p.name for p in sorted((a.out / scan).glob("*.png"))]
        rows[scan] = row
        print(f"   {row['bodies']} bodies, {row['free']} free, {row['articulated']} articulated, "
              f"{row['mass_kg']} kg | 5 s gravity: {row['moved_over_10mm']} moved, "
              f"{row['total_motion_mm']:.0f} mm")
    (a.out / "survey.json").write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
