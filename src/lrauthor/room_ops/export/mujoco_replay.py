#!/usr/bin/env python3
"""mujoco_replay.py — simulate in MuJoCo, then render the result in Blender.

    python -m lrauthor.room_ops.export.mujoco_replay --room <room dir> \
        [--amplitude 1.0] [--frames 0,8,36] [--engine CYCLES]

MuJoCo's renderer is OpenGL: diffuse, specular, shadow maps, and nothing else. No bounced light, no
ambient occlusion, no normal or roughness maps. The authored room has all of those, and its Blender
render is the one that looks like a photograph — so the useful division is to let each side do what
it is actually good at:

    MuJoCo   decides where everything ENDS UP        (contacts, friction, hinges, falling)
    Blender  decides what that LOOKS like            (the authored materials, real light, Cycles)

This is the bridge. It steps the exported scene, records where every body finished, and writes a
Blender script that rebuilds the authored room from `Room.py` — full materials, real lamps — moves
each object to its simulated pose, and renders.

MOVEMENT IS APPLIED AS A DELTA, never as an absolute pose. The two worlds agree on the room but not
on how each object's origin was chosen: MuJoCo bodies sit at the bbox centre the exporter picked,
Blender objects keep whatever origin the builder gave them. Sending absolute poses would silently
teleport every object by its own origin offset. A delta — where it moved FROM, to where it moved TO
— needs no such agreement, and an object the simulation never touched receives exactly nothing.

ARTICULATED PARTS MOVE SEPARATELY FROM THEIR OBJECT. A door that swung 90 degrees is not a door
that moved; its leaf rotated about a hinge while the frame stayed in the wall. The exporter gives
each moving part its own body (`Door0_door_leaf`), and its delta is applied to the part's own
Blender objects — matched by the node names the room builder produced, hex suffix and all — so the
render shows a door standing open in its frame rather than a whole door lying at an angle.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def simulate(scene: Path, *, settle_s: float = 1.0, shake_s: float = 5.0, calm_s: float = 3.0,
             amplitude: float = 1.0, frequency: float = 1.5, ajar: float = 0.25) -> dict:
    """Run the shake and return each body's motion as a delta from where it started."""
    import mujoco

    from lrauthor.room_ops.export import mujoco_shake

    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    report = mujoco_shake.run(scene, settle_s=settle_s, shake_s=shake_s, calm_s=calm_s,
                              amplitude=amplitude, frequency=frequency, ajar=ajar)

    # Re-run in-process so the final state is in hand rather than only its summary. Cheap next to
    # the render that follows, and it keeps `mujoco_shake` a pure reporter.
    mujoco.mj_forward(model, data)
    if ajar:
        for j in range(model.njnt):
            if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE):
                lo, hi = model.jnt_range[j]
                data.qpos[model.jnt_qposadr[j]] = lo + ajar * (hi - lo)
        mujoco.mj_forward(model, data)
    start_pos, start_quat = data.xpos.copy(), data.xquat.copy()

    room_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "room")
    drives = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"drive_room_{a}")
              for a in "xyz"]
    drives = [d for d in drives if d >= 0]
    amp_m = amplitude / (2.0 * np.pi * frequency) ** 2
    dt = model.opt.timestep
    import math

    for name, seconds, shaking in (("settle", settle_s, False), ("shake", shake_s, True),
                                   ("calm", calm_s, False)):
        for step in range(int(seconds / dt)):
            if shaking and drives:
                t = step * dt
                for act, value in zip(drives, (
                        amp_m * math.sin(2 * math.pi * frequency * t),
                        amp_m * math.sin(2 * math.pi * frequency * 1.41 * t + 1.1),
                        0.9 * amp_m * math.sin(2 * math.pi * frequency * 1.93 * t + 2.3))):
                    data.ctrl[act] = value
            elif drives:
                for act in drives:
                    data.ctrl[act] = 0.0
            mujoco.mj_step(model, data)

    deltas = {}
    for b in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
        if not name or b == room_id:
            continue
        d_pos = (data.xpos[b] - start_pos[b]).tolist()
        d_quat = _quat_mul(data.xquat[b], _quat_conj(start_quat[b]))
        if np.linalg.norm(d_pos) < 1e-4 and abs(abs(d_quat[0]) - 1.0) < 1e-5:
            continue                                     # untouched: send nothing
        deltas[name] = {"pos": [round(v, 5) for v in d_pos],
                        "quat": [round(float(v), 6) for v in d_quat],
                        "pivot": data.xpos[b].tolist()}
    return {"report": report, "deltas": deltas}


BLENDER_APPLY = r'''
import bpy, json, math, os, sys
from mathutils import Vector, Quaternion

sys.path.insert(0, os.environ["RR_MODULE_DIR"])
from render_room_cameras import build_scene, render_room_from_cameras   # noqa: E402

state = json.load(open(os.environ["RR_STATE"]))
scene = build_scene(os.environ["RR_SCAN"], os.environ["RR_ASSETS"], os.environ["RR_OUT"])

# SHELL(x, y, z) is Blender's own frame here — `Room.py` builds directly in it — so a MuJoCo delta
# needs no change of basis, only the right objects to apply it to.
def targets(name):
    """Blender objects belonging to one MuJoCo body name."""
    exact = [o for o in bpy.data.objects if o.get("room_id") == name]
    if exact:
        return [o for h in exact for o in ([h] + list(h.children_recursive))]
    # an articulated part: `Door0_door_leaf` -> the door's `door_leaf*` nodes, hex suffix and all
    if "_" in name:
        handle, part = name.split("_", 1)
        owner = [o for o in bpy.data.objects if o.get("room_id") == handle]
        pool = [o for h in owner for o in h.children_recursive] or list(bpy.data.objects)
        return [o for o in pool if o.name == part or o.name.startswith(part + "_")]
    return []

applied, missed = 0, []
for name, d in state["deltas"].items():
    objs = targets(name)
    if not objs:
        missed.append(name); continue
    shift = Vector(d["pos"])
    rot = Quaternion(d["quat"])
    pivot = Vector(d["pivot"]) - shift          # where the body was BEFORE it moved
    for o in objs:
        if o.parent is not None and o.parent in objs:
            continue                             # a parent carries its children already
        o.matrix_world = (
            __import__("mathutils").Matrix.Translation(shift + pivot)
            @ rot.to_matrix().to_4x4()
            @ __import__("mathutils").Matrix.Translation(-pivot)
            @ o.matrix_world)
    applied += 1

print(f"[replay] applied {applied} body deltas, {len(missed)} unmatched: {missed[:8]}")
frames = [int(f) for f in os.environ.get("RR_FRAMES", "").split(",") if f.strip().isdigit()]
render_room_from_cameras(scene, os.environ["RR_OUT"], frames=frames or None,
                         res_div=int(os.environ.get("RR_RESDIV", "2")),
                         engine=os.environ.get("RR_ENGINE", "EEVEE"))
'''


def render(room: Path, state_path: Path, out: Path, *, scan: Path, frames: str = "",
           engine: str = "EEVEE", res_div: int = 2) -> int:
    """Rebuild the authored room in Blender, apply the simulated poses, render."""
    from lrauthor.room_ops.paths import find_blender
    from lrauthor.room_ops.rendering import room_render

    module_dir = Path(room_render.__file__).parent
    script = out / "_apply_sim_state.py"
    out.mkdir(parents=True, exist_ok=True)
    script.write_text(BLENDER_APPLY, encoding="utf-8")

    env = dict(os.environ)
    env.update({
        "RR_MODULE_DIR": str(module_dir), "RR_STATE": str(state_path),
        "RR_SCAN": str(scan), "RR_ASSETS": str(room.parent / "room_preview"),
        "RR_OUT": str(out), "RR_FRAMES": frames, "RR_ENGINE": engine,
        "RR_RESDIV": str(res_div), "SB_ROOM_PY": str(room / "Room.py"),
        "LITEREALITY_ROOM_DIR": str(room),
    })
    binary = find_blender()
    return subprocess.run([binary, "-b", "--python", str(script)], env=env).returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--room", required=True, type=Path, help="room dir holding Room.py")
    ap.add_argument("--scene", type=Path, default=None, help="MJCF (default <room>/../mujoco/scene.xml)")
    ap.add_argument("--scan", type=Path, required=True, help="capture dir (frames + poses)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--amplitude", type=float, default=1.0)
    ap.add_argument("--frequency", type=float, default=1.5)
    ap.add_argument("--ajar", type=float, default=0.25)
    ap.add_argument("--frames", default="", help="capture frames to render, e.g. 4,12,36")
    ap.add_argument("--engine", default="EEVEE", choices=["EEVEE", "CYCLES"])
    ap.add_argument("--res-div", type=int, default=2)
    a = ap.parse_args(argv)

    scene = a.scene or (a.room.parent / "mujoco" / "scene.xml")
    out = a.out or (a.room.parent / "sim_render")
    out.mkdir(parents=True, exist_ok=True)

    result = simulate(scene, amplitude=a.amplitude, frequency=a.frequency, ajar=a.ajar)
    state_path = out / "sim_state.json"
    state_path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"[replay] {len(result['deltas'])} bodies moved -> {state_path}")
    return render(a.room, state_path, out, scan=a.scan, frames=a.frames,
                  engine=a.engine, res_div=a.res_div)


if __name__ == "__main__":
    raise SystemExit(main())
