"""simulate — the finished room as a MuJoCo scene a robot can be trained in.

`publish` ends with a room that renders: one glb, every object baked into it, no notion of what is
furniture and what is wall, no mass and no joints. That is a picture of a room. Training anything
in it needs the opposite — a scene of BODIES, each with its own collision geometry, its own mass
and inertia, and a joint (or deliberately none) saying how it may move.

This stage produces that, and the point of it running LAST is that by now every piece of the answer
already exists and has been checked:

    Room.py `SHELL`              the walls, openings and heights, metric and exact
    room_layout.json             every object's pose, category, and what holds it up
    Room.glb                     the appearance, per named handle
    Objects/<name>/sim/          the object's OWN physics — mass, inertia, friction, joints and
                                 convex colliders, compiled at reconstruct time and gated by a
                                 solver before it was ever placed in a room

Nothing here re-derives what those files already say. That is the whole difference between a scene
that happens to load in MuJoCo and one that is sim-ready: a room where the mass of a chair, the
friction of a worktop and the pivot of a door came out of the asset that was built, rather than out
of a category table consulted at export time.

The shake is optional and off by default. It is a MEASUREMENT, not a build step — it steps the
scene under a rising ground acceleration and reports what moved — and it costs minutes. A room that
exports is usable; a room that also survives a shake is one whose supports are true.
"""

from __future__ import annotations

import json

from litereality_agent.pipeline.context import RunContext
from litereality_agent.pipeline.result import StageResult, StageStatus
from litereality_agent.pipeline.support import run_module


def scene_dir(context: RunContext):
    return context.authoring_root / "mujoco"


def complete(context: RunContext) -> bool:
    return (scene_dir(context) / "scene.xml").is_file()


def _report(context: RunContext) -> dict:
    path = scene_dir(context) / "export_report.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:                                        # noqa: BLE001 — no report, no summary
        return {}


def run(context: RunContext, options: dict) -> StageResult:
    room = context.authored_room
    if not (room / "Room.py").is_file():
        return StageResult("simulate", StageStatus.FAILED,
                           error=f"no authored room at {room} — run `author` first")
    if not (context.preview_dir / "Room.glb").is_file():
        return StageResult("simulate", StageStatus.FAILED,
                           error=f"no built room at {context.preview_dir / 'Room.glb'} — "
                                 f"run `publish` first")

    argv = ["--room", str(room), "--out", str(scene_dir(context))]
    # Decomposition is only reached for objects with NO physics sidecar, and it is the slowest
    # thing in the export by two orders of magnitude. Turning it off makes those objects collide as
    # their convex hulls — a table becomes a solid block — so it stays on and is opt-out.
    if options.get("no_decompose"):
        argv.append("--no-decompose")
    if options.get("reuse_meshes"):
        argv.append("--reuse-meshes")
    rc, log = run_module(context, "litereality_agent.room_ops.export.mujoco_scene", argv,
                         log_name="simulate_export")
    if rc or not complete(context):
        return StageResult("simulate", StageStatus.FAILED,
                           error=f"MuJoCo export failed; see {log}")

    warnings: list[str] = []
    report = _report(context)
    # An object that fell back to the category table is not a failure — every room built before the
    # sidecars existed did exactly that — but it IS the difference between a scene whose physics
    # came from its assets and one whose physics was invented here, so it is said out loud.
    for key, note in (("no_sidecar", "no compiled physics"),
                      ("unreadable_sidecar", "unreadable physics sidecar"),
                      ("unplaceable", "physics could not be placed"),
                      ("placement_rejected", "physics rejected as mis-placed")):
        names = report.get(key) or []
        if names:
            warnings.append(f"{len(names)} object(s) with {note}: {', '.join(map(str, names[:6]))}")

    if options.get("shake"):
        # The report is the run's stdout, which `run_module` tees into the log. A video is asked
        # for because a number saying "Chair0 moved 0.42 m" is not reviewable and a clip is.
        shake_argv = ["--scene", str(scene_dir(context) / "scene.xml"),
                      "--video", str(scene_dir(context) / "shake.mp4"), "--cutaway"]
        shake_rc, shake_log = run_module(context, "litereality_agent.room_ops.export.mujoco_shake",
                                         shake_argv, log_name="simulate_shake")
        if shake_rc:
            warnings.append(f"shake exited {shake_rc}; see {shake_log}")

    details = {"scene": str(scene_dir(context) / "scene.xml"),
               "from_sidecar": len(report.get("from_sidecar") or []),
               "bodies": {k: report.get(k) for k in ("structure", "free", "attached",
                                                     "articulated", "colliders")}}
    return StageResult("simulate", StageStatus.COMPLETED, details=details, warnings=warnings)
