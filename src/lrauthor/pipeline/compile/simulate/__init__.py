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

from lrauthor.pipeline.context import RunContext
from lrauthor.pipeline.result import StageResult, StageStatus
from lrauthor.pipeline.support import run_module


def scene_dir(context: RunContext, *, seed: bool = False):
    return context.scene_dir / "mujoco_seed" if seed else context.authoring_root / "mujoco"


def source_room(context: RunContext, *, seed: bool = False):
    """Which room to export.

    The AUTHORED room by default. But authoring and simulation answer different questions, and a
    room that has not been authored is still a complete physical scene: the shell with its openings
    cut out, and every reconstructed object in its measured box carrying the mass, friction,
    colliders and joints its own build compiled. What authoring adds is materials, wall fixtures and
    the small objects — appearance and clutter — none of which a policy needs to be trained against.

    So `--from-seed` exports the room straight out of `scene_init`, which is the fastest way to get
    a scan into a simulator and the cleanest test of the objects themselves: nothing in the scene
    came from anywhere but the reconstruction.
    """
    if seed:
        return context.seed_room, context.seed_room.parent / "room_preview"
    return context.authored_room, context.preview_dir


def complete(context: RunContext) -> bool:
    return (scene_dir(context) / "scene.xml").is_file()


def _report_path(context: RunContext, seed: bool):
    return scene_dir(context, seed=seed) / "export_report.json"


def _report(context: RunContext, seed: bool = False) -> dict:
    path = _report_path(context, seed)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:                                        # noqa: BLE001 — no report, no summary
        return {}


def run(context: RunContext, options: dict) -> StageResult:
    seed = bool(options.get("from_seed"))
    room, preview = source_room(context, seed=seed)
    if not (room / "Room.py").is_file():
        return StageResult("simulate", StageStatus.FAILED,
                           error=f"no room at {room} — run "
                                 f"`{'seed' if seed else 'author'}` first")
    if not (preview / "Room.glb").is_file():
        return StageResult("simulate", StageStatus.FAILED,
                           error=f"no built room at {preview / 'Room.glb'} — "
                                 f"run `{'seed' if seed else 'publish'}` first")

    argv = ["--room", str(room), "--out", str(scene_dir(context, seed=seed))]
    # Decomposition is only reached for objects with NO physics sidecar, and it is the slowest
    # thing in the export by two orders of magnitude. Turning it off makes those objects collide as
    # their convex hulls — a table becomes a solid block — so it stays on and is opt-out.
    if options.get("no_decompose"):
        argv.append("--no-decompose")
    if options.get("reuse_meshes"):
        argv.append("--reuse-meshes")
    rc, log = run_module(context, "lrauthor.room_ops.export.mujoco_scene", argv,
                         log_name="simulate_export_seed" if seed else "simulate_export")
    if rc or not (scene_dir(context, seed=seed) / "scene.xml").is_file():
        return StageResult("simulate", StageStatus.FAILED,
                           error=f"MuJoCo export failed; see {log}")

    warnings: list[str] = []
    report = _report(context, seed)
    # An object that fell back to the category table is not a failure — every room built before the
    # sidecars existed did exactly that — but it IS the difference between a scene whose physics
    # came from its assets and one whose physics was invented here, so it is said out loud.
    for key, note in (("no_sidecar", "no compiled physics"),
                      ("unreadable_sidecar", "unreadable physics sidecar"),
                      ("unplaceable", "physics could not be placed"),
                      ("placement_rejected", "physics rejected as mis-placed"),
                      ("authored_no_package", "authored into Room.py, so no object package")):
        names = report.get(key) or []
        if names:
            warnings.append(f"{len(names)} object(s) with {note}: {', '.join(map(str, names[:6]))}")

    # A SCENE THAT STARTS INTERPENETRATING IS NOT SIM-READY, HOWEVER WELL IT EXPORTED. Overlap at
    # t=0 is stored energy the solver has to discharge, so the first thing an episode does is throw
    # furniture. This is the one check that speaks for the whole file rather than for one object.
    if report.get("loads_clean") is False:
        worst = report.get("initial_overlaps") or []
        detail = "; ".join(f"{o['bodies']} {o['mm']}mm" for o in worst[:3])
        warnings.append(f"scene starts interpenetrating in {len(worst)} place(s): {detail}")
    elif report.get("loads_clean") is None and report.get("load_check_error"):
        warnings.append(f"could not load the exported scene to check it: "
                        f"{report['load_check_error']}")

    # THE HEADLINE NUMBER, SAID WHETHER OR NOT ANYTHING FAILED. Every warning above fires on a
    # named object going wrong; none of them fires on the ordinary case of an authored room whose
    # fixtures never had a package to begin with, and that case is the majority of the colliders.
    # A scene where most of the physics was invented at export time is a worse scene, and a run
    # that does not say so is reporting a success it has not earned.
    coverage = report.get("sidecar_coverage")
    if coverage is not None and coverage < 1.0:
        warnings.append(
            f"{report.get('derived_colliders', 0)} of {report.get('colliders', 0)} colliders "
            f"({(1 - coverage) * 100:.0f}%) were derived at export time, not read from a compiled "
            f"sidecar — mass, friction and pivots for those came from a category table"
        )

    if options.get("shake"):
        # The report is the run's stdout, which `run_module` tees into the log. A video is asked
        # for because a number saying "Chair0 moved 0.42 m" is not reviewable and a clip is.
        shake_argv = ["--scene", str(scene_dir(context, seed=seed) / "scene.xml"),
                      "--video", str(scene_dir(context, seed=seed) / "shake.mp4"), "--cutaway"]
        shake_rc, shake_log = run_module(context, "lrauthor.room_ops.export.mujoco_shake",
                                         shake_argv, log_name="simulate_shake")
        if shake_rc:
            warnings.append(f"shake exited {shake_rc}; see {shake_log}")

    details = {"scene": str(scene_dir(context, seed=seed) / "scene.xml"),
               "source": "seed" if seed else "authored",
               "from_sidecar": len(report.get("from_sidecar") or []),
               "sidecar_coverage": report.get("sidecar_coverage"),
               "derived_colliders": report.get("derived_colliders"),
               "bodies": {k: report.get(k) for k in ("structure", "free", "attached",
                                                     "articulated", "colliders")}}
    return StageResult("simulate", StageStatus.COMPLETED, details=details, warnings=warnings)
