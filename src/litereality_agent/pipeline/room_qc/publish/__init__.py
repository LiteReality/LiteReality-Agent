"""Compile and publish the final room, plus the review artifacts that go with it.

Lives under `room_qc`, not `realism_authoring`, because publishing is not authoring: by the time
this runs the room is finished being built. What it actually does is gate the room (collision
correction, then the quality report) and then produce the things a human looks at to judge it.
Running last in the flow is not the same as belonging to the flow that precedes it.

Everything publishable lands in `room_preview/` next to the build it describes. `room/` stays the
one editable source; the preview tree is the one place to LOOK. Publishing used to also copy
`Room.py` and `Room.glb` up to the authoring root, which left the same room in three places with
nothing marking which was canonical.

One thing is made here that the room itself is not: the real-vs-render comparison, the room
rendered from the capture's own ARKit poses beside the photo taken there, which is the only
artifact that answers "is this actually the room?" rather than "does this look like a room?". It
costs minutes, so it is last and it is optional.

`Room.web.glb` — the Draco copy a browser is handed — is NOT made here. It is derived from
`Room.glb` and nothing but a viewer ever reads it, so building it at publish time spent a Blender
launch on every run whether or not anyone ever looked at the room. `viewable_room` makes it on the
first `view` of a given build instead, and `compress.compressed` keeps it for every later one.
"""

from __future__ import annotations

import os

from litereality_agent.pipeline.context import RunContext
from litereality_agent.pipeline.result import StageResult, StageStatus
from litereality_agent.pipeline.support import run_module

# The room as a browser wants it: same geometry and clips, roughly a quarter the bytes. Named
# beside `Room.glb` rather than hidden, because it is the one to upload if a room is ever served
# from somewhere other than this repo — a cache by lifecycle, an artifact by usefulness.
WEB_GLB_NAME = "Room.web.glb"
# Enough frames to see whether the room matches the capture, few enough not to dominate a publish.
DEFAULT_COMPARE_FRAMES = 6


def compare_frames(options: dict) -> int:
    """How many real-vs-render pairs to build. `0` (or `COMPARE_FRAMES=0`) skips the pass."""
    raw = options.get("compare_frames")
    if raw is None:
        raw = os.environ.get("COMPARE_FRAMES", DEFAULT_COMPARE_FRAMES)
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return DEFAULT_COMPARE_FRAMES


def complete(context: RunContext) -> bool:
    return (context.preview_dir / "Room.glb").is_file()


def published_room(context: RunContext):
    """The room to hand a viewer, smallest first, or None if this run has not published one.

    Publishing names both files, so choosing between them belongs here rather than in whatever is
    doing the looking. The web copy wins only while it is at least as new as the build it came
    from — otherwise a rebuilt room would be shown inside last publish's geometry.
    """
    built = context.preview_dir / "Room.glb"
    web = context.preview_dir / WEB_GLB_NAME
    try:
        if web.is_file() and built.is_file() and web.stat().st_mtime >= built.stat().st_mtime:
            return web
    except OSError:
        pass
    return built if built.is_file() else None


def require_published(context: RunContext):
    room = published_room(context)
    if room is None:
        raise SystemExit(
            f"no published room for {context.scan} — expected {context.preview_dir / 'Room.glb'}\n"
            f"  build it with:  litereality stage publish {context.scene_dir}"
        )
    return room


def viewable_room(context: RunContext):
    """The body to hand a browser, compressing one if this build has not been viewed yet.

    `require_published` answers "which of the two files is current"; this answers "and make the
    small one if it isn't there", which is the whole difference between a viewer that needs a
    publish flag set and one that just works. Costs a Blender launch on the first view of a given
    build and nothing on every later one, because `compressed` caches beside the room.

    No Blender is a slower page, not a failure: `compress` hands back the full-size room and the
    viewer serves that.
    """
    from litereality_agent.room_ops import compress

    room = require_published(context)
    if room.name == WEB_GLB_NAME:
        return room
    print("compressing the room for the browser — once per build, then cached")
    made = compress.compressed(room, context.preview_dir / WEB_GLB_NAME)
    if made == room:
        print("  no Blender for the web copy; serving the full-size room")
    return made


def run(context: RunContext, options: dict) -> StageResult:
    warnings: list[str] = []
    collision_rc, collision_log = run_module(
        context,
        "litereality_agent.pipeline.room_qc.correct",
        ["--room", context.authored_room, "--apply"],
        log_name="publish_collision",
    )
    if collision_rc:
        warnings.append(f"collision correction exited {collision_rc}; see {collision_log}")
    quality_rc, quality_log = run_module(
        context,
        "litereality_agent.pipeline.room_qc.checks",
        ["--room", context.authored_room],
        log_name="publish_quality",
    )
    if quality_rc:
        warnings.append(f"scene quality checks reported violations; see {quality_log}")
    rc, log = run_module(
        context, "litereality_agent.room_ops.compile.build_from_room",
        ["--room", context.authored_room, "--out", context.preview_dir, "--regenerate"],
        log_name="publish_compile",
    )
    glb = context.preview_dir / "Room.glb"
    if rc or not glb.is_file():
        return StageResult("publish", StageStatus.FAILED, error=f"final compile failed; see {log}")
    # Flatten node-graph SHELL materials into the glb. `build_from_room` only EXPORTS; the bake is
    # the step `compile_room` adds on top, and the agent's own compile/render tool goes through
    # `compile_room` — so every render the author sees is baked. Publishing without it silently
    # shipped a different room: a procedural wall/floor/ceiling material (`two_tone_mat`,
    # `carpet_mat`, `ceiling_tile_mat`) has no glTF representation, so it exports with neither a
    # texture nor a baseColorFactor and renders WHITE. Flat-RGB and fetched-image materials
    # survive either way, which is why this stayed invisible until a room used procedural ones.
    from litereality_agent.room_ops import api

    bake_rc = api.bake_room(context.preview_dir / "Room.blend", glb)
    if bake_rc:
        warnings.append(f"material bake exited {bake_rc}; see {glb.parent / 'bake.log'}")

    frames = compare_frames(options)
    if frames and context.capture_dir.is_dir():
        compare = context.authoring_root / "compare"
        print(f"rendering {frames} real-vs-render comparisons")
        compare_rc, compare_log = run_module(
            context, "litereality_agent.room_ops.rendering.room_render.render_vs_capture",
            ["--scan", context.capture_dir, "--room", context.authored_room,
             "--out", compare, "--frames", frames],
            log_name="publish_compare",
        )
        if compare_rc:
            warnings.append(f"comparison renders exited {compare_rc}; see {compare_log}")

    print(f"\nwalk it:  uv run litereality view {context.scene_dir}")
    return StageResult(
        "publish", StageStatus.COMPLETED,
        artifacts={
            "room_source": str(context.authored_room / "Room.py"),
            "room_glb": str(glb),
        },
        warnings=warnings,
    )
