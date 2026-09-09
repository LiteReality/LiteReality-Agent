"""Scene-definition and compilation API for LiteReality-Agent.

A room is defined by a compact, editable **program**: `Room.py` (the assembler + a semantic
SHELL) plus per-object `object.py`/GLBs and a manifest. `room_ops` is the library that
(a) **exports** that definition from a scan and (b) **compiles** it to a previewable `room.glb`.
It's the room analog of an SDK + compiler — `Room.py` is authored against it, and it builds the
artifact. This is the one place scene definition lives.

LAYER 2 — a library used by scene_init, authoring and scripts/ops. It should depend on
nothing else in this repo.

ONE tool, two consumers:
  • `scene_init`       — exports the SEED `Room.py` + builds the first `room.glb` (once)
  • `authoring`        — re-compiles the agent's edited `Room.py` → `room.glb` EVERY loop iteration

Public API (everything else — export_room, build_room, wall_*, materials… — stays importable):
    export_scene(scan)              scan → Room/<scan>/   (Room.py + SHELL + manifest)
    compile_room(room_dir)          Room.py → <preview>/Room.glb   (assemble in Blender, baked)
    bake_room(blend, glb)           bake SHELL node materials into the glb (faithful preview)
    compress.compressed(glb, out)   Room.glb → the Draco copy `litereality view` serves

`build_from_room` / `bake_glb` spawn Blender, so they run as subprocesses; the helpers wrap them.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

__all__ = ["export_scene", "compile_room", "bake_room", "blender_bin"]


def blender_bin() -> str:
    """Resolve the Blender executable through the room-ops path configuration.

    This used to fall back to the bare string `"blender"`, which is not a path: with Blender
    installed as a macOS app bundle and nothing on PATH, `subprocess.run` raised
    `FileNotFoundError: 'blender'` from inside the bake, *after* the room had already compiled.
    `$LR_BLENDER` is kept as an accepted spelling for callers that used it.
    """
    v = os.environ.get("LR_BLENDER", "").strip()
    if v:
        return str(Path(v) / "blender") if Path(v).is_dir() else v
    from lrauthor.room_ops.paths import find_blender

    return find_blender()


def export_scene(scan: str, out_root: "Path | None" = None) -> "Path | None":
    """scan → `Room/<scan>/` definition (Room.py + SHELL + manifest). Returns the Room dir."""
    from lrauthor.room_ops.export import export_room

    return export_room.export(scan, out_root)


def compile_room(
    room_dir,
    out_dir=None,
    *,
    bake: bool = True,
    blender: "str | None" = None,
    regenerate: bool = False,
) -> "Path | None":
    """Compile a `Room/` definition → `Room.glb` (the scene-definition compiler).

    Runs `build_from_room` (materialize each object.py + assemble the SHELL in Blender), then
    optionally `bake_room` (flatten node-graph SHELL materials so the glb == the render). Returns
    the `Room.glb` path, or None if the assemble step failed.
    """
    room_dir = Path(room_dir)
    preview = Path(out_dir) if out_dir else (room_dir.parent / "room_preview")
    cmd = [sys.executable, "-m", "lrauthor.room_ops.compile.build_from_room", "--room", str(room_dir)]
    if out_dir:
        cmd += ["--out", str(out_dir)]
    if regenerate:
        cmd.append("--regenerate")
    if subprocess.run(cmd).returncode != 0:
        return None
    glb = preview / "Room.glb"
    if not glb.is_file():
        return None
    if bake:
        bake_room(preview / "Room.blend", glb, blender=blender)
    return glb


def bake_room(blend, glb, *, blender: "str | None" = None, resolution: int = 1024) -> int:
    """Bake the SHELL's node-graph materials into a built `Room.glb` (faithful preview).
    No-op (returns 0) if the `Room.blend` isn't present. Returns the Blender exit code."""
    blend, glb = Path(blend), Path(glb)
    if not blend.is_file():
        return 0
    # Blender's bake floods stdout with per-object `Fra: … | Updating Images | Loading bake_…`
    # progress. bake_room runs on EVERY `compile` the authoring/materials agent makes, so if that
    # inherits the parent stdout it drowns the step log (authoring.log hit 164 MB this way). Route
    # it to a sibling bake.log — same precedent as the render engine's render.log — so the step log
    # stays readable and the bake output is still there to inspect on failure.
    log_path = glb.parent / "bake.log"
    with open(log_path, "w") as log:
        return subprocess.run(
            [
                blender or blender_bin(),
                "-b",
                str(blend),
                "--python",
                str(_HERE / "compile" / "bake_glb.py"),
                "--",
                str(glb),
                str(resolution),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        ).returncode
