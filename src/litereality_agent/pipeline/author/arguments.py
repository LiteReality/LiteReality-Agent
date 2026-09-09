"""Fill a stage-2 entry point's path arguments from a SCENE PACKAGE.

Every authoring stage used to require the same four flags — `--room`, `--surface-ref`, `--scan`,
`--refroot` — and getting one wrong fails late and quietly (a `render` that never works, an
authoring pass that never reads a stitch). All four are recorded in the package `init` wrote, so
`--scene <dir>` supplies them, and an explicitly-passed flag still wins over the package.

Usage in an entry point::

    from litereality_agent.room_ops import manifest
    manifest.bootstrap()            # BEFORE importing anything that reads $LITEREALITY_*
    ...
    ap.add_argument("--scene", default=None)
    ap.add_argument("--room", default=None)      # was: required=True
    a = ap.parse_args()
    stage_args.bind(a, need=("room", "surface_ref", "scan"))
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

# Argument name -> the manifest path key that supplies it.
FROM_PACKAGE = {
    "surface_ref": "surface_ref",
    "scan": "capture",
    "refroot": "refroot",
    "room_init": "room_init",
    "results": None,  # special-cased: <authoring>/obj_refine, beside the work room (see bind)
}


def work_room(pkg, *, create: bool = True) -> Path:
    """The room this run EDITS — `realism_authoring[_<RUN_TAG>]/room`.

    Never the seed: `room_init/room` is the deterministic output of scene_init and stays pristine
    so a re-run can always start over. This must be the SAME directory the CLI's `do_author`
    prepares (`$AUTHORING/room`, where `AUTHORING=<scan>/realism_authoring$SUF`), or a
    package-launched stage and a the CLI-launched stage quietly fork into two rooms and the
    later stages export whichever one they happen to find.

    Older packages — written before stage 2 got its own output tree — record no `authoring_room`
    and keep the work room at `scene_stage/_oneshot/`. Those still resolve to their own room, so
    a half-finished run from before the move is not stranded.
    """
    authoring_room = pkg.path("authoring_room")
    tag = (os.environ.get("RUN_TAG") or "").strip()
    if authoring_room is not None:
        room = authoring_room if not tag else (
            authoring_room.parent.with_name(authoring_room.parent.name + f"_{tag}") / authoring_room.name
        )
    else:
        scene_stage = pkg.path("scene_stage")
        if scene_stage is None:
            raise SystemExit(f"package {pkg.root} records no scene_stage — was init interrupted?")
        room = scene_stage / (f"_oneshot_{tag}" if tag else "_oneshot") / "room"
    if create and not room.exists():
        seed = pkg.path("room_init")
        if seed is None or not seed.is_dir():
            raise SystemExit(f"no seed room in {pkg.root} — run `uv run litereality run <scan> --through seed` first")
        room.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(seed, room)
        print(f"[scene] work room seeded from init: {room}", flush=True)
    return room


def bind(a: argparse.Namespace, *, need: tuple[str, ...] = (), pkg=None) -> argparse.Namespace:
    """Fill any unset path argument on `a` from the scene package. Returns `a`.

    `need` names the arguments this stage genuinely cannot run without; anything still unset
    afterwards fails here, with the package that was consulted named in the message, rather than
    surfacing later as an empty render or a missing stitch.
    """
    from litereality_agent.room_ops import manifest

    hint = getattr(a, "scene", None)
    bindable = ("room", *FROM_PACKAGE)
    unset = [n for n in bindable if hasattr(a, n) and getattr(a, n) is None]
    if pkg is None and (hint or any(n in need for n in unset)):
        # Go looking only when a REQUIRED path is missing (or a package was named). A
        # fully-explicit invocation — the CLI passes every flag it needs — must not be retargeted
        # by a package that merely happens to be discoverable from the current directory, and an
        # unset optional like `--results` is not reason enough to demand one.
        pkg = manifest.discover(hint) or manifest.require(hint)
        manifest.apply_env(pkg, override=hint is not None)

    if pkg is not None:
        for name in unset:
            if name == "room":
                value = work_room(pkg)
            else:
                key = FROM_PACKAGE[name]
                # `results` sits NEXT TO the work room, for the same reason work_room() exists:
                # the CLI passes `--results $AUTHORING/obj_refine`, so defaulting to the package
                # root instead sent a `--scene` run's before/after sheets somewhere the CLI never
                # looks — two copies of the evidence in two places, and the run summary's
                # `before/after` line silently absent. Deriving it from work_room() also picks up
                # $RUN_TAG and the legacy pre-move layout for free.
                value = (work_room(pkg, create=False).parent / "obj_refine") if key is None \
                    else pkg.path(key)
            if value is not None:
                setattr(a, name, str(value))

    missing = [n for n in need if not getattr(a, n, None)]
    if missing:
        where = f" (scene package: {pkg.root})" if pkg is not None else ""
        raise SystemExit(
            f"missing required path(s): {', '.join('--' + m.replace('_', '-') for m in missing)}"
            f"{where}\n  pass them explicitly, or point --scene at a package that records them."
        )
    return a


def add_scene_arg(ap: argparse.ArgumentParser) -> None:
    """The one flag that replaces the rest."""
    ap.add_argument(
        "--scene",
        default=None,
        help="scene package dir (holds scene.json) — supplies --room/--surface-ref/--scan/"
             "--refroot. Default: $LR_SCENE, the current directory, or the only package on disk.",
    )
