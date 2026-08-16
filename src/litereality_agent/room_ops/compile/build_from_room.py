#!/usr/bin/env python3
"""build_from_room.py — rebuild a Room/ definition into Room_preview/ (the build artifacts).

(en) Compile a Room/<scan>/ definition into Room_preview/<scan>/. For EVERY object (static or procedural)
it runs that object's `object.py` to materialize <name>.glb (procedural objects also materialize textures),
then assembles SHELL + objects into Room.glb (+ Room.blend). Cached object glbs are reused unless --regenerate.

This is what proves the definition is self-contained: for EVERY object, static or procedural
alike, it runs that object's `object.py` to produce a glb at Object/<name>.glb. A procedural
object's textures are materialized into Object/<name>_textures/ as well. The shell and the
objects are then assembled into Room.glb (and Room.blend alongside it).

    Room/<scan>/  (definition)  ---->  Room_preview/<scan>/

    python -m litereality_agent.room_ops.compile.build_from_room --room Room/office-elliott
    python -m litereality_agent.room_ops.compile.build_from_room --room Room/tea_room --regenerate

By default a cached object glb is reused; --regenerate forces every object.py to run again.
Cameras are included when the original scan frames are present, otherwise geometry only.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from litereality_agent import console

from .. import paths as config

HERE = Path(__file__).resolve().parent


def find_blender() -> str:
    """Delegates to the canonical room-ops Blender resolver; see the note about the six
    copies that disagreed about macOS."""
    from litereality_agent.room_ops.paths import find_blender as _canonical

    return _canonical()

def run_object_py(
    obj_dir: Path,
    name: str,
    is_proc: bool,
    out_glb: Path,
    texdir: Path,
    blender: str,
    regenerate: bool,
) -> bool:
    """The uniform interface: run obj_dir/object.py -- <texdir> <out_glb> to emit its glb."""
    if out_glb.exists() and not regenerate:
        print(console.row("reused", name, "cached  (--regenerate to rebuild)", "cyan"))
        return True
    script = obj_dir / "object.py"
    if not script.exists():
        print(console.row("error", name, "no object.py — nothing to build from", "cyan"))
        return False
    out_glb.parent.mkdir(parents=True, exist_ok=True)
    cmd = [blender, "-b", "--python", str(script), "--", str(texdir), str(out_glb)]
    t0 = time.time()
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    took = f"{console.colour('dim')}{time.time() - t0:5.1f}s{console.colour('off')}"
    if r.returncode != 0 or not out_glb.exists():
        tail = r.stderr.decode(errors="replace").strip().splitlines()[-1:] or ["(no stderr)"]
        print(console.row("error", name, f"{'regen' if is_proc else 'wrap'} FAILED — {tail[0][:60]}",
                          "cyan"))
        return False
    size = console.human_bytes(out_glb.stat().st_size)
    kind = "regen" if is_proc else "wrap "
    print(f"{console.row('done', name, f'{kind}  {size:<9}', 'cyan')}{took}")
    return True


def materialize_textures(obj_dir: Path, texdir: Path) -> int:
    """Rebuild the textures into texdir from the textures.json recipe (download from Poly Haven,
    then recolour). Bundled fallback files are copied across too. Returns the texture count."""
    recipe_p = obj_dir / "textures.json"
    if not recipe_p.exists():
        return 0
    try:
        from .fetch_textures import materialize
    except ImportError:
        from fetch_textures import materialize  # type: ignore
    made = materialize(json.loads(recipe_p.read_text()), texdir)
    bundled = obj_dir / "textures"
    if bundled.is_dir():
        texdir.mkdir(parents=True, exist_ok=True)
        for bf in bundled.glob("*"):
            shutil.copy2(bf, texdir / bf.name)
    return len(made)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--room", required=True, help="the Room/<scan> definition directory")
    ap.add_argument("--out", help="output dir (default Room_preview/<scan>)")
    ap.add_argument("--regenerate", action="store_true", help="force every object.py to run again")
    args = ap.parse_args()

    roomdir = Path(args.room).resolve()
    # the room/ dir can be the baseline (…/<scan>/scene_init/scene_stage/room_init/room) or a harness
    # iteration (…/<scan>/scene_init/scene_stage/stage_N/iteration_M/room) — derive <scan> as the
    # folder directly under the output root.
    out_root = config.output_root()
    scan = None
    p = roomdir
    while p.parent != p:
        if p.parent == out_root:
            scan = p.name
            break
        p = p.parent
    if scan is None:  # fallback (Week3 layout Room/<scan> or loose dir)
        scan = roomdir.parent.name if roomdir.name == "room" else roomdir.name
    manifest = json.loads((roomdir / "manifest.json").read_text())
    blender = find_blender()

    # the preview is a SIBLING of the room/ it's built from (so each iteration owns its own).
    preview = Path(args.out) if args.out else (roomdir.parent / "room_preview")
    obj_out = preview / "Object"
    obj_out.mkdir(parents=True, exist_ok=True)

    # 1. Run each object's object.py, materializing its glb (+ procedural textures) into Object/
    print(console.rule(f"build · {scan} · {len(manifest['assets'])} objects"))
    built, failed = 0, []
    textures_note: dict[str, int] = {}
    pv_manifest = {"scene": scan, "glb_dir": "Object", "assets": []}
    for a in manifest["assets"]:
        name = a["object"]
        is_proc = a["source"] == "procedural"
        sub = "Procedural" if is_proc else "Static"
        obj_dir = roomdir / "Objects" / sub / name
        out_glb = obj_out / f"{name}.glb"
        if is_proc:  # rebuild the textures into the preview from the textures.json recipe
            texdir = obj_out / f"{name}_textures"
            n = materialize_textures(obj_dir, texdir)
            if n:
                textures_note[name] = n
        else:
            texdir = obj_dir  # static: texdir is a placeholder (object.py loads its bundled glb)
        ok = run_object_py(obj_dir, name, is_proc, out_glb, texdir, blender, args.regenerate)
        built += ok
        if not ok:
            failed.append(name)
        e = dict(a)
        e["glb"] = f"Object/{name}.glb"
        pv_manifest["assets"].append(e)

    # 1b. ROOM-LEVEL materials (walls/floor/ceiling PBR the authoring fetched): re-materialize from
    # the room's textures.json recipe into <roomdir>/materials, so a COMPACT room definition that
    # ships the RECIPE only (no image binaries) still rebuilds. Room.py's MAT_DIR = <roomdir>/materials.
    room_recipe = roomdir / "textures.json"
    if room_recipe.exists():
        matdir = roomdir / "materials"
        recipe = json.loads(room_recipe.read_text())
        # skip the download when the images are already present (normal in-place builds), unless forced
        if args.regenerate or any(not (matdir / fn).exists() for fn in recipe):
            try:
                from .fetch_textures import materialize
            except ImportError:
                from fetch_textures import materialize  # type: ignore
            made = materialize(recipe, matdir)
            print(f"  room materials: rebuilt {len(made)} textures -> {matdir}")

    # 2. Write the preview manifest (each glb points at Object/<name>.glb)
    (preview / "manifest.json").write_text(json.dumps(pv_manifest, indent=2), encoding="utf-8")

    # 3. Shell + cameras: include cameras when the original frames exist, otherwise build
    #    geometry only — the SHELL in the definition is enough for the shell itself.
    scan_frames = config.resolve_scan_dir(scan)
    scan_dir = scan_frames if (scan_frames / "room.usdz").exists() else roomdir

    out_glb = preview / "Room.glb"
    out_glb.unlink(missing_ok=True)
    # Run Room/<scan>/Room.py (its own copy of the SHELL); the shell rebuilds from SHELL, no usdz
    cmd = [
        blender,
        "-b",
        "--python",
        str(roomdir / "Room.py"),
        "--",
        str(scan_dir),
        str(preview),
        str(out_glb),
    ]
    log = preview / "build.log"
    t0 = time.time()
    r = console.run_logged(cmd, log, "assembling", "yellow")
    took = f"{console.colour('dim')}{time.time() - t0:5.1f}s{console.colour('off')}"
    # Whether the capture's ARKit cameras were found is worth stating: without them the room
    # still assembles, just without the viewpoints everything downstream compares against.
    detail = "with cameras" if scan_dir is scan_frames else "NO frames — geometry only"
    if r.returncode != 0:
        print(console.row("error", "assemble", f"Blender exited {r.returncode}", "yellow"))
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]:
            print(f"     {console.colour('dim')}{line}{console.colour('off')}")
        print(f"     {console.colour('dim')}… full log: {console.short(log)}{console.colour('off')}")
        return r.returncode

    if not out_glb.is_file():
        print(console.row("error", "Room.glb", f"missing   {detail}", "yellow"))
        return 1

    # Carry the SHELL facts the collision check needs INSIDE the glb (glTF `extras`), so the gate
    # is one deterministic function of one file instead of a glb plus whichever Room.py happens to
    # be newest. Stamped here rather than inside Room.py because this side of the subprocess
    # already holds both paths and needs no bpy. Never fatal: an unstamped glb still checks fine
    # via the Room.py fallback, it just is not self-describing.
    try:
        from litereality_agent.room_ops import glb_meta
        from litereality_agent.room_ops.shell import extract_shell

        c = glb_meta.stamp(out_glb, extract_shell((roomdir / "Room.py").read_text()))
        print(console.row("done", "glb metadata",
                          f"{c['objects']} objects, {c['walls']} walls, "
                          f"{c['openings']} openings", "yellow"))
    except Exception as exc:  # noqa: BLE001
        print(console.row("warn", "glb metadata", f"not stamped ({exc})", "yellow"))

    size = console.human_bytes(out_glb.stat().st_size)
    print(f"{console.row('done' if not failed else 'error', 'Room.glb', f'{size:<9} {detail}', 'yellow')}{took}")
    mark = "done" if built == len(manifest["assets"]) else "error"
    summary = f"{built}/{len(manifest['assets'])} objects" + (f"  ✗ {', '.join(failed)}" if failed else "")
    print(console.row(mark, "objects", summary, "yellow"))
    print(f"   {console.colour('dim')}preview → {console.short(preview)}{console.colour('off')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
