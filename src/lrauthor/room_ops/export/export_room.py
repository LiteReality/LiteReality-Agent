#!/usr/bin/env python3
"""export_room.py — export one scan into the most compact room definition, `Room/<scan>/`.

The definition is SOURCE ONLY: it holds nothing that a build can regenerate.

    Room/<scan>/
      Room.py            the assembler (each object into its own RoomPlan box -> the scene)
      Room.md            notes for this room + object list + rebuild steps
      room.usdz          the RoomPlan shell (walls / floor / openings / a box per object)
      manifest.json      which object goes in which box (+ where it came from)
      Objects/
        Procedural/<name>/  object.py  object.md  textures/   <- code, no final glb
        Static/<name>/      object.py  object.md  <name>.glb   <- the glb IS the source

One interface for everything: every object, static or procedural, exposes an `object.py`
(`blender -b --python object.py -- <texdir> <out.glb>` emits that object's glb), so the
assembler can treat them all alike. A static object.py loads its bundled glb (materials and
orientation fixes can go here later); a procedural one builds the geometry from nothing.

Rebuild with `python -m lrauthor.room_ops.compile.build_from_room --room Room/<scan>`.

    python -m lrauthor.room_ops.export.export_room --scan office-elliott
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from .. import paths as config
from ..compile import pack_assets

HERE = Path(__file__).resolve().parent


# The uniform format: a static object's object.py (load the bundled mesh -> export; materials and
# orientation fixes go here later).
STATIC_OBJECT_PY = '''"""__NAME__ — a static object (mesh source, in the uniform object.py format).

STATIC: the geometry is a neurally generated or scanned mesh, `__NAME__.glb` — there is no
procedural modelling code. It is still wrapped in the same object.py interface a procedural
object exposes, so the assembler can treat every object alike.

Run:
  blender -b --python object.py                       # reads ./__NAME__.glb, writes ./__NAME___built.glb
  blender -b --python object.py -- <texdir> <out.glb>
"""
import bpy, os, sys

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
_argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
SRC = os.path.join(HERE, "__NAME__.glb")               # the source mesh (materials included)
OUT = _argv[1] if len(_argv) > 1 else os.path.join(HERE, "__NAME___built.glb")


def build(out_glb=OUT):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=SRC)
    # Placeholder for the uniform format: materials-as-code and the orientation fix (canonical
    # alignment) belong here.
    bpy.ops.object.select_all(action="SELECT")
    os.makedirs(os.path.dirname(out_glb) or ".", exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=out_glb, export_format="GLB",
                              use_selection=True, export_apply=True)
    print(f"[static] __NAME__ -> {out_glb}")
    return out_glb


if __name__ == "__main__":
    build()
'''

STATIC_OBJECT_MD = """# __NAME__ — a static object (mesh source)

**STATIC**: the geometry is a neurally generated or scanned mesh, `__NAME__.glb`. There is no
procedural modelling code, so the glb itself is the source and has to be kept. It still exposes
the same `object.py` interface every other object does.

Category: `__CAT__`

## Rebuild
```bash
blender -b --python object.py -- . __NAME___built.glb   # loads __NAME__.glb and exports it
```
Under the materials-as-code policy, `build()` in `object.py` will eventually apply the materials
here (a real base texture plus an LAB recolour) and correct the orientation to canonical, rather
than relying on whatever textures and orientation the mesh arrived with.
"""


def copy_sim_sidecar(src_dir: Path, dest: Path, name: str) -> int:
    """Carry one object's compiled physics into the Room package. Returns the files copied.

    `object_generation.sim` leaves a `sim/` directory beside each generated glb holding the object's
    own mass, inertia, friction, joints and CONVEX COLLIDERS — the statement of what it is
    physically, already gated by a solver. Without it in here, a Room directory describes how the
    room LOOKS and nothing about how it behaves, and the MuJoCo export has to re-derive every
    number from a category table at the last moment.

    Copied FILE BY FILE from the model rather than as a whole directory, because a generative
    object (a TRELLIS chair) is a bare glb at the top of `reconstruct/` and shares one `sim/` with
    every other one: copying the directory would put all six chairs' colliders into each chair.
    """
    physics = src_dir / f"{name}.physics.json"
    if not physics.is_file():
        return 0
    try:
        model = json.loads(physics.read_text(encoding="utf-8"))
    except Exception as exc:                                  # noqa: BLE001 — never fail an export
        print(f"  ! {name}: unreadable physics sidecar ({exc})")
        return 0
    wanted = {physics.name, f"{name}.urdf", f"{name}.sim_check.json"}
    for link in model.get("links") or []:
        wanted.update(c["file"] for c in (link.get("colliders") or []))
        if link.get("visual"):
            wanted.add(link["visual"])
    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    for filename in sorted(wanted):
        source = src_dir / filename
        if source.is_file():
            shutil.copy2(source, dest / filename)
            copied += 1
    return copied


def reset_dir(d: Path):
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)


def find_blender() -> str:
    """Delegates to the canonical room-ops Blender resolver; see the note about the six
    copies that disagreed about macOS."""
    from lrauthor.room_ops.paths import find_blender as _canonical

    return _canonical()

def extract_shell_json(usdz: Path) -> dict:
    """Run extract_shell.py under Blender to turn the usdz into a semantic SHELL dict."""
    import tempfile

    tmp = Path(tempfile.mkdtemp()) / "shell.json"
    subprocess.run(
        [
            find_blender(),
            "-b",
            "--python",
            str(HERE / "extract_shell.py"),  # sibling in room_ops/export/
            "--",
            str(usdz),
            str(tmp),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return json.loads(tmp.read_text())


def write_room_md(roomdir: Path, scan: str, manifest: dict):
    rows = []
    for a in manifest["assets"]:
        boxes = a.get("represents_prims") or [a["maps_to_prim"]]
        kind = "Procedural" if a["source"] == "procedural" else "Static"
        rows.append(f"| `{a['object']}` | {a['category']} | **{kind}** | {', '.join(boxes)} |")
    md = f"""# Room definition — {scan}

The most compact, source-only definition of this room. Everything needed to rebuild
the scene lives here as plain text; derived artifacts (procedural meshes, texture
images, the assembled `Room.glb`) are NOT stored — they are regenerated into
`Room_preview/` by `build_from_room.py`.

## Layout
- **`Room.py`** — the assembler **and** this room's geometry. At the bottom it carries
  a `SHELL` dict holding the walls, openings and object boxes (see *Editing* below).
  Running it reads `SHELL` and rebuilds the shell — **no `room.usdz` needed**.
- `manifest.json` — which object fills which RoomPlan box.
- `Objects/Procedural/<name>/` — `object.py` (build code) + `object.md` + `textures.json`
  (a material *recipe*; no image files — textures are downloaded / recolored at build).
- `Objects/Static/<name>/` — `<name>.glb` (neural-mesh source) + a uniform `object.py`.

Every object exposes the same `object.py` interface, so the build treats them alike.

## Editing the scene
Everything is plain text, so an agent can edit the room directly.

**Reshape the room / move walls / move openings** — edit `SHELL` in `Room.py`:
- `walls`: `{{"start":[x,y], "end":[x,y], "thickness":t}}` on the floor plane. Move a
  wall by changing its endpoints. All wall heights are `ceiling_z - floor_z`.
- `openings`: `{{"wall":"WallN", "type":"door|window", "offset":d, "width":w,
  "height":h, "sill":s}}`. `offset` = distance along the wall from its `start`;
  `sill` = height of the opening's bottom above the floor. Slide a door along its wall
  by changing `offset`; raise/lower a window with `sill`.
- `objects`: `{{"category":..., "center":[x,y,z], "size":[w,d,h], "yaw":deg}}`.
  Move/rotate/resize a piece of furniture by editing `center` / `yaw` / `size`.

**Change how an object looks** — edit its `Objects/.../object.py` (procedural build
code) or `textures.json` (material recipe: a Poly Haven base id + an optional LAB
color shift).

## Object inventory
| object | category | type | RoomPlan box |
|--------|----------|------|--------------|
{chr(10).join(rows)}

## Rebuild
```bash
uv run litereality stage seed run/{scan} --force
# -> run/{scan}/room_preview/Room.glb + Room.blend
```
The shell comes from `SHELL` in `Room.py`; cameras come from the original scan frames
if present. Coordinate frame: Blender Z-up, meters; object build frame faces −Y.
"""
    (roomdir / "Room.md").write_text(md, encoding="utf-8")


def _union_box(boxes: list[dict]) -> dict:
    """Oriented union of same-yaw object boxes (center/size/yaw), in the SHELL frame (horizontal
    = center[0],center[1]; vertical = center[2]). Rotate horizontals into the yaw frame, take the
    axis-aligned min/max, rotate the union centre back."""
    import math

    yaw = boxes[0].get("yaw", 0.0)
    a = math.radians(yaw)
    ca, sa = math.cos(a), math.sin(a)
    xmn = ymn = zmn = 1e9
    xmx = ymx = zmx = -1e9
    for b in boxes:
        cx, cy, cz = b["center"]
        sx, sy, sz = b["size"]
        rx, ry = ca * cx + sa * cy, -sa * cx + ca * cy   # rotate horizontal by -yaw
        xmn, xmx = min(xmn, rx - sx / 2), max(xmx, rx + sx / 2)
        ymn, ymx = min(ymn, ry - sy / 2), max(ymx, ry + sy / 2)
        zmn, zmx = min(zmn, cz - sz / 2), max(zmx, cz + sz / 2)
    rcx, rcy = (xmn + xmx) / 2, (ymn + ymx) / 2
    cx, cy = ca * rcx - sa * rcy, sa * rcx + ca * rcy      # rotate centre back by +yaw
    # keep the category of the largest-volume member (drives placement collection)
    cat = max(boxes, key=lambda b: b["size"][0] * b["size"][1] * b["size"][2]).get("category", "storage")
    return {"category": cat,
            "center": [round(cx, 4), round(cy, 4), round((zmn + zmx) / 2, 4)],
            "size": [round(xmx - xmn, 4), round(ymx - ymn, 4), round(zmx - zmn, 4)],
            "yaw": yaw}


def _apply_shell_merges(shell: dict, scan: str) -> None:
    """In place: replace SHELL member boxes with one union box per merged object, read from each
    merged object's members.json marker under object_refs/<scan>/<merged>/members.json."""
    import glob
    import json as _json

    objs = shell.get("objects") or {}
    refs = config.object_init_dir(scan) / "object_refs" / scan
    for mj in glob.glob(str(refs / "*" / "members.json")):
        merged = Path(mj).parent.name
        try:
            members = _json.loads(Path(mj).read_text())
        except Exception:
            continue
        present = [objs[m] for m in members if m in objs]
        if not present:
            continue
        union = _union_box(present)
        for m in members:
            objs.pop(m, None)
        objs[merged] = union
        print(f"  [merge] SHELL box {merged} <- {members}")
    shell["objects"] = objs


def _apply_layout(shell: dict, scan: str) -> None:
    """In place: adopt the object boxes the layout pass settled, from `scene_data/layout_shell.json`.

    The SHELL above is re-extracted from `room.usdz` under Blender, which is right for walls,
    openings and the floor and wrong for the objects. `scene_init/layout` repairs the object boxes
    at ingest — before crops — so every crop, reference image and generated GLB is built against the
    REPAIRED box. Placing those assets back into the scanned box undoes the repair at the last step
    and, worse, disagrees with the asset it is placing: Kitchen's counter run was generated for a
    63 cm depth and would be scaled into the 87 cm box the scan measured.

    So the object boxes come from the layout pass and everything else keeps coming from the usdz.
    On the merged units this also settles a disagreement between two union computations —
    `merge_boxes._union_obb` unions the ARKit pkl entries, `_union_box` above unions the extracted
    SHELL boxes, and on Kitchen they land 33 cm apart. The pkl-derived one wins because it is the
    box the object stage actually used.

    Never raises: a missing or malformed file leaves the export exactly as it was.
    """
    path = config.scene_data_dir(scan) / "layout_shell.json"
    if not path.is_file():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  [layout] ignoring unreadable {path.name}: {exc}")
        return

    objs = shell.get("objects") or {}
    adopted, moved = 0, 0
    for name, box in (payload.get("objects") or {}).items():
        current = objs.get(name)
        if not current or "center" not in box or "size" not in box:
            continue
        if (current.get("center") != box["center"] or current.get("size") != box["size"]
                or current.get("yaw") != box.get("yaw")):
            moved += 1
        current["center"] = list(box["center"])
        current["size"] = list(box["size"])
        if box.get("yaw") is not None:
            current["yaw"] = box["yaw"]
        adopted += 1
    for name in (payload.get("dropped") or []):
        if objs.pop(name, None) is not None:
            print(f"  [layout] SHELL box {name} removed — the layout pass dropped it")
    shell["objects"] = objs
    if adopted:
        print(f"  [layout] SHELL adopted {adopted} object boxes from the layout pass "
              f"({moved} differ from the scan)")


def export(scan: str, out_root: Path | None = None) -> Path | None:
    scan = config.scan_name(scan)  # accept a name or a path to the scan folder
    recon = config.reconstruct_dir(scan)
    if not recon.is_dir():
        print(
            f"\n  ✗ no objects for '{scan}': {recon} does not exist.\n"
            f"    Run the object stage first, then re-run export:\n"
            f"      uv run litereality run {scan} --through reconstruct\n"
        )
        return None

    # 1. pack the scan's reconstruct GLBs -> box mapping + per-object source (in-process)
    manifest = pack_assets.pack(scan=scan, out=config.assets_dir(scan))
    if not manifest["assets"]:
        print(
            f"\n  ⚠ {recon} has no usable GLBs yet — writing a shell-only room "
            f"(run the object stage to fill it).\n"
        )

    roomdir = Path(out_root) if out_root else config.room_dir(scan)
    reset_dir(roomdir)
    proc_root = roomdir / "Objects" / "Procedural"
    static_root = roomdir / "Objects" / "Static"
    proc_root.mkdir(parents=True)
    static_root.mkdir(parents=True)

    # 2. The assembler (with this room's semantic SHELL embedded) + the manifest. The binary
    #    usdz is no longer stored: the SHELL rebuilds the shell without it.
    usdz = config.resolve_usdz(scan)
    shell = extract_shell_json(usdz)
    # Fold any object-stage MERGES (e.g. Sink_Storage0 = Sink0 + Storage2, recorded in each merged
    # object's members.json) into the SHELL's object boxes, so the assembler has a box named for the
    # merged object to place its GLB into (else the merged GLB has no box and is dropped).
    _apply_shell_merges(shell, scan)
    # AFTER the merges: the layout pass works on merged units, so its box for `Sink_Storage0` only
    # has somewhere to land once the merge has created that name in the SHELL.
    _apply_layout(shell, scan)
    # build_room.py lives in room_ops/compile; HERE is room_ops/export.
    room_py = (HERE.parent / "compile" / "build_room.py").read_text(encoding="utf-8")
    # SHELL MUST be defined BEFORE `ROOM = main()` runs — else _load_shell can't see the embedded
    # SHELL and falls back to room.usdz (the room-py-shell-ordering gotcha). build_room.py ends with
    # the `if __name__ … ROOM = main()` guard, so strip it, append SHELL, then re-add the guard.
    _guard = 'if __name__ in ("__main__", "room_py_exec"):'
    if _guard in room_py:
        room_py = room_py[: room_py.rindex(_guard)].rstrip() + "\n"
    room_py += (
        "\n\n# ===========================================================================\n"
        "# COMPACT SHELL of THIS room — walls / openings / object boxes, fully editable.\n"
        "# Edit these numbers to edit the scene (see Room.md). _load_shell() reads SHELL first.\n"
        "# ===========================================================================\n"
        f"SHELL = {json.dumps(shell, indent=2)}\n"
        '\n\nif __name__ in ("__main__", "room_py_exec"):   # SHELL is now defined above\n'
        "    ROOM = main()\n"
    )
    (roomdir / "Room.py").write_text(room_py, encoding="utf-8")
    (roomdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # 3. The source for each object
    summary = {"procedural": 0, "static": 0, "missing": []}
    for a in manifest["assets"]:
        name = a["object"]
        cat = a["category"]
        if a["source"] == "procedural":
            src = config.reconstruct_dir(scan) / name
            d = proc_root / name
            d.mkdir(parents=True, exist_ok=True)
            for f in ("object.py", "object.md"):
                if (src / f).exists():
                    shutil.copy2(src / f, d / f)
            # Textures are not stored as files: they are reversed into a textures.json recipe
            # that the build re-downloads and recolours.
            tex = src / "textures"
            if tex.is_dir() and any(tex.iterdir()):
                try:
                    from ..compile.texture_recipe import gen_recipe
                except ImportError:
                    from lrauthor.room_ops.compile.texture_recipe import (
                        gen_recipe,  # type: ignore
                    )
                recipe, bundled = gen_recipe(tex)
                (d / "textures.json").write_text(json.dumps(recipe, indent=2), encoding="utf-8")
                summary["recipe_imgs"] = summary.get("recipe_imgs", 0) + (
                    len(recipe) - len(bundled)
                )
                if bundled:  # the few that cannot be rebuilt: keep the real file as a fallback
                    (d / "textures").mkdir(exist_ok=True)
                    for bf in bundled:
                        shutil.copy2(tex / bf, d / "textures" / bf)
                    summary["bundled"] = summary.get("bundled", 0) + len(bundled)
            summary["sim"] = summary.get("sim", 0) + bool(
                copy_sim_sidecar(src / "sim", d / "sim", name))
            summary["procedural"] += 1
        else:  # static: the glb is the source; generate the uniform object.py / object.md
            d = static_root / name
            d.mkdir(parents=True, exist_ok=True)
            glb_src = None
            for cand in (
                config.reconstruct_dir(scan) / f"{name}.glb",
                config.assets_dir(scan) / "glb" / f"{name}.glb",
            ):
                if cand.exists():
                    glb_src = cand
                    break
            if glb_src is None:
                summary["missing"].append(name)
                continue
            shutil.copy2(glb_src, d / f"{name}.glb")
            (d / "object.py").write_text(
                STATIC_OBJECT_PY.replace("__NAME__", name), encoding="utf-8"
            )
            (d / "object.md").write_text(
                STATIC_OBJECT_MD.replace("__NAME__", name).replace("__CAT__", cat), encoding="utf-8"
            )
            summary["sim"] = summary.get("sim", 0) + bool(
                copy_sim_sidecar(glb_src.parent / "sim", d / "sim", name))
            summary["static"] += 1

    # 4. Notes for the room
    write_room_md(roomdir, scan, manifest)

    print(f"Room definition -> {roomdir}")
    print(
        f"  Procedural: {summary['procedural']}  (object.py + textures.json recipe; no glb, no texture files)"
    )
    print(f"  Static:     {summary['static']}  (glb + the uniform object.py)")
    print(f"  Physics:    {summary.get('sim', 0)} objects carry a compiled sim/ sidecar "
          f"(mass, inertia, friction, joints, convex colliders)")
    print(
        f"  Texture recipes: {summary.get('recipe_imgs', 0)} rebuildable"
        + (f", {summary['bundled']} stored as files" if summary.get("bundled") else ", 0 stored as files")
    )
    if summary["missing"]:
        print(f"  ! source not found: {summary['missing']}")
    return roomdir


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--scan", required=True)
    ap.add_argument("--out", help="room definition dir (default run/<scan>/room)")
    args = ap.parse_args()
    export(args.scan, Path(args.out) if args.out else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
