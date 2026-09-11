"""The evidence pack: everything an authoring model may look at or measure, in ONE directory.

    evidence/
      README.md          the data formats and coordinate conventions, stated once
      scan/         ->   the capture (frame_*.jpg/json, depth_*.png, conf_*.png, pointcloud.pcd, room.usdz)
      stitches/     ->   the head-on per-surface stitches (surface_ref)
      helpers/           a COPY of litereality_agent.evidence_kit — read_scan, measure, rectify,
                         contact_sheets, arkit_cameras — importable with sys.path.insert(0, "evidence/helpers")
      contact_sheets/    every photograph, upright, a dozen per sheet

Why a directory of files rather than more MCP tools: a tool answers one question the way its
author framed it; a helper is a function the model combines in a script of its own. The measured
GPT-6 one-shot run got its proportions right by probing depth, slicing the point cloud and
triangulating pixels in numpy — none of which a tool offered — and the `open` authoring profile
is built around giving the model that freedom with the infra's evidence, materials and gates.

The pack lives BESIDE the room (`<authoring_root>/evidence`), never inside it: the room dir is
copied and scanned wholesale downstream and a stray file in it is a trap. `helpers/` is a copy,
not a symlink, so a run directory stays self-describing after the package moves on.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

README = """\
# Evidence pack — {scan_name}

All units metres. Two world frames appear in this data:
* **ARKit world** (Y up): `frame_NNNNN.json["cameraPoseARFrame"]` (row-major 4x4 camera->world),
  `pointcloud.pcd`, `room.usdz`.
* **Blender world** (Z up), which `Room.py` and every helper output use:
  `world_blender = Rx(+90) @ world_arkit`  (y_arkit -> z_blender, z_arkit -> -y_blender).
The ARKit camera looks down its local -Z with +Y up — Blender's camera convention — so a photo's
Blender camera matrix is simply `Rx(90) @ cameraPoseARFrame`. `helpers/arkit_cameras.py` builds
those cameras inside Blender; `Room.py` already carries them as `cam_NNNNN`.

## scan/
* `frame_NNNNN.jpg` — {n_frames} photographs, stored LANDSCAPE ({w}x{h}); the phone was held
  portrait so they look rotated. Pixel (u right, v down) coordinates in the helpers refer to
  this stored image. `frame_NNNNN.json` — its pose and 3x3 `intrinsics` (fx 0 cx / 0 fy cy / 0 0 1).
* `depth_NNNNN.png` — 16-bit LiDAR depth in millimetres at {dw}x{dh}, same field of view as the jpg
  (scale the intrinsics by {dw}/{w}). `conf_NNNNN.png` — ARKit confidence 0..2 per depth pixel.
* `pointcloud.pcd` — {n_points} ARKit-world points. `room.usdz` — RoomPlan's walls, openings and
  one oriented box per detected furniture item (already turned into `SHELL` in `Room.py`).

## stitches/
One head-on (rectified) image per wall/floor/ceiling plus an unknown-mask (white = not observed).
Furniture smears onto the plane; the material is what is behind it.

## helpers/  (`import sys; sys.path.insert(0, "evidence/helpers")`)
```python
from read_scan import Scan; import measure, rectify, contact_sheets
S = Scan("evidence/scan")
measure.probe_depth(S, 12, 960, 720)                 # LiDAR point under a jpg pixel -> Blender xyz
measure.triangulate(S, [(12, (u, v)), (14, (u, v))]) # same feature in 2+ photos -> xyz (for what LiDAR misses)
measure.height_profile(S, (x0, y0, x1, y1), floor_z=SHELL["floor_z"])   # peaks = horizontal surfaces
measure.pcd_slice(S, 0.70, 0.80, out_png="slice.png", floor_z=SHELL["floor_z"])  # a plan at one height
rectify.rectify_region(S, 12, quad_px, "textures/monitor.png", (1024, 640))     # photo quad -> texture
```
The cloud's own floor estimate is ±3 cm; `SHELL["floor_z"]` from `Room.py` is RoomPlan's and better.
`arkit_cameras.py` runs inside Blender only (`blender -b Room.blend --python your_script.py`).

## contact_sheets/
Every photograph, upright, numbered — look at all of them before building anything.
"""


def has_capture(d: Path) -> bool:
    """A raw RoomPlan capture: frame_NNNNN.jpg + .json pairs (depth, cloud and usdz beside them)."""
    try:
        return any(f.startswith("frame_") and f.endswith(".json") for f in os.listdir(d))
    except OSError:
        return False


def resolve_scan(scan_dir: Path, scene_dir: Path | None = None) -> Path:
    """The RAW capture for a scene — which is not always what the pipeline calls the capture.

    `scene.json`'s `capture` link can point at the usdz-only input dir (Office_room's does:
    `input/usdz_files`, one file), and `input/rgbd/` is the pipeline's own split layout
    (image/ depth/ intrinsic/ extrinsic/), not the frame_NNNNN.* the kit reads. The raw scan
    lives under the scans root recorded in scene.json (`roots.scans/<scan>`), or wherever
    `LITEREALITY_SCAN` points. A pack without frames is worse than no pack — the brief promises
    depth and a cloud — so this raises rather than linking whatever it was handed."""
    tried = []
    for cand in _scan_candidates(Path(scan_dir), scene_dir):
        tried.append(str(cand))
        if has_capture(cand):
            return cand
    raise FileNotFoundError(
        "no raw capture (frame_NNNNN.json) found for the evidence pack; looked in: " + ", ".join(tried))


def _scan_candidates(scan_dir: Path, scene_dir: Path | None):
    yield scan_dir
    env = os.environ.get("LITEREALITY_SCAN")
    if env:
        yield Path(env)
    if scene_dir is not None:
        sj = Path(scene_dir) / "scene.json"
        try:
            meta = json.loads(sj.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
        name = meta.get("scan") or Path(scene_dir).name
        root = (meta.get("roots") or {}).get("scans")
        if root:
            yield Path(root) / name
        src = (meta.get("capture") or {}).get("source")
        if src:
            yield Path(src)
            yield Path(src).parent / name
    try:
        from litereality_agent.settings import settings
        yield settings.resolved_scans_dir() / (Path(scene_dir).name if scene_dir else scan_dir.name)
    except Exception:  # noqa: BLE001 — settings are optional here
        pass


def build(authoring_root: Path, scan_dir: Path, surface_ref: Path | None, *, sheets: bool = True,
          force: bool = False, scene_dir: Path | None = None) -> Path:
    """Create `<authoring_root>/evidence` and return it. Idempotent unless `force`."""
    from litereality_agent import evidence_kit

    pack = Path(authoring_root) / "evidence"
    if pack.is_dir() and not force and (pack / "README.md").is_file() and has_capture(pack / "scan"):
        return pack
    scan_dir = resolve_scan(Path(scan_dir), scene_dir)
    pack.mkdir(parents=True, exist_ok=True)

    _link(pack / "scan", Path(scan_dir))
    if surface_ref is not None and Path(surface_ref).is_dir():
        _link(pack / "stitches", Path(surface_ref))

    helpers = pack / "helpers"
    if helpers.exists():
        shutil.rmtree(helpers)
    shutil.copytree(Path(evidence_kit.__file__).parent, helpers,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    facts = _facts(Path(scan_dir))
    (pack / "README.md").write_text(README.format(scan_name=Path(scan_dir).name, **facts), encoding="utf-8")

    if sheets:
        try:
            import sys
            sys.path.insert(0, str(helpers))
            from read_scan import Scan  # type: ignore
            import contact_sheets  # type: ignore
            contact_sheets.build(Scan(str(scan_dir)), str(pack / "contact_sheets"))
            shutil.rmtree(helpers / "__pycache__", ignore_errors=True)
        except Exception as exc:  # noqa: BLE001 — sheets are a convenience, never a reason to fail
            (pack / "contact_sheets.SKIPPED").write_text(f"{type(exc).__name__}: {exc}\n")
    return pack


def _link(link: Path, target: Path) -> None:
    if link.is_symlink() or link.exists():
        if link.is_symlink() or link.is_file():
            link.unlink()
        else:
            shutil.rmtree(link)
    os.symlink(os.path.realpath(target), link)


def _facts(scan_dir: Path) -> dict:
    """Numbers the README quotes — read from the files, never assumed."""
    facts = {"n_frames": 0, "w": "?", "h": "?", "dw": "?", "dh": "?", "n_points": "?"}
    try:
        from PIL import Image
        jsons = sorted(f for f in os.listdir(scan_dir) if f.startswith("frame_") and f.endswith(".json"))
        facts["n_frames"] = len(jsons)
        if jsons:
            i = jsons[0][6:11]
            with Image.open(scan_dir / f"frame_{i}.jpg") as im:
                facts["w"], facts["h"] = im.size
            dp = scan_dir / f"depth_{i}.png"
            if dp.is_file():
                with Image.open(dp) as im:
                    facts["dw"], facts["dh"] = im.size
        pcd = scan_dir / "pointcloud.pcd"
        if pcd.is_file():
            with open(pcd, "rb") as fh:
                for _ in range(12):
                    line = fh.readline().decode("ascii", errors="replace")
                    if line.startswith("POINTS"):
                        facts["n_points"] = int(line.split()[1])
                        break
    except Exception:  # noqa: BLE001 — a README with '?' beats no pack
        pass
    return facts
