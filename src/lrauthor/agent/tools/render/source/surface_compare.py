"""surface_compare — per-surface head-on RENDER | REAL(stitch) sheets for the planner / verify.

Renders every wall / floor / ceiling of the CURRENT room ORTHOGRAPHIC (front-on, furniture
hidden, framed to the whole plane) via `authoring/views/image_selection/surface_compare/render_ortho.py`,
and pairs each with its rectified real-surface stitch (from `surface_reference`). A head-on,
like-for-like comparison — far clearer for "does this material / tile match the real wall" than
the oblique whole-frame side-by-sides. The planner reads it to FORM the plan and to VERIFY it.

Disable with `HARNESS_SURFACE_COMPARE=0` (e.g. to save a Blender render per attempt).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from lrauthor.agent.tools.shared import config

# This pointed at `services/`, gone since the layout restructure, so `render_ortho.py` could
# never be found. image_selection is shared tool source now.
_SC_DIR = config.PACKAGE_ROOT / "agent" / "tools" / "shared" / "image_selection" / "surface_compare"


def enabled() -> bool:
    return os.environ.get("HARNESS_SURFACE_COMPARE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def _dump_surf_uv(room_dir: Path, uv_path: Path) -> None:
    """Write the floor/ceiling u/v axes render_ortho expects ($SB_SURF_UV), derived from the SHELL.

    Same convention as the stitcher (dominant_room_axis): the length-weighted dominant wall
    direction is u, its perpendicular v. The 90° ambiguity is resolved against the real stitch's
    known size_m (manifest): the floor's extent along u must match size_m[0]. Axes are written in
    the USD Y-up frame render_ortho converts from (usd = (bx, bz, -by))."""
    import json
    import math
    import re

    src = (Path(room_dir) / "Room.py").read_text()
    segs = []
    for m in re.finditer(
        r'"start":\s*\[\s*([-\d.eE]+)\s*,\s*([-\d.eE]+)\s*\]\s*,\s*"end":\s*\[\s*([-\d.eE]+)\s*,\s*([-\d.eE]+)\s*\]',
        src,
    ):
        x0, y0, x1, y1 = map(float, m.groups())
        segs.append(((x0, y0), (x1, y1)))
    if not segs:
        raise ValueError("no SHELL wall segments found in Room.py")

    pts = [p for s in segs for p in s]

    def ext(a):
        vals = [px * a[0] + py * a[1] for px, py in pts]
        return max(vals) - min(vals)

    # candidate axes: every wall's direction (+ its perpendicular). RoomPlan walls are often NOT
    # mutually parallel, so a mean axis tilts — instead pick the candidate whose floor extents
    # best match the real stitch's known size_m (that IS the stitcher's frame, by construction).
    cands = []
    for (x0, y0), (x1, y1) in segs:
        dx, dy = x1 - x0, y1 - y0
        ln = math.hypot(dx, dy)
        if ln < 0.5:  # skip stub walls — their direction is noise
            continue
        u = (dx / ln, dy / ln)
        cands.append((ln, u))
        cands.append((ln, (-u[1], u[0])))
    if not cands:
        raise ValueError("no usable wall segments")

    W = H = None
    try:
        man = json.loads((config.SURFACE_REF / "surface_ref_manifest.json").read_text())
        items = man if isinstance(man, list) else man.get("surfaces", [])
        fl = next(s for s in items if s.get("name") == "Floor0")
        W, H = fl["size_m"]
    except (StopIteration, FileNotFoundError, KeyError, ValueError):
        pass

    def score(c):
        ln, u = c
        v = (-u[1], u[0])
        if W is None:
            return -ln  # no stitch info: prefer the longest wall's direction
        return abs(ext(u) - W) + abs(ext(v) - H)

    _, u = min(cands, key=score)
    v = (-u[1], u[0])
    eu, ev = ext(u), ext(v)

    to_usd = lambda b: [b[0], 0.0, -b[1]]  # noqa: E731 — blender (x,y,0) → usd (x, 0, -y)
    uv_path.parent.mkdir(parents=True, exist_ok=True)
    uv_path.write_text(json.dumps({"u": to_usd(u), "v": to_usd(v)}))
    print(f"  surf_uv: u={u} v={v} (extents {eu:.2f}×{ev:.2f}) -> {uv_path}", flush=True)


def build(room_dir: Path, out_dir: Path) -> dict:
    """Render the current room's surfaces head-on and pair each with its real stitch.

    Returns {"renders": {surface: ortho_png}, "sheet": <jpg|None>}. The per-surface ortho PNGs
    are what the planner reads (alongside the stitches it already reads); the sheet is for humans."""
    out_dir = Path(out_dir)
    ortho = out_dir / "render"
    ortho.mkdir(parents=True, exist_ok=True)
    for f in ortho.glob("*_ortho.png"):
        f.unlink()

    # Floor/ceiling u/v axes (cached once per scan) — WITHOUT them render_ortho falls back to
    # longest-edge axes, which is 90°-ambiguous: the floor/ceiling ortho comes out ROTATED vs
    # the real stitch and the head-on comparison doesn't line up. room_pipeline (the stitcher's
    # own floor_uv dump) is not shipped in this repo, so derive the SAME convention here:
    # dominant wall axis from the SHELL, 90°-disambiguated against the stitch's known size_m.
    uv = config.SURFACE_REF / "surf_uv.json"
    if not uv.is_file():
        try:
            _dump_surf_uv(Path(room_dir), uv)
        except Exception as e:  # noqa: BLE001 — fall back to derived axes rather than fail
            print(f"  surf_uv derivation failed ({e}); floor/ceiling ortho may be rotated", flush=True)

    env = dict(
        os.environ,
        LITEREALITY_SCAN=config.SCAN,
        SB_ROOM_PY=str(Path(room_dir) / "Room.py"),
        SB_RENDER_DIR=str(ortho),
        LITEREALITY_ROOM_DIR=str(room_dir),
        LITEREALITY_ROOM_PREVIEW=str(Path(room_dir).parent / "room_preview"),
        SB_SURF_UV=str(uv) if uv.is_file() else "",
    )
    log = (out_dir / "render_ortho.log").open("w")
    try:
        subprocess.run(
            [
                config.blender(),
                "-b",
                "--python",
                str(_SC_DIR / "render_ortho.py"),
                "--",
                str(config.SCAN_DIR),
                str(config.ASSETS_DIR),
                str(ortho),
            ],
            env=env,
            check=True,
            cwd=str(config.ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    finally:
        log.close()

    renders = {p.stem.replace("_ortho", ""): str(p) for p in sorted(ortho.glob("*_ortho.png"))}
    sheet_path: str | None = None
    try:  # human-readable RENDER | REAL sheet vs the real stitches (surface_reference)
        import sys

        if str(config.ROOT) not in sys.path:
            sys.path.insert(0, str(config.ROOT))
        from lrauthor.agent.tools.shared.image_selection.surface_compare import sheet

        sp = out_dir / "surface_compare.jpg"
        sheet.compose(ortho, config.SURFACE_REF, sp, config.SCAN)
        sheet_path = str(sp)
    except Exception as exc:
        print(f"     ⚠ surface_compare sheet failed: {exc}", flush=True)
    return {"renders": renders, "sheet": sheet_path}
