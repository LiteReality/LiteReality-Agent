"""render_vs_capture.py — reconstruction vs. real capture, frame by frame at the EXACT poses.

For every captured keyframe in the scan (`frame_NNNNN.jpg` + its `frame_NNNNN.json` ARKit pose), this
renders the reconstructed `Room.py` from that camera's exact pose + intrinsics and places
RENDER | REAL side by side for the SAME viewpoint. It is a true per-frame comparison against ONLY the
captured frames — unlike ``walkthrough_sidebyside``, which interpolates a smooth fly-through and stacks
it against the whole capture video.

    python -m lrauthor.room_ops.rendering.room_render.render_vs_capture --scan <scan_dir> --room <room_dir> \
        --out <out_dir> [--res-div 2] [--fps 6] [--frames all|N|a-b|0,8,30]

Outputs under <out_dir>:  renders/frame_NNNNN.png (raw renders), pairs/pair_NNNNN.png (side-by-side),
vs_capture.mp4 (all pairs as a clip), contact_sheet.png (a static grid for a quick look).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from lrauthor.fonts import font as _shared_font

# Renders landscape (rotate=False) so the render matches the landscape capture jpg directly — the
# ARKit intrinsics (cx≈W/2, cy≈H/2) describe the stored landscape frame, so NO 90° rotation is wanted.
_BLENDER_DRIVER = r"""
import sys, os
sys.path.insert(0, os.environ["RVC_MODULE_DIR"])  # authoring/views/room_render (holds render_room_cameras.py)
from render_room_cameras import build_scene, render_room_from_cameras
a = sys.argv[sys.argv.index("--") + 1:]
scan_dir, assets_dir, out_dir, frames, res_div = a[0], a[1], a[2], a[3], int(a[4])
# assets_dir MUST hold the materialized object glbs (manifest.json + Object/*.glb) — otherwise
# Room.py places nothing and the render is a bare shell (no furniture).
scene = build_scene(scan_dir, assets_dir, out_dir)
render_room_from_cameras(scene, out_dir, frames=frames, res_div=res_div,
                         engine="EEVEE", rotate=False, describe=False)
"""


def _blender_bin() -> str:
    d = os.environ.get("LITEREALITY_BLENDER", "")
    b = Path(d) / "blender" if d else Path("blender")
    return str(b)


def _resolve_assets(room: Path) -> Path:
    """Where the materialized object glbs live. The pipeline puts them in the sibling
    `room_preview/` (manifest.json + Object/*.glb) — that is what Room.py must load, NOT the
    room dir (which only has object.py sources). Falls back to the room dir if a manifest is there."""
    prev = room.parent / "room_preview"
    if (prev / "manifest.json").is_file():
        return prev
    if (room / "manifest.json").is_file():
        return room
    raise SystemExit(f"no assets manifest found (looked in {prev} and {room}); "
                     "run the pipeline's build/export first so objects are materialized.")


def _render_poses(scan_dir: Path, room_py: Path, assets_dir: Path, render_dir: Path,
                  frames: str, res_div: int):
    render_dir.mkdir(parents=True, exist_ok=True)
    scr = Path(tempfile.mkdtemp()) / "_rvc_driver.py"
    scr.write_text(_BLENDER_DRIVER)
    env = dict(os.environ, SB_ROOM_PY=str(room_py),
               RVC_MODULE_DIR=str(Path(__file__).resolve().parent))  # authoring/views/room_render
    cmd = [_blender_bin(), "-b", "--python", str(scr), "--",
           str(scan_dir), str(assets_dir), str(render_dir), frames, str(res_div)]
    print(f"[rvc] rendering poses via {room_py.name} …", flush=True)
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        raise SystemExit(f"[rvc] blender render failed (exit {r.returncode})")


def _label(draw, x, w, text, font):
    from PIL import Image, ImageDraw  # noqa: F401
    tw = draw.textlength(text, font=font)
    draw.text((x + (w - tw) / 2, 6), text, fill=(240, 240, 240), font=font)


def _font(sz):
    from PIL import ImageFont
    try:
        return _shared_font(sz)
    except Exception:  # noqa: BLE001
        return ImageFont.load_default()


def _upright_roll(scan_dir: Path, idx: str) -> int:
    """Degrees to rotate frame `idx` so world-up points up in the image, snapped to a right angle.

    ARKit stores every frame 1920x1440 landscape regardless of how the phone was held, so a scan
    taken in portrait comes out rolled 90 degrees — the room appears on its side. The pose tells us
    which way is up: express world-up (+Y) in camera axes (x right, y up) and read off the roll.
    Both the render and the real photo come from the SAME pose, so rotating both by this keeps them
    aligned.
    """
    try:
        import json
        import math

        d = json.loads((scan_dir / f"{idx}.json").read_text())
        M = d.get("cameraPoseARFrame")
        if not M:
            return 0
        m = [float(x) for x in (M if isinstance(M, list) else [])]
        if len(m) != 16:
            return 0
        # column-major 4x4 -> rotation rows
        r = [[m[0], m[4], m[8]], [m[1], m[5], m[9]], [m[2], m[6], m[10]]]
        # world up (0,1,0) in camera axes == column 1 of R transposed == r[.][1]
        ux, uy = r[0][1], r[1][1]
        return int(round(math.degrees(math.atan2(ux, uy)) / 90.0)) * 90
    except Exception:  # noqa: BLE001
        return 0


def _make_pairs(scan_dir: Path, render_dir: Path, pairs_dir: Path, height: int = 720):
    """Stitch each rendered frame next to its real capture jpg (same pose). Returns sorted pair paths."""
    from PIL import Image, ImageDraw
    pairs_dir.mkdir(parents=True, exist_ok=True)
    bar = 34
    font = _font(22)
    small = _font(18)
    pairs = []
    for rp in sorted(render_dir.glob("frame_*.png")):
        idx = rp.stem  # frame_NNNNN
        real = scan_dir / f"{idx}.jpg"
        if not real.is_file():
            continue
        ri = Image.open(rp).convert("RGB")
        gi = Image.open(real).convert("RGB")
        # Un-roll both by the same pose-derived angle so the room reads upright (see _upright_roll).
        roll = _upright_roll(scan_dir, idx)
        if roll % 360:
            ri = ri.rotate(roll, expand=True)
            gi = gi.rotate(roll, expand=True)
        w = round(height * ri.width / ri.height)
        ri = ri.resize((w, height))
        gi = gi.resize((round(height * gi.width / gi.height), height))
        gw = gi.width
        canvas = Image.new("RGB", (w + gw + 8, height + bar), (15, 15, 15))
        d = ImageDraw.Draw(canvas)
        _label(d, 0, w, "RENDER · reconstruction", font)
        _label(d, w + 8, gw, "REAL · capture", font)
        d.text((8, height + bar - 24), idx, fill=(150, 150, 150), font=small)
        canvas.paste(ri, (0, bar))
        canvas.paste(gi, (w + 8, bar))
        out = pairs_dir / f"pair_{idx.split('_')[1]}.png"
        canvas.save(out)
        pairs.append(out)
    return pairs


def _contact_sheet(pairs, out_png: Path, cols: int = 2, n: int = 12):
    """A static grid of up to n evenly-spaced pairs — a one-glance overview without playing the clip."""
    from PIL import Image
    if not pairs:
        return None
    step = max(1, len(pairs) // n)
    picks = pairs[::step][:n]
    thumbs = [Image.open(p).convert("RGB") for p in picks]
    tw = 900
    thumbs = [t.resize((tw, round(tw * t.height / t.width))) for t in thumbs]
    th = max(t.height for t in thumbs)
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw + (cols + 1) * 6, rows * th + (rows + 1) * 6), (30, 30, 30))
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet.paste(t, (6 + c * (tw + 6), 6 + r * (th + 6)))
    sheet.save(out_png)
    return out_png


def _encode(pairs, out_mp4: Path, fps: float):
    """ffmpeg the pairs into a clip. Renders are static per pose, so a low fps reads best."""
    if not pairs:
        raise SystemExit("[rvc] no render/real pairs to encode")
    lst = out_mp4.parent / "_pairs.txt"
    lst.write_text("".join(f"file '{p.resolve()}'\nduration {1.0 / fps:.4f}\n" for p in pairs)
                   + f"file '{pairs[-1].resolve()}'\n")
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p",
           "-r", str(max(fps, 24)), str(out_mp4)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("[rvc] ffmpeg failed:\n" + r.stderr[-800:])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True, type=Path, help="scan dir (frame_*.jpg + frame_*.json)")
    ap.add_argument("--room", required=True, type=Path, help="room dir containing Room.py")
    ap.add_argument("--out", required=True, type=Path, help="output dir")
    ap.add_argument("--frames", default="all", help="all | N evenly-spaced | a-b | 0,8,30")
    ap.add_argument("--res-div", type=int, default=2, help="render downscale (1=native 1920×1440)")
    ap.add_argument("--fps", type=float, default=6.0, help="pairs shown per second in the clip")
    ap.add_argument("--assets", type=Path, default=None,
                    help="materialized assets dir (default: sibling room_preview/)")
    a = ap.parse_args(argv)

    room_py = a.room / "Room.py"
    if not room_py.is_file():
        raise SystemExit(f"no Room.py at {room_py}")
    assets_dir = a.assets or _resolve_assets(a.room)
    print(f"[rvc] assets: {assets_dir}", flush=True)
    render_dir = a.out / "renders"
    pairs_dir = a.out / "pairs"
    a.out.mkdir(parents=True, exist_ok=True)

    _render_poses(a.scan, room_py, assets_dir, render_dir, a.frames, a.res_div)
    pairs = _make_pairs(a.scan, render_dir, pairs_dir)
    print(f"[rvc] {len(pairs)} render/real pairs", flush=True)
    sheet = _contact_sheet(pairs, a.out / "contact_sheet.png")
    mp4 = a.out / "vs_capture.mp4"
    _encode(pairs, mp4, a.fps)
    print(f"[rvc] DONE\n  video   {mp4}\n  sheet   {sheet}\n  pairs   {pairs_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
