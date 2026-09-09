"""_blender_frames.py — Blender-side worker: build the room + render specific capture frames.

Runs INSIDE Blender (``blender -b --python _blender_frames.py -- <repo-data-root> <room.py> <scan_dir>
<assets_dir> <out_dir> <frames_spec>``). Reuses render_room_cameras to rebuild the exact ARKit
camera poses and render the room from each requested frame, upright (matching the real photos).
Called by compose.render_frames — not meant to be run by hand.
"""

import os
import sys

_argv = sys.argv[sys.argv.index("--") + 1 :]
repo, room_py, scan_dir, assets, out_dir, frames_spec = _argv[:6]

# render_room_cameras is package code under ``room_ops``, not under the caller-supplied <repo>
# (which is the data root). This used to join "..", because the worker sat beside `room_render`;
# moving the tools' source under `agent/` broke that sibling relationship silently — Blender exits
# 0 on a raised script, so it surfaced as a missing render_manifest.json. Anchor on the package
# directory by NAME instead, so the next move cannot reintroduce the same failure.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = _HERE
while os.path.basename(_PKG) != "lrauthor" and os.path.dirname(_PKG) != _PKG:
    _PKG = os.path.dirname(_PKG)
sys.path.insert(0, os.path.join(_PKG, "room_ops", "rendering", "room_render"))
import render_room_cameras as R  # noqa: E402

os.environ["SB_ROOM_PY"] = room_py

def _die(stage, exc):
    # Blender exits 0 even when a -b --python script raises — which upstream turns into a
    # confusing FileNotFoundError on render_manifest.json. Fail LOUD with the real cause.
    import traceback
    print(f"ROOM RENDER FAILED during {stage}: {type(exc).__name__}: {exc}", flush=True)
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)

try:
    scene = R.build_scene(scan_dir, assets, out_dir)
except BaseException as e:  # noqa: BLE001 — includes SystemExit from a broken Room.py
    _die("room build (Room.py)", e)
R.setup_render(engine="EEVEE", light=True)

# Parse frames_spec to EXPLICIT frame indices. This tool takes frame NUMBERS, so a bare
# number means "frame index N" — NOT select_frames' default of "N evenly-spaced frames".
# We hand render_room_from_cameras an explicit list so it can't reinterpret a count.
#   'all'/'' -> all   |   'a-b' -> that range   |   '10' or '10,20,30' -> those exact indices
if frames_spec in ("all", ""):
    frames = "all"
elif "-" in frames_spec and all(p.isdigit() for p in frames_spec.split("-", 1)):
    frames = frames_spec  # a range; render_room_from_cameras/select_frames handles it
else:
    frames = [int(x) for x in frames_spec.split(",") if x.strip()]  # exact indices (list)

# describe=True writes render_manifest.json (per-frame VISIBLE objects + their projected label
# points), which annotate_views reads to label every object (walls, doors, furniture) on the render.
try:
    R.render_room_from_cameras(
        scene, out_dir, frames=frames, engine="EEVEE", light=True, rotate=True, describe=True
    )
except BaseException as e:  # noqa: BLE001
    _die("frame rendering", e)
print("FRAMES RENDER DONE", flush=True)
