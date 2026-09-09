#!/usr/bin/env bash
# Smooth side-by-side walkthrough: RENDER (reconstruction, interpolated camera) | REAL (video).
# Aligns the render to scan_video.mp4 via one anchor (video time of keyframe 0) + ARKit time-deltas.
#
#   make_walkthrough_video.sh <scan_dir> <room_dir> <assets_dir> <out_dir> [fps] [anchor_s]
#
# anchor_s = video timestamp of frame_00000.jpg (the capture's first keyframe). If omitted it is
# found by matching frame_00000.jpg against the video (handles the black intro).
set -euo pipefail
SCAN="$1"; ROOM="$2"; ASSETS="$3"; OUT="$4"; FPS="${5:-24}"; ANCHOR="${6:-}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${LR_PY:-python3}"
BLENDER="${LITEREALITY_BLENDER:?set LITEREALITY_BLENDER to your Blender install dir or binary}"
mkdir -p "$OUT"

# span of the keyframe timeline (seconds) + anchor (video time of keyframe 0)
read SPAN ANCHOR_AUTO < <("$PY" - "$SCAN" "${ANCHOR:-}" <<'PY'
import sys, glob, json, numpy as np, subprocess, os
from PIL import Image
scan, anchor = sys.argv[1], sys.argv[2]
js = sorted(glob.glob(os.path.join(scan, "frame_*.json")))
ts = [json.load(open(j))["time"] for j in js]
span = max(ts) - min(ts)
if anchor:
    print(f"{span:.4f} {float(anchor):.4f}"); sys.exit()
# coarse-match frame_00000.jpg against the video at 3 fps to find its timestamp
import tempfile
d = tempfile.mkdtemp()
subprocess.run(["ffmpeg","-v","error","-i",os.path.join(scan,"scan_video.mp4"),
                "-vf","fps=3,scale=96:72", os.path.join(d,"v_%05d.png")], check=True)
R = np.asarray(Image.open(os.path.join(scan,"frame_00000.jpg")).convert("L").resize((96,72)),float)
best=None
for f in sorted(glob.glob(os.path.join(d,"v_*.png"))):
    n=int(f.split("_")[-1].split(".")[0]); t=(n-1)/3.0
    v=np.asarray(Image.open(f).convert("L"),float); mse=((v-R)**2).mean()
    if best is None or mse<best[0]: best=(mse,t)
print(f"{span:.4f} {best[1]:.4f}")
PY
)
ANCHOR="${ANCHOR:-$ANCHOR_AUTO}"
N=$(printf '%.0f' "$(echo "$SPAN * $FPS" | bc -l)")
echo "span=${SPAN}s  anchor=${ANCHOR}s  fps=${FPS}  frames=${N}"

echo "[1/4] render reconstruction along the interpolated camera path ($N frames)…"
env LITEREALITY_SCAN="$(basename "$SCAN")" SB_ROOM_PY="$ROOM/Room.py" \
    LITEREALITY_ROOM_DIR="$ROOM" LITEREALITY_ROOM_PREVIEW="$ROOM/../room_preview" \
  "$BLENDER" -b --python "$REPO/authoring/views/room_render/render_walkthrough.py" -- \
    "$SCAN" "$ASSETS" "$OUT/render" "$N" >"$OUT/render.log" 2>&1
echo "    rendered $(ls "$OUT/render"/render_*.png 2>/dev/null | wc -l) frames"

echo "[2/4] extract matching real video segment [${ANCHOR}, +${SPAN}] at ${FPS} fps…"
mkdir -p "$OUT/real"
ffmpeg -v error -ss "$ANCHOR" -t "$SPAN" -i "$SCAN/scan_video.mp4" -vf "fps=$FPS" "$OUT/real/real_%05d.png" -y
echo "    extracted $(ls "$OUT/real"/real_*.png 2>/dev/null | wc -l) real frames"

echo "[3/4] composite RENDER | REAL side-by-side + encode…"
# LEFT = reconstruction, RIGHT = real video. The capture is held portrait but stored landscape,
# so rotate each frame 90° CW (transpose=1) to upright, then hstack at a common height. Plain
# hstack (no drawtext — its font dep fails on headless boxes).
ffmpeg -v error -y \
  -framerate "$FPS" -start_number 0 -i "$OUT/render/render_%05d.png" \
  -framerate "$FPS" -start_number 1 -i "$OUT/real/real_%05d.png" \
  -filter_complex "[0:v]transpose=1,scale=-2:820[L];[1:v]transpose=1,scale=-2:820[R];[L][R]hstack=inputs=2,format=yuv420p" \
  -c:v libx264 -crf 20 -r "$FPS" "$OUT/walkthrough.mp4"
echo "[4/4] DONE -> $OUT/walkthrough.mp4"
ls -la "$OUT/walkthrough.mp4" | awk '{print "    "$5" bytes"}'
