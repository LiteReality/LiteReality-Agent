# room_render

Render the **`room.py`** scene (the assembled room — RoomPlan shell + generated
GLB assets) from the **ARKit capture cameras**, so each render reproduces the
viewpoint of `frame_NNNNN.jpg`. Reports render speed.

`room.py` builds geometry + cameras but sets no lights or engine, so this adds
fast lighting (ambient world + a sun) and uses EEVEE — then renders one image per
capture camera and times it.

## Requirements

- **Blender** (tested with 5.1). `room.py` uses `bpy`, so this runs *inside*
  Blender — there is no pure-Python path.
- `room.py` at the project root, plus its inputs: the scan folder (`room.usdz` +
  `frame_*.json`) and `input/assets/` (the GLBs + `manifest.json`).

## Run (headless)

```bash
/Applications/Blender.app/Contents/MacOS/Blender -b \
  --python authoring/views/room_render/render_room_cameras.py -- \
  [scan_dir] [assets_dir] [out_dir] [frames] [engine] [res_div]
```

All args optional (pass `""` to keep a default while setting a later one):

| Arg | Default | Meaning |
|---|---|---|
| `scan_dir` | auto (the `input/…` folder with `room.usdz`) | scan folder |
| `assets_dir` | `input/assets` | folder with `manifest.json` + `glb/*.glb` |
| `out_dir` | `output/room_render` | where PNGs are written |
| `frames` | `all` | `all` · `N` (N evenly-spaced) · `a-b` range · `0,8,30` list |
| `engine` | `EEVEE` | `EEVEE` (fast) or `CYCLES` (slower, CPU-reliable headless) |
| `res_div` | `1` | downscale: `1` = native 1920×1440, `2` = 960×720, … |

Examples:

```bash
# all 59 cameras, full res, EEVEE  ->  output/room_render/frame_NNNNN.png
… --python authoring/views/room_render/render_room_cameras.py -- "" "" output/room_render all

# quick look: 8 evenly-spaced cameras at half res
… --python authoring/views/room_render/render_room_cameras.py -- "" "" output/room_render 8 EEVEE 2

# one specific camera (matches frame_00030.jpg)
… --python authoring/views/room_render/render_room_cameras.py -- "" "" output/room_render 30
```

## Use as a function (inside Blender)

```python
import sys; sys.path.insert(0, "authoring/views/room_render")
from render_room_cameras import build_scene, render_room_from_cameras

scene = build_scene()                       # execs room.py -> built RoomScene
stats = render_room_from_cameras(
    scene, "output/room_render",
    frames=[0, 8, 30],                      # or None/'all', or N, or 'a-b'
    res_div=1, engine="EEVEE",
)
print(stats["per_frame_mean_sec"], stats["fps"])
```

`render_room_from_cameras` returns a stats dict (`total_sec`, `per_frame_mean_sec`,
`per_frame_median/min/max_sec`, `fps`, `est_all_cameras_sec`, `resolution`, …). It
relies on `room.py`'s own `scene.render_from(frame_index, out_path, res_div)`.

## Measured speed (Blender 5.1, EEVEE, this Mac)

Building the scene (USD import + thicken walls + cut openings + place 9 GLB
assets + 59 cameras) takes **~1.7 s**, once per session.

| Resolution | Per frame (mean) | Throughput | All 59 cameras |
|---|---|---|---|
| **1920×1440** (full, `res_div=1`) | **~1.42 s** | ~0.71 fps | **~84 s** render (~86 s incl. build+startup) |
| 960×720 (`res_div=2`) | ~0.63 s | ~1.58 fps | ~37 s |

EEVEE works in `blender -b` (background) here. If a given machine's EEVEE can't
get a GPU context headless, pass `CYCLES` (CPU, slower) or run in the GUI Blender.

## Output

```
output/room_render/
  frame_NNNNN.png             render per camera, UPRIGHT (rotated 90° CW)
  annotated/frame_NNNNN.png   render with each visible object LABELLED on it
  sidebyside/frame_NNNNN.jpg  annotated render | the real photo, for comparison
  svg/frame_NNNNN.svg         labels as editable vector (id + edits as attributes)
  scene_manifest.json         the editable object map (once)
  render_manifest.json        per-view: visible objects + their label points
  room.glb                    room.py's own export (side effect of build)
```

Renders are rotated **90° clockwise** so they're upright (ARKit stores the
landscape sensor frame). numpy-only (Blender has no PIL); pass `rotate=False` to
keep the raw frame.

## Agent-facing labels (what each view shows + how to edit it)

The idea is dead simple: an LLM/agent looks at the **labelled render next to the
real photo** and knows directly which object is which and how to edit it — no
coordinate math. Generate the labels after rendering (needs Pillow, NOT Blender):

```bash
python3 authoring/views/room_render/annotate_views.py <out_dir> <scan_dir> [frames]
```

**The labelled image** (`annotated/` + the `sidebyside/` comparison): every
object that's actually visible gets its **room.py id drawn on it** (`Table0`,
`Chair2`, `Wall2`, `Window0`, …) — coloured text with a dark halo, **no boxes or
fills**, so the render stays clear. "Visible" is decided honestly: furniture is
labelled only where enough of it is on-screen and unoccluded (the label sits on
the part you can see); the wall/floor/ceiling you face is labelled at its centre.

**`scene_manifest.json`** — emitted once, the *map of what exists & how to edit
it*. Each object: its stable `id` (the same handle the edit API takes),
`category`, world `center`/`size`/`bbox`, `top_z`, `placeable_surface`,
`source_glb`, and the `edits` that make sense for it (table → `move_xy, rotate_z,
scale, swap_glb, recolor, place_on_top(z=…)`; door → `articulate(open/close),
swap_glb, recolor`; wall → `recolor, retexture`).

`render_manifest.json` backs the labels (per view: visible objects with their
`label_pct`, `dist_m`, `edits`). The **SVG** carries the same — each label is
`<g data-id=… data-category=… data-edits=…><text>…</text></g>` — so it's both
visual and machine-editable.

Typical loop: show the VLM `sidebyside/frame_NNNNN.jpg`; it spots a difference
("`Table0` is round in the render but the photo's desk is rectangular"), looks up
`Table0` in `scene_manifest.json` for the legal edits, and edits room.py.

**Iterating on labels without re-rendering:** the manifests can be regenerated
fast (no render) with the `describe` mode, then re-run `annotate_views.py`:

```bash
<blender> -b --python authoring/views/room_render/render_room_cameras.py -- "" "" output/room_render all EEVEE 1 describe
```
