# image_render_tool

Build **render-vs-real comparison images on demand** for a room. Everything renders the
**current** room (a `.../room/Room.py`) and pairs it with the **real capture evidence**. The scan
is inferred from the room path, so you only pass the room dir.

## Five modes

Four **frame** modes (render | real photo, from the same camera pose — so the annotation is
drawn on **BOTH** sides and lines up), plus a **wall-reference** mode:

| Mode | Call | What it highlights |
|---|---|---|
| **scene** | `render_scene(room, frames)` | every visible object (walls, doors, windows, furniture) as a chip |
| **wall** | `render_wall(room, frames)` | **all** visible walls — outline + name |
| **wall-focus** | `render_wall_focus(room, frames, "Wall2")` | **ONLY** that one wall — outline + name, both sides |
| **object** | `render_object(room, frames, "Table0")` | ONE object — a bbox + its name to the side |
| **ref** | `render_wall_reference(room, walls)` | per wall: head-on `ortho render \| stitch`, then a WALL-FOCUSED `render \| real` row per covering reference frame |

- The four frame modes annotate **both** the render and the real photo (same pose → same
  positions), so you can match object-for-object between synthetic and real.
- **scene** chips: dark rounded box + category-coloured text; walls placed at their true
  projected centre; nearer objects first; chips nudged apart so they never overlap.
- **wall**: every visible wall's projected outline + name chip.
- **wall-focus**: like **wall**, but isolates a SINGLE named wall — only that wall is outlined
  (orange) + named, on both sides. Use it when working one wall at a time (e.g. stage-2 wall
  fixtures) so the other walls' outlines don't clutter the comparison.
- **object**: a rectangle around only that object (from its visible extent) + its name on a chip
  to the side.
- **ref**: wall-focused — only the wall of interest is outlined on each reference frame, matching
  the boxed real reference tile. Each comparison is a **separate image**: the head-on
  `ortho | stitch` under the wall name (e.g. `Wall0`), plus one file per covering reference frame
  keyed `Wall0_frame_00034` — so `render_wall_reference("Wall0")` returns several entries, not one
  tall stacked sheet.

> **wall-focus / wall gotcha — the wall must be VISIBLE in the frame you pass.** The outline is
> the wall's geometry projected into that frame's camera; if the wall isn't in view, the pair is
> still produced but with **no outline** (that means "not visible here", not a failure). Pick
> frames that show the wall — eyeball with `render_wall(room, frames)` first (all walls named),
> or find them without a Blender render:
>
> ```python
> from lrauthor.agent.tools.render.source.compose import _config_for, _scan_from_room
> from lrauthor.agent.tools.render.source import _overlay
> config = _config_for(_scan_from_room(room))
> planes = _overlay.load_planes(config)
> frames = [int(jf.stem.split("_")[1]) for jf in sorted(config.SCAN_DIR.glob("frame_*.json"))
>           if "Wall2" in _overlay.project_walls(jf, planes, config, 864, 576)]
> ```

`render_frames` / `render_surface` remain as back-compat aliases of `render_scene` /
`render_wall_reference`.

## Call it (Python)

```python
from lrauthor.agent.tools.render.source import render_frames, render_surface

room = "run/<scan>/scene_init/scene_stage/stage_2/iteration_2/room"

from lrauthor.agent.tools.render.source import (
    render_scene, render_wall, render_wall_focus, render_object, render_wall_reference)

# scene / wall — any number of frames at once
render_scene(room, [10])                  # one frame, all objects labelled (both sides)
render_scene(room, [10, 12, 18, 24, 30])  # five frames, rendered together
render_wall(room, "0-40")                 # a range;  "all" = every frame — walls only

# wall-focus — isolate ONE wall (both sides). Use frames where that wall is visible.
render_wall_focus(room, [34, 35, 36], "Wall2")

# object — bbox on ONE object (both sides)
render_object(room, [10, 20], "Table0")

# ref — per wall: ortho|stitch + wall-focused ref frames
render_wall_reference(room, "Wall0")
render_wall_reference(room, ["Wall0", "Wall2"])
render_wall_reference(room)               # all walls
```

Each call returns `{frame|wall: path_to_side_by_side_png}` and writes the PNGs under
`<room>/../_image_render_tool/<mode>/`.

## Call it (CLI)

```bash
python -m lrauthor.agent.tools.render.source scene  --room <room> --frames 10,20,30
python -m lrauthor.agent.tools.render.source wall   --room <room> --frames 10,20
python -m lrauthor.agent.tools.render.source wall-focus --room <room> --frames 34,35 --wall Wall2
python -m lrauthor.agent.tools.render.source object --room <room> --frames 10 --object Table0
python -m lrauthor.agent.tools.render.source ref    --room <room> --walls Wall0,Wall2
python -m lrauthor.agent.tools.render.source ref    --room <room>                # all walls
```

## Prerequisites — what must already exist

All three layers rebuild the scene from `Room.py`, so they share a base requirement; layers 2/3
need the real-surface evidence on top.

| Needed by | Artifact | Where it comes from |
|---|---|---|
| **all layers** | `<room>/Room.py` | `uv run litereality run <scan> --through seed` |
| **all layers** | `scene_stage/_scene_assets/glb/*.glb` + manifest | object_init + pack (the placed objects) |
| **all layers** | raw capture frames `frame_*.jpg` / `frame_*.json` | the scan folder (ARKit capture) |
| **layer 2 & 3** | surface stitches `scene_stage/_harness/surface_ref/*_stitched.jpg` | harness `surface_reference` (once per scan) |
| **layer 3 only** | wall references (built on first call, cached) | `wall_refs` (auto, from `room.usdz` + frames) |

If the surface stitches are missing, layer 2/3 silently skip that wall (no stitch to compare
against). Layer 1 needs none of the harness evidence — just the room + assets + frames.

## Timing — how long to wait (measured, ~59-frame scan, 7 objects)

The **scene build in Blender (~5-6s) dominates and happens once per call**, so batching is far
cheaper than repeated calls.

| Call | Time | Note |
|---|---|---|
| `render_frames(room, [10])` | **~6 s** | scene build + 1 render |
| `render_frames(room, [10,20,30,40,50])` | **~12 s** | +~1.3 s per extra frame — **5 frames in ONE call ≈ 12 s vs ~30 s as five calls** |
| `render_surface(room, "Wall0")` | **~6 s** | renders ALL surfaces in one pass |
| `render_surface(room)` (all walls) | **~7 s** | same render, just more compositing — asking for all walls is nearly free |
| `render_surface(room, "Wall0", with_refs=True)` | **~6 s** | + a **one-time ~5 s** wall-refs build on the first call per scan (cached after) |

**Rule of thumb for agents:** ~6 s per call floor; pass a **list** of frames / walls in a single
call to amortize the build. Budget ~5-15 s for a typical request, +5 s the first time you use
`with_refs` on a scan.

## Notes

- Needs Blender (`$LITEREALITY_BLENDER` or `$BLENDER`) — layers 1 and 2 each do one Blender pass.
- Layer 2/3 rely on the scan's **surface stitches** (`scene_stage/_harness/surface_ref/`), built
  once per scan by the harness. Layer 3 also builds the **wall references** on first use.
- Real photos are the ARKit landscape frames rotated upright, matching the render orientation.
- This wraps the harness's own machinery (`render_room_cameras`, `surface_compare`, `wall_refs`) —
  it's the same rendering the loop uses, just exposed as an on-demand comparison tool.
