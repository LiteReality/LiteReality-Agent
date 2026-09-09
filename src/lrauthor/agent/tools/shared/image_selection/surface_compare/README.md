# surface_compare — render vs real-stitch, per surface (locked in)

For a scan's BUILT room (stage-2 `Room.py`), render every **wall / floor / ceiling** head-on
(orthographic, front-on) and pair each with its **room_pipeline rectified stitch** — a
`RENDER | REAL` sheet, identically oriented. Fully **deterministic** (no LLM). Use it to verify
stage-2: do the reconstructed materials + fixtures match the real surface?

## Run

```bash
python -m lrauthor.authoring.views.image_selection.surface_compare.run <scan> \
    [--room ROOM_DIR] [--stitch STITCH_DIR] [--out OUT_DIR] [--ppm 160] [--max-frames 60]
```
- `<scan>` resolves the latest `run/<scan>/scene_init/scene_stage/stage_2/iteration_*/room/Room.py` (or `--room`).
- Output (default `run/<scan>/scene_init/scene_stage/_surface_compare/`): `render/` (per-surface PNGs +
  `orient.json`), `stitch/`, and `<scan>_surface_compare.jpg` (the sheet).
- Blender: `$BLENDER` → repo default → `which blender`.

## How it works (4 steps)

```
1. STITCH    room_pipeline.py        rectified real image per surface (skipped if present)
2. FLOOR-UV  floor_uv.py             dump room_pipeline's floor u/v (dominant_room_axis)
3. RENDER    render_ortho.py (Blender) ortho front-on render of every surface
4. SHEET     sheet.py                composite RENDER | REAL, apply flip flags
```

## The geometry that makes it correct (don't regress these)

- **Z-up**: derive each surface plane from the BUILT Blender mesh, not raw `room.usdz` (Y-up) → distortion.
- **wall = group** of all `Wall<N>(_|$)` meshes (solid + glazing/mullions/frames), so glazed
  partitions render and aren't mis-counted as fixtures; **normal = PCA thinnest axis** (largest
  face fails on mullioned/thin walls), **image-up = world-up projected into the plane**.
- **floor/ceiling orientation** must come from the stitch's own plane — `floor_uv.py` dumps
  room_pipeline's `floor_ceiling_from_walls`/`dominant_room_axis` u/v (NOT `floor_plane_from_mesh`,
  90°-ambiguous on square rooms); the render transforms it `(x,y,z)_usd → (x,-z,y)_blender`.
  Ceiling reuses the floor u/v and the render is flipped (`rflip`, viewed from below).
- **furniture hidden** by scene-registry category; render framed to the whole plane (wall ∪ its fixtures).
- **flip flags** (`orient.json`): `sflip` flips the stitch to match the render (room_pipeline lays
  walls out by world-X, mirroring some); `rflip` flips the render (ceiling).

## Files
`run.py` (orchestrator) · `render_ortho.py` (Blender) · `floor_uv.py` (venv) · `sheet.py` (venv)
