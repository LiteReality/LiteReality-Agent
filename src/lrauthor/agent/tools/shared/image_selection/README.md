# image_selection — surface geometry and head-on comparison

Shared tool source: the wall/floor/ceiling geometry the render tools reason about, and the
head-on RENDER | REAL sheets used to verify a built room against its capture.

## What is here

| Module | Used by | What it provides |
|---|---|---|
| `surface_views.py` | [`../overlay.py`](../overlay.py) · [`../../render/source/wall_refs.py`](../../render/source/wall_refs.py) | wall corners and edge projection (`wall_corners`, `_proj_edge`), floor/ceiling derivation, per-surface frame `analyze`/`select`, the boxed and labelled `highlight` tile, and the tiny-wall / blur / coverage thresholds |
| `surface_compare/` | [`../../render/source/surface_compare.py`](../../render/source/surface_compare.py) | `render_ortho.py` renders each surface head-on in Blender; `sheet.py` composes it against the real stitch |

Scoring principle, unchanged: rank a view by how well the target **fills the frame (size) × is
centred × is visible**, then take the best few.

## What used to be here

A three-layer reference-selection package — `select_references.py` (a ROOM/WALLS/OBJECTS
orchestrator writing one `references.json`) and `object_views.py` — plus `surface_views.run()`
and its contact sheet.

All of it was reachable from nothing, and it duplicated the [`select_views`](../../select_views/)
tool, which does room / wall / object frame selection for the model at authoring time. It was
removed rather than kept as a second implementation of the same idea that nothing exercised.

What remains is the part that is *not* frame selection: surface geometry, and comparison sheets.

## Note on the scorer

Three implementations of "how well does this frame see this target" still exist:
`select_views.quality`, `surface_views.analyze`/`select`, and a hand-maintained copy in the init
preprocessing (`pipeline/scene_init/ingest/preprocessing/vendor/litereality/object_image_extraction.py`, whose
comment asks you to keep it consistent with `select_views.quality` by hand). Worth unifying, but
it spans `agent` and `pipeline`, so it needs a shared home first.

## surface_compare/

Renders each wall/floor/ceiling of a BUILT room head-on and pairs it with its rectified real
stitch. See [`surface_compare/README.md`](surface_compare/README.md) for the geometry — Z-up
planes, wall = group + PCA normal, floor/ceiling oriented from the dominant room axis, flip flags.
