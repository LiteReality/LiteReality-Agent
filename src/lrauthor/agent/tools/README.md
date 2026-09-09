# authoring/tools — the agent's capability tools

The tools the authoring / materials / QC agent is handed each run, on top of `Read`/`Edit`/`Write`/
`Glob`. `default_registry.build_default_registry()` assembles them into a `ToolRegistry`, and
`author.build_capability_server` exposes them to the claude_agent_sdk as `mcp__cap__*`.

Shared plumbing: `base.py` defines tools and strict pydantic parameters. `registry.py` builds the
tool registry and OpenAI schemas. `_scene.py` resolves the active room, and `_vlm.py` makes a
self-contained vision call.

**Where the backing code lives.** Each tool owns its source under its own folder:
[`render/source/`](render/source/) is the render/annotate engine plus `wall_refs` and
`surface_compare` (CLI: `python -m lrauthor.agent.tools.render.source <mode> ...`), and
[`select_views/source/`](select_views/source/) holds `select_for.py`.

Three primitives are shared rather than owned by one tool, because more than one needs them and a
per-tool copy is a bug waiting to happen:

| Module in [`shared/`](shared/) | Why it is shared |
|---|---|
| `config.py` | harness paths and knobs — four tools plus the pipeline's `evidence.py` |
| `scan.py` | `scan_from_room` / `config_for`; `tests/test_scan_inference.py` pins that exactly one copy exists |
| `overlay.py` | wall-plane projection — both `compose` and `select_views` |
| `image_selection/` | surface geometry + head-on comparison — reaches `render` (via `wall_refs`) and `select_views` (via `overlay`) |
| `stitch_wall_image/` | rectified wall stitches — `overlay`, `wall_refs`, `surface_views`, and the pipeline's `evidence.py` |

Some tools still wrap `room_ops` rather than absorbing it, deliberately: `compile` →
`room_ops.compile_room` (the format's own compiler), `fetch_material` →
`compile/fetch_textures` (`textures.json` is part of the Room format contract).

What is left in `room_ops/rendering/` is genuinely room-ops work: `room_render/` (capture-pose
renders, ranking, render-vs-capture) and the object turntables.

## Capability tools

| Tool | What it does |
|---|---|
| [`fetch_material`](fetch_material/) | real Poly Haven PBR set (diffuse+rough+normal), optionally recoloured |
| [`select_views`](select_views/) | best frames for `room` / a wall / an object — never guess frames |
| [`render`](render/) | render\|photo comparison for ONE target layer (auto-frames + recompiles) |
| [`grid`](grid/) | metric ruler on a surface stitch — READ a fixture's (u, z) in metres instead of estimating |
| [`critic`](critic/) | VLM **grade** vs a goal → `{pass, score, issues}` — the judge |
| [`check_collisions`](check_collisions/) | TRUE-MESH clash/containment (default) from Room.glb — object↔object, through-wall, outside-room, grounding + robotics-style articulated door/window checks (see [`../scene_collision.py`](../scene_collision.py)); each with a metric snap/resize fix. `method='box'` = fast SHELL-box fallback when no glb |

`compile/` stays a module (render recompiles through it) but is not a registered tool — the agent
compiles via `render`, not a standalone call.

## retired tools

The old **closed-loop driver** (`agent/harness/loop.py`, `run_agent(tools_mode="closed")`) drove an
"11 primitives + 3 composites" closed set with a raw-API loop. That driver is gone; the live stages
run on the claude_agent_sdk with the capability tools above. The retired primitives (`read_code`,
`edit_code`, `read_image`, `plan`, `fetch_object`) and composites (`inspect`, `fix`, `survey`) are
parked at the repo-root `legacy/lrauthor/realism_authoring/tools/` for reference — nothing in the
live path imports them.
