# Architecture

LiteReality is one resumable pipeline with two explicit phases and five public stages:

```text
cli.py → PipelineRunner → scene_init                    → realism_authoring
                          ingest → reconstruct → seed      author → publish
```

Each stage exposes `run(context, options) -> StageResult`. `PipelineRunner` owns ordering,
prerequisites, reuse, failure handling, and state in `run/<scene>/.litereality/pipeline.json`.

## Package layout

```text
src/lrauthor/
├── cli.py               the supported command surface
├── settings.py          Pydantic environment settings
├── pipeline/            workflow ordering, state, and phase implementations
│   ├── scene_init/      capture ingest, reconstruction, and deterministic seed room
│   ├── realism_authoring/
│   │   ├── author/      evidence and optional realism passes
│   │   └── publish/     final compilation and viewer
│   └── room_qc/         final checks and placement correction
├── agent/               agent runners, tools (each owning its source), traces, and narration
│   └── providers/       which coding agent drives a session (`claude/` and `codex/`)
├── room_ops/            Room.py manifest, compilation, rendering, and export
├── models/              DINOv2, GroundingDINO, TRELLIS, and object generation
└── runtimes/            hosted execution transports such as Modal

scripts/                  standalone capture and publishing utilities
deploy/modal/             hosted model wrappers; never imported by application code
```

There are no top-level `services`, `adapters`, or `shared` packages. There is also no nested
`pipeline/stages` package. Phase decisions belong in `pipeline`. Reusable agent capabilities
belong in `agent`, and the portable room representation belongs in `room_ops`. Models own
inference, while runtimes own execution location.

`pipeline/realism_authoring/author` owns the optional post-authoring passes (`refine_objects`,
`materials`, and model-driven `quality`). They remain part of the authoring phase rather than
separate public pipeline stages. The agent session and its capability-tool framework live in
`agent/`, so adding another agent workflow does not expand a pipeline-stage package.

## Agent harnesses

The *harness* is the agent loop (file tools, capability tools, hooks, event stream); the *model*
is the brain inside it. They are independent knobs. `agent/providers/` is the seam for the first:
a `SessionSpec` goes in, a normalised event stream comes out, and `resolve(role)` picks the
harness from `LR_<ROLE>_PROVIDER > LR_AGENT_PROVIDER > claude`.

Harnesses are not feature-equivalent, so each declares a `supports` set and call sites branch on
it rather than assuming. Claude Code hosts capability tools in-process, can steer a session's
ending with a `PreToolUse` hook, reports cost, honours a tool allowlist, and surfaces file reads
as observable `Read` calls. Codex runs out-of-process: capability tools are bridged through
`agent/tools/mcp_server.py` over stdio MCP, the step budget degrades to a hard stop, shell access
cannot be withheld, and no cost is reported. `providers.describe()` prints those gaps into the
stage header so a degraded run never looks like a normal one.

Object refinement is the one pass that is Claude-only: its `render_object` tool is built per
object around live session state, so there is no registry entry for the stdio bridge to rebuild.
It fails with an explicit message rather than running without its only self-check tool.

## Agent capability tools

`agent/tools/default_registry.py` is the source of truth for agent tools. The authoring pass uses
this registry, and the materials and quality passes use the same one:

1. `fetch_material` searches Poly Haven for a PBR material and can recolour it.
2. `render` builds the room, selects useful views, and returns render and photo comparisons.
3. `critic` grades images against a stated goal and returns issues to fix.
4. `select_views` chooses capture frames for a room, wall, or object.
5. `grid` draws a metric grid over a surface reference.
6. `check_collisions` checks mesh or box collisions and suggests placement fixes.

The `compile` package is shared code used by `render`. It is not a registered agent tool. Shared
image selection and wall stitching code lives under `agent/tools/shared` because several tools and
the evidence stage use it.

Imports follow one direction, enforced by `tests/test_architecture.py`:

```text
cli.py → pipeline → agent → room_ops
                ↘ models → runtimes
                ↘ room_ops
```

Arrows point from caller to dependency. Models, room-ops code, and reusable agents never import
pipeline code; the pipeline owns the CLI adapters that bind scene-package arguments before calling
an agent. Runtime selection happens in `models/registry.py`; the pipeline consumes that small
composition boundary.

## Room ownership and quality control

`room_ops` owns the portable room representation and operations that work without pipeline state:

1. It reads and writes the Room.py manifest.
2. It compiles Blender and GLB files.
3. It renders rooms and exports assets.
4. It serves the walkable viewer.

Quality control has two scopes. `pipeline/scene_init/reconstruct/mesh_qc` checks each generated
asset before it enters the room. `pipeline/room_qc` checks the assembled room and corrects object
placement. The collision tool keeps its reusable geometry code under `agent/tools`, while the
pipeline decides what to do with each result.

## Configuration and runtimes

`LiteRealitySettings` uses `pydantic-settings` and loads values in this order:

```text
process environment > .env > models.env > typed defaults
```

Heavy inference is isolated from the main environment. A model package owns one inference path
regardless of where it executes. A local adapter can run it in a separately configured process;
a hosted adapter sends its request contract through `runtimes/`. Modal wraps canonical TRELLIS,
GroundingDINO, and DINOv2 inference under `deploy/modal`; DINO also supports an explicitly selected
local model environment.
Deployment files stay outside application source. The CLI and unit tests do not start DINO,
TRELLIS, Blender, or paid model calls.

## Output compatibility

Generated paths remain stable so previous work can be resumed:

```text
run/<scene>/
├── scene.json
├── scene_init/          crops, references, reconstructed objects, seed room
├── realism_authoring/  authored room and final deliverables
└── .litereality/        resume state
```

Some legacy artifact filenames such as `nano_banana_raw.png` are still read for compatibility;
they are data-format details, not supported providers or package boundaries.
