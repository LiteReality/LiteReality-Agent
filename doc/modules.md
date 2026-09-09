# System modules

LiteReality-Agent is an agentic system that turns a real-world scan into a simulation-ready 3D
environment. The modules below are split for development purposes. Every update should tie to
exactly one module. If a change has no obvious home here, the map is wrong and should be fixed
first.

The modules are grouped into three phases, in the order a run passes through them, plus the layers
that every phase uses. Phases depend forwards only: `author` reads what `measure` established,
`compile` reads what `author` produced, and nothing reads backwards.

```text
LiteReality-Agent
│
├── measure — establish what the room contains
│   │
│   ├── Scene understanding
│   │      └── layout and oriented 3D object boxes — currently straight from Apple RoomPlan
│   │
│   ├── Image selection
│   │      └── pick the reference frames for each detected object
│   │
│   ├── Object merging
│   │      └── merge objects that are built into each other, e.g. an oven that is built
│   │          within a cabinet — two objects become one
│   │
│   └── Grouping
│          └── identical objects in the scene — mostly chairs for now
│
├── author — produce an asset for every measured object
│   │
│   ├── Layout agent
│   │      └── make the layout simulation-ready with the multi-stage layout agent
│   │
│   └── Object reconstruction
│          ├── generative objects
│          └── procedural objects
│
├── compile — link the authored assets into one deliverable scene
│   │
│   ├── Room definition
│   │      └── the room built from its objects and shells
│   │
│   ├── QC checks
│   │      └── the quality gates the scene has to pass at the end
│   │
│   ├── Integration
│   │      └── reconstruction results into Blender, MuJoCo, three.js, etc.
│   │
│   └── Simulation export
│          └── the room as a scene of bodies with mass, collision geometry and
│              joints — optional, and last: a room that never reaches a
│              simulator is still a finished room
│
└── across every phase — layers, not stages
    │
    ├── Agent tools
    │      └── the tools we define for the agent to use
    │
    ├── Harness
    │      ├── the design of the agentic loops
    │      └── the prompts
    │
    ├── Viewing and telemetry
    │      ├── viewers for interactive viewing
    │      └── agent trace recording and live viewing
    │
    └── Editing
           └── keep editing the room after it is built, with text or images
```

## Why the grouping

The previous version of this map numbered ten modules 0 through 9, then noted that they "are not
ten sequential parts of the pipeline". The numbers said otherwise, so they are gone: a module is
referred to by name, which is also stabler than a position in a list.

Three of the ten were never stages. **Agent tools** and **Harness** are used by every phase that
runs a model — which is what makes the harness swappable, and why `claude/` and `codex/` can both
back the same pipeline. **Editing** happens after a run finishes, not during one. Grouping them
separately says so.

One module split. The **Layout agent** was doing two jobs: merging objects that are built into each
other, which establishes what the room contains, and making the layout simulation-ready, which
changes it. The first is now **Object merging** under `measure` — where `merge_boxes.py` already
lives, under `ingest`. The second stays the layout agent, under `author`.

## Where each module is written up

- **QC checks** — [`qc/`](qc): [collision check](qc/collision_check.md),
  [supporting relationship](qc/supporting_relationship.md).
- **Agent tools** — [`tools/`](tools): one page per tool, named after the tool.
  [fetch material](tools/fetch_material.md), [image selection](tools/image_selection.md),
  [rendering and compare](tools/rendering_and_compare.md), [3D generation](tools/threed_gen.md).
- **Integration** and **Simulation export** — [`integration/`](integration):
  [the room as a MuJoCo scene](integration/mujoco.md),
  [what an asset must carry](integration/metadata.md).
- **Layout agent** — not written up yet.

Directory names are lowercase, so a link is guessable from a module name. The
sim-ready pages moved out of `Sim-Ready-intergration/`, which was misspelled.

## How the phases map onto the source tree

The source tree predates this map and is not moved by this change. Today:

| phase | packages |
|---|---|
| measure | `pipeline/scene_init/ingest` |
| author | `pipeline/scene_init/reconstruct`, `pipeline/scene_init/seed`, `pipeline/realism_authoring/author` |
| compile | `pipeline/room_qc/publish`, `pipeline/simulate`, `room_ops` |
| across every phase | `agent/`, `agent/providers`, `models/`, `telemetry.py` |

`scene_init` spans two phases because `reconstruct` and `seed` produce assets
rather than establish what the room contains. Aligning the tree with the map is
a follow-up: `scene_init` and `realism_authoring` are named in 114 files, and
two open pull requests touch more than seventy files each, so moving packages
now would conflict with almost everything in flight. The map is agreed first,
the move follows.
