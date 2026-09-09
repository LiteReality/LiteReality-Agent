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
│          └── the room as a scene of bodies, each with mass, collision geometry
│              and a joint — optional, and last
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
