# System modules

These are the components of the pipeline that turns a real-world scan into a simulation-ready 3D
environment. The ten below are a split for development purposes, not ten sequential steps — the
system is one integrated, multi-stage loop that keeps checking itself, so that what comes out is a
high-quality, reusable environment for robot training.

Every update should tie to exactly one module. If a change has no obvious home here, the map is
wrong and should be fixed first.

```text
LiteReality-Agent
│
├── 0. Scene understanding
│      └── layout and oriented 3D object boxes — currently straight from Apple RoomPlan
│
├── 1. Image selection
│      └── pick the reference frames for each detected object
│
├── 2. Layout agent
│      ├── merge objects that are built into each other, e.g. an oven that is built
│      │   within a cabinet, two object will be merge into one
│      └── make the layout simulation-ready with the multi-stage layout agent
│
├── 3. Grouping
│      └── identical objects in the scene — mostly chairs for now
│
├── 4. Object reconstruction
│      ├── generative objects
│      └── procedural objects
│
├── 5. Agent tools
│      └── the tools we define for the agent to use
│
├── 6. QC checks
│      └── the quality gates the scene has to pass at the end 
│
├── 7. Harness
│      ├── the design of the agentic loops
│      └── the prompts
│
├── 8. Infrastructure
│      ├── room definition from objects and shells
│      ├── viewers for interactive viewing
│      └── agent trace recording and live viewing
│
└── 9. Integration
       ├── reconstruction results into Blender, MuJoCo, three.js, etc.
       └── editing tools — keep editing the room after it is built, with text or images
```

## Where each module is written up

- **QC checks** — [`QC/`](QC): [collision check](QC/collision_check.md), [supporting relationship](QC/supporting_relationship.md).
- **Agent tools** — [`Tools/`](Tools): one page per tool, named after the tool.
- **Layout Agents** 
- **Sim-Ready-Intergation**

