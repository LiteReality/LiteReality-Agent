# System modules

We break down the compelent of the pipeline that converting real world scans into simmulation-ready 3D enviroments. This is not a 10 steps pipeline but The the split are four developing prupose, while the entire system in intergated and multi-stage loop checking to ensure a highqautiy, resulable simluation enviroments for robotic trianing


The system splits into the ten modules below. Every update should tie to exactly one of them — if
a change doesn't have an obvious home here, the map is wrong and should be fixed first.

```text
LiteReality
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
│      └── make the layout simulation-ready with the multi-stage layout agent (src/litereality_agent/pipeline/scene_init/layout)
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

