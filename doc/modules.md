# System modules

We split the system into the modules below. Every update should tie to one of them.

```
LiteReality
│
├── 1. Image selection
│      └── pick the reference images for each object
│
├── 2. Object reconstruction
│      ├── generative objects
│      └── procedural objects
│
├── 3. Grouping
│      └── identical objects in the scene, mostly chairs for now
│
├── 4. Agent tools
│      └── the customised tools we define for the agent to use
│
├── 5. QC checks
│      └── the quality gates the scene has to pass at the end
│
├── 6. Harness
│      ├── the design of the agentic loops
│      └── the prompts
│
├── 7. Infrastructure
│      ├── room definition from objects and shells
│      ├── viewers
│      └── agent trace recording and live viewing
│
└── 8. Integration
       ├── reconstruction results into Blender, MuJoCo, three.js, etc.
       └── editing tools — keep editing the room after it is built, with text or images
```
