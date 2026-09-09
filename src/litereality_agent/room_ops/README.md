# Room ops

`room_ops` owns LiteReality's portable, editable room representation:

```text
Room.py + scene.json + object definitions/assets
```

It provides the operations needed to create, inspect, render, validate, compile, and export that
representation. Pipeline decisions and model/agent execution do not belong here.

## Public API

```python
from litereality_agent.room_ops import compile_room, export_scene

room = export_scene("office-elliott")
glb = compile_room(room)
```

The equivalent module entry points are:

```bash
uv run python -m litereality_agent.room_ops.export.export_room --scan office-elliott
uv run python -m litereality_agent.room_ops.compile.build_from_room --room /path/to/Room
```

## Package layout

```text
room_ops/
├── api.py                 stable Python API
├── manifest.py            scene.json schema and package discovery
├── paths.py               room-ops input/output path resolution
├── surfaces.py            Room.py surface discovery
├── procedural_materials.py
├── viewer.py              self-contained Three.js HTML export
├── export/                capture/reconstructed assets → editable Room.py,
│                          and the authored room → MuJoCo (`mujoco_scene`, `mujoco_shake`)
├── compile/               Room.py → Room.glb/Room.blend
└── rendering/             render and reference-view utilities
```

## Format contract

The editable room directory is the source of truth:

```text
Room/
├── Room.py                semantic shell and object placement program
├── Room.md                editing guide
├── manifest.json          object-to-RoomPlan mapping
└── Objects/
    ├── Procedural/<name>/  object.py, object.md, textures.json
    │   └── sim/            the object's own physics: mass, inertia, friction,
    │                       joints and convex colliders (+ URDF)
    └── Static/<name>/      source GLB plus a uniform object.py wrapper
        └── sim/            the same, for a generated mesh
```

The `sim/` sidecar is what makes a Room package a complete statement of the room including how it
behaves. `export/mujoco_scene.py` reads it rather than re-deriving mass, friction, colliders and
hinge pivots at export time — see [`docs/integration/mujoco.md`](../../../docs/integration/mujoco.md).

Compilation produces regenerable output separately:

```text
room_preview/
├── Room.glb
├── Room.blend
├── room_layout.json
└── Object/
```

`Room.py` embeds the room shell as readable geometry: wall endpoints, openings, floor/ceiling
height, and object bounding boxes. Procedural objects retain editable Blender build code and texture
recipes; static neural assets retain their source GLB.

Blender compilation is invoked only by explicit compile/render operations. Importing
`litereality_agent.room_ops` does not start Blender.
