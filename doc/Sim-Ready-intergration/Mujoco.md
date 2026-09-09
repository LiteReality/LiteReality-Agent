# The room as a MuJoCo scene

`Room.glb` is a picture of a room. One file, every object baked into it, no notion of what is
furniture and what is wall, no mass, no joints. Training anything in it needs the opposite — a
scene of **bodies**, each with its own collision geometry, its own mass and inertia, and a joint
(or deliberately none) saying how it may move.

The `simulate` stage produces that:

```bash
uv run litereality stage simulate run/<scan>          # -> realism_authoring/mujoco/scene.xml
uv run litereality stage simulate run/<scan> --shake  # ...and measure what moves
uv run litereality run <scan> --through simulate      # the whole pipeline, ending here
```

**Without the authoring stage at all.** Authoring adds materials, wall fixtures and the small
objects — appearance and clutter. The physical scene is complete without any of it: the shell with
its openings cut out, and every reconstructed object in its measured box carrying the mass,
friction, colliders and joints its own build compiled. `--from-seed` exports that room instead, out
of `scene_init`, into `mujoco_seed/`:

```bash
uv run litereality stage simulate run/<scan> --from-seed
```

It is the fastest way to get a scan into a simulator, and the cleanest test of the objects
themselves — nothing in the scene came from anywhere but the reconstruction. Measured over 5 s of
gravity with nothing touching them, four scans exported this way:

| room | bodies | free | articulated | mass | drift over 5 s |
|---|---|---|---|---|---|
| Office-Elliott | 15 | 5 | 4 | 172 kg | 91 mm |
| fallside-kitchen | 24 | 4 | 9 | 313 kg | 7 mm |
| fallside-office | 12 | 3 | 2 | 515 kg | 107 mm |
| MIL-Meeting | 32 | 16 | 6 | 484 kg | 1234 mm |

MIL-Meeting is the outlier and it is not an export fault: one chair was scanned 44 mm inside a
window reveal, and a 3.6 kg body squeezed out of static geometry travels. That is authored (here,
scanned) interpenetration, and it belongs upstream in the layout repair.

Directly, without the pipeline:

```bash
uv run python -m litereality_agent.room_ops.export.mujoco_scene --room run/<scan>/realism_authoring/room
uv run python -m litereality_agent.room_ops.export.mujoco_shake --scene .../mujoco/scene.xml --cutaway --video shake.mp4
```

## Where the physics comes from

Nothing in the export invents a physical number that the assets already state. Four files are read
together, and each answers the question it is the authority on:

| file | what it settles |
|---|---|
| `Room.py` → `SHELL` | walls, openings, floor and ceiling heights — metric and exact |
| `room_layout.json` | every object's pose and category, and its `rests_on` / `attached_to` |
| `Room.glb` | the appearance, per named handle |
| `Objects/<name>/sim/<name>.physics.json` | **the object's own physics** |

That last one is the difference between a scene that happens to load in MuJoCo and a scene that is
sim-ready. It is written at reconstruct time by
[`models/object_generation/sim`](../../src/litereality_agent/models/object_generation/sim), from
the GLB the recipe actually built, and gated by a solver *before* the object was ever placed in a
room. It carries, per link:

- **mass** — occupancy density over the object's own bounding box, or a figure the recipe read off
  a real label, then split across the links by volume;
- **inertia** and **centre of mass**, integrated from the mesh;
- **friction** and **restitution**, from the material the recipe deliberately chose — glass grips
  at 0.30, carpet at 0.85;
- **convex colliders**, already decomposed;

and per joint: **type, axis, pivot, limits, damping, joint friction, effort and velocity**.

### Placing it

An object is built in its own frame and then *fitted* into the RoomPlan box it belongs to — a
rotation about the vertical and a per-axis scale, because the box the scan measured and the
proportions the asset was built to rarely agree. So every number has to travel through one affine
on the way in. That affine is recovered exactly, not estimated:

```
T = T_room(node) · T_object(node)⁻¹
```

for any single node present in both `Room.glb` and the object's own GLB. trimesh renames duplicated
nodes with a random suffix on every load, so only names that are unique inside their own file can
be matched across the two — one is enough, and every other node then maps through it.

Two things follow from the scale being non-uniform:

- a **derived** mass is multiplied by the volume ratio (occupancy density over a box is a rule, so
  at a different size it is a different mass); an **authored** mass is a fact about the real object
  and survives unchanged;
- the inertia tensor is recomputed from the placed geometry rather than transformed, which is both
  simpler and more honest than pushing a tensor through a non-uniform scale.

The result is then **checked before it is trusted**: the union of the placed colliders has to land
on the bounding box the room recorded for that handle. It is derived from one node, so a room that
had moved an object's parts relative to each other would otherwise be silently mis-collided.

### When there is no sidecar

Objects the authoring agent wrote directly into `Room.py` — the mugs, the cables, the signage —
have no object package and therefore no compiled physics. So do rooms exported before any of this
existed. Those fall back to what the export did before: a mass from a category occupancy table, a
CoACD decomposition run at export time, one friction for the whole building, and a hinge pivot
inferred from the shape of the leaf.

That fallback is a worse scene, not a broken one, and `export_report.json` says exactly which
objects took it (`from_sidecar`, `no_sidecar`, `unplaceable`, `placement_rejected`). The stage
repeats the counts as warnings rather than burying them.

## The four kinds of body

Decided from the data, not from guessing at names.

| kind | what it is | how it is emitted |
|---|---|---|
| `structure` | walls, floor, ceiling | static **boxes** built from the `SHELL` numbers, with openings subtracted — a wall with a door becomes jamb / jamb / lintel, so the door has somewhere to swing |
| `articulated` | doors, windows, drawers | frame static; each moving part its own body on a hinge or slide, at the pivot the recipe authored |
| `free` | anything that `rests_on` something | a free joint, mesh colliders, and its own mass — the things that can be pushed and knocked over |
| `attached` | anything `attached_to` a wall or ceiling | static geometry. A socket is not a rigid body that can fall off |

Collision is never the visual mesh. MuJoCo collides a mesh as its **convex hull**, and the hull of
a table is a solid block — a chair tucked under it would be launched on the first step, and a shelf
would be a filled cupboard. Every concave body is therefore several convex pieces, and the original
mesh is kept as a visual-only geom.

The room itself is a body on three slide joints and a yaw hinge, driven by stiff position servos.
That is what makes `--shake` an earthquake rather than a change in the direction of gravity: the
floor really accelerates under the furniture, and a hung door really swings because its frame is
what is being shaken.

## Reading the result

`realism_authoring/mujoco/` holds `scene.xml`, its `meshes/`, and `export_report.json`. The
support relations the authoring pass recorded are what make the scene start **at rest**: an object
whose `rests_on` surface is genuinely under it does not fall, and one authored floating does. That
is reported rather than silently corrected — a scene that needs 200 steps to stop twitching is
telling you something true about the room.

## Related

- [`meta_data.md`](meta_data.md) — what an asset has to carry to be sim-ready.
- [`QC/supporting_relationship.md`](../QC/supporting_relationship.md) — how `rests_on` /
  `attached_to` are checked.
- [`QC/collision_check.md`](../QC/collision_check.md) — nothing may interpenetrate before it is
  simulated.
