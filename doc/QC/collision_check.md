# Rules

Once a `Room.py` reconstruction is finished, the collision check makes sure nothing collides with anything else.

We have the following entities:

1. **Walls and wall fixtures.** The whole wall plus everything mounted on it is assumed fixed to the
   wall. They don't move and they don't interact.
2. **Windows and doors.** Built independently, so they never clash with the wall.
3. **Detected furniture.** Each one is its own collision body. 
4. **Objects the agent adds** that weren't detected. Also their own individual bodies.

- Everything is inside the room. Nothing sticks out.
- Objects don't clash with each other.
- Objects don't clash with walls.



In the layout stage we already use layout agent to make the layout simulation-ready. the new intorucing of the the collison are mostly from

1. chair and table as they are allow to calssh at bbox level.
2. newly authored objects that is not part of the raw detections. 


Small items the agent puts on tables and shelves are out of scope for now — but they must sit
attached to the table, not embedded into it.

# The deterministic function

```python
check_glb("…/room_preview/Room.glb")   ->  [findings]
```

```bash
python -m litereality_agent.pipeline.room_qc.collision --glb <Room.glb>
# exit 0 = pass, 1 = fail
```

**One file in.** The glb carries everything the check needs — wall centrelines, floor and ceiling
height, each object's category — written into it as glTF `extras` when it is built. That matters
because the alternative was a glb plus whichever `Room.py` was newest, and those two can describe
different moments. A glb built before this existed still works: the check falls back to reading
`Room.py`.

**What fails, and what is only reported.** Not every finding is the collision check's to fix:

| finding | fails the gate? | why |
|---|---|---|
| `object_clash`, `wall_clash`, `opening_blocked` | **yes** | two things occupy the same space |
| `floating`, `sunk`, `above_ceiling` | no | usually a MISSING support object, not a misplaced one |
| `outside_room` | no | the scan's wall loop is wrong — upstream of us |

Failing on the last two would block every publish on a problem this stage cannot repair.

## What a failure hands the fixer

Every clash comes with the metric move that clears it, so the repair step never has to guess:

```json
{ "id": "Chair1", "kind": "object_clash", "with": "Table0",
  "overlap_m": 0.209,
  "fix": { "move":    { "id": "Chair1", "world_dxdy": [0.099, 0.032], "dist_m": 0.209 },
           "or_move":  { "id": "Table0", "world_dxdy": [-0.099, -0.032] },
           "or_resize":{ "shrink_m": 0.209 } } }
```

Contact is decided from the real triangles, because that is the only model that lets a chair tuck
under a table. The distance is measured from a convex decomposition of the same meshes, because the
raw meshes are not watertight and a penetration depth taken from them is meaningless. The convex
model is never allowed to decide *whether* two things touch — only how far to move them — so its
approximation can size a real clash but can never invent one.

## Fixing is a separate function

The check reports. `resolve` repairs, and it is the only thing here that writes `Room.py`:

```
detect  ->  nudge  ->  detect again      until clean, or provably stuck
```

It runs in memory (a fix only slides objects along the floor, so the same slide applied to the
meshes is what a rebuild would produce), then writes the accumulated moves once. Afterwards the room
is rebuilt and the gate is run again — against the real artifact, not against the simulation.

When it cannot finish it says which kind of stuck it is, because "give me more rounds" and "this
needs a human" look identical from a clash count alone:

- `converged` — nothing left
- `capped` — the move needed is bigger than a nudge; that is a placement error upstream
- `nothing_movable` — both pieces are anchored to walls
- `max_rounds` — hit the iteration limit, worth retrying
