"""ROOM-level QC: is the assembled scene in a sane state, and what does it look like?

This is the pipeline's final gate. It checks the authored `Room.py` and its compiled `Room.glb`.
Repairs adjust object placement but never change an object.

Distinct from `scene_init/reconstruct/mesh_qc`, which gates each generated asset as it is made
and asks whether the MESH is usable. Same kind of gate, different subject and scale.

DETECT — read only, never edits a room:

    collision.py  the gate: one glb in, findings + an exit code out
    checks.py     the box report over room_layout.json + SHELL (no compile needed)
    support.py    what holds every object up — nothing floats. Report only: a floating object is a
                  MISSING SUPPORT, not a misplaced one, so there is nothing safe to auto-fix.

REPAIR — the only things here that write `Room.py`:

    resolve.py    detect -> nudge -> re-detect until clean or provably stuck (what to run)
    correct.py    a SINGLE true-mesh pass; `resolve` is this in a loop
    fix.py        the SHELL-box resolver, plus the validators the others share

PUBLISH — compile the final room and the artifacts a human reviews it by:

    publish/      lives here rather than under `realism_authoring` because it is not authoring: by
                  the time it runs the room is finished being built. Running last in the flow is
                  not the same as belonging to the flow that precedes it.
"""
