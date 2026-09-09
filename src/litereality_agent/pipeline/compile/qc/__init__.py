"""ROOM-level QC: is the assembled scene in a sane state?

This is the pipeline's final gate. It checks the authored `Room.py` and its compiled `Room.glb`.
Repairs adjust object placement but never change an object.

Distinct from `scene_init/reconstruct/mesh_qc`, which gates each generated asset as it is made
and asks whether the MESH is usable. Same kind of gate, different subject and scale.

    checks.py   report the violations (no writes)
    fix.py      resolve them from the SHELL boxes (fast, coarse)
    correct.py  resolve them from the TRUE meshes in the compiled Room.glb (needs python-fcl)
    publish/    compile the final room and the artifacts a human reviews it by

`publish/` lives here rather than under `realism_authoring` because it is not authoring: by the
time it runs the room is finished being built. It gates the room and produces what you LOOK at.
Running last in the flow is not the same as belonging to the flow that precedes it.
"""
