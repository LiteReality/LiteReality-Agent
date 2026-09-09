"""MESH-level QC: is each generated asset a usable mesh?

Runs per GLB during reconstruction, where the defects are TRELLIS artefacts — a floor slab
fused on, background welded in, floating fragments, a squat blob. The repair is to make the
asset again from a better reference image.

Distinct from `pipeline/room_qc`, which gates the assembled `Room.py` at the very end and asks
a different question — is this furniture in a sane PLACE — and repairs by nudging placements.
Same kind of gate, different subject and different scale, hence the qualified names.

    checks.py         the per-mesh geometry tests (qa_glb)
    chair_repair.py   the repair loop: regenerate the reference image, re-run TRELLIS, re-check
"""
