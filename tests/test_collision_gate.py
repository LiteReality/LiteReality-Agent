"""room_qc/collision.py — the read-only collision gate.

The gate's contract is its exit code, so that is what these pin: a clash fails, a clean room
passes, and grounding/placement findings are REPORTED but do not fail. That last one is a
deliberate policy, not an oversight — a `floating` object means a missing support and
`outside_room` means the scan's wall loop is wrong, and neither is something the collision
resolver can repair, so failing on them would block every publish on an upstream problem.
"""

from __future__ import annotations

import trimesh

from litereality_agent.pipeline.room_qc import collision
from litereality_agent.room_ops import glb_meta


def _room(tmp_path, chair_x: float):
    """A 1-wall room with a table and a chair; `chair_x` decides whether they interpenetrate."""
    scene = trimesh.Scene()
    for name, pos, ext in (
        ("Floor0", (0, -0.05, 0), (8.0, 0.1, 8.0)),
        ("Wall0", (0, 1.25, -3.0), (8.0, 2.5, 0.1)),
        ("Table0", (0, 0.4, 0), (1.2, 0.8, 0.8)),
        ("Chair0", (chair_x, 0.4, 0), (0.5, 0.8, 0.5)),
    ):
        m = trimesh.creation.box(extents=ext)
        m.apply_translation(pos)
        scene.add_geometry(m, node_name=name, geom_name=f"{name}_mesh")
    glb = tmp_path / "Room.glb"
    glb.write_bytes(scene.export(file_type="glb"))
    glb_meta.stamp(glb, {
        "floor_z": 0.0, "ceiling_z": 2.5,
        "walls": {"Wall0": {"start": [-4.0, 3.0], "end": [4.0, 3.0], "thickness": 0.1}},
        "objects": {"Table0": {"category": "table", "yaw": 0.0},
                    "Chair0": {"category": "chair", "yaw": 0.0}},
    })
    return glb


def test_clean_room_passes(tmp_path):
    glb = _room(tmp_path, chair_x=3.0)          # chair well clear of the table
    findings = collision.check(glb)
    assert collision.report(glb, findings, quiet=True) == 0


def test_interpenetrating_pair_fails(tmp_path):
    glb = _room(tmp_path, chair_x=0.4)          # chair driven into the table
    findings = collision.check(glb)
    clashes = [f for f in findings if f["kind"] == "object_clash"]
    assert clashes, "overlapping meshes must be reported"
    assert collision.report(glb, findings, quiet=True) >= 1


def test_grounding_is_reported_but_does_not_fail_the_gate(tmp_path):
    """A floating object is a missing SUPPORT, not a clash — report it, do not block the build."""
    findings = [{"id": "Sink0", "kind": "floating", "detail": "0.55 m above floor"}]
    assert collision.report(tmp_path / "x.glb", findings, quiet=True) == 0


def test_placement_is_reported_but_does_not_fail_the_gate(tmp_path):
    findings = [{"id": "Chair2", "kind": "outside_room", "detail": "outside the outline"}]
    assert collision.report(tmp_path / "x.glb", findings, quiet=True) == 0


def test_wall_clash_fails(tmp_path):
    findings = [{"id": "Bed0", "kind": "wall_clash", "wall": "Wall6", "penetration_m": 1.797}]
    assert collision.report(tmp_path / "x.glb", findings, quiet=True) == 1


def test_gate_needs_no_room_py(tmp_path):
    """The whole point: a stamped glb is checkable on its own."""
    glb = _room(tmp_path, chair_x=0.4)
    assert not (tmp_path / "Room.py").exists()
    assert collision.check(glb), "a stamped glb must be checkable with no Room.py present"
