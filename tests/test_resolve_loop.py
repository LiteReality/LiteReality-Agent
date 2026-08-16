"""room_qc/resolve.py — the detect -> nudge -> re-detect loop.

What matters here is not that it moves things, but that it TERMINATES for a knowable reason. A
clash count alone cannot tell "give me more rounds" from "this needs a human", and only one of
those is worth retrying — so `stop` is the field under test.

Built on synthetic boxes with a hand-written SHELL, so no Blender and no scan data.
"""

from __future__ import annotations

import json
import math

import pytest
import trimesh

from litereality_agent.pipeline.room_qc import resolve as R
from litereality_agent.room_ops import glb_meta


def _room(tmp_path, objects, walls=None, extra_shell=None):
    """Build a stamped Room.glb + Room.py from {name: (glb_xyz_pos, extents, category)}."""
    walls = walls if walls is not None else {
        "Wall0": {"start": [-5.0, 5.0], "end": [5.0, 5.0], "thickness": 0.1}}
    scene = trimesh.Scene()
    floor = trimesh.creation.box(extents=(20.0, 0.1, 20.0))
    floor.apply_translation((0, -0.05, 0))
    scene.add_geometry(floor, node_name="Floor0", geom_name="Floor0_mesh")

    shell_objects = {}
    for name, (pos, ext, cat) in objects.items():
        m = trimesh.creation.box(extents=ext)
        m.apply_translation(pos)
        scene.add_geometry(m, node_name=name, geom_name=f"{name}_mesh")
        # glb (x, y, z) -> SHELL (x, -z, y): centre in the SHELL plane, height as z
        shell_objects[name] = {"category": cat, "yaw": 0.0,
                               "center": [pos[0], -pos[2], pos[1]],
                               "size": [ext[0], ext[2], ext[1]]}

    glb = tmp_path / "room_preview" / "Room.glb"
    glb.parent.mkdir(parents=True, exist_ok=True)
    glb.write_bytes(scene.export(file_type="glb"))

    shell = {"floor_z": 0.0, "ceiling_z": 3.0, "walls": walls, "objects": shell_objects}
    shell.update(extra_shell or {})
    glb_meta.stamp(glb, shell)

    roomdir = tmp_path / "room"
    roomdir.mkdir(parents=True, exist_ok=True)
    # JSON-style, matching how the builder actually writes SHELL — `_replace_center` rewrites the
    # literal in place with a regex keyed on `"objects"`, so single-quoted repr() would not match.
    (roomdir / "Room.py").write_text("SHELL = " + json.dumps(shell, indent=2) + "\n")
    return roomdir


def test_converges_and_reports_why(tmp_path):
    """Two overlapping free-standing tables must be separated and the loop must say `converged`."""
    room = _room(tmp_path, {
        "Table0": ((0.0, 0.4, 0.0), (1.0, 0.8, 1.0), "table"),
        "Table1": ((0.5, 0.4, 0.0), (1.0, 0.8, 1.0), "table"),
    })
    p = R.resolve(room)
    assert p["stop"] == "converged"
    assert p["converged"]
    assert p["rounds"][0]["clashes"] >= 1, "must actually start dirty"
    assert p["rounds"][-1]["clashes"] == 0
    assert p["moves"], "convergence must come from real moves"


def test_already_clean_room_stops_immediately(tmp_path):
    room = _room(tmp_path, {
        "Table0": ((0.0, 0.4, 0.0), (1.0, 0.8, 1.0), "table"),
        "Table1": ((4.0, 0.4, 0.0), (1.0, 0.8, 1.0), "table"),
    })
    p = R.resolve(room)
    assert p["stop"] == "converged"
    assert len(p["rounds"]) == 1, "a clean room must not iterate"
    assert p["moves"] == []


def test_overlap_too_large_reports_capped_not_a_silent_pass(tmp_path):
    """A 0.9 m interpenetration exceeds MAX_NUDGE — the loop must refuse, and SAY so."""
    room = _room(tmp_path, {
        "Table0": ((0.0, 0.4, 0.0), (1.0, 0.8, 1.0), "table"),
        "Table1": ((0.1, 0.4, 0.0), (1.0, 0.8, 1.0), "table"),
    })
    p = R.resolve(room, max_nudge=0.05)
    assert p["stop"] == "capped"
    assert not p["converged"]


def test_round_cap_is_reported_distinctly(tmp_path):
    """Hitting the cap is retryable; being stuck is not. They must not look the same."""
    room = _room(tmp_path, {
        "Table0": ((0.0, 0.4, 0.0), (1.0, 0.8, 1.0), "table"),
        "Table1": ((0.5, 0.4, 0.0), (1.0, 0.8, 1.0), "table"),
    })
    p = R.resolve(room, max_rounds=1)
    assert p["stop"] == "max_rounds"
    assert not p["converged"]


def test_moves_stay_within_the_nudge_cap(tmp_path):
    room = _room(tmp_path, {
        "Table0": ((0.0, 0.4, 0.0), (1.0, 0.8, 1.0), "table"),
        "Table1": ((0.5, 0.4, 0.0), (1.0, 0.8, 1.0), "table"),
    })
    p = R.resolve(room)
    for _oid, old, new, dist in p["moves"]:
        assert dist <= R.MAX_NUDGE + 1e-9
        assert dist == pytest.approx(math.dist(old, new), abs=1e-6)


def test_apply_writes_only_the_moved_centres(tmp_path):
    room = _room(tmp_path, {
        "Table0": ((0.0, 0.4, 0.0), (1.0, 0.8, 1.0), "table"),
        "Table1": ((0.5, 0.4, 0.0), (1.0, 0.8, 1.0), "table"),
    })
    p = R.resolve(room)
    out = R.apply(p)
    assert out != p["src"], "a converged solve with moves must change the source"
    assert out.count("SHELL") == p["src"].count("SHELL"), "must edit in place, not reserialise"
