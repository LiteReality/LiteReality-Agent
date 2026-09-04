"""The layout stage's two contracts: it never makes a room worse, and it never breaks a run.

Both are properties rather than examples, because the stage runs by default on every scan and the
failure modes that matter are the ones no fixture would happen to contain. Fast and offline — no
scan, no Blender, no model.
"""

from __future__ import annotations

import math
import pickle

import numpy as np
import pytest

from litereality_agent.pipeline.scene_init.layout import adapter
from litereality_agent.pipeline.scene_init.layout.adjust import check
from litereality_agent.pipeline.scene_init.layout.repair import repair, score, attachments
from litereality_agent.pipeline.scene_init.layout.stage import run_layout


def errors(shell):
    return [v for v in check(shell) if v.severity == "error"]


def room(objects, size=6.0):
    """A square room with the given objects, in SHELL form."""
    corners = [(0.0, 0.0), (size, 0.0), (size, size), (0.0, size)]
    walls = {}
    for i in range(4):
        walls[f"Wall{i}"] = {"start": list(corners[i]), "end": list(corners[(i + 1) % 4]),
                             "thickness": 0.1, "height": 2.5}
    verts = [[x, y, 0.0] for x, y in corners]
    return {"walls": walls, "openings": {}, "objects": objects,
            "floor": {"verts": verts, "faces": [[0, 1, 2], [0, 2, 3]]},
            "floor_z": 0.0, "ceiling_z": 2.5}


def box(x, y, w=0.8, d=0.6, h=0.8, category="storage", yaw=0.0):
    return {"category": category, "center": [x, y, h / 2], "size": [w, d, h], "yaw": yaw}


def test_a_sound_room_is_left_alone():
    shell = room({"Storage0": box(2.0, 2.0), "Table0": box(4.0, 4.0, category="table")})
    assert not errors(shell)
    out, _moves, log = repair(shell)
    assert not log, f"nothing to do, but it acted: {log}"
    for oid, obj in out["objects"].items():
        assert math.dist(obj["center"][:2], shell["objects"][oid]["center"][:2]) < 1e-6


@pytest.mark.parametrize("seed", range(12))
def test_repair_never_returns_a_worse_room(seed):
    """The contract the whole action-and-gate design exists to provide.

    Randomised because the interesting failures are compositional: a push that separates one pair
    drives an object into a third, and a fixture that only ever ran on the captures we happened to
    have would not show it.
    """
    rng = np.random.default_rng(seed)
    objects = {}
    for i in range(rng.integers(3, 8)):
        objects[f"Storage{i}"] = box(float(rng.uniform(0.2, 5.8)), float(rng.uniform(0.2, 5.8)),
                                     w=float(rng.uniform(0.4, 1.6)), d=float(rng.uniform(0.4, 1.2)),
                                     yaw=float(rng.uniform(-180, 180)))
    shell = room(objects)
    held = {oid: set(attachments(o, shell["walls"])) for oid, o in shell["objects"].items()}
    held = {k: v for k, v in held.items() if v}
    out, _moves, _log = repair(shell)
    assert score(out, shell, held) <= score(shell, shell, held)


def test_the_stage_never_raises_into_the_pipeline(tmp_path):
    """A layout failure must degrade to an unrepaired run, never take init down."""
    (tmp_path / "objects.pkl").write_bytes(b"not a pickle")
    result = run_layout("broken", scene_data_dir=tmp_path)
    assert "error" in result or "skipped" in result

    assert run_layout("empty", scene_data_dir=tmp_path / "nope").get("skipped")


def test_the_stage_can_be_switched_off(tmp_path, monkeypatch):
    monkeypatch.setenv("LR_LAYOUT", "0")
    assert run_layout("anything", scene_data_dir=tmp_path).get("disabled")


def test_objects_pkl_round_trips_through_the_shell(tmp_path):
    """Every field extraction wrote must survive, and only pose and size may change."""
    entries = [{"object_type": "Storage0", "position": np.array([1.0, 0.4, 2.0]),
                "bbox": np.array([0.8, 0.8, 0.6]), "rotation": 0.0,
                "file": "./x.usda", "mesh_id": "Storage0", "top_down_rect": [(0, 0)]}]
    shell = room({"Storage0": box(1.0, 2.0)})
    shell["objects"]["Storage0"]["center"] = [1.5, 2.0, 0.4]      # pretend the repair moved it
    changed = adapter.apply_to_objects(entries, shell)
    assert changed["moved"] == ["Storage0"]
    assert entries[0]["file"] == "./x.usda" and entries[0]["mesh_id"] == "Storage0"
    assert entries[0]["top_down_rect"] == [(0, 0)]
    # ARKit y stays the height, x/z the floor plane
    assert entries[0]["position"][0] == pytest.approx(1.5)
    assert entries[0]["position"][2] == pytest.approx(2.0)


def test_a_dropped_object_is_removed_and_nothing_is_invented():
    entries = [{"object_type": "A", "position": np.zeros(3), "bbox": np.ones(3), "rotation": 0.0},
               {"object_type": "B", "position": np.zeros(3), "bbox": np.ones(3), "rotation": 0.0}]
    shell = room({"A": box(1.0, 1.0)})                            # B was dropped by the repair
    changed = adapter.apply_to_objects(entries, shell)
    assert changed["dropped"] == ["B"]
    assert [e["object_type"] for e in entries] == ["A"]
