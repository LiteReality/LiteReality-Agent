"""The layout stage's two contracts: it never makes a room worse, and it never breaks a run.

Both are properties rather than examples, because the stage runs by default on every scan and the
failure modes that matter are the ones no fixture would happen to contain. Fast and offline — no
scan, no Blender, no model.
"""

from __future__ import annotations

import copy
import math
import pickle
import re

import numpy as np
import pytest

from litereality_agent.pipeline.scene_init.layout import adapter, report
from litereality_agent.pipeline.scene_init.layout.adjust import check
from litereality_agent.pipeline.scene_init.layout.graph import expected_pair, in_category
from litereality_agent.pipeline.scene_init.layout.repair import (SHRINKABLE, attachments,
                                                                 duplicates, repair, score)
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
    # SHELL (x, y, z) -> ARKit (x, z, -y): height swaps back to ARKit's y, and the plan axis is
    # negated, because ARKIT_TO_Z_UP is (x, y, z) -> (x, -z, y) and this is its inverse.
    assert entries[0]["position"][0] == pytest.approx(1.5)
    assert entries[0]["position"][1] == pytest.approx(0.4)
    assert entries[0]["position"][2] == pytest.approx(-2.0)


def test_a_dropped_object_is_removed_and_nothing_is_invented():
    entries = [{"object_type": "A", "position": np.zeros(3), "bbox": np.ones(3), "rotation": 0.0},
               {"object_type": "B", "position": np.zeros(3), "bbox": np.ones(3), "rotation": 0.0}]
    shell = room({"A": box(1.0, 1.0)})                            # B was dropped by the repair
    changed = adapter.apply_to_objects(entries, shell)
    assert changed["dropped"] == ["B"]
    assert [e["object_type"] for e in entries] == ["A"]


# ── the drawn report ─────────────────────────────────────────────────────────
def test_the_report_draws_both_rooms_and_names_every_object():
    before = room({"Storage0": box(2.0, 2.0), "Table0": box(4.0, 4.0, category="table")})
    after = room({"Storage0": box(2.4, 2.0), "Table0": box(4.0, 4.0, category="table")})
    page = report.render("scan", before, after)
    assert page.count("<svg") == 2, "a change was made, so both plans belong on the page"
    for object_id in ("Storage0", "Table0"):
        assert page.count(object_id) >= 2
    assert 'class="ghost"' in page, "the scanned position has to stay visible under the repair"
    assert 'class="move"' in page, "and the correction has to be drawn as a move"


def test_an_unchanged_room_is_drawn_once():
    """Two identical plans are not a comparison — they invite a hunt for a difference."""
    shell = room({"Storage0": box(2.0, 2.0)})
    page = report.render("scan", shell, shell)
    assert page.count("<svg") == 1
    assert 'class="move"' not in page


def test_a_mixed_winding_floor_still_fills():
    """The floor is one path, and SVG's nonzero rule reads opposed triangles as a hole.

    RoomPlan's floor mesh has no consistent orientation, so emitted as they come the room's own
    triangles cancel and the plate renders as nothing at all — an invisible failure a picture is
    meant to catch rather than contain. Every triangle must leave here wound the same way.
    """
    shell = room({})
    shell["floor"]["faces"] = [[0, 1, 2], [2, 1, 0], [0, 2, 3]]   # deliberately opposed
    page = report.render("scan", shell, shell)
    path = re.search(r'class="floor" d="([^"]+)"', page).group(1)
    for subpath in path.split("Z")[:-1]:
        points = [tuple(map(float, p.split())) for p in subpath.lstrip("M").split("L") if p.strip()]
        area = sum(points[i][0] * points[(i + 1) % len(points)][1]
                   - points[(i + 1) % len(points)][0] * points[i][1]
                   for i in range(len(points)))
        assert area >= 0, f"triangle wound the other way: {subpath}"


def test_a_dropped_object_is_drawn_where_it_was():
    """Deleting a duplicate is the most invisible thing the pass does and the easiest to get wrong."""
    before = room({"Storage0": box(2.0, 2.0), "Storage1": box(2.05, 2.0)})
    after = room({"Storage0": box(2.0, 2.0)})
    page = report.render("scan", before, after)
    assert 'class="dropped"' in page
    assert "Storage1" in page


def scene_data(monkeypatch, tmp_path, objects):
    """A scan directory the stage will accept, without a RoomPlan capture to build one from.

    `shell_from_scene_data` is the loader's job and is tested where the loader is; these are about
    what the STAGE does with the shell it gets back, so it is handed one directly.
    """
    shell = room(objects)
    monkeypatch.setattr(adapter, "shell_from_scene_data",
                        lambda *args, **kwargs: copy.deepcopy(shell))
    entries = [{"object_type": oid, "position": np.array([o["center"][0], o["center"][2],
                                                          -o["center"][1]]),
                "bbox": np.array([o["size"][0], o["size"][2], o["size"][1]]),
                "rotation": 0.0, "mesh_id": oid} for oid, o in objects.items()]
    (tmp_path / "objects.pkl").write_bytes(pickle.dumps(entries))
    return shell


def test_the_stage_writes_the_plan_and_can_be_told_not_to(tmp_path, monkeypatch):
    scene_data(monkeypatch, tmp_path, {"Storage0": box(2.0, 2.0)})

    monkeypatch.setenv("LR_LAYOUT_VIZ", "0")
    assert "visualization" not in run_layout("scan", scene_data_dir=tmp_path)
    assert not (tmp_path / "layout.html").exists()

    monkeypatch.setenv("LR_LAYOUT_VIZ", "1")
    result = run_layout("scan", scene_data_dir=tmp_path)
    assert (tmp_path / "layout.html").is_file(), "a sound room is a claim, and needs its picture too"
    assert result["visualization"].endswith("layout.html")


def test_a_failed_drawing_does_not_fail_the_repair(tmp_path, monkeypatch):
    """The pkl is already written by the time the plan is drawn. A broken picture is not a broken
    repair, and reporting it as one would send a good room back through init."""
    scene_data(monkeypatch, tmp_path,
               {"Storage0": box(2.0, 2.0), "Storage1": box(2.05, 2.0)})
    monkeypatch.setattr(report, "render", lambda *args, **kwargs: 1 / 0)

    result = run_layout("scan", scene_data_dir=tmp_path)
    assert "error" not in result
    assert "visualization" not in result
    assert result["after"] == 0, "the repair itself still has to have happened"
    assert (tmp_path / "layout_report.json").is_file()


# ── what may be deleted, and what a merged run is ────────────────────────────
def test_nothing_is_ever_deleted_by_default():
    """Two boxes on the same spot ARE one object detected twice, and it still keeps both.

    A redundant generated asset is visible, cheap and reversible. A wrong deletion is an object
    that is simply not in the room, and nothing downstream reports the absence.
    """
    shell = room({"Storage0": box(2.0, 2.0), "Storage1": box(2.05, 2.02)})
    assert duplicates(shell), "it can still SEE the duplicate"
    out, _moves, log = repair(shell)
    assert "drop" not in [a["action"] for a in log]
    assert set(out["objects"]) == {"Storage0", "Storage1"}


def test_dropping_can_be_armed_deliberately(monkeypatch):
    monkeypatch.setenv("LR_LAYOUT_DROP", "1")
    shell = room({"Storage0": box(2.0, 2.0), "Storage1": box(2.05, 2.02)})
    out, _moves, log = repair(shell)
    assert [a["action"] for a in log] == ["drop"]
    assert len(out["objects"]) == 1


def test_a_run_that_merely_contains_a_smaller_box_is_not_a_duplicate():
    """Kitchen's failure, as geometry: a cabinet standing inside a counter run.

    Overlap normalised by the smaller box is 1.00 for containment — indistinguishable from a
    perfect duplicate by that measure alone, which is why size has to be part of the test.
    """
    shell = room({"Storage0": box(2.0, 2.0, w=1.23, d=0.67, h=0.89),
                  "Oven_Storage_Stove0": box(2.0, 2.0, w=2.53, d=1.62, h=1.05,
                                             category="oven_storage_stove")})
    assert not duplicates(shell)


def test_a_merged_run_is_never_the_box_that_gets_deleted(monkeypatch):
    """Even armed, the merge fused this unit on evidence the repair does not have."""
    monkeypatch.setenv("LR_LAYOUT_DROP", "1")
    merged = box(2.0, 2.0)
    merged["merged_from"] = ["Oven1", "Oven2", "Stove0"]
    shell = room({"Oven_Storage_Stove0": merged, "Storage1": box(2.05, 2.02)})
    out, _moves, log = repair(shell)
    assert [a["action"] for a in log] == ["drop"]
    assert "Oven_Storage_Stove0" in out["objects"], "it deleted the assembled unit"
    assert "Storage1" not in out["objects"]


def test_the_category_tables_see_through_a_merged_name():
    """A merged run is named after its members, and every category rule keyed on an exact string
    silently stopped applying to it — no error, just knowledge switching off."""
    assert expected_pair("sink", "sink_storage"), "a basin in its own counter run"
    assert expected_pair("storage", "oven_storage_stove"), "a cabinet in a counter run"
    assert in_category("sink_storage", SHRINKABLE), "the shrink action exists FOR merged runs"
    assert not expected_pair("sink_storage", "oven_storage_stove"), \
        "two runs matched loosely enough would exempt a genuine interpenetration"


def test_a_cabinet_inside_a_merged_counter_run_is_not_a_collision():
    """The whole chain: it was reported as a clash, and the repair answered by deleting the run."""
    shell = room({"Storage0": box(2.0, 2.0, w=1.23, d=0.67, h=0.89),
                  "Oven_Storage_Stove0": box(2.0, 2.0, w=2.53, d=1.62, h=1.05,
                                             category="oven_storage_stove")})
    assert not [v for v in errors(shell) if v.kind == "object_clash"]
