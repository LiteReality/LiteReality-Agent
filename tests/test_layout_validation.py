"""Layout acceptance is mandatory even when a repair or cached report claims success."""

import copy
import json
from types import SimpleNamespace

import pytest

from litereality_agent.pipeline.scene_init import ingest
from litereality_agent.pipeline.scene_init.layout import adapter
from litereality_agent.pipeline.scene_init.layout.validation import (
    LayoutValidationError,
    inspect_layout,
    require_valid_layout,
)


@pytest.fixture
def saved_layout(tmp_path, monkeypatch):
    root = tmp_path / "object_init/input/scene_data/Room"
    root.mkdir(parents=True)
    for name in ("objects", "walls", "floor", "wall_holes"):
        (root / f"{name}.pkl").touch()
    shell = {
        "walls": {"Wall0": {"start": [0, 0], "end": [4, 0], "thickness": 0.1,
                            "height": 2.5}},
        "objects": {"Storage0": {"category": "storage", "center": [2, 2, 0.4],
                                 "size": [0.8, 0.6, 0.8], "yaw": 0}},
        "openings": {}, "floor_z": 0, "ceiling_z": 2.5,
        "floor": {"verts": [[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0]],
                  "faces": [[0, 1, 2], [0, 2, 3]]},
    }
    monkeypatch.setattr(adapter, "shell_from_scene_data", lambda path: copy.deepcopy(shell))
    return root, shell


def test_clean_layout_passes_and_saves_verdict(saved_layout):
    root, _ = saved_layout
    result = {"after": 0}
    report = require_valid_layout("Room", result, scene_data_dir=root)
    assert report["passed"]
    assert report["error_count"] == report["wall_clashes"] == report["object_clashes"] == 0
    assert json.loads((root / "layout_validation.json").read_text()) == result["validation"]


@pytest.mark.parametrize("kind", ["object_clash", "wall_clash"])
def test_saved_collisions_override_stale_zero_result(saved_layout, kind):
    root, shell = saved_layout
    if kind == "object_clash":
        shell["objects"]["Storage1"] = copy.deepcopy(shell["objects"]["Storage0"])
    else:
        shell["objects"]["Storage0"]["center"][1] = 0.1
    result = {"after": 0}
    with pytest.raises(LayoutValidationError, match="Layout validation failed"):
        require_valid_layout("Room", result, scene_data_dir=root)
    report = json.loads((root / "layout_validation.json").read_text())
    assert not report["passed"]
    assert kind in {v["kind"] for v in report["violations"]}


@pytest.mark.parametrize("result", [{"disabled": True}, {"skipped": "no data"},
                                   {"error": "repair failed"}, {"after": 1}])
def test_failed_or_incomplete_repair_is_not_success(saved_layout, result):
    root, _ = saved_layout
    with pytest.raises(LayoutValidationError):
        require_valid_layout("Room", dict(result), scene_data_dir=root)


def test_missing_or_unreadable_geometry_cannot_pass(saved_layout, monkeypatch):
    root, _ = saved_layout
    monkeypatch.setattr(adapter, "shell_from_scene_data", lambda path: 1 / 0)
    assert not inspect_layout(root)["passed"]
    (root / "walls.pkl").unlink()
    assert "Missing scene data" in inspect_layout(root)["error"]


def test_cached_ingest_is_rechecked_after_geometry_changes(saved_layout, tmp_path):
    from litereality_agent.pipeline.scene_init.ingest import merge_boxes

    root, shell = saved_layout
    (root / "layout_baseline.json").write_text(json.dumps({"shell": shell, "source_key": None}))
    (root / "merge_review.json").write_text(json.dumps({
        "policy_version": merge_boxes.POLICY_VERSION, "settings": merge_boxes.policy_settings()}))
    work = tmp_path / "object_init"
    (work / "object_init_summary.json").write_text("{}")
    (work / "object_refs/Room").mkdir(parents=True)
    context = SimpleNamespace(object_root=tmp_path, scan="Room")
    assert ingest.complete(context)
    # Neither the old success verdict nor the ingest summary can mask edited geometry.
    require_valid_layout("Room", {"after": 0}, scene_data_dir=root)
    shell["objects"]["Storage1"] = copy.deepcopy(shell["objects"]["Storage0"])
    assert not ingest.complete(context)


def test_flow_stops_before_crops_on_failed_layout(saved_layout, monkeypatch):
    from litereality_agent.pipeline.scene_init import flow

    root, shell = saved_layout
    shell["objects"]["Storage1"] = copy.deepcopy(shell["objects"]["Storage0"])
    (root / "merge_review.json").write_text(json.dumps({
        "policy_version": flow.merge_boxes.POLICY_VERSION, "settings": flow.merge_boxes.policy_settings()}))
    monkeypatch.setattr(flow.config, "enter_work_root", lambda scan: None)
    monkeypatch.setattr(flow.config, "scene_data_complete", lambda scan: True)
    monkeypatch.setattr(flow.config, "scene_data_dir", lambda scan: root)
    monkeypatch.setattr(flow.telemetry, "start", lambda *a, **kw: None)
    monkeypatch.setattr(flow.telemetry, "event", lambda *a, **kw: None)
    stages = []
    monkeypatch.setattr(flow.telemetry, "stage", lambda *a, **kw: stages.append(a))
    monkeypatch.setattr(flow.merge_boxes, "merge_for_scan", lambda scan: {})
    monkeypatch.setattr(flow.layout, "run_layout", lambda scan: {"after": 0})
    monkeypatch.setattr(flow.crop_objects, "crop", lambda *a, **kw: pytest.fail("crop ran"))
    with pytest.raises(LayoutValidationError):
        flow._process_scan("Room", root, SimpleNamespace(skip_extract=True))
    assert ("layout", "Room", "failed") in stages
    assert ("layout", "Room", "done") not in stages


def test_new_extraction_cannot_reuse_old_fidelity_baseline(saved_layout):
    root, shell = saved_layout
    (root / "layout_baseline.json").write_text(json.dumps({"shell": shell, "source_key": None}))
    (root / "objects.extracted.pkl").write_bytes(b"new extraction")
    verdict = inspect_layout(root)
    assert not verdict["passed"]
    assert "baseline changed" in verdict["error"]


def test_failed_handoff_replaces_stale_success_report(saved_layout, monkeypatch):
    from litereality_agent.pipeline.scene_init.layout import stage

    root, _ = saved_layout
    require_valid_layout("Room", {"after": 0}, scene_data_dir=root)
    monkeypatch.setattr(stage, "_write_shell", lambda *a: (_ for _ in ()).throw(OSError("disk failure")))
    result = stage.run_layout("Room", scene_data_dir=root, use_agent=False)
    assert result["error"] == "disk failure"
    assert not json.loads((root / "layout_validation.json").read_text())["passed"]
    assert json.loads((root / "layout_report.json").read_text())["error"] == "disk failure"
