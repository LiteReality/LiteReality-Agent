"""Iterative repair must converge without erasing the scan's identity or measurements."""

import copy
import importlib
import json
import subprocess

import numpy as np
import pytest
from PIL import Image

from litereality_agent.pipeline.scene_init.layout import agent, evidence
from litereality_agent.pipeline.scene_init.layout.converge import fidelity_errors, quality, solve


def room(tmp_path):
    return {
        "objects": {"Storage0": {"category": "storage", "center": [2, 0.1, 0.4],
                                 "size": [0.8, 0.6, 0.8], "yaw": 0}},
        "walls": {"Wall0": {"start": [0, 0], "end": [4, 0], "thickness": 0.1, "height": 2.5}},
        "openings": {}, "floor_z": 0, "ceiling_z": 2.5,
        "floor": {"verts": [[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0]],
                  "faces": [[0, 1, 2], [0, 2, 3]]}, "meta": {"batch_dir": str(tmp_path)},
    }


@pytest.fixture
def agent_room(tmp_path, monkeypatch):
    shell = room(tmp_path)
    folder = tmp_path / "references/Storage0"
    folder.mkdir(parents=True)
    Image.new("RGB", (40, 40), "blue").save(folder / "rank0.jpg")
    repair = importlib.import_module("litereality_agent.pipeline.scene_init.layout.repair")
    monkeypatch.setattr(repair, "repair", lambda s: (copy.deepcopy(s), [], []))
    return shell


def test_agent_can_improve_same_error_then_retry_to_zero(agent_room, monkeypatch):
    calls = []

    def propose(shell, *args, **kwargs):
        calls.append(shell["objects"]["Storage0"]["center"][1])
        return {"object_id": "Storage0", "action": "translate",
                "center": [2, 0.2 if len(calls) == 1 else 0.4, 0.4]}

    monkeypatch.setattr(agent, "propose", propose)
    result, _, _, report = solve(agent_room, copy.deepcopy(agent_room), use_agent=True)
    assert calls == [0.1, 0.2]
    assert quality(result)[0] == 0
    assert report == {"agent_calls": 2, "stop_reason": "validated", "fidelity_errors": []}


def test_no_progress_stops_without_claiming_success(agent_room, monkeypatch):
    monkeypatch.setattr(agent, "propose", lambda *a, **kw: {"action": "none"})
    result, _, _, report = solve(agent_room, copy.deepcopy(agent_room), use_agent=True)
    assert report["agent_calls"] == 2 and report["stop_reason"] == "no_progress"
    assert quality(result)[0] > 0
    assert result["objects"] == agent_room["objects"]


def test_rejected_proposal_feedback_reaches_next_round(agent_room, monkeypatch):
    calls = []

    def propose(shell, *args, **kwargs):
        calls.append(copy.deepcopy(shell['meta'].get('layout_feedback', {})))
        return {"object_id": "Storage0", "action": "translate",
                "center": [2, 2 if len(calls) == 1 else 0.4, 0.4]}

    monkeypatch.setattr(agent, "propose", propose)
    result, _, _, report = solve(agent_room, copy.deepcopy(agent_room), use_agent=True)
    assert report['stop_reason'] == 'validated'
    assert quality(result)[0] == 0
    assert calls[1]['Storage0'][0]['reason'] == 'outside the stated bounds'


def test_resize_does_not_snap_to_unrelated_nearby_wall(tmp_path):
    shell = room(tmp_path)
    shell['objects']['Storage0'].update(center=[1, 0.6, 0.4], size=[1, 0.6, 0.8])
    shell['objects']['Storage1'] = {**shell['objects']['Storage0'], 'center': [1.9, 0.6, 0.4]}
    result, decisions = agent.apply_proposals(shell, [{
        'object_id': 'Storage0', 'action': 'resize', 'size': [0.7, 0.6, 0.8], 'wall': 'Storage1'}])
    assert decisions[0]['accepted']
    assert result['objects']['Storage0']['center'] == [1, 0.6, 0.4]
    assert quality(result)[0] == 0


def test_missing_evidence_does_not_call_agent(tmp_path, monkeypatch):
    shell = room(tmp_path)
    repair = importlib.import_module("litereality_agent.pipeline.scene_init.layout.repair")
    monkeypatch.setattr(repair, "repair", lambda s: (copy.deepcopy(s), [], []))
    monkeypatch.setattr(agent, "propose", lambda *a, **kw: pytest.fail("no photos"))
    result, _, _, report = solve(shell, copy.deepcopy(shell), use_agent=True)
    assert report["agent_calls"] == 0
    assert quality(result)[0] > 0


def test_fidelity_compares_to_original_not_previous_iteration(tmp_path):
    baseline = room(tmp_path)
    baseline["objects"]["Storage0"]["category"] = "table"
    first = copy.deepcopy(baseline)
    first["objects"]["Storage0"]["size"][0] *= 0.9
    assert fidelity_errors(first, baseline) == []
    second = copy.deepcopy(first)
    second["objects"]["Storage0"]["size"][0] *= 0.9
    assert fidelity_errors(second, first) == []
    assert fidelity_errors(second, baseline) == ["Storage0"]
    second["objects"].clear()
    assert fidelity_errors(second, baseline) == ["Storage0"]


def test_bad_merge_can_split_back_to_measured_members(tmp_path):
    shell = room(tmp_path)
    unit = {"category": "storage", "center": [0.8, 2, 0.4], "size": [0.5, 0.8, 0.8], "yaw": 0}
    right = {**unit, "center": [3.2, 2, 0.4]}
    shell["objects"] = {
        "StorageRun0": {**unit, "center": [2, 2, 0.4], "size": [3, 0.8, 0.8],
                        "merged_members": {"Storage0": unit, "Storage1": right}},
        "Table0": {**unit, "category": "table", "center": [2, 2, 0.4]},
    }
    assert quality(shell)[0] > 0
    result, decisions = agent.apply_proposals(shell, [{"object_id": "StorageRun0", "action": "split"}])
    assert decisions[0]["accepted"]
    assert quality(result)[0] == 0
    assert fidelity_errors(result, shell) == []
    assert set(result["objects"]) == {"Storage0", "Storage1", "Table0"}


@pytest.mark.parametrize("confidence,expected", [(0.9, "merge"), (0.5, "separate"),
                                                (float("nan"), "separate"), (2, "separate")])
def test_merge_review_demands_confident_visual_approval(agent_room, tmp_path, monkeypatch,
                                                      confidence, expected):
    def run(command, **kwargs):
        assert "Read ALL listed scan images" in command[2]
        assert str(tmp_path / "references/Storage0/rank0.jpg") in command[2]
        return subprocess.CompletedProcess(command, 0, json.dumps({
            "action": "merge", "confidence": confidence, "why": "test evidence"}), "")

    monkeypatch.setattr(subprocess, "run", run)
    assert evidence.review_merge(agent_room, ["Storage0"], tmp_path)["action"] == expected


def test_merge_review_missing_photos_is_separate(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("no photos"))
    assert evidence.review_merge(room(tmp_path), ["Storage0"], tmp_path)["action"] == "separate"


def test_real_projection_prepares_and_reuses_review_photos(tmp_path):
    scene = tmp_path / "scene_data/Room"
    capture = tmp_path / "rgbd/Room"
    for name in ("image", "intrinsic", "extrinsic"):
        (capture / name).mkdir(parents=True)
    Image.new("RGB", (256, 192), "blue").save(capture / "image/frame_0.jpg")
    np.save(capture / "intrinsic/intrinsic_0.npy", [[100, 0, 128], [0, 100, 96], [0, 0, 1]])
    np.save(capture / "extrinsic/extrinsic_0.npy", np.eye(4))
    shell = room(tmp_path)
    shell["objects"]["Storage0"].update(center=[0, -2, 0], size=[0.8, 0.8, 0.8])
    assert evidence.prepare("Room", scene, shell) == {"images": 1}
    picture = scene / "references/Storage0/rank0.jpg"
    before = picture.stat().st_mtime_ns
    assert evidence.prepare("Room", scene, shell) == {"images": 1}
    assert picture.stat().st_mtime_ns == before
    with Image.open(picture) as image:
        assert image.size == (192, 256)
