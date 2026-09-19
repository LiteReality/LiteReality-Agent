"""The diagnostic observer preserves solver decisions and records CLI evidence offline."""

import importlib
import importlib.util
import json
import subprocess
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from litereality_agent.pipeline.scene_init.layout.adjust import check

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/capture/layout_batch.py"
spec = importlib.util.spec_from_file_location("layout_batch", SCRIPT)
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


@pytest.mark.parametrize("backend", [None, "local", "modal"])
def test_batch_dino_defaults_to_auto_and_allows_explicit_override(
    tmp_path, monkeypatch, capsys, backend,
):
    from litereality_agent import settings

    capture = tmp_path / "Room"
    capture.mkdir()
    (capture / "room.usdz").touch()
    monkeypatch.setattr(settings, "load_settings", lambda: SimpleNamespace())
    args = ["Room", "--scans-dir", str(tmp_path), "--use-dino", "--dry-run"]
    if backend is not None:
        args.extend(["--dino-backend", backend])

    assert batch.main(args) == 0
    assert f"DINO backend: {backend or 'auto'}" in capsys.readouterr().out


def room():
    return {
        "walls": {"Wall0": {"start": [0, 0], "end": [4, 0], "thickness": 0.1,
                            "height": 2.5}},
        "objects": {"Storage0": {"category": "storage", "center": [2, 0, 0.4],
                                 "size": [0.8, 0.6, 0.8], "yaw": 0}},
        "openings": {}, "floor_z": 0, "ceiling_z": 2.5,
        "floor": {"verts": [[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0]],
                  "faces": [[0, 1, 2], [0, 2, 3]]},
    }


def test_observation_preserves_repair_and_records_search(tmp_path):
    repair = importlib.import_module("litereality_agent.pipeline.scene_init.layout.repair")
    expected = repair.repair(room())
    audit = batch.Audit(tmp_path)
    with ExitStack() as stack:
        audit.instrument(stack)
        actual = repair.repair(room())
    assert actual == expected
    events = [json.loads(line) for line in (tmp_path / "steps.jsonl").read_text().splitlines()]
    assert {e["event"] for e in events} >= {
        "deterministic_start", "candidate", "score", "deterministic_finish",
    }


def test_agent_stream_is_saved_and_result_reaches_parser(tmp_path, monkeypatch):
    agent = importlib.import_module("litereality_agent.pipeline.scene_init.layout.agent")
    references = tmp_path / "references/Storage0"
    references.mkdir(parents=True)
    (references / "rank0.jpg").touch()
    reply = {"action": "none", "why": "The scan supports this size."}
    stream = "\n".join(json.dumps(e) for e in [
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Read"}]}},
        {"type": "result", "result": json.dumps(reply), "is_error": False},
    ])

    def fake_run(command, **kwargs):
        assert command[command.index("--output-format") + 1] == "stream-json"
        assert "--verbose" in command
        return subprocess.CompletedProcess(command, 0, stream, "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    violation = next(v for v in check(room()) if v.object == "Storage0" and v.severity == "error")
    audit = batch.Audit(tmp_path)
    with ExitStack() as stack:
        audit.instrument(stack)
        result = agent.propose(room(), violation, tmp_path)
    assert result["why"] == reply["why"]
    assert result["object_id"] == "Storage0"
    assert (tmp_path / "agent_001.events.jsonl").read_text() == stream
    assert "Storage0" in (tmp_path / "agent_001.prompt.txt").read_text()
    assert audit.calls == 1


def test_dino_refinement_preserves_original_and_records_actual_crop(tmp_path, monkeypatch):
    from litereality_agent.pipeline.scene_init import paths as config
    from litereality_agent.pipeline.scene_init.ingest.detect import bbox_polish

    parsed = tmp_path / "parsed"
    obj = parsed / "Storage0"
    camera = obj / "camera"
    camera.mkdir(parents=True)
    source = tmp_path / "frame.jpg"
    Image.new("RGB", (100, 100), "blue").save(source)
    name = "frame_0_ranking_0"
    Image.new("RGB", (80, 80), "blue").save(obj / f"{name}.jpg")
    Image.new("RGB", (80, 80), "blue").save(obj / "stitched_image.jpg")
    batch.write_json(camera / f"{name}.json", {
        "original_image_path": str(source), "resized_bbox": [10, 10, 90, 90], "ranking": 0,
    })
    monkeypatch.setattr(config, "parsed_images_dir", lambda scan: parsed)
    monkeypatch.setattr(config, "work_root", lambda: tmp_path)
    monkeypatch.setenv("LR_ENLARGED_CROP_OBJECTS", "")
    monkeypatch.setattr(bbox_polish.detector, "using_service", lambda: True)
    monkeypatch.setattr(bbox_polish.detector, "available", lambda: True)
    monkeypatch.setattr(bbox_polish.detector, "detect", lambda *args: [
        SimpleNamespace(box=[20, 20, 70, 70]),
    ])
    trace = tmp_path / "trace"
    trace.mkdir()
    audit = batch.Audit(trace)
    result = batch.refine_crops("test", config, audit)
    assert result["refined_total"] == 1
    with Image.open(trace / "crops_before_dino/Storage0" / f"{name}.jpg") as before:
        assert before.size == (80, 80)
    with Image.open(trace / "crops_after_dino/Storage0" / f"{name}.jpg") as after:
        assert after.size == (50, 50)
    metadata = json.loads((camera / f"{name}.json").read_text())
    assert metadata["dino_refined"] is True
    assert metadata["resized_bbox"] == [20, 20, 70, 70]
    assert "dino_seconds" in audit.timings
    assert not (obj / "stitched_image.jpg").exists()
    # Evidence sent to the agent includes the real refined crop, alongside full-frame context.
    sheet = batch.reference_sheet(Image.new("RGB", (100, 100), "black"),
                                  (10, 10, 90, 90), metadata, obj / f"{name}.jpg", "Storage0")
    assert sheet.size == (170, 150)
    assert sheet.getpixel((130, 60))[2] > 240


def test_requested_dino_cannot_silently_skip(tmp_path, monkeypatch):
    from litereality_agent.pipeline.scene_init.ingest.detect import bbox_polish

    parsed = tmp_path / "parsed"
    parsed.mkdir()
    trace = tmp_path / "trace"
    trace.mkdir()
    config = SimpleNamespace(parsed_images_dir=lambda scan: parsed)
    monkeypatch.setattr(bbox_polish, "polish", lambda *args, **kwargs: {
        "skipped": "detector unavailable", "objects": [],
    })
    with pytest.raises(RuntimeError, match="GroundingDINO requested but skipped"):
        batch.refine_crops("test", config, batch.Audit(trace))
    assert json.loads((trace / "dino_refinement.json").read_text())["skipped"]


def test_batch_keeps_diagnostics_but_rejects_colliding_scene(tmp_path, monkeypatch):
    from litereality_agent.pipeline.scene_init import layout
    from litereality_agent.pipeline.scene_init import paths as config
    from litereality_agent.pipeline.scene_init.ingest import merge_boxes
    from litereality_agent.pipeline.scene_init.ingest.extract import extract_scene

    scene = tmp_path / "scene_data"
    scene.mkdir()
    for name in ("objects", "walls", "floor", "wall_holes"):
        (scene / f"{name}.pkl").touch()
    monkeypatch.setattr(config, "enter_work_root", lambda scan: None)
    monkeypatch.setattr(config, "scene_data_dir", lambda scan: scene)
    monkeypatch.setattr(extract_scene, "extract", lambda *a: None)
    monkeypatch.setattr(merge_boxes, "merge_for_scan", lambda scan: {})
    monkeypatch.setattr(layout.adapter, "shell_from_scene_data", lambda path: room())
    monkeypatch.setattr(layout, "run_layout", lambda *a, **kw: {"after": 0})
    with pytest.raises(layout.LayoutValidationError):
        batch.run_scene(tmp_path / "Room", tmp_path / "output", False)
    trace = tmp_path / "output/Room/layout_trace"
    assert (trace / "02_merged.json").is_file()
    assert (trace / "03_final.json").is_file()
    assert (trace / "timings.json").is_file()
    summary = json.loads((trace / "summary.json").read_text())
    assert summary["error"]
    assert not summary["validation"]["passed"]
    assert json.loads((trace / "layout_validation.json").read_text())["wall_clashes"] > 0
