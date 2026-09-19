"""Exercise the public CLI -> runner -> ingest -> real layout/refinement stage path offline."""

import contextlib
import copy
import importlib
import json
import pickle

import pytest
from PIL import Image

from litereality_agent import cli
from litereality_agent.models.grounding_dino import modal
from litereality_agent.pipeline.context import RunContext
from litereality_agent.pipeline.scene_init import flow, ingest
from litereality_agent.pipeline.scene_init.ingest.crop import crop_objects
from litereality_agent.pipeline.scene_init.ingest.detect import detector
from litereality_agent.pipeline.scene_init.layout import adapter, agent, evidence
from litereality_agent.settings import LiteRealitySettings


@pytest.mark.parametrize("command", ["run", "stage"])
@pytest.mark.parametrize("failure", [None, "unresolved", "dino_unavailable"])
def test_main_entry_reaches_layout_agent_and_modal(tmp_path, monkeypatch, command, failure):
    monkeypatch.chdir(tmp_path)  # restore cwd after flow.enter_work_root
    monkeypatch.setenv("LR_LAYOUT_AGENT", "1")
    monkeypatch.setenv("LR_LAYOUT", "1")
    monkeypatch.setenv("LR_LAYOUT_VIZ", "0")
    monkeypatch.setenv("LR_ENLARGED_CROP_OBJECTS", "")
    monkeypatch.setattr(detector, "_SERVICE", None)
    settings = LiteRealitySettings(
        _env_file=None, repo_root=tmp_path, output_root=tmp_path / "run",
        dino_backend="auto", dino_python="/must-not-start/python", modal_profile="test-profile",
        modal_token_id="test-token", modal_token_secret="test-secret",
    )
    context = RunContext("Room", tmp_path / "capture", tmp_path / "run/Room",
                         tmp_path / "run", settings=settings)
    monkeypatch.setattr(cli, "_context", lambda args: context)
    monkeypatch.setattr(cli, "_live_viewer", lambda *args: contextlib.nullcontext(lambda *args: None))
    monkeypatch.setattr(flow, "resolve_scan", lambda *args: (context.capture_dir, context.scan))
    monkeypatch.setattr(flow, "summarize", lambda *args: None)
    events = []

    def extract(raw, scan):
        scene = flow.config.scene_data_dir(scan)
        scene.mkdir(parents=True, exist_ok=True)
        entries = [{"object_type": "Storage0", "position": [2, 0.4, -0.1],
                    "bbox": [0.8, 0.8, 0.6], "rotation": 0.0}]
        for name, value in {"objects": entries, "objects.extracted": entries,
                            "walls": [], "floor": {}, "wall_holes": []}.items():
            (scene / f"{name}.pkl").write_bytes(pickle.dumps(value))

    def read_shell(scene):
        entries = pickle.loads((scene / "objects.pkl").read_bytes())
        return {
            "objects": {e["object_type"]: adapter.object_from_entry(e) for e in entries},
            "walls": {"Wall0": {"start": [0, 0], "end": [4, 0], "thickness": 0.1,
                                "height": 2.5}}, "openings": {},
            "floor_z": 0, "ceiling_z": 2.5,
            "floor": {"verts": [[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0]],
                      "faces": [[0, 1, 2], [0, 2, 3]]},
        }

    def prepare(scan, scene, shell):
        refs = scene / "references/Storage0"
        refs.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (40, 40), "blue").save(refs / "rank0.jpg")

    def propose(*args, **kwargs):
        events.append("layout_agent")
        return {"object_id": "Storage0", "action": "none" if failure == "unresolved" else "translate",
                "center": [2, 0.4, 0.4], "why": "test photograph"}

    def crop(scan, **kwargs):
        events.append("crop")
        parsed = flow.config.parsed_images_dir(scan)
        obj = parsed / "Storage0"
        camera = obj / "camera"
        camera.mkdir(parents=True, exist_ok=True)
        source = tmp_path / "frame.jpg"
        Image.new("RGB", (100, 100), "blue").save(source)
        name = "frame_0_ranking_0"
        Image.new("RGB", (80, 80), "blue").save(obj / f"{name}.jpg")
        (camera / f"{name}.json").write_text(json.dumps({
            "original_image_path": str(source), "resized_bbox": [10, 10, 90, 90], "ranking": 0,
        }))
        (parsed / crop_objects.STAMP).write_text('{"generation":"test"}')

    class Client:
        def __init__(self, *args, **kwargs):
            assert kwargs["profile"] == "test-profile"

        def map(self, payloads):
            events.append("modal_dino")
            assert payloads[0]["op"] == "detect"
            return [{"detections": [{"box": [20, 20, 70, 70], "score": 0.95, "label": "storage"}]}]

    def run_flow(context, module, args, **kwargs):
        assert module == "litereality_agent.pipeline.scene_init.flow"
        assert "--use-dino" in args
        # Stop before reference/mesh generation, but execute real parsing and processing.
        return flow.main([str(a) for a in args] + ["--crops-only", "--skip-openings"]), None

    monkeypatch.setattr(ingest, "run_module", run_flow)
    monkeypatch.setattr(flow.extract_scene, "extract", extract)
    monkeypatch.setattr(flow.merge_boxes, "merge_for_scan", lambda scan: {})
    monkeypatch.setattr(adapter, "shell_from_scene_data", read_shell)
    monkeypatch.setattr(evidence, "prepare", prepare)
    repair = importlib.import_module("litereality_agent.pipeline.scene_init.layout.repair")
    monkeypatch.setattr(repair, "repair", lambda shell: (copy.deepcopy(shell), [], []))
    monkeypatch.setattr(agent, "propose", propose)
    monkeypatch.setattr(crop_objects, "crop", crop)
    monkeypatch.setattr(modal, "ModalClient", Client)
    if failure == "dino_unavailable":
        from litereality_agent.models import registry

        monkeypatch.setattr(registry, "detection_from_settings", lambda settings: None)
        monkeypatch.setattr(detector, "available", lambda: False)
    argv = (["run", "Room", "--through", "ingest"] if command == "run"
            else ["stage", "ingest", "Room"])
    args = cli._parser().parse_args(argv + ["--use-dino"])
    rc = args.handler(args)
    assert rc == (1 if failure else 0)
    work = context.object_root / "object_init"
    verdict = json.loads((work / "input/scene_data/Room/layout_validation.json").read_text())
    assert verdict["passed"] == (failure != "unresolved")
    if failure == "unresolved":
        assert events == ["layout_agent", "layout_agent"]
    elif failure == "dino_unavailable":
        assert events == ["layout_agent", "crop"]
    else:
        assert events == ["layout_agent", "crop", "modal_dino"]
        assert isinstance(detector._SERVICE, modal.ModalDinoService)
        meta = work / "input/parsed_images/Room/Storage0/camera/frame_0_ranking_0.json"
        assert json.loads(meta.read_text())["dino_refined"]
        with Image.open(meta.parent.parent / "frame_0_ranking_0.jpg") as image:
            assert image.size == (50, 50)
