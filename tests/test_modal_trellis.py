from __future__ import annotations

import base64
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from lrauthor.settings import LiteRealitySettings


def _settings(**overrides) -> LiteRealitySettings:
    """Real settings, isolated from the developer's own .env, so runtime selection is exercised
    through `modal_configured()` rather than a hand-built stand-in that can drift from it."""
    return LiteRealitySettings(_env_file=None, **overrides)


class FakeClient:
    def __init__(self, outputs):
        self.outputs = outputs
        self.inputs = None

    def map(self, inputs):
        self.inputs = list(inputs)
        return self.outputs


def test_pipeline_installs_background_bypass_before_model_construction(monkeypatch):
    from lrauthor.models.trellis import inference

    calls = []

    class Pipeline:
        @classmethod
        def from_pretrained(cls, model_name):
            calls.append(("load", model_name))
            return cls()

        def cuda(self):
            calls.append(("cuda", None))

    torch = ModuleType("torch")
    torch.backends = SimpleNamespace(cudnn=SimpleNamespace(enabled=True))
    trellis = ModuleType("trellis2")
    pipelines = ModuleType("trellis2.pipelines")
    pipelines.Trellis2ImageTo3DPipeline = Pipeline
    trellis.pipelines = pipelines
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "trellis2", trellis)
    monkeypatch.setitem(sys.modules, "trellis2.pipelines", pipelines)
    monkeypatch.setattr(inference, "install_noop_rembg", lambda: calls.append(("bypass", None)))

    inference.load_pipeline("model")

    assert calls == [("bypass", None), ("load", "model"), ("cuda", None)]


def test_generate_exports_portable_glb_without_webp_extension(monkeypatch, tmp_path):
    from PIL import Image

    from lrauthor.models.trellis import inference

    calls = {}

    class Mesh:
        vertices = faces = attrs = coords = layout = voxel_size = object()

        def simplify(self, target):
            calls["simplify"] = target

    class Pipeline:
        def run(self, image, **options):
            calls["pipeline"] = options
            return [Mesh()]

    class GLB:
        def export(self, path, **options):
            calls["export"] = (path, options)

    torch = ModuleType("torch")
    torch.manual_seed = lambda seed: calls.setdefault("seed", seed)
    o_voxel = ModuleType("o_voxel")
    o_voxel.postprocess = SimpleNamespace(to_glb=lambda **options: GLB())
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "o_voxel", o_voxel)
    image = tmp_path / "input.png"
    output = tmp_path / "output.glb"
    Image.new("RGBA", (2, 2)).save(image)

    inference.generate(
        Pipeline(),
        image,
        output,
        seed=7,
        decimation_target=20_000,
        texture_size=512,
        pipeline_type="512",
    )

    assert calls["pipeline"] == {"seed": 7, "pipeline_type": "512"}
    assert calls["export"] == (str(output), {"extension_webp": False})


def test_modal_trellis_maps_model_payloads_and_writes_outputs(tmp_path):
    from lrauthor.models.trellis.modal import ModalTrellisService

    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.write_bytes(b"first-image")
    second.write_bytes(b"second-image")
    client = FakeClient(
        [
            {"glb_b64": base64.b64encode(b"first-glb").decode()},
            {"glb_b64": base64.b64encode(b"second-glb").decode()},
        ]
    )
    service = ModalTrellisService(app_name="unused", client=client)

    results = service.reconstruct_many(
        {"Chair0": first, "Chair1": second},
        out_dir=str(tmp_path / "out"),
        seed=7,
        simplify=0.9,
        texture_size=512,
        pipeline_type="512",
    )

    assert base64.b64decode(client.inputs[0]["image_b64"]) == b"first-image"
    assert client.inputs[0] == {
        "image_b64": client.inputs[0]["image_b64"],
        "seed": 7,
        "simplify": 0.9,
        "texture_size": 512,
        "pipeline_type": "512",
    }
    assert Path(results["Chair0"]).read_bytes() == b"first-glb"
    assert Path(results["Chair1"]).read_bytes() == b"second-glb"
    assert service.last_report.ok_count == 2


def test_modal_trellis_isolates_model_reported_errors(tmp_path):
    from lrauthor.models.trellis.modal import ModalTrellisService

    client = FakeClient([{"error": "generation failed"}])
    service = ModalTrellisService(app_name="unused", client=client)
    results = service.reconstruct_many({"Chair0": b"image"}, out_dir=str(tmp_path))

    assert results == {"Chair0": ""}
    assert service.last_report.ok_count == 0
    assert service.last_report.assets[0].error == "generation failed"


def test_registry_selects_modal_trellis(monkeypatch):
    from lrauthor.models import registry

    captured = {}

    class Service:
        def __init__(self, **options):
            captured.update(options)

    monkeypatch.setattr("lrauthor.models.trellis.modal.ModalTrellisService", Service)
    settings = _settings(modal_profile="huangzhening")

    assert isinstance(registry.gen3d_from_settings(settings), Service)
    assert captured == {
        "app_name": "litereality-trellis",
        "function_name": "generate",
        "environment_name": "main",
        "profile": "huangzhening",
        "credentials": None,
    }


def test_registry_never_implicitly_runs_trellis_locally():
    from lrauthor.models import registry

    # The app name now carries a default, so absent credentials leave TRELLIS unconfigured.
    settings = _settings()

    with pytest.raises(RuntimeError, match="TRELLIS is not configured"):
        registry.gen3d_from_settings(settings)


def test_registry_prefers_modal_when_only_a_profile_is_set(monkeypatch):
    """MODAL_PROFILE alone selects hosted TRELLIS: Modal is the default runtime."""
    from lrauthor.models import registry

    class Service:
        def __init__(self, **options):
            pass

    monkeypatch.setattr("lrauthor.models.trellis.modal.ModalTrellisService", Service)
    # trellis_python is set and still not used — configured Modal wins over the local runtime.
    settings = _settings(modal_profile="a-workspace", trellis_python=sys.executable)

    assert isinstance(registry.gen3d_from_settings(settings), Service)


def test_registry_falls_back_to_local_trellis_without_a_profile():
    """An explicit local GPU runtime stays reachable now that the app name always has a value."""
    from lrauthor.models import registry
    from lrauthor.models.trellis.service import LocalTrellisService

    settings = _settings(trellis_python=sys.executable)

    assert isinstance(registry.gen3d_from_settings(settings), LocalTrellisService)


def test_modal_client_uses_named_function_map():
    from lrauthor.runtimes.modal import ModalClient

    class Function:
        def map(self, inputs, *, return_exceptions):
            assert return_exceptions is True
            return ({"value": item["value"] * 2} for item in inputs)

    client = ModalClient("unused", "unused", function=Function())
    assert client.map([{"value": 2}, {"value": 3}]) == [{"value": 4}, {"value": 6}]


def test_modal_client_pins_the_configured_profile(monkeypatch):
    from lrauthor.runtimes.modal import ModalClient

    captured = {}
    modal = ModuleType("modal")

    class Function:
        @staticmethod
        def from_name(app_name, function_name, *, environment_name):
            captured.update(
                app_name=app_name,
                function_name=function_name,
                environment_name=environment_name,
            )
            return object()

    modal.Function = Function
    monkeypatch.setitem(sys.modules, "modal", modal)
    monkeypatch.delenv("MODAL_PROFILE", raising=False)

    ModalClient("app", "function", environment_name="main", profile="huangzhening")

    assert captured == {
        "app_name": "app",
        "function_name": "function",
        "environment_name": "main",
    }
    assert os.environ["MODAL_PROFILE"] == "huangzhening"
