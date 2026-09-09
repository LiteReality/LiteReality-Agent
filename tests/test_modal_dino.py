from __future__ import annotations

import base64

from lrauthor.models.grounding_dino.modal import ModalDinoService
from lrauthor.settings import LiteRealitySettings


class Client:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.inputs = []

    def map(self, inputs):
        self.inputs.extend(inputs)
        return [self.outputs.pop(0)]


def test_detect_serializes_image_and_parses_boxes():
    client = Client([{"detections": [{"box": [1, 2, 3, 4], "score": 0.9, "label": "chair"}]}])
    service = ModalDinoService(app_name="unused", client=client, model_id="detector")

    detections = service.detect(b"png", "chair .", upright=False)

    assert base64.b64decode(client.inputs[0]["image_b64"]) == b"png"
    assert client.inputs[0]["model_id"] == "detector"
    assert detections[0].box_xyxy == (1.0, 2.0, 3.0, 4.0)


def test_embed_uses_embedding_model():
    client = Client([{"embeddings": [[0.25, 0.75]]}])
    service = ModalDinoService(app_name="unused", client=client, embed_model_id="embedder")

    assert service.embed([b"one"]) == [[0.25, 0.75]]
    assert client.inputs[0]["op"] == "embed"
    assert client.inputs[0]["model_id"] == "embedder"


def test_registry_prefers_modal_dino(monkeypatch):
    from lrauthor.models import registry

    captured = {}

    class Service:
        def __init__(self, **options):
            captured.update(options)

    monkeypatch.setattr("lrauthor.models.grounding_dino.modal.ModalDinoService", Service)
    settings = LiteRealitySettings(
        _env_file=None,
        modal_profile="huangzhening",
        dino_model="detector",
        dino_embed_model="embedder",
    )

    assert isinstance(registry.detection_from_settings(settings), Service)
    assert captured == {
        "app_name": "litereality-dino",
        "function_name": "infer",
        "environment_name": "main",
        "profile": "huangzhening",
        "credentials": None,
        "model_id": "detector",
        "embed_model_id": "embedder",
    }


def test_registry_selects_modal_dino_from_tokens_alone(monkeypatch):
    """A token pair in .env is credentials enough — no ~/.modal.toml profile required."""
    from lrauthor.models import registry

    captured = {}

    class Service:
        def __init__(self, **options):
            captured.update(options)

    monkeypatch.setattr("lrauthor.models.grounding_dino.modal.ModalDinoService", Service)
    settings = LiteRealitySettings(
        _env_file=None, MODAL_TOKEN_ID="ak-1", MODAL_TOKEN_SECRET="as-2"
    )

    assert isinstance(registry.detection_from_settings(settings), Service)
    assert captured["credentials"] == ("ak-1", "as-2")
    assert captured["profile"] is None
