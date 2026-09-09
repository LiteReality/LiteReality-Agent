from __future__ import annotations

from litereality_agent.pipeline.measure.ingest.preprocessing import object_images


def test_object_image_adapter_passes_project_crop_setting(monkeypatch):
    received = {}

    def process(*args, **kwargs):
        received["args"] = args
        received["kwargs"] = kwargs
        return "done"

    monkeypatch.setenv("LR_ENLARGED_CROP_OBJECTS", "Sink0,Storage0")
    monkeypatch.setattr(object_images, "_process_object_images", process)

    result = object_images.process_object_images(
        "office",
        ["walls"],
        ["objects"],
        {"holes": []},
        include_walls=True,
    )

    assert result == "done"
    assert received["args"] == (
        "office",
        ["walls"],
        ["objects"],
        {"holes": []},
    )
    assert received["kwargs"]["include_walls"] is True
    assert set(received["kwargs"]["enlarged_crop_objects"]) == {"Sink0", "Storage0"}
