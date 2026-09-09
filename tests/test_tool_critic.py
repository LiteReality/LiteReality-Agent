"""critic — the judge: grade render|photo images against a goal, returning {pass, score, issues}.

The verdict itself needs a VLM call, so grading is `-m live`. What tests offline is everything
around it: the tool must not send a request it cannot pay off (missing images), and a failed
judgement must come back as a readable error rather than an exception — the critic is called
inside the authoring loop, and an exception there ends the pass.
"""

from __future__ import annotations

import asyncio

import pytest

from lrauthor.agent.tools.critic.tool import (
    CriticInvocation,
    CriticParams,
    CriticTool,
)


def _run(**kwargs):
    return asyncio.run(CriticInvocation(CriticParams(**kwargs)).execute())


def test_schema_is_well_formed():
    fn = CriticTool().schema["function"]
    assert fn["name"] == "critic"
    props = fn["parameters"]["properties"]
    for required in ("images", "goal"):
        assert required in props


def test_goal_is_required():
    """Grading without a goal is grading against nothing — the verdict would be unfalsifiable."""
    with pytest.raises(Exception):
        CriticParams(images=["a.png"])


def test_optional_focus_and_labels_default_cleanly():
    p = CriticParams(images=["a.png"], goal="does the wall colour match the photo?")
    assert p.target is None
    assert p.labels is None


def test_missing_image_is_reported_as_an_error(monkeypatch):
    """A path that does not exist must come back as a readable error, not a crash.

    Note the tool has no pre-flight existence check: it guards only the empty list and hands the
    paths to `vision()`, so a missing file is caught by whatever the backend raises. That is safe
    but late — the request is already being assembled. Worth a cheap `is_file()` guard if these
    ever cost a call.
    """

    async def _reject(images, *a, **k):
        raise FileNotFoundError(images[0])

    monkeypatch.setattr("lrauthor.agent.tools._vlm.vision", _reject, raising=False)

    res = _run(images=["/definitely/not/here.png"], goal="does this match?")
    assert not res.is_success(), "grading a nonexistent image cannot succeed"
    assert "critic failed" in (res.error or ""), res.error


def test_empty_image_list_is_rejected():
    res = _run(images=[], goal="does this match?")
    assert not res.is_success()


def test_vlm_failure_is_reported_not_raised(tmp_path, monkeypatch):
    """An exception here ends the authoring pass; an error string lets the model carry on."""
    from PIL import Image

    img = tmp_path / "render.png"
    Image.new("RGB", (32, 32), (200, 200, 200)).save(img)

    async def _boom(*a, **k):
        raise RuntimeError("model overloaded")

    monkeypatch.setattr("lrauthor.agent.tools._vlm.vision", _boom, raising=False)

    res = _run(images=[str(img)], goal="does the wall colour match?")
    assert not res.is_success()
    assert isinstance(res.error, str) and res.error.strip()


@pytest.mark.live
def test_grades_a_real_image(tmp_path):
    """Makes a paid VLM call. Run with `-m live`."""
    from PIL import Image

    img = tmp_path / "flat.png"
    Image.new("RGB", (256, 256), (30, 90, 200)).save(img)

    res = _run(images=[str(img)], goal="the wall should be a plain blue surface")
    assert res.is_success(), f"critic failed on a real call: {res.error}"
    assert str(res.output).strip(), "a verdict with no content is not a verdict"
