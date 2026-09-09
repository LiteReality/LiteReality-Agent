"""grid — put a metric ruler on a rectified surface stitch.

Fully offline: the tool is PIL plus two paths off the harness config, so patching those gives a
real functional test rather than a schema smoke test. What matters is that the ruler is *metric* —
a grid drawn at the wrong pixels-per-metre is worse than none, because the model reads a confident
wrong measurement off it and edits Room.py to match.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from lrauthor.agent.tools.grid.tool import GridInvocation, GridParams, GridTool

PPM = 100.0  # pixels per metre — a 4 m wall at 2.6 m high renders 400 × 260


@pytest.fixture
def surface_ref(tmp_path, monkeypatch):
    """A stitch and its manifest, wired into the harness config the tool reads."""
    from PIL import Image

    from lrauthor.agent.tools.shared import config

    ref = tmp_path / "surface_ref"
    ref.mkdir()
    Image.new("RGB", (400, 260), (128, 128, 128)).save(ref / "Wall0_stitched.jpg")
    manifest = ref / "surface_ref_manifest.json"
    manifest.write_text(json.dumps({"ppm": PPM}), encoding="utf-8")

    monkeypatch.setattr(config, "SURFACE_REF", ref, raising=False)
    monkeypatch.setattr(config, "SURFACE_REF_MANIFEST", manifest, raising=False)
    return ref


def _run(**kwargs):
    return asyncio.run(GridInvocation(GridParams(**kwargs)).execute())


def test_schema_is_well_formed():
    fn = GridTool().schema["function"]
    assert fn["name"] == "grid"
    assert "surface" in fn["parameters"]["properties"]


def test_draws_a_grid_over_the_stitch(surface_ref):
    from PIL import Image

    res = _run(surface="Wall0", major_m=0.5, minor_m=0.1)
    assert res.is_success(), f"grid failed on a valid stitch: {res.error}"

    out = [p for p in surface_ref.rglob("*.jpg") if "grid" in p.name.lower()]
    out += [p for p in surface_ref.rglob("*.png") if "grid" in p.name.lower()]
    assert out, f"no gridded image written; tool said: {res.output}"

    gridded = Image.open(out[0]).convert("RGB")
    assert gridded.size == (400, 260), "the ruler must not resize the surface it measures"
    assert len(gridded.getcolors(maxcolors=1 << 16) or []) > 1, (
        "output is a flat image — nothing was actually drawn"
    )


@pytest.mark.parametrize("major,minor", [(0, 0.1), (0.5, 0), (-1, 0.1)])
def test_rejects_non_positive_spacing(surface_ref, major, minor):
    """A zero or negative step is a division trap, and silently drawing nothing is worse."""
    res = _run(surface="Wall0", major_m=major, minor_m=minor)
    assert not res.is_success()
    assert "positive" in (res.error or "").lower()


def test_missing_stitch_says_which_surface(surface_ref):
    res = _run(surface="Wall99")
    assert not res.is_success()
    assert "Wall99" in (res.error or ""), "the error has to name the surface the model asked for"


def test_missing_manifest_is_actionable(tmp_path, monkeypatch):
    """No ppm means no metric scale. Refuse rather than draw an unlabelled decorative grid."""
    from lrauthor.agent.tools.shared import config

    monkeypatch.setattr(config, "SURFACE_REF", tmp_path, raising=False)
    monkeypatch.setattr(config, "SURFACE_REF_MANIFEST", tmp_path / "missing.json", raising=False)

    res = _run(surface="Wall0")
    assert not res.is_success()
    assert "stitch" in (res.error or "").lower() or "manifest" in (res.error or "").lower()
