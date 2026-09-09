"""render — the render|photo comparison for one target layer: room, a wall, or an object.

This is the tool the authoring model uses to look at its own work, so the failure that matters is
not a bad image — it is the tool refusing to start. In the recorded fallside run `render` raised
`could not infer scan` on its first call and the model never tried again: 88 tool calls, zero
render-verify. The offline tier pins that path; the real render needs Blender and a capture.
"""

from __future__ import annotations

import asyncio

import pytest

from lrauthor.agent.tools.render.tool import (
    RenderInvocation,
    RenderParams,
    RenderTool,
)


def _run(room, **kwargs):
    inv = RenderInvocation(RenderParams(**kwargs))
    inv.bind(str(room), None)
    return asyncio.run(inv.execute())


def test_schema_is_well_formed():
    fn = RenderTool().schema["function"]
    assert fn["name"] == "render"
    props = fn["parameters"]["properties"]
    assert "target" in props
    assert "frames" in props, "the model must be able to pin frames, not only auto-select"


@pytest.mark.parametrize("target", ["room", "Wall0", "Wall12", "Table0"])
def test_params_accept_every_target_layer(target):
    assert RenderParams(target=target).target == target


def test_frames_default_to_auto_selection():
    """Omitted frames means 'ask select_views' — the prompt tells the model to call it that way."""
    assert RenderParams(target="room").frames is None


def test_reaches_its_data_layer_through_a_symlinked_room(stage_tree):
    """Must fail on missing capture data, never on the shape of the path it was handed."""
    res = _run(stage_tree.symlinked_room, target="room", n=1)
    assert not res.is_success(), "fixture has no capture; success here means the assert is stale"
    assert "could not infer scan" not in (res.error or ""), (
        f"scan inference regressed — this is the failure that cost a whole authoring run: {res.error}"
    )


def test_failure_is_reported_not_raised(stage_tree):
    """The model has to receive a readable error it can act on, not an exception."""
    res = _run(stage_tree.symlinked_room, target="Wall0", n=1)
    assert res.error, "a failed render must carry an error string for the model to read"
    assert isinstance(res.error, str) and res.error.strip()


@pytest.mark.blender
@pytest.mark.scan
@pytest.mark.parametrize("target", ["room", "Wall0"])
def test_renders_a_real_room(example_scan, target):
    """Full comparison render. Run with `-m 'blender and scan'`."""
    res = _run(example_scan.room, target=target, n=2)
    assert res.is_success(), f"render failed on {example_scan.name}/{target}: {res.error}"
    assert str(res.output).strip(), "render succeeded but produced nothing to look at"
