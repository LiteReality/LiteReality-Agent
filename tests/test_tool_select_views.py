"""select_views — pick the best capture frames for a target, so the model never guesses frames.

Two tiers. The offline tier binds the tool to the synthetic `stage_tree` and asserts it reaches its
data layer: the failure that matters here is not a wrong frame, it is `could not infer scan`, which
in the recorded fallside run left the model with 88 tool calls and zero render-verify. The `-m scan`
tier runs the real thing on a real capture and asserts it returns usable frame indices.

The previous version of the scan test pointed at `_oneshot/room`, a path the layout restructure
removed, and read `$LR_SCANS_DIR` straight from `os.environ`, which pytest never populates from
`.env`. It therefore skipped on every machine while looking like coverage. Both are fixed by
binding to the `example_scan` fixture instead.
"""

from __future__ import annotations

import asyncio

import pytest

from lrauthor.agent.tools.select_views.tool import (
    SelectViewsInvocation,
    SelectViewsParams,
    SelectViewsTool,
)


def _run(room, **kwargs):
    inv = SelectViewsInvocation(SelectViewsParams(**kwargs))
    inv.bind(str(room), None)
    return asyncio.run(inv.execute())


# ── offline: schema and layout ────────────────────────────────────────────────────────────────
def test_schema_is_well_formed():
    """OpenAI function shape, since that is what the registry exports to every harness."""
    fn = SelectViewsTool().schema["function"]
    assert fn["name"] == "select_views"
    assert fn["description"].strip(), "a tool with no description is a tool the model misuses"
    assert "target" in fn["parameters"]["properties"]


@pytest.mark.parametrize("target", ["room", "Wall0", "Table0"])
def test_params_accept_every_target_shape_the_prompt_asks_for(target):
    """room / a wall / an object — the three layers the prompt tells the model to use."""
    assert SelectViewsParams(target=target, n=3).target == target


def test_reaches_its_data_layer_through_a_symlinked_room(stage_tree):
    """The fixture has no capture, so this must fail — but on missing DATA, not on layout.

    `could not infer scan` means the tool never got as far as looking for frames, which is the
    regression this pins.
    """
    res = _run(stage_tree.symlinked_room, target="room", n=2)
    assert not res.is_success(), "fixture has no capture data; success here means the assert is stale"
    assert "could not infer scan" not in (res.error or ""), (
        f"scan inference regressed — every image tool is dead when this breaks: {res.error}"
    )


# ── with a real capture ───────────────────────────────────────────────────────────────────────
@pytest.mark.scan
@pytest.mark.parametrize("target", ["room", "Wall0"])
def test_picks_frames_on_real_capture(example_scan, target):
    """End to end on a real room. No Blender: view selection only reads the capture poses."""
    res = _run(example_scan.room, target=target, n=3)
    assert res.is_success(), f"select_views failed on {example_scan.name}/{target}: {res.error}"

    frames = res.output["frames"]
    assert frames, f"no frames picked for {target}"
    assert all(isinstance(f["frame"], int) for f in frames), (
        "render indexes the capture with these — a non-int is unusable downstream"
    )


@pytest.mark.scan
def test_every_frame_it_picks_exists_on_disk(example_scan):
    """A frame render cannot open is the same as no frame, but fails later and less clearly."""
    res = _run(example_scan.room, target="room", n=3)
    assert res.is_success(), res.error

    available = {int(p.stem.split("_")[-1]) for p in example_scan.frames}
    assert available, f"no frame_*.jpg under {example_scan.capture}"

    picked = {f["frame"] for f in res.output["frames"]}
    assert picked <= available, (
        f"select_views named frames that are not in {example_scan.name}: "
        f"{sorted(picked - available)[:8]}"
    )
