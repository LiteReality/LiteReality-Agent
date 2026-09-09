"""Liveness of the reconstruction phase in the trace the live viewer tails.

The viewer calls the agent quiet when the newest trace event is older than 120s. Builders only
wrote `agent_step` on completion, and an object takes 9-20 minutes, so a phase with sixteen agents
working reported "quiet" for nearly all of it. What matters is therefore not that events exist but
that the GAP between them stays under the threshold — which is what these pin.
"""

from __future__ import annotations

import json

import pytest

from lrauthor import telemetry

# live/page.py: const LIVE_S = 120 — the viewer's quiet threshold.
LIVE_S = 120


@pytest.fixture
def trace(tmp_path):
    telemetry.start("office", tmp_path)
    return tmp_path / "trace.jsonl"


def _events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_heartbeat_reaches_the_trace_the_viewer_tails(trace):
    telemetry.event("build_progress", scan="office", built=3, total=26, branches={}, active=[])
    kinds = [e["kind"] for e in _events(trace)]
    assert "build_progress" in kinds


def test_heartbeat_carries_what_a_long_phase_needs_to_show(trace):
    telemetry.event(
        "build_progress", scan="office", built=14, total=26,
        branches={"procedural": "14/16", "openings": "0/7"}, active=["Storage8"],
    )
    rec = [e for e in _events(trace) if e["kind"] == "build_progress"][-1]
    assert rec["built"] == 14 and rec["total"] == 26
    assert rec["branches"]["procedural"] == "14/16"
    assert rec["active"] == ["Storage8"]


def test_beat_interval_keeps_gaps_under_the_viewers_quiet_threshold():
    """A heartbeat slower than LIVE_S would leave the viewer reporting quiet between beats."""
    from lrauthor.pipeline.measure import flow

    source = flow._reconstruction_phase.__code__.co_consts
    interval = next(c for c in source if isinstance(c, float) and 0 < c <= LIVE_S)
    assert interval < LIVE_S / 2, "heartbeat must beat well inside the viewer's quiet threshold"


def test_the_gap_agent_step_alone_leaves_is_what_broke_this():
    """Regression record: consecutive per-object completions on Elliott-Studio, in seconds.

    Both exceed LIVE_S by an order of magnitude, which is why completion events alone cannot carry
    liveness. If someone removes the heartbeat, this is the shape of what comes back.
    """
    observed_gaps_s = [19.0 * 60, 38.3 * 60]
    assert all(gap > LIVE_S for gap in observed_gaps_s)
