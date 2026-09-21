"""Acceptance must fail closed, be bounded, and belong to the exact exported build."""

import asyncio
import json
from pathlib import Path

import pytest

from litereality_agent.agent.providers.base import SessionResult, SessionSpec
from litereality_agent.agent.providers.codex import _codex_model, _mcp_config
from litereality_agent.pipeline.context import RunContext
from litereality_agent.pipeline.realism_authoring import acceptance as A
from litereality_agent.pipeline.result import StageStatus
from litereality_agent.pipeline.room_qc import export_support as E
from litereality_agent.pipeline.room_qc import publish


def context(tmp_path):
    c = RunContext("test", tmp_path / "capture", tmp_path / "scene", tmp_path)
    c.authored_room.mkdir(parents=True)
    c.seed_room.mkdir(parents=True)
    for room in (c.seed_room, c.authored_room):
        (room / "Room.py").write_text("SHELL = {}")
        (room / "manifest.json").write_text('{"assets": []}')
    return c


@pytest.mark.parametrize(
    "review",
    [
        None,
        {},
        {"pass": True},
        {"pass": True, "score": 9, "all_objects_reviewed": True, "issues": ["floating"]},
        {"pass": True, "score": float("nan"), "all_objects_reviewed": True, "issues": []},
        {"pass": "true", "score": 9, "all_objects_reviewed": True, "issues": []},
    ],
)
def test_visual_acceptance_requires_complete_typed_evidence(review):
    assert not A.visual_accepted(review or {})


def test_visual_acceptance_passes_complete_review():
    assert A.visual_accepted(dict(pass_=False)) is False
    assert A.visual_accepted({"pass": True, "score": 8, "all_objects_reviewed": True, "issues": []})


def test_repair_loop_stops_at_cap_without_accepting(tmp_path, monkeypatch):
    c = context(tmp_path)
    calls = []
    monkeypatch.setattr(A, "run_module", lambda *a, **kw: (0, None))
    monkeypatch.setattr(A, "assess", lambda *a: {"accepted": False})

    async def repair(*a, **kw):
        calls.append(kw)

    monkeypatch.setattr(A, "repair", repair)
    r = A.finish(c, {"repair_rounds": 2, "repair_steps": 7, "repair_seconds": 20})
    assert not r["accepted"] and len(calls) == 2
    assert calls == [{"steps": 7, "seconds": 20}] * 2
    assert (c.authored_room / "Room.py").is_file()


def test_interrupted_repair_is_never_accepted(tmp_path, monkeypatch):
    c = context(tmp_path)
    monkeypatch.setattr(A, "run_module", lambda *a, **kw: (0, None))
    monkeypatch.setattr(A, "assess", lambda *a: {"accepted": False})

    async def repair(*a, **kw):
        raise RuntimeError("wall time budget reached")

    monkeypatch.setattr(A, "repair", repair)
    r = A.finish(c, {})
    assert r["status"] == "needs_repair" and not r["accepted"]


def test_failed_build_does_not_validate_stale_export(tmp_path, monkeypatch):
    c = context(tmp_path)
    monkeypatch.setattr(A, "run_module", lambda *a, **kw: (1, Path("build.log")))
    monkeypatch.setattr(A, "assess", lambda *a: pytest.fail("stale build evaluated"))
    assert not A.finish(c, {})["accepted"]


def test_failed_export_acceptance_blocks_publish(tmp_path, monkeypatch):
    c = context(tmp_path)

    def run(*args, **kwargs):
        c.preview_dir.mkdir(parents=True, exist_ok=True)
        (c.preview_dir / "Room.glb").write_bytes(b"candidate")
        return 0, None

    monkeypatch.setattr(publish, "run_module", run)
    monkeypatch.setattr("litereality_agent.room_ops.api.bake_room", lambda *a: 0)
    monkeypatch.setattr(A, "assess", lambda *a: {"accepted": False, "support_pass": False})
    result = publish.run(c, {"compare_frames": 0})
    assert result.status is StageStatus.FAILED
    assert (c.preview_dir / "Room.glb").is_file()
    assert not publish.complete(c)


def test_merged_membership_cannot_be_removed(tmp_path):
    c = context(tmp_path)
    (c.seed_room / "manifest.json").write_text(
        json.dumps({"assets": [{"object": "Chairs", "represents_prims": ["Chair0", "Chair1"]}]})
    )
    assert not A.preserved_instances(c.seed_room, c.authored_room)


def moving_scene(parent_contents=False, part=True):
    nodes = [{"name": "Table", "children": [1]}, {"name": "Desktop"}, {"name": "Cup"}]
    if parent_contents:
        nodes[1]["children"] = [2]
    return (
        {"nodes": nodes, "animations": [{"channels": [{"target": {"node": 1}}]}]},
        [
            {"id": "Table"},
            {"id": "Cup", "rests_on": "Table", **({"support_part": "Desktop"} if part else {})},
        ],
    )


def test_moving_support_requires_the_named_link_and_parenting():
    doc, objects = moving_scene(part=False)
    assert E.check(doc, objects, {})[0]["kind"] == "moving_support_part_required"
    doc, objects = moving_scene()
    assert E.check(doc, objects, {})[0]["kind"] == "support_motion_not_preserved"
    doc, objects = moving_scene(parent_contents=True)
    assert E.check(doc, objects, {}) == []


def test_export_cannot_drop_merged_instances():
    findings = E.check(
        {"nodes": [{"name": "Chair0"}]},
        [{"id": "Chair0"}],
        {"assets": [{"object": "Group", "represents_prims": ["Chair0", "Chair1"]}]},
    )
    assert findings == [{"id": "Chair1", "kind": "source_instance_missing_or_ambiguous"}]


def test_codex_is_explicit_and_per_object_tool_is_bridged(monkeypatch):
    monkeypatch.delenv("LR_CODEX_MODEL", raising=False)
    assert _codex_model() == "gpt-6-astra"
    spec = SessionSpec(
        prompt="x",
        cwd=Path("."),
        stdio_mcp={"obj": {"command": "python", "args": ["-m", "object_server"]}},
    )
    flags = _mcp_config(spec)
    assert 'mcp_servers.obj.command="python"' in flags
    assert 'mcp_servers.obj.args=["-m", "object_server"]' in flags


def test_visual_critic_routes_through_quality_without_fallback(monkeypatch):
    from litereality_agent.agent.tools import _vlm

    seen = []

    class Fake:
        async def run(self, spec):
            seen.append(spec)
            yield SessionResult(result='{"pass":true}')

    monkeypatch.setattr(
        "litereality_agent.agent.providers.resolve",
        lambda role: Fake() if role == "quality" else None,
    )
    result = asyncio.run(_vlm.vision(["/tmp/image.png"], "judge", json_mode=True))
    assert result["pass"] is True and seen[0].read_only
    assert seen[0].timeout_seconds == 240
