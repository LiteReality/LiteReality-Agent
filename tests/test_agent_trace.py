"""The run trace — the record of what an authoring pass actually did.

Two failures already happened here and both were invisible from inside a run: the passes shared one
trace file so the later pass wiped the earlier one, and the tool accounting reads fields off a block
type that does not carry them. Neither breaks a run; both destroy the evidence.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from lrauthor.agent.trace import AgentTrace


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_each_pass_writes_its_own_file(tmp_path):
    """THE REGRESSION (fixed in 30a1778): every pass truncates its file on start, so a shared path
    meant the materials pass destroyed the authoring pass's trace mid-run."""
    author = AgentTrace("author", room=tmp_path)
    author.start(model="m")
    author.tool("Read", {"file_path": "/x/Wall0_stitched.jpg"})

    materials = AgentTrace("materials", room=tmp_path)
    materials.start(model="m")

    assert author.path != materials.path
    assert len(_events(author.path)) == 2, "the second pass truncated the first pass's trace"


def test_edit_events_name_the_file_and_the_size_of_the_change(tmp_path):
    tr = AgentTrace("author", room=tmp_path)
    tr.tool("Edit", {"file_path": "/room/Room.py", "old_string": "a\nb", "new_string": "a\nb\nc\nd"})

    (event,) = [e for e in _events(tr.path) if e["kind"] == "tool"]
    assert event["tool"] == "Edit" and event["file"] == "Room.py"
    assert event["delta_lines"] == 2


def test_mcp_prefixes_are_stripped_so_tools_count_under_one_name(tmp_path):
    """The model calls `mcp__cap__render`; the report is about `render`. Without stripping, the same
    tool is counted twice under two names."""
    tr = AgentTrace("author", room=tmp_path)
    tr.tool("mcp__cap__render", {"target": "Wall0"})
    tr.tool("render", {"target": "Wall1"})

    events = [e for e in _events(tr.path) if e["kind"] == "tool"]
    assert [e["tool"] for e in events] == ["render", "render"]
    assert [e["n"] for e in events] == [1, 2]


def test_end_records_the_totals(tmp_path):
    tr = AgentTrace("author", room=tmp_path)
    tr.tool("Read", {"file_path": "/x.jpg"})
    tr.end(calls=1, cost_usd=1.5, summary="done")

    (end,) = [e for e in _events(tr.path) if e["kind"] == "session_end"]
    assert end["calls"] == 1 and end["cost_usd"] == 1.5 and end["counts"] == {"Read": 1}


def test_a_broken_trace_never_breaks_the_run(tmp_path):
    """A trace is telemetry. If its directory is unwritable the pass must continue regardless."""
    tr = AgentTrace("author", room=tmp_path)
    tr.path.parent.chmod(0o500)
    try:
        tr.tool("Read", {"file_path": "/x.jpg"})  # must not raise
        tr.end(calls=1)
    finally:
        tr.path.parent.chmod(0o700)


def test_author_attributes_a_successful_stitch_read(tmp_path, monkeypatch, capsys):
    """The author must recover a result's name and input from its matching use block."""
    from claude_agent_sdk.types import ToolResultBlock, ToolUseBlock

    from lrauthor.agent import author, providers

    room = tmp_path / "run" / "scan" / "realism_authoring" / "room"
    room.mkdir(parents=True)
    (room / "Room.py").write_text("SHELL = {'walls': {}}\n", encoding="utf-8")
    refs = tmp_path / "refs"
    refs.mkdir()
    stitch = refs / "Wall0_stitched.jpg"
    stitch.write_bytes(b"reference")
    scan = tmp_path / "scan"
    scan.mkdir()

    class ReadHarness:
        name = "test"
        supports = frozenset({"read_events"})

        def effective_model(self, spec):
            return spec.model or "test"

        async def run(self, _spec):
            use = ToolUseBlock(id="read-1", name="Read", input={"file_path": str(stitch)})
            result = ToolResultBlock(tool_use_id="read-1", content="ok", is_error=False)
            yield SimpleNamespace(content=[use, result])

    monkeypatch.setattr(author, "surfaces_for", lambda _room: ["Wall0"])
    monkeypatch.setattr(providers, "resolve", lambda *_args: ReadHarness())
    monkeypatch.setattr("lrauthor.agent.scratch.bind", lambda **_kwargs: None)
    monkeypatch.setattr("lrauthor.agent.scratch.prompt_line", lambda: "")
    monkeypatch.setattr("lrauthor.agent.scratch.rescue", lambda *_args, **_kwargs: [])

    assert asyncio.run(author.run(room, refs, scan, "test", 1)) == 0
    output = capsys.readouterr().out
    assert "calls=1 {'Read': 1}" in output
    assert "✓ Wall0" in output
    assert "NEVER OPENED" not in output
