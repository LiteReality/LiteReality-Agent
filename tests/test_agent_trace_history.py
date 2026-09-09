"""A re-run must not destroy the trace of the run before it.

`AgentTrace` truncated its file on start, so only the most recent run was ever traced — and
re-running to check something was the act that deleted the record of the run that raised the
question. Traces are archived under `<traces>/history/` instead, tagged with the same `run_NNN`
that names the run's scratch directory so the events and the images stay paired.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lrauthor.agent import scratch
from lrauthor.agent.trace import AgentTrace, _run_id_of

SCAN = "test-scan-Room"


@pytest.fixture
def traces(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A trace dir the AgentTrace path resolution will find, plus a bound scratch root."""
    out = tmp_path / "run"
    (out / SCAN / "scene_init" / "obj_stage" / "traces").mkdir(parents=True)
    monkeypatch.setenv("LITEREALITY_FINAL", str(out))
    monkeypatch.setenv("LITEREALITY_OUTPUT", str(out))
    monkeypatch.setenv("LR_AUTHORING", str(out / SCAN / "realism_authoring"))
    monkeypatch.delenv("LR_SCRATCH", raising=False)
    monkeypatch.delenv("LR_RUN_ID", raising=False)
    return out / SCAN / "scene_init" / "obj_stage" / "traces"


def _session(pass_name: str = "author") -> AgentTrace:
    """One full pass: bind scratch (as the stages do), trace a call, end."""
    where = scratch.bind()
    tr = AgentTrace(pass_name, scan=SCAN)
    tr.start(model="claude-opus-5", scratch=str(where) if where else None)
    tr.tool("Bash", {"command": "echo hi"}, tool_id="tu_1")
    tr.raw({"msg": "verbatim"})   # before end(): end() closes the handles
    tr.end(calls=1)
    return tr


def test_a_rerun_archives_the_previous_trace(traces: Path):
    first = _session()
    first_text = first.path.read_text()
    _session()

    archived = [p for p in sorted((traces / "history").glob("*.jsonl")) if ".raw." not in p.name]
    assert archived, "the previous run's trace must survive"
    assert archived[0].read_text() == first_text


def test_the_live_path_keeps_its_name(traces: Path):
    """The viewer and the report read a fixed filename; archiving must not move the current run."""
    _session()
    _session()
    assert (traces / "authoring_trace.author.jsonl").is_file()


def test_the_archive_is_tagged_with_the_run_that_wrote_it(traces: Path):
    """Not `$LR_RUN_ID` — by archive time that names the run doing the archiving, one ahead of
    the run being archived. Off by one, and the trace would point at the wrong images."""
    first = _session()
    wrote = json.loads(first.path.read_text().splitlines()[0])["scratch"]
    _session()

    archived = sorted((traces / "history").glob("authoring_trace.author.*.jsonl"))[0]
    assert Path(wrote).name in archived.name
    assert "run_001" in archived.name and "run_002" not in archived.name


def test_the_raw_sidecar_archives_under_the_same_id(traces: Path):
    _session()
    _session()

    hist = sorted(p.name for p in (traces / "history").glob("*.jsonl"))
    assert any(n.startswith("authoring_trace.author.raw.") for n in hist), hist
    ids = {n.split(".")[-2] for n in hist}
    assert ids == {"run_001"}, f"curated and raw must share one id: {hist}"


def test_three_runs_leave_two_archives_and_one_live(traces: Path):
    for _ in range(3):
        _session()
    assert len(sorted((traces / "history").glob("authoring_trace.author.run_*.jsonl"))) == 2
    assert (traces / "authoring_trace.author.jsonl").is_file()


def test_an_empty_trace_is_not_archived(traces: Path):
    """A crashed start leaves a zero-byte file; archiving it just clutters history."""
    AgentTrace("author", scan=SCAN)
    AgentTrace("author", scan=SCAN)
    assert not list((traces / "history").glob("*.jsonl"))


def test_passes_archive_independently(traces: Path):
    """One file per pass — the materials pass once wiped the authoring pass's record."""
    _session("author")
    _session("qc")
    _session("author")
    hist = [p.name for p in (traces / "history").glob("*.jsonl")]
    assert any("author" in n for n in hist)
    assert (traces / "authoring_trace.qc.jsonl").is_file()


def test_run_id_of_a_trace_without_scratch_is_empty(traces: Path, tmp_path: Path):
    """Traces written before per-run scratch existed fall back to sequential numbering."""
    old = tmp_path / "old.jsonl"
    old.write_text(json.dumps({"kind": "session_start", "model": "x"}) + "\n", encoding="utf-8")
    assert _run_id_of(old) == ""


def test_both_halves_of_one_session_archive_under_one_id(traces: Path):
    """The raw sidecar has no session_start and borrows the curated trace's. Resolving that
    AFTER the curated file was blanked fell back to counting, so one session's two files landed
    under different ids (`...run_001_2.jsonl` beside `...raw.run_002.jsonl`)."""
    _session()
    _session()

    hist = sorted(p.name for p in (traces / "history").glob("*.jsonl"))
    assert len(hist) == 2, hist
    ids = {n.rsplit(".", 2)[-2] for n in hist}
    assert ids == {"run_001"}, f"one session must archive under one id, got {hist}"
    assert not any("_2." in n for n in hist), f"collision suffix means the ids diverged: {hist}"


def test_a_rotation_underneath_a_live_session_does_not_cross_contaminate(traces: Path):
    """Observed for real: a 37-minute run was still writing when shorter runs rotated the trace
    path underneath it, so its later events landed in THEIR files — one trace holding two
    interleaved sessions, both unreadable. Events must follow the handle, not the name."""
    live = AgentTrace("author", scan=SCAN)
    live.start(model="long-run")
    live.tool("Read", {"file_path": "a.jpg"}, tool_id="tu_a")

    _session()  # a second run starts: archives and re-creates the same path

    live.tool("Edit", {"file_path": "Room.py"}, tool_id="tu_b")  # the long run continues
    live.end(calls=2)

    fresh = json.loads((traces / "authoring_trace.author.jsonl").read_text().splitlines()[0])
    assert fresh["kind"] == "session_start"
    for path in (traces / "authoring_trace.author.jsonl", *(traces / "history").glob("*.jsonl")):
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        seqs = [json.loads(ln)["seq"] for ln in lines]
        assert seqs == sorted(seqs), f"{path.name} interleaves two sessions: {seqs}"
        starts = [ln for ln in lines if '"session_start"' in ln]
        assert len(starts) <= 1, f"{path.name} holds {len(starts)} sessions"


def test_events_are_flushed_as_they_happen(traces: Path):
    """A killed run — the common case for a long authoring session — must still leave a trace."""
    tr = AgentTrace("author", scan=SCAN)
    tr.start(model="x")
    tr.tool("Bash", {"command": "echo hi"}, tool_id="tu_1")
    assert "echo hi" in tr.path.read_text(), "unflushed events are lost on kill -9"
