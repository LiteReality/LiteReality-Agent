"""A long authoring session must not be able to lose its work.

`the CLI` runs authoring as a HARD stage — a non-zero exit aborts the whole pipeline. The SDK
raises when `--max-turns` is reached, so a 200-turn session that used its budget threw away
every edit it had made and killed the run, despite `Room.py` being edited IN PLACE and sitting
valid on disk the entire time. Running out of turns means "time's up", not "this is broken".

What actually matters downstream is narrow: every later stage (materials, refine, qc, export)
builds `Room.py`, so it must be valid Python. That is what these pin.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from litereality_agent.agent.author import checkpoint, room_compiles

GOOD = "SHELL = {'walls': {}}\n\n\ndef build():\n    return SHELL\n"
BROKEN = "SHELL = {'walls': {\n\ndef build(:\n"


@pytest.fixture
def room(tmp_path: Path) -> Path:
    d = tmp_path / "realism_authoring" / "room"
    d.mkdir(parents=True)
    (d / "Room.py").write_text(GOOD, encoding="utf-8")
    return d


def test_a_valid_room_reports_no_error(room: Path):
    assert room_compiles(room) == ""


def test_a_broken_room_is_reported_with_the_line(room: Path):
    (room / "Room.py").write_text(BROKEN, encoding="utf-8")
    err = room_compiles(room)
    assert err and "line" in err


def test_a_missing_room_is_an_error_not_a_crash(tmp_path: Path):
    assert room_compiles(tmp_path) != ""


def test_only_a_compiling_room_is_checkpointed(room: Path, tmp_path: Path):
    """Saving a broken file as last-known-good would defeat the entire mechanism."""
    ckpt = tmp_path / ".room_checkpoint.py"
    assert checkpoint(room, ckpt) is True
    assert ckpt.read_text() == GOOD

    (room / "Room.py").write_text(BROKEN, encoding="utf-8")
    assert checkpoint(room, ckpt) is False
    assert ckpt.read_text() == GOOD, "the checkpoint must still hold the last GOOD version"


def test_the_checkpoint_advances_with_each_good_edit(room: Path, tmp_path: Path):
    """A break should cost one edit, not the run — so the newest compiling version wins."""
    ckpt = tmp_path / ".room_checkpoint.py"
    checkpoint(room, ckpt)
    later = GOOD + "\nWALL_COLOUR = (0.8, 0.8, 0.75)\n"
    (room / "Room.py").write_text(later, encoding="utf-8")
    assert checkpoint(room, ckpt) is True
    assert "WALL_COLOUR" in ckpt.read_text()


def test_restoring_a_checkpoint_yields_a_compiling_room(room: Path, tmp_path: Path):
    """The recovery path the run performs when a session dies mid-edit."""
    ckpt = tmp_path / ".room_checkpoint.py"
    checkpoint(room, ckpt)
    (room / "Room.py").write_text(BROKEN, encoding="utf-8")
    assert room_compiles(room) != ""

    (room / "Room.py").write_bytes(ckpt.read_bytes())  # what run() does
    assert room_compiles(room) == ""


def test_the_checkpoint_lives_outside_the_room(room: Path):
    """The room dir is copied and scanned wholesale downstream; a second Room-ish .py inside it
    would be picked up as scene code."""
    ckpt = room.parent / ".room_checkpoint.py"
    checkpoint(room, ckpt)
    assert ckpt.is_file()
    assert ckpt.parent != room
    assert not list(room.glob("*checkpoint*"))


class _ResultHarness:
    name = "test"
    supports = frozenset()

    def __init__(self, result):
        self.result = result

    def effective_model(self, spec):
        return spec.model or "test"

    async def run(self, _spec):
        yield self.result


class _CancelledHarness:
    name = "test"
    supports = frozenset()

    def effective_model(self, spec):
        return spec.model or "test"

    async def run(self, _spec):
        if False:
            yield None
        raise asyncio.CancelledError


def _run_with_result(room, tmp_path, monkeypatch, result):
    from litereality_agent.agent import author, providers

    refs = tmp_path / "refs"
    scan = tmp_path / "scan"
    refs.mkdir(exist_ok=True)
    scan.mkdir(exist_ok=True)
    monkeypatch.setattr(author, "surfaces_for", lambda _room: [])
    monkeypatch.setattr(providers, "resolve", lambda *_args: _ResultHarness(result))
    monkeypatch.setattr("litereality_agent.agent.scratch.bind", lambda **_kwargs: None)
    monkeypatch.setattr("litereality_agent.agent.scratch.prompt_line", lambda: "")
    return asyncio.run(author.run(room, refs, scan, "test", 1))


def test_cancelling_authoring_stops_the_run(room, tmp_path, monkeypatch):
    from litereality_agent.agent import author, providers

    refs = tmp_path / "refs"
    scan = tmp_path / "scan"
    refs.mkdir(exist_ok=True)
    scan.mkdir(exist_ok=True)
    monkeypatch.setattr(author, "surfaces_for", lambda _room: [])
    monkeypatch.setattr(providers, "resolve", lambda *_args: _CancelledHarness())
    monkeypatch.setattr("litereality_agent.agent.scratch.bind", lambda **_kwargs: None)
    monkeypatch.setattr("litereality_agent.agent.scratch.prompt_line", lambda: "")

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(author.run(room, refs, scan, "test", 1))


def test_cancelling_object_refinement_is_not_retried(room, tmp_path, monkeypatch):
    from PIL import Image

    from litereality_agent.agent import providers
    from litereality_agent.pipeline.author.realism import refine_objects

    name = "Chair0"
    obj_dir = room / "Objects" / "Procedural" / name
    obj_dir.mkdir(parents=True)
    (obj_dir / "object.py").write_text("def build():\n    return None\n", encoding="utf-8")
    selected = tmp_path / "refs" / name / "images" / "frame_00001.jpg"
    selected.parent.mkdir(parents=True)
    selected.write_bytes(b"selected")
    scan = tmp_path / "scan"
    scan.mkdir()
    Image.new("RGB", (4, 4), "white").save(scan / "frame_00001.jpg")

    harness = _CancelledHarness()
    harness.supports = frozenset({"inproc_tools"})
    monkeypatch.setattr(providers, "resolve", lambda *_args: harness)
    monkeypatch.setattr(refine_objects, "build_server", lambda *_args: (object(), []))
    monkeypatch.setattr(refine_objects, "_snapshot_views", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(refine_objects, "_is_articulated", lambda *_args: False)
    monkeypatch.setattr(refine_objects, "RESULTS", tmp_path / "results")
    refine_objects.RESULTS.mkdir()
    monkeypatch.setattr(refine_objects, "_SEM", asyncio.Semaphore(1))

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            refine_objects.refine_one(
                name,
                room,
                tmp_path / "refs",
                scan,
                "blender",
                "test",
                1,
                {},
            )
        )


def test_terminal_provider_error_fails_authoring(room, tmp_path, monkeypatch):
    from litereality_agent.agent.providers import SessionResult

    before = (room / "Room.py").read_text(encoding="utf-8")
    result = SessionResult(result="authentication failed", is_error=True)

    assert _run_with_result(room, tmp_path, monkeypatch, result) == 1
    assert (room / "Room.py").read_text(encoding="utf-8") == before


def test_deliberate_provider_stop_keeps_partial_success(room, tmp_path, monkeypatch):
    from litereality_agent.agent.providers import SessionResult

    result = SessionResult(result="budget reached", stopped="step budget")

    assert _run_with_result(room, tmp_path, monkeypatch, result) == 0


def test_polish_passes_remain_in_the_author_flow(tmp_path: Path, monkeypatch):
    from litereality_agent.pipeline.author import realism as author
    from litereality_agent.pipeline.context import RunContext
    from litereality_agent.settings import LiteRealitySettings

    settings = LiteRealitySettings(repo_root=tmp_path, output_root=tmp_path / "run")
    context = RunContext(
        "scan",
        tmp_path / "capture" / "scan",
        tmp_path / "run" / "scan",
        tmp_path / "run",
        repo_root=tmp_path,
        settings=settings,
    )
    context.seed_room.mkdir(parents=True)
    (context.seed_room / "Room.py").write_text(GOOD, encoding="utf-8")
    evidence = context.authoring_root / "surface_ref" / "surface_ref_manifest.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("{}", encoding="utf-8")

    calls = []

    def fake_run_module(_context, module, args, *, log_name=None):
        calls.append((module, list(args), log_name))
        return 0, None

    monkeypatch.setattr(author, "run_module", fake_run_module)
    result = author.run(
        context,
        {"refine_objects": True, "materials": True, "quality_pass": True},
    )

    assert result.ok
    assert [module for module, _, _ in calls] == [
        "litereality_agent.pipeline.author.realism.entrypoint",
        "litereality_agent.pipeline.author.realism.refine_objects",
        "litereality_agent.pipeline.author.realism.materials",
        "litereality_agent.pipeline.author.realism.quality",
    ]
