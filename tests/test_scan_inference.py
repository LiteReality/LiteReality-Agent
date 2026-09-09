"""Scan inference — the single choke point every image tool passes through.

`_scan_from_room` is called by `render`, `select_views`, `survey` and `compose` itself (6 call
sites). When it raises, the authoring model has no way to look at its own work: in the recorded
fallside stage-3 run it called `render` once, got

    render failed: RuntimeError: view selection failed: view selection failed:
    ValueError: could not infer scan from room dir .../run/<scan>/scene_init/scene_stage/_oneshot/room

and never tried again — 88 tool calls with zero render-verify. These tests pin the layout contract
so that failure cannot come back quietly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lrauthor.agent.tools.render.source.compose import _scan_from_room


def test_scan_from_physical_deliverables_path(stage_tree):
    assert _scan_from_room(stage_tree.physical_room) == stage_tree.scan


def test_scan_from_output_symlink_path(stage_tree):
    """THE REGRESSION. The path as given and its `.resolve()` name the same room through two
    different roots. Matching the stage root against only one of the two spellings raised — while
    printing the unresolved path in the message, which is why the error read as nonsense.
    """
    assert stage_tree.symlinked_room.is_dir(), "fixture must reproduce the the CLI symlink"
    assert "staging" in stage_tree.symlinked_room.parts
    assert "deliverables" in stage_tree.symlinked_room.resolve().parts
    assert _scan_from_room(stage_tree.symlinked_room) == stage_tree.scan


def test_scan_from_room_py_file_path(stage_tree):
    """Tools bind either the room dir or its Room.py; `room_dir_from` normalizes, but callers in
    the wild pass both, so both spellings must land on the same scan."""
    from lrauthor.agent.tools._scene import room_dir_from

    assert _scan_from_room(room_dir_from(str(stage_tree.symlinked_room / "Room.py"))) == stage_tree.scan


def test_deeper_room_paths_still_resolve(stage_tree):
    """Per-object rooms sit further down the tree (`.../room/Objects/Procedural/Table0`) and are
    passed to the same tools; the nearest `<root>/<scan>` ancestor still names the scan."""
    deep = stage_tree.physical_room / "Objects" / "Procedural" / "Table0"
    deep.mkdir(parents=True)
    assert _scan_from_room(deep) == stage_tree.scan


def test_env_is_the_fallback_outside_the_stage_tree(tmp_path, monkeypatch):
    """A room outside any stage tree (a scratch copy, a test) must not kill the tool: the CLI
    exports `$LITEREALITY_SCAN` for every stage, so use it rather than raising."""
    room = tmp_path / "loose" / "room"
    room.mkdir(parents=True)
    monkeypatch.setenv("LITEREALITY_SCAN", "some-scan")
    assert _scan_from_room(room) == "some-scan"


def test_path_wins_over_env(stage_tree, monkeypatch):
    """The room dir is explicit input; the env var is ambient and gets overwritten by
    `_config_for`. A stale env value must never rename the room the caller asked about."""
    monkeypatch.setenv("LITEREALITY_SCAN", "a-different-scan")
    assert _scan_from_room(stage_tree.physical_room) == stage_tree.scan


def test_raises_only_when_genuinely_unknowable(tmp_path):
    room = tmp_path / "nowhere" / "room"
    room.mkdir(parents=True)
    with pytest.raises(ValueError, match="could not infer scan"):
        _scan_from_room(room)


def test_every_caller_uses_this_helper():
    """Guard against a tool growing its own copy of the inference: one choke point is why a
    one-line fix restored render, select_views and survey together.
    """
    root = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "lrauthor"
        / "scene"
        / "rendering"
    )
    offenders = [
        str(p.relative_to(root.parent))
        for p in root.rglob("*.py")
        if p.name != "compose.py" and 'name == "output"' in p.read_text(encoding="utf-8")
    ]
    assert not offenders, f"inline scan sniffing found in {offenders} — call _scan_from_room instead"
