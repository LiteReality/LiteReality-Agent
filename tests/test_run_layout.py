"""The Python composition root owns the canonical run layout."""

from pathlib import Path

import pytest

from lrauthor.pipeline.context import RunContext
from lrauthor.settings import LiteRealitySettings


def context(tmp_path: Path) -> RunContext:
    capture = tmp_path / "captures" / "Office"
    capture.mkdir(parents=True)
    (capture / "room.usdz").touch()
    settings = LiteRealitySettings(
        repo_root=tmp_path,
        scans_dir=capture.parent,
        output_root=tmp_path / "run",
    )
    return RunContext.resolve(capture, settings=settings)


def test_stage_one_and_authoring_are_siblings(tmp_path):
    ctx = context(tmp_path)
    assert ctx.init_root.parent == ctx.authoring_root.parent == ctx.scene_dir
    assert ctx.authoring_root not in ctx.init_root.parents
    assert ctx.init_root not in ctx.authoring_root.parents


def test_context_has_one_canonical_output_tree(tmp_path):
    ctx = context(tmp_path)
    assert ctx.scene_dir == tmp_path / "run" / "Office"
    assert ctx.object_root == ctx.scene_dir / "scene_init" / "obj_stage"
    assert ctx.seed_room == ctx.scene_dir / "scene_init" / "scene_stage" / "room_init" / "room"


def test_missing_capture_path_is_rejected_before_output_is_created(tmp_path):
    missing = tmp_path / "missing capture"
    output = tmp_path / "run"
    settings = LiteRealitySettings(
        repo_root=tmp_path,
        scans_dir=tmp_path / "captures",
        output_root=output,
    )

    with pytest.raises(ValueError, match="no such capture or scene path"):
        RunContext.resolve(missing, settings=settings)

    assert not missing.exists()
    assert not output.exists()


def test_missing_relative_capture_path_is_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = LiteRealitySettings(repo_root=tmp_path, output_root=tmp_path / "run")

    with pytest.raises(ValueError, match="no such capture or scene path"):
        RunContext.resolve("captures/missing", settings=settings)


def test_bare_scan_name_still_resolves_under_the_scan_root(tmp_path):
    settings = LiteRealitySettings(
        repo_root=tmp_path,
        scans_dir=tmp_path / "captures",
        output_root=tmp_path / "run",
    )

    ctx = RunContext.resolve("Office", settings=settings)

    assert ctx.scan == "Office"
    assert ctx.capture_dir == (tmp_path / "captures" / "Office").resolve()
    assert ctx.scene_dir == (tmp_path / "run" / "Office").resolve()
