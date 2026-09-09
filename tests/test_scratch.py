"""Working images the model MAKES must survive the run.

An authoring session manufactures its own evidence — it crops a wall band, white-balances a
frame, tiles two views — then reads the result back and decides something from it. Those images
were going to `/tmp` (`f02.png`, `f02_dado.png`, `f02_wb.png` on a real run) and being lost, so
the record showed a decision with no way to see what it was made from.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lrauthor.agent import scratch


@pytest.fixture
def bound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A package-bound run: `$LR_SCRATCH` under a realism_authoring tree."""
    authoring = tmp_path / "run" / "Office_room" / "realism_authoring"
    monkeypatch.setenv("LR_AUTHORING", str(authoring))
    monkeypatch.setenv("LR_SCRATCH", str(authoring / "_scratch"))
    return authoring


def test_scratch_dir_lands_inside_the_scene_package(bound: Path):
    where = scratch.scratch_dir()
    assert where == bound / "_scratch"
    assert where.is_dir(), "must exist by the time the prompt names it"


def test_each_run_gets_its_own_numbered_directory(bound: Path):
    """One shared `_scratch` meant each run overwrote the last — the model reuses names like
    `f02.png` every session, so yesterday's evidence vanished and the images stopped matching
    the trace beside them."""
    first = scratch.bind()
    second = scratch.bind()
    third = scratch.bind()
    assert [p.name for p in (first, second, third)] == ["run_001", "run_002", "run_003"]
    assert all(p.parent == bound / "_scratch" for p in (first, second, third))
    assert all(p.is_dir() for p in (first, second, third))


def test_binding_twice_does_not_nest_run_dirs(bound: Path):
    """`$LR_SCRATCH` already points at a run dir on the second call; the base must be recovered
    from it rather than treated as the new base (`run_001/run_002`)."""
    scratch.bind()
    second = scratch.bind()
    assert second == bound / "_scratch" / "run_002"
    assert "run_001" not in str(second)


def test_latest_points_at_the_newest_run(bound: Path):
    scratch.bind()
    newest = scratch.bind()
    link = bound / "_scratch" / "latest"
    assert link.is_symlink() and link.resolve() == newest.resolve()


def test_earlier_runs_survive_a_later_one(bound: Path):
    """The whole point: an image from run_001 is still there after run_002 writes the same name."""
    first = scratch.bind()
    (first / "f02.png").write_bytes(b"first-run")
    second = scratch.bind()
    (second / "f02.png").write_bytes(b"second-run")
    assert (first / "f02.png").read_bytes() == b"first-run"
    assert (second / "f02.png").read_bytes() == b"second-run"


def test_rescue_lands_in_this_runs_directory(bound: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LR_SCENE", str(bound.parent))
    run = scratch.bind()
    made = tmp_path / "outside" / "crop.png"
    made.parent.mkdir(parents=True)
    made.write_bytes(b"\x89PNG\r\n\x1a\n")
    kept = scratch.rescue(str(made))
    assert kept and kept[0].parent == run / "rescued"


def test_falls_back_to_authoring_for_packages_written_before_the_key(bound: Path, monkeypatch):
    monkeypatch.delenv("LR_SCRATCH")
    assert scratch.scratch_dir() == bound / "_scratch"


def test_no_package_bound_is_not_an_error(monkeypatch: pytest.MonkeyPatch):
    """Stages run standalone in tests and one-offs; absence of a package must not raise."""
    monkeypatch.delenv("LR_SCRATCH", raising=False)
    monkeypatch.delenv("LR_AUTHORING", raising=False)
    assert scratch.scratch_dir() is None
    assert scratch.prompt_line() == ""
    assert scratch.rescue({"command": "python3 -c '...' /tmp/x.png"}) == []


def test_the_tmp_image_that_started_this_is_rescued(bound: Path, tmp_path: Path):
    """The exact shape of the loss: Bash writes a crop outside the run, then Reads it back."""
    made = tmp_path / "elsewhere" / "f02_dado.png"
    made.parent.mkdir(parents=True)
    made.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)

    kept = scratch.rescue({"command": f"python3 -c 'crop' && echo {made}"}, "wrote " + str(made))
    assert len(kept) == 1
    assert kept[0].parent == bound / "_scratch" / "rescued"
    assert kept[0].read_bytes() == made.read_bytes()


def test_images_already_inside_the_run_are_not_copied(bound: Path):
    """Renders already live in the package; copying them would double the tree for nothing."""
    render = bound / "_image_render_tool" / "scene" / "scene_frame_00020.png"
    render.parent.mkdir(parents=True)
    render.write_bytes(b"x")
    assert scratch.rescue(f"rendered {render}") == []


def test_capture_frames_are_not_rescued(bound: Path, monkeypatch: pytest.MonkeyPatch):
    """The boundary is the SCENE PACKAGE, not the authoring subtree. Using the latter copied
    every capture frame the model opened — `<scene>/capture/` is already durable, and a run
    that reads 30 frames would duplicate all of them into scratch."""
    scene = bound.parent
    monkeypatch.setenv("LR_SCENE", str(scene))
    frame = scene / "capture" / "Office_room" / "frame_00000.jpg"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"\xff\xd8\xff" + b"0" * 32)
    assert scratch.rescue({"file_path": str(frame)}) == []


def test_an_image_outside_the_package_is_still_rescued(bound: Path, tmp_path: Path, monkeypatch):
    """The widened boundary must not swallow the case the net exists for."""
    monkeypatch.setenv("LR_SCENE", str(bound.parent))
    outside = tmp_path / "somewhere-else" / "crop.png"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
    assert len(scratch.rescue(str(outside))) == 1


def test_repeated_names_are_kept_as_separate_versions(bound: Path, tmp_path: Path):
    """A session rewrites `f02.png` for each crop. Which one it was looking at is the point, so
    the second must not silently replace the first."""
    src = tmp_path / "out" / "f02.png"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"first")
    first = scratch.rescue(str(src))
    src.write_bytes(b"second")
    second = scratch.rescue(str(src))

    assert first and second and first[0] != second[0]
    assert first[0].read_bytes() == b"first"
    assert second[0].read_bytes() == b"second"


def test_only_images_are_copied(bound: Path, tmp_path: Path):
    """`rescue` copies files — a loose pattern would start hoovering up meshes and logs."""
    for name in ("room.glb", "render.log", "Room.py"):
        (tmp_path / name).write_bytes(b"x")
    text = " ".join(str(tmp_path / n) for n in ("room.glb", "render.log", "Room.py"))
    assert scratch.rescue(text) == []


def test_oversize_files_are_skipped(bound: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(scratch, "_MAX_BYTES", 10)
    big = tmp_path / "big.png"
    big.write_bytes(b"0" * 64)
    assert scratch.rescue(str(big)) == []


def test_a_missing_path_is_ignored_not_raised(bound: Path):
    assert scratch.rescue("/tmp/never-existed-abc123.png") == []


def test_prompt_line_names_the_directory_and_forbids_tmp(bound: Path):
    line = scratch.prompt_line()
    assert str(bound / "_scratch") in line
    assert "/tmp" in line, "the habit being corrected must be named explicitly"


def test_relative_image_names_resolve_against_the_stage_roots(tmp_path, monkeypatch):
    """`_save_image` checked `Path('conf_00000.png').is_file()` against the process CWD, so every
    relative name failed: 1 of 7 image events in a real trace saved anything."""
    from lrauthor.agent.trace import AgentTrace

    surface_ref = tmp_path / "surface_ref"
    surface_ref.mkdir()
    (surface_ref / "Wall0_stitched.jpg").write_bytes(b"x")
    monkeypatch.setenv("LR_SURFACE_REF", str(surface_ref))
    monkeypatch.chdir(tmp_path / "..")

    tr = AgentTrace("author", room=tmp_path / "room", scan="test-scan-Room")
    assert tr._locate("Wall0_stitched.jpg") == surface_ref / "Wall0_stitched.jpg"
    assert tr._locate("no_such_image.png") is None


def test_an_empty_run_dir_is_pruned(bound: Path):
    """A short run that only reads stitches makes no images; one empty run_NNN per attempt is
    the clutter per-run directories exist to avoid."""
    run = scratch.bind()
    assert scratch.prune_if_empty(run) is True
    assert not run.exists()
    assert scratch.bind().name == "run_001", "the number is reused"


def test_a_run_dir_with_images_is_kept(bound: Path):
    run = scratch.bind()
    (run / "crop.png").write_bytes(b"x")
    assert scratch.prune_if_empty(run) is False
    assert run.is_dir()


def test_pruning_repoints_latest_instead_of_leaving_it_dangling(bound: Path):
    first = scratch.bind()
    (first / "keep.png").write_bytes(b"x")
    second = scratch.bind()
    scratch.prune_if_empty(second)
    link = bound / "_scratch" / "latest"
    assert link.resolve() == first.resolve()


def test_files_left_at_the_scratch_root_are_claimed_by_the_run(bound: Path):
    """The prompt names the run dir exactly and the model still writes to the parent — observed
    with `Wall0_cmp.jpg` and a `jview.py` helper. Without sweeping, the next run's files sit
    beside them with no way to tell the two apart."""
    run = scratch.bind()
    root = bound / "_scratch"
    (root / "Wall0_cmp.jpg").write_bytes(b"cmp")
    (root / "jview.py").write_text("helper", encoding="utf-8")

    moved = scratch.collect_strays(run)
    assert {p.name for p in moved} == {"Wall0_cmp.jpg", "jview.py"}
    assert (run / "Wall0_cmp.jpg").read_bytes() == b"cmp"
    assert not (root / "Wall0_cmp.jpg").exists()


def test_sweeping_never_touches_other_run_dirs_or_latest(bound: Path):
    first = scratch.bind()
    (first / "keep.png").write_bytes(b"x")
    second = scratch.bind()
    scratch.collect_strays(second)
    assert (first / "keep.png").is_file(), "an earlier run must not be absorbed"
    assert (bound / "_scratch" / "latest").is_symlink()


def test_finish_claims_strays_before_deciding_the_dir_is_empty(bound: Path):
    """Order matters: pruning first would delete the directory the strays belong in."""
    run = scratch.bind()
    (bound / "_scratch" / "stray.png").write_bytes(b"x")
    scratch.finish(run)
    assert run.is_dir() and (run / "stray.png").is_file()


def test_finish_prunes_when_there_was_genuinely_nothing(bound: Path):
    run = scratch.bind()
    scratch.finish(run)
    assert not run.exists()
