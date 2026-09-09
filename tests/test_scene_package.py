"""The scene package — the manifest that makes an init'd folder launchable on its own.

These guard the two ways this fails silently rather than loudly:

  * **path shape.** A package lives in `run/<scan>/` while half the tree is normally
    addressed through the `run/<scan>/…` symlinks the CLI creates. Discovery and relativization
    have to work from either spelling, or a package built by the CLI is unreadable from a shell
    sitting in `output/`, and one built by a bare `uv run -m litereality_agent scene_init` records absolute paths and stops
    being movable.
  * **precedence.** A discovered package must never retarget an environment the caller exported
    on purpose — that would silently author the wrong scan.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from litereality_agent.room_ops import manifest

SCAN = "test-scan-Room"


@pytest.fixture
def package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A package in the shape the CLI produces: real tree under run/<scan>-<tag>, symlinked run/<scan>."""
    final = tmp_path / "deliverables" / "run" / SCAN
    out = tmp_path / "staging" / "run" / SCAN
    scans = tmp_path / "scans_uploaded" / SCAN

    room = final / "scene_init" / "scene_stage" / "room_init" / "room"
    room.mkdir(parents=True)
    (room / "Room.py").write_text("SHELL = {}\n", encoding="utf-8")
    (final / "scene_init" / "obj_stage" / "object_init").mkdir(parents=True)
    (final / "scene_init" / "obj_stage" / "reconstructed_objs" / "Table0").mkdir(parents=True)
    (final / "scene_init" / "obj_stage" / "reconstructed_objs" / "Sofa0.glb").write_bytes(b"")
    scans.mkdir(parents=True)
    (scans / "room.usdz").write_bytes(b"")
    (scans / "frame_00000.jpg").write_bytes(b"")
    (scans / "frame_00000.json").write_text("{}", encoding="utf-8")

    # Stage 1's own root has to exist on the staging side before a stage dir links into it.
    (out / "scene_init").mkdir(parents=True)
    (out / "scene_init" / "obj_stage").symlink_to(final / "scene_init" / "obj_stage")
    (out / "scene_init" / "scene_stage").symlink_to(
        Path("..") / ".." / ".." / ".." / "deliverables" / "run" / SCAN / "scene_init" / "scene_stage"
    )

    monkeypatch.setenv("LITEREALITY_FINAL", str(tmp_path / "deliverables" / "run"))
    monkeypatch.setenv("LITEREALITY_OUTPUT", str(tmp_path / "staging" / "run"))
    monkeypatch.delenv("LR_SCENE", raising=False)
    monkeypatch.delenv("LITEREALITY_SCAN", raising=False)

    capture = manifest.embed_capture(final, scans, SCAN, "link")
    manifest.write(
        final,
        scan=SCAN,
        paths={
            "obj_stage": final / "scene_init" / "obj_stage",
            "refroot": final / "scene_init" / "obj_stage" / "object_init",
            "reconstructed_objs": final / "scene_init" / "obj_stage" / "reconstructed_objs",
            "scene_stage": final / "scene_init" / "scene_stage",
            "room_init": room,
            "room_py": room / "Room.py",
            "surface_ref": final / "scene_init" / "scene_stage" / "_harness" / "surface_ref",
            "capture": capture,
        },
        roots={
            "output": tmp_path / "staging" / "run",
            "final": tmp_path / "deliverables" / "run",
            "scans": tmp_path / "scans_uploaded",
        },
    )
    return final


# --- what the manifest stores ------------------------------------------------ #
def test_in_package_paths_are_relative(package: Path):
    """Absolute paths would pin the package to the machine that wrote it."""
    data = json.loads((package / manifest.MANIFEST_NAME).read_text())
    for key, value in data["paths"].items():
        assert not Path(value).is_absolute(), f"{key} was stored absolute: {value}"


def test_package_survives_being_moved(package: Path, tmp_path: Path):
    """The whole point of relative paths: copy the folder elsewhere and it still resolves."""
    moved = tmp_path / "elsewhere" / SCAN
    moved.parent.mkdir()
    package.rename(moved)

    pkg = manifest.read(moved)
    assert pkg.scan == SCAN
    assert pkg.room_py == moved / "scene_init" / "scene_stage" / "room_init" / "room" / "Room.py"
    assert pkg.room_py.is_file()


def test_capture_is_embedded_under_its_scan_name(package: Path):
    """`$LR_SCANS_DIR/<scan>` is how every consumer spells the capture, so the embedded copy has
    to sit at `capture/<scan>/` for the scans root to simply point at `capture/`."""
    pkg = manifest.read(package)
    assert pkg.capture == package / "capture" / SCAN
    assert (pkg.capture / "room.usdz").exists()
    assert pkg.env()["LR_SCANS_DIR"] == str(package / "capture")


# --- discovery --------------------------------------------------------------- #
def test_discovers_from_inside_the_package(package: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(package / "scene_init" / "scene_stage" / "room_init" / "room")
    pkg = manifest.discover()
    assert pkg is not None and pkg.root == package


def test_discovers_through_the_output_symlink(package: Path, monkeypatch: pytest.MonkeyPatch):
    """A shell inside `run/<scan>/scene_init/scene_stage/…` walks up to `run/<scan>/`, which holds no
    manifest — only following the symlink back into the deliverables root finds one. This is the exact
    path-shape trap the tools README warns about."""
    out_room = package.parents[1] / "run" / SCAN / "scene_init" / "scene_stage" / "room_init" / "room"
    assert out_room.is_dir(), "fixture symlink is wrong"
    monkeypatch.chdir(out_room)
    pkg = manifest.discover()
    assert pkg is not None and pkg.root.name == SCAN


def test_discovers_by_scan_name_under_the_roots(package: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(package.parents[2])
    monkeypatch.setenv("LITEREALITY_SCAN", SCAN)
    pkg = manifest.discover()
    assert pkg is not None and pkg.root == package


def test_require_lists_candidates_when_nothing_resolves(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LITEREALITY_FINAL", str(tmp_path / "nowhere"))
    monkeypatch.setenv("LITEREALITY_OUTPUT", str(tmp_path / "nowhere2"))
    monkeypatch.delenv("LR_SCENE", raising=False)
    monkeypatch.delenv("LITEREALITY_SCAN", raising=False)
    with pytest.raises(SystemExit) as exc:
        manifest.require()
    assert "--scene" in str(exc.value)


# --- precedence -------------------------------------------------------------- #
def test_explicit_scene_overrides_the_environment(package: Path, monkeypatch):
    monkeypatch.setenv("LITEREALITY_SCAN", "some-other-scan")
    pkg = manifest.bootstrap(["--scene", str(package)])
    assert pkg is not None
    assert os.environ["LITEREALITY_SCAN"] == SCAN


def test_discovered_package_never_overrides_the_environment(package: Path, monkeypatch):
    """the CLI exports the scan itself. A package that merely happens to be findable must fill
    gaps only — retargeting here would author a different room than the caller asked for."""
    monkeypatch.chdir(package)
    monkeypatch.setenv("LITEREALITY_SCAN", "some-other-scan")
    manifest.bootstrap([])
    assert os.environ["LITEREALITY_SCAN"] == "some-other-scan"
    assert os.environ["LR_SCANS_DIR"] == str(package / "capture")  # the gap IS filled


def test_bootstrap_does_not_bind_to_an_unnamed_package(package: Path, tmp_path, monkeypatch):
    """`discover(deep=True)` will latch onto the only package on disk — handy at a prompt, wrong
    for a library. The passive bootstrap must stop before that rule."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LITEREALITY_SCAN", raising=False)
    assert manifest.bootstrap([]) is None
    assert manifest.discover() is not None  # …but an explicit lookup still finds it


# --- the two spellings of a stage dir ---------------------------------------- #
def test_stage_links_exist_before_anything_writes(tmp_path: Path, monkeypatch):
    """The object half WRITES `run/<scan>/scene_init/obj_stage`; the room export READS the same
    path through the other root. the CLI links them up front, so a run driven from the CLI has to as
    well — otherwise init reconstructs every object and the export then finds an empty tree and
    returns None, throwing the whole stage away at the last step."""
    from litereality_agent.pipeline.measure import paths as oi_config
    from litereality_agent.pipeline.measure.artifacts import ensure_stage_links
    from litereality_agent.room_ops import paths as sb_config

    monkeypatch.setenv("LITEREALITY_FINAL", str(tmp_path / "run"))
    monkeypatch.setenv("LITEREALITY_OUTPUT", str(tmp_path / "run"))
    ensure_stage_links(SCAN)

    # what the object stage writes …
    written = oi_config.obj_stage_dir(SCAN) / "reconstructed_objs"
    written.mkdir(parents=True)
    (written / "Table0.glb").write_bytes(b"")

    # … is what the room export reads, through the other spelling
    read = sb_config.reconstruct_dir(SCAN)
    assert read.is_dir(), f"{read} does not exist — the export would return None"
    assert (read / "Table0.glb").is_file()
    assert read.resolve() == written.resolve()


def test_stage_links_adopt_an_existing_staging_tree(tmp_path: Path, monkeypatch):
    """A tree from before the links existed has REAL directories under the staging root. Those must be
    adopted, never shadowed by a fresh empty one — that would strand the previous run's objects."""
    from litereality_agent.pipeline.measure import paths as oi_config
    from litereality_agent.pipeline.measure.artifacts import ensure_stage_links

    monkeypatch.setenv("LITEREALITY_FINAL", str(tmp_path / "deliverables" / "run"))
    monkeypatch.setenv("LITEREALITY_OUTPUT", str(tmp_path / "staging" / "run"))
    legacy = tmp_path / "staging" / "run" / SCAN / "scene_init" / "obj_stage" / "reconstructed_objs"
    legacy.mkdir(parents=True)
    (legacy / "Sofa0.glb").write_bytes(b"")

    ensure_stage_links(SCAN)
    adopted = oi_config.obj_stage_dir(SCAN) / "reconstructed_objs" / "Sofa0.glb"
    assert adopted.is_file(), "the pre-existing staging tree was shadowed instead of adopted"


# --- what stage 2 actually gets ---------------------------------------------- #
def _stage_parser():
    """The argument surface every stage-2 entry point now shares."""
    import argparse

    from litereality_agent.pipeline.author import arguments as stage_args

    ap = argparse.ArgumentParser()
    stage_args.add_scene_arg(ap)
    for flag in ("room", "surface-ref", "scan", "refroot", "results"):
        ap.add_argument(f"--{flag}", default=None)
    return ap


@pytest.mark.parametrize(
    "stage,need",
    [
        ("author", ("room", "surface_ref", "scan")),
        ("materials_pass", ("room", "surface_ref", "scan", "refroot")),
        ("qc_pass", ("room", "surface_ref", "scan", "refroot")),
        ("refine_objects", ("room", "refroot", "scan")),
    ],
)
def test_every_stage_resolves_from_the_scene_alone(package: Path, stage, need):
    """The promise of the package: `--scene <dir>` replaces the per-stage path flags."""
    from litereality_agent.pipeline.author import arguments as stage_args

    a = stage_args.bind(_stage_parser().parse_args(["--scene", str(package)]), need=need)
    for name in need:
        assert getattr(a, name), f"{stage} left --{name} unresolved"
    assert Path(a.room).name == "room" and "_oneshot" in a.room


def test_explicit_flags_beat_the_package(package: Path):
    from litereality_agent.pipeline.author import arguments as stage_args

    argv = ["--scene", str(package), "--room", "/explicit/room"]
    a = stage_args.bind(_stage_parser().parse_args(argv), need=("room", "scan"))
    assert a.room == "/explicit/room"
    assert a.scan.startswith(str(package))  # the rest still comes from the package


def test_full_flag_invocation_needs_no_package(monkeypatch, tmp_path):
    """the CLI passes every flag. That must keep working with no package anywhere on disk."""
    from litereality_agent.pipeline.author import arguments as stage_args

    monkeypatch.setenv("LITEREALITY_FINAL", str(tmp_path / "nowhere"))
    monkeypatch.setenv("LITEREALITY_OUTPUT", str(tmp_path / "nowhere2"))
    monkeypatch.chdir(tmp_path)
    argv = ["--room", "/r", "--surface-ref", "/s", "--scan", "/c", "--refroot", "/f"]
    a = stage_args.bind(_stage_parser().parse_args(argv), need=("room", "surface_ref", "scan"))
    assert (a.room, a.scan) == ("/r", "/c")


def test_work_room_is_seeded_once_and_never_clobbered(package: Path):
    """Authoring edits `_oneshot/room`, not the seed — and re-running must not throw away the
    edits by re-copying over them, which is how a resumed run would silently lose a pass."""
    from litereality_agent.pipeline.author import arguments as stage_args

    pkg = manifest.read(package)
    room = stage_args.work_room(pkg)
    assert (room / "Room.py").is_file()

    (room / "Room.py").write_text("EDITED\n", encoding="utf-8")
    assert (stage_args.work_room(pkg) / "Room.py").read_text() == "EDITED\n"
    assert (pkg.room_init / "Room.py").read_text() == "SHELL = {}\n"  # seed untouched


# --- health ------------------------------------------------------------------ #
def test_check_reports_missing_required_paths(package: Path):
    pkg = manifest.read(package)
    required, _ = pkg.check()
    assert required == []

    (package / "scene_init" / "scene_stage" / "room_init" / "room" / "Room.py").unlink()
    required, _ = manifest.read(package).check()
    assert any("room_py" in item for item in required)
