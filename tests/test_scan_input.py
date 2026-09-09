"""A scan can be named or handed over as a folder — and the two must mean the same thing.

Everything downstream resolves a capture as `$LR_SCANS_DIR/<name>`, so a folder is translated
once, at the CLI edge, into exactly that pair. These guard the translation, because getting it
wrong does not crash: it points the whole pipeline at a directory that isn't there and the first
symptom is an empty stitch or a render with no photos to compare against.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from lrauthor.cli import looks_like_scan_dir, resolve_scan


@pytest.fixture
def capture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A minimal RoomPlan capture: the usdz + one frame pair."""
    scan = tmp_path / "captures" / "Office_room"
    scan.mkdir(parents=True)
    (scan / "room.usdz").write_bytes(b"")
    (scan / "frame_00000.jpg").write_bytes(b"")
    (scan / "frame_00000.json").write_text("{}", encoding="utf-8")
    monkeypatch.delenv("LR_SCANS_DIR", raising=False)
    return scan


def test_bare_name_is_left_alone(capture: Path):
    """A name is a name: it must not be probed against the CWD, or a run's meaning would depend
    on where you happened to be standing."""
    assert resolve_scan("Office_room") == "Office_room"
    assert "LR_SCANS_DIR" not in os.environ


@pytest.mark.parametrize("spelling", ["absolute", "relative", "trailing-slash", "dot-relative"])
def test_folder_becomes_name_plus_scans_root(capture: Path, monkeypatch, spelling: str):
    monkeypatch.chdir(capture.parent.parent)
    value = {
        "absolute": str(capture),
        "relative": "captures/Office_room",
        "trailing-slash": "captures/Office_room/",
        "dot-relative": "./captures/Office_room",
    }[spelling]

    assert resolve_scan(value) == "Office_room"
    # the pair the rest of the pipeline consumes must reconstitute the folder exactly
    assert Path(os.environ["LR_SCANS_DIR"]) / "Office_room" == capture.resolve()


def test_missing_folder_fails_loudly(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="no such scan folder"):
        resolve_scan("captures/nope")


def test_a_folder_that_is_not_a_capture_is_refused(tmp_path: Path, monkeypatch):
    """Pointing at the wrong directory is the likely mistake, and it has to fail HERE — a bad
    $LR_SCANS_DIR otherwise surfaces stages later as missing frames, not as a bad argument."""
    (tmp_path / "not_a_scan").mkdir()
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="does not look like a RoomPlan capture"):
        resolve_scan("./not_a_scan")


def test_a_bare_name_is_never_probed_against_the_cwd(tmp_path: Path, monkeypatch):
    """`not_a_scan` names a scan under $LR_SCANS_DIR even when a directory of that name happens
    to sit in the CWD — otherwise the same command means different things in different shells."""
    (tmp_path / "not_a_scan").mkdir()
    monkeypatch.chdir(tmp_path)
    assert resolve_scan("not_a_scan") == "not_a_scan"
    assert "LR_SCANS_DIR" not in os.environ


def test_scan_dir_detection(capture: Path, tmp_path: Path):
    assert looks_like_scan_dir(capture)
    assert not looks_like_scan_dir(tmp_path)
    assert not looks_like_scan_dir(capture / "room.usdz")  # a file is not a capture

    frames_only = tmp_path / "frames_only"
    frames_only.mkdir()
    (frames_only / "frame_00007.jpg").write_bytes(b"")
    assert looks_like_scan_dir(frames_only), "a frame-only capture (no usdz yet) still counts"


def test_resolution_is_idempotent(capture: Path, monkeypatch):
    """`refine <folder>` re-resolves what `init <folder>` already resolved; the second pass must
    not nest the scans root one level deeper."""
    name = resolve_scan(str(capture))
    root = os.environ["LR_SCANS_DIR"]
    assert resolve_scan(name) == name
    assert os.environ["LR_SCANS_DIR"] == root


# --- the capture must be findable from wherever it was given -------------------- #
def test_resolve_scan_dir_honours_the_scans_root(tmp_path, monkeypatch):
    """It used to hardcode `<repo>/scans_uploaded`, so a capture given as a FOLDER was never
    found. That does not fail loudly: the room assembles without the capture's ARKit cameras and
    comes out "geometry only" — a quietly worse build, from a command that reported success."""
    from lrauthor.room_ops import paths as config

    capture = tmp_path / "example_scans" / "Office_room"
    capture.mkdir(parents=True)
    (capture / "room.usdz").write_bytes(b"")
    (capture / "frame_00000.jpg").write_bytes(b"")

    monkeypatch.setenv("LR_SCANS_DIR", str(tmp_path / "example_scans"))
    assert config.resolve_scan_dir("Office_room") == capture


def test_the_staged_usdz_copy_is_recognised(tmp_path, monkeypatch):
    """object_init stages the capture as `<scan>.usdz`, not `room.usdz` — the old probe only
    looked for the latter, so the fallback location never matched either."""
    from lrauthor.room_ops import paths as config

    monkeypatch.setenv("LR_SCANS_DIR", str(tmp_path / "nowhere"))
    monkeypatch.setenv("LITEREALITY_OUTPUT", str(tmp_path / "out"))
    monkeypatch.setenv("LITEREALITY_FINAL", str(tmp_path / "out"))
    staged = config.object_init_dir("Office_room") / "input" / "usdz_files"
    staged.mkdir(parents=True)
    (staged / "Office_room.usdz").write_bytes(b"")
    assert config.resolve_scan_dir("Office_room") == staged


def test_a_missing_capture_returns_the_expected_location(tmp_path, monkeypatch):
    """Not found should point at where it was looked for, so the error names a fixable path."""
    from lrauthor.room_ops import paths as config

    monkeypatch.setenv("LR_SCANS_DIR", str(tmp_path / "captures"))
    monkeypatch.setenv("LITEREALITY_OUTPUT", str(tmp_path / "out"))
    monkeypatch.setenv("LITEREALITY_FINAL", str(tmp_path / "out"))
    assert config.resolve_scan_dir("Nope") == tmp_path / "captures" / "Nope"
