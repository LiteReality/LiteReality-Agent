"""Authoring stages must be handed the scene package, and say so plainly when they are not.

`author` and `publish` read the tree `scene_init` produced under the output root. A capture folder
resolves to a valid context too — same scan name, same default output root — so `stage author
scans/<name>` looks like it works. It stops working the moment `--output-root` differs from the
default: the stage then authors one tree while the viewer watches another, and nothing reports it.

So the package path is required, and the two ways of getting it wrong get different answers — one
is "you named the capture", the other is "scene init has not run yet".
"""

from __future__ import annotations

import argparse
import json

import pytest

from lrauthor import cli


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A capture at `scans/Studio`, and an output root that may or may not hold its package."""
    monkeypatch.setenv("LR_COLOR", "1")  # the message is meant to be red; prove it can be
    capture = tmp_path / "scans" / "Studio"
    capture.mkdir(parents=True)
    (capture / "room.usdz").write_bytes(b"")
    (tmp_path / "run").mkdir()
    return tmp_path


def _args(tmp_path, target: str, stage: str = "author", **extra):
    return argparse.Namespace(
        target=str(tmp_path / target), stage=stage, output_root=str(tmp_path / "run"),
        from_stage=None, **extra,
    )


def _seed_package(tmp_path, body: dict | None = None):
    """The marker `scene_init` leaves behind — what makes a directory a scene package."""
    package = tmp_path / "run" / "Studio"
    package.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "litereality.scene/1", "scan": "Studio",
        "paths": {"capture": "capture/Studio"},
        "roots": {"output": str(tmp_path / "run"), "scans": str(tmp_path / "scans")},
        "capture": {"mode": "link", "source": str(tmp_path / "scans" / "Studio")},
    }
    (package / "scene.json").write_text(json.dumps(body if body is not None else manifest),
                                        encoding="utf-8")
    return package


def test_a_half_written_package_does_not_die_on_a_null_capture(tree):
    """A scene.json readable but naming nothing — an interrupted write — used to raise a bare
    `TypeError: argument should be a str …, not 'NoneType'` from `Path(None)`, mentioning neither
    the file nor the scan."""
    _seed_package(tree, body={})
    args = _args(tree, "run/Studio")

    context = cli._context(args)  # must resolve rather than explode

    assert context.scan == "Studio", "the directory name stands in for a manifest that has no scan"
    assert context.capture_dir.name == "Studio"


def test_the_package_path_is_accepted(tree):
    _seed_package(tree)
    args = _args(tree, "run/Studio")

    cli._require_scene_package(args.target, cli._context(args), "author")  # must not raise


def test_naming_the_capture_is_refused_and_names_the_package(tree, capsys):
    """The failure that looks like success until an --output-root differs."""
    _seed_package(tree)
    args = _args(tree, "scans/Studio")

    with pytest.raises(SystemExit) as exit_info:
        cli._require_scene_package(args.target, cli._context(args), "author")

    err = capsys.readouterr().err
    assert exit_info.value.code == 2
    assert "runs on the scene package, not the capture" in err
    assert "\033[1;31m" in err, "the refusal is meant to be red"
    assert "stage author" in err and "Studio" in err, "it must show the command that works"
    assert "--through seed" not in err, "the package exists; seeding is not the advice"


def test_a_missing_package_says_scene_init_has_not_run(tree, capsys):
    """Better here than three steps later as `missing seed room at …`."""
    args = _args(tree, "scans/Studio")

    with pytest.raises(SystemExit):
        cli._require_scene_package(args.target, cli._context(args), "author")

    err = capsys.readouterr().err
    assert "scene.json does not exist" in err
    assert "--through seed" in err, "the fix is to produce the package first"


def test_publish_is_held_to_the_same_rule(tree, capsys):
    args = _args(tree, "scans/Studio", stage="publish")

    with pytest.raises(SystemExit):
        cli._require_scene_package(args.target, cli._context(args), "publish")

    assert "publish runs on the scene package" in capsys.readouterr().err


def test_a_full_run_may_still_start_from_a_capture(tree, monkeypatch):
    """`run scans/<name>` produces the package on the way through — requiring one would be absurd."""
    called = []
    monkeypatch.setattr(cli, "_require_scene_package", lambda *a: called.append(a))
    monkeypatch.setattr(cli.PipelineRunner, "run", lambda *a, **k: [])

    args = argparse.Namespace(
        target=str(tree / "scans" / "Studio"), output_root=str(tree / "run"),
        from_stage=None, through="seed", force=None, strict=False, live=False,
        polish=False, refine_objects=False, materials=False, quality_pass=False,
        compare_frames=None,
    )
    cli._run(args)

    assert called == [], "a run that starts at ingest must not demand a package"


def test_a_run_that_starts_at_authoring_is_checked(tree, monkeypatch):
    checked = []
    monkeypatch.setattr(cli, "_require_scene_package", lambda *a: checked.append(a[2]))
    monkeypatch.setattr(cli.PipelineRunner, "run", lambda *a, **k: [])

    args = argparse.Namespace(
        target=str(tree / "scans" / "Studio"), output_root=str(tree / "run"),
        from_stage="author", through=None, force=None, strict=False, live=False,
        polish=False, refine_objects=False, materials=False, quality_pass=False,
        compare_frames=None,
    )
    cli._run(args)

    assert checked == ["author"]
