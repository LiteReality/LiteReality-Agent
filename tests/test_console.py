"""The stage display: readable, never load-bearing.

This renders progress, so its failure mode is subtle — a display that raises takes the pipeline
down with it, and a display that lies is worse than none. Two invariants: it never throws
whatever it is handed, and it never emits escape codes into something that is not a terminal
(a log file full of `\\033[1;32m` is unreadable, and CI diffs become noise).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from lrauthor import console


@pytest.fixture
def tty(monkeypatch):
    # capsys swaps sys.stdout for its own object, so patching isatty on the real one does not
    # survive into the test — $LR_COLOR is the supported way to force the decision.
    monkeypatch.setenv("LR_COLOR", "1")
    monkeypatch.delenv("LR_QUIET", raising=False)
    console._started.clear()
    console._announced.clear()


@pytest.fixture
def piped(monkeypatch):
    monkeypatch.setenv("LR_COLOR", "0")
    monkeypatch.delenv("LR_QUIET", raising=False)
    console._started.clear()
    console._announced.clear()


def test_no_escape_codes_when_not_a_terminal(piped, capsys):
    console.stage_event("crop_objects", "scan", "start", {})
    console.stage_event("crop_objects", "scan", "done", {"n_objects": 9})
    print(console.rule("title"))
    print(console.row("done", "label", "value", "cyan"))
    out = capsys.readouterr().out
    assert "\033" not in out, "escape codes leaked into a non-tty stream"
    assert "9 objects" in out


def test_each_stage_keeps_its_own_colour(tty, capsys):
    """The point of the colours: two adjacent stages must not look alike."""
    for name in ("crop_objects", "object_references", "reconstruct"):
        console.stage_event(name, "scan", "start", {})
        console.stage_event(name, "scan", "done", {"n_objects": 1})
    out = capsys.readouterr().out
    hues = {console.STAGES[n][1] for n in ("crop_objects", "object_references", "reconstruct")}
    assert len(hues) == 3
    for hue in hues:
        assert console._C[hue] in out


def test_a_build_stage_that_produced_nothing_is_marked_failed(tty, capsys):
    """`0 GLBs` from a build stage is the exact shape of the procedural failure that shipped —
    it must not render as a tick."""
    console.stage_event("procedural", "scan", "start", {})
    console.stage_event("procedural", "scan", "done", {"n_glb": 0})
    out = capsys.readouterr().out
    assert "✗" in out and "✓" not in out


@pytest.mark.parametrize("name,data,expected", [
    ("box_merge", {"merged": 0}, "no overlaps to merge"),
    ("opening_references", {"n_openings": 0}, "no doors or windows"),
    ("bbox_polish", {"refined": 0}, "nothing to refine"),
    ("crop_objects", {"n_objects": 0}, "no objects found"),
])
def test_a_legitimate_zero_says_what_it_means(tty, capsys, name, data, expected):
    """Most stages have nothing to do some of the time — a room with no counter runs to merge,
    no doors, nothing to refine. Rendering those as ✗ trains you to ignore the ✗, and it is the
    build stages' ✗ that actually matters."""
    console.stage_event(name, "scan", "done", data)
    out = capsys.readouterr().out
    assert expected in out, out
    assert "✗" not in out and "0 " not in out


def test_an_unregistered_stage_treats_zero_as_benign(tty, capsys):
    """A new stage should not start its life reporting failures nobody wrote."""
    console.stage_event("some_new_stage", "scan", "done", {"n_objects": 0})
    out = capsys.readouterr().out
    assert "✗" not in out and "nothing to do" in out


def test_a_stage_that_produced_something_is_marked_ok(tty, capsys):
    console.stage_event("procedural", "scan", "start", {})
    console.stage_event("procedural", "scan", "done", {"n_glb": 2})
    assert "✓" in capsys.readouterr().out


def test_quiet_silences_everything(monkeypatch, capsys):
    monkeypatch.setenv("LR_QUIET", "1")
    monkeypatch.setenv("LR_COLOR", "1")
    console.stage_event("crop_objects", "scan", "start", {})
    console.stage_event("crop_objects", "scan", "done", {"n_objects": 9})
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("payload", [
    {"n_glb": None}, {"chairs": None, "clusters": None}, {"weird": object()},
    {}, {"n_objects": "not-a-number"},
])
def test_never_raises_on_a_payload_it_did_not_expect(tty, payload):
    """A stage adding a counter must never be able to break the run through the display."""
    console.stage_event("chair_clusters", "scan", "start", {})
    console.stage_event("chair_clusters", "scan", "done", payload)
    console.stage_event("a_stage_with_no_entry", "scan", "done", payload)


def test_unknown_stage_still_renders(tty, capsys):
    """A new stage should show up as itself rather than vanish until someone registers it."""
    console.stage_event("brand_new_stage", "scan", "done", {"n_objects": 3})
    out = capsys.readouterr().out
    assert "brand new stage" in out and "3 objects" in out


def test_summaries_read_as_english(tty, capsys):
    for name, data, expected in (
        ("classify", {"procedural": 2, "trellis": 2}, "2 procedural · 2 trellis"),
        ("chair_clusters", {"chairs": 5, "clusters": 2}, "5 chairs → 2 groups"),
        ("chair_qc", {"repaired": 1, "still_failing": 0}, "1 repaired"),
        ("reconstruct", {"n_glb": 1}, "1 GLB"),
        ("bbox_polish", {"refined": 41}, "41 boxes refined"),
    ):
        console.stage_event(name, "scan", "done", data)
        out = capsys.readouterr().out
        assert expected in out, f"{name}: {out!r}"
    # the zero that is noise, not news
    console.stage_event("chair_qc", "scan", "done", {"repaired": 1, "still_failing": 0})
    assert "still failing" not in capsys.readouterr().out


def test_short_keeps_paths_outside_the_checkout_absolute(tmp_path):
    """Relativising buys nothing without a shared prefix.

    An `--output-root` on another volume came out as `../../../../../private/var/folders/…` —
    longer than the absolute path it replaced, and it buries the part that identifies the run.
    """
    from lrauthor import REPO_ROOT

    inside = Path(REPO_ROOT) / "run" / "Scan" / "Room.py"
    assert console.short(inside) == os.path.join("run", "Scan", "Room.py")

    outside = tmp_path / "elsewhere" / "Room.py"
    assert console.short(outside) == str(outside)
    assert not console.short(outside).startswith("..")


def test_colour_override_wins_both_ways(monkeypatch, capsys):
    """`| less -R` wants colour; `> run.log` never does. Neither is a tty."""
    monkeypatch.setenv("LR_COLOR", "1")
    assert console.use_colour() is True
    monkeypatch.setenv("LR_COLOR", "0")
    assert console.use_colour() is False


# --- the reconstruction phase's bookkeeping ---------------------------------- #
def _run_module():
    """object_init.run pulls in cv2/open3d/trimesh. They are real dependencies present in the
    project venv; a stripped environment skips rather than pretending to have checked."""
    pytest.importorskip("cv2", reason="object_init.run needs the full preprocessing stack")
    pytest.importorskip("trimesh", reason="object_init.run needs the full preprocessing stack")
    import sys
    import types

    sys.modules.setdefault("open3d", types.ModuleType("open3d"))
    from lrauthor.pipeline.measure import flow as run

    return run


def test_expected_assets_covers_both_routes_and_openings(tmp_path, monkeypatch):
    """The progress denominator must count everything the run intends to build — a routed object
    from either route, plus every opening — or the bar finishes while work is still going."""
    import json

    run = _run_module()
    monkeypatch.setenv("LITEREALITY_FINAL", str(tmp_path / "final"))
    monkeypatch.setenv("LITEREALITY_OUTPUT", str(tmp_path / "out"))
    from lrauthor.pipeline.measure import paths as config

    config.set_scan("Sim")
    routing = config.work_root() / "routing"
    routing.mkdir(parents=True)
    (routing / "routing_manifest.json").write_text(json.dumps({"scans": {"Sim": [
        {"name": "Table0", "route": "procedural"}, {"name": "Sofa0", "route": "trellis"}]}}))
    for opening in ("Wall1_Door_0", "Wall5_Window_0"):
        (config.opening_refs_root() / "Sim" / opening).mkdir(parents=True)

    assert run._expected_assets("Sim") == ["Sofa0", "Table0", "Wall1_Door_0", "Wall5_Window_0"]


def test_built_count_recognises_both_glb_layouts(tmp_path, monkeypatch):
    """TRELLIS writes `<name>.glb` — one baked mesh IS the artifact. The articulated agent writes
    `<name>/` with a glb plus the editable `object.py`/`object.md`, and is only finished when all
    of it is there (see test_a_half_written_object_does_not_count_as_built)."""
    run = _run_module()
    monkeypatch.setenv("LITEREALITY_FINAL", str(tmp_path / "final"))
    monkeypatch.setenv("LITEREALITY_OUTPUT", str(tmp_path / "out"))
    from lrauthor.pipeline.measure import paths as config

    config.set_scan("Sim")
    recon = config.reconstruct_dir("Sim")
    recon.mkdir(parents=True)
    expected = ["Sofa0", "Table0", "Wall1_Door_0"]
    assert run._built_count("Sim", expected) == 0

    (recon / "Sofa0.glb").write_bytes(b"")                      # neural: complete
    (recon / "Table0").mkdir()
    (recon / "Table0" / "Table0.glb").write_bytes(b"")          # articulated: glb only …
    (recon / "Table0" / "object.py").write_text("")             # … plus its editable source
    (recon / "Table0" / "object.md").write_text("")
    (recon / "Wall1_Door_0").mkdir()                            # started, nothing emitted yet
    assert run._built_count("Sim", expected) == 2


# --- the parallel phase's display ------------------------------------------- #
def test_bar_names_the_branches_still_running(tty, capsys):
    """A bar sitting at 4/6 for ten minutes is indistinguishable from a hang unless it says who
    is still working — which is exactly what the long articulated-agent branch looks like once
    the fast branches have finished."""
    import time

    live = {"openings": time.time() - 587}
    bar = console.Progress(6, "reconstruction")
    bar.update(4, console._still_running(lambda: dict(live)))
    out = capsys.readouterr().out
    assert "4/6" in out and "openings" in out and "9m47s" in out


def test_branch_status_is_never_load_bearing(tty):
    """The status callback runs inside the render loop; a throwing one must not kill the phase."""
    def boom():
        raise RuntimeError("branch bookkeeping exploded")

    assert console._still_running(boom) == ""
    assert console._still_running(None) == ""


def test_elapsed_reads_as_minutes_past_a_minute():
    assert console.fmt_elapsed(45) == "45s"
    assert console.fmt_elapsed(587) == "9m47s"
    assert console.fmt_elapsed(0) == "0s"


def test_thread_tee_isolates_each_branch():
    """`redirect_stdout` swaps a process-global, so with three branches running it sends every
    thread's output to whichever log was bound last. Routing by thread is the whole point."""
    import io
    import threading

    terminal = io.StringIO()
    terminal.isatty = lambda: True
    tee = console.ThreadTee(terminal)
    logs: dict[str, io.StringIO] = {}

    def branch(name):
        logs[name] = io.StringIO()
        tee.bind(logs[name])
        tee.write(f"[{name}] chatter\n")
        assert tee.isatty() is False, "a thread writing to a log must report non-tty"
        tee.unbind()

    threads = [threading.Thread(target=branch, args=(n,)) for n in ("a", "b", "c")]
    for t in threads:
        t.start()
    tee.write("BAR\n")  # main thread keeps the terminal
    for t in threads:
        t.join()

    assert terminal.getvalue() == "BAR\n"
    assert all(f"[{n}] chatter" in logs[n].getvalue() for n in logs)
    assert not any("BAR" in f.getvalue() for f in logs.values())
    assert tee.isatty() is True  # main thread, unbound


# --- one Blender resolver ----------------------------------------------------- #
def test_only_one_module_resolves_blender():
    """There were seven of these and they disagreed: only the authoring copy knew that a stock
    macOS install hides the binary inside /Applications/Blender.app/Contents/MacOS/. With
    BLENDER_PATH blank, stage 2 found Blender and stage 1 did not — scene_init packed every
    object, then died reading the usdz and left a room directory with no Room.py."""
    import re
    from pathlib import Path as P

    pkg = P(__file__).resolve().parents[1] / "src" / "lrauthor"
    body = re.compile(r"def _?find_blender\(\)[^\n]*\n(?:[ \t]+[^\n]*\n|\n)*")
    implementations = []
    for f in pkg.rglob("*.py"):
        for m in body.finditer(f.read_text(encoding="utf-8", errors="replace")):
            # A delegate just calls the canonical one; a real implementation probes the system.
            if "_canonical()" not in m.group(0) and "shutil" in m.group(0) + f.read_text(errors="replace")[:0]:
                implementations.append(str(f.relative_to(pkg)))
            elif "_canonical()" not in m.group(0) and "os.environ" in m.group(0):
                implementations.append(str(f.relative_to(pkg)))
    assert implementations == ["room_ops/paths.py"], (
        f"Blender is resolved in more than one place: {implementations}"
    )


def test_the_resolver_finds_a_macos_app_bundle(monkeypatch, tmp_path):
    """The case that broke a real run: nothing in the environment, nothing on PATH, Blender
    installed as an app bundle."""
    from lrauthor.room_ops import paths as ic

    bundle = tmp_path / "Blender.app" / "Contents" / "MacOS"
    bundle.mkdir(parents=True)
    binary = bundle / "Blender"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)

    monkeypatch.delenv("BLENDER", raising=False)
    monkeypatch.setenv("LITEREALITY_BLENDER", str(bundle))  # the install DIR, as the README says
    assert ic.find_blender() == str(binary)


def test_the_error_says_what_to_set(monkeypatch, tmp_path):
    """`Blender not found` with no next step is what sent this investigation the long way round."""
    import pytest as _pytest

    from lrauthor.room_ops import paths as ic

    monkeypatch.delenv("BLENDER", raising=False)
    monkeypatch.delenv("LITEREALITY_BLENDER", raising=False)
    monkeypatch.setattr("shutil.which", lambda _: None)
    monkeypatch.setattr(ic.Path, "is_file", lambda self: False)
    with _pytest.raises(SystemExit) as exc:
        ic.find_blender()
    assert "BLENDER_PATH in .env" in str(exc.value) and "checked:" in str(exc.value)


# --- stage output capture ----------------------------------------------------- #
def test_a_nested_stage_row_still_reaches_the_terminal(tty, tmp_path, capsys):
    """Stages nest: the crop pass emits opening-reference events inside its own bracket. A nested
    row printed through the redirected `sys.stdout` lands in the OUTER stage's log instead of on
    screen — which reads exactly like the run stopping dead, and is how a live run appeared to
    hang at `▶ dino polish…` while it was in fact still working."""
    console.capture_stages(tmp_path)
    try:
        console.stage_event("crop_objects", "scan", "start", {})       # outer: capture opens
        console.stage_event("opening_references", "scan", "start", {})  # nested
        console.stage_event("opening_references", "scan", "done", {"n_openings": 2})
        console.stage_event("crop_objects", "scan", "done", {"n_objects": 9})
    finally:
        console.restore_capture()

    out = capsys.readouterr().out
    assert "opening refs" in out and "2 openings" in out, "nested row never reached the terminal"
    assert "crops" in out and "9 objects" in out
    leaked = [f.name for f in tmp_path.glob("*.log") if "openings" in f.read_text()]
    assert not leaked, f"stage rows leaked into {leaked}"


def test_captured_output_lands_in_the_stage_log(tty, tmp_path, capsys):
    console.capture_stages(tmp_path)
    try:
        console.stage_event("crop_objects", "scan", "start", {})
        print("🖼️ Starting to obtain 2D images for objects...")
        print("  ✓ Processed Table0 (0.14s)")
        console.stage_event("crop_objects", "scan", "done", {"n_objects": 9})
    finally:
        console.restore_capture()

    out = capsys.readouterr().out
    assert "Processed Table0" not in out, "stage narration reached the terminal"
    assert "Processed Table0" in (tmp_path / "crop_objects.log").read_text()


def test_a_failing_stage_shows_its_tail(tty, tmp_path, capsys):
    """Hiding output is only acceptable if a failure still surfaces."""
    console.capture_stages(tmp_path)
    try:
        console.stage_event("crop_objects", "scan", "start", {})
        print("Traceback (most recent call last):")
        print("ValueError: the thing broke")
        console.stage_event("crop_objects", "scan", "error", {})
    finally:
        console.restore_capture()

    out = capsys.readouterr().out
    assert "ValueError: the thing broke" in out and "full log" in out


def test_restore_is_unconditional(tty, tmp_path, capsys):
    """A stage that raises never reaches its `done` event. Without an explicit restore, stdout
    stays pointed at a log file and the traceback itself disappears into it."""
    import sys

    real = sys.stdout
    console.capture_stages(tmp_path)
    console.stage_event("crop_objects", "scan", "start", {})
    assert sys.stdout is not real, "capture did not engage"
    console.restore_capture()
    assert sys.stdout is real, "stdout was left redirected"
    print("visible again")
    assert "visible again" in capsys.readouterr().out


# --- the live status line ----------------------------------------------------- #
def _run_stage(tmp_path, lines, stage="chair_clusters", pace=0.1):
    """`pace` spaces the lines out past the redraw throttle — a real stage prints seconds apart,
    and the throttle exists so a chatty one cannot flicker."""
    import time

    console.capture_stages(tmp_path)
    try:
        console.stage_event(stage, "s", "start", {})
        for line in lines:
            print(line)
            time.sleep(pace)
        console.stage_event(stage, "s", "done", {"chairs": 5, "clusters": 2})
    finally:
        console.restore_capture()


def test_live_status_shows_the_newest_meaningful_line(tty, tmp_path, capsys):
    """A slow stage sat on a static `▶ chair groups…` for ninety seconds with no sign of life.
    Its own narration says what it is doing, so that becomes the status."""
    _run_stage(tmp_path, ["[chair-cluster] DINOv2 embeddings for 5 chairs",
                          "[chair-judge] claude-opus-5: 5 chairs -> 2 types"])
    out = capsys.readouterr().out
    assert "DINOv2 embeddings" in out and "claude-opus-5" in out
    assert out.count("\r") >= 2, "status is not redrawing in place"


def test_the_final_row_lands_on_a_clean_line(tty, tmp_path, capsys):
    _run_stage(tmp_path, ["some chatter"])
    out = capsys.readouterr().out
    assert "5 chairs → 2 groups" in out
    assert out.rstrip().endswith("\033[0m") or "✓" in out
    # the row is preceded by a clear, so no status text is left stranded on its line
    assert "\r\033[K" in out


def test_progress_bar_noise_is_not_used_as_status(tty, tmp_path, capsys):
    """tqdm redraws thousands of times a second and says nothing about what is happening."""
    _run_stage(tmp_path, [" 40%|##########        | 4/10 [00:00<00:00,  9.78it/s]",
                          "  ✓ Processed Chair2 (0.09s)"])
    out = capsys.readouterr().out
    assert "Processed Chair2" in out
    assert "it/s" not in out


def test_nothing_is_drawn_when_output_is_redirected(piped, tmp_path, capsys):
    """A log file must not fill with half-drawn status lines."""
    _run_stage(tmp_path, ["[chair-judge] working"])
    out = capsys.readouterr().out
    assert "\r" not in out and "[chair-judge]" not in out
    assert "5 chairs → 2 groups" in out  # the row itself still prints


def test_the_log_keeps_everything_the_status_summarised(tty, tmp_path, capsys):
    """The status shows one line at a time and throttles; the log must still have them all."""
    lines = [f"[nano-banana] ChairCluster{i}" for i in range(6)]
    _run_stage(tmp_path, lines, pace=0.0)  # faster than the throttle: most never render
    capsys.readouterr()
    log = (tmp_path / "chair_clusters.log").read_text()
    for line in lines:
        assert line in log, f"{line} was shown but never recorded"


def test_a_long_status_is_truncated_not_wrapped(tty, tmp_path, capsys, monkeypatch):
    """A wrapped status leaves fragments behind when the next redraw only clears one line."""
    monkeypatch.setattr("shutil.get_terminal_size", lambda *_: os.terminal_size((80, 24)))
    _run_stage(tmp_path, ["x" * 400])
    out = capsys.readouterr().out
    longest = max((len(seg) for seg in out.split("\r")), default=0)
    assert longest < 200, f"a {longest}-char status line will wrap"


# --- no filesystem side effects from importing --------------------------------- #
def test_importing_the_harness_config_creates_nothing(tmp_path, monkeypatch):
    """`run/test-scan-Room/` turned up in a user's data tree: the harness config mkdir'd under
    $LITEREALITY_OUTPUT at import time, so merely importing it — from a test, a --help, an
    editor's autocomplete — left an empty scan tree next to their scans."""
    import subprocess
    import sys

    probe = (
        "import os, sys; from pathlib import Path;"
        f"os.environ.update(LITEREALITY_SCAN='probe', LITEREALITY_OUTPUT={str(tmp_path)!r},"
        " LITEREALITY_BLENDER='/nonexistent');"
        "sys.modules.setdefault('bpy', type(sys)('bpy'));"
        "import lrauthor.agent.tools.shared.config as c;"
        f"print(sorted(str(p) for p in Path({str(tmp_path)!r}).rglob('*')))"
    )
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, env=env)
    if r.returncode != 0:  # Blender is resolved at import; skip where it cannot be
        pytest.skip(f"config import needs a resolvable Blender: {r.stderr.strip()[-120:]}")
    assert r.stdout.strip().endswith("[]"), f"import created: {r.stdout.strip()}"


def test_ensure_dirs_is_what_creates_them(tmp_path, monkeypatch):
    """The creation did not vanish — it moved to an explicit call the writers make."""
    import subprocess
    import sys

    probe = (
        "import os, sys; from pathlib import Path;"
        f"os.environ.update(LITEREALITY_SCAN='probe', LITEREALITY_OUTPUT={str(tmp_path)!r},"
        " LITEREALITY_BLENDER='/nonexistent');"
        "sys.modules.setdefault('bpy', type(sys)('bpy'));"
        "import lrauthor.agent.tools.shared.config as c; c.ensure_dirs();"
        f"print(sorted(str(p.relative_to({str(tmp_path)!r})) for p in Path({str(tmp_path)!r}).rglob('*')))"
    )
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        pytest.skip("config import needs a resolvable Blender")
    assert "_harness" in r.stdout


# --- reconstruction concurrency ------------------------------------------------ #
def _run_mod():
    pytest.importorskip("cv2", reason="object_init.run needs the full preprocessing stack")
    pytest.importorskip("trimesh")
    import sys
    import types

    sys.modules.setdefault("open3d", types.ModuleType("open3d"))
    from lrauthor.pipeline.measure import flow as run

    return run


@pytest.mark.parametrize("objects,expected", [(0, 1), (1, 1), (2, 2), (6, 6), (8, 8), (9, 8), (40, 8)])
def test_a_branch_gets_one_worker_per_object_up_to_the_cap(objects, expected):
    """A flat 2 left a 6-object room mostly idle. The cap is real though: each worker is a
    Blender process plus its own agent session, so past it they contend rather than help."""
    run = _run_mod()
    assert run.branch_concurrency(objects) == expected
    assert run.MAX_PARALLEL_OBJECTS == 8


def test_an_explicit_concurrency_still_wins():
    run = _run_mod()
    assert run.branch_concurrency(40, explicit=3) == 3
    assert run.branch_concurrency(1, explicit=6) == 6


def test_branches_own_disjoint_objects(tmp_path, monkeypatch):
    """Each row counts ITS work. A shared denominator says the phase is 4/6 but not who is behind."""
    import json

    run = _run_mod()
    monkeypatch.setenv("LITEREALITY_FINAL", str(tmp_path))
    monkeypatch.setenv("LITEREALITY_OUTPUT", str(tmp_path))
    from lrauthor.pipeline.measure import paths as config

    config.set_scan("Sim")
    routing = config.work_root() / "routing"
    routing.mkdir(parents=True)
    (routing / "routing_manifest.json").write_text(json.dumps({"scans": {"Sim": [
        {"name": "Table0", "route": "procedural"}, {"name": "Table1", "route": "procedural"},
        {"name": "Sofa0", "route": "trellis"}]}}))
    for opening in ("Wall1_Door_0", "Wall5_Window_0"):
        (config.opening_refs_root() / "Sim" / opening).mkdir(parents=True)

    per = run._expected_by_branch("Sim")
    assert per["trellis"] == ["Sofa0"]
    assert per["procedural"] == ["Table0", "Table1"]
    assert per["openings"] == ["Wall1_Door_0", "Wall5_Window_0"]
    everything = [o for objs in per.values() for o in objs]
    assert len(everything) == len(set(everything)), "an object is claimed by two branches"


def _branch(total, built=(), active=(), queued=(), started=None, finished=None, result="",
            activity=None):
    """One branch as the reconstruction loop reports it: three sets, read off disk."""
    return {"total": total, "done": len(built), "built": list(built), "active": list(active),
            "activity": dict(activity or {}),
            "queued": list(queued), "started": started, "finished": finished, "result": result}


@pytest.fixture
def window(monkeypatch):
    """A terminal of known size. `shutil.get_terminal_size` reads these before it measures."""
    def _set(columns=100, lines=40):
        monkeypatch.setenv("COLUMNS", str(columns))
        monkeypatch.setenv("LINES", str(lines))
    _set()
    return _set


def test_a_finished_branch_stays_on_the_board(tty, window, tmp_path, capsys):
    """TRELLIS did its two chairs in 118s and vanished from the display — the phase then looked
    like two slow branches when three had run."""
    import time as _t

    board = console.BranchBoard(["trellis", "procedural"], 4, workers={"procedural": 2})
    now = _t.time()
    board.render({
        "trellis": _branch(2, built=["chair_a", "chair_b"], started=now - 118, finished=now),
        "procedural": _branch(2, built=["Table0"], active=["Table1"], started=now - 273),
    })
    out = capsys.readouterr().out
    assert "TRELLIS" in out and "2 / 2" in out
    assert "Procedural" in out and "1 / 2" in out
    assert "Table1" in out, "the object actually being built is the point of the board"
    assert "3 / 26" not in out and "3 / 4" in out  # the overall total still adds up


def test_a_full_bar_with_branches_running_says_verifying(tty, window, tmp_path, capsys):
    """`2/2` with the clock still climbing reads as a hang. It is not: the count is GLBs on disk,
    and the articulated agent writes one, probe-checks it, renders it for a VLM completeness
    check, and rewrites it if either gate fails."""
    import time as _t

    board = console.BranchBoard(["procedural", "openings"], 4)
    now = _t.time()
    board.render({
        "procedural": _branch(2, built=["a", "b"], started=now - 486),
        "openings": _branch(2, built=["c", "d"], started=now - 486),
    })
    out = capsys.readouterr().out
    assert "4 / 4" in out and "verifying" in out
    assert "Procedural" in out and "Openings" in out


def test_a_finished_phase_does_not_claim_to_be_verifying(tty, window, tmp_path, capsys):
    import time as _t

    board = console.BranchBoard(["procedural"], 2)
    now = _t.time()
    board.render({"procedural": _branch(2, built=["a", "b"], started=now - 60, finished=now)})
    out = capsys.readouterr().out
    assert "2 / 2" in out and "verifying" not in out


def test_a_branch_that_finished_short_is_counted_as_failed(tty, window, tmp_path, capsys):
    """Objects still queued have not failed. Objects a FINISHED branch never produced have."""
    import time as _t

    board = console.BranchBoard(["procedural", "openings"], 4)
    now = _t.time()
    board.render({
        "procedural": _branch(2, built=["a"], started=now - 60, finished=now, result="1/2 built"),
        "openings": _branch(2, built=[], queued=["c", "d"], started=now - 60),
    })
    out = capsys.readouterr().out
    assert "1 failed" in out, "the one object the finished branch never built"
    assert "2 queued" in out


# --- the board holds still --------------------------------------------------- #
STATE = {
    "openings": _branch(5, built=["w1", "w2", "w3", "w4"], active=["door_01"], queued=["window_03"]),
    "procedural": _branch(16, built=[f"o{i}" for i in range(8)], active=["wardrobe_02"],
                          queued=[f"q{i}" for i in range(7)]),
    "trellis": _branch(5, built=["sofa", "armchair_02"], active=["armchair_01"],
                       queued=["side_table", "plant"]),
}


@pytest.mark.parametrize("columns,rows", [(120, 50), (100, 40), (84, 20), (70, 13), (60, 9),
                                          (60, 5), (40, 4)])
def test_the_board_never_outgrows_the_window(tty, tmp_path, columns, rows):
    """The whole reason this is height-aware.

    The board redraws by moving the cursor up one line per ROW it drew. A block taller than the
    window cannot be reached that way, and a row wider than the window soft-wraps into two — both
    leave the cursor short, and the board then marches down the screen a frame at a time instead
    of refreshing. Neither dimension may ever be exceeded, at any size.
    """
    board = console.BranchBoard(["openings", "procedural", "trellis"], 26)
    board._observe(STATE)
    lines = board._fit(STATE, columns, rows - 1)

    assert 0 < len(lines) <= rows - 1, "taller than the window it has to redraw inside"
    for parts in lines:
        visible = console._ANSI.sub("", console._paint(parts, columns))
        assert len(visible) <= columns, f"{visible!r} is wider than the window and will wrap"


def test_a_short_window_gives_up_the_queue_before_the_live_objects(tty, tmp_path):
    """What is building now outranks what is waiting — you can only act on the former."""
    board = console.BranchBoard(["openings", "procedural", "trellis"], 26)
    board._observe(STATE)
    roomy = "\n".join(console._paint(p, 100) for p in board._fit(STATE, 100, 39))
    cramped = "\n".join(console._paint(p, 100) for p in board._fit(STATE, 100, 17))

    assert "Queued" in roomy and "wardrobe_02" in roomy
    assert "Queued" not in cramped, "the queue is the first thing a short window should shed"
    assert "wardrobe_02" in cramped, "but not at the cost of what is building right now"


def test_the_smallest_window_keeps_the_counts_rather_than_the_title(tty, tmp_path):
    """Trimming the other end would spend the last lines of a tiny window on a banner."""
    board = console.BranchBoard(["openings", "procedural", "trellis"], 26)
    board._observe(STATE)
    text = "\n".join(console._paint(p, 60) for p in board._fit(STATE, 60, 3))

    assert "completed" in text
    assert "LiteReality-Agent" not in text


def test_the_board_shows_what_each_object_is_doing(tty, window, tmp_path, capsys):
    """A caption that cannot change is not status.

    "building frame" sat beside four openings for seven straight minutes — same words whether an
    object was writing its build script or waiting on a VLM check. The caller works the phase out
    from the object's own directory; the branch's generic verb is only the fallback.
    """
    board = console.BranchBoard(["openings"], 2, workers={"openings": 2})
    board.render({"openings": _branch(
        2, active=["Wall1_Door_1", "Wall4_Door_2"], started=1.0,
        activity={"Wall1_Door_1": "checking result", "Wall4_Door_2": "rendering previews"},
    )})
    out = capsys.readouterr().out

    assert "checking result" in out and "rendering previews" in out
    assert "building frame" not in out


def test_an_object_with_no_readable_phase_falls_back_to_the_branch_verb(tty, window, capsys):
    """TRELLIS leaves nothing partial to read, so there is nothing to derive — say something."""
    board = console.BranchBoard(["trellis"], 2, workers={})
    board.render({"trellis": _branch(2, active=["sofa"], started=1.0)})

    assert "reconstructing asset" in capsys.readouterr().out


@pytest.mark.parametrize("files,dirs,expected", [
    (["Wall1.glb"], ["previews"], "checking result"),
    (["Wall1.glb"], [], "rendering previews"),
    (["build_Wall1.py"], [], "running the build"),
    ([], ["textures"], "preparing textures"),
    ([], [], "starting up"),
])
def test_the_object_phase_reads_the_newest_milestone(tmp_path, files, dirs, expected):
    """Builders write build script → GLB → previews → object.py/md, so the last one present wins."""
    run = _run_module()
    for name in files:
        (tmp_path / name).write_bytes(b"")
    for name in dirs:
        (tmp_path / name).mkdir()

    assert run._object_phase(tmp_path) == expected


# --- estimates are earned, not invented --------------------------------------- #
def test_no_estimate_until_something_has_actually_finished(tty, window, tmp_path, capsys):
    """A remaining-time figure extrapolated from zero completions is a number-shaped guess."""
    board = console.BranchBoard(["procedural"], 4)
    board.render({"procedural": _branch(4, built=[], active=["a"], queued=["b", "c", "d"])})

    assert "remaining" not in capsys.readouterr().out


def test_objects_already_on_disk_do_not_count_toward_the_rate(tty, window, tmp_path, capsys):
    """A resumed run starts with output already there. Counting it as work done in the last
    instant makes the first estimate absurdly optimistic."""
    board = console.BranchBoard(["procedural"], 4)
    board.render({"procedural": _branch(4, built=["a", "b", "c"], active=["d"])})
    out = capsys.readouterr().out

    assert "3 / 4" in out
    assert "remaining" not in out, "three objects appeared at once because they were already there"


# --- what counts as built ------------------------------------------------------ #
def test_a_half_written_object_does_not_count_as_built(tmp_path, monkeypatch):
    """The agent writes the GLB and previews BEFORE object.py/object.md, so an interrupted run
    leaves a directory that looks finished. Counting the glb alone reported 6/6 while two
    objects were being regenerated from scratch — thirteen minutes of work the bar called done."""
    run = _run_mod()
    monkeypatch.setenv("LITEREALITY_FINAL", str(tmp_path))
    monkeypatch.setenv("LITEREALITY_OUTPUT", str(tmp_path))
    from lrauthor.pipeline.measure import paths as config

    config.set_scan("Sim")
    recon = config.reconstruct_dir("Sim")
    recon.mkdir(parents=True)

    (recon / "Sofa0.glb").write_bytes(b"")            # neural: one baked mesh is the whole artifact
    for name, files in (
        ("Table0", ("Table0.glb", "object.py", "object.md")),   # complete
        ("Table1", ("Table1.glb",)),                            # killed before object.py
        ("Door0", ("Door0.glb", "object.py")),                  # killed before object.md
    ):
        d = recon / name
        d.mkdir()
        for f in files:
            (d / f).write_text("")

    assert run._built_count("Sim", ["Sofa0"]) == 1
    assert run._built_count("Sim", ["Table0"]) == 1
    assert run._built_count("Sim", ["Table1"]) == 0, "a glb without its source is not built"
    assert run._built_count("Sim", ["Door0"]) == 0, "object.md missing means the agent redoes it"
    assert run._built_count("Sim", ["Sofa0", "Table0", "Table1", "Door0"]) == 2


def test_the_counter_matches_what_the_builder_would_skip(tmp_path, monkeypatch):
    """The bar and --skip-existing must agree on 'built', or the bar promises work the builder is
    about to redo."""
    run = _run_mod()
    monkeypatch.setenv("LITEREALITY_FINAL", str(tmp_path))
    monkeypatch.setenv("LITEREALITY_OUTPUT", str(tmp_path))
    from lrauthor.pipeline.measure import paths as config

    config.set_scan("Sim")
    recon = config.reconstruct_dir("Sim")
    (recon / "Table1").mkdir(parents=True)
    (recon / "Table1" / "Table1.glb").write_bytes(b"")
    previews = recon / "Table1" / "previews"
    previews.mkdir()
    for i in range(3):
        (previews / f"{i}.png").write_bytes(b"")

    # the builder's own requirement set (generate_procedural.verify_reasons)
    d = recon / "Table1"
    builder_says_built = (
        (d / "Table1.glb").is_file()
        and len(list(previews.glob("*.png"))) >= 2
        and (d / "object.py").is_file()
        and (d / "object.md").is_file()
    )
    assert builder_says_built is False
    assert run._built_count("Sim", ["Table1"]) == 0, "the bar disagreed with the builder"


def test_warnings_are_not_used_as_the_live_status(tty, tmp_path, capsys):
    """A stale HF rate-limit notice sitting on the status line for a minute is worse than
    showing nothing — the status should say what the stage is DOING. The warning is still logged."""
    _run_stage(tmp_path, [
        "Warning: You are sending unauthenticated requests to the HF Hub. Set a HF_TOKEN.",
        "Floor            refined 2/4 dropped 0+2blur (max_iou=0.932)",
    ], stage="bbox_polish")
    out = capsys.readouterr().out
    assert "refined 2/4" in out
    assert "HF Hub" not in out
    assert "HF Hub" in (tmp_path / "bbox_polish.log").read_text(), "the warning was lost"


# --- bbox polish reuse --------------------------------------------------------- #
def test_polish_reuses_a_recorded_run(tmp_path, monkeypatch):
    """The refinement is written back into each crop and meta, so re-running it re-detects every
    box to reach the state it is already in — a minute of DINO for nothing."""
    pytest.importorskip("cv2")
    import json as _json
    import sys
    import types

    sys.modules.setdefault("open3d", types.ModuleType("open3d"))
    monkeypatch.setenv("LITEREALITY_FINAL", str(tmp_path))
    monkeypatch.setenv("LITEREALITY_OUTPUT", str(tmp_path))
    from lrauthor.pipeline.measure import paths as config
    from lrauthor.pipeline.measure.ingest.detect import bbox_polish

    config.set_scan("Sim")
    marker = bbox_polish.marker_path("Sim")
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(_json.dumps({"scan": "Sim", "refined_total": 29}))

    out = bbox_polish.polish("Sim")
    assert out["reused"] is True and out["refined_total"] == 29


@pytest.mark.live
def test_force_ignores_the_marker(tmp_path, monkeypatch):
    """After new crops or a different detector, the recorded run is stale.

    Marked live: unlike the reuse case above, force=True skips the marker and runs the real
    detector, which needs the Modal extra and Modal credentials.
    """
    pytest.importorskip("cv2")
    import json as _json
    import sys
    import types

    sys.modules.setdefault("open3d", types.ModuleType("open3d"))
    monkeypatch.setenv("LITEREALITY_FINAL", str(tmp_path))
    monkeypatch.setenv("LITEREALITY_OUTPUT", str(tmp_path))
    from lrauthor.pipeline.measure import paths as config
    from lrauthor.pipeline.measure.ingest.detect import bbox_polish

    config.set_scan("Sim")
    marker = bbox_polish.marker_path("Sim")
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(_json.dumps({"scan": "Sim", "refined_total": 29}))

    out = bbox_polish.polish("Sim", force=True)
    assert not out.get("reused"), "force must not return the recorded run"
