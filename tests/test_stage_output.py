"""Stage output reaches the terminal and the log — both, live.

Every row `console` renders is printed by a subprocess that `support.run_module` spawns, so where
that subprocess's stdout goes decides whether a run looks alive. Handing it the log file directly
put the whole display in `realism_authoring/logs/` and left the terminal blank for the minutes a
stage takes, which reads exactly like a hang. These tests pin the three things that made the tee
worth having over the redirect: the output is duplicated, it arrives while the child is still
running, and the escapes stay out of the file.
"""

from __future__ import annotations

import fcntl
import os
import pty
import struct
import sys
import termios
import textwrap
import time

import pytest

from lrauthor.pipeline import support
from lrauthor.pipeline.context import RunContext

# Colours, an in-place redraw, a pause long enough that buffering is unmistakable, and a
# non-zero exit — a pipeline stage in miniature.
PROBE = '''\
import os, shutil, sys, time
print(f"colour={os.environ.get('LR_COLOR')} unbuffered={os.environ.get('PYTHONUNBUFFERED')}")
print(f"width={shutil.get_terminal_size((100, 24)).columns}")
print("FIRST")
print("\\r\\033[K   \\033[1;36mcrops\\033[0m  1/2", end="")
sys.stdout.flush()
time.sleep(0.6)
print("\\r\\033[K   \\033[1;36mcrops\\033[0m  2/2")
print("SECOND")
sys.exit(3)
'''


class Recorder:
    """Stands in for the terminal, remembering WHEN each write landed."""

    def __init__(self) -> None:
        self.events: list[tuple[float, str]] = []
        self._t0 = time.monotonic()

    def write(self, text: str) -> int:
        self.events.append((time.monotonic() - self._t0, text))
        return len(text)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return True

    @property
    def text(self) -> str:
        return "".join(chunk for _, chunk in self.events)

    def when(self, needle: str) -> float:
        return next(at for at, chunk in self.events if needle in chunk)


@pytest.fixture
def probe(tmp_path, monkeypatch):
    """A RunContext whose repo root holds `probe.py`, and a `run()` that records the terminal."""
    monkeypatch.setenv("LR_COLOR", "1")  # decide colour as if the parent were a terminal
    monkeypatch.delenv("LR_QUIET", raising=False)
    (tmp_path / "probe.py").write_text(textwrap.dedent(PROBE), encoding="utf-8")
    context = RunContext(
        scan="Probe",
        capture_dir=tmp_path / "capture",
        scene_dir=tmp_path / "scene",
        output_root=tmp_path / "out",
        repo_root=tmp_path,
    )

    def run(**kwargs):
        # Swapped around the call rather than in fixture setup: pytest re-installs its own capture
        # object on sys.stdout when the test body begins, which would undo an earlier swap.
        recorder = Recorder()
        real, sys.stdout = sys.stdout, recorder
        try:
            rc, log = support.run_module(context, "probe", **kwargs)
        finally:
            sys.stdout = real
        return recorder, rc, log

    return context, run


def test_stage_output_reaches_both_the_terminal_and_the_log(probe):
    context, run = probe
    recorder, rc, log = run(log_name="probe")

    assert rc == 3, "the stage's exit code is what marks the result failed; it must survive the pipe"
    assert log == context.authoring_root / "logs" / "probe.log"
    for line in ("FIRST", "SECOND", "crops"):
        assert line in recorder.text, f"{line!r} never reached the terminal"
        assert line in log.read_text(encoding="utf-8"), f"{line!r} never reached the log"


def test_output_arrives_while_the_stage_is_still_running(probe):
    """A tee that only flushes at exit is the blank terminal all over again, just better hidden."""
    recorder, _, _ = probe[1](log_name="probe")

    gap = recorder.when("SECOND") - recorder.when("FIRST")
    assert gap > 0.4, f"output was batched, not streamed (FIRST and SECOND {gap:.2f}s apart)"


def test_the_child_is_told_to_stay_unbuffered_and_colourful(probe):
    """Both are inferred from the child's OWN stdout, which is a pipe — so we override them."""
    recorder, _, _ = probe[1](log_name="probe")

    assert "colour=1 unbuffered=1" in recorder.text
    assert "\033[1;36m" in recorder.text, "the terminal keeps the escapes the log gives up"


def test_the_log_keeps_the_text_and_drops_the_escapes(probe):
    """`\\033[1;36m` in a file is unreadable, and a bare `\\r` collapses it into one long line."""
    _, _, log = probe[1](log_name="probe")
    written = log.read_text(encoding="utf-8")

    assert "\033" not in written and "\r" not in written
    assert "   crops  1/2" in written, "the redraw frames become ordinary lines, not nothing"


@pytest.mark.skipif(os.name != "posix", reason="needs a pty to fake a terminal of known width")
def test_the_child_is_told_how_wide_the_terminal_is(tmp_path, monkeypatch):
    """The regression the branch board scrolls on.

    A pipe has no width, so the child's `get_terminal_size` falls back to 100 columns. On a
    narrower window every board row wraps to two physical lines while the redraw still moves the
    cursor up one line per ROW — it lands short, and the board marches down the screen instead of
    refreshing. The parent can still measure, so it measures and passes the answer down.
    """
    monkeypatch.setenv("LR_COLOR", "1")
    for inherited in ("COLUMNS", "LINES"):  # else the child would read them rather than be told
        monkeypatch.delenv(inherited, raising=False)
    (tmp_path / "probe.py").write_text(textwrap.dedent(PROBE), encoding="utf-8")
    context = RunContext(
        scan="Probe", capture_dir=tmp_path / "capture", scene_dir=tmp_path / "scene",
        output_root=tmp_path / "out", repo_root=tmp_path,
    )

    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 72, 0, 0))
    terminal = os.fdopen(slave, "w")
    # `shutil.get_terminal_size` measures `sys.__stdout__`, not `sys.stdout` — under pytest the
    # real one is a capture file, which is exactly as unmeasurable as the pipe.
    monkeypatch.setattr(sys, "__stdout__", terminal)
    recorder = Recorder()
    real, sys.stdout = sys.stdout, recorder
    try:
        support.run_module(context, "probe", log_name="probe")
    finally:
        sys.stdout = real
        terminal.close()
        os.close(master)

    assert "width=72" in recorder.text, "the child fell back to its pipe default and will wrap"


def test_a_stage_without_a_log_name_still_inherits_the_terminal(tmp_path, monkeypatch):
    """No log to write means no reason to pipe: let the child own the real terminal."""
    (tmp_path / "probe.py").write_text(textwrap.dedent(PROBE), encoding="utf-8")
    context = RunContext(
        scan="Probe", capture_dir=tmp_path / "capture", scene_dir=tmp_path / "scene",
        output_root=tmp_path / "out", repo_root=tmp_path,
    )
    rc, log = support.run_module(context, "probe")

    assert (rc, log) == (3, None)
    assert not (context.authoring_root / "logs").exists()
