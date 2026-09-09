"""Compact, colour-coded stage progress for the terminal.

A pipeline run prints a lot: subprocess chatter, per-object logs, warnings. The stage boundaries
were buried in it, so "where is this run, and did that step actually do anything" needed reading
a screen of text. This renders one line per stage instead — a stable colour per stage, the result
it produced, and how long it took:

    ── scene_init · Office_room ───────────────────────────────────
       ✓ extract        reused                              0.1s
       ✓ crops          9 objects                          12.0s
       ✓ object refs    2 objects                          48.0s
       ✓ chair groups   5 chairs → 2 groups                62.1s
       ✗ procedural     0 GLB                               0.2s

The colour is per stage and stable across runs, grouped by hue: preparation is cyan/blue,
image-generation magenta, 3D generation green. You learn the shape of a run by its colours.

Everything degrades: no colour when stdout is not a tty (`$LR_COLOR=1/0` forces it either
way, for `| less -R` or for a log file), and `$LR_QUIET=1` turns the display off entirely. The
JSONL trace is written regardless — this is presentation, never the record.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import time
from pathlib import Path

__all__ = ["stage_event", "rule", "row", "colour", "use_colour", "enabled", "short", "plain"]

_C = {
    "dim": "\033[2m", "bold": "\033[1m", "off": "\033[0m",
    "red": "\033[1;31m", "green": "\033[1;32m", "yellow": "\033[1;33m",
    "blue": "\033[1;34m", "magenta": "\033[1;35m", "cyan": "\033[1;36m",
    "grey": "\033[0;37m",
}

# stage name -> (label, colour, announce-on-start, what a ZERO result means).
#
# "Announce" marks the slow stages, which say so when they begin — minutes of subprocess output
# with no header is the thing this replaces; the fast ones would just double their line count.
#
# The zero column matters more than it looks. A build stage that emitted 0 GLBs has failed and
# must not render as a tick — that is the signal that caught the procedural crash. But most
# stages have nothing to do some of the time: a room with no counter runs to merge, no openings,
# no chairs needing repair. Reporting those as failures trains you to ignore the ✗.
STAGES: dict[str, tuple[str, str, bool, str]] = {
    "extract_scene":      ("extract",      "cyan",    False, "nothing to extract"),
    "box_merge":          ("box merge",    "cyan",    False, "no overlaps to merge"),
    "crop_objects":       ("crops",        "cyan",    True,  "no objects found"),
    "bbox_polish":        ("dino polish",  "blue",    True,  "nothing to refine"),
    "object_references":  ("object refs",  "magenta", True,  "no objects to reference"),
    "chair_clusters":     ("chair groups", "magenta", True,  "no chairs"),
    "opening_references": ("opening refs", "magenta", True,  "no doors or windows"),
    "classify":           ("routing",      "yellow",  False, "nothing to route"),
    "reconstruct":        ("trellis",      "green",   True,  ""),   # "" = a zero here is a FAILURE
    "chair_qc":           ("chair qc",     "green",   True,  "all pass"),
    "procedural":         ("procedural",   "green",   True,  ""),
    "build_openings":     ("openings",     "green",   True,  ""),
}

# How each stage's payload reads as one short phrase. Keys are the kwargs the stage already
# passes to `telemetry.stage(...)`, so adding a counter there is all it takes to surface it.
def _summary(name: str, data: dict) -> str:
    def n(key, one, many=None):
        if key not in data:
            return None
        v = data[key]
        return f"{v} {one if v == 1 else (many or one)}"

    if name == "classify":
        parts = [f"{data[k]} {k}" for k in ("procedural", "trellis") if data.get(k)]
        return " · ".join(parts) or "nothing to route"
    if name == "chair_clusters":
        chairs, groups = data.get("chairs"), data.get("clusters")
        return f"{chairs} chairs → {groups} groups" if chairs is not None else ""
    if name == "chair_qc":
        # A zero is only worth the space when it is bad news: "0 still failing" is noise,
        # "0 repaired" alongside failures is not.
        bits = [x for x in (n("repaired", "repaired"),
                            n("still_failing", "still failing") if data.get("still_failing") else None)
                if x]
        return " · ".join(bits) or "all pass"
    for key, one, many in (
        ("n_glb", "GLB", "GLBs"), ("n_objects", "object", "objects"),
        ("n_openings", "opening", "openings"), ("merged", "merged", "merged"),
        ("refined", "box refined", "boxes refined"),
    ):
        got = n(key, one, many)
        if got:
            return got
    return ""


_MARK = {"done": ("✓", None), "reused": ("·", "dim"), "error": ("✗", "red"),
         "skipped": ("⊘", "dim")}
_started: dict[str, float] = {}
_announced: set[str] = set()

# --- stage output capture ---------------------------------------------------- #
# The vendored preprocessing narrates: a line per object, a tqdm bar per loop, a line per
# stitched folder. Forty lines of it to say "9 crops in 1.4s". Since every stage is already
# bracketed by telemetry.stage(start) / telemetry.stage(done), that bracket is exactly where the
# noise can be diverted to a log, leaving the one summary row. $LR_VERBOSE=1 turns it off.
_capture_dir: "Path | None" = None
_capture: dict[str, object] = {}


def restore_capture() -> None:
    """Force stdout/stderr back to the terminal. Safe to call when nothing is captured.

    A stage that raises never reaches its `telemetry.stage(done)`, so the paired `_end_capture`
    never runs and every later byte — including the traceback — lands in a log file nobody is
    looking at. Callers put this in a `finally`.
    """
    if _capture:
        _end_capture(str(_capture.get("name")), failed=True)
    capture_stages(None)


def capture_stages(directory) -> None:
    """Send stage output to `<directory>/<stage>.log` instead of the terminal. None disables.

    Disabled inside the parallel reconstruction phase, which routes per BRANCH via ThreadTee —
    swapping a process-global from a worker thread there would cross the streams.
    """
    global _capture_dir
    _capture_dir = Path(directory) if directory else None


def _begin_capture(name: str) -> None:
    if _capture_dir is None or _capture or os.environ.get("LR_VERBOSE", "").strip() in ("1", "true"):
        return  # already capturing (a nested stage) or asked not to
    try:
        _capture_dir.mkdir(parents=True, exist_ok=True)
        handle = open(_capture_dir / f"{name}.log", "a", encoding="utf-8", errors="replace")
    except OSError:
        return
    label, hue = (STAGES.get(name) or (name.replace("_", " "), "grey"))[:2]
    stream = _StageStream(handle, label, hue, sys.stdout)
    _capture.update(name=name, handle=handle, stream=stream, out=sys.stdout, err=sys.stderr)
    sys.stdout = sys.stderr = stream



_ANSI = re.compile(r"\033\[[0-9;]*[A-Za-z]")
# Not worth a status line: separators, bare tqdm bars, and third-party warnings. The status
# should say what the stage is DOING — a stale HF rate-limit notice sitting there for a minute
# is worse than showing nothing, and every one of these is still in the log.
_NOISE = re.compile(
    r"^[\s\-=_.·•]*$"
    r"|^\s*\d+%\|"
    r"|^(warning|userwarning|futurewarning|deprecationwarning)\b"
    r"|^\S+warning:"
    r"|huggingface|hf_token|hf hub"
    r"|^\s*warnings?\.warn",
    re.I,
)


def _clean(line: str) -> str:
    return _ANSI.sub("", line).replace("\t", " ").strip()


def plain(text: str) -> str:
    """`text` with the escapes removed and in-place redraws unrolled — what a log file wants.

    Unlike `_clean` this keeps whole streams rather than one line: no stripping, no tab handling,
    and the line structure preserved. A live display redraws by returning to the start of the line
    and moving the cursor up; written verbatim into a file that collapses into one unreadable
    mega-line, so each redraw becomes an ordinary line instead.
    """
    return _ANSI.sub("", text).replace("\r\n", "\n").replace("\r", "\n")


class _StageStream:
    """Tees a captured stage's output: everything to the log, the newest line to a live status.

    A slow stage used to sit on a static `▶ chair groups…` for ninety seconds with no sign of
    life. Its own narration already says what it is doing — DINOv2 embeddings, the judge call,
    each hosted reference image — so that narration is reused as a one-line status that refreshes in
    place, and still lands in the log in full.
    """

    def __init__(self, handle, label: str, hue: str, out):
        self.handle, self.label, self.hue, self.out = handle, label, hue, out
        self._buf = ""
        self._last = 0.0
        self.live = False

    def write(self, s: str) -> int:
        self.handle.write(s)
        if not use_colour():
            return len(s)  # a redirected run gets the log, not a screenful of half-drawn lines
        self._buf += s
        if "\n" in self._buf or "\r" in self._buf:
            parts = re.split(r"[\r\n]", self._buf)
            self._buf = parts[-1]
            for raw in reversed(parts[:-1]):
                line = _clean(raw)
                if line and not _NOISE.match(line):
                    self._status(line)
                    break
        return len(s)

    def _status(self, text: str) -> None:
        now = time.time()
        if now - self._last < 0.08:  # a stage can print faster than anyone can read
            return
        self._last = now
        width = shutil.get_terminal_size((100, 24)).columns
        head = f"   {colour('dim')}▶{colour('off')} {colour(self.hue)}{self.label:<14}{colour('off')} "
        room = max(10, width - len(self.label) - 8)
        if len(text) > room:
            text = text[: room - 1] + "…"
        print(f"\r\033[K{head}{colour('dim')}{text}{colour('off')}", end="", file=self.out, flush=True)
        self.live = True

    def clear(self) -> None:
        if self.live:
            print("\r\033[K", end="", file=self.out, flush=True)
            self.live = False

    def flush(self) -> None:
        try:
            self.handle.flush()
        except Exception:  # noqa: BLE001
            pass

    def isatty(self) -> bool:
        return False  # the stage is writing to a log: no colour codes in it

    def __getattr__(self, name):
        return getattr(self.handle, name)


def _out():
    """The real terminal stream, even mid-capture.

    Stage rows must never travel through the redirected `sys.stdout`: stages nest (the crop pass
    emits opening-reference events inside its own bracket), and a nested row printed while the
    outer stage is captured lands in the outer stage's log instead of on screen — which reads
    exactly like the run stopping dead.
    """
    return _capture.get("out") or sys.stdout


def _end_capture(name: str, failed: bool) -> None:
    if _capture.get("name") != name:
        return
    handle, out, err = _capture["handle"], _capture["out"], _capture["err"]
    stream = _capture.get("stream")
    if stream is not None:
        stream.clear()  # wipe the live line so the final row lands on a clean one
    sys.stdout, sys.stderr = out, err
    _capture.clear()
    try:
        handle.close()
        if failed:  # never swallow a failure: show the tail, and where the rest is
            path = _capture_dir / f"{name}.log"
            tail = path.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
            for line in tail:
                print(f"     {colour('dim')}{line}{colour('off')}", file=out)
            print(f"     {colour('dim')}… full log: {short(path)}{colour('off')}", file=out)
    except OSError:
        pass


def enabled() -> bool:
    return os.environ.get("LR_QUIET", "").strip() not in ("1", "true", "yes")


def use_colour() -> bool:
    """Colour when writing to a terminal, or when `$LR_COLOR` says so either way.

    The override is not only for tests: piping a run through `less -R` or `tee` is exactly when
    you still want the colours, and redirecting to a log file is when you never do.
    """
    forced = os.environ.get("LR_COLOR", "").strip().lower()
    if forced in ("1", "true", "yes", "always"):
        return True
    if forced in ("0", "false", "no", "never"):
        return False
    try:
        return _out().isatty()
    except Exception:  # noqa: BLE001 — a replaced stdout without isatty is not a reason to fail
        return False


def colour(name: str) -> str:
    return _C.get(name, "") if use_colour() else ""


def rule(title: str, width: int = 72) -> str:
    c, off = colour("cyan"), colour("off")
    return f"\n{c}── {title} {'─' * max(0, width - len(title) - 4)}{off}"


def row(mark: str, label: str, value: str, label_colour: str = "") -> str:
    """`   ✓ label          value` — glyph coloured by outcome, label by stage."""
    glyph, glyph_colour = _MARK.get(mark, ("·", "dim"))
    gc = colour(glyph_colour) if glyph_colour else colour("green")
    off = colour("off")
    lc = colour(label_colour) if label_colour else ""
    # Only close a colour that was actually opened — a bare reset next to an uncoloured label
    # prints as a literal escape on terminals that do not swallow it.
    label_part = f"{lc}{label:<14}{off}" if lc else f"{label:<14}"
    return f"   {gc}{glyph}{off} {label_part} {value}"


def short(path) -> str:
    """A path relative to the checkout — absolute ones wrap and bury the part that differs.

    Only when it IS under the checkout. A `--output-root` somewhere else relativises to
    `../../../../../private/var/folders/…`, which is longer than the absolute path and harder to
    read: the point was to drop a shared prefix, and there is no shared prefix to drop.
    """
    from lrauthor import REPO_ROOT

    if not path:
        return ""
    try:
        relative = os.path.relpath(os.path.realpath(str(path)), os.path.realpath(str(REPO_ROOT)))
    except ValueError:  # different drive on Windows: no relative path exists at all
        return str(path)
    return str(path) if relative.startswith("..") else relative


def stage_event(name: str, scan: str, status: str, data: dict) -> None:
    """Render one stage boundary. Called from `telemetry.stage`; never raises."""
    if not enabled():
        return
    try:
        label, hue, announce, zero_means = STAGES.get(
            name, (name.replace("_", " "), "grey", False, "nothing to do")
        )
        key = f"{scan}/{name}"

        if status == "start":
            _started[key] = time.time()
            if announce and key not in _announced:
                _announced.add(key)
                # No newline on a terminal: the live status line overwrites this in place, and
                # the final row clears it. Piped output keeps the newline and stays a plain log.
                end = "" if use_colour() else "\n"
                print(f"   {colour('dim')}▶ {label}…{colour('off')}", end=end,
                      file=_out(), flush=True)
            _begin_capture(name)
            return

        _end_capture(name, failed=status == "error")
        elapsed = time.time() - _started.pop(key, time.time())
        _announced.discard(key)
        detail = _summary(name, data) or ("reused" if status == "reused" else "")
        mark = status if status in _MARK else "done"
        if mark == "done" and detail.startswith("0 "):
            if zero_means:
                # A legitimate empty result: say what it means, and stay neutral rather than
                # spending a ✗ on it.
                detail, mark = zero_means, "reused"
            else:
                mark = "error"  # a build stage that produced nothing HAS failed
        stamp = f"{colour('dim')}{elapsed:6.1f}s{colour('off')}"
        lead = "\r\033[K" if use_colour() else ""  # wipe the announce / live status line
        print(f"{lead}{row(mark, label, f'{detail:<34}', hue)}{stamp}", file=_out(), flush=True)
    except Exception:  # noqa: BLE001 — display must never break a run
        pass


# ── live progress ────────────────────────────────────────────────────────────
class Progress:
    """A single-line progress bar for a phase whose work is counted in finished items.

    Written for the reconstruction phase, where the workers are subprocesses that each take tens
    of seconds and print heavily. It counts what actually landed on disk rather than what a
    worker claims, so it stays honest across three different builders (TRELLIS, the articulated
    agent, the openings pass) without any of them reporting in.

    Degrades to one line per change when stdout is not a terminal — `\r` into a log file
    produces a single unreadable mega-line.
    """

    def __init__(self, total: int, label: str = "reconstruction", width: int = 28):
        self.total = max(0, int(total))
        self.label = label
        self.width = width
        self.done = 0
        self.detail = ""
        self._t0 = time.time()
        self._last = ""

    def update(self, done: int, detail: str = "") -> None:
        self.done, self.detail = min(done, self.total), detail
        self._render()

    def _bar(self) -> str:
        if not self.total:
            return ""
        filled = int(self.width * self.done / self.total)
        return "█" * filled + "░" * (self.width - filled)

    def _render(self, final: bool = False) -> None:
        if not enabled():
            return
        elapsed = time.time() - self._t0
        line = (f"   {colour('cyan')}{self.label:<14}{colour('off')} "
                f"{self._bar()} {self.done}/{self.total}"
                f"  {colour('dim')}{elapsed:5.0f}s{colour('off')}")
        if self.detail:
            line += f"  {colour('dim')}{self.detail}{colour('off')}"
        if use_colour():
            print(f"\r\033[K{line}", end="\n" if final else "", file=_out(), flush=True)
        elif line != self._last:  # a log file gets one line per actual change, not per poll
            print(line, file=_out(), flush=True)
        self._last = line

    def finish(self, detail: str = "") -> None:
        self.detail = detail or self.detail
        self._render(final=True)


def fmt_elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def progress_over(count_done, total: int, label: str, stop, interval: float = 1.0,
                  running=None) -> None:
    """Render `count_done()` against `total` until `stop()` is true. Blocking; run it in a thread.

    `count_done` is polled rather than pushed so the workers need no instrumentation — several of
    them are subprocesses in other interpreters.

    `running()` returns `{branch: started_at}` for the branches still working. Without it a bar
    that sits at 4/6 for ten minutes is indistinguishable from a hang, which is exactly what the
    long articulated-agent branch looks like once the fast ones have finished.
    """
    bar = Progress(total, label)
    try:
        while not stop():
            bar.update(count_done(), _still_running(running))
            time.sleep(interval)
        bar.update(count_done(), _still_running(running))
    finally:
        bar.finish()


def _still_running(running) -> str:
    if running is None:
        return ""
    try:
        now = time.time()
        live = sorted(running().items())
        return " · ".join(f"{name} {fmt_elapsed(now - t0)}" for name, t0 in live)
    except Exception:  # noqa: BLE001
        return ""


class ThreadTee:
    """Per-thread stdout routing, installed for the duration of a parallel phase.

    `contextlib.redirect_stdout` swaps a process-global, so with several branches running it
    sends every thread's output to whichever log was bound last. This routes by thread instead:
    a bound thread writes to its log, everything else (notably the progress bar on the main
    thread) keeps the terminal.
    """

    def __init__(self, default):
        import threading

        self._threading = threading
        self._default = default
        self._targets: dict[int, object] = {}

    def bind(self, handle) -> None:
        self._targets[self._threading.get_ident()] = handle

    def unbind(self) -> None:
        self._targets.pop(self._threading.get_ident(), None)

    def _target(self):
        return self._targets.get(self._threading.get_ident(), self._default)

    def write(self, s):
        return self._target().write(s)

    def flush(self):
        try:
            self._target().flush()
        except Exception:  # noqa: BLE001
            pass

    def isatty(self):
        # A bound thread is writing to a file: report that, so colour codes stay out of the log
        # while the bar on the main thread keeps them.
        return self._target() is self._default and self._default.isatty()

    def __getattr__(self, name):
        return getattr(self._default, name)


# ── the reconstruction board ─────────────────────────────────────────────────
BRANCH_LABEL = {"trellis": "TRELLIS", "procedural": "Procedural", "openings": "Openings"}
BRANCH_ACTIVITY = {"trellis": "reconstructing asset", "procedural": "generating geometry",
                   "openings": "building frame"}
_BRANCH_STAGE = {"trellis": "reconstruct", "procedural": "procedural", "openings": "build_openings"}


def fmt_clock(seconds: float) -> str:
    """`02:44` — a stopwatch, for something being measured rather than guessed."""
    minutes, secs = divmod(int(max(0.0, seconds)), 60)
    if minutes < 60:
        return f"{minutes:02d}:{secs:02d}"
    return f"{minutes // 60}:{minutes % 60:02d}:{secs:02d}"


def fmt_rough(seconds: float) -> str:
    """`3m 20s` — spaced, so an estimate does not read like a stopwatch."""
    minutes, secs = divmod(int(max(0.0, seconds)), 60)
    return f"{minutes}m {secs:02d}s" if minutes else f"{secs}s"


def bar(done: int, total: int, cells: int) -> str:
    filled = int(cells * done / total) if total else 0
    return "█" * min(cells, filled) + "░" * max(0, cells - filled)


def _paint(parts, width: int) -> str:
    """Render `(text, colour)` segments, clipped to `width` VISIBLE characters.

    Width is counted on the text alone. Measuring the escape codes too — which is what happens
    if you build the string first and slice it — makes every coloured line clip early, and a
    line that overshoots instead is worse: it wraps, and a display that redraws by moving the
    cursor up one line per ROW then lands short and walks down the screen.
    """
    out, used = [], 0
    for text, hue in parts:
        if used >= width:
            break
        room = width - used
        if len(text) > room:
            text = text[: max(0, room - 1)] + "…"
        used += len(text)
        out.append(f"{colour(hue)}{text}{colour('off')}" if hue else text)
    return "".join(out)


class BranchBoard:
    """The reconstruction phase as a live board: what is building, what is done, what is waiting.

        LiteReality-Agent Obj Reconstruction                        02:44 elapsed

        Overall   ███████████░░░░░░░░░░░░░  11 / 26   42%
                  ~3m 20s remaining

        Generating now
          ◉ Openings      door_01      building frame           32s   ~18s left
          ◉ Procedural    wardrobe_02  generating geometry    1m42s   ~40s left

        Progress
          Openings       ████████░░   4 / 5
          Procedural     █████░░░░░   8 / 16
          TRELLIS        ████░░░░░░   2 / 5   · verifying

        Queued
          Procedural     desk_lamp · shelf · radiator · +5
        ──────────────────────────────────────────────────────────────────────
        3 active  ·  11 completed  ·  12 queued  ·  0 failed

    Every count comes from disk, not from a worker reporting in: the builders are subprocesses in
    other interpreters, and one is an agent that rewrites its own output when a QC gate fails. The
    caller hands over the three sets each poll and this only renders them.

    Timings are therefore INFERRED, and the `~` on each estimate is not decoration. An object is
    timed from the first poll that saw it building, and its estimate is the average of the ones its
    branch has already finished. TRELLIS writes `<name>.glb` in one step, leaving nothing partial,
    so its objects step from queued to done with no visible building phase — the board shows what
    it can rather than inventing a name to put there.

    It redraws by moving the cursor up, which only reaches while the whole block is on screen. So
    it measures the window every frame and sheds detail — the airy spacing, then the queue, then
    the per-object rows — until the block fits. A taller board that scrolls is worse than a short
    one that holds still. Degrades to one line per change when stdout is not a terminal.
    """

    # Richest first; the first layout that fits the window wins. Each step gives up the least
    # useful thing left: the airy spacing, then the queue, then per-object rows, then the blank
    # lines between sections, and finally the headings themselves.
    # (spaced, show queue, cap on per-object rows, no blank lines, no headings)
    TIERS = (
        (True,  True,  None, False, False),
        (False, True,  None, False, False),
        (False, True,  6,    False, False),
        (False, False, 6,    False, False),
        (False, False, 3,    False, False),
        (False, False, 0,    False, False),
        (False, False, 0,    True,  False),
        (False, False, 0,    True,  True),
    )

    # Below this the per-object rows lose their activity phrase rather than clipping the estimate
    # off the right-hand end — "wardrobe_02 … 1m42s ~40s left" is the useful part of that line.
    PHRASE_MIN_WIDTH = 76

    def __init__(self, branches: list[str], total: int, workers=None,
                 title: str = "Obj Reconstruction"):
        self.branches = branches
        self.total = total
        self.workers = workers or {}  # {branch: worker count}
        self.title = title
        self._drawn = 0
        self._t0 = time.time()
        self._started_at: dict[str, float] = {}       # "branch/object" -> first seen building
        self._durations: dict[str, list[float]] = {}  # branch -> how long its objects took
        self._finished: set[str] = set()
        self._completions: list[float] = []           # when each object landed, for the rate
        self._primed = False
        self._last_summary = ""

    # --- what the board has learned by watching ------------------------------ #
    def _observe(self, state: dict) -> None:
        now = time.time()
        for branch in self.branches:
            item_state = state.get(branch) or {}
            for name in item_state.get("active") or ():
                self._started_at.setdefault(f"{branch}/{name}", now)
            for name in item_state.get("built") or ():
                key = f"{branch}/{name}"
                if key in self._finished:
                    continue
                self._finished.add(key)
                if not self._primed:
                    continue  # a resumed run starts with objects already on disk: not our rate
                self._completions.append(now)
                start = self._started_at.get(key)
                if start:
                    self._durations.setdefault(branch, []).append(now - start)
        self._primed = True

    # Below a second an estimate is a number-shaped nothing — "~0s left" next to a spinner reads
    # as a stall. Say nothing instead.
    ETA_FLOOR = 1.0

    def _overall_eta(self, done: int) -> float:
        """Seconds left, from the rate THIS session has actually achieved. 0 when unknowable."""
        remaining = self.total - done
        elapsed = time.time() - self._t0
        if remaining <= 0 or len(self._completions) < 2 or elapsed <= 0:
            return 0.0
        eta = remaining * elapsed / len(self._completions)
        return eta if eta >= self.ETA_FLOOR else 0.0

    def _item_eta(self, branch: str, elapsed: float) -> float:
        seen = self._durations.get(branch)
        if not seen:
            return 0.0
        eta = sum(seen) / len(seen) - elapsed
        return eta if eta >= self.ETA_FLOOR else 0.0

    @staticmethod
    def _hue(branch: str) -> str:
        return STAGES.get(_BRANCH_STAGE.get(branch, branch), (None, "grey"))[1]

    @staticmethod
    def _names(state: dict, branch: str, key: str) -> list:
        return list((state.get(branch) or {}).get(key) or ())

    # --- composition ---------------------------------------------------------- #
    def _compose(self, state: dict, width: int, *, airy: bool, queue: bool, cap,
                 tight: bool, dense: bool) -> list:
        now = time.time()
        done = sum(len(self._names(state, b, "built")) for b in self.branches)
        eta = self._overall_eta(done)
        blank: list = []
        lines: list = []

        def gap() -> None:
            if not (tight or dense):
                lines.append(blank)

        def spaced() -> None:
            if airy:
                lines.append(blank)

        stamp = f"{fmt_clock(now - self._t0)} elapsed"
        title = f"LiteReality-Agent {self.title}"
        lines.append([(title, "bold"), (" " * max(1, width - len(title) - len(stamp)), ""),
                      (stamp, "dim")])
        gap()

        pct = int(100 * done / self.total) if self.total else 0
        lines.append([("Overall   ", ""), (bar(done, self.total, 24), "cyan"),
                      (f"  {done} / {self.total}   {pct}%", "bold")])
        if eta and not dense:
            lines.append([(f"          ~{fmt_rough(eta)} remaining", "dim")])
        gap()

        active = [(b, name) for b in self.branches for name in self._names(state, b, "active")]
        if active and cap != 0:
            shown = active if cap is None else active[:cap]
            lines.append([("Generating now", "bold")])
            spaced()
            for branch, name in shown:
                lines.append(self._active_row(state, branch, name, now, width))
                spaced()
            if len(active) > len(shown):
                lines.append([(f"    … +{len(active) - len(shown)} more building", "dim")])
            if not airy:
                gap()

        if not dense:
            lines.append([("Progress", "bold")])
            spaced()
        for branch in self.branches:
            lines.append(self._progress_row(state, branch))
        gap()

        if queue:
            waiting = [(b, names) for b in self.branches
                       if (names := self._names(state, b, "queued"))]
            if waiting:
                lines.append([("Queued", "bold")])
                spaced()
                for branch, names in waiting:
                    lines.append([(f"  {BRANCH_LABEL.get(branch, branch):<15}", self._hue(branch)),
                                  (self._queue_text(names), "dim")])
                gap()

        lines.append([("─" * width, "dim")])
        footer = self._footer(state, done)
        if eta and dense:  # nowhere else left for it, and it is the reason anyone is watching
            footer.append((f"  ·  ~{fmt_rough(eta)} left", "dim"))
        lines.append(footer)
        if eta and not dense:
            lines.append([(f"Estimated completion: ~{fmt_rough(eta)}", "dim")])
        return lines

    def _active_row(self, state: dict, branch: str, name: str, now: float, width: int) -> list:
        started = self._started_at.get(f"{branch}/{name}")
        elapsed = now - started if started else 0.0
        eta = self._item_eta(branch, elapsed)
        row = [("  ◉ ", self._hue(branch)),
               (f"{BRANCH_LABEL.get(branch, branch):<15}", self._hue(branch)),
               (f"{name:<18}", "")]
        if width >= self.PHRASE_MIN_WIDTH:
            # What THIS object is doing, when the caller could work it out; the branch's generic
            # verb only as a fallback. A caption that cannot change is not status.
            phase = ((state.get(branch) or {}).get("activity") or {}).get(name)
            row.append((f"{phase or BRANCH_ACTIVITY.get(branch, 'building'):<22}", "dim"))
        row.append((f"{fmt_elapsed(elapsed):>6}", ""))
        row.append((f"   ~{fmt_rough(eta)} left" if eta else "", "dim"))
        return row

    def _progress_row(self, state: dict, branch: str) -> list:
        item_state = state.get(branch) or {}
        built = len(self._names(state, branch, "built"))
        total = item_state.get("total", 0)
        # A full bar with the clock still running reads as a hang. It is not: the count is GLBs on
        # disk, and the articulated agent writes one, probe-checks it, renders it for a VLM
        # completeness check, and rewrites it if either gate fails. The file exists long before the
        # object is accepted, so say which of the two we are looking at.
        note, hue = "", "dim"
        if item_state.get("finished"):
            note = "   ✓ done" if built >= total else f"   ✗ {item_state.get('result', 'incomplete')}"
            hue = "green" if built >= total else "red"
        elif item_state.get("started") and total and built >= total:
            note, hue = "   · verifying", "yellow"
        elif self.workers.get(branch, 1) > 1:
            # Why one branch outruns another: eight objects at a time, not one.
            note = f"   · {self.workers[branch]} workers"
        return [(f"  {BRANCH_LABEL.get(branch, branch):<15}", self._hue(branch)),
                (bar(built, total, 10), self._hue(branch)),
                (f"   {built} / {total}", ""), (note, hue)]

    @staticmethod
    def _queue_text(names: list, show: int = 3) -> str:
        head = " · ".join(names[:show])
        return f"{head} · +{len(names) - show}" if len(names) > show else head

    def _footer(self, state: dict, done: int) -> list:
        active = sum(len(self._names(state, b, "active")) for b in self.branches)
        queued = sum(len(self._names(state, b, "queued")) for b in self.branches)
        # Only a FINISHED branch can have failures — an object still queued has not failed yet.
        failed = sum(max(0, (state.get(b) or {}).get("total", 0) - len(self._names(state, b, "built")))
                     for b in self.branches if (state.get(b) or {}).get("finished"))
        return [(f"{active} active  ·  {done} completed  ·  {queued} queued  ·  ", ""),
                (f"{failed} failed", "red" if failed else "dim")]

    # --- drawing --------------------------------------------------------------- #
    def _fit(self, state: dict, width: int, height: int) -> list:
        lines: list = []
        for airy, queue, cap, tight, dense in self.TIERS:
            lines = self._compose(state, width, airy=airy, queue=queue, cap=cap,
                                  tight=tight, dense=dense)
            if len(lines) <= height:
                return lines
        # Even the densest layout overflows. Keep the TAIL: the counts and the estimate are what
        # a window this short still has room to be useful for — trimming the other end would
        # spend the last few lines on a title.
        return lines[-max(1, height):]

    def render(self, state: dict) -> None:
        """`state[branch]` = {total, built, active, queued, started, finished, result}."""
        if not enabled():
            return
        self._observe(state)
        out = _out()
        if not use_colour():  # a log gets one line per change, never a redrawn frame
            line = self._summary_line(state)
            if line != self._last_summary:
                print(line, file=out, flush=True)
                self._last_summary = line
            return

        size = shutil.get_terminal_size((100, 24))
        lines = self._fit(state, size.columns, max(1, size.lines - 1))
        if self._drawn:
            print(f"\033[{self._drawn}A", end="", file=out)
        painted = 0
        for parts in lines:
            print(f"\033[K{_paint(parts, size.columns)}", file=out)
            painted += 1
        while painted < self._drawn:  # a shorter tier than last frame: wipe what it left behind
            print("\033[K", file=out)
            painted += 1
        self._drawn = painted
        out.flush()

    def _summary_line(self, state: dict) -> str:
        done = sum(len(self._names(state, b, "built")) for b in self.branches)
        per = "  ·  ".join(
            f"{BRANCH_LABEL.get(b, b)} {len(self._names(state, b, 'built'))}"
            f"/{(state.get(b) or {}).get('total', 0)}"
            for b in self.branches
        )
        return f"   reconstruction {done}/{self.total} built  ·  {per}"

    def finish(self, state: dict) -> None:
        self.render(state)
        self._drawn = 0


def run_logged(cmd, log_path, label: str, hue: str = "cyan", **kwargs):
    """Run a subprocess with its output in a log, and a live status line on screen.

    For the Blender calls: they emit hundreds of glTF/USD INFO lines that bury whatever else is
    on screen, but going silent for a minute is worse. The output goes to `log_path`, the newest
    meaningful line drives a status that redraws in place, and the caller prints the result row.

    Returns the CompletedProcess. The child is forced unbuffered — a redirected Python child
    block-buffers, so its log stays empty until it exits and the status never moves.
    """
    import subprocess
    import threading

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()

    def tail():
        stream = _StageStream(open(os.devnull, "w"), label, hue, _out())
        while not stop.wait(0.4):
            try:
                lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for raw in reversed(lines[-40:]):
                line = _clean(raw)
                if line and not _NOISE.match(line):
                    stream._status(line)
                    break
        stream.clear()

    watcher = threading.Thread(target=tail, daemon=True)
    if use_colour() and enabled():
        watcher.start()
    env = dict(kwargs.pop("env", None) or os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    try:
        with open(log_path, "a", encoding="utf-8", errors="replace") as handle:
            return subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, env=env, **kwargs)
    finally:
        stop.set()
        if watcher.is_alive():
            watcher.join(timeout=1.0)


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"
