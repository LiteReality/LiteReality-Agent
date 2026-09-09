from __future__ import annotations

import codecs
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from lrauthor import console
from lrauthor.pipeline.context import RunContext
from lrauthor.pipeline.result import StageResult, StageStatus


def run_module(
    context: RunContext,
    module: str,
    args: Iterable[object] = (),
    *,
    log_name: str | None = None,
) -> tuple[int, Path | None]:
    command = [sys.executable, "-m", module, *(str(arg) for arg in args)]
    if not log_name:
        proc = subprocess.run(command, cwd=context.repo_root, env=context.environment)
        return proc.returncode, None

    log = context.authoring_root / "logs" / f"{log_name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    return _tee(command, context, log), log


def _tee(command: list[str], context: RunContext, log: Path) -> int:
    """Run `command`, sending its output to the terminal AND to `log`.

    Handing the child a file as its stdout — which is what this used to do — puts the entire stage
    display in the log and leaves the terminal blank for the minutes a stage takes. Every row
    `console` renders is written by these subprocesses, so that redirect was hiding the run. The
    output is piped instead, and forwarded to both.

    Four details keep the live display intact across the pipe. A Python child block-buffers when
    its stdout is not a tty, so it is forced unbuffered or nothing appears until it exits. It turns
    colour off for the same reason, so `$LR_COLOR` is set from OUR stdout rather than from its own
    — which also means `LR_COLOR=0` still gets you the plain one-line-per-change form, in both
    places at once. The forwarding is chunk-wise rather than line-wise: the stage rows, the status
    line and the branch board all redraw with carriage returns and cursor escapes, so reading by
    line would hold each frame back until some later newline.

    And the child cannot MEASURE a pipe, so `shutil.get_terminal_size` there falls back to a
    hardcoded 100 columns. On a narrower window every board row soft-wraps to two physical lines
    while the redraw still moves the cursor up by one line per ROW — it lands short, and the board
    scrolls down the screen instead of refreshing in place. Measure here, where the terminal is
    still a terminal, and hand the answer down.
    """
    env = dict(context.environment)
    env["PYTHONUNBUFFERED"] = "1"
    env["LR_COLOR"] = "1" if console.use_colour() else "0"
    size = shutil.get_terminal_size(fallback=(0, 0))
    if size.columns and size.lines:
        env["COLUMNS"], env["LINES"] = str(size.columns), str(size.lines)

    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    with log.open("w", encoding="utf-8", errors="replace") as stream:
        proc = subprocess.Popen(
            command, cwd=context.repo_root, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        try:
            while True:
                # os.read returns as soon as ANY bytes are there; a buffered `.read(n)` would sit
                # on a partial frame waiting for n of them.
                chunk = os.read(proc.stdout.fileno(), 65536)
                if not chunk:
                    break
                text = decoder.decode(chunk)
                sys.stdout.write(text)
                sys.stdout.flush()
                stream.write(console.plain(text))  # the log keeps the text, never the escapes
                stream.flush()  # so `tail -f` and a crashed run both see it
            stream.write(console.plain(decoder.decode(b"", final=True)))
        finally:
            proc.stdout.close()
            returncode = proc.wait()  # reap the child even when we are unwinding
    return returncode


def command_result(
    stage: str,
    rc: int,
    *,
    artifacts: dict[str, Path] | None = None,
    log: Path | None = None,
    allow_nonzero: bool = False,
) -> StageResult:
    paths = {name: str(path) for name, path in (artifacts or {}).items() if path.exists()}
    warnings = []
    if rc and allow_nonzero:
        warnings.append(f"command exited {rc}" + (f"; see {log}" if log else ""))
    return StageResult(
        stage,
        StageStatus.COMPLETED if rc == 0 or allow_nonzero else StageStatus.FAILED,
        artifacts=paths,
        warnings=warnings,
        error=None if rc == 0 or allow_nonzero else f"command exited {rc}" + (f"; see {log}" if log else ""),
    )
