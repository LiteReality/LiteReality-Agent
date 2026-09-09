"""Lightweight, always-on process event log.

Pipeline boundaries and model calls are appended as one JSON line to

    <repo>/run/<scan>/traces/trace.jsonl

so a downstream viewer can reconstruct the whole timeline. Per-call payloads
(prompts, agent thoughts) can also be dropped under ``traces/images/`` and
``traces/agent/``.

The tracer is a process-global keyed to the *current scan*. Call :func:`start`
(the scene-initialization flow does this per scan) to point it at that scan's ``traces/`` dir; if it
was never started, every hook is a cheap no-op, so instrumented code never has to care whether
telemetry is active.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from lrauthor import REPO_ROOT, console

_lock = threading.Lock()
_state: dict = {"scan": None, "dir": None, "t0": None}


def start(scan: str, directory: str | Path | None = None) -> Path:
    """Begin (or resume) a trace for ``scan``; returns the traces directory."""
    if directory is None:
        root = Path(
            os.environ.get("LITEREALITY_FINAL")
            or os.environ.get("LITEREALITY_OUTPUT")
            or (REPO_ROOT / "run")
        )
        d = root / scan / "scene_init" / "obj_stage" / "traces"
    else:
        d = Path(directory)
    (d / "images").mkdir(parents=True, exist_ok=True)
    (d / "agent").mkdir(parents=True, exist_ok=True)
    _state.update(scan=scan, dir=d, t0=time.time())
    event("trace_start", scan=scan)
    return d


def active_dir() -> Path | None:
    return _state["dir"]


def event(kind: str, **data) -> None:
    """Append one event to ``trace.jsonl`` (no-op if telemetry was never started)."""
    d = _state["dir"]
    if d is None:
        return
    t0 = _state["t0"] or time.time()
    rec = {"t": round(time.time(), 3), "dt": round(time.time() - t0, 3), "kind": kind, **data}
    line = json.dumps(rec, default=str)
    with _lock:
        with (d / "trace.jsonl").open("a", encoding="utf-8") as f:
            f.write(line + "\n")


# ── convenience hooks ────────────────────────────────────────────────────────
def on_image_generation(
    prompt: str,
    sheet,
    out,
    status: str,
    *,
    model: str | None = None,
    elapsed_sec: float | None = None,
    attempt: int | None = None,
    usage: dict | None = None,
    error: str | None = None,
) -> None:
    """Record one hosted reference-image generation call."""
    out = Path(out)
    event(
        "image_generation",
        name=out.parent.name,
        status=status,
        model=model,
        sheet=str(sheet),
        out=str(out),
        prompt_chars=len(prompt or ""),
        prompt_preview=(prompt or "").strip().replace("\n", " ")[:240],
        elapsed_sec=elapsed_sec,
        attempt=attempt,
        usage=usage,
        error=error,
    )


def stage(name: str, scan: str, status: str = "start", **data) -> None:
    """Mark a pipeline stage boundary (extract / crop / object_refs / ...).

    Writes the trace line AND renders the terminal's stage row. Every stage in the pipeline
    already calls this, so it is the one place a progress display can hook without touching a
    single stage's control flow. The trace is the record; the display is presentation and is
    allowed to be absent ($LR_QUIET=1, or a non-tty losing its colour).
    """
    event("stage", stage=name, scan=scan, status=status, **data)
    console.stage_event(name, scan, status, data)


def agent_step(agent: str, step: str, *, payload: dict | None = None, **data) -> None:
    """Record an agent action (reserved for articulated-object generation tracing).

    A larger ``payload`` (the agent's reasoning, tool args) is written as a
    sidecar JSON under ``traces/agent/`` and referenced from the timeline line.
    """
    ref = None
    d = _state["dir"]
    if payload is not None and d is not None:
        n = sum(1 for _ in (d / "agent").glob("*.json"))
        ref = f"agent/{n:05d}_{agent}_{step}.json"
        (d / ref).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    event("agent_step", agent=agent, step=step, payload_ref=ref, **data)
