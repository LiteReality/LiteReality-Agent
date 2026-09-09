"""The procedural builder's QC gates must not stall the event loop.

`process()` is async and several jobs run under one semaphore, but its gates — `verify_reasons`,
`verify` (probe_glb) and `completeness_report` — are blocking `subprocess.run` calls, and the last
one is a whole agent session. Called directly they hold the event loop, so while ONE object is
being QC'd every OTHER object's agent session is frozen: `--concurrency N` buys nothing during
precisely the phase that dominates the stage. On a two-opening room that turned ~2 concurrent
builds into a serial chain and the reconstruct stage ran for 38 minutes.

This is invisible by inspection — the code reads as concurrent and the output is identical, only
the wall time differs — so it is pinned here: two jobs whose gates each sleep must finish in about
ONE gate's time, not two.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

from lrauthor.models.object_generation import generate

GATE_SECONDS = 0.4


def _job(name: str) -> generate.Job:
    return generate.Job(
        scan="test-scan", name=name, category="door",
        image=Path(f"/nonexistent/{name}.png"), glb=Path(f"/nonexistent/{name}/{name}.glb"),
        dims="1.00 (X) x 0.10 (Y) x 2.00 (Z)",
    )


def _args():
    return SimpleNamespace(
        skip_existing=True, retries=0, model=None, max_turns=1,
        max_budget_usd=None, provider=None, no_completeness=False,
    )


def test_qc_gates_do_not_block_other_jobs(monkeypatch, tmp_path):
    """Two jobs, each with a blocking gate: wall time must be ~1 gate, not ~2."""

    def slow_blocking_gate(job):
        time.sleep(GATE_SECONDS)  # stands in for probe_glb / completeness_check subprocesses
        return {"pass": True, "violations": [], "missing": []}

    # Force the build path (not the "already built" skip), and make the agent run a no-op so the
    # gates are the only thing on the clock.
    monkeypatch.setattr(generate, "verify_reasons", lambda job: ["forced rebuild"])
    monkeypatch.setattr(generate, "probe_glb", slow_blocking_gate)
    monkeypatch.setattr(generate, "completeness_report", slow_blocking_gate)
    monkeypatch.setattr(generate, "verify", lambda job: bool(slow_blocking_gate(job)))
    monkeypatch.setattr(generate, "drop_build_recipe", lambda d: [])

    async def noop_run_one(job, args, feedback=""):
        return None

    monkeypatch.setattr(generate, "run_one", noop_run_one)

    jobs = [_job("Wall1_Door_0"), _job("Wall5_Window_0")]
    args = _args()

    async def drive():
        sem = asyncio.Semaphore(len(jobs))
        started = time.monotonic()
        await asyncio.gather(*(generate.process(j, sem, args) for j in jobs))
        return time.monotonic() - started

    elapsed = asyncio.run(drive())

    assert all(j.status == "ok" for j in jobs), [j.status for j in jobs]
    # Each job runs two gates (verify + completeness) = 2 * GATE_SECONDS of blocking work. Run
    # concurrently that is ~2 gates of wall time; serialised on a held event loop it is ~4. The
    # midpoint is a wide enough bar to survive a slow CI box and still fail a reverted to_thread.
    serial_floor = 4 * GATE_SECONDS
    assert elapsed < serial_floor * 0.75, (
        f"QC gates serialised the jobs: {elapsed:.2f}s for 2 jobs "
        f"(serial would be ~{serial_floor:.2f}s) — did a gate lose its asyncio.to_thread?"
    )
