"""Work that appears after the builder started still gets built.

The openings pass reads its job list from `opening_refs/*/reference_1024.png`, which the reference
stage is still writing. On a seven-opening room the last two references landed 27s and 49s after
this process listed its work — and because the list was a single startup snapshot, those two
openings were never built, never reported, and never counted as failed. The board showed them
queued forever behind five workers that had free slots.

A worker finishing is the natural moment to look again, so that is when the list is re-read.
"""

from __future__ import annotations

import asyncio

import pytest

from lrauthor.models.object_generation.generate import Job, run_all


def _job(name: str) -> Job:
    return Job(scan="Sim", name=name, category="door", image=None, glb=None, dims="")


class Args:
    concurrency = 3


@pytest.fixture
def runner(monkeypatch):
    """Drive run_all with a recording stand-in for `process`, so nothing spawns an agent."""
    started: list[str] = []

    async def fake_process(job, sem, args):
        async with sem:
            started.append(job.name)
            await asyncio.sleep(0)  # yield, so the dispatcher gets to look again
            job.status = "ok"

    monkeypatch.setattr("lrauthor.models.object_generation.generate.process",
                        fake_process)
    return started


def test_a_job_that_appears_after_the_start_is_picked_up(runner):
    """The bug: `reference_1024.png` for two openings was written 27s and 49s too late."""
    visible = [_job("Wall1_Door_1"), _job("Wall2_Door_4")]
    late = [_job("Wall9_Door_3"), _job("Wall14_Door_0")]

    def collect():
        if len(runner) >= 1:      # the reference stage finishes while the first object builds
            return visible + late
        return list(visible)

    everything = asyncio.run(run_all(list(visible), collect, asyncio.Semaphore(3), Args()))

    assert sorted(runner) == ["Wall14_Door_0", "Wall1_Door_1", "Wall2_Door_4", "Wall9_Door_3"]
    assert [j.name for j in everything][-2:] == ["Wall9_Door_3", "Wall14_Door_0"]
    assert all(j.status == "ok" for j in everything), "a late job must be reported like any other"


def test_a_job_is_never_dispatched_twice(runner):
    """`collect()` keeps returning finished work — it reads a directory, not a queue."""
    jobs = [_job("Wall1_Door_1"), _job("Wall2_Door_4")]

    asyncio.run(run_all(list(jobs), lambda: list(jobs), asyncio.Semaphore(3), Args()))

    assert sorted(runner) == ["Wall1_Door_1", "Wall2_Door_4"]


def test_the_concurrency_limit_still_holds_for_late_arrivals(monkeypatch):
    """Late work must queue behind the semaphore, not slip past it — each job is an agent
    session and a Blender process, and the cap is what stops them thrashing."""
    peak, live = 0, 0

    async def fake_process(job, sem, args):
        nonlocal peak, live
        async with sem:
            live += 1
            peak = max(peak, live)
            await asyncio.sleep(0.01)
            live -= 1
            job.status = "ok"

    monkeypatch.setattr("lrauthor.models.object_generation.generate.process",
                        fake_process)
    first = [_job(f"o{i}") for i in range(2)]
    everything = [*first, *(_job(f"late{i}") for i in range(6))]

    asyncio.run(run_all(list(first), lambda: list(everything), asyncio.Semaphore(2), Args()))

    assert peak <= 2, f"{peak} jobs ran at once against a limit of 2"


def test_a_crashing_job_is_not_swallowed(monkeypatch):
    """`asyncio.wait` keeps an exception inside its task until someone asks for the result."""
    async def boom(job, sem, args):
        raise RuntimeError("blender died")

    monkeypatch.setattr("lrauthor.models.object_generation.generate.process", boom)

    with pytest.raises(RuntimeError, match="blender died"):
        asyncio.run(run_all([_job("o0")], list, asyncio.Semaphore(2), Args()))
