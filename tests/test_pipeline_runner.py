from __future__ import annotations

import os
from pathlib import Path

from lrauthor.pipeline.context import RunContext
from lrauthor.pipeline.result import StageResult, StageStatus
from lrauthor.pipeline.runner import PipelineRunner
from lrauthor.pipeline.stage import Stage
from lrauthor.settings import LiteRealitySettings


def context(tmp_path: Path) -> RunContext:
    settings = LiteRealitySettings(repo_root=tmp_path, output_root=tmp_path / "run")
    return RunContext("scan", tmp_path / "capture", tmp_path / "run" / "scan", tmp_path / "run", settings=settings)


def result(name: str, status: StageStatus = StageStatus.COMPLETED) -> StageResult:
    return StageResult(name, status, artifacts={"marker": __file__})


def test_optional_failure_does_not_stop_later_stages(tmp_path):
    called = []
    stages = (
        Stage("required", lambda *_: called.append("required") or result("required")),
        Stage(
            "optional",
            lambda *_: called.append("optional") or result("optional", StageStatus.FAILED),
            ("required",),
            required=False,
        ),
        Stage("publish", lambda *_: called.append("publish") or result("publish"), ("required",)),
    )
    results = PipelineRunner(stages).run(context(tmp_path))
    assert called == ["required", "optional", "publish"]
    assert results[1].details["fatal"] is False


def test_strict_promotes_optional_failure_and_stops(tmp_path):
    called = []
    stages = (
        Stage("required", lambda *_: result("required")),
        Stage(
            "optional",
            lambda *_: result("optional", StageStatus.FAILED),
            ("required",),
            required=False,
        ),
        Stage("publish", lambda *_: called.append("publish") or result("publish"), ("required",)),
    )
    results = PipelineRunner(stages).run(context(tmp_path), strict=True)
    assert [item.stage for item in results] == ["required", "optional"]
    assert results[-1].details["fatal"] is True
    assert called == []


def test_completed_stage_is_reused_when_artifacts_remain(tmp_path):
    calls = []
    stage = Stage(
        "only",
        lambda *_: calls.append(1) or result("only"),
        is_complete=lambda _: True,
    )
    runner = PipelineRunner((stage,))
    first = runner.run(context(tmp_path))[0]
    second = runner.run(context(tmp_path))[0]
    assert first.status is StageStatus.COMPLETED
    assert second.status is StageStatus.REUSED
    assert calls == [1]


def test_force_is_forwarded_to_stage_and_invalidates_dependents(tmp_path):
    received = []
    stages = (
        Stage("first", lambda _, options: received.append(options) or result("first")),
        Stage("second", lambda *_: result("second"), ("first",)),
    )
    runner = PipelineRunner(stages)
    runner.run(context(tmp_path))
    runner.run(context(tmp_path), force={"first"})
    assert received == [{}, {"force": True}]


def test_stage_receives_context_settings_without_leaking_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("LITEREALITY_OUTPUT", "callers-value")
    observed = []
    stage = Stage(
        "only",
        lambda *_: observed.append(os.environ["LITEREALITY_OUTPUT"]) or result("only"),
    )
    PipelineRunner((stage,)).run(context(tmp_path))
    assert observed == [str(tmp_path / "run")]
    assert os.environ["LITEREALITY_OUTPUT"] == "callers-value"


def test_single_stage_discovers_existing_prerequisite_artifacts(tmp_path):
    called = []
    stages = (
        Stage("seed", lambda *_: result("seed"), is_complete=lambda _: True),
        Stage("publish", lambda *_: called.append("publish") or result("publish"), ("seed",)),
    )
    output = PipelineRunner(stages).run_stage(context(tmp_path), "publish")
    assert output.status is StageStatus.COMPLETED
    assert called == ["publish"]
