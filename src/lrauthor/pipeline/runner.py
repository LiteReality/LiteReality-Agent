"""Dependency-aware orchestration for the complete reconstruction workflow."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable
from typing import Any
from unittest.mock import patch

from lrauthor.pipeline.context import RunContext
from lrauthor.pipeline.result import StageResult, StageStatus
from lrauthor.pipeline.stage import Stage
from lrauthor.pipeline.stages import STAGES

PIPELINE = tuple(STAGES)


class PipelineError(RuntimeError):
    pass


class PipelineRunner:
    def __init__(self, stages: Iterable[Stage] = PIPELINE):
        self.stages = tuple(stages)
        self.by_name = {stage.name: stage for stage in self.stages}
        if len(self.by_name) != len(self.stages):
            raise ValueError("pipeline stage names must be unique")

    def _read_state(self, context: RunContext) -> dict[str, Any]:
        try:
            return json.loads(context.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"version": 1, "scan": context.scan, "stages": {}}

    def _write_state(self, context: RunContext, state: dict[str, Any]) -> None:
        context.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = context.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(context.state_path)

    def _selection(self, start: str | None, through: str | None) -> tuple[Stage, ...]:
        names = [stage.name for stage in self.stages]
        if start and start not in self.by_name:
            raise PipelineError(f"unknown stage {start!r}; choose from {', '.join(names)}")
        if through and through not in self.by_name:
            raise PipelineError(f"unknown stage {through!r}; choose from {', '.join(names)}")
        left = names.index(start) if start else 0
        right = names.index(through) + 1 if through else len(names)
        if left >= right:
            raise PipelineError("--from must occur before --through")
        return self.stages[left:right]

    def run(
        self,
        context: RunContext,
        *,
        start: str | None = None,
        through: str | None = None,
        force: set[str] | None = None,
        strict: bool = False,
        options: dict[str, dict[str, Any]] | None = None,
    ) -> list[StageResult]:
        selected = self._selection(start, through)
        state = self._read_state(context)
        prior = state.setdefault("stages", {})
        force = force or set()
        unknown_force = force - self.by_name.keys()
        if unknown_force:
            raise PipelineError(f"unknown forced stage(s): {', '.join(sorted(unknown_force))}")

        forced_indexes = [i for i, stage in enumerate(self.stages) if stage.name in force]
        invalidate_from = min(forced_indexes) if forced_indexes else len(self.stages)
        for stage in self.stages[invalidate_from:]:
            prior.pop(stage.name, None)

        results: list[StageResult] = []
        completed = {
            name
            for name, value in prior.items()
            if value.get("status") in {StageStatus.COMPLETED.value, StageStatus.REUSED.value}
        }
        for stage in selected:
            for prerequisite in stage.prerequisites:
                dependency = self.by_name[prerequisite]
                if (
                    prerequisite not in completed
                    and dependency.is_complete is not None
                    and dependency.is_complete(context)
                ):
                    reused = StageResult(prerequisite, StageStatus.REUSED)
                    prior[prerequisite] = reused.to_dict()
                    completed.add(prerequisite)
            missing = [name for name in stage.prerequisites if name not in completed]
            if missing:
                raise PipelineError(
                    f"stage {stage.name!r} requires {', '.join(missing)}; "
                    "run the prerequisites or start earlier"
                )

            old = prior.get(stage.name)
            complete_on_disk = stage.is_complete(context) if stage.is_complete else False
            if stage.name not in force and old and complete_on_disk:
                result = StageResult.from_dict(old)
                result.status = StageStatus.REUSED
            else:
                started = time.monotonic()
                try:
                    stage_options = dict((options or {}).get(stage.name, {}))
                    if stage.name in force:
                        stage_options.setdefault("force", True)
                    # Several ported stage implementations still read canonical environment
                    # names. Keep that compatibility bridge at the orchestration boundary and
                    # restore the caller's process immediately after the stage returns.
                    with patch.dict(os.environ, context.environment, clear=True):
                        result = stage.run(context, stage_options)
                except Exception as exc:  # one normalization point for stage failures
                    result = StageResult(stage.name, StageStatus.FAILED, error=str(exc))
                result.duration_seconds = time.monotonic() - started

            results.append(result)
            if not result.ok:
                result.details["fatal"] = bool(stage.required or strict)
            prior[stage.name] = result.to_dict()
            self._write_state(context, state)
            if result.ok:
                completed.add(stage.name)
                continue
            if stage.required or strict:
                break
        return results

    def run_stage(
        self,
        context: RunContext,
        name: str,
        *,
        force: bool = False,
        strict: bool = False,
        options: dict[str, Any] | None = None,
    ) -> StageResult:
        if name not in self.by_name:
            raise PipelineError(f"unknown stage {name!r}")
        results = self.run(
            context,
            start=name,
            through=name,
            force={name} if force else set(),
            strict=strict,
            options={name: options or {}},
        )
        return results[0]
