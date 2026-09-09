"""Stage protocol and declarative metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from lrauthor.pipeline.context import RunContext
from lrauthor.pipeline.result import StageResult


class StageCallable(Protocol):
    def __call__(self, context: RunContext, options: dict[str, Any]) -> StageResult: ...


@dataclass(frozen=True, slots=True)
class Stage:
    name: str
    run: StageCallable
    prerequisites: tuple[str, ...] = ()
    required: bool = True
    is_complete: Callable[[RunContext], bool] | None = None
