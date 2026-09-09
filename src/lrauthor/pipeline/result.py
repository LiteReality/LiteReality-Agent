"""Stable result types shared by all pipeline stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class StageStatus(str, Enum):
    COMPLETED = "completed"
    REUSED = "reused"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(slots=True)
class StageResult:
    stage: str
    status: StageStatus
    duration_seconds: float = 0.0
    artifacts: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in {StageStatus.COMPLETED, StageStatus.REUSED, StageStatus.SKIPPED}

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StageResult":
        return cls(
            stage=str(data["stage"]),
            status=StageStatus(data["status"]),
            duration_seconds=float(data.get("duration_seconds", 0.0)),
            artifacts={str(k): str(v) for k, v in data.get("artifacts", {}).items()},
            warnings=[str(item) for item in data.get("warnings", [])],
            error=data.get("error"),
            details=dict(data.get("details", {})),
        )

    def artifacts_exist(self) -> bool:
        return bool(self.artifacts) and all(Path(value).exists() for value in self.artifacts.values())
