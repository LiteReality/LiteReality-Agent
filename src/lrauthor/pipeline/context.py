"""Resolved, explicit context passed to every pipeline stage."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from lrauthor import REPO_ROOT
from lrauthor.settings import LiteRealitySettings, load_settings

SCAN_MARKERS = ("room.usdz", "roomplan/room.usdz")


def _looks_like_scan(path: Path) -> bool:
    return any((path / marker).is_file() for marker in SCAN_MARKERS) or bool(
        next(path.glob("frame_*.jpg"), None)
    )


def _looks_like_path(value: str | os.PathLike[str]) -> bool:
    text = os.fspath(value)
    return (
        Path(text).expanduser().is_absolute()
        or "/" in text
        or "\\" in text
        or text.startswith((".", "~"))
    )


@dataclass(frozen=True, slots=True)
class RunContext:
    scan: str
    capture_dir: Path
    scene_dir: Path
    output_root: Path
    repo_root: Path = REPO_ROOT
    settings: LiteRealitySettings = field(default_factory=load_settings)

    @classmethod
    def resolve(
        cls,
        target: str | os.PathLike[str],
        *,
        output_root: str | os.PathLike[str] | None = None,
        settings: LiteRealitySettings | None = None,
    ) -> "RunContext":
        """Resolve a scan name, capture directory, or existing scene package once.

        Environment variables are accepted at this composition boundary for
        compatibility. Stage modules receive paths from this object instead of
        deriving them again at import time.
        """
        settings = settings or load_settings()
        raw = Path(target).expanduser()
        out = Path(output_root or settings.resolved_output_root()).resolve()

        if raw.exists() and raw.is_dir():
            candidate = raw.resolve()
            if (candidate / "scene.json").is_file():
                scene = candidate
                scan = candidate.name
                capture = None
                try:
                    from lrauthor.room_ops.manifest import read

                    package = read(candidate)
                    scan = package.scan or scan
                    capture = package.capture
                except Exception:  # malformed packages are validated by the stage that consumes them
                    capture = None
                if not capture:
                    # A scene.json that is readable but names nothing — half-written by an
                    # interrupted run, or hand-edited — used to reach `Path(None)` and die with a
                    # TypeError mentioning neither the file nor the scan. Fall back to where the
                    # capture would live, and let the consuming stage report what is missing.
                    capture = settings.resolved_scans_dir() / scan
                return cls(scan, Path(capture).resolve(), scene, out, settings.repo_root, settings)
            if not _looks_like_scan(candidate):
                raise ValueError(f"{candidate} is neither a RoomPlan capture nor a scene package")
            scan, capture = candidate.name, candidate
        else:
            if _looks_like_path(target):
                raise ValueError(f"no such capture or scene path: {raw}")
            scan = str(target)
            capture = settings.resolved_scans_dir() / scan

        return cls(scan, capture.resolve(), out / scan, out, settings.repo_root, settings)

    @property
    def state_path(self) -> Path:
        return self.scene_dir / ".litereality" / "pipeline.json"

    @property
    def init_root(self) -> Path:
        return self.scene_dir / "scene_init"

    @property
    def object_root(self) -> Path:
        return self.init_root / "obj_stage"

    @property
    def seed_room(self) -> Path:
        return self.init_root / "scene_stage" / "room_init" / "room"

    @property
    def authoring_root(self) -> Path:
        return self.scene_dir / "realism_authoring"

    @property
    def authored_room(self) -> Path:
        return self.authoring_root / "room"

    @property
    def preview_dir(self) -> Path:
        return self.authoring_root / "room_preview"

    @property
    def environment(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update(self.settings.as_environment())
        env.update(
            {
                "LR_REPO_ROOT": str(self.repo_root),
                "LR_SCANS_DIR": str(self.capture_dir.parent),
                "LITEREALITY_SCAN": self.scan,
                "LITEREALITY_OUTPUT": str(self.output_root),
                "LITEREALITY_FINAL": str(self.output_root),
                "LR_SCENE": str(self.scene_dir),
                "LR_AUTHORING": str(self.authoring_root),
            }
        )
        return env
