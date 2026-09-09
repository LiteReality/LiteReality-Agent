"""Assemble reconstructed assets into the editable seed scene."""

from pathlib import Path

from lrauthor.pipeline.context import RunContext
from lrauthor.pipeline.result import StageResult, StageStatus


def complete(context: RunContext) -> bool:
    return (context.seed_room / "Room.py").is_file() and (context.scene_dir / "scene.json").is_file()


def run(context: RunContext, options: dict) -> StageResult:
    from lrauthor.pipeline.measure import paths
    from lrauthor.pipeline.measure.artifacts import ensure_stage_links, finalize
    from lrauthor.room_ops import compile_room, export_scene

    ensure_stage_links(context.scan)
    paths.set_scan(context.scan)
    room = export_scene(context.scan)
    if room is None or not (Path(room) / "Room.py").is_file():
        return StageResult("seed", StageStatus.FAILED, error="seed export did not produce Room.py")
    artifacts = {"room": str(room)}
    warnings: list[str] = []
    if options.get("preview", True):
        try:
            preview = compile_room(Path(room), bake=True)
            if preview:
                artifacts["preview"] = str(preview)
        except Exception as exc:
            warnings.append(f"seed preview unavailable: {exc}")
    package = finalize(context.scan, capture_mode=options.get("capture_mode", "link"))
    artifacts["manifest"] = str(package.manifest)
    return StageResult("seed", StageStatus.COMPLETED, artifacts=artifacts, warnings=warnings)
