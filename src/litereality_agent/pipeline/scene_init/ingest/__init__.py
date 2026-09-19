"""Capture ingestion: extraction, crops, detection, and reference evidence."""

from litereality_agent.pipeline.context import RunContext
from litereality_agent.pipeline.result import StageResult
from litereality_agent.pipeline.support import command_result, run_module


def complete(context: RunContext) -> bool:
    root = context.object_root / "object_init"
    if not ((root / "object_init_summary.json").is_file()
            and (root / "object_refs" / context.scan).is_dir()):
        return False
    from litereality_agent.pipeline.scene_init.layout.validation import inspect_layout

    from . import merge_boxes

    scene = root / "input" / "scene_data" / context.scan
    return ((scene / "layout_baseline.json").is_file()
            and merge_boxes.policy_current(scene) and inspect_layout(scene)["passed"])


def run(context: RunContext, options: dict) -> StageResult:
    args: list[object] = [
        "--scan", context.capture_dir, "--name", context.scan,
        "--output-root", context.output_root,
    ]
    if options.get("skip_image_generation"):
        args.append("--skip-image-generation")
    if options.get("use_dino"):
        args.append("--use-dino")
    if options.get("force"):
        args.extend(("--force-extract", "--force-crop", "--force-image-generation"))
    rc, log = run_module(
        context, "litereality_agent.pipeline.scene_init.flow", args, log_name="ingest"
    )
    return command_result(
        "ingest", rc,
        artifacts={"object_work": context.object_root / "object_init"}, log=log,
    )
