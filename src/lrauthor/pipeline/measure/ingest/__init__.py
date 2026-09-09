"""Capture ingestion: extraction, crops, detection, and reference evidence."""

from lrauthor.pipeline.context import RunContext
from lrauthor.pipeline.result import StageResult
from lrauthor.pipeline.support import command_result, run_module


def complete(context: RunContext) -> bool:
    root = context.object_root / "object_init"
    return (root / "object_init_summary.json").is_file() and (root / "object_refs" / context.scan).is_dir()


def run(context: RunContext, options: dict) -> StageResult:
    args: list[object] = [
        "--scan", context.capture_dir, "--name", context.scan,
        "--output-root", context.output_root,
    ]
    if options.get("skip_image_generation"):
        args.append("--skip-image-generation")
    if options.get("force"):
        args.extend(("--force-extract", "--force-crop", "--force-image-generation"))
    rc, log = run_module(
        context, "lrauthor.pipeline.measure.flow", args, log_name="ingest"
    )
    return command_result(
        "ingest", rc,
        artifacts={"object_work": context.object_root / "object_init"}, log=log,
    )
