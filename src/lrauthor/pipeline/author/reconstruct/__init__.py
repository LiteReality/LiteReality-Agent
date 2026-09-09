"""Route and reconstruct every detected scene object."""

from lrauthor.pipeline.context import RunContext
from lrauthor.pipeline.result import StageResult
from lrauthor.pipeline.support import command_result, run_module


def complete(context: RunContext) -> bool:
    root = context.object_root / "reconstructed_objs"
    return root.is_dir() and bool(list(root.glob("*.glb")) + list(root.glob("*/*.glb")))


def run(context: RunContext, options: dict) -> StageResult:
    args: list[object] = [
        "--scan", context.capture_dir, "--name", context.scan,
        "--output-root", context.output_root,
        "--skip-extract", "--skip-crop", "--skip-references",
        "--classify", "--reconstruct",
        "--procedural", "--build-openings",
        "--agent-model", context.settings.procedural_model,
    ]
    if options.get("chair_qc"):
        args.append("--chair-qc")
    if options.get("force"):
        args.append("--force-reconstruct")
    if options.get("concurrency"):
        args.extend(("--concurrency", options["concurrency"]))
    rc, log = run_module(
        context, "lrauthor.pipeline.measure.flow", args, log_name="reconstruct"
    )
    return command_result(
        "reconstruct", rc,
        artifacts={"objects": context.object_root / "reconstructed_objs"}, log=log,
    )
