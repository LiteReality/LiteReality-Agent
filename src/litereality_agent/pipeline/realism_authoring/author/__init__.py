"""Run Room.py authoring and its optional polish passes."""

import json
import shutil

from litereality_agent.pipeline.context import RunContext
from litereality_agent.pipeline.result import StageResult, StageStatus
from litereality_agent.pipeline.support import command_result, run_module


def complete(context: RunContext) -> bool:
    return (context.authored_room / "Room.py").is_file()


def run(context: RunContext, options: dict) -> StageResult:
    if not (context.seed_room / "Room.py").is_file():
        return StageResult("author", StageStatus.FAILED, error=f"missing seed room at {context.seed_room}")
    evidence = context.authoring_root / "surface_ref" / "surface_ref_manifest.json"
    if not evidence.is_file() or options.get("force"):
        evidence_rc, evidence_log = run_module(
            context,
            "litereality_agent.pipeline.realism_authoring.author.evidence",
            ["--scene", context.scene_dir],
            log_name="evidence",
        )
        if evidence_rc:
            return StageResult(
                "author",
                StageStatus.FAILED,
                error=f"scene evidence failed; see {evidence_log}",
            )
    if context.authored_room.exists():
        shutil.rmtree(context.authored_room)
    context.authored_room.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(context.seed_room, context.authored_room)
    args: list[object] = [
        "--scene", context.scene_dir,
        "--room", context.authored_room,
        "--surface-ref", context.authoring_root / "surface_ref",
        "--scan", context.capture_dir,
        "--profile", options.get("profile", "base"),
        "--max-turns", options.get("max_turns", 140),
        "--step-budget", options.get("step_budget", 100),
    ]
    rc, log = run_module(
        context,
        "litereality_agent.pipeline.realism_authoring.author.entrypoint",
        args,
        log_name="author",
    )
    result = command_result(
        "author", rc, artifacts={"room": context.authored_room}, log=log,
    )
    if rc:
        return result

    # A ROOM THAT RAN OUT OF STEPS IS NOT A FINISHED ROOM. The step budget lands the session
    # gracefully and exits 0, which is right — hitting it means "time's up", not "this is broken",
    # and the work so far is kept. But it exits 0 through the same path as a room the model
    # considered done, so without this the two are indistinguishable from outside and a truncated
    # room is reported as a completed stage.
    try:
        summary = json.loads(
            (context.authored_room.parent / ".author_result.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        summary = {}
    if summary.get("ended_early"):
        result.warnings.append(
            f"authoring stopped on {summary['ended_early']} after "
            f"{summary.get('calls', '?')} tool-calls (budget "
            f"{summary.get('step_budget', '?')}) — the room is as far as it got, not finished. "
            f"Raise it with --author-steps."
        )

    passes: list[tuple[str, str, list[object]]] = []
    if options.get("refine_objects"):
        refine_args: list[object] = [
            "--scene", context.scene_dir,
            "--room", context.authored_room,
            "--refroot", context.object_root / "object_init",
            "--scan", context.capture_dir,
            "--results", context.authoring_root / "obj_refine",
            "--concurrency", options.get("refine_concurrency", 2),
            "--budget", options.get("refine_budget", 8),
        ]
        if options.get("objects"):
            refine_args.extend(("--objects", options["objects"]))
        passes.append(
            (
                "object refinement",
                "litereality_agent.pipeline.realism_authoring.author.refine_objects",
                refine_args,
            )
        )
    if options.get("materials"):
        passes.append(
            (
                "materials",
                "litereality_agent.pipeline.realism_authoring.author.materials",
                [
                    "--scene", context.scene_dir,
                    "--room", context.authored_room,
                    "--surface-ref", context.authoring_root / "surface_ref",
                    "--scan", context.capture_dir,
                    "--refroot", context.object_root / "object_init",
                ],
            )
        )
    if options.get("quality_pass"):
        passes.append(
            (
                "model quality",
                "litereality_agent.pipeline.realism_authoring.author.quality",
                [
                    "--scene", context.scene_dir,
                    "--room", context.authored_room,
                    "--surface-ref", context.authoring_root / "surface_ref",
                    "--scan", context.capture_dir,
                    "--refroot", context.object_root / "object_init",
                ],
            )
        )

    for name, module, pass_args in passes:
        pass_rc, pass_log = run_module(
            context, module, pass_args, log_name=name.replace(" ", "_")
        )
        if pass_rc:
            result.warnings.append(f"{name} exited {pass_rc}; see {pass_log}")
    refinement = context.authoring_root / "obj_refine"
    if refinement.is_dir():
        result.artifacts["object_refinement"] = str(refinement)
    return result
