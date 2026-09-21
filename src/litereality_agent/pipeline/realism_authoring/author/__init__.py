"""Run Room.py authoring and its optional polish passes."""

import json
import shutil
import tempfile
from pathlib import Path

from litereality_agent.pipeline.context import RunContext
from litereality_agent.pipeline.result import StageResult, StageStatus
from litereality_agent.pipeline.support import command_result, run_module


def complete(context: RunContext) -> bool:
    from litereality_agent.pipeline.realism_authoring.acceptance import is_accepted
    preview = context.authoring_root / "room_preview"
    return (is_accepted(context.authored_room, preview)
            or is_accepted(context.authored_room, preview, "acceptance.json"))


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
    if context.authored_room.exists() and options.get("force"):
        backup = Path(tempfile.mkdtemp(prefix="before-author-", dir=context.authored_room.parent))
        shutil.move(str(context.authored_room), str(backup / "room"))
    context.authored_room.parent.mkdir(parents=True, exist_ok=True)
    if not context.authored_room.exists():
        shutil.copytree(context.seed_room, context.authored_room)
    args: list[object] = [
        "--scene", context.scene_dir,
        "--room", context.authored_room,
        "--surface-ref", context.authoring_root / "surface_ref",
        "--scan", context.capture_dir,
        "--profile", options.get("profile", "open"),
    ]
    # Budgets only when asked for: the entrypoint knows each profile's own defaults.
    for key, flag in (("max_turns", "--max-turns"), ("step_budget", "--step-budget")):
        if options.get(key):
            args += [flag, options[key]]
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
        result.status = StageStatus.FAILED
        result.error = "needs_repair: authoring session stopped early; saved room retained"
        return result

    # THE GATE. Whatever the brief asked for, the question at the end is the same: could a physics
    # engine take this room? Answered from the build and written down beside it. It cannot run
    # before a compile has produced room_preview/, and the polish passes below rebuild, so it runs
    # last (see the end of this function).
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
            result.status = StageStatus.FAILED
            result.error = f"needs_repair: {name} did not finish; saved room retained"
            return result
    refinement = context.authoring_root / "obj_refine"
    if refinement.is_dir():
        result.artifacts["object_refinement"] = str(refinement)

    preview = context.authoring_root / "room_preview"
    # The gate reads the BUILD. A session that rendered has one; a session that stopped early
    # or only edited may not, and an unbuilt room is exactly the one whose claims nobody checked.
    # `Room.py` is valid Python here (author.run guarantees it), so build it once (~1-2 min).
    from litereality_agent.pipeline.realism_authoring.acceptance import finish
    report = finish(context, options)
    result.artifacts["acceptance"] = str(preview / "author_acceptance.json")
    result.details["acceptance"] = report
    if not report.get("accepted"):
        result.status = StageStatus.FAILED
        result.error = "needs_repair: support/visual acceptance failed; source and evidence retained"
    return result
