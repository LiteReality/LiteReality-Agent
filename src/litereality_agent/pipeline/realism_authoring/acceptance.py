"""Bounded support-first repair and independent review, with build-bound acceptance.

No missing, failed or interrupted check is a pass. Sources and review candidates survive
failure; only an exact, validated build can be reused as a completed author/publish stage.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import shutil
from pathlib import Path

from litereality_agent.agent import providers
from litereality_agent.pipeline.room_qc.validate import validate
from litereality_agent.pipeline.support import run_module

SCHEMA = "support-first/1"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(room: Path, preview: Path) -> dict:
    files = sorted(
        p
        for p in room.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and not p.name.startswith(".")
    )
    return {
        "source": {str(p.relative_to(room)): digest(p) for p in files},
        "build": {
            name: digest(preview / name)
            for name in ("Room.glb", "room_layout.json", "manifest.json")
        },
    }


def is_accepted(room: Path, preview: Path, name="author_acceptance.json") -> bool:
    try:
        report = json.loads((preview / name).read_text())
        return (
            report.get("schema") == SCHEMA
            and report.get("accepted") is True
            and report.get("fingerprint") == fingerprint(room, preview)
        )
    except (OSError, ValueError, TypeError):
        return False


def visual_accepted(review: dict) -> bool:
    score = review.get("score")
    return (
        review.get("pass") is True
        and isinstance(score, (int, float))
        and not isinstance(score, bool)
        and math.isfinite(score)
        and score >= 8
        and review.get("issues") == []
        and review.get("all_objects_reviewed") is True
    )


def preserved_instances(seed: Path, room: Path) -> bool:
    """Manifest identities/membership cannot be edited away to satisfy the gate."""

    def identities(root):
        assets = json.loads((root / "manifest.json").read_text())["assets"]
        return {
            a["object"]: {k: a.get(k) for k in ("kind", "maps_to_prim", "represents_prims")}
            for a in assets
        }

    if identities(seed) != identities(room):
        return False
    # Editable procedural recipes may improve; bundled source mesh references may not vanish
    # or be replaced. Matching paths also prevents a repair from quietly swapping chair assets.
    for source in seed.rglob("*.glb"):
        target = room / source.relative_to(seed)
        if not target.is_file() or digest(source) != digest(target):
            return False
    return True


async def repair(room: Path, scan: Path, report: Path, *, steps: int, seconds: int):
    harness = providers.resolve("author")
    spec = providers.SessionSpec(
        cwd=room,
        read_roots=(scan, report.parent),
        capability_scene=room,
        capability_tools=("render", "select_views", "check_collisions"),
        model="claude-opus-5",
        step_budget=steps,
        timeout_seconds=seconds,
        max_turns=steps + 10,
        prompt=f"""Repair this room's remaining support and visual defects.
Read {report} and its referenced validation/review evidence. Read Room.md and capture evidence
under {scan}. Edit only this room directory. Fix support FIRST: floor contacts, appropriate
table/shelf surfaces, wall attachments, every member of merged groups and chair stacks.
Use actual geometry, not bounding-box top heights. For articulated furniture declare
support_part on each supported group so it follows the correct moving link.
Preserve manifest identities, represents_prims, source neural meshes and articulated joints.
Do not delete difficult objects, add invisible supports, change checks or edit reports.
Inspect renders against photographs. Save valid Room.py frequently. Do not launch another
model. The supervisor will rebuild and independently check your result. Budget: {steps} tool
calls / {seconds} seconds. Return a concise summary of fixes and unresolved problems.""",
    )
    result = None
    async for event in harness.run(spec):
        if isinstance(event, providers.SessionResult):
            result = event
    if result is None or result.is_error or result.stopped:
        raise RuntimeError(
            f"repair incomplete: {getattr(result, 'stopped', '') or getattr(result, 'result', 'no result')}"
        )


def assess(context, preview: Path, evidence: Path) -> dict:
    from litereality_agent.agent.tools._vlm import vision

    evidence.mkdir(parents=True, exist_ok=True)
    report = {"schema": SCHEMA, "accepted": False, "status": "needs_repair"}
    try:
        report["fingerprint"] = fingerprint(context.authored_room, preview)
        report["instances_preserved"] = preserved_instances(
            context.seed_room, context.authored_room
        )
        geometry = validate(context.authored_room, preview)
        (evidence / "validation.json").write_text(json.dumps(geometry, indent=2))
        report["validation"] = str(evidence / "validation.json")
        report["support_pass"] = geometry.get("pass") is True
        if not report["instances_preserved"] or not report["support_pass"]:
            return report
        rc, log = run_module(
            context,
            "litereality_agent.room_ops.rendering.room_render.render_vs_capture",
            [
                "--scan",
                context.capture_dir,
                "--room",
                context.authored_room,
                "--out",
                evidence / "compare",
                "--frames",
                "6",
            ],
            log_name="acceptance_render",
        )
        pairs = sorted((evidence / "compare" / "pairs").glob("pair_*.png"))
        if rc or len(pairs) < 3:
            raise RuntimeError(
                f"independent review needs at least three successful capture pairs: {log}"
            )
        layout = json.loads((preview / "room_layout.json").read_text())
        review = asyncio.run(
            vision(
                [str(p) for p in pairs],
                "Independently judge this reconstruction against the paired capture photographs. "
                "Check EVERY listed object, merged member, chair stack, support surface and opening. "
                "Check shape, scale, missing objects, materials and excessive lighting. "
                "Do not approve invisible/uncertain support or insufficient evidence. "
                "Return JSON: pass (boolean), score (0-10), all_objects_reviewed (boolean), "
                "issues (list of concrete unresolved defects with object IDs). "
                "A pass needs score >=8, no issues, and all objects reviewed. Layout: "
                + json.dumps(layout["objects"]),
                json_mode=True,
            )
        )
        (evidence / "visual_review.json").write_text(json.dumps(review, indent=2))
        report["review"] = str(evidence / "visual_review.json")
        report["visual_pass"] = visual_accepted(review)
        # Review/rendering must not change what was checked.
        unchanged = fingerprint(context.authored_room, preview) == report["fingerprint"]
        report["accepted"] = report["visual_pass"] and unchanged
        report["status"] = "accepted" if report["accepted"] else "needs_repair"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def finish(context, options: dict) -> dict:
    preview = context.authoring_root / "room_preview"
    rounds = int(options.get("repair_rounds", 2))
    steps = int(options.get("repair_steps", 40))
    seconds = int(options.get("repair_seconds", 600))
    if not 0 <= rounds <= 3 or not 1 <= steps <= 150 or not 1 <= seconds <= 1800:
        raise ValueError("repair caps: rounds 0..3, steps 1..150, seconds 1..1800")
    root = context.authoring_root / "acceptance"
    root.mkdir(parents=True, exist_ok=True)
    # Never overwrite an earlier attempt's evidence.
    import tempfile

    attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=root))
    for number in range(rounds + 1):
        rc, log = run_module(
            context,
            "litereality_agent.room_ops.compile.build_from_room",
            ["--room", context.authored_room, "--out", preview, "--regenerate"],
            log_name="acceptance_build",
        )
        evidence = attempt / f"round_{number:02d}"
        evidence.mkdir()
        shutil.copytree(context.authored_room, evidence / "room", symlinks=True)
        if not rc:
            for name in ("Room.glb", "room_layout.json", "manifest.json"):
                if (preview / name).is_file():
                    shutil.copy2(preview / name, evidence / name)
        report = (
            assess(context, preview, evidence)
            if not rc
            else {
                "schema": SCHEMA,
                "accepted": False,
                "status": "needs_repair",
                "error": f"build failed: {log}",
            }
        )
        report["round"] = number
        path = evidence / "assessment.json"
        path.write_text(json.dumps(report, indent=2))
        preview.mkdir(parents=True, exist_ok=True)
        (preview / "author_acceptance.json").write_text(json.dumps(report, indent=2))
        if report["accepted"] or number == rounds or rc:
            return report
        try:
            asyncio.run(
                repair(
                    context.authored_room, context.capture_dir, path, steps=steps, seconds=seconds
                )
            )
        except Exception as exc:
            report.update(accepted=False, status="needs_repair", error=str(exc))
            (preview / "author_acceptance.json").write_text(json.dumps(report, indent=2))
            return report
    raise AssertionError("unreachable")
