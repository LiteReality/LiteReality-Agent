#!/usr/bin/env python3
"""Category-aware procedural generation — RoomPlan object -> articulated GLB.

For every object the complexity router sent to the **procedural** path, this builds
a detailed, CATEGORY-SPECIFIC prompt (geometry + materials + exact articulation
from category_specs.py + the object's real RoomPlan dimensions + its clean
reference image) and drives the `image-to-articulated-glb` agent to produce a GLB
that both looks right and moves correctly (dishwasher door drops, drawer pulls out,
cabinet door swings on its outer edge).

Run (from repo root, with Blender 4.x/5.x on PATH):
    PATH=/path/to/blender_dir:$PATH \
      python -m lrauthor.models.object_generation.generate
      python -m lrauthor.models.object_generation.generate --only storage dishwasher
      python -m lrauthor.models.object_generation.generate --scan tea_room --dry-run

Needs claude_agent_sdk (uses logged-in Claude Code creds if no ANTHROPIC_API_KEY).
Outputs: procedural/glb/<scan>/<name>/<name>.glb (+ object.py/md, previews, textures).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pickle
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from lrauthor import REPO_ROOT, telemetry

# claude_agent_sdk is imported lazily inside _run_claude so the `codex` backend needs no SDK.
from .category_specs import CANONICAL_ORIENTATION, get_spec

HERE = Path(__file__).resolve().parent
# The bundled image-to-articulated-glb skill package used by this agent workflow.
AGENT_PKG = HERE / "articulated-glb-agent"
SKILL_NAME = "image-to-articulated-glb"


def default_work():
    """Compatibility default for direct CLI use, independent of pipeline packages."""
    output = Path(
        os.environ.get("LITEREALITY_FINAL")
        or os.environ.get("LITEREALITY_OUTPUT")
        or (REPO_ROOT / "run")
    )
    scan = os.environ.get("LITEREALITY_SCAN", "_shared")
    return output / scan / "scene_init" / "obj_stage" / "object_init"

# default LLM for the procedural articulated-GLB agent (override with --model / $LR_PROCEDURAL_MODEL)
DEFAULT_PROCEDURAL_MODEL = os.environ.get("LR_PROCEDURAL_MODEL", "claude-opus-5")

PROMPT_TEMPLATE = """\
Run the {skill} skill end-to-end to build ONE procedural 3D GLB of a {category} from this reference image:
  {image}

The category is FIXED and known (RoomPlan: {category}). Follow this category spec exactly.

TARGET — match the reference image's shape, proportions, part count/layout, materials, and colours.
REAL-WORLD SIZE from the room scan (metres): {dims}. Build to roughly these overall dimensions.

{canonical}

GEOMETRY ({category}):
{geometry}

MATERIALS:
{materials}

ARTICULATION — this is critical; the object must move correctly:
{articulation}
Tag every moving part with glTF node extras: articulation_type (revolute|prismatic), articulation_axis,
limit_min, limit_max, and add a closed->open animation. If the spec says the object is STATIC, add no
joints and render `preview_closed.png` (iso) + `preview_front.png` instead of an open view.

Pipeline: analyse the image, fetch PBR textures, write & run the Blender build script, validate the GLB,
render-check (Read the previews and compare to the reference; iterate until it matches, the articulation
opens the right way, AND the canonical orientation holds — the `--view front` render shows the real FRONT,
not the back or a side), then emit a self-contained object.py + object.md.

Write the final GLB to exactly:
  {glb}
Put every artifact (textures/, previews/, build recipe, object.py, object.md) in that GLB's directory.
End your reply with a single line:
DONE {glb}
"""


@dataclass
class Job:
    scan: str
    name: str
    category: str
    image: Path
    glb: Path
    dims: str
    status: str = "pending"
    attempts: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0
    error: str = ""
    log: list = field(default_factory=list)
    probe: dict = field(default_factory=dict)  # last QC-probe report (set by verify())
    completeness: dict = field(default_factory=dict)  # last reference-vs-render VLM report


def _dims_str(meta: dict) -> str:
    bbox = meta.get("bbox")
    if bbox is None:
        bbox = meta.get("size")
    try:
        import numpy as np

        d = [float(v) for v in np.asarray(bbox).reshape(-1)[:3]]
        if len(d) == 3:
            return f"{d[0]:.2f} (X) x {d[1]:.2f} (Y) x {d[2]:.2f} (Z)"
    except Exception:
        pass
    return "unknown (match the reference proportions)"


def load_scene_objects(work: Path, scan: str) -> dict:
    path = work / "input" / "scene_data" / scan / "objects.pkl"
    if not path.exists():
        return {}
    with path.open("rb") as f:
        objects = pickle.load(f)
    out = {}
    for obj in objects:
        nm = obj.get("mesh_id") or obj.get("object_type")
        if nm:
            out[nm] = obj
    return out


def collect_jobs(
    work: Path, scans: list[str] | None, only_cats: set[str] | None, out_root: Path
) -> list[Job]:
    manifest = json.loads((work / "routing" / "routing_manifest.json").read_text())
    jobs = []
    for scan, recs in manifest["scans"].items():
        if scans and scan not in scans:
            continue
        meta_by_name = load_scene_objects(work, scan)
        for r in recs:
            if r.get("route") != "procedural":
                continue
            cat = r["category"]
            if only_cats and cat not in only_cats:
                continue
            ref = Path(r["reference"])
            glb = out_root / r["name"] / f"{r['name']}.glb"
            jobs.append(
                Job(
                    scan=scan,
                    name=r["name"],
                    category=cat,
                    image=ref,
                    glb=glb,
                    dims=_dims_str(meta_by_name.get(r["name"], {})),
                )
            )
    return jobs


def _opening_dims(work: Path, scan: str, name: str) -> str:
    """W x thickness x H (metres) for an opening, from its wall-local dimension box."""
    path = work / "input" / "scene_data" / scan / "wall_holes.pkl"
    if not path.exists():
        return "unknown (match the reference proportions)"
    with path.open("rb") as f:
        wall_holes = pickle.load(f)
    for rec in wall_holes.values():
        if not isinstance(rec, dict) or "name" not in rec:
            continue
        for nm, dim in zip(rec["name"], rec["dimension"]):
            if nm == name:
                (x0, y0), (x1, y1) = dim
                w, h = abs(x1 - x0), abs(y1 - y0)
                return f"{w:.2f} (X, width) x 0.10 (Y, through-wall depth) x {h:.2f} (Z, height)"
    return "unknown (match the reference proportions)"


def collect_opening_jobs(
    work: Path, scans: list[str] | None, only_cats: set[str] | None, out_root: Path
) -> list[Job]:
    """Door/window jobs from opening_refs/ (openings are not in the routing manifest).

    Scans the directories directly (not the per-scan json, which a --only regen may
    have trimmed) and infers the category from the opening name (`*_Door_*` /
    `*_Window_*`)."""
    jobs = []
    refs_root = work / "opening_refs"
    if not refs_root.exists():
        return jobs
    for scan_dir in sorted(refs_root.iterdir()):
        if not scan_dir.is_dir() or (scans and scan_dir.name not in scans):
            continue
        for op_dir in sorted(scan_dir.iterdir()):
            ref = op_dir / "reference_1024.png"
            if not op_dir.is_dir() or not ref.exists():
                continue
            name = op_dir.name
            cat = "door" if "_Door_" in name else "window" if "_Window_" in name else "door"
            if only_cats and cat not in only_cats:
                continue
            glb = out_root / name / f"{name}.glb"
            jobs.append(
                Job(
                    scan=scan_dir.name,
                    name=name,
                    category=cat,
                    image=ref,
                    glb=glb,
                    dims=_opening_dims(work, scan_dir.name, name),
                )
            )
    return jobs


def build_prompt(job: Job) -> str:
    spec = get_spec(job.category)
    return PROMPT_TEMPLATE.format(
        skill=SKILL_NAME,
        category=job.category,
        image=job.image,
        glb=job.glb,
        dims=job.dims,
        canonical=CANONICAL_ORIENTATION,
        geometry=spec["geometry"],
        materials=spec["materials"],
        articulation=spec["articulation"],
    )


PROBE_SCRIPT = AGENT_PKG / ".claude" / "skills" / SKILL_NAME / "scripts" / "probe_glb.py"


def probe_glb(glb: Path) -> dict:
    """Run the semantic QC probe on a built GLB → its report dict. Never hard-gate on our own
    tooling breaking: if the probe itself errors, return a pass-through report with a note."""
    if not glb.is_file():
        return {"pass": False, "violations": [
            {"severity": "hard", "check": "exists", "part": glb.name, "detail": "GLB not produced"}]}
    try:
        proc = subprocess.run(
            [sys.executable, str(PROBE_SCRIPT), str(glb), "--json"],
            capture_output=True, text=True, timeout=120,
        )
        return json.loads(proc.stdout)
    except Exception as e:  # noqa: BLE001
        return {"pass": True, "violations": [], "probe_error": f"{type(e).__name__}: {e}"}


def probe_feedback(report: dict) -> str:
    """Turn probe violations into an instruction block appended to the retry prompt."""
    vs = report.get("violations", [])
    hard = [v for v in vs if v["severity"] == "hard"]
    soft = [v for v in vs if v["severity"] == "soft"]
    if not hard and not soft:
        return ""
    lines = ["\n\n## QC PROBE (scripts/probe_glb.py) FOUND ISSUES — fix and rebuild before finishing:"]
    for v in hard:
        lines.append(f"- [MUST FIX] {v['check']} · {v['part']}: {v['detail']}")
    for v in soft:
        lines.append(f"- [review]  {v['check']} · {v['part']}: {v['detail']}")
    lines.append("Rebuild the GLB and re-run probe_glb.py until it exits 0 (no [hard]) before you finish.")
    return "\n".join(lines)


COMPLETENESS_SCRIPT = AGENT_PKG / ".claude" / "skills" / SKILL_NAME / "scripts" / "completeness_check.py"
_PREVIEW_PREF = ("preview_front.png", "preview_closed.png", "preview_open.png")


def completeness_report(job: Job) -> dict:
    """VLM check: does the render capture every salient feature of the reference? (probe_glb is
    geometry-only and blind to missing hooks/handles/trim.) Never hard-fails on VLM errors."""
    pv = job.glb.parent / "previews"
    renders = [pv / n for n in _PREVIEW_PREF if (pv / n).is_file()]
    if not renders and pv.is_dir():
        renders = sorted(pv.glob("*.png"))[:3]
    if not job.image or not Path(job.image).is_file() or not renders:
        return {"pass": True, "missing": [], "notes": "no reference or renders"}
    cmd = [sys.executable, str(COMPLETENESS_SCRIPT), "--ref", str(job.image), "--json"]
    for p in renders:
        cmd += ["--render", str(p)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        return json.loads(proc.stdout)
    except Exception as e:  # noqa: BLE001
        return {"pass": True, "missing": [], "notes": f"completeness error: {type(e).__name__}"}


def completeness_feedback(report: dict) -> str:
    missing = report.get("missing") or []
    if not missing:
        return ""
    lines = ["\n\n## COMPLETENESS (reference vs render) — the build is MISSING features the reference shows:"]
    lines += [f"- {m}" for m in missing]
    if report.get("notes"):
        lines.append(f"note: {report['notes']}")
    lines.append("Add these parts to the build recipe, rebuild, re-render, and re-check until nothing "
                 "salient is missing (do not drop features that are already correct).")
    return "\n".join(lines)


def resume_note(job: Job) -> str:
    """Point a retry at the previous attempt's artifacts instead of letting it start over.

    The base prompt says "build ONE procedural 3D GLB", and a retry re-sends it to a FRESH session
    that has no memory of the last attempt — so a gate rejecting one missing handle threw away a
    correct build and paid for the whole thing again (the window rebuild cost ~600s and a second
    full agent session). The recipe, object.py and previews are all still on disk; `object.py` only
    gets swept away by drop_build_recipe once the object PASSES. Naming those files turns the retry
    into an edit. Only the listed files are mentioned, so a genuinely empty directory still
    produces a from-scratch build.
    """
    d = job.glb.parent
    existing = [p.name for p in sorted(d.glob("build_*.py"))]
    existing += [n for n in ("object.py",) if (d / n).is_file()]
    if not existing:
        return ""
    return (
        f"\n\n## PREVIOUS ATTEMPT — do NOT start from scratch.\n"
        f"Attempt {job.attempts} of this object is already on disk in {d}:\n"
        + "".join(f"  - {n}\n" for n in existing)
        + "Read the build recipe, make the SMALLEST change that fixes the issues below, then "
        "re-run it, re-render and re-check. Keep everything that is already correct."
    )


def verify_reasons(job: Job) -> list[str]:
    """Why this object is NOT considered built. Empty means it is.

    Split out of `verify` so a rebuild can say what it is for. A killed run leaves a directory
    that looks finished — the GLB and previews are written before `object.py` and `object.md` —
    so every later run silently regenerates it from scratch, which is minutes of agent time with
    no explanation and a progress count that already reads complete.
    """
    d = job.glb.parent
    previews = d / "previews"
    n_prev = len(list(previews.glob("*.png"))) if previews.is_dir() else 0
    missing = []
    if not job.glb.is_file():
        missing.append("no glb")
    if n_prev < 2:
        missing.append(f"{n_prev} preview(s), need 2")
    for name in ("object.py", "object.md"):
        if not (d / name).is_file():
            missing.append(f"no {name}")
    report = probe_glb(job.glb) if job.glb.is_file() else {"pass": False}
    job.probe = report
    if not report.get("pass", True):
        missing.append("failed the geometry probe")
    return missing


def verify(job: Job) -> bool:
    # No GLB size floor: a thin flat object (a TV panel, a sign) legitimately makes a
    # tiny GLB, and a size check false-fails it. Require the deliverables exist AND the
    # semantic QC probe passes (no HARD violations) — the probe report is stashed on the
    # job so process() can feed the specifics back into the next attempt.
    d = job.glb.parent
    previews = d / "previews"
    n_prev = len(list(previews.glob("*.png"))) if previews.is_dir() else 0
    files_ok = (
        job.glb.is_file()
        and n_prev >= 2
        and (d / "object.py").is_file()
        and (d / "object.md").is_file()
    )
    report = probe_glb(job.glb)
    job.probe = report
    return files_ok and bool(report.get("pass", True))


def drop_build_recipe(obj_dir: Path) -> list[str]:
    """Delete the agent's ``build_<name>.py`` recipe once ``object.py`` exists.

    The recipe is a build-time INTERMEDIATE: make_selfcontained.py inlines blender_lib.py
    into it to produce object.py, and object.py is the only definition anything downstream
    reads. Shipping both leaves two near-identical definitions in the object dir, and the
    recipe is the unusable one — it imports blender_lib through an absolute path into the
    authoring machine's skill dir, so it breaks on any other checkout.

    Guarded on object.py being present: if the inlining step never ran, the recipe is still
    the ONLY definition and must survive.
    """
    if not (obj_dir / "object.py").is_file():
        return []
    removed = []
    for p in sorted(obj_dir.glob("build_*.py")):
        p.unlink()
        removed.append(p.name)
    return removed


async def run_one(job: Job, args, feedback: str = "") -> None:
    """Generate one articulated GLB via the chosen agent backend (claude | codex).

    Both run the same bundled ``image-to-articulated-glb`` skill: Claude
    Code loads it natively via claude_agent_sdk; Codex has no skill system, so the SKILL.md is
    injected as instructions. Keep the skill as the single shared playbook for both.

    ``feedback`` (non-empty on retries) carries the previous attempt's QC-probe violations so the
    agent knows exactly what to fix instead of blindly regenerating.
    """
    from lrauthor.agent import providers

    job.glb.parent.mkdir(parents=True, exist_ok=True)
    harness = providers.resolve("procedural", getattr(args, "provider", None))

    # Both harnesses run the SAME bundled playbook; only the delivery differs. Claude Code loads
    # `image-to-articulated-glb` natively as a skill; Codex has no skill system, so SKILL.md is
    # injected as prompt text. Keeping one playbook is the point — do not fork it per harness.
    prompt = build_prompt(job) + feedback
    if "skills" not in harness.supports:
        scripts = AGENT_PKG / ".claude" / "skills" / SKILL_NAME / "scripts"
        prompt = (
            "Run this SKILL end-to-end, fully autonomously, without asking for confirmation.\n\n"
            f"===== SKILL: {SKILL_NAME} =====\n{_skill_instructions()}\n===== END SKILL =====\n\n"
            f"{build_prompt(job)}\n\n"
            f"Reference image: {job.image}\nOutput GLB: {job.glb}\n"
            f"Skill helper scripts live in: {scripts}\n"
            f"{feedback}"
        )

    spec = providers.SessionSpec(
        prompt=prompt,
        cwd=AGENT_PKG,
        read_roots=(job.image.parent, job.glb.parent),
        file_tools=("Bash", "Read", "Write", "Edit", "Glob", "Grep", "Skill", "TodoWrite", "WebFetch"),
        skills=(SKILL_NAME,),
        setting_sources=("project",),
        model=args.model or DEFAULT_PROCEDURAL_MODEL,
        max_turns=args.max_turns,
        # A dollar cap only means something on metered API access. The default path
        # is the logged-in `claude` CLI (subscription), where the SDK's cost figure is
        # an API-list-price equivalent and capping on it truncates a build for no real
        # saving -- the door job finished at $4.53 against the old $5.00 default.
        # `max_turns` is the runaway guard; pass --max-budget-usd to opt back in.
        budget_usd=args.max_budget_usd or None,
    )

    t0 = time.time()
    async for message in harness.run(spec):
        if isinstance(message, providers.SessionResult):
            job.cost_usd += message.total_cost_usd or 0.0
            entry = {
                "attempt": job.attempts,
                "is_error": message.is_error,
                "num_turns": message.num_turns,
                "cost_usd": message.total_cost_usd,
                "result_tail": (message.result or "")[-200:],
                "seconds": round(time.time() - t0, 1),
                "provider": harness.name,
            }
            job.log.append(entry)
            if message.is_error:
                job.error = (message.result or "agent error")[-200:]
            # record the agent run in the scan's process trace (for later visualization)
            telemetry.agent_step(
                "image-to-articulated-glb",
                f"{job.name}:attempt{job.attempts}",
                payload={
                    "job": {
                        "scan": job.scan,
                        "name": job.name,
                        "category": job.category,
                        "dims": job.dims,
                        "image": str(job.image),
                        "glb": str(job.glb),
                    },
                    "result": entry,
                },
                name=job.name,
                category=job.category,
                is_error=message.is_error,
                num_turns=message.num_turns,
                cost_usd=message.total_cost_usd,
            )


def _skill_instructions() -> str:
    """The skill playbook (SKILL.md), injected into the prompt for harnesses without skills."""
    sk = AGENT_PKG / ".claude" / "skills" / SKILL_NAME / "SKILL.md"
    try:
        return sk.read_text(encoding="utf-8")
    except OSError:
        return ""


async def process(job: Job, sem: asyncio.Semaphore, args) -> None:
    # The QC gates below (probe_glb, completeness_report) are blocking subprocess.run calls, and
    # one of them is a whole agent session. Called directly they stall the event loop, which
    # freezes every OTHER job's agent session too — `--concurrency N` then buys nothing during
    # the gates, and the gates are where most of the wall time is. Hand them to a thread.
    async with sem:
        reasons = (
            await asyncio.to_thread(verify_reasons, job)
            if args.skip_existing
            else ["--skip-existing not set"]
        )
        if args.skip_existing and not reasons:
            job.status = "skipped"
            drop_build_recipe(job.glb.parent)  # clean recipes left by earlier runs too
            print(f"[skip] {job.scan}/{job.name} ({job.category}) — already built", flush=True)
            return
        if args.skip_existing and job.glb.is_file():
            # Partially-built: say what is missing. Otherwise this is minutes of silent agent
            # work on an object whose GLB is already on disk and already counted as done.
            print(f"[redo] {job.scan}/{job.name}: rebuilding — {', '.join(reasons)}", flush=True)
        t0 = time.time()
        feedback = ""
        for attempt in range(1, args.retries + 2):
            job.attempts = attempt
            job.error = ""
            print(f"[run ] {job.scan}/{job.name} ({job.category}) attempt {attempt}")
            try:
                await run_one(job, args, feedback)
            except Exception as e:  # noqa: BLE001
                job.error = f"{type(e).__name__}: {e}"
            # deliverables exist AND geometric QC probe passes (stashes job.probe)
            if await asyncio.to_thread(verify, job):
                # Passed geometry; now the completeness gate (VLM: does it match the reference?).
                comp = (
                    {"pass": True}
                    if getattr(args, "no_completeness", False)
                    else await asyncio.to_thread(completeness_report, job)
                )
                job.completeness = comp
                if comp.get("pass", True):
                    job.status = "ok"
                    drop_build_recipe(job.glb.parent)
                    break
                job.status = "failed"
                job.error = "incomplete: " + "; ".join(comp.get("missing") or [])[:150]
                # retry to ADD the missing features, editing the build already on disk
                feedback = resume_note(job) + completeness_feedback(comp)
                print(f"[fail] {job.scan}/{job.name}: {job.error}", file=sys.stderr)
                continue
            job.status = "failed"
            # Prefer the probe's specific reason; carry it into the next attempt as feedback.
            hard = [v for v in (job.probe.get("violations") or []) if v["severity"] == "hard"]
            job.error = (
                f"QC failed: {'; '.join(v['check'] + ':' + v['part'] for v in hard)}"
                if hard else (job.error or "GLB/previews/deliverables missing")
            )
            feedback = resume_note(job) + probe_feedback(job.probe)
            print(f"[fail] {job.scan}/{job.name}: {job.error}", file=sys.stderr)
        job.seconds = time.time() - t0
        mark = "ok  " if job.status == "ok" else "FAIL"
        # `~$` because on the CLI path this is a list-price equivalent, not a charge.
        print(
            f"[{mark}] {job.scan}/{job.name} ({job.category})  "
            f"{job.seconds:.0f}s  (~${job.cost_usd:.2f} list-price equiv)"
        )


async def run_all(jobs: list[Job], collect, sem: asyncio.Semaphore, args) -> list[Job]:
    """Run every job, re-reading the work list each time one finishes.

    The list used to be a single snapshot taken at startup, and the references it is built from
    are written by a stage that can still be running. On a seven-opening room two reference
    images landed 27s and 49s after this process listed its work: those two openings were never
    built, never reported, and never counted as failed. They simply were not there when we looked,
    and nothing looked again.

    A free worker is the natural moment to look again, so that is when it re-collects. Anything
    that appeared since is picked up by the slot that just opened. `seen` is keyed on the job name
    rather than the job, so a finished or failed object is never dispatched twice.
    """
    seen = {job.name for job in jobs}
    everything = list(jobs)
    pending = {asyncio.create_task(process(job, sem, args)) for job in jobs}
    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()  # a crashed worker surfaces here rather than being swallowed
        for job in collect():
            if job.name in seen:
                continue
            seen.add(job.name)
            everything.append(job)
            print(f"[new ] {job.scan}/{job.name} ({job.category}) — appeared after the run started",
                  flush=True)
            pending.add(asyncio.create_task(process(job, sem, args)))
    return everything


async def main_async(args) -> int:
    work = (args.work or default_work()).resolve()
    # per-scan layout: out defaults to the scan's reconstructed_objs/ dir (sibling of object_init/)
    out_root = (args.out or (work.parent / "reconstructed_objs")).resolve()
    only_cats = set(args.only) if args.only else None
    if args.openings:
        def collect():
            return collect_opening_jobs(work, args.scan, only_cats, out_root)
        jobs = collect()
        if not jobs:
            print(
                "no opening jobs found (run object_init.references.opening_references first)",
                file=sys.stderr,
            )
            return 2
    else:
        def collect():
            return collect_jobs(work, args.scan, only_cats, out_root)
        jobs = collect()
        if not jobs:
            print(
                "no procedural-route jobs found (run object_init.classify.classify_complexity first)",
                file=sys.stderr,
            )
            return 2

    print(f"{len(jobs)} procedural job(s):")
    for j in jobs:
        print(f"  {j.scan}/{j.name}  [{j.category}]  {j.dims}")
    if args.dry_run:
        print("\n--- example prompt ---\n" + build_prompt(jobs[0]))
        return 0

    # open the scan's process trace so each agent run is recorded under traces/agent/
    telemetry.start(jobs[0].scan, work.parent / "traces")
    telemetry.event(
        "procedural_start", scan=jobs[0].scan, n_jobs=len(jobs), openings=bool(args.openings)
    )

    sem = asyncio.Semaphore(args.concurrency)
    jobs = await run_all(jobs, collect, sem, args)

    report = {
        "out": str(out_root),
        "jobs": [vars(j) | {"image": str(j.image), "glb": str(j.glb)} for j in jobs],
    }
    # Per-branch filename. The objects pass and the openings pass share one `out_root` and used to
    # write the same `procedural_report.json`, so whichever finished last silently replaced the
    # other's report — the last run's file listed only the two openings, with Table0/Table1 gone.
    # Serially that was a deterministic overwrite; now that the branches run together it is a race.
    (out_root / f"{'openings' if args.openings else 'procedural'}_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    ok = sum(1 for j in jobs if j.status == "ok")
    cost = sum(j.cost_usd for j in jobs)
    print(f"\n{ok}/{len(jobs)} ok · ~${cost:.2f} list-price equiv · {out_root}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--work", type=Path, default=None,
                    help="per-scan object work root (default: derived from pipeline environment)")
    ap.add_argument("--out", type=Path, help="output root (default procedural/glb)")
    ap.add_argument("--scan", nargs="*", help="limit to these scans")
    ap.add_argument(
        "--only",
        nargs="*",
        help="limit to these categories (e.g. storage dishwasher / door window)",
    )
    ap.add_argument(
        "--openings",
        action="store_true",
        help="generate doors/windows from opening_refs instead of routing-manifest objects",
    )
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--max-turns", type=int, default=80)
    ap.add_argument(
        "--max-budget-usd",
        type=float,
        default=None,
        help="Stop an agent run at this API-list-price cost. Off by default: the "
        "logged-in `claude` CLI is a subscription, so the figure is notional and "
        "capping on it only truncates builds. Set it when running on metered API access.",
    )
    ap.add_argument("--model", default=None)
    ap.add_argument(
        "--provider",
        choices=["claude", "codex"],
        default=None,
        help="agent backend for articulated generation (default $LR_PROCEDURAL_PROVIDER or claude)",
    )
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--no-completeness", action="store_true",
                    help="skip the reference-vs-render completeness VLM gate (faster/cheaper, "
                         "geometry probe only)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
