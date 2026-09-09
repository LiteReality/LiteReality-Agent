#!/usr/bin/env python3
"""Batch image -> articulated GLB, driven by the Claude Agent SDK.

Self-contained package: the `image-to-articulated-glb` skill ships in
`.claude/skills/` next to this script and is loaded from there — no
Claude Code installation or ~/.claude setup required on the host.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 batch_glb.py /path/to/images/*.png
    python3 batch_glb.py /path/to/images/ --concurrency 2 --retries 1
    python3 batch_glb.py img.png --out-base ./glb_out --dry-run

Output convention (same as the skill default):
    <image_dir>/<stem>_glb/<stem>.glb     (+ textures/, previews/, build recipe,
                                            object.py, object.md)
With --out-base:  <out_base>/<stem>/<stem>.glb

object.py is a self-contained Blender script that rebuilds the object on its own
(`blender -b --python object.py`); object.md documents how to edit it.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

PACKAGE_ROOT = Path(__file__).resolve().parent
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
SKILL_NAME = "image-to-articulated-glb"

PROMPT_TEMPLATE = """\
Run the {skill} skill end-to-end on this reference image:
  {image}
Write the final GLB to exactly this path:
  {glb}
Put every artifact (textures/, previews/, build recipe, object.py, object.md)
in that GLB's directory.

Run the FULL pipeline autonomously without pausing for confirmation:
analyze the image, fetch PBR textures, write and run the build script,
validate the GLB, render-check (Read both preview PNGs and compare against the
reference; iterate until correct), then emit a self-contained object.py
(via scripts/make_selfcontained.py) and an object.md describing how to edit it.

When finished, end your reply with a single line:
DONE {glb}
"""


@dataclass
class Job:
    image: Path
    glb: Path
    status: str = "pending"  # ok | failed | skipped
    cost_usd: float = 0.0
    seconds: float = 0.0
    attempts: int = 0
    error: str = ""
    log: list = field(default_factory=list)


def collect_images(inputs: list[str]) -> list[Path]:
    images = []
    for raw in inputs:
        p = Path(raw).expanduser()
        if p.is_dir():
            images += sorted(q for q in p.iterdir() if q.suffix.lower() in IMAGE_EXTS)
        elif p.is_file():
            images.append(p)
        else:
            print(f"warning: {raw} not found, skipping", file=sys.stderr)
    seen, out = set(), []
    for p in images:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(rp)
    return out


def glb_path_for(image: Path, out_base: Path | None) -> Path:
    if out_base:
        return out_base.resolve() / image.stem / f"{image.stem}.glb"
    return image.parent / f"{image.stem}_glb" / f"{image.stem}.glb"


def verify(job: Job) -> bool:
    """Success = GLB exists, render-check previews were produced (>=2, any names —
    static objects have no open/closed state and emit e.g. closed+front instead),
    and the self-contained object.py + object.md deliverables were written."""
    d = job.glb.parent
    previews = d / "previews"
    n_previews = len(list(previews.glob("*.png"))) if previews.is_dir() else 0
    return (
        job.glb.is_file()
        and job.glb.stat().st_size > 100_000
        and n_previews >= 2
        and (d / "object.py").is_file()
        and (d / "object.md").is_file()
    )


async def run_one(job: Job, args) -> None:
    options = ClaudeAgentOptions(
        cwd=str(PACKAGE_ROOT),  # anchor: skill discovery via .claude/skills/
        add_dirs=[str(job.image.parent), str(job.glb.parent)],  # grant access to in/out dirs
        system_prompt={"type": "preset", "preset": "claude_code"},
        setting_sources=["project"],  # package-local settings only -> self-contained
        skills=[SKILL_NAME],
        allowed_tools=[
            "Bash",
            "Read",
            "Write",
            "Edit",
            "Glob",
            "Grep",
            "Skill",
            "TodoWrite",
            "WebFetch",
        ],
        permission_mode="bypassPermissions",
        max_turns=args.max_turns,
        max_budget_usd=args.max_budget_usd,
        model=args.model,
    )
    prompt = PROMPT_TEMPLATE.format(skill=SKILL_NAME, image=job.image, glb=job.glb)

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, ResultMessage):
            job.cost_usd += message.total_cost_usd or 0.0
            job.log.append(
                {
                    "attempt": job.attempts,
                    "is_error": message.is_error,
                    "num_turns": message.num_turns,
                    "cost_usd": message.total_cost_usd,
                    "result_tail": (message.result or "")[-300:],
                }
            )
            if message.is_error:
                job.error = (message.result or "agent reported an error")[-300:]


async def process(job: Job, sem: asyncio.Semaphore, args) -> None:
    async with sem:
        if args.skip_existing and verify(job):
            job.status = "skipped"
            print(f"[skip] {job.image.name} -> {job.glb} (already complete)")
            return
        t0 = time.time()
        for attempt in range(1, args.retries + 2):
            job.attempts = attempt
            job.error = ""
            print(f"[run ] {job.image.name} (attempt {attempt})")
            try:
                await run_one(job, args)
            except Exception as e:  # SDK/transport errors
                job.error = f"{type(e).__name__}: {e}"
            if verify(job):
                job.status = "ok"
                break
            job.status = "failed"
            if not job.error:
                job.error = "pipeline finished but GLB/previews missing"
            print(f"[fail] {job.image.name}: {job.error}", file=sys.stderr)
        job.seconds = time.time() - t0
        mark = "ok  " if job.status == "ok" else "FAIL"
        print(f"[{mark}] {job.image.name}  ${job.cost_usd:.2f}  {job.seconds:.0f}s  -> {job.glb}")


async def main_async(args) -> int:
    images = collect_images(args.inputs)
    if not images:
        print("no input images found", file=sys.stderr)
        return 2

    out_base = Path(args.out_base).expanduser() if args.out_base else None
    jobs = [Job(image=img, glb=glb_path_for(img, out_base)) for img in images]

    if args.dry_run:
        for j in jobs:
            print(f"{j.image}  ->  {j.glb}")
        print(f"{len(jobs)} job(s)")
        return 0

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "note: ANTHROPIC_API_KEY is not set — the SDK will fall back to any "
            "logged-in Claude Code credentials on this machine.",
            file=sys.stderr,
        )

    sem = asyncio.Semaphore(args.concurrency)
    await asyncio.gather(*(process(j, sem, args) for j in jobs))

    ok = [j for j in jobs if j.status == "ok"]
    skipped = [j for j in jobs if j.status == "skipped"]
    failed = [j for j in jobs if j.status == "failed"]
    total_cost = sum(j.cost_usd for j in jobs)
    print(
        f"\nsummary: {len(ok)} ok, {len(skipped)} skipped, {len(failed)} failed"
        f" | total cost ${total_cost:.2f}"
    )
    report = {
        "jobs": [
            {
                "image": str(j.image),
                "glb": str(j.glb),
                "status": j.status,
                "attempts": j.attempts,
                "cost_usd": round(j.cost_usd, 4),
                "seconds": round(j.seconds, 1),
                "error": j.error,
                "log": j.log,
            }
            for j in jobs
        ]
    }
    report_path = Path(args.report)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"report: {report_path}")
    if failed:
        for j in failed:
            print(f"  failed: {j.image.name}: {j.error}", file=sys.stderr)
        return 1
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("inputs", nargs="+", help="image files and/or directories")
    ap.add_argument("--out-base", help="output base dir (default: next to each image, <stem>_glb/)")
    ap.add_argument(
        "--concurrency", type=int, default=2, help="parallel jobs (Blender is CPU-heavy; default 2)"
    )
    ap.add_argument(
        "--retries",
        type=int,
        default=1,
        help="retries per image after a failed attempt (default 1)",
    )
    ap.add_argument(
        "--max-turns", type=int, default=80, help="max agent turns per attempt (default 80)"
    )
    ap.add_argument(
        "--max-budget-usd", type=float, default=5.0, help="USD budget per attempt (default 5.0)"
    )
    ap.add_argument("--model", default=None, help="model override (default: SDK default)")
    ap.add_argument(
        "--skip-existing", action="store_true", help="skip images whose GLB+previews already exist"
    )
    ap.add_argument("--dry-run", action="store_true", help="list planned jobs and exit")
    ap.add_argument("--report", default="batch_glb_report.json", help="JSON report path")
    sys.exit(asyncio.run(main_async(ap.parse_args())))


if __name__ == "__main__":
    main()
