"""Fresh, isolated scan-to-publish smoke test. Never modifies the supplied capture.

Usage: python scripts/smoke_support_first.py CAPTURE --workdir NEW_DIR
       [--settings-from EXISTING_CHECKOUT] [--use-dino]
Model-backed stages consume account usage. Results/status stay in NEW_DIR.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from litereality_agent.pipeline.context import RunContext
    from litereality_agent.pipeline.realism_authoring.acceptance import digest
    from litereality_agent.pipeline.runner import PipelineRunner
    from litereality_agent.settings import load_settings

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--settings-from", type=Path, default=ROOT)
    parser.add_argument("--use-dino", action="store_true")
    args = parser.parse_args()
    source, work = args.capture.resolve(), args.workdir.resolve()
    if not source.is_dir() or source == work or source in work.parents:
        parser.error("capture must exist; workdir must be outside the capture")
    work.mkdir(parents=True, exist_ok=True)
    capture = work / "input" / source.name
    output = work / "output"
    if capture.exists() or output.exists():
        parser.error(
            "this is a fresh-run harness: input/output already exists; choose a new workdir"
        )
    originals = {str(p.relative_to(source)): digest(p) for p in source.rglob("*") if p.is_file()}
    shutil.copytree(source, capture)
    settings = load_settings(
        args.settings_from,
        repo_root=ROOT,
        output_root=output,
        scans_dir=capture.parent,
        author_provider="codex",
        materials_provider="codex",
        quality_provider="codex",
        refine_provider="codex",
        codex_model="gpt-6-astra",
    )
    # Every subprocess, including render/tool hosts, must use this worktree's code.
    os.environ["PYTHONPATH"] = str(ROOT / "src")
    os.environ["PYTHONUNBUFFERED"] = "1"
    context = RunContext.resolve(capture, output_root=output, settings=settings)
    status = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source": str(source),
        "capture_copy": str(capture),
        "scene": str(context.scene_dir),
        "author_model": settings.codex_model,
        "use_dino": args.use_dino,
    }
    status_path = work / "smoke_status.json"
    status_path.write_text(json.dumps(status, indent=2))
    (work / "source_checksums.json").write_text(json.dumps(originals, indent=2))
    try:
        results = PipelineRunner().run(
            context,
            through="publish",
            options={
                "ingest": {"use_dino": args.use_dino},
                "reconstruct": {"chair_qc": True},
                "author": {
                    "profile": "open",
                    "repair_rounds": 2,
                    "repair_steps": 40,
                    "repair_seconds": 600,
                },
            },
        )
        status["stages"] = [r.to_dict() for r in results]
        status["status"] = (
            "accepted"
            if results and results[-1].stage == "publish" and results[-1].ok
            else "needs_attention"
        )
    except Exception as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        current = {str(p.relative_to(source)): digest(p) for p in source.rglob("*") if p.is_file()}
        status["source_unchanged"] = originals == current
        if not status["source_unchanged"]:
            status["status"] = "failed"
        status["finished_utc"] = datetime.now(timezone.utc).isoformat()
        status_path.write_text(json.dumps(status, indent=2))
        print(
            json.dumps({k: status[k] for k in ("status", "scene", "source_unchanged")}), flush=True
        )
    return 0 if status["status"] == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
