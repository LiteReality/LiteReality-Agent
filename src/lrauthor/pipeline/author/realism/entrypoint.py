"""Bind pipeline scene-package arguments and launch the reusable authoring agent."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from lrauthor.agent import scratch
from lrauthor.agent.author import PROFILES, run
from lrauthor.pipeline.author import arguments as stage_args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    stage_args.add_scene_arg(parser)
    parser.add_argument(
        "--room",
        default=None,
        help="room dir to edit (default: the package's work room)",
    )
    parser.add_argument("--surface-ref", default=None)
    parser.add_argument("--scan", default=None, help="capture dir (default: package capture)")
    parser.add_argument("--model", default=os.environ.get("HARNESS_MODEL", "claude-opus-5"))
    parser.add_argument(
        "--max-turns",
        type=int,
        default=140,
        help="hard SDK backstop; the step budget lands the run before this",
    )
    parser.add_argument(
        "--step-budget",
        type=int,
        default=int(os.environ.get("AUTHOR_STEPS", "100")),
        help="tool-call budget with graceful wind-down (0 disables)",
    )
    parser.add_argument(
        "--step-reserve",
        type=int,
        default=int(os.environ.get("AUTHOR_STEP_RESERVE", "15")),
        help="steps reserved for final edits after self-check tools switch off",
    )
    parser.add_argument("--profile", default="base", choices=list(PROFILES))
    parser.add_argument(
        "--provider",
        default=None,
        choices=["claude", "codex"],
        help="agent harness (default: $LR_AUTHOR_PROVIDER, else $LR_AGENT_PROVIDER, else claude)",
    )
    args = stage_args.bind(parser.parse_args(), need=("room", "surface_ref", "scan"))

    try:
        rc = asyncio.run(
            run(
                Path(args.room),
                Path(args.surface_ref),
                Path(args.scan),
                args.model,
                args.max_turns,
                args.profile,
                step_budget=args.step_budget,
                step_reserve=args.step_reserve,
                provider=args.provider,
            )
        )
        raise SystemExit(rc or 0)
    finally:
        scratch.finish()


if __name__ == "__main__":
    main()
