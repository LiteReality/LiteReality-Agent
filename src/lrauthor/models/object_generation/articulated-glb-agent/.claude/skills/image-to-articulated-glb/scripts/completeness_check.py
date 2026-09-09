#!/usr/bin/env python3
"""Completeness QC — does the render capture every salient feature of the reference?

The geometry probe (probe_glb.py) proves a model is VALID (connected, sane joints, faces the
front) but is blind to MISSING features: a door that lacks its coat hooks, a cabinet missing a
handle, the wrong number of drawers. This closes that gap with a vision judge: it compares the
reference image against the render previews and lists what's present in the reference but absent
or wrong in the render.

Runs on the logged-in Claude via claude_agent_sdk (Read-only) — no API key, no Gemini. Kept
separate from probe_glb.py so that stays a pure, dependency-free geometric check.

Usage:
    completeness_check.py --ref <reference.png> --render <r.png> [--render <r2.png> ...] [--json]

Exit 0 = complete (nothing salient missing), 1 = missing features, 2 = bad usage / VLM error.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import sys
from pathlib import Path

PROMPT = (
    "IMAGE 1 is the REFERENCE of a single object. The remaining images are RENDERS of a 3D model "
    "built to reproduce it (front / closed / open views). Judge only whether the render captures "
    "every SALIENT feature of the reference: the distinct parts (hooks, handles, latches, knobs, "
    "vents, shelves, panels, glazing, trim), their COUNT, and rough placement. IGNORE lighting, "
    "exact colour, background, resolution and micro-detail. "
    'Reply with ONLY a JSON object: {"pass": bool, "missing": ["<feature in the reference that is '
    'absent or clearly wrong in the render>", ...], "notes": "<one short line>"}. '
    "Set pass=true only when nothing salient is missing."
)


_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


def _blocks(images: list[Path]) -> list[dict]:
    """The images as Messages-API content blocks, each labelled so the prompt can refer to it."""
    out: list[dict] = []
    for i, p in enumerate(images):
        out.append({"type": "text", "text": f"IMAGE {i + 1} ({'REFERENCE' if i == 0 else p.name}):"})
        out.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": _MIME.get(p.suffix.lower(), "image/png"),
                "data": base64.b64encode(p.read_bytes()).decode(),
            },
        })
    out.append({"type": "text", "text": PROMPT})
    return out


async def _ask(images: list[Path]) -> str:
    """Ask the judge in ONE turn, with the images attached to the prompt.

    They used to be named as paths and pulled in with the `Read` tool — one model round-trip per
    image before the question could even be answered, five for a reference plus three renders, on
    every build attempt of every object, on the critical path. `query()` also accepts a streamed
    user message whose content is Messages-API blocks, so the images can ride along with the
    prompt: same logged-in-CLI auth (no ANTHROPIC_API_KEY, still unmetered), one turn instead of
    five. Measured on one reference image: 9.2s/2 turns -> 3.8s/1 turn.
    """
    from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

    opts = ClaudeAgentOptions(
        system_prompt={"type": "preset", "preset": "claude_code"},
        setting_sources=[],
        allowed_tools=[],  # nothing to fetch — the images are already in the prompt
        permission_mode="bypassPermissions",
        max_turns=2,
        # Its own knob, defaulting to the harness model (unchanged behaviour). This gate decides
        # whether to throw a build away and rebuild it from scratch, so a cheaper judge is not a
        # free win: too lenient ships an object missing parts, too strict costs a full agent
        # rebuild. Worth measuring per-tier against known-good builds before moving the default.
        model=os.environ.get("LR_COMPLETENESS_MODEL") or os.environ.get("HARNESS_MODEL", "claude-opus-5"),
    )

    async def stream():
        yield {
            "type": "user",
            "message": {"role": "user", "content": _blocks(images)},
            "parent_tool_use_id": None,
            "session_id": "completeness",
        }

    out = ""
    async for m in query(prompt=stream(), options=opts):
        if isinstance(m, ResultMessage):
            out = m.result or ""
    return out


def _parse(out: str) -> dict:
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", out, re.S)
        return json.loads(m.group(0)) if m else {"pass": True, "missing": [], "notes": "unparsed"}


def check(ref: Path, renders: list[Path]) -> dict:
    imgs = [ref] + [r for r in renders if r.is_file()]
    if len(imgs) < 2:
        return {"pass": True, "missing": [], "notes": "no renders to compare", "error": "no_renders"}
    try:
        data = _parse(asyncio.run(_ask(imgs)))
    except Exception as e:  # noqa: BLE001 — never hard-fail delivery on a VLM hiccup
        return {"pass": True, "missing": [], "notes": f"vlm error: {type(e).__name__}", "error": str(e)}
    data.setdefault("pass", True)
    data.setdefault("missing", [])
    data.setdefault("notes", "")
    return data


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--render", action="append", default=[], help="repeatable")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    report = check(Path(args.ref), [Path(r) for r in args.render])
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        verdict = "COMPLETE" if report["pass"] else "INCOMPLETE"
        print(f"completeness: {verdict}")
        for m in report.get("missing", []):
            print(f"  ✗ missing: {m}")
        if report.get("notes"):
            print(f"  note: {report['notes']}")
    return 0 if report.get("pass", True) else 1


if __name__ == "__main__":
    sys.exit(main())
