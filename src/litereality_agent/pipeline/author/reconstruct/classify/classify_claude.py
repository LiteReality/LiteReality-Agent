"""Hosted Claude vision classifier.

Runs on the logged-in Claude CLI via ``claude_agent_sdk`` (read-only, one-shot).

Deliberately does NOT import the harness (``litereality_agent.agent.tools.shared.config`` requires ``$LITEREALITY_SCAN``);
init must stay decoupled. It reproduces the minimal Claude-vision call inline, the same way
:mod:`openai_classify` talks to OpenAI directly.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path

from litereality_agent import REPO_ROOT as _ROOT  # the checkout — the agent works from there


def _model() -> str:
    return os.environ.get("HARNESS_MODEL", "claude-opus-5")


async def _ask(image_path: Path, prompt: str) -> str:
    from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

    opts = ClaudeAgentOptions(
        cwd=str(_ROOT),
        add_dirs=[str(_ROOT), str(image_path.parent)],
        system_prompt={"type": "preset", "preset": "claude_code"},
        setting_sources=[],  # one-shot grade: skip project/user config (faster, reproducible)
        allowed_tools=["Read"],
        permission_mode="bypassPermissions",
        max_turns=4,
        model=_model(),
    )
    full = (
        f"First Read (view) this image, then answer.\n- IMAGE: {image_path}\n\n{prompt}\n\n"
        "Reply with ONLY the JSON object — no prose, no code fences."
    )
    out = ""
    async for m in query(prompt=full, options=opts):
        if isinstance(m, ResultMessage):
            out = m.result or ""
    return out


def _parse(out: str) -> dict:
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", out, re.S)
        if m:
            return json.loads(m.group(0))
        raise


def classify_image(
    image_path: Path,
    prompt: str,
    response_schema: dict,
    *,
    model: str | None = None,  # accepted for signature parity; ignored (model via HARNESS_MODEL)
    api_key: str | None = None,  # unused — Claude runs via the logged-in CLI
    timeout: int = 60,
    retries: int = 3,
) -> dict:
    """Send one image + prompt to Claude vision, return the parsed JSON decision."""
    image_path = Path(image_path)
    keys = ", ".join(response_schema.get("properties", {}).keys()) or "route, reason"
    full_prompt = (
        f"{prompt}\n\nReply with ONLY a JSON object containing exactly these keys: {keys}."
    )

    last_err = None
    for _ in range(retries):
        try:
            data = _parse(asyncio.run(_ask(image_path, full_prompt)))
            if not isinstance(data, dict):
                raise RuntimeError(f"non-dict reply: {type(data).__name__}")
            for req in response_schema.get("required", []):
                data.setdefault(req, None)
            return data
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"

    raise RuntimeError(f"Claude classify failed after {retries} tries: {last_err}")
