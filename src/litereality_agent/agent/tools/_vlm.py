"""Provider-routed vision calls used by the authoring critic and acceptance review."""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path

_SEMAPHORE = None


def _semaphore():
    global _SEMAPHORE
    if _SEMAPHORE is None:
        _SEMAPHORE = asyncio.Semaphore(int(os.environ.get("HARNESS_VLM_CONCURRENCY", "6")))
    return _SEMAPHORE


async def vision(
    images: list[str],
    prompt: str,
    *,
    json_mode: bool = False,
    labels: list[str] | None = None,
):
    """Use the quality role (GPT-6/Codex by default), never an implicit Claude fallback."""
    from litereality_agent.agent import providers
    from litereality_agent.agent.tools.shared import config

    lines = [
        f"- IMAGE {index + 1}{f' — {labels[index]}' if labels and labels[index] else ''}: {path}"
        for index, path in enumerate(images)
    ]
    request = "First Read every image, then answer.\n" + "\n".join(lines) + "\n\n" + prompt
    if json_mode:
        request += "\n\nReply with only JSON."
    request += "\nRead-only review: do not edit files or call another model."
    harness = providers.resolve("quality")
    spec = providers.SessionSpec(
        prompt=request, cwd=Path(config.ROOT),
        read_roots=tuple(Path(p).resolve().parent for p in images),
        file_tools=("Read",), setting_sources=(), model=config.MODEL,
        read_only=True, max_turns=max(8, len(images) + 4),
        step_budget=max(14, len(images) + 4), timeout_seconds=240,
    )
    output = ""
    finished = False
    async with _semaphore():
        async for message in harness.run(spec):
            if isinstance(message, providers.SessionResult):
                if message.is_error or message.stopped:
                    raise RuntimeError(f"visual review incomplete: {message.stopped or message.result}")
                finished = True
                output = message.result or ""
    if not finished or not output.strip():
        raise RuntimeError("visual review returned no completed result")
    if not json_mode:
        return output
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}|\[.*\]", output, re.S)
        if not match:
            raise ValueError("visual review did not return JSON")
        return json.loads(match.group(0))
