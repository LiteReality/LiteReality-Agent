"""Claude vision calls used by the authoring critic tool."""

from __future__ import annotations

import asyncio
import json
import os
import re

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
    """Read every image with the configured hosted Claude model."""
    from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

    from lrauthor.agent.tools.shared import config

    options = ClaudeAgentOptions(
        cwd=str(config.ROOT),
        add_dirs=[str(config.ROOT)],
        system_prompt={"type": "preset", "preset": "claude_code"},
        setting_sources=[],
        allowed_tools=["Read"],
        permission_mode="bypassPermissions",
        max_turns=max(4, len(images) + 2),
        model=config.MODEL,
    )
    lines = [
        f"- IMAGE {index + 1}{f' — {labels[index]}' if labels and labels[index] else ''}: {path}"
        for index, path in enumerate(images)
    ]
    request = "First Read every image, then answer.\n" + "\n".join(lines) + "\n\n" + prompt
    if json_mode:
        request += "\n\nReply with only JSON."
    output = ""
    async with _semaphore():
        async for message in query(prompt=request, options=options):
            if isinstance(message, ResultMessage):
                output = message.result or ""
    if not json_mode:
        return output
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}|\[.*\]", output, re.S)
        return json.loads(match.group(0)) if match else {}
