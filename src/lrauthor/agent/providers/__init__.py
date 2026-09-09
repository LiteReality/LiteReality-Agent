"""Agent harnesses — which coding agent drives a file-editing session (Claude Code | Codex).

See `base.py` for the seam: `SessionSpec` in, a normalised event stream out, `resolve(role)` to
pick the harness from config.
"""

from lrauthor.agent.providers.base import (
    DEFAULT_PROVIDER,
    PROVIDERS,
    AgentHarness,
    AgentMessage,
    SessionResult,
    SessionSpec,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    describe,
    normalise_blocks,
    provider_name,
    resolve,
)

__all__ = [
    "DEFAULT_PROVIDER",
    "PROVIDERS",
    "AgentHarness",
    "AgentMessage",
    "SessionResult",
    "SessionSpec",
    "TextBlock",
    "ToolResultBlock",
    "ToolUseBlock",
    "describe",
    "normalise_blocks",
    "provider_name",
    "resolve",
]
