"""Which coding agent drives a file-editing session — Claude Code or Codex.

Every authoring pass in this repo is the same shape: put a model in a directory, give it
file-editing tools plus our capability tools (`fetch_material`, `render`, `critic`, …), stream
back what it did, and keep `Room.py` compiling. Only the *harness* underneath differs.

    harness   the agent loop: file tools, permission model, MCP wiring, hooks, event stream
    model     the brain inside it (`HARNESS_MODEL`, `LR_PROCEDURAL_MODEL`, …)

Those are independent knobs. This module is the seam for the first one; the model stays where
it already was. Selection is config, resolved by :func:`resolve`:

    LR_<ROLE>_PROVIDER  >  LR_AGENT_PROVIDER  >  "claude"

Harnesses are NOT feature-equivalent, and pretending otherwise is how a step budget silently
stops existing. Each declares what it can do in `supports`, and callers branch on that rather
than assuming:

    hooks          can deny/steer a tool call before it runs (the step budget's wind-down)
    inproc_tools   capability tools run in THIS process (no stdio MCP subprocess)
    cost           reports what the session cost
    skills         loads a `.claude/skills` package natively
    restricted     the tool allowlist is honoured (Codex always has shell access)

The event stream is normalised so consumers — `ToolNarrator`, `AgentTrace`, the stitch-coverage
check, the `Room.py` checkpointer — work unchanged against either harness. `AgentMessage.raw`
carries the harness's own object so the raw trace sidecar stays verbatim.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

# ── normalised event stream ────────────────────────────────────────────────────────────────
# Field names deliberately match the claude_agent_sdk block shapes: ToolNarrator and AgentTrace
# read them with getattr, so one vocabulary keeps both harnesses working through the same code.


@dataclass
class TextBlock:
    """Model prose / reasoning."""

    text: str


@dataclass
class ToolUseBlock:
    """A tool the model decided to call. `id` pairs it with its ToolResultBlock."""

    id: str
    name: str
    input: dict


@dataclass
class ToolResultBlock:
    """What a tool call answered. Carries `tool_use_id`, never the tool's name — see
    tool_narration.py for what reading a name off this block cost us."""

    tool_use_id: str
    content: Any
    is_error: bool = False


@dataclass
class AgentMessage:
    """One streamed message: a list of blocks, plus the harness's own object for the raw trace."""

    content: list = field(default_factory=list)
    raw: Any = None


@dataclass
class SessionResult:
    """Terminal message. `stopped` is non-empty when the HARNESS ended the run deliberately
    (step budget) rather than the model finishing — a graceful landing, not a failure."""

    result: str = ""
    total_cost_usd: float | None = None
    is_error: bool = False
    num_turns: int | None = None
    stopped: str = ""
    raw: Any = None


# ── what a session needs, harness-agnostic ─────────────────────────────────────────────────


@dataclass
class SessionSpec:
    """One file-editing agent session.

    `capability_tools` names entries of the closed tool registry (`agent/tools`). How they reach
    the model is the harness's problem: Claude Code gets them in-process, Codex over a stdio MCP
    subprocess. `capability_scene` is what those tools bind to — the room directory — and
    defaults to `cwd`.
    """

    prompt: str
    cwd: Path
    read_roots: tuple[Path, ...] = ()
    capability_tools: tuple[str, ...] = ()
    capability_scene: Path | None = None
    # A bespoke, session-scoped tool host (refine_objects' per-object `render_object`). Only a
    # harness advertising `inproc_tools` can serve one — there is no registry entry behind it to
    # rebuild in a subprocess. Callers must check `supports` before relying on it.
    extra_mcp: dict = field(default_factory=dict)
    extra_allowed: tuple[str, ...] = ()
    file_tools: tuple[str, ...] = ("Read", "Edit", "Write", "Glob")
    skills: tuple[str, ...] = ()
    setting_sources: tuple[str, ...] = ("project", "user")
    model: str | None = None
    max_turns: int = 80
    budget_usd: float | None = None
    # Graceful landing: at `step_budget` tool calls the session ends; for the last `step_reserve`
    # of them the capability tools are switched off so what remains goes to final edits.
    # 0 disables. Honoured natively where "hooks" is supported, emulated (hard stop) otherwise.
    step_budget: int = 0
    step_reserve: int = 0
    log: Callable[[str], None] = print

    def scene(self) -> Path:
        return self.capability_scene or self.cwd

    def roots(self) -> list[str]:
        return sorted({str(p) for p in self.read_roots})


class AgentHarness(Protocol):
    """A coding agent that can be pointed at a directory and streamed."""

    name: str
    supports: frozenset[str]

    def effective_model(self, spec: SessionSpec) -> str:
        """What this harness will ACTUALLY run. `spec.model` holds the role's model in Claude
        vocabulary, which Codex ignores — printing it there names a model that never ran."""
        ...

    def run(self, spec: SessionSpec) -> AsyncIterator[AgentMessage | SessionResult]: ...


# ── selection ──────────────────────────────────────────────────────────────────────────────

PROVIDERS = ("claude", "codex")
DEFAULT_PROVIDER = "claude"


def provider_name(role: str = "", override: str | None = None) -> str:
    """Resolve the harness NAME for a role: explicit > `LR_<ROLE>_PROVIDER` > `LR_AGENT_PROVIDER`.

    Role names match the knobs in `models.env` (`author`, `materials`, `quality`, `refine`,
    `procedural`). `LR_PROCEDURAL_PROVIDER` predates this module and keeps working unchanged.
    """
    candidates = [override]
    if role:
        candidates.append(os.environ.get(f"LR_{role.upper()}_PROVIDER"))
    candidates.append(os.environ.get("LR_AGENT_PROVIDER"))
    for value in candidates:
        if value and value.strip():
            chosen = value.strip().lower()
            if chosen not in PROVIDERS:
                raise ValueError(
                    f"unsupported agent provider {chosen!r} (expected one of {', '.join(PROVIDERS)})"
                )
            return chosen
    return DEFAULT_PROVIDER


def resolve(role: str = "", override: str | None = None) -> AgentHarness:
    """The harness for a role. Import is lazy so selecting Codex needs no claude_agent_sdk."""
    chosen = provider_name(role, override)
    if chosen == "codex":
        from lrauthor.agent.providers.codex import CodexHarness

        return CodexHarness()
    from lrauthor.agent.providers.claude import ClaudeHarness

    return ClaudeHarness()


def describe(harness: AgentHarness, spec: SessionSpec) -> str:
    """One line for the stage header, and a WARNING where the harness cannot honour the spec.

    A missing capability must be visible in the log. The step budget is the case that motivated
    this: without hooks it degrades from "wind down, then stop" to a hard stop, and a run that
    quietly lost its graceful landing looks identical to one that never had it.
    """
    parts = [f"provider={harness.name}", f"model={harness.effective_model(spec)}"]
    if spec.step_budget > 0:
        native = "hooks" in harness.supports
        parts.append(
            f"step budget={spec.step_budget}"
            + (
                f" (wind-down at {max(1, spec.step_budget - max(0, spec.step_reserve))})"
                if native
                else f" (HARD STOP — {harness.name} has no pre-tool hook, no wind-down)"
            )
        )
    else:
        parts.append("step budget=off")
    if spec.capability_tools and "inproc_tools" not in harness.supports:
        parts.append("tools=stdio MCP")
    if spec.file_tools and "restricted" not in harness.supports:
        parts.append("WARNING: tool allowlist not enforced (shell access is always on)")
    return "  |  ".join(parts)


def normalise_blocks(blocks: Sequence[Any]) -> list[Any]:
    """Map harness-native content blocks onto our vocabulary, passing unknown kinds through.

    Unknown blocks are kept rather than dropped: consumers ignore what they do not recognise,
    but the raw trace should never lose something because this mapping had not heard of it.
    """
    out: list[Any] = []
    for block in blocks or []:
        kind = type(block).__name__
        if kind == "TextBlock":
            out.append(TextBlock(text=getattr(block, "text", "") or ""))
        elif kind == "ToolUseBlock":
            out.append(
                ToolUseBlock(
                    id=getattr(block, "id", "") or "",
                    name=getattr(block, "name", "") or "",
                    input=getattr(block, "input", {}) or {},
                )
            )
        elif kind == "ToolResultBlock":
            out.append(
                ToolResultBlock(
                    tool_use_id=getattr(block, "tool_use_id", "") or "",
                    content=getattr(block, "content", None),
                    is_error=bool(getattr(block, "is_error", False)),
                )
            )
        else:
            out.append(block)
    return out
