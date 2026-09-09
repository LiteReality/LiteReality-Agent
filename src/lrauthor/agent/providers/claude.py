"""Claude Code harness — `claude_agent_sdk` against the logged-in `claude` CLI.

The reference implementation: capability tools run IN THIS PROCESS (no subprocess, no schema
round-trip), a `PreToolUse` hook can steer the session's ending, and the SDK reports what the
run cost. Everything the other harnesses have to approximate is native here.

The model is a separate knob. `spec.model` is passed through to the CLI verbatim, so an
Anthropic-compatible endpoint configured via `ANTHROPIC_BASE_URL` serves a non-Claude model
through this same harness, keeping hooks and in-process tools intact.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

from lrauthor.agent.providers.base import (
    AgentMessage,
    SessionResult,
    SessionSpec,
    normalise_blocks,
)

# The capability tools are the model's self-check and material sourcing (mcp__cap__render |
# critic | select_views | grid | fetch_material | check_collisions). In wind-down they are
# switched off so the remaining budget goes to FINAL Room.py edits; Read/Edit/Write/Glob stay
# on so the model can still land its work.
_EXPLORE_PREFIX = "mcp__cap__"


def build_capability_server(scene_path: Path, tool_names):
    """In-process MCP server exposing a SUBSET of the closed registry, scene-bound.

    Returns `(server, allowed_tool_names)`. The registry is the single source of truth for
    schemas — see `agent/tools/default_registry.py`.
    """
    from claude_agent_sdk import create_sdk_mcp_server
    from claude_agent_sdk import tool as sdk_tool

    from lrauthor.agent.tools import build_default_registry

    registry = build_default_registry()

    def _make_handler(name: str):
        async def _handler(args):
            inv = await registry.build_invocation(name, args or {})
            if inv is None:
                return {
                    "content": [{"type": "text", "text": f"unknown tool {name}"}],
                    "is_error": True,
                }
            if hasattr(inv, "bind"):
                inv.bind(str(scene_path), None)
            try:
                res = await inv.execute()
            except Exception as exc:  # noqa: BLE001
                return {
                    "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
                    "is_error": True,
                }
            return {
                "content": [{"type": "text", "text": json.dumps(res.to_dict(), default=str)}],
                "is_error": not res.is_success(),
            }

        return _handler

    sdk_tools = []
    for name in tool_names:
        decl = registry.get_tool(name)
        fn = decl.schema["function"]
        sdk_tools.append(sdk_tool(name, fn["description"], fn["parameters"])(_make_handler(name)))
    server = create_sdk_mcp_server(name="cap", version="1.0.0", tools=sdk_tools)
    allowed = [f"mcp__cap__{n}" for n in tool_names]
    return server, allowed


def _make_step_budget_hook(state: dict, budget: int, reserve: int, log):
    """PreToolUse hook that turns a `budget` of tool-calls into a graceful landing.

    The only pre-existing control was `--max-turns`, a hard SDK cap that just throws and abandons
    the loop (whatever compiling Room.py was last checkpointed is kept). This instead COUNTS every
    tool call and steers the ending:
      • last `reserve` steps  → DENY the self-check tools so the model spends what's left making its
        final Room.py edits, with a message telling it to wind down;
      • at `budget`           → end the session cleanly (Room.py is already edited in place).
    `state` carries the live count back out for the run summary.
    """
    finalize_at = max(1, budget - max(0, reserve))

    async def hook(input_data, tool_use_id, context):
        state["calls"] += 1
        n = state["calls"]
        tool = (
            input_data.get("tool_name", "")
            if isinstance(input_data, dict)
            else getattr(input_data, "tool_name", "") or ""
        )

        if n >= budget:
            if not state.get("hard_logged"):
                state["hard_logged"] = True
                state["stopped"] = f"step budget {budget} reached"
                log(f"\n  ⛔ step budget {budget} reached (call {n}) — ending the authoring session.")
            msg = (
                f"STEP BUDGET EXHAUSTED ({n}/{budget}). Stop now — do not call any more tools; "
                f"Room.py is saved as-is."
            )
            return {
                "continue_": False,
                "stopReason": msg,
                "systemMessage": f"authoring: step budget {budget} reached — ending run",
            }

        if n >= finalize_at:
            fin = (
                f"STEP BUDGET: step {n}/{budget}. WIND DOWN NOW — stop rendering / critiquing / "
                f"fetching. Use your remaining {max(0, budget - n)} step(s) to make ONLY your final "
                f"Room.py edits (the most important remaining fixes), then finish with your per-wall "
                f"summary. The self-check tools are disabled for the rest of the run."
            )
            if tool.startswith(_EXPLORE_PREFIX):
                if not state.get("winddown_logged"):
                    state["winddown_logged"] = True
                    log(
                        f"\n  ⏳ step {n}/{budget} — wind-down: self-check tools disabled, "
                        f"{max(0, budget - n)} step(s) left for final edits."
                    )
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": fin,
                    }
                }
            # An allowed edit/read during wind-down: nudge once so the model knows it's landing.
            if not state.get("winddown_ctx_sent"):
                state["winddown_ctx_sent"] = True
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "allow",
                        "additionalContext": fin,
                    }
                }
        return {}

    return hook


class ClaudeHarness:
    """`claude_agent_sdk` — the full-capability harness."""

    name = "claude"
    # `read_events` — file reads surface as observable `Read` tool calls. The stitch-coverage
    # guardrail in author.py is built on that; a harness that reads files internally cannot be
    # audited the same way and must not be reported as having skipped every reference.
    supports = frozenset(
        {"hooks", "inproc_tools", "cost", "skills", "restricted", "images", "read_events"}
    )

    def effective_model(self, spec: SessionSpec) -> str:
        return spec.model or "(claude CLI default)"

    async def run(self, spec: SessionSpec) -> AsyncIterator[AgentMessage | SessionResult]:
        from claude_agent_sdk import ClaudeAgentOptions, HookMatcher, ResultMessage, query

        allowed = list(spec.file_tools)
        servers = dict(spec.extra_mcp)
        allowed += list(spec.extra_allowed)
        if spec.capability_tools:
            server, cap_allowed = build_capability_server(spec.scene(), spec.capability_tools)
            servers["cap"] = server
            allowed += cap_allowed

        budget_state = {"calls": 0}
        hooks = None
        if spec.step_budget > 0:
            hooks = {
                "PreToolUse": [
                    HookMatcher(
                        hooks=[
                            _make_step_budget_hook(
                                budget_state, spec.step_budget, spec.step_reserve, spec.log
                            )
                        ]
                    )
                ]
            }

        options = ClaudeAgentOptions(
            cwd=str(spec.cwd),
            add_dirs=spec.roots(),
            system_prompt={"type": "preset", "preset": "claude_code"},
            setting_sources=list(spec.setting_sources),
            allowed_tools=allowed,
            mcp_servers=servers,
            permission_mode="bypassPermissions",
            max_turns=spec.max_turns,
            model=spec.model,
            hooks=hooks,
            **({"skills": list(spec.skills)} if spec.skills else {}),
            **({"max_budget_usd": spec.budget_usd} if spec.budget_usd else {}),
        )

        result: SessionResult | None = None
        async for message in query(prompt=spec.prompt, options=options):
            if isinstance(message, ResultMessage):
                result = SessionResult(
                    result=message.result or "",
                    total_cost_usd=getattr(message, "total_cost_usd", None),
                    is_error=bool(getattr(message, "is_error", False)),
                    num_turns=getattr(message, "num_turns", None),
                    raw=message,
                )
                continue
            blocks = getattr(message, "content", None)
            if blocks:
                yield AgentMessage(content=normalise_blocks(blocks), raw=message)

        # The hook's own stop is a deliberate, graceful end — surface it on the terminal message
        # so callers treat it as "time's up", not "this crashed".
        if result is None:
            result = SessionResult()
        result.stopped = budget_state.get("stopped", "")
        yield result
