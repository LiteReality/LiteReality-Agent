"""Codex harness — the `codex` CLI driven non-interactively via `codex exec --json`.

Codex runs as a SEPARATE PROGRAM, and every difference from the Claude Code harness follows from
that one fact:

  • capability tools     reached over a stdio MCP subprocess (`agent/tools/mcp_server.py`), not
                         in-process. Same registry, same scene binding — one pipe further away.
  • no pre-tool hook     the step budget cannot wind down gracefully. It is emulated by counting
                         tool events and terminating the session at the cap. `Room.py` is edited
                         in place and checkpointed per edit, so a hard stop costs one edit — but
                         the "spend your last N steps on final edits" behaviour is simply gone,
                         and `providers.describe()` says so in the stage header.
  • no tool allowlist    Codex always has shell access. A session that Claude Code would run with
                         Read/Edit/Write/Glob only cannot be restricted the same way here.
  • no cost              `codex exec` does not report spend; `SessionResult.total_cost_usd` is None.
  • no skills            there is no skill system; callers inject the playbook as prompt text
                         (see `models/object_generation/generate.py:_skill_instructions`).

EVENT MAPPING IS VERSION-SENSITIVE. `codex exec --json` has shipped two families of event shape:
a nested `{"msg": {"type": ...}}` form and a flatter `{"type": "item.completed", "item": {...}}`
form. Both are handled below; anything unrecognised degrades to a text block rather than being
dropped. The flat family was verified against codex-cli 0.146.0, which emits:

    {"type":"thread.started","thread_id":…}
    {"type":"turn.started"}
    {"type":"item.started",  "item":{"id":"item_0","type":"command_execution",
                                     "command":"/bin/zsh -lc 'cat note.txt'","exit_code":null,…}}
    {"type":"item.completed","item":{"id":"item_0","type":"command_execution",
                                     "aggregated_output":"hello\\n","exit_code":0,…}}
    {"type":"item.completed","item":{"id":"item_1","type":"agent_message","text":"hello"}}
    {"type":"turn.completed","usage":{"input_tokens":…,"output_tokens":…}}

If a Codex run traces as prose with no tool calls, the shape has moved: run
`codex exec --json --skip-git-repo-check 'say hi'` and compare against `_normalise`.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path
from collections.abc import AsyncIterator

from litereality_agent.agent.providers.base import (
    AgentMessage,
    SessionResult,
    SessionSpec,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)

# Codex names its own file-edit and shell actions differently from Claude Code. Callers key off
# the Claude vocabulary — `author.py` checkpoints `Room.py` when it sees an Edit/Write/Bash
# result — so the mapping normalises INTO that vocabulary rather than inventing a third one.
_EDIT_TOOL = "Edit"
_SHELL_TOOL = "Bash"

# One `codex exec --json` event is one line, and an event carries whole command output — a
# `cat Room.py` or a stitch listing runs to hundreds of KB. asyncio's StreamReader defaults to a
# 64 KiB line cap and raises `ValueError: Separator is found, but chunk is longer than limit` on
# the first line over it, which killed the session a couple of tool calls in and reported success.
# The cap has to clear the largest event a session can emit, not the typical one.
_EVENT_LINE_LIMIT = 64 * 1024 * 1024


def _toml(value) -> str:
    """A TOML scalar/array literal for `codex -c key=<value>`. JSON is valid TOML for these."""
    return json.dumps(value)


def _mcp_config(spec: SessionSpec) -> list[str]:
    """`-c` overrides registering our capability tools as a Codex MCP server.

    The server is spawned with THIS interpreter, so it imports the same `litereality_agent` the
    pipeline is running from — not whatever `python` resolves to on Codex's PATH.
    """
    if not spec.capability_tools:
        return []
    args = [
        "-m",
        "litereality_agent.agent.tools.mcp_server",
        "--scene",
        str(spec.scene()),
        "--tools",
        ",".join(spec.capability_tools),
    ]
    return [
        "-c",
        f"mcp_servers.cap.command={_toml(sys.executable)}",
        "-c",
        f"mcp_servers.cap.args={_toml(args)}",
    ]


def _text_of(value) -> str:
    """Flatten a Codex field to text, unwrapping the MCP result envelope.

    An `mcp_tool_call` result arrives as the raw MCP payload —
    `{"content": [{"type": "text", "text": "<our json>"}], "isError": false}` — and stringifying
    that verbatim buried our tool's actual output inside a Python dict repr, which is what both
    the live trace line and the model's own view of the result would then show.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if "content" in value:
            return _text_of(value["content"])
        if "text" in value:
            return _text_of(value["text"])
        return json.dumps(value, default=str)
    if isinstance(value, list):
        return " ".join(
            _text_of(v.get("text", "")) if isinstance(v, dict) else str(v) for v in value
        )
    return "" if value is None else str(value)


def _mcp_failed(item: dict) -> bool:
    """A tool call fails either at the Codex layer (`status`) or inside MCP (`isError`)."""
    if (item.get("status") or "") == "failed":
        return True
    result = item.get("result")
    return bool(isinstance(result, dict) and result.get("isError"))


def _normalise(event: dict, counter: dict) -> list:
    """One Codex JSON event → our blocks. Unknown shapes become a TextBlock, never nothing.

    `counter` supplies monotonic ids so a use block and its result can be paired the way
    `ToolNarrator` and `AgentTrace` expect (Codex's own call ids are not present on every shape).
    """

    def next_id(prefix: str) -> str:
        counter["n"] = counter.get("n", 0) + 1
        return f"{prefix}_{counter['n']:04d}"

    # Family A: {"msg": {"type": ...}} — the older experimental stream.
    msg = event.get("msg")
    if isinstance(msg, dict):
        kind = msg.get("type") or ""
        if kind in ("agent_message", "agent_reasoning"):
            return [TextBlock(text=_text_of(msg.get("message") or msg.get("text")))]
        if kind == "exec_command_begin":
            command = msg.get("command")
            command = " ".join(command) if isinstance(command, list) else _text_of(command)
            call_id = msg.get("call_id") or next_id("cmd")
            counter[call_id] = _SHELL_TOOL
            return [ToolUseBlock(id=call_id, name=_SHELL_TOOL, input={"command": command})]
        if kind == "exec_command_end":
            call_id = msg.get("call_id") or next_id("cmd")
            return [
                ToolResultBlock(
                    tool_use_id=call_id,
                    content=_text_of(msg.get("stdout") or msg.get("aggregated_output")),
                    is_error=bool(msg.get("exit_code")),
                )
            ]
        if kind == "patch_apply_begin":
            blocks = []
            changes = msg.get("changes") or {}
            paths = changes.keys() if isinstance(changes, dict) else changes
            for path in paths:
                blocks.append(
                    ToolUseBlock(
                        id=next_id("edit"), name=_EDIT_TOOL, input={"file_path": str(path)}
                    )
                )
            return blocks
        if kind == "task_complete":
            return [TextBlock(text=_text_of(msg.get("last_agent_message")))]
        return []

    # Family B: {"type": "item.completed", "item": {...}} — the current stream.
    etype = event.get("type") or ""
    item = event.get("item")
    if etype.startswith("item.") and isinstance(item, dict):
        itype = item.get("type") or ""
        started = etype == "item.started"
        if itype in ("agent_message", "reasoning"):
            return [TextBlock(text=_text_of(item.get("text")))] if not started else []
        if itype == "command_execution":
            call_id = item.get("id") or next_id("cmd")
            command = item.get("command")
            command = " ".join(command) if isinstance(command, list) else _text_of(command)
            if started:
                return [ToolUseBlock(id=call_id, name=_SHELL_TOOL, input={"command": command})]
            return [
                ToolResultBlock(
                    tool_use_id=call_id,
                    content=_text_of(item.get("aggregated_output")),
                    is_error=bool(item.get("exit_code")),
                )
            ]
        if itype == "file_change":
            if started:
                return []
            blocks = []
            for change in item.get("changes") or []:
                path = change.get("path") if isinstance(change, dict) else change
                cid = next_id("edit")
                blocks.append(
                    ToolUseBlock(id=cid, name=_EDIT_TOOL, input={"file_path": str(path)})
                )
                blocks.append(ToolResultBlock(tool_use_id=cid, content="applied"))
            return blocks
        if itype == "mcp_tool_call":
            call_id = item.get("id") or next_id("mcp")
            name = f"mcp__{item.get('server') or 'cap'}__{item.get('tool') or '?'}"
            if started:
                return [
                    ToolUseBlock(id=call_id, name=name, input=item.get("arguments") or {})
                ]
            return [
                ToolResultBlock(
                    tool_use_id=call_id,
                    content=_text_of(item.get("result") or item.get("output")),
                    is_error=_mcp_failed(item),
                )
            ]
        return []

    if etype == "turn.failed":
        error = event.get("error") or {}
        return [TextBlock(text=f"[codex error] {_text_of(error.get('message') or error)}")]
    # thread.started / turn.started / turn.completed carry no per-step content; `run` reads
    # turn.completed's token usage directly.
    return []


class CodexHarness:
    """`codex exec` — out-of-process, capability tools bridged over stdio MCP."""

    name = "codex"
    supports = frozenset({"images"})

    def effective_model(self, spec: SessionSpec) -> str:
        # Deliberately NOT `spec.model`: that is the role's Claude-vocabulary model, which this
        # harness ignores. Reporting it would name a model that never ran.
        return _codex_model() or "(codex default)"

    async def run(self, spec: SessionSpec) -> AsyncIterator[AgentMessage | SessionResult]:
        if shutil.which("codex") is None:
            raise RuntimeError(
                "codex CLI not on PATH — install it (`npm install -g @openai/codex`) and sign in, "
                "or set LR_AGENT_PROVIDER=claude."
            )

        cmd = [
            "codex",
            "exec",
            "--json",
            "--cd",
            str(spec.cwd),
            # A room directory is not a git checkout, and without this Codex refuses to start at
            # all: "Not inside a trusted directory and --skip-git-repo-check was not specified."
            "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox",
            "-c",
            f"model_reasoning_effort={_toml(_effort())}",
            *_mcp_config(spec),
        ]
        # Claude Code's `add_dirs` equivalent. Note these become WRITABLE to Codex — there is no
        # read-only variant — so the roots a session is given are wider here than on Claude Code.
        for root in spec.roots():
            if root != str(spec.cwd):
                cmd += ["--add-dir", root]
        # `spec.model` carries the ROLE's model in Claude vocabulary (`claude-opus-5`), which is
        # meaningless to Codex — passing it through would fail the run. Codex takes its model from
        # `LR_CODEX_MODEL`, or its own default when that is unset.
        model = _codex_model()
        if model:
            cmd += ["-m", model]
        cmd.append(spec.prompt)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            # The prompt goes as an argument; leaving stdin inherited makes `codex exec` block
            # reading it ("Reading additional input from stdin…") and the session never starts.
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=_EVENT_LINE_LIMIT,
        )

        counter: dict = {}
        calls = 0
        turns = 0
        stopped = ""
        summary = ""
        unparsed = 0
        matched = 0
        usage: dict = {}
        thread_id = ""

        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            raw = line.decode(errors="replace").strip()
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                # Not every line is an event (banners, progress). Keep it as prose so the
                # trace still shows what the session said.
                unparsed += 1
                yield AgentMessage(content=[TextBlock(text=raw)], raw=raw)
                continue
            if event.get("type") == "turn.completed":
                turns += 1
                usage = event.get("usage") or usage
            if event.get("type") == "thread.started":
                thread_id = str(event.get("thread_id") or "")
            blocks = _normalise(event, counter)
            if blocks:
                matched += 1
            for block in blocks:
                if isinstance(block, TextBlock) and block.text.strip():
                    summary = block.text
            if blocks:
                yield AgentMessage(content=blocks, raw=event)

            # Step budget, emulated: no hook exists, so the only lever is ending the session.
            calls += sum(1 for b in blocks if isinstance(b, ToolUseBlock))
            if spec.step_budget > 0 and calls >= spec.step_budget and not stopped:
                stopped = f"step budget {spec.step_budget} reached"
                spec.log(
                    f"\n  ⛔ step budget {spec.step_budget} reached ({calls} tool-calls) — "
                    f"terminating the codex session (no wind-down: codex has no pre-tool hook)."
                )
                proc.terminate()
                break

        err = b""
        if proc.stderr is not None:
            err = await proc.stderr.read()
        await proc.wait()

        if matched == 0 and unparsed == 0:
            spec.log(
                "  ⚠ codex produced no recognisable events — the --json event shape may have "
                "changed; see agent/providers/codex.py:_normalise"
            )

        failed = proc.returncode not in (0, None) and not stopped
        yield SessionResult(
            result=summary,
            # `codex exec` reports TOKENS, not spend, and the per-token price depends on the
            # account's plan — deriving a dollar figure here would be a guess presented as a fact.
            total_cost_usd=None,
            is_error=failed,
            num_turns=turns or None,
            stopped=stopped,
            raw={
                "returncode": proc.returncode,
                "usage": usage,
                "stderr": err.decode(errors="replace")[-2000:],
                "thread_id": thread_id,
                # Codex's own transcript: every message, tool call, tool output and — unlike our
                # normalised view — every image content item the model was actually shown.
                "rollout": rollout_path(thread_id),
            },
        )


def _effort() -> str:
    # Higher reasoning effort is noticeably better on geometry/material work — see the note in
    # models/object_generation/generate.py where this default came from.
    return os.environ.get("LR_CODEX_EFFORT", "high")


def rollout_path(thread_id: str) -> str | None:
    """Codex writes `~/.codex/sessions/YYYY/MM/DD/rollout-<stamp>-<thread_id>.jsonl`. Find it."""
    if not thread_id:
        return None
    root = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex") / "sessions"
    if not root.is_dir():
        return None
    hits = sorted(root.glob(f"*/*/*/rollout-*-{thread_id}.jsonl"))
    return str(hits[-1]) if hits else None


def _codex_model() -> str | None:
    """`LR_CODEX_MODEL`, or None to let `codex` pick its own default."""
    value = (os.environ.get("LR_CODEX_MODEL") or "").strip()
    return value or None
