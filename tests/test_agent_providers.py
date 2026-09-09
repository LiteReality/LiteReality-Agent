"""The agent-harness seam: config selection, event normalisation, and the stdio tool bridge.

Three things have to hold for Claude Code and Codex to be interchangeable, and each has a way of
failing silently:

1. SELECTION follows config. A typo in `LR_AGENT_PROVIDER` must fail loudly, not fall back to the
   default and run the wrong agent for two paid hours.
2. The NORMALISED blocks stay duck-type compatible with `ToolNarrator` / `AgentTrace`. Those read
   everything with `getattr`, so a renamed field does not raise — it silently narrates `?` and
   traces nothing. tool_narration.py's own docstring records what that cost the last time.
3. Codex's file edits map onto the `Edit` vocabulary. `author.py` checkpoints the last COMPILING
   `Room.py` when it sees an Edit/Write/Bash result; if Codex's edits arrive under any other name
   the checkpoint never fires and an interrupted session loses the whole run instead of one edit.
"""

from __future__ import annotations

import asyncio
import sys

import pytest

from lrauthor.agent import providers
from lrauthor.agent.providers import base
from lrauthor.agent.providers.codex import _normalise

# ── 1. selection ───────────────────────────────────────────────────────────────────────────


def test_defaults_to_claude(monkeypatch):
    monkeypatch.delenv("LR_AGENT_PROVIDER", raising=False)
    monkeypatch.delenv("LR_AUTHOR_PROVIDER", raising=False)
    assert providers.provider_name("author") == "claude"


def test_role_override_beats_global(monkeypatch):
    monkeypatch.setenv("LR_AGENT_PROVIDER", "codex")
    monkeypatch.setenv("LR_AUTHOR_PROVIDER", "claude")
    assert providers.provider_name("author") == "claude"
    assert providers.provider_name("materials") == "codex"


def test_explicit_argument_beats_environment(monkeypatch):
    monkeypatch.setenv("LR_AGENT_PROVIDER", "codex")
    assert providers.provider_name("author", "claude") == "claude"


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("LR_AGENT_PROVIDER", "codexx")
    with pytest.raises(ValueError, match="unsupported agent provider"):
        providers.provider_name("author")


def test_settings_reject_unknown_provider(monkeypatch):
    """The typo must fail at the composition boundary, not inside a paid session."""
    from lrauthor.settings import load_settings

    monkeypatch.setenv("LR_AGENT_PROVIDER", "gpt")
    with pytest.raises(Exception, match="unsupported agent provider"):
        load_settings()


def test_settings_roles_inherit_the_global_provider(monkeypatch):
    from lrauthor.settings import load_settings

    monkeypatch.setenv("LR_AGENT_PROVIDER", "codex")
    monkeypatch.setenv("LR_AUTHOR_PROVIDER", "claude")
    settings = load_settings()
    assert settings.author_provider == "claude"
    assert settings.materials_provider == "codex"
    assert settings.as_environment()["LR_QUALITY_PROVIDER"] == "codex"


def test_unexpanded_dotenv_default_is_treated_as_unset(monkeypatch):
    """`models.env` is read by python-dotenv, which leaves `${A:-$B}` as the literal `$B`.

    `critic_model` already carries a workaround for this; a provider arriving as `$lr_agent_provider`
    must read as "unset", not as an invalid harness name.
    """
    from lrauthor.settings import LiteRealitySettings

    assert LiteRealitySettings._checked_provider("$LR_AGENT_PROVIDER") is None
    assert LiteRealitySettings._checked_provider("  ") is None
    assert LiteRealitySettings._checked_provider("CODEX") == "codex"


# ── 2. normalised blocks stay compatible with the existing consumers ────────────────────────


def test_blocks_narrate_and_attribute_like_sdk_blocks():
    from lrauthor.agent.tool_narration import ToolNarrator

    nar = ToolNarrator()
    line = nar.use(base.ToolUseBlock(id="t1", name="mcp__cap__render", input={"target": "Wall0"}))
    assert "render" in line and "Wall0" in line
    assert nar.calls == 1 and nar.counts == {"render": 1}

    name, inp = nar.result(base.ToolResultBlock(tool_use_id="t1", content="ok"))
    assert name == "render" and inp == {"target": "Wall0"}


def test_error_results_are_reported():
    from lrauthor.agent.tool_narration import ToolNarrator

    nar = ToolNarrator()
    nar.use(base.ToolUseBlock(id="t1", name="Edit", input={"file_path": "/x/Room.py"}))
    block = base.ToolResultBlock(tool_use_id="t1", content="boom", is_error=True)
    name, _ = nar.result(block)
    assert nar.error_line(name, block) == "      ↳ Edit failed: boom"


def test_trace_records_normalised_blocks(tmp_path):
    """`AgentTrace` walks dataclasses for the raw sidecar — our blocks must survive that."""
    from lrauthor.agent.trace import AgentTrace

    tr = AgentTrace("test", room=tmp_path)
    tr.start(model="m")
    tr.raw(base.AgentMessage(content=[base.TextBlock(text="hello")]))
    tr.tool("mcp__cap__render", {"target": "Wall0"}, tool_id="t1")
    tr.result(base.ToolResultBlock(tool_use_id="t1", content="done"))
    tr.end(calls=1)
    lines = [ln for ln in tr.path.read_text().splitlines() if ln.strip()]
    kinds = [__import__("json").loads(ln)["kind"] for ln in lines]
    assert "tool" in kinds and "result" in kinds and "session_end" in kinds


def test_claude_blocks_are_mapped_not_dropped():
    """Unknown block kinds pass through rather than vanishing from the stream."""

    class ThinkingBlock:
        def __init__(self):
            self.thinking = "…"

    class TextBlock:
        def __init__(self):
            self.text = "hi"

    out = base.normalise_blocks([TextBlock(), ThinkingBlock()])
    assert isinstance(out[0], base.TextBlock) and out[0].text == "hi"
    assert type(out[1]).__name__ == "ThinkingBlock"


# ── 3. codex event mapping ─────────────────────────────────────────────────────────────────


def test_codex_current_stream_maps_tool_calls():
    counter: dict = {}
    started = _normalise(
        {"type": "item.started", "item": {"id": "c1", "type": "command_execution",
                                          "command": ["bash", "-lc", "ls"]}}, counter)
    assert isinstance(started[0], base.ToolUseBlock)
    assert started[0].name == "Bash" and started[0].id == "c1"

    done = _normalise(
        {"type": "item.completed", "item": {"id": "c1", "type": "command_execution",
                                            "aggregated_output": "Room.py", "exit_code": 0}}, counter)
    assert isinstance(done[0], base.ToolResultBlock)
    assert done[0].tool_use_id == "c1" and done[0].is_error is False


def test_codex_file_changes_map_to_the_edit_vocabulary():
    """This is what keeps `author.py`'s Room.py checkpointing alive on the codex harness."""
    blocks = _normalise(
        {"type": "item.completed",
         "item": {"type": "file_change", "changes": [{"path": "/room/Room.py", "kind": "modify"}]}},
        {})
    uses = [b for b in blocks if isinstance(b, base.ToolUseBlock)]
    results = [b for b in blocks if isinstance(b, base.ToolResultBlock)]
    assert uses and uses[0].name == "Edit"
    assert uses[0].input["file_path"] == "/room/Room.py"
    assert results and results[0].tool_use_id == uses[0].id


def test_codex_mcp_calls_keep_the_cap_prefix():
    """`tool_label` splits on `__`, so capability tools must arrive fully qualified."""
    from lrauthor.agent.tool_narration import tool_label

    blocks = _normalise(
        {"type": "item.started",
         "item": {"id": "m1", "type": "mcp_tool_call", "server": "cap", "tool": "fetch_material",
                  "arguments": {"query": "oak"}}}, {})
    assert blocks[0].name == "mcp__cap__fetch_material"
    assert tool_label(blocks[0].name) == "fetch_material"


def test_codex_legacy_stream_is_also_handled():
    counter: dict = {}
    use = _normalise({"msg": {"type": "exec_command_begin", "call_id": "x",
                              "command": ["ls", "-l"]}}, counter)
    assert use[0].name == "Bash" and use[0].input["command"] == "ls -l"
    res = _normalise({"msg": {"type": "exec_command_end", "call_id": "x",
                              "stdout": "out", "exit_code": 1}}, counter)
    assert res[0].is_error is True
    text = _normalise({"msg": {"type": "agent_message", "message": "done"}}, counter)
    assert isinstance(text[0], base.TextBlock)


def test_codex_unknown_events_do_not_raise():
    assert _normalise({"type": "thread.started", "thread_id": "t"}, {}) == []
    assert _normalise({}, {}) == []
    assert _normalise({"type": "turn.failed", "error": {"message": "nope"}}, {})[0].text


def test_codex_without_the_cli_fails_with_an_actionable_message(monkeypatch):
    from lrauthor.agent.providers.codex import CodexHarness

    monkeypatch.setattr("shutil.which", lambda _: None)
    spec = base.SessionSpec(prompt="hi", cwd=".")

    async def drain():
        async for _ in CodexHarness().run(spec):
            pass

    with pytest.raises(RuntimeError, match="codex CLI not on PATH"):
        asyncio.run(drain())


# ── capability reporting ───────────────────────────────────────────────────────────────────


def test_describe_flags_a_degraded_step_budget():
    from lrauthor.agent.providers.claude import ClaudeHarness
    from lrauthor.agent.providers.codex import CodexHarness

    spec = base.SessionSpec(prompt="p", cwd=".", step_budget=100, step_reserve=15,
                            capability_tools=("render",))
    claude = providers.describe(ClaudeHarness(), spec)
    codex = providers.describe(CodexHarness(), spec)
    assert "wind-down at 85" in claude
    assert "HARD STOP" in codex, "a lost graceful landing must be visible in the log"
    assert "stdio MCP" in codex
    assert "tool allowlist not enforced" in codex


def test_claude_supports_everything_the_call_sites_rely_on():
    from lrauthor.agent.providers.claude import ClaudeHarness

    assert {"hooks", "inproc_tools", "cost", "skills", "read_events"} <= ClaudeHarness().supports


# ── the stdio bridge serves the same tools as the in-process server ─────────────────────────


def test_stdio_bridge_matches_the_registry():
    """Codex reaches the capability tools only through this server; a schema that drifts from the
    registry is a tool the model calls wrongly with no local error to catch it."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    from lrauthor.agent.author import CAPABILITY_TOOLS
    from lrauthor.agent.tools import build_default_registry

    async def listing():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "lrauthor.agent.tools.mcp_server", "--scene", "/tmp/room"],
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return (await session.list_tools()).tools

    served = asyncio.run(asyncio.wait_for(listing(), 60))
    registry = build_default_registry()
    assert {t.name for t in served} == set(CAPABILITY_TOOLS)
    for tool in served:
        expected = registry.get_tool(tool.name).schema["function"]
        assert tool.description == expected["description"]
        assert tool.inputSchema == expected["parameters"]


# ── the claude path still builds the session it always did ─────────────────────────────────


def _captured_options(spec, messages=()):
    """Run the claude harness with `query` stubbed; return the ClaudeAgentOptions it built."""
    import claude_agent_sdk

    from lrauthor.agent.providers.claude import ClaudeHarness

    seen = {}

    def fake_query(prompt, options):
        seen["prompt"] = prompt
        seen["options"] = options

        async def stream():
            for message in messages:
                yield message

        return stream()

    original = claude_agent_sdk.query
    claude_agent_sdk.query = fake_query
    try:

        async def drain():
            return [m async for m in ClaudeHarness().run(spec)]

        seen["out"] = asyncio.run(drain())
    finally:
        claude_agent_sdk.query = original
    return seen


def test_claude_session_matches_the_pre_refactor_shape(tmp_path):
    """The authoring session's wiring is load-bearing: the room as cwd, the stitch/scan/repo roots
    readable, ONLY Read/Edit/Write/Glob plus the six `mcp__cap__*` tools, and the step-budget hook
    installed. Rebuilding that from a SessionSpec must produce the same thing."""
    from lrauthor.agent.author import CAPABILITY_TOOLS

    spec = base.SessionSpec(
        prompt="author the room",
        cwd=tmp_path / "room",
        read_roots=(tmp_path / "scan", tmp_path / "refs"),
        capability_tools=CAPABILITY_TOOLS,
        model="claude-opus-5",
        max_turns=140,
        step_budget=100,
        step_reserve=15,
    )
    options = _captured_options(spec)["options"]

    assert options.cwd == str(tmp_path / "room")
    assert options.add_dirs == sorted([str(tmp_path / "scan"), str(tmp_path / "refs")])
    assert options.allowed_tools == ["Read", "Edit", "Write", "Glob"] + [
        f"mcp__cap__{n}" for n in CAPABILITY_TOOLS
    ]
    assert set(options.mcp_servers) == {"cap"}
    assert options.permission_mode == "bypassPermissions"
    assert options.model == "claude-opus-5"
    assert options.max_turns == 140
    assert options.setting_sources == ["project", "user"]
    assert options.system_prompt == {"type": "preset", "preset": "claude_code"}
    assert "PreToolUse" in options.hooks


def test_no_step_budget_installs_no_hook(tmp_path):
    spec = base.SessionSpec(prompt="p", cwd=tmp_path, step_budget=0)
    assert not _captured_options(spec)["options"].hooks


def test_bespoke_tool_hosts_are_merged(tmp_path):
    """refine_objects hands in its own per-object `render_object` server."""
    spec = base.SessionSpec(
        prompt="p",
        cwd=tmp_path,
        extra_mcp={"obj": object()},
        extra_allowed=("mcp__obj__render_object",),
    )
    options = _captured_options(spec)["options"]
    assert set(options.mcp_servers) == {"obj"}
    assert "mcp__obj__render_object" in options.allowed_tools


def test_session_result_carries_cost_and_summary(tmp_path):
    from claude_agent_sdk import ResultMessage

    result = ResultMessage(
        subtype="success", duration_ms=1, duration_api_ms=1, is_error=False, num_turns=3,
        session_id="s", total_cost_usd=1.25, result="all done",
    )
    out = _captured_options(base.SessionSpec(prompt="p", cwd=tmp_path), [result])["out"]
    final = out[-1]
    assert isinstance(final, base.SessionResult)
    assert final.result == "all done" and final.total_cost_usd == 1.25
    assert final.num_turns == 3 and final.stopped == ""


def test_step_budget_hook_winds_down_then_stops(tmp_path):
    """The graceful landing: capability tools denied in the reserve, clean stop at the cap."""
    from lrauthor.agent.providers.claude import _make_step_budget_hook

    state = {"calls": 0}
    hook = _make_step_budget_hook(state, budget=5, reserve=2, log=lambda _: None)

    async def call(tool):
        return await hook({"tool_name": tool}, "id", None)

    assert asyncio.run(call("Edit")) == {}  # 1
    assert asyncio.run(call("mcp__cap__render")) == {}  # 2

    denied = asyncio.run(call("mcp__cap__render"))  # 3 — inside the reserve
    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "WIND DOWN" in denied["hookSpecificOutput"]["permissionDecisionReason"]

    allowed = asyncio.run(call("Edit"))  # 4 — edits still allowed while landing
    assert allowed["hookSpecificOutput"]["permissionDecision"] == "allow"

    stop = asyncio.run(call("Edit"))  # 5 — the cap
    assert stop["continue_"] is False
    assert state["stopped"] == "step budget 5 reached"


def test_codex_mcp_results_unwrap_the_envelope():
    """Verified against codex-cli 0.146.0: an mcp_tool_call result arrives as the raw MCP payload.

    Stringifying it verbatim buried the tool's real output inside a dict repr — the trace line and
    the scratch-image rescue both read that text, so the envelope has to come off here.
    """
    blocks = _normalise(
        {"type": "item.completed",
         "item": {"id": "m1", "type": "mcp_tool_call", "server": "cap", "tool": "render",
                  "status": "completed",
                  "result": {"content": [{"type": "text", "text": '{"result": "/x/wall0.png"}'}],
                             "isError": False}}}, {})
    assert blocks[0].content == '{"result": "/x/wall0.png"}'
    assert blocks[0].is_error is False


def test_codex_mcp_error_is_detected_inside_the_envelope():
    """`status` can read "completed" while the tool itself failed — isError is the real signal."""
    blocks = _normalise(
        {"type": "item.completed",
         "item": {"id": "m1", "type": "mcp_tool_call", "server": "cap", "tool": "check_collisions",
                  "status": "completed",
                  "result": {"content": [{"type": "text", "text": '{"error": "no Room.py"}'}],
                             "isError": True}}}, {})
    assert blocks[0].is_error is True
    assert blocks[0].content == '{"error": "no Room.py"}'
