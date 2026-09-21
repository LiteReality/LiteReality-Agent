"""Exercise local caps without calling a model or consuming account usage."""

import asyncio
import sys
import time

from litereality_agent.agent.providers.base import SessionResult, SessionSpec
from litereality_agent.agent.providers.codex import CodexHarness, _budgeted_prompt


def test_session_can_see_its_time_and_tool_limits():
    prompt = _budgeted_prompt(SessionSpec(prompt="Required task", cwd=".", step_budget=40), 600)
    assert "600 wall-clock seconds" in prompt and "deadline" in prompt
    assert "40 tool calls" in prompt and "Never skip required checks" in prompt
    assert prompt.endswith("Required task")


def run_fake(monkeypatch, program, seconds, steps):
    real_spawn = asyncio.create_subprocess_exec

    async def fake_spawn(*args, **kwargs):
        return await real_spawn(sys.executable, "-u", "-c", program, **kwargs)

    monkeypatch.setattr("shutil.which", lambda _: "/fake/codex")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)
    monkeypatch.setenv("LR_CODEX_SESSION_SECONDS", str(seconds))

    async def collect():
        return [
            event
            async for event in CodexHarness().run(
                SessionSpec(prompt="test", cwd=".", step_budget=steps)
            )
        ]

    started = time.monotonic()
    events = asyncio.run(collect())
    assert time.monotonic() - started < 5
    return next(e for e in events if isinstance(e, SessionResult))


def test_wall_cap_interrupts_a_silent_session(monkeypatch):
    result = run_fake(monkeypatch, "import time; time.sleep(60)", 0.1, 25)
    assert "wall time budget" in result.stopped
    assert result.raw["returncode"] != 0


def test_tool_cap_interrupts_a_busy_session(monkeypatch):
    program = """import json, time
print(json.dumps({'type':'item.started','item':{'id':'1','type':'command_execution','command':'true'}}), flush=True)
time.sleep(60)
"""
    result = run_fake(monkeypatch, program, 60, 1)
    assert "step budget 1" in result.stopped


def test_completed_session_keeps_usage(monkeypatch):
    program = """import json
print(json.dumps({'type':'turn.completed','usage':{'input_tokens':123,'output_tokens':4}}))
"""
    result = run_fake(monkeypatch, program, 60, 25)
    assert not result.stopped and not result.is_error
    assert result.raw["usage"]["input_tokens"] == 123
