"""agent_loop.py — hand the room to a reasoning agent and let it work.

The difference from :mod:`physical_engine.agent` is the shape of the interaction. That module asks
one question per violation and takes one answer; this one gives the agent a scene, a set of verbs,
and the photographs, and lets it look, act, re-check and change its mind until the room is right.
That matters because the remaining failures are not single-object questions: on Airbnb-Cam a
wardrobe, a table and a second table are wrong TOGETHER, and any repair that considers one at a
time scores worse at every individual step.

The safety property is unchanged and is enforced where it belongs — at the end. The agent works on
a copy in its own session directory; whatever it produces is scored against the deterministic
result and is kept ONLY if it is better. A confused agent costs time and nothing else.
"""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .adjust import check
from .repair import attachments, repair, score
from .shell import save_shell

__all__ = ["run_agent", "solve_v14"]

TIMEOUT = 900

BRIEF = """You are fixing the geometry of a room reconstructed from an iPhone RoomPlan scan, so a
physics engine will accept it. Work in this directory: {session}

Your tools — run them with Bash, exactly as written:

  {py} -m physical_engine.toolcli {session} summary
  {py} -m physical_engine.toolcli {session} object <ObjectId>
  {py} -m physical_engine.toolcli {session} check
  {py} -m physical_engine.toolcli {session} attach <ObjectId> <Wall> [<Wall2>]
  {py} -m physical_engine.toolcli {session} move <ObjectId> <dx> <dy>
  {py} -m physical_engine.toolcli {session} resize <ObjectId> <x> <y> <z>
  {py} -m physical_engine.toolcli {session} drop <ObjectId> <reason>
  {py} -m physical_engine.toolcli {session} undo

`object` prints paths to reference photographs — the scan frames with that object's reconstructed
box drawn on them. READ THEM with the Read tool. They are the only evidence for whether a box
actually bounds the thing it claims to, and they are why you are being asked rather than the
solver.

What counts as done: zero violations, reported by `check`.

What counts as a GOOD repair, and this part matters more than the count:

* Furniture against a wall belongs against that wall. A unit in a corner belongs in that corner.
  Prefer `attach` over `move` — it computes the distance for you and preserves the alignment that
  a raw translation destroys. `attach X WallA WallB` fits X into the corner of those two walls,
  trimming it slightly if it must.
* RoomPlan merges adjacent installed units and records them too deep — a counter fused with the
  wall behind it, an oven fused into its cabinet run. That reads as "inside the wall" and the
  repair is `resize`, not `move`. A standalone table or chair is NOT merged that way, so do not
  resize one to escape a wall.
* Two boxes that were already stacked on each other in the scan are one object detected twice.
  `drop` the redundant one and say so. Do not shuffle duplicates apart.
* A chair overlapping the table it is tucked under is correct. Leave it.
* Do not slide furniture across the room. Every verb prints the score; `defects` counts collisions
  PLUS units knocked off their wall PLUS anything moved implausibly far, so a big move that clears
  a clash can still make the number worse. If it does, `undo`.

Start with `summary`. Look at the photographs for anything you are unsure about. Work until `check`
reports nothing, or until you are confident the rest is a reconstruction fault rather than a
placement one — if so, say which and stop. Finish by printing a one-paragraph summary of what you
changed and why."""


def run_agent(shell: dict[str, Any], session: Path, *, references: Path | None = None,
              model: str = "sonnet", python: str | None = None) -> tuple[dict, str]:
    """Give one scene to the agent in its own session directory. Returns (state, transcript)."""
    session.mkdir(parents=True, exist_ok=True)
    save_shell(shell, session / "state.json")
    save_shell(shell, session / "baseline.json")
    (session / "history.json").write_text(
        json.dumps([{"note": "initial", "state": shell}]), encoding="utf-8")
    if references and references.is_dir() and not (session / "references").exists():
        shutil.copytree(references, session / "references")

    interpreter = python or "python3"
    brief = BRIEF.format(session=session, py=interpreter)
    try:
        done = subprocess.run(
            ["claude", "-p", brief, "--output-format", "text", "--model", model,
             "--allowed-tools", "Read", "Bash", "Glob"],
            capture_output=True, text=True, timeout=TIMEOUT, cwd=str(session.parent))
    except subprocess.TimeoutExpired:
        return shell, "agent timed out"
    from .shell import load_shell
    final = load_shell(session / "state.json")
    return final, (done.stdout or "")[-4000:]


def solve_v14(shell: dict[str, Any], *, session_root: Path | None = None,
              model: str = "sonnet", detail: bool = False):
    """Deterministic repair, then a reasoning agent on whatever it could not finish.

    The agent is given the DETERMINISTIC result rather than the raw scan: everything cheap and
    checkable has already been done, so its attention goes to the part that actually needs
    judgement. Its output is scored the same way and kept only if it wins.
    """
    import sys as _sys

    baseline = copy.deepcopy(shell)
    solved, moves, log = repair(shell)
    errors = [v for v in check(solved) if v.severity == "error"]
    root = Path((shell.get("meta") or {}).get("batch_dir") or ".")
    if not errors:
        return (solved, moves, log) if detail else solved

    session = Path(session_root or (root / "agent_session"))
    final, transcript = run_agent(solved, session, references=root / "references",
                                  model=model, python=_sys.executable)

    held = {oid: set(attachments(o, baseline.get("walls") or {}))
            for oid, o in (baseline.get("objects") or {}).items()}
    held = {k: v for k, v in held.items() if v}
    # let the deterministic actions tidy whatever the agent left, then judge the pair
    tidied, tidy_moves, tidy_log = repair(final)
    if score(tidied, baseline, held) < score(solved, baseline, held):
        return ((tidied, tidy_moves, log + [{"action": "agent", "detail": transcript[-400:]}]
                 + tidy_log) if detail else tidied)
    return (solved, moves, log) if detail else solved
