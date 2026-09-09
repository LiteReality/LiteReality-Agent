"""One readable line per tool call — for the live status row and the stage logs.

Why this is a module and not three inline prints: a tool's NAME and INPUT exist **only** on
`ToolUseBlock` (`id`, `name`, `input`). `ToolResultBlock` carries `tool_use_id`, `content`,
`is_error` — and nothing else. Narrating from the result block is therefore silently empty:
`getattr(block, "name", "?")` returns the default, so the authoring row read

    ▶ authoring        [5] ?

for an entire run. It poisoned everything derived from that name, too. The per-tool `counts`
in the done line collapsed to `{'?': N}`, and `author.py`'s stitch-coverage guardrail — which
watches for `Read(<surface>_stitched.jpg)` to prove the model opened its head-on references —
never observed a single read, so it warned that every surface had been authored blind. That
warning was an artifact of reading the wrong block, not a quality signal.

The trace file was correct throughout, because `AgentTrace.tool()` was already called at use
time. That asymmetry is the tell: same run, right names on disk, `?` on screen.

So: narrate at USE time, keep the id, and let the result attribute back to it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from lrauthor import REPO_ROOT

_REPO = str(REPO_ROOT).rstrip("/")

# What each tool is working ON — first key present wins, so order is "most worth reading
# first". `critic` carries both `target` and `goal`; the goal is the line worth showing.
_HINT_KEYS: dict[str, tuple[str, ...]] = {
    "render": ("target",),
    "select_views": ("target",),
    "critic": ("goal", "target"),
    "fetch_material": ("query", "name"),
    "fetch_object": ("query", "name"),
    "inspect": ("target",),
    "Read": ("file_path",),
    "Edit": ("file_path",),
    "Write": ("file_path",),
    "NotebookEdit": ("notebook_path",),
    "Glob": ("pattern",),
    "Grep": ("pattern",),
    "Bash": ("description", "command"),
}
# For anything not in the table (new capability tools, SDK built-ins we have not enumerated).
_FALLBACK_KEYS = ("target", "query", "goal", "file_path", "pattern", "name", "command")
# Shown as a basename: an absolute room path eats the whole line and the leaf is the content.
_PATH_KEYS = frozenset({"file_path", "path", "notebook_path", "image", "room"})


# Shell/python boilerplate that carries no information about WHAT a command does. Every
# analysis the model writes opens the same way — `cd <60-char abs path> && python3 - <<'EOF'`
# then a block of imports — so the first 44 characters of the raw command are identical across
# calls that do completely different things.
_CD_PREFIX = re.compile(r"^\s*cd\s+(\S+)\s*&&\s*")
_PY_WRAPPER = re.compile(r"^\s*python3?\s*(?:-c\s*[\"']|-\s*<<\s*[\"']?\w+[\"']?)\s*", re.M)
_BOILERPLATE = re.compile(r"^\s*(?:import\s|from\s+\S+\s+import\s|#|EOF|PY|\"|')", re.M)
# The statement that says what the script was ASKING, as opposed to what it was setting up.
_OUTPUT_STMT = re.compile(r"\b(?:print|save|savefig|write|write_text|to_csv)\s*\(")


def _bash_summary(command: str, width: int) -> str:
    """A one-line "what is this doing" for a shell command with no `description`.

    Strips the wrapper and the imports, then shows the statements that actually differ between
    one call and the next. Long paths collapse to basenames — the directory is shared by every
    command in the run, so it is the part with no information in it.
    """
    text = command.strip()
    cd_into = ""
    m = _CD_PREFIX.match(text)
    if m:
        cd_into = Path(m.group(1)).name
        text = text[m.end():]
    text = _PY_WRAPPER.sub("", text, count=1)
    lines = [ln.strip() for ln in text.splitlines()]
    keep = [ln for ln in lines if ln and not _BOILERPLATE.match(ln)]
    # Two calls that load the same manifest and then ask different questions of it are
    # indistinguishable from their opening line. What the script PRINTS or SAVES is the question
    # it was asking, so lead with that and let the setup fall off the end.
    output = [ln for ln in keep if _OUTPUT_STMT.search(ln)]
    ordered = output + [ln for ln in keep if ln not in output] if output else keep
    body = " ".join(ordered) if ordered else " ".join(ln for ln in lines if ln)
    # Absolute paths are mostly the same shared prefix; the leaf is what identifies the file.
    body = re.sub(r"/[\w./+-]*/([\w.+-]+)", r"\1", body)
    body = " ".join(body.split())
    # Reserve the tag's room BEFORE truncating. Appending it after meant it was cut off in
    # exactly the case it exists for — the long commands, which is nearly all of them.
    tag = f"  ({cd_into})" if cd_into and body else ""
    room = max(12, width - len(tag))
    if len(body) > room:
        body = body[: room - 1] + "…"
    return body + tag


def describe_tools_line() -> str:
    """Ask the model to label its own shell calls.

    `_bash_summary` is a decent guess, but it is reverse-engineering intent from syntax. The
    `Bash` tool takes a `description`, and when the model fills it in the row reads "Detect
    whiteboard columns on Wall0" instead of the first statement of the script — no heuristic
    competes with that. It supplies one inconsistently, so ask.
    """
    return (
        "\nPROGRESS LABELS: every `Bash` call must pass a short `description` (5-8 words) saying\n"
        "what that command is FOR — \"measure dado band on Wall2\", not \"run python\". It is shown\n"
        "as the live progress line and kept in the run trace, so it is how anyone later\n"
        "reconstructs your reasoning.\n"
    )


def tool_label(name: str) -> str:
    """`mcp__cap__render` → `render`. In-process MCP tools arrive fully qualified."""
    return (name or "?").split("__")[-1] or "?"


def hint_for(name: str, inp: dict[str, Any] | None, width: int = 44) -> str:
    """A short "on what" for one tool call. Never raises — this is narration, not logic."""
    inp = inp or {}
    for key in _HINT_KEYS.get(name, ()) + _FALLBACK_KEYS:
        raw = inp.get(key)
        if raw in (None, "", [], {}):
            continue
        if isinstance(raw, (list, tuple)):
            raw = raw[0] if len(raw) == 1 else f"{len(raw)} items"
        text = str(raw)
        if key == "command":
            return _bash_summary(text, width)
        if key in _PATH_KEYS:
            text = Path(text).name or text
        text = " ".join(text.split())  # a multi-line goal must not break the status row
        # A `Bash` command is mostly absolute repo path — 60 characters of prefix every stage
        # shares, crowding out the part that says what it is doing. Every path here is inside
        # the repo, so the prefix carries no information.
        text = text.replace(f"{_REPO}/", "").replace(str(_REPO), ".")
        return text if len(text) <= width else text[: width - 1] + "…"
    return ""


class ToolNarrator:
    """Accumulates the tool stream: prints at use time, attributes results back by id.

    `calls` and `counts` are what the stage's done line reports, so they count *invocations*.
    A call whose result never arrives (the run hit its turn cap mid-tool) still counts — it
    happened.
    """

    def __init__(self, hint_width: int = 44) -> None:
        self.calls = 0
        self.counts: dict[str, int] = {}
        self._hint_width = hint_width
        self._pending: dict[str, tuple[str, dict]] = {}

    def use(self, block) -> str:
        """Record a `ToolUseBlock` and return the line to print for it."""
        name = tool_label(getattr(block, "name", "") or "?")
        inp = getattr(block, "input", {}) or {}
        self.calls += 1
        self.counts[name] = self.counts.get(name, 0) + 1
        bid = getattr(block, "id", "") or ""
        if bid:
            self._pending[bid] = (name, inp)
        # Right-aligned index: the counter runs past 9 within the first minute, and a shifting
        # column smears the status row that redraws in place.
        return f"  [{self.calls:>3}] {name:<14} {hint_for(name, inp, self._hint_width)}".rstrip()

    def result(self, block) -> tuple[str, dict]:
        """The `(name, input)` of the call this `ToolResultBlock` answers.

        `("?", {})` when the id is unknown — a resumed session can deliver a result whose use
        block predates this narrator. Callers must treat that as "no attribution", never as a
        tool literally named `?`.
        """
        bid = getattr(block, "tool_use_id", "") or ""
        return self._pending.pop(bid, ("?", {}))

    def error_line(self, name: str, block) -> str | None:
        """A line for a failed tool result, or None when it succeeded."""
        if not getattr(block, "is_error", False):
            return None
        content = getattr(block, "content", None)
        if isinstance(content, list):
            content = " ".join(str(c.get("text", "")) for c in content if isinstance(c, dict))
        text = " ".join(str(content or "").split())[:90]
        return f"      ↳ {name} failed: {text}" if text else f"      ↳ {name} failed"
