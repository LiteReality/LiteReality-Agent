"""Tool narration must read names off the block that actually has one.

The bug: `ToolResultBlock` has exactly three fields — `tool_use_id`, `content`, `is_error`. The
authoring loop read `name` and `input` off it, so every line printed `[5] ?` with no hint, the
done-line `counts` collapsed to `{'?': N}`, and the stitch-coverage guardrail (which watches for
`Read(<surface>_stitched.jpg)`) saw zero reads and warned that every surface was authored blind.

These tests use the REAL SDK dataclasses, not hand-rolled stand-ins. A fake result block with a
`name` attribute would pass while the shipped code stayed broken — that is precisely the mistake
being pinned here.
"""

from __future__ import annotations

import pytest

from lrauthor.agent.tool_narration import ToolNarrator, hint_for, tool_label

sdk_types = pytest.importorskip("claude_agent_sdk.types")
ToolUseBlock = sdk_types.ToolUseBlock
ToolResultBlock = sdk_types.ToolResultBlock


def test_result_block_really_has_no_name_or_input():
    """The premise. If the SDK ever adds these fields this test should fail loudly, because the
    narrator's whole id-tracking design exists to work around their absence."""
    fields = set(ToolResultBlock.__annotations__)
    assert fields == {"tool_use_id", "content", "is_error"}
    assert "name" not in fields and "input" not in fields


def test_use_block_narrates_tool_and_target():
    nar = ToolNarrator()
    line = nar.use(ToolUseBlock(id="t1", name="mcp__cap__render", input={"target": "Wall3"}))
    assert "render" in line and "Wall3" in line
    assert "?" not in line
    assert line.startswith("  [  1] "), "index is right-aligned so the column cannot shift"


def test_counts_are_named_not_question_marks():
    nar = ToolNarrator()
    for i, (name, inp) in enumerate([
        ("mcp__cap__render", {"target": "Wall0"}),
        ("Read", {"file_path": "/a/b/Wall0_stitched.jpg"}),
        ("mcp__cap__render", {"target": "Wall1"}),
    ]):
        nar.use(ToolUseBlock(id=f"t{i}", name=name, input=inp))
    assert nar.counts == {"render": 2, "Read": 1}
    assert nar.calls == 3
    assert "?" not in nar.counts


def test_result_attributes_back_to_its_use_block():
    """This is what makes stitch coverage work: the result carries only an id, so the input has
    to be recovered from the use block that opened the call."""
    nar = ToolNarrator()
    nar.use(ToolUseBlock(id="tu_9", name="Read", input={"file_path": "/x/Wall2_stitched.jpg"}))
    name, inp = nar.result(ToolResultBlock(tool_use_id="tu_9", content="ok", is_error=False))
    assert name == "Read"
    assert inp["file_path"].endswith("Wall2_stitched.jpg")


def test_unknown_result_id_is_not_a_tool_named_question_mark():
    """A resumed session can deliver a result whose use block predates this narrator. Callers
    must be able to tell "no attribution" from a real tool."""
    nar = ToolNarrator()
    name, inp = nar.result(ToolResultBlock(tool_use_id="never-seen", content=None, is_error=False))
    assert (name, inp) == ("?", {})
    assert nar.counts == {}, "an unattributed result must not invent a call"


def test_error_result_is_reported_with_the_right_tool_name():
    nar = ToolNarrator()
    nar.use(ToolUseBlock(id="e1", name="mcp__cap__render", input={"target": "Wall9"}))
    blk = ToolResultBlock(tool_use_id="e1", content="Blender exited 1", is_error=True)
    name, _ = nar.result(blk)
    line = nar.error_line(name, blk)
    assert line is not None and "render" in line and "Blender exited 1" in line


def test_successful_result_produces_no_error_line():
    nar = ToolNarrator()
    nar.use(ToolUseBlock(id="s1", name="Read", input={"file_path": "/x/y.jpg"}))
    blk = ToolResultBlock(tool_use_id="s1", content="...", is_error=False)
    assert nar.error_line(*nar.result(blk)[:1], blk) is None


@pytest.mark.parametrize("name,inp,want", [
    ("render", {"target": "Wall3"}, "Wall3"),
    ("critic", {"images": ["a.png"], "goal": "radiator under the window?"}, "radiator under the window?"),
    ("fetch_material", {"query": "oak floor", "name": "Floor0"}, "oak floor"),
    ("Read", {"file_path": "/very/long/abs/path/Wall0_stitched.jpg"}, "Wall0_stitched.jpg"),
    ("Glob", {"pattern": "**/*.py"}, "**/*.py"),
    ("Write", {"file_path": "/room/Room.py"}, "Room.py"),
])
def test_hints_show_the_thing_being_worked_on(name, inp, want):
    assert hint_for(name, inp) == want


def test_hint_never_breaks_the_status_row():
    """the CLI redraws ONE line; a newline in a goal would smear the display."""
    hint = hint_for("critic", {"goal": "line one\nline two\n\tline three", "images": []})
    assert "\n" not in hint and "\t" not in hint
    assert hint == "line one line two line three"


def test_a_supplied_description_always_wins():
    """The model's own label beats any amount of syntax analysis, so it is the first choice."""
    hint = hint_for("Bash", {"description": "Measure dado band on Wall2",
                             "command": "python3 - <<'PY'\nimport json\nprint(1)\nPY"})
    assert hint == "Measure dado band on Wall2"


def test_paths_collapse_to_basenames_so_the_command_is_visible():
    """A Bash hint is mostly a shared absolute prefix; without collapsing it the truncation
    window closes before the command itself appears."""
    from lrauthor import REPO_ROOT

    hint = hint_for("Bash", {"command": f"ls {REPO_ROOT}/run/Office_room/realism_authoring/room"})
    assert hint == "ls room"
    assert str(REPO_ROOT) not in hint


def test_python_boilerplate_does_not_eat_the_line():
    """`cd <60-char path> && python3 - <<'PY'` + imports is identical across every analysis the
    model writes, so the raw command's first 44 characters say nothing about what it does."""
    hint = hint_for("Bash", {"command":
        "cd /a/b/c/_scratch && python3 - <<'PY'\n"
        "import json\nfrom PIL import Image\n"
        "d=json.load(open('surface_ref_manifest.json'))\n"
        "print('ppm', d['ppm'])\nPY"})
    assert "import" not in hint and "python3" not in hint
    assert "print('ppm'" in hint
    assert "(_scratch)" in hint, "the directory it ran in is worth one trailing tag"


def test_two_scripts_reading_the_same_file_are_distinguishable():
    """Both open the manifest; what they ASK of it differs, and that is what must be visible."""
    base = "python3 -c \"\nimport json\nd=json.load(open('m.json'))\n"
    a = hint_for("Bash", {"command": base + "print(type(d), list(d.keys()))\n\""})
    b = hint_for("Bash", {"command": base + "print('ppm', d['ppm'])\n\""})
    assert a != b, f"indistinguishable rows: {a!r}"


def test_long_hints_are_truncated_to_fit():
    hint = hint_for("critic", {"goal": "x" * 500, "images": []}, width=20)
    assert len(hint) <= 20 and hint.endswith("…")


def test_missing_and_empty_inputs_degrade_quietly():
    assert hint_for("render", None) == ""
    assert hint_for("render", {}) == ""
    assert hint_for("render", {"target": ""}) == ""
    assert hint_for("brand_new_tool", {"target": "Wall1"}) == "Wall1", "unknown tools use fallbacks"


def test_the_tool_column_stays_put_as_the_counter_grows():
    """the CLI redraws this line in place; a column that shifts at 10 and again at 100 reads as
    flicker rather than progress."""
    nar = ToolNarrator()
    lines = [nar.use(ToolUseBlock(id=f"t{i}", name="Read", input={"file_path": "/x/y.jpg"}))
             for i in range(120)]
    starts = {line.index("Read") for line in (lines[0], lines[9], lines[10], lines[99], lines[119])}
    assert len(starts) == 1, f"tool name column moved: {starts}"


def test_tool_label_strips_the_mcp_prefix():
    assert tool_label("mcp__cap__fetch_material") == "fetch_material"
    assert tool_label("Read") == "Read"
    assert tool_label("") == "?"
