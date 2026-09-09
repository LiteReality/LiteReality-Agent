"""compile — build the active Room.py into Room.glb, and surface build errors for the model to fix.

The freshness stamp is pure filesystem, so it tests offline. It matters more than it looks: the
model calls `render` in a loop, and every render recompiles unless the stamp says the room is
unchanged. A stamp that wrongly reports "fresh" means the model renders a stale room and reasons
about work it already replaced.

The Blender build itself is `-m blender`.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from lrauthor.agent.tools.compile.tool import (
    CompileInvocation,
    CompileParams,
    CompileTool,
    compile_is_fresh,
)

ROOM_PY = '''\
SHELL = {"walls": {}, "openings": {}, "floor_height": 0.0, "ceiling_height": 2.6}


def build():
    return SHELL
'''


@pytest.fixture
def room(tmp_path):
    d = tmp_path / "room"
    d.mkdir()
    (d / "Room.py").write_text(ROOM_PY, encoding="utf-8")
    return d


def test_schema_is_well_formed():
    fn = CompileTool().schema["function"]
    assert fn["name"] == "compile"
    assert "regenerate" in fn["parameters"]["properties"]


def test_a_room_that_never_compiled_is_not_fresh(room):
    assert compile_is_fresh(room) is False


def test_fresh_only_while_room_py_is_unchanged(room):
    from lrauthor.agent.tools.compile.tool import _write_compile_stamp

    _write_compile_stamp(room)
    assert compile_is_fresh(room) is True, "a stamped, unedited room should not rebuild"

    (room / "Room.py").write_text(ROOM_PY + "\n# the model edited a material\n", encoding="utf-8")
    assert compile_is_fresh(room) is False, (
        "stamp survived an edit — the model would render a stale room and reason about "
        "work it had already replaced"
    )


def test_stamp_never_raises_on_an_unwritable_room(room, monkeypatch):
    """The stamp is an optimisation. If it cannot be written the compile must still succeed."""
    from lrauthor.agent.tools.compile.tool import _write_compile_stamp

    def _boom(*a, **k):
        raise OSError("read-only filesystem")

    monkeypatch.setattr("pathlib.Path.write_text", _boom)
    _write_compile_stamp(room)  # must not raise


def test_missing_room_py_is_not_fresh_rather_than_an_error(tmp_path):
    assert compile_is_fresh(tmp_path / "no-such-room") is False


def test_compile_api_rejects_a_missing_glb(room, tmp_path, monkeypatch):
    from lrauthor.room_ops import api

    monkeypatch.setattr(api.subprocess, "run", lambda _cmd: SimpleNamespace(returncode=0))
    assert api.compile_room(room, out_dir=tmp_path / "preview", bake=False) is None


def test_compile_tool_does_not_stamp_a_missing_glb(room, tmp_path, monkeypatch):
    missing = tmp_path / "preview" / "Room.glb"
    monkeypatch.setattr("lrauthor.room_ops.compile_room", lambda *_a, **_k: missing)

    inv = CompileInvocation(CompileParams(regenerate=False))
    inv.bind(str(room), None)
    result = asyncio.run(inv.execute())

    assert not result.is_success()
    assert result.build == {"status": "error", "glb_path": None}
    assert not (room / ".compile_stamp").exists()


@pytest.mark.blender
@pytest.mark.scan
def test_compiles_a_real_seed_room(example_scan):
    """Full build through Blender. Run with `-m 'blender and scan'`."""
    inv = CompileInvocation(CompileParams(regenerate=False))
    inv.bind(str(example_scan.room), None)
    res = asyncio.run(inv.execute())
    assert res.is_success(), f"seed room failed to compile: {res.error}"
    assert compile_is_fresh(example_scan.room), "a successful compile must leave a valid stamp"
