"""compile — PRIMITIVE. Build the active Room.py → Room.glb; surface build errors to fix.

Thin wrapper over `room_ops.compile_room` (the existing scene build). The INTERCEPTED tool:
the harness can watch it to update build freshness / write revisions. Lazy import (needs Blender).
"""

from __future__ import annotations

from pathlib import Path

from lrauthor.agent.tools._scene import room_dir_from
from lrauthor.agent.tools.base import (
    BaseDeclarativeTool,
    SceneToolInvocation,
    ToolParamsModel,
    ToolResult,
    make_tool_schema,
    validate_tool_params,
)


def _room_hash(room_dir) -> str:
    import hashlib

    return hashlib.sha1((room_dir / "Room.py").read_bytes()).hexdigest()


def _write_compile_stamp(room_dir) -> None:
    try:
        (room_dir / ".compile_stamp").write_text(_room_hash(room_dir))
    except Exception:  # noqa: BLE001 — the stamp is an optimization, never a failure
        pass


def compile_is_fresh(room_dir) -> bool:
    """True iff Room.py is unchanged since the last SUCCESSFUL compile."""
    try:
        return (room_dir / ".compile_stamp").read_text().strip() == _room_hash(room_dir)
    except Exception:  # noqa: BLE001
        return False


class CompileParams(ToolParamsModel):
    regenerate: bool = False


class CompileInvocation(SceneToolInvocation):
    def get_description(self) -> str:
        return "Compile Room.py → Room.glb"

    async def execute(self) -> ToolResult:
        from lrauthor.room_ops import compile_room  # lazy: Blender subprocess

        try:
            room_dir = room_dir_from(self.scene_path)
        except ValueError as e:
            return ToolResult(error=str(e))
        try:
            glb = compile_room(room_dir, regenerate=self.params.regenerate)
        except Exception as e:  # noqa: BLE001
            return ToolResult(
                error=f"compile raised {type(e).__name__}: {e}",
                build={"status": "error", "error": str(e)},
            )
        if glb is None or not Path(glb).is_file():
            return ToolResult(
                error="compile failed — Room.py did not produce Room.glb (see build errors).",
                build={"status": "error", "glb_path": None},
            )
        _write_compile_stamp(room_dir)
        return ToolResult(output=f"Compiled Room.glb at {glb}", build={"status": "ok", "glb_path": str(glb)})


class CompileTool(BaseDeclarativeTool):
    def __init__(self) -> None:
        schema = make_tool_schema(
            name="compile",
            description=(
                "Build the active Room.py into Room.glb (the scene build). Returns status 'ok' with "
                "the glb path, or 'error' with the message to fix. Recompile after edits before rendering."
            ),
            parameters={
                "regenerate": {"type": "boolean", "description": "Force-regenerate per-object materials (default false)."}
            },
            required=[],
        )
        super().__init__("compile", schema)

    async def build(self, params: dict) -> CompileInvocation:
        return CompileInvocation(validate_tool_params(CompileParams, params))
