"""render — PRIMITIVE. Render|photo comparison for ONE target layer: room / wall / object.

Layered by target (there is no aimless render):
  - "room"    → scene mode, every object chip-labelled
  - "Wall<N>" → wall_focus mode, only that wall outlined (both sides)
  - <Object>  → object mode, bbox on that object
If `frames` is omitted the best frames are auto-picked via select_views. The room-ops
rendering engine is imported lazily because it needs Blender.
"""

from __future__ import annotations

import asyncio

from lrauthor.agent.tools._scene import room_dir_from
from lrauthor.agent.tools.base import (
    BaseDeclarativeTool,
    SceneToolInvocation,
    ToolParamsModel,
    ToolResult,
    make_tool_schema,
    validate_tool_params,
)


def _sync_stale_objects(room_dir) -> list[str]:
    """HONEST-RENDER invariant: the render loads each object from the PACKED assets
    (`_scene_assets/glb/<name>.glb`), NOT from its `object.py`. So an edit to a procedural object's
    `object.py` would silently NOT appear in the render. Fix: rebuild + repack any procedural object
    whose `object.py` is newer than its packed GLB, so object edits actually show up. No-op when all
    objects are fresh (cheap mtime check). Never raises — a sync failure must not break the render."""
    import json
    import shutil
    import subprocess
    from pathlib import Path

    room_dir = Path(room_dir)
    proc = room_dir / "Objects" / "Procedural"
    if not proc.is_dir():
        return []
    try:
        from lrauthor.agent.tools.render.source.compose import (
            _config_for,
            _scan_from_room,
        )

        config = _config_for(_scan_from_room(room_dir))
        pack = Path(config.ASSET_GLB_DIR)
        blender = config.blender()
    except Exception:  # noqa: BLE001
        return []
    synced = []
    for od in sorted(proc.iterdir()):
        opy = od / "object.py"
        dst = pack / f"{od.name}.glb"
        if not opy.is_file() or not dst.is_file():
            continue  # only repack objects already in the packed set
        if opy.stat().st_mtime <= dst.stat().st_mtime:
            continue  # object.py not touched since last pack → fresh
        tj = od / "textures.json"
        if tj.is_file():
            try:
                from lrauthor.room_ops.compile.fetch_textures import materialize

                materialize(json.loads(tj.read_text()), od / "textures")
            except Exception:  # noqa: BLE001
                pass
        built = od / "_synced.glb"
        try:
            r = subprocess.run(
                [str(blender), "-b", "--python", str(opy), "--", str(od / "textures"), str(built)],
                capture_output=True, text=True, timeout=300,
            )
            if r.returncode == 0 and built.is_file():
                shutil.copy(built, dst)  # repack → the render now loads the rebuilt object
                synced.append(od.name)
        except Exception:  # noqa: BLE001
            pass
    return synced


async def render_target(scene_path, target: str, frames: list[int] | None, n: int = 4) -> dict:
    """Shared engine (also used by the composites): returns
    {'frames': [...], 'selection': [{frame, score, sees?}] | None, 'images': {label: png}}.
    `selection` records WHY each frame was picked (its visibility score) when auto-selected."""
    room_dir = room_dir_from(scene_path)

    # HONEST RENDER: reflect any edited procedural object.py into the packed assets the render reads.
    await asyncio.to_thread(_sync_stale_objects, room_dir)

    # INVARIANT: never render a room whose Room.py changed since its last successful compile —
    # a runtime-broken edit would crash Blender mid-render with a confusing error. Auto-recompile.
    from lrauthor.agent.tools.compile.tool import (
        CompileInvocation,
        CompileParams,
        compile_is_fresh,
    )

    if not compile_is_fresh(room_dir):
        comp = CompileInvocation(CompileParams())
        comp.bind(str(room_dir), None)
        cres = await comp.execute()
        if not cres.is_success():
            raise RuntimeError(
                f"Room.py changed since the last successful compile and auto-compile FAILED — "
                f"fix the build first: {cres.error}"
            )
    selection = None
    if not frames:
        from lrauthor.agent.tools.select_views.tool import (
            SelectViewsInvocation,
            SelectViewsParams,
        )

        sel = SelectViewsInvocation(SelectViewsParams(target=target, n=n))
        sel.bind(str(room_dir), None)
        res = await sel.execute()
        if not res.is_success():
            raise RuntimeError(f"view selection failed: {res.error}")
        selection = res.output["frames"]
        frames = [r["frame"] for r in selection]

    from lrauthor.agent.tools.render import source as irt  # lazy: Blender

    room = str(room_dir)
    if target.lower() == "room" or target in ("Floor0", "Ceiling0"):
        out = irt.render_scene(room, frames)
    elif target.startswith("Wall"):
        out = irt.render_wall_focus(room, frames, target)
    else:
        out = irt.render_object(room, frames, target)
    # descriptive keys — they double as the labels the VLM sees for each image
    images = {f"{target} · frame {k} (render | real photo)": v for k, v in out.items()}

    # For surfaces, attach the HEAD-ON comparison: the current surface rendered ORTHO side-by-side
    # with the real rectified STITCH — the like-for-like reference for colour/pattern/fixtures.
    if target.startswith("Wall") or target in ("Floor0", "Ceiling0"):
        attached = False
        try:
            ref = irt.render_wall_reference(room, target, with_refs=False)
            if target in ref:
                images[f"{target}_headon (ortho RENDER | real STITCH)"] = ref[target]
                attached = True
        except Exception:  # noqa: BLE001 — an aid, never a failure
            pass
        if not attached:
            # No ortho render for this surface yet (e.g. Ceiling0 before its mesh is created —
            # RoomPlan has no ceiling) → attach the raw REAL stitch so the reference is never missing.
            try:
                from lrauthor.agent.tools.render.source.compose import (
                    _config_for,
                    _scan_from_room,
                )

                stitch = _config_for(_scan_from_room(room_dir)).SURFACE_REF / f"{target}_stitched.jpg"
                if stitch.is_file():
                    from PIL import Image as _Im

                    if min(_Im.open(stitch).size) >= 40:  # a ~20px sliver smear is not a reference
                        images[f"{target}_stitch (head-on REAL reference — no render side yet)"] = str(stitch)
            except Exception:  # noqa: BLE001
                pass
    return {"frames": frames, "selection": selection, "images": images}


class RenderParams(ToolParamsModel):
    target: str  # "room" | "Wall<N>" | object name
    frames: list[int] | None = None  # omit → auto-select via select_views
    n: int = 4


class RenderInvocation(SceneToolInvocation):
    def get_description(self) -> str:
        return f"render target={self.params.target}"

    async def execute(self) -> ToolResult:
        try:
            out = await render_target(self.scene_path, self.params.target, self.params.frames, self.params.n)
        except Exception as e:  # noqa: BLE001
            return ToolResult(error=f"render failed: {type(e).__name__}: {e}")
        return ToolResult(output=out)


class RenderTool(BaseDeclarativeTool):
    def __init__(self) -> None:
        schema = make_tool_schema(
            name="render",
            description=(
                "Render the current room paired with the real photo for ONE target layer: 'room' "
                "(everything labelled), a wall ('Wall3' — only it outlined), or an object ('Table0' — "
                "bbox). Omit `frames` to auto-pick the best views. Returns {frames, images:{label: png}}. "
                "Render only a room that currently compiles."
            ),
            parameters={
                "target": {"type": "string", "description": "'room', a wall (Wall3), or an object (Table0)."},
                "frames": {"type": "array", "items": {"type": "integer"},
                           "description": "Frame indices; omit to auto-select the best."},
                "n": {"type": "integer", "description": "How many auto-selected frames (default 4)."},
            },
            required=["target"],
        )
        super().__init__("render", schema)

    async def build(self, params: dict) -> RenderInvocation:
        return RenderInvocation(validate_tool_params(RenderParams, params))
