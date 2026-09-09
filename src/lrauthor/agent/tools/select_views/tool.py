"""select_views — PRIMITIVE. Pick the BEST capture frames for a target — no frame-guessing.

Wraps the repo's view-selection stack:
  - target "room"    → greedy cover: fewest frames that together see every wall well
  - target "Wall<N>" → frames where that wall projects large & on-screen (wall-plane projection,
                       no Blender render needed — reuses agent/overlay.py)
  - target <Object>  → authoring/views/room_render/select_for.views_for over a render_manifest with object
                       visibility (needs a prior render('room') pass that wrote the manifest)

Returns [{frame, score}] so render/inspect/measure can consume the frames directly.
"""

from __future__ import annotations

import json
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


def _wall_scores(room_dir: Path) -> dict[int, dict[str, float]]:
    """{frame -> {wall_name -> on-screen coverage 0..1}} via wall-plane projection (no Blender)."""
    from lrauthor.agent.tools.render.source.compose import (
        _config_for,
        _scan_from_room,
    )
    from lrauthor.agent.tools.shared import overlay as _overlay

    config = _config_for(_scan_from_room(room_dir))
    planes = _overlay.load_planes(config)
    Wu, Hu = 864, 576
    out: dict[int, dict[str, float]] = {}
    for jf in sorted(config.SCAN_DIR.glob("frame_*.json")):
        try:
            frame = int(jf.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        proj = _overlay.project_walls(jf, planes, config, Wu, Hu)
        scores = {}
        for name, d in proj.items():
            pts = [pt for seg in d.get("edges", []) for pt in seg]
            if not pts:
                continue
            xs = [max(0.0, min(float(Wu), x)) for x, _ in pts]
            ys = [max(0.0, min(float(Hu), y)) for _, y in pts]
            scores[name] = ((max(xs) - min(xs)) * (max(ys) - min(ys))) / float(Wu * Hu)
        if scores:
            out[frame] = scores
    return out


def _frame_gap(per_frame: dict) -> int:
    """Minimum frame-index separation between chosen views. Adjacent ARKit frames are near-identical
    (~1 frame/s sweep), so enforcing a temporal gap is a cheap, effective viewpoint-diversity proxy."""
    return max(3, len(per_frame) // 15)


def _select_room(room_dir: Path, n: int) -> list[dict]:
    """Greedy weighted set-cover over walls: FEWEST DIVERSE frames that together see every wall.
    Stops when coverage is complete — returns fewer than n rather than padding with near-duplicates."""
    per_frame = _wall_scores(room_dir)
    gap = _frame_gap(per_frame)
    walls = sorted({w for s in per_frame.values() for w in s})
    chosen: list[dict] = []
    covered: set[str] = set()

    def too_close(f):
        return any(abs(f - c["frame"]) < gap for c in chosen)

    while len(chosen) < n and len(covered) < len(walls):
        best, best_gain = None, 0.0
        for f, scores in per_frame.items():
            if too_close(f):
                continue
            gain = sum(v for w, v in scores.items() if w not in covered)
            if gain > best_gain:
                best, best_gain = f, gain
        if best is None:
            break
        chosen.append({"frame": best, "score": round(best_gain, 3),
                       "sees": sorted(w for w in per_frame[best] if w not in covered)})
        covered |= set(per_frame[best])
    # top-up BEYOND coverage when more views were asked for — widest remaining views,
    # still diversity-gapped so they are genuinely different viewpoints (never near-dups).
    if len(chosen) < n:
        rest = sorted(((f, sum(s.values())) for f, s in per_frame.items() if not too_close(f)),
                      key=lambda t: -t[1])
        for f, sc in rest:
            if len(chosen) >= n:
                break
            if too_close(f):
                continue
            chosen.append({"frame": f, "score": round(sc, 3), "sees": sorted(per_frame[f])})
    return chosen


def _select_wall(room_dir: Path, wall: str, n: int) -> list[dict]:
    """Frames seeing the wall largest — with the same diversity gap, so we get the wall from
    genuinely different viewpoints instead of three adjacent shutter clicks."""
    per_frame = _wall_scores(room_dir)
    gap = _frame_gap(per_frame)
    ranked = sorted(
        ({"frame": f, "score": round(s[wall], 3)} for f, s in per_frame.items() if wall in s),
        key=lambda r: -r["score"],
    )
    chosen: list[dict] = []
    for r in ranked:
        if any(abs(r["frame"] - c["frame"]) < gap for c in chosen):
            continue
        chosen.append(r)
        if len(chosen) >= n:
            break
    return chosen or ranked[:n]


def _find_manifest(room_dir: Path) -> Path | None:
    """Newest render_manifest.json with object visibility near this room (from a render pass)."""
    cands = sorted(
        room_dir.parent.glob("_image_render_tool/*/_renders/render_manifest.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return cands[0] if cands else None


def _select_object(room_dir: Path, obj: str, n: int) -> list[dict] | str:
    mf = _find_manifest(room_dir)
    if mf is None:
        return "no render manifest with object visibility yet — call render(target='room') once first"
    from lrauthor.agent.tools.select_views.source.select_for import views_for

    manifest = json.loads(mf.read_text())
    rows = views_for(manifest, obj, n=n, min_quality=0.0)
    if not rows:
        return f"'{obj}' not visible in any rendered frame (per {mf})"
    out = []
    for r in rows:
        fr = r["frame"]
        num = int(fr.split("_")[1]) if isinstance(fr, str) and "_" in fr else int(fr)
        out.append({"frame": num, "score": r.get("quality")})
    return out


class SelectViewsParams(ToolParamsModel):
    target: str  # "room" | "Wall<N>" | an object name (Table0, ...)
    n: int = 4


class SelectViewsInvocation(SceneToolInvocation):
    def get_description(self) -> str:
        return f"select_views target={self.params.target}"

    async def execute(self) -> ToolResult:
        try:
            room_dir = room_dir_from(self.scene_path)
        except ValueError as e:
            return ToolResult(error=str(e))
        t, n = self.params.target, max(1, min(self.params.n, 10))
        try:
            if t.lower() == "room":
                rows = _select_room(room_dir, n)
            elif t.startswith("Wall") or t in ("Floor0", "Ceiling0"):
                # floor/ceiling aren't planes in the overlay — fall back to room coverage
                rows = _select_wall(room_dir, t, n) if t.startswith("Wall") else _select_room(room_dir, n)
            else:
                rows = _select_object(room_dir, t, n)
                if isinstance(rows, str):
                    return ToolResult(error=rows)
        except Exception as e:  # noqa: BLE001
            return ToolResult(error=f"view selection failed: {type(e).__name__}: {e}")
        if not rows:
            return ToolResult(error=f"no frames found that see {t!r}")
        return ToolResult(output={"target": t, "frames": rows})


class SelectViewsTool(BaseDeclarativeTool):
    def __init__(self) -> None:
        schema = make_tool_schema(
            name="select_views",
            description=(
                "Pick the BEST capture frames for a target — never guess frame numbers. target='room' "
                "→ the minimal set covering every wall; target='Wall3' → frames seeing that wall "
                "largest; target='Table0' → frames where that object is clearest (needs one prior "
                "render of the room). Returns {frames:[{frame,score}]}."
            ),
            parameters={
                "target": {"type": "string", "description": "'room', a wall (Wall3), or an object (Table0)."},
                "n": {"type": "integer", "description": "How many frames (default 4, max 10)."},
            },
            required=["target"],
        )
        super().__init__("select_views", schema)

    async def build(self, params: dict) -> SelectViewsInvocation:
        return SelectViewsInvocation(validate_tool_params(SelectViewsParams, params))
