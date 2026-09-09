"""check_collisions — PRIMITIVE: deterministic clash + containment report with snap/resize fixes.

The geometry QC the authoring/QC model cannot do reliably by eye: does any furniture piece
interpenetrate another, bury itself in a wall, or sit outside the room footprint — and if so, the
exact metric MOVE that fixes it (and the equivalent RESIZE). Pure arithmetic over the CURRENT
`SHELL` in Room.py (oriented boxes: center / size / yaw), so it reflects the model's latest edit
with no compile in between, and re-running it after an edit confirms the clash is gone.

Three kinds, all in world metres:
  * object_clash → two furniture boxes overlap. Fix = the minimum translation vector (shortest push,
    via the separating-axis theorem) that parts them, or shrink either footprint by that much.
  * wall_clash   → a furniture centre sits inside a wall slab. Fix = the push along the wall's INWARD
    normal that seats the piece flush against the wall (snap-to-wall), or trim its depth.
  * outside_room → a furniture centre falls outside the wall footprint. Fix = the move that brings it
    back onto the footprint (toward the nearest wall).

It DETECTS and SUGGESTS; it does not edit. The caller chooses move-vs-resize by eye against the
head-on stitches / photos (render + read_image) — a chair that should tuck under a desk moves, a
cabinet drawn too deep resizes — applies the change to `SHELL` with edit_code, then re-runs this.

Reuses the exact overlap math (and the "correct overlap" allow-list — a chair tucked under a table,
a sink in its counter) from authoring/qc_room.py, so what it flags matches the deterministic linter.
"""

from __future__ import annotations

import math
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

# metres — mirror qc_room's thresholds so this tool and the linter agree on what "a clash" is.
OBJ_PEN = 0.08       # min footprint interpenetration to report (below this = touching, not clashing)
Z_BAND = 0.05        # min shared height band for two boxes to be able to collide at all
WALL_PEN_TOL = 0.05  # how far an object's face may poke past a wall plane before it counts as through it
FOOT_MARGIN = 0.10   # how far a centre may sit past the footprint before it is "outside the room"

_HOW_TO_APPLY = (
    "Every fix is a world-space (Δx, Δy) in metres in the SHELL frame. To MOVE/snap: add it to that "
    "object's SHELL['objects'][id]['center'][0:2]. To RESIZE: reduce that object's 'size' by the "
    "shrink amount. Decide move-vs-resize by eye — render the wall/room and read_image against the "
    "head-on stitch: snap a piece that just drifted, resize one drawn too big/deep. Apply with "
    "edit_code, recompile, then re-run check_collisions to confirm 0 clashes."
)


class CheckCollisionsParams(ToolParamsModel):
    method: str = "mesh"  # "mesh" (default) = true meshes from Room.glb; "box" = fast SHELL-box fallback
    include_grounding: bool = True  # also report floor grounding (floating / sunk / poke-through)


def _obb(o: dict):
    """SHELL object → (cx, cy, hx, hy, yaw_rad, cz, hz), or None if it has no box."""
    c = o.get("center")
    s = o.get("size")
    if not c or not s:
        return None
    return (c[0], c[1], s[0] / 2, s[1] / 2, math.radians(o.get("yaw", 0) or 0), c[2], s[2] / 2)


def _support(obb, nx: float, ny: float) -> float:
    """Half-extent of an oriented box along the unit direction (nx, ny) — its support radius."""
    _cx, _cy, hx, hy, r, *_ = obb
    ux, uy = math.cos(r), math.sin(r)
    return hx * abs(ux * nx + uy * ny) + hy * abs(-uy * nx + ux * ny)


def _footprint(walls: dict):
    """Room footprint (xmin, ymin, xmax, ymax) + centre, from the wall endpoints. None if no walls."""
    pts = [p for w in walls.values() for p in (w.get("start"), w.get("end")) if p]
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys), (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2


def _round2(v):
    return round(float(v), 3)


class CheckCollisionsInvocation(SceneToolInvocation):
    def get_description(self) -> str:
        return "check_collisions: furniture clash + containment report"

    async def execute(self) -> ToolResult:
        # Lazy: geometry is light, but keep tool import time flat and match the other tools' pattern.
        from .source import geometry as qc_room

        try:
            room_dir = room_dir_from(self.scene_path)
        except ValueError as e:
            return ToolResult(error=str(e))
        room_py = room_dir / "Room.py"
        if not room_py.is_file():
            return ToolResult(error=f"no Room.py at {room_py}")

        SHELL = qc_room._extract_shell(room_py.read_text())
        objs = SHELL.get("objects") or {}
        walls = SHELL.get("walls") or {}
        if not objs:
            return ToolResult(error=(
                "SHELL has no readable 'objects' — Room.py may be mid-edit or SHELL is malformed. "
                "Fix Room.py so it imports, then retry."
            ))

        # DEFAULT: true-mesh. Everything (object↔object, wall, containment, grounding, openings) is
        # decided from the real meshes in Room.glb — no bounding boxes. `method='box'` is only a fast,
        # geometry-free fallback over the SHELL boxes when there is no compiled glb.
        if self.params.method == "mesh":
            return self._run_mesh(room_dir, SHELL)

        floor_z = SHELL.get("floor_z")
        ceil_z = SHELL.get("ceiling_z")
        foot = _footprint(walls)

        furn = [(oid, o) for oid, o in objs.items() if o.get("category") in qc_room.FURNITURE]
        boxes = [(oid, o, _obb(o)) for oid, o in furn]
        boxes = [b for b in boxes if b[2] is not None]

        clashes: list[dict] = []

        # ---- object ↔ object: oriented-box footprint overlap (SAT) within a shared height band ----
        # Collect EVERY overlapping box pair as a candidate (including chair↔table). How the
        # candidate is resolved depends on `method`: "box" drops the category allow-list pairs;
        # "mesh" keeps only pairs whose real meshes actually touch (so a tuck clears, a jam stays).
        obj_candidates = []  # (pair_frozenset, in_allowlist, finding)
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                (idi, oi, bi), (idj, oj, bj) = boxes[i], boxes[j]
                zov = min(bi[5] + bi[6], bj[5] + bj[6]) - max(bi[5] - bi[6], bj[5] - bj[6])
                if zov <= Z_BAND:
                    continue  # no shared height → cannot collide
                mtv = qc_room._obb_mtv_2d(bi[:5], bj[:5])
                if not mtv or mtv[0] <= OBJ_PEN:
                    continue
                depth, (ux, uy) = mtv  # axis oriented i → j: +axis pushes j away from i
                finding = {
                    "id": idj,
                    "kind": "object_clash",
                    "with": idi,
                    "overlap_m": _round2(depth),
                    "fix": {
                        "move": {"id": idj, "world_dxdy": [_round2(depth * ux), _round2(depth * uy)],
                                 "dist_m": _round2(depth)},
                        "or_move": {"id": idi, "world_dxdy": [_round2(-depth * ux), _round2(-depth * uy)]},
                        "or_resize": {"shrink_m": _round2(depth),
                                      "note": f"reduce {idi} or {idj} 'size' by ~{depth:.2f} m along the push axis"},
                    },
                }
                obj_candidates.append((frozenset((idi, idj)),
                                       qc_room._expected(oi.get("category"), oj.get("category")),
                                       finding))

        clashes.extend(f for _pair, allow, f in obj_candidates if not allow)

        # ---- furniture ↔ wall: object EXTENT crosses a wall plane to the outside → snap inward -------
        # Not a centre test: RoomPlan walls are ~0 mm thick, so "centre inside the slab" never fires.
        # We flag when the box's exterior face pokes PAST the wall plane (within the wall's span) — a
        # table poking through a wall.
        if walls:
            for oid, o, b in boxes:
                cx, cy = b[0], b[1]
                if foot and not (foot[0] - FOOT_MARGIN <= cx <= foot[2] + FOOT_MARGIN
                                 and foot[1] - FOOT_MARGIN <= cy <= foot[3] + FOOT_MARGIN):
                    continue  # centre is outside the room → that's outside_room's job, not wall_clash
                best = None
                for wid, w in walls.items():
                    s, e = w.get("start"), w.get("end")
                    if not s or not e:
                        continue
                    dx, dy = e[0] - s[0], e[1] - s[1]
                    L = math.hypot(dx, dy)
                    if L < 1e-9:
                        continue
                    ux, uy = dx / L, dy / L
                    nx, ny = -uy, ux
                    if foot and ((foot[4] - s[0]) * nx + (foot[5] - s[1]) * ny) < 0:
                        nx, ny = -nx, -ny  # toward the room interior
                    t = ((cx - s[0]) * ux + (cy - s[1]) * uy) / L
                    if not (-0.15 <= t <= 1.15):
                        continue  # object doesn't sit along this wall's span
                    sp = (cx - s[0]) * nx + (cy - s[1]) * ny   # centre offset, interior positive
                    pen = _support(b, nx, ny) - sp             # how far the exterior face pokes through
                    if pen > WALL_PEN_TOL and (best is None or pen > best[0]):
                        best = (pen, wid, nx, ny)
                if best:
                    pen, wid, nx, ny = best
                    mv = pen + 0.02
                    clashes.append({
                        "id": oid, "kind": "wall_clash", "wall": wid, "penetration_m": _round2(pen),
                        "fix": {
                            "snap_to_wall": {"id": oid, "world_dxdy": [_round2(nx * mv), _round2(ny * mv)],
                                             "dist_m": _round2(mv)},
                            "or_resize": {"shrink_m": _round2(pen),
                                          "note": f"trim {oid} depth so it clears {wid}"},
                        },
                    })

        # ---- furniture ↔ footprint: centre outside the room → snap back onto it --------------------
        if foot:
            xmin, ymin, xmax, ymax = foot[0], foot[1], foot[2], foot[3]
            for oid, o, b in boxes:
                cx, cy = b[0], b[1]
                inside = (xmin - FOOT_MARGIN <= cx <= xmax + FOOT_MARGIN
                          and ymin - FOOT_MARGIN <= cy <= ymax + FOOT_MARGIN)
                if inside:
                    continue
                tx = min(max(cx, xmin), xmax)
                ty = min(max(cy, ymin), ymax)
                clashes.append({
                    "id": oid,
                    "kind": "outside_room",
                    "fix": {"move_inside": {"id": oid,
                                            "world_dxdy": [_round2(tx - cx), _round2(ty - cy)],
                                            "dist_m": _round2(math.hypot(tx - cx, ty - cy))}},
                })

        # ---- grounding (secondary): floating / sunk / poking through floor or ceiling --------------
        grounding: list[dict] = []
        if self.params.include_grounding and floor_z is not None:
            for oid, o, b in boxes:
                bottom, top = b[5] - b[6], b[5] + b[6]
                gap = bottom - floor_z
                if gap > 0.08:
                    grounding.append({"id": oid, "check": "floating", "detail": f"{gap:.2f} m above floor"})
                elif gap < -0.05:
                    grounding.append({"id": oid, "check": "sunk", "detail": f"{-gap:.2f} m into floor"})
                if ceil_z is not None and top > ceil_z + 0.05:
                    grounding.append({"id": oid, "check": "above_ceiling",
                                      "detail": f"top {top:.2f} > ceiling {ceil_z:.2f}"})

        clashes.sort(key=lambda c: -(c.get("overlap_m") or c.get("fix", {}).get("snap_to_wall", {}).get("dist_m")
                                     or c.get("fix", {}).get("move_inside", {}).get("dist_m") or 0))
        KINDS = ("object_clash", "wall_clash", "outside_room", "opening_blocked", "open_swing_blocked")
        counts = {k: sum(c["kind"] == k for c in clashes) for k in KINDS}

        summary = (
            f"{len(clashes)} clash(es): {counts['object_clash']} object–object, "
            f"{counts['wall_clash']} into-wall, {counts['outside_room']} outside-room, "
            f"{counts['opening_blocked'] + counts['open_swing_blocked']} opening"
            + (f"; {len(grounding)} grounding issue(s)" if grounding else "")
        ) if clashes or grounding else "clean — no clashes, everything inside the walls and grounded"

        return ToolResult(output={
            "method": "box",
            "summary": summary,
            "n_furniture": len(boxes),
            "clashes": clashes,
            "grounding": grounding,
            "how_to_apply": _HOW_TO_APPLY,
        })

    def _run_mesh(self, room_dir, SHELL):
        """The default: every check from the real meshes in Room.glb — no bounding boxes."""
        import glob

        cands = sorted(glob.glob(str(room_dir.parent) + "/**/Room.glb", recursive=True),
                       key=lambda p: Path(p).stat().st_mtime, reverse=True)
        if not cands:
            return ToolResult(error=(
                "mesh collision needs a compiled Room.glb and none was found — run compile first, "
                "or pass method='box' for a quick SHELL-box pre-check."
            ))
        glb = Path(cands[0])
        try:
            from .source import collision_mesh as sc
            bodies = sc.build_bodies(glb, SHELL)
            clashes = sc.check_all(bodies, SHELL)
            for w in sc.open_clearance(bodies, SHELL):
                clashes.append({"id": w["id"], "kind": "open_swing_blocked",
                                "opening": w["opening"], "detail": w["detail"]})
        except ImportError:
            return ToolResult(error="mesh collision needs python-fcl (uv pip install python-fcl), "
                                    "or pass method='box'.")
        except Exception as e:  # noqa: BLE001 — a mesh hiccup should be a clear error, not a crash
            return ToolResult(error=f"mesh collision failed ({type(e).__name__}: {e}). "
                                    "Recompile Room.glb, or pass method='box'.")

        grounding = [c for c in clashes if c["kind"] in ("floating", "sunk", "above_ceiling")]
        clashes = [c for c in clashes if c["kind"] not in ("floating", "sunk", "above_ceiling")]
        clashes.sort(key=lambda c: -(c.get("overlap_m") or c.get("penetration_m")
                                     or c.get("fix", {}).get("snap_to_wall", {}).get("dist_m") or 0))
        KINDS = ("object_clash", "wall_clash", "outside_room", "opening_blocked", "open_swing_blocked")
        counts = {k: sum(c["kind"] == k for c in clashes) for k in KINDS}
        stale = (room_dir / "Room.py").stat().st_mtime > glb.stat().st_mtime
        summary = (
            f"{len(clashes)} clash(es): {counts['object_clash']} object–object, "
            f"{counts['wall_clash']} through-wall, {counts['outside_room']} outside-room, "
            f"{counts['opening_blocked'] + counts['open_swing_blocked']} opening"
            + (f"; {len(grounding)} grounding issue(s)" if grounding else "")
        ) if clashes or grounding else "clean — no mesh clashes, everything inside the walls and grounded"
        out = {
            "method": "mesh", "source": glb.name, "summary": summary,
            "n_furniture": len(bodies["furniture"]), "clashes": clashes, "grounding": grounding,
            "how_to_apply": _HOW_TO_APPLY,
        }
        if stale:
            out["stale_warning"] = "Room.glb is older than Room.py — recompile for accurate results."
        return ToolResult(output=out)


class CheckCollisionsTool(BaseDeclarativeTool):
    def __init__(self) -> None:
        schema = make_tool_schema(
            name="check_collisions",
            description=(
                "TRUE-MESH furniture collision check (default): from the real meshes in the compiled "
                "Room.glb — no bounding boxes — reports every furniture piece that interpenetrates "
                "another (object_clash), pokes through a wall (wall_clash), sits outside the room "
                "(outside_room), floats/sinks, or fouls an articulated door/window (opening_blocked / "
                "open_swing_blocked). Each comes with the exact metric MOVE (snap) and RESIZE that "
                "clears it. A chair correctly tucked under a desk reads CLEAR (real triangles, not "
                "boxes). Needs a compiled Room.glb — recompile after edits, then re-run to confirm 0. "
                "method='box' is only a fast SHELL-box pre-check when no glb is available."
            ),
            parameters={
                "method": {
                    "type": "string",
                    "enum": ["mesh", "box"],
                    "description": "'mesh' (default) = true meshes from Room.glb (accurate; needs a "
                                   "recent compile). 'box' = fast SHELL-box fallback, no compile.",
                },
                "include_grounding": {
                    "type": "boolean",
                    "description": "Also report floor grounding (floating / sunk / poke-through). Default true.",
                },
            },
            required=[],
        )
        super().__init__("check_collisions", schema)

    async def build(self, params: dict) -> CheckCollisionsInvocation:
        return CheckCollisionsInvocation(validate_tool_params(CheckCollisionsParams, params))
