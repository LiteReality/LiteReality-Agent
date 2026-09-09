"""qc_collision.py — DETERMINISTIC TRUE-MESH collision resolution for a built room. No model.

The box resolver (`qc_fix.py`) is fast but coarse: a bounding box fills a chair's or table's
concavity, so a chair correctly tucked under a desk reads as a clash. This resolves collisions from
the REAL meshes in the compiled `Room.glb` (`scene_collision.py`, FCL — the engine MoveIt uses), then
writes the resulting SHELL `center` nudges back into `Room.py`. It moves PLACEMENTS only and NEVER
touches an object.py — the reconstructed objects are final. Same reviewable two-lines-per-object diff
as `qc_fix`, and it reuses qc_fix's writer / validator / anchoring so behaviour stays consistent.

Auto-fixed (both carry an exact metric fix from the mesh):
  * object_clash — two furniture meshes interpenetrate → move the MOBILE piece off the anchored one.
  * wall_clash   — a piece pokes through a wall → snap it back inward.
Reported but not auto-fixed (no safe deterministic move): outside_room, opening_blocked, floating /
sunk / above_ceiling (grounding is `Room.py::_ground_furniture`'s job), and clashes where both pieces
are anchored to walls (needs a human / the model pass).

Needs a compiled `Room.glb` (build the room first) and `python-fcl`.

    python -m lrauthor.pipeline.compile.quality_check.correct --room <dir>          # dry run: print the plan
    python -m lrauthor.pipeline.compile.quality_check.correct --room <dir> --apply  # write the nudges into Room.py
"""

from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path

from lrauthor.agent.tools.check_collisions.source import collision_mesh as sc
from lrauthor.agent.tools.check_collisions.source.geometry import _extract_shell
from lrauthor.pipeline.compile.quality_check.fix import (
    FREE_STANDING,
    MAX_NUDGE,
    _anchored,
    _obb,
    _replace_center,
    _validate,
)

# scene_collision findings that come with a metric fix we can apply deterministically.
_FIXABLE = ("object_clash", "wall_clash")


def _find_glb(room_dir: Path) -> Path | None:
    """Newest Room.glb at or beside the room dir — same discovery the check_collisions tool uses."""
    cands = sorted(
        glob.glob(str(room_dir.parent) + "/**/Room.glb", recursive=True),
        key=lambda p: Path(p).stat().st_mtime,
        reverse=True,
    )
    return Path(cands[0]) if cands else None


def plan(room_dir=None, shell_path=None, max_nudge=MAX_NUDGE):
    """Detect true-mesh clashes and turn each fixable one into a validated SHELL nudge.

    Returns the same shape as `qc_fix.plan` (shell_path/src/objects/pos/before/after/moves/rejected/
    unresolvable/unmovable) so the CLI and `apply` are shared in spirit — plus `findings` (the full
    mesh report) and `error` when no glb is available.
    """
    room_dir = Path(room_dir) if room_dir else None
    shell_path = Path(shell_path or (room_dir / "Room.py"))
    src = shell_path.read_text()
    SHELL = _extract_shell(src)
    walls = SHELL.get("walls", {})
    shobjs = SHELL.get("objects") or {}

    glb = _find_glb(room_dir) if room_dir else None
    if glb is None:
        return {"error": "no compiled Room.glb found — build the room first", "shell_path": shell_path,
                "src": src, "objects": shobjs, "pos": {}, "before": [], "after": [], "moves": [],
                "rejected": [], "unresolvable": [], "unmovable": [], "findings": []}

    bodies = sc.build_bodies(glb, SHELL)
    findings = sc.check_all(bodies, SHELL)

    pos = {oid: [so["center"][0], so["center"][1]] for oid, so in shobjs.items()}
    start = {oid: list(p) for oid, p in pos.items()}
    # Mobility is judged ONCE from the original placement, exactly as qc_fix does: an object against
    # its wall is anchored and stays; the other piece in the pair gives way.
    anchored = {
        oid: (so.get("category") not in FREE_STANDING and _anchored(_obb(oid, so), walls))
        for oid, so in shobjs.items()
    }

    fixable = [f for f in findings if f["kind"] in _FIXABLE]
    req: dict[str, list] = {}          # oid -> [(dx, dy), ...] world-frame move vectors (already scaled)
    unresolvable: list = []
    unmovable: list = []

    for f in fixable:
        if f["kind"] == "object_clash":
            a, b = f["with"], f["id"]                 # scene_collision's vector moves b off a
            dx, dy = f["fix"]["move"]["world_dxdy"]
            ina, inb = a in pos, b in pos
            wa = 0.0 if (not ina or anchored.get(a)) else 1.0
            wb = 0.0 if (not inb or anchored.get(b)) else 1.0
            if wa + wb == 0:
                why = "neither has a SHELL box" if not (ina or inb) else "both against walls"
                unresolvable.append((a, b, f.get("overlap_m", 0.0), why))
                continue
            if inb and wb:
                req.setdefault(b, []).append((dx * wb / (wa + wb), dy * wb / (wa + wb)))
            if ina and wa:
                req.setdefault(a, []).append((-dx * wa / (wa + wb), -dy * wa / (wa + wb)))
        else:  # wall_clash — the object moves inward; the wall is fixed
            oid = f["id"]
            dx, dy = f["fix"]["snap_to_wall"]["world_dxdy"]
            if oid in pos:
                req.setdefault(oid, []).append((dx, dy))
            else:
                unmovable.append((oid, f.get("wall"), "no SHELL box (code-built fixture)"))

    # Merge requirements pointing the SAME way by taking the LARGEST, never summing — an object
    # wedged into a corner is reported once per wall, all pushing the same way; summing over-moves it.
    for oid, vecs in req.items():
        merged: dict = {}
        for dx, dy in vecs:
            L = math.hypot(dx, dy)
            if L < 1e-6:
                continue
            k = (round(dx / L, 2), round(dy / L, 2))
            merged[k] = max(merged.get(k, 0.0), L)
        for (ux, uy), mag in merged.items():
            pos[oid][0] += ux * mag
            pos[oid][1] += uy * mag

    moves, rejected = _validate(pos, start, shobjs, SHELL, walls, max_nudge)
    return {"shell_path": shell_path, "src": src, "objects": shobjs, "pos": pos,
            "before": fixable, "after": [], "moves": moves, "rejected": rejected,
            "unresolvable": unresolvable, "unmovable": unmovable, "findings": findings}


def apply(p: dict) -> str:
    """Write only the moved objects' `center` X/Y back into Room.py — height and all else untouched."""
    src = p["src"]
    for oid, _old, new, _dist in p["moves"]:
        src = _replace_center(src, oid, new)
    return src


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--room", help="room dir containing Room.py (and a compiled Room.glb nearby)")
    ap.add_argument("--shell", help="Room.py path (overrides --room's Room.py)")
    ap.add_argument("--apply", action="store_true", help="write the nudges into Room.py")
    ap.add_argument("--out", help="write to this path instead of in place")
    ap.add_argument("--max-nudge", type=float, default=MAX_NUDGE,
                    help=f"m; refuse corrections larger than this (default {MAX_NUDGE})")
    a = ap.parse_args()
    if not (a.room or a.shell):
        ap.error("need --room or --shell")
    p = plan(a.room, a.shell, max_nudge=a.max_nudge)

    if p.get("error"):
        print(f"COLLISION FIX (mesh): {p['error']}")
        return 0  # not a hard failure — the deterministic box pass / model pass can still run

    grounding = [f for f in p["findings"] if f["kind"] in ("floating", "sunk", "above_ceiling")]
    other = [f for f in p["findings"]
             if f["kind"] in ("outside_room", "opening_blocked", "open_swing_blocked")]
    print(f"COLLISION FIX (true-mesh) {p['shell_path']}")
    print(f"  mesh clashes: {len(p['before'])}   moves: {len(p['moves'])}   "
          f"other: {len(other)}   grounding: {len(grounding)}")
    for oid, o, n, d in p["moves"]:
        print(f"  ↔ {oid:16} {d:.3f} m   ({o[0]:.3f}, {o[1]:.3f}) → ({n[0]:.3f}, {n[1]:.3f})")
    for a_, b_, depth, why in p["unresolvable"]:
        print(f"  ! {a_:16} vs {b_}: {why}, {depth:.2f} m — needs the model pass / a human")
    for oid, wid, why in p["unmovable"]:
        print(f"  ! {oid:16} vs {wid}: {why}")
    for oid, dist, why in p["rejected"]:
        print(f"  ! {oid:16} reverted: {why}")
    for f in other:
        print(f"  · {f['id']:16} {f['kind']}: {f.get('detail', '')}")
    if not p["before"] and not other:
        print("  ✓ no mesh collisions to fix")
        return 0

    if (a.apply or a.out) and p["moves"]:
        dst = Path(a.out) if a.out else p["shell_path"]
        dst.write_text(apply(p))
        print(f"  wrote {dst}  — rebuild the room, then re-run to verify 0 clashes")
    elif p["moves"]:
        print("  (dry run — pass --apply to write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
