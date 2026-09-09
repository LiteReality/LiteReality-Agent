"""checks.py — DETERMINISTIC geometry QC for a built room. No model. Reads room_layout.json
(world-space AABB per object) + the SHELL (floor_z/ceiling_z/walls/openings) and flags:

  below_floor / above_ceiling  — object pokes through the floor or ceiling
  floating / sunk              — floor-standing furniture not resting on the floor
  outside_room                 — object centre outside the room footprint
  wall_clash                   — furniture penetrates a wall slab
  object_clash                 — two furniture pieces interpenetrate
  fixture_over_opening         — a wall fixture sits over a door/window opening

Overlap that is CORRECT is not reported: an undermount sink inside its counter, a built-in oven in
the cabinet run, a chair tucked under a table, a socket on its trunking (see EXPECTED_CONTAINMENT /
PASSTHROUGH). `pipeline/room_qc/fix.py` resolves what IS reported by nudging furniture apart.

Everything is pure arithmetic over AABBs, so it's fast, exact, and needs no LLM.

The box arithmetic itself lives with the tool that runs it on every agent turn
(`agent/tools/check_collisions/source/geometry.py`). This module owns the report because the
pipeline decides whether the room passes.

    python -m lrauthor.pipeline.compile.qc.checks --room <room dir>
    python -m lrauthor.pipeline.compile.qc.checks --layout <room_layout.json> --shell <Room.py>
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

from lrauthor.agent.tools.check_collisions.source.geometry import (
    EXPECTED_FIXTURE_OVERLAP,
    FIXTURE_PEN,
    FLOAT_TOL,
    FLOOR_STANDING,
    FURNITURE,
    PASSTHROUGH,
    SUNK_TOL,
    WALL_MOUNTED,
    Z_TOL,
    _expected,
    _extract_shell,
    _obb_penetration_2d,
    assign_wall,
    wall_span,
)


def _load(room_dir=None, layout=None, shell=None):
    if layout is None:
        cands = sorted(glob.glob(str(Path(room_dir).parent) + "/**/room_layout.json", recursive=True),
                       key=lambda p: Path(p).stat().st_mtime, reverse=True)
        layout = cands[0] if cands else None
    if shell is None:
        shell = str(Path(room_dir) / "Room.py")
    objs = json.loads(Path(layout).read_text())
    objs = objs if isinstance(objs, list) else objs.get("objects", [])
    SHELL = _extract_shell(Path(shell).read_text())
    return objs, SHELL, layout


def qc(room_dir=None, layout=None, shell=None):
    objs, SHELL, layout_path = _load(room_dir, layout, shell)
    byid = {o["id"]: o for o in objs}
    floor_z = SHELL.get("floor_z", byid.get("Floor0", {}).get("top_z", -1e9))
    ceil_z = SHELL.get("ceiling_z", byid.get("Ceiling0", {}).get("top_z", 1e9))
    floor = byid.get("Floor0")
    fmin, fmax = (floor["bbox_min"], floor["bbox_max"]) if floor else ([-1e9] * 3, [1e9] * 3)
    furn = [o for o in objs if o["category"] in FURNITURE]

    V = []  # (object, check, detail)
    for o in furn:
        bmin, bmax, c = o["bbox_min"], o["bbox_max"], o["center"]
        if bmin[2] < floor_z - Z_TOL:
            V.append((o["id"], "below_floor", f"bottom {bmin[2]:.2f} < floor {floor_z:.2f}"))
        if bmax[2] > ceil_z + Z_TOL:
            V.append((o["id"], "above_ceiling", f"top {bmax[2]:.2f} > ceiling {ceil_z:.2f}"))
        if o["category"] in FLOOR_STANDING:
            gap = bmin[2] - floor_z
            if gap > FLOAT_TOL:
                V.append((o["id"], "floating", f"{gap:.2f} m above the floor"))
            elif gap < -SUNK_TOL:
                V.append((o["id"], "sunk", f"{-gap:.2f} m into the floor"))
        if not (fmin[0] - 0.10 <= c[0] <= fmax[0] + 0.10 and fmin[1] - 0.10 <= c[1] <= fmax[1] + 0.10):
            V.append((o["id"], "outside_room", f"centre ({c[0]:.2f},{c[1]:.2f}) outside footprint"))
        # wall clash: use the ACTUAL wall segment (SHELL start/end/thickness), not the AABB — walls are
        # often diagonal, so AABBs are fat and useless. Flag when the object CENTRE lies inside the wall
        # slab (perp distance to the wall line < thickness/2, within the segment span). Furniture resting
        # against a wall has its centre ~half-its-depth away, well outside the thin slab → no false hit.
        for wid, w in (SHELL.get("walls") or {}).items():
            sx, sy = w["start"]
            ex, ey = w["end"]
            dx, dy = ex - sx, ey - sy
            L2 = dx * dx + dy * dy
            if L2 == 0:
                continue
            t = ((c[0] - sx) * dx + (c[1] - sy) * dy) / L2
            if not (0.0 <= t <= 1.0):
                continue  # centre projects off the ends of this wall
            px, py = sx + t * dx, sy + t * dy
            perp = ((c[0] - px) ** 2 + (c[1] - py) ** 2) ** 0.5
            if perp < w.get("thickness", 0.1) / 2 + 0.02:  # centre is inside the wall slab
                V.append((o["id"], "wall_clash", f"centre inside {wid} (perp {perp:.2f} m)"))
                break
    # object clash: ORIENTED-box footprint overlap (SAT) + vertical overlap, from SHELL center/size/yaw.
    # Skips chair-under-table tucks. Oriented boxes remove the rotated-furniture false positives.
    shobjs = SHELL.get("objects", {})

    def _obb(o):
        so = shobjs.get(o["id"])
        if not so:
            return None
        cx, cy, cz = so["center"]
        w, d, h = so["size"]
        return (cx, cy, w / 2, d / 2, math.radians(so.get("yaw", 0)), cz, h / 2)

    fo = [(o, _obb(o)) for o in furn]
    for i in range(len(fo)):
        for j in range(i + 1, len(fo)):
            (oi, bi), (oj, bj) = fo[i], fo[j]
            if bi is None or bj is None or _expected(oi["category"], oj["category"]):
                continue
            zov = min(bi[5] + bi[6], bj[5] + bj[6]) - max(bi[5] - bi[6], bj[5] - bj[6])
            if zov <= 0.05:
                continue  # don't share a height band → can't collide
            pen = _obb_penetration_2d(bi[:5], bj[:5])
            if pen and pen > 0.08:
                V.append((oi["id"], "object_clash", f"footprint into {oj['id']} ~{pen:.2f} m"))

    # ---- wall fixtures -----------------------------------------------------------------------
    # Everything below works in each wall's OWN plane (along-wall metres × height), never in world
    # AABBs: walls are often diagonal, and a diagonal thin plate's world AABB is fat enough to
    # overlap half the room.
    walls_shell = SHELL.get("walls", {})
    fixtures = [o for o in objs if o["category"] in WALL_MOUNTED]
    mounted = []  # (obj, wall_id, t0, t1, z0, z1, d0, d1)
    for o in fixtures:
        hit = assign_wall(o["center"], walls_shell)
        if not hit:
            continue  # not near any wall (ceiling-mounted, or free-standing) — not our business
        span = wall_span(o, walls_shell[hit[0]])
        if span:
            mounted.append((o, hit[0], *span))

    # NB: fixture ↔ fixture overlap is intentionally NOT flagged — two wall fixtures sharing a patch
    # of wall happens legitimately (e.g. a socket on trunking, stacked plates) and was noisy, so it
    # was removed. Only fixtures over an OPENING are still reported below.

    # fixture over a door/window opening (opening rect taken from the SHELL, same wall frame)
    for oid, op in (SHELL.get("openings") or {}).items():
        w = walls_shell.get(op["wall"])
        if not w:
            continue
        o0, o1 = op["offset"], op["offset"] + op["width"]
        sill = SHELL["floor_z"] + op.get("sill", 0)
        z0, z1 = sill, sill + op["height"]
        for o, wid, t0, t1, fz0, fz1, _d0, _d1 in mounted:
            # a window's own frame belongs over its opening — that's the frame doing its job
            if (wid != op["wall"] or o["category"] in PASSTHROUGH
                    or _expected(o["category"], op.get("type", ""), EXPECTED_FIXTURE_OVERLAP)):
                continue
            ov_t = min(t1, o1) - max(t0, o0)
            ov_z = min(fz1, z1) - max(fz0, z0)
            if ov_t > FIXTURE_PEN and ov_z > FIXTURE_PEN:
                V.append((o["id"], "fixture_over_opening",
                          f"sits over {oid} on {op['wall']} ~{min(ov_t, ov_z):.2f} m"))

    return {"layout": layout_path, "n_furniture": len(furn), "n_fixtures": len(mounted),
            "violations": V}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--room")
    ap.add_argument("--layout")
    ap.add_argument("--shell")
    a = ap.parse_args()
    r = qc(a.room, a.layout, a.shell)
    V = r["violations"]
    print(f"QC {r['layout']}\n  furniture: {r['n_furniture']}  "
          f"wall fixtures: {r.get('n_fixtures', 0)}  violations: {len(V)}")
    for oid, chk, detail in V:
        print(f"  ✗ {oid:16} {chk:20} {detail}")
    if not V:
        print("  ✓ clean — no geometric violations")
    return 1 if V else 0


if __name__ == "__main__":
    raise SystemExit(main())
