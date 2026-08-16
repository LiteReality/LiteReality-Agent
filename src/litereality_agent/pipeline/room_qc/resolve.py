"""resolve.py — detect, nudge, re-detect, until the room is clean or it stops improving.

`correct.py` does ONE pass: it reads the built glb, computes a nudge per clash, writes it into
`Room.py`, and prints "rebuild the room, then re-run to verify 0 clashes". Nothing ever did that.
`publish` runs it exactly once, so the room ships with whatever the first pass left behind — which
is why Panda-2 went 8 moves, then 6, then still 6 across successive runs. Each pass helps and
nothing converges.

This closes the loop. The trick is that verification does NOT need a Blender rebuild: a fix only
translates objects in the floor plane, so the same translation applied to the in-memory meshes
gives exactly the geometry the rebuild would produce. So the loop runs entirely in memory —

    detect (true mesh)  ->  weight & validate  ->  translate the meshes  ->  detect again

— and `Room.py` is written ONCE, with the accumulated offsets, after it converges. One rebuild at
the end instead of one per round, and the convex-decomposition cache still hits every round
because a translated mesh has the same canonical (centred) geometry.

Mobility weights are frozen at the ORIGINAL placement, as the collision-check doc requires: if they
were re-derived each round, an object nudged off its wall would become "mobile" mid-solve and drift
across the room.

    python -m litereality_agent.pipeline.room_qc.resolve --room <room dir>
    python -m litereality_agent.pipeline.room_qc.resolve --room <room dir> --apply
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from litereality_agent.agent.tools.check_collisions.source import collision_mesh as sc
from litereality_agent.pipeline.room_qc.correct import _FIXABLE, _find_glb
from litereality_agent.pipeline.room_qc.fix import (
    FREE_STANDING,
    MAX_NUDGE,
    SEPARATION,
    _anchored,
    _obb,
    _replace_center,
    _validate,
)

MAX_ROUNDS = 12          # Panda-2 needs 7; the cap is a runaway guard, not a budget


def _round_moves(findings, pos, anchored, shobjs):
    """One round's requested translation per object, in the SHELL plane.

    Same weighting as `correct.plan`: the anchored side of a pair holds, the mobile side gives way,
    and corrections pointing the same way are merged by taking the LARGEST rather than summing (an
    object wedged into a corner is reported once per wall, all pushing the same direction).
    """
    req: dict[str, list] = {}
    unresolvable = []
    for f in (x for x in findings if x["kind"] in _FIXABLE):
        if f["kind"] == "object_clash":
            a, b = f["with"], f["id"]
            dx, dy = f["fix"]["move"]["world_dxdy"]
            wa = 0.0 if (a not in pos or anchored.get(a)) else 1.0
            wb = 0.0 if (b not in pos or anchored.get(b)) else 1.0
            if wa + wb == 0:
                unresolvable.append((a, b, f.get("overlap_m", 0.0), "both against walls"))
                continue
            if wb:
                req.setdefault(b, []).append((dx * wb / (wa + wb), dy * wb / (wa + wb)))
            if wa:
                req.setdefault(a, []).append((-dx * wa / (wa + wb), -dy * wa / (wa + wb)))
        else:  # wall_clash — the wall is fixed, the object moves inward
            oid = f["id"]
            if oid in pos:
                req.setdefault(oid, []).append(tuple(f["fix"]["snap_to_wall"]["world_dxdy"]))

    merged: dict[str, tuple] = {}
    for oid, vecs in req.items():
        by_dir: dict = {}
        for dx, dy in vecs:
            L = math.hypot(dx, dy)
            if L < 1e-6:
                continue
            k = (round(dx / L, 2), round(dy / L, 2))
            by_dir[k] = max(by_dir.get(k, 0.0), L)
        mx = my = 0.0
        for (ux, uy), mag in by_dir.items():
            mx += ux * mag
            my += uy * mag
        L = math.hypot(mx, my)
        if L > 1e-6:
            # Overshoot by SEPARATION. The MTV is by definition the move that brings the pair to
            # EXACTLY touching, which FCL still reports as contact — so without this the loop
            # nudges forever and never clears the clash it just resolved.
            s = (L + SEPARATION) / L
            merged[oid] = (mx * s, my * s)
    return merged, unresolvable


def resolve(room_dir=None, shell_path=None, max_nudge=MAX_NUDGE, max_rounds=MAX_ROUNDS):
    """Iterate detect->nudge until no clashes remain, no move is available, or the cap is hit."""
    room_dir = Path(room_dir) if room_dir else None
    shell_path = Path(shell_path or (room_dir / "Room.py"))
    src = shell_path.read_text()

    glb = _find_glb(room_dir) if room_dir else None
    if glb is None:
        return {"error": "no compiled Room.glb found — build the room first",
                "shell_path": shell_path, "src": src, "moves": [], "rounds": [],
                "converged": False}

    shell = sc.shell_for(glb)
    if not shell:
        from litereality_agent.room_ops.shell import extract_shell

        shell = extract_shell(src)
    shobjs = shell.get("objects") or {}
    walls = shell.get("walls") or {}

    bodies = sc.build_bodies(glb, shell)
    fur = bodies["furniture"]

    # positions we are solving for; `_obb`/`_validate` need center+size, which a stamped glb does
    # not carry (deliberately — it recovers from the mesh). Fall back to Room.py for those.
    if not all("center" in o and "size" in o for o in shobjs.values()):
        from litereality_agent.room_ops.shell import extract_shell

        py = extract_shell(src).get("objects") or {}
        for oid, o in py.items():
            shobjs.setdefault(oid, {}).update(o)

    pos = {oid: [so["center"][0], so["center"][1]] for oid, so in shobjs.items() if "center" in so}
    start = {oid: list(p) for oid, p in pos.items()}
    anchored = {oid: (so.get("category") not in FREE_STANDING and _anchored(_obb(oid, so), walls))
                for oid, so in shobjs.items() if "center" in so}

    # WHY the loop stopped is the useful part of the report: "give me more rounds" and "this needs
    # a human" look identical from a clash count alone, and only one of them is worth retrying.
    rounds, converged, unresolvable, stop = [], False, [], "max_rounds"
    for n in range(max_rounds):
        findings = sc.check_all(bodies, shell)
        clashes = [f for f in findings if f["kind"] in _FIXABLE]
        rounds.append({"round": n, "clashes": len(clashes), "findings": findings})
        if not clashes:
            converged, stop = True, "converged"
            break

        merged, unres = _round_moves(clashes, pos, anchored, shobjs)
        unresolvable = unres
        if not merged:
            stop = "nothing_movable"  # both sides of every remaining pair are pinned to a wall
            break

        applied = 0
        for oid, (dx, dy) in merged.items():
            trial = [pos[oid][0] + dx, pos[oid][1] + dy]
            probe = dict(pos)
            probe[oid] = trial
            ok, _rej = _validate(probe, start, shobjs, shell, walls, max_nudge)
            if not any(m[0] == oid for m in ok) and math.hypot(
                    trial[0] - start[oid][0], trial[1] - start[oid][1]) > max_nudge:
                continue  # this object cannot legally get there; leave it for the report
            pos[oid] = trial
            # keep the in-memory mesh in step: SHELL(x, y) = glb(x, -z)
            if oid in fur:
                fur[oid].apply_translation((dx, 0.0, -dy))
            applied += 1
        if not applied:
            stop = "capped"  # every remaining move is larger than MAX_NUDGE — a placement error
            break

    moves, rejected = _validate(pos, start, shobjs, shell, walls, max_nudge)
    return {"shell_path": shell_path, "src": src, "glb": glb, "objects": shobjs,
            "moves": moves, "rejected": rejected, "unresolvable": unresolvable,
            "rounds": rounds, "converged": converged, "stop": stop,
            "final": rounds[-1]["findings"] if rounds else []}


def apply(p: dict) -> str:
    src = p["src"]
    for oid, _old, new, _dist in p["moves"]:
        src = _replace_center(src, oid, new)
    return src


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--room", required=True)
    ap.add_argument("--apply", action="store_true", help="write the accumulated nudges into Room.py")
    ap.add_argument("--max-rounds", type=int, default=MAX_ROUNDS)
    ap.add_argument("--max-nudge", type=float, default=MAX_NUDGE)
    a = ap.parse_args()

    p = resolve(a.room, max_nudge=a.max_nudge, max_rounds=a.max_rounds)
    if p.get("error"):
        print(f"RESOLVE: {p['error']}")
        return 0  # not a hard failure — the box pass can still run

    print(f"RESOLVE {p['shell_path']}")
    for r in p["rounds"]:
        print(f"  round {r['round']}: {r['clashes']} clash(es)")
    for oid, o, n, d in p["moves"]:
        print(f"  ↔ {oid:16} {d:.3f} m   ({o[0]:.3f}, {o[1]:.3f}) → ({n[0]:.3f}, {n[1]:.3f})")
    for a_, b_, depth, why in p["unresolvable"]:
        print(f"  ! {a_:16} vs {b_}: {why}, {depth:.2f} m — needs a human")
    for oid, dist, why in p["rejected"]:
        print(f"  ! {oid:16} reverted: {why}")

    left = p["rounds"][-1]["clashes"] if p["rounds"] else "?"
    why = {
        "converged": f"✓ converged in {len(p['rounds']) - 1} round(s) — 0 clashes remain",
        "nothing_movable": f"✗ stuck — {left} clash(es) where BOTH pieces are wall-anchored; "
                           "needs the model pass or a human",
        "capped": f"✗ stuck — {left} clash(es) need a move larger than {a.max_nudge} m; "
                  "that is a placement error upstream, not a nudge",
        "max_rounds": f"✗ hit the {a.max_rounds}-round cap with {left} clash(es) left "
                      "— retry with --max-rounds",
    }[p["stop"]]
    print(f"  {why}")

    if a.apply and p["moves"]:
        p["shell_path"].write_text(apply(p))
        print(f"  wrote {p['shell_path']} — rebuild, then the collision gate should pass")
    elif p["moves"]:
        print("  (dry run — pass --apply to write)")
    return 0 if p["converged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
