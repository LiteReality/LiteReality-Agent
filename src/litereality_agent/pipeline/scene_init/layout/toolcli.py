"""toolcli.py — the tool surface a reasoning agent uses to repair a room.

The deterministic solver decides what to do from the geometry alone, and on the hard scenes that is
not enough: it can tell that a wardrobe overlaps a table, but not that the wardrobe belongs in the
corner and the table does not belong where the scan put it. That judgement needs someone to look at
the photographs and think, which is what this file is for.

It is a COMMAND LINE, not a JSON schema, because an agent that can run commands can look before it
acts, act, look again, and change its mind — and every one of those steps is recorded. The agent
never edits the layout: it calls these verbs, and the geometry underneath them keeps the arithmetic
honest (a snap computes its own distance; a corner fit computes its own trim; every mutation is
scored and can be undone).

    python -m physical_engine.toolcli <session> summary
    python -m physical_engine.toolcli <session> object Storage1
    python -m physical_engine.toolcli <session> attach Storage1 Wall3 Wall7
    python -m physical_engine.toolcli <session> undo

Every mutating verb prints the score before and after, so the agent always knows whether what it
just did helped. Lower is better, and the first number — defects — is collisions plus units knocked
off their wall plus furniture moved implausibly far.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from .adjust import check
from .repair import _corner_fit, _snap_delta, attachments, score
from .shell import load_shell, save_shell, wall_frame
from .graph import wall_distance


def _paths(session: Path):
    return session / "state.json", session / "baseline.json", session / "history.json"


def _load(session: Path):
    state_p, base_p, _ = _paths(session)
    state, baseline = load_shell(state_p), load_shell(base_p)
    held = {oid: set(attachments(o, baseline.get("walls") or {}))
            for oid, o in (baseline.get("objects") or {}).items()}
    return state, baseline, {k: v for k, v in held.items() if v}


def _save(session: Path, state, note: str):
    state_p, _, hist_p = _paths(session)
    history = json.loads(hist_p.read_text()) if hist_p.is_file() else []
    history.append({"note": note, "state": state})
    hist_p.write_text(json.dumps(history[-25:]), encoding="utf-8")
    save_shell(state, state_p)


def _score_line(state, baseline, held, prefix="score") -> str:
    defects, penetration, moved = score(state, baseline, held)
    return (f"{prefix}: defects={defects} penetration={penetration:.3f}m "
            f"displacement={moved:.3f}m")


def _violations(state) -> list[str]:
    return [f"{v.object} {v.kind} ({v.other or '-'}) {v.detail}"
            for v in check(state) if v.severity == "error"]


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    session, verb, args = Path(argv[1]), argv[2], argv[3:]
    state, baseline, held = _load(session)
    walls = state.get("walls") or {}
    objects = state.get("objects") or {}

    def finish(note):
        before = _score_line(_load(session)[0], baseline, held, "before")
        _save(session, state, note)
        print(before)
        print(_score_line(state, baseline, held, "after "))
        remaining = _violations(state)
        print(f"violations remaining: {len(remaining)}")
        for line in remaining:
            print("  " + line)
        return 0

    if verb == "summary":
        print(f"room: {len(walls)} walls, {len(objects)} objects")
        print(_score_line(state, baseline, held))
        print("\nobjects (category, size, walls it is installed against):")
        for oid, obj in sorted(objects.items()):
            held_now = sorted(attachments(obj, walls))
            print(f"  {oid:14} {obj.get('category',''):12} "
                  f"size={[round(v,2) for v in obj['size']]} walls={held_now or '-'}")
        print("\nviolations:")
        for line in _violations(state):
            print("  " + line)
        return 0

    if verb == "object":
        oid = args[0]
        obj = objects.get(oid)
        if not obj:
            print(f"no such object: {oid}")
            return 1
        print(json.dumps({"id": oid, "category": obj.get("category"),
                          "size": [round(v, 3) for v in obj["size"]],
                          "center": [round(v, 3) for v in obj["center"]],
                          "yaw": obj.get("yaw")}, indent=2))
        print("\nwalls (gap from this object's face, negative = it is inside the wall):")
        for wall_id, wall in sorted(walls.items()):
            measured = wall_distance(obj, wall)
            length = wall_frame(wall)[3]
            if measured is not None and abs(measured[0]) < 1.5:
                print(f"  {wall_id:8} gap {measured[0]*100:+7.1f} cm   length {length:.2f} m")
        print("\nneighbours within 2 m:")
        for other_id, other in sorted(objects.items()):
            if other_id == oid:
                continue
            d = math.dist(other["center"][:2], obj["center"][:2])
            if d < 2.0:
                print(f"  {other_id:14} {other.get('category',''):12} {d:.2f} m")
        refs = sorted((session / "references" / oid).glob("rank*.jpg"))[:3]
        print("\nreference photographs (this object's box drawn on the scan) — Read them:")
        for ref in refs:
            print(f"  {ref}")
        if not refs:
            print("  (none)")
        return 0

    if verb == "check":
        for line in _violations(state):
            print(line)
        print(_score_line(state, baseline, held))
        return 0

    if verb == "attach":
        oid, names = args[0], [a for a in args[1:] if a in walls]
        obj = objects.get(oid)
        if not obj or not names:
            print("usage: attach <object> <wall> [wall2]   (wall2 makes it a corner fit)")
            return 1
        if len(names) == 2:
            fitted = _corner_fit(obj, walls[names[0]], walls[names[1]])
            if fitted:
                obj["size"] = fitted
        for wall_id in names:
            delta = _snap_delta(obj, walls[wall_id])
            obj["center"][0] = round(obj["center"][0] + float(delta[0]), 4)
            obj["center"][1] = round(obj["center"][1] + float(delta[1]), 4)
        return finish(f"attach {oid} {' '.join(names)}")

    if verb == "move":
        oid, dx, dy = args[0], float(args[1]), float(args[2])
        obj = objects.get(oid)
        if not obj:
            return 1
        obj["center"][0] = round(obj["center"][0] + dx, 4)
        obj["center"][1] = round(obj["center"][1] + dy, 4)
        return finish(f"move {oid} {dx} {dy}")

    if verb == "resize":
        oid = args[0]
        obj = objects.get(oid)
        if not obj:
            return 1
        obj["size"] = [round(float(v), 4) for v in args[1:4]]
        return finish(f"resize {oid} {obj['size']}")

    if verb == "drop":
        oid = args[0]
        if oid not in objects:
            return 1
        objects.pop(oid)
        return finish(f"drop {oid} ({' '.join(args[1:]) or 'no reason given'})")

    if verb == "undo":
        _, _, hist_p = _paths(session)
        history = json.loads(hist_p.read_text()) if hist_p.is_file() else []
        if len(history) < 2:
            print("nothing to undo")
            return 1
        history.pop()
        save_shell(history[-1]["state"], _paths(session)[0])
        hist_p.write_text(json.dumps(history), encoding="utf-8")
        state, baseline, held = _load(session)
        print(f"undone. now: {history[-1]['note']}")
        print(_score_line(state, baseline, held))
        return 0

    print(f"unknown verb: {verb}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
