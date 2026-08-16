"""collision.py — THE collision gate: one glb in, pass or fail out. Reads only, never edits.

This is the deterministic half of collision QC, kept deliberately separate from the half that
repairs anything:

    collision.py   is this room physically possible?   -> findings + exit code   (no writes)
    correct.py     make it possible                    -> nudges written into Room.py

They run together in `publish`, but they are different questions and only one of them is allowed
to change the room. Anything that decides "clean or not" belongs here.

A stamped glb carries everything the checks need (`room_ops/glb_meta`), so this takes a FILE, not
a room directory — no `Room.py`, no `room_layout.json`, no picking the newest of five candidates
by mtime. `--room` remains for glbs built before stamping existed.

    python -m litereality_agent.pipeline.room_qc.collision --glb <Room.glb>
    python -m litereality_agent.pipeline.room_qc.collision --glb <Room.glb> --room <room dir>
    python -m litereality_agent.pipeline.room_qc.collision --run-root run      # every built room

Exit code is the gate: 0 clean, 1 violations, 2 nothing to check.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

# grouped for reporting: a clash is a fault in the room, grounding is usually a MISSING support
# object, and containment means the reconstruction put something outside its own walls.
_CLASH = ("object_clash", "wall_clash", "opening_blocked")
_GROUNDING = ("floating", "sunk", "above_ceiling")
_PLACEMENT = ("outside_room", "open_swing_blocked")

_MARK = {"clash": "✗", "grounding": "▲", "placement": "·"}


def _bucket(kind: str) -> str:
    if kind in _CLASH:
        return "clash"
    if kind in _GROUNDING:
        return "grounding"
    return "placement"


def check(glb: Path, room: Path | None = None) -> list[dict]:
    """Every collision finding for one built room. Pure: reads the glb (and `Room.py` only when the
    glb predates stamping), writes nothing except the convex-decomposition cache beside it."""
    from litereality_agent.agent.tools.check_collisions.source import collision_mesh as sc

    shell = None
    if room and (room / "Room.py").is_file():
        from litereality_agent.room_ops.shell import extract_shell

        shell = extract_shell((room / "Room.py").read_text())
    return sc.check_glb(glb, shell)


def report(glb: Path, findings: list[dict], *, quiet: bool = False) -> int:
    """Print the findings; return the count that should FAIL the gate (clashes only).

    Grounding and placement are reported but do not fail: a `floating` object usually means a
    missing support, and `outside_room` means the scan's wall loop is wrong — neither is something
    the collision resolver can or should fix, so failing on them would block every publish on a
    problem that lives upstream.
    """
    by = Counter(_bucket(f["kind"]) for f in findings)
    if not quiet:
        print(f"COLLISION {glb}")
        print(f"  clashes: {by['clash']}   grounding: {by['grounding']}   "
              f"placement: {by['placement']}")
        for f in sorted(findings, key=lambda f: (_bucket(f["kind"]), f["kind"], f["id"])):
            b = _bucket(f["kind"])
            metric = f.get("overlap_m") or f.get("penetration_m")
            detail = f.get("detail") or (f"into {f.get('with') or f.get('wall')}"
                                         + (f" ~{metric:.3f} m" if metric else ""))
            how = f" [{f['measured_by']}]" if f.get("measured_by") else ""
            print(f"  {_MARK[b]} {f['id']:16} {f['kind']:18} {detail}{how}")
        if not findings:
            print("  ✓ clean — no collisions")
    return by["clash"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glb", type=Path, help="a built Room.glb")
    ap.add_argument("--room", type=Path, help="room dir, for glbs built before metadata stamping")
    ap.add_argument("--run-root", type=Path, help="check every built room under this directory")
    ap.add_argument("--json", action="store_true", help="emit findings as JSON instead")
    a = ap.parse_args()

    targets: list[tuple[Path, Path | None]] = []
    if a.run_root:
        for glb in sorted(a.run_root.glob("*/realism_authoring/room_preview/Room.glb")):
            targets.append((glb, glb.parent.parent / "room"))
    elif a.glb:
        targets = [(a.glb, a.room)]
    else:
        ap.error("need --glb or --run-root")
    if not targets:
        print("no Room.glb to check")
        return 2

    failed, out = 0, {}
    for glb, room in targets:
        findings = check(glb, room if a.room or a.run_root else None)
        out[str(glb)] = findings
        failed += report(glb, findings, quiet=a.json)
        if not a.json and len(targets) > 1:
            print()
    if a.json:
        print(json.dumps(out, indent=2))
    if len(targets) > 1 and not a.json:
        print(f"{len(targets)} rooms checked — {failed} clash(es) total")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
