"""support.py — DETERMINISTIC support check for a built room. No model.

Nothing floats. Every object rests on the floor, hangs on a wall, sits on top of another object, or
hangs from the ceiling — and this reports the ones that do none of those, plus the ones that sink
into their support instead of resting on it, and the ones balanced off the edge of it.

Report only: unlike `fix.py` / `correct.py` there is no auto-repair here. A floating object is
almost never a placement that needs nudging down — it is a MISSING SUPPORT (the shelf it belongs on
was never built), and silently dropping it to the floor would hide the real defect. So this prints
what is wrong and leaves the repair to the authoring pass.

Needs a compiled `Room.glb`, and it must be NEWER than `Room.py`: `correct.py` moves objects after
the build, so a stale glb describes a room that no longer exists. Run this after the final compile.

    python -m litereality_agent.pipeline.room_qc.support --room <room dir>
    python -m litereality_agent.pipeline.room_qc.support --room <room dir> --graph
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

from litereality_agent.agent.tools.check_collisions.source import collision_mesh as sc
from litereality_agent.agent.tools.check_collisions.source import support as sp
from litereality_agent.agent.tools.check_collisions.source.geometry import _extract_shell


def _find_glb(room_dir: Path) -> Path | None:
    """Newest Room.glb at or beside the room dir — same discovery the other QC modules use."""
    cands = sorted(glob.glob(str(room_dir.parent) + "/**/Room.glb", recursive=True),
                   key=lambda p: Path(p).stat().st_mtime, reverse=True)
    return Path(cands[0]) if cands else None


def check(room_dir=None, shell_path=None) -> dict:
    room_dir = Path(room_dir) if room_dir else None
    shell_path = Path(shell_path or (room_dir / "Room.py"))
    SHELL = _extract_shell(shell_path.read_text())

    glb = _find_glb(room_dir) if room_dir else None
    if glb is None:
        return {"error": "no compiled Room.glb found — build the room first",
                "supports": {}, "graph": {}, "findings": []}

    bodies = sc.build_bodies(glb, SHELL)
    out = sp.find_supports(bodies, SHELL)
    out["source"] = glb.name
    out["stale"] = shell_path.stat().st_mtime > glb.stat().st_mtime
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--room", help="room dir containing Room.py (and a compiled Room.glb nearby)")
    ap.add_argument("--shell", help="Room.py path (overrides --room's Room.py)")
    ap.add_argument("--graph", action="store_true", help="also print the support scene graph")
    a = ap.parse_args()
    if not (a.room or a.shell):
        ap.error("need --room or --shell")
    r = check(a.room, a.shell)

    if r.get("error"):
        print(f"SUPPORT CHECK: {r['error']}")
        return 0

    findings, supports = r["findings"], r["supports"]
    print(f"SUPPORT CHECK {r['source']}")
    print(f"  objects: {len(supports)}   issues: {len(findings)}")
    if r.get("stale"):
        print("  ! Room.glb is older than Room.py — recompile for accurate results")
    if a.graph:
        for oid, s in sorted(supports.items()):
            arrow = f"→ {s['parent']}" if s["parent"] else "→ (nothing)"
            print(f"  {oid:16} {s['kind']:10} {arrow}")
    for f in findings:
        print(f"  ✗ {f['id']:16} {f['kind']:18} {f['detail']}")
    if not findings:
        print("  ✓ clean — everything rests on something")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
