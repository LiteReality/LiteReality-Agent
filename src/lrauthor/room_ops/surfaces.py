"""Which surfaces a room actually has, parsed straight from its `Room.py`.

Deliberately dependency-free (`re` + `math`, no pydantic, no tool registry) so the prompt-building
passes — author / materials / qc / eval — can ask "what surfaces am I authoring?" without importing
the whole tool stack. `lrauthor.agent.tools.survey` re-exports `surface_ids` for its own callers.

The count is DISCOVERED, never assumed. A hardcoded `Wall0..WallN` silently drops every wall past
the cap from the prompt and from the coverage checks, so a run reports full coverage while most of
the room was never referenced — a 30-wall scan authored as if it had 9.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

# RoomPlan corner artifacts: sub-threshold stub walls whose stitches are ~20px smears and whose
# ortho renders are skipped anyway (render_ortho's own sliver threshold).
SLIVER_WALL_M = 0.35


def surface_ids(room_py: Path | str) -> list[str]:
    """REAL surface ids in this Room.py: Wall0..N (sliver stubs excluded) + Floor0 + Ceiling0.

    RoomPlan emits corner-artifact stub walls (e.g. a 0.2m 'Wall9'); their stitches are useless
    ~20px smears and their ortho renders are skipped anyway (<0.35m = render_ortho's own sliver
    threshold) — surveying/planning them wastes reads and fix attempts on junk."""
    try:
        src = Path(room_py).read_text()
    except Exception:  # noqa: BLE001
        return ["Floor0", "Ceiling0"]
    lengths = {}
    for m in re.finditer(
        r'"(Wall\d+)":\s*\{\s*"start":\s*\[\s*([-\d.eE]+)\s*,\s*([-\d.eE]+)\s*\]\s*,'
        r'\s*"end":\s*\[\s*([-\d.eE]+)\s*,\s*([-\d.eE]+)\s*\]', src,
    ):
        name, x0, y0, x1, y1 = m.group(1), *map(float, m.groups()[1:])
        lengths[name] = math.hypot(x1 - x0, y1 - y0)
    all_walls = sorted(set(re.findall(r"\bWall\d+\b", src)), key=lambda w: int(w[4:]))
    walls = [w for w in all_walls if lengths.get(w, 1.0) >= SLIVER_WALL_M]
    skipped = [w for w in all_walls if w not in walls]
    if skipped:
        print(f"  surface_ids: skipping sliver stub wall(s) {skipped} (<{SLIVER_WALL_M}m)", flush=True)
    return walls + ["Floor0", "Ceiling0"]
