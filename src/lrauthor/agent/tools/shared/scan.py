"""Resolve which scan a room belongs to — the single choke point every image tool passes through.

`render`, `select_views` and the render engine all need the same two answers: which scan is this
room part of, and which harness config does that scan resolve to. This lived inside
`engine/compose.py`, which was fine while the engine was one shared package. Now that each tool
owns its own `source/`, keeping it there would mean `select_views` importing out of `render`'s
source — so it moves up to where both can reach it without depending on each other.

It stays ONE copy on purpose. `tests/test_scan_inference.py` pins that, and records why: when
this raised, the authoring model lost render-verify after a single failed call and authored a room
blind for 88 tool calls. A per-tool copy is exactly how that comes back.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["STAGE_ROOTS", "config_for", "scan_from_room"]

# `run/` is the one stage root. It used to be two -- the CLI symlinked the staging and
# deliverables spellings of `scene_init/scene_stage`, so the same room had two legitimate
# spellings and matching only one of them raised for every real run. Since `render`,
# `select_views` and `survey` all funnel through here, the model lost render-verify after one
# failed call and authored the room blind. Keep the realpath check below even with a single
# root: a room dir reached through any symlink still has to resolve.
STAGE_ROOTS = ("run",)


def config_for(scan: str):
    """Import the harness config with LITEREALITY_SCAN set to `scan` (config reads it at import)."""
    os.environ["LITEREALITY_SCAN"] = scan
    from lrauthor.agent.tools.shared import config

    config.ensure_dirs()  # the render writes here

    return config


def scan_from_room(room_dir: Path) -> str:
    """Recover the scan name from a room dir: `run/<scan>/scene_init/scene_stage/.../room`.

    Checks the path AS GIVEN and its realpath, so a room reached through a symlink still resolves.
    Falls back to `$LITEREALITY_SCAN`, which the CLI exports for every stage, so a room dir outside
    the stage tree still resolves instead of killing the tool.
    """
    room_dir = Path(room_dir)
    for candidate in (room_dir, room_dir.resolve()):
        for p in candidate.parents:
            if p.parent.name in STAGE_ROOTS:
                return p.name
    scan = os.environ.get("LITEREALITY_SCAN")
    if scan:
        return scan
    raise ValueError(f"could not infer scan from room dir {room_dir} — pass scan= explicitly")
