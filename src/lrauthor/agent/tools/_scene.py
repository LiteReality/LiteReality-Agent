"""Shared helpers to resolve the active scene the harness binds to a tool.

A `SceneToolInvocation` is bound with `scene_path` — either the `Room.py` file or the room
directory that contains it. These normalize both.
"""

from __future__ import annotations

from pathlib import Path


def room_dir_from(scene_path: str | None) -> Path:
    if not scene_path:
        raise ValueError("no scene bound — harness must bind the room path")
    p = Path(scene_path)
    return p.parent if p.suffix == ".py" else p


def room_py_from(scene_path: str | None) -> Path:
    return room_dir_from(scene_path) / "Room.py"
