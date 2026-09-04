"""adapter.py — the RoomPlan pkl set on disk, as the SHELL the repair works on, and back again.

Two coordinate conventions meet here and the whole module exists to keep them apart.

``scene_data`` is ARKit's: **Y is up**, the floor is the XZ plane, and a wall carries a centre, a
rotation about Y and a bbox. The SHELL is the repair's: **Z is up**, the floor is XY, walls are
2-D centrelines with a thickness, objects are oriented footprints with a yaw. Every conversion is
in this file and nowhere else, so a sign error has one place to hide rather than a dozen.

Reading is delegated to :mod:`.scene_data`, which recovers a wall's centreline from its transform
matrix rather than its Euler angle — ``R = Ry(theta) . Rz(180)`` does not decompose the obvious
way, and getting it wrong tilts every wall in the room by a few degrees.

Writing back is deliberately narrow. The repair may move an object, resize it, or drop it; it may
not invent one, and it never touches walls, floor or openings — those are the room's structure and
a layout pass has no business rewriting them. So ``apply_to_objects`` edits ``position`` and
``bbox`` in place on the entries it recognises and removes the dropped ones, leaving every other
field (``file``, ``mesh_id``, ``rotation``, ``top_down_rect``) exactly as extraction wrote it.
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["shell_from_scene_data", "apply_to_objects", "load_objects"]


def _as_list(value) -> list[float]:
    return [float(v) for v in np.asarray(value).reshape(-1)]


def load_objects(scene_data_dir: str | Path) -> list[dict]:
    path = Path(scene_data_dir) / "objects.pkl"
    return pickle.load(open(path, "rb")) if path.is_file() else []


def shell_from_scene_data(scene_data_dir: str | Path, *, thickness: float = 0.16) -> dict[str, Any]:
    """Build a SHELL from one scan's ``scene_data`` folder."""
    from . import scene_data as loader

    scene_data_dir = Path(scene_data_dir)
    geometry = loader._build_from_scene_data(str(scene_data_dir), thickness=thickness)
    walls_org = geometry.get("walls_org") or {}
    floor_pkl = geometry.get("floor_pkl") or {}

    walls: dict[str, Any] = {}
    floor_y: list[float] = []
    for index in sorted(walls_org):
        wall = walls_org[index]
        (x1, z1), (x2, z2) = wall["pose"]["2d_line"]
        bbox = _as_list(wall["pose"]["bbox"])
        # ARKit XZ is the SHELL's XY; the wall's own thickness is its third bbox term, which
        # RoomPlan reports as ~0.0001 m because it models walls as planes.
        walls[f"Wall{index}"] = {"start": [x1, z1], "end": [x2, z2],
                                 "thickness": float(bbox[2]) if len(bbox) > 2 else 0.0001,
                                 "height": float(bbox[1]) if len(bbox) > 1 else 2.5}
        centre = _as_list(wall["pose"]["position"])
        if len(centre) > 1 and len(bbox) > 1:
            floor_y.append(centre[1] - bbox[1] / 2.0)

    verts: list[list[float]] = []
    faces: list[list[int]] = []
    for patch in floor_pkl.values():
        points = np.asarray(patch, dtype=float).reshape(-1, 3)
        if len(points) < 3:
            continue
        base = len(verts)
        # ARKit (x, y, z) -> SHELL (x, z, y): the horizontal plane becomes XY, height becomes Z
        verts += [[float(p[0]), float(p[2]), float(p[1])] for p in points]
        faces += [[base, base + i, base + i + 1] for i in range(1, len(points) - 1)]

    floor_z = min((v[2] for v in verts), default=min(floor_y, default=0.0))
    ceiling_z = floor_z + max((w["height"] for w in walls.values()), default=2.5)

    objects: dict[str, Any] = {}
    for entry in load_objects(scene_data_dir):
        object_id = entry.get("object_type") or entry.get("mesh_id")
        if not object_id:
            continue
        position, bbox = _as_list(entry["position"]), _as_list(entry["bbox"])
        if len(position) < 3 or len(bbox) < 3:
            continue
        rotation = entry.get("rotation", 0.0)
        yaw = float(np.asarray(rotation).reshape(-1)[0]) if np.size(rotation) else 0.0
        objects[object_id] = {
            "category": _category(object_id),
            # position is the box CENTRE in ARKit; swap Z and Y to reach the SHELL frame
            "center": [position[0], position[2], position[1]],
            # bbox is (width, height, depth) in ARKit; the SHELL wants (width, depth, height)
            "size": [bbox[0], bbox[2], bbox[1]],
            "yaw": yaw,
        }

    return {"walls": walls, "openings": {}, "objects": objects,
            "floor": {"verts": verts, "faces": faces},
            "floor_z": floor_z, "ceiling_z": ceiling_z}


def _category(object_id: str) -> str:
    """RoomPlan names an object `<Category><index>`; the rules key on the category."""
    return "".join(c for c in object_id if not c.isdigit()).strip("_").lower()


def apply_to_objects(entries: list[dict], shell: dict[str, Any]) -> dict[str, Any]:
    """Fold a repaired SHELL back into the ``objects.pkl`` list, in place.

    Returns a summary of what changed. Only centre and size are written, and only for entries the
    SHELL still contains — an object the repair dropped is removed, and nothing is ever added.
    """
    objects = shell.get("objects") or {}
    moved, resized, dropped = [], [], []
    kept: list[dict] = []
    for entry in entries:
        object_id = entry.get("object_type") or entry.get("mesh_id")
        repaired = objects.get(object_id)
        if repaired is None:
            dropped.append(object_id)
            continue
        position, bbox = _as_list(entry["position"]), _as_list(entry["bbox"])
        want_position = [repaired["center"][0], repaired["center"][2], repaired["center"][1]]
        want_bbox = [repaired["size"][0], repaired["size"][2], repaired["size"][1]]
        if math.dist(position[:3], want_position) > 1e-4:
            entry["position"] = np.asarray(want_position, dtype=float)
            moved.append(object_id)
        if any(abs(a - b) > 1e-4 for a, b in zip(bbox[:3], want_bbox)):
            entry["bbox"] = np.asarray(want_bbox, dtype=float)
            resized.append(object_id)
        kept.append(entry)
    entries[:] = kept
    return {"moved": moved, "resized": resized, "dropped": dropped}
