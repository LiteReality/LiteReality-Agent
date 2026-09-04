"""adapter.py — the RoomPlan pkl set on disk, as the SHELL the repair works on, and back again.

Two coordinate conventions meet here and the whole module exists to keep them apart.

``scene_data`` is ARKit's: **Y is up**, the floor is the XZ plane, and a wall carries a centre, a
rotation about Y and a bbox. The SHELL is the repair's: **Z is up**, the floor is XY, walls are
2-D centrelines with a thickness, objects are oriented footprints with a yaw. Every conversion is
in this file and nowhere else, so a sign error has one place to hide rather than a dozen.

The mapping is ``(x, y, z)_ARKit -> (x, -z, y)_SHELL``. The negation is the whole reason this
paragraph exists. ARKit is right-handed with +Z toward the viewer; the SHELL's XY is a plan view,
so preserving orientation requires flipping that axis. Getting it wrong does not mirror the room
into something obviously broken — it produces walls of exactly the right LENGTHS that no longer
meet at their corners, which reads as a plausible-looking room that is subtly, uselessly wrong.
The check: a room's wall endpoints coincide to within centimetres. If they are half a metre apart,
this sign is why.

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

MIN_PLATE_FRACTION = 0.30   # a floor covering less of its own walls than this is not believable


def _polygon_area(verts: list[list[float]], faces: list[list[int]]) -> float:
    """Total XY area of the floor triangles, for sanity-checking the plate."""
    total = 0.0
    for a, b, c in faces:
        (ax, ay), (bx, by), (cx, cy) = verts[a][:2], verts[b][:2], verts[c][:2]
        total += abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay)) / 2.0
    return total


def _as_list(value) -> list[float]:
    return [float(v) for v in np.asarray(value).reshape(-1)]


def load_objects(scene_data_dir: str | Path) -> list[dict]:
    path = Path(scene_data_dir) / "objects.pkl"
    return pickle.load(open(path, "rb")) if path.is_file() else []


def shell_from_scene_data(scene_data_dir: str | Path, *, thickness: float = 0.16) -> dict[str, Any]:
    """Build a SHELL from one scan's ``scene_data`` folder."""
    from . import scene_data as loader
    from .shell import ARKIT_TO_Z_UP, _box_frame, _r

    scene_data_dir = Path(scene_data_dir)
    raw_walls = []
    walls_pkl = scene_data_dir / "walls.pkl"
    if walls_pkl.is_file():
        raw_walls = pickle.load(open(walls_pkl, "rb"))
    geometry = loader._build_from_scene_data(str(scene_data_dir), thickness=thickness)
    walls_org = geometry.get("walls_org") or {}
    floor_pkl = geometry.get("floor_pkl") or {}

    walls: dict[str, Any] = {}
    floor_y: list[float] = []
    for index in sorted(walls_org):
        wall = walls_org[index]
        # the loader drops `transform` from its pose, so the raw pkl entry is what carries it
        pose = dict(wall["pose"])
        if index < len(raw_walls):
            pose.setdefault("transform", (raw_walls[index].get("pose") or {}).get("transform"))
        matrix = _world_matrix(pose)
        centre, along, width, depth, height = _box_frame(matrix, prefer_long=True)
        half = along * (width / 2.0)
        walls[f"Wall{index}"] = {
            "start": [_r(centre[0] - half[0]), _r(centre[1] - half[1])],
            "end": [_r(centre[0] + half[0]), _r(centre[1] + half[1])],
            "thickness": _r(depth), "height": _r(height)}
        floor_y.append(float(centre[2]) - height / 2.0)

    # The floor comes from its OWN mesh, put through the same change of basis as everything else.
    # The loader's clipped patches are a convex quad; a real room is rarely one, and on Tea_room's
    # L-shape that quad is a small rotated diamond covering neither half of the room, which reads
    # as most of the furniture standing off the plate.
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    floor_file = scene_data_dir / "floor.pkl"
    if floor_file.is_file():
        raw_floor = pickle.load(open(floor_file, "rb"))
        points = np.asarray(raw_floor.get("points", []), dtype=float).reshape(-1, 3)
        indices = np.asarray(raw_floor.get("faces", []), dtype=int).reshape(-1, 3)
        transform = raw_floor.get("transform")
        if len(points) >= 3 and len(indices):
            if transform is not None:
                matrix = np.asarray(transform, dtype=float).reshape(4, 4).T
                points = (matrix @ np.c_[points, np.ones(len(points))].T).T[:, :3]
            world = (ARKIT_TO_Z_UP @ np.c_[points, np.ones(len(points))].T).T[:, :3]
            verts = [[_r(p[0]), _r(p[1]), _r(p[2])] for p in world]
            faces = [[int(a), int(b), int(c)] for a, b, c in indices
                     if max(a, b, c) < len(verts)]
    if not faces:                                  # fall back to whatever the loader could clip
        for patch in floor_pkl.values():
            points = np.asarray(patch, dtype=float).reshape(-1, 3)
            if len(points) < 3:
                continue
            base = len(verts)
            verts += [[_r(p[0]), _r(-p[2]), _r(p[1])] for p in points]
            faces += [[base, base + i, base + i + 1] for i in range(1, len(points) - 1)]

    # A floor plate has to be plausible before it is trusted. Some captures hand back a floor
    # entity whose transform collapses it: Airbnb-Cam's is 0.27 m2 and 4 cm wide against a room
    # nearly 3 m across, and taking it at face value put EVERY object "off the floor plate" —
    # seven violations that were all artefacts of one bad mesh. A plate that covers almost none of
    # the area its own walls enclose is not a plate, and the honest response is to decline the test
    # rather than to report what it says. `check` then falls back to the wall bounds, which is
    # weaker but true.
    if verts:
        plate = _polygon_area(verts, faces)
        wall_points = [p for w in walls.values() for p in (w["start"], w["end"])]
        if wall_points:
            xs = [p[0] for p in wall_points]
            ys = [p[1] for p in wall_points]
            enclosed = (max(xs) - min(xs)) * (max(ys) - min(ys))
            if enclosed > 0 and plate < MIN_PLATE_FRACTION * enclosed:
                verts, faces = [], []

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
        matrix = _world_matrix({"position": position, "bbox": bbox,
                                "rotation": entry.get("rotation"),
                                "transform": entry.get("transform")})
        # prefer_long=False: an object keeps its OWN local axes. Taking the longer one instead
        # rotates any object whose local Y is longer by 90 degrees — the chair-orientation bug.
        centre, along, width, depth, height = _box_frame(matrix, prefer_long=False)
        objects[object_id] = {
            "category": _category(object_id),
            "center": [_r(centre[0]), _r(centre[1]), _r(centre[2])],
            "size": [_r(width), _r(depth), _r(height)],
            "yaw": _r(math.degrees(math.atan2(along[1], along[0])), 2),
        }

    return {"walls": walls, "openings": {}, "objects": objects,
            "floor": {"verts": verts, "faces": faces},
            "floor_z": floor_z, "ceiling_z": ceiling_z}


def _world_matrix(pose: dict) -> np.ndarray:
    """A scene_data pose -> its 4x4 box matrix in the SHELL's Z-up world, scale baked in.

    Going through the matrix is the point. The obvious shortcut — read the pose's centre and
    extents, swap the axes by hand — gets the CENTRES right and the DIRECTIONS wrong, because the
    handedness flip in ``(x, y, z) -> (x, -z, y)`` also reverses the sense of rotation about the
    vertical. The symptom is walls of exactly the correct lengths whose endpoints no longer meet,
    which looks like a plausible room until you try to stand in it. Applying the change of basis to
    the whole transform and reading the columns back out cannot make that mistake.
    """
    from .shell import ARKIT_TO_Z_UP

    bbox = _as_list(pose.get("bbox") or [1.0, 1.0, 1.0])
    transform = pose.get("transform")
    if transform is not None:
        # ROW-major, translation in the LAST ROW — USD's convention, preserved verbatim in the pkl.
        # The whole 4x4 transposes; taking only the rotation block and reading the translation from
        # the last column silently inverts the rotation, which leaves every centre correct and every
        # DIRECTION reversed about the vertical. Walls then have exactly the right lengths and no
        # longer meet at their corners.
        matrix = np.asarray(transform, dtype=float).reshape(4, 4).T.copy()
    else:                                   # only a Y-rotation was recorded
        rotation = pose.get("rotation", 0.0)
        yaw = float(np.asarray(rotation).reshape(-1)[1 if np.size(rotation) >= 3 else 0])
        c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        matrix = np.eye(4)
        matrix[:3, :3] = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        matrix[:3, 3] = _as_list(pose["position"])[:3]
    # the rotation part is orthonormal, so bake the extents into the columns: their LENGTHS then
    # are the box's extents, which is what _box_frame reads
    for axis in range(3):
        column = matrix[:3, axis]
        norm = float(np.linalg.norm(column)) or 1.0
        matrix[:3, axis] = column / norm * (bbox[axis] if axis < len(bbox) else 1.0)
    return ARKIT_TO_Z_UP @ matrix


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
        want_position = [repaired["center"][0], repaired["center"][2], -repaired["center"][1]]
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
