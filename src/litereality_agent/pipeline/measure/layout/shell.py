"""shell.py — the RoomPlan layout → the SHELL: a flat, editable, Z-up floor plan.

STAGE 1b. :mod:`physical_engine.roomplan` gives 4x4 matrices in ARKit's Y-up frame — faithful, but
nothing you can reason about. The SHELL is the same room stated as floor-plan semantics::

    walls:    WallN  -> {start: [x, y], end: [x, y], thickness}      (height = ceiling_z - floor_z)
    openings: DoorN  -> {wall, type, offset, width, height, sill}    offset = metres along the wall
    objects:  ChairN -> {category, center: [x, y, z], size: [w, d, h], yaw}   yaw in degrees
    floor:    {verts, faces}                    + floor_z / ceiling_z / root_matrix

This is deliberately **the exact schema** ``LiteReality-Agent`` writes into ``Room.py`` as its
``SHELL`` dict, so anything produced here drops straight into that pipeline and anything it has
already produced can be read back here.

The difference is how it is computed. Upstream, ``room_ops/export/extract_shell.py`` imports the
usdz into **Blender** and reads ``matrix_world`` off the imported objects. That is a 400 MB
dependency and a subprocess for what is, underneath, one rotation and some trigonometry. Here the
Blender frame is reconstructed directly:

* Blender's USD importer turns the stage's ``upAxis = "Y"`` into a +90 degrees X rotation on the
  root prim — that is exactly the ``root_matrix`` upstream records, and :data:`ARKIT_TO_Z_UP`
  reproduces it: ``(x, y, z)_arkit -> (x, -z, y)``.
* Extents and orientation come from the columns of the world matrix, whose lengths ARE the box
  extents (RoomPlan's rotation part is orthonormal).

Two axis conventions are preserved from upstream because they encode real fixes:

* **Walls** take their long horizontal axis as the wall direction. A wall's local X is its length,
  so this agrees with the file and stays right for the degenerate stubs RoomPlan emits at corners.
* **Objects** keep their OWN local axes instead. Picking the longer axis rotates any object whose
  local Y is longer by 90 degrees — the chair-orientation bug. The matrix already carries the
  RoomPlan rotation; re-deriving it is what breaks it.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "ARKIT_TO_Z_UP",
    "load_shell",
    "save_shell",
    "read_room_py_shell",
    "wall_frame",
    "opening_span",
    "facing_yaw",
    "object_footprint",
    "shell_summary",
]

# ARKit (Y-up, right-handed) -> Blender/USD Z-up. Identical to the ``root_matrix`` Blender's USD
# importer writes onto the root prim, and recorded in the SHELL so downstream cameras — which are
# still in ARKit space, straight off ``cameraPoseARFrame`` — can be brought into the same frame.
ARKIT_TO_Z_UP = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])

_ROUND = 4


def _r(value: float, digits: int = _ROUND) -> float:
    return round(float(value), digits)


def _box_frame(matrix: np.ndarray, prefer_long: bool) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Decompose a Z-up world box matrix → (centre, along-axis unit vector, width, depth, height).

    ``prefer_long`` picks the longer horizontal axis as "width" (right for walls, wrong for
    objects — see the module docstring).
    """
    cols = [matrix[:3, k] for k in range(3)]
    lengths = [float(np.linalg.norm(c)) for c in cols]
    units = [c / (n if n > 1e-12 else 1.0) for c, n in zip(cols, lengths)]

    vertical = max(range(3), key=lambda k: abs(units[k][2]))
    horizontal = [k for k in range(3) if k != vertical]
    if prefer_long and lengths[horizontal[1]] > lengths[horizontal[0]]:
        horizontal = horizontal[::-1]
    wi, di = horizontal

    along = np.array([cols[wi][0], cols[wi][1], 0.0])
    if np.linalg.norm(along) < 1e-9:
        along = np.array([1.0, 0.0, 0.0])
    along = along / np.linalg.norm(along)
    return matrix[:3, 3], along, lengths[wi], lengths[di], lengths[vertical]


def save_shell(shell: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(shell, indent=2), encoding="utf-8")
    return path


def load_shell(path: str | Path) -> dict[str, Any]:
    """Read a SHELL from ``shell.json`` or straight out of a LiteReality ``Room.py``."""
    path = Path(path)
    if path.suffix == ".py" or path.name == "Room.py":
        return read_room_py_shell(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_room_py_shell(path: str | Path) -> dict[str, Any]:
    """Extract the ``SHELL = {...}`` literal from a LiteReality ``Room.py``.

    Brace-matched rather than regex-terminated, and parsed as a Python literal before JSON: the
    authoring agent edits ``SHELL`` as Python and readily writes something valid-Python-but-not-
    JSON (a trailing comma, an adjacent-string note), at which point a JSON-only reader returns
    ``{}`` and every downstream check silently passes on an empty room.
    """
    import ast

    source = Path(path).read_text(encoding="utf-8")
    match = re.search(r"\bSHELL\s*=\s*\{", source)
    if not match:
        return {}
    start = source.index("{", match.start())
    depth = 0
    for i in range(start, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                blob = source[start:i + 1]
                for parse in (json.loads, ast.literal_eval):
                    try:
                        return parse(blob)
                    except Exception:  # noqa: BLE001
                        continue
                return {}
    return {}


# ── small geometric helpers every later stage shares ─────────────────────────
def wall_frame(wall: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """(start, along-unit, outward-normal-unit, length) for one SHELL wall."""
    start = np.asarray(wall["start"], dtype=float)
    end = np.asarray(wall["end"], dtype=float)
    delta = end - start
    length = float(np.linalg.norm(delta))
    along = delta / length if length > 1e-9 else np.array([1.0, 0.0])
    normal = np.array([along[1], -along[0]])
    return start, along, normal, length


def facing_yaw(obj: dict[str, Any]) -> float:
    """Which way an object FACES, in degrees. **Not** the same as its ``yaw``.

    ``yaw`` is the heading of the box's local **+X** axis — its WIDTH direction. RoomPlan puts an
    object's front along its local **+Z** (depth) axis, and because the frame is right-handed with
    local +Y mapped to world +Z::

        Z_world = X_world x Y_world = (cos y, sin y, 0) x (0, 0, 1) = (sin y, -cos y, 0)

    the front is exactly ``yaw - 90 deg``. Verified two ways: the rotation determinant is +1.000000
    for all 102 objects sampled, and the +Z column sits within 0.005 deg of ``yaw - 90`` for every
    one of them. Independently, across six scans, ``yaw - 90`` matches the room-facing wall normal
    to 0.0 deg for 19 of the 22 objects that stand against a wall (a TV, a sofa, a fridge, a run of
    kitchen units all face out of the wall behind them).

    The remaining few are off by 180 deg: RoomPlan cannot always tell an object's front from its
    back, so treat this as an AXIS with a probable sign, not a guarantee. Drawing ``yaw`` itself as
    an arrow is simply wrong — it points sideways across the object.
    """
    return (obj.get("yaw", 0.0) - 90.0 + 180.0) % 360.0 - 180.0


def opening_span(opening: dict[str, Any]) -> tuple[float, float]:
    """An opening's real extent along its wall → (t0, t1) metres from the wall's start.

    READ THIS BEFORE USING ``offset``. In this schema — and in LiteReality's ``Room.py``, which
    defines it — ``offset`` is the distance to the opening's **CENTRE**, not to its leading edge::

        t0 = offset - width / 2        t1 = offset + width / 2

    ``room_ops/export/extract_shell.py:113`` writes the centre projection, and
    ``room_ops/compile/build_room.py:582`` reads it back as a centre, so the compiled room is
    correct. But ``Room.md`` describes it only as "distance along the wall from its start", and
    ``pipeline/room_qc/checks.py:156`` takes that literally (``o0, o1 = offset, offset + width``),
    which slides the tested interval half an opening-width toward the wall's end. Its
    ``fixture_over_opening`` check therefore compares against the wrong patch of wall, and any
    plot drawn the same way shows doors overhanging corners that they do not actually overhang.

    Every consumer in this package goes through here so the mistake cannot be made twice.
    """
    half = opening["width"] / 2.0
    return opening["offset"] - half, opening["offset"] + half


def object_footprint(obj: dict[str, Any]) -> np.ndarray:
    """The 4 XY corners of an object's oriented footprint, counter-clockwise."""
    cx, cy = obj["center"][0], obj["center"][1]
    hw, hd = obj["size"][0] / 2.0, obj["size"][1] / 2.0
    yaw = math.radians(obj.get("yaw", 0.0))
    c, s = math.cos(yaw), math.sin(yaw)
    local = np.array([[-hw, -hd], [hw, -hd], [hw, hd], [-hw, hd]])
    rotation = np.array([[c, -s], [s, c]])
    return local @ rotation.T + np.array([cx, cy])


def shell_summary(shell: dict[str, Any]) -> dict[str, Any]:
    """Counts + room extents, for the report line at the end of extraction."""
    walls = shell.get("walls") or {}
    objects = shell.get("objects") or {}
    openings = shell.get("openings") or {}
    lengths = {wid: float(np.linalg.norm(np.asarray(w["end"], float) - np.asarray(w["start"], float)))
               for wid, w in walls.items()}
    categories: dict[str, int] = {}
    for obj in objects.values():
        categories[obj["category"]] = categories.get(obj["category"], 0) + 1
    verts = np.asarray((shell.get("floor") or {}).get("verts") or [], dtype=float)
    extent = None
    if len(verts):
        extent = [_r(v) for v in (verts[:, :2].max(axis=0) - verts[:, :2].min(axis=0))]
    return {
        "walls": len(walls),
        "wall_length_total": _r(sum(lengths.values()), 2),
        "sliver_walls": sorted(w for w, n in lengths.items() if n < 0.35),
        "openings": {k: sum(1 for o in openings.values() if o["type"] == k)
                     for k in ("door", "window", "opening")},
        "objects": len(objects),
        "object_categories": dict(sorted(categories.items())),
        "height": _r(shell["ceiling_z"] - shell["floor_z"]),
        "floor_extent_xy": extent,
    }
