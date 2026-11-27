"""A built-in fridge column that RoomPlan boxed twice: the refrigerator, the wall cabinet on top of
it, and one storage box around both.

In a fully built-in kitchen the refrigerator stands in a column of the cabinetry, with a wall cabinet
above it. RoomPlan reports the refrigerator and that cabinet, and sometimes also a storage box that
spans the whole column, from the floor to the cabinet's top. That box duplicates the other two. Left
in, it overlaps both, and the layout repair cannot clear it: no move or trim takes a box out of the
inside of another, deleting is opt-in there, and the layout check then fails the whole room before
anything is built.

So the duplicate is taken out here, before the measurements are frozen, and recorded in
scene_data/<scan>/column_duplicates.json, but only when it matches the column on every count:

* a refrigerator, and a storage unit or cabinet sitting on it (the gap between the refrigerator's top
  and the cabinet's bottom at most STACK_GAP of the refrigerator's height), each overlap the box by
  at least DUP_SHARE of the smaller of the two (min-normalised, the layout's duplicate measure);
* the box spans the column: its bottom at the refrigerator's, its top at the cabinet's, each within
  SPAN_TOL of the box's own height;
* nothing else clashes with the box under the layout check's pair rule.

A pantry beside a refrigerator, a tall unit next to one, or a box that anything else runs into stays,
and the layout check still reports it.
"""

from __future__ import annotations

from typing import Any

from litereality_agent.pipeline.scene_init.layout.adapter import object_from_entry
from litereality_agent.pipeline.scene_init.layout.adjust import _obb
from litereality_agent.pipeline.scene_init.layout.graph import (
    CLASH_TOL,
    OPEN_FRAME,
    PASSTHROUGH,
    Z_BAND_TOL,
    expected_pair,
    obb_mtv,
)
from litereality_agent.pipeline.scene_init.layout.repair import DUP_SAME, _overlap_fraction

STACK_GAP = 0.05     # of the refrigerator's height: a cabinet this close above it sits on it
SPAN_TOL = 0.10      # of the box's height: how far its ends may miss the column's
DUP_SHARE = DUP_SAME  # overlap / smaller volume that makes the box a second detection
UNITS = {"storage", "cabinet"}


def _span(obj: dict[str, Any]) -> tuple[float, float]:
    half = obj["size"][2] / 2
    return obj["center"][2] - half, obj["center"][2] + half


def _clashes(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """The layout check's pair rule (adjust.check): shared height and overlapping footprints."""
    cat_a, cat_b = a.get("category", ""), b.get("category", "")
    if expected_pair(cat_a, cat_b) or {cat_a, cat_b} & (PASSTHROUGH | OPEN_FRAME):
        return False
    (a0, a1), (b0, b1) = _span(a), _span(b)
    if min(a1, b1) - max(a0, b0) <= Z_BAND_TOL:
        return False
    mtv = obb_mtv(_obb(a), _obb(b))
    return bool(mtv) and mtv[0] > CLASH_TOL


def column_duplicates(objects: dict[str, dict[str, Any]]) -> dict[str, tuple[str, str]]:
    """{box: (refrigerator, cabinet)} for every box that duplicates a fridge column.

    ``objects`` are layout-shell objects: category, centre and size (z up, metres) and yaw.
    """
    found: dict[str, tuple[str, str]] = {}
    fridges = [n for n, o in objects.items() if o.get("category") == "refrigerator"]
    units = [n for n, o in objects.items() if o.get("category") in UNITS]
    for f in fridges:
        f0, f1 = _span(objects[f])
        for c in units:
            c0, c1 = _span(objects[c])
            if abs(c0 - f1) > STACK_GAP * (f1 - f0):
                continue
            for s in units:
                if s == c or s in found:
                    continue
                box = objects[s]
                s0, s1 = _span(box)
                if abs(s0 - f0) > SPAN_TOL * (s1 - s0) or abs(s1 - c1) > SPAN_TOL * (s1 - s0):
                    continue
                if min(_overlap_fraction(box, objects[f]), _overlap_fraction(box, objects[c])) < DUP_SHARE:
                    continue
                if any(_clashes(box, o) for n, o in objects.items() if n not in (s, f, c)):
                    continue
                found[s] = (f, c)
    return found


def drop_column_duplicates(entries: list) -> tuple[list, list]:
    """(entries to keep, entries dropped); a dropped entry carries why in ``drop_reason``.

    Never raises: if the boxes cannot be read, every object is kept and the layout check decides, as
    it did before this step existed.
    """
    try:
        objects = {e["object_type"]: object_from_entry(e) for e in entries}
        found = column_duplicates(objects)
    except Exception as exc:  # noqa: BLE001
        print(f"  ! fridge column check skipped: {type(exc).__name__}: {exc}", flush=True)
        return entries, []
    kept, dropped = [], []
    for entry in entries:
        hit = found.get(entry["object_type"])
        if hit is None:
            kept.append(entry)
            continue
        reason = f"a second box around the fridge column of {hit[0]} and {hit[1]}"
        dropped.append({**entry, "drop_reason": reason})
        print(f"  {entry['object_type']} not modelled: {reason}", flush=True)
    return kept, dropped
