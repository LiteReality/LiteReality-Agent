"""merge_boxes.py — fuse badly-overlapping RoomPlan boxes into ONE object before crops/references.

RoomPlan emits a kitchen counter run as a pile of interpenetrating boxes: a sink and the base
cabinet under it come back as separate objects with the same yaw and overlapping footprints, one
dipping into the other. Cropped and reconstructed separately, they produce assets that occupy the
same space and read as garbage in the room.

Merging keys on GENUINE 3D overlap — shared yaw, overlapping footprint, AND overlapping height —
not footprint alone. A wall/upper cabinet projects to the same wall footprint as the base cabinet
below it but floats a real gap above; fusing the two (the earlier footprint-only test did, via a
sink that overlaps both) welds a separate object into the run. The vertical gate (see DEFAULT_VGAP)
keeps stacked-but-separate cabinets apart while still fusing a sink resting on its base cabinet.

This is the fix, and it must run in a specific window: AFTER `extract_scene` writes
``objects.pkl`` and BEFORE `crop_objects` reads it, so the crop, the reference image, the
complexity route and the reconstruction all see one merged unit. `run.py` calls `merge_for_scan`
at exactly that point; `scripts/ops/merge_objects.py` is a CLI over the same functions for
repairing a scan after the fact.

The merge writes two things besides the rewritten pkl:
  · ``object_refs/<scan>/<merged>/members.json`` — the marker `export_room._apply_shell_merges`
    globs for, so the SHELL's member boxes are unioned into one box at export time.
  · the merged ids into ``$LR_ENLARGED_CROP_OBJECTS`` — so the crop takes the FULL projected box
    (not just visible points) and `bbox_polish` leaves it alone instead of re-tightening it back
    down to the sub-part its detector recognises (the sink).

Set ``LR_BOX_MERGE=0`` to disable, or ``LR_BOX_MERGE_THRESH`` to change the overlap ratio.
``LR_BOX_MERGE_WALL_GUARD=0`` turns off the wall gate (see `_separated_by_wall`), which is on by
default and can only ever REMOVE a merge — with it off, and with no walls on disk, this module
behaves exactly as it did before that gate existed.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import shutil
from pathlib import Path

# Categories that form counter runs. Restricting to these keeps the auto-merge away from
# free-standing furniture, where an overlapping footprint usually means a bad box, not one object.
KITCHEN_PREFIXES = (
    "Storage", "Sink", "Stove", "Oven", "Hob", "Dishwasher", "Cooktop", "Counter",
)
DEFAULT_THRESH = 0.15  # min footprint-overlap ratio (of the SMALLER box) to fuse
YAW_TOLERANCE_DEG = 3.0  # boxes must share a yaw to be part of one run
# Vertical gate: footprint overlap alone fuses a WALL/UPPER cabinet into the BASE unit below it,
# because both project to the same wall footprint (a sink over the base cabinet then bridges them
# transitively). Genuinely one object overlaps in 3D, so also require the boxes' vertical extents
# to overlap — allowing a small gap so a sink resting on the counter still fuses with its cabinet,
# while an upper cabinet a real gap above the base does NOT. Override with $LR_BOX_MERGE_VGAP.
DEFAULT_VGAP = 0.12  # m: max vertical gap between extents still treated as overlapping (touching)
# Wall gate: two boxes with a WALL BETWEEN THEM are not one counter run, whatever their footprints
# do. RoomPlan sometimes detects a unit twice and records the second copy far too deep — deep
# enough to punch through the wall behind it — and that smeared copy then overlaps a genuine unit
# in the NEXT room. Union-find welds the two runs into a single box spanning the wall, and every
# later stage reads the result as one counter recorded too deep. Set $LR_BOX_MERGE_WALL_GUARD=0
# to disable. See `_separated_by_wall`.
WALL_GUARD_EPS = 1e-6  # keeps a shared endpoint from reading as a crossing


def _union_obb(members: list[dict]) -> dict:
    """Union OBB of same-yaw members (bbox=[w,h,d] full sizes, position=centre, rotation=yaw deg).

    Rotate the centres into the shared yaw frame, take the axis-aligned min/max ± half-size there,
    then rotate the resulting centre back out. Doing it in the yaw frame is what keeps the union a
    tight oriented box instead of an axis-aligned one that would bulge off the wall.
    """
    import numpy as np

    yaw = float(members[0]["rotation"])
    a = math.radians(yaw)
    ca, sa = math.cos(a), math.sin(a)
    xmn = ymn = zmn = 1e9
    xmx = ymx = zmx = -1e9
    for m in members:
        px, py, pz = (float(v) for v in m["position"])
        w, hh, d = (float(v) for v in m["bbox"])
        rx = ca * px + sa * pz
        rz = -sa * px + ca * pz
        xmn, xmx = min(xmn, rx - w / 2), max(xmx, rx + w / 2)
        zmn, zmx = min(zmn, rz - d / 2), max(zmx, rz + d / 2)
        ymn, ymx = min(ymn, py - hh / 2), max(ymx, py + hh / 2)
    rcx, rcz = (xmn + xmx) / 2, (zmn + zmx) / 2
    cx = ca * rcx - sa * rcz
    cz = sa * rcx + ca * rcz
    cy = (ymn + ymx) / 2
    size = [round(xmx - xmn, 4), round(ymx - ymn, 4), round(zmx - zmn, 4)]
    hw, hd = size[0] / 2, size[2] / 2
    corners = []
    for sx, sz in ((-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)):
        wx = ca * (rcx + sx) - sa * (rcz + sz)
        wz = sa * (rcx + sx) + ca * (rcz + sz)
        corners.append((wx, wz))
    # largest footprint drives which member's source mesh file the merged entry points at
    base = max(members, key=lambda m: float(m["bbox"][0]) * float(m["bbox"][2]))
    return {
        "object_type": None,
        "bbox": np.array(size, dtype=float),
        "rotation": yaw,
        "top_down_rect": corners,
        "position": np.array([round(cx, 4), round(cy, 4), round(cz, 4)], dtype=float),
        "file": base["file"],
        "mesh_id": None,
        "room_plan_rotation": base.get("room_plan_rotation"),
    }


def _footprint_overlap(a: dict, b: dict) -> float:
    """Footprint overlap as a fraction of the SMALLER box, via an axis-aligned test in a's yaw frame.

    Returns 0 when the yaws differ: two boxes at different yaws are not one counter run, and the
    same-yaw assumption is what makes the cheap axis-aligned test valid.
    """
    if abs(float(a["rotation"]) - float(b["rotation"])) > YAW_TOLERANCE_DEG:
        return 0.0
    ang = math.radians(float(a["rotation"]))
    ca, sa = math.cos(ang), math.sin(ang)

    def box(o):
        px, _, pz = (float(v) for v in o["position"])
        w, _, d = (float(v) for v in o["bbox"])
        rx = ca * px + sa * pz
        rz = -sa * px + ca * pz
        return rx - w / 2, rx + w / 2, rz - d / 2, rz + d / 2, w * d

    ax0, ax1, az0, az1, aa = box(a)
    bx0, bx1, bz0, bz1, ba = box(b)
    ox = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    oz = max(0.0, min(az1, bz1) - max(az0, bz0))
    smaller = min(aa, ba)
    return (ox * oz) / smaller if smaller > 0 else 0.0


def _vertical_gap(a: dict, b: dict) -> float:
    """Signed vertical distance between two boxes' height extents (bbox[1]=height, position[1]=cy).

    Negative/zero = the extents overlap; positive = a clear gap between them. Used to keep an upper
    cabinet (footprint over the base, but a real gap above it) from fusing into the base unit.
    """
    acy, ah = float(a["position"][1]), float(a["bbox"][1])
    bcy, bh = float(b["position"][1]), float(b["bbox"][1])
    atop, abot = acy + ah / 2, acy - ah / 2
    btop, bbot = bcy + bh / 2, bcy - bh / 2
    return max(abot, bbot) - min(atop, btop)  # >0 only when there is a gap between the extents


def wall_segments(scene_data_dir: str | Path) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """This scan's wall centrelines as 2-D segments in the ARKit ground plane (x, z).

    The centrelines come from :mod:`..layout.adapter`, which recovers each wall from its transform
    MATRIX rather than its Euler angle — ``R = Ry(theta) . Rz(180)`` does not decompose the obvious
    way, and doing it by hand tilts every wall a few degrees. That module works in the SHELL frame
    (Z up, floor on XY) where ``(x, y, z)_ARKit -> (x, -z, y)_SHELL``, so coming back to the ground
    plane this file works in is ``(x, z)_ARKit = (X, -Y)_SHELL``.

    Returns an empty list rather than raising. No walls simply means no wall guard: the merge then
    behaves exactly as it did before this function existed, which is the right way to fail for a
    check that only ever REMOVES merges.
    """
    try:
        from ..layout import adapter

        shell = adapter.shell_from_scene_data(Path(scene_data_dir))
        return [((float(w["start"][0]), -float(w["start"][1])),
                 (float(w["end"][0]), -float(w["end"][1])))
                for w in (shell.get("walls") or {}).values()]
    except Exception as exc:  # noqa: BLE001 — a missing or unreadable room is not a merge failure
        print(f"  [box-merge] no walls available for the wall guard ({type(exc).__name__}: {exc})",
              flush=True)
        return []


def _separated_by_wall(a: dict, b: dict, walls: list) -> bool:
    """Is there a wall standing between these two boxes?

    Deliberately the crude, physical test: draw the straight line between the two centres and ask
    whether it crosses a wall SEGMENT. Not "which side of the wall's infinite line" — a room is
    full of walls whose infinite lines cut everything in half, and the question here is only ever
    about the wall actually standing in the way.

    This is what Kitchen-Xiaoyang_Lyu needed. One oven was detected twice and the second copy came
    back 1.46 m deep, straight through the wall behind it. That copy genuinely overlaps the counter
    run on one side AND a storage unit on the other, so it bridges them, and the merged box —
    2.53 x 1.62 m in the shipped run — spans a wall. No later stage can undo that: the layout pass
    reads it as a counter recorded too deep and every repair it can make is the wrong one.
    """
    if not walls:
        return False
    ax, az = float(a["position"][0]), float(a["position"][2])
    bx, bz = float(b["position"][0]), float(b["position"][2])
    dx, dz = bx - ax, bz - az
    for (sx, sz), (ex, ez) in walls:
        wx, wz = ex - sx, ez - sz
        denominator = dx * wz - dz * wx
        if abs(denominator) < 1e-12:  # parallel: the wall lies beside them, not between them
            continue
        ox, oz = sx - ax, sz - az
        t = (ox * wz - oz * wx) / denominator
        u = (ox * dz - oz * dx) / denominator
        if WALL_GUARD_EPS < t < 1.0 - WALL_GUARD_EPS and WALL_GUARD_EPS < u < 1.0 - WALL_GUARD_EPS:
            return True
    return False


def auto_groups(
    objs: list[dict], thresh: float = DEFAULT_THRESH, vgap: float = DEFAULT_VGAP,
    walls: list | None = None
) -> list[list[str]]:
    """Cluster counter-run boxes that genuinely overlap in 3D (same yaw, footprints, heights).

    A pair fuses only when it shares a yaw, its footprints overlap more than `thresh`, AND its
    vertical extents overlap (or sit within `vgap` of touching). Union-find then fuses transitively
    so a real chain — sink over the base cabinet, base cabinet abutting the next base cabinet —
    still merges; the vertical gate is what stops a wall/upper cabinet, whose footprint overlaps but
    which floats a real gap above, from being dragged in through a box between them.

    `walls`, when given, adds the horizontal twin of that gate: a pair with a wall BETWEEN them
    never fuses, however well their boxes overlap. The vertical gate cannot express this — the
    bridging box overlaps its neighbour on each side perfectly plausibly, and the resulting group
    is no taller than a single cabinet. Omit `walls` and the behaviour is exactly as before.
    """
    ids = [
        o["object_type"]
        for o in objs
        if o.get("object_type") and any(str(o["object_type"]).startswith(k) for k in KITCHEN_PREFIXES)
    ]
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by = {o["object_type"]: o for o in objs}
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = by[ids[i]], by[ids[j]]
            if _footprint_overlap(a, b) > thresh and _vertical_gap(a, b) <= vgap:
                if _separated_by_wall(a, b, walls or []):
                    continue
                parent[find(ids[i])] = find(ids[j])
    groups: dict = {}
    for i in ids:
        groups.setdefault(find(i), []).append(i)
    return [sorted(g) for g in groups.values() if len(g) > 1]


def name_for(members: list[str], taken: object = ()) -> str:
    """Readable id for a merged group: `Sink_Storage0` from Sink0 + Storage1.

    Named from the distinct categories rather than `Merged0` so the object is identifiable in the
    reference sheet, the viewer outliner and the SHELL without cross-referencing members.json.

    Two collisions to avoid, both of which produce a merged object indistinguishable from an
    ordinary box:
      · a single-category run (Storage0 + Storage5) would be named `Storage0` — the same id as one
        of its own members. Those get `StorageRun0` instead.
      · any candidate already in use by another object gets its index bumped.
    """
    kinds: list[str] = []
    for m in members:
        kind = "".join(c for c in m if not c.isdigit()) or m
        if kind not in kinds:
            kinds.append(kind)
    base = f"{kinds[0]}Run" if len(kinds) == 1 else "_".join(kinds)
    taken = set(taken)
    i = 0
    while f"{base}{i}" in taken:
        i += 1
    return f"{base}{i}"


def apply_merges(
    pkl_path: Path, refroot: Path, groups: dict[str, list[str]], *, backup: bool = True
) -> dict:
    """Rewrite `objects.pkl` with one union entry per group and drop the members.

    Returns {"merged": {name: members}, "skipped": {name: reason}}. Groups with fewer than two
    present members are skipped rather than collapsing to a rename of a single box.
    """
    objs = pickle.load(open(pkl_path, "rb"))
    by = {o["object_type"]: o for o in objs}
    merged: dict[str, list[str]] = {}
    skipped: dict[str, str] = {}
    consumed: set[str] = set()
    new_entries: list[dict] = []

    for name, mem in groups.items():
        present = [m for m in mem if m in by]
        if len(present) < 2:
            skipped[name] = f"only {len(present)} of {len(mem)} members present"
            continue
        entry = _union_obb([by[m] for m in present])
        entry["object_type"] = name
        entry["mesh_id"] = name
        # Provenance travels WITH the box. `members.json` beside the references records the same
        # thing, but a later stage holding only objects.pkl cannot reach it, and the layout pass
        # that runs next needs to know this unit was assembled rather than detected — otherwise it
        # reads a run swallowing its own cabinet as a duplicate and deletes the run.
        entry["merged_from"] = list(present)
        new_entries.append(entry)
        consumed.update(present)
        merged[name] = present
        (refroot / name).mkdir(parents=True, exist_ok=True)
        (refroot / name / "members.json").write_text(json.dumps(present), encoding="utf-8")

    if not new_entries:
        return {"merged": merged, "skipped": skipped}

    if backup:
        shutil.copy2(pkl_path, pkl_path.with_suffix(".pkl.bak"))
    kept = [o for o in objs if o["object_type"] not in consumed]
    pickle.dump(kept + new_entries, open(pkl_path, "wb"))
    return {"merged": merged, "skipped": skipped}


def merge_for_scan(scan: str, *, thresh: float | None = None, enabled: bool | None = None) -> dict:
    """The pipeline entry point: auto-detect counter runs for `scan`, fuse them, and arm the crop.

    Called from `run.py` between extract and crop. Never raises into the caller — a merge failure
    must degrade to the old unmerged behaviour rather than take down init, so the worst case is the
    bug this function exists to fix, not a dead run.
    """
    from litereality_agent.pipeline.measure import paths as config

    if enabled is None:
        enabled = os.environ.get("LR_BOX_MERGE", "1") not in ("0", "false", "no")
    if not enabled:
        print("  [box-merge] disabled ($LR_BOX_MERGE=0)", flush=True)
        return {"merged": {}, "skipped": {}, "disabled": True}
    if thresh is None:
        try:
            thresh = float(os.environ.get("LR_BOX_MERGE_THRESH", DEFAULT_THRESH))
        except ValueError:
            thresh = DEFAULT_THRESH
    try:
        vgap = float(os.environ.get("LR_BOX_MERGE_VGAP", DEFAULT_VGAP))
    except ValueError:
        vgap = DEFAULT_VGAP
    guard = os.environ.get("LR_BOX_MERGE_WALL_GUARD", "1") not in ("0", "false", "no")

    try:
        pkl = config.scene_data_dir(scan) / "objects.pkl"
        if not pkl.is_file():
            print(f"  [box-merge] no objects.pkl at {pkl} — skipping", flush=True)
            return {"merged": {}, "skipped": {}}
        objs = pickle.load(open(pkl, "rb"))
        # Walls are read from the same scene_data folder the objects came from, so this needs no
        # new plumbing and no ordering change: `walls.pkl` is written by extraction, before the
        # window this function runs in.
        walls = wall_segments(pkl.parent) if guard else []
        if guard and not walls:
            print("  [box-merge] wall guard inactive — no wall centrelines for this scan", flush=True)
        detected = auto_groups(objs, thresh, vgap, walls)
        if not detected:
            print(f"  [box-merge] no overlapping counter runs (thresh={thresh})", flush=True)
            return {"merged": {}, "skipped": {}}
        taken = {o.get("object_type") for o in objs}
        groups: dict[str, list[str]] = {}
        for mem in detected:
            name = name_for(mem, taken | set(groups))
            groups[name] = mem
        refroot = config.object_refs_root() / scan
        res = apply_merges(pkl, refroot, groups)
        for name, mem in res["merged"].items():
            print(f"  [box-merge] {name} <- {', '.join(mem)}", flush=True)
        for name, why in res["skipped"].items():
            print(f"  [box-merge] ⚠ skipped {name}: {why}", flush=True)
        if res["merged"]:
            arm_enlarged_crops(res["merged"])
        return res
    except Exception as exc:  # noqa: BLE001 — telemetry-grade: never break init over a merge
        print(f"  [box-merge] FAILED (non-fatal, continuing unmerged): {exc}", flush=True)
        return {"merged": {}, "skipped": {}, "error": str(exc)}


def arm_enlarged_crops(merged: dict[str, list[str]]) -> str:
    """Add the merged ids to `$LR_ENLARGED_CROP_OBJECTS`, preserving anything already there.

    Both consumers read this env var inside their functions (`preprocessing.process_object_images`,
    `bbox_polish.polish`), so setting it here — after extract, before crop — reaches them.
    """
    current = set(filter(None, os.environ.get("LR_ENLARGED_CROP_OBJECTS", "").split(",")))
    current.update(merged)
    value = ",".join(sorted(current))
    os.environ["LR_ENLARGED_CROP_OBJECTS"] = value
    print(f"  [box-merge] LR_ENLARGED_CROP_OBJECTS={value}", flush=True)
    return value
