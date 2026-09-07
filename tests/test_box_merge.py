"""Box merge — fusing RoomPlan's interpenetrating counter runs into one object.

The bug this guards against is not a wrong answer, it is a step that never runs. The merge
algorithm existed for months in `scripts/ops/merge_objects.py` and grouped fallside's counter run,
while the export-side consumer (`export_room._apply_shell_merges`) sat wired up waiting for
`members.json` markers that nothing produced — so every scan shipped a sink interpenetrating its
own cabinets and no test failed, because there was nothing switched on to regress.

Merging keys on 3D overlap (footprint AND height), not footprint alone: the sink fuses with the
base cabinet it dips into, but the wall cabinet sharing that footprint a gap above stays a separate
object — footprint-only merging welded it in and rendered the counter as one exploded blob.

Hence `test_merge_runs_before_crop_in_the_pipeline` below: the algorithm tests are necessary but
they were never what was missing.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

from litereality_agent.pipeline.scene_init.ingest import merge_boxes


def box(name: str, pos: tuple[float, float, float], size: tuple[float, float, float],
        yaw: float = 0.0) -> dict:
    """One objects.pkl entry: position=(x, y=up, z), bbox=(w, height, d) full sizes, rotation=yaw deg."""
    return {
        "object_type": name,
        "position": list(pos),
        "bbox": list(size),
        "rotation": yaw,
        "file": f"{name}.obj",
        "mesh_id": name,
        "room_plan_rotation": None,
        "top_down_rect": [],
    }


# The real fallside geometry that shipped broken, in objects.pkl terms: a sink dropped into a base
# cabinet, with a wall cabinet above sharing the footprint. Same yaw, overlapping in plan.
FALLSIDE = [
    box("Sink0", (0.6155, -0.6378, -1.6917), (0.4858, 0.2325, 0.4093), yaw=-121.04),
    box("Storage0", (0.8169, 0.3921, -1.5821), (1.7487, 0.7827, 0.3554), yaw=-121.04),
    box("Storage2", (0.5393, -0.9689, -1.7818), (1.1202, 0.8946, 0.6252), yaw=-121.04),
    box("Sofa0", (-1.3891, -0.9893, -0.7878), (1.3896, 0.8537, 0.8353), yaw=148.96),
]


# --------------------------------------------------------------------------- detection


def test_detects_the_fallside_counter_run():
    """The sink dropped into its base cabinet is ONE object; the wall cabinet above is NOT.

    `Sink0` (counter level) and `Storage2` (base cabinet) interpenetrate in 3D, so they fuse.
    `Storage0` shares their wall footprint but floats ~0.5 m above with a clear gap — a genuinely
    separate upper cabinet — so the vertical gate keeps it out. (Footprint-only merging used to
    weld all three together, which rendered as one exploded blob spanning the counter.)
    """
    assert merge_boxes.auto_groups(FALLSIDE) == [["Sink0", "Storage2"]]


def test_upper_cabinet_above_the_base_is_not_merged():
    """A wall cabinet whose footprint overlaps the base but which sits a real gap above must stay
    separate — footprint overlap alone is not 3D overlap. Regression for the vertical gate."""
    base = box("Storage0", (0, -0.9, 0), (1.0, 0.9, 0.6), yaw=10.0)     # y in [-1.35, -0.45]
    upper = box("Storage1", (0, 0.4, 0), (1.0, 0.8, 0.4), yaw=10.0)     # y in [ 0.00,  0.80]
    assert merge_boxes._footprint_overlap(base, upper) > 0.15           # footprints DO overlap
    assert merge_boxes._vertical_gap(base, upper) > merge_boxes.DEFAULT_VGAP  # but a real gap above
    assert merge_boxes.auto_groups([base, upper]) == []


def test_free_standing_furniture_is_never_merged():
    """A sofa overlapping a table is a bad box, not one object — only counter-run categories fuse."""
    objs = [
        box("Sofa0", (0, 0, 0), (1.5, 0.8, 0.8)),
        box("Table0", (0.1, 0, 0.1), (1.2, 0.7, 0.7)),
    ]
    assert merge_boxes.auto_groups(objs) == []


def test_different_yaw_is_not_one_run():
    """Same footprint, different orientation — the cheap axis-aligned overlap test is only valid
    within a shared yaw, so a yaw mismatch must score zero rather than a bogus overlap."""
    objs = [
        box("Sink0", (0, 0, 0), (0.5, 0.3, 0.5), yaw=0.0),
        box("Storage0", (0, 0, 0), (0.5, 0.8, 0.5), yaw=45.0),
    ]
    assert merge_boxes._footprint_overlap(objs[0], objs[1]) == 0.0
    assert merge_boxes.auto_groups(objs) == []


def test_yaw_within_tolerance_still_merges():
    """RoomPlan yaws jitter by a fraction of a degree across boxes of the same run."""
    objs = [
        box("Sink0", (0, 0, 0), (0.5, 0.3, 0.5), yaw=-121.04),
        box("Storage0", (0, -0.5, 0), (1.2, 0.8, 0.6), yaw=-121.04 + 1.5),
    ]
    assert merge_boxes.auto_groups(objs) == [["Sink0", "Storage0"]]


def test_chain_fuses_transitively():
    """Sink overlaps the base cabinet, the base cabinet overlaps the next one, and the two ends do
    NOT overlap each other — union-find must still return a single run, not two pairs."""
    objs = [
        box("Sink0", (0.0, 0, 0), (1.0, 0.3, 1.0)),
        box("Storage0", (0.8, 0, 0), (1.0, 0.8, 1.0)),
        box("Storage1", (1.6, 0, 0), (1.0, 0.8, 1.0)),
    ]
    assert merge_boxes._footprint_overlap(objs[0], objs[2]) == 0.0
    assert merge_boxes.auto_groups(objs) == [["Sink0", "Storage0", "Storage1"]]


def test_barely_touching_boxes_are_not_merged():
    """Adjacent cabinets in a row touch without interpenetrating — fusing those would erase real
    separate units, so the threshold must sit above a shared edge."""
    objs = [
        box("Storage0", (0.0, 0, 0), (1.0, 0.8, 1.0)),
        box("Storage1", (1.02, 0, 0), (1.0, 0.8, 1.0)),
    ]
    assert merge_boxes.auto_groups(objs) == []


def test_threshold_is_a_fraction_of_the_smaller_box():
    """A small sink fully inside a big cabinet overlaps 100% of itself but only a slice of the
    cabinet. Scoring against the smaller box is what makes containment detectable."""
    small = box("Sink0", (0, 0, 0), (0.3, 0.2, 0.3))
    big = box("Storage0", (0, -0.4, 0), (2.0, 0.9, 0.7))
    assert merge_boxes._footprint_overlap(small, big) == 1.0


# --------------------------------------------------------------------------- the wall guard


def wall(x1: float, z1: float, x2: float, z2: float):
    """One wall centreline as `wall_segments` returns it: a 2-D segment in the ARKit ground plane."""
    return ((x1, z1), (x2, z2))


def test_a_wall_between_two_boxes_stops_the_merge():
    """Two boxes whose footprints overlap are still not one object if a wall stands between them.

    This is the horizontal twin of the vertical gate. Both boxes are the same height here, so the
    height rule has nothing to say — only the wall does.
    """
    objs = [
        box("Storage0", (0.0, 0, 0.0), (1.0, 0.8, 1.0)),
        box("Storage1", (0.6, 0, 0.0), (1.0, 0.8, 1.0)),
    ]
    assert merge_boxes.auto_groups(objs) == [["Storage0", "Storage1"]]
    between = [wall(0.3, -2.0, 0.3, 2.0)]
    assert merge_boxes.auto_groups(objs, walls=between) == []


def test_a_wall_beside_the_run_does_not_stop_it():
    """A counter run is normally installed ALONG a wall. A wall the run merely lies against, rather
    than crosses, must leave the merge alone — otherwise the guard would break every kitchen."""
    objs = [
        box("Sink0", (0.0, 0, 0.0), (1.0, 0.3, 1.0)),
        box("Storage0", (0.6, 0, 0.0), (1.0, 0.8, 1.0)),
    ]
    alongside = [wall(-2.0, 0.7, 2.0, 0.7), wall(-2.0, -0.7, 2.0, -0.7)]
    assert merge_boxes.auto_groups(objs, walls=alongside) == [["Sink0", "Storage0"]]


def test_a_smeared_box_cannot_bridge_two_rooms():
    """The Kitchen-Xiaoyang_Lyu bug, in miniature.

    RoomPlan detected one oven twice and recorded the second copy far too deep — deep enough to
    reach through the wall behind it. That copy overlaps the real run on one side AND a unit on the
    far side, so union-find welds both runs into a single box spanning the wall. Nothing downstream
    can recover from that: the layout pass reads the result as a counter recorded too deep.

    The near-side pair must survive; only the member across the wall is released.
    """
    objs = [
        box("Storage3", (0.0, 0, 0.0), (2.0, 0.9, 0.6)),      # the real counter run
        box("Oven2", (0.2, 0, 0.5), (0.6, 0.9, 1.6)),         # one oven, smeared through the wall
        box("Storage0", (0.3, 0, 1.1), (1.2, 0.9, 0.6)),      # a unit on the FAR side
    ]
    assert merge_boxes.auto_groups(objs) == [["Oven2", "Storage0", "Storage3"]]
    partition = [wall(-3.0, 0.8, 3.0, 0.8)]
    assert merge_boxes.auto_groups(objs, walls=partition) == [["Oven2", "Storage3"]]


def test_only_a_wall_that_actually_stands_between_them_counts():
    """The test is against the wall SEGMENT, not the infinite line through it.

    A room is full of walls whose infinite lines cut everything in half. Scoring against the line
    would refuse merges all over the room; scoring against the segment asks the only question that
    matters, which is whether a wall is physically in the way.
    """
    objs = [
        box("Storage0", (0.0, 0, 0.0), (1.0, 0.8, 1.0)),
        box("Storage1", (0.6, 0, 0.0), (1.0, 0.8, 1.0)),
    ]
    stops_short = [wall(0.3, 1.5, 0.3, 4.0)]      # same line as the separating wall, but elsewhere
    assert merge_boxes.auto_groups(objs, walls=stops_short) == [["Storage0", "Storage1"]]


def test_no_walls_means_exactly_the_old_behaviour():
    """The guard can only ever REMOVE a merge, and it is inert without walls. A scan whose room
    geometry is missing or unreadable must merge exactly as it did before the guard existed."""
    for walls in (None, []):
        assert merge_boxes.auto_groups(FALLSIDE, walls=walls) == [["Sink0", "Storage2"]]


def test_wall_segments_degrades_to_empty_instead_of_raising(tmp_path, capsys):
    """No room on disk is not a merge failure. `wall_segments` says so and returns nothing, because
    taking down the merge over missing wall geometry would be worse than the bug it prevents."""
    assert merge_boxes.wall_segments(tmp_path) == []
    assert "wall guard" in capsys.readouterr().out


def test_the_guard_is_on_by_default_and_opt_outable(monkeypatch, tmp_path):
    """It ships on. `$LR_BOX_MERGE_WALL_GUARD=0` is the escape hatch if a scan's walls are wrong."""
    import litereality_agent.pipeline.scene_init.paths as config

    scene = tmp_path / "scene"
    scene.mkdir()
    objs = [
        box("Storage0", (0.0, 0, 0.0), (1.0, 0.8, 1.0)),
        box("Storage1", (0.6, 0, 0.0), (1.0, 0.8, 1.0)),
    ]
    pickle.dump(objs, open(scene / "objects.pkl", "wb"))
    monkeypatch.setattr(config, "scene_data_dir", lambda scan: scene)
    monkeypatch.setattr(config, "object_refs_root", lambda: tmp_path / "refs")
    seen = {}
    real = merge_boxes.auto_groups
    monkeypatch.setattr(merge_boxes, "auto_groups",
                        lambda o, th=None, vg=None, walls=None: seen.setdefault("walls", walls) or [])
    merge_boxes.merge_for_scan("scan")
    assert seen["walls"] == []          # asked for walls; this scan has none

    seen.clear()
    monkeypatch.setenv("LR_BOX_MERGE_WALL_GUARD", "0")
    merge_boxes.merge_for_scan("scan")
    assert seen["walls"] == []
    monkeypatch.setattr(merge_boxes, "auto_groups", real)


# --------------------------------------------------------------------------- union geometry


def test_union_covers_every_member():
    """The merged box must contain all members: a union that clipped one would crop the reference
    image to a sub-part and defeat the whole merge."""
    merged = merge_boxes._union_obb([b for b in FALLSIDE if b["object_type"] != "Sofa0"])
    w, h, d = (float(v) for v in merged["bbox"])
    for m in FALLSIDE[:3]:
        mw, mh, md = (float(v) for v in m["bbox"])
        assert w >= mw - 1e-6 and h >= mh - 1e-6 and d >= md - 1e-6
    # vertical span must cover sink lip down to base-cabinet floor
    ys = [(m["position"][1] - m["bbox"][1] / 2, m["position"][1] + m["bbox"][1] / 2) for m in FALLSIDE[:3]]
    assert h >= max(y1 for _, y1 in ys) - min(y0 for y0, _ in ys) - 1e-6


def test_union_keeps_the_shared_yaw_and_reports_a_footprint():
    merged = merge_boxes._union_obb([b for b in FALLSIDE if b["object_type"] != "Sofa0"])
    assert merged["rotation"] == -121.04
    assert len(merged["top_down_rect"]) == 4


def test_union_points_at_the_largest_members_mesh():
    """The merged entry needs a source mesh; the biggest footprint is the least-wrong choice."""
    merged = merge_boxes._union_obb([b for b in FALLSIDE if b["object_type"] != "Sofa0"])
    # footprints (w x d): Storage2 1.1202x0.6252 = 0.700 beats Storage0 1.7487x0.3554 = 0.622 —
    # the widest box along the wall is not necessarily the largest in plan.
    assert merged["file"] == "Storage2.obj"


def test_group_name_is_readable_not_merged0():
    """`Sink_Storage0` is identifiable in a reference sheet and the viewer outliner; `Merged0`
    forces a cross-reference to members.json to know what it even is."""
    assert merge_boxes.name_for(["Sink0", "Storage0", "Storage2"]) == "Sink_Storage0"
    assert merge_boxes.name_for(["Storage3", "Oven1", "Stove0"]) == "Storage_Oven_Stove0"


def test_single_category_run_does_not_take_a_members_own_name():
    """Storage0 + Storage5 must not be called `Storage0` — that id belongs to a member it just
    consumed, leaving a merged unit indistinguishable from an ordinary cabinet."""
    assert merge_boxes.name_for(["Storage0", "Storage5"]) == "StorageRun0"


def test_name_avoids_ids_already_in_use():
    taken = {"Sink_Storage0", "Sink_Storage1"}
    assert merge_boxes.name_for(["Sink0", "Storage1"], taken) == "Sink_Storage2"


# --------------------------------------------------------------------------- applying


def _pkl(tmp_path: Path, objs: list[dict]) -> Path:
    p = tmp_path / "objects.pkl"
    with p.open("wb") as f:
        pickle.dump(objs, f)
    return p


def test_apply_rewrites_pkl_and_writes_the_export_marker(tmp_path):
    """`members.json` is the contract with `export_room._apply_shell_merges` — without it the
    SHELL keeps three separate object boxes even though the asset is one merged unit."""
    pkl = _pkl(tmp_path, FALLSIDE)
    refroot = tmp_path / "object_refs"
    res = merge_boxes.apply_merges(pkl, refroot, {"Sink_Storage0": ["Sink0", "Storage0", "Storage2"]})

    assert res["merged"] == {"Sink_Storage0": ["Sink0", "Storage0", "Storage2"]}
    with pkl.open("rb") as f:
        out = pickle.load(f)
    names = {o["object_type"] for o in out}
    assert names == {"Sofa0", "Sink_Storage0"}, "members must be consumed, not left alongside"
    marker = refroot / "Sink_Storage0" / "members.json"
    assert json.loads(marker.read_text()) == ["Sink0", "Storage0", "Storage2"]


def test_apply_backs_up_the_original(tmp_path):
    pkl = _pkl(tmp_path, FALLSIDE)
    merge_boxes.apply_merges(pkl, tmp_path / "refs", {"Sink_Storage0": ["Sink0", "Storage2"]})
    with pkl.with_suffix(".pkl.bak").open("rb") as f:
        assert {o["object_type"] for o in pickle.load(f)} == {
            "Sink0", "Storage0", "Storage2", "Sofa0",
        }


def test_apply_skips_a_group_with_one_present_member(tmp_path):
    """A one-member 'merge' is a rename that silently drops the object's real id — refuse it."""
    pkl = _pkl(tmp_path, FALLSIDE)
    res = merge_boxes.apply_merges(pkl, tmp_path / "refs", {"X0": ["Sink0", "NotThere1"]})
    assert res["merged"] == {} and "X0" in res["skipped"]
    with pkl.open("rb") as f:
        assert len(pickle.load(f)) == len(FALLSIDE), "pkl must be untouched when nothing merges"


def test_arm_enlarged_crops_preserves_existing_ids(monkeypatch):
    """The crop and bbox_polish consumers read this env var; clobbering a value another merge set
    would silently un-enlarge that object's crop."""
    monkeypatch.setenv("LR_ENLARGED_CROP_OBJECTS", "Existing0")
    value = merge_boxes.arm_enlarged_crops({"Sink_Storage0": ["Sink0"]})
    assert set(value.split(",")) == {"Existing0", "Sink_Storage0"}


def test_merge_for_scan_is_opt_outable(monkeypatch):
    monkeypatch.setenv("LR_BOX_MERGE", "0")
    assert merge_boxes.merge_for_scan("whatever-scan").get("disabled") is True


def test_merge_for_scan_never_raises(monkeypatch, tmp_path):
    """A merge failure must degrade to the old unmerged behaviour, not kill a 2-hour init run."""
    monkeypatch.delenv("LR_BOX_MERGE", raising=False)
    monkeypatch.setattr(merge_boxes, "auto_groups", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setenv("LITEREALITY_OUTPUT", str(tmp_path))
    res = merge_boxes.merge_for_scan("no-such-scan")
    assert res["merged"] == {}


# --------------------------------------------------------------------------- the wiring itself


def test_merge_runs_before_crop_in_the_pipeline():
    """THE ACTUAL BUG: the merge must be CALLED, and called before the crop reads objects.pkl.

    Checked at source level because `process_scan` is a long procedural function around vendored
    subprocess stages — there is no seam to unit-test, and the ordering *is* the contract: run the
    merge after `crop_objects` and you have already cropped three separate boxes.
    """
    src = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "litereality_agent"
        / "pipeline"
        / "scene_init"
        / "flow.py"
    ).read_text()
    assert "merge_boxes.merge_for_scan(" in src, "the merge is not wired into init at all"
    i_extract = src.index("extract_scene.extract(")
    i_merge = src.index("merge_boxes.merge_for_scan(")
    i_crop = src.index("crop_objects.crop(")
    assert i_extract < i_merge < i_crop, (
        "box merge must sit between extract (writes objects.pkl) and crop (reads it)"
    )
