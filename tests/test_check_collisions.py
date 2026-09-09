"""check_collisions — the deterministic clash/containment tool the QC pass uses to make furniture
clash-free and inside the walls.

Offline and synthetic: a hand-built SHELL with one of each defect (two tables overlapping, a chair
buried in a wall, a table past the footprint, a chair floating) plus a chair correctly tucked under
a table. Asserts the tool flags exactly the real defects, ignores the correct overlap, and returns a
metric fix pointing the right way. Also pins the `_extract_shell` robustness fix: a SHELL the model
has annotated with Python-but-not-JSON must still parse (or the whole check silently reads empty).
"""

from __future__ import annotations

import asyncio
import json

# python-fcl is a core dependency (the true-mesh backend the QC clash gate runs on), so a
# missing one must FAIL these tests rather than skip them — a skipped clash gate is invisible.
import fcl  # noqa: F401
import pytest

from lrauthor.agent.tools.check_collisions.source import geometry as qc_room
from lrauthor.agent.tools.check_collisions.tool import (
    CheckCollisionsInvocation,
    CheckCollisionsParams,
)

# A 4 m × 3 m room (footprint x[0,4] y[0,3], centre (2,1.5)), floor at z=0.
_WALLS = {
    "Wall0": {"start": [0, 0], "end": [4, 0], "thickness": 0.1},
    "Wall1": {"start": [4, 0], "end": [4, 3], "thickness": 0.1},
    "Wall2": {"start": [4, 3], "end": [0, 3], "thickness": 0.1},
    "Wall3": {"start": [0, 3], "end": [0, 0], "thickness": 0.1},
}
_OBJECTS = {
    # two tables overlapping ~0.4 m along +x  → object_clash
    "TableA": {"category": "table", "center": [1.0, 1.0, 0.4], "size": [1.0, 1.0, 0.8], "yaw": 0},
    "TableB": {"category": "table", "center": [1.6, 1.0, 0.4], "size": [1.0, 1.0, 0.8], "yaw": 0},
    # chair centre 0.03 m into Wall0 (half-thickness 0.05) → wall_clash, snap +y into the room
    "ChairW": {"category": "chair", "center": [3.0, 0.03, 0.4], "size": [0.5, 0.5, 0.8], "yaw": 0},
    # table centre at x=5, past the footprint → outside_room, move −x back in
    "TableOut": {"category": "table", "center": [5.0, 1.5, 0.4], "size": [0.8, 0.8, 0.8], "yaw": 0},
    # chair tucked under a table → CORRECT overlap, must NOT be flagged
    "TableC": {"category": "table", "center": [3.5, 2.5, 0.4], "size": [1.0, 1.0, 0.8], "yaw": 0},
    "ChairC": {"category": "chair", "center": [3.5, 2.5, 0.3], "size": [0.5, 0.5, 0.6], "yaw": 0},
    # chair 0.75 m above the floor → floating (grounding, not a clash)
    "ChairFloat": {"category": "chair", "center": [1.0, 2.5, 1.0], "size": [0.5, 0.5, 0.5], "yaw": 0},
}
_SHELL = {"floor_z": 0.0, "ceiling_z": 3.0, "walls": _WALLS, "objects": _OBJECTS}


def _write_room(tmp_path, shell_literal: str):
    room = tmp_path / "room"
    room.mkdir()
    (room / "Room.py").write_text(f"SHELL = {shell_literal}\n\nif __name__ == '__main__':\n    pass\n")
    return room


def _run(room, method="box"):
    # These synthetic rooms have no compiled Room.glb, so they exercise the fast box fallback path
    # (the default method is now 'mesh', which requires a glb — covered by the scene_collision tests).
    inv = CheckCollisionsInvocation(CheckCollisionsParams(method=method))
    inv.bind(str(room), None)
    return asyncio.run(inv.execute())


def test_flags_each_real_defect_once(tmp_path):
    room = _write_room(tmp_path, json.dumps(_SHELL, indent=2))
    res = _run(room)
    assert res.is_success(), res.error
    out = res.output
    kinds = sorted(c["kind"] for c in out["clashes"])
    assert kinds == ["object_clash", "outside_room", "wall_clash"], kinds
    assert out["n_furniture"] == 7


def test_correct_tuck_is_not_flagged(tmp_path):
    room = _write_room(tmp_path, json.dumps(_SHELL, indent=2))
    out = _run(room).output
    flagged = {c["id"] for c in out["clashes"]} | {c.get("with") for c in out["clashes"]}
    assert "ChairC" not in flagged, "a chair tucked under a table is a correct overlap, not a clash"
    assert "TableC" not in flagged


def test_object_clash_pushes_along_the_overlap_axis(tmp_path):
    room = _write_room(tmp_path, json.dumps(_SHELL, indent=2))
    out = _run(room).output
    oc = next(c for c in out["clashes"] if c["kind"] == "object_clash")
    assert {oc["id"], oc["with"]} == {"TableA", "TableB"}
    assert oc["overlap_m"] == 0.4  # boxes overlap 0.4 m along x
    dx, dy = oc["fix"]["move"]["world_dxdy"]
    assert abs(dx) == 0.4 and abs(dy) < 1e-6, "separation is along x, the overlap axis"


def test_wall_clash_snaps_into_the_room(tmp_path):
    room = _write_room(tmp_path, json.dumps(_SHELL, indent=2))
    out = _run(room).output
    wc = next(c for c in out["clashes"] if c["kind"] == "wall_clash")
    assert wc["id"] == "ChairW" and wc["wall"] == "Wall0"
    dx, dy = wc["fix"]["snap_to_wall"]["world_dxdy"]
    # Wall0 lies on y=0 with the room to +y, so the snap must push the chair in +y, out of the slab.
    assert dy > 0 and abs(dx) < 1e-6


def test_outside_room_moves_back_onto_the_footprint(tmp_path):
    room = _write_room(tmp_path, json.dumps(_SHELL, indent=2))
    out = _run(room).output
    orr = next(c for c in out["clashes"] if c["kind"] == "outside_room")
    assert orr["id"] == "TableOut"
    dx, dy = orr["fix"]["move_inside"]["world_dxdy"]
    assert dx == -1.0 and abs(dy) < 1e-6  # x=5 clamps back to the footprint edge x=4


def test_floating_is_reported_under_grounding_not_as_a_clash(tmp_path):
    room = _write_room(tmp_path, json.dumps(_SHELL, indent=2))
    out = _run(room).output
    assert any(g["id"] == "ChairFloat" and g["check"] == "floating" for g in out["grounding"])
    assert "ChairFloat" not in {c["id"] for c in out["clashes"]}


def test_include_grounding_false_suppresses_grounding(tmp_path):
    room = _write_room(tmp_path, json.dumps(_SHELL, indent=2))
    inv = CheckCollisionsInvocation(CheckCollisionsParams(method="box", include_grounding=False))
    inv.bind(str(room), None)
    out = asyncio.run(inv.execute()).output
    assert out["grounding"] == []


def test_empty_or_unreadable_shell_errors_cleanly(tmp_path):
    room = tmp_path / "room"
    room.mkdir()
    (room / "Room.py").write_text("SHELL = {}\n")
    res = _run(room)
    assert not res.is_success()
    assert "objects" in (res.error or "")


# --------------------------------------------------------------------------- #
# _extract_shell robustness — the reason the tool reads the live SHELL at all
# --------------------------------------------------------------------------- #
def test_extract_shell_parses_python_but_not_json():
    """The model annotates SHELL as PYTHON. An adjacent-string note (`"a" "b"`) is valid Python and
    invalid JSON; the old json-only parse returned {} and blinded every geometry check."""
    literal = (
        '{\n'
        '  "floor_z": 0.0,\n'
        '  "note": "moved ChairW toward the wall start;" " see Wall0 stitch",\n'  # adjacent strings
        '  "walls": {"Wall0": {"start": [0, 0], "end": [4, 0], "thickness": 0.1,}},\n'  # trailing comma
        '  "objects": {"TableA": {"category": "table", "center": [1, 1, 0.4], "size": [1, 1, 0.8], "yaw": 0}}\n'
        '}'
    )
    import json as _json

    try:
        _json.loads(literal)
        raise AssertionError("fixture no longer exercises the JSON-invalid path")
    except ValueError:
        pass
    shell = qc_room._extract_shell(f"SHELL = {literal}\n")
    assert shell.get("walls") and shell.get("objects"), "ast fallback must recover the SHELL"
    assert shell["note"] == "moved ChairW toward the wall start; see Wall0 stitch"


def test_tool_works_on_a_python_annotated_shell(tmp_path):
    """End to end: the tool must still find a planted clash when SHELL carries a Python-only note."""
    literal = (
        '{\n'
        '  "floor_z": 0.0, "ceiling_z": 3.0,\n'
        '  "walls": {"Wall0": {"start": [0, 0], "end": [4, 0], "thickness": 0.1},\n'
        '            "Wall1": {"start": [4, 0], "end": [4, 3], "thickness": 0.1},\n'
        '            "Wall2": {"start": [4, 3], "end": [0, 3], "thickness": 0.1},\n'
        '            "Wall3": {"start": [0, 3], "end": [0, 0], "thickness": 0.1}},\n'
        '  "objects": {\n'
        '     "TableA": {"category": "table", "center": [1.0, 1.0, 0.4], "size": [1.0, 1.0, 0.8], "yaw": 0,\n'
        '                "note": "kept where the" " stitch shows it"},\n'  # Python-only note
        '     "TableB": {"category": "table", "center": [1.6, 1.0, 0.4], "size": [1.0, 1.0, 0.8], "yaw": 0}}\n'
        '}'
    )
    room = _write_room(tmp_path, literal)
    out = _run(room).output
    assert any(c["kind"] == "object_clash" for c in out["clashes"]), out["summary"]


# --------------------------------------------------------------------------- #
# mesh method — the true-mesh narrow-phase confirm (scene_collision)
# --------------------------------------------------------------------------- #
def test_glb_to_shell_xy_axis_map():
    """The verified frame map SHELL(x, y) = glb(x, -z) — a mesh fix in glb space must land in the
    SHELL plane the model edits, or every mesh-mode suggestion points the wrong way."""
    from lrauthor.agent.tools.check_collisions.source.collision_mesh import glb_to_shell_xy

    assert glb_to_shell_xy(0.3, 0.0) == (0.3, 0.0)
    assert glb_to_shell_xy(0.0, 0.4) == (0.0, -0.4)


def test_mesh_contacts_booleans():
    """mesh_contacts returns true-mesh contact booleans (not FCL depth): overlapping furniture boxes
    are a pair, separated ones are not."""
    trimesh = pytest.importorskip("trimesh")
    from lrauthor.agent.tools.check_collisions.source import collision_mesh as sc

    a = trimesh.creation.box((1, 1, 1))
    b = trimesh.creation.box((1, 1, 1))
    b.apply_translation((0.6, 0, 0))  # overlaps a
    c = trimesh.creation.box((1, 1, 1))
    c.apply_translation((5, 0, 0))    # far from both
    bodies = {"furniture": {"A": a, "B": b, "C": c}, "structure": None, "floor": None, "leaves": {}}
    contacts = sc.mesh_contacts(bodies)
    assert frozenset(("A", "B")) in contacts["object_pairs"]
    assert frozenset(("A", "C")) not in contacts["object_pairs"]
    assert frozenset(("B", "C")) not in contacts["object_pairs"]


def test_table_poking_through_a_wall_is_wall_clash_not_missed(tmp_path):
    """The bug the user caught: a table whose CENTRE is well inside the room but whose BODY pokes
    through a ~0 mm-thick wall. The old centre-in-slab test missed it entirely; the extent test must
    flag it as wall_clash (snapping inward), and NOT as outside_room (the centre is inside)."""
    shell = {
        "floor_z": 0.0, "ceiling_z": 3.0,
        "walls": {"Wall0": {"start": [0, 0], "end": [4, 0], "thickness": 0.0001},
                  "Wall1": {"start": [4, 0], "end": [4, 3], "thickness": 0.0001},
                  "Wall2": {"start": [4, 3], "end": [0, 3], "thickness": 0.0001},
                  "Wall3": {"start": [0, 3], "end": [0, 0], "thickness": 0.0001}},
        # centre 0.30 m off Wall0, but 0.50 m half-depth → body pokes ~0.20 m through the wall
        "objects": {"Desk0": {"category": "desk", "center": [2.0, 0.30, 0.4],
                              "size": [1.2, 1.0, 0.8], "yaw": 0}},
    }
    room = _write_room(tmp_path, json.dumps(shell))
    out = _run(room).output
    wc = [c for c in out["clashes"] if c["kind"] == "wall_clash"]
    assert wc and wc[0]["id"] == "Desk0" and wc[0]["wall"] == "Wall0"
    assert wc[0]["penetration_m"] == pytest.approx(0.20, abs=0.01)
    assert wc[0]["fix"]["snap_to_wall"]["world_dxdy"][1] > 0  # snap in +y, into the room
    assert not any(c["kind"] == "outside_room" for c in out["clashes"])  # centre is inside


def test_wall_penetrations_mesh_extent(tmp_path):
    """scene_collision.wall_penetrations flags a furniture MESH crossing a wall plane (glb frame,
    SHELL(x,y)=glb(x,-z)), which the box centre tests can't see."""
    trimesh = pytest.importorskip("trimesh")
    from lrauthor.agent.tools.check_collisions.source import collision_mesh as sc

    shell = {"walls": {"Wall0": {"start": [0, 0], "end": [4, 0]},
                       "Wall2": {"start": [4, 3], "end": [0, 3]}}}
    # a desk box in glb space centred at (2, 0.4, -0.3): its z spans [-0.8, 0.2], so it pokes 0.2 m
    # past Wall0 (glb-plan line z=0; room interior is −z) to the exterior.
    desk = trimesh.creation.box((1.0, 0.8, 1.0))
    desk.apply_translation((2.0, 0.4, -0.3))
    bodies = {"furniture": {"Desk0": desk}, "structure": None, "floor": None, "leaves": {}}
    found = sc.wall_penetrations(bodies, shell)
    assert found and found[0]["id"] == "Desk0" and found[0]["wall"] == "Wall0"
    assert found[0]["penetration_m"] == pytest.approx(0.20, abs=0.03)


def test_mesh_without_a_glb_errors_clearly(tmp_path):
    """The default (mesh) needs a compiled Room.glb. Without one it must error with a clear, actionable
    message (compile, or use method='box') rather than silently doing something coarse."""
    room = _write_room(tmp_path, json.dumps(_SHELL, indent=2))
    inv = CheckCollisionsInvocation(CheckCollisionsParams())  # default = mesh
    inv.bind(str(room), None)
    res = asyncio.run(inv.execute())
    assert not res.is_success()
    assert "Room.glb" in res.error and "method='box'" in res.error


def test_check_all_mesh_end_to_end():
    """scene_collision.check_all over real trimesh bodies (no boxes): a chair jammed into a table is an
    object_clash, a chair tucked UNDER it is not, a desk poking through a wall is a wall_clash, and a
    piece past the wall loop is outside_room."""
    trimesh = pytest.importorskip("trimesh")
    from lrauthor.agent.tools.check_collisions.source import collision_mesh as sc

    shell = {
        "floor_z": 0.0, "ceiling_z": 3.0,
        "walls": {"W0": {"start": [0, 0], "end": [4, 0]}, "W1": {"start": [4, 0], "end": [4, 3]},
                  "W2": {"start": [4, 3], "end": [0, 3]}, "W3": {"start": [0, 3], "end": [0, 0]}},
        "objects": {"Table0": {"category": "table"}, "ChairJam": {"category": "chair"},
                    "DeskThru": {"category": "desk"}, "Stray": {"category": "chair"}},
    }
    # glb frame: SHELL(x,y) = glb(x,-z). Room interior spans glb z in [-3, 0].
    table = trimesh.creation.box((1.2, 0.75, 0.8))
    table.apply_translation((2.0, 0.4, -1.5))
    jam = trimesh.creation.box((0.5, 0.9, 0.5))
    jam.apply_translation((2.4, 0.45, -1.5))   # jammed into the table
    desk = trimesh.creation.box((1.0, 0.8, 1.0))
    desk.apply_translation((1.0, 0.4, -0.3))   # pokes past W0 (glb-plan line z=0)
    stray = trimesh.creation.box((0.4, 0.8, 0.4))
    stray.apply_translation((2.0, 0.4, 1.0))   # outside the wall loop (z>0)
    bodies = {"furniture": {"Table0": table, "ChairJam": jam, "DeskThru": desk, "Stray": stray},
              "structure": None, "floor": None, "leaves": {}}

    kinds = {(f["id"], f["kind"]) for f in sc.check_all(bodies, shell)}
    assert ("ChairJam", "object_clash") in kinds or ("Table0", "object_clash") in kinds
    assert ("DeskThru", "wall_clash") in kinds
    assert ("Stray", "outside_room") in kinds
