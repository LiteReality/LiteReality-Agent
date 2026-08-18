"""support — nothing floats: every object traces back to the floor, a wall or the ceiling.

Offline and synthetic: boxes assembled by hand into the cases that actually matter, so the test
needs no capture, no Blender and no compiled Room.glb.

  Table0 on the floor, Book0 on the table            the ordinary chain
  Shelf0 on a wall, Mug0 on the shelf                the case the bottom face CANNOT answer —
                                                     a wall-hung shelf has nothing beneath it, and
                                                     the chain still has to reach the wall
  Plank0 across two boxes                            multi-support: its centre is in the GAP between
                                                     the two contact patches and must read stable
  Cup0 half off the table edge                       resting, but unstable
  Bowl0 sunk into the tabletop                       embedded, not resting on
  Lamp0 in mid-air                                   floating
"""

from __future__ import annotations

import numpy as np
import pytest

# trimesh (and the r-tree its ray caster needs) are core dependencies — the support check IS ray
# casts, so a missing one must FAIL these tests rather than skip them. A skipped support gate is an
# invisible one, exactly as in test_check_collisions.
import trimesh

from litereality_agent.agent.tools.check_collisions.source import support as sp

# A 4 m x 3 m room. SHELL plan (x, y) maps into the glb as (x, -y), so the room's glb footprint is
# x in [0, 4], z in [-3, 0]; height is glb y.
SHELL = {
    "floor_z": 0.0,
    "ceiling_z": 2.6,
    "walls": {
        "Wall0": {"start": [0, 0], "end": [4, 0], "thickness": 0.1},
        "Wall1": {"start": [4, 0], "end": [4, 3], "thickness": 0.1},
        "Wall2": {"start": [4, 3], "end": [0, 3], "thickness": 0.1},
        "Wall3": {"start": [0, 3], "end": [0, 0], "thickness": 0.1},
    },
}


def _box(x0, x1, y0, y1, z0, z1):
    """Axis-aligned box from its glb bounds (y is height)."""
    return trimesh.creation.box(
        extents=(x1 - x0, y1 - y0, z1 - z0),
        transform=trimesh.transformations.translation_matrix(
            ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)),
    )


@pytest.fixture(scope="module")
def result():
    bodies = {
        "floor": _box(0, 4, -0.05, 0.0, -3, 0),
        "structure": None,
        "leaves": {},
        "furniture": {
            "Table0": _box(0.5, 1.5, 0.0, 0.75, -1.5, -0.5),      # on the floor
            "Book0": _box(0.9, 1.1, 0.75, 0.80, -1.1, -0.9),      # on the table
            "Bowl0": _box(0.6, 0.8, 0.70, 0.85, -1.4, -1.2),      # sunk 0.05 m into the tabletop
            "Cup0": _box(1.45, 1.65, 0.75, 0.85, -1.1, -0.9),     # half off the table edge
            "Shelf0": _box(0.5, 1.5, 1.20, 1.25, -0.3, 0.0),      # hung on Wall0 (glb z = 0)
            "Mug0": _box(0.9, 1.1, 1.25, 1.35, -0.2, -0.1),       # on the wall-hung shelf
            "BoxA": _box(2.0, 2.3, 0.0, 0.5, -0.5, -0.2),
            "BoxB": _box(3.2, 3.5, 0.0, 0.5, -0.5, -0.2),
            "Plank0": _box(2.05, 3.45, 0.50, 0.55, -0.45, -0.25),  # spans BoxA and BoxB
            "Lamp0": _box(2.9, 3.1, 0.90, 1.10, -2.1, -1.9),      # mid-air
        },
    }
    return sp.find_supports(bodies, SHELL)


def _kinds(result, oid):
    return {f["kind"] for f in result["findings"] if f["id"] == oid}


def test_floor_and_stacking_chain(result):
    """The ordinary case: a table on the floor, a book on the table."""
    assert result["supports"]["Table0"]["kind"] == "floor"
    assert result["supports"]["Book0"] == {"kind": "on_object", "parent": "Table0",
                                           "gap_m": 0.0, "also_on": []}
    assert result["graph"]["Table0"] == "Floor"
    assert result["graph"]["Book0"] == "Table0"
    assert not _kinds(result, "Book0")


def test_wall_hung_shelf_is_found_from_its_back_face(result):
    """The case the bottom face cannot answer — nothing is underneath a wall-hung shelf, so the
    probe has to come off its BACK face instead, and the chain still has to reach the wall."""
    assert result["supports"]["Shelf0"]["kind"] == "wall"
    assert result["supports"]["Shelf0"]["parent"] == "Wall0"
    assert result["graph"]["Mug0"] == "Shelf0"
    assert not _kinds(result, "Shelf0")
    assert not _kinds(result, "Mug0")


def test_multi_support_plank_is_stable(result):
    """A plank across two boxes rests on BOTH. Judging it against one patch would put its centre in
    mid-air and call it unstable."""
    s = result["supports"]["Plank0"]
    assert {s["parent"], *s["also_on"]} == {"BoxA", "BoxB"}
    assert "unstable" not in _kinds(result, "Plank0")


def test_off_the_edge_is_unstable(result):
    """Touching is not enough — the centre has to be over what it touches."""
    assert result["supports"]["Cup0"]["parent"] == "Table0"
    assert "unstable" in _kinds(result, "Cup0")


def test_sunk_into_its_support_is_embedded(result):
    assert "embedded" in _kinds(result, "Bowl0")


def test_mid_air_is_floating(result):
    assert result["supports"]["Lamp0"]["kind"] == "floating"
    assert "floating" in _kinds(result, "Lamp0")


def test_every_chain_reaches_the_structure(result):
    """No cycles and no dangling chains among the well-formed objects."""
    bad = {f["id"] for f in result["findings"]
           if f["kind"] in ("support_cycle", "unsupported_chain")}
    assert bad == set()


def test_support_pairs_are_expected_contact():
    """Resting IS touching, so the clash check needs these pairs on its allow-list."""
    graph = {"Book0": "Table0", "Table0": "Floor", "Shelf0": "Wall0"}
    pairs = sp.support_pairs(graph, {"Book0", "Table0", "Shelf0"})
    assert pairs == {frozenset(("Book0", "Table0"))}


def test_support_cycle_is_reported():
    """Two objects resting on each other reach nothing — the graph check is what catches it."""
    issues = sp._graph_issues({"A": "B", "B": "A"}, {"A", "B"})
    assert {i["kind"] for i in issues} == {"support_cycle"}


def test_contact_face_follows_the_probe_direction():
    """The point of the whole design: which vertices are the contact patch depends on where the
    support is expected to be, not on the object's lowest z."""
    m = _box(0, 1, 0, 2, 0, 1)
    assert np.allclose(sp._contact_face(m, sp.DOWN)[:, 1].max(), 0.0)
    assert np.allclose(sp._contact_face(m, sp.UP)[:, 1].min(), 2.0)
