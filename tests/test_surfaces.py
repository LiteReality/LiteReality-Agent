"""Surface discovery — how many surfaces a room has decides what the prompt covers.

`surface_ids` feeds the stitch list in every authoring prompt and the coverage report afterwards.
A wrong answer is invisible: the run reports full coverage over the surfaces it knew about while the
rest of the room was never referenced.
"""

from __future__ import annotations

from lrauthor.room_ops.surfaces import SLIVER_WALL_M, surface_ids


def test_discovers_the_rooms_real_surfaces(stage_tree):
    ids = surface_ids(stage_tree.physical_room / "Room.py")
    assert ids == ["Wall0", "Wall1", "Wall3", "Wall10", "Floor0", "Ceiling0"]


def test_excludes_roomplan_sliver_stubs(stage_tree):
    """Wall2 in the fixture is a 0.21 m corner artifact. Its stitch is a ~20px smear and
    `render_ortho` skips it anyway, so surveying it burns reads and fix attempts on junk."""
    assert "Wall2" not in surface_ids(stage_tree.physical_room / "Room.py")
    assert SLIVER_WALL_M == 0.35


def test_orders_walls_numerically_not_lexicographically(stage_tree):
    """`Wall10` must come after `Wall3`. Sorting as strings puts Wall10 second, which scrambles
    every per-surface list the model is handed."""
    walls = [s for s in surface_ids(stage_tree.physical_room / "Room.py") if s.startswith("Wall")]
    assert walls == sorted(walls, key=lambda w: int(w[4:]))


def test_opening_keys_do_not_mint_phantom_walls(stage_tree):
    """`Wall1_Door_0` is an opening, not a wall. A phantom entry would demand a stitch that can
    never exist and be reported as a permanent coverage gap."""
    src = (stage_tree.physical_room / "Room.py").read_text()
    assert "Wall1_Door_0" in src
    assert len([s for s in surface_ids(stage_tree.physical_room / "Room.py") if s.startswith("Wall")]) == 4


def test_no_hidden_wall_cap(tmp_path):
    """THE REGRESSION the discovery rewrite exists for: a hardcoded `Wall0..Wall8` silently dropped
    every wall past the cap from the prompt AND from the coverage denominator, so a 30-wall flat was
    authored as if it had 9 and still reported 100% coverage.
    """
    walls = "\n".join(
        f'        "Wall{i}": {{"start": [{i}.0, 0.0], "end": [{i}.0, 2.0], "height": 2.5}},'
        for i in range(30)
    )
    room_py = tmp_path / "Room.py"
    room_py.write_text('SHELL = {\n    "walls": {\n' + walls + "\n    },\n}\n")

    ids = surface_ids(room_py)
    assert len([s for s in ids if s.startswith("Wall")]) == 30
    assert ids[-2:] == ["Floor0", "Ceiling0"]


def test_missing_room_py_degrades_instead_of_raising(tmp_path):
    """Prompt building must not crash on a half-built room: floor and ceiling always exist."""
    assert surface_ids(tmp_path / "does-not-exist.py") == ["Floor0", "Ceiling0"]


def test_surfaces_for_bridges_the_room_dir(stage_tree):
    """`author`/`materials_pass`/`qc_pass`/`eval_room` all call this one bridge, so it must accept a
    room DIRECTORY (not a Room.py path) and agree with the parser."""
    from lrauthor.agent.author import surfaces_for

    assert surfaces_for(stage_tree.physical_room) == surface_ids(stage_tree.physical_room / "Room.py")


def test_eval_room_grades_only_real_walls(stage_tree):
    """`eval_room` averages scores over the walls it grades; inventing Wall0..Wall8 for a smaller
    room makes `mean_score` meaningless."""
    from lrauthor.agent.author import surfaces_for

    walls = [s for s in surfaces_for(stage_tree.physical_room) if s.startswith("Wall")]
    assert walls == ["Wall0", "Wall1", "Wall3", "Wall10"]
