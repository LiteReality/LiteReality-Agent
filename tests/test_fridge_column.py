"""A storage box RoomPlan drew around a built-in fridge column is a duplicate; a pantry or a box
something else runs into is not."""

import copy

from litereality_agent.pipeline.scene_init.ingest.extract.fridge_column import column_duplicates


def box(category, center, size, yaw=0.0):
    return {"category": category, "center": list(center), "size": list(size), "yaw": yaw}


# A captured built-in kitchen, as the layout read it: the refrigerator, the wall cabinet on it and a
# storage box RoomPlan drew around the whole column (narrower than the fridge and set off from it).
CAPTURED = {
    "Refrigerator0": box("refrigerator", (1.6643, 3.6626, -0.5802), (0.7271, 0.633, 1.8408), -34.56),
    "Storage5": box("storage", (1.5459, 3.4998, 0.6283), (0.7927, 0.6377, 0.5762), -34.56),
    "Storage2": box("storage", (1.4714, 3.5566, -0.2921), (0.6052, 0.6377, 2.417), -34.56),
}


def column(pantry_x=None, extra=None):
    """A fridge 1.8 m tall with a 0.6 m cabinet on it; optionally a tall unit beside it."""
    objects = {
        "Refrigerator0": box("refrigerator", (0, 0, 0.9), (0.75, 0.65, 1.8)),
        "Storage0": box("storage", (0, 0, 2.1), (0.75, 0.65, 0.6)),
    }
    if pantry_x is not None:
        objects["Storage1"] = box("storage", (pantry_x, 0, 1.2), (0.6, 0.65, 2.4))
    if extra:
        objects.update(extra)
    return objects


def scaled(objects, k):
    out = copy.deepcopy(objects)
    for o in out.values():
        o["center"] = [v * k for v in o["center"]]
        o["size"] = [v * k for v in o["size"]]
    return out


def test_a_box_around_the_fridge_and_its_cabinet_is_a_duplicate():
    assert column_duplicates(CAPTURED) == {"Storage2": ("Refrigerator0", "Storage5")}


def test_the_decision_does_not_depend_on_the_room_s_scale():
    assert column_duplicates(scaled(CAPTURED, 2.0)) == column_duplicates(CAPTURED)


def test_a_pantry_a_little_into_the_fridge_stays():
    # beside the fridge, 5 cm into it, floor to the cabinet's top: a real unit, not the column again
    assert column_duplicates(column(pantry_x=0.375 + 0.3 - 0.05)) == {}


def test_a_tall_unit_next_to_the_fridge_stays():
    assert column_duplicates(column(pantry_x=0.375 + 0.3)) == {}


def test_a_column_box_something_else_runs_into_stays():
    # the same column box, but a table also stands 10 cm into it: the layout check must still see that
    table = {"Table0": box("table", (0, 0.65 / 2 + 0.4 - 0.1, 0.4), (1.2, 0.8, 0.8))}
    objects = column(pantry_x=0.0, extra=table)
    assert column_duplicates(column(pantry_x=0.0)) == {"Storage1": ("Refrigerator0", "Storage0")}
    assert column_duplicates(objects) == {}
