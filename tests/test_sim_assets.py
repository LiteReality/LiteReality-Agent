"""The room export reading a generated object's OWN physics, rather than inventing it.

Every assertion here is about a number surviving the trip from the asset into the scene: the mass
the object was compiled with, the friction of the material its recipe chose, the pivot its build
put the hinge on, and the convex pieces a solver already accepted. Getting any of them wrong is
silent — the scene still loads, still renders, and behaves like a different room.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from litereality_agent.room_ops.export import sim_assets

trimesh = pytest.importorskip("trimesh")


def _model_json(name="Widget0", *, mass_source="derived", joints=True):
    doc = {
        "schema_version": 1,
        "name": name,
        "category": "cabinet",
        "root": "base_link",
        "bbox": [[-0.3, -0.2, 0.0], [0.3, 0.2, 0.9]],
        "total_mass": 12.0,
        "notes": [],
        "links": [
            {"name": "base_link", "mass": 8.0, "com": [0, 0, 0.45],
             "inertia": [1.0, 1.0, 0.5, 0, 0, 0], "friction": 0.55, "restitution": 0.15,
             "material": "Oak", "visual": f"{name}_base_link_vis.obj",
             "colliders": [{"file": f"{name}_base_link_col0.obj", "volume": 0.05}],
             "mass_source": mass_source, "inertia_source": "geometry"},
            {"name": "door_leaf", "mass": 4.0, "com": [0, 0, 0.45],
             "inertia": [0.4, 0.4, 0.1, 0, 0, 0], "friction": 0.30, "restitution": 0.30,
             "material": "Glazing", "visual": f"{name}_door_leaf_vis.obj",
             "colliders": [{"file": f"{name}_door_leaf_col0.obj", "volume": 0.01}],
             "mass_source": mass_source, "inertia_source": "geometry"},
        ],
        "joints": [
            {"name": "door_leaf_joint", "type": "revolute", "parent": "base_link",
             "child": "door_leaf", "origin": [0.3, -0.2, 0.05], "axis": [0.0, 0.0, 1.0],
             "limit_lower": 0.0, "limit_upper": 1.5708, "damping": 0.05, "friction": 0.02,
             "effort": 20.0, "velocity": 3.0, "origin_source": "authored"},
        ] if joints else [],
    }
    return doc


def _write_sidecar(directory: Path, doc: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for link in doc["links"]:
        for collider in link["colliders"]:
            box = trimesh.creation.box(extents=(0.2, 0.1, 0.3))
            box.export(directory / collider["file"])
    path = directory / f"{doc['name']}.physics.json"
    path.write_text(json.dumps(doc, indent=2))
    return path


def test_a_sidecar_from_a_schema_this_does_not_know_is_refused(tmp_path):
    """Reading it anyway would put a wrong mass or a wrong hinge in and report success."""
    doc = _model_json()
    doc["schema_version"] = 99
    path = _write_sidecar(tmp_path, doc)
    with pytest.raises(ValueError, match="schema"):
        sim_assets.load(path)


def test_link_frames_accumulate_down_the_joint_chain(tmp_path):
    """A joint's origin is stated in its PARENT's frame, so a link two joints down is the sum."""
    doc = _model_json()
    doc["links"].append({"name": "handle", "mass": 0.3, "com": [0, 0, 0],
                         "inertia": [0.01, 0.01, 0.01, 0, 0, 0], "friction": 0.4,
                         "restitution": 0.2, "material": "", "visual": "",
                         "colliders": [], "mass_source": "derived",
                         "inertia_source": "geometry"})
    doc["joints"].append({"name": "handle_joint", "type": "prismatic", "parent": "door_leaf",
                          "child": "handle", "origin": [0.0, 0.1, 0.4], "axis": [0, 1, 0],
                          "limit_lower": 0.0, "limit_upper": 0.05, "damping": 0.05,
                          "friction": 0.02, "effort": 20.0, "velocity": 1.0,
                          "origin_source": "authored"})
    model = sim_assets.load(_write_sidecar(tmp_path, doc))
    frames = model.frames()
    assert np.allclose(frames["base_link"], [0, 0, 0])
    assert np.allclose(frames["door_leaf"], [0.3, -0.2, 0.05])
    assert np.allclose(frames["handle"], [0.3, -0.1, 0.45])


def test_a_cycle_in_the_joints_does_not_hang_or_raise(tmp_path):
    """A malformed sidecar is a bad file, not a reason for an export to never finish."""
    doc = _model_json()
    doc["joints"].append({"name": "loop", "type": "revolute", "parent": "door_leaf",
                          "child": "base_link", "origin": [1, 1, 1], "axis": [0, 0, 1],
                          "limit_lower": 0.0, "limit_upper": 1.0, "damping": 0.05,
                          "friction": 0.02, "effort": 20.0, "velocity": 3.0,
                          "origin_source": "authored"})
    model = sim_assets.load(_write_sidecar(tmp_path, doc))
    frames = model.frames()
    assert set(frames) >= {"base_link", "door_leaf"}


# ------------------------------------------------------------------ placement


def _scene(nodes):
    """A trimesh Scene with one box per (name, transform)."""
    scene = trimesh.Scene()
    for name, transform in nodes:
        scene.add_geometry(trimesh.creation.box(extents=(0.2, 0.2, 0.2)),
                           node_name=name, geom_name=f"g_{name}", transform=transform)
    return scene


def _yaw_scale(yaw, scale, translation):
    m = np.eye(4)
    c, s = math.cos(yaw), math.sin(yaw)
    m[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ np.diag(scale)
    m[:3, 3] = translation
    return m


def test_the_placement_transform_is_recovered_exactly_from_one_shared_node():
    """`build_room` fits an asset into a measured box — a yaw and a per-axis scale. One node
    present in both files pins the whole affine down, and every other node then maps through it."""
    expected = _yaw_scale(0.7, (1.3, 0.8, 1.1), (2.0, -1.0, 0.4))
    object_scene = _scene([("Carcass", np.eye(4)),
                           ("Leaf", trimesh.transformations.translation_matrix([0.5, 0, 0]))])
    room_scene = _scene([("Carcass", expected),
                         ("Leaf", expected @ trimesh.transformations.translation_matrix(
                             [0.5, 0, 0]))])
    got = sim_assets.placement_transform(object_scene, room_scene, ["Carcass", "Leaf"])
    assert got is not None
    assert np.allclose(got, expected, atol=1e-9)


def test_no_matchable_node_returns_none_rather_than_guessing():
    """trimesh renames a duplicated node with a random suffix per load, so an object whose every
    node was duplicated somewhere in the room has nothing to match on. Placing it anyway would put
    its colliders where the object is not."""
    object_scene = _scene([("Leaf_aaaaaa", np.eye(4)), ("Leaf_bbbbbb", np.eye(4))])
    room_scene = _scene([("Leaf_cccccc", np.eye(4)), ("Leaf_dddddd", np.eye(4))])
    assert sim_assets.placement_transform(object_scene, room_scene,
                                          ["Leaf_cccccc", "Leaf_dddddd"]) is None


def test_place_carries_colliders_pivots_and_axes_into_the_room(tmp_path):
    model = sim_assets.load(_write_sidecar(tmp_path, _model_json()))
    # A quarter turn, no scale, moved across the room: the pivot must arrive rotated, not raw.
    transform_zup = _yaw_scale(math.pi / 2, (1.0, 1.0, 1.0), (5.0, 1.0, 0.0))
    transform_yup = np.linalg.inv(sim_assets._A4) @ transform_zup @ sim_assets._A4
    placed = sim_assets.place(model, transform_yup)

    assert set(placed.links) == {"base_link", "door_leaf"}
    # origin [0.3, -0.2, 0.05] rotated +90 deg about Z is [0.2, 0.3, 0.05], then translated.
    assert np.allclose(placed.pivots["door_leaf"], [5.2, 1.3, 0.05], atol=1e-6)
    # A vertical hinge stays vertical.
    assert np.allclose(placed.axes["door_leaf"], [0, 0, 1], atol=1e-6)
    # The friction is the one the recipe's material implies, per link — not one for the room.
    assert placed.links["base_link"].friction == pytest.approx(0.55)
    assert placed.links["door_leaf"].friction == pytest.approx(0.30)
    assert placed.links["door_leaf"].colliders


def test_a_derived_mass_scales_with_the_box_but_an_authored_one_does_not(tmp_path):
    """Occupancy density over a bounding box is a rule, so at a different size it is a different
    mass. A mass the recipe read off the real object is a fact, and the fit does not change it."""
    scale = (2.0, 1.0, 1.0)                                   # the room stretched it 2x in X
    transform_zup = _yaw_scale(0.0, scale, (0, 0, 0))
    transform_yup = np.linalg.inv(sim_assets._A4) @ transform_zup @ sim_assets._A4

    derived = sim_assets.place(sim_assets.load(
        _write_sidecar(tmp_path / "d", _model_json())), transform_yup)
    authored = sim_assets.place(sim_assets.load(
        _write_sidecar(tmp_path / "a", _model_json(mass_source="authored"))), transform_yup)

    assert derived.links["base_link"].mass == pytest.approx(16.0)
    assert authored.links["base_link"].mass == pytest.approx(8.0)


def test_a_missing_collider_file_drops_the_link_rather_than_emitting_a_ghost(tmp_path):
    """A body with no collider falls through everything it touches, silently. Dropping the link
    welds its geometry into the carcass instead, which is wrong in a way you can see."""
    doc = _model_json()
    path = _write_sidecar(tmp_path, doc)
    (tmp_path / "Widget0_door_leaf_col0.obj").unlink()
    placed = sim_assets.place(sim_assets.load(path), np.eye(4))
    assert "door_leaf" not in placed.links
    assert "base_link" in placed.links


def test_nodes_are_assigned_to_links_past_the_room_glbs_rename(tmp_path):
    """The room appends `_<6 hex>` to any node name it has seen before, and longest match wins so
    a part with its own joint is not swallowed by the link it hangs on."""
    doc = _model_json()
    doc["links"].append({"name": "door_leaf_handle", "mass": 0.2, "com": [0, 0, 0],
                         "inertia": [0.01, 0.01, 0.01, 0, 0, 0], "friction": 0.4,
                         "restitution": 0.2, "material": "", "visual": "", "colliders": [],
                         "mass_source": "derived", "inertia_source": "geometry"})
    model = sim_assets.load(_write_sidecar(tmp_path, doc))
    owned = sim_assets.assign_nodes(
        ["Carcass", "door_leaf_a1b2c3", "door_leaf_panel", "door_leaf_handle_ff0011"], model)
    assert owned["base_link"] == ["Carcass"]
    assert sorted(owned["door_leaf"]) == ["door_leaf_a1b2c3", "door_leaf_panel"]
    assert owned["door_leaf_handle"] == ["door_leaf_handle_ff0011"]


# ------------------------------------------------------- into the Room package


def test_only_this_objects_files_are_carried_into_the_room_package(tmp_path):
    """A generative object is a bare glb at the top of `reconstruct/`, so every chair in the room
    shares one `sim/`. Copying the DIRECTORY would put all of them into each object's package."""
    from litereality_agent.room_ops.export.export_room import copy_sim_sidecar

    shared = tmp_path / "sim"
    _write_sidecar(shared, _model_json("ChairCluster0"))
    _write_sidecar(shared, _model_json("ChairCluster1"))
    (shared / "ChairCluster0.urdf").write_text("<robot/>")

    dest = tmp_path / "package" / "sim"
    copied = copy_sim_sidecar(shared, dest, "ChairCluster0")

    names = sorted(p.name for p in dest.iterdir())
    assert copied == len(names)
    assert "ChairCluster0.physics.json" in names
    assert "ChairCluster0.urdf" in names
    assert not any("ChairCluster1" in n for n in names)


def test_an_object_with_no_compiled_physics_copies_nothing_and_does_not_raise(tmp_path):
    """Physics is new; every room exported before it exists without one, and an export that
    refused those would be an export nobody could run."""
    from litereality_agent.room_ops.export.export_room import copy_sim_sidecar

    assert copy_sim_sidecar(tmp_path / "absent", tmp_path / "dest", "Table0") == 0


def test_sidecar_files_are_named_for_the_object_so_two_cannot_overwrite_each_other(tmp_path):
    """Both objects have a `base_link`. When they share an output directory — which every
    generative object does — link-named files meant the second one's geometry replaced the first
    one's while both physics.json files went on claiming their own volumes, and both objects
    passed their own gate."""
    from litereality_agent.models.object_generation.sim import properties

    first = {c["file"] for c in _model_json("ChairCluster0")["links"][0]["colliders"]}
    second = {c["file"] for c in _model_json("ChairCluster1")["links"][0]["colliders"]}
    assert not (first & second), "the fixture must exercise distinct names"
    # And the compiler is what produces them: the stem it builds colliders under carries the
    # object's name, not just the link's.
    source = Path(properties.__file__).read_text(encoding="utf-8")
    assert 'stem = f"{name}_{link}"' in source
    assert "_colliders(local, stem," in source


def test_an_instanced_object_is_placed_from_geometry_when_no_name_survives(tmp_path):
    """Four chairs from one cluster: every node is renamed for every instance after the first, and
    a cluster whose own parts repeat has nothing unique inside its own file either. The placement
    still has a known shape — a yaw and a per-axis scale — so it comes out of the two bounding
    boxes instead. Rejecting here would have left exactly the objects a room has several of
    collided by guesswork."""
    doc = _model_json()
    doc["bbox"] = [[-0.3, -0.2, 0.0], [0.3, 0.2, 0.9]]
    model = sim_assets.load(_write_sidecar(tmp_path, doc))

    yaw, scale, translation = 0.6, (1.5, 0.9, 1.2), (3.0, -2.0, 0.1)
    truth = _yaw_scale(yaw, scale, translation)
    lo, hi = np.asarray(doc["bbox"][0]), np.asarray(doc["bbox"][1])
    centre = (lo + hi) / 2.0
    # The object's own bounding box, carried into the room the way `build_room` carries it.
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                        for z in (lo[2], hi[2])])
    world = (corners - centre) @ truth[:3, :3].T + truth[:3, 3]
    placed_mesh = trimesh.PointCloud(world).convex_hull

    recovered = sim_assets.placement_from_bbox(model, placed_mesh, yaw)
    assert recovered is not None
    as_zup = sim_assets._A4 @ recovered @ sim_assets._A4_INV
    assert np.allclose(as_zup[:3, :3], truth[:3, :3], atol=1e-6)
    # `truth` maps the asset's CENTRED box, so its translation is the room position of that centre.
    assert np.allclose(as_zup[:3, :3] @ centre + as_zup[:3, 3], translation, atol=1e-6)


def test_a_model_with_no_recorded_bbox_cannot_be_placed_from_geometry(tmp_path):
    """Without the object's own extent there is no second box to compare against, and inventing
    one would scale the colliders by whatever the room happened to measure."""
    doc = _model_json()
    doc["bbox"] = []
    model = sim_assets.load(_write_sidecar(tmp_path, doc))
    box = trimesh.creation.box(extents=(1, 1, 1))
    assert sim_assets.placement_from_bbox(model, box, 0.0) is None
