"""The physical-property compiler and its exports.

No Blender and no fixture GLBs: the model is built directly, which is the point — every exporter
is a pure function of it, so they can be tested without building anything.
"""

from __future__ import annotations

import importlib.util
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from litereality_agent.models.object_generation.sim.mjcf import to_mjcf  # noqa: E402
from litereality_agent.models.object_generation.sim.properties import (  # noqa: E402
    DEFAULT_FRICTION,
    MATERIAL_FRICTION,
    Collider,
    Joint,
    Link,
    SimModel,
    _inertia_from_geometry,
    _lookup,
)
from litereality_agent.models.object_generation.sim.urdf import to_urdf  # noqa: E402

trimesh = pytest.importorskip("trimesh")
np = pytest.importorskip("numpy")


def _cabinet(tmp_path: Path) -> SimModel:
    """A carcass and one drawer, with real meshes on disk so the exports resolve."""
    for name, extents in (("base_link", (0.6, 0.5, 0.8)), ("drawer", (0.55, 0.45, 0.2))):
        box = trimesh.creation.box(extents=extents)
        box.export(tmp_path / f"{name}_col0.obj")
        box.export(tmp_path / f"{name}_vis.obj")
    links = [
        Link(name="base_link", mass=30.0, com=[0, 0, 0.4], inertia=[2.1, 2.4, 1.5, 0, 0, 0],
             friction=0.55, restitution=0.2, visual="base_link_vis.obj",
             colliders=[Collider("base_link_col0.obj", 0.24)]),
        Link(name="drawer", mass=3.0, com=[0, 0, 0], inertia=[0.08, 0.09, 0.13, 0, 0, 0],
             friction=0.45, restitution=0.1, visual="drawer_vis.obj",
             colliders=[Collider("drawer_col0.obj", 0.05)]),
    ]
    joints = [Joint(name="drawer_joint", type="prismatic", parent="base_link", child="drawer",
                    origin=[0.0, -0.25, 0.5], axis=[0, -1, 0],
                    limit_lower=0.0, limit_upper=0.4)]
    return SimModel(name="Cabinet", category="storage", root="base_link", links=links,
                    joints=joints, bbox=[[-0.3, -0.25, 0.0], [0.3, 0.25, 0.8]], total_mass=33.0)


def test_material_friction_is_keyed_on_the_name_not_an_exact_match():
    assert _lookup(MATERIAL_FRICTION, "Oak_Veneer_Top", DEFAULT_FRICTION) == 0.55
    assert _lookup(MATERIAL_FRICTION, "Brushed_Steel", DEFAULT_FRICTION) == 0.42
    assert _lookup(MATERIAL_FRICTION, "Glazing_Centre", DEFAULT_FRICTION) == 0.30
    assert _lookup(MATERIAL_FRICTION, "SomethingUnnamed", DEFAULT_FRICTION) == DEFAULT_FRICTION


def test_inertia_falls_back_rather_than_returning_a_tensor_mujoco_will_reject():
    """A flat visual panel has no volume. Every engine here refuses a non-positive inertia, so
    the ladder has to end somewhere that always works."""
    flat = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
                           faces=[[0, 1, 2], [0, 2, 3]])
    inertia, com, source = _inertia_from_geometry(flat, mass=2.0)
    assert source == "bounding_box"
    assert np.all(np.linalg.eigvalsh(inertia) > 0)

    solid = trimesh.creation.box(extents=(0.4, 0.3, 0.2))
    inertia, com, source = _inertia_from_geometry(solid, mass=6.0)
    assert source == "geometry"
    # A uniform box: ixx = m/12 (dy^2 + dz^2).
    assert inertia[0, 0] == pytest.approx(6.0 / 12 * (0.3 ** 2 + 0.2 ** 2), rel=1e-3)


def test_urdf_carries_the_joint_origin_explicitly(tmp_path):
    """The origin is the field that made a URDF worth emitting: without it a consumer has to
    re-derive the pivot from the mesh, which is what everything downstream used to do."""
    path = to_urdf(_cabinet(tmp_path), tmp_path)
    root = ET.parse(path).getroot()
    joint = root.find("joint")
    assert joint.get("type") == "prismatic"
    assert joint.find("origin").get("xyz") == "0 -0.25 0.5"
    assert joint.find("axis").get("xyz") == "0 -1 0"
    assert joint.find("limit").get("upper") == "0.4"
    assert joint.find("dynamics").get("damping") == "0.05"
    # Per-link friction survives; restitution has no home in URDF and must not be invented.
    mus = {c.find("contact_coefficients").get("mu")
           for link in root.findall("link") for c in link.findall("collision")}
    assert mus == {"0.55", "0.45"}
    assert root.find(".//restitution") is None


def test_urdf_declares_a_positive_inertia_for_every_link(tmp_path):
    root = ET.parse(to_urdf(_cabinet(tmp_path), tmp_path)).getroot()
    links = root.findall("link")
    assert len(links) == 2
    for link in links:
        inertial = link.find("inertial")
        assert float(inertial.find("mass").get("value")) > 0
        for axis in ("ixx", "iyy", "izz"):
            assert float(inertial.find("inertia").get(axis)) > 0


def test_mjcf_compiles_and_the_drawer_actually_slides(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    xml = to_mjcf(_cabinet(tmp_path), tmp_path, drop_height=0.0)
    model = mujoco.MjModel.from_xml_path(str(xml))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt)]
    assert "drawer_joint" in names
    j = names.index("drawer_joint")
    assert int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_SLIDE)
    assert list(model.jnt_range[j]) == pytest.approx([0.0, 0.4])


def test_mjcf_excludes_contact_between_a_leaf_and_what_it_hangs_on(tmp_path):
    """A drawer touches its carcass along the whole runner. Without the exclude the solver spends
    every step pushing apart two bodies that are meant to be in contact."""
    root = ET.parse(to_mjcf(_cabinet(tmp_path), tmp_path)).getroot()
    pairs = {(e.get("body1"), e.get("body2")) for e in root.findall("./contact/exclude")}
    assert ("base_link", "drawer") in pairs


def test_scenario_variants_stay_beside_their_meshes(tmp_path):
    """meshdir is relative to the XML, so a scenario written into its own directory silently
    loses every collider it references."""
    model = _cabinet(tmp_path)
    drop = to_mjcf(model, tmp_path, drop_height=0.05, suffix="_drop")
    release = to_mjcf(model, tmp_path, free=False, suffix="_release")
    assert drop.parent == release.parent == tmp_path
    assert drop != release


def test_a_missing_native_dependency_is_loud(monkeypatch):
    """The two failure modes that matter are both silent by default.

    Without coacd every concave link falls back to its own convex hull and the gate still reports a
    clean pass — measured on Table1, 11 colliders became 2 and nothing said so. Without mujoco the
    physics json and URDF are still written, so the output looks finished and was never checked.
    Both are declared dependencies, so a missing one is an install fault and must say so.
    """
    from litereality_agent.models.object_generation.sim import properties

    real = importlib.util.find_spec

    def missing(name, *a, **kw):
        return None if name in ("coacd", "mujoco") else real(name, *a, **kw)

    monkeypatch.setattr(importlib.util, "find_spec", missing)

    with pytest.raises(RuntimeError, match="coacd is not installed"):
        properties.require("coacd", "concave links would collapse")
    with pytest.raises(RuntimeError, match="mujoco is not installed"):
        properties.require("mujoco", "the gate cannot run")
    # and it must point at the fix rather than leaving someone to work around it
    try:
        properties.require("coacd", "x")
    except RuntimeError as exc:
        assert "uv sync" in str(exc)


def test_require_passes_for_something_installed():
    from litereality_agent.models.object_generation.sim.properties import require

    require("json", "this can never happen")
