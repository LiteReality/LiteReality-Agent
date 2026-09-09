"""The authoring half: what a recipe is allowed to declare, and what validate() rejects.

`object_model` imports bpy only inside build(), so the declaration and its validation are testable
without Blender — which is the reason the module is split that way.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = (Path(__file__).resolve().parents[1] / "src/lrauthor/models/object_generation"
           / "articulated-glb-agent/.claude/skills/image-to-articulated-glb/scripts")
sys.path.insert(0, str(SCRIPTS))

import object_model as om  # noqa: E402


def _model(**over):
    parts = [om.Part(name="Carcass", shape="box", size=(0.6, 0.5, 0.8), at=(0, 0, 0.4),
                     material="Wood"),
             om.Part(name="Drawer", shape="box", size=(0.55, 0.1, 0.15), at=(0, 0.2, 0.6),
                     material="Wood", **over.pop("part", {}))]
    mats = [om.Material(name="Wood", **over.pop("material", {}))]
    joints = [om.Joint(part="Drawer", type="prismatic", axis=(0, -1, 0), limit_max=0.4,
                       **over.pop("joint", {}))]
    return om.ArticulatedModel(name="Unit", parts=parts, materials=mats, joints=joints)


def test_a_plain_model_still_validates():
    assert om.validate(_model())["pass"]


def test_mass_and_density_together_is_rejected():
    """They are two statements of the same fact and nothing downstream can decide which wins."""
    report = om.validate(_model(part={"mass": 4.0, "density": 600.0}))
    assert not report["pass"]
    assert any(v["check"] == "physics" for v in report["violations"])


def test_contact_coefficients_are_range_checked():
    assert not om.validate(_model(material={"restitution": 1.4}))["pass"]
    assert not om.validate(_model(material={"friction": -0.2}))["pass"]
    assert om.validate(_model(material={"friction": 0.62, "restitution": 0.2}))["pass"]


def test_a_revolute_joint_without_a_stated_origin_is_flagged_but_not_fatal():
    """Soft, not hard: the pivot can still be recovered from the node transform. Stating it is
    better, and every consumer that has had to recover it has had to guess."""
    model = _model()
    model.joints = [om.Joint(part="Drawer", type="revolute", axis=(0, 0, 1), limit_max=1.57)]
    report = om.validate(model)
    origin = [v for v in report["violations"] if v["check"] == "joint_origin"]
    assert origin and origin[0]["severity"] == "soft"
    assert report["pass"]

    model.joints[0].origin = (-0.27, 0.2, 0.6)
    assert not [v for v in om.validate(model)["violations"] if v["check"] == "joint_origin"]


def test_declared_physics_survives_onto_the_dataclasses():
    m = _model(part={"mass": 3.2}, material={"friction": 0.5, "restitution": 0.1},
               joint={"damping": 1.4, "effort": 60.0})
    assert m.parts[1].mass == 3.2 and m.parts[1].density is None
    assert m.materials[0].friction == 0.5 and m.materials[0].restitution == 0.1
    assert m.joints[0].damping == 1.4 and m.joints[0].effort == 60.0
