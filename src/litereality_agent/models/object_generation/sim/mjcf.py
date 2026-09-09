"""SimModel -> MJCF, with a floor. The other derived export, and the one the checks run on.

MuJoCo can import the URDF directly, but a URDF describes an object and not a WORLD: no ground,
no gravity to fall under, no light. The physics gate needs all three, so the object is written
straight to MJCF instead of being imported and then wrapped. It is the same model either way.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from .properties import SimModel

MJ_JOINT = {"revolute": "hinge", "continuous": "hinge", "prismatic": "slide"}


def _fmt(vals) -> str:
    return " ".join(f"{float(v):.6g}" for v in vals)


def to_mjcf(model: SimModel, out_dir: Path, *, drop_height: float = 0.0,
            floor_friction: float | None = None, free: bool = True, suffix: str = "") -> Path:
    """`drop_height` lifts the object that far above the floor; 0 rests it exactly on it.

    Scenario variants take a `suffix` rather than their own directory: `meshdir` is relative to
    the XML, so a scenario written one level down silently loses every collider it references.
    """
    root = ET.Element("mujoco", model=model.name)
    ET.SubElement(root, "compiler", angle="radian", meshdir=".", autolimits="true")
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast")

    asset = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 0 3", dir="0 0 -1", directional="true")

    base_friction = floor_friction if floor_friction is not None else _root_friction(model)
    ET.SubElement(world, "geom", name="floor", type="plane", size="8 8 0.1",
                  friction=f"{base_friction:.4g} 0.03 0.005", rgba="0.6 0.6 0.62 1")

    by_name = {l.name: l for l in model.links}
    children: dict[str, list] = {}
    for j in model.joints:
        children.setdefault(j.parent, []).append(j)

    # The object sits with its own base exactly on z=0 before any drop is added, so "how far did
    # it fall" is a number about the asset rather than about where the harness happened to put it.
    lift = -float(model.bbox[0][2]) + drop_height
    body = ET.SubElement(world, "body", name=model.root, pos=f"0 0 {lift:.6g}")
    if free:
        ET.SubElement(body, "freejoint", name="root_free")
    _emit(body, by_name[model.root], asset, model)
    _emit_children(body, model.root, children, by_name, asset, model)

    # A leaf touches the carcass it hangs on at every point along its hinge; without this the
    # solver spends every step pushing two bodies apart that are meant to be in contact.
    contact = ET.SubElement(root, "contact")
    for j in model.joints:
        ET.SubElement(contact, "exclude", body1=j.parent, body2=j.child)

    ET.indent(root, space="  ")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{model.name}{suffix}.xml"
    path.write_text(ET.tostring(root, encoding="unicode") + "\n")
    return path


def _root_friction(model: SimModel) -> float:
    return next((l.friction for l in model.links if l.name == model.root), 0.55)


def _emit_children(parent_el, parent_name, children, by_name, asset, model):
    for joint in children.get(parent_name, []):
        link = by_name[joint.child]
        el = ET.SubElement(parent_el, "body", name=link.name, pos=_fmt(joint.origin))
        if joint.type != "fixed":
            ET.SubElement(el, "joint", name=joint.name, type=MJ_JOINT.get(joint.type, "hinge"),
                          axis=_fmt(joint.axis),
                          range=f"{joint.limit_lower:.6g} {joint.limit_upper:.6g}",
                          damping=f"{joint.damping:.4g}", frictionloss=f"{joint.friction:.4g}",
                          armature="0.002")
        _emit(el, link, asset, model)
        _emit_children(el, link.name, children, by_name, asset, model)


def _emit(body_el, link, asset, model):
    ixx, iyy, izz, ixy, ixz, iyz = link.inertia
    ET.SubElement(body_el, "inertial", pos=_fmt(link.com), mass=f"{link.mass:.6g}",
                  fullinertia=f"{ixx:.8g} {iyy:.8g} {izz:.8g} {ixy:.8g} {ixz:.8g} {iyz:.8g}")
    for i, col in enumerate(link.colliders):
        mesh_name = f"{link.name}_c{i}"
        ET.SubElement(asset, "mesh", name=mesh_name, file=col.file)
        # Restitution is deliberately not written: MuJoCo has no such parameter. It stays in the
        # physics json for the engines that do.
        ET.SubElement(body_el, "geom", type="mesh", mesh=mesh_name, condim="4",
                      friction=f"{link.friction:.4g} 0.03 0.005",
                      solref="0.01 1", solimp="0.9 0.95 0.001")
