"""SimModel -> URDF. A derived export, never a source of truth.

URDF is what the rest of the ecosystem reads — Isaac, PyBullet, Drake, ROS — so an asset that
cannot produce one is an asset those users cannot open. It is emitted, not maintained: rebuild the
object and the URDF is rebuilt with it, so the two can never drift.

What URDF cannot carry, and where it goes instead: restitution has no home in the spec at all (it
stays in the physics json, which is the actual source of truth), and sliding friction goes in
`<contact_coefficients mu=...>`, which the spec does define but most consumers ignore.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from .properties import SimModel


def _fmt(vals) -> str:
    return " ".join(f"{float(v):.6g}" for v in vals)


def to_urdf(model: SimModel, out_dir: Path, *, mesh_dir: str = ".") -> Path:
    robot = ET.Element("robot", name=model.name)

    # MuJoCo reads URDF, but only with its own compiler block present to say where the meshes are.
    # Other parsers ignore an unknown top-level element, so this costs nothing and means the same
    # single file loads in MuJoCo and in everything else.
    mj = ET.SubElement(robot, "mujoco")
    ET.SubElement(mj, "compiler", meshdir=mesh_dir, balanceinertia="true", discardvisual="false")

    for link in model.links:
        el = ET.SubElement(robot, "link", name=link.name)
        inertial = ET.SubElement(el, "inertial")
        ET.SubElement(inertial, "origin", xyz=_fmt(link.com), rpy="0 0 0")
        ET.SubElement(inertial, "mass", value=f"{link.mass:.6g}")
        ixx, iyy, izz, ixy, ixz, iyz = link.inertia
        ET.SubElement(inertial, "inertia", ixx=f"{ixx:.6g}", iyy=f"{iyy:.6g}", izz=f"{izz:.6g}",
                      ixy=f"{ixy:.6g}", ixz=f"{ixz:.6g}", iyz=f"{iyz:.6g}")
        if link.visual:
            vis = ET.SubElement(el, "visual")
            ET.SubElement(ET.SubElement(vis, "geometry"), "mesh",
                          filename=f"{mesh_dir}/{link.visual}" if mesh_dir != "." else link.visual)
        for col in link.colliders:
            c = ET.SubElement(el, "collision")
            ET.SubElement(ET.SubElement(c, "geometry"), "mesh",
                          filename=f"{mesh_dir}/{col.file}" if mesh_dir != "." else col.file)
            ET.SubElement(c, "contact_coefficients", mu=f"{link.friction:.4g}")

    for joint in model.joints:
        el = ET.SubElement(robot, "joint", name=joint.name, type=joint.type)
        ET.SubElement(el, "parent", link=joint.parent)
        ET.SubElement(el, "child", link=joint.child)
        ET.SubElement(el, "origin", xyz=_fmt(joint.origin), rpy="0 0 0")
        if joint.type != "fixed":
            ET.SubElement(el, "axis", xyz=_fmt(joint.axis))
            if joint.type != "continuous":
                ET.SubElement(el, "limit", lower=f"{joint.limit_lower:.6g}",
                              upper=f"{joint.limit_upper:.6g}", effort=f"{joint.effort:.6g}",
                              velocity=f"{joint.velocity:.6g}")
            ET.SubElement(el, "dynamics", damping=f"{joint.damping:.4g}",
                          friction=f"{joint.friction:.4g}")

    ET.indent(robot, space="  ")
    path = Path(out_dir) / f"{model.name}.urdf"
    path.write_text(ET.tostring(robot, encoding="unicode") + "\n")
    return path
