"""Regenerate the figures in this directory from the compiled objects under run/.

    python assets/sim-ready/make_figures.py [--out assets/sim-ready]

`colliders.png` — what the solver actually collides, closed and driven to every joint's limit.
The open pose is the one worth having: a wrong joint origin makes a leaf swing about the wrong
line, which is invisible in a closed render and unmistakable at the stop.

`physics.png` — the drop and tilt traces behind the gate's verdict.

MuJoCo's own renderer needs a GL context, which a headless box does not have, so these are drawn
from the compiled collider meshes directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import mujoco
import numpy as np
import trimesh

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

RUN = Path("/scratch2/LiteReality-Agent/run")
OBJECTS = [
    ("MIL-Meeting-Zhening", "Wall4_Door_0", "door · 1 hinge"),
    ("fallside-kitchen-Zhening", "Dishwasher0", "dishwasher · hinge + slide"),
    ("Kitchen-Xiaoyang_Lyu", "Refrigerator0", "fridge · 4 doors"),
    ("Airbnb-Cam-Zhening", "Storage1", "storage · 6 joints"),
    ("Airbnb-Room-Zhening", "Bed0", "bed · 4 drawers"),
]
PALETTE = plt.get_cmap("tab20").colors
MOVING_OPEN = "#e8663a"
MOVING_SHUT = "#f0b8a4"


def simdir(scan: str, obj: str) -> Path:
    return RUN / scan / "scene_init/obj_stage/reconstructed_objs" / obj / "sim"


def rodrigues(points: np.ndarray, axis, angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    c, s = np.cos(angle), np.sin(angle)
    return points * c + np.cross(a, points) * s + np.outer(points @ a, a) * (1 - c)


def chain_origin(model: dict, link: str) -> np.ndarray:
    """A link's frame in object coordinates, walking its joints back to the root."""
    offset, current = np.zeros(3), link
    for _ in range(8):
        joint = next((j for j in model["joints"] if j["child"] == current), None)
        if joint is None:
            break
        offset = offset + np.asarray(joint["origin"], dtype=float)
        current = joint["parent"]
    return offset


def posed(model: dict, d: Path, opened: bool):
    """(vertices, faces, is_moving) per collider, in object coordinates."""
    out = []
    for link in model["links"]:
        joint = next((j for j in model["joints"] if j["child"] == link["name"]), None)
        base = chain_origin(model, link["name"])
        for col in link["colliders"]:
            mesh = trimesh.load(str(d / col["file"]), process=False)
            verts = np.asarray(mesh.vertices, dtype=float)
            if joint is not None and opened:
                if joint["type"] == "prismatic":
                    verts = verts + np.asarray(joint["axis"], dtype=float) * joint["limit_upper"]
                else:
                    verts = rodrigues(verts, joint["axis"], joint["limit_upper"])
            out.append((verts + base, mesh.faces, joint is not None))
    return out


def draw_colliders(out: Path) -> None:
    fig = plt.figure(figsize=(19.5, 8.6))
    for k, (scan, obj, subtitle) in enumerate(OBJECTS):
        d = simdir(scan, obj)
        model = json.loads((d / f"{obj}.physics.json").read_text())
        n = sum(len(link["colliders"]) for link in model["links"])
        for row, opened in ((0, False), (1, True)):
            ax = fig.add_subplot(2, 5, row * 5 + k + 1, projection="3d")
            corners = []
            for i, (verts, faces, moving) in enumerate(posed(model, d, opened)):
                corners.append(verts)
                if moving:
                    colour = MOVING_OPEN if opened else MOVING_SHUT
                else:
                    colour = PALETTE[(i * 3) % 20]
                ax.add_collection3d(Poly3DCollection(
                    verts[faces], alpha=.62 if moving else .42,
                    facecolor=colour, edgecolor="k", linewidths=.12))
            box = np.vstack(corners)
            low, high = box.min(0), box.max(0)
            centre, radius = (low + high) / 2, (high - low).max() / 2 * 1.02
            ax.set_xlim(centre[0] - radius, centre[0] + radius)
            ax.set_ylim(centre[1] - radius, centre[1] + radius)
            ax.set_zlim(centre[2] - radius, centre[2] + radius)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=16, azim=-62)
            ax.set_axis_off()
            if row == 0:
                ax.set_title(f"{obj}\n{subtitle}\n{n} convex colliders · "
                             f"{model['total_mass']:.1f} kg", fontsize=9.5, pad=-4)
            else:
                limits = ", ".join(f"{j['limit_upper']:.2f}" for j in model["joints"][:4])
                more = " …" if len(model["joints"]) > 4 else ""
                ax.set_title(f"open at limit  ({limits}{more})", fontsize=8.5, pad=-4)
    fig.text(.5, .975,
             "What the solver sees — convex decomposition (top, closed) and the same colliders "
             "driven to each joint's limit (bottom, moving parts in orange)",
             ha="center", fontsize=13)
    fig.subplots_adjust(left=.01, right=.99, top=.93, bottom=.01, wspace=.02, hspace=.02)
    fig.savefig(out / "colliders.png", dpi=115)
    plt.close(fig)


def draw_physics(out: Path) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 4.5))
    for k, (scan, obj, _subtitle) in enumerate(OBJECTS):
        d = simdir(scan, obj)
        model = mujoco.MjModel.from_xml_path(str(d / f"{obj}_drop.xml"))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        times, lows = [], []
        for i in range(int(1.0 / model.opt.timestep)):
            mujoco.mj_step(model, data)
            times.append(i * model.opt.timestep)
            lows.append(float(np.asarray(data.xpos[1:])[:, 2].min()))
        a1.plot(times, np.asarray(lows) * 1000, lw=1.6, color=PALETTE[k * 2], label=obj)
    a1.axhline(0, color="k", lw=.8, ls="--")
    a1.set_xlabel("time (s)")
    a1.set_ylabel("lowest body origin (mm)")
    a1.grid(alpha=.25)
    a1.set_title("Drop — released 50 mm up; all five settle within ~0.2 s", fontsize=11)
    a1.text(.42, .42,
            "resting heights differ because a body origin sits on its own joint,\n"
            "not on the floor — penetration is measured separately (max 0.03 mm)",
            transform=a1.transAxes, fontsize=7.5, color="#555")
    a1.legend(fontsize=8, frameon=False)

    names, measured, predicted = [], [], []
    for scan, obj, _subtitle in OBJECTS:
        report = json.loads((simdir(scan, obj) / f"{obj}.sim_check.json").read_text())
        tilt = report["scenarios"]["tilt"]
        names.append(obj)
        measured.append(tilt["measured_slide_deg"])
        predicted.append(tilt["expected_slide_deg"])
    x = np.arange(len(names))
    a2.bar(x - .19, predicted, .38, color="#9aa7b8",
           label="predicted from stated friction · atan(µ)")
    a2.bar(x + .19, measured, .38, color="#2f6f9f", label="measured slide angle")
    a2.set_xticks(x)
    a2.set_xticklabels(names, rotation=18, ha="right", fontsize=8.5)
    a2.set_ylabel("degrees")
    a2.grid(axis="y", alpha=.25)
    a2.legend(fontsize=8, frameon=False)
    a2.set_title("Tilt — does it slide at the angle its own friction claims?", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "physics.png", dpi=115)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).parent)
    args = ap.parse_args()
    draw_colliders(args.out)
    print("colliders.png")
    draw_physics(args.out)
    print("physics.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
