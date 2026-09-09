"""How many convex parts does a carcass actually need, and what do they cost?

Fifty colliders on one cabinet looks like over-segmentation. It is the opposite: below a
dozen parts CoACD returns fat convex chunks that each span the cavity, so their union still
fills the hull completely and the cupboard is a solid block. This measures that directly
rather than arguing about it.

    interior kept = how much of the object's open volume survives decomposition
                    0%   = the collider is a solid block
                    100% = the collider encloses exactly the material the mesh does

    python assets/sim-ready/measure_collider_budget.py [--out assets/sim-ready]

Writes budget.json and budget.png. Needs a compiled object under run/ (see
`python -m lrauthor.models.object_generation.sim build <glb>`).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import ConvexHull

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from lrauthor.models.object_generation.sim.properties import _decompose  # noqa: E402

RUN = Path("/scratch2/LiteReality-Agent/run")
CARCASS = ("Airbnb-Cam-Zhening", "Storage1", "base_link")
OBJECTS = [("MIL-Meeting-Zhening", "Wall4_Door_0"), ("fallside-kitchen-Zhening", "Dishwasher0"),
           ("Kitchen-Xiaoyang_Lyu", "Refrigerator0"), ("Airbnb-Cam-Zhening", "Storage1"),
           ("Airbnb-Room-Zhening", "Bed0")]
BUDGETS = (1, 4, 8, 12, 16, 24)
SAMPLES = 120_000


def simdir(scan: str, obj: str) -> Path:
    return RUN / scan / "scene_init/obj_stage/reconstructed_objs" / obj / "sim"


def inside(points: np.ndarray, verts: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Convex-hull membership from the face planes — no rtree, no ray casting."""
    eq = ConvexHull(np.asarray(verts)).equations
    return np.all(points @ eq[:, :3].T + eq[:, 3] <= tol, axis=1)


def sweep() -> dict:
    scan, obj, link = CARCASS
    mesh = trimesh.load(str(simdir(scan, obj) / f"{link}_vis.obj"), process=False)
    hull = mesh.convex_hull
    material = float(mesh.volume / hull.volume)

    rng = np.random.default_rng(0)
    pts = rng.uniform(*hull.bounds, size=(SAMPLES, 3))
    pts = pts[inside(pts, hull.vertices)]

    rows = []
    for budget in BUDGETS:
        t = time.time()
        parts = [(hull.vertices, hull.faces)] if budget == 1 else _decompose(mesh, budget)
        elapsed = time.time() - t
        occupied = np.zeros(len(pts), bool)
        for verts, _faces in parts:
            try:
                occupied |= inside(pts, verts)
            except Exception:                                # noqa: BLE001 — degenerate part
                continue
        filled = float(occupied.mean())
        rows.append({"budget": budget, "parts": len(parts), "seconds": round(elapsed, 1),
                     "filled_fraction": round(filled, 4),
                     "interior_kept": round((1 - filled) / (1 - material), 4)})
    return {"object": f"{obj}/{link}", "mesh_volume": round(float(mesh.volume), 4),
            "hull_volume": round(float(hull.volume), 4),
            "material_fraction": round(material, 4), "samples": int(len(pts)), "rows": rows}


def cost() -> list:
    import mujoco

    out = []
    for scan, obj in OBJECTS:
        d = simdir(scan, obj)
        model_json = json.loads((d / f"{obj}.physics.json").read_text())
        n = sum(len(link["colliders"]) for link in model_json["links"])
        mo = mujoco.MjModel.from_xml_path(str(d / f"{obj}_drop.xml"))
        da = mujoco.MjData(mo)
        mujoco.mj_forward(mo, da)
        for _ in range(400):                                 # settle before timing
            mujoco.mj_step(mo, da)
        contacts, t = [], time.perf_counter()
        for _ in range(1000):
            mujoco.mj_step(mo, da)
            contacts.append(da.ncon)
        ms = (time.perf_counter() - t)
        volumes = sorted(c["volume"] for link in model_json["links"] for c in link["colliders"])
        total = sum(volumes) or 1.0
        slivers = [v for v in volumes if v < total * 0.005]
        out.append({"object": obj, "colliders": n, "mean_contacts": round(float(np.mean(contacts)), 1),
                    "ms_per_step": round(ms, 4),
                    "times_realtime": round(mo.opt.timestep / (ms / 1000), 0),
                    "slivers_under_half_percent": len(slivers),
                    "sliver_volume_share": round(sum(slivers) / total, 4)})
    return out


def plot(data: dict, costs: list, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.3))
    rows = data["rows"]
    x = [r["parts"] for r in rows]
    a1.plot(x, [r["interior_kept"] * 100 for r in rows], "o-", color="#2f6f9f", lw=2)
    a1.axhspan(0, 2, color="#c0392b", alpha=.13)
    a1.text(5.5, 6, "solid block — the cupboard has no inside", fontsize=8.5, color="#8c2f24")
    a1.set_xlabel("convex parts")
    a1.set_ylabel("open interior kept (%)")
    a1.set_title(f"{data['object']} — real material is "
                 f"{data['material_fraction'] * 100:.0f}% of its hull", fontsize=10.5)
    a1.grid(alpha=.25)
    a1.set_ylim(-3, 100)

    names = [c["object"] for c in costs]
    a2.bar(names, [c["ms_per_step"] for c in costs], color="#2f6f9f")
    for i, c in enumerate(costs):
        a2.text(i, c["ms_per_step"], f"  {c['colliders']} parts\n  {c['mean_contacts']:.0f} contacts",
                ha="center", va="bottom", fontsize=7.5)
    a2.set_ylabel("ms per solver step")
    a2.set_ylim(0, max(c["ms_per_step"] for c in costs) * 1.7)
    a2.set_xticklabels(names, rotation=18, ha="right", fontsize=8.5)
    a2.set_title("Cost — part count is not what drives contact work", fontsize=10.5)
    a2.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(out / "budget.png", dpi=115)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).parent)
    args = ap.parse_args()
    data = sweep()
    costs = cost()
    (args.out / "budget.json").write_text(json.dumps({"sweep": data, "cost": costs}, indent=2) + "\n")
    plot(data, costs, args.out)
    for r in data["rows"]:
        print(f"  {r['parts']:>3} parts  filled {r['filled_fraction'] * 100:5.1f}%  "
              f"interior kept {r['interior_kept'] * 100:5.1f}%")
    for c in costs:
        print(f"  {c['object']:16s} {c['colliders']:>3} colliders  {c['mean_contacts']:>5.1f} contacts  "
              f"{c['ms_per_step']:.3f} ms/step  {c['slivers_under_half_percent']} slivers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
