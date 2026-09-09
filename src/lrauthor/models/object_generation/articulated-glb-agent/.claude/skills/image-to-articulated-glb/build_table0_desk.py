"""Build recipe: large light-oak laminate office desk (RoomPlan category: table).
STATIC object -- no articulation (drawers modelled as fixed faces).

Reference: LiteReality .../Table0/reference_1024.png
A rectangular softened-edge oak top on a solid waterfall PANEL END at the left
and TWO 3-drawer pedestals under the right portion. Everything light oak.

Canonical frame (Blender Z-up, meters): UP=+Z, FRONT=-Y (drawer faces toward -Y),
footprint centred at X=0,Y=0, lowest point at Z=0.
Real-world overall size: X(width)=3.35, Y(depth)=0.82, Z(height)=0.74 m.

Run: blender -b --python build_table0_desk.py -- <texture_dir> <out.glb>
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
from blender_lib import (
    add_box,
    add_rounded_box,
    export_glb,
    join_parts,
    make_pbr_material,
    reset_scene,
    shade_smooth_auto,
    uv_cube_project,
)

argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
TEXDIR = argv[0] if argv else "textures"
OUT_GLB = argv[1] if len(argv) > 1 else "Table0.glb"

scene = reset_scene()

# ---------------- dimensions (meters; Z up, front faces -Y) ----------------
TOP_X, TOP_Y = 3.35, 0.82  # width (X) / depth (Y)
TOP_T = 0.045  # tabletop thickness
TOP_Z = 0.74  # top surface height (overall height)
UNDER = TOP_Z - TOP_T  # underside of top = 0.695
HX, HY = TOP_X / 2, TOP_Y / 2

# ---------------- materials (all light-oak laminate) ----------------
WOOD = make_pbr_material(
    "OakLaminate",
    diff_path=os.path.join(TEXDIR, "oak_diff_1k.jpg"),
    rough_path=os.path.join(TEXDIR, "plywood_rough_1k.jpg"),
    normal_path=os.path.join(TEXDIR, "plywood_nor_gl_1k.png"),
    normal_strength=0.30,
)
WOOD_BASE = make_pbr_material(
    "OakLaminateBase",
    diff_path=os.path.join(TEXDIR, "oak_diff_1k.jpg"),
    rough_path=os.path.join(TEXDIR, "plywood_rough_1k.jpg"),
    normal_path=os.path.join(TEXDIR, "plywood_nor_gl_1k.png"),
    normal_strength=0.30,
)

# ---------------- tabletop (rectangular, softened edges) ----------------
top = add_rounded_box(
    "Top", (TOP_X, TOP_Y, TOP_T), (0, 0, TOP_Z - TOP_T / 2), WOOD, radius=0.008, segments=3
)
uv_cube_project(top, size=1.0)
shade_smooth_auto(top, angle=40)

# ---------------- left waterfall panel end (solid side) ----------------
PANEL_T = 0.05  # panel thickness (along X)
PANEL_Y = TOP_Y - 0.03  # depth (slightly inset from top edges)
px = -HX + PANEL_T / 2
panel = add_rounded_box(
    "Left_Panel",
    (PANEL_T, PANEL_Y, UNDER),
    (px, 0.0, UNDER / 2),
    WOOD_BASE,
    radius=0.006,
    segments=2,
)
uv_cube_project(panel)
shade_smooth_auto(panel, angle=45)

# ---------------- two 3-drawer pedestals under the right portion ----------------
PED_W = 0.42  # pedestal width (X)
PED_Y0 = -HY + 0.02  # pedestal front (toward -Y)
PED_Y1 = HY - 0.04  # pedestal back
PED_DEPTH = PED_Y1 - PED_Y0
GAP = 0.012  # gap between the two pedestals
N_DRAWERS = 3
DRW_GAP = 0.008  # reveal between drawer faces
REVEAL = 0.012  # side reveal of drawer faces inside carcass
PROUD = 0.018  # drawer face stands proud of carcass front

# right edge of the right pedestal sits inboard of the top's right end
PED_RIGHT_EDGE = 1.00
ped_centers = [
    PED_RIGHT_EDGE - PED_W / 2 - (PED_W + GAP),  # left pedestal
    PED_RIGHT_EDGE - PED_W / 2,
]  # right pedestal

for pi, cx in enumerate(ped_centers):
    tag = "L" if pi == 0 else "R"
    parts = []
    # carcass body (solid box; front inset so drawer faces sit proud of it)
    body_front = PED_Y0 + PROUD
    body = add_box(
        f"ped{tag}_body",
        (PED_W, PED_Y1 - body_front, UNDER),
        (cx, (body_front + PED_Y1) / 2, UNDER / 2),
        WOOD_BASE,
    )
    parts.append(body)
    # 3 stacked drawer faces on the -Y front
    area_z0, area_z1 = 0.03, UNDER - 0.01
    dh = (area_z1 - area_z0 - (N_DRAWERS - 1) * DRW_GAP) / N_DRAWERS
    for di in range(N_DRAWERS):
        z1 = area_z1 - di * (dh + DRW_GAP)
        zc = z1 - dh / 2
        face = add_rounded_box(
            f"ped{tag}_drawer{di + 1}",
            (PED_W - 2 * REVEAL, PROUD, dh - 0.004),
            (cx, PED_Y0 + PROUD / 2, zc),
            WOOD,
            radius=0.004,
            segments=2,
        )
        parts.append(face)
    for p in parts:
        uv_cube_project(p)
        shade_smooth_auto(p, angle=45)
    ped = join_parts(parts, f"Pedestal_{tag}", (cx, 0, 0))

export_glb(os.path.abspath(OUT_GLB))
