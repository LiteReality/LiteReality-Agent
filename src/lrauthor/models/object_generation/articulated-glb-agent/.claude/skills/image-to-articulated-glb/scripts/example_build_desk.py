"""Example build script (template): office desk with a fixed 3-drawer
pedestal and a mobile 2-drawer pedestal on casters, from a reference photo.

Copy this file as the starting point for new objects; keep the structure:
materials -> static body -> carcasses -> drawers/doors -> articulation tags
-> animation -> export.

Run: blender -b --python example_build_desk.py -- <texture_dir> <out.glb>
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blender_lib import (
    add_box,
    add_cylinder,
    animate_prismatic,
    build_carcass,
    build_drawer,
    export_glb,
    join_parts,
    make_pbr_material,
    make_plain_material,
    parent_keep_world,
    reset_scene,
    set_articulation,
    uv_cube_project,
)

argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
TEXDIR = argv[0] if argv else "textures"
OUT_GLB = argv[1] if len(argv) > 1 else "desk.glb"

scene = reset_scene()

# ---------- materials ----------
WOOD = make_pbr_material(
    "BeechWood",
    diff_path=os.path.join(TEXDIR, "plywood_diff_1k.jpg"),
    rough_path=os.path.join(TEXDIR, "plywood_rough_1k.jpg"),
    normal_path=os.path.join(TEXDIR, "plywood_nor_gl_1k.png"),
    tint=(1.0, 0.82, 0.62),
    tint_fac=0.45,  # warm beech tone
)
HANDLE = make_plain_material("DarkHandle", (0.16, 0.16, 0.18), 0.4, 0.45)
PLASTIC = make_plain_material("BlackPlastic", (0.05, 0.05, 0.05), 0.0, 0.65)

# ---------- desk body (static) ----------
TOP_Z, T = 0.74, 0.03
DESK_L, DESK_D = 2.4, 0.8
parts = [
    add_box("top", (DESK_L, DESK_D, T), (0, 0, TOP_Z - T / 2), WOOD),
    add_box(
        "panel_l", (T, DESK_D - 0.02, TOP_Z - T), (-DESK_L / 2 + T / 2, 0, (TOP_Z - T) / 2), WOOD
    ),
    add_box(
        "panel_r", (T, DESK_D - 0.02, TOP_Z - T), (DESK_L / 2 - T / 2, 0, (TOP_Z - T) / 2), WOOD
    ),
]
for p in parts:
    uv_cube_project(p)
desk = join_parts(parts, "Desk_Body", (0, 0, 0))

# cable grommets, flush with the desktop
add_cylinder("Grommet_L", 0.040, 0.034, (-0.85, 0.25, TOP_Z - 0.015), PLASTIC)
add_cylinder("Grommet_R", 0.040, 0.034, (0.62, 0.25, TOP_Z - 0.015), PLASTIC)

# ---------- fixed pedestal: 3 drawers, suspended under the top ----------
FP_X0, FP_X1, FP_YF, FP_YB = -0.62, -0.17, -0.37, 0.27
FP_Z0, FP_Z1 = 0.19, TOP_Z - T
ped_fixed = build_carcass("Pedestal_Fixed", FP_X0, FP_X1, FP_YF, FP_YB, FP_Z0, FP_Z1, WOOD)
parent_keep_world(ped_fixed, desk)

fixed_drawers = []
n, gap = 3, 0.006
dh = (FP_Z1 - FP_Z0 - 0.03 - (n - 1) * gap) / n
for i in range(n):
    z1 = FP_Z1 - 0.015 - i * (dh + gap)
    d = build_drawer(
        f"Drawer_Fixed_{i + 1}",
        FP_X0 + 0.015,
        FP_X1 - 0.015,
        FP_YF,
        0.55,
        z1 - dh,
        z1,
        WOOD,
        HANDLE,
    )
    parent_keep_world(d, ped_fixed)
    fixed_drawers.append(d)

# ---------- mobile pedestal: 2 drawers + casters ----------
MP_X0, MP_X1, MP_YF, MP_YB = 0.42, 0.84, -0.34, 0.23
MP_Z0, MP_Z1 = 0.08, 0.62
ped_mob = build_carcass("Pedestal_Mobile", MP_X0, MP_X1, MP_YF, MP_YB, MP_Z0, MP_Z1, WOOD)

for ix, x in enumerate((MP_X0 + 0.05, MP_X1 - 0.05)):
    for iy, y in enumerate((MP_YF + 0.06, MP_YB - 0.06)):
        w = add_cylinder(
            f"Caster_{ix}{iy}", 0.028, 0.024, (x, y, 0.032), PLASTIC, rot=(0, math.pi / 2, 0)
        )
        m = add_box(f"CasterMount_{ix}{iy}", (0.02, 0.02, 0.03), (x, y, 0.066), PLASTIC)
        parent_keep_world(w, ped_mob)
        parent_keep_world(m, ped_mob)

mz_top, d1_h = MP_Z1 - 0.015, 0.17
d1 = build_drawer(
    "Drawer_Mobile_Top",
    MP_X0 + 0.015,
    MP_X1 - 0.015,
    MP_YF,
    0.48,
    mz_top - d1_h,
    mz_top,
    WOOD,
    HANDLE,
)
d2 = build_drawer(
    "Drawer_Mobile_File",
    MP_X0 + 0.015,
    MP_X1 - 0.015,
    MP_YF,
    0.48,
    MP_Z0 + 0.015,
    mz_top - d1_h - gap,
    WOOD,
    HANDLE,
)
parent_keep_world(d1, ped_mob)
parent_keep_world(d2, ped_mob)

# ---------- articulation tags + slide animation ----------
items = [(d, 0.42) for d in fixed_drawers] + [(d1, 0.38), (d2, 0.38)]
for ob, ext in items:
    set_articulation(ob, "prismatic", (0, -1, 0), 0.0, ext)
animate_prismatic(scene, items)

export_glb(os.path.abspath(OUT_GLB))
