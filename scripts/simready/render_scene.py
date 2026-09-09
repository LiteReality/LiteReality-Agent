#!/usr/bin/env python3
"""render_scene.py — look at a room that is actually being simulated.

    <python> render_scene.py <mujoco/scene.xml> --out <dir> [--settle 2.0]

Not a picture of the geometry: the scene is STEPPED first, so what comes out is the room as the
physics engine holds it rather than as the exporter wrote it. If a shelf is not really under the
mug, this is where you see the mug on the floor.

Four views, because no single one answers the question:

``eye``        standing in the room at 1.68 m. The only view that says whether it looks like the
               place. Walls and ceiling are kept, because they are most of what you see.
``overview``   the same corner, just under the ceiling.
``dollhouse``  from outside and above, walls and ceiling hidden. Shows the whole plan and every
               object's footprint at once — the view that shows something standing in mid-air.
``topdown``    straight down. Best coverage of the floor, and the honest check on layout.

The cutaway is a VIEW, not a change to the model: the walls stay in the scene and keep colliding,
they are simply not drawn. MuJoCo's default `geomgroup` hides groups 3-5, so the walls (group 1)
and ceiling (group 0) are turned off explicitly rather than by moving them.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")   # headless: no display on this machine

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

WIDTH, HEIGHT = 1280, 960
# What each view hides. The room body's own walls are group 1 and the ceiling slab group 0; the
# collision geoms are group 3 and are never drawn.
CUTAWAY = {0, 1}
VIEWS = (("eye", frozenset()), ("overview", frozenset()),
         ("dollhouse", frozenset(CUTAWAY)), ("topdown", frozenset({0})))


def _pin_room(model, data) -> None:
    """Hold the room's drive joints at zero.

    The room is a 50 t body on position servos, so with no controls set it sags under its own
    weight for the first few hundred steps and the whole building drifts down through the render.
    """
    for name in ("room_x", "room_y", "room_z", "room_yaw"):
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint >= 0:
            data.qpos[model.jnt_qposadr[joint]] = 0.0
            data.qvel[model.jnt_dofadr[joint]] = 0.0


def _ceiling_cut(model, data) -> float:
    """The height above which a doll's house view should draw nothing.

    Hiding the ceiling SLAB is not enough: the suspended grid, the downlights and the service
    boxing are authored fixtures in their own right, so they go on being drawn and a plan view
    ends up looking through a lattice. The exporter puts its `topdown` camera 150 mm under the
    ceiling, which is the one number in the file that says where the ceiling is.
    """
    camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "topdown")
    if camera >= 0:
        return float(model.cam_pos[camera][2]) - 0.30
    return float(np.max(data.geom_xpos[:, 2])) - 0.30 if model.ngeom else 1e9


def _geom_floor(model, data, geom: int) -> float:
    """The lowest point a geom reaches, in world z.

    Not `geom_rbound`, which is the radius of the geom's bounding SPHERE: a suspended ceiling grid
    is a single wide, flat mesh, so its sphere reaches most of the way to the floor and the geom
    tests as low-lying however high it hangs. The grid survived the cutaway for exactly that
    reason. `geom_aabb` is the box, and its eight corners carried through the geom's own rotation
    are where the thing actually is.
    """
    centre = model.geom_aabb[geom][:3]
    half = model.geom_aabb[geom][3:]
    corners = np.array([[centre[0] + sx * half[0], centre[1] + sy * half[1], centre[2] + sz * half[2]]
                        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    world = corners @ np.asarray(data.geom_xmat[geom]).reshape(3, 3).T + data.geom_xpos[geom]
    return float(world[:, 2].min())


def render(scene: Path, out: Path, *, settle: float = 2.0, views=VIEWS,
           hide_above: float | None = None) -> list[Path]:
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    moved = {}
    if settle > 0:
        start = data.xpos.copy()
        for _ in range(int(settle / model.opt.timestep)):
            _pin_room(model, data)
            mujoco.mj_step(model, data)
        moved = {b: float(np.linalg.norm(data.xpos[b] - start[b])) for b in range(1, model.nbody)}

    # Everything at ceiling height, remembered so it can be put back between views. Moving a geom
    # into a group nothing draws is a change to the VIEW; the geometry stays in the model and goes
    # on colliding, which is the whole difference between a cutaway and a room with no ceiling.
    cut = _ceiling_cut(model, data) if hide_above is None else hide_above
    original = model.geom_group.copy()
    high = [g for g in range(model.ngeom) if _geom_floor(model, data, g) > cut]

    out.mkdir(parents=True, exist_ok=True)
    written = []
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    for name, hidden in views:
        camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if camera < 0:
            continue
        options = mujoco.MjvOption()
        mujoco.mjv_defaultOption(options)
        for group in hidden:
            options.geomgroup[group] = 0
        model.geom_group[:] = original
        if hidden:                       # a cutaway view: lose the ceiling fittings as well
            for g in high:
                model.geom_group[g] = 5
        renderer.update_scene(data, camera=camera, scene_option=options)
        path = out / f"{name}.png"
        _save(renderer.render(), path)
        written.append(path)
    renderer.close()
    model.geom_group[:] = original

    if moved:
        worst = sorted(moved.items(), key=lambda kv: -kv[1])[:5]
        names = [(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "?", v) for b, v in worst]
        print(f"  settled {settle:.1f}s — worst movers: "
              + ", ".join(f"{n} {v * 1000:.0f}mm" for n, v in names))
    return written


def _save(pixels, path: Path) -> None:
    from PIL import Image

    Image.fromarray(pixels).save(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("scene", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--settle", type=float, default=2.0,
                    help="seconds of gravity before rendering (0 shows the export as written)")
    ap.add_argument("--hide-above", type=float, default=None,
                    help="cutaway views draw nothing above this z (default: just under the ceiling)")
    a = ap.parse_args(argv)
    for path in render(a.scene, a.out, settle=a.settle, hide_above=a.hide_above):
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
