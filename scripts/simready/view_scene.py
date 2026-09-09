"""Open a LiteReality MuJoCo scene interactively.

MuJoCo's managed viewer (`python -m mujoco.viewer`) crashes constructing its window on this macOS,
but the PASSIVE viewer works — so this owns the loop instead of handing it to MuJoCo.

    ./.venv/bin/mjpython scripts/simready/view_scene.py [scene.xml]             # frozen: look, don't touch
    ./.venv/bin/mjpython scripts/simready/view_scene.py [scene.xml] --physics   # gravity on, things can be pushed

Frozen is the default because it answers a different question: it shows the room exactly as the
exporter wrote it, with nothing settled, nudged or knocked over. Nothing is stepped, so a drag
cannot move anything and no contact force is ever resolved.
"""
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer

DEFAULT = "run/Office-Elliott/realism_authoring/mujoco/scene.xml"
args = [a for a in sys.argv[1:] if not a.startswith("-")]
physics = "--physics" in sys.argv
scene = Path(args[0]) if args else Path(DEFAULT)

model = mujoco.MjModel.from_xml_path(str(scene))
data = mujoco.MjData(model)
drives = [j for j in (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                      for n in ("room_x", "room_y", "room_z", "room_yaw")) if j >= 0]

mujoco.mj_forward(model, data)
print(f"{scene}\n  {model.nbody} bodies · {model.ngeom} geoms · {model.njnt} joints")
print(f"  mode: {'PHYSICS — gravity on, bodies can be pushed' if physics else 'FROZEN — nothing is stepped, nothing can move'}")
print("  drag to orbit · scroll to zoom · double-click a body to select and name it")
print("  in the window: press F1 for help, Tab for the panel, ctrl-A to see contact forces")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t0 = time.time()
        if physics:
            for j in drives:
                data.qpos[model.jnt_qposadr[j]] = 0.0
                data.qvel[model.jnt_dofadr[j]] = 0.0
            mujoco.mj_step(model, data)
        else:
            # Kinematics only. Poses stay exactly as exported; no force is ever integrated, so a
            # mouse drag cannot displace anything and the scene cannot settle out from under you.
            data.xfrc_applied[:] = 0.0
            data.qacc[:] = 0.0
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
        viewer.sync()
        wait = model.opt.timestep - (time.time() - t0)
        if wait > 0:
            time.sleep(wait)
