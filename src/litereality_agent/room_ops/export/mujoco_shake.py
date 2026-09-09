#!/usr/bin/env python3
"""mujoco_shake.py — shake the exported room and see what actually moves.

    python -m litereality_agent.room_ops.export.mujoco_shake --scene <mujoco/scene.xml> [--video out.mp4]

A room that LOADS in MuJoCo has proved almost nothing. The interesting failures all pass loading:
every prop welded into the walls, a table whose convex hull swallows its own chairs, a mug authored
two centimetres inside the desk it "rests on", a hundred bodies that never had a joint and so can
never move. Each of those is a still image of a room that looks perfect and a simulation that is
inert or explodes.

So the check is a shake. Gravity is swung horizontally for a few seconds — the whole room feels a
lateral acceleration, exactly as if the building were moving — and then released. Three numbers
come out of that, and they are the ones worth having:

* **settle** — how far things drift with NO shaking. A scene that is already at rest reports
  millimetres. Centimetres mean objects were authored floating and are falling; a big number means
  something is interpenetrating and being pushed apart on the first step.
* **shake** — how much the shaking actually moved things. Near zero means the bodies are not free
  at all, whatever the XML says.
* **what stayed put** — the walls, the sockets, the skirting. If those move, something that should
  be static was emitted with a joint.

The video is the other half. Numbers say the scene is dynamic; watching a desk's clutter slide and
topple says it is dynamic in the way a room is.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def _free_bodies(model) -> list[tuple[int, str, int]]:
    """(body id, name, qpos address) for every body with a free joint."""
    import mujoco

    out = []
    for b in range(model.nbody):
        if model.body_jntnum[b] != 1:
            continue
        j = model.body_jntadr[b]
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"
        out.append((b, name, model.jnt_qposadr[j]))
    return out


def _pick_camera(model) -> int:
    """A camera that can actually see the room. -1 (the free camera) usually cannot.

    The exported scene carries every capture camera, and the useful one is whichever sees the most
    of the room — approximated by the camera furthest from the walls it is looking at, which in a
    hand-held scan is a corner shot rather than a close-up of a desk.
    """

    best, best_score = -1, -1.0
    for c in range(model.ncam):
        pos = model.cam_pos[c]
        score = float(np.linalg.norm(pos - model.stat.center))
        if score > best_score:
            best, best_score = c, score
    return best


def _lift_out_of_walls(model, data, *, limit: float = 0.15) -> dict:
    """Move any free body that STRADDLES a wall back to the side it mostly lies on.

    The relax pass below can separate objects that merely touch, but it is helpless against a body
    that passes clean THROUGH a wall: an interior partition is 80 mm thick and a floor cabinet is
    700 mm deep, so contacts on the two faces push in exactly opposite directions and cancel. The
    cabinet then sits in a stable equilibrium with half its geometry inside the plaster, and no
    amount of relaxation time or damping changes it — measured at 1.2 s and 4 s, at every damping
    from 0.55 to 0.99, the result was identical to the millimetre.

    So this is geometric, not physical. Each wall is a box, and its thinnest axis is the one through
    the plaster; a body's vertices projected onto that axis say which side it belongs to, and it is
    translated along that axis until the last vertex clears the face. The shift is capped, because a
    body needing more than `limit` is not slightly misplaced — it is somewhere else entirely, and
    quietly teleporting it would hide an authoring fault rather than report one.
    """
    import mujoco

    walls = []
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if name.lower().startswith("wall") and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            walls.append(g)
    free_adr = {}
    for b in range(model.nbody):
        for j in range(int(model.body_jntadr[b]), int(model.body_jntadr[b]) + int(model.body_jntnum[b])):
            if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
                free_adr[b] = int(model.jnt_qposadr[j])
    if not walls or not free_adr:
        return {}

    def collision_verts(b):
        pts = []
        for g in range(int(model.body_geomadr[b]), int(model.body_geomadr[b]) + int(model.body_geomnum[b])):
            if model.geom_group[g] != 3 or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mid = int(model.geom_dataid[g])
            a, n = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
            v = model.mesh_vert[a:a + n].reshape(-1, 3)
            pts.append(v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])
        return np.vstack(pts) if pts else None

    lifted: dict[str, float] = {}

    def _stand_on_fixtures():
        """Raise a free body that starts INSIDE a fixture until it sits on top of it.

        A shelf unit's convex decomposition fills its own shelves: the hulls that approximate an
        open carcass close it up, so a mug authored ON a shelf is geometrically INSIDE the collider.
        MuJoCo then ejects it and it falls the height of the desk — fallside-office drifted 72 cm
        with nobody shaking it, and half its free bodies never came to rest at all.

        The direction is UP, not the contact normal. These objects are meant to be supported, and
        the shortest way out of a hull is usually sideways, which slides a keyboard off the desk it
        belongs on. Lifting it clear and letting gravity bring it back down puts it where the author
        meant it to be, on top of whatever it was standing in.
        """
        raised = 0
        for _ in range(12):
            mujoco.mj_forward(model, data)
            worst = {}
            for c in range(data.ncon):
                con = data.contact[c]
                if con.dist >= -0.004:
                    continue
                b1, b2 = int(model.geom_bodyid[con.geom1]), int(model.geom_bodyid[con.geom2])
                for body, other in ((b1, b2), (b2, b1)):
                    if body in free_adr and other not in free_adr:
                        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other) or ""
                        if name in ("room", "world"):
                            continue        # walls are handled above, with a proper thin axis
                        if body not in worst or con.dist < worst[body]:
                            worst[body] = con.dist
            if not worst:
                break
            for body, dist in worst.items():
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or str(body)
                if lifted.get(name, 0.0) >= limit:
                    continue
                adr = free_adr[body]
                data.qpos[adr + 2] += -dist + 0.002
                lifted[name] = round(lifted.get(name, 0.0) + float(-dist), 4)
                raised += 1
        return raised

    for _ in range(4):
        mujoco.mj_forward(model, data)
        worst = None
        for b in sorted(free_adr):
            verts = collision_verts(b)
            if verts is None:
                continue
            for w in walls:
                rot = data.geom_xmat[w].reshape(3, 3)
                local = (verts - data.geom_xpos[w]) @ rot
                size = model.geom_size[w]
                if int(np.all(np.abs(local) <= size, axis=1).sum()) < 3:
                    continue
                thin = int(np.argmin(size))
                along = local[:, thin]
                side = 1.0 if float(along.mean()) > 0 else -1.0
                clear = (size[thin] + 0.004) - (float(along.min()) if side > 0 else -float(along.max()))
                if clear <= 0:
                    continue
                if worst is None or clear > worst[0]:
                    worst = (clear, b, rot[:, thin] * side)
        if worst is None:
            break
        clear, b, direction = worst
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or str(b)
        if clear > limit:
            lifted[name] = -round(float(clear), 4)   # negative: reported, deliberately NOT moved
            break
        adr = free_adr[b]
        data.qpos[adr:adr + 3] += direction * clear
        lifted[name] = round(lifted.get(name, 0.0) + float(clear), 4)
    _stand_on_fixtures()
    mujoco.mj_forward(model, data)
    return lifted


def run(scene: Path, *, video: Path | None = None, camera: int | str | None = None,
        settle_s: float = 1.0,
        shake_s: float = 3.0, calm_s: float = 2.0, amplitude: float = 6.0,
        # 1280x960 is what the model's offscreen framebuffer is sized for; rendering 960x720 into
        # it threw away a third of the resolution that was already being paid for.
        frequency: float = 2.2, fps: int = 30, width: int = 1280, height: int = 960,
        hide_groups: tuple[int, ...] = (), ajar: float = 0.0,
        # A PICTURE HOOK IS NOT A BOLT. Measured against the real constraint force: at 1.6x the
        # frame's own weight nothing came down at any amplitude, at 0.2 they fell in a tremor you
        # could sleep through. 0.4 leaves a still room untouched, survives a gentle shake, and drops
        # every frame in a real one — which is what a nail through a plasterboard wall does.
        hold_ratio: float = 0.4,
        relax_s: float = 1.2) -> dict:
    """Settle, shake, let it calm. Returns a report; writes a video if asked."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    free = _free_bodies(model)
    dt = model.opt.timestep
    g = float(abs(model.opt.gravity[2])) or 9.81

    room_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "room")
    drives = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"drive_room_{a}")
              for a in "xyz"]
    drives = [d for d in drives if d >= 0]
    yaw_drive = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "drive_room_yaw")
    mujoco.mj_forward(model, data)
    # WHAT THE ROOM WAS HANDED IN. A settle drift only says objects moved; this says whether they
    # were overlapping before anything ran. A room authored under the `simulation` profile reports
    # zero here because that brief required `check_collisions`; MIL-Meeting, authored before it
    # existed, starts with 28 overlaps — chairs 2.8 cm inside a projection screen — and MuJoCo
    # resolves those by pushing things apart, which reads as the simulation misbehaving when it is
    # in fact the room being unphysical to begin with.
    def _penetrations():
        out = []
        for i in range(data.ncon):
            c = data.contact[i]
            if c.dist < -0.001:
                out.append((float(c.dist),
                            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                              model.geom_bodyid[c.geom1]) or "?",
                            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                              model.geom_bodyid[c.geom2]) or "?"))
        return sorted(out)

    if ajar:
        for j in range(model.njnt):
            if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE):
                lo, hi = model.jnt_range[j]
                data.qpos[model.jnt_qposadr[j]] = lo + ajar * (hi - lo)
        mujoco.mj_forward(model, data)
        start = data.xpos.copy()

    # `int(...)`: `jnt_type` is a numpy int32 and `mjtJoint` is a pybind enum, so `x in (ENUM_A,
    # ENUM_B)` is quietly False for every joint. It raises nothing — the list simply comes back
    # empty and every articulated joint in the room reports a swing of zero.
    moving_kinds = {int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)}

    # AFTER opening the joints, not before. `--ajar` snaps a drawer 30% out in a single step, and a
    # drawer with a table in front of it lands 3 cm inside that table — an overlap this code
    # created and then did not count, because the census ran first. Opening the room and then
    # measuring it means the relaxation below pushes those drawers back to where they actually fit.
    initial_overlaps = _penetrations()

    # HANGINGS FAIL UNDER LOAD. Each picture is welded to the room, and the weld is switched off
    # the moment the inertial pull on it exceeds what a picture hook holds: F = m * |a|, with the
    # acceleration the one the room is actually being driven at. That is a real criterion rather
    # than a timer — a heavy mirror comes down before a small photo, and in a gentle tremor nothing
    # does. Once broken it stays broken: a nail does not re-seat itself.
    # WELD **OR** CONNECT. A hanging used to be welded to the room; it is now held by a `connect`,
    # a point constraint that lets the picture swing on its nail instead of being clamped flat.
    # Matching on the weld type alone silently found none of them, and every picture in the room
    # became unbreakable — the failure looked like a physics result rather than a filter that had
    # stopped matching. The name is what identifies a hanging; the constraint type is an
    # implementation detail of how it is held.
    holds = (int(mujoco.mjtEq.mjEQ_WELD), int(mujoco.mjtEq.mjEQ_CONNECT))
    welds = [e for e in range(model.neq)
             if int(model.eq_type[e]) in holds
             and (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, e) or "").startswith("hang_")]
    # A HOOK HOLDS A FEW TIMES WHAT HANGS ON IT — not a fixed number of newtons. The threshold used
    # to be a flat 22 N for everything, which is a wall bolt, not a picture hook: a 0.31 kg mirror
    # pulls 1.9 N at this shake and a 1.22 kg picture 7.3 N, so NOTHING could ever come down however
    # hard the room was driven, and the stillness read as a physics result. Scaling the hold to each
    # frame's own weight is what a hook actually does, and it scales with the object: a heavy mirror
    # needs more force to shift but also weighs more, so both are equally precarious.
    weld_limit = {}
    for e in welds:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, e) or ""
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name[len("hang_"):])
        mass = float(model.body_mass[body]) if body >= 0 else 1.0
        weld_limit[e] = max(hold_ratio * mass * g, 0.5)
    pull_ema: dict[int, float] = {}
    pull_base: dict[int, float] = {}
    ema_alpha = dt / (dt + 0.02)
    fallen: list[str] = []
    # a = A(2*pi*f)^2 -> the displacement that delivers the requested lateral acceleration
    amp_m = amplitude / (2.0 * math.pi * frequency) ** 2
    start = data.xpos.copy()

    cam_id = camera if isinstance(camera, int) else (
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera) if camera else _pick_camera(model))
    renderer = None
    frames_dir = None
    # A cutaway is a rendering choice, never a model change: the walls stay in the physics and keep
    # colliding, they are simply not drawn. Hiding them by deleting geoms would be simulating a
    # different room from the one being watched.
    scene_option = mujoco.MjvOption()
    for g in hide_groups:
        if 0 <= g < len(scene_option.geomgroup):
            scene_option.geomgroup[g] = 0
    if video:
        renderer = mujoco.Renderer(model, height=height, width=width)
        frames_dir = Path(tempfile.mkdtemp())
    every = max(1, int(round(1.0 / (fps * dt))))

    def phase(name, seconds, shaking):
        nonlocal frame_no
        for step in range(int(seconds / dt)):
            if shaking:
                t = step * dt
                if drives:
                    # MOVE THE BUILDING. Displacement, not gravity: a = A(2*pi*f)^2, so the
                    # requested acceleration is converted to the amplitude that produces it. The
                    # contents then slide because the floor moved under them, which is what an
                    # earthquake is; tilting gravity instead leaves the walls standing still and
                    # looks like the room's contents deciding to migrate.
                    #
                    # THE VERTICAL TERM IS NOT A GARNISH. A chair does not slide because the floor is
                    # dragged sideways under it; it slides when the floor drops away and unloads its
                    # feet. Friction 0.55 breaks traction at mu*g = 5.4 m/s^2 and this drive peaks at
                    # 3.0, so a level shake never breaks it — every chair rides with the floor and
                    # reads as glued to it. A vertical term of comparable size modulates the normal
                    # force instead of fighting it, and the feet let go at the top of each cycle. It
                    # is how furniture actually walks across a room in a quake, it costs no extra
                    # violence in the horizontal, and it needs no fudging of the friction.
                    targets = (amp_m * math.sin(2 * math.pi * frequency * t),
                               amp_m * math.sin(2 * math.pi * frequency * 1.41 * t + 1.1),
                               0.9 * amp_m * math.sin(2 * math.pi * frequency * 1.93 * t + 2.3))
                    for act, value in zip(drives, targets):
                        data.ctrl[act] = value
                    if yaw_drive >= 0:
                        # A twist about the vertical, at a fourth incommensurate frequency. Its
                        # amplitude is set so a point 2 m from the centre sees roughly the same
                        # acceleration as the sliding axes deliver — enough to make where an object
                        # STANDS matter, which pure translation never does.
                        data.ctrl[yaw_drive] = (amp_m / 2.0) * math.sin(
                            2 * math.pi * frequency * 0.77 * t + 0.6)
                # Three INCOMMENSURATE frequencies, one per axis. A single frequency in x and y
                # traces a smooth ellipse: every object gets the same push at the same moment and
                # the room sways as one piece, which looks like a camera move rather than a shake.
                # Ratios that never repeat make the direction wander, so objects load against each
                # other and topple instead of sliding in convoy. The vertical term is deliberately
                # weaker than gravity — it should unstick things, not throw them at the ceiling.
            elif drives:
                for act in drives:
                    data.ctrl[act] = 0.0
                if yaw_drive >= 0:
                    data.ctrl[yaw_drive] = 0.0
            if welds and name != "settle":
                # THE FORCE THE NAIL IS ACTUALLY CARRYING, not a guess from the drive signal.
                # `m * |a_commanded|` describes how hard the ROOM is being pushed and says nothing
                # about the picture: a frame that has begun to swing loads its hook far harder than
                # its own mass times the room's acceleration, and one hanging in a sheltered corner
                # far less. MuJoCo already solves for the constraint force every step — a `connect`
                # contributes three rows, and their norm IS the pull on the nail. Checking it also
                # lets a picture come off during the calm, while it is still swinging, which is when
                # a real one usually goes.
                for e in list(welds):
                    if not data.eq_active[e]:
                        continue
                    rows = [i for i in range(data.nefc)
                            if int(data.efc_type[i]) == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
                            and int(data.efc_id[i]) == e]
                    if not rows:
                        continue
                    # SMOOTHED, NOT INSTANTANEOUS. A constraint solver produces single-step spikes
                    # when it first catches a body — Picture0 peaked at 2718 N, 226x its own weight,
                    # while the room was standing perfectly still — and breaking on the raw value
                    # dropped two pictures off a wall that was never shaken. A hook fails under load
                    # it is actually carrying, so the pull is low-passed over ~20 ms: a real overload
                    # lasts, a solver artefact does not survive the filter.
                    pull = float(np.linalg.norm(data.efc_force[rows]))
                    smooth = pull_ema.get(e)
                    smooth = pull if smooth is None else smooth + ema_alpha * (pull - smooth)
                    pull_ema[e] = smooth
                    # AGAINST WHAT IT WAS ALREADY CARRYING. Some pictures are authored INSIDE the
                    # rail they hang on — Picture1 starts 23.6 mm into PictureRail4 — so the rail
                    # shoves and the nail holds it back at 69x the frame's own weight before anything
                    # has moved. Judged on the absolute pull those two dropped off a wall that was
                    # standing still, which says nothing about the hook and everything about the
                    # authoring. A hook fails on the load ADDED to whatever preload it already sits
                    # under, so the baseline is taken the moment the shaking starts.
                    if e not in pull_base:
                        pull_base[e] = smooth
                    if smooth - pull_base[e] > weld_limit[e]:
                        data.eq_active[e] = 0
                        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, e) or ""
                        fallen.append(name[len("hang_"):])
                        welds.remove(e)
            mujoco.mj_step(model, data)
            if hinges:
                hinge_track.append([float(data.qpos[model.jnt_qposadr[j]]) for j in hinges])
            if renderer is not None and step % every == 0:
                renderer.update_scene(data, camera=cam_id, scene_option=scene_option)
                import PIL.Image
                PIL.Image.fromarray(renderer.render()).save(frames_dir / f"f_{frame_no:05d}.png")
                frame_no += 1

    hinge_track: list[float] = []
    # A CLOSED door cannot respond to anything: it starts hard against its own limit stop, wedged
    # in the frame, and a shake can only rattle it. Reporting that as "not responsive" would be
    # measuring a latch, not the joint. Setting each articulated joint part-open gives the shake
    # something to act on — which is also the more useful demonstration, since the question is
    # whether the ARTICULATION works, not whether shut doors stay shut.
    # HINGES ONLY. A prismatic joint in a room is usually something that CARRIES things — a desk's
    # lift top, a drawer — and opening it a third of the way lifts a laptop, a keyboard and a mug
    # 8.8 cm into the air before the clock starts. That measured as 88 cm of "settle drift" and was
    # entirely self-inflicted. A door ajar disturbs nothing; a desk raised does.
    hinges = [j for j in range(model.njnt) if int(model.jnt_type[j]) in moving_kinds]
    # RELAX FIRST. Rooms authored before the `simulation` profile ship with objects already inside
    # each other — Airbnb-Cam starts with 87 overlapping contacts, a chair 2.6 cm inside a table —
    # and MuJoCo's first instinct is to resolve that penetration by firing them apart. Stepping
    # briefly with NO gravity and bleeding off velocity each step lets the same overlaps separate
    # gently instead: the contact still pushes, nothing accelerates far, and by the time gravity is
    # switched on the room is merely untidy rather than exploding. It cannot fix a room that was
    # authored wrong; it stops that wrongness becoming a launch.
    frame_no = 0
    lifted = _lift_out_of_walls(model, data)
    if initial_overlaps:
        saved_gravity = model.opt.gravity.copy()
        model.opt.gravity[:] = 0.0
        for _ in range(int(relax_s / dt)):
            mujoco.mj_step(model, data)
            data.qvel[:] *= 0.55
        model.opt.gravity[:] = saved_gravity
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        start = data.xpos.copy()          # the room as the simulation actually begins
        relaxed_overlaps = len(_penetrations())
    else:
        relaxed_overlaps = 0

    phase("settle", settle_s, False)
    settled = data.xpos.copy()
    phase("shake", shake_s, True)
    phase("calm", calm_s, False)
    final = data.xpos.copy()
    # PER JOINT. Collapsing every joint into one number each step measures whichever joint happens
    # to sit furthest from zero, not how far anything actually travelled: a door swinging 110 deg
    # while a desk's lift top holds still reported a swing of zero.
    # np.ptp(...), not arr.ptp(...): the method was removed from ndarray in NumPy 2.
    swings = (np.ptp(np.asarray(hinge_track), axis=0) if hinge_track else np.array([]))

    def moved(a, b, ids):
        return {n: float(np.linalg.norm(b[i] - a[i])) for i, n, _ in ids}

    drift = moved(start, settled, free)
    shook = moved(settled, final, free)
    # The room is DRIVEN, so it is not a witness to whether static things stayed put — its
    # children are. Excluding it keeps `static_moved` meaningful.
    static_ids = [(b, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}", 0)
                  for b in range(model.nbody)
                  if model.body_jntnum[b] == 0 and b != room_id]
    static_move = moved(start, final, static_ids)

    report = {
        "bodies": int(model.nbody),
        "free_bodies": len(free),
        "static_bodies": len(static_ids),
        "geoms": int(model.ngeom),
        "settle_drift_max_cm": round(max(drift.values(), default=0.0) * 100, 2),
        "settle_drift_mean_cm": round(float(np.mean(list(drift.values()))) * 100, 2) if drift else 0.0,
        "settled_within_1cm": sum(1 for v in drift.values() if v < 0.01),
        "shake_moved_max_cm": round(max(shook.values(), default=0.0) * 100, 2),
        "shake_moved_over_1cm": sum(1 for v in shook.values() if v > 0.01),
        "static_moved_max_cm": round(max(static_move.values(), default=0.0) * 100, 2),
        "room_shake_amplitude_cm": round(amp_m * 100, 2),
        "initial_overlaps": len(initial_overlaps),
        "overlaps_after_relax": relaxed_overlaps,
        "lifted_out_of_walls": lifted,
        "hangings": len(weld_limit),
        "hangings_fell": fallen,
        "worst_initial_overlaps": [(round(d * 100, 1), a, b) for d, a, b in initial_overlaps[:5]],
        "hinge_swing_deg": {
            (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}"):
                (round(float(np.degrees(swings[i])), 1)
                 if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE)
                 else round(float(swings[i]) * 100, 1))     # a slide travels in cm, not degrees
            for i, j in enumerate(hinges)
            if not (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "").startswith("room_")
        } if len(swings) else {},
        "worst_settlers": sorted(((round(v * 100, 1), n) for n, v in drift.items()), reverse=True)[:8],
        "most_shaken": sorted(((round(v * 100, 1), n) for n, v in shook.items()), reverse=True)[:8],
    }

    if renderer is not None and frames_dir is not None:
        video = Path(video)
        video.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(fps),
                        "-i", str(frames_dir / "f_%05d.png"), "-c:v", "libx264",
                        "-preset", "slow", "-pix_fmt", "yuv420p", "-crf", "17",
                        str(video)], check=True)
        report["video"] = str(video)
        report["frames"] = frame_no
        # DELETE THE FRAMES. `mkdtemp` does not clean up after itself, and a render of this room is
        # ~130 MB of PNGs. Left behind, forty-seven renders filled the root filesystem to 4 KB free
        # and the next export died with "No space left on device" — a failure with no connection to
        # anything it was doing. The mp4 is the artefact; the frames are scaffolding.
        shutil.rmtree(frames_dir, ignore_errors=True)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", required=True, type=Path)
    ap.add_argument("--video", type=Path, default=None)
    ap.add_argument("--camera", default=None, help="camera name (default: the widest capture view)")
    # 12 m/s^2 was a demolition setting: it emptied the shelves and threw a chair across the room.
    # A tremor you can watch without the room being destroyed is around 1-2.
    ap.add_argument("--amplitude", type=float, default=1.5,
                    help="lateral m/s^2 during the shake (~1 a tremor, ~5 furniture moves, 12+ destructive)")
    ap.add_argument("--cutaway", action="store_true",
                    help="hide the walls (group 1) and ceiling (group 0) for a cutaway view")
    ap.add_argument("--hide", default="", help="extra geom groups to hide, comma separated")
    ap.add_argument("--ajar", type=float, default=0.0,
                    help="start every articulated joint this fraction open (0.3 = a door ajar)")
    ap.add_argument("--frequency", type=float, default=1.5, help="Hz")
    ap.add_argument("--shake", type=float, default=3.0, help="seconds of shaking")
    # THE SHAKE SHOULD BE MOST OF THE VIDEO. Every second of settle and calm is a still room, and a
    # clip that spends 1 s arriving and 2 s subsiding around 2 s of movement reads as a static scene
    # with a wobble in the middle — the objects are moving correctly and nobody can tell. These are
    # exposed because they set what fraction of the result is worth watching.
    ap.add_argument("--settle", type=float, default=1.0,
                    help="seconds of stillness before the shake (proves the room starts at rest)")
    ap.add_argument("--calm", type=float, default=2.0,
                    help="seconds of stillness after the shake (shows where everything ended up)")
    a = ap.parse_args(argv)
    hide = {int(g) for g in a.hide.split(",") if g.strip().isdigit()}
    if a.cutaway:
        hide |= {0, 1}
    report = run(a.scene, video=a.video, camera=a.camera, shake_s=a.shake,
                 settle_s=a.settle, calm_s=a.calm,
                 amplitude=a.amplitude, frequency=a.frequency, hide_groups=tuple(sorted(hide)),
                 ajar=a.ajar)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
