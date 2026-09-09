"""
render_room_cameras.py — render the room.py scene from the ARKit capture cameras.

`room.py` builds the assembled room (RoomPlan shell + generated GLB assets) in
Blender and rebuilds one camera per capture frame (`cam_00000` … matching
`frame_00000.jpg`). This module renders that scene from those cameras and reports
the render speed.

room.py itself sets no lights/engine, so this adds fast lighting (ambient world +
a sun) and uses EEVEE by default — the same recipe as the blender_recon skill.

Run headless (needs Blender; room.py needs bpy):
    /Applications/Blender.app/Contents/MacOS/Blender -b \
      --python authoring/views/room_render/render_room_cameras.py -- \
      [scan_dir] [assets_dir] [out_dir] [frames] [engine] [res_div]

All args optional:
    scan_dir    default: room.py's DEF_SCAN (the bundled scan)
    assets_dir  default: room.py's DEF_ASSETS (input/assets)
    out_dir     default: <project>/run/room_render
    frames      "all" (default) | N evenly-spaced (e.g. 8) | "a-b" | "0,8,30"
    engine      "EEVEE" (default) | "CYCLES"
    res_div     integer downscale, 1 = full native res (default), 2 = half, …

Programmatic use (inside Blender):
    import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
    from render_room_cameras import build_scene, render_room_from_cameras
    scene = build_scene(scan_dir, assets_dir, out_dir)
    stats = render_room_from_cameras(scene, out_dir, frames=[0, 8, 30])
"""

import json
import math
import os
import re
import statistics
import sys
import time

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


# --------------------------------------------------------------------------- #
# Build the room.py scene
# --------------------------------------------------------------------------- #
def _project_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))  # authoring/views/room_render
    return os.path.abspath(os.path.join(here, os.pardir, os.pardir))


def _default_scan(root) -> str:
    base = os.path.join(root, "input")
    if os.path.isdir(base):
        for name in sorted(os.listdir(base)):
            d = os.path.join(base, name)
            if os.path.isdir(d) and os.path.isfile(os.path.join(d, "room.usdz")):
                return d
    return os.path.join(base, "LiteReality-LiteReality_Scanner-Room_2732-20260603-154053")


def build_scene(scan_dir=None, assets_dir=None, out_dir=None):
    """Execute room.py to build the assembled scene + cameras, and return the
    RoomScene instance (room.py's global ``ROOM``)."""
    root = _project_root()
    room_py = os.environ.get("SB_ROOM_PY") or os.path.join(root, "room.py")
    if not os.path.isfile(room_py):
        raise SystemExit(f"room.py not found at {room_py}")

    out_dir = out_dir or os.path.join(root, "run", "room_render")
    os.makedirs(out_dir, exist_ok=True)
    out_glb = os.path.join(out_dir, "room.glb")

    # room.py reads scan_dir / assets_dir / out_glb positionally from argv after
    # "--" (an empty entry is taken literally, not as "use default"), so resolve
    # real paths here.
    scan_dir = scan_dir or _default_scan(root)
    assets_dir = assets_dir or os.path.join(root, "input", "assets")

    ns = {"__file__": room_py, "__name__": "room_py_exec"}
    argv = ["room.py", "--", scan_dir, assets_dir, out_glb]

    saved = sys.argv
    sys.argv = argv
    try:
        with open(room_py) as fh:
            code = compile(fh.read(), room_py, "exec")
        exec(code, ns)  # builds the scene -> ns["ROOM"]
    finally:
        sys.argv = saved

    scene = ns.get("ROOM")
    if scene is None:
        raise SystemExit("room.py did not produce a ROOM scene")
    return scene


# --------------------------------------------------------------------------- #
# Render setup
# --------------------------------------------------------------------------- #
def _set_engine(want: str) -> str:
    sc = bpy.context.scene
    valid = [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]
    if want.upper().startswith("CYC"):
        candidates = ["CYCLES"]
    else:
        candidates = ["BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"]
    for cand in candidates:
        if cand in valid:
            sc.render.engine = cand
            break
    else:
        sc.render.engine = valid[0]
    return sc.render.engine


def _kelvin_to_rgb(k: float):
    """Cheap blackbody Kelvin -> linear RGB (approx), for physically-plausible lamp colour."""
    t = k / 100.0
    if t <= 66:
        r = 255.0
        g = 99.47 * math.log(t) - 161.12 if t > 0 else 255.0
    else:
        r = 329.7 * ((t - 60) ** -0.1332)
        g = 288.12 * ((t - 60) ** -0.0755)
    b = 255.0 if t >= 66 else (138.5 * math.log(t - 10) - 305.04 if t > 19 else 0.0)
    c = [max(0.0, min(255.0, x)) / 255.0 for x in (r, g, b)]
    return tuple((v**2.2) for v in c)  # to ~linear


def _nishita_world(sc, sun_elevation_deg=40.0, sun_rotation_deg=35.0, strength=0.25):
    """Physical Nishita sky as the world light (indoor daylight)."""
    world = sc.world or bpy.data.worlds.new("World")
    sc.world = world
    world.use_nodes = True
    nt = world.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs[1].default_value = strength
    try:
        sky = nt.nodes.new("ShaderNodeTexSky")
        sky.sky_type = "NISHITA"
        sky.sun_elevation = math.radians(sun_elevation_deg)
        sky.sun_rotation = math.radians(sun_rotation_deg)
        nt.links.new(sky.outputs[0], bg.inputs[0])
    except Exception:  # noqa: BLE001 — fall back to a flat sky colour
        bg.inputs[0].default_value = (0.9, 0.93, 0.98, 1.0)
    nt.links.new(bg.outputs[0], out.inputs[0])


def _window_portals(sc):
    """One is_portal AREA light per window opening — guides the world light in physically."""
    n = 0
    for obj in list(bpy.data.objects):
        if obj.type != "MESH" or "window" not in obj.name.lower():
            continue
        dims = obj.dimensions
        ld = bpy.data.lights.new(f"portal_{obj.name}", "AREA")
        ld.shape = "RECTANGLE"
        ld.size = max(dims.x, dims.y, 0.4)
        ld.size_y = max(dims.z, 0.4)
        try:
            ld.cycles.is_portal = True
        except Exception:  # noqa: BLE001
            ld.energy = 20.0  # EEVEE has no portals — a soft fill instead
        lo = bpy.data.objects.new(f"portal_{obj.name}", ld)
        lo.location = obj.matrix_world.translation
        lo.rotation_euler = obj.rotation_euler
        sc.collection.objects.link(lo)
        n += 1
    return n


def _ceiling_lamps(sc, watts=60.0, temp_k=4700.0):
    """Warm point lamps just under the ceiling (physically-based watt + Kelvin)."""
    ceils = [o for o in bpy.data.objects if o.type == "MESH" and o.name.lower().startswith("ceiling")]
    if not ceils:
        return 0
    n = 0
    for c in ceils:
        bb = [c.matrix_world @ Vector(corner) for corner in c.bound_box]
        cx = sum(v.x for v in bb) / 8
        cy = sum(v.y for v in bb) / 8
        cz = min(v.z for v in bb) - 0.05
        ld = bpy.data.lights.new("ceil_lamp", "POINT")
        ld.energy = watts
        ld.color = _kelvin_to_rgb(temp_k)
        ld.shadow_soft_size = 0.05
        lo = bpy.data.objects.new("ceil_lamp", ld)
        lo.location = (cx, cy, cz)
        sc.collection.objects.link(lo)
        n += 1
    return n


def _apply_clay(color=(0.2, 0.05, 0.01, 1.0)):
    """Assign a single flat clay material to every mesh — judge shape/placement, not material."""
    clay = bpy.data.materials.get("__clay__") or bpy.data.materials.new("__clay__")
    clay.use_nodes = True
    bsdf = clay.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = 0.9
    for o in bpy.data.objects:
        if o.type == "MESH":
            o.data.materials.clear()
            o.data.materials.append(clay)


def setup_render(
    engine="EEVEE",
    light=True,
    ambient=0.6,
    sun_energy=3.0,
    lighting="simple",
    view_transform="Standard",
    exposure=0.0,
    film_exposure=1.0,
    clay=False,
    samples=None,
):
    """Configure the render: colour management, lighting rig, engine + quality, so the room.py
    scene renders (room.py builds geometry only). Safe to call once per session.

    lighting:
      "simple"   — one sun + flat white world (fast, the original recipe)
      "physical" — Nishita sky + an is_portal light per window + warm watt/Kelvin ceiling lamps
                   (photorealistic indoor light; best with engine="CYCLES")
    view_transform: MATCH the capture photos' colour domain ("Standard" ≈ sRGB JPEG; "Filmic"/"AgX"
                    for wider-gamut sources). film_exposure is a scene-linear pre-tonemap gain.
    clay:    render every surface as one flat clay material (shape/placement critique, no material).
    samples: Cycles sample cap; None → 4096 hero default on Cycles.
    """
    sc = bpy.context.scene
    eng = _set_engine(engine)

    if lighting == "physical":
        _nishita_world(sc, strength=0.25)
        if not any(o.type == "LIGHT" for o in bpy.data.objects):
            _window_portals(sc)
            _ceiling_lamps(sc)
    elif light:
        world = sc.world or bpy.data.worlds.new("World")
        sc.world = world
        world.use_nodes = True
        bg = world.node_tree.nodes.get("Background")
        if bg is None:
            bg = world.node_tree.nodes.new("ShaderNodeBackground")
        bg.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
        bg.inputs[1].default_value = ambient
        if not any(o.type == "LIGHT" for o in bpy.data.objects):
            ld = bpy.data.lights.new("SunKey", "SUN")
            ld.energy = sun_energy
            try:
                ld.angle = math.radians(3.0)  # soft shadows
            except Exception:
                pass
            sun = bpy.data.objects.new("SunKey", ld)
            sc.collection.objects.link(sun)
            sun.rotation_euler = (math.radians(50), 0.0, math.radians(35))

    if clay:
        _apply_clay()

    # EEVEE niceties (ignored on Cycles).
    try:
        sc.eevee.use_raytracing = True
        sc.eevee.use_shadows = True
    except Exception:
        pass
    # Cycles quality: adaptive sampling + brute-force clean (matches Infinigen's indoor recipe).
    if eng == "CYCLES":
        try:
            sc.cycles.samples = samples or 4096
            sc.cycles.adaptive_threshold = 0.005
            sc.cycles.time_limit = 0
            sc.cycles.use_denoising = True
            try:
                sc.cycles.denoiser = "OPTIX"
            except Exception:
                pass
        except Exception:
            pass
    # Colour management — the systematic render-vs-photo bias lever.
    try:
        sc.view_settings.view_transform = view_transform
        sc.view_settings.exposure = exposure
        sc.cycles.film_exposure = film_exposure
    except Exception:
        pass
    return eng


def camera_indices():
    """Frame indices of the built ARKit cameras (cam_NNNNN), sorted."""
    idx = []
    for obj in bpy.data.objects:
        if obj.name.startswith("cam_") and obj.name[4:].isdigit():
            idx.append(int(obj.name[4:]))
    return sorted(idx)


def select_frames(all_idx, spec):
    """spec: None/'all' | int N (N evenly spaced) | 'a-b' range | '0,8,30' list."""
    if spec in (None, "", "all"):
        return list(all_idx)
    spec = str(spec).strip()
    if "," in spec:
        want = {int(x) for x in spec.split(",") if x.strip()}
        return [i for i in all_idx if i in want]
    if "-" in spec and all(p.isdigit() for p in spec.split("-", 1)):
        a, b = (int(p) for p in spec.split("-", 1))
        return [i for i in all_idx if a <= i <= b]
    n = int(spec)
    if n >= len(all_idx) or n <= 0:
        return list(all_idx)
    if n == 1:
        return [all_idx[0]]
    step = (len(all_idx) - 1) / (n - 1)
    return [all_idx[int(round(k * step))] for k in range(n)]


# --------------------------------------------------------------------------- #
# Upright output: rotate the render 90 deg clockwise (portrait-held capture)
# --------------------------------------------------------------------------- #
def rotate_png_cw(path):
    """Rotate a saved PNG 90 deg clockwise in place (numpy only; Blender has no
    PIL). ARKit stores the landscape sensor frame, so this makes the render
    upright, matching the rotated overlays. Returns (new_w, new_h)."""
    import numpy as np

    img = bpy.data.images.load(path, check_existing=False)
    w, h = img.size
    arr = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)  # bottom-up
    out = np.rot90(arr[::-1], k=-1)[::-1]  # 90 deg CW
    new_w, new_h = h, w
    ni = bpy.data.images.new("rot_tmp", width=new_w, height=new_h, alpha=True)
    ni.colorspace_settings.name = img.colorspace_settings.name
    ni.pixels = out.reshape(-1)
    ni.filepath_raw = path
    ni.file_format = "PNG"
    ni.save()
    bpy.data.images.remove(img)
    bpy.data.images.remove(ni)
    return new_w, new_h


# --------------------------------------------------------------------------- #
# Agent-facing scene description (what each view looks at + how to edit it)
# --------------------------------------------------------------------------- #
_STATIC_CATS = {"wall", "floor", "ceiling"}


def edit_affordances(rec):
    """The edits that make sense for this object, by category + flags — the
    handle + legal moves an agent needs. Actual ops live on RoomScene / room.py."""
    cat = rec["category"]
    if cat in _STATIC_CATS:
        return ["recolor", "retexture"]
    if cat == "door":
        ops = ["articulate(open/close)", "swap_glb", "recolor"]
    elif cat == "window":
        ops = ["swap_glb", "recolor", "blinds(open/close)"]
    else:
        ops = ["move_xy", "rotate_z", "scale", "swap_glb", "recolor"]
    if rec.get("placeable_surface"):
        ops.append(f"place_on_top(z={rec['top_z']})")
    return ops


def _obj_to_id(obj, objects):
    """Map a ray-hit Blender object to a registry id (walk up to the handle)."""
    o = obj
    while o is not None:
        if o.get("room_id"):
            return o["room_id"]
        if re.match(r"(Wall|Floor)\d+$", o.name):
            return o.name
        o = o.parent
    return obj.name if obj else None


def _bbox_corners(rec):
    lo, hi = rec["bbox_min"], rec["bbox_max"]
    return [
        Vector((lo[0] if i & 1 else hi[0], lo[1] if i & 2 else hi[1], lo[2] if i & 4 else hi[2]))
        for i in range(8)
    ]


_PTS_CACHE = {}


def _object_world_points(rec, cap=600):
    """Sampled world-space mesh vertices of an object (cached) for a tight
    silhouette box. Falls back to bbox corners if it has no mesh."""
    rid = rec["id"]
    if rid in _PTS_CACHE:
        return _PTS_CACHE[rid]
    deps = bpy.context.evaluated_depsgraph_get()
    handle = bpy.data.objects.get(rec["handle"])
    meshes, stack = [], ([handle] if handle else [])
    while stack:
        o = stack.pop()
        if o.type == "MESH":
            meshes.append(o)
        stack.extend(o.children)
    pts = []
    for o in meshes:
        ev = o.evaluated_get(deps)
        me = ev.to_mesh()
        mat, vs = o.matrix_world, me.vertices
        step = max(1, len(vs) // cap)
        for i in range(0, len(vs), step):
            pts.append(mat @ vs[i].co.copy())
        ev.to_mesh_clear()
    _PTS_CACHE[rid] = pts or _bbox_corners(rec)
    return _PTS_CACHE[rid]


def describe_view(scene, frame_index):
    """Agent-facing description of capture view <frame_index>: what the camera
    looks at, the visible editable objects (where in the UPRIGHT image, distance,
    state) and what is out of frame. Image coords are in the rotated (90 deg CW)
    frame, matching the saved render."""
    sc = bpy.context.scene
    cam = bpy.data.objects.get(f"cam_{frame_index:05d}")
    if cam is None:
        return None
    deps = bpy.context.evaluated_depsgraph_get()
    origin = cam.matrix_world.translation.copy()
    cam_inv = cam.matrix_world.inverted()
    near = cam.data.clip_start or 0.01

    forward = -(cam.matrix_world.to_3x3().col[2]).normalized()
    look = None
    hit, loc, _n, _i, hobj, _m = sc.ray_cast(deps, origin, forward)
    if hit:
        look = {"id": _obj_to_id(hobj, scene.objects), "dist": round((loc - origin).length, 2)}

    visible = []
    for rid, rec in scene.objects.items():
        center = Vector(rec["center"])
        if rec["category"] in ("floor", "ceiling"):
            # Large planes with only sparse perimeter vertices (often occluded by
            # furniture) — a vertex-based occlusion test over-drops them, and cross-room
            # floor/ceiling ghosting is rare. Keep centre projection.
            cu = world_to_camera_view(sc, cam, center)
            if cu.z <= 0 or not (0.0 <= cu.x <= 1.0 and 0.0 <= cu.y <= 1.0):
                continue
            lx, ly = cu.y, cu.x  # rotated frame: x=v, y=u
            bbox = (lx, ly, lx, ly)
        else:
            # Walls + furniture: keep the mesh points that are on-screen AND not occluded
            # by a nearer surface, label at the centroid of those. This is the fix for
            # multi-room scans — a WALL hidden behind another room's wall used to be
            # labelled (walls were centre-projected with no occlusion test). A partly
            # hidden object still gets a label, on the part you can actually see.
            allp = _object_world_points(rec)
            samp = allp[:: max(1, len(allp) // 40)]
            seen = []
            for p in samp:
                if (cam_inv @ p).z >= -near:
                    continue  # behind the camera
                co = world_to_camera_view(sc, cam, p)
                if not (0.0 <= co.x <= 1.0 and 0.0 <= co.y <= 1.0):
                    continue  # off-screen
                d = p - origin
                r = sc.ray_cast(deps, origin, d.normalized())
                if (not r[0]) or (r[1] - origin).length >= d.length - 0.05:
                    seen.append((co.y, co.x))  # visible point (rotated x,y)
            if len(seen) < max(2, len(samp) // 12):
                continue  # not enough of it is visible (occluded/off-screen)
            lx = sum(s[0] for s in seen) / len(seen)
            ly = sum(s[1] for s in seen) / len(seen)
            xs, ys = [s[0] for s in seen], [s[1] for s in seen]
            bbox = (min(xs), min(ys), max(xs), max(ys))  # visible extent (rotated x,y)
        visible.append(
            {
                "id": rid,
                "category": rec["category"],
                "label_pct": [round(lx * 100, 1), round(ly * 100, 1)],
                "bbox_pct": [round(v * 100, 1) for v in bbox],  # x0,y0,x1,y1 of visible extent
                "dist_m": round((center - origin).length, 2),
                "edits": edit_affordances(rec),
            }
        )
    visible.sort(key=lambda d: d["dist_m"])
    return {
        "frame": f"frame_{frame_index:05d}",
        "camera_pos": [round(v, 3) for v in origin],
        "looking_at": look,
        "visible": visible,
    }


def scene_manifest(scene):
    """The editable object graph, emitted once (per-view files reference these
    ids). The agent's map of WHAT exists + HOW to edit each thing."""
    objs = []
    for rid, rec in sorted(scene.objects.items()):
        objs.append(
            {
                "id": rid,
                "category": rec["category"],
                "handle": rec["handle"],
                "center": rec["center"],
                "size": rec["size"],
                "bbox_min": rec["bbox_min"],
                "bbox_max": rec["bbox_max"],
                "top_z": rec["top_z"],
                "placeable_surface": rec.get("placeable_surface", False),
                "source_glb": rec.get("source_glb"),
                "edits": edit_affordances(rec),
            }
        )
    cams = sorted(
        int(o.name[4:])
        for o in bpy.data.objects
        if o.name.startswith("cam_") and o.name[4:].isdigit()
    )
    return {
        "objects": objs,
        "counts": {
            c: sum(1 for o in objs if o["category"] == c)
            for c in sorted({o["category"] for o in objs})
        },
        "n_objects": len(objs),
        "cameras": cams,
        "render_resolution_native": [scene.render_w, scene.render_h],
    }


# --------------------------------------------------------------------------- #
# Render loop + timing
# --------------------------------------------------------------------------- #
def render_room_from_cameras(
    scene, out_dir, frames=None, res_div=1, engine="EEVEE", light=True, rotate=True, describe=True
):
    """Render the built scene from the chosen capture cameras into out_dir and
    report timing. Returns a stats dict.

    frames: None/'all' | int N | 'a-b' | '0,8,30' | explicit list of ints.
    """
    os.makedirs(out_dir, exist_ok=True)
    t_setup = time.perf_counter()
    eng = setup_render(engine=engine, light=light)
    setup_sec = time.perf_counter() - t_setup

    all_idx = camera_indices()
    if not all_idx:
        raise SystemExit("no ARKit cameras (cam_NNNNN) in the scene")
    chosen = frames if isinstance(frames, (list, tuple)) else select_frames(all_idx, frames)
    chosen = [i for i in chosen if i in set(all_idx)]

    res = (scene.render_w, scene.render_h)
    eff = (res[0] // res_div, res[1] // res_div) if all(res) else res
    print(
        f"\n=== render {len(chosen)} cam(s) | engine {eng} | "
        f"res {eff[0]}x{eff[1]} (res_div={res_div}) -> {out_dir} ==="
    )

    upright_res = (eff[1], eff[0]) if rotate else eff
    if describe and not getattr(scene, "objects", None):
        scene.index()

    times, post = [], 0.0
    descriptions = []
    for n, fi in enumerate(chosen, 1):
        out_path = os.path.join(out_dir, f"frame_{fi:05d}.png")
        t0 = time.perf_counter()
        try:
            scene.render_from(fi, out_path, res_div=res_div)
        except Exception as exc:  # keep the batch going
            print(f"  [!] cam_{fi:05d} failed: {exc}")
            continue
        dt = time.perf_counter() - t0
        times.append(dt)
        tp = time.perf_counter()
        if rotate:
            rotate_png_cw(out_path)
        if describe:
            desc = describe_view(scene, fi)
            if desc is not None:
                descriptions.append(desc)
        post += time.perf_counter() - tp
        if n == 1 or n == len(chosen) or n % 10 == 0:
            print(f"  [{n}/{len(chosen)}] cam_{fi:05d}  {dt:.2f}s")

    if describe:
        with open(os.path.join(out_dir, "scene_manifest.json"), "w") as fh:
            json.dump(scene_manifest(scene), fh, indent=2)
        with open(os.path.join(out_dir, "render_manifest.json"), "w") as fh:
            json.dump(
                {
                    "out_dir": out_dir,
                    "engine": eng,
                    "upright_resolution": list(upright_res),
                    "rotated_cw": rotate,
                    "frames": descriptions,
                },
                fh,
                indent=2,
            )

    stats = {
        "engine": eng,
        "resolution": list(upright_res),
        "res_div": res_div,
        "rendered": len(times),
        "rotated_cw": rotate,
        "described": describe,
        "out_dir": out_dir,
        "setup_sec": round(setup_sec, 3),
        "post_sec": round(post, 2),
    }
    if times:
        total = sum(times)
        stats.update(
            {
                "total_sec": round(total, 2),
                "per_frame_mean_sec": round(statistics.mean(times), 3),
                "per_frame_median_sec": round(statistics.median(times), 3),
                "per_frame_min_sec": round(min(times), 3),
                "per_frame_max_sec": round(max(times), 3),
                "fps": round(len(times) / total, 2) if total else None,
                "est_all_cameras_sec": round(statistics.mean(times) * len(all_idx), 1),
                "n_cameras_total": len(all_idx),
            }
        )
        print("\n=== SPEED ===")
        print(
            f"  rendered {len(times)} frame(s) in {total:.2f}s  "
            f"({stats['fps']} fps, {stats['per_frame_mean_sec'] * 1000:.0f} ms/frame mean)"
        )
        print(
            f"  per-frame  mean {stats['per_frame_mean_sec']:.3f}s  "
            f"median {stats['per_frame_median_sec']:.3f}s  "
            f"min {stats['per_frame_min_sec']:.3f}s  max {stats['per_frame_max_sec']:.3f}s"
        )
        print(
            f"  rotate+describe overhead {post:.2f}s total ({post / len(times) * 1000:.0f} ms/frame)"
        )
        print(
            f"  setup (lighting+engine) {setup_sec:.2f}s  |  "
            f"est. all {len(all_idx)} cams ≈ {stats['est_all_cameras_sec']:.0f}s render"
        )
    return stats


def describe_only(scene, out_dir, frames=None):
    """Re-derive the per-view descriptions WITHOUT rendering (fast). Reuses the
    existing (already-upright) PNGs; only rewrites the manifests. Lets you iterate
    on the boxes/labels without paying for a re-render."""
    os.makedirs(out_dir, exist_ok=True)
    if not getattr(scene, "objects", None):
        scene.index()
    all_idx = camera_indices()
    chosen = frames if isinstance(frames, (list, tuple)) else select_frames(all_idx, frames)
    chosen = [i for i in chosen if i in set(all_idx)]
    upright_res = (scene.render_h, scene.render_w)  # rotated (upright)
    descriptions = [d for fi in chosen if (d := describe_view(scene, fi)) is not None]
    with open(os.path.join(out_dir, "scene_manifest.json"), "w") as fh:
        json.dump(scene_manifest(scene), fh, indent=2)
    with open(os.path.join(out_dir, "render_manifest.json"), "w") as fh:
        json.dump(
            {
                "out_dir": out_dir,
                "engine": "(describe-only)",
                "upright_resolution": list(upright_res),
                "rotated_cw": True,
                "frames": descriptions,
            },
            fh,
            indent=2,
        )
    print(f"described {len(descriptions)} view(s) -> {out_dir}/render_manifest.json")
    return descriptions


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_argv():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    scan_dir = argv[0] if len(argv) > 0 and argv[0] else None
    assets_dir = argv[1] if len(argv) > 1 and argv[1] else None
    out_dir = (
        argv[2]
        if len(argv) > 2 and argv[2]
        else os.path.join(_project_root(), "run", "room_render")
    )
    frames = argv[3] if len(argv) > 3 and argv[3] else "all"
    engine = argv[4] if len(argv) > 4 and argv[4] else "EEVEE"
    res_div = int(argv[5]) if len(argv) > 5 and argv[5] else 1
    mode = argv[6] if len(argv) > 6 and argv[6] else "render"
    return scan_dir, assets_dir, out_dir, frames, engine, res_div, mode


def main():
    scan_dir, assets_dir, out_dir, frames, engine, res_div, mode = _parse_argv()
    t0 = time.perf_counter()
    scene = build_scene(scan_dir, assets_dir, out_dir)
    build_sec = time.perf_counter() - t0
    print(f"\nbuilt room.py scene in {build_sec:.2f}s")
    if mode == "describe":
        describe_only(scene, out_dir, frames=frames)
        print(f"\nDONE  build {build_sec:.1f}s + describe-only")
        return
    stats = render_room_from_cameras(scene, out_dir, frames=frames, res_div=res_div, engine=engine)
    stats["build_sec"] = round(build_sec, 2)
    print(f"\nDONE  build {build_sec:.1f}s + render {stats.get('total_sec', 0)}s")
    return stats


if __name__ == "__main__":
    main()
