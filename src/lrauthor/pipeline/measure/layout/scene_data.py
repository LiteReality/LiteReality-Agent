"""scene_data.py — read the RoomPlan pkl set that `extract_scene` writes.

Ported from the physics-agent branch (`agent/physics/scene_loader.py`), which had already solved
the part that is easy to get subtly wrong: walls arrive as a centre, a Y-rotation and a bbox, and
recovering their centreline needs the transform matrix rather than the Euler angle, because
`R = Ry(theta) . Rz(180)` does not decompose the naive way. Re-deriving that by hand invites a sign
error that would put every wall in the room a few degrees out.

Original docstring follows.

Scene geometry / object I/O (port of the pipeline's _geom_io).

Input data lives under  input/<project>/  (one folder per scene, the former
result/<project>/preprocess contents):
    input/<project>/scene_data/{walls,wall_holes,floor,objects}.pkl
    input/<project>/object_stage/<obj>/images_boxed/   (photos for the agent)
    input/<project>/parsed_images/<obj>/

Outputs still go to result/<project>/stage2/ (commit contract unchanged).
When a previous stage2/phase1 run exists under result/, its organized wall
geometry is preferred; otherwise the raw scene_data is converted in-memory.
"""
from __future__ import annotations

import logging
import math
import os
import pickle

import numpy as np

log = logging.getLogger(__name__)


def resolve_scene_data(input_root: str, project: str) -> str:
    """Locate a scan's scene_data dir under either supported layout.

    The Concerto-R bundle used ``<input_root>/<project>/scene_data``; the
    scene_init work tree uses ``<input_root>/scene_data/<project>``. Return
    whichever exists (preferring the bundle layout), else the bundle spelling
    so error messages name the canonical path.
    """
    for candidate in (os.path.join(input_root, project, "scene_data"),
                      os.path.join(input_root, "scene_data", project)):
        if os.path.isdir(candidate):
            return candidate
    return os.path.join(input_root, project, "scene_data")


def _clip_floor_to_walls(floor_verts_3d: list, walls_org: dict,
                         close_radius: float = 0.8) -> list | None:
    """Clip a rectangular floor polygon to the wall envelope.

    AEO scans give a single rectangular floor that covers the scan extent,
    so the floor often pokes out beyond walled rooms. We rebuild it by
    morphological closing on the wall footprints (bridges door openings)
    then take the outer hull of every component, intersected with the
    original rect. Returns a list of polygon-vertex-lists (one per patch)
    or None if clipping failed / wasn't beneficial.
    """
    try:
        from shapely.geometry import MultiPolygon, Polygon
        from shapely.ops import unary_union
    except ImportError:
        return None

    if len(floor_verts_3d) < 3 or len(walls_org) < 4:
        return None

    floor_y = float(floor_verts_3d[0][1])
    floor_rect = Polygon([(float(v[0]), float(v[2])) for v in floor_verts_3d])
    if not floor_rect.is_valid or floor_rect.area < 1e-6:
        return None

    wall_polys = []
    for w in walls_org.values():
        pose = w["pose"]
        line = pose.get("2d_line")
        if line is None:
            continue
        try:
            thickness = float(pose["bbox"][2])
        except (KeyError, IndexError, TypeError):
            thickness = 0.16
        (x0, z0), (x1, z1) = line
        dx, dz = x1 - x0, z1 - z0
        L = math.hypot(dx, dz)
        if L < 1e-6:
            continue
        tx, tz = dx / L, dz / L
        nx, nz = -tz, tx
        h = thickness / 2.0
        wall_polys.append(Polygon([
            (x0 + nx * h, z0 + nz * h),
            (x1 + nx * h, z1 + nz * h),
            (x1 - nx * h, z1 - nz * h),
            (x0 - nx * h, z0 - nz * h),
        ]))
    if not wall_polys:
        return None

    wall_union = unary_union(wall_polys)
    # Closing bridges door gaps and corner gaps so adjacent walls become
    # one connected ring per "room cluster".
    closed = wall_union.buffer(close_radius, join_style=2).buffer(
        -close_radius, join_style=2)
    if closed.is_empty:
        return None
    components = list(closed.geoms) if isinstance(closed, MultiPolygon) \
        else [closed]
    envelopes = [Polygon(list(c.exterior.coords)) for c in components
                 if c.area > 0.5]
    if not envelopes:
        return None
    envelope = unary_union(envelopes)

    # Adaptive choice: hull/envelope ratio measures how "open" the layout
    # is. Closed scenes have ratio ≈ 1.0–1.15 (envelope is a tight fit);
    # open scenes (U-shape, partial enclosures) have ratio ≥ 1.2 where the
    # convex hull covers the implied interior the envelope misses.
    hull = wall_union.convex_hull
    ratio = hull.area / envelope.area if envelope.area > 0 else 1.0
    open_threshold = float(os.environ.get("AEO_FLOOR_CLIP_OPEN_RATIO", "1.20"))
    mode = os.environ.get("AEO_FLOOR_CLIP_MODE", "auto").lower()
    if mode == "tight":
        coverage, scheme = envelope, "tight (forced)"
    elif mode == "loose":
        coverage, scheme = hull, "loose (forced)"
    elif ratio >= open_threshold:
        coverage, scheme = hull, f"loose (open layout, ratio={ratio:.2f})"
    else:
        coverage, scheme = envelope, f"tight (closed layout, ratio={ratio:.2f})"

    clipped = coverage.intersection(floor_rect)
    if clipped.is_empty:
        return None
    patches = list(clipped.geoms) if isinstance(clipped, MultiPolygon) \
        else [clipped]
    patches = [p for p in patches if isinstance(p, Polygon)
               and p.area >= 1e-3]
    if not patches:
        return None

    total = sum(p.area for p in patches)
    # Skip if clipping removed less than 2% — not worth the topology change.
    if total > 0.98 * floor_rect.area:
        return None

    out = []
    for p in patches:
        out.append([[float(x), floor_y, float(z)]
                    for (x, z) in list(p.exterior.coords)[:-1]])
    parts = ", ".join(f"{p.area:.2f}" for p in patches)
    log.info(f"  [geom] floor clipped: {floor_rect.area:.2f} -> "
          f"{total:.2f} m^2 ({len(patches)} patch[es]: {parts}, "
          f"r={close_radius}, scheme={scheme})")
    return out


def _apply_uniform_wall_height(walls_org: dict, walls_hole: dict) -> None:
    """Raise every wall's height to max(bbox[1]) in-place, preserving each
    wall's world-Y bottom and every hole's world-Y position.
    Toggle with env var UNIFORM_WALL_HEIGHT=1.
    """
    if os.environ.get("UNIFORM_WALL_HEIGHT", "").lower() not in \
            ("1", "true", "yes"):
        return
    if not walls_org:
        return

    max_h = max(float(w["pose"]["bbox"][1]) for w in walls_org.values())
    n_changed = 0
    for idx, w in walls_org.items():
        pose = w["pose"]
        old_h = float(pose["bbox"][1])
        old_cy = float(pose["position"][1])
        bottom = old_cy - old_h / 2.0
        new_cy = bottom + max_h / 2.0
        if abs(new_cy - old_cy) < 1e-6 and abs(old_h - max_h) < 1e-6:
            continue
        new_bbox = list(pose["bbox"])
        new_bbox[1] = max_h
        pose["bbox"] = new_bbox
        new_pos = list(pose["position"])
        new_pos[1] = new_cy
        pose["position"] = new_pos
        # Compensate every hole in this wall: keep hole world-Y identical.
        # old_world_v = old_cy + old_v   →   new_v = old_cy + old_v - new_cy
        delta = old_cy - new_cy
        hole_entry = walls_hole.get(idx)
        if isinstance(hole_entry, dict):
            holes = hole_entry.get("holes")
            if isinstance(holes, dict):
                dims = holes.get("dimension")
                if dims:
                    new_dims = []
                    for ((u_min, v_min), (u_max, v_max)) in dims:
                        new_dims.append((
                            (float(u_min), float(v_min) + delta),
                            (float(u_max), float(v_max) + delta),
                        ))
                    holes["dimension"] = new_dims
        n_changed += 1
    if n_changed:
        log.info(f"  [uniform-wall-height] raised {n_changed}/{len(walls_org)} "
              f"walls to h={max_h:.3f} m")


def _wall_yaw_and_line(pose: dict, length: float) -> tuple[float, list]:
    """Extract yaw (deg, LiteReality Ry convention) and XZ centerline from a
    stage1 wall pose dict. Uses the transform matrix when available so
    R ≈ Ry(θ)·Rz(180°) decomposes correctly."""
    position = [float(v) for v in pose["position"]]
    if "transform" in pose and pose["transform"] is not None:
        R = np.asarray(pose["transform"])[:3, :3]
        length_dir = R @ np.array([1.0, 0.0, 0.0])
        ux, uz = float(length_dir[0]), float(length_dir[2])
        n = math.hypot(ux, uz)
        if n < 1e-9:
            ux, uz = 1.0, 0.0
        else:
            ux, uz = ux / n, uz / n
        yaw_deg = math.degrees(math.atan2(-uz, ux))
    else:
        rot = pose.get("rotation", [0, 0, 0])
        yaw_deg = float(rot[1]) if hasattr(rot, "__len__") and len(rot) >= 3 \
            else float(rot)
        yaw_rad = math.radians(yaw_deg)
        ux, uz = math.cos(yaw_rad), -math.sin(yaw_rad)

    cx, cz = position[0], position[2]
    dx, dz = ux * length / 2.0, uz * length / 2.0
    line = [[cx - dx, cz - dz], [cx + dx, cz + dz]]
    return yaw_deg, line


def _build_from_scene_data(scene_data_dir: str,
                           thickness: float = 0.16) -> dict:
    """Convert stage1 scene_data/*.pkl → in-memory PHASE1-equivalent dicts."""
    walls_src = pickle.load(
        open(os.path.join(scene_data_dir, "walls.pkl"), "rb"))
    wall_holes_src = pickle.load(
        open(os.path.join(scene_data_dir, "wall_holes.pkl"), "rb"))
    floor_src = None
    floor_pkl_path = os.path.join(scene_data_dir, "floor.pkl")
    if os.path.exists(floor_pkl_path):
        floor_src = pickle.load(open(floor_pkl_path, "rb"))

    walls_org: dict = {}
    walls_hole: dict = {}
    wall_hole_reorder: list = []

    for idx, wall in enumerate(walls_src):
        pose = wall["pose"]
        bbox = [float(v) for v in pose["bbox"]]
        length = float(bbox[0])
        yaw_deg, line = _wall_yaw_and_line(pose, length)

        walls_org[idx] = {
            "file": idx,
            "pose": {
                "position": [float(v) for v in pose["position"]],
                "rotation": [0.0, yaw_deg, 0.0],
                "bbox": [length, float(bbox[1]), thickness],
                "2d_line": line,
                "area_belongs": [0],
            },
        }

        wall_key = wall.get("file", f"aeo://wall_{idx}.usda")
        holes = wall_holes_src.get(wall_key, []) \
            if isinstance(wall_holes_src, dict) else []
        if isinstance(holes, dict) and holes.get("dimension"):
            wall_hole_reorder.append(wall_key.split("/")[-1].split(".")[0])
            walls_hole[idx] = {"file": idx, "holes": holes}
        else:
            walls_hole[idx] = {"file": idx, "holes": []}

    floor_pkl: dict = {}
    if floor_src is not None:
        fpos = np.asarray(floor_src.get("position", [0, 0, 0]), dtype=float)
        floor_y = float(fpos[1])
        # AEO's floor.position[1] is the slab CENTER, not the top surface;
        # snap to the median wall bottom so the floor mesh meets the walls
        # flush. Opt out with AEO_NO_FLOOR_SNAP=1.
        if walls_org and os.environ.get("AEO_NO_FLOOR_SNAP") != "1":
            wall_bottoms = []
            for w in walls_org.values():
                wp = w["pose"]
                wpos, wbb = wp.get("position"), wp.get("bbox")
                if wpos is None or wbb is None or len(wpos) < 2 or len(wbb) < 2:
                    continue
                wall_bottoms.append(float(wpos[1]) - float(wbb[1]) / 2.0)
            if wall_bottoms:
                snapped = float(np.median(wall_bottoms))
                if abs(snapped - floor_y) > 1e-3:
                    log.info(f"  [floor-snap] floor_y {floor_y:.4f} → "
                          f"{snapped:.4f} (median of {len(wall_bottoms)} "
                          f"wall bottoms)")
                floor_y = snapped
        tdr = floor_src.get("top_down_rect")
        if tdr is not None:
            verts_3d = [[float(x), floor_y, float(z)] for (x, z) in tdr]
        else:
            fbbox = np.asarray(floor_src.get("bbox", [1, 0.02, 1]),
                               dtype=float)
            # The thickness axis is whichever bbox dim is smallest; the floor
            # rectangle lies in the plane perpendicular to it. LiteReality
            # stores floors Y-up (thickness=bbox[1]); AEO/RoomPlan Z-up
            # (thickness=bbox[2]). Detect from bbox so both work.
            thickness_idx = int(np.argmin(fbbox))
            half = fbbox / 2.0
            corners_local = np.zeros((4, 3))
            ax = [i for i in range(3) if i != thickness_idx]
            signs = [(+1, +1), (+1, -1), (-1, -1), (-1, +1)]
            for k, (s0, s1) in enumerate(signs):
                corners_local[k, ax[0]] = s0 * half[ax[0]]
                corners_local[k, ax[1]] = s1 * half[ax[1]]
            # Prefer the full transform matrix (captures rpy beyond pure yaw,
            # e.g. the Rx(180) flip commonly seen on AEO floors).
            T = floor_src.get("transform")
            if T is not None:
                R = np.asarray(T)[:3, :3]
            else:
                rot = floor_src.get("rotation", [0, 0, 0])
                yaw_deg = float(rot[1]) \
                    if hasattr(rot, "__len__") and len(rot) >= 3 else float(rot)
                yaw = math.radians(yaw_deg)
                c, s = math.cos(yaw), math.sin(yaw)
                R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
            corners_world = (R @ corners_local.T).T + fpos.reshape(1, 3)
            verts_3d = [[float(p[0]), floor_y, float(p[2])]
                        for p in corners_world]

        floor_pkl = {0: verts_3d}
        if os.environ.get("AEO_NO_FLOOR_CLIP") != "1":
            try:
                radius = float(os.environ.get("AEO_FLOOR_CLIP_RADIUS", "0.8"))
            except ValueError:
                radius = 0.8
            clipped_patches = _clip_floor_to_walls(verts_3d, walls_org,
                                                   close_radius=radius)
            if clipped_patches:
                floor_pkl = {i: patch for i, patch in
                             enumerate(clipped_patches)}

    _apply_uniform_wall_height(walls_org, walls_hole)

    return {
        "walls_org": walls_org,
        "walls_hole": walls_hole,
        "floor_pkl": floor_pkl,
        "wall_hole_reorder": wall_hole_reorder,
    }


def load_geometry(project: str, input_root: str, result_root: str,
                  thickness: float = 0.16) -> dict:
    """Return walls_org / walls_hole / floor_pkl / wall_hole_reorder.

    Prefers a previous stage2/phase1 output under result_root; falls back
    to the raw scan at input_root/<project>/scene_data.
    """
    phase1 = os.path.join(result_root, project, "stage2", "phase1")
    walls_path = os.path.join(phase1, "walls_organized.pkl")

    if os.path.exists(walls_path):
        walls_org = pickle.load(open(walls_path, "rb"))
        walls_hole = pickle.load(
            open(os.path.join(phase1, "walls_hole_organized.pkl"), "rb"))
        floor_path = os.path.join(phase1, "floor_new_final.pkl")
        floor_pkl = pickle.load(open(floor_path, "rb")) \
            if os.path.exists(floor_path) else {}
        reorder_path = os.path.join(phase1, "wall_hole_reorder.txt")
        wall_hole_reorder = []
        if os.path.exists(reorder_path):
            wall_hole_reorder = [ln.strip() for ln in open(reorder_path)
                                 if ln.strip()]
        log.info("  [geom] source: stage2/phase1")
        _apply_uniform_wall_height(walls_org, walls_hole)
        return {
            "walls_org": walls_org,
            "walls_hole": walls_hole,
            "floor_pkl": floor_pkl,
            "wall_hole_reorder": wall_hole_reorder,
        }

    scene_data = resolve_scene_data(input_root, project)
    if not os.path.exists(os.path.join(scene_data, "walls.pkl")):
        raise FileNotFoundError(
            f"No geometry found: neither {walls_path} nor "
            f"{scene_data}/walls.pkl")
    log.info(f"  [geom] source: {scene_data} (raw scan)")
    return _build_from_scene_data(scene_data, thickness=thickness)


def load_objects(project: str, input_root: str, result_root: str) -> list:
    """Load object list. Prefer stage5 augmented > stage2/phase3 optimized >
    stage2/phase1 step1 > the raw scene_data objects.pkl."""
    candidates = [
        os.path.join(result_root, project, "stage5",
                     "objects_augmented.pkl"),
        os.path.join(result_root, project, "stage2", "phase3",
                     "objects_optimized.pkl"),
        os.path.join(result_root, project, "stage2", "phase1",
                     "objects_step1.pkl"),
        os.path.join(resolve_scene_data(input_root, project), "objects.pkl"),
    ]
    for path in candidates:
        if os.path.exists(path):
            objs = pickle.load(open(path, "rb"))
            rel = os.path.relpath(path)
            log.info(f"  [geom] objects: {len(objs)} from {rel}")
            return objs
    log.info("  [geom] objects: 0 (no source file found)")
    return []
