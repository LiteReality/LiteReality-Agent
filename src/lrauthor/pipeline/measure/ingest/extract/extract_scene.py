"""Stage 1 — raw scan folder -> RoomPlan scene data (walls/objects/wall_holes/floor).

Ported from the v2 ``step1_scene_extraction``. Copies the scan's USDZ + point
cloud, extracts the RGBD frames, parses the RoomPlan layout, and rebuilds the
door/window openings from the wall-child USDAs. Writes, relative to the work root:

    input/usdz_files/<scan>.usdz
    input/scan_geometry/<scan>/scene_points.obj
    input/rgbd/<scan>/...
    input/scene_data/<scan>/{walls,objects,wall_holes,floor}.pkl

Heavy USD/torch/GroundingDINO code is NOT needed here. RoomPlan USDZ is parsed by
the ported implementation under ``ingest/preprocessing/vendor``.
"""

from __future__ import annotations

import glob
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image

from lrauthor.pipeline.measure import paths as config
from lrauthor.pipeline.measure.ingest.preprocessing.scene_data import (
    Colors,
    extract_rgbd,
    extract_scene_data,
    get_object_pose,
    get_scene_usda,
    save_scene_data,
)

# ── point-cloud helpers (a scan may ship pointcloud.pcd, an OBJ, or only depth) ──
_PCD_DTYPES = {
    ("I", 1): "i1",
    ("I", 2): "i2",
    ("I", 4): "i4",
    ("I", 8): "i8",
    ("U", 1): "u1",
    ("U", 2): "u2",
    ("U", 4): "u4",
    ("U", 8): "u8",
    ("F", 4): "f4",
    ("F", 8): "f8",
}


def _parse_pcd_header(raw: bytes) -> tuple[dict, int]:
    """PCD header fields + the byte offset where the payload starts. The header is
    always ASCII lines, whatever DATA mode follows."""
    header, offset = {}, 0
    for line in raw.split(b"\n"):
        offset += len(line) + 1
        text = line.decode("ascii", "replace").strip()
        if not text or text.startswith("#"):
            continue
        key, _, rest = text.partition(" ")
        header[key.upper()] = rest.strip()
        if key.upper() == "DATA":
            break
    return header, offset


def _pcd_binary_xyz(raw: bytes, header: dict, offset: int) -> np.ndarray:
    """(n,3) float array from a `DATA binary` payload, via the header's field layout."""
    fields = header.get("FIELDS", "").split()
    sizes = [int(v) for v in header.get("SIZE", "").split()]
    types = [v.upper() for v in header.get("TYPE", "").split()]
    counts = [int(v) for v in header.get("COUNT", "").split()] or [1] * len(fields)
    if not (len(fields) == len(sizes) == len(types) == len(counts)):
        raise RuntimeError("PCD header FIELDS/SIZE/TYPE/COUNT disagree")

    # COUNT>1 fields occupy that many consecutive values; name them uniquely so the
    # structured dtype keeps the byte layout exact.
    spec = []
    for name, size, typ, cnt in zip(fields, sizes, types, counts):
        key = _PCD_DTYPES.get((typ, size))
        if key is None:
            raise RuntimeError(f"unsupported PCD field type {typ}{size} for '{name}'")
        spec.extend([(name if cnt == 1 else f"{name}__{i}", key) for i in range(cnt)])
    for axis in ("x", "y", "z"):
        if axis not in [n for n, _ in spec]:
            raise RuntimeError(f"PCD has no '{axis}' field (FIELDS: {' '.join(fields)})")

    dtype = np.dtype(spec)
    n = int(header.get("POINTS") or 0) or (
        int(header.get("WIDTH", 0) or 0) * int(header.get("HEIGHT", 0) or 0)
    )
    body = raw[offset:]
    avail = len(body) // dtype.itemsize
    if n <= 0 or n > avail:
        n = avail  # trust the payload over a header that over-promises
    rec = np.frombuffer(body, dtype=dtype, count=n)
    return np.stack([rec["x"], rec["y"], rec["z"]], axis=1).astype(np.float64)


def _pcd_to_vertex_obj(pcd_path: Path, obj_path: Path) -> int:
    """RoomPlanScanner ships `DATA ascii`, but scripts/shrink_scan.py --extras rewrites
    it to `DATA binary` — handle both, or a shrunk scan silently has no point cloud."""
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    raw = pcd_path.read_bytes()
    header, offset = _parse_pcd_header(raw)
    mode = header.get("DATA", "ascii").lower()

    if mode == "binary_compressed":
        raise RuntimeError(
            f"{pcd_path} is DATA binary_compressed (LZF), which is not supported — "
            "re-export as `ascii` or `binary`"
        )

    if mode == "binary":
        pts = _pcd_binary_xyz(raw, header, offset)
        pts = pts[np.isfinite(pts).all(axis=1)]
        count = int(len(pts))
        if count == 0:
            raise RuntimeError(f"No points found in {pcd_path}")
        with obj_path.open("w", encoding="utf-8") as dst:
            dst.write("# Generated from RoomPlanScanner pointcloud.pcd for object_init\n")
            dst.writelines(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in pts)
        return count

    count = 0
    data = False
    with (
        pcd_path.open("r", encoding="utf-8", errors="ignore") as src,
        obj_path.open("w", encoding="utf-8") as dst,
    ):
        dst.write("# Generated from RoomPlanScanner pointcloud.pcd for object_init\n")
        for line in src:
            stripped = line.strip()
            if not stripped:
                continue
            if not data:
                if stripped.lower().startswith("data"):
                    data = True
                continue
            parts = stripped.split()
            if len(parts) < 3:
                continue
            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            except ValueError:
                continue
            dst.write(f"v {x} {y} {z}\n")
            count += 1
    if count == 0:
        raise RuntimeError(f"No points found in {pcd_path}")
    return count


def _depth_frames_to_vertex_obj(
    raw: Path, obj_path: Path, frame_stride: int = 4, pixel_stride: int = 3
) -> int:
    """Unproject RoomPlan depth frames into a world-space point cloud OBJ.

    Used when a scan ships depth + ARKit poses but no precomputed pointcloud.pcd /
    textured_output.obj. ARKit: intrinsics + cameraPoseARFrame column-major; camera
    space +x right, +y up, -z forward; depth uint16 mm at 256x192 while intrinsics
    are for the 1920x1440 RGB frame.
    """
    import json

    frames = sorted(raw.glob("frame_*.json")) or sorted((raw / "images").glob("frame_*.json"))
    if not frames:
        return 0
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with obj_path.open("w", encoding="utf-8") as dst:
        dst.write("# Generated from RoomPlan depth frames for object_init\n")
        for fi, fjson in enumerate(frames):
            if fi % frame_stride:
                continue
            depth_png = fjson.with_name(fjson.stem.replace("frame", "depth") + ".png")
            if not depth_png.exists():
                continue
            try:
                meta = json.loads(fjson.read_text())
                K = np.asarray(meta["intrinsics"], dtype=float).reshape(3, 3)
                pose = np.asarray(meta["cameraPoseARFrame"], dtype=float).reshape(4, 4)
                depth = np.asarray(Image.open(depth_png), dtype=np.float32) / 1000.0
            except Exception:
                continue
            dh, dw = depth.shape
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            sx, sy = dw / 1920.0, dh / 1440.0
            fx, fy, cx, cy = fx * sx, fy * sy, cx * sx, cy * sy
            ys, xs = np.mgrid[0:dh:pixel_stride, 0:dw:pixel_stride]
            d = depth[ys, xs]
            valid = (d > 0.1) & (d < 8.0)
            xs, ys, d = xs[valid], ys[valid], d[valid]
            if d.size == 0:
                continue
            xc = (xs - cx) / fx * d
            yc = -((ys - cy) / fy * d)
            cam = np.stack([xc, yc, -d, np.ones_like(d)], axis=1)
            world = (pose @ cam.T).T[:, :3]
            for p in world:
                dst.write(f"v {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
                written += 1
    return written


def _first_frame(raw: Path) -> Path | None:
    for pattern in ("images/frame_*.jpg", "frame_*.jpg", "*.jpg"):
        for path in sorted(raw.glob(pattern)):
            return path
    return None


def copy_scan_files(raw_scan_folder: str, scan: str) -> str:
    """Copy USDZ + point cloud into the work tree and extract RGBD. Returns USDZ path."""
    raw = Path(raw_scan_folder)
    started = time.time()

    original_usdz = raw / "roomplan" / "room.usdz"
    if not original_usdz.exists():
        original_usdz = raw / "room.usdz"
    if not original_usdz.exists():
        raise FileNotFoundError(f"Missing room.usdz under {raw}")

    scene_usdz = Path("input") / "usdz_files" / f"{scan}.usdz"
    scene_usdz.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(original_usdz, scene_usdz)
    print(f"{Colors.GREEN}✓{Colors.RESET} USDZ copied ({time.time() - started:.2f}s)")

    target_folder = Path("input") / "scan_geometry" / scan
    target_folder.mkdir(parents=True, exist_ok=True)
    legacy_obj = raw / "textured_output.obj"
    if legacy_obj.exists():
        shutil.copy(legacy_obj, target_folder / "scene_points.obj")
        for ext, dst in ((".mtl", "scene_points.mtl"), (".jpg", "preview.jpg")):
            src = raw / f"textured_output{ext}"
            if src.exists():
                shutil.copy(src, target_folder / dst)
    else:
        pcd = raw / "pointcloud.pcd"
        if pcd.exists():
            count = _pcd_to_vertex_obj(pcd, target_folder / "scene_points.obj")
            source = "pointcloud.pcd"
        else:
            count = _depth_frames_to_vertex_obj(raw, target_folder / "scene_points.obj")
            source = "depth frames"
            if count == 0:
                raise FileNotFoundError(
                    f"Missing scan point cloud under {raw} (no textured_output.obj, "
                    "pointcloud.pcd, or usable depth frames)"
                )
        (target_folder / "scene_points.mtl").write_text(
            "# Placeholder material for vertex-only point cloud OBJ\n", encoding="utf-8"
        )
        frame = _first_frame(raw)
        if frame:
            shutil.copy(frame, target_folder / "preview.jpg")
        print(f"{Colors.YELLOW}⚠{Colors.RESET} scene_points.obj from {source} ({count} points)")

    extract_rgbd(str(raw), f"input/rgbd/{scan}")
    print(f"{Colors.GREEN}✓{Colors.RESET} RGBD extracted ({time.time() - started:.2f}s)")
    return str(scene_usdz)


# ── RoomPlan door/window openings, rebuilt from the wall-child USDAs ──────────
def _bbox_world_points(pose: dict) -> list[list[float]]:
    bbox = pose["bbox"]
    half = bbox / 2.0
    local = []
    for z in (-half[2], half[2]):
        local.extend(
            [
                [-half[0], -half[1], z, 1.0],
                [half[0], -half[1], z, 1.0],
                [half[0], half[1], z, 1.0],
                [-half[0], half[1], z, 1.0],
            ]
        )
    points = pose["transform"].T @ np.array(local, dtype=float).T
    return points[:3, :].T.tolist()


def _wall_local_dimension(wall_pose: dict, world_points: list[list[float]]) -> list[list[float]]:
    world = np.asarray([[p[0], p[1], p[2], 1.0] for p in world_points], dtype=float)
    local = np.linalg.inv(wall_pose["transform"].T) @ world.T
    xy = local[:2, :].T
    mn = xy.min(axis=0)
    mx = xy.max(axis=0)
    return [[float(mn[0]), float(mn[1])], [float(mx[0]), float(mx[1])]]


def rebuild_roomplan_wall_holes(scene_usdz: str, walls: list[dict]) -> dict:
    """Read RoomPlan wall-child Door/Window USDAs into projection-ready openings."""
    result: dict = {wall["file"]: [] for wall in walls}
    get_scene_usda(scene_usdz)
    for wall in walls:
        wall_file = Path(wall["file"])
        records = []
        for child in sorted(glob.glob(str(wall_file.parent / "*.usda"))):
            child_path = Path(child)
            stem = child_path.stem
            if not (stem.startswith("Door") or stem.startswith("Window")):
                continue
            opening_type = "Door" if stem.startswith("Door") else "Window"
            pose = get_object_pose(str(child_path))
            bbox_points = _bbox_world_points(pose)
            records.append(
                {
                    "name": f"{wall_file.stem}_{opening_type}_{stem[len(opening_type) :]}",
                    "type": opening_type,
                    "dimension": _wall_local_dimension(wall["pose"], bbox_points),
                    "bbox_points": bbox_points,
                }
            )
        if records:
            result[wall["file"]] = {
                "name": [r["name"] for r in records],
                "type": [r["type"] for r in records],
                "dimension": [r["dimension"] for r in records],
                "bbox_points": [r["bbox_points"] for r in records],
            }
    return result


def extract(raw_scan_folder: str, scan: str) -> Path:
    """Run scene extraction for one scan. Assumes CWD == work root (enter_work_root)."""
    scene_usdz = copy_scan_files(raw_scan_folder, scan)
    walls, objects, _wall_holes, floor = extract_scene_data(scene_usdz)
    wall_holes = rebuild_roomplan_wall_holes(scene_usdz, walls)
    opening_count = sum(len(v.get("type", [])) for v in wall_holes.values() if isinstance(v, dict))
    print(f"{Colors.GREEN}✓{Colors.RESET} {len(objects)} objects, {opening_count} openings")
    save_scene_data(scan, walls, objects, wall_holes, floor)
    return config.scene_data_dir(scan)
