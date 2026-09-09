"""wall_stitch_sdk -- standalone rectified wall-image stitching from a RoomPlan scan.

A single-file, dependency-light SDK that turns a captured RoomPlan scene into a
flat, head-on ("rectified ortho") image per wall, plus known/unknown coverage
masks. No dependency on any other module in this repo.

It is NOT panorama/feature stitching. For each wall it:
  1. reads the wall's plane + size from the RoomPlan USDZ geometry,
  2. lays a metric ortho pixel grid on that plane (each output pixel is a 3D
     point on the wall),
  3. projects every capture frame onto the grid, scoring each frame per pixel
     (head-on, central, close, and depth-visible win),
  4. fills each output pixel with the single best frame -- best-frames-first,
     fill-only, no blending. Unobserved pixels stay blank.

Dependencies: numpy, Pillow.  (Python 3.9+)

Inputs -- a captured scene folder containing:
    room.usdz                       RoomPlan geometry (walls)
    frame_<id>.jpg, frame_<id>.json  RGB + {"intrinsics":[9], "cameraPoseARFrame":[16]}
    depth_<id>.png, conf_<id>.png    ARKit depth (uint16 mm) + confidence  [optional]

Library use:
    from wall_stitch_sdk import stitch_walls
    results = stitch_walls("/path/to/scene", out_dir="out", pixels_per_meter=220)

CLI:
    python wall_stitch_sdk.py /path/to/scene --out out --ppm 220
"""

from __future__ import annotations

import argparse
import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw

__all__ = ["WallPlane", "find_frame_pairs", "load_wall_planes", "stitch_wall", "stitch_walls"]

_NUMBER_RE = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


# --------------------------------------------------------------------------- #
# 1. RoomPlan USDZ -> wall planes
# --------------------------------------------------------------------------- #
@dataclass
class WallPlane:
    """An oriented, rectified wall surface: a frame + metric size."""

    name: str
    center: np.ndarray  # world-space center (3,)
    u_axis: np.ndarray  # unit horizontal axis (3,), +x-ish
    v_axis: np.ndarray  # unit vertical axis (3,), +up
    normal: np.ndarray  # unit outward normal (3,)
    width_m: float
    height_m: float


_CUBE_LOCAL = np.array(
    [
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ],
    dtype=float,
)


def _parse_numbers(text: str) -> list[float]:
    return [float(v) for v in re.findall(_NUMBER_RE, text)]


def _parse_matrix4d(usda: str) -> np.ndarray:
    m = re.search(r"matrix4d\s+xformOp:transform\s*=\s*\(\s*\((.*?)\)\s*\)", usda, re.S)
    if not m:
        return np.eye(4, dtype=float)
    vals = _parse_numbers(m.group(1))
    if len(vals) != 16:
        raise ValueError(f"expected 16 matrix values, found {len(vals)}")
    return np.array(vals, dtype=float).reshape(4, 4)


def _wall_corners(scale: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """8 world-space corners of a RoomPlan wall box. USD places translation in the
    final row, so transform with row-vectors (homogeneous @ matrix)."""
    local = _CUBE_LOCAL * scale
    homo = np.c_[local, np.ones(len(local))]
    return (homo @ transform)[:, :3]


def load_wall_planes(usdz_path: str | Path) -> list[WallPlane]:
    """Parse a RoomPlan USDZ and return one WallPlane per wall."""
    usdz_path = Path(usdz_path)
    planes: list[WallPlane] = []
    with zipfile.ZipFile(usdz_path) as archive:
        for name in archive.namelist():
            if not re.search(r"assets/Parametric/Walls/(Wall\d+)/(Wall\d+)\.usda$", name):
                continue
            usda = archive.read(name).decode("utf-8")
            scale_m = re.search(r"double3\s+xformOp:scale\s*=\s*\((.*?)\)", usda, re.S)
            if not scale_m:
                continue
            scale = np.array(_parse_numbers(scale_m.group(1)), dtype=float)
            if len(scale) != 3:
                continue
            corners = _wall_corners(scale, _parse_matrix4d(usda))
            planes.append(_plane_from_corners(Path(name).stem, corners))
    planes.sort(key=lambda p: p.name)
    return planes


def _plane_from_corners(name: str, corners: np.ndarray) -> WallPlane:
    """Oriented plane (horizontal/vertical/normal + size) from a wall's 8 corners,
    via SVD of the centered point cloud (a wall is a thin, rotated box)."""
    center = corners.mean(axis=0)
    centered = corners - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axes = vh
    extents = np.array([np.ptp(centered @ ax) for ax in axes])

    v_idx = int(np.argmax(np.abs(axes @ np.array([0.0, 1.0, 0.0]))))
    vertical = axes[v_idx]
    if vertical[1] < 0:
        vertical = -vertical
    remaining = [i for i in range(3) if i != v_idx]
    h_idx = max(remaining, key=lambda i: extents[i])
    horizontal = axes[h_idx]
    if horizontal[0] < 0:
        horizontal = -horizontal
    normal = np.cross(horizontal, vertical)
    n = np.linalg.norm(normal)
    if n < 1e-8:
        raise ValueError(f"degenerate wall normal for {name}")
    normal = normal / n
    return WallPlane(
        name, center, horizontal, vertical, normal, float(extents[h_idx]), float(extents[v_idx])
    )


# --------------------------------------------------------------------------- #
# 2. Frames: discovery, intrinsics/pose, projection, sampling
# --------------------------------------------------------------------------- #
def find_frame_pairs(scene_dir: str | Path) -> list[tuple[Path, Path]]:
    """All (rgb_jpg, meta_json) capture pairs in a scene dir (also checks images/)."""
    scene_dir = Path(scene_dir)
    pairs: list[tuple[Path, Path]] = []
    for json_path in sorted(scene_dir.glob("frame_*.json")):
        jpg = json_path.with_suffix(".jpg")
        if jpg.exists():
            pairs.append((jpg, json_path))
        elif (scene_dir / "images" / jpg.name).exists():
            pairs.append((scene_dir / "images" / jpg.name, json_path))
    return pairs


def _load_frame_meta(json_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(json_path.read_text())
    intrinsics = np.array(data["intrinsics"], dtype=float).reshape(3, 3)
    pose = np.array(data["cameraPoseARFrame"], dtype=float).reshape(4, 4)
    return intrinsics, pose


def _world_to_camera(points: np.ndarray, camera_pose: np.ndarray) -> np.ndarray:
    homo = np.c_[np.nan_to_num(points), np.ones(len(points))]
    cam_from_world = np.nan_to_num(np.linalg.inv(camera_pose))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        cam = (cam_from_world @ homo.T).T[:, :3]
    cam[~np.isfinite(cam)] = np.nan
    return cam


def _project(camera_points: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    depth = -camera_points[:, 2]
    px = np.empty((len(camera_points), 2), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        px[:, 0] = intrinsics[0, 0] * camera_points[:, 0] / depth + intrinsics[0, 2]
        px[:, 1] = intrinsics[1, 1] * (-camera_points[:, 1]) / depth + intrinsics[1, 2]
    px[~np.isfinite(px)] = np.nan
    return px


def _bilinear_sample(image: np.ndarray, pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    x, y = pixels[:, 0], pixels[:, 1]
    valid = (x >= 0) & (x <= w - 2) & (y >= 0) & (y <= h - 2)
    colors = np.zeros((len(pixels), 3), dtype=np.float32)
    if not np.any(valid):
        return colors, valid
    xv, yv = x[valid], y[valid]
    x0 = np.floor(xv).astype(np.int32)
    y0 = np.floor(yv).astype(np.int32)
    dx = (xv - x0)[:, None]
    dy = (yv - y0)[:, None]
    c00 = image[y0, x0].astype(np.float32)
    c10 = image[y0, x0 + 1].astype(np.float32)
    c01 = image[y0 + 1, x0].astype(np.float32)
    c11 = image[y0 + 1, x0 + 1].astype(np.float32)
    colors[valid] = (
        c00 * (1 - dx) * (1 - dy) + c10 * dx * (1 - dy) + c01 * (1 - dx) * dy + c11 * dx * dy
    )
    return colors, valid


def _sample_depth_m(depth_image, conf_image, pixels, image_size):
    image_w, image_h = image_size
    dh, dw = depth_image.shape[:2]
    finite = np.all(np.isfinite(pixels), axis=1) & np.all(np.abs(pixels) < 1e6, axis=1)
    x = np.zeros(len(pixels), dtype=np.int32)
    y = np.zeros(len(pixels), dtype=np.int32)
    if np.any(finite):
        x[finite] = np.rint(pixels[finite, 0] * dw / image_w).astype(np.int32)
        y[finite] = np.rint(pixels[finite, 1] * dh / image_h).astype(np.int32)
    valid = finite & (x >= 0) & (x < dw) & (y >= 0) & (y < dh)
    depth_m = np.full(len(pixels), np.nan, dtype=np.float32)
    if np.any(valid):
        raw = depth_image[y[valid], x[valid]].astype(np.float32)
        depth_m[valid] = raw / 1000.0  # ARKit depth PNG is uint16 millimetres
        valid[valid] &= raw > 0
    if conf_image is not None and np.any(valid):
        conf = np.zeros(len(pixels), dtype=np.uint8)
        conf[valid] = conf_image[y[valid], x[valid]]
        valid &= conf > 0
    return depth_m, valid


def _depth_path(image_path: Path, prefix: str) -> Path:
    return image_path.with_name(f"{prefix}_{image_path.stem.replace('frame_', '')}.png")


# --------------------------------------------------------------------------- #
# 3. The stitch
# --------------------------------------------------------------------------- #
def _plane_grid(plane: WallPlane, ppm: int):
    """Regular ortho grid of world points across the wall (and an all-True mask)."""
    w = max(64, int(math.ceil(plane.width_m * ppm)))
    h = max(64, int(math.ceil(plane.height_m * ppm)))
    xs = (np.arange(w, dtype=np.float32) + 0.5) / w - 0.5
    ys = 0.5 - (np.arange(h, dtype=np.float32) + 0.5) / h
    uu, vv = np.meshgrid(xs * plane.width_m, ys * plane.height_m)
    points = plane.center + uu[..., None] * plane.u_axis + vv[..., None] * plane.v_axis
    return points.reshape(-1, 3), w, h


def stitch_wall(
    plane: WallPlane,
    frame_pairs: Sequence[tuple[Path, Path]],
    out_path: str | Path,
    pixels_per_meter: int = 220,
    max_frames: int | None = None,
    use_depth_occlusion: bool = True,
    depth_tolerance_m: float = 0.08,
    annotate: bool = True,
    fill_holes: bool = True,
    relax_facing: float = 0.05,
) -> dict:
    """Stitch one wall -> writes <out_path> (+ _known_mask.png / _unknown_mask.png).
    Returns a result dict (coverage, size, paths, frames_used).

    Coverage is built in two best-frames-first passes: a STRICT pass (depth-verified,
    head-on facing > 0.12) lays down clean pixels, then — if ``fill_holes`` — a RELAXED
    pass (depth-occlusion dropped, grazing allowed down to ``relax_facing``) fills only
    the pixels the strict pass left blank. Clean observations always win; the relaxed
    pixels just plug the holes that noisy/missing ARKit depth and oblique-only views
    would otherwise leave unobserved."""
    out_path = Path(out_path)
    points, W, H = _plane_grid(plane, pixels_per_meter)
    n = len(points)
    best_score = np.zeros(n, dtype=np.float32)
    best_color = np.zeros((n, 3), dtype=np.float32)
    observations = np.zeros(n, dtype=np.uint16)
    frame_obs: list[dict] = []
    used = 0

    pairs = frame_pairs[:max_frames] if max_frames is not None else frame_pairs
    for image_path, json_path in pairs:
        image = np.asarray(Image.open(image_path).convert("RGB"))
        img_h, img_w = image.shape[:2]
        intrinsics, pose = _load_frame_meta(json_path)
        cam = _world_to_camera(points, pose)
        depth = -cam[:, 2]
        in_front = depth > 0.08
        pixels = _project(cam, intrinsics)
        sampled, inside = _bilinear_sample(image, pixels)

        depth_visible = np.ones(n, dtype=bool)
        # OCCLUDED = depth data EXISTS and measures something clearly in FRONT of the wall point
        # (a chair/desk/etc). Used to keep foreground objects out of BOTH passes — the relaxed
        # hole-fill must still drop these, only relaxing where depth is genuinely MISSING.
        confirmed_occluded = np.zeros(n, dtype=bool)
        if use_depth_occlusion:
            dpath, cpath = _depth_path(image_path, "depth"), _depth_path(image_path, "conf")
            if dpath.exists():
                depth_img = np.asarray(Image.open(dpath))
                conf_img = np.asarray(Image.open(cpath)) if cpath.exists() else None
                obs_depth, dvalid = _sample_depth_m(depth_img, conf_img, pixels, (img_w, img_h))
                depth_visible = dvalid & (obs_depth + depth_tolerance_m >= depth)
                # wall point is >occl_margin BEHIND the measured surface → a foreground object
                # (chair/desk) sits in front. Margin > tol so depth noise doesn't punch holes.
                occl_margin = max(depth_tolerance_m, 0.15)
                confirmed_occluded = dvalid & (depth - obs_depth > occl_margin)

        cam_pos = pose[:3, 3]
        to_cam = cam_pos[None, :] - points
        dist = np.linalg.norm(to_cam, axis=1)
        to_cam /= np.maximum(dist[:, None], 1e-6)
        to_cam = np.nan_to_num(to_cam)
        with np.errstate(invalid="ignore", over="ignore"):
            facing = np.abs(np.sum(to_cam * np.nan_to_num(plane.normal)[None, :], axis=1))
        facing[~np.isfinite(facing)] = 0.0
        margin = np.minimum.reduce(
            [pixels[:, 0], pixels[:, 1], img_w - 1 - pixels[:, 0], img_h - 1 - pixels[:, 1]]
        )
        edge_weight = np.clip(margin / 120.0, 0.0, 1.0)
        score = (facing**2) * edge_weight / np.maximum(dist, 0.3)  # head-on, central, close
        score[~np.isfinite(score)] = 0.0

        # relaxed (hole-fill) base now EXCLUDES depth-confirmed foreground occluders, so chairs /
        # desks in front of the wall no longer get smeared onto the plane; only genuinely
        # missing-depth holes get filled from oblique frames.
        base = in_front & inside & (facing > relax_facing) & ~confirmed_occluded
        strict = base & depth_visible & (facing > 0.12)
        scount = int(np.count_nonzero(strict))
        rcount = int(np.count_nonzero(base))
        if scount or rcount:
            frame_obs.append(
                {
                    "strict": strict,
                    "relaxed": base,
                    "score": score,
                    "sampled": sampled,
                    "squality": float(np.mean(score[strict]) * scount) if scount else 0.0,
                    "rquality": float(np.mean(score[base]) * rcount) if rcount else 0.0,
                }
            )
        observations += strict.astype(np.uint16)
        used += 1

    # best-frames-first, fill-only (no blending, no overwrite). Pass 1 STRICT lays clean
    # depth-verified pixels; pass 2 RELAXED fills only the still-blank holes.
    passes = [("strict", "squality")] + ([("relaxed", "rquality")] if fill_holes else [])
    for mask_key, q_key in passes:
        for it in sorted(frame_obs, key=lambda d: d[q_key], reverse=True):
            fill = it[mask_key] & (best_score <= 0)
            if np.any(fill):
                best_score[fill] = np.maximum(it["score"][fill], 1e-6)  # >0 marks it filled
                best_color[fill] = it["sampled"][fill]

    filled = best_score > 0
    canvas = np.full((n, 3), 236, dtype=np.uint8)  # unobserved = light gray
    canvas[filled] = np.clip(best_color[filled], 0, 255).astype(np.uint8)
    out = canvas.reshape(H, W, 3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(out)
    if annotate:
        d = ImageDraw.Draw(img, "RGBA")
        d.rectangle((0, 0, img.width - 1, img.height - 1), outline=(40, 40, 40, 220), width=2)
        d.text(
            (12, 10),
            plane.name,
            fill=(255, 255, 255, 255),
            stroke_width=3,
            stroke_fill=(0, 0, 0, 180),
        )
    img.save(out_path, quality=94)

    known = (filled.astype(np.uint8) * 255).reshape(H, W)
    unknown = ((~filled).astype(np.uint8) * 255).reshape(H, W)
    known_path = out_path.with_name(f"{out_path.stem}_known_mask.png")
    unknown_path = out_path.with_name(f"{out_path.stem}_unknown_mask.png")
    Image.fromarray(known).save(known_path)
    Image.fromarray(unknown).save(unknown_path)

    return {
        "name": plane.name,
        "image": str(out_path),
        "known_mask": str(known_path),
        "unknown_mask": str(unknown_path),
        "size_px": [W, H],
        "surface_size_m": [plane.width_m, plane.height_m],
        "frames_used": used,
        "coverage": float(np.count_nonzero(filled) / n),
        "strict_coverage": float(np.count_nonzero(observations > 0) / n),
        "mean_observations_per_pixel": float(np.mean(observations)),
        "stitch_strategy": (
            "best_frames_first_fill_only_no_blend"
            + ("_strict+relaxed_holefill" if fill_holes else "")
        ),
    }


def stitch_walls(
    scene_dir: str | Path,
    out_dir: str | Path | None = None,
    usdz: str | Path | None = None,
    pixels_per_meter: int = 220,
    max_frames: int | None = None,
    use_depth_occlusion: bool = True,
    depth_tolerance_m: float = 0.08,
    wall_names: Sequence[str] | None = None,
    annotate: bool = True,
    fill_holes: bool = True,
    relax_facing: float = 0.05,
) -> list[dict]:
    """Stitch every wall of a captured RoomPlan scene. Returns per-wall result dicts."""
    scene_dir = Path(scene_dir)
    usdz = Path(usdz) if usdz else scene_dir / "room.usdz"
    out_dir = Path(out_dir) if out_dir else scene_dir / "wall_stitches"
    planes = load_wall_planes(usdz)
    if wall_names is not None:
        want = {w.lower() for w in wall_names}
        planes = [p for p in planes if p.name.lower() in want]
    if not planes:
        raise ValueError(f"no walls found in {usdz}")
    frame_pairs = find_frame_pairs(scene_dir)
    if not frame_pairs:
        raise ValueError(f"no frame_*.jpg/json pairs in {scene_dir}")
    results = []
    for plane in planes:
        info = stitch_wall(
            plane,
            frame_pairs,
            out_dir / f"{plane.name}_stitched.jpg",
            pixels_per_meter=pixels_per_meter,
            max_frames=max_frames,
            use_depth_occlusion=use_depth_occlusion,
            depth_tolerance_m=depth_tolerance_m,
            annotate=annotate,
            fill_holes=fill_holes,
            relax_facing=relax_facing,
        )
        results.append(info)
    return results


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stitch rectified wall images from a RoomPlan scan.")
    ap.add_argument(
        "scene_folder",
        type=Path,
        help="dir with room.usdz + frame_*.jpg/json (+ depth_*/conf_*.png)",
    )
    ap.add_argument("--usdz", type=Path, default=None, help="defaults to <scene>/room.usdz")
    ap.add_argument("--out", type=Path, default=None, help="defaults to <scene>/wall_stitches")
    ap.add_argument("--ppm", type=int, default=220, help="pixels per meter (ortho resolution)")
    ap.add_argument("--wall", action="append", default=None, help="only this wall (repeatable)")
    ap.add_argument("--limit-frames", type=int, default=None)
    ap.add_argument("--no-depth-occlusion", action="store_true")
    ap.add_argument("--depth-tolerance-m", type=float, default=0.08)
    ap.add_argument(
        "--no-fill-holes",
        action="store_true",
        help="disable the relaxed second pass (strict pixels only — more holes)",
    )
    ap.add_argument(
        "--relax-facing",
        type=float,
        default=0.05,
        help="min facing for the hole-fill pass (lower = fills more, more grazing)",
    )
    ap.add_argument(
        "--no-annotate", action="store_true", help="omit the name/border overlay (clean texture)"
    )
    a = ap.parse_args(argv)
    results = stitch_walls(
        a.scene_folder,
        a.out,
        usdz=a.usdz,
        pixels_per_meter=a.ppm,
        max_frames=a.limit_frames,
        use_depth_occlusion=not a.no_depth_occlusion,
        depth_tolerance_m=a.depth_tolerance_m,
        wall_names=a.wall,
        annotate=not a.no_annotate,
        fill_holes=not a.no_fill_holes,
        relax_facing=a.relax_facing,
    )
    for r in results:
        print(
            f"  {r['name']:8} {r['surface_size_m'][0]:.2f}x{r['surface_size_m'][1]:.2f}m  "
            f"coverage {r['coverage']:.1%} -> {Path(r['image']).name}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
