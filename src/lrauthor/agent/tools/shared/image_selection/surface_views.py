"""surface_views.py — per-surface image selection for walls / floors / ceilings.

For each surface it picks the MINIMAL set of SHARP capture frames that together COVER the
surface, draws the surface BOXED + labelled (reference-overlay style), and reports two
quality signals the prompt needs:
  - sharpness  → a per-image `blurry` flag when the best available view is soft
  - coverage % → how much of the surface the chosen images actually show (low = thin capture)

Tiny walls (< MIN_WALL_AREA m^2 or any side < MIN_WALL_SIDE m) are skipped — not worth a pass.

  python authoring/views/room_render/surface_views.py <scan>          # -> sheet.jpg + manifest.json
Reuses the wall-plane + camera + depth math from stitch_wall (floor/ceiling planes are
derived from the wall corners — RoomPlan has no ceiling and a mesh floor).
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lrauthor.agent.tools.shared.stitch_wall_image import stitch_wall as SW
from lrauthor.fonts import font as _shared_font

NEAR = 0.05
BIN_M = 0.45  # ~bin size on the surface (m); coverage measured over these bins
BLUR_THRESH = 60.0  # Laplacian-variance below this (on the surface region) = blurry
LOWCOV_PCT = 60.0  # selected images cover < this % of the surface = low coverage
MIN_BIN_FRAC = 0.12  # a frame must show >= this fraction of bins to be a candidate
MIN_WALL_AREA = 0.8  # m^2 — skip walls smaller than this
MIN_WALL_SIDE = 0.4  # m
TOL = 0.2  # depth-occlusion tolerance (m)
MAXN = 3  # at most this many frames per surface


def _font(s):
    try:
        return _shared_font(s)
    except Exception:
        return ImageFont.load_default()


FONT, CHIP, HFONT = _font(28), _font(46), _font(38)


def wall_corners(p):
    u = p.u_axis * p.width_m / 2
    v = p.v_axis * p.height_m / 2
    return np.array([p.center - u - v, p.center + u - v, p.center + u + v, p.center - u + v])


def derive_floor_ceiling(planes):
    c = np.vstack([wall_corners(p) for p in planes])
    yf, yc = float(c[:, 1].min()), float(c[:, 1].max())
    xmn, xmx = float(c[:, 0].min()), float(c[:, 0].max())
    zmn, zmx = float(c[:, 2].min()), float(c[:, 2].max())
    cx, cz = (xmn + xmx) / 2, (zmn + zmx) / 2
    w, dpt = max(xmx - xmn, 0.2), max(zmx - zmn, 0.2)
    F = SW.WallPlane(
        "Floor0",
        np.array([cx, yf, cz]),
        np.array([1.0, 0, 0]),
        np.array([0, 0, -1.0]),
        np.array([0, 1.0, 0]),
        w,
        dpt,
    )
    C = SW.WallPlane(
        "Ceiling0",
        np.array([cx, yc, cz]),
        np.array([1.0, 0, 0]),
        np.array([0, 0, 1.0]),
        np.array([0, -1.0, 0]),
        w,
        dpt,
    )
    return F, C


def surface_grid(p):
    Nu = min(10, max(2, round(p.width_m / BIN_M)))
    Nv = min(10, max(2, round(p.height_m / BIN_M)))
    pts, binof = [], []
    for iu in range(Nu):
        for iv in range(Nv):
            u, v = (iu + 0.5) / Nu, (iv + 0.5) / Nv
            pts.append(
                p.center + (u - 0.5) * p.width_m * p.u_axis + (v - 0.5) * p.height_m * p.v_axis
            )
            binof.append(iu * Nv + iv)
    return np.array(pts), np.array(binof), Nu * Nv


def analyze(plane, pairs):
    pts, binof, total = surface_grid(plane)
    min_bins = max(2, int(total * MIN_BIN_FRAC))
    rows = []
    for img_p, json_p in pairs:
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        H, W = img.shape[:2]
        intr, pose = SW._load_frame_meta(json_p)
        cam = SW._world_to_camera(pts, pose)
        depth = -cam[:, 2]
        px = SW._project(cam, intr)
        fin = np.all(np.isfinite(px), axis=1)
        inimg = fin & (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
        cp = pose[:3, 3]
        tc = cp[None, :] - pts
        d = np.linalg.norm(tc, axis=1)
        tc = np.nan_to_num(tc / np.maximum(d[:, None], 1e-6))
        facing = np.abs(np.sum(tc * np.nan_to_num(plane.normal)[None, :], axis=1)) > 0.12
        dvis = np.ones(len(pts), bool)
        dp = SW._depth_path(img_p, "depth")
        if dp.exists():
            dm = np.asarray(Image.open(dp))
            cpath = SW._depth_path(img_p, "conf")
            cm = np.asarray(Image.open(cpath)) if cpath.exists() else None
            od, ok = SW._sample_depth_m(dm, cm, px, (W, H))
            dvis = ok & (od + TOL >= depth)
        vis = (depth > NEAR) & inimg & facing & dvis
        bins = set(binof[vis].tolist())
        if len(bins) < min_bins:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        mask = np.zeros((H, W), np.uint8)
        vp = px[vis].astype(int)
        vp = vp[(vp[:, 0] >= 0) & (vp[:, 0] < W) & (vp[:, 1] >= 0) & (vp[:, 1] < H)]
        mask[vp[:, 1], vp[:, 0]] = 1
        mask = cv2.dilate(mask, np.ones((15, 15), np.uint8))
        sharp = float(lap[mask > 0].var()) if mask.sum() > 200 else 0.0
        rows.append(
            {
                "frame": img_p.stem,
                "img_p": img_p,
                "json_p": json_p,
                "bins": bins,
                "sharp": round(sharp, 1),
                "ncov": len(bins),
            }
        )
    return rows, total


def select(rows):
    """Greedy set-cover preferring sharper frames; includes a soft frame only when it adds
    unique coverage (then it's flagged blurry). Cap MAXN."""
    if not rows:
        return []
    cand = list(rows)
    coverable = set().union(*[r["bins"] for r in rows])
    remaining = set(coverable)
    chosen = []
    while remaining and len(chosen) < MAXN:
        best = max(cand, key=lambda r: (len(r["bins"] & remaining), r["sharp"]))
        if not (best["bins"] & remaining):
            break
        chosen.append(best)
        remaining -= best["bins"]
        cand.remove(best)
    return chosen


def _proj_edge(aw, bw, pose, intr):
    cam = SW._world_to_camera(np.array([aw, bw]), pose).astype(float)
    za, zb = cam[0, 2], cam[1, 2]
    da, db = -za, -zb
    if da < NEAR and db < NEAR:
        return None
    if da < NEAR or db < NEAR:
        t = (-NEAR - za) / (zb - za + 1e-9)
        clip = cam[0] + t * (cam[1] - cam[0])
        cam = np.array([clip, cam[1]]) if da < NEAR else np.array([cam[0], clip])
    p = SW._project(cam, intr)
    return [tuple(p[0]), tuple(p[1])] if np.all(np.isfinite(p)) else None


def _chip(d, name, x, y, W, H):
    tb = d.textbbox((0, 0), name, font=CHIP)
    w, h = tb[2] - tb[0], tb[3] - tb[1]
    x = min(max(x, w / 2 + 14), W - w / 2 - 14)
    y = min(max(y, 24), H - h - 24)
    x0, y0 = x - w / 2 - 10, y - 6
    d.rounded_rectangle((x0, y0, x0 + w + 20, y0 + h + 14), radius=9, fill=(0, 0, 0))
    d.text((x0 + 10, y0 + 4), name, fill=(255, 170, 60), font=CHIP)


def highlight(plane, r):
    img = Image.open(r["img_p"]).convert("RGB")
    W0, H0 = img.size
    intr, pose = SW._load_frame_meta(r["json_p"])
    c = wall_corners(plane)
    d = ImageDraw.Draw(img)
    pts = []
    for i in range(4):
        seg = _proj_edge(c[i], c[(i + 1) % 4], pose, intr)
        if seg:
            d.line(seg, fill=(255, 150, 40), width=7)
            pts += seg
    out = img.transpose(Image.Transpose.ROTATE_270)
    Wr, Hr = out.size
    d2 = ImageDraw.Draw(out)
    if pts:
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        _chip(d2, plane.name, H0 - 1 - cy, cx, Wr, Hr)
    blur = " · BLURRY" if r["sharp"] < BLUR_THRESH else ""
    cap = f"{plane.name} · {r['frame']} · sharp {r['sharp']}{blur}"
    d2.rectangle((0, Hr - 40, Wr, Hr), fill=(0, 0, 0))
    d2.text((12, Hr - 36), cap, fill=(255, 180, 120) if blur else (220, 226, 234), font=FONT)
    return out
