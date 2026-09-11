"""Measure the room from the scan. Everything returned is in BLENDER world (Z up), metres.

    from read_scan import Scan; import measure
    S = Scan("evidence/scan")
    measure.probe_depth(S, 12, 960, 700)        -> {"point": [x,y,z], "depth_m": 1.83, "conf": 2}
    measure.triangulate(S, [(12, (960,700)), (14, (1010,650))])   -> {"point": [...], "residual_m": 0.01}
    measure.pcd_slice(S, z0=0.70, z1=0.80)       -> occupancy map of everything between two heights
    measure.height_profile(S, (x0,y0,x1,y1))     -> histogram of point heights inside a footprint

Pixel coordinates are in the STORED landscape jpg (u right, v down, e.g. 1920x1440). A pixel in a
contact-sheet thumbnail must be scaled back to the full frame first.
"""
from __future__ import annotations

import numpy as np


def _floor_z(S, bin_m: float = 0.02) -> float | None:
    """Height of the floor plane from the cloud: the strongest band in the lowest dense 15 cm.

    LiDAR leaves stray returns well below the real floor (Office_room: 30 cm under it) and the
    floor band itself is ~±3 cm wide, so neither the lowest point nor a low percentile works. The
    floor is the largest plane in the room, so it is the strongest bin near the bottom. Expect
    ±3 cm; when Room.py's `SHELL["floor_z"]` (RoomPlan's floor) is available, prefer it and pass
    it as `floor_z=` to the helpers below."""
    p = S.pcd_blender()
    if len(p) < 100:
        return None
    z = p[:, 2]
    edges = np.arange(float(z.min()), float(z.max()) + bin_m, bin_m)
    hist, _ = np.histogram(z, bins=edges)
    dense = np.flatnonzero(hist >= 0.10 * hist.max())
    if not len(dense):
        return float(np.percentile(z, 1.0))
    lo = dense[0]
    hi = min(len(hist), lo + int(round(0.15 / bin_m)))
    j = lo + int(np.argmax(hist[lo:hi]))
    a, b = max(0, j - 1), min(len(hist), j + 2)
    w = hist[a:b].astype(float)
    centres = edges[a:b] + bin_m / 2
    return float((w * centres).sum() / w.sum())


def floor_z(S) -> float:
    """Blender-world z of the floor, estimated from the point cloud (±3 cm)."""
    return _floor_z(S) or 0.0


def probe_depth(S, i: int, u: float, v: float, radius_px: int = 2) -> dict:
    """The 3D point the LiDAR saw at jpg pixel (u, v) of frame i (median over a small window)."""
    d = S.depth(i)
    c = S.conf(i)
    rgb_h, rgb_w = S.rgb(i).shape[:2]
    h, w = d.shape
    du, dv = u * w / rgb_w, v * h / rgb_h
    ui, vi = int(round(du)), int(round(dv))
    u0, u1 = max(0, ui - radius_px), min(w, ui + radius_px + 1)
    v0, v1 = max(0, vi - radius_px), min(h, vi + radius_px + 1)
    win = d[v0:v1, u0:u1]
    cw = c[v0:v1, u0:u1]
    ok = win > 0
    if not ok.any():
        return {"point": None, "depth_m": None, "conf": None, "note": "no depth at this pixel"}
    z = float(np.median(win[ok]))
    k = S.K_depth(i)
    x = (du - k[0, 2]) / k[0, 0] * z
    y = (dv - k[1, 2]) / k[1, 1] * z
    cam = np.array([x, -y, -z, 1.0])
    P = (S.pose_blender(i) @ cam)[:3]
    return {"point": [round(float(a), 4) for a in P], "depth_m": round(z, 4),
            "conf": int(cw[ok].min()), "frame": i, "pixel": [u, v]}


def ray(S, i: int, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
    """(origin, unit direction) of the pixel ray in Blender world."""
    k = S.K(i)
    dcam = np.array([(u - k[0, 2]) / k[0, 0], -(v - k[1, 2]) / k[1, 1], -1.0])
    M = S.pose_blender(i)
    dw = M[:3, :3] @ dcam
    return M[:3, 3].copy(), dw / np.linalg.norm(dw)


def triangulate(S, observations: list[tuple[int, tuple[float, float]]]) -> dict:
    """Least-squares intersection of the pixel rays [(frame, (u, v)), ...] from 2+ frames.

    Use it for things LiDAR misses (dark fabric, glass, thin legs): click the same feature in two
    or more photos. `residual_m` is the RMS distance from the point to the rays — above ~5 cm the
    clicks probably do not show the same feature."""
    if len(observations) < 2:
        raise ValueError("triangulate needs the same feature in at least two frames")
    A = np.zeros((3, 3))
    b = np.zeros(3)
    rays = []
    for i, (u, v) in observations:
        o, d = ray(S, i, u, v)
        rays.append((o, d))
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ o
    X = np.linalg.lstsq(A, b, rcond=None)[0]
    res = [float(np.linalg.norm((np.eye(3) - np.outer(d, d)) @ (X - o))) for o, d in rays]
    return {"point": [round(float(a), 4) for a in X],
            "residual_m": round(float(np.sqrt(np.mean(np.square(res)))), 4),
            "n_rays": len(rays)}


def pcd_slice(S, z0: float, z1: float, cell_m: float = 0.02, floor_relative: bool = True,
              out_png: str | None = None, floor_z: float | None = None) -> dict:
    """Occupancy map of the point cloud between heights z0..z1 (metres above the floor by default).

    Returns {"grid": HxW uint8 counts, "x0","y0","cell_m", "floor_z"} and optionally writes a PNG
    (white = points). A slice at tabletop height draws every tabletop; at 0.05-0.30 m it draws
    legs and plinths; a thin slice just under a shelf finds its underside."""
    p = S.pcd_blender()
    fz = (floor_z if floor_z is not None else _floor_z(S)) if floor_relative else 0.0
    fz = fz or 0.0
    sel = p[(p[:, 2] >= fz + z0) & (p[:, 2] <= fz + z1)]
    x0, y0 = float(p[:, 0].min()), float(p[:, 1].min())
    W = int(np.ceil((p[:, 0].max() - x0) / cell_m)) + 1
    H = int(np.ceil((p[:, 1].max() - y0) / cell_m)) + 1
    grid = np.zeros((H, W), dtype=np.uint16)
    if len(sel):
        ix = ((sel[:, 0] - x0) / cell_m).astype(int)
        iy = ((sel[:, 1] - y0) / cell_m).astype(int)
        np.add.at(grid, (iy, ix), 1)
    out = {"grid": grid, "x0": x0, "y0": y0, "cell_m": cell_m, "floor_z": fz, "n_points": int(len(sel)),
           "note": "grid[row, col]: row = y index (world y = y0 + row*cell), col = x index"}
    if out_png:
        from PIL import Image
        img = (np.clip(grid, 0, 1) * 255).astype(np.uint8)[::-1]  # y up on the page
        Image.fromarray(img).save(out_png)
        out["png"] = out_png
    return out


def height_profile(S, footprint_xy: tuple[float, float, float, float], bin_m: float = 0.02,
                   floor_z: float | None = None) -> dict:
    """Histogram of point heights (above the floor) inside the footprint (x0, y0, x1, y1).

    Peaks are horizontal surfaces: a desk shows one at ~0.72-0.76 m, a shelf unit one per board.
    Pass `floor_z=SHELL["floor_z"]` from Room.py when you have it; the cloud's own estimate is ±3 cm.
    `surfaces_m` lists the peak heights, strongest first."""
    p = S.pcd_blender()
    fz = floor_z if floor_z is not None else (_floor_z(S) or 0.0)
    x0, y0, x1, y1 = footprint_xy
    sel = p[(p[:, 0] >= min(x0, x1)) & (p[:, 0] <= max(x0, x1)) & (p[:, 1] >= min(y0, y1)) & (p[:, 1] <= max(y0, y1))]
    if not len(sel):
        return {"surfaces_m": [], "hist": [], "floor_z": fz, "n_points": 0}
    h = sel[:, 2] - fz
    edges = np.arange(0.0, float(h.max()) + bin_m, bin_m)
    hist, _ = np.histogram(h, bins=edges)
    peaks = []
    for j in range(1, len(hist) - 1):
        if hist[j] >= hist[j - 1] and hist[j] >= hist[j + 1] and hist[j] > max(5, 0.05 * hist.max()):
            peaks.append((int(hist[j]), round(float(edges[j] + bin_m / 2), 3)))
    peaks.sort(reverse=True)
    return {"surfaces_m": [z for _, z in peaks[:8]], "counts": [n for n, _ in peaks[:8]],
            "hist": hist.tolist(), "bin_m": bin_m, "floor_z": fz, "n_points": int(len(sel))}
