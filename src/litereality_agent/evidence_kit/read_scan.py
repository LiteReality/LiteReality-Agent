"""Read a RoomPlan/ARKit capture directory without Blender.

    from read_scan import Scan
    S = Scan("evidence/scan")
    S.indices()            -> [0, 1, ...]           every frame_NNNNN.json
    S.rgb(i)               -> HxWx3 uint8 (stored LANDSCAPE, e.g. 1920x1440; the phone was portrait)
    S.depth(i)             -> hxw float32 metres     (256x192 LiDAR; same field of view as the jpg)
    S.conf(i)              -> hxw uint8 0..2         (ARKit confidence; 2 = high)
    S.K(i)                 -> 3x3 intrinsics for the jpg;  S.K_depth(i) scaled for the depth map
    S.pose(i)              -> 4x4 camera->world, ARKit world (Y up)
    S.pose_blender(i)      -> 4x4 camera->world, Blender world (Z up) = Rx(90) @ pose
    S.pcd()                -> Nx3 float32 ARKit-world points;  S.pcd_blender() in Blender world

Frames: `cameraPoseARFrame` is ROW-MAJOR 4x4 camera->world (translation in the last column). The
camera looks down its local -Z with +Y up — Blender's camera convention — so the same matrix
rotated into Z-up IS the Blender camera matrix. All units metres.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

# ARKit (Y-up) -> Blender (Z-up): y -> z, z -> -y.
C_ARKIT_TO_BLENDER = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)


class Scan:
    def __init__(self, scan_dir: str):
        self.dir = os.path.abspath(scan_dir)
        if not os.path.isdir(self.dir):
            raise FileNotFoundError(self.dir)
        self._pcd = None

    # -- frames -------------------------------------------------------------------------------
    def indices(self) -> list[int]:
        return sorted(int(m.group(1)) for f in os.listdir(self.dir)
                      if (m := re.match(r"frame_(\d+)\.json$", f)))

    def frame(self, i: int) -> dict:
        with open(os.path.join(self.dir, f"frame_{i:05d}.json")) as fh:
            return json.load(fh)

    def rgb_path(self, i: int) -> str:
        return os.path.join(self.dir, f"frame_{i:05d}.jpg")

    def rgb(self, i: int) -> np.ndarray:
        from PIL import Image
        return np.asarray(Image.open(self.rgb_path(i)).convert("RGB"))

    def depth(self, i: int) -> np.ndarray:
        from PIL import Image
        return np.asarray(Image.open(os.path.join(self.dir, f"depth_{i:05d}.png")), dtype=np.float32) / 1000.0

    def conf(self, i: int) -> np.ndarray:
        from PIL import Image
        return np.asarray(Image.open(os.path.join(self.dir, f"conf_{i:05d}.png")))

    def K(self, i: int) -> np.ndarray:
        return np.array(self.frame(i)["intrinsics"], dtype=np.float64).reshape(3, 3)

    def K_depth(self, i: int) -> np.ndarray:
        """Intrinsics rescaled to the depth map's resolution (same field of view as the jpg)."""
        k = self.K(i).copy()
        d = self.depth(i)
        r = self.rgb(i)
        sx, sy = d.shape[1] / r.shape[1], d.shape[0] / r.shape[0]
        k[0, :] *= sx
        k[1, :] *= sy
        return k

    def pose(self, i: int) -> np.ndarray:
        return np.array(self.frame(i)["cameraPoseARFrame"], dtype=np.float64).reshape(4, 4)

    def pose_blender(self, i: int) -> np.ndarray:
        return C_ARKIT_TO_BLENDER @ self.pose(i)

    # -- point cloud --------------------------------------------------------------------------
    def pcd(self) -> np.ndarray:
        if self._pcd is None:
            self._pcd = _read_pcd(os.path.join(self.dir, "pointcloud.pcd"))
        return self._pcd

    def pcd_blender(self) -> np.ndarray:
        p = self.pcd()
        return p @ C_ARKIT_TO_BLENDER[:3, :3].T

    # -- geometry -----------------------------------------------------------------------------
    def unproject_depth(self, i: int, min_conf: int = 1) -> np.ndarray:
        """Every depth pixel of frame i as an Nx3 point in Blender world (Z up)."""
        d = self.depth(i)
        c = self.conf(i)
        k = self.K_depth(i)
        h, w = d.shape
        v, u = np.mgrid[0:h, 0:w]
        ok = (d > 0) & (c >= min_conf)
        z = d[ok]
        x = (u[ok] - k[0, 2]) / k[0, 0] * z
        y = (v[ok] - k[1, 2]) / k[1, 1] * z
        # camera looks down -Z with +Y up: image v grows downward, so flip y and z
        cam = np.stack([x, -y, -z, np.ones_like(z)], axis=1)
        return (self.pose_blender(i) @ cam.T).T[:, :3].astype(np.float32)


def _read_pcd(path: str) -> np.ndarray:
    """ASCII or binary `.pcd` with at least x y z fields (open3d if present, else a small parser)."""
    try:
        import open3d as o3d  # type: ignore
        return np.asarray(o3d.io.read_point_cloud(path).points, dtype=np.float32)
    except Exception:  # noqa: BLE001 — open3d is optional
        pass
    with open(path, "rb") as fh:
        header, fields, data = [], [], "ascii"
        while True:
            line = fh.readline().decode("ascii", errors="replace").strip()
            header.append(line)
            if line.startswith("FIELDS"):
                fields = line.split()[1:]
            if line.startswith("DATA"):
                data = line.split()[1]
                break
        if data != "ascii":
            raise ValueError(f"{path}: binary PCD needs open3d")
        arr = np.loadtxt(fh, dtype=np.float32)
    ix = [fields.index(a) for a in ("x", "y", "z")]
    return arr[:, ix]
