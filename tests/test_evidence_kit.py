"""The evidence kit measures the scan the way the one-shot run did by hand — and gets the same numbers.

A synthetic capture with known geometry: a flat floor at z=0 and a 0.75 m desk slab, seen by two
cameras with ARKit conventions (Y up, camera looks down -Z). Every helper is checked against the
truth it was built from, not against itself.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from litereality_agent import evidence_kit

KIT = Path(evidence_kit.__file__).parent
sys.path.insert(0, str(KIT))
from read_scan import Scan  # noqa: E402
import measure  # noqa: E402
import rectify  # noqa: E402
import contact_sheets  # noqa: E402

W, H, DW, DH = 640, 480, 160, 120
FX = 500.0
DESK_Z_ARKIT = 0.75  # metres above the floor; ARKit Y-up, so this is the y coordinate


def _pose_looking_down(x: float, z: float, height: float) -> np.ndarray:
    """A camera at (x, height, z) in ARKit world looking straight down (-Y), image 'up' = -Z."""
    M = np.eye(4)
    # camera axes in world: right=+X, up=-Z, forward(-Zcam)=-Y  => Zcam = +Y
    M[:3, 0] = [1, 0, 0]
    M[:3, 1] = [0, 0, -1]
    M[:3, 2] = [0, 1, 0]
    M[:3, 3] = [x, height, z]
    return M


def _render_depth(pose: np.ndarray, K: np.ndarray, size) -> np.ndarray:
    """Ray-cast the synthetic world: floor at y=0, a desk slab (top y=0.75) over x∈[0,1.6], z∈[0,0.8]."""
    w, h = size
    v, u = np.mgrid[0:h, 0:w]
    d = np.stack([(u - K[0, 2]) / K[0, 0], -(v - K[1, 2]) / K[1, 1], -np.ones_like(u, dtype=float)], -1)
    dw = d @ pose[:3, :3].T
    o = pose[:3, 3]
    out = np.zeros((h, w), dtype=np.float32)
    for plane_y, box in ((DESK_Z_ARKIT, ((0.0, 1.6), (0.0, 0.8))), (0.0, None)):
        t = (plane_y - o[1]) / dw[..., 1]
        pts = o + t[..., None] * dw
        ok = (t > 0) & (out == 0)
        if box:
            ok &= (pts[..., 0] >= box[0][0]) & (pts[..., 0] <= box[0][1]) & (pts[..., 2] >= box[1][0]) & (pts[..., 2] <= box[1][1])
        depth = -(np.linalg.inv(pose) @ np.concatenate([pts, np.ones_like(t)[..., None]], -1)[..., None])[..., 2, 0]
        out[ok] = depth[ok]
    return out


@pytest.fixture(scope="module")
def scan(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("scan")
    K = np.array([[FX, 0, W / 2], [0, FX, H / 2], [0, 0, 1]])
    rng = np.random.default_rng(0)
    for i, (x, z) in enumerate([(0.8, 0.4), (1.1, 0.6)]):
        pose = _pose_looking_down(x, z, 2.4)
        (d / f"frame_{i:05d}.json").write_text(json.dumps({
            "frame_index": i, "cameraPoseARFrame": pose.flatten().tolist(), "intrinsics": K.flatten().tolist()}))
        Image.fromarray(rng.integers(0, 255, (H, W, 3), dtype=np.uint8)).save(d / f"frame_{i:05d}.jpg")
        kd = K.copy(); kd[0, :] *= DW / W; kd[1, :] *= DH / H
        depth = _render_depth(pose, kd, (DW, DH))
        Image.fromarray((depth * 1000).astype(np.uint16)).save(d / f"depth_{i:05d}.png")
        Image.fromarray(np.full((DH, DW), 2, dtype=np.uint8)).save(d / f"conf_{i:05d}.png")
    # point cloud: floor + desk top, ARKit frame
    floor = np.c_[rng.uniform(-1, 3, 4000), rng.normal(0, 0.005, 4000), rng.uniform(-1, 3, 4000)]
    desk = np.c_[rng.uniform(0, 1.6, 1500), DESK_Z_ARKIT + rng.normal(0, 0.005, 1500), rng.uniform(0, 0.8, 1500)]
    pts = np.vstack([floor, desk])
    with open(d / "pointcloud.pcd", "w") as fh:
        fh.write("# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
                 f"WIDTH {len(pts)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(pts)}\nDATA ascii\n")
        np.savetxt(fh, pts, fmt="%.4f")
    return d


def test_scan_reads_frames_poses_and_the_cloud(scan):
    S = Scan(str(scan))
    assert S.indices() == [0, 1]
    assert S.rgb(0).shape == (H, W, 3)
    assert S.depth(0).shape == (DH, DW) and S.depth(0).max() > 2.0
    assert S.pcd().shape[1] == 3 and len(S.pcd()) == 5500
    # ARKit y (height) becomes Blender z
    assert np.allclose(S.pcd_blender()[:, 2], S.pcd()[:, 1])


def test_probe_depth_lands_on_the_desk_top(scan):
    S = Scan(str(scan))
    # camera 0 is above (0.8, 0.4): the image centre looks straight down onto the desk
    p = measure.probe_depth(S, 0, W / 2, H / 2)
    assert p["point"] is not None
    assert p["point"][2] == pytest.approx(DESK_Z_ARKIT, abs=0.01)     # Blender z = height
    assert p["point"][0] == pytest.approx(0.8, abs=0.01)


def test_triangulate_recovers_a_point_seen_twice(scan):
    S = Scan(str(scan))
    X = np.array([0.5, 0.3, DESK_Z_ARKIT])   # Blender world (x, y=-z_arkit, z=height)
    obs = []
    for i in (0, 1):
        M = np.linalg.inv(S.pose_blender(i)); K = S.K(i)
        c = M @ np.append(X, 1.0)
        obs.append((i, (K[0, 0] * c[0] / -c[2] + K[0, 2], K[1, 1] * -c[1] / -c[2] + K[1, 2])))
    t = measure.triangulate(S, obs)
    assert np.allclose(t["point"], X, atol=1e-3) and t["residual_m"] < 1e-3


def test_height_profile_finds_the_desk_and_floor(scan):
    S = Scan(str(scan))
    assert measure.floor_z(S) == pytest.approx(0.0, abs=0.02)
    prof = measure.height_profile(S, (0.0, -0.8, 1.6, 0.0), floor_z=0.0)   # desk footprint (Blender y = -z_arkit)
    assert any(abs(h - DESK_Z_ARKIT) < 0.02 for h in prof["surfaces_m"][:2])


def test_pcd_slice_draws_only_the_desk_at_desk_height(scan, tmp_path):
    S = Scan(str(scan))
    sl = measure.pcd_slice(S, 0.70, 0.80, floor_z=0.0, out_png=str(tmp_path / "s.png"))
    assert sl["n_points"] > 1000 and (tmp_path / "s.png").is_file()
    rows, cols = np.nonzero(sl["grid"])
    xs = sl["x0"] + cols * sl["cell_m"]
    assert xs.min() > -0.1 and xs.max() < 1.7           # the desk's x extent, nothing else


def test_rectify_writes_a_texture_of_the_requested_size(scan, tmp_path):
    S = Scan(str(scan))
    out = rectify.rectify_region(S, 0, [(100, 100), (300, 110), (290, 300), (110, 290)], str(tmp_path / "t.png"), (256, 200))
    assert Image.open(out).size == (256, 200)


def test_contact_sheets_cover_every_frame(scan, tmp_path):
    S = Scan(str(scan))
    sheets = contact_sheets.build(S, str(tmp_path / "cs"), per_sheet=1)
    assert len(sheets) == 2 and all(Path(p).is_file() for p in sheets)


def test_evidence_pack_is_self_contained(scan, tmp_path):
    from litereality_agent.agent import evidence_pack
    stitches = tmp_path / "surface_ref"; stitches.mkdir()
    pack = evidence_pack.build(tmp_path / "authoring", scan, stitches)
    assert (pack / "README.md").is_file() and "Rx(+90)" in (pack / "README.md").read_text()
    assert (pack / "scan").is_symlink() and (pack / "scan" / "pointcloud.pcd").is_file()
    assert (pack / "helpers" / "measure.py").is_file() and not (pack / "helpers").is_symlink()
    assert len(list((pack / "contact_sheets").glob("*.jpg"))) == 1
    assert "2 photographs" in (pack / "README.md").read_text()
    # idempotent: a second build keeps the pack
    assert evidence_pack.build(tmp_path / "authoring", scan, stitches) == pack
