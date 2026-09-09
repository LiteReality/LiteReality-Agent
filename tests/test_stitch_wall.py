"""The wall-image stitch (`views/stitch_wall_image/stitch_wall.py`) — the head-on `<wall>_stitched.jpg`
references stage-2 authoring treats as its DECISIVE evidence (see author.py's stitch-coverage guard).

A silently-wrong stitch (mirrored, mis-scaled, or blank) sends the authoring model a lie it cannot
tell from truth, so the geometry and the fill are worth pinning. The SDK is dependency-light
(numpy + Pillow) and pure, so these run offline and fast:

  * the plane fit from a wall's 8 corners (axes orthonormal, size recovered),
  * the ortho pixel grid (shape from pixels-per-meter, every sample ON the plane),
  * a full one-frame stitch of a head-on view (coverage lands, image + masks written),
  * frame-pair discovery, and the empty-input guards.

One check parses the committed `example_scans/Office_room/room.usdz` to keep the real RoomPlan USDZ
parser honest; it skips if that capture is absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from lrauthor.agent.tools.shared.stitch_wall_image.stitch_wall import (
    WallPlane,
    _plane_from_corners,
    _plane_grid,
    find_frame_pairs,
    load_wall_planes,
    stitch_wall,
    stitch_walls,
)

_EXAMPLE_USDZ = Path(__file__).resolve().parents[1] / "example_scans" / "Office_room" / "room.usdz"


def _axis_aligned_wall(width_m: float = 2.0, height_m: float = 2.5) -> WallPlane:
    """A wall centered at the origin, facing +Z, u=+X, v=+Y — the frame the synthetic camera below
    looks at head-on."""
    return WallPlane(
        name="Wall0",
        center=np.zeros(3),
        u_axis=np.array([1.0, 0.0, 0.0]),
        v_axis=np.array([0.0, 1.0, 0.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        width_m=width_m,
        height_m=height_m,
    )


# --------------------------------------------------------------------------- #
# plane fit
# --------------------------------------------------------------------------- #
def test_plane_from_corners_recovers_axes_and_size():
    """A thin 3 m (X) x 2 m (Y) x 0.1 m (Z) box → horizontal=+X, vertical=+Y, outward normal, and the
    metric extents read back as width/height. This is the map from RoomPlan geometry to the ortho
    frame everything downstream is rectified into."""
    w, h, t = 3.0, 2.0, 0.1
    signs = np.array([[sx, sy, sz] for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)], float)
    corners = signs * np.array([w / 2, h / 2, t / 2])

    plane = _plane_from_corners("Wall0", corners)

    assert plane.width_m == pytest.approx(w, abs=1e-6)
    assert plane.height_m == pytest.approx(h, abs=1e-6)
    assert abs(plane.v_axis[1]) == pytest.approx(1.0, abs=1e-6)  # vertical is world-up
    assert plane.v_axis[1] > 0  # and points up, not down
    assert abs(plane.u_axis[0]) == pytest.approx(1.0, abs=1e-6)  # horizontal is world-X
    # right-handed, unit-length frame
    assert np.linalg.norm(plane.normal) == pytest.approx(1.0, abs=1e-6)
    assert np.dot(np.cross(plane.u_axis, plane.v_axis), plane.normal) == pytest.approx(1.0, abs=1e-6)


def test_coincident_corners_collapse_to_zero_size():
    """Eight coincident corners carry no metric extent. SVD still hands back an orthonormal frame,
    so this doesn't raise — but the recovered footprint must be zero, which downstream size/coverage
    logic treats as an empty wall rather than silently inventing dimensions."""
    plane = _plane_from_corners("Wall0", np.zeros((8, 3)))
    assert plane.width_m == pytest.approx(0.0, abs=1e-9)
    assert plane.height_m == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# ortho grid
# --------------------------------------------------------------------------- #
def test_plane_grid_shape_and_points_lie_on_plane():
    """The grid resolution follows pixels-per-meter, and every world point it emits sits exactly on
    the wall plane (zero component along the normal). If a point drifted off-plane the stitch would
    sample the wrong depth and blend in foreground clutter."""
    plane = _axis_aligned_wall(width_m=3.0, height_m=2.0)
    ppm = 50
    points, w, h = _plane_grid(plane, ppm)

    assert (w, h) == (150, 100)  # both above the 64-px floor, so ppm drives the size
    assert points.shape == (w * h, 3)
    # centered at origin with normal +Z: on-plane means z == 0 for every sample
    assert np.allclose(points[:, 2], 0.0, atol=1e-6)
    # samples stay within the wall's metric footprint (half-pixel inset from the edges)
    assert np.all(np.abs(points[:, 0]) <= plane.width_m / 2)
    assert np.all(np.abs(points[:, 1]) <= plane.height_m / 2)


def test_plane_grid_enforces_a_floor_resolution():
    """A tiny stub wall must not collapse to a 1-pixel image (the min(64, …) guard)."""
    _, w, h = _plane_grid(_axis_aligned_wall(width_m=0.05, height_m=0.05), ppm=10)
    assert w >= 64 and h >= 64


# --------------------------------------------------------------------------- #
# full stitch, one head-on frame
# --------------------------------------------------------------------------- #
def _write_head_on_frame(dir_: Path, color=(120, 180, 90), dist_m: float = 4.0, size: int = 1024):
    """A single capture looking straight at the +Z wall from (0,0,dist): identity rotation (camera
    axes = world axes, so it views down -Z toward the wall), translation on +Z. Solid color so the
    assertion is about coverage geometry, not pixels."""
    img = Image.new("RGB", (size, size), color)
    jpg = dir_ / "frame_00000.jpg"
    img.save(jpg, quality=95)
    fx = fy = 1000.0
    cx = cy = size / 2
    intrinsics = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    pose = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, dist_m, 0.0, 0.0, 0.0, 1.0]
    meta = dir_ / "frame_00000.json"
    meta.write_text(json.dumps({"intrinsics": intrinsics, "cameraPoseARFrame": pose}))
    return jpg, meta


def test_stitch_wall_fills_a_head_on_view(tmp_path):
    """End to end on one synthetic frame: a camera square-on to the wall should observe (nearly) the
    whole plane, and the tool must write the stitch plus its known/unknown coverage masks, at the
    resolution the ppm implies."""
    plane = _axis_aligned_wall(width_m=2.0, height_m=2.0)
    frame = _write_head_on_frame(tmp_path)
    out = tmp_path / "Wall0_stitched.jpg"

    result = stitch_wall(plane, [frame], out, pixels_per_meter=40)

    assert out.is_file()
    assert Path(result["known_mask"]).is_file()
    assert Path(result["unknown_mask"]).is_file()
    assert result["size_px"] == [80, 80]  # 2 m * 40 ppm
    assert result["frames_used"] == 1
    assert 0.0 <= result["coverage"] <= 1.0
    assert result["coverage"] > 0.5, "a head-on frame should cover most of the wall"
    # the written image matches the reported grid size
    assert Image.open(out).size == (80, 80)


def test_stitch_wall_leaves_an_unobserved_wall_blank(tmp_path):
    """A camera facing AWAY from the wall observes nothing; coverage must read ~0 rather than the
    fill silently smearing an oblique frame across the plane."""
    plane = _axis_aligned_wall()
    # same frame but placed behind the wall on -Z looking further away → nothing projects in front
    img = Image.new("RGB", (256, 256), (10, 10, 10))
    jpg = tmp_path / "frame_00000.jpg"
    img.save(jpg)
    intr = [500.0, 0.0, 128.0, 0.0, 500.0, 128.0, 0.0, 0.0, 1.0]
    # translation on -Z, identity rotation → camera views down -Z, AWAY from the wall at origin
    pose = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, -3.0, 0, 0, 0, 1]
    (tmp_path / "frame_00000.json").write_text(
        json.dumps({"intrinsics": intr, "cameraPoseARFrame": pose})
    )
    result = stitch_wall(plane, [(jpg, tmp_path / "frame_00000.json")], tmp_path / "w.jpg",
                         pixels_per_meter=20)
    assert result["coverage"] == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# discovery + guards
# --------------------------------------------------------------------------- #
def test_find_frame_pairs_matches_jpg_to_json(tmp_path):
    """Only frames with BOTH the jpg and its json count; a lone json is skipped."""
    (tmp_path / "frame_00000.jpg").write_bytes(b"x")
    (tmp_path / "frame_00000.json").write_text("{}")
    (tmp_path / "frame_00001.json").write_text("{}")  # no jpg → dropped
    pairs = find_frame_pairs(tmp_path)
    assert [j.name for _, j in pairs] == ["frame_00000.json"]


def test_stitch_walls_needs_frames(tmp_path):
    """No capture pairs → a clear error, not an empty silent run. (Uses the real example USDZ so
    there ARE walls; the failure must be the missing frames.)"""
    if not _EXAMPLE_USDZ.is_file():
        pytest.skip("example_scans/Office_room/room.usdz not present")
    with pytest.raises(ValueError, match="frame"):
        stitch_walls(tmp_path, out_dir=tmp_path / "out", usdz=_EXAMPLE_USDZ)


# --------------------------------------------------------------------------- #
# real RoomPlan parser
# --------------------------------------------------------------------------- #
def test_load_wall_planes_parses_the_example_capture():
    """Guard the USDZ parser against a real RoomPlan export: every wall comes back oriented (unit,
    right-handed frame; vertical points up) with a positive metric footprint."""
    if not _EXAMPLE_USDZ.is_file():
        pytest.skip("example_scans/Office_room/room.usdz not present")
    planes = load_wall_planes(_EXAMPLE_USDZ)
    assert planes, "no walls parsed from the example RoomPlan capture"
    for p in planes:
        assert p.width_m > 0 and p.height_m > 0
        assert p.v_axis[1] > 0  # vertical axis points up
        for ax in (p.u_axis, p.v_axis, p.normal):
            assert np.linalg.norm(ax) == pytest.approx(1.0, abs=1e-6)
        assert np.dot(np.cross(p.u_axis, p.v_axis), p.normal) == pytest.approx(1.0, abs=1e-3)
    assert [p.name for p in planes] == sorted(p.name for p in planes)  # returned sorted
