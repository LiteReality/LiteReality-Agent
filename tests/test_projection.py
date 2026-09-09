"""Points behind the camera must not count as visible.

`compute_mapping` / `compute_mapping_for_3D_bbox` decide, per frame, whether an object is in
view — and that decision picks which frames become the object's evidence crops, which become the
reference image, which becomes the reconstructed GLB. A false "visible" therefore does not fail;
it quietly seeds the whole object from a frame the object is not in.

The pinhole projection divides by camera-space depth, which is only defined in front of the
camera. `z == 0` gives nan and is loud (the RuntimeWarnings); `z < 0` mirrors the point through
the principal point onto a perfectly plausible pixel and is silent. This pins the silent one.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

# The vendored preprocessing imports open3d/cv2 at module scope. They are real dependencies and
# present in the project venv; `setdefault` leaves those alone and only stubs them when a
# stripped environment is running the suite, since neither is used by the code under test.
for _heavy in ("open3d", "cv2"):
    sys.modules.setdefault(_heavy, types.ModuleType(_heavy))

from lrauthor.pipeline.measure.ingest.preprocessing.vendor.litereality.object_image_extraction import (  # noqa: E402
    project_to_pixels,
    round_to_int,
)

FX = FY = 1000.0
CX, CY = 960.0, 720.0
W, H = 1920, 1440
POSE = np.eye(4)  # camera at the origin, +Z forward — the convention p[2] > 0 encodes


def visible(points: np.ndarray) -> np.ndarray:
    p, in_front = project_to_pixels(np.asarray(points, dtype=float), POSE, FX, FY, CX, CY)
    return in_front * (p[0] > 0) * (p[1] > 0) * (p[0] < W - 1) * (p[1] < H - 1)


def test_points_in_front_project_where_they_always_did():
    """The fix must not move a single valid point."""
    p, in_front = project_to_pixels(
        np.array([[0.0, 0.0, 3.0], [0.5, 0.3, 4.0]]), POSE, FX, FY, CX, CY
    )
    assert in_front.all()
    assert p[0][0] == pytest.approx(CX) and p[1][0] == pytest.approx(CY)
    assert p[0][1] == pytest.approx(CX + 0.5 * FX / 4.0)
    assert p[1][1] == pytest.approx(CY + 0.3 * FY / 4.0)


def test_a_point_behind_the_camera_is_not_visible():
    """Without the guard these land at (893, 687) and (1010, 745) — inside the frame, and marked
    visible. That is the whole bug: an object behind the camera scoring as seen."""
    behind = np.array([[0.2, 0.1, -3.0], [-0.2, -0.1, -4.0]])
    assert not visible(behind).any()


def test_a_point_on_the_camera_plane_is_not_visible():
    assert not visible(np.array([[0.0, 0.0, 0.0]])).any()


def test_no_runtime_warnings_on_degenerate_input():
    """The warnings were the symptom that led here; a silent run is how we know the guard fires
    before the division rather than after it."""
    points = np.array([[0.0, 0.0, 3.0], [0.0, 0.0, 0.0], [0.2, 0.1, -3.0]])
    # errstate(raise) turns a stray invalid operation into an exception rather than a warning,
    # so this fails whether numpy is configured to warn or not.
    with np.errstate(all="raise"), _no_warnings():
        p, _ = project_to_pixels(points, POSE, FX, FY, CX, CY)
        round_to_int(p)


class _no_warnings:
    def __enter__(self):
        import warnings

        self._ctx = warnings.catch_warnings(record=True)
        self._log = self._ctx.__enter__()
        warnings.simplefilter("always")
        return self

    def __exit__(self, *exc):
        caught = [str(w.message) for w in self._log]
        self._ctx.__exit__(*exc)
        assert not caught, f"projection still warns: {caught}"
        return False


def test_round_to_int_never_casts_nan():
    """`np.round(nan).astype(int)` is undefined — it warns and yields INT_MIN, which would be a
    catastrophic array index if a mask ever let it through."""
    p = np.array([[1.4, np.nan, np.inf], [2.6, -np.inf, np.nan]])
    out = round_to_int(p)
    assert out.dtype == np.int64 or out.dtype == np.int32
    assert out[0][0] == 1 and out[1][0] == 3
    assert (np.abs(out[:, 1:]) < 1000).all(), "non-finite values must collapse to 0, not INT_MIN"


def test_mixed_batch_keeps_only_the_valid_points():
    points = np.array(
        [
            [0.0, 0.0, 3.0],  # visible
            [0.0, 0.0, 0.0],  # on the plane
            [0.2, 0.1, -3.0],  # behind
            [0.5, 0.3, 4.0],  # visible
            [50.0, 0.0, 3.0],  # in front but far outside the frame
        ]
    )
    assert list(visible(points)) == [True, False, False, True, False]
