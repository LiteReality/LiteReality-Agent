"""convex.py — the decomposition cache and the minimum-translation-vector the collision fix uses.

Pins the two-stage split: RAW triangles decide whether two things touch, convex pieces decide how
far to move them. These tests are about the second half, so they hand-build convex boxes rather
than decomposing anything — the MTV arithmetic is what regresses, not CoACD.

The one decomposition test uses a unit box (sub-second) purely to pin the cache round-trip; the
real assets take 8-16 s each and belong in a benchmark, not the suite.
"""

from __future__ import annotations

import math

# coacd is a core dependency (without it the fix silently reverts to the AABB estimate that
# over-states the move by 1.6-2.8x), so a missing one must FAIL rather than skip.
import coacd  # noqa: F401
import pytest
import trimesh

from litereality_agent.agent.tools.check_collisions.source import collision_mesh as sc
from litereality_agent.agent.tools.check_collisions.source import convex


def _box(cx, cy, cz, sx, sy, sz):
    """An axis-aligned convex box in the glb frame (Y up)."""
    m = trimesh.creation.box(extents=(sx, sy, sz))
    m.apply_translation((cx, cy, cz))
    return m


def test_mtv_picks_the_shortest_horizontal_move():
    """Two 1 m boxes overlapping 0.2 m in x and 0.6 m in z must be pushed 0.2 m along x."""
    a = [_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)]
    b = [_box(0.8, 0.0, 0.4, 1.0, 1.0, 1.0)]
    gx, gz, dist = convex.mtv_xy(a, b)
    assert dist == pytest.approx(0.2, abs=0.02)
    assert abs(gx) > abs(gz)      # pushed along x, the shallower overlap
    assert gx > 0                 # b sits at +x, so it keeps going +x


def test_mtv_leaves_the_world_axes_for_rotated_furniture():
    """Rotated furniture is where the AABB estimate falls apart, and it is the common case.

    Two unit boxes yawed 45 deg, separated by 0.8 along their OWN x-axis: they overlap 0.2, and the
    correct push is 0.2 along that rotated axis. Their world AABBs are 1.414 wide, so the old
    axis-aligned estimate pays 0.848 — over 4x too far, and past MAX_NUDGE, so it gets reverted.
    """
    r = trimesh.transformations.rotation_matrix(math.radians(45), [0, 1, 0])
    a_mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    a_mesh.apply_transform(r)
    b_mesh = a_mesh.copy()
    step = 0.8 / math.sqrt(2.0)
    b_mesh.apply_translation((step, 0.0, step))

    gx, gz, dist = convex.mtv_xy([a_mesh], [b_mesh])
    assert dist == pytest.approx(0.2, abs=0.02), "shortest move is along the boxes' own axis"
    assert gx > 0.05 and gz > 0.05, "the move must be diagonal in world coordinates"

    amin, amax = a_mesh.bounds
    bmin, bmax = b_mesh.bounds
    aabb_estimate = min(min(amax[0], bmax[0]) - max(amin[0], bmin[0]),
                        min(amax[2], bmax[2]) - max(amin[2], bmin[2]))
    assert aabb_estimate > 4 * dist, "this is the over-estimate the convex MTV removes"


def test_mtv_is_none_when_nothing_touches():
    """The measure model never invents contact — that is the detector's job, on raw triangles."""
    a = [_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)]
    b = [_box(3.0, 0.0, 0.0, 1.0, 1.0, 1.0)]
    assert convex.mtv_xy(a, b) is None


def test_decomposition_preserves_a_tuck_that_a_single_hull_would_break():
    """A C-shaped 'table' (top + two legs) with a 'chair' in the gap.

    One convex hull of the table fills the gap and reports contact; the 3-piece decomposition does
    not. This is the property that lets convex geometry measure without corrupting detection.
    """
    table = [_box(0.0, 0.9, 0.0, 2.0, 0.1, 1.0),      # top
             _box(-0.9, 0.4, 0.0, 0.1, 0.9, 1.0),      # left leg
             _box(0.9, 0.4, 0.0, 0.1, 0.9, 1.0)]       # right leg
    chair = [_box(0.0, 0.3, 0.0, 0.6, 0.6, 0.6)]       # sits in the gap, touching nothing

    assert convex.mtv_xy(table, chair) is None, "decomposed table must leave the tuck alone"

    whole = trimesh.util.concatenate(table).convex_hull
    assert convex.mtv_xy([whole], chair) is not None, "a single hull swallows the gap"


def test_separation_falls_back_to_aabb_without_convex_parts():
    """No coacd (or a decomposition that failed) must degrade to the old behaviour, not crash."""
    fur = {"A": _box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
           "B": _box(0.8, 0.0, 0.0, 1.0, 1.0, 1.0)}
    sep = sc._separation("A", "B", fur, {})
    assert sep["method"] == "aabb"
    assert sep["depth"] == pytest.approx(0.2, abs=1e-6)

    sep2 = sc._separation("A", "B", fur, {"A": [fur["A"]], "B": [fur["B"]]})
    assert sep2["method"] == "convex"


def test_decomposition_cache_round_trips(tmp_path):
    """Second call must come from disk — the cache is what makes 8-16 s per asset affordable."""
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    first = convex.decompose(box, cache_dir=tmp_path)
    assert first, "a unit box must decompose"
    assert len(list(tmp_path.glob("*.npz"))) == 1

    second = convex.decompose(box, cache_dir=tmp_path)
    assert len(second) == len(first)
    assert len(list(tmp_path.glob("*.npz"))) == 1, "must reuse, not add a second entry"


def test_cache_key_is_shared_across_placements_of_one_asset():
    """Eight chairs cut from one asset must hash to ONE key, or the cache never pays for itself."""
    base = trimesh.creation.box(extents=(0.5, 0.9, 0.5))

    def placed(x, z, yaw_deg):
        m = base.copy()
        m.apply_transform(trimesh.transformations.rotation_matrix(math.radians(yaw_deg), [0, 1, 0]))
        m.apply_translation((x, 0.0, z))
        return m

    keys = set()
    for x, z, yaw in ((0.0, 0.0, 0.0), (3.0, -2.0, 25.16), (-1.5, 4.0, -154.84)):
        local, _ = convex._canonical(placed(x, z, yaw), yaw)
        keys.add(convex._key(local, convex.THRESHOLD, convex.PREPROCESS_RESOLUTION))
    assert len(keys) == 1, f"same asset at 3 placements produced {len(keys)} cache keys"


def test_mtv_direction_maps_into_the_shell_frame():
    """The fix is written into SHELL coordinates, where SHELL(x, y) = glb(x, -z)."""
    a = [_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)]
    b = [_box(0.0, 0.0, 0.8, 1.0, 1.0, 1.0)]
    gx, gz, _dist = convex.mtv_xy(a, b)
    sdx, sdy = sc.glb_to_shell_xy(gx, gz)
    assert sdx == pytest.approx(gx)
    assert sdy == pytest.approx(-gz)


def test_only_restricts_decomposition_to_bodies_in_contact():
    """Decomposition is ~20 s per asset, so it must run for the touching bodies and nobody else."""
    fur = {"A": _box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
           "B": _box(0.8, 0.0, 0.0, 1.0, 1.0, 1.0),
           "Far": _box(9.0, 0.0, 0.0, 1.0, 1.0, 1.0)}
    bodies = {"furniture": fur, "glb": None}
    shell = {"objects": {k: {"yaw": 0.0} for k in fur}}

    parts = sc._convex_parts(bodies, shell, only={"A", "B"})
    assert set(parts) == {"A", "B"}, "must not decompose bodies that touch nothing"

    assert set(sc._convex_parts(bodies, shell, only=set())) == set()
