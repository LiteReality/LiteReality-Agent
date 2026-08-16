"""convex.py — convex decomposition cache, and the minimum translation vector the FIX uses.

Two-stage collision, and the split is the whole point:

  DETECT   `collision_mesh.mesh_contacts` — FCL on the RAW triangles. Exact, and the only model
           that gets a tucked chair right (a chair 80 mm clear of its table reads CLEAR).
  MEASURE  this module — a convex decomposition of the same meshes, asked only HOW FAR to move a
           pair the detector has ALREADY confirmed is touching.

Because the convex model never decides *whether* two things collide, its over-approximation
cannot invent a clash. It only sizes a fix. That is what makes it safe to use here when a single
convex hull would be far too coarse to detect with: measured on Panda-2, one hull per object
swallows the entire 80 mm gap under a tucked chair and reports contact, while a decomposition
holds the error to ~17 mm — still too much to DETECT with, fine to MEASURE with.

Why this exists at all: FCL's own penetration depth is unusable on these meshes. They are not
watertight (Table0 alone has 846 connected components), and on open surfaces FCL's depth is
garbage — a desk flush against a wall once reported 1.16 m. The previous fallback was the two
meshes' world-AABB overlap, which over-states the move by 1.6–2.8x (Chair1/Table0 asked for
0.591 m against a true 0.208 m) and so blew through MAX_NUDGE and got reverted. Convex pieces
are closed by construction, so EPA/projection arithmetic on them is meaningful.

Degrades gracefully: with no `coacd` installed every entry point returns None and the caller
keeps its previous AABB behaviour.

    python -m litereality_agent.agent.tools.check_collisions.source.convex --glb <Room.glb>
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

# CoACD voxel-remeshes non-watertight input to make it manifold, and THAT grid — not the
# decomposition `threshold` — is the accuracy floor. Turning `threshold` down is wasted effort:
# 24 -> 128 pieces moved the clearance error 17.2 -> 17.4 mm, for 2x the time.
# `preprocess_resolution` is the real knob, and the error tracks its cell size ~1:1. Measured on
# Panda-2's Chair2/Table0, whose true clearance is 80.2 mm:
#
#     pre_res    cell     error    time/2 assets
#         50   16.2mm    17.2mm            17 s   <- CoACD default, too coarse
#        100    8.1mm     8.7mm            38 s   <- here: under a third of OBJ_PEN
#        200    4.1mm     4.3mm           103 s
#        400    2.0mm     2.1mm           343 s
#
# 100 is the knee. This number only sizes a fix that is then judged against MAX_NUDGE = 0.30 m, so
# millimetres are irrelevant next to the 0.38 m error it exists to remove — and cost is per-asset
# on a cold cache, which is the thing a user actually waits for.
THRESHOLD = 0.05
PREPROCESS_RESOLUTION = 100
MAX_CONVEX_HULL = 32
CACHE_DIRNAME = ".convex_cache"


def available() -> bool:
    """Is convex decomposition usable? Callers fall back to AABB arithmetic when not."""
    try:
        import coacd  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def _canonical(mesh, yaw_deg: float):
    """Object-local copy of a world mesh, so the 8 chairs cut from ONE asset share one cache entry.

    Undo the placement: move the centre to the origin, then un-yaw. The glb frame is Y-up with
    SHELL(x, y) = glb(x, -z), under which a SHELL yaw of theta is a rotation of theta about glb +Y,
    so the inverse is -theta about +Y.

    The centre MUST be the vertex mean, not the bbox midpoint. Only the mean is rotation-
    equivariant: the AABB of a rotated mesh is not the rotated AABB, so centring on it leaves a
    yaw-dependent residual offset and every yaw hashes differently. Measured before this fix,
    Panda-2's eight chairs — one asset, four yaws — produced four cache entries instead of one.
    Returns (local_mesh, forward_matrix) mapping the local frame back to world.
    """
    import trimesh

    centre = np.asarray(mesh.vertices).mean(axis=0)
    to_origin = trimesh.transformations.translation_matrix(-centre)
    unyaw = trimesh.transformations.rotation_matrix(-math.radians(yaw_deg or 0.0), [0, 1, 0])
    local = mesh.copy()
    local.apply_transform(unyaw @ to_origin)
    forward = np.linalg.inv(unyaw @ to_origin)
    return local, forward


def _key(local, threshold: float, pre_res: int) -> str:
    """Fingerprint of the canonical geometry + the settings that shaped the decomposition.

    Per-VERTEX hashing does not work here, and the reason is worth recording. Two placements of one
    asset differ by a rotation, which leaves residuals of ~1e-7 on every coordinate. Across a
    chair's 30 260 vertices, ANY fixed rounding grid puts hundreds of those residuals on opposite
    sides of a boundary — measured 787 expected crossings at 1e-5 — so the hashes differ even
    though the geometry is identical to within a micron. Rounding coarser only moves the boundary;
    sorting or using rotation-invariant radii does not help either, because order statistics carry
    the same per-value noise.

    So fingerprint on AGGREGATES, whose noise averages down by sqrt(n): the mean radius of 30 k
    vertices is stable to ~1e-9, comfortably inside a 1e-6 grid. Tolerances are deliberately loose
    relative to that noise, because the two failure modes are wildly asymmetric — a false MISS
    costs one 20 s decomposition, a false HIT silently reuses the wrong geometry.
    """
    v = np.asarray(local.vertices, dtype=np.float64)
    radii = np.linalg.norm(v - v.mean(axis=0), axis=1)
    h = hashlib.sha1()
    h.update(np.round(np.sort(local.extents), 3).tobytes())     # 1 mm
    h.update(np.round([radii.mean(), radii.std(), radii.max()], 6).tobytes())
    h.update(np.round(float(local.area), 4).tobytes())
    h.update(f"{len(v)}:{len(local.faces)}:{threshold}:{pre_res}:{MAX_CONVEX_HULL}".encode())
    return h.hexdigest()[:16]


def _load_cache(path: Path):
    import trimesh

    if not path.is_file():
        return None
    try:
        z = np.load(path, allow_pickle=False)
        n = int(z["n"])
        return [trimesh.Trimesh(vertices=z[f"v{i}"], faces=z[f"f{i}"], process=False)
                for i in range(n)]
    except Exception:  # noqa: BLE001
        return None  # a corrupt entry is a cache miss, never a crash


def _save_cache(path: Path, parts) -> None:
    payload = {"n": np.array(len(parts))}
    for i, p in enumerate(parts):
        payload[f"v{i}"] = np.asarray(p.vertices, dtype=np.float64)
        payload[f"f{i}"] = np.asarray(p.faces, dtype=np.int32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def decompose(mesh, *, yaw_deg: float = 0.0, cache_dir: Path | None = None,
              threshold: float = THRESHOLD, pre_res: int = PREPROCESS_RESOLUTION):
    """Convex pieces of one world-space object mesh, cached across instances and across runs.

    Costs 8–16 s per DISTINCT asset, so the cache is what makes this usable: the 8 chairs in a
    room are one asset, and re-running the pass on an unchanged room is free.
    """
    import trimesh

    try:
        import coacd
    except Exception:  # noqa: BLE001
        return None

    local, forward = _canonical(mesh, yaw_deg)
    cache_path = (Path(cache_dir) / f"{_key(local, threshold, pre_res)}.npz") if cache_dir else None

    parts = _load_cache(cache_path) if cache_path else None
    if parts is None:
        coacd.set_log_level("error")
        try:
            raw = coacd.run_coacd(
                coacd.Mesh(np.asarray(local.vertices), np.asarray(local.faces)),
                threshold=threshold, max_convex_hull=MAX_CONVEX_HULL,
                preprocess_mode="auto", preprocess_resolution=pre_res,
            )
        except Exception:  # noqa: BLE001
            return None
        parts = []
        for v, f in raw:
            try:
                parts.append(trimesh.Trimesh(vertices=v, faces=f, process=False).convex_hull)
            except Exception:  # noqa: BLE001
                continue
        if not parts:
            return None
        if cache_path:
            _save_cache(cache_path, parts)

    return [p.copy().apply_transform(forward) for p in parts]


def _colliding_pairs(parts_a, parts_b):
    """Indices of piece pairs actually in contact — the only ones a translation must clear."""
    from trimesh.collision import CollisionManager

    cm = CollisionManager()
    for i, p in enumerate(parts_a):
        cm.add_object(f"a{i}", p)
    out = []
    for j, q in enumerate(parts_b):
        hit, names = cm.in_collision_single(q, return_names=True)
        if hit:
            out.extend((int(n[1:]), j) for n in names)
    return out


def mtv_xy(parts_a, parts_b, *, directions: int = 180):
    """Smallest HORIZONTAL translation of B that clears it from A. Returns (dx, dz, dist) in the
    glb frame, or None.

    Movement is XY-only by design (vertical placement is grounding's job — see the collision-check
    doc §6), which turns the general minimum-translation problem into a 1-D search over horizontal
    directions. For each candidate direction n, separating a convex pair along n needs exactly
    `max(A·n) - min(B·n)`; the union needs the max of that over the pairs in contact. Taking the
    direction that minimises it gives the shortest move — which is what MAX_NUDGE should be judged
    against, and what the AABB fallback got wrong by pushing along a world axis instead.
    """
    pairs = _colliding_pairs(parts_a, parts_b)
    if not pairs:
        return None

    N = _candidate_directions(parts_a, parts_b, pairs, directions)
    need = np.zeros(len(N))
    for ia, jb in pairs:
        va = np.asarray(parts_a[ia].vertices)
        vb = np.asarray(parts_b[jb].vertices)
        # push B along +n until its near face clears A's far face
        need = np.maximum(need, (va @ N.T).max(axis=0) - (vb @ N.T).min(axis=0))

    need = np.where(need > 0, need, np.inf)
    k = int(np.argmin(need))
    if not np.isfinite(need[k]):
        return None
    d = float(need[k])
    return float(N[k, 0] * d), float(N[k, 2] * d), d


def _candidate_directions(parts_a, parts_b, pairs, directions: int):
    """Horizontal directions to test: a uniform fan PLUS the pieces' own face normals.

    The fan alone quantises the answer — at 2 deg steps a 45 deg-yawed pair is measured 8% long,
    and every chair in a scan sits at some arbitrary yaw. For box-like furniture the true minimum
    lies exactly along a face normal, so feeding those in makes the common case exact instead of
    merely close.
    """
    ang = np.linspace(0.0, 2.0 * math.pi, directions, endpoint=False)
    fan = np.stack([np.cos(ang), np.zeros_like(ang), np.sin(ang)], axis=1)

    normals = []
    for parts, idx in ((parts_a, {i for i, _ in pairs}), (parts_b, {j for _, j in pairs})):
        for i in idx:
            n = np.asarray(parts[i].face_normals)
            n = n[np.abs(n[:, 1]) < 0.35]        # keep the near-vertical faces: horizontal normals
            if len(n):
                normals.append(n)
    if normals:
        n = np.vstack(normals)
        n[:, 1] = 0.0
        L = np.linalg.norm(n, axis=1)
        n = n[L > 1e-9] / L[L > 1e-9, None]
        n = np.vstack([n, -n])                   # a face normal separates in either sense
        # dedupe at ~0.6 deg so a 50k-triangle hull does not blow the direction set up
        n = np.unique(np.round(n, 2), axis=0)
        fan = np.vstack([fan, n])
    return fan


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glb", required=True)
    ap.add_argument("--shell")
    a = ap.parse_args()
    if not available():
        print("coacd is not installed — the fix falls back to AABB arithmetic")
        return 1

    from litereality_agent.agent.tools.check_collisions.source import collision_mesh as sc
    from litereality_agent.agent.tools.check_collisions.source.geometry import _extract_shell

    glb = Path(a.glb)
    shell = _extract_shell(Path(a.shell).read_text()) if a.shell else {}
    bodies = sc.build_bodies(glb, shell)
    yaws = {i: o.get("yaw", 0.0) for i, o in (shell.get("objects") or {}).items()}
    cache = glb.parent / CACHE_DIRNAME
    for oid, m in sorted(bodies["furniture"].items()):
        parts = decompose(m, yaw_deg=yaws.get(oid, 0.0), cache_dir=cache)
        print(f"  {oid:16} {len(m.faces):7} tris -> "
              f"{len(parts) if parts else 'FAILED':>4} convex pieces")
    print(f"cache: {cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
