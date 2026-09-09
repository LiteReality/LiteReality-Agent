"""_overlay.py — project wall planes onto a frame to get each wall's outline + TRUE centre.

The frame render is from the exact ARKit camera pose, so we project the wall planes (room.usdz)
with that pose to get, per visible wall, its outline and geometric centre in UPRIGHT pixels.
Two uses:
  - label walls at their real centre on the frame annotation (the manifest's label point is the
    centroid of *visible* pixels, which drifts high when furniture hides a wall's base)
  - wall-FOCUS: outline just the wall of interest on the surface reference frames.
"""

from __future__ import annotations

from pathlib import Path

_LOADED = None


def _modules(config):
    global _LOADED
    if _LOADED is None:
        # Real imports now that both live under agent/tools/shared — these were `sys.path`
        # inserts back when they sat outside the package the tools could reach.
        from lrauthor.agent.tools.shared.image_selection import surface_views as SV
        from lrauthor.agent.tools.shared.stitch_wall_image import stitch_wall as SW

        _LOADED = (SW, SV)
    return _LOADED


def load_planes(config):
    """Wall planes for the scan, from room.usdz (walls only — floor/ceiling not needed here)."""
    SW, _SV = _modules(config)
    usdz = config.SCAN_DIR / "room.usdz"
    if not usdz.exists():
        usdz = config.SCAN_DIR / "roomplan" / "room.usdz"
    return [p for p in SW.load_wall_planes(usdz) if p.name.startswith("Wall")]


def project_walls(frame_json, planes, config, Wu: int, Hu: int, only: str | None = None) -> dict:
    """Per VISIBLE wall (or just `only`): its outline segments + geometric centre, in UPRIGHT
    pixels. Returns {name: {"edges": [[(x,y),(x,y)], …], "cx": float, "cy": float}}."""
    SW, SV = _modules(config)
    intr, pose = SW._load_frame_meta(Path(frame_json))
    H0 = Wu  # landscape height = upright width (rotate-90-CW transform: (x,y) -> (H0-1-y, x))
    out: dict[str, dict] = {}
    for p in planes:
        if only and p.name != only:
            continue
        corners = SV.wall_corners(p)
        edges = []
        for i in range(4):
            seg = SV._proj_edge(corners[i], corners[(i + 1) % 4], pose, intr)
            if seg:
                edges.append([(H0 - 1 - y, x) for (x, y) in seg])
        if not edges:
            continue
        pts = [pt for seg in edges for pt in seg]
        xs = [px for px, _ in pts]
        ys = [py for _, py in pts]
        # ON-SCREEN coverage: clip the projected wall's bbox to the frame. A wall that FILLS the
        # view has its raw centroid off-screen (far corners project way outside), but its
        # on-screen area is large — so gate on the clipped coverage, not the centroid, or big
        # walls get wrongly culled exactly when they dominate the frame.
        ix0, iy0 = max(0.0, min(xs)), max(0.0, min(ys))
        ix1, iy1 = min(float(Wu), max(xs)), min(float(Hu), max(ys))
        ivw, ivh = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
        if only is None:
            # keep any wall covering a meaningful area of THIS frame; drop off-frame walls and
            # thin edge-on slivers (one on-screen dimension tiny).
            on_frac = (ivw * ivh) / float(Wu * Hu)
            if on_frac < 0.03 or min(ivw, ivh) < 0.04 * min(Wu, Hu):
                continue
        # label at the centre of the ON-SCREEN region (always in-frame, over the visible wall);
        # fall back to the clamped raw centroid if nothing is on-screen (only= focus mode).
        if ivw > 0 and ivh > 0:
            cx, cy = (ix0 + ix1) / 2, (iy0 + iy1) / 2
        else:
            cx = min(max(0.0, sum(xs) / len(xs)), float(Wu))
            cy = min(max(0.0, sum(ys) / len(ys)), float(Hu))
        out[p.name] = {"edges": edges, "cx": cx, "cy": cy}
    return out


def draw_wall_outlines(draw, projections: dict, focus: bool = False) -> None:
    """Stroke each projected wall's outline. focus=True (single wall of interest) draws thicker."""
    width = 9 if focus else 5
    for _name, pr in projections.items():
        for seg in pr["edges"]:
            draw.line(seg, fill=(255, 150, 40), width=width)
