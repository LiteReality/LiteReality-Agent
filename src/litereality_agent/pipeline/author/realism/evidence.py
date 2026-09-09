"""Stage-1 surface reference + triage — the agent's MATERIAL ground truth.

Oblique full-room photos are a poor substrate for "what material is this wall": the
surface is seen at an angle, under uneven lighting, distorted by perspective. This
module instead builds, for **every wall + the floor + the ceiling**, a rectified,
head-on ("ortho") image of that surface, plus a known/unknown mask marking which
pixels are real observations vs unobserved. The agent reads colour, pattern and
finish straight off the rectified image instead of inferring them through perspective.

It reuses the standalone SDK stitcher (`authoring/views/stitch_wall_image/stitch_wall.py`) — which
already does the projection/scoring/fill — and adds the two things the stitcher lacks
for this stage:

  1. floor + ceiling planes. RoomPlan's floor is a polygon mesh (no box scale) and the
     ceiling is not in the usdz at all, so both planes are DERIVED from the wall
     corners (axis-aligned ground rectangle at the wall-bottom / wall-top height).
  2. triage. Per surface, from the KNOWN region of its stitch: coverage, colour spread
     and edge energy -> plain | detailed | low_coverage, and a pooled shared-wall
     colour. This tells the agent where to spend effort (cheap on plain walls, focused
     on the few detailed ones) instead of treating every wall the same.

Outputs land in ``config.SURFACE_REF`` (per-scan scratch, shared across iterations —
the reference is ground truth, not iteration state):

    <Surface>_stitched.jpg        rectified head-on image
    <Surface>_stitched_unknown_mask.png   white = NOT observed (do not match to it)
    surface_ref_manifest.json     per-surface triage + paths + the shared wall colour
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from litereality_agent.agent.tools.shared import config

# --- tunable triage thresholds (env-overridable) --------------------------- #
# A real stitch is contaminated by wall-mounted fixtures (whiteboards, shelves) and
# foreground occluders smeared onto the plane, plus smooth lighting gradients. So
# MEAN colour-std / MEAN edge energy are useless as a plain/detailed gate (they're
# dominated by that noise). The robust discriminator is the MEDIAN gradient: a plain
# painted wall stays near-zero even with a whiteboard on 20% of it, while an all-over
# pattern (tile, wood planks, brick) lifts the median. Colour spread is reported but
# NOT gated. Triage is a HINT — the agent sees the rectified image and overrides it.
LOW_COVERAGE = float(os.environ.get("HARNESS_TRIAGE_LOWCOV", "0.22"))  # < this -> barely observed
PLAIN_EDGE_MED = float(os.environ.get("HARNESS_TRIAGE_EDGE", "3.0"))  # median gradient (0..255/px)

# --- band profile (localize a SUB-WALL material zone) ---------------------- #
# The whole-wall triage above misses a tiled splashback / wainscot / dado / feature band,
# because it only covers PART of the wall height — the plain majority dominates the median.
# So also slice each wall into N horizontal bands (floor->top), and flag the band(s) whose
# colour or texture deviates from the wall's dominant band as a likely own-material region.
BAND_N = int(os.environ.get("HARNESS_BAND_N", "6"))  # horizontal strips per wall
BAND_COLOR_DELTA = float(os.environ.get("HARNESS_BAND_COLOR", "26"))  # RGB dist from wall colour
BAND_EDGE_DELTA = float(os.environ.get("HARNESS_BAND_EDGE", "1.6"))  # extra median gradient vs wall


def _load_stitch():
    """Import the standalone SDK stitcher by path (no package install needed)."""
    spec = importlib.util.spec_from_file_location("stitch_wall", str(config.STITCH_TOOL))
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("stitch_wall", mod)
    spec.loader.exec_module(mod)
    return mod


def _wall_world_corners(plane) -> np.ndarray:
    """The 4 world-space corners of a WallPlane (centre +/- half-width u +/- half-height v)."""
    u = plane.u_axis * (plane.width_m / 2.0)
    v = plane.v_axis * (plane.height_m / 2.0)
    c = plane.center
    return np.array([c - u - v, c + u - v, c + u + v, c - u + v], dtype=float)


def _derive_floor_ceiling(WallPlane, planes):
    """Floor + ceiling planes from the wall corners: an axis-aligned ground rectangle
    (XZ bounding box of the walls) placed at the wall-bottom and wall-top heights.
    Floor normal points up (+Y), ceiling normal points down (-Y) — both face the room,
    so the cameras see them. Slightly larger than the true floor polygon, but the
    known/unknown mask marks the unobserved corners, so the over-extent is harmless."""
    corners = np.vstack([_wall_world_corners(p) for p in planes])
    y_floor, y_ceil = float(corners[:, 1].min()), float(corners[:, 1].max())
    xmin, xmax = float(corners[:, 0].min()), float(corners[:, 0].max())
    zmin, zmax = float(corners[:, 2].min()), float(corners[:, 2].max())
    cx, cz = (xmin + xmax) / 2.0, (zmin + zmax) / 2.0
    width, depth = max(xmax - xmin, 0.2), max(zmax - zmin, 0.2)
    floor = WallPlane(
        "Floor0",
        np.array([cx, y_floor, cz]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, -1.0]),
        np.array([0.0, 1.0, 0.0]),
        width,
        depth,
    )
    ceiling = WallPlane(
        "Ceiling0",
        np.array([cx, y_ceil, cz]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, -1.0, 0.0]),
        width,
        depth,
    )
    return floor, ceiling


def project_to_surface(surface: dict, points_world, image_size) -> np.ndarray:
    """World points -> (col, row) pixels in that surface's rectified reference image.

    The exact inverse of stitch_wall._plane_grid, which rasterised the image as::

        u = ((col + 0.5) / w - 0.5) * width_m        # +u_axis runs left -> right
        v = (0.5 - (row + 0.5) / h) * height_m       # +v_axis runs bottom -> top

    Use this instead of re-deriving the basis. The floor and ceiling share a centre and size
    but have OPPOSITE ``v_axis`` ([0,0,-1] vs [0,0,+1]) so both face into the room; assuming
    one sign for the other mirrors every overlay vertically, which on a yawed room looks like
    a large rotation error rather than an obvious flip.

    ``surface`` is one entry of surface_ref_manifest.json's ``surfaces`` (needs center/u_axis/
    v_axis/size_m). ``image_size`` is the reference image's (width, height) in pixels.
    """
    for key in ("center", "u_axis", "v_axis", "size_m"):
        if key not in surface:
            raise KeyError(
                f"surface {surface.get('name', '?')!r} has no {key!r} — it predates the basis "
                "being recorded; rebuild the surface reference (build(force=True))"
            )
    c = np.asarray(surface["center"], dtype=float)
    u_ax = np.asarray(surface["u_axis"], dtype=float)
    v_ax = np.asarray(surface["v_axis"], dtype=float)
    width_m, height_m = (float(v) for v in surface["size_m"])
    w, h = (int(v) for v in image_size)

    pts = np.atleast_2d(np.asarray(points_world, dtype=float))
    rel = pts - c
    u = rel @ u_ax
    v = rel @ v_ax
    col = (u / width_m + 0.5) * w - 0.5
    row = (0.5 - v / height_m) * h - 0.5
    return np.stack([col, row], axis=1)


def _band_profile(img: np.ndarray, known: np.ndarray, height_m: float) -> dict:
    """Slice the rectified wall into BAND_N horizontal strips (floor->top) and flag the band(s)
    whose colour / texture deviates from the wall's dominant strips — a likely tiled splashback,
    wainscot, dado or feature band that the whole-wall triage flattens away. Returns
    {bands:[{v_lo,v_hi,color_hex,edge_med,coverage}], flag:{v_lo,v_hi,color_hex,reason}|None}."""
    H, W = known.shape
    gray = img.mean(axis=2)
    gy, gx = np.gradient(gray)
    grad = np.hypot(gx, gy)
    bands = []
    for i in range(BAND_N):  # band i = i-th strip up from the FLOOR (image row 0 = wall TOP)
        v_lo, v_hi = height_m * i / BAND_N, height_m * (i + 1) / BAND_N
        r0 = int(round(H * (1.0 - (i + 1) / BAND_N)))
        r1 = int(round(H * (1.0 - i / BAND_N)))
        m = known[r0:r1]
        rec = {
            "v_lo": round(v_lo, 2),
            "v_hi": round(v_hi, 2),
            "coverage": round(float(m.mean()), 2),
        }
        if m.sum() >= 32:
            px = img[r0:r1][m]
            col = [int(v) for v in np.median(px, axis=0)]
            rec |= {
                "color_rgb": col,
                "color_hex": "#%02x%02x%02x" % tuple(col),
                "edge_med": round(float(np.median(grad[r0:r1][m])), 2),
            }
        else:
            rec |= {"color_rgb": None, "color_hex": None, "edge_med": None}
        bands.append(rec)

    valid = [b for b in bands if b["color_rgb"] and b["coverage"] >= 0.25]
    if len(valid) < 3:
        return {"bands": bands, "flag": None}
    base_col = np.median([b["color_rgb"] for b in valid], axis=0)  # the plain-wall colour/texture
    base_edge = float(np.median([b["edge_med"] for b in valid]))
    flagged = []
    for b in valid:
        dcol = float(np.linalg.norm(np.array(b["color_rgb"], float) - base_col))
        dedge = b["edge_med"] - base_edge
        if dcol >= BAND_COLOR_DELTA or dedge >= BAND_EDGE_DELTA:
            flagged.append((b, dcol, dedge))
    if not flagged:
        return {"bands": bands, "flag": None}
    lo = min(b["v_lo"] for b, _, _ in flagged)
    hi = max(b["v_hi"] for b, _, _ in flagged)
    b0, dc, de = max(flagged, key=lambda t: max(t[1] / BAND_COLOR_DELTA, t[2] / BAND_EDGE_DELTA))
    why = []
    if dc >= BAND_COLOR_DELTA:
        why.append(f"colour {b0['color_hex']} differs from the wall (Δ{dc:.0f})")
    if de >= BAND_EDGE_DELTA:
        why.append(f"more pattern/texture here (edge {b0['edge_med']} vs {base_edge:.1f})")
    return {
        "bands": bands,
        "flag": {
            "v_lo": round(lo, 2),
            "v_hi": round(hi, 2),
            "color_hex": b0["color_hex"],
            "reason": "; ".join(why),
        },
    }


def _triage_one(
    image_path: Path, unknown_mask_path: Path, coverage: float, height_m: float | None = None
) -> dict:
    """Classify a surface from the KNOWN region of its stitch.
    plain: low colour spread + low edge energy (uniform paint/plaster).
    detailed: structured colour/pattern (tile, wood planks, feature wall, splashback).
    low_coverage: barely observed — don't focus; lean on the shared material / obliques."""
    img = np.asarray(Image.open(image_path).convert("RGB")).astype(np.float32)
    unknown = np.asarray(Image.open(unknown_mask_path).convert("L")) > 127
    known = ~unknown
    if known.sum() < 64:
        return {
            "class": "low_coverage",
            "coverage": round(coverage, 3),
            "color_rgb": None,
            "color_hex": None,
            "color_mad": None,
            "edge_med": None,
            "edge_p90": None,
        }
    px = img[known]
    color_rgb = [int(v) for v in np.median(px, axis=0)]
    color_mad = float(np.median(np.abs(px - np.median(px, axis=0))))  # robust spread (reported)
    gray = img.mean(axis=2)
    gy, gx = np.gradient(gray)
    g = np.hypot(gx, gy)[known]
    edge_med = float(np.median(g))  # robust pattern signal -> the gate
    edge_p90 = float(np.percentile(g, 90))  # how strong the strongest structure is

    if coverage < LOW_COVERAGE:
        cls = "low_coverage"
    elif edge_med < PLAIN_EDGE_MED:
        cls = "plain"
    else:
        cls = "detailed"
    # localize a sub-wall material band (tiled splashback / wainscot / dado) — walls only
    band = _band_profile(img, known, height_m) if height_m else {"bands": [], "flag": None}
    return {
        "class": cls,
        "coverage": round(coverage, 3),
        "color_rgb": color_rgb,
        "color_hex": "#%02x%02x%02x" % tuple(color_rgb),
        "color_mad": round(color_mad, 1),
        "edge_med": round(edge_med, 2),
        "edge_p90": round(edge_p90, 2),
        "bands": band["bands"],
        "band_flag": band["flag"],
    }


def _pooled_wall_color(surfaces: list[dict]) -> dict | None:
    """One shared wall colour from the pooled plain walls (most rooms = one wall finish).
    Median of the plain walls' measured colours, weighted by coverage."""
    plains = [
        s
        for s in surfaces
        if s["name"].startswith("Wall")
        and s["triage"]["class"] == "plain"
        and s["triage"]["color_rgb"]
    ]
    if not plains:
        return None
    cols = np.array([s["triage"]["color_rgb"] for s in plains], dtype=float)
    rgb = [int(v) for v in np.median(cols, axis=0)]
    return {
        "color_rgb": rgb,
        "color_hex": "#%02x%02x%02x" % tuple(rgb),
        "from_walls": [s["name"] for s in plains],
    }


def build(force: bool = False) -> dict:
    """Stitch every surface + triage. Idempotent: skips if the manifest already exists
    (the reference is ground truth — same across iterations) unless ``force``."""
    config.ensure_dirs()  # this stage writes the stitches + manifest
    if config.SURFACE_REF_MANIFEST.is_file() and not force:
        return json.loads(config.SURFACE_REF_MANIFEST.read_text())

    stitch = _load_stitch()
    usdz = config.SCAN_DIR / "room.usdz"
    planes = stitch.load_wall_planes(usdz)
    if not planes:
        raise SystemExit(f"no wall planes parsed from {usdz}")
    floor, ceiling = _derive_floor_ceiling(stitch.WallPlane, planes)
    frame_pairs = stitch.find_frame_pairs(config.SCAN_DIR)
    if not frame_pairs:
        raise SystemExit(f"no frame_*.jpg/json pairs in {config.SCAN_DIR}")

    config.SURFACE_REF.mkdir(parents=True, exist_ok=True)
    surfaces: list[dict] = []
    for plane in list(planes) + [floor, ceiling]:
        out = config.SURFACE_REF / f"{plane.name}_stitched.jpg"
        # clean texture (no name/border overlay) so colour sampling is honest
        info = stitch.stitch_wall(
            plane, frame_pairs, out, pixels_per_meter=config.STITCH_PPM, annotate=False
        )
        # band profile is vertical → walls only (height_m = the wall's real height)
        h_m = info["surface_size_m"][1] if plane.name.startswith("Wall") else None
        tri = _triage_one(Path(info["image"]), Path(info["unknown_mask"]), info["coverage"], h_m)
        surfaces.append(
            {
                "name": plane.name,
                "kind": (
                    "floor"
                    if plane.name.startswith("Floor")
                    else "ceiling"
                    if plane.name.startswith("Ceiling")
                    else "wall"
                ),
                "image": info["image"],
                "unknown_mask": info["unknown_mask"],
                "size_m": info["surface_size_m"],
                # The ortho basis this image was rasterised on. Recorded because size_m alone is
                # NOT enough to put world geometry back onto the picture: the caller would have to
                # re-derive centre/axes, and the floor and ceiling deliberately differ in v_axis
                # sign ([0,0,-1] vs [0,0,+1], so both face the room). Guessing that sign mirrors
                # the overlay vertically. Use project_to_surface() rather than reading these raw.
                "center": [float(v) for v in plane.center],
                "u_axis": [float(v) for v in plane.u_axis],
                "v_axis": [float(v) for v in plane.v_axis],
                "triage": tri,
            }
        )
        print(
            f"    {plane.name:9} cov={info['coverage']:.0%} -> {tri['class']:12} "
            f"{tri.get('color_hex') or '--'}",
            flush=True,
        )

    manifest = {
        "scan": config.SCAN,
        "ppm": config.STITCH_PPM,
        "shared_wall_color": _pooled_wall_color(surfaces),
        "surfaces": surfaces,
    }
    config.SURFACE_REF_MANIFEST.write_text(json.dumps(manifest, indent=2))
    return manifest


def evidence_block(manifest: dict | None = None) -> str:
    """Markdown injected into the stage-1 prompt: the rectified reference per surface,
    its triage class, and the shared wall colour — so the agent reads the right pixels
    and spends effort where the triage says it matters."""
    if manifest is None:
        if not config.SURFACE_REF_MANIFEST.is_file():
            return ""
        manifest = json.loads(config.SURFACE_REF_MANIFEST.read_text())
    surfaces = manifest.get("surfaces", [])
    if not surfaces:
        return ""

    lines = [
        "## Rectified per-surface reference (an aid — use alongside the photos/overlay)",
        f"Each surface below has a head-on (rectified) stitch + an unknown-mask in "
        f"`{config.SURFACE_REF}`, so you can read its colour / pattern / finish square-on "
        "when that's easier than off the obliques. WHITE in the unknown mask = not "
        "observed; furniture/fixtures also smear onto the plane — neither is the material. "
        "The triage tag is a HINT on where effort likely matters:",
        "- **plain** → looks like uniform paint/plaster: the shared wall material (below) probably fits.",
        "- **detailed** → looks like real pattern/structure (tile, wood, splashback, feature wall): "
        "likely wants its own matched PBR + real-world pattern scale.",
        "- **low_coverage** → barely seen edge-on: lean on the shared material / the oblique frames.",
    ]
    shared = manifest.get("shared_wall_color")
    if shared:
        lines.append(
            f"\n**Shared wall material colour** = `{shared['color_hex']}` "
            f"(RGB {shared['color_rgb']}, pooled from plain walls "
            f"{', '.join(shared['from_walls'])}). Use it for every plain / low-coverage wall."
        )

    order = (
        [s for s in surfaces if s["kind"] == "wall"]
        + [s for s in surfaces if s["kind"] == "floor"]
        + [s for s in surfaces if s["kind"] == "ceiling"]
    )
    lines.append("")
    for s in order:
        t = s["triage"]
        stem = Path(s["image"]).name
        meta = (
            f"cov {t['coverage']:.0%}, colour {t.get('color_hex') or '--'}, "
            f"pattern(edge_med {t.get('edge_med')}, p90 {t.get('edge_p90')})"
        )
        lines.append(
            f"- **{s['name']}** [{t['class']}] — {meta}\n"
            f"  ref `{stem}` (+ `{Path(s['unknown_mask']).name}`), "
            f"size {s['size_m'][0]:.2f}×{s['size_m'][1]:.2f}m"
        )
    # The whole-wall [plain]/[detailed] tag HIDES a sub-wall material band (a tiled
    # splashback, wainscot, dado, wood/enamel lower section) — it only covers part of the
    # height, so the plain majority dominates. Auto-detecting it off this (furniture-smeared)
    # stitch is unreliable, so YOU judge it: for every wall — kitchens & bathrooms especially
    # — scan its rectified ref + photos top-to-bottom for a horizontal material boundary
    # (counter splashback ~0.9–1.5 m, dado/wainscot ~0.9–1.1 m). When you see one, give that
    # band its OWN recoloured PBR as a sub-wall zone with `add_wall_panel` (see the How section).
    lines.append(
        "\n**Sub-wall bands:** the tag above is WHOLE-wall — it won't tell you about a tiled "
        "splashback / wainscot / dado that covers only part of the height. Inspect each wall's "
        "ref top-to-bottom for a horizontal material boundary (esp. kitchens/bathrooms) and "
        "paint any real band as its own zone (`add_wall_panel`)."
    )
    return "\n".join(lines)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Build the stage-1 rectified surface references + triage."
    )
    ap.add_argument("--force", action="store_true", help="rebuild even if the manifest exists")
    # Declared so `--scene` is accepted and documented here too. It is acted on earlier than this:
    # `config` bootstraps from it at IMPORT time, because every path in this module is already
    # resolved by the time main() runs.
    ap.add_argument("--scene", default=None, help="scene package dir (holds scene.json)")
    a = ap.parse_args()
    man = build(force=a.force)
    print(f"\n{len(man['surfaces'])} surfaces -> {config.SURFACE_REF_MANIFEST}")
    print(evidence_block(man))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
