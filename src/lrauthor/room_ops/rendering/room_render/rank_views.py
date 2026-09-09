"""
rank_views.py — rank the capture views by how much of the room they cover.

Goal (per the plan): score every capture view so the ones that show the *most*
float to the top — a view that covers a large area of the room AND a lot of
distinct objects AND fills the frame with stuff.

It does this WITHOUT re-rendering: `authoring/views/room_render/render_room_cameras.py`
already walked every camera and wrote, per object, whether it is visible and
where in the frame — with real mesh ray-cast occlusion (so an object hidden
behind a wall/table is correctly marked `occluded`). That lives in
`output/room_render/render_manifest.json`. We read it and turn each view's
visibility into a coverage score.

Three terms, one per thing the plan asked for, each normalised across all views
to [0,1] then averaged (balanced weights):

  1. objects   "cover a lot of objects"        # distinct non-shell objects seen
  2. area      "cover a large area of the room" floor's screen coverage
  3. fill      "cover a lot of stuff overall"   frame fraction filled by objects

Outputs:
  output/view_ranking.json   full ranking + per-view breakdown
  output/view_ranking.csv    same, as a flat table
  (montage is built separately by make_view_montage.py — needs PIL)

Run (no Blender, no numpy needed):
    python3 rank_views.py
"""

import csv
import json
import os

from lrauthor import REPO_ROOT as _REPO

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
# Fall back to the repo's run/ tree, never to HERE: HERE is inside the package, and an
# installed package has no business being written to (site-packages is often read-only).
_RENDER = os.environ.get("SB_RENDER_DIR") or os.path.join(_REPO, "run", "room_render")
_VOUT = os.environ.get("SB_VIEW_OUT") or os.path.join(_REPO, "run")
MANIFEST = os.path.join(_RENDER, "render_manifest.json")
SCENE_MANIFEST = os.path.join(_RENDER, "scene_manifest.json")
OUT_JSON = os.path.join(_VOUT, "view_ranking.json")
OUT_CSV = os.path.join(_VOUT, "view_ranking.csv")

# what counts as "an object" (the stuff) vs the room shell
SHELL_CATS = {"wall", "floor", "ceiling"}

# how much a visible object is worth, by how clearly it is seen
STATE_WEIGHT = {"full": 1.0, "partial": 0.5, "occluded": 0.0}

# balanced weights (sum to 1). Retune here if you want to favour one criterion.
W_OBJECTS = 1.0 / 3.0  # cover a lot of objects
W_AREA = 1.0 / 3.0  # cover a large area of the room
W_FILL = 1.0 / 3.0  # cover a lot of stuff overall

FILL_GRID = 100  # rasterisation resolution for the union-of-boxes fill


# --------------------------------------------------------------------------- #
# Per-view raw terms
# --------------------------------------------------------------------------- #
def _box_frac(img_pct):
    """Screen-area fraction [0,1] of an object's image bbox (img_pct in 0..100)."""
    x0, x1 = img_pct["x"]
    y0, y1 = img_pct["y"]
    return max(0.0, (x1 - x0)) * max(0.0, (y1 - y0)) / 10000.0


def _union_fill(boxes):
    """Fraction of the frame covered by the UNION of image boxes (so overlapping
    objects aren't double-counted). Coarse raster — exact enough for ranking."""
    if not boxes:
        return 0.0
    g = FILL_GRID
    covered = bytearray(g * g)
    for b in boxes:
        x0 = max(0, min(g, int(b["x"][0] * g / 100)))
        x1 = max(0, min(g, int(round(b["x"][1] * g / 100))))
        y0 = max(0, min(g, int(b["y"][0] * g / 100)))
        y1 = max(0, min(g, int(round(b["y"][1] * g / 100))))
        for yy in range(y0, y1):
            row = yy * g
            for xx in range(x0, x1):
                covered[row + xx] = 1
    return sum(covered) / float(g * g)


def _vis_point(v):
    """Label point (x%, y%) of a visible entry, from either manifest schema."""
    if "label_pct" in v:
        return float(v["label_pct"][0]), float(v["label_pct"][1])
    if "img_pct" in v:
        bx, by = v["img_pct"]["x"], v["img_pct"]["y"]
        return (bx[0] + bx[1]) / 2.0, (by[0] + by[1]) / 2.0
    return 50.0, 50.0


def _spread(points):
    """Frame-area fraction [0,1] of the bbox enclosing a set of label points —
    how widely the content is spread across the frame."""
    if len(points) < 2:
        return 0.0
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return (max(xs) - min(xs)) * (max(ys) - min(ys)) / 10000.0


def raw_terms(view, n_target_objects, rich):
    """The three raw coverage terms for one view, plus bookkeeping.

    rich schema (img_pct + state): objects=state-weighted count, area=floor box
    coverage, fill=union of object boxes (the original, most faithful version).
    lean schema (label_pct only; occlusion already gated by the render): objects=
    visible-object count, area=spread of ALL visible labels (how much of the room
    is in view), fill=spread of OBJECT labels (how spread the stuff is)."""
    vis = view["visible"]
    objs = [v for v in vis if v["category"] not in SHELL_CATS]
    floor = next((v for v in vis if v["category"] == "floor"), None)

    if rich:
        seen = [v for v in objs if STATE_WEIGHT.get(v["state"], 0.0) > 0.0]
        obj_score = sum(STATE_WEIGHT.get(v["state"], 0.0) for v in objs)
        area_term = _box_frac(floor["img_pct"]) if floor else 0.0
        fill_term = _union_fill([v["img_pct"] for v in seen])
        nf = sum(1 for v in objs if v["state"] == "full")
        npd = sum(1 for v in objs if v["state"] == "partial")
        noc = sum(1 for v in objs if v["state"] == "occluded")
    else:
        seen = objs  # presence == genuinely visible
        obj_score = len(objs)
        area_term = _spread([_vis_point(v) for v in vis])
        fill_term = _spread([_vis_point(v) for v in objs])
        nf, npd, noc = len(objs), 0, 0

    obj_term = min(1.0, obj_score / n_target_objects) if n_target_objects else 0.0
    return {
        "objects_raw": round(obj_term, 4),
        "area_raw": round(area_term, 4),
        "fill_raw": round(fill_term, 4),
        "n_objects_seen": len(seen),
        "n_objects_full": nf,
        "n_objects_partial": npd,
        "n_objects_occluded": noc,
        "objects_seen": [
            {"id": v["id"], "cat": v["category"], "state": v.get("state", "visible")}
            for v in sorted(seen, key=lambda d: d.get("dist_m", 9e9))
        ],
        "floor_visible": floor is not None,
        "looking_at": view.get("looking_at"),
    }


# --------------------------------------------------------------------------- #
# Normalise + combine
# --------------------------------------------------------------------------- #
def _normalize(values):
    """Min-max each term to [0,1] across all views, so the balanced average is a
    fair blend (a term with a naturally small spread doesn't get drowned out).
    Returns a function value -> normalised."""
    lo, hi = min(values), max(values)
    span = hi - lo
    if span < 1e-9:
        return lambda x: 0.0
    return lambda x: (x - lo) / span


def _is_rich(manifest):
    """Rich schema = visible entries carry image bboxes (img_pct) + state."""
    for fr in manifest["frames"]:
        for v in fr["visible"]:
            return "img_pct" in v and "state" in v
    return False


def rank(manifest, n_target_objects):
    rich = _is_rich(manifest)
    rows = [
        dict(
            frame=v["frame"],
            image=v["frame"] + ".jpg",
            camera_pos=v.get("camera_pos"),
            **raw_terms(v, n_target_objects, rich),
        )
        for v in manifest["frames"]
    ]

    norm_o = _normalize([r["objects_raw"] for r in rows])
    norm_a = _normalize([r["area_raw"] for r in rows])
    norm_f = _normalize([r["fill_raw"] for r in rows])

    for r in rows:
        r["objects_n"] = round(norm_o(r["objects_raw"]), 4)
        r["area_n"] = round(norm_a(r["area_raw"]), 4)
        r["fill_n"] = round(norm_f(r["fill_raw"]), 4)
        r["score"] = round(
            W_OBJECTS * r["objects_n"] + W_AREA * r["area_n"] + W_FILL * r["fill_n"], 4
        )

    rows.sort(key=lambda r: r["score"], reverse=True)
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def write_outputs(rows, n_target_objects):
    doc = {
        "weights": {"objects": W_OBJECTS, "area": W_AREA, "fill": W_FILL},
        "terms": {
            "objects": "weighted count of distinct non-shell objects visible "
            "(full=1.0, partial=0.5, occluded=0.0), / total in scene",
            "area": "floor's image-space bbox area fraction (visible room area proxy)",
            "fill": "frame fraction covered by the union of visible object boxes",
            "note": "each raw term is min-max normalised across views, then the "
            "score is their weighted average (the *_n fields).",
        },
        "n_target_objects": n_target_objects,
        "n_views": len(rows),
        "ranking": rows,
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(doc, fh, indent=2)

    cols = [
        "rank",
        "frame",
        "score",
        "objects_n",
        "area_n",
        "fill_n",
        "objects_raw",
        "area_raw",
        "fill_raw",
        "n_objects_seen",
        "n_objects_full",
        "n_objects_partial",
        "n_objects_occluded",
    ]
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def print_table(rows):
    print(
        f"\n{'#':>3} {'frame':<13} {'score':>6} | {'obj':>5} {'area':>5} "
        f"{'fill':>5} | {'seen':>4} {'F/P/O':>8}  looking_at"
    )
    print("-" * 78)
    for r in rows:
        la = (r["looking_at"] or {}).get("id", "-")
        fpo = f"{r['n_objects_full']}/{r['n_objects_partial']}/{r['n_objects_occluded']}"
        print(
            f"{r['rank']:>3} {r['frame']:<13} {r['score']:>6.3f} | "
            f"{r['objects_n']:>5.2f} {r['area_n']:>5.2f} {r['fill_n']:>5.2f} | "
            f"{r['n_objects_seen']:>4} {fpo:>8}  {la}"
        )


def main():
    manifest = json.load(open(MANIFEST))
    counts = json.load(open(SCENE_MANIFEST))["counts"]
    n_target = sum(n for c, n in counts.items() if c not in SHELL_CATS)
    print(
        f"scene: {sum(counts.values())} objects | {n_target} non-shell "
        f"(targets): { {c: n for c, n in counts.items() if c not in SHELL_CATS} }"
    )
    schema = (
        "rich (img_pct+state)"
        if _is_rich(manifest)
        else "lean (label_pct: area=label-spread, fill=object-spread)"
    )
    print(
        f"views: {len(manifest['frames'])} | weights obj/area/fill = "
        f"{W_OBJECTS:.2f}/{W_AREA:.2f}/{W_FILL:.2f} | manifest schema: {schema}"
    )

    rows = rank(manifest, n_target)
    write_outputs(rows, n_target)
    print_table(rows)
    print(f"\nwrote {os.path.relpath(OUT_JSON, HERE)} and {os.path.relpath(OUT_CSV, HERE)}")
    print("TOP 5:", ", ".join(r["frame"] for r in rows[:5]))
    return rows


if __name__ == "__main__":
    main()
