"""
select_views.py — minimal set of capture views that covers the whole room.

Given the per-view visibility already computed in
`output/room_render/render_manifest.json` (every object/wall/floor tagged
full/partial/occluded with its image bbox, via real mesh ray-cast occlusion),
pick the SMALLEST set of capture views such that every main object and (most of)
the walls are seen *well* in at least one chosen view — so you can reconstruct /
texture the scene from just those images.

This is the classic **greedy weighted set-cover** algorithm:

  universe  = the things to cover: each object (chair/table/door/window) and
              each wall. Floor is covered too but weighted low. (Ceiling is NOT
              in the RoomPlan reconstruction — walls+floor only — so it cannot be
              scored here; see the note at the bottom of this file.)
  covered   = an element counts as covered by a view only if its QUALITY in that
              view clears a bar: state (full>partial, occluded=0) x size-in-frame
              x how centred it is. So "covered" means "seen at a good angle",
              not merely "in frame".
  solve     = repeatedly take the view that adds the most NEW weighted coverage
              until every (coverable) object + wall is covered, then a backward
              pass drops any view that later picks made redundant -> minimal set.

Want each object from several angles (for multi-view reconstruction)? bump
MIN_VIEWS["object"] to 2+: a 2nd view only counts for an object if its viewing
direction differs from the already-chosen ones by >= DIVERSITY_DEG.

Outputs:
  output/view_selection.json   chosen views (in pick order) + full coverage report
  output/view_selection.csv    one row per chosen view
  output/view_selection.png    contact sheet of the chosen set (needs PIL)

Run (no Blender, no numpy):
    python3 select_views.py
"""

import csv
import json
import math
import os
import sys

from lrauthor import REPO_ROOT as _REPO
from lrauthor.fonts import font as _shared_font

HERE = os.path.dirname(os.path.abspath(__file__))
# Render dir + write dir are configurable via env (the harness sets these per scan). Fall back
# to the repo's run/ tree, never to HERE: HERE is inside the package, and an installed package
# has no business being written to (site-packages is often read-only).
_RENDER = os.environ.get("SB_RENDER_DIR") or os.path.join(_REPO, "run", "room_render")
_VOUT = os.environ.get("SB_VIEW_OUT") or os.path.join(_REPO, "run")
MANIFEST = os.path.join(_RENDER, "render_manifest.json")
SCENE = os.path.join(_RENDER, "scene_manifest.json")
FRAMES = os.environ.get("SB_SCAN_DIR", "")
OUT_JSON = os.path.join(_VOUT, "view_selection.json")
OUT_CSV = os.path.join(_VOUT, "view_selection.csv")
OUT_PNG = os.path.join(_VOUT, "view_selection.png")

# --------------------------------------------------------------------------- #
# Tunables — the whole policy lives here
# --------------------------------------------------------------------------- #
SHELL_CATS = {"wall", "floor", "ceiling"}

# value of covering one element of each type (drives the greedy priority)
WEIGHT = {"object": 3.0, "wall": 1.5, "floor": 0.5, "ceiling": 0.3}

# how many DISTINCT good views each element type wants (angular diversity kicks
# in when > 1). Floor/wall = 1; bump object to 2 for multi-angle reconstruction.
MIN_VIEWS = {"object": 1, "wall": 1, "floor": 1, "ceiling": 1}

# which types MUST be covered before we stop (the rest are bonus coverage)
REQUIRED_TYPES = {"object", "wall"}

# quality model: an element counts as covered only if quality >= COVER_THRESH.
# Two manifest schemas are supported:
#   rich (old): each visible entry has `img_pct` (bbox) + `state`  -> detail from
#               box size, occlusion from state.
#   lean (new): each visible entry has `label_pct` (point) + `dist_m`, and the
#               render already drops occluded/barely-visible objects -> presence
#               means "genuinely visible", detail from proximity.
STATE_W = {"full": 1.0, "partial": 0.6, "occluded": 0.0}  # rich schema only
REF_SIZE = {
    "object": 0.06,
    "wall": 0.12,
    "floor": 0.15,
    "ceiling": 0.15,
}  # frame-frac that = full detail (rich)
REF_DIST = {
    "object": 1.8,
    "wall": 2.5,
    "floor": 2.5,
    "ceiling": 3.0,
}  # metres that = full detail (lean)
COVER_THRESH = {"object": 0.20, "wall": 0.20, "floor": 0.15, "ceiling": 0.15}

DIVERSITY_DEG = 40.0  # min angular separation between two views of one object

# budget: how many views to return.
#   BUDGET        = the cap (max # of views).  CLI: `python3 select_views.py 10`
#   PAD_TO_BUDGET = False -> stop as soon as everything is covered (<= BUDGET).
#                 = True  -> always return exactly BUDGET, spending the extra
#                           views on the best additional angles. CLI: `... 10 exact`
# When padding, each ADDITIONAL diverse view of an element is worth DECAY^k of the
# first (diminishing returns), so the budget goes where it helps most.
BUDGET = 25
PAD_TO_BUDGET = False
DECAY = 0.5
EPS = 1e-6


# --------------------------------------------------------------------------- #
def etype(cat):
    return cat if cat in SHELL_CATS else "object"


def _vis_point(v):
    """Label/anchor point (x%, y%) in the upright frame, from either schema."""
    if "label_pct" in v:
        return float(v["label_pct"][0]), float(v["label_pct"][1])
    if "img_pct" in v:  # old box -> its centre
        bx, by = v["img_pct"]["x"], v["img_pct"]["y"]
        return (bx[0] + bx[1]) / 2.0, (by[0] + by[1]) / 2.0
    return 50.0, 50.0


def _size_term(v, t):
    """Detail the view gives of the element, in [0,1]: image-bbox fraction if the
    manifest has a box (rich), else a proximity proxy from dist_m (lean)."""
    if "img_pct" in v:
        bx, by = v["img_pct"]["x"], v["img_pct"]["y"]
        size = max(0.0, bx[1] - bx[0]) * max(0.0, by[1] - by[0]) / 10000.0
        return min(1.0, size / REF_SIZE[t])
    dist = float(v.get("dist_m", REF_DIST[t]))
    return max(0.3, min(1.0, REF_DIST[t] / max(dist, 1e-3)))


def quality(v):
    """How well element `v` (a manifest 'visible' entry) is seen in a view:
    presence x detail x centredness, in [0,1]. 0 => not usable. (In the lean
    schema, occlusion is already gated upstream, so presence => visible.)"""
    t = etype(v["category"])
    sw = STATE_W.get(v["state"], 0.0) if "state" in v else 1.0
    if sw <= 0.0:
        return 0.0
    x, y = _vis_point(v)
    centred = max(0.0, 1.0 - math.hypot(x - 50.0, y - 50.0) / 70.71)
    return sw * _size_term(v, t) * (0.85 + 0.15 * centred)


def _unit(a, b):
    d = [b[i] - a[i] for i in range(3)]
    n = math.sqrt(sum(c * c for c in d)) or 1.0
    return [c / n for c in d]


def _angle_deg(u, w):
    dot = max(-1.0, min(1.0, sum(u[i] * w[i] for i in range(3))))
    return math.degrees(math.acos(dot))


# --------------------------------------------------------------------------- #
# Build coverage tables
# --------------------------------------------------------------------------- #
def build(manifest, scene):
    centers = {o["id"]: o["center"] for o in scene["objects"]}
    types = {o["id"]: etype(o["category"]) for o in scene["objects"]}

    # cov[frame][eid] = quality ; campos[frame] = camera position
    cov, campos = {}, {}
    for fr in manifest["frames"]:
        f = fr["frame"]
        campos[f] = fr.get("camera_pos")
        seen = {}
        for v in fr["visible"]:
            q = quality(v)
            if q >= COVER_THRESH[etype(v["category"])]:
                seen[v["id"]] = max(seen.get(v["id"], 0.0), q)
        cov[f] = seen

    # required elements that are coverable by at least one view
    required = [
        eid for eid, t in types.items() if t in REQUIRED_TYPES and any(eid in cov[f] for f in cov)
    ]
    uncoverable = [eid for eid, t in types.items() if t in REQUIRED_TYPES and eid not in required]
    return cov, campos, centers, types, required, uncoverable


def view_dir(campos, centers, frame, eid):
    return _unit(campos[frame], centers[eid])


def diverse_enough(campos, centers, frame, eid, already):
    """Is `frame`'s angle on `eid` far enough from every already-chosen view?"""
    if eid not in centers or campos.get(frame) is None:
        return True
    d_new = view_dir(campos, centers, frame, eid)
    for f_old in already:
        if campos.get(f_old) is None:
            continue
        if _angle_deg(d_new, view_dir(campos, centers, f_old, eid)) < DIVERSITY_DEG:
            return False
    return True


# --------------------------------------------------------------------------- #
# Greedy weighted set-cover  (+ redundancy trim)
# --------------------------------------------------------------------------- #
def _dir_safe(campos, centers, frame, eid):
    if eid not in centers or campos.get(frame) is None:
        return None
    return view_dir(campos, centers, frame, eid)


def marginal(frame, eid, q, t, covered_by, campos, centers):
    """Value of adding `frame` for element `eid`, given the diverse views already
    covering it: first cover = full weight; each ADDITIONAL diverse angle =
    DECAY**k of it; a redundant (similar) angle = 0. Returns (value, depth)."""
    existing = covered_by[eid]
    depth = len(existing)
    if depth == 0:
        return WEIGHT[t] * q, depth
    d_new = _dir_safe(campos, centers, frame, eid)
    if d_new is not None:
        for e in existing:
            if e["dir"] is not None and _angle_deg(d_new, e["dir"]) < DIVERSITY_DEG:
                return 0.0, depth  # an angle we already have
    return WEIGHT[t] * q * (DECAY**depth), depth


def gain_of(frame, cov, covered_by, campos, centers, types):
    """Marginal value `frame` adds, split so NEW coverage always beats extra
    angles: returns (first_cover_value, extra_angle_value, [(eid, q, depth), ...]).
    Greedy compares these lexicographically -> a view that covers something not
    yet covered is always preferred over one that only adds another angle."""
    first, extra, contrib = 0.0, 0.0, []
    for eid, q in cov[frame].items():
        v, depth = marginal(frame, eid, q, types[eid], covered_by, campos, centers)
        if v <= 0.0:
            continue
        if depth == 0:
            first += v
        else:
            extra += v
        contrib.append((eid, round(q, 3), depth))
    return first, extra, contrib


def _required_satisfied(covered_by, types, required):
    return all(len(covered_by[e]) >= MIN_VIEWS[types[e]] for e in required)


def solve(cov, campos, centers, types, required):
    covered_by = {eid: [] for eid in types}
    chosen = []
    # greedy: each round take the view with the best (new-coverage, extra-angle)
    # value, lexicographically. New coverage dominates, so everything required is
    # covered as fast as possible; then we stop (cap mode) or keep spending the
    # budget on the best extra angles (pad mode). Always bounded by BUDGET.
    while len(chosen) < BUDGET:
        satisfied = _required_satisfied(covered_by, types, required)
        if satisfied and not PAD_TO_BUDGET:
            break
        best, best_key, best_c = None, (-1.0, -1.0), None
        for f in cov:
            if f in chosen:
                continue
            first, extra, c = gain_of(f, cov, covered_by, campos, centers, types)
            if (first > EPS or extra > EPS) and (first, extra) > best_key:
                best, best_key, best_c = f, (first, extra), c
        if best is None:  # nothing more worth adding
            break
        if not satisfied and best_key[0] <= EPS:  # remaining required uncoverable
            break
        chosen.append(best)
        for eid, q, depth in best_c:
            covered_by[eid].append(
                {"frame": best, "q": q, "dir": _dir_safe(campos, centers, best, eid)}
            )

    # backward trim (cap/minimal mode only — padding wants to keep the budget):
    # drop any view a later pick made redundant for REQUIRED coverage.
    if not PAD_TO_BUDGET:
        for f in list(reversed(chosen)):
            if _still_covered(chosen, f, cov, campos, centers, types, required):
                chosen.remove(f)
    return finalize(chosen, cov, campos, centers, types, required)


def _still_covered(chosen, drop, cov, campos, centers, types, required):
    """Would every required element still be covered (with MIN_VIEWS + diversity)
    if `drop` were removed from `chosen`?"""
    keep = [f for f in chosen if f != drop]
    covered_by = {eid: [] for eid in types}
    for f in keep:
        for eid, q in cov[f].items():
            t = types[eid]
            if len(covered_by[eid]) >= MIN_VIEWS[t]:
                continue
            if MIN_VIEWS[t] > 1 and not diverse_enough(campos, centers, f, eid, covered_by[eid]):
                continue
            covered_by[eid].append(f)
    return all(len(covered_by[e]) >= MIN_VIEWS[types[e]] for e in required)


def finalize(chosen, cov, campos, centers, types, required):
    """Recompute, for the FINAL set, what each view contributes (in order),
    tagging each pick core/extra and each add with its angle depth (0 = first
    time the element is covered, >=1 = an additional angle)."""
    covered_by = {eid: [] for eid in types}
    picks = []
    for f in chosen:
        phase = "core" if not _required_satisfied(covered_by, types, required) else "extra"
        adds = []
        for eid, q in sorted(cov[f].items(), key=lambda kv: -kv[1]):
            v, depth = marginal(f, eid, q, types[eid], covered_by, campos, centers)
            if v <= 0.0:
                continue
            covered_by[eid].append(
                {"frame": f, "q": round(q, 3), "dir": _dir_safe(campos, centers, f, eid)}
            )
            adds.append((eid, round(q, 3), depth))
        picks.append({"frame": f, "phase": phase, "adds": adds})
    return chosen, picks, covered_by


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def report(chosen, picks, covered_by, cov, types, required, uncoverable):
    def split(ids):
        d = {"object": [], "wall": [], "floor": []}
        for e in ids:
            d.get(types.get(e, "object"), d["object"]).append(e)
        return {k: sorted(v) for k, v in d.items() if v}

    elements = {}
    for eid in types:
        covs = sorted(
            ((f, round(cov[f][eid], 3)) for f in chosen if eid in cov[f]), key=lambda t: -t[1]
        )
        if covs or types[eid] in REQUIRED_TYPES:
            elements[eid] = {
                "type": types[eid],
                "covered": len(covs) > 0,
                "best_quality": covs[0][1] if covs else 0.0,
                "views": [f for f, _ in covs],
            }

    req_covered = [e for e in required if elements.get(e, {}).get("covered")]
    rep_picks = []
    for p in picks:
        new_ids = [e for e, _, d in p["adds"] if d == 0]
        extra_ids = [e for e, _, d in p["adds"] if d >= 1]
        rep_picks.append(
            {
                "frame": p["frame"],
                "phase": p["phase"],
                "new": split(new_ids),
                "n_new": len(new_ids),
                "extra_angles": split(extra_ids),
                "n_extra": len(extra_ids),
            }
        )
    return {
        "policy": {
            "budget": BUDGET,
            "pad_to_budget": PAD_TO_BUDGET,
            "weight": WEIGHT,
            "min_views": MIN_VIEWS,
            "required_types": sorted(REQUIRED_TYPES),
            "cover_thresh": COVER_THRESH,
            "diversity_deg": DIVERSITY_DEG,
            "decay": DECAY,
        },
        "n_chosen": len(chosen),
        "chosen": chosen,
        "summary": {
            "required_total": len(required),
            "required_covered": len(req_covered),
            "fully_covered": len(req_covered) == len(required),
            "uncoverable_by_any_view": sorted(uncoverable),
        },
        "picks": rep_picks,
        "elements": elements,
    }


def write_outputs(rep):
    with open(OUT_JSON, "w") as fh:
        json.dump(rep, fh, indent=2)
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "order",
                "frame",
                "phase",
                "n_new",
                "n_extra_angles",
                "new_objects",
                "new_walls",
                "new_floor",
                "extra_angle_objects",
            ]
        )
        for i, p in enumerate(rep["picks"], 1):
            n, e = p["new"], p["extra_angles"]
            w.writerow(
                [
                    i,
                    p["frame"],
                    p["phase"],
                    p["n_new"],
                    p["n_extra"],
                    " ".join(n.get("object", [])),
                    " ".join(n.get("wall", [])),
                    " ".join(n.get("floor", [])),
                    " ".join(e.get("object", [])),
                ]
            )


def print_report(rep):
    s = rep["summary"]
    mode = (
        f"exactly {rep['policy']['budget']}"
        if rep["policy"]["pad_to_budget"]
        else f"minimal (<= {rep['policy']['budget']})"
    )
    head = "FULL" if s["fully_covered"] else "PARTIAL"
    print(
        f"\n{head} COVERING SET [{mode}]: {rep['n_chosen']} views -> "
        f"{s['required_covered']}/{s['required_total']} required elements covered"
    )
    print("-" * 74)
    for i, p in enumerate(rep["picks"], 1):
        parts = []
        for t in ("object", "wall", "floor"):
            if p["new"].get(t):
                parts.append(f"{t}: {', '.join(p['new'][t])}")
        line = f"  {i}. {p['frame']}  [{p['phase']:<5}] +{p['n_new']:>2} new"
        if parts:
            line += "  | " + " | ".join(parts)
        if p["n_extra"]:
            ang = ", ".join(p["extra_angles"].get("object", []) + p["extra_angles"].get("wall", []))
            line += f"   (+{p['n_extra']} extra angle: {ang})"
        print(line)
    if s["uncoverable_by_any_view"]:
        print(
            f"\n  CAPTURE GAPS (no view sees these well): {', '.join(s['uncoverable_by_any_view'])}"
        )
    weak = [
        e
        for e, r in rep["elements"].items()
        if r["type"] in REQUIRED_TYPES and r["covered"] and r["best_quality"] < 0.5
    ]
    if weak:
        print(f"  WEAKLY covered (best quality < 0.5): {', '.join(sorted(weak))}")


# --------------------------------------------------------------------------- #
# Montage of the chosen set (optional; needs PIL)
# --------------------------------------------------------------------------- #
def montage(rep):
    try:
        from PIL import Image, ImageDraw
    except Exception:
        print("  (PIL not available -> skipping montage)")
        return

    def font(sz, bold=False):  # noqa: ARG001 — the shared resolver is bold-first already
        return _shared_font(sz)

    COLS = 3
    TW = 360
    TH = TW * 4 // 3
    CAP = 96
    PAD = 14
    TITLE = 56
    BG, FG, SUB = (24, 26, 30), (236, 238, 242), (150, 156, 166)
    CORE, EXTRA = (120, 200, 140), (110, 160, 220)  # green=core, blue=extra angle
    f_title, f_rank, f_body = font(28, True), font(22, True), font(15)

    picks = rep["picks"]
    nrows = (len(picks) + COLS - 1) // COLS
    cw, ch = TW + PAD, TH + CAP + PAD
    W, H = COLS * cw + PAD, TITLE + nrows * ch + PAD
    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)
    mode = "exactly " if rep["policy"]["pad_to_budget"] else "minimal — "
    d.text(
        (PAD, 14),
        f"Covering set ({mode}{rep['n_chosen']} views)  green=core  blue=extra angle",
        font=f_title,
        fill=FG,
    )

    for i, p in enumerate(picks):
        acc = CORE if p["phase"] == "core" else EXTRA
        im = Image.open(os.path.join(FRAMES, p["frame"] + ".jpg")).convert("RGB")
        im = im.transpose(Image.ROTATE_270).resize((TW, TH), Image.LANCZOS)
        tile = Image.new("RGB", (TW, TH + CAP), BG)
        tile.paste(im, (0, 0))
        dd = ImageDraw.Draw(tile)
        dd.rectangle([0, 0, TW - 1, 5], fill=acc)
        y = TH + 6
        dd.text((6, y), f"{i + 1}. {p['frame']}", font=f_rank, fill=acc)
        for t, lbl in (("object", "obj"), ("wall", "wall"), ("floor", "floor")):
            if p["new"].get(t):
                y += 23
                dd.text((6, y), f"{lbl}: {', '.join(p['new'][t])}"[:54], font=f_body, fill=SUB)
        if p["n_extra"]:
            y += 23
            ang = ", ".join(p["extra_angles"].get("object", []) + p["extra_angles"].get("wall", []))
            dd.text((6, y), f"+angle: {ang}"[:54], font=f_body, fill=EXTRA)
        r, c = divmod(i, COLS)
        canvas.paste(tile, (PAD + c * cw, TITLE + r * ch))

    canvas.save(OUT_PNG)
    print(f"  montage -> {os.path.relpath(OUT_PNG, HERE)} ({W}x{H})")


# --------------------------------------------------------------------------- #
def main():
    global BUDGET, PAD_TO_BUDGET
    args = [a for a in sys.argv[1:]]
    if args and args[0].lstrip("-").isdigit():
        BUDGET = int(args[0])
    if any(a.lower() in ("exact", "pad", "fill") for a in args):
        PAD_TO_BUDGET = True

    manifest = json.load(open(MANIFEST))
    scene = json.load(open(SCENE))
    cov, campos, centers, types, required, uncoverable = build(manifest, scene)

    counts = {}
    for t in types.values():
        counts[t] = counts.get(t, 0) + 1
    print(
        f"scene elements: {counts} | required (coverable): {len(required)} "
        f"| budget {BUDGET} | pad_to_budget {PAD_TO_BUDGET} | min_views {MIN_VIEWS}"
    )

    chosen, picks, covered_by = solve(cov, campos, centers, types, required)
    rep = report(chosen, picks, covered_by, cov, types, required, uncoverable)
    write_outputs(rep)
    print_report(rep)
    montage(rep)
    print(f"\nwrote {os.path.relpath(OUT_JSON, HERE)} + {os.path.relpath(OUT_CSV, HERE)}")
    print("SELECTED:", ", ".join(chosen))
    return rep


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------
# Note on CEILING: the RoomPlan reconstruction is walls + floor only — there is
# no ceiling mesh, so no element exists to score "ceiling coverage" against. If
# you want ceiling shots too, the cheap proxy is to pick the few views whose
# camera looks most steeply upward (forward.z large) from room_layout.json's
# camera list and append them — say the word and I'll wire that in.
# -----------------------------------------------------------------------------
