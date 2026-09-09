#!/usr/bin/env python3
"""select_for.py — pick the photos that show one target most clearly (by wall, or by object).

The global `select_views.py` picks the smallest set of views covering the whole room. This does
the opposite: given ONE target — a wall (Wall1) or an object (Table0) — it ranks every photo by
how clearly that target is visible and returns the best N. It reuses `select_views.quality`
(size as a distance proxy x centredness as label_pct), so the measure matches global selection.

The harness uses this to choose verification photos when it is working on one wall (stage 2
fixtures) or one object (stage 3 refine).

  python authoring/views/room_render/select_for.py --target Wall1 --n 3
  python authoring/views/room_render/select_for.py --target Table0 --manifest <render_manifest.json>

stdout: comma-separated frame indices, ready to feed to render_room_cameras.
stderr: the quality and distance for each.
"""

import argparse
import json
import os
import sys

from lrauthor import REPO_ROOT

# `select_views` sits beside this file and is now reached by a real import rather than the
# `sys.path` insert it needed when the two lived apart. Only `quality()` is used from it — the
# scorer both the whole-room cover and this per-target ranking share, so the two agree.
from lrauthor.agent.tools.select_views.source import select_views as SV


def views_for(manifest: dict, target_id: str, n: int = 3, min_quality: float = 0.5) -> list[dict]:
    """How well the target is visible in each photo, best first. Keeps only those scoring at
    least min_quality — clear enough to be useful — and at most N of them."""
    rows = []
    for fr in manifest["frames"]:
        for v in fr.get("visible", []):
            if v["id"] == target_id:
                rows.append(
                    {
                        "frame": fr["frame"],
                        "quality": round(SV.quality(v), 3),
                        "dist_m": v.get("dist_m"),
                        "label_pct": v.get("label_pct"),
                    }
                )
                break
    rows = [r for r in rows if r["quality"] >= min_quality]  # below the bar: too unclear to use
    rows.sort(key=lambda r: -r["quality"])
    return rows[:n]


def _num(frame_id: str) -> str:
    return str(int(frame_id.split("_")[1]))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--target", required=True, help="id of a wall (e.g. Wall1) or an object (e.g. Table0)")
    ap.add_argument("--n", type=int, default=3, help="how many views to return at most (default 3)")
    ap.add_argument(
        "--min-quality",
        type=float,
        default=0.5,
        help="drop views scoring below this quality — too unclear to use (default 0.5)",
    )
    ap.add_argument(
        "--manifest",
        # Anchored on the data root. This used to join a `ROOT` derived from __file__, which
        # resolved inside the package rather than the data tree — the default never existed.
        default=os.path.join(REPO_ROOT, "run", "room_render", "render_manifest.json"),
        help="render_manifest.json (needs the full visibility pass from describe-all)",
    )
    args = ap.parse_args()

    if not os.path.isfile(args.manifest):
        print(f"manifest not found: {args.manifest} (render with describe-all first)", file=sys.stderr)
        return 2
    m = json.load(open(args.manifest))
    rows = views_for(m, args.target, args.n, args.min_quality)
    if not rows:
        ids = sorted({v["id"] for fr in m["frames"] for v in fr.get("visible", [])})
        print(
            f"# no view of {args.target} reaches quality>={args.min_quality} — none is clear "
            f"enough. Visible ids: {ids}",
            file=sys.stderr,
        )
        return 2

    print(",".join(_num(r["frame"]) for r in rows) + ",")  # stdout: frame spec
    for r in rows:
        print(
            f"  {r['frame']}  quality={r['quality']}  dist={r['dist_m']}m  pos={r['label_pct']}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
