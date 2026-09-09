"""wall_refs.py — curated WALL image references for the agent (image_selection/surface_views).

Per wall/floor/ceiling: the minimal set of SHARP real frames that COVER it, with the wall
OUTLINED (orange box) + labelled, plus coverage% / blur notices. Exported as per-surface
reference tiles + a contact sheet + a manifest, and surfaced as an evidence block the prompt
injects in BOTH stage 1 (materials) and stage 2 (wall fixtures) — so the agent compares the
curated reference frame against the render for every wall.

This is the wall-selection we built (sharp + boxed + minimal-covering), distinct from the older
`surface_reference` stitch (a holey, furniture-smeared mosaic). Outputs land in
`scene_stage/_harness/wall_refs/` (shared, built once; recorded via the prompt + trajectory).
"""

from __future__ import annotations

import json

from lrauthor.agent.tools.shared import config
from lrauthor.fonts import font as _shared_font

REFS_DIR = config.HARNESS_OUT / "wall_refs"
MANIFEST = REFS_DIR / "wall_refs.json"
SHEET = REFS_DIR / "wall_refs_sheet.jpg"


def _load():
    from lrauthor.agent.tools.shared.image_selection import surface_views as SV
    from lrauthor.agent.tools.shared.stitch_wall_image import stitch_wall as SW

    return SV, SW


def build(force: bool = False) -> dict:
    """Select sharp covering frames per surface, export boxed reference tiles + sheet + manifest.
    Idempotent — the reference is ground truth, shared across iterations."""
    if MANIFEST.exists() and not force:
        return json.loads(MANIFEST.read_text())
    SV, SW = _load()
    REFS_DIR.mkdir(parents=True, exist_ok=True)
    planes = SW.load_wall_planes(config.SCAN_DIR / "room.usdz")
    F, C = SV.derive_floor_ceiling(planes)
    pairs = SW.find_frame_pairs(config.SCAN_DIR)
    out = {"scan": config.SCAN, "surfaces": []}
    sheet_tiles = []
    for p in list(planes) + [F, C]:
        if p.name.startswith("Wall") and (
            p.width_m * p.height_m < SV.MIN_WALL_AREA
            or min(p.width_m, p.height_m) < SV.MIN_WALL_SIDE
        ):
            continue
        rows, total = SV.analyze(p, pairs)
        chosen = SV.select(rows)
        if not chosen:
            continue
        cov = set().union(*[c["bins"] for c in chosen])
        cov_pct = round(100 * len(cov) / total, 0)
        blurry = all(c["sharp"] < SV.BLUR_THRESH for c in chosen)
        low = cov_pct < SV.LOWCOV_PCT
        tiles = []
        for i, c in enumerate(chosen):
            tile = SV.highlight(p, c)  # the wall-boxed, labelled reference frame
            tp = REFS_DIR / f"{p.name}_ref{i}.jpg"
            tile.save(tp, quality=90)
            tiles.append(
                {
                    "path": str(tp),
                    "frame": c["frame"],
                    "sharp": round(c["sharp"], 1),
                    "blurry": c["sharp"] < SV.BLUR_THRESH,
                }
            )
            sheet_tiles.append(
                (
                    f"{p.name}  {cov_pct:.0f}% cover"
                    + ("  blurry" if c["sharp"] < SV.BLUR_THRESH else ""),
                    tile,
                )
            )
        out["surfaces"].append(
            {
                "name": p.name,
                "kind": (
                    "floor"
                    if p.name.startswith("Floor")
                    else "ceiling"
                    if p.name.startswith("Ceiling")
                    else "wall"
                ),
                "coverage_pct": cov_pct,
                "blurry": blurry,
                "low_coverage": low,
                "tiles": tiles,
            }
        )
    MANIFEST.write_text(json.dumps(out, indent=2))
    _build_sheet(sheet_tiles, SHEET)
    return out


def _build_sheet(named_tiles, out_path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    if not named_tiles:
        return
    try:
        F = _shared_font(30)
    except Exception:
        F = ImageFont.load_default()
    TW, HEAD = 460, 40
    rows = []
    for label, tile in named_tiles:
        im = tile.resize((TW, int(TW * tile.height / tile.width)))
        row = Image.new("RGB", (TW, im.height + HEAD), (8, 10, 14))
        d = ImageDraw.Draw(row)
        d.rectangle((0, 0, TW, HEAD), fill=(34, 40, 52))
        d.text((8, 6), label, fill=(255, 220, 150), font=F)
        row.paste(im, (0, HEAD))
        rows.append(row)
    W = max(r.width for r in rows)
    sheet = Image.new("RGB", (W, sum(r.height for r in rows) + 6 * (len(rows) + 1)), (8, 10, 14))
    y = 6
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height + 6
    sheet.save(out_path, quality=88)


def evidence_block() -> str:
    """The prompt block — per-wall reference tiles + the compare instruction. Used in BOTH stages."""
    man = build()
    lines = [
        "## Wall image references — SHARP, wall-BOXED frames that COVER each surface",
        "Per surface: the minimal set of sharp real frames that cover it, with the wall OUTLINED "
        "(orange box) + labelled so you can read its TRUE material + what is mounted on it. "
        "For EVERY surface, open BOTH its reference frame(s) below AND the matching render view, and "
        "compare them. `coverage%` = how much of the surface the frames show; heed the blur/low-coverage flags.",
        f"Contact sheet (all surfaces at a glance): `{SHEET}`",
    ]
    for s in man["surfaces"]:
        fl = []
        if s["low_coverage"]:
            fl.append("LOW COVERAGE")
        if s["blurry"]:
            fl.append("blurry")
        flag = ("  ⚠ " + ", ".join(fl)) if fl else ""
        paths = "; ".join(f"`{t['path']}` (sharp {t['sharp']})" for t in s["tiles"])
        lines.append(f"- **{s['name']}** — {s['coverage_pct']:.0f}% covered{flag}: {paths}")
    return "\n".join(lines)


if __name__ == "__main__":
    m = build(force=True)
    print(f"built wall_refs for {m['scan']}: {len(m['surfaces'])} surfaces -> {REFS_DIR}")
    for s in m["surfaces"]:
        print(
            f"  {s['name']}: {len(s['tiles'])} ref(s), {s['coverage_pct']:.0f}% cover"
            + (" blurry" if s["blurry"] else "")
            + (" lowcov" if s["low_coverage"] else "")
        )
