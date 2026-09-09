"""sheet.py — composite the per-surface RENDER | REAL(stitch) comparison sheet.

Pairs each <Surface>_ortho.png with its <Surface>_stitched.jpg, applies the per-surface flip
flags from orient.json (sflip = flip the stitch to match the render; rflip = flip the render,
used for ceilings viewed from below), skips sliver walls, and stacks one row per surface.
"""

import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from lrauthor.fonts import font as _shared_font


def _font(s):
    try:
        return _shared_font(s)
    except Exception:
        return ImageFont.load_default()


FONT, HF = _font(26), _font(36)
ORDER = {"Wall": 0, "Floor": 1, "Ceiling": 2}


def _stitch_for(stitch_dir, name):
    for cand in (name, f"{name}_derived", "Ceiling0_derived"):
        p = stitch_dir / f"{cand}_stitched.jpg"
        if p.exists():
            return p
    return None


def compose(ortho_dir, stitch_dir, out_path, title):
    ortho_dir, stitch_dir = Path(ortho_dir), Path(stitch_dir)
    flips = {}
    oj = ortho_dir / "orient.json"
    if oj.exists():
        flips = json.loads(oj.read_text())
    files = sorted(
        ortho_dir.glob("*_ortho.png"),
        key=lambda r: (ORDER.get(re.match(r"[A-Za-z]+", r.stem).group(), 9), r.stem),
    )
    rows = []
    for r in files:
        name = r.stem.replace("_ortho", "")
        st = _stitch_for(stitch_dir, name)
        if st is None:
            continue
        a = Image.open(r).convert("RGB")
        b = Image.open(st).convert("RGB")
        if a.width / a.height < 0.35:  # skip sliver walls
            continue
        fl = flips.get(name, {})
        if isinstance(fl, dict):
            if fl.get("rflip"):
                a = a.transpose(Image.FLIP_LEFT_RIGHT)
            if fl.get("sflip"):
                b = b.transpose(Image.FLIP_LEFT_RIGHT)
        elif fl:
            b = b.transpose(Image.FLIP_LEFT_RIGHT)
        H = 420
        a = a.resize((int(a.width * H / a.height), H))
        b = b.resize((int(b.width * H / b.height), H))
        for im, lab, col in [(a, "RENDER", (150, 220, 255)), (b, "REAL (stitch)", (160, 255, 170))]:
            d = ImageDraw.Draw(im)
            d.rectangle((0, 0, im.width, 30), fill=(0, 0, 0))
            d.text((5, 3), lab, fill=col, font=FONT)
        row = Image.new("RGB", (a.width + b.width + 18, H + 8 + 44), (8, 10, 14))
        dh = ImageDraw.Draw(row)
        dh.rectangle((0, 0, row.width, 44), fill=(34, 40, 52))
        dh.text((10, 5), name, fill=(255, 220, 150), font=HF)
        row.paste(a, (6, 48))
        row.paste(b, (a.width + 12, 48))
        rows.append(row)
    if not rows:
        print(f"{title}: no surfaces")
        return None
    W = max(r.width for r in rows)
    TT = 50
    sh = Image.new("RGB", (W, TT + sum(r.height for r in rows) + 6 * (len(rows) + 1)), (8, 10, 14))
    ImageDraw.Draw(sh).text((12, 8), title, fill=(255, 255, 255), font=HF)
    y = TT
    for r in rows:
        sh.paste(r, (0, y))
        y += r.height + 6
    sh.save(out_path, quality=88)
    print(f"{title} -> {out_path}  {len(rows)} surfaces  {sh.size}")
    return out_path
