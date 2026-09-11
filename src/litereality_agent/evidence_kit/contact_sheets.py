"""Every photograph of the capture on a handful of sheets, upright, numbered by frame index.

    import contact_sheets; contact_sheets.build(S, "evidence/contact_sheets")   -> [sheet paths]

Look at ALL of them before building anything: the scan's boxes say where the big furniture is,
the photographs say what the room actually contains (the coat on the door, the heater, the mug)."""
from __future__ import annotations

import os

from PIL import Image, ImageDraw


def build(S, out_dir: str, per_sheet: int = 12, cols: int = 4, thumb_w: int = 360,
          rotate_cw: bool = True) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    idx = S.indices()
    paths = []
    for s in range(0, len(idx), per_sheet):
        chunk = idx[s:s + per_sheet]
        thumbs = []
        for i in chunk:
            im = Image.open(S.rgb_path(i)).convert("RGB")
            if rotate_cw:
                im = im.rotate(-90, expand=True)
            im.thumbnail((thumb_w, thumb_w * 2))
            d = ImageDraw.Draw(im)
            d.rectangle([0, 0, 96, 22], fill=(0, 0, 0))
            d.text((4, 4), f"frame {i:05d}", fill=(255, 255, 255))
            thumbs.append(im)
        tw = max(t.width for t in thumbs)
        th = max(t.height for t in thumbs)
        rows = (len(thumbs) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * (tw + 6), rows * (th + 6)), (24, 24, 24))
        for n, t in enumerate(thumbs):
            sheet.paste(t, ((n % cols) * (tw + 6), (n // cols) * (th + 6)))
        p = os.path.join(out_dir, f"sheet_{s // per_sheet:02d}_frames_{chunk[0]:05d}-{chunk[-1]:05d}.jpg")
        sheet.save(p, quality=85)
        paths.append(p)
    return paths
