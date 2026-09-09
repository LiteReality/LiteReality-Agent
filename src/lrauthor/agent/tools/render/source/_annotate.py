"""_annotate.py — drawing primitives for the three frame modes.

All operate IN PLACE on a PIL image and are applied to BOTH the render and the real photo (they
share the camera pose, so projected/label positions match):
  - draw_chips(img, marks)              scene mode: a highlighted chip per object at its centre
  - draw_wall_edges(img, edges)         wall  mode: orange wall outlines
  - draw_object_box(img, box, name)     object mode: a bbox on one object + its name to the side
"""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from lrauthor.fonts import font as _shared_font

# category → colour (mirrors authoring/views/room_render/annotate_views)
CATEGORY_COLOR = {
    "table": (250, 170, 70),
    "desk": (250, 170, 70),
    "chair": (90, 170, 255),
    "sofa": (90, 170, 255),
    "door": (90, 215, 130),
    "window": (90, 195, 235),
    "wall": (195, 150, 240),
    "floor": (60, 210, 195),
    "ceiling": (235, 150, 215),
}
WALL_ORANGE = (255, 150, 40)


def _font(size: int):
    try:
        return _shared_font(size)
    except Exception:
        return ImageFont.load_default()


# These annotations are drawn on the FULL-RESOLUTION render (1440x1920 for a phone capture), but
# what anyone actually looks at is the side-by-side composite, where `compose._hstack` resizes each
# panel to PANEL_H. That is a ~0.29x downscale, so a size chosen against the render is barely a
# third of itself by the time it is read. Size against the final view instead: pick the px we want
# IN THE COMPOSITE and scale it back up by the same ratio the resize will scale it down.
PANEL_H = 560  # keep in step with compose._hstack(height=...)
CHIP_PX_IN_COMPOSITE = 20  # object-name chips, as read in the composite
BOX_PX_IN_COMPOSITE = 24  # the single-object callout label, a touch larger


def _at_render_scale(img_h: int, wanted_in_composite: int) -> int:
    """`wanted_in_composite` px after the panel is resized to PANEL_H — expressed at img_h."""
    return max(wanted_in_composite, round(wanted_in_composite * img_h / PANEL_H))


def chip_px(img_h: int) -> int:
    return _at_render_scale(img_h, CHIP_PX_IN_COMPOSITE)


def box_label_px(img_h: int) -> int:
    return _at_render_scale(img_h, BOX_PX_IN_COMPOSITE)


def _overlap(a, b, pad: int = 4) -> bool:
    return not (a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1])


def draw_chips(img: Image.Image, marks: list[dict]) -> Image.Image:
    """A highlighted chip (dark rounded box + category-coloured text) per mark, at its `label`
    pixel position. Nearer objects (smaller dist_m) placed first; colliding chips nudged down."""
    W, H = img.size
    d = ImageDraw.Draw(img)
    font = _font(chip_px(H))
    padx, pady = max(9, font.size // 3), max(6, font.size // 5)
    placed: list[tuple] = []
    for m in sorted(marks, key=lambda m: (m.get("dist_m") if m.get("dist_m") is not None else 1e9)):
        text = m["id"]
        col = CATEGORY_COLOR.get(m.get("category"), (235, 235, 235))
        tb = d.textbbox((0, 0), text, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        cx, cy = m["label"]

        def box_at(x, y, tw=tw, th=th):
            return (x - tw / 2 - padx, y - th / 2 - pady, x + tw / 2 + padx, y + th / 2 + pady)

        cx = min(max(tw / 2 + padx + 2, cx), W - tw / 2 - padx - 2)
        cy = min(max(th / 2 + pady + 2, cy), H - th / 2 - pady - 2)
        guard = 0
        while any(_overlap(box_at(cx, cy), b) for b in placed) and guard < 40:
            cy += th + 2 * pady + 4
            if cy > H - th / 2 - pady - 2:
                cy = th / 2 + pady + 6
                cx = min(cx + tw + 24, W - tw / 2 - padx - 2)
            guard += 1
        box = box_at(cx, cy)
        d.rounded_rectangle(box, radius=max(9, font.size // 3), fill=(0, 0, 0))
        d.text((cx - tw / 2, cy - th / 2 - tb[1]), text, fill=col, font=font)
        placed.append(box)
    return img


def draw_wall_edges(img: Image.Image, edges: list, focus: bool = False) -> Image.Image:
    """Stroke projected wall outline segments (list of [(x,y),(x,y)])."""
    W = img.size[0]
    d = ImageDraw.Draw(img)
    for seg in edges:
        d.line(seg, fill=WALL_ORANGE, width=max(4, W // (260 if focus else 340)))
    return img


def draw_object_box(img: Image.Image, box, name: str, color=(255, 210, 90)) -> Image.Image:
    """A rectangle around ONE object (box = x0,y0,x1,y1 px) + its name on a chip to the SIDE."""
    W, H = img.size
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = box
    d.rectangle((x0, y0, x1, y1), outline=color, width=max(4, W // 300))
    font = _font(box_label_px(H))
    tb = d.textbbox((0, 0), name, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    # place the label just to the RIGHT of the box (or left if it would run off-frame)
    lx, ly = x1 + 10, (y0 + y1) / 2 - th / 2
    if lx + tw + 18 > W:
        lx = x0 - tw - 28
    lx = max(4, min(lx, W - tw - 22))
    ly = max(4, min(ly, H - th - 16))
    d.rounded_rectangle((lx, ly, lx + tw + 18, ly + th + 14), radius=8, fill=(0, 0, 0))
    d.text((lx + 9, ly + 5 - tb[1]), name, fill=color, font=font)
    return img
