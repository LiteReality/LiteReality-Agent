"""
annotate_views.py — turn each rendered view into a VLM-friendly, *visual* label.

Reads render_manifest.json (written by render_room_cameras.py) and, per frame:
  1. draws labelled boxes (the room.py id on each object) on the upright render
     -> annotated/frame_NNNNN.png        ("set-of-mark": what to edit, on the image)
  2. writes an editable vector overlay   -> svg/frame_NNNNN.svg
     (each object is <g data-id=.. data-category=.. data-edits=..>; the id/edits
      travel with the box, so the file is both visual and machine-editable)
  3. stitches it next to the real capture photo (rotated upright)
     -> sidebyside/frame_NNNNN.jpg       (compare render vs photo, decide edits)

Only objects that are *actually visible* are marked: occluded ones are dropped,
tiny slivers below a coverage threshold are dropped, and the list is capped, so
the annotation stays readable (unlike the old text table).

Run (numpy + Pillow; NOT Blender):
    python3 authoring/views/room_render/annotate_views.py <out_dir> <scan_dir> [frames]
    frames: 'all' (default) | '0,11,30'
"""

import json
import os
import re
import sys

from PIL import Image, ImageDraw

from lrauthor.fonts import font as _shared_font

# Label ONLY the original RoomPlan/USDZ objects: the shell + the canonical RoomPlan
# object/opening categories (name = <Category><index>, e.g. Wall0, Table0, Door0).
# Everything an agent adds later (Socket_*, Trunk_*, Skirt_*, Radiator0, Shelf2,
# Whiteboard0, …) is NOT a RoomPlan category, so it's skipped — otherwise the
# image drowns in overlapping fixture labels.
MAIN_RE = re.compile(
    r"^(Wall|Floor|Ceiling|Door|Window|Opening|Table|Chair|Sofa|Bed|Storage|"
    r"Refrigerator|Stove|Oven|Washer|Dishwasher|Sink|Toilet|Bathtub|Stairs|"
    r"Fireplace|Television)\d+$"
)

CATEGORY_COLOR = {
    "table": (245, 155, 45),
    "desk": (245, 155, 45),
    "chair": (40, 145, 255),
    "sofa": (40, 145, 255),
    "door": (70, 200, 110),
    "window": (60, 175, 215),
    "wall": (165, 110, 215),
    "floor": (0, 185, 165),
    "ceiling": (220, 120, 200),
}


def _color(cat):
    return CATEGORY_COLOR.get(cat, (235, 235, 235))


def _font(size):
    return _shared_font(size)


def _marks(frame, W, H):
    """Manifest objects -> drawable labels. Visibility is already decided upstream
    in describe_view, so here we just place each object's id at its label point."""
    out = []
    for o in frame["visible"]:
        if not MAIN_RE.match(o["id"]):  # skip agent-added fixtures
            continue
        lx, ly = o["label_pct"]
        out.append(
            {
                "id": o["id"],
                "category": o["category"],
                "edits": o.get("edits", []),
                "dist_m": o.get("dist_m"),
                "label": [lx / 100 * W, ly / 100 * H],
            }
        )
    return out


def _label(draw, text, x, y, col, font, W, H, sw, center=False):
    """A label as colored text with a dark halo — legible but it does NOT cover
    the object with a filled chip, so the render stays clearly visible."""
    tb = draw.textbbox((0, 0), text, font=font, stroke_width=sw)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    if center:
        x, y = x - tw / 2, y - th / 2
    x = min(max(2, x), W - tw - 2)
    y = min(max(2, y), H - th - 2)
    draw.text(
        (x, y), text, font=font, fill=col + (255,), stroke_width=sw, stroke_fill=(0, 0, 0, 235)
    )


def draw_annotated(render_path, marks, out_path):
    img = Image.open(render_path).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    font = _font(max(26, W // 30))
    sw = max(2, W // 420)
    for m in marks:
        _label(
            draw,
            m["id"],
            m["label"][0],
            m["label"][1],
            _color(m["category"]),
            font,
            W,
            H,
            sw,
            center=True,
        )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path, quality=92)
    return img


def write_svg(render_rel, marks, W, H, out_path):
    el = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
        f'  <image href="{render_rel}" x="0" y="0" width="{W}" height="{H}"/>',
    ]
    for m in marks:
        r, g, b = _color(m["category"])
        col = f"rgb({r},{g},{b})"
        lx, ly = m["label"]
        edits = ",".join(m["edits"])
        el.append(
            f'  <g data-id="{m["id"]}" data-category="{m["category"]}" '
            f'data-dist="{m["dist_m"]}" data-edits="{edits}">'
        )
        el.append(
            f'    <text x="{lx:.0f}" y="{ly:.0f}" font-size="40" font-family="Arial" '
            f'font-weight="bold" text-anchor="middle" fill="{col}" '
            f'stroke="black" stroke-width="1" paint-order="stroke">{m["id"]}</text>'
        )
        el.append("  </g>")
    el.append("</svg>")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write("\n".join(el))


def side_by_side(annotated_img, photo_path, out_path):
    if not os.path.isfile(photo_path):
        return
    photo = Image.open(photo_path).convert("RGB").transpose(Image.Transpose.ROTATE_270)
    H = annotated_img.height
    photo = photo.resize((int(photo.width * H / photo.height), H))
    gap = 12
    canvas = Image.new("RGB", (annotated_img.width + gap + photo.width, H), (255, 255, 255))
    canvas.paste(annotated_img, (0, 0))
    canvas.paste(photo, (annotated_img.width + gap, 0))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path, quality=90)


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "output/room_render"
    scan_dir = sys.argv[2] if len(sys.argv) > 2 else None
    frames_spec = sys.argv[3] if len(sys.argv) > 3 else "all"

    man = json.load(open(os.path.join(out_dir, "render_manifest.json")))
    W, H = man["upright_resolution"]
    want = None if frames_spec == "all" else {f"frame_{int(x):05d}" for x in frames_spec.split(",")}

    n = 0
    for frame in man["frames"]:
        fid = frame["frame"]
        if want is not None and fid not in want:
            continue
        render_path = os.path.join(out_dir, f"{fid}.png")
        if not os.path.isfile(render_path):
            continue
        marks = _marks(frame, W, H)
        img = draw_annotated(render_path, marks, os.path.join(out_dir, "annotated", f"{fid}.png"))
        write_svg(f"../{fid}.png", marks, W, H, os.path.join(out_dir, "svg", f"{fid}.svg"))
        if scan_dir:
            side_by_side(
                img,
                os.path.join(scan_dir, f"{fid}.jpg"),
                os.path.join(out_dir, "sidebyside", f"{fid}.jpg"),
            )
        n += 1
    print(
        f"annotated {n} frame(s) -> {out_dir}/annotated, /svg"
        + (", /sidebyside" if scan_dir else "")
    )


if __name__ == "__main__":
    main()
