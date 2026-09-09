#!/usr/bin/env python3
"""recolor.py — shift a real base texture to a target colour while keeping its pattern.

(en) Shift a real texture's overall colour to a target RGB while preserving its pattern.
Ported from studio LR_mat_painting: in LAB space, move the mean to the target RGB while keeping each pixel's
deviation from the mean (= keeps the pattern/light variation). `pattern_strength` (1.0=full, 0=solid).

Ported from the studio's `LR_mat_painting/Material_refinements.apply_color_to_texture`: in LAB
space, the mean of the whole texture is translated to the target RGB. Each pixel keeps its
offset from that mean, which is what preserves the pattern and the light/dark variation —
rather than producing a flat, lifeless block of colour.

    # as a library
    from recolor import recolor_to_rgb
    out_bgr = recolor_to_rgb(src_bgr, (204, 191, 172))           # target RGB

    # from the command line
    python recolor.py <src.jpg> <r,g,b> <out.jpg> [pattern_strength]

pattern_strength (default 1.0): 1.0 keeps the texture's variation intact; below 1 flattens it
toward a solid colour, which suits very smooth paint or enamel; 0 collapses to a flat colour.
"""

from __future__ import annotations

import sys

import cv2
import numpy as np


def recolor_to_rgb(src_bgr: np.ndarray, rgb, pattern_strength: float = 1.0) -> np.ndarray:
    """src_bgr: HxWx3 BGR uint8. rgb: the target colour (R,G,B) 0-255. Returns BGR uint8."""
    src = src_bgr.astype(np.uint8)
    lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    target = cv2.cvtColor(np.uint8([[list(rgb)]]), cv2.COLOR_RGB2LAB)[0][0].astype(np.float32)
    out = np.empty_like(lab)
    for i in range(3):  # L, a, b: translate each channel's mean onto the target
        ch = lab[:, :, i]
        mean = ch.mean()
        # Keep each pixel's offset from the mean (that IS the pattern), scale it by strength,
        # then re-centre everything on the target mean.
        out[:, :, i] = target[i] + pattern_strength * (ch - mean)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)


def recolor_file(src_path: str, rgb, out_path: str, pattern_strength: float = 1.0):
    src = cv2.imread(src_path, cv2.IMREAD_COLOR)
    if src is None:
        raise SystemExit(f"could not read the base texture: {src_path}")
    cv2.imwrite(out_path, recolor_to_rgb(src, rgb, pattern_strength))
    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        raise SystemExit(1)
    src, rgb_s, out = sys.argv[1], sys.argv[2], sys.argv[3]
    strength = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    rgb = tuple(int(x) for x in rgb_s.split(","))
    recolor_file(src, rgb, out, strength)
    print(f"recolored {src} -> {out}  (rgb={rgb}, strength={strength})")
