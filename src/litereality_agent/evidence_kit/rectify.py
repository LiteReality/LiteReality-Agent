"""Lift a planar region out of a photograph as a straight-on texture.

    import rectify
    rectify.rectify_region(S, 12, [(880,300),(1310,290),(1320,560),(875,570)], "textures/monitor.png",
                           size_px=(1024, 640))

`quad_px` is the region's four corners in the STORED jpg, in order top-left, top-right,
bottom-right, bottom-left AS THE OBJECT IS ORIENTED (the phone was held portrait, so 'top' of a
monitor is usually the jpg's left or right edge — pick the corners by what they are on the object,
not on the page). Use the result for screens, notice boards, pictures, labels and the view through
a window; keep the aspect of `size_px` close to the object's real aspect so it is not stretched."""
from __future__ import annotations

import os

import numpy as np


def _homography(src, dst):
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _, _, vt = np.linalg.svd(np.array(A, dtype=np.float64))
    H = vt[-1].reshape(3, 3)
    return H / H[2, 2]


def rectify_region(S, i: int, quad_px, out_path: str, size_px=(1024, 1024)) -> str:
    """Warp the quad of frame i to a size_px image and save it. Returns out_path."""
    from PIL import Image
    img = Image.open(S.rgb_path(i)).convert("RGB")
    W, Hh = size_px
    dst = [(0, 0), (W - 1, 0), (W - 1, Hh - 1), (0, Hh - 1)]
    # PIL's PERSPECTIVE wants the inverse map (output -> input) as 8 coefficients
    Hm = _homography(dst, [tuple(map(float, p)) for p in quad_px])
    coeffs = (Hm / Hm[2, 2]).flatten()[:8]
    out = img.transform((W, Hh), Image.PERSPECTIVE, tuple(coeffs), Image.BICUBIC)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    out.save(out_path)
    return out_path
