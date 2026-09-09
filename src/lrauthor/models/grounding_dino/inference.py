"""Canonical GroundingDINO inference (HF transformers backend).

One open-vocabulary detector used by both bbox-refinement paths:
  - :mod:`bbox_polish` — tighten projected object boxes to the real 2D detection
  - the openings matcher — tighten projected door/window boxes

We use the **HF transformers** GroundingDINO (``IDEA-Research/grounding-dino-tiny``
by default) rather than the upstream clone: transformers + torch are already in the
env, the model auto-downloads, and it runs on GPU — no building the clone / fetching
the 660 MB SwinT weights. Same algorithm as the v1 ``bbox_polish.py`` / v2 opening
matcher, just the lighter dependency.

The model is loaded once and cached. ``$LR_DINO_MODEL`` overrides the checkpoint
(e.g. ``IDEA-Research/grounding-dino-base`` for higher recall).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

_MODEL = None  # (processor, model, device, model_id) — lazily loaded, process-wide cache


@dataclass
class Detection:
    box: list[float]  # [x1, y1, x2, y2] in pixels
    score: float
    label: str


def default_model_id() -> str:
    return os.environ.get("LR_DINO_MODEL", "IDEA-Research/grounding-dino-tiny")


def available() -> bool:
    """True if the detector deps are importable (torch + transformers)."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except Exception:
        return False


def _load(model_id: str | None = None):
    global _MODEL
    model_id = model_id or default_model_id()
    if _MODEL is not None and _MODEL[3] == model_id:
        return _MODEL
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    from lrauthor.models.device import pick_device

    device = pick_device()  # cuda > mps (Apple Silicon) > cpu
    proc = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()
    _MODEL = (proc, model, device, model_id)
    return _MODEL


def detect(
    image,
    prompt: str,
    box_threshold: float = 0.30,
    text_threshold: float = 0.20,
    model_id: str | None = None,
    upright: bool = True,
) -> list[Detection]:
    """Open-vocab detect ``prompt`` in a PIL ``image``.

    ``prompt`` is GroundingDINO caption form: lowercase phrases separated by '. '
    (e.g. ``"chair ."`` or ``"storage . cabinet . cupboard ."``). Returns pixel-space
    [x1,y1,x2,y2] boxes (in the ORIGINAL image's coordinates) with scores + label.

    ``upright`` (default True): the rgbd frames are stored LANDSCAPE (a portrait phone
    capture laid on its side), so objects sit sideways — bad for the detector. We rotate
    the frame 90° CW to upright for inference (as v1 bbox_polish did), then map the boxes
    BACK to the original landscape frame so the IoU-match vs the projected box still holds.
    """
    import torch
    from PIL import Image as _Image

    proc, model, device, _ = _load(model_id)
    W0, H0 = image.size
    img = image.transpose(_Image.Transpose.ROTATE_270) if upright else image  # 90° clockwise
    inp = proc(images=img, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inp)
    # transformers >=4.51: pass `threshold`; string names come back under `text_labels`.
    res = proc.post_process_grounded_object_detection(
        out,
        inp.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[img.size[::-1]],
    )[0]
    labels = res.get("text_labels", res.get("labels"))
    dets = []
    for box, score, lab in zip(res["boxes"].tolist(), res["scores"].tolist(), labels):
        b = [float(v) for v in box]
        if upright:
            # undo the 90° CW rotation: upright (xu,yu) -> landscape (x,y)=(yu, H0-1-xu)
            xu1, yu1, xu2, yu2 = b
            lx1, ly1, lx2, ly2 = yu1, H0 - 1 - xu2, yu2, H0 - 1 - xu1
            b = [min(lx1, lx2), min(ly1, ly2), max(lx1, lx2), max(ly1, ly2)]
        dets.append(Detection(box=b, score=float(score), label=str(lab)))
    return dets


def iou(a: list[float], b: list[float]) -> float:
    """IoU of two [x1,y1,x2,y2] boxes."""
    xi1, yi1 = max(a[0], b[0]), max(a[1], b[1])
    xi2, yi2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
