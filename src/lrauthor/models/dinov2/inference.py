"""Canonical DINOv2 image embeddings for chair grouping.

The chair clusterer needs a similarity that says "same chair *design*", not "photographed
in the same pose". The old hand-crafted features (32x32 grayscale template + Canny edge
marginals + colour hist) measured pose alignment, so identical chairs shot from different
angles scored low and the thresholds had to be hand-tuned per scene. A self-supervised ViT
(DINOv2) CLS embedding captures shape+material identity and is robust to viewpoint/lighting,
so a single cosine threshold separates types cleanly.

Runs in the ISOLATED detect env (torch + transformers), same interpreter as GroundingDINO —
``dino_worker`` exposes it as ``op:"embed"`` so the light loop env never imports torch. The
model loads once and is cached process-wide. ``$LR_DINO_EMBED_MODEL`` overrides the checkpoint
(default ``facebook/dinov2-small``; ``-base`` for a stronger, slower descriptor).
"""

from __future__ import annotations

import os

_MODEL = None  # (processor, model, device, model_id) — lazily loaded, process-wide cache


def default_model_id() -> str:
    return os.environ.get("LR_DINO_EMBED_MODEL", "facebook/dinov2-small")


def available() -> bool:
    """True if the embedder deps are importable. DINOv2's AutoImageProcessor requires
    **torchvision** — check it explicitly so this never reports ready and then falls back to
    the CV features at run time (a silent quality regression)."""
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401  — required by DINOv2's image processor
        import transformers  # noqa: F401

        return True
    except Exception:
        return False


def _load(model_id: str | None = None):
    global _MODEL
    model_id = model_id or default_model_id()
    if _MODEL is not None and _MODEL[3] == model_id:
        return _MODEL
    from transformers import AutoImageProcessor, AutoModel

    from lrauthor.models.device import pick_device

    device = pick_device()  # cuda > mps (Apple Silicon) > cpu
    proc = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()
    _MODEL = (proc, model, device, model_id)
    return _MODEL


def embed_paths(
    image_paths: list[str],
    *,
    model_id: str | None = None,
    upright: bool = True,
    batch_size: int = 16,
) -> list[list[float]]:
    """L2-normalised CLS embedding for each image path (one row per path, order preserved).

    ``upright`` (default True): the rgbd crops are stored LANDSCAPE (a portrait phone capture
    laid on its side), so we rotate 90° CW to upright before encoding — the same convention
    ``dino_detect`` uses. Unreadable images yield a zero vector so indexing stays aligned.
    """
    import numpy as np
    import torch
    from PIL import Image as _Image

    proc, model, device, _ = _load(model_id)

    imgs: list = []
    ok: list[bool] = []
    for p in image_paths:
        try:
            im = _Image.open(p).convert("RGB")
            if upright:
                im = im.transpose(_Image.Transpose.ROTATE_270)
            imgs.append(im)
            ok.append(True)
        except Exception:
            imgs.append(None)
            ok.append(False)

    dim = int(getattr(model.config, "hidden_size", 384))
    out: list[list[float] | None] = [None] * len(image_paths)
    valid_idx = [i for i, good in enumerate(ok) if good]
    for start in range(0, len(valid_idx), batch_size):
        chunk = valid_idx[start : start + batch_size]
        batch = [imgs[i] for i in chunk]
        inp = proc(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            res = model(**inp)
        cls = res.last_hidden_state[:, 0].float().cpu().numpy()  # CLS token per image
        cls = cls / (np.linalg.norm(cls, axis=1, keepdims=True) + 1e-8)
        for row, i in zip(cls, chunk):
            out[i] = [float(v) for v in row]

    zero = [0.0] * dim
    return [row if row is not None else zero for row in out]
