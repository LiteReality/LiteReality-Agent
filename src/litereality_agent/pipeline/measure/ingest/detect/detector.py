"""Single detection entry for the init stage — call THIS, not dino_detect directly.

Routes to the isolated DINO **tool** (`models.dino.DinoSubprocessService`, registered
via `set_service`) when configured, else falls back to in-process `dino_detect` (back-compat,
needs torch in this env). Either way it returns `dino_detect.Detection`-shaped objects
(`.box / .score / .label`) so callers (`bbox_polish`, the opening matcher) are unchanged.

Importing this module does NOT import torch — `dino_detect`'s heavy imports are lazy, and the
service path runs torch in its own interpreter.
"""

from __future__ import annotations

from typing import Any

from litereality_agent.models.grounding_dino import inference as dino_detect

_SERVICE: Any = None  # a DetectionService (DinoSubprocessService); None → in-process fallback


def set_service(service: Any) -> None:
    """Install the detect tool (or clear with None)."""
    global _SERVICE
    _SERVICE = service


def using_service() -> bool:
    return _SERVICE is not None


def available() -> bool:
    return _SERVICE is not None or dino_detect.available()


def default_model_id() -> str:
    return dino_detect.default_model_id()


def iou(a: list[float], b: list[float]) -> float:
    return dino_detect.iou(a, b)


def embed_available() -> bool:
    """True if DINOv2 embeddings can be produced (via the tool, or in-process torch)."""
    if _SERVICE is not None and hasattr(_SERVICE, "embed"):
        return True
    from litereality_agent.models.dinov2 import inference as dino_embed

    return dino_embed.available()


def embed(images: list, *, upright: bool = True) -> list[list[float]]:
    """L2-normalised DINOv2 CLS embedding per image, via the isolated tool if installed, else
    in-process. Order preserved. Raises if no embedding backend is available (callers should
    gate on ``embed_available()`` and fall back)."""
    if _SERVICE is not None and hasattr(_SERVICE, "embed"):
        return _SERVICE.embed(images, upright=upright)

    # in-process fallback: dino_embed wants paths; write non-path images to temp PNGs
    import os
    import tempfile

    from litereality_agent.models.dinov2 import inference as dino_embed

    paths: list[str] = []
    temps: list[str] = []
    for img in images:
        if isinstance(img, (str, os.PathLike)):
            paths.append(str(img))
        else:
            from PIL import Image

            pil = img if hasattr(img, "save") else Image.fromarray(img)
            fd, p = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            pil.convert("RGB").save(p)
            paths.append(p)
            temps.append(p)
    try:
        return dino_embed.embed_paths(paths, upright=upright)
    finally:
        for p in temps:
            try:
                os.unlink(p)
            except OSError:
                pass


def detect(
    image,
    prompt: str,
    box_threshold: float = 0.30,
    text_threshold: float = 0.20,
    model_id: str | None = None,
    upright: bool = True,
) -> list[dino_detect.Detection]:
    """Detect via the tool if installed, else in-process. Returns dino_detect.Detection list."""
    if _SERVICE is not None:
        dets = _SERVICE.detect(
            image,
            prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            upright=upright,
        )
        return [
            dino_detect.Detection(box=list(d.box_xyxy), score=float(d.score), label=d.label)
            for d in dets
        ]
    return dino_detect.detect(
        image,
        prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        model_id=model_id,
        upright=upright,
    )
