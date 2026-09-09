"""Torch device selection with Apple Silicon (MPS) support.

GroundingDINO detect + DINOv2 grouping run in the main env; on a MacBook (Apple Silicon) they should
use the **MPS** GPU rather than falling back to CPU. A few of their ops aren't implemented on MPS yet,
so we enable PyTorch's CPU fallback for just those ops — set at import (before torch loads) so it
always takes effect.
"""

from __future__ import annotations

import os

# Set BEFORE torch is imported anywhere: unsupported MPS ops fall back to CPU instead of erroring.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def pick_device() -> str:
    """Best available torch device: ``cuda`` (NVIDIA) > ``mps`` (Apple Silicon) > ``cpu``."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available() and mps.is_built():
        return "mps"
    return "cpu"
