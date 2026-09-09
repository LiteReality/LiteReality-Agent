"""Environment wiring for canonical TRELLIS.2 inference.

The ``TRELLIS.2/`` clone is kept pristine so it can be reproduced on another machine.
This module makes that clone usable without modifying it by:

  * putting the clone package on ``sys.path`` so ``import trellis2`` resolves,
  * pointing the HuggingFace cache at ``backends/weights/`` so model weights
    **auto-download there on first run** (nothing is vendored into git),
  * setting the GPU-friendly defaults TRELLIS.2 expects (sdpa attention, EXR I/O,
    expandable CUDA segments, a writable flex-gemm autotune cache).

Import this first, before importing ``trellis2``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from lrauthor import REPO_ROOT

LAUNCHER_DIR = Path(__file__).resolve().parent
# The clones and the downloaded weights are DATA, not part of the package: they live at the
# CHECKOUT root (which `.gitignore` already excludes), never inside src/. Deriving them from
# `__file__` would follow the code into the wheel and download gigabytes into site-packages.
THIRD_PARTY = REPO_ROOT
TRELLIS2_DIR = THIRD_PARTY / "TRELLIS.2"
WEIGHTS_DIR = Path(os.environ.get("LITEREALITY_WEIGHTS", THIRD_PARTY / "weights")).resolve()

# TRELLIS.2's pipeline.json references the GATED `facebook/dinov3-vitl16-pretrain-lvd1689m`
# image-cond model. We repoint it at a LOCAL (ungated) copy so loading needs no HF auth.
LOCAL_MODEL_DIR = WEIGHTS_DIR / "trellis2_4b_local"


def resolve_dinov3() -> Path | None:
    """Locate a local DINOv3 dir (config.json + weights): ``$LITEREALITY_DINOV3`` ->
    ``backends/weights/dinov3``. None if nothing local."""
    for cand in (os.environ.get("LITEREALITY_DINOV3"), WEIGHTS_DIR / "dinov3"):
        if cand and Path(cand).exists():
            return Path(cand).resolve()
    return None


def configure_hf_cache() -> Path:
    """Route every HuggingFace download into ``backends/weights/``."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(WEIGHTS_DIR))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(WEIGHTS_DIR / "hub"))
    os.environ.setdefault("TORCH_HOME", str(WEIGHTS_DIR / "torch"))
    return WEIGHTS_DIR


def configure_trellis_runtime() -> None:
    """Env defaults TRELLIS.2 needs, plus make the clone's ``trellis2`` importable."""
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # Dense attention: sdpa (PyTorch built-in) is always available.
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    # Sparse attention only accepts xformers / flash_attn / flash_attn_3 (no sdpa) — use
    # xformers (prebuilt wheel for torch 2.6/cu124; flash_attn would need a long build).
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")
    # flex_gemm autotune cache defaults to ~/.flex_gemm, often a broken symlink; redirect it.
    cache = WEIGHTS_DIR / ".flex_gemm" / "autotune_cache.json"
    os.environ.setdefault("FLEX_GEMM_AUTOTUNE_CACHE_PATH", str(cache))
    cache.parent.mkdir(parents=True, exist_ok=True)
    configure_hf_cache()
    # Resolve `import trellis2` to the pristine clone. NOTE: o_voxel / natten / cumesh
    # / flex_gemm / nvdiffrast are COMPILED extensions — they must come from the
    # installed env (site-packages), so we deliberately do NOT add the clone's
    # source `o-voxel/` dir to sys.path (its `_C` isn't built there).
    sp = str(TRELLIS2_DIR)
    if TRELLIS2_DIR.exists() and sp not in sys.path:
        sys.path.insert(0, sp)
