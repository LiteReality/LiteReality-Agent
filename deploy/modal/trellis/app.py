"""Modal deployment wrapper for the canonical TRELLIS.2 implementation.

Deploys into whichever workspace the active ``MODAL_PROFILE`` names; the weights are ungated, so
no Hugging Face account or secret is required. See README.md.
"""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path

import modal

APP_NAME = "litereality-trellis"  # included into `litereality` by deploy/modal/app.py
FUNCTION_NAME = "trellis"
MODEL_VOLUME = "litereality-trellis-models"
REMOTE_REPO_ROOT = Path("/root")
WEIGHTS_ROOT = Path("/models")

# TRELLIS.2's image encoder is the gated ``facebook/dinov3-vitl16-pretrain-lvd1689m``. The DINOv3
# licence grants the right to redistribute the trained weights provided the licence travels with
# them, so we bake an ungated mirror into the image instead of requiring every deployer to hold an
# HF account. Pinned by commit and verified by digest: a mirror owner can force-push or delete the
# repo, and four independent uploads of these weights agree on this digest.
DINOV3_DIR = Path("/opt/dinov3")
DINOV3_REPO = "camenduru/dinov3-vitl16-pretrain-lvd1689m"
DINOV3_REVISION = "3c276edd87d6f6e569ff0c4400e086807d0f3881"
DINOV3_SHA256 = "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179"

deployment_root = Path(__file__).resolve().parent
image = (
    modal.Image.from_dockerfile(
        deployment_root / "Dockerfile",
        context_dir=deployment_root,
    )
    .run_commands("python3.10 -m pip install --no-cache-dir 'setuptools>=64' wheel")
    .run_commands(
        "bash -lc 'cd /root/TRELLIS.2 && "
        ". ./setup.sh --cumesh --o-voxel --flexgemm --nvdiffrast --nvdiffrec'",
        # Upstream checks nvidia-smi before installing its native extensions. Scope the
        # billable builder GPU to this cached layer instead of the complete Docker build.
        gpu="H100",
        env={"TORCH_CUDA_ARCH_LIST": "9.0"},
    )
    # TRELLIS.2 reaches into DINOv3 internals that changed in Transformers 5.x.
    # Keep this after the cached native-extension layer so the compatibility pin
    # remains a CPU-only image operation and cannot trigger another GPU build.
    .pip_install("transformers==4.57.5")
    # CPU-only, and deliberately after the cached GPU layer so pulling 1.2GB of weights can never
    # trigger another H100 build. Fails the build on a digest mismatch rather than at inference.
    .run_commands(
        f"python3.10 -c \"from huggingface_hub import snapshot_download; \
snapshot_download('{DINOV3_REPO}', revision='{DINOV3_REVISION}', local_dir='{DINOV3_DIR}')\"",
        f"echo '{DINOV3_SHA256}  {DINOV3_DIR}/model.safetensors' | sha256sum -c -",
    )
    .env(
        {
            "LR_REPO_ROOT": str(REMOTE_REPO_ROOT),
            "LITEREALITY_WEIGHTS": str(WEIGHTS_ROOT),
            "HF_XET_HIGH_PERFORMANCE": "1",
            # resolve_dinov3() reads this first; with it set, pipeline.json is repointed at the
            # local copy and loading never reaches the gated repo, so no HF token is needed.
            "LITEREALITY_DINOV3": str(DINOV3_DIR),
        }
    )
    .add_local_python_source("litereality_agent")
)
weights = modal.Volume.from_name(MODEL_VOLUME, create_if_missing=True)
app = modal.App(APP_NAME)
_pipeline = None


def _load_pipeline():
    global _pipeline
    if _pipeline is None:
        from litereality_agent.models.trellis.inference import (
            DEFAULT_MODEL,
            load_pipeline,
            prepare_local_model,
        )

        _pipeline = load_pipeline(prepare_local_model(DEFAULT_MODEL))
        weights.commit()
    return _pipeline


@app.function(
    name=FUNCTION_NAME,
    image=image,
    gpu="H100",
    volumes={str(WEIGHTS_ROOT): weights},
    timeout=20 * 60,
    retries=0,
    scaledown_window=10,
    # Serialize cold starts so workers cannot race while populating the shared model Volume.
    # Raise this deliberately only after the cache has been pre-populated and costs reviewed.
    max_containers=1,
)
def generate(request: dict) -> dict:
    """Convert the transport payload into one GLB using model-owned inference code."""
    from litereality_agent.models.trellis.inference import generate as generate_glb

    with tempfile.TemporaryDirectory() as temporary_dir:
        temp = Path(temporary_dir)
        image_path = temp / "input.png"
        output_path = temp / "output.glb"
        image_path.write_bytes(base64.b64decode(request["image_b64"], validate=True))
        generate_glb(
            _load_pipeline(),
            image_path,
            output_path,
            seed=int(request.get("seed", 42)),
            decimation_target=_decimation_target(request.get("simplify", 0.95)),
            texture_size=int(request.get("texture_size", 1024)),
            pipeline_type=request.get("pipeline_type"),
        )
        return {"glb_b64": base64.b64encode(output_path.read_bytes()).decode("ascii")}


def _decimation_target(simplify: float) -> int:
    """Translate the legacy ratio into the canonical launcher's face target."""
    ratio = min(max(float(simplify), 0.0), 1.0)
    return max(10_000, round(1_000_000 * (1.0 - ratio)))
