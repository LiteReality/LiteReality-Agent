"""Scale-to-zero Modal wrapper for GroundingDINO and DINOv2."""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "litereality-dino"  # included into `litereality` by deploy/modal/app.py
FUNCTION_NAME = "dino"
MODEL_VOLUME = "litereality-dino-models"
WEIGHTS_ROOT = Path("/models")

deployment_root = Path(__file__).resolve().parent
image = (
    modal.Image.from_dockerfile(
        deployment_root / "Dockerfile",
        context_dir=deployment_root,
    )
    .env({"HF_HOME": str(WEIGHTS_ROOT / "huggingface")})
    .add_local_python_source("litereality_agent")
)
weights = modal.Volume.from_name(MODEL_VOLUME, create_if_missing=True)
app = modal.App(APP_NAME)


@app.function(
    name=FUNCTION_NAME,
    image=image,
    gpu="L4",
    volumes={str(WEIGHTS_ROOT): weights},
    timeout=10 * 60,
    scaledown_window=60,
    max_containers=4,
)
def infer(request: dict) -> dict:
    """Serve either a detection or embedding request using shared model code."""
    from litereality_agent.models.grounding_dino.worker import hosted_handler

    response = hosted_handler(request)
    weights.commit()
    return response
