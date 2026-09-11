"""GroundingDINO and DINOv2 through one scale-to-zero Modal function."""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Any

from litereality_agent.models.grounding_dino.service import Detection
from litereality_agent.runtimes.modal import ModalClient


class ModalDinoService:
    """Detection and embedding adapter for the deployed Modal DINO function."""

    name = "dino-modal"

    def __init__(
        self,
        *,
        app_name: str,
        function_name: str = "dino",
        environment_name: str = "main",
        profile: str | None = None,
        credentials: tuple[str, str] | None = None,
        model_id: str | None = None,
        embed_model_id: str | None = None,
        client: Any = None,
    ) -> None:
        self.model_id = model_id
        self.embed_model_id = embed_model_id
        self.client = client or ModalClient(
            app_name,
            function_name,
            environment_name=environment_name,
            profile=profile,
            credentials=credentials,
        )

    def _run(self, payload: dict) -> dict:
        output = self.client.map([payload])[0]
        if isinstance(output, BaseException):
            raise RuntimeError(f"Modal DINO failed: {output}") from output
        if output.get("error"):
            raise RuntimeError(str(output["error"]))
        return output

    def detect(
        self,
        image: Any,
        prompt: str,
        *,
        box_threshold: float = 0.30,
        text_threshold: float = 0.20,
        upright: bool = True,
    ) -> list[Detection]:
        output = self._run(
            {
                "op": "detect",
                "image_b64": _image_b64(image),
                "prompt": prompt,
                "box_threshold": box_threshold,
                "text_threshold": text_threshold,
                "upright": upright,
                "model_id": self.model_id,
            }
        )
        return [
            Detection(
                label=str(item["label"]),
                box_xyxy=tuple(float(value) for value in item["box"]),
                score=float(item["score"]),
            )
            for item in output.get("detections", [])
        ]

    def embed(self, images: list[Any], *, upright: bool = True) -> list[list[float]]:
        output = self._run(
            {
                "op": "embed",
                "images_b64": [_image_b64(image) for image in images],
                "upright": upright,
                "model_id": self.embed_model_id,
            }
        )
        return [[float(value) for value in row] for row in output.get("embeddings", [])]

    def close(self) -> None:
        """Match the local worker lifecycle API; Modal holds no local process."""


def _image_b64(image: Any) -> str:
    """Encode supported image inputs for the JSON-safe Modal request contract."""
    if isinstance(image, (str, os.PathLike)):
        raw = Path(image).read_bytes()
    elif isinstance(image, bytes):
        raw = image
    else:
        from PIL import Image

        pil = image if hasattr(image, "save") else Image.fromarray(image)
        buffer = io.BytesIO()
        pil.convert("RGB").save(buffer, format="PNG")
        raw = buffer.getvalue()
    return base64.b64encode(raw).decode()
