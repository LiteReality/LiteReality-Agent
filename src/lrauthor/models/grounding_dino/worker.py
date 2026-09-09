"""GroundingDINO worker shared by local subprocess and hosted runtimes.

A persistent JSON-lines server over stdin/stdout: the model loads once and serves many
detect requests warm (bbox_polish calls DINO on many crops per scan). Spawned by
`models/dino.py:DinoSubprocessService`. Run as:

    $LR_DINO_PYTHON -u -m lrauthor.models.grounding_dino.worker

Protocol — one JSON object per line in, one per line out:
    {"op":"ping"}                              -> {"ok":true,"model":...,"cuda":bool}
    {"op":"detect","image_path":..,"prompt":..,"box_threshold":..,"text_threshold":..,
     "upright":true,"model_id":null,"id":N}    -> {"ok":true,"detections":[{box,score,label}],"id":N}
    {"op":"embed","image_paths":[..],"upright":true,"model_id":null,"id":N}
                                               -> {"ok":true,"embeddings":[[..],..],"id":N}
On error: {"ok":false,"error":"..."}.
"""

from __future__ import annotations

import json
import sys


def _handle(req: dict) -> dict:
    op = req.get("op", "detect")
    if op == "ping":
        from . import inference

        try:
            import torch

            cuda = bool(torch.cuda.is_available())
        except Exception:
            cuda = False
        return {"ok": True, "model": inference.default_model_id(), "cuda": cuda}
    if op == "detect":
        from PIL import Image

        from . import inference

        img = Image.open(req["image_path"]).convert("RGB")
        dets = inference.detect(
            img,
            req["prompt"],
            box_threshold=req.get("box_threshold", 0.30),
            text_threshold=req.get("text_threshold", 0.20),
            model_id=req.get("model_id"),
            upright=req.get("upright", True),
        )
        return {
            "ok": True,
            "detections": [
                {"box": list(d.box), "score": float(d.score), "label": d.label} for d in dets
            ],
        }
    if op == "embed":
        from lrauthor.models.dinov2 import inference

        embs = inference.embed_paths(
            list(req["image_paths"]),
            model_id=req.get("model_id"),
            upright=req.get("upright", True),
        )
        return {"ok": True, "embeddings": embs}
    return {"ok": False, "error": f"unknown op {op!r}"}


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req: dict | None = None
        try:
            req = json.loads(line)
            resp = _handle(req)
        except Exception as e:  # noqa: BLE001 — report, keep serving
            resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        if isinstance(req, dict) and "id" in req:
            resp["id"] = req["id"]
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()
    return 0


def hosted_handler(request: dict) -> dict:
    """Run one JSON-safe hosted request through the canonical inference functions."""
    import base64
    import io
    import tempfile
    import traceback
    from pathlib import Path

    from PIL import Image

    try:
        operation = request.get("op", "detect")
        if operation == "detect":
            from . import inference

            image = Image.open(io.BytesIO(base64.b64decode(request["image_b64"]))).convert("RGB")
            detections = inference.detect(
                image,
                request["prompt"],
                box_threshold=float(request.get("box_threshold", 0.30)),
                text_threshold=float(request.get("text_threshold", 0.20)),
                model_id=request.get("model_id"),
                upright=bool(request.get("upright", True)),
            )
            return {
                "detections": [
                    {"box": item.box, "score": item.score, "label": item.label}
                    for item in detections
                ]
            }
        if operation == "embed":
            from lrauthor.models.dinov2 import inference

            with tempfile.TemporaryDirectory() as directory:
                paths: list[str] = []
                for index, encoded in enumerate(request.get("images_b64") or []):
                    path = Path(directory) / f"{index}.png"
                    image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
                    image.save(path)
                    paths.append(str(path))
                embeddings = inference.embed_paths(
                    paths,
                    model_id=request.get("model_id"),
                    upright=bool(request.get("upright", True)),
                )
            return {"embeddings": embeddings}
        return {"error": f"unknown op {operation!r}"}
    except Exception as exc:  # keep the remote worker alive and return useful diagnostics
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc()[-2000:],
        }


if __name__ == "__main__":
    raise SystemExit(main())
