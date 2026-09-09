"""GroundingDINO service for an isolated local subprocess runtime.
Mirrors the TRELLIS isolation pattern: the heavy torch+transformers model lives in its own
interpreter (`$LR_DINO_PYTHON`), and we talk to the canonical persistent JSON-lines worker so the
application environment never imports torch/CUDA. One warm process
serves many crops.
    svc = DinoSubprocessService()                 # python from $LR_DINO_PYTHON (else sys.executable)
    registry.register("detect", svc)
    dets = registry.detect().detect(img, "chair")  # -> list[Detection]
    svc.close()
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any

from lrauthor import REPO_ROOT  # worker cwd: the checkout, so relative data paths resolve


@dataclass(slots=True)
class Detection:
    label: str
    box_xyxy: tuple[float, float, float, float]
    score: float


class DinoSubprocessService:
    name = "grounding-dino"

    def __init__(
        self,
        *,
        python: str | None = None,
        command: list[str] | None = None,
        model_id: str | None = None,
        cwd: str | os.PathLike | None = None,
        env: dict[str, str] | None = None,
        start_timeout: float = 180.0,
    ) -> None:
        self.python = python or os.environ.get("LR_DINO_PYTHON") or sys.executable
        # default: run the worker as a module under the isolated interpreter
        self.command = command or [
            self.python,
            "-u",
            "-m",
            "lrauthor.models.grounding_dino.worker",
        ]
        self.model_id = model_id or os.environ.get("LR_DINO_MODEL")
        self.cwd = str(cwd or REPO_ROOT)
        self.env = env
        self.start_timeout = start_timeout
        self._proc: subprocess.Popen | None = None

    # -- lifecycle --------------------------------------------------------------
    def _ensure(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        environ = {**os.environ, **(self.env or {})}
        environ.setdefault("PYTHONPATH", self.cwd)
        self._proc = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=self.cwd,
            env=environ,
            bufsize=1,
        )
        hello = self._rpc({"op": "ping"})  # blocks until the model env is importable
        if not hello.get("ok"):
            raise RuntimeError(f"DINO worker failed to start: {hello}")

    def _rpc(self, req: dict) -> dict:
        assert self._proc and self._proc.stdin and self._proc.stdout
        self._proc.stdin.write(json.dumps(req) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            err = self._proc.stderr.read() if self._proc.stderr else ""
            raise RuntimeError(f"DINO worker died (no response). stderr tail:\n{err[-800:]}")
        return json.loads(line)

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            self._proc.kill()
        self._proc = None

    # -- DetectionService -------------------------------------------------------
    def detect(
        self,
        image: Any,
        prompt: str,
        *,
        box_threshold: float = 0.30,
        text_threshold: float = 0.20,
        upright: bool = True,
    ) -> list[Detection]:
        self._ensure()
        path, is_tmp = self._as_path(image)
        try:
            resp = self._rpc(
                {
                    "op": "detect",
                    "image_path": path,
                    "prompt": prompt,
                    "box_threshold": box_threshold,
                    "text_threshold": text_threshold,
                    "upright": upright,
                    "model_id": self.model_id,
                }
            )
        finally:
            if is_tmp:
                os.unlink(path)
        if not resp.get("ok"):
            raise RuntimeError(f"DINO detect failed: {resp.get('error')}")
        return [
            Detection(label=d["label"], box_xyxy=tuple(d["box"]), score=float(d["score"]))
            for d in resp.get("detections", [])
        ]

    # -- EmbeddingService (DINOv2, same warm process) ---------------------------
    def embed(
        self,
        images: list[Any],
        *,
        upright: bool = True,
    ) -> list[list[float]]:
        """L2-normalised DINOv2 CLS embedding per image (paths pass through; PIL/ndarray are
        written to a temp PNG). Order preserved. Runs in the isolated torch env alongside detect."""
        self._ensure()
        paths: list[str] = []
        temps: list[str] = []
        for img in images:
            p, is_tmp = self._as_path(img)
            paths.append(p)
            if is_tmp:
                temps.append(p)
        try:
            resp = self._rpc(
                {"op": "embed", "image_paths": paths, "upright": upright, "model_id": None}
            )
        finally:
            for p in temps:
                try:
                    os.unlink(p)
                except OSError:
                    pass
        if not resp.get("ok"):
            raise RuntimeError(f"DINO embed failed: {resp.get('error')}")
        return [[float(v) for v in row] for row in resp.get("embeddings", [])]

    @staticmethod
    def _as_path(image: Any) -> tuple[str, bool]:
        """Return (path, is_temp). Paths pass through; PIL/ndarray are written to a temp PNG."""
        if isinstance(image, (str, os.PathLike)):
            return str(image), False
        from PIL import Image

        img = image if hasattr(image, "save") else Image.fromarray(image)
        fd, p = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        img.convert("RGB").save(p)
        return p, True
