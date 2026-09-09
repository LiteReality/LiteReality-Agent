"""TRELLIS service for an isolated local process runtime.

Symmetric with ModalTrellisService (same Generation3DService interface), so `reconstructor`
calls it the same way.

Needs the TRELLIS.2 GPU environment selected by `$LITEREALITY_TRELLIS_PYTHON`. The canonical
inference module is used by default and can be overridden with `$LITEREALITY_TRELLIS_LAUNCHER`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from lrauthor import PACKAGE_ROOT  # the launcher is CODE, shipped in the wheel

DEFAULT_LAUNCHER = PACKAGE_ROOT / "models" / "trellis" / "inference.py"


def _resolve_python(explicit: str | None) -> str:
    for c in (
        explicit,
        os.environ.get("LITEREALITY_TRELLIS_PYTHON"),
    ):
        if c and Path(c).exists():
            return str(c)
    raise RuntimeError(
        "Local TRELLIS requires an explicit TRELLIS_PYTHON/LITEREALITY_TRELLIS_PYTHON."
    )


class LocalTrellisService:
    name = "trellis-local"

    def __init__(
        self, *, python: str | None = None, launcher: str | None = None, parallel: int = 1
    ) -> None:
        self.python = _resolve_python(python)
        self.launcher = Path(
            launcher or os.environ.get("LITEREALITY_TRELLIS_LAUNCHER") or DEFAULT_LAUNCHER
        )
        self.parallel = parallel
        self.last_report: dict | None = None

    # -- Generation3DService -----------------------------------------------------
    def reconstruct(
        self, images: list[Any], *, out_dir: str, asset_id: str = "asset", **opts: Any
    ) -> str:
        image = images[0] if isinstance(images, (list, tuple)) else images
        return self.reconstruct_many({asset_id: image}, out_dir=out_dir, **opts)[asset_id]

    def reconstruct_many(
        self,
        items: dict[str, Any],
        *,
        out_dir: str,
        seed: int = 42,
        simplify: float = 0.95,  # simplify accepted (hosted parity), unused locally
        decimation: int = 50000,
        texture_size: int = 4096,
    ) -> dict[str, str]:
        """{asset_id: image(path|PIL)} → {asset_id: glb_path} via the local launcher (batch)."""
        if not self.launcher.is_file():
            raise FileNotFoundError(f"TRELLIS launcher not found: {self.launcher}")
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        refs = Path(tempfile.mkdtemp(prefix="trellis_refs_"))
        try:
            for aid, img in items.items():
                dst = refs / f"{aid}.png"
                if isinstance(img, (str, os.PathLike)):
                    shutil.copy(img, dst)
                else:  # PIL
                    img.convert("RGB").save(dst)
            cmd = [
                self.python,
                str(self.launcher),
                "-i",
                str(refs),
                "-o",
                str(out),
                "--seed",
                str(seed),
                "--decimation",
                str(decimation),
                "--texture-size",
                str(texture_size),
                "--parallel",
                str(self.parallel),
                "--skip-existing",
            ]
            print(f"  [trellis-local] {len(items)} asset(s) via {self.launcher.name}", flush=True)
            t0 = time.time()
            rc = subprocess.run(cmd).returncode
            self.last_report = {"rc": rc, "seconds": round(time.time() - t0, 1), "n": len(items)}
        finally:
            shutil.rmtree(refs, ignore_errors=True)
        return {aid: str(out / f"{aid}.glb") for aid in items}
