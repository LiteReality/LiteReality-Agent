"""Single reconstruction entry for the init stage — call THIS, not the launcher directly.

Routes to the configured hosted `gen3d` tool when installed via
`set_service`, else returns None so the caller falls back to the local TRELLIS launcher. Same
pattern as `detector.py` for DINO. Installing this module imports no torch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_SERVICE: Any = None  # a Generation3DService; None → local launcher


def set_service(service: Any) -> None:
    global _SERVICE
    _SERVICE = service


def using_service() -> bool:
    return _SERVICE is not None


def service() -> Any:
    return _SERVICE


def reconstruct_refs(
    refs: list[tuple[str, Path]],
    out_dir: Path | str,
    *,
    seed: int = 42,
    simplify: float = 0.95,
    texture_size: int = 1024,
) -> dict[str, str] | None:
    """refs = [(asset_id, png_path)] → {asset_id: glb_path}, via the tool (parallel). Returns
    None if no tool is installed (caller should use the local launcher)."""
    if _SERVICE is None:
        return None
    items = {aid: str(png) for aid, png in refs}
    return _SERVICE.reconstruct_many(
        items, out_dir=str(out_dir), seed=seed, simplify=simplify, texture_size=texture_size
    )
