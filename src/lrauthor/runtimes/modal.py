"""Small Modal transport shared by hosted model services.

The transport knows how to find and invoke a deployed Modal function.  Model-specific
serialization stays beside the model implementation.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any


class ModalClient:
    """Look up one deployed function and invoke batches through Modal's map API."""

    def __init__(
        self,
        app_name: str,
        function_name: str,
        *,
        environment_name: str = "main",
        profile: str | None = None,
        credentials: tuple[str, str] | None = None,
        function: Any = None,
    ) -> None:
        if function is None:
            if not profile and not credentials:
                raise RuntimeError(
                    "Modal calls need credentials: set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET "
                    "in .env, or MODAL_PROFILE for a ~/.modal.toml profile. No workspace "
                    "default is used."
                )
            if credentials:
                # A token pair identifies the workspace on its own, so it takes precedence and no
                # ~/.modal.toml need exist.
                os.environ["MODAL_TOKEN_ID"], os.environ["MODAL_TOKEN_SECRET"] = credentials
            if profile:
                # Pin both lookup and billing to the explicitly configured shared profile.
                # Modal otherwise silently uses whichever local profile happens to be active.
                os.environ["MODAL_PROFILE"] = profile
            try:
                import modal
            except ImportError as exc:  # pragma: no cover - depends on optional install
                raise RuntimeError(
                    "Modal runtime support is not installed; run `uv sync --extra modal`."
                ) from exc
            function = modal.Function.from_name(
                app_name,
                function_name,
                environment_name=environment_name,
            )
        self.function = function

    def map(self, inputs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run inputs in parallel while preserving input order and per-item failures."""
        return list(self.function.map(inputs, return_exceptions=True))
