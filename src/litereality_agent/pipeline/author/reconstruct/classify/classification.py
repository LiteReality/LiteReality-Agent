"""Select the hosted complexity classifier (Claude by default, OpenAI optional)."""

from __future__ import annotations

import os

DEFAULT_CLASSIFY_MODEL = "claude"  # a non-None default; each backend resolves its own model


def provider() -> str:
    # Provider policy: classification (reasoning) is CLAUDE by default; OpenAI is image-gen only.
    # An explicit LR_CLASSIFY_PROVIDER override still wins for users who want it, but the default
    # never routes classify to OpenAI just because image-gen is OpenAI.
    p = os.environ.get("LR_CLASSIFY_PROVIDER")
    selected = p.strip().lower() if p else "claude"
    if selected not in {"claude", "openai"}:
        raise ValueError(f"unsupported classifier provider: {selected}")
    return selected


def classify_image(*args, **kwargs):
    p = provider()
    if p == "openai":
        from litereality_agent.pipeline.author.reconstruct.classify import (
            classify_openai as classify,
        )

        return classify.classify_image(*args, **kwargs)
    from litereality_agent.pipeline.author.reconstruct.classify import (
        classify_claude as classify,
    )

    return classify.classify_image(*args, **kwargs)
