"""Hosted OpenAI vision classifier.

Same signature; returns a parsed JSON decision. Uses an OpenAI vision model
(``$LR_OPENAI_VLM_MODEL``, default ``gpt-4o``) with JSON-object output. The response schema is
described to the model in the prompt (the caller's prompt already asks for those
fields); we validate presence of the required keys and fill sane defaults if missing.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

DEFAULT_OPENAI_VLM_MODEL = "gpt-4o"


def _data_uri(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def classify_image(
    image_path: Path,
    prompt: str,
    response_schema: dict,
    *,
    model: str | None = None,
    api_key: str | None = None,
    timeout: int = 60,
    retries: int = 3,
) -> dict:
    """Send one image + prompt to an OpenAI vision model, return the parsed JSON decision."""
    from openai import OpenAI

    image_path = Path(image_path)
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OpenAI classification needs $OPENAI_API_KEY.")
    if not model:
        model = os.environ.get("LR_OPENAI_VLM_MODEL", DEFAULT_OPENAI_VLM_MODEL)
    client = OpenAI(api_key=key, timeout=timeout)

    keys = ", ".join(response_schema.get("properties", {}).keys()) or "route, reason"
    sys_hint = (
        "You are a strict classifier. Reply with ONLY a JSON object containing exactly these "
        f"keys: {keys}. No prose, no markdown."
    )

    last_err = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.1,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": sys_hint},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": _data_uri(image_path)}},
                        ],
                    },
                ],
            )
            text = resp.choices[0].message.content or ""
            data = json.loads(text)
            for req in response_schema.get("required", []):
                data.setdefault(req, None)
            return data
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"

    raise RuntimeError(f"OpenAI classify failed after {retries} tries: {last_err}")
