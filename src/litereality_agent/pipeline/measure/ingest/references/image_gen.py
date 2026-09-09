"""Hosted reference-image generation, on OpenAI or Gemini.

One call per object: the evidence sheet plus the prompt go up, one clean object render comes back.
Pick the backend with ``$LR_IMAGE_PROVIDER`` (``openai`` by default, or ``gemini``); naming a
``gemini-*`` model is enough on its own. The retry ladder, telemetry, and black-background
normalisation are shared, so the rest of the pipeline cannot tell which one produced a reference.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from litereality_agent import telemetry

DEFAULT_IMAGE_MODEL = "gpt-image-1"
DEFAULT_GEMINI_IMAGE_MODEL = "gemini-2.5-flash-image"


def resolve_provider(model: str | None = None) -> str:
    """Which backend to call: explicit ``$LR_IMAGE_PROVIDER`` wins, else infer from the model name.

    Inferring matters because the model is what callers actually pass around (``--image-model``),
    so naming a ``gemini-*`` model without also setting the provider should just work rather than
    sending a Gemini model id to OpenAI and failing on an unhelpful 404.
    """
    explicit = (os.environ.get("LR_IMAGE_PROVIDER") or "").strip().lower()
    if explicit in ("openai", "gemini"):
        return explicit
    name = (model or os.environ.get("LR_OPENAI_IMAGE_MODEL") or "").lower()
    return "gemini" if name.startswith("gemini") else "openai"

# Reference generation is one hosted image call per object, ~45s each, and every caller had it in
# a serial loop. The ceiling is OpenAI's rate limit, not local CPU, so the cap is about not opening
# an unbounded number of requests at once on a large room rather than about cores.
#
# 6 meant a typical 17-object room ran three serial waves — about 135s, most of a cold scene_init.
# 20 covers a normal room in one wave. It is safe to raise only because the retry path below now
# waits out a 429 instead of burning its attempts in nine seconds: exceeding the account's budget
# costs time, not quality. Lower it with $LR_IMAGE_WORKERS on a constrained tier.
MAX_IMAGE_WORKERS = int(os.environ.get("LR_IMAGE_WORKERS", "20"))

# A rate-limited request needs to wait out the account's per-minute image budget, so it gets its
# own ladder (20s, 40s, ...) rather than the short one used for a timeout or a transient 5xx.
_RATE_LIMIT_BACKOFF = 20.0
_MAX_BACKOFF = 90.0


def image_workers(n_items: int) -> int:
    """Pool size for a batch of `n_items` reference images."""
    return max(1, min(n_items, MAX_IMAGE_WORKERS))


def _retry_after_seconds(exc: Exception) -> float | None:
    """The server's own Retry-After, when it sent one. Trust it over any guess we make."""
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    for key, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        raw = headers.get(key)
        if raw is None:
            continue
        try:
            return float(raw) * scale
        except (TypeError, ValueError):
            continue
    return None


def _is_rate_limit(exc: Exception) -> bool:
    # google-genai raises ClientError with `.code`, and says RESOURCE_EXHAUSTED rather than 429 in
    # the body, so neither of the OpenAI tests alone catches a throttled Gemini call.
    return (
        getattr(exc, "status_code", None) == 429
        or getattr(exc, "code", None) == 429
        or type(exc).__name__ == "RateLimitError"
        or "RESOURCE_EXHAUSTED" in str(exc)
    )


# OpenAI answers 429 for two unrelated conditions: "you are going too fast" and "your balance is
# empty". Only the first clears by waiting.
_OUT_OF_CREDIT = ("insufficient_quota", "credit_balance_exhausted", "billing_hard_limit_reached")


def is_out_of_credit(exc: Exception) -> bool:
    """True for a 429 that means no credits rather than too many requests.

    Worth separating because the two need opposite handling, and the backoff above makes getting
    it wrong expensive: waiting 20s then 40s for a balance that will never refill costs a minute
    per object before the identical failure lands anyway. Fail on the first attempt instead, and
    say what to do about it — an empty account otherwise reads as a mysterious rate limit.
    """
    body = getattr(exc, "body", None)
    detail = ""
    if isinstance(body, dict):
        err = body.get("error") or {}
        detail = f"{err.get('type', '')} {err.get('code', '')}"
    haystack = f"{detail} {exc}".lower()
    return any(marker in haystack for marker in _OUT_OF_CREDIT)


def retry_delay(exc: Exception, attempt: int) -> float:
    """Seconds to wait before retrying `exc`.

    A 429 is a different failure from a timeout: it says the account's images-per-minute budget is
    spent, and no amount of retrying inside that minute will help. The old fixed 1.5/3/4.5s ladder
    spent all three attempts in nine seconds and then raised — which object_references turns into a
    PLACEHOLDER reference, so a rate limit degraded the room silently instead of failing loudly.
    """
    explicit = _retry_after_seconds(exc)
    if explicit is not None:
        return min(explicit + 0.5, _MAX_BACKOFF)
    if _is_rate_limit(exc):
        return min(_RATE_LIMIT_BACKOFF * (2 ** attempt), _MAX_BACKOFF)
    return 1.5 * (attempt + 1)

# gpt-image-1 token pricing ($/1M tokens); output image tokens dominate. Used only for the
# rough cost line we log — override via env if OpenAI changes rates.
_PRICE_IN_TEXT = float(os.environ.get("LR_OPENAI_PRICE_IN_TEXT", "5.0"))
_PRICE_IN_IMAGE = float(os.environ.get("LR_OPENAI_PRICE_IN_IMAGE", "10.0"))
_PRICE_OUT_IMAGE = float(os.environ.get("LR_OPENAI_PRICE_OUT_IMAGE", "40.0"))


def _cost_usd(usage: dict) -> float:
    """Rough USD from a gpt-image-1 usage block."""
    it = usage.get("input_tokens_details", {}) or {}
    text_in = it.get("text_tokens", 0)
    image_in = it.get("image_tokens", 0)
    out = usage.get("output_tokens", 0)
    return round(
        (text_in * _PRICE_IN_TEXT + image_in * _PRICE_IN_IMAGE + out * _PRICE_OUT_IMAGE) / 1e6, 5
    )


# gemini-2.5-flash-image token pricing ($/1M); one 1024px image is ~1290 output tokens (~$0.04).
_PRICE_GEMINI_IN = float(os.environ.get("LR_GEMINI_PRICE_IN", "0.30"))
_PRICE_GEMINI_OUT = float(os.environ.get("LR_GEMINI_PRICE_OUT", "30.0"))


def _gemini_cost_usd(usage: dict) -> float:
    return round(
        (usage.get("input_tokens", 0) * _PRICE_GEMINI_IN
         + usage.get("output_tokens", 0) * _PRICE_GEMINI_OUT) / 1e6, 5
    )


def _flatten_on_black(png_bytes: bytes) -> bytes:
    """Composite a possibly-transparent render onto solid black so normalize's black-key crop works."""
    im = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    canvas = Image.new("RGBA", im.size, (0, 0, 0, 255))
    canvas.alpha_composite(im)
    out = io.BytesIO()
    canvas.convert("RGB").save(out, format="PNG")
    return out.getvalue()


def _call_openai(sheet_path, full_prompt, img_model, key, timeout, use_transparent):
    """One OpenAI image edit. Returns (png_bytes, usage, cost_usd, extra_response_fields)."""
    from openai import OpenAI

    quality = os.environ.get("LR_OPENAI_IMAGE_QUALITY", "medium")
    client = OpenAI(api_key=key, timeout=timeout)
    edit_kwargs = dict(model=img_model, prompt=full_prompt, size="1024x1024", quality=quality)
    if use_transparent:
        edit_kwargs["background"] = "transparent"
    with open(sheet_path, "rb") as fh:
        result = client.images.edit(image=fh, **edit_kwargs)
    png = base64.b64decode(result.data[0].b64_json)
    usage = result.usage.model_dump() if getattr(result, "usage", None) is not None else {}
    return png, usage, _cost_usd(usage), {"quality": quality}


def _call_gemini(sheet_path, full_prompt, img_model, key, timeout):
    """One Gemini image edit. Returns (png_bytes, usage, cost_usd, extra_response_fields).

    Gemini returns the render as an inline_data part alongside any commentary parts, so the image
    has to be picked out rather than read off a fixed index. A response with no image part at all
    is normally a safety block, which the retry loop should treat as a failed attempt.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=timeout * 1000))
    mime = "image/jpeg" if Path(sheet_path).suffix.lower() in (".jpg", ".jpeg") else "image/png"
    result = client.models.generate_content(
        model=img_model,
        contents=[
            full_prompt,
            types.Part.from_bytes(data=Path(sheet_path).read_bytes(), mime_type=mime),
        ],
        # We pass no tools, and leaving AFC on makes the SDK warn once per call — one line of noise
        # per object, so a 20-object room buries its own progress output.
        config=types.GenerateContentConfig(
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )
    png = None
    for candidate in (result.candidates or []):
        for part in ((candidate.content.parts if candidate.content else None) or []):
            blob = getattr(part, "inline_data", None)
            if blob is not None and blob.data:
                png = blob.data
                break
        if png:
            break
    if png is None:
        reason = getattr(result, "prompt_feedback", None) or getattr(
            (result.candidates or [None])[0], "finish_reason", None)
        raise RuntimeError(f"Gemini returned no image part (finish_reason={reason})")
    meta = getattr(result, "usage_metadata", None)
    usage = {
        "input_tokens": getattr(meta, "prompt_token_count", 0) or 0,
        "output_tokens": getattr(meta, "candidates_token_count", 0) or 0,
        "total_tokens": getattr(meta, "total_token_count", 0) or 0,
    }
    return png, usage, _gemini_cost_usd(usage), {}


def generate_reference(
    sheet_path: Path,
    prompt: str,
    out_path: Path,
    *,
    model: str | None = None,
    api_key: str | None = None,
    timeout: int = 180,
    retries: int = 3,
    force: bool = False,
    response_dir: Path | None = None,
) -> dict | None:
    """Generate an isolated object reference and write it to ``out_path``."""
    sheet_path = Path(sheet_path)
    out_path = Path(out_path)
    call_t0 = time.time()
    provider = resolve_provider(model)
    if out_path.exists() and not force:
        telemetry.on_image_generation(prompt, sheet_path, out_path, "skipped_exists", model=provider)
        return None

    if provider == "gemini":
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("Gemini image generation needs $GEMINI_API_KEY.")
        img_model = (os.environ.get("LR_GEMINI_IMAGE_MODEL")
                     or (model if (model or "").lower().startswith("gemini") else None)
                     or DEFAULT_GEMINI_IMAGE_MODEL)
        # No transparent-background option on the image models, so take the black-background path.
        use_transparent = False
    else:
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OpenAI image generation needs $OPENAI_API_KEY.")
        img_model = os.environ.get("LR_OPENAI_IMAGE_MODEL") or model or DEFAULT_IMAGE_MODEL
        # gpt-image-2 rejects transparent backgrounds; only the gpt-image-1 family supports them.
        # With transparent we composite the exact object onto black; otherwise we ask the model for
        # a solid black background (normalize keys on black either way).
        use_transparent = "gpt-image-2" not in img_model
    label = f"{provider}:{img_model}"
    # A generic clause kills the hallucinated hooks/posters/props gpt-image-1 tends to bolt on.
    clean = (
        "\n\nRender ONLY this single object as a clean photoreal PBR asset — remove loose clutter and "
        "scene (coats, bags, papers, people, room background, floor, added text or borders), but KEEP "
        "every fixture that is physically part of the object (e.g. hooks, handles, hinges, rails). Even "
        "studio lighting, no cast shadow, centered and fully visible. Reproduce the object's real "
        "structure and the EXACT COUNT of every repeated part (panels, glazed panes, drawers, doors, "
        "shelves, legs, hooks); do NOT simplify, drop, merge, or invent any real feature (e.g. never "
        "flatten a glazed/panelled surface into a plain one, and never add glazing that is not there)."
    )
    bg = (
        " Fully transparent background." if use_transparent
        else " Place it on a SOLID PURE BLACK (#000000) background."
    )
    full_prompt = prompt + clean + bg

    last_err = None
    for attempt in range(retries):
        try:
            if provider == "gemini":
                png, usage, cost, extra = _call_gemini(
                    sheet_path, full_prompt, img_model, key, timeout)
            else:
                png, usage, cost, extra = _call_openai(
                    sheet_path, full_prompt, img_model, key, timeout, use_transparent)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(_flatten_on_black(png) if use_transparent else png)
            if response_dir is not None:
                response_dir.mkdir(parents=True, exist_ok=True)
                (response_dir / f"{provider}_response_{attempt}.json").write_text(
                    json.dumps({"model": img_model, "usage": usage, "cost_usd": cost, **extra},
                               indent=2),
                    encoding="utf-8",
                )
            telemetry.on_image_generation(
                prompt, sheet_path, out_path, "ok", model=label,
                elapsed_sec=round(time.time() - call_t0, 3), attempt=attempt, usage=usage,
            )
            print(f"    [{provider}-image] {out_path.name}  ${cost}  "
                  f"({usage.get('output_tokens','?')} out tok)", flush=True)
            return {
                "status": "ok",
                "model": label,
                "attempt": attempt,
                "output": str(out_path),
                "elapsed_sec": round(time.time() - call_t0, 3),
                "usage": usage,
                "cost_usd": cost,
            }
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
            if provider == "openai" and is_out_of_credit(exc):
                telemetry.on_image_generation(
                    prompt, sheet_path, out_path, "error", model=label,
                    elapsed_sec=round(time.time() - call_t0, 3), error=str(last_err)[:400],
                )
                raise RuntimeError(
                    "OpenAI has no credits remaining, so every reference from here will fail. "
                    "Top up at https://platform.openai.com/settings/organization/billing/ and "
                    f"re-run — finished references are kept and skipped. ({last_err})"
                ) from exc
            if attempt == retries - 1:
                break  # nothing left to wait for; the old code slept anyway
            delay = retry_delay(exc, attempt)
            if _is_rate_limit(exc):
                print(
                    f"    [{provider}-image] rate limited on {out_path.name}; "
                    f"waiting {delay:.0f}s (attempt {attempt + 1}/{retries})",
                    flush=True,
                )
            time.sleep(delay)

    telemetry.on_image_generation(
        prompt, sheet_path, out_path, "error", model=label,
        elapsed_sec=round(time.time() - call_t0, 3), error=str(last_err)[:400],
    )
    raise RuntimeError(f"{provider} image returned nothing after {retries} tries: {last_err}")


def normalize_reference(
    source: Path,
    destination: Path,
    canvas_size: int = 1024,
    target_fill: float = 0.78,
    pad_ratio: float = 0.08,
) -> None:
    """Crop an object from its black background and center it on a square canvas."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    pixels = np.asarray(image)
    mask = (pixels[:, :, 0] > 14) | (pixels[:, :, 1] > 14) | (pixels[:, :, 2] > 14)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        image.resize((canvas_size, canvas_size), Image.Resampling.LANCZOS).save(destination)
        return

    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
    width, height = x1 - x0, y1 - y0
    padding = int(max(width, height) * pad_ratio)
    bounds = (
        max(0, x0 - padding),
        max(0, y0 - padding),
        min(image.width, x1 + padding),
        min(image.height, y1 + padding),
    )
    crop = image.crop(bounds)
    scale = target_fill * canvas_size / max(crop.width, crop.height)
    resized = crop.resize(
        (max(1, int(crop.width * scale)), max(1, int(crop.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (canvas_size, canvas_size), (0, 0, 0))
    canvas.paste(resized, ((canvas_size - resized.width) // 2, (canvas_size - resized.height) // 2))
    canvas.save(destination)
