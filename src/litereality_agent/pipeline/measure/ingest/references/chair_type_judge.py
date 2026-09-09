"""Chair type grouping via a Claude Opus 4.8 agent — the judgment layer.

The DINOv2 embedding groups by *global appearance*, which blurs the attributes that actually
define a chair type in a real room: frame material (wood vs chrome vs black metal), armrest
material/presence, upholstery colour, base style. A meeting room of visually-similar chairs then
either over-splits (occlusion) or over-merges (distinct materials collapse). Those distinctions are
exactly what a vision LLM reads reliably and an embedding cannot.

So the type decision is delegated to an agent: it Reads one best-views montage per chair and returns
a structured grouping keyed on those attributes. Runs through ``claude_agent_sdk`` on the logged-in
Claude subscription (no ``ANTHROPIC_API_KEY`` ⇒ not metered), same path as the procedural agent.
Default model ``claude-opus-5`` (``$LR_CHAIR_JUDGE_MODEL``).

The embedding is not wasted — :mod:`chair_clusters` still uses it to pick each chair's clearest
representative view and as the deterministic fallback when the SDK/subscription is unavailable.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path

from PIL import Image

from litereality_agent.fonts import font as _shared_font

DEFAULT_JUDGE_MODEL = os.environ.get("LR_CHAIR_JUDGE_MODEL", "claude-opus-5")
MONTAGE_VIEWS = int(os.environ.get("LR_CHAIR_JUDGE_VIEWS", "3"))  # best views per chair in the montage
JUDGE_TIMEOUT = float(os.environ.get("LR_CHAIR_JUDGE_TIMEOUT", "600"))  # wall-clock cap for the agent


def available() -> bool:
    """True if the agent SDK is importable (grouping can be delegated to Claude)."""
    try:
        import claude_agent_sdk  # noqa: F401

        return True
    except Exception:
        return False


def _ranking_key(path: Path) -> tuple[int, str]:
    """Sort crops by their ranking index (0 = best view), then name."""
    m = re.search(r"ranking_(\d+)", path.name)
    return (int(m.group(1)) if m else 99, path.name)


def _oriented(path: Path) -> Image.Image:
    # crops are stored landscape (portrait capture on its side); rotate upright for the LLM.
    with Image.open(path) as im:
        return im.convert("RGB").transpose(Image.Transpose.ROTATE_270)


def build_montages(
    chairs: list[tuple[str, list[Path]]], montage_dir: Path, *, view_size: int = 340
) -> dict[str, Path]:
    """One labeled montage per chair (its best ``MONTAGE_VIEWS`` views side by side). Returns
    ``{chair_name: montage_path}``."""
    from PIL import ImageDraw, ImageFont

    montage_dir.mkdir(parents=True, exist_ok=True)
    try:
        font = _shared_font(30)
    except Exception:
        font = ImageFont.load_default()
    out: dict[str, Path] = {}
    for name, images in chairs:
        views = sorted(images, key=_ranking_key)[:MONTAGE_VIEWS]
        n = max(len(views), 1)
        canvas = Image.new("RGB", (n * view_size, view_size + 40), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        for i, p in enumerate(views):
            im = _oriented(p)
            im.thumbnail((view_size - 8, view_size - 8), Image.Resampling.LANCZOS)
            canvas.paste(im, (i * view_size + (view_size - im.width) // 2, 4 + (view_size - im.height) // 2))
        draw.text((6, view_size + 6), name, fill=(0, 0, 0), font=font)
        path = montage_dir / f"{name}.jpg"
        canvas.save(path, quality=90)
        out[name] = path
    return out


def _build_prompt(montage_dir: Path, names: list[str]) -> str:
    listing = ", ".join(f"{n}.jpg" for n in names)
    return f"""You are grading a 3D-reconstruction pipeline. A meeting room was scanned and each chair \
instance was cropped out. In `{montage_dir}` there is ONE montage image per chair: {listing}.
Each montage shows up to {MONTAGE_VIEWS} views of the SAME chair instance, labeled with its name.

Your job: group these {len(names)} chair instances by TYPE (which physical product model they are), \
so downstream we build ONE 3D asset per type and place it at every instance of that type.

Read EVERY montage with the Read tool before deciding. Two instances are the SAME type ONLY if ALL of \
these agree:
  - frame / leg material: wood  vs  bright metal/chrome  vs  black metal/plastic
  - armrests: none / wood / metal / padded-black  (arms vs no-arms is a real type difference)
  - seat + back upholstery colour: e.g. blue vs black vs grey
  - base style: cantilever (sled) vs four-leg vs swivel
Crops differ in viewpoint, lighting, blur and occlusion — judge the actual object, not the photo. A very \
occluded crop (e.g. only an armrest + wall) should still be assigned to the type it most plausibly is, \
from whatever material/colour cues are visible; mark those low confidence.

Prefer to SPLIT when materials clearly differ — wood, chrome, and black are different types even if the \
seat colour matches.

Output ONLY a fenced ```json block (no prose) with this exact shape:
{{
  "types": [
    {{"type_id": "T0",
      "signature": {{"frame_material": "...", "armrests": "...", "seat_color": "...", "base": "..."}},
      "members": ["Chair0", "..."],
      "confidence": "high|medium|low",
      "notes": "one short line"}}
  ]
}}
Every chair name must appear in exactly one type."""


async def _run_agent(montage_dir: Path, names: list[str], model: str) -> tuple[dict, float]:
    from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

    options = ClaudeAgentOptions(
        cwd=str(montage_dir),
        add_dirs=[str(montage_dir)],
        system_prompt={"type": "preset", "preset": "claude_code"},
        allowed_tools=["Read", "Glob"],
        permission_mode="bypassPermissions",
        max_turns=max(40, 3 * len(names)),
        model=model,
    )
    result_text = ""
    cost = 0.0
    async for message in query(prompt=_build_prompt(montage_dir, names), options=options):
        if isinstance(message, ResultMessage):
            result_text = message.result or ""
            cost = message.total_cost_usd or 0.0
    m = re.search(r"```json\s*(.+?)```", result_text, re.S) or re.search(r"(\{.*\})", result_text, re.S)
    if not m:
        raise RuntimeError(f"judge agent returned no JSON: {result_text[:300]}")
    return json.loads(m.group(1)), cost


def judge_types(
    chairs: list[tuple[str, list[Path]]],
    montage_dir: Path,
    *,
    model: str | None = None,
) -> list[dict]:
    """Group chairs by type via the Opus 4.8 agent. ``chairs`` is ``[(name, [crop_paths]), ...]``.

    Returns a list of groups, each ``{"members": [...], "signature": {...}, "confidence": str,
    "notes": str}``, ordered by first-member natural order. Raises on hard agent/parse failure so the
    caller can fall back to the deterministic embedding clustering. Any chair the agent omits is
    appended as its own single-member group (never silently dropped)."""
    model = model or DEFAULT_JUDGE_MODEL
    names = [name for name, _ in chairs]
    build_montages(chairs, montage_dir)

    data, cost = asyncio.run(asyncio.wait_for(_run_agent(montage_dir, names, model), JUDGE_TIMEOUT))
    print(f"  [chair-judge] {model}: {len(names)} chairs -> "
          f"{len(data.get('types', []))} types  (cost_usd≈{round(cost, 4)}, subscription)", flush=True)

    groups: list[dict] = []
    assigned: set[str] = set()
    for t in data.get("types", []):
        members = [m for m in t.get("members", []) if m in names and m not in assigned]
        if not members:
            continue
        assigned.update(members)
        groups.append(
            {
                "members": members,
                "signature": t.get("signature", {}),
                "confidence": t.get("confidence"),
                "notes": t.get("notes"),
            }
        )
    # never drop a chair the agent forgot: give it its own group, flagged.
    for name in names:
        if name not in assigned:
            groups.append(
                {"members": [name], "signature": {}, "confidence": "low",
                 "notes": "unassigned by judge; kept as singleton"}
            )
    return groups
