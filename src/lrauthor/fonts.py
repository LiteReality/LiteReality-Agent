"""One font resolver for every image the pipeline draws on.

Fourteen modules used to hardcode `/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf` and fall
back to `ImageFont.load_default()` when it was missing. That path is Linux-only, so on macOS every
one of them silently dropped to the default bitmap face — a FIXED 10 px that ignores the size the
caller asked for. Labels came out roughly half their intended size on every Mac, and nothing
failed, so nothing said so.

So: resolve against a list that covers both platforms, cache the result, and only fall back to
`load_default(size)` (Pillow >= 10.1, which IS scalable) when no real face exists.

    from lrauthor.fonts import font
    draw.text((x, y), "Wall1", font=font(28), fill="white")
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

# Ordered by preference, bold first — these labels sit on top of photographs and need the weight.
_CANDIDATES = (
    # Linux (Debian/Ubuntu, and most CI images)
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    # macOS
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    # last resort, regular weight
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


@lru_cache(maxsize=1)
def font_path() -> str | None:
    """The first usable face on this machine, or None if there is genuinely nothing."""
    for candidate in _CANDIDATES:
        if Path(candidate).is_file():
            try:
                ImageFont.truetype(candidate, 12)  # a path can exist and still fail to load
                return candidate
            except OSError:
                continue
    return None


@lru_cache(maxsize=64)
def font(size: int) -> ImageFont.FreeTypeFont:
    """A real face at `size` px. Falls back to a SCALABLE default rather than the 10 px bitmap."""
    path = font_path()
    if path:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size)  # Pillow >= 10.1 accepts a size and scales
    except TypeError:
        return ImageFont.load_default()
