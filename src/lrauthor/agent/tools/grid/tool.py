"""grid — PRIMITIVE: put a metric ruler on a rectified surface stitch.

The stitch is head-on and metric (`ppm` = pixels per metre, from the surface_ref manifest), so a
grid drawn on it in metres lets a position be READ off the photograph rather than estimated. That
matters because the model is weak at judging proportion by eye. It is deterministic and free —
no model call — and answers the common case directly: "where along this wall, and how high?"

Output is a JPEG with:
  - major lines every `major_m` (default 0.5 m), labelled in metres
  - minor lines every `minor_m` (default 0.1 m), unlabelled
  - u across (distance along the wall from its start) and z up (height above the floor)

Read a fixture's position straight off it and write those numbers into SHELL.

Invented by the authoring agent itself: it hand-rolled this in Bash 13 times in one run (see
run/<scan>/.../traces/img/*_grid.jpg) before this tool existed.
"""

from __future__ import annotations

import json

from lrauthor.agent.tools.base import (
    BaseDeclarativeTool,
    BaseToolInvocation,
    ToolParamsModel,
    ToolResult,
    make_tool_schema,
    validate_tool_params,
)

# Drawn onto an alpha overlay, not straight onto the photo: a ruler you cannot see THROUGH is
# useless, and at 0.1 m on a 160 ppm stitch the minor lines alone are ~40 verticals.
MAJOR_U = (220, 40, 40, 235)  # metres along the wall
MAJOR_Z = (40, 90, 230, 235)  # metres above the floor
MINOR = (150, 175, 255, 70)  # faint — for counting, not for reading


class GridParams(ToolParamsModel):
    surface: str
    major_m: float = 0.5
    minor_m: float = 0.1


class GridInvocation(BaseToolInvocation):
    def get_description(self) -> str:
        return f"grid: {self.params.surface} @ {self.params.major_m}m"

    async def execute(self) -> ToolResult:
        from PIL import Image, ImageDraw

        from lrauthor.agent.tools.shared import config
        from lrauthor.fonts import font as _font

        p = self.params
        if p.major_m <= 0 or p.minor_m <= 0:
            return ToolResult(error="major_m and minor_m must be positive")

        man = config.SURFACE_REF_MANIFEST
        if not man.is_file():
            return ToolResult(error=f"no surface_ref manifest at {man} — build the stitches first")
        try:
            ppm = float(json.loads(man.read_text()).get("ppm") or 0)
        except (OSError, ValueError) as e:
            return ToolResult(error=f"could not read ppm from {man}: {e}")
        if ppm <= 0:
            return ToolResult(error="surface_ref manifest has no usable 'ppm'")

        stitch = config.SURFACE_REF / f"{p.surface}_stitched.jpg"
        if not stitch.is_file():
            return ToolResult(error=f"no stitch for {p.surface} at {stitch}")

        im = Image.open(stitch).convert("RGB")
        W, H = im.size
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        # Label size follows the image, so the ruler stays readable whatever the stitch resolution.
        f = _font(max(13, W // 70))

        def label(x: int, y: int, text: str, rgba) -> None:
            """Text on its own solid chip, so a number is never lost against the photo."""
            tb = d.textbbox((x, y), text, font=f)
            d.rectangle((tb[0] - 3, tb[1] - 2, tb[2] + 3, tb[3] + 2), fill=(0, 0, 0, 200))
            d.text((x, y), text, fill=rgba[:3] + (255,), font=f)

        def lines(step_m: float, width: int, labelled: bool) -> None:
            step_px = step_m * ppm
            if step_px < 6:  # denser than this is noise, not a ruler
                return
            n = 0
            while n * step_px <= W:  # verticals: u, metres along the wall from its start
                x = n * step_px
                d.line([(x, 0), (x, H)], fill=MAJOR_U if labelled else MINOR, width=width)
                if labelled and n:
                    label(int(x) + 4, 3, f"{n * step_m:.1f}", MAJOR_U)
                n += 1
            n = 0
            while n * step_px <= H:  # horizontals: z, metres above the floor (drawn bottom-up)
                y = H - n * step_px
                d.line([(0, y), (W, y)], fill=MAJOR_Z if labelled else MINOR, width=width)
                if labelled and n:
                    label(4, int(y) - f.size - 3, f"z{n * step_m:.1f}", MAJOR_Z)
                n += 1

        lines(p.minor_m, 1, False)
        lines(p.major_m, 2, True)
        im = Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")

        out_dir = config.SURFACE_REF / "_grid"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{p.surface}_grid.jpg"
        im.save(out, quality=92)

        return ToolResult(
            output={
                "image": str(out),
                "surface": p.surface,
                "ppm": round(ppm, 2),
                "extent_m": {"u": round(W / ppm, 3), "z": round(H / ppm, 3)},
                "major_m": p.major_m,
                "minor_m": p.minor_m,
                "how_to_read": (
                    "Red verticals are u = metres along the wall from its start; blue horizontals "
                    "are z = metres above the floor. Read a fixture's (u, z) straight off the image "
                    "and use those numbers directly in SHELL."
                ),
            }
        )


class GridTool(BaseDeclarativeTool):
    def __init__(self) -> None:
        schema = make_tool_schema(
            name="grid",
            description=(
                "Overlay a METRIC grid on a surface's rectified stitch and return the image. Use it "
                "to READ a fixture's position instead of estimating: red verticals are metres along "
                "the wall (u), blue horizontals are metres above the floor (z). Call this before "
                "placing or resizing anything on a wall — the numbers you read are the numbers to "
                "write into SHELL."
            ),
            parameters={
                "surface": {
                    "type": "string",
                    "description": "Surface id as named in SHELL, e.g. Wall1, Ceiling0, Floor0.",
                },
                "major_m": {
                    "type": "number",
                    "description": "Labelled line spacing in metres (default 0.5).",
                },
                "minor_m": {
                    "type": "number",
                    "description": "Unlabelled line spacing in metres (default 0.1).",
                },
            },
            required=["surface"],
        )
        super().__init__("grid", schema)

    async def build(self, params: dict) -> GridInvocation:
        return GridInvocation(validate_tool_params(GridParams, params))
