"""critic — PRIMITIVE. The JUDGE: grade render|photo image(s) against a goal.

Unlike read_image (open answer to an objective), the critic returns a VERDICT the loop can act on:
{pass, score 0-10, issues[]}. Used standalone as the gate ("is this item/stage done?") and inside
`fix` to decide whether the change landed. The tool owns the grading prompt; the agent supplies
only the goal + images.
"""

from __future__ import annotations

from lrauthor.agent.tools.base import (
    BaseDeclarativeTool,
    BaseToolInvocation,
    ToolParamsModel,
    ToolResult,
    make_tool_schema,
    validate_tool_params,
)

_PROMPT = (
    "You are a strict reconstruction critic. MOST images are a side-by-side: LEFT = the current 3D "
    "render, RIGHT = the real photo from the same pose. Each image's label (shown with it) states its "
    "layout — trust the label. SOME images are a SINGLE real-only reference (e.g. labelled 'REAL "
    "reference — no render side yet' or a real 'stitch'): those have NO render side to grade — use "
    "them only as ground-truth for what the real surface looks like, never fail the item from such an "
    "image alone. Grade ONLY against this goal:\n"
    "GOAL: {goal}\n{target_line}"
    "Judge whether the RENDER satisfies the goal when compared to the PHOTO (colour, lightness, "
    "finish, presence/placement — whatever the goal names). Be strict: near-misses fail.\n"
    'Reply as JSON: {{"pass": bool, "score": 0-10, "issues": ["concrete, actionable problem", ...]}}. '
    "score 8+ only when it genuinely matches; issues empty when pass=true."
)


class CriticParams(ToolParamsModel):
    images: list[str]
    goal: str
    target: str | None = None  # optional focus (Wall3, Table0) named in the verdict
    labels: list[str] | None = None  # per-image caption shown TO the VLM (defaults to filename)


class CriticInvocation(BaseToolInvocation):
    def get_description(self) -> str:
        return f"critic: {self.params.goal[:60]}"

    async def execute(self) -> ToolResult:
        from lrauthor.agent.tools._vlm import vision

        p = self.params
        if not p.images:
            return ToolResult(error="critic requires at least one image")
        tl = f"TARGET: focus only on {p.target}.\n" if p.target else ""
        try:
            verdict = await vision(p.images, _PROMPT.format(goal=p.goal, target_line=tl), json_mode=True, labels=p.labels)
        except Exception as e:  # noqa: BLE001
            return ToolResult(error=f"critic failed: {type(e).__name__}: {e}")
        # A vision backend may return one verdict per image; merge those conservatively.
        if isinstance(verdict, list) and verdict and all(isinstance(v, dict) and "pass" in v for v in verdict):
            merged_issues: list = []
            for v in verdict:
                for i in v.get("issues") or []:
                    if i not in merged_issues:
                        merged_issues.append(i)
            verdict = {
                "pass": all(v.get("pass") for v in verdict),
                "score": min(v.get("score", 0) for v in verdict),
                "issues": merged_issues,
            }
        if not isinstance(verdict, dict) or "pass" not in verdict:
            return ToolResult(error=f"critic returned malformed verdict: {verdict!r}"[:400])
        verdict.setdefault("score", 0)
        verdict.setdefault("issues", [])
        return ToolResult(output=verdict)


class CriticTool(BaseDeclarativeTool):
    def __init__(self) -> None:
        schema = make_tool_schema(
            name="critic",
            description=(
                "Grade render|photo comparison image(s) against a goal — the judge. Returns "
                '{pass, score 0-10, issues[]}. Use to decide whether a fix landed and to gate "done". '
                "Strict: near-misses fail."
            ),
            parameters={
                "images": {"type": "array", "items": {"type": "string"},
                           "description": "Side-by-side comparison image paths (render outputs)."},
                "goal": {"type": "string", "description": "What must be true for a pass."},
                "target": {"type": "string", "description": "Optional focus: Wall3 / Table0 / ..."},
            },
            required=["images", "goal"],
        )
        super().__init__("critic", schema)

    async def build(self, params: dict) -> CriticInvocation:
        return CriticInvocation(validate_tool_params(CriticParams, params))
