"""Rear views must be available to inspect hidden panels and supports."""
import ast
from pathlib import Path


def test_preview_has_front_and_rear_directions():
    script = Path(__file__).resolve().parents[1] / (
        "src/litereality_agent/models/object_generation/articulated-glb-agent/"
        ".claude/skills/image-to-articulated-glb/scripts/render_glb_preview.py")
    tree = ast.parse(script.read_text())
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "VIEW_DIRS" for t in n.targets))
    directions = ast.literal_eval(assignment.value)
    assert directions['front'][1] < 0 < directions['back'][1]
    assert directions['iso'][1] < 0 < directions['back_iso'][1]
