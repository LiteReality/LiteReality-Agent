"""Keep package ownership and dependency direction explicit."""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "litereality_agent"
LAYERS = {"agent", "models", "pipeline", "room_ops", "runtimes"}
# `models` may reach `agent` because procedural object generation is a model that happens to be
# produced by an agent session: it drives a harness from `agent/providers`. The arrow is safe
# because `agent` imports nothing from `models`, so the two cannot form a cycle.
ALLOWED = {
    "agent": {"room_ops"},
    "models": {"agent", "runtimes"},
    "pipeline": {"agent", "models", "room_ops"},
    "room_ops": set(),
    "runtimes": set(),
}


def imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def test_layer_imports_only_point_inward():
    violations = []
    for path in PACKAGE.rglob("*.py"):
        relative = path.relative_to(PACKAGE)
        owner = relative.parts[0]
        if owner not in LAYERS:
            continue
        for imported in imports(path):
            if not imported.startswith("litereality_agent."):
                continue
            target = imported.split(".")[1]
            if target in LAYERS and target != owner and target not in ALLOWED[owner]:
                violations.append(f"{relative}:{owner} -> {target}")
    assert not violations, "invalid layer imports:\n" + "\n".join(violations)


def test_only_supported_package_areas_exist():
    retired = {"adapters", "services", "shared"}
    present = {path.name for path in PACKAGE.iterdir() if path.is_dir() and any(path.rglob("*.py"))}
    assert not present & retired


def test_ported_preprocessing_stays_behind_project_adapters():
    preprocessing = PACKAGE / "pipeline" / "measure" / "ingest" / "preprocessing"
    vendor = preprocessing / "vendor" / "litereality"
    assert {path.name for path in preprocessing.glob("*.py")} == {
        "__init__.py",
        "object_images.py",
        "scene_data.py",
    }
    assert not (
        PACKAGE / "pipeline" / "measure" / "ingest" / "extract" / "lr_preprocessing"
    ).exists()

    violations = []
    for path in vendor.glob("*.py"):
        for imported in imports(path):
            if imported.startswith("litereality_agent"):
                violations.append(f"{path.name}: {imported}")
        if "LR_ENLARGED_CROP_OBJECTS" in path.read_text(encoding="utf-8"):
            violations.append(f"{path.name}: project environment setting")
    assert not violations, "project logic found in preprocessing vendor code:\n" + "\n".join(
        violations
    )
