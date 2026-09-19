"""Required acceptance gate for saved layout geometry, including reused ingest outputs."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import adapter
from .adjust import Violation, check


class LayoutValidationError(RuntimeError):
    """A scene must not advance beyond layout with failed or incomplete validation."""


def inspect_layout(scene_data_dir: str | Path) -> dict[str, Any]:
    """Read and check the persisted boxes; never trust a previous report's zero count.

    This is the layout checker's proxy geometry, tolerances and category exemptions,
    not a certification that reconstructed meshes have no intersections.
    """
    root = Path(scene_data_dir)
    report: dict[str, Any] = {
        "passed": False,
        "scope": "layout boxes; existing checker tolerances and category exemptions",
        "error_count": None,
        "object_clashes": None,
        "wall_clashes": None,
        "violations": [],
        "notes": [],
    }
    try:
        missing = [name for name in ("objects", "walls", "floor", "wall_holes")
                   if not (root / f"{name}.pkl").is_file()]
        if missing:
            raise ValueError(f"Missing scene data: {', '.join(missing)}")
        shell = adapter.shell_from_scene_data(root)
        if not shell.get("objects") or not shell.get("walls"):
            raise ValueError("Cannot validate a layout without both objects and walls")
        for oid, obj in shell['objects'].items():
            if (any(not math.isfinite(v) for v in obj['center'] + obj['size'])
                    or any(v <= 0 for v in obj['size']) or not math.isfinite(obj.get('yaw', 0))):
                raise ValueError(f"Invalid object geometry: {oid}")
        findings = check(shell)
        baseline = root / 'layout_baseline.json'
        if baseline.is_file():
            from .converge import fidelity_errors

            saved = json.loads(baseline.read_text())
            source = root / 'objects.extracted.pkl'
            source_key = hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None
            if saved.get('source_key') != source_key:
                raise ValueError('Extraction baseline changed; re-extract before validation')
            original = saved['shell']
            findings.extend(Violation(oid, 'fidelity', 'cumulative change exceeds scan limits')
                            for oid in fidelity_errors(shell, original))
        errors = [v for v in findings if v.severity == "error"]
        report.update(
            passed=not errors,
            error_count=len(errors),
            object_clashes=sum(v.kind == "object_clash" for v in errors),
            wall_clashes=sum(v.kind == "wall_clash" for v in errors),
            violations=[asdict(v) for v in errors],
            notes=[asdict(v) for v in findings if v.severity != "error"],
        )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def require_valid_layout(scan: str, result: dict[str, Any], *,
                         scene_data_dir: str | Path | None = None) -> dict[str, Any]:
    """Save a verdict and raise before downstream work if layout did not pass."""
    if scene_data_dir is None:
        from litereality_agent.pipeline.scene_init import paths

        scene_data_dir = paths.scene_data_dir(scan)
    root = Path(scene_data_dir)
    report = inspect_layout(root)
    if "error" in result or "skipped" in result or result.get("disabled"):
        report.update(passed=False, stage_error=result.get("error") or result.get("skipped")
                      or "layout repair was disabled")
    elif result.get("after", 0) > 0:
        report.update(passed=False, stage_error=f"repair reported {result['after']} remaining errors")
    result["validation"] = report
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "layout_validation.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["passed"]:
        reason = report.get("error") or report.get("stage_error") or (
            f"{report['error_count']} layout errors remain "
            f"({report['object_clashes']} object clashes, {report['wall_clashes']} wall clashes)"
        )
        raise LayoutValidationError(f"Layout validation failed for {scan}: {reason}. "
                                    f"See {destination}")
    print(f"  [layout] validation passed: zero layout errors ({destination})", flush=True)
    return report
