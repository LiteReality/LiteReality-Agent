"""Compile physics for every procedural object a scan produced.

Called at the end of the procedural reconstruct stage, where the objects have just been built and
the recipe that built them is still the thing anyone would go and fix. A failure here never fails
the stage — an object without a physics sidecar is exactly as usable as it was before this
existed — but it is recorded per object so the gap is visible rather than silent.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

from .checks import check, write_report
from .properties import build_model, write_model
from .urdf import to_urdf


def compile_object(glb: Path, *, run_checks: bool = True, decompose: bool = True) -> dict:
    """One GLB -> physics json + URDF + MJCF (+ the gate). Never raises."""
    glb = Path(glb)
    out = glb.parent / "sim"
    try:
        model = build_model(glb, out, decompose=decompose)
        write_model(model, out)
        to_urdf(model, out)
        row = {"object": model.name, "status": "ok", "links": len(model.links),
               "joints": len(model.joints), "mass_kg": model.total_mass,
               "colliders": sum(len(l.colliders) for l in model.links)}
        if run_checks:
            report = check(model, out)
            write_report(report, out)
            row["check"] = "pass" if report.passed else "fail"
            row["findings"] = report.findings
        return row
    except Exception as exc:                                 # noqa: BLE001 — never fail the stage
        return {"object": glb.stem, "status": "error", "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=3)}


def compile_directory(recon_dir: Path, *, run_checks: bool = True) -> dict:
    """Every procedural object under `reconstruct/` — the ones that have their own directory,
    which is exactly the set that was built from a recipe and can carry articulation."""
    recon_dir = Path(recon_dir)
    rows = [compile_object(glb, run_checks=run_checks)
            for glb in sorted(recon_dir.glob("*/*.glb"))]
    summary = {"objects": len(rows),
               "ok": sum(r["status"] == "ok" for r in rows),
               "passed": sum(r.get("check") == "pass" for r in rows),
               "failed": sum(r.get("check") == "fail" for r in rows),
               "errors": sum(r["status"] == "error" for r in rows),
               "rows": rows}
    (recon_dir / "sim_report.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
