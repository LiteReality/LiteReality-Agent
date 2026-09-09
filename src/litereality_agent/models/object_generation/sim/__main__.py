"""Compile an object's physics, export it, and gate it.

    python -m litereality_agent.models.object_generation.sim build <object.glb> [--out DIR]
    python -m litereality_agent.models.object_generation.sim check <object.glb> [--out DIR]

`build` writes <name>.physics.json, <name>.urdf, <name>.xml and the meshes they reference.
`check` does that and then runs the drop / release / tilt gate. Exit 0 = pass, 1 = fail.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .checks import check, write_report
from .properties import build_model, write_model
from .urdf import to_urdf


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sim", description=__doc__)
    ap.add_argument("action", choices=("build", "check"))
    ap.add_argument("glb", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="default: <glb dir>/sim")
    ap.add_argument("--category", default="", help="override the category used for density")
    ap.add_argument("--no-decompose", action="store_true",
                    help="convex hulls only — faster, and wrong for anything you put things in")
    args = ap.parse_args(argv)

    out = args.out or args.glb.parent / "sim"
    out.mkdir(parents=True, exist_ok=True)
    model = build_model(args.glb, out, category=args.category, decompose=not args.no_decompose)
    write_model(model, out)
    to_urdf(model, out)

    print(f"{model.name}: {len(model.links)} links, {len(model.joints)} joints, "
          f"{model.total_mass:.2f} kg, "
          f"{sum(len(link.colliders) for link in model.links)} colliders -> {out}")
    for note in model.notes:
        print(f"  note: {note}")
    if args.action == "build":
        return 0

    report = check(model, out)
    write_report(report, out)
    print(json.dumps(report.to_json()["scenarios"], indent=2))
    for f in report.findings:
        print(f"  {f['severity'].upper():4s} {f['check']}: {f['detail']}")
    print("PASS" if report.passed else "FAIL")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
