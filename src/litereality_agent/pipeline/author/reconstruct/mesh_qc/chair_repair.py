"""Chair-only geometry QC + repair.

TRELLIS is the chair path and the usual source of bad geometry (squat blobs, fused
floor slabs, floating fragments). This QCs **only** the chair clusters
(``ChairCluster*``) — never procedural objects or openings — and repairs the bad
ones by regenerating the hosted reference image (a fresh, cleaner image —
NOT just re-seeding TRELLIS on the same picture), then re-running TRELLIS, until the
chair passes geometry QA or attempts run out.

    repair_scan(scan)  ->  per-cluster QC + repair report

Run as a stage of object_init (``run.py --chair-qc`` / part of ``--full``) or
standalone. Needs hosted image generation and the TRELLIS runtime.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from litereality_agent import telemetry
from litereality_agent.pipeline.author.reconstruct import flow as reconstruct
from litereality_agent.pipeline.measure import paths as config
from litereality_agent.pipeline.measure.ingest.references import chair_clusters

from . import checks


def _clusters(scan: str) -> dict[str, dict]:
    p = config.chair_clusters_root() / scan / "chair_clusters.json"
    if not p.exists():
        return {}
    return {c["cluster_id"]: c for c in json.loads(p.read_text()).get("clusters", [])}


def _normalized_ref(scan: str, cid: str) -> Path:
    root = config.chair_clusters_root() / scan
    current = root / "references" / f"{cid}.png"
    legacy = root / "nano_banana_normalized_1024" / f"{cid}.png"
    return current if current.exists() or not legacy.exists() else legacy


def _qa(glb: Path, scan: str, cid: str) -> dict:
    return checks.qa_glb(glb, {"scan": scan, "object": cid, "category": "chair"})


def _trellis_one(
    scan: str, ref_png: Path, out_glb: Path, *, python: str | None, seed: int, decimation: int
) -> int:
    """Re-run TRELLIS on a single (new) reference image -> out_glb."""
    from litereality_agent.settings import load_settings

    settings = load_settings()
    if settings.modal_trellis_app or settings.trellis_python:
        from litereality_agent.models.registry import gen3d_from_settings

        service = gen3d_from_settings(settings)
        generated = service.reconstruct(
            [ref_png],
            out_dir=str(out_glb.parent),
            asset_id=out_glb.stem,
            seed=seed,
            simplify=0.95,
        )
        return 0 if generated and Path(generated).is_file() else 1

    if python is None:
        raise RuntimeError(
            "Chair repair requires MODAL_TRELLIS_APP or an explicit TRELLIS_PYTHON."
        )
    interp = reconstruct.resolve_python(python)
    cmd = [
        interp,
        str(reconstruct.LAUNCHER),
        "-i",
        str(ref_png),
        "-o",
        str(out_glb),
        "--seed",
        str(seed),
        "--decimation",
        str(decimation),
    ]
    return subprocess.run(cmd, cwd=str(config.REPO_ROOT)).returncode


def repair_scan(
    scan: str,
    *,
    max_attempts: int = 2,
    include_warn: bool = False,
    python: str | None = None,
    model: str | None = None,
    seed: int = 42,
    decimation: int = 50000,
    dry_run: bool = False,
) -> dict:
    """QC every chair of ``scan``; repair the failing ones via image-regen + TRELLIS."""
    config.set_scan(scan)
    telemetry.start(scan)
    recon = config.reconstruct_dir(scan)
    clusters = _clusters(scan)
    chair_root = config.chair_clusters_root()
    model = model or config.DEFAULT_IMAGE_MODEL

    chairs = sorted(recon.glob("ChairCluster*.glb"))
    results = []
    print(f"[chair-qc] {scan}: {len(chairs)} chair(s)", flush=True)
    for glb in chairs:
        cid = glb.stem
        qa = _qa(glb, scan, cid)
        bad = qa["status"] == "fail" or (include_warn and qa["status"] == "warn")
        rec = {
            "cluster": cid,
            "initial": qa["status"],
            "initial_flags": qa["flags"],
            "attempts": [],
        }
        print(f"  {cid}: {qa['status']} {qa['flags']}", flush=True)
        telemetry.event("chair_qc", cluster=cid, status=qa["status"], flags=qa["flags"])
        if not bad:
            rec["final"] = qa["status"]
            results.append(rec)
            continue
        cluster = clusters.get(cid)
        if cluster is None:
            rec["final"] = "no_cluster_meta"
            results.append(rec)
            continue
        if dry_run:
            rec["final"] = "would_repair"
            results.append(rec)
            continue

        shutil.copy2(glb, glb.with_suffix(".glb.orig"))  # back up the bad one
        passed = False
        for attempt in range(1, max_attempts + 1):
            # 1. Regenerate the reference (force → a fresh, different image).
            chair_clusters.generate_for_cluster(scan, cluster, chair_root, model, force=True)
            ref = _normalized_ref(scan, cid)
            # 2. re-run TRELLIS on the NEW image (vary the seed per attempt)
            rc = _trellis_one(
                scan, ref, glb, python=python, seed=seed + attempt, decimation=decimation
            )
            qa2 = (
                _qa(glb, scan, cid)
                if rc == 0 and glb.is_file()
                else {"status": "fail", "flags": ["trellis_failed"]}
            )
            rec["attempts"].append(
                {"attempt": attempt, "status": qa2["status"], "flags": qa2["flags"]}
            )
            print(f"    attempt {attempt}: {qa2['status']} {qa2['flags']}", flush=True)
            telemetry.event(
                "chair_qc_retry",
                cluster=cid,
                attempt=attempt,
                status=qa2["status"],
                flags=qa2["flags"],
            )
            if qa2["status"] != "fail":
                passed = True
                rec["final"] = qa2["status"]
                break
        if not passed:
            # none of the re-rolls passed — restore the backup (don't ship a worse one)
            shutil.copy2(glb.with_suffix(".glb.orig"), glb)
            rec["final"] = "still_failing_restored_original"
        results.append(rec)

    report = {
        "scan": scan,
        "chairs": results,
        "repaired": sum(
            1 for r in results if r["attempts"] and r["final"] != "still_failing_restored_original"
        ),
        "still_failing": sum(1 for r in results if r["final"] == "still_failing_restored_original"),
    }
    (recon / "chair_qc_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"[chair-qc] {scan}: {report['repaired']} repaired, {report['still_failing']} still failing "
        f"-> {recon / 'chair_qc_report.json'}",
        flush=True,
    )
    return report


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--scan", required=True)
    ap.add_argument("--attempts", type=int, default=2, help="re-roll attempts per failing chair.")
    ap.add_argument(
        "--include-warn", action="store_true", help="also repair 'warn' chairs, not just 'fail'."
    )
    ap.add_argument("--trellis-python", help="interpreter for the TRELLIS launcher.")
    ap.add_argument("--model", default=None, help="Image model override for regeneration.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--decimation", type=int, default=50000)
    ap.add_argument(
        "--dry-run", action="store_true", help="QC + list what would be repaired, don't re-roll."
    )
    args = ap.parse_args()
    scan = config.scan_name(args.scan) if hasattr(config, "scan_name") else args.scan
    repair_scan(
        scan,
        max_attempts=args.attempts,
        include_warn=args.include_warn,
        python=args.trellis_python,
        model=args.model,
        seed=args.seed,
        decimation=args.decimation,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
