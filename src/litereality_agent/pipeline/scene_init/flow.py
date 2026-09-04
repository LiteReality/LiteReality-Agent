#!/usr/bin/env python3
"""Detailed object workflow used by the ingest and reconstruct stage boundaries.

Pipeline per scan:
    extract_scene -> box_merge -> crop_objects -> object_references -> chair_clusters
                  -> opening_references -> (optional) reconstruct   # refs -> GLBs via TRELLIS/local

Run (from the repo root):
    python -m litereality_agent.pipeline.scene_init.flow --scan office-elliott
    python -m litereality_agent.pipeline.scene_init.flow --scan office-elliott --skip-image-generation
    python -m litereality_agent.pipeline.scene_init.flow --scan /abs/path/to/raw_scan --name my_room
    python -m litereality_agent.pipeline.scene_init.flow --scan office-elliott --reconstruct

A ``--scan`` value is either an absolute path to a raw scan folder, or the name of
a folder under ``<repo>/scans_uploaded/``. **Everything** for a scan — crops,
references, reconstructed GLBs, and the process trace — lands under one place:
``<repo>/run/<scan>/`` (override with ``--output-root``). The final summary
prints every path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from litereality_agent import console, telemetry
from litereality_agent.pipeline.scene_init import paths as config
from litereality_agent.pipeline.scene_init.ingest import merge_boxes
from litereality_agent.pipeline.scene_init import layout
from litereality_agent.pipeline.scene_init.ingest.crop import crop_objects
from litereality_agent.pipeline.scene_init.ingest.detect import bbox_polish
from litereality_agent.pipeline.scene_init.ingest.extract import extract_scene
from litereality_agent.pipeline.scene_init.ingest.references import (
    chair_clusters,
    object_references,
    opening_references,
)
from litereality_agent.pipeline.scene_init.reconstruct import flow as reconstruct
from litereality_agent.pipeline.scene_init.reconstruct.classify import classify_complexity
from litereality_agent.pipeline.scene_init.reconstruct.mesh_qc import chair_repair


def resolve_scan(value: str, name: str | None) -> tuple[Path, str]:
    """Return (raw_folder, scan_name) for a --scan value (path or scans_uploaded name)."""
    candidate = Path(value).expanduser()
    if candidate.exists() and candidate.is_dir():
        raw = candidate.resolve()
    else:
        raw = (config.SCANS_ROOT / value).resolve()
        if not raw.exists():
            raise SystemExit(f"Scan not found: {value} (looked at {candidate} and {raw})")
    scan_name = name or raw.name
    # sanitize to a safe directory key
    scan_name = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in scan_name)
    return raw, scan_name


def _references_on_disk(scan: str) -> bool:
    """True when a previous run left the reference manifests `--skip-references` would reuse.

    Falls back to regenerating rather than proceeding on a partial tree: a missing manifest means
    the routing and the reconstruction would disagree about which objects exist.
    """
    return (
        (config.object_refs_root() / scan / "object_references.json").is_file()
        and (config.chair_clusters_root() / scan / "chair_clusters.json").is_file()
    )


def _load_reference_json(path: Path, default: dict) -> dict:
    """Read a reference manifest, falling back to `default` when it is absent or unreadable.

    Openings legitimately have no manifest on a room without doors or windows, so a miss here is
    not an error — but a CORRUPT manifest silently becoming an empty object list would drop every
    object from the run, so say so rather than swallowing it.
    """
    if not path.is_file():
        return dict(default)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        print(f"[references] {path} is unreadable ({exc}); regenerating instead", flush=True)
        raise


def process_scan(raw: Path, scan: str, args: argparse.Namespace) -> dict:
    print(console.rule(f"scene_init · {scan}"))
    print(f"   {console.colour('dim')}capture: {raw}{console.colour('off')}", flush=True)
    # Divert each stage's narration to traces/stages/<stage>.log; the stage rows are the run.
    console.capture_stages(config.traces_dir(scan) / "stages")
    try:
        return _process_scan(scan, raw, args)
    finally:
        # A stage that raises never reaches its `done` event, so restore explicitly or the
        # traceback itself is written into the log we were hiding the noise in.
        console.restore_capture()


def _process_scan(scan: str, raw: Path, args: argparse.Namespace) -> dict:

    # All paths for this scan resolve under <output>/<scan>/; chdir into its
    # object_init work tree so the vendored preprocessing's relative input/ paths
    # land there too, and open a trace for the whole run.
    config.enter_work_root(scan)
    telemetry.start(scan)
    telemetry.event("scan_start", scan=scan, raw=str(raw))

    # 1. scene extraction (skip if already present unless forced)
    telemetry.stage("extract_scene", scan, "start")
    if args.skip_extract and config.scene_data_complete(scan):
        print("[extract] reusing existing scene_data")
        telemetry.stage("extract_scene", scan, "reused")
    elif config.scene_data_complete(scan) and not args.force_extract:
        print("[extract] scene_data present — reusing (use --force-extract to redo)")
        telemetry.stage("extract_scene", scan, "reused")
    else:
        extract_scene.extract(str(raw), scan)
        telemetry.stage("extract_scene", scan, "done")

    # 1b. box merge — fuse RoomPlan's interpenetrating counter runs (sink + base cabinet + wall
    # cabinet come back as separate same-yaw boxes) into ONE object. This has to sit HERE, between
    # extract writing objects.pkl and crop reading it, so the crop, the reference image, the
    # complexity route and the reconstruction all see the merged unit. Run after crop and you get
    # three interpenetrating assets; that was the state before this call existed. $LR_BOX_MERGE=0
    # opts out. See scene_init/object_init/merge_boxes.py.
    telemetry.stage("box_merge", scan, "start")
    merge_res = merge_boxes.merge_for_scan(scan)
    telemetry.stage("box_merge", scan, "done", merged=len(merge_res.get("merged", {})))

    # 1c. layout repair — settle the LAYOUT while it is still only boxes. Same window as the merge
    # above and for the same reason: after this point every mistake in a box gets paid for
    # repeatedly. A duplicate detection becomes two generated assets; a counter run recorded half a
    # metre too deep becomes an asset generated at the wrong extent and then squashed to fit; a box
    # that moves after its crop was cut leaves the reference image describing geometry that no
    # longer exists. Repairing here costs nothing and the correction propagates to the crop, the
    # references, the chair clustering and the reconstruction for free.
    # $LR_LAYOUT=0 opts out; $LR_LAYOUT_AGENT=1 lets it consult the reference photographs.
    telemetry.stage("layout", scan, "start")
    layout_res = layout.run_layout(scan)
    telemetry.stage("layout", scan, "done",
                    before=layout_res.get("before", 0), after=layout_res.get("after", 0),
                    moved=len(layout_res.get("moved", [])),
                    resized=len(layout_res.get("resized", [])),
                    dropped=len(layout_res.get("dropped", [])))

    # 2. per-object crops
    # Reuse mirrors extract_scene above: the stage inspects the disk and decides for itself, rather
    # than depending on the caller passing a flag. It only reused with an explicit --skip-crop
    # before, and the ingest stage never passes one, so every run re-cropped from scratch — ~11s a
    # run on a 28-object scan even when nothing had changed. `crops_current` compares the crops
    # against the scene data they were built from, so a box_merge that reshapes the object set
    # still invalidates them.
    telemetry.stage("crop_objects", scan, "start")
    if args.skip_crop:
        if not config.parsed_images_dir(scan).exists():
            raise SystemExit(f"--skip-crop but no crops at {config.parsed_images_dir(scan)}")
        telemetry.stage("crop_objects", scan, "reused")
    elif crop_objects.crops_current(scan, args.include_walls) and not args.force_crop:
        print("[crop] crops match the current scene data — reusing (use --force-crop to redo)")
        telemetry.stage("crop_objects", scan, "reused")
    else:
        crop_objects.crop(scan, include_walls=args.include_walls)
        cropped = config.parsed_images_dir(scan)
        telemetry.stage("crop_objects", scan, "done",
                      n_objects=sum(1 for d in cropped.iterdir() if d.is_dir())
                      if cropped.is_dir() else 0)

    # 2b. Optional GroundingDINO bbox refinement. RoomPlan's projected crop is the
    # default path; DINO is an explicit enhancement for environments that provide it.
    polish_res = None
    use_dino = args.use_dino and not args.skip_bbox_polish
    if use_dino:
        telemetry.stage("bbox_polish", scan, "start")
        polish_res = bbox_polish.polish(scan, force=args.force_bbox_polish)
        telemetry.stage("bbox_polish", scan,
                      "reused" if polish_res.get("reused") else "done",
                      refined=polish_res.get("refined_total", 0))

    only = set(args.only) if args.only else None

    # crops-only: stop after cropping + bbox refinement (object crops are already on disk;
    # also produce the DINO-refined OPENING crops, but NO nano references / object generation).
    if getattr(args, "crops_only", False):
        op = {"openings": []}
        if not args.skip_openings:
            telemetry.stage("opening_references", scan, "start")
            op = opening_references.generate_for_scan(
                scan,
                config.opening_refs_root(),
                skip_image_generation=True,
                crops_only=True,
                only=only,
                refine_dino=use_dino,
            )
            telemetry.stage("opening_references", scan, "done", n_openings=len(op["openings"]))
        print(
            "\n  [crops-only] object crops refined + opening crops written — "
            "stopping before references/generation.",
            flush=True,
        )
        telemetry.event("scan_done_crops_only", scan=scan)
        return {
            "scan": scan,
            "raw": str(raw),
            "crops_only": True,
            "bbox_polish": polish_res,
            "openings": op["openings"],
        }

    # 3-5. references. `--skip-references` reuses what is already on disk. The reconstruct stage
    # passes it: it re-enters this module only to reach classify/build, and re-deriving references
    # over crops that --skip-crop has already frozen is not merely wasted time. Chair grouping is
    # an LLM judgment, so it returns a DIFFERENT answer often enough to matter (2/3/3/2 clusters
    # across four runs of the same five chairs) and `cluster_and_generate` rewrites
    # chair_clusters.json unconditionally — so the later stage can overwrite the grouping that
    # ingest committed to and that the references and routing were built against.
    reuse = args.skip_references and _references_on_disk(scan)

    # 3. object references (non-chair objects)
    telemetry.stage("object_references", scan, "start")
    if reuse:
        obj_result = _load_reference_json(
            config.object_refs_root() / scan / "object_references.json",
            {"scan": scan, "objects": []},
        )
        telemetry.stage("object_references", scan, "reused")
    else:
        obj_result = object_references.generate_for_scan(
            scan,
            config.object_refs_root(),
            skip_image_generation=args.skip_image_generation,
            force_image_generation=args.force_image_generation,
            model=args.image_model,
            max_images=args.max_images,
            only=only,
            include_openings=args.include_openings,
        )
        telemetry.stage("object_references", scan, "done", n_objects=len(obj_result["objects"]))

    # 4. chair grouping + per-cluster reference
    telemetry.stage("chair_clusters", scan, "start")
    if reuse:
        chair_result = _load_reference_json(
            config.chair_clusters_root() / scan / "chair_clusters.json",
            {"chair_count": 0, "clusters": []},
        )
        telemetry.stage("chair_clusters", scan, "reused")
    else:
        chair_result = chair_clusters.cluster_and_generate(
            scan,
            config.chair_clusters_root(),
            skip_image_generation=args.skip_image_generation,
            force_image_generation=args.force_image_generation,
            model=args.image_model,
            use_dino=use_dino,
        )
        telemetry.stage(
            "chair_clusters",
            scan,
            "done",
            chairs=chair_result["chair_count"],
            clusters=len(chair_result["clusters"]),
        )

    # 5. door/window references (projected 3D opening boxes → clean hosted references)
    opening_result = {"openings": []}
    if reuse:
        telemetry.stage("opening_references", scan, "start")
        opening_result = _load_reference_json(
            config.opening_refs_root() / scan / "opening_references.json", {"openings": []}
        )
        telemetry.stage("opening_references", scan, "reused")
    elif not args.skip_openings:
        telemetry.stage("opening_references", scan, "start")
        opening_result = opening_references.generate_for_scan(
            scan,
            config.opening_refs_root(),
            skip_image_generation=args.skip_image_generation,
            force_image_generation=args.force_image_generation,
            model=args.image_model,
            refine_dino=use_dino,
        )
        telemetry.stage(
            "opening_references", scan, "done", n_openings=len(opening_result["openings"])
        )

    result = {
        "scan": scan,
        "raw": str(raw),
        "box_merge": merge_res,
        "objects": obj_result["objects"],
        "chairs": chair_result,
        "openings": opening_result["openings"],
        "paths": {
            "output": str(config.scan_output_dir(scan)),
            "crops": str(config.parsed_images_dir(scan)),
            "object_refs": str(config.object_refs_root() / scan),
            "chair_clusters": str(config.chair_clusters_root() / scan),
            "chair_grouping_json": str(config.chair_clusters_root() / scan / "chair_clusters.json"),
            "opening_refs": str(config.opening_refs_root() / scan),
            "reconstruct": str(config.reconstruct_dir(scan)),
            "traces": str(config.traces_dir(scan)),
        },
    }

    full = args.full
    # 6. complexity routing: VLM decides procedural (box geometry) vs trellis (organic) per object
    if args.classify or full:
        telemetry.stage("classify", scan, "start")
        route = classify_complexity.classify_for_scan(scan, model=args.classify_model)
        result["routing"] = {"procedural": route["procedural"], "trellis": route["trellis"]}
        telemetry.stage("classify", scan, "done", **result["routing"])

    # 7-9. RECONSTRUCTION — every object's geometry, built at once.
    #
    # Three builders, and they contend for different resources: TRELLIS is GPU/network, the
    # articulated agent is API + CPU Blender, openings likewise. Run serially they idle waiting
    # on each other; run together the phase takes about as long as its slowest branch. The one
    # real ordering constraint is chair_qc, which repairs what TRELLIS produced and so has to
    # follow it — that stays a sequential pair inside its own branch.
    #
    #     ┌ trellis ──▶ chair_qc ┐
    #     ├ procedural           ├──▶ all done ──▶ scene assembly
    #     └ openings             ┘
    #
    # $LR_SEQUENTIAL=1 restores the old one-at-a-time order (and the live subprocess output with
    # it) when a branch needs watching.
    _reconstruction_phase(scan, args, result, full)

    # Per-scan roll-up manifest: the one file a viewer needs to find everything.
    config.manifest_path(scan).write_text(json.dumps(result, indent=2), encoding="utf-8")
    (config.work_root() / "object_init_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    console.capture_stages(None)
    telemetry.event("scan_done", scan=scan)
    return result



def _expected_assets(scan: str) -> list[str]:
    """Every asset this run intends to build — the progress bar's denominator.

    Read from the routing manifest and the opening references rather than from any builder, so
    the count is right no matter which branch produces a given file.
    """
    names: list[str] = []
    routing = config.work_root() / "routing" / "routing_manifest.json"
    if routing.is_file():
        try:
            recs = json.loads(routing.read_text(encoding="utf-8")).get("scans", {}).get(scan, [])
            names += [r["name"] for r in recs if r.get("name")]
        except (OSError, ValueError):
            pass
    refs = config.opening_refs_root() / scan
    if refs.is_dir():
        names += [d.name for d in refs.iterdir() if d.is_dir()]
    return sorted(set(names))


# One worker per object rather than a flat 2 — a 6-object room spent most of the phase with
# idle capacity. Capped because each worker is a Blender process plus its own agent session:
# past this, Blender contends for CPU and the sessions contend for API rate limits, and the
# phase gets slower rather than faster. Override with --concurrency.
MAX_PARALLEL_OBJECTS = 8


def branch_concurrency(n_objects: int, explicit: int = 0) -> int:
    """Workers for a branch: `explicit` if the user set one, else one per object up to the cap."""
    if explicit and explicit > 0:
        return explicit
    return max(1, min(n_objects, MAX_PARALLEL_OBJECTS))


def _expected_by_branch(scan: str) -> dict[str, list[str]]:
    """Which objects each branch is responsible for.

    A shared denominator tells you the phase is 4/6 but not which branch is behind. Routing says
    trellis-vs-procedural per object; everything under opening_refs belongs to the openings pass.
    """
    branches: dict[str, list[str]] = {"trellis": [], "procedural": [], "openings": []}
    routing = config.work_root() / "routing" / "routing_manifest.json"
    if routing.is_file():
        try:
            for rec in json.loads(routing.read_text(encoding="utf-8")).get("scans", {}).get(scan, []):
                name, route = rec.get("name"), rec.get("route")
                if name:
                    branches["trellis" if route == "trellis" else "procedural"].append(name)
        except (OSError, ValueError):
            pass
    refs = config.opening_refs_root() / scan
    if refs.is_dir():
        branches["openings"] = sorted(d.name for d in refs.iterdir() if d.is_dir())
    return {k: sorted(set(v)) for k, v in branches.items()}


def _asset_is_built(recon: Path, name: str) -> bool:
    if (recon / f"{name}.glb").is_file():
        return True
    d = recon / name
    return any(d.glob("*.glb")) and (d / "object.py").is_file() and (d / "object.md").is_file()


def _built_count(scan: str, expected: list[str]) -> int:
    """How many of `expected` are actually FINISHED.

    Not just "a .glb exists". The articulated agent writes the GLB and its previews before
    `object.py` / `object.md`, so an interrupted run leaves a directory that looks complete: the
    count read 6/6 while two objects were being regenerated from scratch, and the phase appeared
    to hang for thirteen minutes on work the progress bar said was done. An articulated object
    counts when its editable source is there too — which is also exactly what the builder's own
    --skip-existing check requires, so the bar and the builder now agree.
    """
    recon = config.reconstruct_dir(scan)
    if not recon.is_dir():
        return 0
    return sum(_asset_is_built(recon, name) for name in expected)


def _missing_assets(scan: str, expected: list[str]) -> list[str]:
    """Expected assets that do not have a complete reconstruction on disk."""
    recon = config.reconstruct_dir(scan)
    return [name for name in expected if not _asset_is_built(recon, name)]


def _object_phase(directory: Path) -> str:
    """What a half-finished object directory says it is actually doing.

    Every builder writes the same sequence into `<name>/`: a build script, the GLB that script
    produces, preview renders of it, and finally the editable `object.py`/`object.md` once the
    probe and VLM gates pass. So the newest milestone present names the phase.

    This replaces a per-branch caption. "building frame" sat next to four openings for seven
    straight minutes regardless of what any of them were doing, which is worse than no caption —
    it looks like live status and is not.
    """
    if any(directory.glob("*.glb")):
        return "checking result" if (directory / "previews").is_dir() else "rendering previews"
    if any(directory.glob("build_*.py")):
        return "running the build"
    if (directory / "textures").is_dir():
        return "preparing textures"
    return "starting up"


def _branch_split(scan: str, names: list[str]) -> tuple[list[str], list[str], list[str], dict]:
    """Split a branch's objects into finished, building, and waiting — all read from disk.

    Nothing reports in: every builder is a subprocess in another interpreter, and one of them is
    an agent that rewrites its own output when a QC gate fails. What IS observable is the work
    tree. The procedural and openings passes create `<name>/` and fill it, so a directory without
    the complete set of artifacts is an object being built right now.

    TRELLIS is the honest gap. It writes `<name>.glb` in a single step and leaves nothing partial
    behind, so its objects appear here as waiting and then as finished, with no building phase for
    the board to show. Reporting them as building on the strength of "the branch is running" would
    be a guess at WHICH objects, and a wrong name on screen is worse than no name.
    """
    recon = config.reconstruct_dir(scan)
    built, building, waiting, phase = [], [], [], {}
    for name in names:
        if _asset_is_built(recon, name):
            built.append(name)
        elif (recon / name).is_dir():
            building.append(name)
            phase[name] = _object_phase(recon / name)
        else:
            waiting.append(name)
    return built, building, waiting, phase


def _reconstruction_phase(scan: str, args: argparse.Namespace, result: dict, full: bool) -> None:
    """Build every object, in parallel branches, behind one progress bar."""
    from concurrent.futures import ThreadPoolExecutor


    # The parallel phase routes output per BRANCH (ThreadTee); per-STAGE capture swaps a
    # process-global and would cross the streams with three workers running.
    console.capture_stages(None)

    def trellis_branch():
        if args.reconstruct or full:
            telemetry.stage("reconstruct", scan, "start")
            try:
                recon = reconstruct.run_for_scan(
                    scan, python=args.trellis_python, seed=args.seed,
                    decimation=args.decimation, skip_existing=not args.force_reconstruct,
                    dry_run=args.dry_run_reconstruct,
                )
                result["reconstruct"] = recon
                telemetry.stage("reconstruct", scan, "done", n_glb=recon.get("generated", 0))
            except Exception as exc:  # TRELLIS is non-fatal: the other branches still land
                result["reconstruct"] = {"error": str(exc), "generated": 0}
                telemetry.stage("reconstruct", scan, "error", error=str(exc))
        if args.chair_qc or full:  # must follow TRELLIS — it repairs what TRELLIS made
            telemetry.stage("chair_qc", scan, "start")
            qc = chair_repair.repair_scan(
                scan, max_attempts=args.qc_attempts, include_warn=args.qc_include_warn,
                python=args.trellis_python, model=args.image_model, seed=args.seed,
                decimation=args.decimation, dry_run=args.dry_run_reconstruct,
            )
            result["chair_qc"] = qc
            telemetry.stage("chair_qc", scan, "done", repaired=qc.get("repaired", 0),
                          still_failing=qc.get("still_failing", 0))

    def agent_branch(stage_name: str, openings: bool, enabled_flag: bool, workers: int):
        if not (enabled_flag or full):
            return
        telemetry.stage(stage_name, scan, "start")
        out = reconstruct.run_procedural_for_scan(
            scan, openings=openings, python=args.procedural_python,
            blender_path=args.blender_path, concurrency=workers,
            retries=args.retries, budget=args.budget, model=args.agent_model,
            skip_existing=not args.force_reconstruct, dry_run=args.dry_run_reconstruct,
        )
        result[stage_name] = out
        telemetry.stage(stage_name, scan, "done", n_glb=out.get("generated", 0))

    per_branch = _expected_by_branch(scan)
    workers = {name: branch_concurrency(len(per_branch.get(name, [])), args.concurrency)
               for name in ("procedural", "openings")}
    branches = [
        ("trellis", trellis_branch),  # The configured runtime handles its own batch.
        ("procedural", lambda: agent_branch("procedural", False, args.procedural,
                                            workers["procedural"])),
        ("openings", lambda: agent_branch("build_openings", True, args.build_openings,
                                          workers["openings"])),
    ]

    if os.environ.get("LR_SEQUENTIAL", "").strip() in ("1", "true", "yes"):
        for _, fn in branches:
            fn()
        return

    logs = config.traces_dir(scan) / "build"
    logs.mkdir(parents=True, exist_ok=True)
    expected = _expected_assets(scan)
    plan = " · ".join(f"{n} ×{workers.get(n, 1)}" for n in ("procedural", "openings")
                      if per_branch.get(n))
    print(console.rule(f"reconstruction · {len(expected)} objects · trellis + {plan}"))
    print(f"   {console.colour('dim')}per-branch output → {console.short(logs)}/"
          f"{console.colour('off')}", flush=True)

    state: dict[str, dict] = {
        name: {"done": 0, "total": len(per_branch.get(name, [])), "started": None,
               "finished": None, "result": "", "built": [], "active": [], "activity": {},
               "queued": list(per_branch.get(name, []))}
        for name, _ in branches
    }

    def poll() -> None:
        """Re-read what is on disk into `state`; the board renders, it does not look."""
        for branch, names in per_branch.items():
            if branch not in state:
                continue
            built, building, waiting, phase = _branch_split(scan, names)
            state[branch].update(done=len(built), built=built, active=building,
                                 queued=waiting, activity=phase)

    # The live viewer decides whether anything is alive from the age of the newest trace event, and
    # the only build events reaching the trace were the per-object `agent_step` lines a builder
    # writes when it FINISHES. One object takes 9-20 minutes; on Elliott-Studio the gaps between
    # consecutive lines ran to 19 and 38 minutes, far past the viewer's 120s liveness threshold, so
    # a phase with sixteen agents working reported "quiet" for nearly all of it. `poll()` already
    # re-reads this state every second for the board — put it in the trace too, so liveness follows
    # the work rather than the moments it happens to complete.
    BEAT_S = 20.0
    beat = {"at": 0.0, "counts": None}

    def heartbeat() -> None:
        """Emit build progress on change, and at least every BEAT_S while the phase runs."""
        counts = {name: state[name]["done"] for name, _ in branches}
        now = time.time()
        if counts == beat["counts"] and now - beat["at"] < BEAT_S:
            return
        telemetry.event(
            "build_progress",
            scan=scan,
            built=sum(counts.values()),
            total=len(expected),
            branches={n: f"{counts[n]}/{state[n]['total']}" for n, _ in branches},
            # Which objects are mid-build, so a long phase says WHAT it is working on. TRELLIS
            # contributes none by design — see _branch_split.
            active=sorted(name for n, _ in branches for name in state[n]["active"]),
        )
        beat["at"], beat["counts"] = now, counts

    board = console.BranchBoard([n for n, _ in branches], len(expected), workers=workers)
    tee = console.ThreadTee(sys.stdout)
    real_stdout, sys.stdout = sys.stdout, tee

    def run(name, fn):
        # Both the branch's own prints and its subprocess output go to its log: three streams
        # interleaved on the terminal are unreadable and fight the board for the cursor. The
        # board itself renders from the main thread, which stays bound to the terminal.
        handle = open(logs / f"{name}.log", "a", encoding="utf-8", errors="replace")
        tee.bind(handle)
        reconstruct.log_to(logs / f"{name}.log")
        state[name]["started"] = time.time()
        try:
            fn()
            state[name]["result"] = f"{state[name]['done']}/{state[name]['total']} built"
        except Exception as exc:  # noqa: BLE001 — one branch failing must not sink the others
            telemetry.event("branch_error", scan=scan, branch=name, error=str(exc))
            state[name]["result"] = f"FAILED: {exc}"
            print(f"[{name}] BRANCH FAILED: {exc}", flush=True)
        finally:
            state[name]["finished"] = time.time()
            reconstruct.log_to(None)
            tee.unbind()
            handle.close()

    try:
        with ThreadPoolExecutor(max_workers=len(branches)) as pool:
            futures = [pool.submit(run, name, fn) for name, fn in branches]
            while not all(f.done() for f in futures):
                poll()
                heartbeat()
                board.render(state)
                time.sleep(1.0)
            poll()
            heartbeat()
            board.finish(state)
            for f in futures:
                f.result()
    finally:
        sys.stdout = real_stdout



def summarize(results: list[dict]) -> None:
    print("\n" + "=" * 64)
    print("object_init complete")
    print("=" * 64)
    for r in results:
        if r.get("crops_only"):
            openings = r.get("openings", [])
            refined = (r.get("bbox_polish") or {}).get("refined_total", 0)
            op_refined = sum(o.get("n_dino_refined", 0) for o in openings)
            print(f"\n{r['scan']}:  [crops-only]")
            print(f"  object crops:  {refined} frame box(es) DINO-refined")
            print(
                f"  opening crops: {len(openings)} doors/windows ({op_refined} frame box(es) DINO-refined)"
            )
            print(
                f"  crops under:   {config.parsed_images_dir(r['scan'])}  +  {config.opening_refs_root() / r['scan']}"
            )
            continue
        n_obj = len(r["objects"])
        ok_obj = sum(1 for o in r["objects"] if o["status"] == "ok")
        n_chairs = r["chairs"]["chair_count"]
        n_clusters = len(r["chairs"]["clusters"])
        print(f"\n{r['scan']}:")
        print(f"  output dir:    {r['paths']['output']}")
        print(f"  objects:       {n_obj} ({ok_obj} references ok)")
        print(f"  chairs:        {n_chairs} chairs -> {n_clusters} groups")
        for c in r["chairs"]["clusters"]:
            print(
                f"      {c['cluster_id']}: {', '.join(c['members'])}  (rep {c['representative']})"
            )
        openings = r.get("openings", [])
        ok_op = sum(1 for o in openings if o["status"] == "ok")
        print(f"  openings:      {len(openings)} doors/windows ({ok_op} references ok)")
        if "routing" in r:
            print(
                f"  routing:       {r['routing']['procedural']} procedural, {r['routing']['trellis']} trellis"
            )
        if "reconstruct" in r:
            print(
                f"  trellis:       {r['reconstruct'].get('generated', 0)} GLB(s) -> {r['paths']['reconstruct']}"
            )
        if "procedural" in r:
            print(f"  procedural:    {r['procedural'].get('generated', 0)} articulated GLB(s)")
        if "build_openings" in r:
            print(f"  built openings:{r['build_openings'].get('generated', 0)} door/window GLB(s)")
        print(f"  manifest:      {config.manifest_path(r['scan'])}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--scan",
        nargs="+",
        required=True,
        help="Raw scan folder path(s) or scans_uploaded name(s).",
    )
    parser.add_argument("--name", help="Override scan name (only valid with a single --scan).")
    parser.add_argument(
        "--output-root", type=Path, help="Where all per-scan output goes (default <repo>/run)."
    )
    parser.add_argument(
        "--skip-extract", action="store_true", help="Reuse existing scene_data, never extract."
    )
    parser.add_argument(
        "--force-extract", action="store_true", help="Re-run scene extraction even if present."
    )
    parser.add_argument(
        "--skip-crop", action="store_true", help="Reuse existing crops, never crop."
    )
    parser.add_argument(
        "--force-crop",
        action="store_true",
        help="Re-crop even when the crops on disk match the current scene data.",
    )
    parser.add_argument(
        "--skip-references",
        action="store_true",
        help="Reuse the references already on disk instead of regenerating them. The reconstruct "
        "stage passes this: crops are frozen by then, so re-deriving them re-runs the chair "
        "judge and can reshuffle the clustering ingest already committed to.",
    )
    parser.add_argument(
        "--use-dino",
        action="store_true",
        help="Opt in to GroundingDINO crop refinement and DINOv2 chair embeddings.",
    )
    parser.add_argument(
        "--skip-bbox-polish",
        action="store_true",
        help="Deprecated compatibility flag; DINO refinement is skipped by default.",
    )
    parser.add_argument(
        "--crops-only",
        action="store_true",
        help="Stop after cropping (plus DINO refinement when --use-dino is set). "
        "No nano references, no object generation.",
    )
    parser.add_argument(
        "--skip-image-generation", action="store_true", help="Skip hosted image calls; write placeholders."
    )
    parser.add_argument(
        "--force-image-generation", action="store_true", help="Regenerate references even if they exist."
    )
    parser.add_argument("--image-model", default=config.DEFAULT_IMAGE_MODEL)
    parser.add_argument(
        "--max-images", type=int, default=5, help="Max evidence crops per object sheet."
    )
    parser.add_argument("--only", nargs="*", help="Only generate references for these object ids.")
    parser.add_argument(
        "--include-walls",
        action="store_true",
        help="Also crop per-wall folders (material evidence).",
    )
    parser.add_argument(
        "--include-openings",
        action="store_true",
        help="Also reference Wall*_Door/Window objects in the object pass.",
    )
    parser.add_argument(
        "--skip-openings",
        action="store_true",
        help="Skip the door/window (projected-box) reference stage.",
    )
    # generation stages (after references)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the whole pipeline: classify -> trellis -> procedural -> openings.",
    )
    parser.add_argument(
        "--classify", action="store_true", help="VLM-route each object procedural vs trellis."
    )
    parser.add_argument("--classify-model", default=None, help="Model for the complexity router.")
    # neural reconstruction (trellis-route -> GLBs via integrations/trellis/local)
    parser.add_argument(
        "--reconstruct", action="store_true", help="Build trellis-route GLBs via TRELLIS.2."
    )
    parser.add_argument(
        "--trellis-python",
        help="Interpreter for the TRELLIS launcher ($LITEREALITY_TRELLIS_PYTHON / .venv).",
    )
    parser.add_argument("--seed", type=int, default=42, help="TRELLIS seed.")
    parser.add_argument(
        "--decimation", type=int, default=50000, help="TRELLIS mesh decimation target."
    )
    # chair QC (chairs only): geometry-check + repair by regenerating the image + re-TRELLIS
    parser.add_argument(
        "--chair-qc",
        action="store_true",
        help="QC TRELLIS chairs; repair bad ones (image regen + re-TRELLIS).",
    )
    parser.add_argument(
        "--qc-attempts", type=int, default=2, help="Re-roll attempts per failing chair."
    )
    parser.add_argument(
        "--qc-include-warn", action="store_true", help="Repair 'warn' chairs too, not just 'fail'."
    )
    # articulated build (procedural-route + openings -> articulated GLBs via the agent)
    parser.add_argument(
        "--procedural",
        action="store_true",
        help="Build procedural-route objects via the articulated agent.",
    )
    parser.add_argument(
        "--build-openings",
        action="store_true",
        help="Build doors/windows via the articulated agent.",
    )
    parser.add_argument(
        "--procedural-python",
        help="Interpreter with claude_agent_sdk ($LITEREALITY_PROCEDURAL_PYTHON).",
    )
    parser.add_argument(
        "--blender-path",
        help="Dir with the blender binary to prepend to PATH ($LITEREALITY_BLENDER).",
    )
    parser.add_argument(
        "--agent-model", default=None, help="Model override for the articulated agent."
    )
    parser.add_argument(
        "--force-bbox-polish", action="store_true",
        help="Redo the DINO bbox refinement even if a previous run is recorded.",
    )
    parser.add_argument(
        "--concurrency", type=int, default=0,
        help=f"objects built at once per branch (0 = auto: one per object, capped at "
             f"{MAX_PARALLEL_OBJECTS})."
    )
    parser.add_argument(
        "--retries", type=int, default=1, help="Articulated-agent retries per object."
    )
    parser.add_argument(
        "--budget", type=float, default=5.0, help="Articulated-agent per-object USD budget."
    )
    # shared
    parser.add_argument(
        "--force-reconstruct", action="store_true", help="Re-generate GLBs even if they exist."
    )
    parser.add_argument(
        "--dry-run-reconstruct",
        action="store_true",
        help="Show what generation would run, don't launch.",
    )
    args = parser.parse_args(argv)

    if args.name and len(args.scan) > 1:
        raise SystemExit("--name only works with a single --scan.")
    if args.output_root:
        import os

        os.environ["LITEREALITY_OUTPUT"] = str(args.output_root.expanduser().resolve())

    print(f"output root: {config.output_root()}")

    started = time.time()
    results = []
    for value in args.scan:
        raw, scan = resolve_scan(value, args.name)
        results.append(process_scan(raw, scan, args))

    summarize(results)
    print(f"\nelapsed {time.time() - started:.1f}s")
    successful_reconstruction_statuses = {"ok", "no_references", "dry_run"}
    reconstruction_failed = any(
        "reconstruct" in result
        and (
            result["reconstruct"].get("error")
            or result["reconstruct"].get("status") not in successful_reconstruction_statuses
        )
        for result in results
    )
    builds_requested = args.full or args.reconstruct or args.procedural or args.build_openings
    incomplete = {}
    if builds_requested and not args.dry_run_reconstruct:
        incomplete = {
            result["scan"]: _missing_assets(result["scan"], _expected_assets(result["scan"]))
            for result in results
        }
        incomplete = {scan: names for scan, names in incomplete.items() if names}
        if incomplete:
            for scan, names in incomplete.items():
                print(f"reconstruction incomplete for {scan}: {', '.join(names)}")
    return int(reconstruction_failed or bool(incomplete))


if __name__ == "__main__":
    sys.exit(main())
