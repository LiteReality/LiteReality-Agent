"""Inspect captures through layout only, with an isolated output tree and complete audit files.

Run: uv run python scripts/capture/layout_batch.py --scans-dir data/toyuyan
This diagnostic wrapper calls the production extraction, merge and layout implementations.
It prepares photographic evidence before layout; it never starts reconstruction or authoring.
"""

from __future__ import annotations

import argparse
import atexit
import dataclasses
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]


def encode(value):
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=encode), encoding="utf-8")


class Audit:
    def __init__(self, root, *, started=None):
        self.root = root
        self.sequence = 0
        self.calls = 0
        self.started = time.perf_counter() if started is None else started
        self.timings = {"photos_seconds": 0.0, "agent_seconds": 0.0}

    def elapsed(self):
        return time.perf_counter() - self.started

    @contextmanager
    def phase(self, name):
        started = time.perf_counter()
        self.event(f"{name}_start")
        try:
            yield
        finally:
            duration = time.perf_counter() - started
            key = f"{name}_seconds"
            self.timings[key] = self.timings.get(key, 0.0) + duration
            self.event(f"{name}_end", duration_seconds=duration)

    def event(self, kind, **data):
        self.sequence += 1
        elapsed = self.elapsed()
        row = {"step": self.sequence, "elapsed_seconds": elapsed, "event": kind, **data}
        with (self.root / "steps.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=encode) + "\n")
        if kind not in {"candidate", "score"}:
            duration = f" ({data['duration_seconds']:.2f}s)" if "duration_seconds" in data else ""
            print(f"[layout step {self.sequence}, +{elapsed:.2f}s] {kind}{duration}", flush=True)

    def instrument(self, stack):
        """Observe production functions without replacing their geometry or decision rules."""
        repair = importlib.import_module("litereality_agent.pipeline.scene_init.layout.repair")
        agent = importlib.import_module("litereality_agent.pipeline.scene_init.layout.agent")
        original_candidates = repair._candidates

        def candidates(shell, violation, held, family, dups):
            result = original_candidates(shell, violation, held, family, dups)
            for candidate in result:
                apply = candidate.apply

                def observed(trial, apply=apply, candidate=candidate):
                    apply(trial)
                    self.event("candidate", violation=violation, action=candidate.kind,
                               detail=candidate.detail,
                               objects=trial.get("objects"), openings=trial.get("openings"))

                candidate.apply = observed
            return result

        # Every score is evaluated against the real search baseline by the original solver.
        original_score = repair.score

        def score(*args, **kwargs):
            result = original_score(*args, **kwargs)
            self.event("score", value=result)
            return result

        original_repair = repair.repair

        def solve(shell, **kwargs):
            self.event("deterministic_start", shell=shell)
            result = original_repair(shell, **kwargs)
            self.event("deterministic_finish", shell=result[0], moves=result[1], actions=result[2])
            return result

        original_gate = agent.apply_proposals

        def gate(shell, proposals):
            result = original_gate(shell, proposals)
            self.event("proposal_gate", proposals=proposals, decisions=result[1], shell=result[0])
            return result

        original_propose = agent.propose
        original_run = subprocess.run

        def propose(shell, violation, batch_dir, *args, **kwargs):
            # Never ask for photographic judgement when this object has no photograph.
            if not list((batch_dir / "references" / violation.object).glob("rank*.jpg")):
                self.event("agent_skipped", object=violation.object, reason="no reference photos")
                return None
            self.calls += 1
            prefix = self.root / f"agent_{self.calls:03d}"

            def invoke(command, **options):
                prefix.with_suffix(".prompt.txt").write_text(command[2], encoding="utf-8")
                command = list(command)
                command[command.index("--output-format") + 1] = "stream-json"
                command.append("--verbose")
                self.event("agent_start", object=violation.object, prompt=str(prefix))
                try:
                    with self.phase("agent"):
                        done = original_run(command, **options)
                except subprocess.TimeoutExpired:
                    self.event("agent_timeout", object=violation.object)
                    raise
                prefix.with_suffix(".events.jsonl").write_text(done.stdout or "", encoding="utf-8")
                prefix.with_suffix(".stderr.txt").write_text(done.stderr or "", encoding="utf-8")
                answer = None
                for line in (done.stdout or "").splitlines():
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if event.get("type") == "result":
                        if event.get("is_error"):
                            raise RuntimeError(f"Layout agent failed: {event.get('result', event)}")
                        answer = event.get("result", "")
                if done.returncode or answer is None:
                    raise RuntimeError(f"Layout agent failed; inspect {prefix}.stderr.txt")
                return subprocess.CompletedProcess(command, done.returncode, answer, done.stderr)

            with patch.object(agent.subprocess, "run", invoke):
                result = original_propose(shell, violation, batch_dir, *args, **kwargs)
            self.event("agent_proposal", proposal=result)
            return result

        for module, name, replacement in (
            (repair, "_candidates", candidates), (repair, "score", score),
            (repair, "repair", solve), (agent, "apply_proposals", gate),
            (agent, "propose", propose),
        ):
            stack.enter_context(patch.object(module, name, replacement))


def refine_crops(scan, config, audit):
    """Keep comparable crop snapshots and require the requested detector to run."""
    from litereality_agent.pipeline.scene_init.ingest.detect import bbox_polish

    parsed = config.parsed_images_dir(scan)
    # Stitched sheets were built before refinement and are not the evidence consumed here.
    ignore = shutil.ignore_patterns("stitched_image.jpg")
    shutil.copytree(parsed, audit.root / "crops_before_dino", ignore=ignore)
    with audit.phase("dino"):
        result = bbox_polish.polish(scan, force=True)
    write_json(audit.root / "dino_refinement.json", result)
    audit.event("dino_result", result=result)
    if result.get("skipped"):
        raise RuntimeError(f"GroundingDINO requested but skipped: {result['skipped']}. "
                           "Configure the Modal or local DINO backend before retrying.")
    # Do not leave pre-refinement montages that still show rejected frames in this output tree.
    for sheet in parsed.glob("*/stitched_image.jpg"):
        sheet.unlink()
    shutil.copytree(parsed, audit.root / "crops_after_dino", ignore=ignore)
    return result


def reference_sheet(image, rectangle, info, crop_path, object_id):
    """Show scanned geometry and the actual selected crop together to the layout agent."""
    from PIL import Image, ImageDraw

    draw = ImageDraw.Draw(image)
    draw.rectangle(rectangle, outline="red", width=5)
    if info.get("dino_refined"):
        draw.rectangle(info["resized_bbox"], outline="lime", width=5)
    image = image.rotate(-90, expand=True)
    with Image.open(crop_path) as source:
        crop = source.convert("RGB").rotate(-90, expand=True)
    crop.thumbnail((max(1, image.width // 2), image.height))
    sheet = Image.new("RGB", (image.width + crop.width + 20, image.height + 50), "white")
    sheet.paste(image, (0, 50))
    sheet.paste(crop, (image.width + 20, 50))
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 8), f"{object_id} | Red: scanned 3D box projection", fill="black")
    label = "Green: GroundingDINO box; right: refined crop" if info.get("dino_refined") else "Right: projected crop (no accepted DINO refinement)"
    draw.text((10, 26), label, fill="black")
    return sheet


def prepare_photos(scan, config, audit, *, use_dino=False):
    """Rank scan frames, then draw each scanned box's projected rectangle on full frames."""
    from PIL import Image, ImageDraw

    from litereality_agent.pipeline.scene_init.ingest.crop.crop_objects import crop
    from litereality_agent.pipeline.scene_init.ingest.preprocessing.vendor.litereality.object_image_extraction import (
        FrameStore,
        parse_objects,
        project_bbox_2d,
    )
    from litereality_agent.pipeline.scene_init.layout.adapter import load_objects

    crop(scan)
    refinement = refine_crops(scan, config, audit) if use_dino else None
    scene = config.scene_data_dir(scan)
    boxes, _ = parse_objects(load_objects(scene))
    frames = FrameStore(scan)
    count = 0
    references = []
    for object_dir in sorted(config.parsed_images_dir(scan).iterdir()):
        if object_dir.name not in boxes:
            continue
        metadata = sorted((object_dir / "camera").glob("*_ranking_*.json"),
                          key=lambda p: int(p.stem.rsplit("_", 1)[1]))[:2]
        for item in metadata:
            info = json.loads(item.read_text())
            frame = int(re.search(r"frame_(\d+)", item.name)[1])
            bbox = project_bbox_2d(scan, boxes[object_dir.name], frame, frames=frames)
            if bbox is None:
                continue
            with Image.open(info["original_image_path"]) as source:
                image = source.convert("RGB")
            x0, y0, x1, y1 = bbox
            rectangle = (x0 * image.width / 256, y0 * image.height / 192,
                         x1 * image.width / 256, y1 * image.height / 192)
            crop_path = object_dir / f"{item.stem}.jpg"
            if use_dino:
                image = reference_sheet(image, rectangle, info, crop_path, object_dir.name)
            else:
                draw = ImageDraw.Draw(image)
                draw.rectangle(rectangle, outline="red", width=5)
                draw.text(rectangle[:2], object_dir.name, fill="red")
                image = image.rotate(-90, expand=True)
            destination = scene / "references" / object_dir.name
            destination.mkdir(parents=True, exist_ok=True)
            reference = destination / f"rank{info['ranking']}.jpg"
            image.save(reference)
            references.append({"object": object_dir.name, "reference": str(reference),
                               "crop": str(crop_path), "metadata": info})
            count += 1
    write_json(audit.root / "references.json", references)
    audit.event("photos_ready", images=count, directory=str(scene / "references"))
    return refinement


class Tee:
    def __init__(self, terminal, file):
        self.terminal, self.file = terminal, file

    def write(self, text):
        self.terminal.write(text)
        self.file.write(text)
        self.file.flush()
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.file.flush()


def run_scene(raw, output, use_agent, *, use_dino=False):
    started = time.perf_counter()
    from litereality_agent.pipeline.scene_init import paths as config
    from litereality_agent.pipeline.scene_init.ingest import merge_boxes
    from litereality_agent.pipeline.scene_init.ingest.extract.extract_scene import extract
    from litereality_agent.pipeline.scene_init.layout import (
        adapter,
        check,
        require_valid_layout,
        run_layout,
    )

    scan = raw.name
    trace = output / scan / "layout_trace"
    trace.mkdir(parents=True)
    audit = Audit(trace, started=started)
    result = {}
    previous = Path.cwd()
    from unittest.mock import patch

    try:
        with (trace / "console.log").open("w", encoding="utf-8") as log, \
                patch.dict(os.environ, {"LR_LAYOUT_AGENT": "1" if use_agent else "0"}):
            with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                config.enter_work_root(scan)
                audit.event("capture", path=str(raw))
                with audit.phase("extraction"):
                    extract(str(raw), scan)
                scene = config.scene_data_dir(scan)
                write_json(trace / "01_extracted.json", adapter.shell_from_scene_data(scene))
                with audit.phase("merge"):
                    merged = merge_boxes.merge_for_scan(scan)
                    audit.event("merge", result=merged)
                    if merged.get("error"):
                        raise RuntimeError(f"Box merge failed: {merged['error']}")
                before = adapter.shell_from_scene_data(scene)
                write_json(trace / "02_merged.json", before)
                audit.event("initial_check", violations=check(before))
                refinement = None
                if use_dino or (use_agent and any(v.severity == "error" for v in check(before))):
                    with audit.phase("photos"):
                        refinement = prepare_photos(scan, config, audit, use_dino=use_dino)
                with ExitStack() as stack:
                    audit.instrument(stack)
                    audit.timings["to_layout_seconds"] = audit.elapsed()
                    with audit.phase("layout"):
                        result = run_layout(scan, use_agent=use_agent)
                    audit.timings["through_layout_seconds"] = audit.elapsed()
                result["agent_calls"] = audit.calls
                if refinement is not None:
                    result["dino"] = refinement
                if audit.calls == 0:
                    audit.event("no_agent_calls", reason="disabled, already sound, no suspects, or no usable photos")
                write_json(trace / "03_final.json", adapter.shell_from_scene_data(scene))
                if (scene / "layout.html").exists():
                    shutil.copy2(scene / "layout.html", trace / "layout.html")
                with audit.phase("validation"):
                    require_valid_layout(scan, result, scene_data_dir=scene)
                audit.timings["through_validation_seconds"] = audit.elapsed()
                audit.event("finished", result=result)
                return result
    except Exception as exc:
        result.setdefault("error", str(exc))
        raise
    finally:
        os.chdir(previous)
        config.set_scan(None)
        audit.timings["total_seconds"] = audit.elapsed()
        result.update(agent_calls=audit.calls, timings=audit.timings)
        if "validation" in result:
            write_json(trace / "layout_validation.json", result["validation"])
        write_json(trace / "timings.json", audit.timings)
        write_json(trace / "summary.json", result)
        print(f"[{scan}] total {audit.timings['total_seconds']:.2f}s; "
              f"timings: {trace / 'timings.json'}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenes", nargs="*", help="scene names; default: every capture in scans-dir")
    parser.add_argument("--scans-dir", type=Path)
    parser.add_argument("--output-root", type=Path, help="new directory (must not already exist)")
    parser.add_argument("--no-agent", action="store_true", help="no layout model calls; photos are still prepared with --use-dino")
    parser.add_argument("--use-dino", action="store_true", help="refine object crops with GroundingDINO before layout on every scene")
    parser.add_argument("--dino-backend", choices=("local", "modal", "auto"), default="auto",
                        help="DINO runtime (default: auto, prefers configured Modal; "
                             "local explicitly selects a local GPU)")
    parser.add_argument("--dry-run", action="store_true", help="list selected captures without running anything")
    args = parser.parse_args(argv)
    from litereality_agent.settings import load_settings

    settings = load_settings()
    scans = (args.scans_dir or settings.resolved_scans_dir()).expanduser().resolve()
    if not scans.is_dir():
        parser.error(f"Scan directory does not exist: {scans}")
    selected = [scans / name for name in args.scenes] if args.scenes else sorted(scans.iterdir())
    selected = [p.resolve() for p in selected if p.is_dir() and
                ((p / "room.usdz").is_file() or (p / "roomplan/room.usdz").is_file())]
    if not selected or (args.scenes and len(selected) != len(args.scenes)):
        parser.error("No captures found, or a requested scene is missing room.usdz")
    if len({p.name for p in selected}) != len(selected):
        parser.error("Scene names must be unique")
    output = (args.output_root or REPO / "run-layout" / datetime.now().strftime("%Y%m%d-%H%M%S")).expanduser().resolve()
    print(f"{len(selected)} scenes; output: {output}")
    if args.use_dino:
        print(f"DINO backend: {args.dino_backend}")
    for raw in selected:
        print(f"  {raw.name}")
    if args.dry_run:
        return 0
    if not args.no_agent and not shutil.which("claude"):
        parser.error("The current layout agent requires a logged-in claude CLI; or use --no-agent")
    if output.exists():
        parser.error(f"Use a new output directory: {output}")
    settings.apply_environment()
    dino_runtime = None
    if args.use_dino:
        os.environ["LR_DINO_BACKEND"] = args.dino_backend
        from litereality_agent.models.registry import detection_from_settings
        from litereality_agent.pipeline.scene_init.ingest.detect import detector

        service = detection_from_settings(load_settings())
        if service is not None:
            atexit.register(service.close)
            dino_runtime = {"backend": args.dino_backend, "service": service.name}
            if args.dino_backend == "local":
                # Fail before extraction if CUDA is unavailable; no inference.
                dino_runtime.update(service.health(), python=service.python)
            detector.set_service(service)
    os.environ.update(LITEREALITY_OUTPUT=str(output), LITEREALITY_FINAL=str(output),
                      LR_LAYOUT="1", LR_LAYOUT_VIZ="1", LR_BOX_MERGE="1")
    output.mkdir(parents=True)
    if dino_runtime is not None:
        write_json(output / "dino_runtime.json", dino_runtime)
    results = {}
    batch_started = time.perf_counter()
    for raw in selected:
        try:
            results[raw.name] = run_scene(raw, output, not args.no_agent, use_dino=args.use_dino)
        except Exception as exc:
            traceback.print_exc()
            summary = output / raw.name / "layout_trace/summary.json"
            results[raw.name] = json.loads(summary.read_text()) if summary.is_file() else {}
            results[raw.name]["error"] = str(exc)
        write_json(output / "summary.json", results)
        write_json(output / "timings.json", {
            "batch_elapsed_seconds": time.perf_counter() - batch_started,
            "completed_scenes": len(results),
            "scenes": {name: value.get("timings", {}) for name, value in results.items()},
        })
    print(f"Batch finished. Reports: {output}")
    return int(any("error" in result for result in results.values()))


if __name__ == "__main__":
    raise SystemExit(main())
