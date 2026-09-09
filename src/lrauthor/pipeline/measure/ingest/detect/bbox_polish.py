"""Stage 2b — GroundingDINO bbox refinement for normal objects.

Ported from LiteReality v1 ``LR_preprocessing/bbox_polish.py``. The crop stage gives
each object a 2D box per frame by PROJECTING its RoomPlan 3D box / visible point cloud
— which is often loose or contaminated (sparse points, occlusion, neighbouring
clutter). This pass TIGHTENS those boxes to the real object:

  1. derive a text prompt from the object category (``Storage0`` -> ``"storage .
     cabinet . cupboard . chest . shelf . wardrobe ."``),
  2. run GroundingDINO on each selected full frame,
  3. take the detection with the best IoU vs the projected box,
  4. if IoU >= ``IOU_THRESHOLD`` replace the projected box with the detection and
     re-crop the full frame; if every frame scores low, drop to a permissive fallback;
     otherwise keep the original projected crop.

Operates IN PLACE on the crop stage's output
(``parsed_images/<scan>/<Object>/frame_*_ranking_*.jpg`` + ``camera/*.json``), so it
slots between :mod:`crop_objects` and :mod:`object_references`. ON by default; the
projected crop is always the fallback, so a detector miss never makes things worse.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2

from lrauthor.pipeline.measure import paths as config

from . import detector

BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.20
IOU_THRESHOLD = 0.20  # keep the detection if it overlaps the projection this much
FALLBACK_THRESHOLD = 0.20  # if the best frame is below this, go permissive (v1)
MIN_IOU_FALLBACK = 0.001  # ...but still demand a little overlap in fallback mode (v1 MIN_IOU_SAVE)
# blur: drop very blurry crops (Laplacian-variance sharpness). A frame is dropped if its box
# region is below BLUR_FLOOR (absolute) OR below BLUR_REL × the sharpest kept frame — but the
# single sharpest frame is always kept, so an object never loses all its evidence.
BLUR_FLOOR = 25.0
BLUR_REL = 0.30


def _sharpness(gray, box) -> float:
    """Laplacian variance of the box region of a grayscale array (higher = sharper)."""
    h, w = gray.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
    y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
    region = gray[y1:y2, x1:x2]
    if region.size == 0:
        return 0.0
    return float(cv2.Laplacian(region, cv2.CV_64F).var())


# category synonyms that help the open-vocab detector (mirrors v1 derive_prompt)
SYNONYMS = {
    "storage": ["cabinet", "cupboard", "chest", "shelf", "wardrobe"],
    "table": ["desk", "table"],
    "sofa": ["couch", "sofa"],
    "television": ["tv", "monitor", "screen"],
}


def derive_prompt(object_id: str) -> str:
    """Object id -> GroundingDINO caption. Strips digits + 'wall', splits camelCase,
    maps ChairCluster->chair, and expands a few categories with synonyms."""
    name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", object_id)  # camelCase -> spaced
    name = re.sub(r"\d+", "", name)
    name = re.sub(r"(?i)\bwall\b", "", name)
    name = re.sub(r"(?i)\bcluster\b", "", name)  # ChairCluster -> chair
    words = [w for w in name.strip().replace("_", " ").lower().split() if w]
    base = words[0] if words else "object"
    terms = [base] + SYNONYMS.get(base, [])
    return " . ".join(dict.fromkeys(terms)) + " ."


def _camera_dir(obj_dir: Path) -> Path:
    return obj_dir / "camera"


def polish_object(obj_dir: Path) -> dict:
    """Refine every selected frame's box for one object folder, in place."""
    name = obj_dir.name
    prompt = derive_prompt(name)
    cam_dir = _camera_dir(obj_dir)
    metas = sorted(cam_dir.glob("frame_*_ranking_*.json")) if cam_dir.is_dir() else []
    if not metas:
        return {"object": name, "frames": 0, "refined": 0, "skipped": "no crops"}

    # pass 1: detect + best-IoU match per frame (don't write yet — need the max over frames)
    frames = []
    for mp in metas:
        meta = json.loads(mp.read_text())
        full = Path(meta.get("original_image_path", ""))
        proj = meta.get("resized_bbox") or meta.get("bbox")
        if not full.is_file() or not proj:
            continue
        import numpy as np
        from PIL import Image

        image = Image.open(full).convert("RGB")
        dets = detector.detect(image, prompt, BOX_THRESHOLD, TEXT_THRESHOLD)
        best_iou, best_box = 0.0, None
        for d in dets:
            v = detector.iou(d.box, [float(x) for x in proj])
            if v > best_iou:
                best_iou, best_box = v, d.box
        gray = np.asarray(image.convert("L"))
        sharp = _sharpness(gray, best_box if best_box else proj)  # sharpness of the object region
        frames.append(
            {
                "meta_path": mp,
                "meta": meta,
                "full": full,
                "proj": proj,
                "best_iou": best_iou,
                "best_box": best_box,
                "sharp": sharp,
            }
        )

    if not frames:
        return {"object": name, "frames": 0, "refined": 0, "skipped": "no readable frames"}

    # v1 modes: all_zero (DINO never found it -> keep the projections, best we have),
    # fallback (best frame weak -> permissive), normal. The key fix vs a naive port: when
    # DINO CONFIRMS the object on some frames, a frame it did NOT confirm has an unreliable
    # projection (e.g. the window's frame_3 landed on a cabinet) -> DROP it, don't keep the
    # wrong box. So every surviving crop is either DINO-confirmed or (if none were) projected.
    max_iou = max(f["best_iou"] for f in frames)
    all_zero = max_iou <= 0.0
    fallback = (max_iou < FALLBACK_THRESHOLD) and not all_zero
    keep_thr = MIN_IOU_FALLBACK if fallback else IOU_THRESHOLD

    # first pass of keep/drop: confirmed DINO box, or (if DINO found nothing) the projection.
    for f in frames:
        f["use_dino"] = bool(f["best_box"]) and f["best_iou"] >= keep_thr
        f["keep"] = f["use_dino"] or all_zero

    # blur filter: among the frames we'd keep, drop very blurry ones — but never the sharpest,
    # so each object keeps at least one (its best) crop.
    kept = [f for f in frames if f["keep"]]
    if kept:
        sharpest = max(kept, key=lambda f: f.get("sharp", 0.0))
        floor = max(BLUR_FLOOR, BLUR_REL * sharpest.get("sharp", 0.0))
        for f in kept:
            if f is not sharpest and f.get("sharp", 0.0) < floor:
                f["keep"], f["blur_drop"] = False, True

    refined = dropped = dropped_blur = 0
    for f in frames:
        if f["keep"]:
            box = f["best_box"] if f["use_dino"] else f["proj"]
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            img = cv2.imread(str(f["full"]))
            if img is None:
                continue
            H, W = img.shape[:2]
            x1, x2 = max(0, min(x1, x2)), min(W, max(x1, x2))
            y1, y2 = max(0, min(y1, y2)), min(H, max(y1, y2))
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            if f["use_dino"]:  # re-crop at the tightened box
                cv2.imwrite(str(obj_dir / (f["meta_path"].stem + ".jpg")), img[y1:y2, x1:x2])
                f["meta"].update(
                    {
                        "resized_bbox": [x1, y1, x2, y2],
                        "dino_refined": True,
                        "dino_iou": round(f["best_iou"], 4),
                        "dino_prompt": prompt,
                    }
                )
                f["meta_path"].write_text(json.dumps(f["meta"], indent=2))
                refined += 1
            # else: projection kept as-is (existing crop + meta)
        else:
            f["meta_path"].unlink(missing_ok=True)  # drop: wrong projection or too blurry
            (obj_dir / (f["meta_path"].stem + ".jpg")).unlink(missing_ok=True)
            if f.get("blur_drop"):
                dropped_blur += 1
            else:
                dropped += 1

    return {
        "object": name,
        "frames": len(frames),
        "refined": refined,
        "dropped": dropped,
        "dropped_blur": dropped_blur,
        "kept_projection": all_zero,
        "max_iou": round(max_iou, 4),
        "fallback": fallback,
        "prompt": prompt,
    }


def marker_path(scan: str) -> Path:
    """Where a completed polish records itself, so a re-run can skip it."""
    from lrauthor.pipeline.measure import paths as config

    return config.work_root() / f"bbox_polish_{scan}.json"


def polish(scan: str, *, force: bool = False) -> dict:
    """Refine all object crops for ``scan`` in place. Returns a per-object summary.

    Idempotent: the refinement is written back into each frame's meta and crop, so running it
    again re-detects every box to reach the state it is already in — a minute or two of DINO for
    nothing. A completed run records itself; `force=True` (or --force-bbox-polish) redoes it,
    which is what you want after new crops or a different detector.
    """
    marker = marker_path(scan)
    if not force and marker.is_file():
        try:
            previous = json.loads(marker.read_text(encoding="utf-8"))
            previous["reused"] = True
            print(f"[bbox-polish] reusing previous refinement "
                  f"({previous.get('refined_total', 0)} boxes) — --force-bbox-polish to redo",
                  flush=True)
            return previous
        except (OSError, ValueError):
            pass  # unreadable marker: fall through and redo it properly
    # Resolve DINO once through the model registry. Hosted Modal wins when configured; an isolated
    # local interpreter remains available for Linux workstations. The light pipeline process never
    # needs to import torch in either case.
    if not detector.using_service():
        from lrauthor.models.registry import detection_from_settings
        from lrauthor.settings import load_settings

        settings = load_settings()
        settings.apply_environment()
        service = detection_from_settings(settings)
        if service is not None:
            detector.set_service(service)
            print(f"  [bbox_polish] DINO backend: {service.name}", flush=True)

    if not detector.available():
        print(
            "  [bbox_polish] torch/transformers unavailable — skipping refinement "
            "(crops stay as the projected boxes).",
            flush=True,
        )
        return {"scan": scan, "skipped": "detector unavailable", "objects": []}

    parsed = config.parsed_images_dir(scan)
    if not parsed.is_dir():
        print(f"  [bbox_polish] no crops at {parsed} — run crop_objects first.", flush=True)
        return {"scan": scan, "skipped": "no crops", "objects": []}

    # openings (Wall*_Door/Window*, Opening*_Door/Window*) are handled by the detection-based
    # opening matcher, not here — skip them.
    # Also skip ENLARGED-crop objects (merged sink+storage sets): their crop is deliberately the full
    # projected 3D box, and open-vocab detection would just re-tighten it back to the sub-part it
    # recognises (the sink) — defeating the enlargement. Keep the geometric crop as-is.
    import os
    enlarged = set(filter(None, os.environ.get("LR_ENLARGED_CROP_OBJECTS", "").split(",")))
    obj_dirs = [
        d
        for d in sorted(parsed.iterdir())
        if d.is_dir() and not re.search(r"_(Door|Window)_", d.name) and d.name not in enlarged
    ]
    print(
        f"  [bbox_polish] {detector.default_model_id()} — refining {len(obj_dirs)} object(s)",
        flush=True,
    )
    results = []
    for d in obj_dirs:
        r = polish_object(d)
        results.append(r)
        if r.get("refined") is not None:
            tag = (
                " KEPT-PROJECTION"
                if r.get("kept_projection")
                else (" FALLBACK" if r.get("fallback") else "")
            )
            print(
                f"    {r['object']:16s} refined {r.get('refined', 0)}/{r.get('frames', 0)} "
                f"dropped {r.get('dropped', 0)}+{r.get('dropped_blur', 0)}blur "
                f"(max_iou={r.get('max_iou')}{tag})",
                flush=True,
            )
    total = sum(r.get("refined", 0) for r in results)
    dropped = sum(r.get("dropped", 0) for r in results)
    blur = sum(r.get("dropped_blur", 0) for r in results)
    print(
        f"  [bbox_polish] done — {total} box(es) tightened, {dropped} wrong-projection + "
        f"{blur} blurry frame(s) dropped",
        flush=True,
    )
    summary = {
        "scan": scan,
        "objects": results,
        "refined_total": total,
        "dropped_total": dropped,
        "dropped_blur_total": blur,
    }
    try:  # record completion so the next run can skip it; never fatal
        marker.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except (OSError, TypeError):
        pass
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--scan", required=True)
    ap.add_argument("--force", action="store_true", help="redo even if a previous run is recorded")
    ap.add_argument("--output-root", help="override the output root")
    ap.add_argument(
        "--model", help="HF GroundingDINO id ($LR_DINO_MODEL); default grounding-dino-tiny"
    )
    args = ap.parse_args()
    if args.output_root:
        import os

        os.environ["LITEREALITY_OUTPUT"] = str(Path(args.output_root).expanduser().resolve())
    if args.model:
        import os

        os.environ["LR_DINO_MODEL"] = args.model
    config.set_scan(args.scan)
    out = polish(args.scan, force=args.force)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
