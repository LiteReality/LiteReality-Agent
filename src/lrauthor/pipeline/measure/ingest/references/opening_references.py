"""Generate clean hosted reference images for doors and windows.

Openings are thin, wall-attached, and point-cloud-sparse, so the ordinary object
crop path misses most of them. Instead of a 2D detector (the original used
GroundingDINO), this projects each RoomPlan **3D opening box** (the 8 world
corners in wall_holes.pkl) into every capture frame using that frame's
intrinsic/extrinsic, scores how well the opening is seen (visible corners,
projected size, centeredness, sharpness), crops the best frames, and feeds a
weighted evidence sheet to the OpenAI image model with a door/window-specific prompt.

Self-contained: numpy + opencv + pillow + the rgbd camera data. No torch / no
GroundingDINO. Writes:

    opening_refs/<scan>/<Opening>/crops/frame_*.jpg     selected evidence crops
    opening_refs/<scan>/<Opening>/input2imagegen.jpg      evidence sheet
    opening_refs/<scan>/<Opening>/clean_obj_reference.png raw generation
    opening_refs/<scan>/<Opening>/reference_1024.png    clean reference
    opening_refs/<scan>/<Opening>/{prompt.txt,reference_meta.json}
    opening_refs/<scan>/opening_references.json
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from lrauthor.pipeline.measure import paths as config
from lrauthor.pipeline.measure.ingest.references import image_gen as image_model

from ..detect import detector

LOW_RES_W = 256  # the resolution the rgbd intrinsics are written for
LOW_RES_H = 192


# ── geometry ─────────────────────────────────────────────────────────────────
def rgbd_dir(scan: str) -> Path:
    return config.input_root() / "rgbd" / scan


def load_openings(scan: str) -> list[dict]:
    with (config.scene_data_dir(scan) / "wall_holes.pkl").open("rb") as f:
        wall_holes = pickle.load(f)
    openings = []
    for rec in wall_holes.values():
        if not isinstance(rec, dict) or not rec.get("name"):
            continue
        for name, kind, pts in zip(rec["name"], rec["type"], rec["bbox_points"]):
            openings.append(
                {"name": name, "type": kind, "bbox_points": np.asarray(pts, dtype=np.float32)}
            )
    return openings


def project_points(points: np.ndarray, scan: str, frame_id: int):
    d = rgbd_dir(scan)
    ip = d / "intrinsic" / f"intrinsic_{frame_id}.npy"
    pp = d / "extrinsic" / f"extrinsic_{frame_id}.npy"
    if not ip.exists() or not pp.exists():
        return None
    intrinsic = np.load(ip)
    pose = np.load(pp)
    points_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    camera = np.linalg.inv(pose) @ points_h.T
    z = camera[2]
    valid = z > 1e-4
    if valid.sum() < 2:
        return None
    u = (camera[0] * intrinsic[0, 0]) / z + intrinsic[0, 2]
    v = (camera[1] * intrinsic[1, 1]) / z + intrinsic[1, 2]
    return np.stack([u, v], axis=1), valid


def clamp_box(box, w, h):
    return [
        max(0.0, min(float(w - 1), box[0])),
        max(0.0, min(float(h - 1), box[1])),
        max(0.0, min(float(w - 1), box[2])),
        max(0.0, min(float(h - 1), box[3])),
    ]


def projected_box(opening, scan, frame_id, image_size, pad_ratio=0.15):
    out = project_points(opening["bbox_points"], scan, frame_id)
    if out is None:
        return None
    uv, valid = out
    in_view = (
        valid
        & (uv[:, 0] >= -LOW_RES_W * pad_ratio)
        & (uv[:, 0] <= LOW_RES_W * (1 + pad_ratio))
        & (uv[:, 1] >= -LOW_RES_H * pad_ratio)
        & (uv[:, 1] <= LOW_RES_H * (1 + pad_ratio))
    )
    if in_view.sum() < 2:
        return None
    pts = uv[in_view]
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    if x1 <= x0 or y1 <= y0:
        return None
    w, h = image_size
    box = clamp_box(
        [x0 * w / LOW_RES_W, y0 * h / LOW_RES_H, x1 * w / LOW_RES_W, y1 * h / LOW_RES_H], w, h
    )
    if (box[2] - box[0]) * (box[3] - box[1]) <= 20:
        return None
    return {"frame_id": frame_id, "box": box, "visible_corners": int(in_view.sum())}


def edge_margin_score(box, w, h):
    margin = min(box[0], box[1], w - box[2], h - box[3])
    return max(0.0, min(1.0, margin / max(1.0, min(w, h) * 0.08)))


def pad_box(box, w, h, ratio=0.18):
    pad = ratio * max(box[2] - box[0], box[3] - box[1])
    return [
        int(round(v))
        for v in clamp_box([box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad], w, h)
    ]


def sharpness(image_bgr, box):
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    crop = image_bgr[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)]
    if crop.size == 0:
        return 0.0
    return float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def score_candidate(cand, image_bgr, image_size):
    w, h = image_size
    box = cand["box"]
    area_frac = ((box[2] - box[0]) * (box[3] - box[1])) / float(w * h)
    area_norm = min(1.0, math.sqrt(area_frac / 0.15))  # ~15% of frame -> 1.0
    corners = cand["visible_corners"] / 8.0
    center = edge_margin_score(box, w, h)
    sharp = max(0.0, min(1.0, math.log1p(sharpness(image_bgr, box)) / math.log1p(900.0)))
    return 0.30 * corners + 0.34 * area_norm + 0.18 * center + 0.18 * sharp


# ── evidence sheet (one big primary + support strip) ─────────────────────────
def _paste(canvas, im, box, bg=(8, 8, 8)):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    tile = Image.new("RGB", (w, h), bg)
    t = im.copy()
    t.thumbnail((w - 16, h - 16), Image.Resampling.LANCZOS)
    tile.paste(t, ((w - t.width) // 2, (h - t.height) // 2))
    canvas.paste(tile, (x0, y0))


def make_sheet(crop_paths, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (1024, 1024), (0, 0, 0))
    if not crop_paths:
        canvas.save(output, quality=95)
        return
    _paste(canvas, Image.open(crop_paths[0]).convert("RGB"), (40, 60, 660, 964))
    for path, y in zip(crop_paths[1:5], (60, 290, 520, 750)):
        _paste(canvas, Image.open(path).convert("RGB"), (700, y, 988, y + 214))
    canvas.save(output, quality=95)


DOOR_PROMPT = (
    "Preserve the leaf type (flush, panelled, or glazed) and the EXACT panel or glazing layout and its proportions. "
    "Keep the handle/lever or knob at its real height and on the correct side, plus any push-bar, lock/escutcheon, "
    "letter plate, vision panel, or kick plate. Show the hinges on the correct side and any visible door closer. "
    "Preserve the door frame/architrave and threshold, and the leaf color and material (wood grain, painted, "
    "laminate). Keep single- vs double-leaf as seen. "
    "If the door is open in the evidence, render it in a clean CLOSED state so it is easy to reconstruct and later "
    "articulate (a hinged door swinging from one side). If it has glazing, render that glass as clear or slightly "
    "tinted, with subtle highlights only. Do NOT copy any recognizable outside scenery, room contents, or reflected "
    "objects into the glass. Do NOT add posters, signs, hooks, notices, or anything hanging on the door unless it is "
    "clearly part of the door — default to a clean leaf."
)

WINDOW_PROMPT = (
    "Identify HOW the window opens from the frame and hardware (casement/side-hung with visible hinges or stays, "
    "top-hung/awning, vertical or horizontal sliding/sash with meeting rails or tracks, or fixed) and render that "
    "mechanism plausibly. Preserve the glazing layout: the number of sashes/lights, the mullions and glazing bars, "
    "and the real PROPORTION of glass area to frame. Include the window SILL/ledge and the surrounding frame or "
    "architrave if visible, and the frame material and color. "
    "CRITICAL — curtains/blinds, reproduce FAITHFULLY from the evidence (do not invent): look carefully at the "
    "photos to see whether this window actually has a covering and, if so, EXACTLY what it is and where it sits. "
    "Match the real TYPE (horizontal venetian slats, vertical louvre blinds, roller blind, roman blind, or fabric "
    "curtains) and the real POSITION/STATE (e.g. vertical blinds drawn and stacked to one side, a blind pulled "
    "halfway down, curtains gathered at the sides). Reproduce that same covering in the same place — it is easy to "
    "miss, so do not drop or flatten a covering that is there. But do NOT fabricate a covering that is not in the "
    "evidence, do NOT change its type (e.g. never turn side vertical blinds into a top roller blind), and do NOT "
    "move it to a different part of the window. If the window clearly has no covering, add none. "
    "CRITICAL — the glass: for the glazing that is actually visible (the panes not covered by blinds or curtains), "
    "render it as CLEAN, EMPTY, UNIFORM transparent or very lightly tinted glass, like a brand-new glazed window "
    "with nothing behind it. Those panes must be completely CONTENT-FREE: absolutely NO outside scenery (buildings, "
    "rooftops, street, trees, sky, clouds), NO interior or next-room content, NO people, NO furniture, NO reflected "
    "objects, and NO copied imagery of any kind — even though the evidence clearly shows such things through or "
    "reflected in the glass. Use the evidence ONLY to tell which regions are glass; treat everything behind the glass "
    "as a plain empty neutral light backdrop, keeping only generic cues (a faint tint, soft gradient, subtle edge "
    "highlight). This applies to interior windows and serving hatches too — clean empty glass, never the room beyond."
)


def build_prompt(opening: dict) -> str:
    kind = opening["type"].lower()
    specifics = DOOR_PROMPT if kind == "door" else WINDOW_PROMPT
    return (
        f"The input image is a visual evidence sheet for one scanned {kind}: {opening['name']}. The large tile is "
        f"the best-aligned view of the opening; the smaller tiles are secondary views. Some views may be blurry, "
        f"off-center, cropped, partly occluded, or seen at an angle — combine them to recover the most complete, "
        f"correct structure. Create one clean object-only 3D-asset reference render of the same {kind}, in a "
        f"front-facing (or very slightly three-quarter) view, upright and centered, with the full opening visible "
        f"top to bottom. {specifics} "
        "Ignore the wall background, floor, room clutter, labels, borders, and the collage layout. Do not invent "
        "extra panels, panes, bars, handles, or decorative details not supported by the references. "
        "STRUCTURE FIDELITY: reproduce the overall construction and the EXACT COUNT of every repeated element the "
        "evidence shows — panels, glazed panes, sashes/lights, mullions/glazing bars, and hinges. Do NOT drop, merge, "
        "omit, or simplify a real feature: if the evidence shows a glazed vision panel or a multi-pane layout, the "
        "render MUST contain the same panels/panes in the same arrangement — never flatten a panelled or glazed leaf "
        "into a plain flush one. Match reality, neither adding nor removing structure. Use a solid pure "
        "black background edge-to-edge."
    )


def refine_box_with_dino(opening: dict, cand: dict, prompt: str) -> bool:
    """Tighten a projected opening box to the real 2D detection (v2's detection-based
    matcher, reusing the shared HF GroundingDINO). Detect doors/windows in the frame,
    take the detection with the best IoU vs the projected box, and replace the box if
    IoU >= 0.20. Returns True if refined. Projection stays as the fallback."""
    from PIL import Image as _Image

    try:
        with _Image.open(cand["path"]) as im:
            image = im.convert("RGB")
        dets = detector.detect(image, prompt, box_threshold=0.25, text_threshold=0.20)
    except Exception:
        return False
    best_iou, best_box = 0.0, None
    for d in dets:
        v = detector.iou(d.box, cand["box"])
        if v > best_iou:
            best_iou, best_box = v, d.box
    if best_box and best_iou >= 0.20:
        cand["box"] = [float(x) for x in best_box]
        cand["dino_iou"] = round(best_iou, 4)
        return True
    return False


def generate_for_scan(
    scan,
    output_root,
    *,
    skip_image_generation=False,
    force_image_generation=False,
    model=config.DEFAULT_IMAGE_MODEL,
    stride=1,
    max_images=5,
    only=None,
    refine_dino=True,
    crops_only=False,
) -> dict:
    output_root = Path(output_root)
    image_dir = rgbd_dir(scan) / "image"
    frames = sorted(image_dir.glob("frame_*.jpg"), key=lambda p: int(p.stem.split("_")[1]))
    if not frames:
        return {"scan": scan, "openings": [], "error": f"no rgbd frames in {image_dir}"}
    with Image.open(frames[0]) as im:
        image_size = im.size  # (W, H)

    openings = load_openings(scan)
    result = {"scan": scan, "openings": []}
    for opening in openings:
        name = opening["name"]
        if only and name not in only:
            continue
        # 1. score every (strided) frame by how well the projected box is seen
        candidates = []
        for fp in frames:
            fid = int(fp.stem.split("_")[1])
            if stride > 1 and fid % stride:
                continue
            cand = projected_box(opening, scan, fid, image_size)
            if cand is None:
                continue
            image_bgr = cv2.imread(str(fp))
            if image_bgr is None:
                continue
            cand["score"] = score_candidate(cand, image_bgr, image_size)
            cand["path"] = fp
            candidates.append(cand)
        candidates.sort(key=lambda c: c["score"], reverse=True)
        selected = candidates[:max_images]

        # 1b. refine each selected box with GroundingDINO (detection IoU-matched to the
        # projection); projection stays the fallback. Doors/windows are where the raw
        # projection is loosest, so this is what tightens the opening crop.
        n_refined = 0
        if refine_dino and selected and detector.available():
            dprompt = f"{opening['type'].lower()} ."
            for cand in selected:
                if refine_box_with_dino(opening, cand, dprompt):
                    n_refined += 1
            # if DINO confirmed the opening on SOME frames, the unconfirmed frames have an
            # unreliable projection (e.g. a window box that landed on a cabinet) -> drop them,
            # keep only the DINO-confirmed crops. If none were confirmed, keep the projections.
            if n_refined > 0:
                selected = [c for c in selected if "dino_iou" in c]

        # blur filter: drop very blurry opening crops (Laplacian-variance), keep the sharpest.
        if len(selected) > 1:
            sh = {}
            for c in selected:
                im = cv2.imread(str(c["path"]))
                sh[id(c)] = sharpness(im, c["box"]) if im is not None else 0.0
            best = max(selected, key=lambda c: sh[id(c)])
            floor = max(25.0, 0.30 * sh[id(best)])
            selected = [c for c in selected if c is best or sh[id(c)] >= floor]

        out_dir = output_root / scan / name
        crops_dir = out_dir / "crops"
        status, error, image_meta = "no_view", None, None
        crop_paths = []
        crops_meta = []
        if selected:
            crops_dir.mkdir(parents=True, exist_ok=True)
            for cand in selected:
                with Image.open(cand["path"]) as im:
                    crop = im.convert("RGB").crop(pad_box(cand["box"], *image_size))
                # The rgbd frames are stored landscape; rotate the crop upright so the
                # Present the opening front-on to the image model.
                crop = crop.transpose(Image.Transpose.ROTATE_270)
                cp = crops_dir / f"frame_{cand['frame_id']}.jpg"
                crop.save(cp, quality=95)
                crop_paths.append(cp)
                crops_meta.append(
                    {
                        "frame_id": cand["frame_id"],
                        "frame": str(cand["path"]),
                        "box": [round(float(v), 2) for v in cand["box"]],  # final (landscape) box
                        "dino_refined": "dino_iou" in cand,
                        "dino_iou": cand.get("dino_iou"),
                    }
                )
            (out_dir / "crops_meta.json").write_text(
                json.dumps(crops_meta, indent=2), encoding="utf-8"
            )

            if crops_only:
                status = "crops_only"
            else:
                sheet = out_dir / config.INPUT_SHEET
                raw = out_dir / config.CLEAN_REFERENCE
                normalized = out_dir / "reference_1024.png"
                prompt = build_prompt(opening)
                make_sheet(crop_paths, sheet)
                (out_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
                if skip_image_generation:
                    status = "skipped_image_generation"
                else:
                    try:
                        image_meta = image_model.generate_reference(
                            sheet,
                            prompt,
                            raw,
                            model=model,
                            force=force_image_generation,
                            response_dir=out_dir / "_responses",
                        )
                        image_model.normalize_reference(raw, normalized)
                        status = "ok"
                    except Exception as exc:  # noqa: BLE001
                        status, error = "error", str(exc)

        record = {
            "scan": scan,
            "opening": name,
            "type": opening["type"],
            "status": status,
            "n_views": len(selected),
            "n_dino_refined": n_refined,
            "crops": [str(p) for p in crop_paths],
        }
        if error:
            record["error"] = error
        if image_meta:
            record["image_meta"] = image_meta
        if selected:
            (out_dir / "reference_meta.json").write_text(
                json.dumps(record, indent=2), encoding="utf-8"
            )
        result["openings"].append(record)
        print(
            f"  [{status}] {name} ({opening['type']}, {len(selected)} views, "
            f"{n_refined} DINO-refined)",
            flush=True,
        )

    (output_root / scan).mkdir(parents=True, exist_ok=True)
    (output_root / scan / "opening_references.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--scan", nargs="*", help="Scan names (default: all with scene_data).")
    ap.add_argument("--skip-image-generation", action="store_true")
    ap.add_argument("--force-image-generation", action="store_true")
    ap.add_argument("--model", default=config.DEFAULT_IMAGE_MODEL)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument(
        "--only", nargs="*", help="Only (re)generate these opening names, e.g. Wall5_Window_0."
    )
    args = ap.parse_args()

    # The work root is per scan, so it has to be selected inside the loop — entering it once up
    # front (as this did) resolved every path against the `_shared` fallback and wrote to a tree
    # no scan reads. Discovering scans needs the OUTPUT root rather than scene_data, for the same
    # reason: there is no scene_data to list until a scan has been chosen.
    only = set(args.only) if args.only else None
    scans = args.scan
    if not scans:
        root = config.output_root()
        scans = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
        if not scans:
            raise SystemExit(f"no scans under {config.output_root()}; pass --scan")

    results = []
    out_root = None
    for scan in scans:
        config.enter_work_root(scan)
        out_root = config.work_root() / "opening_refs"
        print(f"[openings] {scan}", flush=True)
        results.append(
            generate_for_scan(
                scan,
                out_root,
                skip_image_generation=args.skip_image_generation,
                force_image_generation=args.force_image_generation,
                model=args.model,
                stride=args.stride,
                only=only,
            )
        )
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "opening_manifest.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    n_ok = sum(1 for r in results for o in r["openings"] if o["status"] == "ok")
    n_tot = sum(len(r["openings"]) for r in results)
    print(f"\nopenings: {n_ok}/{n_tot} references ok -> {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
