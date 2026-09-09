"""Group repeated chairs, then generate one clean reference per group.

Ported from the v2 ``step2 clustering``. Repeated chairs in a room (the four
identical chairs around a table) should collapse to one shared type. For each
chair it builds visual + color + dimension + spatial features, links pairs with a
conservative gate, runs greedy average-link clustering, picks a representative,
and generates one clean hosted reference per cluster. Legacy output directory names are preserved
so existing reconstruction runs remain resumable. Writes:

    <output_root>/<scan>/chair_clusters.json      # the grouping (members per cluster)
    <output_root>/<scan>/chair_sheets/<Chair>.jpg
    <output_root>/<scan>/cluster_sheets/<ChairCluster>.jpg
    <output_root>/<scan>/input2imagegen/<ChairCluster>.jpg
    <output_root>/<scan>/generated/<ChairCluster>.png
    <output_root>/<scan>/references/<ChairCluster>.png
    <output_root>/chair_clusters_all.json, index.html

The grouping JSON is what downstream 3D generation consumes: generate one asset
per cluster and place it at every member's RoomPlan box.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from litereality_agent.fonts import font as _shared_font
from litereality_agent.pipeline.measure import paths as config
from litereality_agent.pipeline.measure.ingest.references import image_gen as image_model

from ..detect import detector  # DINOv2 embeddings (isolated torch tool) for the visual signal
from . import chair_type_judge  # Opus 4.8 judgment layer — the primary grouping decision

# The chair TYPE decision is delegated to a vision LLM (chair_type_judge): frame material / arms /
# colour / base are what define a type, and an embedding blurs them. The DINOv2 average-linkage path
# below stays as the deterministic FALLBACK (SDK/subscription unavailable) and for representative-view
# selection. Set LR_CHAIR_JUDGE=0 to force the embedding path.
USE_JUDGE = os.environ.get("LR_CHAIR_JUDGE", "1").strip().lower() not in ("0", "false", "no", "off")

# ── clustering knobs (env-overridable) ─────────────────────────────────────────
# The visual signal is a DINOv2 CLS-embedding cosine — semantically "same chair design",
# robust to viewpoint/lighting — so a single threshold separates types (unlike the old
# pose-sensitive template features that needed a per-scene ladder). Two chairs merge iff
# their embeddings agree AND size agrees AND colour isn't wildly different. Colour here is a
# NOISY crop-level signal (floor/table bleed in), so it's only a loose safety gate — the
# embedding carries the decision.
#
# Clustering is AVERAGE-linkage, not complete-linkage: at scale (a meeting room of 16 near-
# identical chairs) many crops are occluded/partial, so some same-type pairs dip below threshold.
# Complete-linkage's weakest-link rule then shatters one chair type into many clusters; averaging
# over the cross-pairs tolerates a few bad views and keeps the type together. The threshold is set
# so a genuinely different chair (mean cross-similarity ~0.58 in tests) still stays separate.
VIS_THRESHOLD = float(os.environ.get("LR_CHAIR_VIS_THRESHOLD", "0.62"))  # mean DINOv2 cosine to merge
DIM_MIN = float(os.environ.get("LR_CHAIR_DIM_MIN", "0.55"))  # reject gross size mismatch
COLOR_MIN = float(os.environ.get("LR_CHAIR_COLOR_MIN", "0.20"))  # reject only extreme colour clash
EMBED_VIEWS = int(os.environ.get("LR_CHAIR_EMBED_VIEWS", "6"))  # crops per chair to mean-pool


@dataclass
class ChairRecord:
    scan: str
    name: str
    object_dir: Path
    images: list[Path]
    bbox: list[float]
    position: list[float]
    sheet_path: Path
    feature: np.ndarray
    color_feature: np.ndarray
    dominant_rgb: list[int]
    view_score: float


def natural_key(path: Path) -> list[object]:
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", path.name)]


def chair_sort_key(name: str) -> list[object]:
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", name)]


def load_scene_objects(scan: str) -> dict[str, dict]:
    with (config.scene_data_dir(scan) / "objects.pkl").open("rb") as f:
        objects = pickle.load(f)
    by_name = {}
    for obj in objects:
        name = obj.get("mesh_id") or obj.get("object_type")
        if name:
            by_name[name] = obj
    return by_name


def oriented_image(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB").transpose(Image.Transpose.ROTATE_270)


# ── features ─────────────────────────────────────────────────────────────────
def dino_features(image_lists: list[list[Path]]) -> list[np.ndarray] | None:
    """Mean-pooled, L2-normed DINOv2 CLS embedding per chair (one vector per input list).

    Batches ALL chairs' crops into a single embed call for GPU efficiency, then averages each
    chair's views. Returns None if no embedding backend is available (caller falls back to the
    hand-crafted :func:`image_feature`)."""
    if not detector.embed_available():
        return None
    flat: list[Path] = []
    spans: list[tuple[int, int]] = []
    for paths in image_lists:
        chosen = list(paths[:EMBED_VIEWS])
        spans.append((len(flat), len(flat) + len(chosen)))
        flat.extend(chosen)
    if not flat:
        return None
    try:
        embs = detector.embed([str(p) for p in flat], upright=True)
    except Exception as exc:  # noqa: BLE001 — fall back to CV features rather than crash init
        print(f"  [chair-cluster] DINOv2 embed failed ({exc}); falling back to CV features", flush=True)
        return None
    arr = np.asarray(embs, dtype=np.float32)
    out = []
    for lo, hi in spans:
        if hi > lo:
            m = arr[lo:hi].mean(axis=0)
        else:
            m = np.zeros(arr.shape[1], dtype=np.float32)
        out.append((m / (np.linalg.norm(m) + 1e-8)).astype(np.float32))
    return out


def image_feature(paths: list[Path]) -> np.ndarray:
    """Fallback CV descriptor (colour hist + downsampled grayscale + edge marginals). Used only
    when DINOv2 is unavailable; it is pose-sensitive, which is why the DINOv2 path is preferred."""
    feats = []
    for path in paths[:4]:
        im = np.asarray(oriented_image(path).resize((128, 128), Image.Resampling.LANCZOS))
        hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 5, 5], [0, 180, 0, 256, 0, 256]).reshape(
            -1
        )
        hist = hist / (np.linalg.norm(hist) + 1e-8)

        gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        small = small - small.mean()
        small = small / (np.linalg.norm(small) + 1e-8)

        edges = cv2.Canny(gray, 80, 160)
        edge = np.concatenate([edges.mean(axis=0), edges.mean(axis=1)]).astype(np.float32)
        edge = edge / (np.linalg.norm(edge) + 1e-8)
        feats.append(np.concatenate([hist, small.reshape(-1), edge]))

    if not feats:
        return np.zeros(12 * 5 * 5 + 32 * 32 + 256, dtype=np.float32)
    feat = np.mean(feats, axis=0).astype(np.float32)
    return feat / (np.linalg.norm(feat) + 1e-8)


def color_signature(paths: list[Path]) -> tuple[np.ndarray, list[int]]:
    hists = []
    dominant_rgbs = []
    for path in paths[:4]:
        im = np.asarray(oriented_image(path).resize((128, 128), Image.Resampling.LANCZOS))
        # The crop border often contains floor/table/background after padding;
        # use the central region so object material matters more.
        crop = im[16:112, 16:112]
        lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)
        hist = cv2.calcHist([lab], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).reshape(-1)
        hist = hist / (np.linalg.norm(hist) + 1e-8)
        hists.append(hist)

        pixels = crop.reshape(-1, 3).astype(np.float32)
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV).reshape(-1, 3)
        mask = (hsv[:, 1] > 18) | (hsv[:, 2] < 210)
        if mask.sum() > 64:
            pixels = pixels[mask]
        dominant_rgbs.append(np.median(pixels, axis=0))

    if not hists:
        return np.zeros(512, dtype=np.float32), [128, 128, 128]
    hist = np.mean(hists, axis=0).astype(np.float32)
    hist = hist / (np.linalg.norm(hist) + 1e-8)
    dominant = np.clip(np.mean(dominant_rgbs, axis=0), 0, 255).astype(np.uint8).tolist()
    return hist, [int(v) for v in dominant]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8))


def color_similarity(
    a_hist: np.ndarray, b_hist: np.ndarray, a_rgb: list[int], b_rgb: list[int]
) -> float:
    hist_sim = max(0.0, min(1.0, cosine(a_hist, b_hist)))
    a_lab = cv2.cvtColor(np.uint8([[a_rgb]]), cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]
    b_lab = cv2.cvtColor(np.uint8([[b_rgb]]), cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]
    dominant_sim = math.exp(-float(np.linalg.norm(a_lab - b_lab)) / 24.0)
    return float(0.65 * hist_sim + 0.35 * dominant_sim)


def dimension_similarity(a: list[float], b: list[float]) -> float:
    da = np.maximum(np.array(a, dtype=np.float32), 1e-4)
    db = np.maximum(np.array(b, dtype=np.float32), 1e-4)
    # Sort horizontal dims so a rotated chair is still comparable.
    sa = np.array([min(da[0], da[2]), da[1], max(da[0], da[2])], dtype=np.float32)
    sb = np.array([min(db[0], db[2]), db[1], max(db[0], db[2])], dtype=np.float32)
    return float(np.exp(-np.mean(np.abs(np.log(sa / sb)))))


def spatial_similarity(a: list[float], b: list[float]) -> float:
    pa = np.array(a, dtype=np.float32)
    pb = np.array(b, dtype=np.float32)
    dist = float(np.linalg.norm(pa[[0, 2]] - pb[[0, 2]]))
    return float(math.exp(-dist / 2.0))


def pair_link(visual: float, dim: float, color: float) -> bool:
    """Per-pair "plausibly the same type" — the DINOv2 embeddings agree, footprints agree
    (guards a stool vs an armchair), and colour isn't wildly off (loose — colour is a noisy
    crop-level signal). Only used for the ``linked`` flag logged in the JSON; the actual
    grouping decision is average-linkage over clusters (:func:`agglomerate_clusters`)."""
    return visual >= VIS_THRESHOLD and dim >= DIM_MIN and color >= COLOR_MIN


def agglomerate_clusters(
    names: list[str], pair_lookup: dict[tuple[str, str], dict]
) -> list[list[str]]:
    """Greedy AVERAGE-linkage clustering. Two clusters merge when the MEAN cross-pair visual
    similarity clears the threshold and the mean footprint/colour agree — averaging tolerates the
    few occluded/partial views that shatter complete-linkage at scale, while a genuinely different
    chair (whose mean similarity stays low) still stays apart. Merge order favours the highest
    mean visual, so the most-confident pairs join first."""

    def key(a: str, b: str) -> tuple[str, str]:
        return tuple(sorted((a, b), key=chair_sort_key))

    def means(left: list[str], right: list[str]) -> tuple[float, float, float]:
        vis, dim, col = [], [], []
        for a in left:
            for b in right:
                pair = pair_lookup[key(a, b)]
                vis.append(pair["visual_similarity"])
                dim.append(pair["dimension_similarity"])
                col.append(pair["color_similarity"])
        return float(np.mean(vis)), float(np.mean(dim)), float(np.mean(col))

    clusters = [[name] for name in names]
    while True:
        best = None
        best_visual = -1.0
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                mvis, mdim, mcol = means(clusters[i], clusters[j])
                if mvis < VIS_THRESHOLD or mdim < DIM_MIN or mcol < COLOR_MIN:
                    continue
                if mvis > best_visual:
                    best = (i, j)
                    best_visual = mvis
        if best is None:
            break
        i, j = best
        clusters[i] = sorted(clusters[i] + clusters[j], key=chair_sort_key)
        del clusters[j]

    return sorted(
        [sorted(c, key=chair_sort_key) for c in clusters], key=lambda c: chair_sort_key(c[0])
    )


# ── sheets ───────────────────────────────────────────────────────────────────
def paste_fit(
    canvas: Image.Image,
    image: Image.Image,
    box: tuple[int, int, int, int],
    bg: tuple[int, int, int],
) -> None:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    tile = Image.new("RGB", (w, h), bg)
    im = image.copy()
    im.thumbnail((w - 20, h - 20), Image.Resampling.LANCZOS)
    tile.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    canvas.paste(tile, (x0, y0))


def make_cluster_generation_sheet(image_paths: list[Path], output: Path) -> None:
    """Primary chair view plus a small ordered support strip for image generation."""
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (1024, 1024), (0, 0, 0))
    if not image_paths:
        canvas.save(output, quality=95)
        return
    paste_fit(canvas, oriented_image(image_paths[0]), (132, 44, 892, 760), (8, 8, 8))
    boxes = [(96, 796, 304, 984), (316, 796, 524, 984), (536, 796, 744, 984), (756, 796, 964, 984)]
    for path, box in zip(image_paths[1:5], boxes):
        paste_fit(canvas, oriented_image(path), box, (8, 8, 8))
    canvas.save(output, quality=95)


def make_sheet(
    image_paths: list[Path],
    output: Path,
    title: str | None = None,
    labels: bool = True,
    tile_size: int = 180,
    background: tuple[int, int, int] = (247, 248, 250),
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    thumbs = []
    for path in image_paths:
        im = oriented_image(path)
        im.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (tile_size, tile_size), background)
        canvas.paste(im, ((tile_size - im.width) // 2, (tile_size - im.height) // 2))
        thumbs.append((path, canvas))
    if not thumbs:
        Image.new("RGB", (360, 220), background).save(output, quality=95)
        return

    columns = min(4, len(thumbs))
    rows = int(math.ceil(len(thumbs) / columns))
    top = 34 if title and labels else 0
    label_space = 38 if labels else 0
    gap = 20 if labels else 12
    margin = 10 if labels else 12
    sheet = Image.new(
        "RGB",
        (
            columns * (tile_size + gap) + margin,
            rows * (tile_size + label_space + gap) + top + margin,
        ),
        background,
    )
    draw = ImageDraw.Draw(sheet)
    try:
        font = _shared_font(14)
        title_font = _shared_font(18)
    except Exception:
        font = title_font = ImageFont.load_default()
    if title and labels:
        draw.text((12, 8), title, fill=(23, 32, 42), font=title_font)
    for i, (path, thumb) in enumerate(thumbs):
        row, col = divmod(i, columns)
        x = margin + col * (tile_size + gap)
        y = top + margin + row * (tile_size + label_space + gap)
        sheet.paste(thumb, (x, y))
        if labels:
            draw.rectangle((x, y, x + tile_size, y + tile_size), outline=(211, 218, 228))
            draw.text(
                (x, y + tile_size + 6),
                path.parent.name + " " + path.stem.split("_ranking_")[-1],
                fill=(101, 113, 132),
                font=font,
            )
    sheet.save(output, quality=95)


# ── per-scan clustering ──────────────────────────────────────────────────────
def build_chair_records(
    scan: str, output_root: Path, *, use_dino: bool = False
) -> list[ChairRecord]:
    objects = load_scene_objects(scan)
    scan_dir = config.parsed_images_dir(scan)
    # First pass: gather each chair's crops + metadata (no per-chair model calls yet).
    metas = []
    for obj_dir in sorted(scan_dir.glob("Chair*"), key=lambda p: chair_sort_key(p.name)):
        if not obj_dir.is_dir():
            continue
        images = sorted(obj_dir.glob("frame_*_ranking_*.jpg"), key=natural_key)
        if not images:
            continue
        meta = objects.get(obj_dir.name, {})
        bbox = np.asarray(meta.get("bbox", [1.0, 1.0, 1.0]), dtype=float).reshape(-1)[:3].tolist()
        position = (
            np.asarray(meta.get("position", [0.0, 0.0, 0.0]), dtype=float).reshape(-1)[:3].tolist()
        )
        metas.append((obj_dir, images, bbox, position))

    # The lightweight CV descriptor is sufficient for the normal path. DINOv2 is an
    # explicit enhancement so ingest never loads a large model unexpectedly.
    features = dino_features([images for _, images, _, _ in metas]) if use_dino else None
    if features is None:
        label = "DINOv2 unavailable" if use_dino else "using lightweight CV features"
        print(f"  [chair-cluster] {label}", flush=True)
        features = [image_feature(images) for _, images, _, _ in metas]
    else:
        print(f"  [chair-cluster] DINOv2 embeddings for {len(metas)} chairs", flush=True)

    records = []
    for (obj_dir, images, bbox, position), feature in zip(metas, features):
        sheet_path = output_root / scan / "chair_sheets" / f"{obj_dir.name}.jpg"
        make_sheet(images[:4], sheet_path, obj_dir.name)
        color_feature, dominant_rgb = color_signature(images)
        records.append(
            ChairRecord(
                scan=scan,
                name=obj_dir.name,
                object_dir=obj_dir,
                images=images,
                bbox=bbox,
                position=position,
                sheet_path=sheet_path,
                feature=feature,
                color_feature=color_feature,
                dominant_rgb=dominant_rgb,
                view_score=float(len(images)),
            )
        )
    return records


def choose_representative(cluster: list[str], records_by_name: dict[str, ChairRecord]) -> str:
    def score(name: str) -> float:
        rec = records_by_name[name]
        area_score = 0.0
        for img_path in rec.images[:4]:
            try:
                with Image.open(img_path) as im:
                    area_score += im.size[0] * im.size[1]
            except Exception:
                pass
        return rec.view_score * 1000.0 + area_score / 10000.0

    return max(cluster, key=score)


def cluster_scan(scan: str, output_root: Path, *, use_dino: bool = False) -> dict:
    records = build_chair_records(scan, output_root, use_dino=use_dino)
    names = [r.name for r in records]
    by_name = {r.name: r for r in records}
    pair_lookup = {}
    pair_scores = []
    for i, a in enumerate(records):
        for b in records[i + 1 :]:
            visual = max(0.0, min(1.0, cosine(a.feature, b.feature)))
            color = color_similarity(
                a.color_feature, b.color_feature, a.dominant_rgb, b.dominant_rgb
            )
            dim = dimension_similarity(a.bbox, b.bbox)
            spatial = spatial_similarity(a.position, b.position)  # logged only; not in identity
            linked = pair_link(visual, dim, color)
            merge_score = 0.62 * visual + 0.28 * dim + 0.10 * color
            pair = {
                "a": a.name,
                "b": b.name,
                "visual_similarity": round(visual, 4),
                "color_similarity": round(color, 4),
                "dimension_similarity": round(dim, 4),
                "spatial_similarity": round(spatial, 4),
                "merge_score": round(merge_score, 4),
                "linked": linked,
            }
            pair_lookup[tuple(sorted((a.name, b.name), key=chair_sort_key))] = pair
            pair_scores.append(pair)

    groups, method = _group_chairs(scan, records, names, pair_lookup, output_root)

    clusters = []
    for idx, group in enumerate(groups):
        members = group["members"]
        representative = choose_representative(members, by_name)
        evidence = []
        # Small ordered evidence set — too many crops make the image model treat the
        # input as a collage instead of one furniture target.
        for member in members:
            evidence.extend(by_name[member].images[:2])
        evidence = evidence[:5]
        sheet_path = output_root / scan / "cluster_sheets" / f"ChairCluster{idx}.jpg"
        sheet_path = output_root / scan / "input2imagegen" / f"ChairCluster{idx}.jpg"
        make_sheet(evidence, sheet_path, f"{scan} ChairCluster{idx}: {', '.join(members)}")
        make_cluster_generation_sheet(evidence, sheet_path)
        clusters.append(
            {
                "cluster_id": f"ChairCluster{idx}",
                "representative": representative,
                "members": members,
                "member_count": len(members),
                "cluster_sheet": str(sheet_path.relative_to(output_root)),
                "input2imagegen_sheet": str(sheet_path.relative_to(output_root)),
                "member_colors": {member: by_name[member].dominant_rgb for member in members},
                # type attributes from the judgment layer (empty on the embedding fallback path)
                "type_signature": group.get("signature", {}),
                "judge_confidence": group.get("confidence"),
                "judge_notes": group.get("notes"),
                "reference_render": None,
            }
        )
    return {
        "scan": scan,
        "category": "Chair",
        "chair_count": len(records),
        "grouping_method": method,
        "clusters": clusters,
        "pair_scores": pair_scores,
    }


def _group_chairs(
    scan: str,
    records: list[ChairRecord],
    names: list[str],
    pair_lookup: dict[tuple[str, str], dict],
    output_root: Path,
) -> tuple[list[dict], str]:
    """Decide the chair groups. Primary path: the Opus 4.8 judgment layer (groups by material /
    arms / colour / base). Falls back to the deterministic DINOv2 average-linkage clustering when
    the judge is disabled, unavailable, or errors. Returns ``(groups, method)`` where each group is
    ``{"members": [...], "signature": {...}, "confidence": str|None, "notes": str|None}`` sorted by
    first-member natural order."""
    def _sorted(groups: list[dict]) -> list[dict]:
        for g in groups:
            g["members"] = sorted(g["members"], key=chair_sort_key)
        return sorted(groups, key=lambda g: chair_sort_key(g["members"][0]))

    if USE_JUDGE and chair_type_judge.available():
        try:
            chairs = [(r.name, r.images) for r in records]
            montage_dir = output_root / scan / "chair_type_montages"
            groups = chair_type_judge.judge_types(chairs, montage_dir)
            return _sorted(groups), f"judge:{chair_type_judge.DEFAULT_JUDGE_MODEL}"
        except Exception as exc:  # noqa: BLE001 — never fail init on the judge; fall back
            print(f"  [chair-judge] failed ({exc}); falling back to embedding clustering", flush=True)
    elif USE_JUDGE:
        print("  [chair-judge] claude_agent_sdk unavailable; using embedding clustering", flush=True)

    groups = [{"members": m, "signature": {}, "confidence": None, "notes": None}
              for m in agglomerate_clusters(names, pair_lookup)]
    return _sorted(groups), "embedding:dinov2-avg-linkage"


def generate_for_cluster(
    scan: str, cluster: dict, output_root: Path, model: str, force: bool
) -> None:
    sheet = output_root / (cluster.get("input2imagegen_sheet")
                       or cluster.get("gemini_input_sheet")  # pre-rename trees
                       or cluster["cluster_sheet"])
    out = output_root / scan / "generated" / f"{cluster['cluster_id']}.png"
    legacy_out = output_root / scan / "nano_banana" / f"{cluster['cluster_id']}.png"
    existing = out if out.exists() else legacy_out
    if existing.exists() and not force:
        cluster["reference_render"] = str(existing.relative_to(output_root))
        return
    colors = list(cluster.get("member_colors", {}).values())
    avg_color = (
        [int(np.mean([rgb[i] for rgb in colors])) for i in range(3)] if colors else [80, 80, 80]
    )
    prompt = (
        "The input image is a visual-only evidence sheet for one repeated chair type from the "
        "same room scan. The large upper tile is the primary chair view. The smaller bottom "
        "tiles are secondary evidence only; use them to verify hidden sides and structural "
        "details, not as separate chairs. Some crops are noisy, partially occluded, or include "
        "nearby table, floor, room background, or other chairs. Infer the shared chair only from "
        "consistent details across the tiles. Preserve the actual chair design exactly when visible: "
        "overall silhouette, backrest shape, seat shape, armrests if present, number and type of "
        "legs or pedestal/base, support bars, proportions, material, and dominant color close to "
        f"RGB {avg_color}. Create one clean object-only 3D-asset reference render of the chair, "
        "upright, centered, fully visible from top to bottom, in a front-biased three-quarter view "
        "about 20 degrees above the horizon. Remove unrelated clutter and any nearby table, floor, "
        "walls, other chairs, people, shadows, reflections, labels, text, borders, and collage layout. "
        "Do not remove attached chair parts. Do not invent decorative details, extra legs, cushions, "
        "arms, wheels, or supports not supported by the references. Use a solid pure black background "
        "edge-to-edge."
    )
    prompt_path = output_root / scan / "prompts" / f"{cluster['cluster_id']}.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt, encoding="utf-8")
    cluster["reference_prompt"] = str(prompt_path.relative_to(output_root))

    try:
        image_model.generate_reference(
            sheet,
            prompt,
            out,
            model=model,
            force=force,
            response_dir=output_root / scan / "generated" / "_responses",
        )
        cluster["reference_render"] = str(out.relative_to(output_root))
        normalized = output_root / scan / "references" / f"{cluster['cluster_id']}.png"
        image_model.normalize_reference(out, normalized)
        cluster["reference_1024"] = str(normalized.relative_to(output_root))
        cluster["reference_status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        cluster["reference_status"] = "error"
        cluster["reference_error"] = str(exc)


def cluster_and_generate(
    scan: str,
    output_root: Path,
    *,
    skip_image_generation: bool = False,
    force_image_generation: bool = False,
    model: str = config.DEFAULT_IMAGE_MODEL,
    use_dino: bool = False,
) -> dict:
    output_root = Path(output_root)
    result = cluster_scan(scan, output_root, use_dino=use_dino)
    print(
        f"  grouped {result['chair_count']} chairs -> {len(result['clusters'])} clusters",
        flush=True,
    )
    if not skip_image_generation:
        # One ~45s hosted image call per cluster, each writing only into its own cluster dict.
        for cluster in result["clusters"]:
            print(f"  [image] {cluster['cluster_id']} {cluster['members']}", flush=True)
        clusters = result["clusters"]
        with ThreadPoolExecutor(max_workers=image_model.image_workers(len(clusters))) as pool:
            list(pool.map(
                lambda c: generate_for_cluster(scan, c, output_root, model, force_image_generation),
                clusters,
            ))

    scan_dir = output_root / scan
    scan_dir.mkdir(parents=True, exist_ok=True)
    (scan_dir / "chair_clusters.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
