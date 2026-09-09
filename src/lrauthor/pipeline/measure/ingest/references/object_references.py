"""Per-object crops → one clean hosted reference image per object.

Ported from the v2 ``step2 references``. For each non-chair object it scores the
crops, builds a weighted evidence sheet (one dominant view + a support strip),
asks the configured OpenAI image model for a clean object-only render, and
normalizes it onto a 1024² black canvas. Chairs are handled by
:mod:`object_init.chair_clusters` (they need grouping first). Writes:

    <output_root>/<scan>/<Object>/input2imagegen.jpg      # evidence sheet sent to the image model
    <output_root>/<scan>/<Object>/clean_obj_reference.png # the raw generation
    <output_root>/<scan>/<Object>/reference_1024.png   # normalized reference
    <output_root>/<scan>/<Object>/{prompt.txt,reference_meta.json}
    <output_root>/object_reference_manifest.json, index.html
"""

from __future__ import annotations

import json
import math
import pickle
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from lrauthor.pipeline.measure import paths as config
from lrauthor.pipeline.measure.ingest.references import image_gen as image_model
from lrauthor.pipeline.measure.ingest.references.image_gen import image_workers

# Per-category guidance: the specific structure to preserve when generating the
# clean reference. Keyed by the lowercased category (digits stripped from the id).
CATEGORY_HINTS = {
    "table": "the tabletop shape (round/oval/rectangular) and thickness; the EXACT support type and count — single pedestal vs 3/4 legs vs trestle vs panel ends; apron and stretcher bars; edge profile; leg cross-section (round/square/tapered); the material and finish of the top vs the base (wood, laminate, metal); remove everything resting on the tabletop (cups, papers, laptops, decor)",
    "desk": "the worktop shape and thickness; the leg or pedestal frame and count; any drawer pedestal or modesty panel; cable grommets; whether it is a sit-stand frame; the materials of top vs frame; remove all items on the desk surface",
    "storage": "the overall cabinet/shelf body proportions; the EXACT count and layout of shelves, drawers, and doors; hinged vs sliding doors; open vs closed compartments; handle/knob style and position; plinth/base or wall-mount; the back panel; the material and color of the carcass vs the fronts; remove loose contents and clutter on top or inside",
    "cabinet": "the cabinet body proportions; the exact number and layout of doors and drawers; hinged vs sliding; handles/knobs; legs/plinth or wall-mount; material and color of carcass vs fronts; remove clutter",
    "cupboard": "the cupboard body; door count and split; handles; shelves if open; plinth or wall-mount; material and color; remove contents/clutter",
    "shelf": "the number of shelves and their spacing; open frame vs backed; uprights and brackets; any doors or drawers at the base; material; remove all shelved items",
    "bookshelf": "the number of shelves and spacing; side panels and back; any base cabinet; material; remove the books and contents, keep the empty shelf structure",
    "wardrobe": "the tall cabinet body; door count and type (hinged/sliding/mirrored); handles; plinth; material and color; remove clutter",
    "sofa": "the seat count and cushion divisions (seat + back cushions); armrest shape and height; backrest profile; the base/legs or skirted base; piping and seams; upholstery material and color; keep attached cushions, remove only loose throw pillows/blankets that are clutter",
    "chair": "the backrest shape and height; seat shape; armrests if present; the EXACT base/leg type — 4 legs vs sled vs cantilever vs 5-star caster base vs pedestal; gas-lift/swivel for an office chair; upholstered vs hard shell; material and color",
    "stool": "the seat shape; the leg type and count; any footrest ring or gas-lift base; height; material",
    "bench": "the seat slab; the leg or support type and count; a back if present; material",
    "bed": "the mattress and base proportions; headboard shape and height; footboard if present; the frame and legs; remove loose bedding clutter but keep the basic mattress and headboard form",
    "nightstand": "the small cabinet body; drawer/door count; handles; legs or plinth; material and color; remove items on top",
    "sink": "the basin shape and count (single/double bowl); the faucet/tap form and mounting (only if part of the fixture); a drainer board; the surrounding counter or cabinet only if it belongs to the unit; material (steel/ceramic) and color; remove dishes, soap, and clutter",
    "sink storage": "this is ONE connected fixture — a sink integrated INTO a base storage cabinet. Render the WHOLE unit from the countertop down to the floor, NOT just the sink: the sink basin(s) and any faucet/tap on top, AND the base cabinet directly below it — its doors and/or drawers, handles/knobs, toe-kick/plinth, and carcass sides. Keep the sink and cabinet as a single attached unit. Material of the sink (steel/ceramic) and of the cabinet fronts and color; remove dishes, soap, and loose clutter",
    "counter": "the countertop slab shape and thickness; the supporting cabinets or legs only if part of the unit; edge profile; material; remove items on the counter",
    "television": "the screen rectangle and aspect; bezel thickness; the stand/feet type or wall mount; the black/glossy panel material; remove any reflected room content on the screen and any media devices around it",
    "dishwasher": "the rectangular appliance body; the front door panel and handle or recess; the control strip and buttons; the toe-kick; integrated vs freestanding; stainless or white finish",
    "refrigerator": "the body proportions; the door split (single/double/French, freezer on top or bottom); handles; a water/ice dispenser if present; the finish and color; the toe grille",
    "oven": "the appliance body; the door with its window; control knobs and panel; the handle; built-in vs freestanding; the finish",
    "microwave": "the box body; the door with window and its handle or release button; the control panel; the finish",
    "washer": "the appliance body; the round door or top lid; the control panel and dispenser drawer; the finish",
    "lamp": "the base, the stem or arm, and the shade shapes; whether it is a floor/table/wall lamp; the material and color of each part",
    "whiteboard": "the flat board rectangle; the frame; the pen tray; wall-mount; keep the board blank and clean",
    "mirror": "the reflective rectangle/oval; the frame profile and material; do not render any reflected room content — keep the glass a plain neutral reflective surface",
    "plant": "the pot shape and material and the plant's overall silhouette and foliage; keep it natural",
}

# What habitually surrounds a category in a real room, named so the generator can be told to drop it.
# Every evidence tile for a meeting table shows chairs, so the generator reads them as part of the
# subject and fuses them in — and because the pipeline reconstructs chairs separately (chair
# clusters), the scene then gets both: eight real chairs placed, six more welded to the tabletop.
# The generic "other furniture" in the removal list does not survive that much visual evidence;
# naming the specific intruder does.
CATEGORY_COMPANIONS = {
    "table": "chairs, stools, benches",
    "desk": "chairs, stools, monitors, desk lamps",
    "counter": "stools, chairs",
    "sink": "stools, chairs",
    "sink storage": "stools, chairs",
    "sofa": "coffee tables, side tables, floor lamps, rugs",
    "bed": "nightstands, benches, rugs, lamps",
    "chair": "tables, desks, and any other chair",
    "stool": "tables, counters, and any other stool",
    "bench": "tables, and any other bench",
    "television": "TV stands, media consoles, soundbars, shelves",
    "nightstand": "beds, lamps",
    "whiteboard": "tables, chairs",
    "mirror": "vanities, sinks, cabinets",
    "lamp": "tables, nightstands, sofas",
    "plant": "tables, shelves, plant stands",
}


def natural_key(path: Path) -> list[object]:
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", path.name)]


def category_from_name(name: str) -> str:
    return re.sub(r"\d+", "", name).replace("_", " ").strip().lower()


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


def extract_rank(path: Path) -> int:
    match = re.search(r"_ranking_(\d+)", path.name)
    return int(match.group(1)) if match else 999


def load_camera_json(image_path: Path) -> dict:
    meta_path = image_path.parent / "camera" / image_path.with_suffix(".json").name
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        return {}


def front_score(meta: dict, object_meta: dict) -> float:
    pose = np.array(meta.get("pose", []), dtype=np.float32)
    obj_pos = np.array(object_meta.get("position", []), dtype=np.float32).reshape(-1)
    if pose.shape != (4, 4) or obj_pos.shape[0] < 3:
        return 0.5
    camera_pos = pose[:3, 3]
    to_camera = camera_pos[[0, 2]] - obj_pos[[0, 2]]
    norm = np.linalg.norm(to_camera)
    if norm < 1e-5:
        return 0.5
    to_camera = to_camera / norm
    rotation_deg = float(object_meta.get("rotation", 0.0))
    theta = math.radians(rotation_deg)
    front = np.array([math.sin(theta), math.cos(theta)], dtype=np.float32)
    dot = float(np.clip(np.dot(front, to_camera), -1.0, 1.0))
    angle = math.degrees(math.acos(dot))
    return float(math.exp(-((angle - 25.0) ** 2) / (2.0 * 38.0 * 38.0)))


def score_crop(image_path: Path, object_meta: dict, max_area: float) -> dict:
    meta = load_camera_json(image_path)
    rank = extract_rank(image_path)
    with Image.open(image_path) as im:
        width, height = im.size
    area = float(width * height)
    area_score = math.sqrt(area / max(max_area, 1.0))
    rank_score = 1.0 / (1.0 + rank)
    aspect = max(width / max(height, 1), height / max(width, 1))
    aspect_score = math.exp(-max(0.0, aspect - 2.4) / 2.0)
    front = front_score(meta, object_meta)
    total = 0.36 * rank_score + 0.30 * area_score + 0.22 * front + 0.12 * aspect_score
    return {
        "image": str(image_path),
        "camera_json": str(image_path.parent / "camera" / image_path.with_suffix(".json").name),
        "rank": rank,
        "width": width,
        "height": height,
        "area_score": round(area_score, 4),
        "rank_score": round(rank_score, 4),
        "front_score": round(front, 4),
        "aspect_score": round(aspect_score, 4),
        "evidence_score": round(total, 4),
    }


def select_evidence_images(object_dir: Path, object_meta: dict, max_images: int) -> list[dict]:
    images = sorted(object_dir.glob("frame_*_ranking_*.jpg"), key=natural_key)
    if not images:
        return []
    areas = []
    for image_path in images:
        with Image.open(image_path) as im:
            areas.append(im.size[0] * im.size[1])
    max_area = float(max(areas) if areas else 1.0)
    scored = [score_crop(path, object_meta, max_area) for path in images]
    scored.sort(key=lambda item: item["evidence_score"], reverse=True)
    return scored[:max_images]


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
    im.thumbnail((w - 24, h - 24), Image.Resampling.LANCZOS)
    tile.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    canvas.paste(tile, (x0, y0))


def make_weighted_input_sheet(scored: list[dict], output: Path) -> None:
    """One dominant view plus a small support strip — a product-reference sheet."""
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (1024, 1024), (0, 0, 0))
    if not scored:
        canvas.save(output)
        return
    primary = oriented_image(Path(scored[0]["image"]))
    paste_fit(canvas, primary, (132, 44, 892, 760), (8, 8, 8))
    boxes = [(96, 796, 304, 984), (316, 796, 524, 984), (536, 796, 744, 984), (756, 796, 964, 984)]
    for item, box in zip(scored[1:5], boxes):
        paste_fit(canvas, oriented_image(Path(item["image"])), box, (8, 8, 8))
    canvas.save(output, quality=95)


def first_metadata_value(object_meta: dict, *keys: str) -> object:
    for key in keys:
        if key in object_meta and object_meta[key] is not None:
            return object_meta[key]
    return []


def object_context(object_meta: dict) -> str:
    bbox = first_metadata_value(object_meta, "bbox", "size")
    position = first_metadata_value(object_meta, "position")
    parts = []
    try:
        dims = [float(v) for v in np.asarray(bbox).reshape(-1)[:3]]
        if len(dims) == 3:
            parts.append(
                f"approximate scan dimensions xyz={dims[0]:.2f},{dims[1]:.2f},{dims[2]:.2f} meters"
            )
    except Exception:
        dims = []
    try:
        pos = [float(v) for v in np.asarray(position).reshape(-1)[:3]]
        if len(pos) == 3 and len(dims) == 3 and pos[1] - dims[1] / 2.0 > 0.18:
            parts.append("scan placement suggests this may be wall-mounted or wall-hanging")
    except Exception:
        pass
    return "; ".join(parts)


def build_prompt(object_name: str, category: str, object_meta: dict, evidence_count: int) -> str:
    hints = CATEGORY_HINTS.get(
        category, "object silhouette, proportions, structure, material, and color"
    )
    context = object_context(object_meta)
    context_sentence = f" Scan metadata: {context}." if context else ""
    companions = CATEGORY_COMPANIONS.get(category)
    # Stated twice, at both ends: the removal list in the middle of the prompt loses to three tiles
    # that all show the object ringed by its companions.
    solo = (
        f" RENDER EXACTLY ONE OBJECT — the {category} alone, nothing else in frame."
        + (f" The evidence almost certainly shows {companions} around it; those are separate objects "
           "that this pipeline captures and rebuilds on their own, so drawing any of them here puts "
           "a duplicate in the scene. Render none of them, not even partially or in shadow." if companions else "")
    )
    return (
        f"The input image is a visual-only evidence sheet for one scanned object: {object_name}, "
        f"category {category}. The large upper tile is the primary view and should define the "
        f"object identity and front-biased orientation. The {max(0, evidence_count - 1)} smaller "
        "bottom tiles are secondary evidence only; use them to verify hidden sides and structural "
        "details, not as separate objects. Some views may be noisy, partially occluded, or contain "
        f"nearby room content.{context_sentence} Create one clean object-only 3D-asset reference "
        f"render of the furniture/object itself.{solo} Use a front-biased three-quarter view, upright and "
        "centered, with the full object visible from top to bottom. Preserve the real object type: "
        "if it is wall-mounted or wall-hanging, keep it wall-mounted/wall-hanging and do not turn it "
        "into a freestanding cabinet; if it is freestanding, keep its base/legs/supports. Preserve "
        "structural facts exactly when visible: number of drawers, doors, shelves, handles, legs, panels, "
        "cushions, and major openings. For this category, preserve: "
        f"{hints}. Remove loose clutter and unrelated items sitting on, inside, beside, or in front of "
        "the furniture, including books, cups, bags, boxes, electronics, decor, papers, room background, "
        "floor, wall, other furniture, labels, text, borders, and the collage layout. Do not remove parts "
        "that are physically attached to the furniture. Do not invent decorative details or extra drawers, "
        "shelves, legs, handles, or objects not supported by the references. "
        "STRUCTURE FIDELITY: reproduce the object's overall construction and the EXACT COUNT of every repeated "
        "element the evidence shows (drawers, doors, shelves, legs, panels, cushions, sink bowls). Do NOT drop, "
        "merge, omit, or simplify a real feature — never turn a panelled, glazed, or multi-drawer piece into a "
        "plain box, and never reduce a 3-leg/4-leg/pedestal base to a different count. Match reality: neither add "
        f"nor remove structure. Use a solid pure black background edge-to-edge. Final check before you render: "
        f"exactly one {category} fills the frame and no other object appears anywhere in the image."
    )


def make_skip_image_generation_reference(source: Path, destination: Path, canvas_size: int = 1024) -> None:
    image = Image.open(source).convert("RGB")
    image.thumbnail((canvas_size, canvas_size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (canvas_size, canvas_size), (0, 0, 0))
    canvas.paste(image, ((canvas_size - image.width) // 2, (canvas_size - image.height) // 2))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def object_dirs(
    scan: str, only: set[str] | None, include_openings: bool, include_chairs: bool
) -> list[Path]:
    dirs = []
    scan_dir = config.parsed_images_dir(scan)
    if not scan_dir.exists():
        return dirs
    for child in sorted(scan_dir.iterdir(), key=natural_key):
        if not child.is_dir():
            continue
        if only and child.name not in only:
            continue
        if child.name == "Floor":
            continue
        if child.name.startswith("Chair") and not include_chairs:
            continue
        if re.match(r"^Wall\d+$", child.name):
            continue
        # openings (Wall*_Door/Window*, Opening*_Door/Window*) -> handled by opening_references
        if re.search(r"_(Door|Window)_", child.name) and not include_openings:
            continue
        if list(child.glob("frame_*_ranking_*.jpg")):
            dirs.append(child)
    return dirs


def generate_for_scan(
    scan: str,
    output_root: Path,
    *,
    skip_image_generation: bool = False,
    force_image_generation: bool = False,
    model: str = config.DEFAULT_IMAGE_MODEL,
    max_images: int = 5,
    only: set[str] | None = None,
    include_openings: bool = False,
    include_chairs: bool = False,
) -> dict:
    output_root = Path(output_root)
    object_meta = load_scene_objects(scan)
    scan_result = {"scan": scan, "objects": []}

    def generate_one(obj_dir: Path) -> dict | None:
        name = obj_dir.name
        category = category_from_name(name)
        meta = object_meta.get(name, {})
        scored = select_evidence_images(obj_dir, meta, max_images)
        if not scored:
            return None

        out_dir = output_root / scan / name
        input_sheet = out_dir / config.INPUT_SHEET
        raw = out_dir / config.CLEAN_REFERENCE
        normalized = out_dir / "reference_1024.png"
        prompt_path = out_dir / "prompt.txt"
        meta_path = out_dir / "reference_meta.json"

        make_weighted_input_sheet(scored, input_sheet)
        prompt = build_prompt(name, category, meta, len(scored))
        out_dir.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(prompt, encoding="utf-8")

        status = "skipped_image_generation"
        error = None
        image_meta = None
        if skip_image_generation:
            make_skip_image_generation_reference(input_sheet, normalized)
        else:
            try:
                image_meta = image_model.generate_reference(
                    input_sheet,
                    prompt,
                    raw,
                    model=model,
                    force=force_image_generation,
                    response_dir=out_dir / "_responses",
                )
                image_model.normalize_reference(raw, normalized)
                status = "ok"
            except Exception as exc:  # noqa: BLE001
                status = "error_fallback"
                error = str(exc)
                make_skip_image_generation_reference(input_sheet, normalized)

        record = {
            "scan": scan,
            "object": name,
            "category": category,
            "status": status,
            "input2imagegen": str(input_sheet.relative_to(output_root)),
            "raw_reference": str(raw.relative_to(output_root)),
            "reference_1024": str(normalized.relative_to(output_root)),
            "prompt": str(prompt_path.relative_to(output_root)),
            "evidence": scored,
        }
        if error:
            record["error"] = error
        if image_meta:
            record["image_meta"] = image_meta
        meta_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"  [{status}] object {name}", flush=True)
        return record

    # One hosted image call per object, ~45s each and fully independent — this loop was serial and
    # was most of the ingest stage's wall time. `map` keeps the manifest order deterministic.
    targets = object_dirs(scan, only, include_openings, include_chairs)
    with ThreadPoolExecutor(max_workers=image_workers(len(targets))) as pool:
        scan_result["objects"] = [r for r in pool.map(generate_one, targets) if r is not None]

    (output_root / scan).mkdir(parents=True, exist_ok=True)
    (output_root / scan / "object_references.json").write_text(
        json.dumps(scan_result, indent=2), encoding="utf-8"
    )
    return scan_result
