"""Stage 2 — scene data -> per-object cropped evidence images.

Ported from the v2 ``step2 cropping``. Drives the vendored preprocessing to crop
each RoomPlan object out of the capture frames (ranked, good-only), stitch them,
and prepare the camera metadata. Writes, relative to the work root:

    input/parsed_images/<scan>/<Object>/frame_*_ranking_*.jpg   # the crops
    input/parsed_images/<scan>/<Object>/camera/frame_*.json
    input/object_stage/<scan>/...
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import time
import uuid

from litereality_agent.pipeline.scene_init import paths as config
from litereality_agent.pipeline.scene_init.ingest.preprocessing.object_images import (
    Colors,
    prepare_camera_data_for_retrieval,
    process_all_folders,
    process_object_images,
)

# Written beside the crops; records the inputs they were built from. Every reader of
# parsed_images/<scan>/ iterates directories only, so a file here is inert.
STAMP = ".crop_inputs.json"


def _evidence_objects(objects):
    """Read immutable scan geometry while preserving current merge/split identities."""
    result = copy.deepcopy(objects)
    for entry in result:
        entry.update(entry.pop("evidence_geometry", {}))
    return result


def _input_fingerprint(scan: str, include_walls: bool) -> dict:
    """Everything the crops are a function of.

    Crops are NOT a pure function of the capture, so "the directory exists" is not a safe reuse
    test. `box_merge` runs between extract and crop and REWRITES objects.pkl — fusing
    interpenetrating counter runs into one unit — and it also sets $LR_ENLARGED_CROP_OBJECTS,
    which changes the crop rectangle for the ids it merged. Reusing on existence alone would hand
    the next stage crops built against a different object set. Key on the inputs instead, so a
    merge (or a re-extract, or a flag change) invalidates them and an unchanged scan does not.
    """
    scene_dir = config.scene_data_dir(scan)
    digests = {}
    for name in ("walls", "objects", "wall_holes", "floor"):
        path = scene_dir / f"{name}.pkl"
        if name == "objects" and path.is_file():
            with path.open("rb") as handle:
                evidence = _evidence_objects(pickle.load(handle))
            data = json.dumps(evidence, sort_keys=True, default=lambda v: v.tolist()).encode()
            digests[name] = hashlib.sha256(data).hexdigest()
        else:
            digests[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    capture = config.input_root() / "rgbd" / scan
    return {
        "scene_data": digests,
        "capture": [(str(p.relative_to(capture)), p.stat().st_size, p.stat().st_mtime_ns)
                    for folder in ("image", "intrinsic", "extrinsic", "depth", "confidence")
                    for p in sorted((capture / folder).glob("*")) if p.is_file()],
        "include_walls": bool(include_walls),
        "enlarged": sorted(filter(None, os.environ.get("LR_ENLARGED_CROP_OBJECTS", "").split(","))),
    }


def crops_current(scan: str, include_walls: bool = False) -> bool:
    """True when the crops on disk were built from exactly the inputs present now.

    The counterpart of `config.scene_data_complete` for the crop stage: it lets the stage decide
    for itself whether to reuse, rather than depending on a caller remembering to pass a flag.
    """
    parsed = config.parsed_images_dir(scan)
    stamp = parsed / STAMP
    if not stamp.is_file() or not any(child.is_dir() for child in parsed.iterdir()):
        return False
    try:
        recorded = json.loads(stamp.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    # JSON turns tuples into lists; compare JSON-native values on both sides.
    return recorded.get("inputs", recorded) == json.loads(json.dumps(
        _input_fingerprint(scan, include_walls)))


def _load_scene_data(scan: str):
    scene_dir = config.scene_data_dir(scan)
    required = {
        name: scene_dir / f"{name}.pkl" for name in ("walls", "objects", "wall_holes", "floor")
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing scene data (run extract_scene first): " + ", ".join(missing)
        )
    loaded = []
    for path in required.values():
        with path.open("rb") as f:
            loaded.append(pickle.load(f))
    loaded[1] = _evidence_objects(loaded[1])
    return loaded  # walls, objects, wall_holes, floor


def crop(scan: str, include_walls: bool = False) -> None:
    """Crop every object's evidence images. Assumes CWD == work root."""
    started = time.time()
    walls, objects, wall_holes, _floor = _load_scene_data(scan)
    # Fingerprint the inputs BEFORE the work, so the stamp records what was actually cropped.
    fingerprint = _input_fingerprint(scan, include_walls)
    parsed = config.parsed_images_dir(scan)
    if parsed.exists():
        history = parsed.parent / '.crop-history' / scan / uuid.uuid4().hex
        history.parent.mkdir(parents=True, exist_ok=True)
        parsed.rename(history)
    # Deleted/renamed merge members must not survive as stale crop directories.

    print(f"{Colors.BLUE}[crop]{Colors.RESET} processing object images...")
    process_object_images(scan, walls, objects, wall_holes, include_walls=include_walls)

    print(f"{Colors.BLUE}[crop]{Colors.RESET} stitching...")
    process_all_folders(f"input/parsed_images/{scan}")

    print(f"{Colors.BLUE}[crop]{Colors.RESET} preparing camera data...")
    prepare_camera_data_for_retrieval(scan)

    # Last, so an interrupted crop leaves no stamp and the next run redoes it.
    parsed.mkdir(parents=True, exist_ok=True)
    (parsed / STAMP).write_text(json.dumps({"inputs": fingerprint, "generation": uuid.uuid4().hex},
                                         indent=2), encoding="utf-8")

    print(
        f"{Colors.GREEN}✓{Colors.RESET} crops in input/parsed_images/{scan} ({time.time() - started:.2f}s)"
    )
