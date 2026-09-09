"""Complexity router — decide procedural vs TRELLIS per static object.

For every clean reference image object_init produced, ask a VLM whether the
object's geometry is simple/regular enough to build **procedurally** (primitives:
boxes, cylinders, revolved/extruded profiles + PBR — exact dimensions, clean,
editable) or whether it is organic/complex enough to need **TRELLIS** (neural
image-to-3D). Prior: chairs, sofas, and abstract/organic objects → trellis;
regular furniture and appliances with box/slab/revolve geometry → procedural.

Writes, under the work root:
    routing/<scan>/routing.json          per-scan decisions
    routing/routing_manifest.json        combined
and prints the split. Downstream: trellis/generate_static.py --route trellis only
builds the complex ones; the procedural ones go to the procedural generator.

Run (from repo root):
    python -m litereality_agent.pipeline.author.reconstruct.classify.classify_complexity                       # all scans
    python -m litereality_agent.pipeline.author.reconstruct.classify.classify_complexity --scan tea_room
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from litereality_agent.pipeline.author.reconstruct.classify import classification
from litereality_agent.pipeline.measure import paths as config

# Each worker is one hosted vision call, so the ceiling is the provider's rate limit rather than
# local CPU. Capped rather than unbounded so a 40-object room does not open 40 sessions at once.
MAX_CLASSIFY_WORKERS = int(os.environ.get("LR_CLASSIFY_WORKERS", "8"))

ROUTE_SCHEMA = {
    "type": "object",
    "properties": {
        "route": {"type": "string", "enum": ["procedural", "trellis"]},
        "complexity": {"type": "string", "enum": ["low", "medium", "high"]},
        "is_organic": {"type": "boolean"},
        "primitive_buildable": {"type": "boolean"},
        "reason": {"type": "string"},
        "primitive_plan": {"type": "string"},
    },
    "required": ["route", "complexity", "is_organic", "primitive_buildable", "reason"],
}

PROMPT_TEMPLATE = """You are deciding HOW to generate a 3D asset for a scanned indoor object.

The image is a clean, object-only reference render of a single {category} (approx real-world dimensions W x H x D = {dims} meters).

Choose one of two generation methods:

- "procedural": build it in Blender from PRIMITIVES (boxes, cylinders, revolved or extruded profiles) plus PBR materials. Pick this when the geometry is SIMPLE and REGULAR — box-like, slab/panel, or a simple revolve/extrude. Good for: rectangular tables, desks, cabinets, shelves, storage units, fridges, dishwashers, ovens, flat-panel TVs, simple counters/sinks, boxes, plain benches. Procedural wins here: exact dimensions, clean topology, editable, lightweight.

- "trellis": neural image-to-3D. Pick this when the geometry is ORGANIC, CURVED, SCULPTED, or visually COMPLEX so that a few primitives cannot capture it well. Good for: sofas and upholstered/ergonomic chairs, anything with soft cushions or molded shells, plants, decorative or abstract objects, complex assemblies.

Strong prior: chairs, sofas, and abstract/organic objects almost always need "trellis". Regular furniture and box/slab/revolve appliances should be "procedural".

Judge from the actual shape in the image, not just the category name. Return JSON with: route, complexity (low/medium/high), is_organic, primitive_buildable, a one-sentence reason, and (only if route=procedural) a short primitive_plan describing the primitives to build it from."""


def _dims_str(meta: dict) -> str:
    bbox = meta.get("bbox")
    if bbox is None:
        bbox = meta.get("size")
    try:
        import numpy as np

        d = [float(v) for v in np.asarray(bbox).reshape(-1)[:3]]
        if len(d) == 3:
            return f"{d[0]:.2f} x {d[1]:.2f} x {d[2]:.2f}"
    except Exception:
        pass
    return "unknown"


def load_scene_objects(scan: str) -> dict[str, dict]:
    path = config.scene_data_dir(scan) / "objects.pkl"
    if not path.exists():
        return {}
    with path.open("rb") as f:
        objects = pickle.load(f)
    out = {}
    for obj in objects:
        name = obj.get("mesh_id") or obj.get("object_type")
        if name:
            out[name] = obj
    return out


def collect(work: Path, scans: list[str] | None, include_chairs: bool):
    """Yield (scan, kind, name, ref_png, meta) for each reference."""
    obj_root = work / "object_refs"
    # object clustering now lives under run/object_clustering/<scan>/ (see config), not the
    # per-scan work tree — resolve through config so this reader tracks the writer's location.
    chair_root = config.chair_clusters_root()
    found = sorted({p.name for p in obj_root.iterdir() if p.is_dir()}) if obj_root.exists() else []
    for scan in scans or found:
        meta_by_name = load_scene_objects(scan)
        for obj_dir in sorted((obj_root / scan).glob("*/")) if (obj_root / scan).exists() else []:
            ref = obj_dir / "reference_1024.png"
            if ref.exists():
                yield scan, "objects", obj_dir.name, ref, meta_by_name.get(obj_dir.name, {})
        if include_chairs:
            clusters_json = chair_root / scan / "chair_clusters.json"
            cl_dir = chair_root / scan / "references"
            if not cl_dir.is_dir():
                cl_dir = chair_root / scan / "nano_banana_normalized_1024"
            reps = {}
            if clusters_json.exists():
                data = json.loads(clusters_json.read_text())
                reps = {c["cluster_id"]: c.get("representative") for c in data.get("clusters", [])}
            for ref in sorted(cl_dir.glob("ChairCluster*.png")) if cl_dir.exists() else []:
                rep_meta = meta_by_name.get(reps.get(ref.stem, ""), {})
                yield scan, "chairs", ref.stem, ref, rep_meta


def category_from_name(name: str, kind: str) -> str:
    if kind == "chairs":
        return "chair"
    import re

    return re.sub(r"\d+", "", name).replace("_", " ").strip().lower() or "object"


# User policy: chairs, sofas, and abstract/organic objects always go to TRELLIS
# (interaction-relevant + shape fidelity matters), regardless of the VLM call.
# Everything else follows the VLM (regular box/slab/revolve geometry -> procedural).
ORGANIC_CATEGORIES = {"sofa", "couch", "armchair", "chair", "plant", "person"}


def apply_policy(kind: str, category: str, decision: dict) -> dict:
    forced = (
        kind == "chairs"
        or any(c in category for c in ORGANIC_CATEGORIES)
        or bool(decision.get("is_organic"))
    )
    decision = dict(decision)
    decision["route_vlm"] = decision.get("route")
    if forced and decision.get("route") != "trellis":
        decision["route"] = "trellis"
        decision["forced_by_policy"] = "chairs/sofas/abstract -> trellis"
    return decision


def classify_for_scan(scan: str, *, model: str | None = None, include_chairs: bool = True) -> dict:
    """Route every reference of one scan, writing routing/ under its work root.

    Sets the scan context, reads ``object_refs`` + ``chair_clusters`` from the scan's
    per-scan work tree, and writes ``routing/routing.json`` and a single-scan
    ``routing/routing_manifest.json`` (the format trellis/procedural stages read).
    """
    from litereality_agent import telemetry

    config.set_scan(scan)
    model = model or classification.DEFAULT_CLASSIFY_MODEL
    work = config.work_root()
    routing_root = work / "routing"

    items = [it for it in collect(work, [scan], include_chairs)]
    print(f"[classify] {scan}: {len(items)} object(s)...", flush=True)

    def classify_one(item) -> dict:
        _scan, kind, name, ref, meta = item
        category = category_from_name(name, kind)
        dims = _dims_str(meta)
        prompt = PROMPT_TEMPLATE.format(category=category, dims=dims)
        try:
            decision = classification.classify_image(ref, prompt, ROUTE_SCHEMA, model=model)
        except Exception as exc:  # noqa: BLE001
            decision = {
                "route": "trellis",
                "complexity": "unknown",
                "is_organic": True,
                "reason": f"classify error, default trellis: {exc}",
            }
        decision = apply_policy(kind, category, decision)
        return {
            "scan": scan,
            "kind": kind,
            "name": name,
            "category": category,
            "dims": dims,
            "reference": str(ref),
            **decision,
        }

    # One independent vision call per object, each ~9s. Run them together: the stage was serial
    # and cost 46s on a 5-object room for no reason. `map` keeps routing order stable, and the
    # per-object lines print after the pool drains so threads don't interleave mid-line.
    with ThreadPoolExecutor(max_workers=max(1, min(len(items), MAX_CLASSIFY_WORKERS))) as pool:
        recs: list[dict] = list(pool.map(classify_one, items))

    for record in recs:
        flag = " [policy]" if record.get("forced_by_policy") else ""
        print(
            f"  {record['name']:16s} -> {record['route']:10s}{flag:9s} "
            f"({record.get('complexity', '?')}) {record.get('reason', '')[:60]}"
        )

    manifest = {"work": str(work), "model": model, "scans": {scan: recs}}
    (routing_root / scan).mkdir(parents=True, exist_ok=True)
    (routing_root / scan / "routing.json").write_text(json.dumps(recs, indent=2), encoding="utf-8")
    (routing_root / "routing_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    n_proc = sum(1 for r in recs if r["route"] == "procedural")
    n_trellis = sum(1 for r in recs if r["route"] == "trellis")
    print(
        f"[classify] {scan}: {n_proc} procedural, {n_trellis} trellis "
        f"-> {routing_root / 'routing_manifest.json'}"
    )
    telemetry.event("classify", scan=scan, procedural=n_proc, trellis=n_trellis, total=len(recs))
    return {
        "scan": scan,
        "records": recs,
        "procedural": n_proc,
        "trellis": n_trellis,
        "manifest": str(routing_root / "routing_manifest.json"),
    }


def _discover_scans() -> list[str]:
    root = config.output_root()
    if not root.is_dir():
        return []
    return sorted(
        p.name for p in root.iterdir() if (p / "scene_init" / "obj_stage" / "object_init" / "object_refs").is_dir()
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--scan", nargs="*", help="Scan names (default: all under the output root).")
    ap.add_argument("--no-chairs", action="store_true")
    ap.add_argument("--model", default=classification.DEFAULT_CLASSIFY_MODEL)
    args = ap.parse_args()

    scans = args.scan or _discover_scans()
    if not scans:
        print("No scans with object_refs found under the output root.", file=sys.stderr)
        return 2
    total = {"procedural": 0, "trellis": 0}
    for scan in scans:
        res = classify_for_scan(scan, model=args.model, include_chairs=not args.no_chairs)
        total["procedural"] += res["procedural"]
        total["trellis"] += res["trellis"]
    print(
        f"\n{'=' * 60}\nrouting total: {total['procedural']} procedural, {total['trellis']} trellis"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
