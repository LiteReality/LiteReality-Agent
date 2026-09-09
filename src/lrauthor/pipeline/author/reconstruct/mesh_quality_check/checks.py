#!/usr/bin/env python3
"""Geometry checks for one generated GLB — flags floor slabs, fused background, floating
fragments, bad aspect, and squat-blob chairs.

This is the MESH-level gate, run per generated asset during reconstruction. Not to be
confused with `pipeline/compile/quality_check`, which gates the assembled `Room.py` at the end: this
one asks "is this chair a usable mesh", that one asks "is this furniture in a sane place".
`chair_repair.py` next door acts on what this reports.

Ported from the studio pipeline's qa/glb_geometry_qa.py. The per-mesh checks
(qa_glb + helpers) are kept verbatim; only the runner is Week3-aware: it scans the
TRELLIS chair GLBs under trellis/out/<scan>/chairs/*.glb (TRELLIS is the chair path
and the usual source of bad geometry), runs the checks, and writes a report.

    python -m lrauthor.pipeline.author.reconstruct.mesh_quality_check.checks
    python -m ...mesh_quality_check.checks --scan tea_room
    python -m ...mesh_quality_check.checks --glb path/to/x.glb     # check arbitrary GLB(s)

GLBs are assumed glTF Y-up (TRELLIS output), so "up" is the Y extent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
TRELLIS_OUT = HERE / "out"

AXES = ("x", "y", "z")


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
        raise ValueError("empty_or_unreadable_mesh")
    return mesh


def component_stats(mesh: trimesh.Trimesh) -> dict[str, Any]:
    # Coarse occupied-voxel component test (cheap vs exact face-component split on
    # 80k-face textured GLBs) — still catches detached slabs/background chunks.
    from scipy import ndimage

    vertices = np.asarray(mesh.vertices)
    bounds = mesh.bounds
    extents = np.maximum(bounds[1] - bounds[0], 1e-9)
    grid_size = 36
    ijk = np.floor((vertices - bounds[0]) / extents * (grid_size - 1)).astype(np.int32)
    ijk = np.clip(ijk, 0, grid_size - 1)
    occ = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
    occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labels, count = ndimage.label(occ, structure=structure)
    sizes = np.bincount(labels.ravel())[1:]
    sizes = sorted([int(v) for v in sizes], reverse=True)
    total = sum(sizes) or 1
    return {
        "method": "voxel_occupancy_36",
        "count": count,
        "largest_area_ratio": round(sizes[0] / total, 4) if sizes else 0.0,
        "second_area_ratio": round(sizes[1] / sizes[0], 4) if len(sizes) > 1 and sizes[0] else 0.0,
        "small_component_count": max(0, len(sizes) - 1),
    }


def extreme_plane_metrics(
    mesh: trimesh.Trimesh, thickness_ratio: float = 0.055
) -> list[dict[str, Any]]:
    bounds = mesh.bounds
    extents = np.maximum(bounds[1] - bounds[0], 1e-9)
    centers = mesh.triangles_center
    normals = mesh.face_normals
    face_areas = mesh.area_faces
    total_area = float(mesh.area) or 1.0
    vertices = np.asarray(mesh.vertices)
    metrics: list[dict[str, Any]] = []

    for axis, name in enumerate(AXES):
        other = [i for i in range(3) if i != axis]
        thickness = thickness_ratio * extents[axis]
        for side, value, selector in (
            ("min", bounds[0, axis], centers[:, axis] <= bounds[0, axis] + thickness),
            ("max", bounds[1, axis], centers[:, axis] >= bounds[1, axis] - thickness),
        ):
            flat = np.abs(normals[:, axis]) > 0.88
            face_sel = selector & flat
            flat_area_ratio = float(face_areas[face_sel].sum() / total_area)

            vert_sel = (
                vertices[:, axis] <= value + thickness
                if side == "min"
                else vertices[:, axis] >= value - thickness
            )
            slab_vertices = vertices[vert_sel]
            if len(slab_vertices):
                spread = np.ptp(slab_vertices[:, other], axis=0) / extents[other]
                footprint_ratio = float(np.prod(np.clip(spread, 0.0, 1.0)))
                thinness = float(thickness / max(extents[axis], 1e-9))
            else:
                footprint_ratio = 0.0
                thinness = 0.0

            metrics.append(
                {
                    "axis": name,
                    "side": side,
                    "flat_area_ratio": round(flat_area_ratio, 4),
                    "footprint_ratio": round(footprint_ratio, 4),
                    "thinness": round(thinness, 4),
                }
            )
    return metrics


def low_support_metrics(
    mesh: trimesh.Trimesh, band_ratio: float = 0.12, grid_size: int = 32
) -> dict[str, Any]:
    """Dense bottom mass that often means floor/background got fused in."""
    bounds = mesh.bounds
    extents = np.maximum(bounds[1] - bounds[0], 1e-9)
    vertices = np.asarray(mesh.vertices)
    centers = mesh.triangles_center
    face_areas = mesh.area_faces
    total_area = float(mesh.area) or 1.0

    bottom_cutoff = bounds[0, 1] + band_ratio * extents[1]
    face_sel = centers[:, 1] <= bottom_cutoff
    vert_sel = vertices[:, 1] <= bottom_cutoff
    low_vertices = vertices[vert_sel]

    if len(low_vertices):
        xz_axes = [0, 2]
        spread = np.ptp(low_vertices[:, xz_axes], axis=0) / extents[xz_axes]
        footprint_ratio = float(np.prod(np.clip(spread, 0.0, 1.0)))
        ij = np.floor(
            (low_vertices[:, xz_axes] - bounds[0, xz_axes]) / extents[xz_axes] * (grid_size - 1)
        ).astype(np.int32)
        ij = np.clip(ij, 0, grid_size - 1)
        occupancy = np.zeros((grid_size, grid_size), dtype=bool)
        occupancy[ij[:, 0], ij[:, 1]] = True
        occupancy_ratio = float(occupancy.sum() / (grid_size * grid_size))
        ij_min = ij.min(axis=0)
        ij_max = ij.max(axis=0)
        rect_cells = max(1, int((ij_max[0] - ij_min[0] + 1) * (ij_max[1] - ij_min[1] + 1)))
        rect_occupancy_ratio = float(occupancy.sum() / rect_cells)
    else:
        footprint_ratio = occupancy_ratio = rect_occupancy_ratio = 0.0

    return {
        "assumed_up_axis": "y",
        "band_ratio": band_ratio,
        "area_ratio": round(float(face_areas[face_sel].sum() / total_area), 4),
        "footprint_ratio": round(footprint_ratio, 4),
        "occupancy_ratio": round(occupancy_ratio, 4),
        "rect_occupancy_ratio": round(rect_occupancy_ratio, 4),
        "vertex_ratio": round(float(vert_sel.sum() / max(len(vertices), 1)), 4),
    }


def qa_glb(path: Path, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata = metadata or {}
    result: dict[str, Any] = {"file": path.name, "status": "pass", "flags": []}
    for k in ("category", "source_group", "object", "scan"):
        if k in metadata:
            result[k] = metadata[k]
    try:
        mesh = load_mesh(path)
    except Exception as exc:
        return {
            "file": path.name,
            "status": "fail",
            "flags": ["load_failed"],
            "error": str(exc),
            **{k: metadata.get(k) for k in ("scan", "object")},
        }

    bounds = mesh.bounds
    extents = bounds[1] - bounds[0]
    extent_min = float(max(np.min(extents), 1e-9))
    extent_max = float(np.max(extents))
    aspect = extent_max / extent_min
    result["extents"] = [round(float(v), 5) for v in extents.tolist()]
    result["bbox_aspect"] = round(float(aspect), 4)
    horizontal_max = float(max(extents[0], extents[2], 1e-9))
    up_extent = float(extents[1])
    result["chair_shape"] = {"height_to_footprint_ratio": round(up_extent / horizontal_max, 4)}

    planes = extreme_plane_metrics(mesh)
    suspicious_planes = [
        p for p in planes if p["flat_area_ratio"] >= 0.09 and p["footprint_ratio"] >= 0.36
    ]
    strong_planes = [
        p for p in planes if p["flat_area_ratio"] >= 0.15 and p["footprint_ratio"] >= 0.46
    ]
    comps = component_stats(mesh)
    result["components"] = comps
    low_support = low_support_metrics(mesh)
    result["low_support"] = low_support

    min_y = float(bounds[0, 1])
    result["grounding"] = {
        "bottom_y": round(min_y, 5),
        "needs_origin_grounding": abs(min_y) > 0.025,
    }

    flags = result["flags"]
    if strong_planes:
        flags.append("fail_large_flat_extreme_plane")
    elif suspicious_planes:
        flags.append("warn_possible_flat_panel")
    if (
        low_support["area_ratio"] >= 0.18
        and low_support["occupancy_ratio"] >= 0.45
        and low_support["footprint_ratio"] >= 0.65
    ):
        flags.append("fail_low_support_mass")
    elif (
        low_support["area_ratio"] >= 0.12
        and low_support["occupancy_ratio"] >= 0.3
        and low_support["footprint_ratio"] >= 0.55
    ):
        flags.append("warn_low_support_mass")
    if comps["second_area_ratio"] >= 0.28:
        flags.append("warn_large_secondary_component")
    if comps["small_component_count"] >= 20:
        flags.append("fail_many_floating_fragments")
    elif comps["small_component_count"] >= 8:
        flags.append("warn_many_floating_fragments")
    if comps["count"] >= 9 and comps["largest_area_ratio"] < 0.9:
        flags.append("warn_many_components")
    if aspect >= 7.5:
        flags.append("warn_extreme_bbox_aspect")
    if up_extent / horizontal_max < 0.68:
        flags.append("fail_squat_chair_blob")
    elif up_extent / horizontal_max < 0.82:
        flags.append("warn_squat_chair_shape")
    if result["grounding"]["needs_origin_grounding"]:
        flags.append("needs_origin_grounding")

    if any(f.startswith("fail_") for f in flags):
        result["status"] = "fail"
    elif any(f.startswith("warn_") for f in flags):
        result["status"] = "warn"
    return result


def collect_chairs(scans: list[str] | None) -> list[tuple[Path, dict]]:
    items = []
    if not TRELLIS_OUT.exists():
        return items
    for scan_dir in sorted(TRELLIS_OUT.iterdir()):
        if not scan_dir.is_dir() or scan_dir.name.startswith("_"):
            continue
        if scans and scan_dir.name not in scans:
            continue
        for glb in (
            sorted((scan_dir / "chairs").glob("*.glb")) if (scan_dir / "chairs").is_dir() else []
        ):
            items.append((glb, {"scan": scan_dir.name, "object": glb.stem, "category": "chair"}))
    return items


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--scan", nargs="*", help="limit to these scans")
    ap.add_argument(
        "--glb", nargs="*", type=Path, help="QA arbitrary GLB(s) instead of the chair set"
    )
    ap.add_argument("--out", type=Path, default=TRELLIS_OUT / "chair_qa_report.json")
    args = ap.parse_args()

    if args.glb:
        items = [(p, {"object": p.stem}) for p in args.glb]
    else:
        items = collect_chairs(args.scan)
    if not items:
        print("no GLBs to QA", flush=True)
        return 2

    report_items = [qa_glb(p, meta) for p, meta in items]
    summary = {
        s: sum(1 for it in report_items if it["status"] == s) for s in ("pass", "warn", "fail")
    }
    summary["total"] = len(report_items)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"summary": summary, "items": report_items}, indent=2), encoding="utf-8"
    )

    print(
        f"\nGLB geometry QA — {summary['total']} chair(s): "
        f"{summary['pass']} pass, {summary['warn']} warn, {summary['fail']} FAIL\n"
    )
    for it in report_items:
        mark = {"pass": "✓", "warn": "▲", "fail": "✗"}[it["status"]]
        loc = f"{it.get('scan', '')}/{it.get('object', '')}".strip("/")
        print(f"  {mark} {it['status']:5} {loc:40} {', '.join(it['flags']) or 'clean'}")
    print(f"\nreport: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
