"""validate — is the built room a room a physics engine could accept? A GATE, not an opinion.

    python -m litereality_agent.pipeline.room_qc.validate --room <room_dir> [--preview <room_preview>] [--json out]

Reads what the build wrote (`room_preview/room_layout.json`, `Room.glb`, `manifest.json`) and
checks the claims the author made against the geometry:

  support     every object that says `rests_on=X` actually has its underside on X's top surface
              (within +1 cm above / -2 cm into it) and over X's footprint; floor-standing furniture
              stands on the floor.
  declared    every added object states `rests_on` or `attached_to` — an object that says neither
              is one nobody can simulate.
  clash       no two objects share volume — the real-mesh contact report from `check_collisions`
              (python-fcl). The axis-aligned overlap list is ADVISORY: a chair tucked under a table
              overlaps its box legitimately, so boxes are a review list, never a verdict.
  articulated the manifest's articulated objects (doors, windows, lift tops …) still reach the
              build as multi-part objects — an authoring pass must not flatten them.
  manifold    open boundary edges per mesh — reported, not failed: reconstructed meshes are not
              always closed and that is a decomposition problem, not an authoring one.

Exit 0 when the gate passes, 2 when it does not, 1 when it could not run. The JSON report is
written next to the layout as `validation.json` (or to --json) so the stage summary, the trace and
the model itself can read the same verdict. The GPT-6 one-shot run rebuilt until its own audit
of exactly these things passed; this makes that audit part of the harness.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

STRUCTURE_CATS = {"wall", "floor", "ceiling", "door", "window", "opening", "room_shell", "shell"}
FLOOR_STANDING = {"table", "chair", "sofa", "storage", "desk", "cabinet", "bed", "refrigerator",
                  "stove", "sink", "toilet", "bathtub", "bookshelf", "wardrobe"}
FLOAT_TOL, SINK_TOL, OVERLAP_SLACK = 0.010, 0.020, 0.010


def _load(preview: Path) -> tuple[list[dict], dict, dict]:
    layout = json.loads((preview / "room_layout.json").read_text(encoding="utf-8"))
    objs = layout["objects"] if isinstance(layout["objects"], list) else list(layout["objects"].values())
    manifest = {}
    mp = preview / "manifest.json"
    if mp.is_file():
        manifest = json.loads(mp.read_text(encoding="utf-8"))
    return objs, layout, manifest


def _footprint_overlap(a: dict, b: dict) -> float:
    """Fraction of a's xy footprint that lies over b's."""
    ax0, ay0 = a["bbox_min"][:2]; ax1, ay1 = a["bbox_max"][:2]
    bx0, by0 = b["bbox_min"][:2]; bx1, by1 = b["bbox_max"][:2]
    ox = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    oy = max(0.0, min(ay1, by1) - max(ay0, by0))
    area = max(1e-9, (ax1 - ax0) * (ay1 - ay0))
    return ox * oy / area


def check_support(objs: list[dict], floor_z: float) -> list[dict]:
    by_id = {o["id"]: o for o in objs}
    findings = []
    for o in objs:
        cat = (o.get("category") or "").lower()
        if cat in STRUCTURE_CATS:
            continue
        sup = o.get("rests_on")
        att = o.get("attached_to")
        bottom = o["bbox_min"][2]
        if sup:
            if sup.lower().startswith("floor"):
                top, over = floor_z, 1.0
            elif sup in by_id:
                s = by_id[sup]
                top, over = s.get("top_z", s["bbox_max"][2]), _footprint_overlap(o, s)
            else:
                findings.append({"id": o["id"], "kind": "unknown_support", "rests_on": sup,
                                 "detail": f"{o['id']} rests_on {sup!r}, which is not in the layout"})
                continue
            gap = bottom - top
            if gap > FLOAT_TOL:
                findings.append({"id": o["id"], "kind": "floating", "rests_on": sup, "gap_m": round(gap, 4),
                                 "detail": f"{o['id']} is {gap*100:.1f} cm above {sup} — drop it by that much"})
            elif gap < -SINK_TOL:
                findings.append({"id": o["id"], "kind": "sunk", "rests_on": sup, "gap_m": round(gap, 4),
                                 "detail": f"{o['id']} is {-gap*100:.1f} cm into {sup} — raise it by that much"})
            if over < 0.5 and not sup.lower().startswith("floor"):
                findings.append({"id": o["id"], "kind": "off_support", "rests_on": sup, "overlap": round(over, 2),
                                 "detail": f"only {over*100:.0f}% of {o['id']}'s footprint is over {sup}"})
        elif att:
            continue
        elif cat in FLOOR_STANDING or o.get("source_glb"):
            gap = bottom - floor_z
            if gap > 0.05:
                findings.append({"id": o["id"], "kind": "floating", "rests_on": "floor", "gap_m": round(gap, 4),
                                 "detail": f"{o['id']} ({cat}) floats {gap*100:.1f} cm above the floor"})
            elif gap < -0.05:
                findings.append({"id": o["id"], "kind": "sunk", "rests_on": "floor", "gap_m": round(gap, 4),
                                 "detail": f"{o['id']} ({cat}) is {-gap*100:.1f} cm into the floor"})
        else:
            findings.append({"id": o["id"], "kind": "undeclared_support",
                             "detail": f"{o['id']} ({cat or 'no category'}) states neither rests_on nor attached_to"})
    return findings


def check_overlaps(objs: list[dict]) -> list[dict]:
    items = [o for o in objs if (o.get("category") or "").lower() not in STRUCTURE_CATS]
    findings = []
    for i, a in enumerate(items):
        for b in items[i + 1:]:
            if a.get("rests_on") == b["id"] or b.get("rests_on") == a["id"]:
                continue
            if a.get("attached_to") == b["id"] or b.get("attached_to") == a["id"]:
                continue
            ov = [min(a["bbox_max"][k], b["bbox_max"][k]) - max(a["bbox_min"][k], b["bbox_min"][k]) for k in range(3)]
            if all(v > OVERLAP_SLACK for v in ov):
                findings.append({"id": a["id"], "with": b["id"], "kind": "aabb_overlap",
                                 "overlap_m": [round(v, 3) for v in ov],
                                 "detail": f"{a['id']} and {b['id']} share {min(ov)*100:.1f} cm of volume "
                                           f"(bounding boxes; confirm on the meshes)"})
    return findings


def check_articulated(manifest: dict, glb: Path) -> list[dict]:
    """Each articulated asset must reach the build as a handle with 2+ mesh parts under it."""
    wanted = [(a["object"], a.get("maps_to_prim") or a["object"])
              for a in manifest.get("assets", []) if a.get("kind") == "articulated"]
    if not wanted or not glb.is_file():
        return []
    try:
        import networkx as nx
        import trimesh
        scene = trimesh.load(str(glb), force="scene")
        G = scene.graph.to_networkx()
        geo = set(scene.graph.nodes_geometry)
    except Exception as exc:  # noqa: BLE001
        return [{"kind": "articulated_unchecked", "detail": f"could not open {glb.name}: {exc}"}]
    findings = []
    for name, handle in wanted:
        if handle not in G:
            findings.append({"id": name, "kind": "articulation_lost",
                             "detail": f"{name} is articulated in the manifest but its handle {handle!r} "
                                       f"is not in Room.glb"})
            continue
        parts = [n for n in nx.descendants(G, handle) if n in geo]
        if len(parts) < 2:
            findings.append({"id": name, "kind": "articulation_lost",
                             "detail": f"{name} ({handle}) reaches Room.glb as {len(parts)} mesh part(s) — "
                                       f"its moving/fixed split is gone"})
    return findings


def check_manifold(glb: Path, limit: int = 12) -> list[dict]:
    if not glb.is_file():
        return []
    try:
        import trimesh
        scene = trimesh.load(str(glb), force="scene")
    except Exception:  # noqa: BLE001
        return []
    rows = []
    for name, geom in scene.geometry.items():
        try:
            if geom.is_watertight:
                continue
            edges = int(len(trimesh.grouping.group_rows(geom.edges_sorted, require_count=1)))
        except Exception:  # noqa: BLE001
            continue
        if edges:
            rows.append({"mesh": name, "open_edges": edges})
    rows.sort(key=lambda r: -r["open_edges"])
    return rows[:limit]


def mesh_clashes(room: Path, glb: Path) -> list[dict] | None:
    """The real-mesh report from check_collisions, when its dependencies are available."""
    try:
        from litereality_agent.agent.tools.check_collisions.source import collision_mesh as cm
        from litereality_agent.agent.tools.check_collisions.source.geometry import _extract_shell
        shell = _extract_shell((room / "Room.py").read_text(encoding="utf-8"))
        bodies = cm.build_bodies(glb, shell)
        out = cm.check_all(bodies, shell)
        out += [{"id": w["id"], "kind": "open_swing_blocked", "opening": w["opening"], "detail": w["detail"]}
                for w in cm.open_clearance(bodies, shell)]
        return out
    except ImportError:
        return None
    except Exception as exc:  # noqa: BLE001 — a mesh hiccup is a note, not a verdict
        return [{"kind": "mesh_check_failed", "detail": f"{type(exc).__name__}: {exc}"}]


def validate(room: Path, preview: Path | None = None) -> dict:
    room = Path(room)
    preview = Path(preview) if preview else room.parent / "room_preview"
    if not (preview / "room_layout.json").is_file():
        return {"ok": False, "error": f"no build to validate: {preview / 'room_layout.json'} missing "
                                      f"(compile the room first)", "pass": False}
    objs, layout, manifest = _load(preview)
    floor_z = float(layout.get("bounds", {}).get("min", [0, 0, 0])[2])
    for o in objs:
        if (o.get("category") or "").lower() == "floor":
            floor_z = float(o.get("top_z", floor_z))
            break
    glb = preview / "Room.glb"
    support = check_support(objs, floor_z)
    overlaps = check_overlaps(objs)
    articulated = check_articulated(manifest, glb)
    manifold = check_manifold(glb)
    clashes = mesh_clashes(room, glb)
    # Bounding boxes cannot tell a chair tucked under a table from a chair through it, so the AABB
    # list is a REVIEW list; only the real-mesh contacts (python-fcl) block.
    failing = [f for f in support if f["kind"] in ("floating", "sunk", "undeclared_support", "unknown_support")]
    failing += [f for f in articulated if f["kind"] == "articulation_lost"]
    if clashes:
        failing += [c for c in clashes if c.get("kind") in ("object_clash", "wall_clash", "opening_blocked")]
    n_decl = sum(1 for o in objs if o.get("rests_on") or o.get("attached_to"))
    report = {
        "ok": True, "pass": not failing, "room": str(room), "preview": str(preview),
        "floor_z": round(floor_z, 4), "objects": len(objs), "declared_support": n_decl,
        "counts": {"support": len(support), "overlap": len(overlaps), "articulation": len(articulated),
                   "mesh_clash": (len(clashes) if clashes is not None else None), "open_meshes": len(manifold)},
        "support": support, "overlap": overlaps, "articulation": articulated,
        "mesh_clash": clashes, "manifold": manifold,
        "failing": failing,
    }
    return report


def summary(report: dict) -> str:
    if not report.get("ok"):
        return f"validate: could not run — {report.get('error')}"
    c = report["counts"]
    verdict = "PASS" if report["pass"] else f"FAIL ({len(report['failing'])} blocking finding(s))"
    lines = [f"validate: {verdict} — {report['objects']} objects, {report['declared_support']} with a declared support; "
             f"support {c['support']}, articulation {c['articulation']}, "
             f"mesh clashes {c['mesh_clash'] if c['mesh_clash'] is not None else 'n/a (no python-fcl)'}, "
             f"open meshes {c['open_meshes']}; {c['overlap']} box overlaps to review"]
    for f in report["failing"][:25]:
        lines.append(f"  ✗ {f.get('kind')}: {f.get('detail')}")
    if len(report["failing"]) > 25:
        lines.append(f"  … {len(report['failing']) - 25} more in validation.json")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--room", required=True, type=Path, help="room dir (Room.py)")
    ap.add_argument("--preview", type=Path, default=None, help="built room dir (default: ../room_preview)")
    ap.add_argument("--json", type=Path, default=None, help="where to write the report (default: <preview>/validation.json)")
    a = ap.parse_args(argv)
    rep = validate(a.room, a.preview)
    out = a.json or (Path(rep.get("preview", a.room.parent / "room_preview")) / "validation.json")
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    except OSError:
        pass
    print(summary(rep))
    if rep.get("ok"):
        print(f"  report → {out}")
    return 0 if rep.get("pass") else (2 if rep.get("ok") else 1)


if __name__ == "__main__":
    sys.exit(main())
