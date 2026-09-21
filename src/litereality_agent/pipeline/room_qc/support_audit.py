"""Finite-mesh support audit used by the mandatory authoring/export gate.

Checks support graph roots, actual attachment distance, and downward contacts from
bottom mesh vertices. This is a static geometric check, not stability certification.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

ROOT_CATEGORIES = {"wall", "floor", "ceiling"}
FLOOR_CATEGORIES = {
    "table",
    "chair",
    "sofa",
    "storage",
    "desk",
    "cabinet",
    "bed",
    "refrigerator",
    "stove",
    "toilet",
    "bathtub",
    "wardrobe",
}
FLEXIBLE_CATEGORIES = {"textile", "cloth", "fabric", "mat", "rug", "pillow", "duvet"}


def support_graph(objects, strict=False):
    by_id = {o["id"]: o for o in objects}
    roots = {o["id"] for o in objects if o.get("category") in ROOT_CATEGORIES}
    floor = next((o["id"] for o in objects if o.get("category") == "floor"), None)
    edges, findings = {}, []
    if len(by_id) != len(objects):
        findings.append({"kind": "duplicate_object_ids"})
    for o in objects:
        name = o["id"]
        if name in roots:
            continue
        rest, attach = o.get("rests_on"), o.get("attached_to")
        if rest and attach:
            findings.append({"id": name, "kind": "ambiguous_support"})
        target = rest or attach
        inferred = False
        if strict and not target and o.get("category") in {"door", "window", "opening"}:
            wall = re.search(r"(Wall\d+)_", o.get("source_glb") or "")
            if wall and wall.group(1) in roots:
                target, attach, inferred = wall.group(1), wall.group(1), True
        if not target and o.get("category") in FLOOR_CATEGORIES and floor:
            target, inferred = floor, True
        if not target:
            # Openings are validated by the articulation gate, not a floor-standing inference.
            if strict or o.get("category") not in {"door", "window", "opening"}:
                findings.append({"id": name, "kind": "undeclared_support"})
            continue
        edges[name] = {
            "target": target,
            "relation": "attached_to" if attach else "rests_on",
            "inferred": inferred,
        }
        if target not in by_id:
            findings.append({"id": name, "kind": "unknown_support", "target": target})
    for name in edges:
        visited, node = set(), name
        while node not in roots:
            if node in visited:
                findings.append({"id": name, "kind": "support_cycle", "at": node})
                break
            visited.add(node)
            if node not in edges:
                findings.append({"id": name, "kind": "unrooted_support", "at": node})
                break
            node = edges[node]["target"]
    return edges, findings


def mesh_bodies(glb, objects):
    from collections import defaultdict

    import trimesh

    scene = trimesh.load(str(glb), force="scene")
    handles = {o.get("handle") or o["id"]: o["id"] for o in objects}
    requested_parts = {o["support_part"] for o in objects if o.get("support_part")}
    parents = scene.graph.transforms.parents
    parts = defaultdict(list)
    for node in scene.graph.nodes_geometry:
        owner, seen = node, set()
        while owner not in handles and owner in parents and owner not in seen:
            seen.add(owner)
            owner = parents[owner]
        if owner not in handles:
            continue
        transform, geometry = scene.graph[node]
        mesh = scene.geometry[geometry].copy()
        mesh.apply_transform(transform)
        # GLB (x,y,z) -> authored Blender/SHELL (x,-z,y).
        v = mesh.vertices.copy()
        mesh.vertices = np.column_stack((v[:, 0], -v[:, 2], v[:, 1]))
        parts[handles[owner]].append(mesh)
        ancestor, visited = node, set()
        while ancestor not in visited:
            visited.add(ancestor)
            if ancestor in requested_parts:
                parts["part:" + ancestor].append(mesh)
            if ancestor == owner or ancestor not in parents:
                break
            ancestor = parents[ancestor]
    return {name: trimesh.util.concatenate(items) for name, items in parts.items()}


def vertical_contacts(child, support, tolerance=0.015, sink_tolerance=0.02, flexible=False):
    """Downward contact rays, with underside patches for draped flexible objects.

    Only upward-facing support triangles can bear weight. In particular a cupboard's
    downward-facing bottom must not validate an object buried inside its volume.
    """
    vertices = np.asarray(child.vertices)
    if flexible and len(child.faces):
        underside = child.faces[np.asarray(child.face_normals)[:, 2] < -0.1]
        if len(underside):
            triangles = vertices[underside]
            weights = np.array(
                [[1 / 3, 1 / 3, 1 / 3], [0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]]
            )
            interior = np.einsum("ij,kjl->kil", weights, triangles).reshape(-1, 3)
            candidates = np.vstack((vertices[np.unique(underside)], interior))
        else:
            candidates = vertices
    else:
        candidates = vertices[vertices[:, 2] <= vertices[:, 2].min() + 0.004]
    samples = np.unique(np.round(candidates, 6), axis=0)
    if len(samples) > 128:
        samples = samples[np.linspace(0, len(samples) - 1, 128).astype(int)]
    if flexible:
        # Never lose real bottom contacts when a generated textile's normals are
        # inconsistent or the contact patch is small relative to its upper shell.
        bottom = np.unique(
            np.round(vertices[vertices[:, 2] <= vertices[:, 2].min() + 0.004], 6), axis=0
        )
        if len(bottom) > 64:
            bottom = bottom[np.linspace(0, len(bottom) - 1, 64).astype(int)]
        samples = np.unique(np.vstack((samples, bottom)), axis=0)
    triangles = np.asarray(support.triangles)[np.asarray(support.face_normals)[:, 2] > 0.1]
    lo, hi = samples[:, :2].min(axis=0) - 1e-5, samples[:, :2].max(axis=0) + 1e-5
    triangles = triangles[
        np.all(triangles[:, :, :2].max(axis=1) >= lo, axis=1)
        & np.all(triangles[:, :, :2].min(axis=1) <= hi, axis=1)
    ]
    heights = np.full(len(samples), -np.inf)
    for start in range(0, len(triangles), 2048):
        tri = triangles[start : start + 2048]
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        u, v = b - a, c - a
        det = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        good = np.abs(det) > 1e-12
        if not good.any():
            continue
        a, u, v, det = a[good], u[good], v[good], det[good]
        dx, dy = samples[:, None, 0] - a[None, :, 0], samples[:, None, 1] - a[None, :, 1]
        s = (dx * v[None, :, 1] - dy * v[None, :, 0]) / det
        t = (u[None, :, 0] * dy - u[None, :, 1] * dx) / det
        z = a[None, :, 2] + s * u[None, :, 2] + t * v[None, :, 2]
        hit = (s >= -1e-6) & (t >= -1e-6) & (s + t <= 1 + 1e-6)
        hit &= z <= samples[:, None, 2] + sink_tolerance + 1e-6
        heights = np.maximum(heights, np.where(hit, z, -np.inf).max(axis=1))
    gaps = samples[:, 2] - heights
    contact = np.isfinite(gaps) & (gaps <= tolerance) & (gaps >= -sink_tolerance - 1e-6)
    area = 0.0
    if contact.sum() >= 3:
        from scipy.spatial import ConvexHull, QhullError

        try:
            area = float(ConvexHull(samples[contact, :2]).volume)
        except QhullError:
            pass
    return {
        "samples": len(samples),
        "contact_samples": int(contact.sum()),
        "sampling": "flexible_underside" if flexible else "bottom_vertices",
        "contact_hull_area_m2": area,
        "contact_fraction": float(contact.mean()) if len(contact) else 0.0,
        "minimum_gap_m": float(gaps.min()) if np.isfinite(gaps).any() else None,
    }


def disconnected_findings(name, child, support, *, limit=128):
    """Every disconnected member must have a contact path to the declared support.

    Connected triangles are not semantic objects. Adjacent components may form a chair
    or stack, so connect touching components before checking reachability. This detects
    floating members but is deliberately not a centre-of-mass/stability certificate.
    """
    from trimesh.collision import CollisionManager

    parts = list(child.split(only_watertight=False))
    if len(parts) > limit:
        return [{"id": name, "kind": "component_check_incomplete", "components": len(parts)}]
    if len(parts) < 2:
        return []
    rooted, links = set(), {i: set() for i in range(len(parts))}
    manager = CollisionManager()
    manager.add_object("support", support)
    for i, part in enumerate(parts):
        if manager.min_distance_single(part) <= 0.005:
            rooted.add(i)
        local = CollisionManager()
        local.add_object(str(i), part)
        for j in range(i):
            # Cheap AABB distance rejection before FCL on potentially complex meshes.
            gap = np.maximum(
                0,
                np.maximum(
                    part.bounds[0] - parts[j].bounds[1], parts[j].bounds[0] - part.bounds[1]
                ),
            )
            if np.linalg.norm(gap) <= 0.005 and local.min_distance_single(parts[j]) <= 0.005:
                links[i].add(j)
                links[j].add(i)
    pending = list(rooted)
    while pending:
        for j in links[pending.pop()] - rooted:
            rooted.add(j)
            pending.append(j)
    return [
        {
            "id": name,
            "kind": "unsupported_disconnected_member",
            "component": i,
            "bounds": part.bounds.tolist(),
        }
        for i, part in enumerate(parts)
        if i not in rooted
    ]


def audit(preview: Path, legacy_report: Path | None = None, *, strict=False):
    from trimesh.collision import CollisionManager

    objects = json.loads((preview / "room_layout.json").read_text())["objects"]
    if isinstance(objects, dict):
        objects = list(objects.values())
    edges, failing = support_graph(objects, strict=strict)
    bodies = mesh_bodies(preview / "Room.glb", objects)
    checks = []
    by_id = {o["id"]: o for o in objects}
    for name, edge in edges.items():
        target = edge["target"]
        geometry_target = (
            "part:" + by_id[name]["support_part"] if by_id[name].get("support_part") else target
        )
        record = {"id": name, **edge}
        if name not in bodies or geometry_target not in bodies:
            failing.append({"id": name, "kind": "support_mesh_missing", "target": target})
            continue
        manager = CollisionManager()
        manager.add_object(target, bodies[geometry_target])
        distance = float(manager.min_distance_single(bodies[name]))
        record["finite_mesh_distance_m"] = distance
        if edge["relation"] == "attached_to":
            if not np.isfinite(distance) or distance > (0.01 if strict else 0.02):
                failing.append(
                    {"id": name, "kind": "attachment_gap", "target": target, "gap_m": distance}
                )
        else:
            flexible = by_id[name].get("category") in FLEXIBLE_CATEGORIES
            contacts = vertical_contacts(
                bodies[name],
                bodies[geometry_target],
                flexible=flexible,
                tolerance=0.005 if strict else 0.015,
                sink_tolerance=0.005 if strict else 0.02,
            )
            record.update(contacts)
            if not contacts["contact_samples"]:
                failing.append(
                    {
                        "id": name,
                        "kind": "no_downward_support_contact",
                        "target": target,
                        **contacts,
                    }
                )
            elif flexible and contacts["contact_hull_area_m2"] < 0.0001:
                failing.append(
                    {
                        "id": name,
                        "kind": "insufficient_flexible_contact_patch",
                        "target": target,
                        **contacts,
                    }
                )
            elif not flexible and contacts["contact_fraction"] < 0.5:
                failing.append(
                    {"id": name, "kind": "limited_bottom_contact", "target": target, **contacts}
                )
        checks.append(record)
        if strict:
            failing.extend(disconnected_findings(name, bodies[name], bodies[geometry_target]))
    # Keep the legacy report intact. A wall-plane finding is contradicted only by
    # independently measured POSITIVE separation from that specific finite wall.
    legacy_unresolved, contradicted, support_rechecked = [], [], []
    if legacy_report is not None:
        legacy = json.loads(legacy_report.read_text())
        if not legacy.get("ok"):
            legacy_unresolved.append({"kind": "legacy_check_unavailable"})
        for finding in legacy.get("failing", []):
            name, wall = finding.get("id"), finding.get("wall")
            if finding.get("kind") in {
                "floating",
                "sunk",
                "off_support",
                "unknown_support",
                "undeclared_support",
            }:
                measured = next((c for c in checks if c["id"] == name), None)
                if measured is not None:
                    # Replace the whole-object top-height approximation with this
                    # audit's contact-surface check; preserve both pieces of evidence.
                    support_rechecked.append({"finding": finding, "contact_check": measured})
                    continue
            if finding.get("kind") == "wall_clash" and name in bodies and wall in bodies:
                manager = CollisionManager()
                manager.add_object(wall, bodies[wall])
                distance = float(manager.min_distance_single(bodies[name]))
                if np.isfinite(distance) and distance > 0.03:
                    contradicted.append({"finding": finding, "finite_mesh_separation_m": distance})
                    continue
            legacy_unresolved.append(finding)
    return {
        "schema": "support_audit/3-strict" if strict else "support_audit/2",
        "strict": strict,
        "coverage": {
            "objects": len(objects),
            "support_relations": len(edges),
            "geometrically_checked": len(checks),
            "roots": sum(o.get("category") in ROOT_CATEGORIES for o in objects),
        },
        "pass": not failing and not legacy_unresolved,
        "support_findings": failing,
        "checks": checks,
        "legacy_unresolved": legacy_unresolved,
        "legacy_contradicted_by_finite_geometry": contradicted,
        "legacy_support_findings_rechecked_on_contact_surface": support_rechecked,
        "limitations": [
            "Static geometric contacts only; not physical stability or force analysis.",
            "Connected-member checks do not establish semantic correctness or structural strength.",
            "Visual appropriateness of the support surface needs photographic review.",
            "No full-scene collision or articulation certification beyond the supplied legacy gate.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", type=Path, required=True)
    parser.add_argument("--legacy-report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    result = audit(args.preview, args.legacy_report, strict=args.strict)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "pass": result["pass"],
                "support_findings": result["support_findings"],
                "legacy_unresolved": result["legacy_unresolved"],
                "contradicted_wall_findings": len(result["legacy_contradicted_by_finite_geometry"]),
            },
            indent=2,
        )
    )
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
