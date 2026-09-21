"""Check named instances and moving-support relationships in the actual exported GLB."""

from __future__ import annotations

import json
import struct
from pathlib import Path


def glb_document(path: Path) -> dict:
    with path.open("rb") as stream:
        header = stream.read(12)
        if len(header) != 12 or struct.unpack("<4sII", header) != (b"glTF", 2, path.stat().st_size):
            raise ValueError("invalid GLB header")
        length, kind = struct.unpack("<II", stream.read(8))
        if kind != 0x4E4F534A:
            raise ValueError("GLB has no JSON chunk")
        return json.loads(stream.read(length))


def check(doc: dict, objects: list[dict], manifest: dict) -> list[dict]:
    nodes = doc.get("nodes", [])
    names, parents, findings = {}, {}, []
    for i, node in enumerate(nodes):
        names.setdefault(node.get("name"), []).append(i)
        for child in node.get("children", []):
            if child in parents:
                findings.append({"kind": "multiply_parented_node", "node": child})
            parents[child] = i

    def unique(name):
        matches = names.get(name, [])
        return matches[0] if len(matches) == 1 else None

    def under(node, ancestor):
        seen = set()
        while node is not None and node not in seen:
            if node == ancestor:
                return True
            seen.add(node)
            node = parents.get(node)
        return False

    animated = {
        c["target"]["node"]
        for a in doc.get("animations", [])
        for c in a.get("channels", [])
        if "node" in c.get("target", {})
    }
    by_id = {o["id"]: o for o in objects}
    for asset in manifest.get("assets", []):
        expected = asset.get("represents_prims") or [asset.get("maps_to_prim") or asset["object"]]
        for name in expected:
            handle = by_id.get(name, {}).get("handle", name)
            i = unique(handle)
            if name not in by_id or i is None:
                findings.append({"id": name, "kind": "source_instance_missing_or_ambiguous"})
            elif asset.get("kind") == "articulated" and not any(under(n, i) for n in animated):
                findings.append({"id": name, "kind": "animation_lost"})
    for obj in objects:
        target = obj.get("rests_on") or obj.get("attached_to")
        if not target or target not in by_id:
            continue
        child = unique(obj.get("handle") or obj["id"])
        support = unique(by_id[target].get("handle") or target)
        if child is None or support is None:
            findings.append({"id": obj["id"], "kind": "support_node_missing_or_ambiguous"})
            continue
        # Ignore animations on supported children: they are not moving parts of the furniture.
        moving = {n for n in animated if under(n, support) and not under(n, child)}
        part_name = obj.get("support_part")
        part = unique(part_name) if part_name else support
        if part is None or not under(part, support):
            findings.append({"id": obj["id"], "kind": "invalid_support_part", "part": part_name})
        elif moving and not part_name:
            findings.append(
                {"id": obj["id"], "kind": "moving_support_part_required", "target": target}
            )
        elif moving or part_name:
            if not under(child, part) or any(under(n, child) for n in animated):
                findings.append(
                    {"id": obj["id"], "kind": "support_motion_not_preserved", "part": part_name}
                )
    return findings
