#!/usr/bin/env python3
"""Validate a GLB's structure without Blender: nodes/hierarchy, animations,
materials, embedded textures, articulation extras. Pure stdlib.

Usage: python3 validate_glb.py <model.glb>
"""

import json
import struct
import sys


def load_glb_json(path):
    with open(path, "rb") as f:
        data = f.read()
    magic, version, _ = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF":
        sys.exit(f"ERROR: not a GLB file: {path}")
    clen, ctype = struct.unpack_from("<I4s", data, 12)
    if ctype != b"JSON":
        sys.exit("ERROR: first chunk is not JSON")
    return json.loads(data[20 : 20 + clen]), len(data)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    j, size = load_glb_json(sys.argv[1])
    nodes = j.get("nodes", [])
    warnings = []

    print(f"file: {sys.argv[1]}  ({size / 1e6:.2f} MB, glTF {j['asset']['version']})")

    print("\n--- node hierarchy ---")
    roots = j["scenes"][j.get("scene", 0)]["nodes"]

    def walk(idx, depth):
        n = nodes[idx]
        t = [round(x, 3) for x in n.get("translation", [0, 0, 0])]
        extra = "  extras=" + json.dumps(n["extras"]) if "extras" in n else ""
        print(f"{'  ' * depth}{n.get('name', f'#{idx}')}  T={t}{extra}")
        for c in n.get("children", []):
            walk(c, depth + 1)

    for r in roots:
        walk(r, 0)

    print("\n--- animations ---")
    anims = j.get("animations", [])
    if not anims:
        warnings.append("no animations — articulated parts will not move")
    for a in anims:
        targets = {nodes[ch["target"]["node"]].get("name") for ch in a["channels"]}
        print(f"{a.get('name')}: {len(a['channels'])} channel(s) -> {sorted(targets)}")

    print("\n--- materials ---")
    for m in j.get("materials", []):
        pbr = m.get("pbrMetallicRoughness", {})
        tex = [k for k in ("baseColorTexture", "metallicRoughnessTexture") if k in pbr]
        if "normalTexture" in m:
            tex.append("normalTexture")
        factors = {
            k: round(v, 3) if isinstance(v, float) else v
            for k, v in pbr.items()
            if k.endswith("Factor")
        }
        print(f"{m.get('name')}: textures={tex or 'none'} factors={factors}")

    print("\n--- embedded images ---")
    imgs = j.get("images", [])
    if not imgs:
        warnings.append("no embedded images — textures missing?")
    for img in imgs:
        print(f"{img.get('name')} ({img.get('mimeType')})")

    articulated = [
        n.get("name") for n in nodes if "extras" in n and "articulation_type" in n["extras"]
    ]
    print(f"\narticulated nodes ({len(articulated)}): {articulated}")
    if not articulated:
        warnings.append("no articulation extras found (was export_extras=True set?)")

    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(" -", w)
    else:
        print("\nOK: structure looks complete")


main()
