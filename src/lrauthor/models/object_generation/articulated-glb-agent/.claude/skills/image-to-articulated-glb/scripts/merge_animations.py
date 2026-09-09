#!/usr/bin/env python3
"""Add a combined "AllParts" animation to GLB files (pure python, no Blender).

The skill exports one animation clip per movable part, but most glTF viewers
only auto-play the FIRST animation, so multi-joint assets appear to have a
single moving part. This script merges the channels/samplers of every existing
animation into one new clip named "AllParts" and inserts it at animations[0]
(keyframe accessors are shared, so the binary chunk is untouched). Original
per-part clips are kept after it. Idempotent: files that already have an
"AllParts" animation are skipped.

Usage:
    python3 merge_animations.py <file.glb | dir> [more files/dirs...]

Directories are searched recursively for *.glb.
"""

import copy
import json
import os
import struct
import sys
import tempfile

JSON_CHUNK = 0x4E4F534A  # 'JSON'
BIN_CHUNK = 0x004E4942  # 'BIN\0'
MERGED_NAME = "AllParts"


def read_glb(path):
    with open(path, "rb") as f:
        data = f.read()
    magic, version, _length = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67:
        raise ValueError(f"{path}: not a GLB file")
    if version != 2:
        raise ValueError(f"{path}: unsupported GLB version {version}")
    chunks = []
    off = 12
    while off < len(data):
        clen, ctype = struct.unpack_from("<II", data, off)
        chunks.append((ctype, data[off + 8 : off + 8 + clen]))
        off += 8 + clen
    return chunks


def write_glb(path, gltf, chunks):
    payload = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    payload += b" " * (-len(payload) % 4)  # pad with spaces per spec
    out = [None]  # placeholder for header
    body = bytearray()
    body += struct.pack("<II", len(payload), JSON_CHUNK) + payload
    for ctype, cdata in chunks:
        if ctype == JSON_CHUNK:
            continue
        cdata = cdata + b"\x00" * (-len(cdata) % 4)
        body += struct.pack("<II", len(cdata), ctype) + cdata
    header = struct.pack("<III", 0x46546C67, 2, 12 + len(body))
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".glb.tmp")
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            f.write(header + body)
        os.replace(tmp_path, path)
    except BaseException:
        os.unlink(tmp_path)
        raise


def merge_animations(path):
    chunks = read_glb(path)
    gltf = json.loads(next(c for t, c in chunks if t == JSON_CHUNK))
    anims = gltf.get("animations", [])
    if len(anims) < 2:
        return "skip (fewer than 2 animations)"
    if any(a.get("name") == MERGED_NAME for a in anims):
        return "skip (already merged)"
    merged = {"name": MERGED_NAME, "samplers": [], "channels": []}
    for anim in anims:
        offset = len(merged["samplers"])
        merged["samplers"].extend(copy.deepcopy(anim["samplers"]))
        for ch in anim["channels"]:
            ch = copy.deepcopy(ch)
            ch["sampler"] += offset
            merged["channels"].append(ch)
    gltf["animations"] = [merged] + anims
    write_glb(path, gltf, chunks)
    return f"merged {len(anims)} clips / {len(merged['channels'])} channels"


def main(argv):
    targets = []
    for arg in argv:
        if os.path.isdir(arg):
            for root, _dirs, files in os.walk(arg):
                targets.extend(os.path.join(root, f) for f in sorted(files) if f.endswith(".glb"))
        else:
            targets.append(arg)
    if not targets:
        print(__doc__)
        return 1
    for path in targets:
        try:
            result = merge_animations(path)
        except Exception as e:  # keep batch going, report at the end
            result = f"ERROR: {e}"
        print(f"{result:50s} {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
