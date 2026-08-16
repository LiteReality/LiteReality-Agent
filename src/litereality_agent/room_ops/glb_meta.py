"""glb_meta.py — the collision model's facts, carried INSIDE the glb as glTF `extras`.

The collision check wants to be one deterministic function of one file: hand it a `Room.glb`, get
back whether anything clashes. It could not be, because three things it needs live only in
`Room.py`'s SHELL literal and are not recoverable from the geometry:

  * wall CENTRELINES — the built walls are extruded solids, and a diagonal wall's mesh tells you
    nothing about its line (Wall0's mesh AABB is 1.25 x 2.67 x 2.50 for a 2.72 m wall)
  * floor_z / ceiling_z — usually derivable from Floor0/Ceiling0, but 2 of 8 rooms have no ceiling
  * per-object CATEGORY — node names cover the common furniture, but `Television0`,
    `Dishwasher0` and `Oven_Stove0` match no pattern and silently drop out of the model

Measured cost of not having them: on Elliott-Studio, running from the glb alone finds 1 of 12
violations and reports a clean room. So this module puts exactly those facts into the file.

Deliberately NOT stored: `center` and `size`. Both recover from the placed mesh's oriented
bounding box to within 0.1-1.1 mm (the builder scales each asset to fill its RoomPlan box), so a
stored copy would be duplication that goes stale the moment the resolver nudges something.

    python -m litereality_agent.room_ops.glb_meta --room <room dir>
    python -m ...room_ops.glb_meta --run-root run          # stamp every built room in place
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

SCHEMA = "litereality/collision/1"
_JSON, _BIN = 0x4E4F534A, 0x004E4942


def read_chunks(path: str | Path) -> tuple[dict, bytes]:
    """(glTF JSON, binary chunk) of a .glb."""
    b = Path(path).read_bytes()
    if b[:4] != b"glTF":
        raise ValueError(f"not a glb: {path}")
    total = struct.unpack("<I", b[8:12])[0]
    off, gltf, binary = 12, None, b""
    while off < min(total, len(b)):
        length, ctype = struct.unpack("<II", b[off:off + 8])
        payload = b[off + 8:off + 8 + length]
        if ctype == _JSON:
            gltf = json.loads(payload)
        elif ctype == _BIN:
            binary = payload
        off += 8 + length
    if gltf is None:
        raise ValueError(f"no JSON chunk in {path}")
    return gltf, binary


def write_chunks(path: str | Path, gltf: dict, binary: bytes) -> None:
    """Rewrite a .glb with a new JSON chunk. Chunks are 4-byte aligned — JSON pads with spaces,
    BIN with zeros; getting that wrong produces a file every loader rejects."""
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    out = bytearray()
    body = bytearray()
    body += struct.pack("<II", len(js), _JSON) + js
    if binary:
        bn = binary + b"\x00" * (-len(binary) % 4)
        body += struct.pack("<II", len(bn), _BIN) + bn
    out += b"glTF" + struct.pack("<II", 2, 12 + len(body)) + body
    Path(path).write_bytes(bytes(out))


def stamp(glb: str | Path, shell: dict, out: str | Path | None = None) -> dict:
    """Write the SHELL facts into the glb's `extras`. Returns what was written, for reporting.

    Idempotent: re-stamping overwrites the same keys. Objects and walls are matched to nodes BY
    NAME, which is the same handle convention `build_bodies` groups on.
    """
    gltf, binary = read_chunks(glb)
    walls = shell.get("walls") or {}
    objects = shell.get("objects") or {}
    openings = shell.get("openings") or {}

    counts = {"walls": 0, "objects": 0, "openings": 0}
    for node in gltf.get("nodes", []):
        name = node.get("name")
        if not name:
            continue
        extras = dict(node.get("extras") or {})
        if name in walls:
            w = walls[name]
            extras.update({"lr_kind": "wall", "lr_start": list(w["start"]),
                           "lr_end": list(w["end"]),
                           "lr_thickness": float(w.get("thickness", 0.0))})
            counts["walls"] += 1
        elif name in objects:
            o = objects[name]
            extras.update({"lr_kind": "object", "lr_category": o.get("category", ""),
                           "lr_yaw": float(o.get("yaw", 0.0))})
            counts["objects"] += 1
        elif name in openings:
            op = openings[name]
            extras.update({"lr_kind": "opening", "lr_wall": op.get("wall", ""),
                           "lr_offset": float(op.get("offset", 0.0)),
                           "lr_width": float(op.get("width", 0.0)),
                           "lr_height": float(op.get("height", 0.0)),
                           "lr_sill": float(op.get("sill", 0.0))})
            counts["openings"] += 1
        else:
            continue
        node["extras"] = extras

    scene = (gltf.get("scenes") or [{}])[gltf.get("scene", 0)]
    scene_extras = dict(scene.get("extras") or {})
    scene_extras.update({"lr_schema": SCHEMA})
    for k in ("floor_z", "ceiling_z"):
        if shell.get(k) is not None:
            scene_extras[f"lr_{k}"] = float(shell[k])
    scene["extras"] = scene_extras

    write_chunks(out or glb, gltf, binary)
    return counts


def shell_from_glb(glb: str | Path) -> dict:
    """Rebuild the SHELL-shaped dict the collision checks want, from the glb alone.

    Returns {} when the file carries no `lr_schema` — the caller then falls back to parsing
    `Room.py`, so a glb built before this existed still works.
    """
    try:
        gltf, _ = read_chunks(glb)
    except Exception:  # noqa: BLE001
        return {}
    scenes = gltf.get("scenes") or [{}]
    scene_extras = scenes[gltf.get("scene", 0)].get("extras") or {}
    if scene_extras.get("lr_schema") != SCHEMA:
        return {}

    shell: dict = {"walls": {}, "objects": {}, "openings": {}}
    for k in ("floor_z", "ceiling_z"):
        if f"lr_{k}" in scene_extras:
            shell[k] = scene_extras[f"lr_{k}"]
    for node in gltf.get("nodes", []):
        name, ex = node.get("name"), (node.get("extras") or {})
        kind = ex.get("lr_kind")
        if not name or not kind:
            continue
        if kind == "wall":
            shell["walls"][name] = {"start": list(ex["lr_start"]), "end": list(ex["lr_end"]),
                                    "thickness": ex.get("lr_thickness", 0.0)}
        elif kind == "object":
            shell["objects"][name] = {"category": ex.get("lr_category", ""),
                                      "yaw": ex.get("lr_yaw", 0.0)}
        elif kind == "opening":
            shell["openings"][name] = {"wall": ex.get("lr_wall", ""),
                                       "offset": ex.get("lr_offset", 0.0),
                                       "width": ex.get("lr_width", 0.0),
                                       "height": ex.get("lr_height", 0.0),
                                       "sill": ex.get("lr_sill", 0.0)}
    return shell


def _shell_of_room(room_dir: Path) -> dict:
    from litereality_agent.room_ops.shell import extract_shell

    return extract_shell((room_dir / "Room.py").read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--room", help="a room dir containing Room.py")
    ap.add_argument("--glb", help="the glb to stamp (default: every Room.glb near --room)")
    ap.add_argument("--run-root", help="stamp every built room under this directory")
    a = ap.parse_args()

    targets: list[tuple[Path, Path]] = []
    if a.run_root:
        for room_py in sorted(Path(a.run_root).glob("*/realism_authoring/room/Room.py")):
            for glb in sorted(room_py.parent.parent.rglob("Room.glb")):
                targets.append((room_py.parent, glb))
    elif a.room:
        room = Path(a.room)
        globs = [Path(a.glb)] if a.glb else sorted(room.parent.rglob("Room.glb"))
        targets = [(room, g) for g in globs]
    else:
        ap.error("need --room or --run-root")

    for room, glb in targets:
        shell = _shell_of_room(room)
        c = stamp(glb, shell)
        back = shell_from_glb(glb)
        ok = (len(back.get("walls", {})) == c["walls"]
              and len(back.get("objects", {})) == c["objects"])
        print(f"{'ok ' if ok else 'ERR'} {glb}  walls={c['walls']} objects={c['objects']} "
              f"openings={c['openings']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
