#!/usr/bin/env python3
"""pack_assets.py — gather a scan's generated GLBs into one flat asset dir + manifest.

build_room.py reads ONE scan's assets from a flat layout:
    <assets>/glb/<prim-named>.glb        (Table0.glb, ChairCluster0.glb, Wall1_Door_0.glb, ...)
    <assets>/manifest.json               (each glb -> its RoomPlan box)

The object stage writes both generators under one ``run/<scan>/reconstruct/`` tree,
so this bridges them:

    reconstruct/<name>.glb           -> TRELLIS (neural) static objects + chairs (flat files)
    reconstruct/<name>/<name>.glb    -> procedural / articulated objects + Wall* openings (dirs)

It maps each glb to its box by PRIM NAME (the filename stem). Chair clusters expand to
all member prims (one chair mesh fills every repeat) via the object stage's
``chair_clusters.json``; openings ``Wall{N}_{Type}_{k}`` map to ``{Type}{k}``.

    uv run litereality stage seed <scene> --force
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

from .. import paths as config

_PREFIX_CAT = [
    ("Wall", "wall"),
    ("Floor", "floor"),
    ("Ceiling", "ceiling"),
    ("Door", "door"),
    ("Window", "window"),
    ("Opening", "opening"),
    ("Table", "table"),
    ("ChairCluster", "chair"),
    ("Chair", "chair"),
    ("Storage", "storage"),
    ("Sofa", "sofa"),
    ("Bed", "bed"),
    ("Television", "television"),
    ("Refrigerator", "refrigerator"),
    ("Oven", "oven"),
    ("Stove", "stove"),
    ("Sink", "sink"),
    ("Toilet", "toilet"),
]


def category_of(name: str) -> str:
    base = re.sub(r"\.\d+$", "", name)
    for prefix, cat in _PREFIX_CAT:
        if base.startswith(prefix):
            return cat
    return "object"


def is_opening(name: str) -> bool:
    return name.startswith("Wall") and bool(re.search(r"_(Door|Window|Opening)_\d+$", name))


def opening_target(name: str) -> str:
    """Wall1_Door_0 -> Door0, Wall5_Window_0 -> Window0."""
    m = re.match(r"Wall\d+_(Door|Window|Opening)_(\d+)$", name)
    return f"{m.group(1)}{m.group(2)}" if m else name


def opening_cat(name: str) -> str:
    return "door" if "_Door_" in name else "window" if "_Window_" in name else "opening"


def cluster_members(scan: str) -> dict[str, list[str]]:
    p = config.chair_clusters_json(scan)
    if not p.exists():
        return {}
    return {c["cluster_id"]: c["members"] for c in json.loads(p.read_text()).get("clusters", [])}


def entry(fname: str, prim_targets: list[str], cat: str, kind: str, source: str) -> dict:
    e = {
        "glb": f"glb/{fname}",
        "object": Path(fname).stem,
        "category": cat,
        "kind": kind,
        "source": source,
        "maps_to_prim": prim_targets[0],
    }
    if len(prim_targets) > 1:
        e["represents_prims"] = prim_targets
    return e


def from_scan(scan: str, glb_out: Path) -> list[dict]:
    """Gather every GLB under run/<scan>/reconstruct/ into glb_out + return entries."""
    recon = config.reconstruct_dir(scan)
    members = cluster_members(scan)
    assets, missing = [], []
    packed: set[str] = set()

    def take(src: Path, name: str, e: dict):
        if not src.exists():
            missing.append(str(src))
            return
        if name in packed:
            return
        packed.add(name)
        shutil.copy2(src, glb_out / f"{name}.glb")
        assets.append(e)

    if not recon.is_dir():
        print(f"  ! no reconstruct dir: {recon}")
        return assets

    # procedural / articulated: reconstruct/<name>/<name>.glb (dirs that carry object.py)
    for d in sorted(p for p in recon.iterdir() if p.is_dir() and p.name != "_refs"):
        name = d.name
        glb = d / f"{name}.glb"
        if not glb.exists():
            continue
        if is_opening(name):
            take(
                glb,
                name,
                entry(
                    f"{name}.glb",
                    [opening_target(name)],
                    opening_cat(name),
                    "articulated",
                    "procedural",
                ),
            )
        else:
            take(glb, name, entry(f"{name}.glb", [name], category_of(name), "static", "procedural"))

    # TRELLIS (neural) static objects + chairs: flat reconstruct/<name>.glb
    for glb in sorted(recon.glob("*.glb")):
        name = glb.stem
        if name in packed:
            continue
        if name.startswith("ChairCluster"):
            take(
                glb,
                name,
                entry(f"{name}.glb", members.get(name, [name]), "chair", "static", "TRELLIS"),
            )
        elif is_opening(name):
            take(
                glb,
                name,
                entry(
                    f"{name}.glb",
                    [opening_target(name)],
                    opening_cat(name),
                    "articulated",
                    "TRELLIS",
                ),
            )
        else:
            take(glb, name, entry(f"{name}.glb", [name], category_of(name), "static", "TRELLIS"))

    if missing:
        print("  (skipped, missing glb):")
        for m in missing:
            print(f"    {m}")
    return assets


def from_glb_dir(src_dir: Path, glb_out: Path, scan_hint: str | None) -> list[dict]:
    members = cluster_members(scan_hint) if scan_hint else {}
    assets = []
    for glb in sorted(src_dir.glob("*.glb")):
        name = glb.stem.split("__")[-1]
        shutil.copy2(glb, glb_out / f"{name}.glb")
        if is_opening(name):
            assets.append(
                entry(
                    f"{name}.glb",
                    [opening_target(name)],
                    opening_cat(name),
                    "articulated",
                    "glb-dir",
                )
            )
        elif name.startswith("ChairCluster"):
            assets.append(
                entry(f"{name}.glb", members.get(name, [name]), "chair", "static", "glb-dir")
            )
        else:
            assets.append(entry(f"{name}.glb", [name], category_of(name), "static", "glb-dir"))
    return assets


def pack(scan: str | None = None, glb_dir: str | None = None, out: Path | None = None) -> dict:
    """Pack a scan's reconstruct GLBs (or a flat glb dir) into ``out`` + write the manifest."""
    assets_dir = Path(out) if out else config.assets_dir(scan)
    glb_out = assets_dir / "glb"
    glb_out.mkdir(parents=True, exist_ok=True)
    for f in glb_out.glob("*.glb"):
        f.unlink()

    if glb_dir:
        assets = from_glb_dir(Path(glb_dir), glb_out, scan)
        scene = scan or Path(glb_dir).name
    else:
        assets = from_scan(scan, glb_out)
        scene = scan

    manifest = {
        "scene": scene,
        "glb_dir": "glb",
        "note": "Packed by room_ops.compile.pack_assets for build_room.py.",
        "assets": assets,
    }
    (assets_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"packed {len(assets)} assets -> {glb_out}")
    for a in assets:
        tgt = a.get("represents_prims") or [a["maps_to_prim"]]
        print(f"  {a['glb']:30} -> {', '.join(tgt):24} [{a['category']}/{a['kind']}/{a['source']}]")
    print(f"manifest -> {assets_dir / 'manifest.json'}")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--scan", help="scan name; gathers run/<scan>/reconstruct/")
    ap.add_argument("--glb-dir", help="instead: a flat folder of prim-named GLBs to pack")
    ap.add_argument("--out", help="assets dir to write (default run/<scan>/room/_assets)")
    args = ap.parse_args()
    if not args.scan and not args.glb_dir:
        ap.error("give --scan or --glb-dir")
    pack(scan=args.scan, glb_dir=args.glb_dir, out=Path(args.out) if args.out else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
