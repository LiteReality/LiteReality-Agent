#!/usr/bin/env python3
"""texture_recipe.py — reverse an object's textures/ directory into a textures.json recipe.

(en) Reverse-engineer an object's textures/ directory into a textures.json recipe.
For each texture: if its base id resolves on Poly Haven → {from:polyhaven, ...}; else if it's a diffuse →
{from:recolor, base:<a polyhaven base from the same object>, rgb:<mean colour>} (real-texture-recoloured, never
solid); else (rare non-polyhaven rough/normal) → {from:bundled} (the only case still stored as a binary).

Each texture is traced back to where it came from:
- the base id in the filename resolves on Poly Haven -> {from:polyhaven, id, map, res}
- otherwise, if it is a diffuse map -> {from:recolor, base:<a polyhaven base from this same
  object>, rgb:<mean colour>, ...}. That is the "real base texture + LAB recolour" rebuild —
  never a flat colour — and it prefers a polyhaven base the object already uses.
- otherwise (the rare non-polyhaven rough/nor) -> {from:bundled}, which still has to be stored
  as a real file in the definition.

So an object directory in the definition carries only a few KB of textures.json, and the image
files themselves are rebuilt at build time.
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import requests

MAP_ALIAS = {
    "diff": "Diffuse",
    "diffuse": "Diffuse",
    "rough": "Rough",
    "roughness": "Rough",
    "nor_gl": "nor_gl",
    "nor": "nor_gl",
    "nor_dx": "nor_dx",
    "ao": "AO",
    "disp": "Displacement",
    "arm": "arm",
    "spec": "Specular",
}
_poly_cache: dict[str, bool] = {}


def is_polyhaven(base_id: str) -> bool:
    if base_id not in _poly_cache:
        try:
            r = requests.get(f"https://api.polyhaven.com/files/{base_id}", timeout=15)
            _poly_cache[base_id] = r.status_code == 200 and "Diffuse" in r.json()
        except Exception:
            _poly_cache[base_id] = False
    return _poly_cache[base_id]


def parse_name(fname: str):
    """-> (base_id, map_type, res). e.g. oak_veneer_01_diff_1k.jpg -> (oak_veneer_01, Diffuse, 1k)."""
    stem = Path(fname).stem
    m = re.match(
        r"(.+?)_(diff|diffuse|rough|roughness|nor_gl|nor_dx|nor|ao|disp|arm|spec)(?:_(\d+k))?$",
        stem,
    )
    if not m:
        return stem, None, "1k"
    return m.group(1), MAP_ALIAS.get(m.group(2), m.group(2)), m.group(3) or "1k"


def _mean_rgb(path: Path):
    im = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    return [int(x) for x in im.reshape(-1, 3).mean(0)]


def _stddev(path: Path) -> float:
    return float(cv2.imread(str(path)).std())


def gen_recipe(tex_dir: Path):
    """-> (recipe dict, list of bundled filenames)."""
    tex_dir = Path(tex_dir)
    files = sorted(
        p.name for p in tex_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    parsed = {f: parse_name(f) for f in files}
    bases = {bid for bid, _, _ in parsed.values()}
    poly = {b: is_polyhaven(b) for b in bases}
    poly_bases = [b for b in bases if poly[b]]

    recipe, bundled = {}, []
    for f, (bid, mp, res) in parsed.items():
        if poly.get(bid):
            recipe[f] = {"from": "polyhaven", "id": bid, "map": mp or "Diffuse", "res": res}
        elif mp == "Diffuse":
            sd = _stddev(tex_dir / f)
            # Base: prefer a polyhaven base this object already uses; otherwise pick the
            # default for a flat colour vs a texture.
            base = (
                poly_bases[0] if poly_bases else ("beige_wall_001" if sd < 8 else "oak_veneer_01")
            )
            recipe[f] = {
                "from": "recolor",
                "base": base,
                "map": "Diffuse",
                "res": res,
                "rgb": _mean_rgb(tex_dir / f),
                "pattern_strength": 0.7 if sd < 8 else 1.0,
            }
        else:
            recipe[f] = {"from": "bundled"}
            bundled.append(f)
    return recipe, bundled
