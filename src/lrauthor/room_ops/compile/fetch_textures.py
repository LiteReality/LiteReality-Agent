#!/usr/bin/env python3
"""fetch_textures.py — rebuild textures from a textures.json recipe, so the definition stores
no image files of its own.

(en) Materialize textures from a textures.json recipe so the definition stores no binaries.
Each recipe entry is either `from:polyhaven` (download a specific map at a given res) or `from:recolor`
(download a Poly Haven base diffuse, then LAB-shift it to a target rgb while preserving its pattern).
Downloads are md5-checked and cached locally at ~/.litereality_texcache so shared bases download once.

The recipe, textures.json — one per object, a few KB, living in the Room/ definition:
{
  "oak_veneer_01_diff_1k.jpg": {"from":"polyhaven","id":"oak_veneer_01","map":"Diffuse","res":"1k"},
  "oak_veneer_01_rough_1k.jpg":{"from":"polyhaven","id":"oak_veneer_01","map":"Rough","res":"1k"},
  "oak_veneer_01_nor_gl_1k.png":{"from":"polyhaven","id":"oak_veneer_01","map":"nor_gl","res":"1k"},
  "paint_white_diff_1k.jpg":   {"from":"recolor","base":"beige_wall_001","map":"Diffuse","res":"1k","rgb":[209,207,208],"pattern_strength":1.0}
}

- from=polyhaven: download one map for that id from Poly Haven.
- from=recolor: download the base's Diffuse, then LAB-recolour it to rgb, keeping the pattern.

Downloads are md5-checked and cached locally (~/.litereality_texcache, keyed by url, so several
objects sharing one base fetch it only once).

    python fetch_textures.py <textures.json> <out_dir>
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import requests

API = "https://api.polyhaven.com/files/{id}"
CACHE = Path(os.environ.get("LITEREALITY_TEXCACHE", Path.home() / ".litereality_texcache"))
_FILES_CACHE: dict[str, dict] = {}


def _api_files(asset_id: str) -> dict:
    if asset_id not in _FILES_CACHE:
        r = requests.get(API.format(id=asset_id), timeout=30)
        r.raise_for_status()
        _FILES_CACHE[asset_id] = r.json()
    return _FILES_CACHE[asset_id]


def _pick_url(asset_id: str, map_name: str, res: str):
    """Returns (url, md5, ext). map_name is e.g. Diffuse/Rough/nor_gl; res is e.g. 1k/2k."""
    files = _api_files(asset_id)
    if map_name not in files:
        raise KeyError(f"{asset_id} has no map '{map_name}' (available: {list(files)})")
    by_res = files[map_name].get(res) or next(iter(files[map_name].values()))
    fmt = "jpg" if "jpg" in by_res else next(iter(by_res))  # prefer jpg
    info = by_res[fmt]
    return info["url"], info.get("md5"), fmt


def _download(url: str, md5: str | None) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(url.encode()).hexdigest()[:16]
    dst = CACHE / f"{key}_{Path(url).name}"
    if dst.exists():
        return dst
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    if md5 and hashlib.md5(r.content).hexdigest() != md5:
        raise IOError(f"md5 mismatch: {url}")
    dst.write_bytes(r.content)
    return dst


def materialize(recipe: dict, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for fname, spec in recipe.items():
        out = out_dir / fname
        if spec["from"] == "polyhaven":
            url, md5, _ = _pick_url(spec["id"], spec["map"], spec.get("res", "1k"))
            src = _download(url, md5)
            out.write_bytes(src.read_bytes())
        elif spec["from"] == "recolor":
            import cv2

            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from recolor import recolor_to_rgb

            url, md5, _ = _pick_url(spec["base"], spec.get("map", "Diffuse"), spec.get("res", "1k"))
            base = _download(url, md5)
            img = cv2.imread(str(base), cv2.IMREAD_COLOR)
            rec = recolor_to_rgb(img, spec["rgb"], spec.get("pattern_strength", 1.0))
            cv2.imwrite(str(out), rec)
        elif spec["from"] == "bundled":
            continue  # a fallback texture that cannot be rebuilt: the caller copies the real
            # file across from the definition's textures/
        else:
            raise ValueError(f"unknown from: {spec['from']} ({fname})")
        made.append(fname)
    return made


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    recipe = json.loads(Path(sys.argv[1]).read_text())
    made = materialize(recipe, Path(sys.argv[2]))
    print(f"rebuilt {len(made)} textures -> {sys.argv[2]}")
    for m in made:
        print(f"  {m}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
