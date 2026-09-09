#!/usr/bin/env python3
"""Download a Poly Haven PBR texture set (diffuse + roughness + normal).

Usage: python3 fetch_polyhaven.py <asset_id> <out_dir> [resolution]
e.g.:  python3 fetch_polyhaven.py plywood ./textures 1k

Exits non-zero if the diffuse map can't be fetched, so callers can fall back
to a procedural texture.
"""

import os
import sys
import urllib.request

BASE = "https://dl.polyhaven.org/file/ph-assets/Textures"


def fetch(url, dest):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
            f.write(r.read())
        print(f"OK  {os.path.basename(dest)}  ({os.path.getsize(dest) // 1024} KB)")
        return True
    except Exception as e:
        print(f"FAIL {url}: {e}")
        return False


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    asset, out = sys.argv[1], sys.argv[2]
    res = sys.argv[3] if len(sys.argv) > 3 else "1k"
    os.makedirs(out, exist_ok=True)

    maps = [
        (
            f"{BASE}/jpg/{res}/{asset}/{asset}_diff_{res}.jpg",
            os.path.join(out, f"{asset}_diff_{res}.jpg"),
            True,
        ),
        (
            f"{BASE}/jpg/{res}/{asset}/{asset}_rough_{res}.jpg",
            os.path.join(out, f"{asset}_rough_{res}.jpg"),
            False,
        ),
        (
            f"{BASE}/png/{res}/{asset}/{asset}_nor_gl_{res}.png",
            os.path.join(out, f"{asset}_nor_gl_{res}.png"),
            False,
        ),
    ]
    ok_all = True
    for url, dest, required in maps:
        ok = fetch(url, dest)
        if not ok:
            if os.path.exists(dest):
                os.remove(dest)
            if required:
                sys.exit(f"ERROR: required diffuse map failed for '{asset}'")
            ok_all = False
    print("DONE" + ("" if ok_all else " (some optional maps missing)"))


main()
