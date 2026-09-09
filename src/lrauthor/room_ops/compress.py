"""Draco-compress a compiled room for the web.

`compile_room` produces a faithful `Room.glb` — uncompressed geometry, PNG textures, tens of
megabytes. That is the right thing to keep and the wrong thing to send a browser, so this makes the
small copy `litereality view` serves: same room, same animation clips, roughly a quarter the bytes.

Blender does the work, because it is already the thing that exported the glb and its glTF exporter
carries the Draco encoder. No Blender means no compression, which is a slower page rather than a
failure — the caller gets the original back and can serve that.
"""

from __future__ import annotations

import shutil
from pathlib import Path

_COMPRESS_SCRIPT = r"""
import bpy, sys
inp, outp = sys.argv[-2], sys.argv[-1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=inp)
kw = dict(filepath=outp, export_format='GLB', export_draco_mesh_compression_enable=True,
          export_draco_mesh_compression_level=6, export_image_format='JPEG',
          export_extras=True, export_animations=True)
try: bpy.ops.export_scene.gltf(export_jpeg_quality=75, **kw)
except TypeError: bpy.ops.export_scene.gltf(**kw)
"""


def compress(glb: Path) -> Path:
    """Draco-compress geometry + JPEG textures, keeping extras and animation clips so doors and
    drawers still articulate. Returns the ORIGINAL when Blender is missing or the result is no
    smaller, so callers can serve whatever comes back without checking which they got.

    Resolved through the canonical `paths.find_blender`, never `$LITEREALITY_BLENDER` raw: reading
    the env var directly meant that on a stock macOS install with Blender in /Applications and the
    variable unset, this silently returned the UNCOMPRESSED room every time."""
    import subprocess
    import tempfile

    from lrauthor.room_ops.paths import find_blender

    try:
        binp = Path(find_blender())
    except Exception:  # noqa: BLE001 — no Blender is a degraded export, not a failure
        return glb
    if not binp.exists():
        return glb
    out = Path(tempfile.mkdtemp()) / (glb.stem + "_c.glb")
    scr = out.parent / "_c.py"
    scr.write_text(_COMPRESS_SCRIPT)
    try:
        subprocess.run([str(binp), "-b", "--factory-startup", "--python", str(scr), "--",
                        str(glb), str(out)], capture_output=True, timeout=600)
    except Exception:  # noqa: BLE001
        return glb
    return out if out.is_file() and out.stat().st_size < glb.stat().st_size else glb


def compressed(glb: Path, cache: Path) -> Path:
    """`compress`, but keeping the result at `cache` so the next caller skips Blender entirely.

    Compressing a room takes a Blender launch; serving one takes none. `view` may be started many
    times against the same published room, and re-compressing on each is minutes of nothing.
    Reused only while the cache is at least as new as the glb, so a rebuilt room is never served
    inside a stale body.
    """
    try:
        if cache.is_file() and cache.stat().st_mtime >= glb.stat().st_mtime:
            return cache
    except OSError:
        pass
    made = compress(glb)
    if made != glb:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(made, cache)
            return cache
        except OSError:  # an unwritable cache is a slow serve, not a failed one
            pass
    return made
