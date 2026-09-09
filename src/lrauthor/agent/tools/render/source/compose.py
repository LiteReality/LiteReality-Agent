"""compose.py — the image-render-tool: build render-vs-real comparison images on demand.

Three layers of comparison, all render the CURRENT room (a ``room/Room.py``) and pair it with
the real capture evidence:

  1. render_frames(room, frames)     — per FRAME: render from that exact ARKit camera pose
                                        | the real photo of that frame                (oblique, in-context)
  2. render_surface(room, walls)     — per WALL: the rendered head-on ORTHO
                                        | the real rectified STITCH of that wall       (flat, like-for-like)
  3. render_surface(room, walls,     — same as (2) PLUS the curated SHARP reference
                    with_refs=True)     frames that cover the wall (boxed + labelled)

Each returns a dict of {frame|wall: output_png}. The tool sets ``$LITEREALITY_SCAN`` so the
harness config resolves the scan's paths (raw frames, surface stitches, wall references).

CLI:  python -m lrauthor.agent.tools.render.source <frames|surface> --room <room dir> [...]
API:  from lrauthor.agent.tools.render.source import render_frames, render_surface
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from lrauthor.agent.tools.shared.scan import config_for as _config_for
from lrauthor.agent.tools.shared.scan import scan_from_room as _scan_from_room
from lrauthor.fonts import font as _shared_font

_HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# small PIL helpers (labelled side-by-side panels)
# --------------------------------------------------------------------------- #
def _font(size: int):
    try:
        return _shared_font(size)
    except Exception:
        return ImageFont.load_default()


def _hstack(items: list[tuple], height: int = 560, gap: int = 8) -> Image.Image:
    """Resize each panel to a common height, THEN draw its label at a fixed readable size,
    and stack left→right on a dark strip.

    items: list of (image, label_text, label_color). Labelling AFTER the resize is deliberate —
    drawing the label on the full-res source then scaling the whole panel down would shrink the
    text to a few invisible pixels.
    """
    bar_h = max(28, height // 18)  # label bar scales a little with panel height, always readable
    font = _font(int(bar_h * 0.68))
    panels = []
    for im, text, color in items:
        p = im.convert("RGB").resize((max(1, int(im.width * height / im.height)), height))
        d = ImageDraw.Draw(p)
        d.rectangle((0, 0, p.width, bar_h), fill=(0, 0, 0))
        d.text((6, max(1, (bar_h - font.size) // 2)), text, fill=color, font=font)
        panels.append(p)
    W = sum(p.width for p in panels) + gap * (len(panels) + 1)
    out = Image.new("RGB", (W, height + 2 * gap), (12, 14, 18))
    x = gap
    for p in panels:
        out.paste(p, (x, gap))
        x += p.width + gap
    return out


def _vstack(rows: list[Image.Image], gap: int = 8, bg=(12, 14, 18)) -> Image.Image:
    """Stack panels top→bottom, centred, on a dark strip (for surface-comparison + ref rows)."""
    W = max(r.width for r in rows)
    H = sum(r.height for r in rows) + gap * (len(rows) + 1)
    out = Image.new("RGB", (W, H), bg)
    y = gap
    for r in rows:
        out.paste(r, ((W - r.width) // 2, y))
        y += r.height + gap
    return out


def _frame_int(frame) -> int | None:
    """'frame_00058' | 58 | '58' -> 58 (or None)."""
    try:
        return int(str(frame).replace("frame_", "").lstrip("0") or "0")
    except ValueError:
        return None







def _run_blender_frames(room_dir: Path, spec: str, config, renders: Path) -> None:
    """Render the given frames (spec: 'all' | '10,20' | '0-40') from room_dir into `renders`,
    upright, + a render_manifest.json (object visibility). Clears stale renders first."""
    renders.mkdir(parents=True, exist_ok=True)
    for f in renders.glob("frame_*.png"):
        f.unlink()
    log_path = renders.parent / "render.log"
    log = log_path.open("w")
    try:
        subprocess.run(
            [
                config.blender(), "-b", "--python", str(_HERE / "_blender_frames.py"), "--",
                str(config.ROOT), str(room_dir / "Room.py"), str(config.SCAN_DIR),
                str(config.ASSETS_DIR), str(renders), spec,
            ],
            check=True, cwd=str(config.ROOT), stdout=log, stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError:
        # surface the ACTUAL failure (usually a runtime error in the agent's Room.py edit)
        tail = log_path.read_text(errors="ignore").splitlines()
        cause = [ln for ln in tail if "ROOM RENDER FAILED" in ln or "Error" in ln][-3:]
        raise RuntimeError(
            "Blender render failed — " + (" | ".join(c.strip() for c in cause) or "see render.log")
            + f" (full log: {log_path})"
        ) from None
    finally:
        log.close()


# --------------------------------------------------------------------------- #
# Frame side-by-side — three modes (scene / wall / object). Each renders the frames once, then
# annotates BOTH the render AND the real photo (same camera pose → the same overlay lines up).
# --------------------------------------------------------------------------- #
_MAIN_RE = None


def _main_re():
    global _MAIN_RE
    if _MAIN_RE is None:
        import re

        _MAIN_RE = re.compile(
            r"^(Wall|Floor|Ceiling|Door|Window|Opening|Table|Chair|Sofa|Bed|Storage|"
            r"Television|Refrigerator|Stove|Oven|Sink|Toilet|Bathtub|Stairs|Fireplace)\d+$"
        )
    return _MAIN_RE


def _frame_prep(room_dir, frames, scan, out_dir, sub):
    room_dir = Path(room_dir).resolve()
    scan = scan or _scan_from_room(room_dir)
    config = _config_for(scan)
    out_dir = Path(out_dir) if out_dir else (room_dir.parent / "_image_render_tool" / sub)
    out_dir.mkdir(parents=True, exist_ok=True)
    renders = out_dir / "_renders"
    spec = frames if isinstance(frames, str) else ",".join(str(int(f)) for f in frames)
    _run_blender_frames(room_dir, spec, config, renders)
    man = json.loads((renders / "render_manifest.json").read_text())
    visible = {fr["frame"]: fr["visible"] for fr in man["frames"]}
    return room_dir, config, out_dir, renders, visible


def _real_upright(config, idx: int, size) -> Image.Image:
    """The real photo for frame idx, rotated upright + resized to match the render (same pose)."""
    im = Image.open(config.SCAN_DIR / f"frame_{idx:05d}.jpg").convert("RGB")
    im = im.transpose(Image.Transpose.ROTATE_270)
    return im.resize(tuple(size)) if im.size != tuple(size) else im


def _iter_frames(renders, config, visible):
    """Yield (idx, fid, render_img, real_img, visible_objs) for each rendered frame with a photo."""
    for rp in sorted(renders.glob("frame_*.png")):
        idx = int(rp.stem.split("_")[1])
        fid = f"frame_{idx:05d}"
        if not (config.SCAN_DIR / f"{fid}.jpg").is_file():
            continue
        rend = Image.open(rp).convert("RGB")
        yield idx, fid, rp, rend, _real_upright(config, idx, rend.size), visible.get(fid, [])


def render_scene(room_dir, frames, *, scan=None, out_dir=None, height: int = 560) -> dict[int, str]:
    """SCENE mode: label EVERY visible object (walls, doors, windows, furniture) as a chip, on
    BOTH the render and the real photo. Returns {frame_index: side_by_side_png}."""
    _, config, out_dir, renders, visible = _frame_prep(room_dir, frames, scan, out_dir, "scene")
    from lrauthor.agent.tools.shared import overlay as _overlay

    from ._annotate import draw_chips, draw_wall_edges

    planes = _overlay.load_planes(config)
    out: dict[int, str] = {}
    for idx, fid, _rp, rend, real, objs in _iter_frames(renders, config, visible):
        W, H = rend.size
        marks = [
            {"id": o["id"], "category": o["category"], "dist_m": o.get("dist_m"),
             "label": [o["label_pct"][0] / 100 * W, o["label_pct"][1] / 100 * H]}
            for o in objs if _main_re().match(o["id"])
        ]
        proj = _overlay.project_walls(config.SCAN_DIR / f"{fid}.json", planes, config, W, H)
        for m in marks:  # walls: use the projected TRUE centre (not the visible-centroid)
            if m["category"] == "wall" and m["id"] in proj:
                m["label"] = [proj[m["id"]]["cx"], proj[m["id"]]["cy"]]
        edges = [seg for pr in proj.values() for seg in pr["edges"]]
        for im in (rend, real):
            draw_wall_edges(im, edges)
            draw_chips(im, [dict(m) for m in marks])
        out[idx] = _save_pair(rend, real, out_dir, f"scene_frame_{idx:05d}", idx, "scene", height)
    return out


def render_wall(room_dir, frames, *, scan=None, out_dir=None, height: int = 560) -> dict[int, str]:
    """WALL mode: outline + name every visible wall on BOTH the render and the real photo."""
    _, config, out_dir, renders, visible = _frame_prep(room_dir, frames, scan, out_dir, "wall")
    from lrauthor.agent.tools.shared import overlay as _overlay

    from ._annotate import draw_chips, draw_wall_edges

    planes = _overlay.load_planes(config)
    out: dict[int, str] = {}
    for idx, fid, _rp, rend, real, _objs in _iter_frames(renders, config, visible):
        W, H = rend.size
        proj = _overlay.project_walls(config.SCAN_DIR / f"{fid}.json", planes, config, W, H)
        edges = [seg for pr in proj.values() for seg in pr["edges"]]
        marks = [{"id": n, "category": "wall", "dist_m": 0, "label": [pr["cx"], pr["cy"]]}
                 for n, pr in proj.items()]
        for im in (rend, real):
            draw_wall_edges(im, edges)
            draw_chips(im, [dict(m) for m in marks])
        out[idx] = _save_pair(rend, real, out_dir, f"wall_frame_{idx:05d}", idx, "walls", height)
    return out


def render_object(room_dir, frames, obj: str, *, scan=None, out_dir=None, height: int = 560) -> dict[int, str]:
    """OBJECT mode: draw a bbox around ONE object (`obj`, e.g. 'Table0') + its name to the side,
    on BOTH the render and the real photo. Frames where it isn't visible are still paired (no box)."""
    _, config, out_dir, renders, visible = _frame_prep(room_dir, frames, scan, out_dir, "object")
    from ._annotate import draw_object_box

    out: dict[int, str] = {}
    for idx, fid, _rp, rend, real, objs in _iter_frames(renders, config, visible):
        W, H = rend.size
        rec = next((o for o in objs if o["id"] == obj), None)
        if rec and rec.get("bbox_pct"):
            x0, y0, x1, y1 = rec["bbox_pct"]
            box = [x0 / 100 * W, y0 / 100 * H, x1 / 100 * W, y1 / 100 * H]
            for im in (rend, real):
                draw_object_box(im, box, obj)
        out[idx] = _save_pair(rend, real, out_dir, f"{obj}_frame_{idx:05d}", idx, obj, height)
    return out


def render_wall_focus(room_dir, frames, wall: str, *, scan=None, out_dir=None, height: int = 560) -> dict[int, str]:
    """WALL-FOCUS mode: draw a BBOX around ONLY `wall` (e.g. 'Wall2') + its name to the side, on
    BOTH the render and the real photo, per frame — like render_object, but for one wall. The box
    is the wall's projected extent (clamped to the frame). Frames where the wall isn't visible are
    still paired (no box). Returns {frame_index: side_by_side_png}."""
    _, config, out_dir, renders, visible = _frame_prep(room_dir, frames, scan, out_dir, "wall_focus")
    from lrauthor.agent.tools.shared import overlay as _overlay

    from ._annotate import WALL_ORANGE, draw_object_box

    planes = _overlay.load_planes(config)
    out: dict[int, str] = {}
    for idx, fid, _rp, rend, real, _objs in _iter_frames(renders, config, visible):
        W, H = rend.size
        proj = _overlay.project_walls(config.SCAN_DIR / f"{fid}.json", planes, config, W, H, only=wall)
        pts = [pt for pr in proj.values() for seg in pr["edges"] for pt in seg]
        if pts:  # bbox of the projected wall, clamped to the frame
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            x0, y0 = max(0.0, min(xs)), max(0.0, min(ys))
            x1, y1 = min(float(W), max(xs)), min(float(H), max(ys))
            if x1 - x0 > 8 and y1 - y0 > 8:  # wall actually occupies a region of THIS frame
                for im in (rend, real):
                    draw_object_box(im, [x0, y0, x1, y1], wall, color=WALL_ORANGE)
        out[idx] = _save_pair(rend, real, out_dir, f"{wall}_frame_{idx:05d}", idx, wall, height)
    return out


def render_walls_focus_batch(
    room_dir, frames_by_wall: dict, *, scan=None, out_dir=None, height: int = 560
) -> dict:
    """BATCHED wall-focus: ONE Blender pass for the union of all walls' frames, then per-wall
    annotation (cheap PIL). Sequentially this costs one scene build + render per WALL; batched it
    is one pass total — the observe phase's main speedup. Returns {wall: {frame: png}}."""
    all_frames = sorted({f for fs in frames_by_wall.values() for f in fs})
    _, config, out_dir, renders, visible = _frame_prep(room_dir, all_frames, scan, out_dir, "wall_focus")
    from lrauthor.agent.tools.shared import overlay as _overlay

    from ._annotate import WALL_ORANGE, draw_object_box

    planes = _overlay.load_planes(config)
    out: dict = {w: {} for w in frames_by_wall}
    for wall, frames in frames_by_wall.items():
        want = set(frames)
        for idx, fid, _rp, rend, real, _objs in _iter_frames(renders, config, visible):
            if idx not in want:
                continue
            W, H = rend.size
            proj = _overlay.project_walls(
                config.SCAN_DIR / f"{fid}.json", planes, config, W, H, only=wall
            )
            pts = [pt for pr in proj.values() for seg in pr["edges"] for pt in seg]
            if pts:
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                x0, y0 = max(0.0, min(xs)), max(0.0, min(ys))
                x1, y1 = min(float(W), max(xs)), min(float(H), max(ys))
                if x1 - x0 > 8 and y1 - y0 > 8:
                    for im in (rend, real):
                        draw_object_box(im, [x0, y0, x1, y1], wall, color=WALL_ORANGE)
            out[wall][idx] = _save_pair(rend, real, out_dir, f"{wall}_frame_{idx:05d}", idx, wall, height)
    return out


def _save_pair(rend, real, out_dir, name, idx, tag, height) -> str:
    sheet = _hstack(
        [
            (rend, f"RENDER · {tag} · frame {idx}", (150, 220, 255)),
            (real, f"REAL · {tag} · frame {idx}", (160, 255, 170)),
        ],
        height=height,
    )
    op = out_dir / f"{name}.png"
    sheet.save(op)
    return str(op)


# back-compat: the old name is the scene mode
render_frames = render_scene


# --------------------------------------------------------------------------- #
# Layers 2 & 3 — per-wall stitch (real) | ortho render (+ references)
# --------------------------------------------------------------------------- #

def _align_floorlike(ortho_im, stitch_im, known_mask, unknown_mask):
    """Floor/Ceiling only: the axis convention fixes 90° ambiguity but not 180°/mirror. Resolve
    them EMPIRICALLY: compare the render's floor SILHOUETTE against the stitch's own coverage
    masks across all 8 orientations (4 rotations × mirror) and keep the best-IoU transform."""
    import numpy as np

    # stitch silhouette = union of the known+unknown coverage masks (the polygon on its canvas)
    sm = None
    for mp in (known_mask, unknown_mask):
        try:
            m = np.array(Image.open(mp).convert("L").resize((128, 128))) > 16
            sm = m if sm is None else (sm | m)
        except Exception:  # noqa: BLE001
            pass
    if sm is None or sm.sum() < 64:
        return ortho_im

    # render silhouette = pixels that differ from the background grey (sampled at the corners)
    a = np.array(ortho_im.convert("RGB").resize((128, 128)), dtype=float)
    corners = np.concatenate([a[:6, :6].reshape(-1, 3), a[:6, -6:].reshape(-1, 3),
                              a[-6:, :6].reshape(-1, 3), a[-6:, -6:].reshape(-1, 3)])
    bg = np.median(corners, axis=0)
    rm = (np.abs(a - bg).sum(axis=2) > 60)
    if rm.sum() < 64:
        return ortho_im

    ops = [[], [Image.ROTATE_90], [Image.ROTATE_180], [Image.ROTATE_270],
           [Image.FLIP_LEFT_RIGHT], [Image.FLIP_LEFT_RIGHT, Image.ROTATE_90],
           [Image.FLIP_LEFT_RIGHT, Image.ROTATE_180], [Image.FLIP_LEFT_RIGHT, Image.ROTATE_270]]

    def iou(mask):
        inter = (mask & sm).sum()
        union = (mask | sm).sum()
        return inter / union if union else 0.0

    best, best_ops = -1.0, []
    rimg = Image.fromarray((rm * 255).astype("uint8"))
    for seq in ops:
        t = rimg
        for op in seq:
            t = t.transpose(op)
        score = iou(np.array(t.resize((128, 128))) > 127)
        if score > best:
            best, best_ops = score, seq
    out = ortho_im
    for op in best_ops:
        out = out.transpose(op)
    return out


def render_wall_reference(
    room_dir: str | Path,
    walls: list[str] | str | None = None,
    *,
    scan: str | None = None,
    out_dir: str | Path | None = None,
    with_refs: bool = True,
    height: int = 480,
) -> dict[str, str]:
    """WALL-REFERENCE mode: per wall, the rendered head-on ORTHO | real rectified STITCH, and (with
    refs, the default) one WALL-FOCUSED render|real image per covering reference frame — each a
    SEPARATE file. walls: ['Wall0','Wall2'] | 'Wall0' | None = every surface.

    Returns {key: png}: the head-on comparison under the wall name (e.g. 'Wall0'), and each
    reference-frame comparison under '<wall>_frame_<NNNNN>' (e.g. 'Wall0_frame_00034').
    """
    room_dir = Path(room_dir).resolve()
    scan = scan or _scan_from_room(room_dir)
    config = _config_for(scan)
    out_dir = Path(out_dir) if out_dir else (room_dir.parent / "_image_render_tool" / "surface")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. render every surface head-on (one Blender pass) via the existing surface_compare
    from lrauthor.agent.tools.render.source import surface_compare

    res = surface_compare.build(room_dir, out_dir / "_sc")
    renders: dict[str, str] = res["renders"]  # {surface: ortho_png}

    # orientation flags (render_ortho writes these) so the ortho render lines up with the real
    # stitch — same convention as the human surface_compare sheet: rflip -> flip the render,
    # sflip -> flip the stitch. Without this the render is mirrored vs the stitch for some walls.
    flips: dict = {}
    oj = out_dir / "_sc" / "render" / "orient.json"
    if oj.is_file():
        try:
            flips = json.loads(oj.read_text())
        except Exception:
            flips = {}

    # 2. curated wall references (only when asked — it's its own selection pass). Keep each
    #    tile's frame number so the panel can be labelled "Wall0 · frame N" like the reference.
    ref_tiles: dict[str, list[tuple]] = {}
    if with_refs:
        from lrauthor.agent.tools.render.source import wall_refs

        for s in wall_refs.build().get("surfaces", []):
            ref_tiles[s["name"]] = [(t["path"], t.get("frame", "")) for t in s.get("tiles", [])]

    want = _normalize_walls(walls, list(renders))

    # with refs: render each wall's reference FRAMES once (one Blender pass), RAW — so we can
    # WALL-FOCUS: box only the wall of interest on each ref frame (not every object).
    raw_dir = out_dir / "_ref_frames" / "_renders"
    planes = None
    if with_refs:
        ref_frames = sorted(
            {fi for name in want for _, f in ref_tiles.get(name, []) if (fi := _frame_int(f)) is not None}
        )
        if ref_frames:
            _run_blender_frames(room_dir, ",".join(map(str, ref_frames)), config, raw_dir)
            from lrauthor.agent.tools.shared import overlay as _overlay

            planes = _overlay.load_planes(config)

    out: dict[str, str] = {}
    for name in want:
        ortho = renders.get(name)
        stitch = config.SURFACE_REF / f"{name}_stitched.jpg"
        if not ortho or not Path(ortho).is_file() or not stitch.is_file():
            continue
        # top: the head-on surface comparison (render ortho | real stitch), wall name on each.
        # Apply the orient.json flips so the render lines up with the stitch (else mirrored).
        ortho_im = Image.open(ortho).convert("RGB")
        stitch_im = Image.open(stitch).convert("RGB")
        fl = flips.get(name) if isinstance(flips.get(name), dict) else {}
        if fl.get("rflip"):
            ortho_im = ortho_im.transpose(Image.FLIP_LEFT_RIGHT)
        if fl.get("sflip"):
            stitch_im = stitch_im.transpose(Image.FLIP_LEFT_RIGHT)
        if name in ("Floor0", "Ceiling0"):
            # 180°/mirror are convention-ambiguous for floor-like surfaces — align by silhouette
            ortho_im = _align_floorlike(
                ortho_im, stitch_im,
                config.SURFACE_REF / f"{name}_stitched_known_mask.png",
                config.SURFACE_REF / f"{name}_stitched_unknown_mask.png",
            )
        surface_row = _hstack(
            [
                (ortho_im, f"RENDER · {name}", (150, 220, 255)),
                (stitch_im, f"REAL stitch · {name}", (160, 255, 170)),
            ],
            height=height,
        )
        op = out_dir / f"{name}_compare.png"
        surface_row.save(op)
        out[name] = str(op)
        if not with_refs:
            continue
        # each covering reference frame -> its OWN render|real image (SEPARATE, not stacked);
        # keyed "{wall}_frame_{NNNNN}" so callers get one file per comparison.
        for tile_path, frame in ref_tiles.get(name, []):
            fi = _frame_int(frame)
            raw = raw_dir / f"frame_{fi:05d}.png" if fi is not None else None
            if not (raw and raw.is_file() and Path(tile_path).is_file()):
                continue
            boxed = _wall_focus_render(raw, name, fi, planes, config, out_dir / "_ref_frames")
            row = _hstack(
                [
                    (Image.open(boxed), f"RENDER · {name} · frame {fi}", (150, 220, 255)),
                    (Image.open(tile_path), f"REAL · {name} · frame {fi}", (255, 210, 130)),
                ],
                height=height,
            )
            rp = out_dir / f"{name}_frame_{fi:05d}.png"
            row.save(rp)
            out[f"{name}_frame_{fi:05d}"] = str(rp)
    return out


def _wall_focus_render(raw_png: Path, wall: str, fi: int, planes, config, dst_dir: Path) -> Path:
    """Outline + name ONLY `wall` on a copy of a raw frame render (wall-focus)."""
    from lrauthor.agent.tools.shared import overlay as _overlay

    from ._annotate import draw_chips, draw_wall_edges

    img = Image.open(raw_png).convert("RGB")
    Wu, Hu = img.size
    proj = _overlay.project_walls(config.SCAN_DIR / f"frame_{fi:05d}.json", planes, config, Wu, Hu, only=wall)
    edges = [seg for pr in proj.values() for seg in pr["edges"]]
    draw_wall_edges(img, edges, focus=True)
    draw_chips(img, [{"id": wall, "category": "wall", "dist_m": 0, "label": [pr["cx"], pr["cy"]]} for pr in proj.values()])
    out = dst_dir / f"{wall}_frame_{fi:05d}.png"
    img.save(out)
    return out


# back-compat alias
render_surface = render_wall_reference


def _normalize_walls(walls, available: list[str]) -> list[str]:
    if walls is None or walls == "all":
        return sorted(available)
    if isinstance(walls, str):
        walls = [w.strip() for w in walls.replace(",", " ").split() if w.strip()]
    return [w for w in walls if w in available] or list(walls)
