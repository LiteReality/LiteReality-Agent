"""report.py — the layout pass, drawn, so the run leaves evidence rather than a number.

``layout_report.json`` says ``4 violations -> 0, moved 1, dropped 2``. That is the right thing to
assert and the wrong thing to trust: a violation count cannot distinguish a repair that put a
counter back against its wall from one that shoved a table half a metre across the room, and both
read as zero. So every run writes a picture of what it actually did.

One self-contained HTML file per scan, two floor plans side by side in the SAME projection:

    as scanned          the boxes the capture produced, with every violating object outlined red
    after the pass      the same room repaired, the scanned positions ghosted underneath in dashed
                        outline and each correction drawn as an arrow from where the box WAS

Ghost-plus-arrow is the whole design. A before/after pair you have to flick between hides small
moves and exaggerates large ones; drawn on top of each other, a 3 cm reseat and a 50 cm slide are
told apart at a glance, which is exactly the judgement the score cannot make.

Deliberately dependency-free: inline SVG built as text, no matplotlib, no CDN, nothing to resolve.
It renders from ``file://``, survives being emailed, and — since it costs a few milliseconds of
string building — can run on every scan without anyone deciding whether it is worth it.
"""

from __future__ import annotations

import html
import math
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from .shell import facing_yaw, object_footprint, opening_span, wall_frame

__all__ = ["render", "write"]

# One palette shared by both panels, so an object is the same colour before and after and the eye
# tracks it across the pair. Shared with `physical_engine.plotting` — keep them in step.
CATEGORY_COLORS = {
    "wall": "#4a5568", "floor": "#cbd5e0", "ceiling": "#e2e8f0",
    "door": "#dd6b20", "window": "#3182ce", "opening": "#805ad5",
    "chair": "#38a169", "stool": "#48bb78", "table": "#d69e2e", "desk": "#b7791f",
    "sofa": "#9f7aea", "bed": "#ed64a6", "storage": "#0987a0", "cabinet": "#00a3c4",
    "television": "#e53e3e", "refrigerator": "#2c7a7b", "stove": "#c05621", "oven": "#9c4221",
    "sink": "#2b6cb0", "toilet": "#4299e1", "bathtub": "#63b3ed", "washer": "#667eea",
    "dishwasher": "#5a67d8", "fireplace": "#c53030", "stairs": "#718096",
}
_FALLBACK = ["#7f9cf5", "#f6ad55", "#68d391", "#fc8181", "#b794f4", "#4fd1c5", "#f687b3"]

BAD = "#e53e3e"          # a violation, before or remaining
FIXED = "#38a169"        # a violation the pass cleared
GHOST = "#a0aec0"        # where the scan put it

PANEL_W = 640.0          # px available to one plan
PANEL_H = 540.0
MARGIN = 0.45            # metres of room drawn around the outermost geometry
MOVED_TOL = 0.01         # under a centimetre is arithmetic, not a move
FLOOR_FACE_CAP = 4000    # a pathological floor mesh must not produce a 50 MB file


def category_color(category: str) -> str:
    """A stable colour for a category, including the compound names a box merge produces.

    ``Oven_Storage_Stove0`` is one object made of three, and colouring it by hash makes the merged
    unit look unrelated to the run it replaced. Taking the first token that IS a known category
    keeps it the colour of the thing it mostly is. The fallback hashes with CRC32 rather than
    ``hash()``, whose string seed is randomised per process — the same room would otherwise come
    out a different colour on every run, and a picture you cannot compare to yesterday's is worth
    much less than one you can.
    """
    if category in CATEGORY_COLORS:
        return CATEGORY_COLORS[category]
    for token in category.split("_"):
        if token in CATEGORY_COLORS:
            return CATEGORY_COLORS[token]
    return _FALLBACK[zlib.crc32(category.encode()) % len(_FALLBACK)]


def _e(text: Any) -> str:
    return html.escape(str(text), quote=True)


def _n(value: float) -> str:
    """Coordinates at 0.1 px. Full float repr triples the file size and shows nothing."""
    return f"{value:.1f}"


class _View:
    """One world→pixel projection, shared by both panels so they are directly comparable.

    Fitting each panel to its own contents would be the obvious thing and is wrong: the repaired
    room is very slightly smaller (a dropped duplicate, a trimmed counter), so a per-panel fit
    scales it up a fraction and every object in the room appears to have shifted.
    """

    def __init__(self, shells: list[dict[str, Any]]):
        xs: list[float] = []
        ys: list[float] = []
        for shell in shells:
            for wall in (shell.get("walls") or {}).values():
                for point in (wall["start"], wall["end"]):
                    xs.append(float(point[0]))
                    ys.append(float(point[1]))
            for vert in ((shell.get("floor") or {}).get("verts") or []):
                xs.append(float(vert[0]))
                ys.append(float(vert[1]))
            for obj in (shell.get("objects") or {}).values():
                for corner in object_footprint(obj):
                    xs.append(float(corner[0]))
                    ys.append(float(corner[1]))
        if not xs:
            xs, ys = [0.0, 1.0], [0.0, 1.0]
        self.min_x, self.max_x = min(xs) - MARGIN, max(xs) + MARGIN
        self.min_y, self.max_y = min(ys) - MARGIN, max(ys) + MARGIN
        span_x = max(self.max_x - self.min_x, 1e-3)
        span_y = max(self.max_y - self.min_y, 1e-3)
        self.scale = min(PANEL_W / span_x, PANEL_H / span_y)
        self.width = span_x * self.scale
        self.height = span_y * self.scale

    def pt(self, x: float, y: float) -> tuple[float, float]:
        """World metres → SVG pixels. Y is flipped: the plan is Z-up, the canvas is not."""
        return ((float(x) - self.min_x) * self.scale,
                (self.max_y - float(y)) * self.scale)

    def poly(self, points) -> str:
        return " ".join(f"{_n(a)},{_n(b)}" for a, b in (self.pt(p[0], p[1]) for p in points))


def _violation_index(violations) -> dict[str, list[str]]:
    """object id → the violations naming it, both as subject and as the thing it collides with."""
    index: dict[str, list[str]] = {}
    for violation in violations or []:
        text = f"{violation.kind}: {violation.detail}"
        index.setdefault(violation.object, []).append(text)
        if violation.other:
            index.setdefault(violation.other, []).append(text)
    return index


def _floor_path(shell: dict[str, Any], view: _View) -> str:
    """The floor mesh as ONE path — every triangle wound the same way, which is load-bearing.

    A RoomPlan floor arrives as a triangle soup with no consistent orientation, and SVG's default
    ``nonzero`` fill rule reads a clockwise triangle laid over a counter-clockwise one as a hole.
    Emitted as they come, the room's triangles cancel each other out and the plate renders as
    *nothing at all* — not a wrong shape, an invisible one, which is exactly the kind of bug a
    picture is supposed to catch rather than contain. Flipping every negative-area triangle makes
    the winding uniform, so nonzero unions them into the one region meant by the mesh.
    """
    verts = (shell.get("floor") or {}).get("verts") or []
    faces = (shell.get("floor") or {}).get("faces") or []
    if not verts or not faces:
        return ""
    parts: list[str] = []
    for face in faces[:FLOOR_FACE_CAP]:
        try:
            corners = [view.pt(verts[i][0], verts[i][1]) for i in face]
        except IndexError:
            continue
        if len(corners) < 3:
            continue
        area = sum(corners[i][0] * corners[(i + 1) % len(corners)][1]
                   - corners[(i + 1) % len(corners)][0] * corners[i][1]
                   for i in range(len(corners)))
        if area < 0:
            corners.reverse()
        head = f"M{_n(corners[0][0])} {_n(corners[0][1])}"
        parts.append(head + "".join(f"L{_n(x)} {_n(y)}" for x, y in corners[1:]) + "Z")
    if not parts:
        return ""
    return f'<path class="floor" d="{"".join(parts)}"/>'


def _walls_svg(shell: dict[str, Any], view: _View) -> str:
    out: list[str] = []
    walls = shell.get("walls") or {}
    label = len(walls) <= 20      # past this the labels only overprint each other
    for wall_id, wall in walls.items():
        start, _along, normal, length = wall_frame(wall)
        x1, y1 = view.pt(start[0], start[1])
        x2, y2 = view.pt(wall["end"][0], wall["end"][1])
        # RoomPlan emits sub-centimetre stubs at corners; drawn solid they read as real walls.
        sliver = length < 0.35
        klass = "wall sliver" if sliver else "wall"
        out.append(f'<line class="{klass}" x1="{_n(x1)}" y1="{_n(y1)}" '
                   f'x2="{_n(x2)}" y2="{_n(y2)}"><title>{_e(wall_id)} · '
                   f'{length:.2f} m</title></line>')
        if label and not sliver:
            mid = (start + np.asarray(wall["end"], dtype=float)) / 2.0 + normal * 0.18
            mx, my = view.pt(mid[0], mid[1])
            out.append(f'<text class="wall-label" x="{_n(mx)}" y="{_n(my)}">'
                       f'{_e(wall_id.replace("Wall", "W"))}</text>')
    return "".join(out)


def _openings_svg(shell: dict[str, Any], view: _View) -> str:
    out: list[str] = []
    walls = shell.get("walls") or {}
    for opening_id, opening in (shell.get("openings") or {}).items():
        wall = walls.get(opening.get("wall") or "")
        if not wall:
            continue
        start, along, _normal, _length = wall_frame(wall)
        # `offset` is the distance to the opening's CENTRE. See shell.opening_span — reading it as
        # a leading edge slides every door half its own width along the wall.
        t0, t1 = opening_span(opening)
        a, b = start + along * t0, start + along * t1
        x1, y1 = view.pt(a[0], a[1])
        x2, y2 = view.pt(b[0], b[1])
        colour = category_color(opening.get("type", "opening"))
        out.append(f'<line class="opening" stroke="{colour}" x1="{_n(x1)}" y1="{_n(y1)}" '
                   f'x2="{_n(x2)}" y2="{_n(y2)}"><title>{_e(opening_id)} · '
                   f'{opening.get("type", "opening")}</title></line>')
    return "".join(out)


def _object_svg(object_id: str, obj: dict[str, Any], view: _View, *,
                edge: str | None, note: str = "") -> str:
    colour = category_color(obj.get("category", ""))
    points = view.poly(object_footprint(obj))
    stroke = edge or colour
    width = 2.4 if edge else 1.0
    centre_x, centre_y = view.pt(obj["center"][0], obj["center"][1])

    # The arrow is FACING (yaw - 90 deg), not `yaw` — `yaw` is the box's width axis and points
    # sideways across the object. Capped, or a 3.5 m counter throws an arrow across the room.
    angle = math.radians(facing_yaw(obj))
    reach = min(max(obj["size"][0], obj["size"][1]) * 0.55, 0.55) * view.scale
    tip_x = centre_x + math.cos(angle) * reach
    tip_y = centre_y - math.sin(angle) * reach       # canvas Y is flipped

    size = obj.get("size", [0, 0, 0])
    tip = (f'{object_id} · {obj.get("category", "?")}\n'
           f'{size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f} m\n'
           f'centre ({obj["center"][0]:.2f}, {obj["center"][1]:.2f}, {obj["center"][2]:.2f})\n'
           f'yaw {obj.get("yaw", 0.0):.1f} deg')
    if note:
        tip += f"\n{note}"
    return (f'<g class="obj" data-id="{_e(object_id)}"><title>{_e(tip)}</title>'
            f'<polygon points="{points}" fill="{colour}" fill-opacity="0.42" '
            f'stroke="{stroke}" stroke-width="{width}"/>'
            f'<line class="facing" x1="{_n(centre_x)}" y1="{_n(centre_y)}" '
            f'x2="{_n(tip_x)}" y2="{_n(tip_y)}"/>'
            f'<text class="obj-label" x="{_n(centre_x)}" y="{_n(centre_y)}">'
            f'{_e(object_id)}</text></g>')


def _panel(shell: dict[str, Any], view: _View, *, highlight: dict[str, str],
           notes: dict[str, list[str]], ghost: dict[str, Any] | None = None,
           arrows: list[tuple[str, dict, dict]] | None = None,
           gone: dict[str, dict] | None = None) -> str:
    """One floor plan. ``ghost`` underlays a second shell; ``arrows`` draw what moved."""
    parts = [f'<svg viewBox="0 0 {_n(view.width)} {_n(view.height)}" '
             f'width="{_n(view.width)}" height="{_n(view.height)}" '
             f'xmlns="http://www.w3.org/2000/svg" class="plan">',
             '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
             'markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
             '<path d="M0 0 L10 5 L0 10 z" fill="#1a202c"/></marker></defs>',
             _floor_path(shell, view), _walls_svg(shell, view), _openings_svg(shell, view)]

    if ghost:
        # A dropped object is drawn as dropped, not as a ghost — two outlines over the same box
        # would read as "it moved and also vanished".
        for object_id, obj in (ghost.get("objects") or {}).items():
            if object_id in (gone or {}):
                continue
            parts.append(f'<polygon class="ghost" points="{view.poly(object_footprint(obj))}"/>')

    for object_id, obj in (gone or {}).items():
        parts.append(f'<polygon class="dropped" points="{view.poly(object_footprint(obj))}">'
                     f'<title>{_e(object_id)} — dropped as a duplicate detection</title></polygon>')

    for object_id, obj in (shell.get("objects") or {}).items():
        parts.append(_object_svg(object_id, obj, view, edge=highlight.get(object_id),
                                 note="; ".join(notes.get(object_id, []))))

    for object_id, was, now in (arrows or []):
        x1, y1 = view.pt(was["center"][0], was["center"][1])
        x2, y2 = view.pt(now["center"][0], now["center"][1])
        distance = math.dist(was["center"][:2], now["center"][:2])
        parts.append(f'<line class="move" x1="{_n(x1)}" y1="{_n(y1)}" x2="{_n(x2)}" '
                     f'y2="{_n(y2)}" marker-end="url(#arrow)"><title>{_e(object_id)} moved '
                     f'{distance * 100:.0f} cm</title></line>')

    parts.append("</svg>")
    return "".join(parts)


def _legend(shells: list[dict[str, Any]]) -> str:
    categories = sorted({obj.get("category", "") for shell in shells
                         for obj in (shell.get("objects") or {}).values()})
    swatches = "".join(
        f'<span class="swatch"><i style="background:{category_color(c)}"></i>{_e(c)}</span>'
        for c in categories if c)
    return (f'<div class="legend">{swatches}'
            f'<span class="swatch"><i class="ghost-key"></i>as scanned</span>'
            f'<span class="swatch"><i style="background:{BAD}"></i>violation</span>'
            f'<span class="swatch"><i class="dropped-key"></i>dropped</span></div>')


def _diff(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """What changed between the two shells, measured rather than taken on trust.

    The stage already has the summary ``adapter.apply_to_objects`` returned, but that describes the
    pkl write. This describes the picture, and deriving it here means the drawing can never claim a
    move the geometry does not show.
    """
    old = before.get("objects") or {}
    new = after.get("objects") or {}
    moved, resized, arrows = [], [], []
    for object_id, now in new.items():
        was = old.get(object_id)
        if not was:
            continue
        distance = math.dist(was["center"][:2], now["center"][:2])
        if distance > MOVED_TOL:
            moved.append((object_id, distance))
            arrows.append((object_id, was, now))
        if any(abs(a - b) > MOVED_TOL for a, b in zip(was["size"], now["size"])):
            resized.append((object_id, was["size"], now["size"]))
    gone = {object_id: obj for object_id, obj in old.items() if object_id not in new}
    return {"moved": moved, "resized": resized, "gone": gone, "arrows": arrows}


def _violation_rows(violations, cleared: set[str]) -> str:
    rows = []
    for violation in violations or []:
        state = "cleared" if str(violation) in cleared else "remaining"
        colour = FIXED if state == "cleared" else BAD
        rows.append(f'<tr><td class="mono">{_e(violation.object)}</td>'
                    f'<td class="mono">{_e(violation.kind)}</td>'
                    f'<td>{_e(violation.detail)}</td>'
                    f'<td style="color:{colour}">{state}</td></tr>')
    return "".join(rows)


def render(scan: str, before: dict[str, Any], after: dict[str, Any], *,
           violations_before=None, violations_after=None,
           actions: list[dict] | None = None, mode: str = "deterministic") -> str:
    """The whole report as one HTML string. Pure function — it reads nothing and writes nothing."""
    from .adjust import check

    if violations_before is None:
        violations_before = check(before)
    if violations_after is None:
        violations_after = check(after)
    errors_before = [v for v in violations_before if v.severity == "error"]
    errors_after = [v for v in violations_after if v.severity == "error"]
    notes_after = [v for v in violations_after if v.severity != "error"]

    view = _View([before, after])
    change = _diff(before, after)
    still = {str(v) for v in errors_after}

    left = _panel(before, view,
                  highlight={v.object: BAD for v in errors_before}
                  | {v.other: BAD for v in errors_before if v.other},
                  notes=_violation_index(errors_before))
    right = _panel(after, view,
                   highlight={v.object: BAD for v in errors_after}
                   | {v.other: BAD for v in errors_after if v.other},
                   notes=_violation_index(errors_after),
                   ghost=before, arrows=change["arrows"], gone=change["gone"])

    # When the pass changed nothing, a second identical plan is not a comparison — it is the same
    # picture twice, and printing it invites the reader to hunt for a difference that is not there.
    touched = bool(change["moved"] or change["resized"] or change["gone"])
    if touched:
        left_title = (f'as scanned <em>&mdash; {len(errors_before)} violations, outlined red</em>'
                      if errors_before else 'as scanned <em>&mdash; no violations</em>')
        panels = (f'<div class="panel"><h3>{left_title}</h3>{left}</div>'
                  f'<div class="panel"><h3>after the pass <em>&mdash; dashed = where the scan put '
                  f'it, arrow = the correction</em></h3>{right}</div>')
    else:
        note = ("nothing needed repair" if not errors_before
                else f"{len(errors_after)} violations the pass could not resolve")
        panels = (f'<div class="panel"><h3>as scanned, and unchanged by the pass '
                  f'<em>&mdash; {_e(note)}</em></h3>{left}</div>')

    moved_cm = ", ".join(f"{oid} {d * 100:.0f} cm" for oid, d in change["moved"]) or "—"
    resized_text = ", ".join(
        f"{oid} {a[0]:.2f}x{a[1]:.2f}x{a[2]:.2f} → {b[0]:.2f}x{b[1]:.2f}x{b[2]:.2f} m"
        for oid, a, b in change["resized"]) or "—"
    dropped_text = ", ".join(change["gone"]) or "—"

    action_rows = "".join(
        f'<tr><td class="mono">{_e(a.get("action", "?"))}</td>'
        f'<td>{_e(a.get("detail", ""))}</td></tr>' for a in (actions or [])
    ) or '<tr><td colspan="2" class="quiet">nothing to do — the scan was already sound</td></tr>'

    note_rows = "".join(
        f'<tr><td class="mono">{_e(v.object)}</td><td class="mono">{_e(v.kind)}</td>'
        f'<td>{_e(v.detail)}</td></tr>' for v in notes_after)

    verdict = ("clean" if not errors_after else f"{len(errors_after)} unresolved")
    verdict_colour = FIXED if not errors_after else BAD

    # Everything the template interpolates is computed here. An f-string cannot carry a
    # multi-line expression on every version this runs under, and a template with logic in it is
    # unreadable anyway.
    cleared = {str(v) for v in errors_before} - still
    violation_rows = (_violation_rows(errors_before, cleared)
                      or '<tr><td colspan="4" class="quiet">none — the scan was already '
                         'sound</td></tr>')
    note_rows = note_rows or '<tr><td colspan="3" class="quiet">none</td></tr>'
    n_objects = len(after.get("objects") or {})
    n_walls = len(after.get("walls") or {})
    legend = _legend([before, after])

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>layout · {_e(scan)}</title>
<style>
  :root {{ color-scheme: light; }}
  body {{ margin: 0; padding: 26px 30px 60px; background: #f7fafc; color: #1a202c;
         font: 13px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }}
  h1 {{ font-size: 19px; margin: 0 0 2px; font-weight: 650; }}
  h2 {{ font-size: 13px; margin: 30px 0 8px; font-weight: 650; letter-spacing: .02em;
        text-transform: uppercase; color: #4a5568; }}
  .sub {{ color: #718096; margin: 0 0 18px; }}
  .stats {{ display: flex; flex-wrap: wrap; gap: 26px; margin: 0 0 20px; padding: 12px 16px;
            background: #fff; border: 1px solid #e2e8f0; border-radius: 6px; }}
  .stat b {{ display: block; font-size: 19px; font-weight: 650; }}
  .stat span {{ color: #718096; font-size: 11px; text-transform: uppercase; letter-spacing: .04em; }}
  .panels {{ display: flex; flex-wrap: wrap; gap: 18px; }}
  .panel {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 12px 14px 8px; }}
  .panel h3 {{ font-size: 12.5px; margin: 0 0 8px; font-weight: 650; }}
  .panel h3 em {{ font-style: normal; color: #718096; font-weight: 400; }}
  svg.plan {{ display: block; max-width: 100%; height: auto; }}
  /* Fill only, no stroke: the plate is a triangle MESH, and stroking each triangle draws every
     internal edge as a starburst across the room that reads as geometry rather than tessellation.
     Unstroked, the triangles merge into the one shape that matters — where the floor actually is,
     which is what an `outside_room` violation is measured against. */
  .floor {{ fill: #e4ebf3; stroke: none; }}
  .wall {{ stroke: #2d3748; stroke-width: 4; stroke-linecap: butt; opacity: .9; }}
  .wall.sliver {{ stroke-width: 1.5; stroke-dasharray: 2 3; }}
  .opening {{ stroke-width: 6; stroke-linecap: butt; }}
  .wall-label {{ font-size: 8px; fill: #4a5568; text-anchor: middle; dominant-baseline: middle; }}
  /* A white halo under the text, so a label over a dark box or a neighbour's edge stays readable
     without moving it somewhere that no longer identifies the box it names. */
  .obj-label {{ font-size: 8px; fill: #1a202c; text-anchor: middle; dominant-baseline: middle;
                pointer-events: none; paint-order: stroke; stroke: #fff; stroke-width: 2.2px;
                stroke-linejoin: round; }}
  .facing {{ stroke: #1a202c; stroke-width: 1.2; opacity: .85; }}
  .ghost {{ fill: none; stroke: {GHOST}; stroke-width: 1.2; stroke-dasharray: 4 3; }}
  .dropped {{ fill: {BAD}; fill-opacity: .10; stroke: {BAD}; stroke-width: 1.4;
              stroke-dasharray: 3 3; }}
  .move {{ stroke: #1a202c; stroke-width: 1.8; }}
  .obj:hover polygon {{ fill-opacity: .75; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 14px 0 0; color: #4a5568;
             font-size: 11px; }}
  .swatch {{ display: inline-flex; align-items: center; gap: 5px; }}
  .swatch i {{ width: 11px; height: 11px; border-radius: 2px; display: inline-block; }}
  .ghost-key {{ border: 1px dashed {GHOST}; }}
  .dropped-key {{ border: 1px dashed {BAD}; background: rgba(229,62,62,.12); }}
  table {{ border-collapse: collapse; background: #fff; border: 1px solid #e2e8f0;
           border-radius: 6px; font-size: 12px; min-width: 520px; }}
  th, td {{ text-align: left; padding: 6px 12px; border-bottom: 1px solid #edf2f7; }}
  th {{ color: #718096; font-weight: 600; font-size: 11px; text-transform: uppercase; }}
  tr:last-child td {{ border-bottom: 0; }}
  .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  .quiet {{ color: #a0aec0; }}
  dl {{ display: grid; grid-template-columns: max-content 1fr; gap: 4px 18px; margin: 0;
        background: #fff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 12px 16px; }}
  dt {{ color: #718096; }}
  dd {{ margin: 0; }}
</style></head><body>
<h1>{_e(scan)} — layout</h1>
<p class="sub">scene_init step 1c · repaired on boxes, before crops and reconstruction · {_e(mode)}</p>

<div class="stats">
  <div class="stat"><b>{len(errors_before)} &rarr; {len(errors_after)}</b><span>violations</span></div>
  <div class="stat"><b style="color:{verdict_colour}">{_e(verdict)}</b><span>verdict</span></div>
  <div class="stat"><b>{len(change["moved"])}</b><span>moved</span></div>
  <div class="stat"><b>{len(change["resized"])}</b><span>resized</span></div>
  <div class="stat"><b>{len(change["gone"])}</b><span>dropped</span></div>
  <div class="stat"><b>{n_objects}</b><span>objects</span></div>
  <div class="stat"><b>{n_walls}</b><span>walls</span></div>
</div>

<div class="panels">{panels}</div>
{legend}

<h2>what changed</h2>
<dl>
  <dt>moved</dt><dd class="mono">{_e(moved_cm)}</dd>
  <dt>resized</dt><dd class="mono">{_e(resized_text)}</dd>
  <dt>dropped</dt><dd class="mono">{_e(dropped_text)}</dd>
</dl>

<h2>actions taken</h2>
<table><tr><th>action</th><th>detail</th></tr>{action_rows}</table>

<h2>violations</h2>
<table><tr><th>object</th><th>kind</th><th>detail</th><th>after</th></tr>
{violation_rows}</table>

<h2>noted, not defects</h2>
<table><tr><th>object</th><th>kind</th><th>detail</th></tr>
{note_rows}</table>
</body></html>
"""


def write(path: str | Path, scan: str, before: dict[str, Any], after: dict[str, Any],
          **kwargs) -> Path:
    """Render and save. Returns the path written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(scan, before, after, **kwargs), encoding="utf-8")
    return path
