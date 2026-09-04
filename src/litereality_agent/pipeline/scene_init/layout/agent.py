"""agent.py — the last resort: ask a model to look at the photograph.

Everything up to v9 is geometry, and geometry has an answer for a box in the wrong PLACE. It has
no answer for a box of the wrong SIZE, and that is what the residual failures turn out to be:
Kitchen's ``Oven_Storage_Stove0`` is a merged run recorded as 2.53 x 1.47 m, and no kitchen counter
is a metre and a half deep. It reads as half a metre inside the wall behind it because the box is
too deep, not because the unit is misplaced, so every translation the solver can make is the wrong
move — slide it out and it collides with the island, slide it along and it is still too deep.

The scan does contain the answer, in the reference crops :mod:`physical_engine.views` already
ranked and wrote out: a photograph of that counter with its box drawn on it. This module hands the
model that image, the geometry, and the violation, and asks what is actually wrong.

Nothing it proposes is trusted. Every edit is applied to a copy and kept ONLY if the violation
count falls, no wall anchor is broken, and the change stays inside stated bounds. A proposal that
fails any of those is dropped and recorded. That gate is the whole reason this is safe to include:
the model chooses what to try, the geometry decides what is true.
"""

from __future__ import annotations

import copy
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from .adjust import ANCHOR_TOL, check
from .graph import wall_distance
from .shell import wall_frame

__all__ = ["propose", "apply_proposals", "solve_v10"]

MAX_SCALE = 0.60          # a proposal may not change a dimension by more than this fraction
MAX_SHIFT = 0.60          # nor move a centre further than this, in metres
TIMEOUT = 420


def _anchors(shell: dict[str, Any]) -> dict[str, set[str]]:
    walls = shell.get("walls") or {}
    out: dict[str, set[str]] = {}
    for oid, obj in (shell.get("objects") or {}).items():
        held = {wid for wid, w in walls.items()
                if (m := wall_distance(obj, w)) is not None and abs(m[0]) <= ANCHOR_TOL}
        if held:
            out[oid] = held
    return out


def _context(shell: dict[str, Any], violation, batch_dir: Path) -> dict[str, Any]:
    objects = shell.get("objects") or {}
    obj = objects.get(violation.object)
    if obj is None:
        return {}
    walls = shell.get("walls") or {}
    near_walls = {}
    for wid, w in walls.items():
        measured = wall_distance(obj, w)
        # the wall named in the violation goes in whatever its distance: the model spotted this
        # itself, replying "Wall6 isn't even in the provided wall list" for a Wall6 collision
        if wid == violation.other or (measured is not None and abs(measured[0]) < 1.2):
            if measured is None:
                start, along, normal, length = wall_frame(w)
                near_walls[wid] = {"start": [round(v, 3) for v in w["start"]],
                                   "end": [round(v, 3) for v in w["end"]],
                                   "length": round(length, 3), "gap_to_object": None}
                continue
            start, along, normal, length = wall_frame(w)
            near_walls[wid] = {"start": [round(v, 3) for v in w["start"]],
                               "end": [round(v, 3) for v in w["end"]],
                               "length": round(length, 3),
                               "gap_to_object": round(measured[0], 3)}
    neighbours = {oid: {"category": o.get("category"), "size": [round(v, 3) for v in o["size"]],
                        "center": [round(v, 3) for v in o["center"]]}
                  for oid, o in objects.items()
                  if oid != violation.object
                  and math.dist(o["center"][:2], obj["center"][:2]) < 2.5}
    refs = sorted((batch_dir / "references" / violation.object).glob("rank*.jpg"))[:2]
    return {"object_id": violation.object,
            "object": {"category": obj.get("category"), "size": [round(v, 3) for v in obj["size"]],
                       "center": [round(v, 3) for v in obj["center"]], "yaw": obj.get("yaw")},
            "violation": {"kind": violation.kind, "detail": violation.detail,
                          "other": violation.other},
            "walls": near_walls, "neighbours": neighbours,
            "reference_images": [str(p) for p in refs]}


PROMPT = """You are correcting a 3-D room reconstructed from an iPhone RoomPlan scan.

The object below is back at ITS ORIGINAL SCANNED POSE. A deterministic solver did produce a legal
arrangement for it, but only by {suspicion} — which a person looking at the room would not accept.
Its attempt has been undone so you are judging the real measurement, not its guess.

{context}

The reference images listed above are photographs from the scan with this object's reconstructed
box drawn on them. READ THEM with the Read tool before answering — they are the evidence for
whether the box actually bounds the thing it claims to.

A solver has already tried translation and failed, so consider that the BOX may be wrong rather
than its position: RoomPlan merges adjacent units and routinely records a counter run deeper than
it is, which makes it read as buried in the wall behind it.

Think about what is actually wrong, the way a person would. Usually it is one of:
  * the box is recorded DEEPER than the thing it bounds, because RoomPlan merged it with the wall
    or the unit beside it  -> "resize"
  * it belongs flush in a corner, touching two walls, and is a few cm out -> "attach" with both
    wall ids; the geometry will trim and seat it for you
  * it belongs against one wall -> "attach" with that single wall id
  * the measurement is simply right and the violation is the wall's fault -> "none"

Reply with ONLY a JSON object, no prose and no code fence:
{{"action": "resize" | "translate" | "attach" | "none",
  "size":   [x, y, z],       // resize only; metres
  "center": [x, y, z],       // translate only; metres
  "walls":  ["Wall3", "Wall7"],  // attach only; one or two wall ids from the list above
  "confidence": 0.0-1.0,
  "why": "one sentence citing what the photograph shows"}}

Prefer "attach" over "translate" for anything that is furniture against a wall — it preserves the
alignment a translation destroys. Use "none" if the images do not justify a change. Do not change a
dimension by more than {scale:.0%} or move a centre more than {shift} m."""


def propose(shell: dict[str, Any], violation, batch_dir: Path,
            model: str = "sonnet", suspicion: str | None = None) -> dict[str, Any] | None:
    """Ask the model what is wrong with one object. Returns its raw proposal, unvalidated."""
    context = _context(shell, violation, batch_dir)
    if not context:
        return None
    prompt = PROMPT.format(context=json.dumps(context, indent=2), scale=MAX_SCALE,
                           shift=MAX_SHIFT, suspicion=suspicion or "moving it a long way")
    try:
        done = subprocess.run(
            ["claude", "-p", prompt, "--output-format", "text", "--model", model,
             "--allowed-tools", "Read"],
            capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return {"action": "none", "why": "agent timed out"}
    text = (done.stdout or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {"action": "none", "why": f"unparseable reply: {text[:120]}"}
    try:
        out = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {"action": "none", "why": "invalid json"}
    out["object_id"] = context["object_id"]
    out["wall"] = violation.other
    return out


def _reflush(shell: dict[str, Any], object_id: str, prefer: str | None = None) -> None:
    """After a resize, put the unit back against the wall it is installed on.

    Shrinking a box about its centre pulls BOTH faces in, so a counter whose depth was over-measured
    by half a metre comes out of the wall by only a quarter of it and stays in violation — which is
    how the first accepted-looking proposal was correctly rejected. A counter's front face is where
    the room says it is and its back face belongs on the wall, so the model is asked only for the
    size and the geometry decides the placement that follows from it.
    """
    from .repair import _flush_to, attachments as _attachments
    from .adjust import _floor_bounds, _floor_triangles

    obj = shell["objects"][object_id]
    walls = shell.get("walls") or {}
    triangles = _floor_triangles(shell)
    bounds = _floor_bounds(shell)
    centroid = (bounds[0] + bounds[1]) / 2.0 if bounds is not None else None
    import numpy as np
    found = _attachments(obj, walls, triangles, centroid, tol=0.75)
    # The wall named in the violation goes first. Flushing against whichever attachment happened to
    # be found first put this counter against the wall it was already fine with and left it a
    # quarter of a metre inside the one it was not.
    found.sort(key=lambda a: a[0] != prefer)
    for _wid, _heading, wall, _along, inward, _gap in found:
        start = wall_frame(wall)[0]
        centre = np.array(obj["center"][:2], float)
        perp = float(np.dot(centre - start, inward))
        want = math.copysign(_flush_to(obj, wall, inward), perp if perp else 1.0)
        moved = centre + inward * (want - perp)
        obj["center"][0], obj["center"][1] = round(float(moved[0]), 4), round(float(moved[1]), 4)


def _bounded(obj, proposal) -> bool:
    if proposal.get("action") == "attach":
        walls = proposal.get("walls")
        return isinstance(walls, list) and 1 <= len(walls) <= 2
    if proposal.get("action") == "resize":
        want = proposal.get("size")
        if not (isinstance(want, list) and len(want) == 3):
            return False
        return all(v > 0.05 and abs(v - o) <= MAX_SCALE * o
                   for v, o in zip(want, obj["size"]))
    if proposal.get("action") == "translate":
        want = proposal.get("center")
        if not (isinstance(want, list) and len(want) == 3):
            return False
        return math.dist(want[:2], obj["center"][:2]) <= MAX_SHIFT
    return False


def apply_proposals(shell: dict[str, Any], proposals: list[dict[str, Any]]) -> tuple[dict, list]:
    """Apply each proposal only if it is bounded, reduces violations, and breaks no anchor."""
    current = copy.deepcopy(shell)
    log = []
    for proposal in proposals:
        oid = proposal.get("object_id")
        obj = (current.get("objects") or {}).get(oid or "")
        if obj is None or proposal.get("action") not in ("resize", "translate", "attach"):
            log.append({**proposal, "accepted": False, "reason": "no actionable change"})
            continue
        if not _bounded(obj, proposal):
            log.append({**proposal, "accepted": False, "reason": "outside the stated bounds"})
            continue
        trial = copy.deepcopy(current)
        target = trial["objects"][oid]
        if proposal["action"] == "resize":
            target["size"] = [round(float(v), 4) for v in proposal["size"]]
            _reflush(trial, oid, proposal.get("wall"))
        elif proposal["action"] == "attach":
            # The model names the walls; the geometry does the fitting. Keeping the arithmetic on
            # this side is the whole point — the model is good at "that wardrobe belongs in the
            # corner" and has no business computing the centimetres.
            from .repair import _corner_fit, _snap_delta
            names = [w for w in proposal["walls"] if w in (trial.get("walls") or {})]
            if not names:
                log.append({**proposal, "accepted": False, "reason": "named no known wall"})
                continue
            if len(names) == 2:
                fitted = _corner_fit(target, trial["walls"][names[0]], trial["walls"][names[1]])
                if fitted:
                    target["size"] = fitted
            for wall_id in names:
                delta = _snap_delta(target, trial["walls"][wall_id])
                target["center"][0] = round(target["center"][0] + float(delta[0]), 4)
                target["center"][1] = round(target["center"][1] + float(delta[1]), 4)
        else:
            target["center"] = [round(float(v), 4) for v in proposal["center"]]
        before = len([v for v in check(current) if v.severity == "error"])
        after = len([v for v in check(trial) if v.severity == "error"])
        held_before, held_after = _anchors(current), _anchors(trial)
        # Anchors of OTHER objects. A unit that is legitimately smaller than it was recorded can no
        # longer reach both walls of the corner it was merged across, and that is the correction
        # working, not damage. What must not happen is an edit to one object quietly moving another
        # off its wall — so the gate is scoped to everything except the object being edited.
        broke = sum(len(v - held_after.get(k, set()))
                    for k, v in held_before.items() if k != oid)
        if after < before and broke == 0:
            current = trial
            log.append({**proposal, "accepted": True,
                        "reason": f"violations {before} -> {after}, no anchor broken"})
        else:
            log.append({**proposal, "accepted": False,
                        "reason": f"violations {before} -> {after}, {broke} anchors broken"})
    return current, log


def _cache_path(root: Path) -> Path:
    return root / "agent_proposals.json"


def solve_v10(shell: dict[str, Any], *, batch_dir: str | Path | None = None,
              model: str = "sonnet", refresh: bool = False, detail: bool = False):
    """v9, then one bounded agent pass over whatever geometry could not settle.

    Accepted proposals are CACHED beside the scene. A model asked the same question twice does not
    give the same answer, and an arena whose score moves between runs cannot tell a real
    improvement from a reroll — fallside-kitchen passed on one run of this and failed the next with
    no code change in between. Caching makes the solver reproducible, makes a rebuild free, and
    means the model is consulted once per finding rather than once per build. Delete the cache, or
    pass ``refresh``, to ask again.
    """
    from .repair import repair as _repair

    out = _repair(shell)[0]
    errors = [v for v in check(out) if v.severity == "error"]
    root = Path(batch_dir) if batch_dir else Path(
        (shell.get("meta") or {}).get("batch_dir") or ".")
    if not errors or not root.is_dir():
        return (out, []) if detail else out

    cache_file = _cache_path(root)
    cached: dict[str, Any] = {}
    if cache_file.is_file() and not refresh:
        try:
            cached = json.loads(cache_file.read_text())
        except json.JSONDecodeError:
            cached = {}

    proposals, keys = [], []
    for violation in errors:
        key = f"{violation.object}:{violation.kind}:{violation.other or ''}"
        if key in cached:
            proposals.append(cached[key])
            keys.append(key)
            continue
        proposal = propose(out, violation, root, model)
        if proposal:
            proposals.append(proposal)
            keys.append(key)

    fixed, log = apply_proposals(out, proposals)

    # ONLY accepted proposals are cached. Storing every reply froze the first thing the model
    # happened to say, so a scene the agent had repaired on one run stayed broken on every run
    # afterwards — the cache was preserving its worst answer as eagerly as its best. A rejected
    # proposal is simply not knowledge, and asking again next time costs one call.
    keep = {k: p for k, p, entry in zip(keys, proposals, log) if entry.get("accepted")}
    if keep != cached:
        cache_file.write_text(json.dumps(keep, indent=1), encoding="utf-8")
    return (fixed, log) if detail else fixed
