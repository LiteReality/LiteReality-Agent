"""qc_pass.py — model-driven QC pass over an ALREADY-authored Room.py (carried out by the harness model).

Not a rebuild — a single self-paced session that reads the authored `Room.py` + `Room.md` and makes
the FEWEST edits that resolve a fixed QC checklist (openings actually open, glazing transparent,
windows/curtains articulate, no fixture over an opening, ceiling materialed, every object matches its
reference at a glance). Ground truth = the head-on surface stitches + each object's reference image.

Same capability tools as authoring (render / critic / select_views / fetch_material). Edits ONLY
Room.py — the reconstructed objects are final, so object.py is never touched; Room.py must stay
valid Python.

(The sibling `authoring/qc_room.py` is a separate DETERMINISTIC geometry linter — no model. This is
the model pass; run the linter first/after if you want the arithmetic checks too.)

    python -m litereality_agent.pipeline.author.realism.quality \
        --room <room dir> --surface-ref <dir> --scan <scan dir> --refroot <object_init dir> \
        --model claude-opus-5
"""
from __future__ import annotations

import argparse
import asyncio
import os
import time
from pathlib import Path

# Append QC events to the application timeline. The detailed agent transcript is recorded
# separately by AgentTrace below.
from litereality_agent import telemetry  # noqa: E402

# reuse the authoring capability server (fetch_material / render / critic / select_views), scene-bound
from litereality_agent.agent.author import (  # noqa: E402
    CAPABILITY_TOOLS,
    surfaces_for,
)

PROMPT = """\
Role: You are doing a QC pass on an already-authored Room.py, not a rebuild. Read Room.md + Room.py in
full, then make the fewest edits that resolve the issues below. Use the head-on surface stitches and
each object's reference image as ground truth. Edit ONLY Room.py — do NOT edit any object.py, the
reconstructed objects are final; place/scale via Room.py only, and it must stay valid Python.

GEOMETRY FIDELITY (evidence-based, from the stitches/photos):
 • You MAY and SHOULD correct OPENINGS — add one the photo clearly shows, and REMOVE one the photo does
   NOT show (a scan-hallucinated door/window, e.g. a window behind a whiteboard that isn't real), and
   fix a clearly-wrong offset/width/height/sill. This is required, not optional (see Openings #10).
 • Fixtures must not be OVER-SIMPLIFIED vs the photo — a fixture the stitch shows as a detailed 3D
   thing must not be left a single flat slab; if it is, give it the missing parts (per Per-item QC).
 • Nothing HALLUCINATED — delete geometry the photos don't support (see Wall fixtures #3).
 • Positions/scale must be pretty much correct against the reference.
Do NOT relocate the furniture object boxes (Table*/Chair*/reconstructed objects) or move a whole wall
plane; keep every correction conservative and metric.

The room is `Room.py` in your working directory (a builder + an embedded `SHELL`). Use the `render`
tool to CHECK your edits: render 'room' or a wall, read the returned PNG, verify against the stitch /
reference, then fix. A couple of render-checks on the things you changed is worth far more than none.

HEAD-ON SURFACE STITCHES (ground truth for walls / openings / fixtures):
{stitch_lines}

PER-OBJECT REFERENCE IMAGES (ground truth for each object — the REAL object, not a render):
{object_lines}

Raw oblique capture frames are in {scan} (optional cross-check).

============================ Openings & glass (correctness — do first) ============================
1. Every opening must actually be open. For a plain Opening (doorway, no leaf) the wall is a true cut
   hole with nothing filling it — no slab, no bright/opaque card blocking it. For Door/Window, the
   leaf/pane sits in the cut, not a solid box.
2. All glazing is transparent glass. Window panes, glazed door panels, glass cabinet fronts → a
   transparent glass material (Principled BSDF Transmission ~1.0, low roughness), lightly tinted at
   most. Frames, mullions, and rails stay opaque. No opaque "glass".
3. Windows should all be articulated. Each window opens (tilt or slide) with a correct hinge/axis via
   the same keyframed-leaf articulation the door/window assets use (a closed->open clip on the moving
   leaf). The transparent pane stays attached to the moving leaf; opened != closed.
4. Unify near-identical windows on the same wall (sometimes across rooms). If two+ windows on one wall
   or in the same room are the same size/type, give them ONE shared design (same geometry + material)
   instead of near-duplicates that differ only by reconstruction noise.
5. Curtains/blinds, if present, are articulated. Roller blinds roll up (prismatic); drapes slide along
   the rod (prismatic). Anchor above the window; when open they must NOT cover the frame/glass.
6. Hydraulic door closer: if the real door has one (arm at the top hinge), model it and attach it to
   the door leaf so it moves with the door.
7. Windows must MATCH THIS scene's real windows. Read the stitch + raw frames: the frame layout,
   proportions, mullion/sash pattern and glazing must match the ACTUAL windows in this room. Do NOT
   leave a generic or mismatched window design, and do NOT let two windows in the same room be
   completely different types when the photos show them identical — rebuild them to the real design.
8. Doors: for EACH door, look at the rendered frames that show it AND the real photo/stitch, then edit
   the door object (leaf, panels, vision/glass panel, rails, handle, hinge side) so it matches the real
   door. Don't leave a door that clearly differs from the photo.
9. Partition vs wall: if a SHELL "wall" is really just a PARTITION in the room (e.g. a glass partition
   or a low screen), do NOT also build a solid wall structure on it — remove the redundant wall
   geometry so only the partition remains. A wall and a partition must not be stacked on the same plane.
10. Per-wall opening STRUCTURE must match the photo. Read EACH wall's stitch on its own: the NUMBER of
   openings on that wall, their left-to-right order and spacing, each one's width/height, and each
   one's type (door vs window; fixed vs casement/sliding; single vs multi-pane) must match what THAT
   wall's photo shows. Walls differ from each other — never copy one wall's opening pattern onto
   another. Pay special attention to walls with MULTIPLE openings (a run of windows, or a door next to
   a window): get the count and the arrangement right. Add an opening the photo clearly shows but the
   scan missed, and remove one the photo does not have. A mismatched opening count/layout on a wall is
   a correctness bug, not a nuance.
   PRIOR: multiple windows on the SAME wall are usually IDENTICAL in structure (same frame, mullion/sash
   pattern, proportions). Default to giving them ONE shared design unless the photo clearly shows they
   differ — this both matches reality and looks clean, so use it to fix a wall whose windows drifted
   apart from reconstruction noise (see also Openings #4).

============================ Materials & lighting ============================
1. Do NOT add light sources that aren't physically in the room. Represent only real emitters.
2. The ceiling must be materialed (not bare/blank).
4. MIRRORS use a MIRROR material — highly reflective (metallic ~1.0, roughness ~0.0-0.05, near-white
   base), so it reads as a real reflective mirror. Never leave a mirror as flat grey paint or plain
   glass.
3. Give ANY geometry you ADD or REBUILD a realistic PBR material — never leave a new fixture on a flat
   solid colour. Window/door frames, mullions, rails, sills, sash surrounds and hydraulic door closers
   → painted or brushed metal (fetch_material e.g. "brushed aluminium" / "painted steel", or a
   procedural metal: metallic ~1.0, roughness ~0.25-0.4); blinds/curtains → woven fabric
   (fetch_material or procedural "fabric"); panes → the transmissive glass above. Match the real
   colour/finish from the stitch. IMPORTANT: only re-material the parts you actually add or touch —
   do NOT restyle existing good surfaces (that would undo the authoring).

============================ Wall fixtures ============================
1. No fixture over an opening. Sockets, switches, trunking, boards, radiators, shelves must not overlap
   a Door/Window (they'd break when the leaf swings/slides); put nothing on the windows. Only
   curtains/blinds/pelmets may sit above a window. Horizontal runs (trunking, skirting, dado) break at
   openings.
2. Notice/white/pin boards and signs must MATCH their reference — correct position, size, and that they
   actually exist where the photo shows one. Fix a board that is wrong (wrong place/size/shape) or
   remove one the photo doesn't have.
3. NO placeholder / filler / decorative objects the reference does not show. The authoring step
   sometimes GENERATES generic props — books, boxes, blocks, cups, bottles, plants, ornaments,
   "items" — and sits them on shelves, desks, cabinets, counters or the floor. These are invented,
   not real. DELETE every object on a shelf/surface that is not clearly visible in that surface's
   reference photo/stitch. When in doubt, leave the shelf/surface EMPTY. Model ONLY what the
   reference actually shows — an empty shelf must stay empty.

============================ Furniture placement (clash-free & inside the walls) ============================
1. Call `check_collisions` and resolve everything it reports. It runs TRUE-MESH by default (over the
   compiled Room.glb — so `compile` first, then call it), reporting per furniture piece any
   object_clash (meshes interpenetrating), wall_clash (a piece poking THROUGH a wall), outside_room,
   floating/sunk, or an articulated door/window fouled — opening_blocked (inside a closed leaf) or
   open_swing_blocked (blocking its open swing). A chair correctly tucked under a desk reads CLEAR
   (real triangles, not boxes). Each finding carries a world-space (Δx, Δy) MOVE/snap and an
   equivalent RESIZE; apply, recompile, and re-run until it reports 0.
2. For each reported clash, decide MOVE vs RESIZE by looking at the evidence — render the wall/room
   and read the stitch/photo. Default to MOVING: add the fix's (Δx, Δy) to that object's
   SHELL['objects'][id]['center'][0:2] to snap it flush against its wall / part it from its neighbour
   / bring it back inside the room. RESIZE (shrink 'size') only when the piece is genuinely drawn too
   big or too deep for what the photo shows. Keep every change small and metric.
3. Correct overlaps are already ignored (a chair tucked under a desk, a sink in its counter) — do not
   "fix" those. Re-run `check_collisions` after your edits until it reports 0 clashes.

============================ Per-item QC against reference (catch MAJOR errors only) ============================
1. Objects: for each procedural object, once placed in the room, compare with the REAL reference image
   (not the synthetic render). Fix only MAJOR differences — wrong type, a missing major part, clearly
   wrong proportions. Ignore minor nuance. ONE check per object; don't iterate endlessly.
2. Openings at scene level: these were generated before scene authoring, so review each door/window
   once in-context. If it's unreasonable in the room (wrong scale, floating, flipped, wrong hinge side
   or structure), correct it. ONE check per object; don't iterate endlessly.

============================ Done when ============================
All glass is transparent · all openings actually open · each wall's openings match its photo (count,
position, type) · no fixture overlaps an opening · windows/curtains articulate · ceiling is materialed ·
no invented/placeholder objects on shelves or surfaces · every object matches its reference at a glance ·
`check_collisions` reports 0 clashes (no furniture interpenetrating, buried in a wall, or outside the room).

IMPORTANT: You do NOT want to overwrite what you already have here — make MINIMAL, targeted edits that
preserve the existing authoring. When done, summarise per surface what you changed, then stop.
"""


def _object_lines(refroot: Path, scan: str) -> tuple[str, int]:
    """List each object's reference_1024.png (the real object) for the model to Read."""
    refs = refroot / "object_refs" / scan
    if not refs.is_dir():
        return "  (no per-object reference images found)", 0
    lines = []
    for d in sorted(p for p in refs.iterdir() if p.is_dir()):
        img = d / "reference_1024.png"
        if img.is_file():
            lines.append(f"  - {d.name}: {img.resolve()}")
    return ("\n".join(lines) or "  (no per-object reference images found)"), len(lines)


async def run(room: Path, surface_ref: Path, scan: Path, refroot: Path, model: str, max_turns: int,
              extra: str = "", provider: str | None = None):
    from litereality_agent.agent import providers

    surfaces = surfaces_for(room)
    stitches = [surface_ref / f"{s}_stitched.jpg" for s in surfaces]
    stitch_lines = "\n".join(f"  - {s} (head-on): {p.resolve()}"
                             for s, p in zip(surfaces, stitches) if p.is_file())
    object_lines, n_obj = _object_lines(refroot, scan.name)
    prompt = PROMPT.format(stitch_lines=stitch_lines, object_lines=object_lines, scan=scan)
    if extra.strip():
        # user-directed fixes OVERRIDE the "minimal edits / don't overwrite / verify-only" posture —
        # the user has personally reviewed this scene and these MUST be actioned, not re-justified.
        prompt = (
            "########################################################################\n"
            "# MANDATORY USER-DIRECTED FIXES — the user reviewed THIS scene in the viewer and\n"
            "# requires the changes below. ACT on every one — actually EDIT Room.py / object.py to\n"
            "# make them. Do NOT re-verify and conclude 'it already matches' or 'it's fine as is';\n"
            "# the user has decided these are wrong. These OVERRIDE the general 'minimal edits /\n"
            "# preserve existing / one check per object' guidance below. Render to confirm each fix.\n"
            "########################################################################\n"
            f"{extra.strip()}\n"
            "########################################################################\n\n"
            + prompt
        )

    from litereality_agent import REPO_ROOT as repo_root
    dirs = {str(repo_root), str(os.path.realpath(room.parents[2])), str(surface_ref), str(scan),
            str(os.path.realpath(surface_ref)), str(os.path.realpath(refroot))}
    harness = providers.resolve("quality", provider)
    spec = providers.SessionSpec(
        prompt=prompt,
        cwd=room,
        read_roots=tuple(Path(d) for d in dirs),
        capability_tools=CAPABILITY_TOOLS,
        model=model,
        max_turns=max_turns,
    )

    n_stitch = sum(1 for p in stitches if p.is_file())
    print(f"== QC pass ==\n  room={room}\n"
          f"  tools=Read,Edit,Write,Glob + {list(CAPABILITY_TOOLS)}\n"
          f"  stitches={n_stitch} · object refs={n_obj}\n"
          f"  {providers.describe(harness, spec)}\n", flush=True)
    try:
        telemetry.start(scan.name)
        telemetry.event(
            "qc_start",
            scan=scan.name,
            model=model,
            room=str(room),
            stitches=n_stitch,
            object_refs=n_obj,
        )
    except Exception:  # noqa: BLE001 — telemetry must not fail an authoring pass
        pass
    t0 = time.monotonic()
    result_text, cost = "", None
    from litereality_agent.agent import scratch
    from litereality_agent.agent.tool_narration import ToolNarrator, hint_for
    scratch_at = scratch.bind(near=room)   # this pass's own run_NNN dir
    nar = ToolNarrator()
    from litereality_agent.agent.trace import AgentTrace
    tr = AgentTrace("qc", room=room, scan=scan.name)
    tr.start(model=model, room=str(room),
             scratch=str(scratch_at) if scratch_at else None)
    async for m in harness.run(spec):
        tr.raw(getattr(m, "raw", None) if getattr(m, "raw", None) is not None else m)
        for block in getattr(m, "content", []) or []:
            b = type(block).__name__
            if b == "TextBlock" and getattr(block, "text", "").strip():
                print(f"  …{block.text.strip()[:150]}", flush=True)
                tr.think(block.text)
                try:
                    telemetry.event("qc_note", text=block.text.strip()[:400])
                except Exception:  # noqa: BLE001 — telemetry is best-effort
                    pass
            elif b == "ToolUseBlock":
                # Narrate at USE time — ToolResultBlock has no name/input to narrate from.
                print(nar.use(block), flush=True)
                name = (getattr(block, "name", "?") or "?").split("__")[-1]
                inp = getattr(block, "input", {}) or {}
                tr.tool(getattr(block, "name", "?"), inp, tool_id=getattr(block, "id", "") or "")
                try:
                    telemetry.event("qc_tool", i=nar.calls, tool=name, hint=hint_for(name, inp, 160))
                except Exception:  # noqa: BLE001 — telemetry is best-effort
                    pass
            elif b == "ToolResultBlock":
                tr.result(block)
                name, used = nar.result(block)
                for kept in scratch.rescue(used, getattr(block, "content", None)):
                    print(f"      ↳ kept {kept.name}", flush=True)
                failed = nar.error_line(name, block)
                if failed:
                    print(failed, flush=True)
        if isinstance(m, providers.SessionResult):
            result_text = m.result or ""
            cost = m.total_cost_usd
    dt = round(time.monotonic() - t0, 1)
    calls, counts = nar.calls, nar.counts
    try:
        telemetry.event(
            "qc_done",
            calls=calls,
            counts=counts,
            cost=cost,
            seconds=dt,
            summary=result_text[:1200],
        )
    except Exception:  # noqa: BLE001 — telemetry must not fail an authoring pass
        pass
    print(f"\n== QC done {int(dt // 60)}m{int(dt % 60):02d}s | calls={calls} {counts} | cost=${cost} ==\n",
          flush=True)
    print("SUMMARY:\n" + result_text[:3000], flush=True)


def main():
    from litereality_agent.pipeline.author import arguments as stage_args

    ap = argparse.ArgumentParser(description=__doc__)
    stage_args.add_scene_arg(ap)
    ap.add_argument("--room", default=None)
    ap.add_argument("--surface-ref", default=None)
    ap.add_argument("--scan", default=None)
    ap.add_argument("--refroot", default=None, help="object_init dir (holds object_refs/<scan>/)")
    ap.add_argument("--model", default=os.environ.get("HARNESS_MODEL", "claude-opus-5"))
    ap.add_argument("--max-turns", type=int, default=160)
    ap.add_argument("--extra", default="", help="mandatory user-directed fixes prepended to the prompt")
    ap.add_argument("--extra-file", default="", help="read the --extra text from a file")
    ap.add_argument("--provider", default=None, choices=["claude", "codex"],
                    help="agent harness (default: $LR_QUALITY_PROVIDER, else $LR_AGENT_PROVIDER)")
    a = stage_args.bind(ap.parse_args(), need=("room", "surface_ref", "scan", "refroot"))
    extra = a.extra
    if a.extra_file:
        extra = Path(a.extra_file).read_text()
    asyncio.run(run(Path(a.room), Path(a.surface_ref), Path(a.scan), Path(a.refroot),
                    a.model, a.max_turns, extra, provider=a.provider))


if __name__ == "__main__":
    main()
