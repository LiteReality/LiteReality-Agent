"""materials_pass.py — give the room's objects REAL PBR materials, after authoring.

Geometry is left alone. Small objects are allowed to stay simple boxes/cylinders — the point of this
pass is that a simple box with a real captured wood/fabric/metal PBR set reads far closer to the
photo than the same box with a flat `_solid_mat` colour.

The authoring pass reliably textures the SHELL (floor, ceiling, walls) because those are large and
obvious, but tends to leave furniture and fixtures on flat colours: in a typical authored room only
`floor_carpet` and `ceiling_tile` end up in materials/. This pass walks what is left, fetches a real
Poly Haven set per material (recoloured in LAB to the colour already chosen, so the room's palette is
preserved), and rewires those objects to use it.

Ground truth = each object's reference image + the head-on surface stitches, same as authoring.
Edits ONLY material wiring — never geometry, never object placement.

    python -m litereality_agent.pipeline.author.realism.materials \
        --room <room dir> --surface-ref <dir> --scan <scan dir> --refroot <object_init dir>
"""
from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import time
from pathlib import Path

from litereality_agent import telemetry  # noqa: E402
from litereality_agent.agent.author import (  # noqa: E402
    CAPABILITY_TOOLS,
    surfaces_for,
)

PROMPT = """\
Role: you are doing a MATERIALS pass on an already-authored Room.py. Geometry is FINISHED — do not
move, resize, add or delete a single object. You are only changing what things are MADE OF.

Simple geometry is FINE and is not your problem to fix. A box with a real captured oak PBR set reads
like a shelf; the same box with a flat brown colour reads like a toy. That difference is the whole
job.

## Your checklist — the room's fixture palette ({n_todo} of {n_keys} still flat)

Every fixture in the room draws its material from this one palette, so texturing a key textures every
fixture that uses it. WORK THROUGH ALL OF THEM. This is a checklist, not a budget.

{key_lines}

## Required outcome, per key

For EVERY unticked key you must do one of two things, and say which in your summary:

  (a) give it a real PBR set — the default, and what you should be doing in most cases; or
  (b) leave it flat, and justify it with a SPECIFIC physical reason from the reference image
      ("bare painted steel radiator, no visible texture at 1 m"). "Small", "not very visible" and
      "nobody would notice" are NOT acceptable reasons — the shelf clutter is exactly what makes a
      room read as real.

Books/box/bottle keys (book_*, box_*, glass_*) are NOT exempt: printed cloth or paper spines, kraft
corrugate, and green glass are all real materials with real PBR sets. Texture them.

## How

1. `mcp__cap__read_image` the reference / stitch to see what the thing actually is.
2. `mcp__cap__fetch_material` with a specific query ("oak veneer", "kraft cardboard", "book cloth",
   "green glass", "white painted steel"). Recolour to the rgb already in the palette so the room's
   colours do not shift. It caches and writes the recipe into textures.json.
3. Rewire that key in the `Fx_` factory, keeping the `_tex_mat(...) or _solid_mat(...)` fallback so a
   missing texture degrades to the current colour instead of breaking the build.
4. `mcp__cap__render` and look. A material that reads wrong is worse than flat — revert that one.

Other flat call sites, for context ({n_flat} total):
{flat_lines}

Reference images:
{object_lines}

Head-on surface stitches (ground truth for wall-mounted fixtures):
{stitch_lines}

Already fetched into this room — reuse before fetching anything new:
{have_materials}

Constraints: Room.py MUST stay valid Python and MUST still build (render reports build errors — fix
them). Keep real-world metre scale. Do not touch geometry, transforms, or the object inventory.

Finish with a table: key | textured with what | or left flat + the physical reason.
"""


def _fixture_keys(room: Path) -> tuple[str, int, int]:
    """Every key in the room's fixture palette, with its colour and whether it already has a PBR set.

    Fixtures don't each get their own `_solid_mat` call — they all funnel through one palette dict
    (`P = {"wood": ((r,g,b), rough, metal), ...}`) and one `Fx_{key}` factory. So the honest unit of
    work is the KEY, not the call site: one key can be every shelf board in the room.
    """
    src = room / "Room.py"
    if not src.is_file():
        return "  (no Room.py)", 0, 0
    text = src.read_text(encoding="utf-8")
    keys = re.findall(r'"([a-z_]+)":\s*\(\(([0-9.,\s]+)\)', text)
    # A key counts as textured if it reaches a _tex_mat call. Three shapes occur in practice:
    #   _tex_mat("Fx_wood", "shelf_beech", ...)                 explicit, one key
    #   if key == "box_kraft":      return self._tex_mat(...)   guarded, one key
    #   if key in ("metal_grey", "metal_dark"): _tex_mat(f"Fx_{key}", key, ...)   guarded, many
    # Matching only the first shape under-reports badly, so scan the guarded branches too.
    textured = set(re.findall(r'_tex_mat\(\s*"Fx_([a-z_]+)"', text))
    for names, body in re.findall(
            r'if\s+key\s*==\s*"([a-z_]+)"\s*:\s*\n((?:.*\n){0,3})', text):
        if "_tex_mat" in body:
            textured.add(names)
    for names, body in re.findall(
            r'if\s+key\s+in\s*\(([^)]*)\)\s*:\s*\n((?:.*\n){0,3})', text):
        if "_tex_mat" in body:
            textured.update(re.findall(r'"([a-z_]+)"', names))
    lines, todo = [], 0
    for k, rgb in keys:
        done = k in textured
        if not done:
            todo += 1
        vals = [f"{float(v):.2f}" for v in rgb.split(",") if v.strip()]
        lines.append(f"  [{'x' if done else ' '}] {k:15} rgb({', '.join(vals)})"
                     + ("  — already textured" if done else ""))
    return ("\n".join(lines) or "  (no fixture palette found)"), len(keys), todo


def _flat_sites(room: Path) -> tuple[str, int]:
    """Every `_solid_mat(...)` call in Room.py — the objects still on flat colour."""
    src = (room / "Room.py")
    if not src.is_file():
        return "  (no Room.py)", 0
    lines = []
    for i, ln in enumerate(src.read_text(encoding="utf-8").splitlines(), 1):
        m = re.search(r'_solid_mat\(\s*[fr]?["\']([^"\']+)', ln)
        if m:
            lines.append(f"  - line {i}: {m.group(1)}")
    return ("\n".join(lines[:60]) or "  (none — every material is already textured)"), len(lines)


def _have_materials(room: Path) -> str:
    d = room / "materials"
    if not d.is_dir():
        return "  (none yet)"
    names = sorted({p.stem.rsplit("_", 1)[0] for p in d.glob("*.jpg")})
    return "\n".join(f"  - {n}" for n in names) or "  (none yet)"


def _object_lines(refroot: Path, scan: str) -> str:
    refs = refroot / "object_refs" / scan
    if not refs.is_dir():
        return "  (no reference images found)"
    out = []
    for d in sorted(p for p in refs.iterdir() if p.is_dir()):
        img = next((d / n for n in ("reference_1024.png", "clean_obj_reference.png",
                                    "nano_banana_raw.png") if (d / n).is_file()), None)
        if img:
            out.append(f"  - {d.name}: {img.resolve()}")
    return "\n".join(out) or "  (no reference images found)"


async def run(room: Path, surface_ref: Path, scan: Path, refroot: Path, model: str, max_turns: int,
              n_target: int = 8,  # n_target kept for CLI compat; the checklist is now per-key
              provider: str | None = None):
    from litereality_agent.agent import providers

    stitch_lines = "\n".join(
        f"  - {s} (head-on): {(surface_ref / f'{s}_stitched.jpg').resolve()}"
        for s in surfaces_for(room) if (surface_ref / f"{s}_stitched.jpg").is_file()) or "  (no stitches)"
    flat_lines, n_flat = _flat_sites(room)
    key_lines, n_keys, n_todo = _fixture_keys(room)
    prompt = PROMPT.format(have_materials=_have_materials(room), flat_lines=flat_lines, n_flat=n_flat,
                           object_lines=_object_lines(refroot, scan.name), stitch_lines=stitch_lines,
                           key_lines=key_lines, n_keys=n_keys, n_todo=n_todo)

    from litereality_agent import REPO_ROOT as repo_root
    dirs = {str(repo_root), str(os.path.realpath(room.parents[2])), str(surface_ref), str(scan),
            str(os.path.realpath(surface_ref)), str(os.path.realpath(refroot))}
    harness = providers.resolve("materials", provider)
    spec = providers.SessionSpec(
        prompt=prompt,
        cwd=room,
        read_roots=tuple(Path(d) for d in dirs),
        capability_tools=CAPABILITY_TOOLS,
        model=model,
        max_turns=max_turns,
    )
    print(f"== materials pass ==\n  room={room}\n"
          f"  {providers.describe(harness, spec)}, max-turns={max_turns}\n", flush=True)
    try:
        telemetry.start(scan.name)
        telemetry.event("materials_pass", scan=scan.name, status="start", n_flat=n_flat)
    except Exception:  # noqa: BLE001 — telemetry must not fail an authoring pass
        pass

    from litereality_agent.agent import scratch
    from litereality_agent.agent.tool_narration import ToolNarrator
    scratch_at = scratch.bind(near=room)   # this pass's own run_NNN dir
    nar = ToolNarrator()
    from litereality_agent.agent.trace import AgentTrace
    tr = AgentTrace('materials', room=room, scan=scan.name)
    tr.start(model=model, room=str(room), scratch=str(scratch_at) if scratch_at else None)
    t0 = time.time()
    summary = ""
    async for msg in harness.run(spec):
        tr.raw(getattr(msg, "raw", None) if getattr(msg, "raw", None) is not None else msg)
        if isinstance(msg, providers.SessionResult):
            summary = (msg.result or "")[:2000]
        for blk in getattr(msg, "content", []) or []:
            kind = type(blk).__name__
            if kind == "ToolUseBlock":
                print(nar.use(blk), flush=True)
                tr.tool(getattr(blk, "name", "?"), getattr(blk, "input", {}) or {},
                        tool_id=getattr(blk, "id", "") or "")
            elif kind == "TextBlock":
                tr.think(getattr(blk, "text", ""))
            elif kind == "ToolResultBlock":
                tr.result(blk)
                name, _ = nar.result(blk)
                failed = nar.error_line(name, blk)
                if failed:
                    print(failed, flush=True)
    dt = time.time() - t0
    _, _, todo_after = _fixture_keys(room)
    print(f"== materials_pass done {dt:.0f}s  fixture keys: {n_todo} flat at start -> {todo_after} now")
    if summary:
        print(summary)
    try:
        telemetry.event("materials_pass", scan=scan.name, status="ok", seconds=round(dt, 1))
    except Exception:  # noqa: BLE001 — telemetry must not fail an authoring pass
        pass
    return 0


def main():
    from litereality_agent.pipeline.author import arguments as stage_args

    ap = argparse.ArgumentParser(description=__doc__)
    stage_args.add_scene_arg(ap)
    # All four come from the scene package when not given; passing one still wins over it.
    ap.add_argument("--room", default=None)
    ap.add_argument("--surface-ref", default=None)
    ap.add_argument("--scan", default=None)
    ap.add_argument("--refroot", default=None)
    ap.add_argument("--model", default=os.environ.get("MATERIALS_MODEL")
                    or os.environ.get("HARNESS_MODEL") or "claude-opus-5")
    ap.add_argument("--max-turns", type=int, default=int(os.environ.get("MATERIALS_TURNS", "80")))
    ap.add_argument("--targets", type=int, default=int(os.environ.get("MATERIALS_TARGETS", "8")),
                    help="how many of the most visible surfaces to aim for (default 8)")
    ap.add_argument("--provider", default=None, choices=["claude", "codex"],
                    help="agent harness (default: $LR_MATERIALS_PROVIDER, else $LR_AGENT_PROVIDER)")
    a = stage_args.bind(ap.parse_args(), need=("room", "surface_ref", "scan", "refroot"))
    return asyncio.run(run(Path(a.room), Path(a.surface_ref), Path(a.scan), Path(a.refroot),
                           a.model, a.max_turns, a.targets, provider=a.provider))


if __name__ == "__main__":
    sys.exit(main())
