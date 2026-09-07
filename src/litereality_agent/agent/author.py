"""One-shot room authoring.

A single self-paced model pass reconstructs the room's shell materials and the wall/ceiling/floor
fixtures by editing `Room.py` in place — there is no staged survey/plan/fix orchestration. The
model is given generic Read/Edit/Write on the room directory plus a set of capability tools that
plain file-editing cannot replace: `fetch_material` (real Poly Haven PBR — diffuse + roughness +
normal, LAB-recoloured to the measured colour) and `render` / `critic` / `select_views` as an
optional self-check it may call to compare its render against the capture photos and correct
colour and placement.

    uv run python -m litereality_agent.pipeline.realism_authoring.author.entrypoint --scene <scene dir>

`--scene` is the scene package the seed stage wrote (the folder holding `scene.json`); it supplies the
room, the surface references and the capture. Omit it entirely when $LR_SCENE is set or the
current directory is inside a package. The explicit spelling still works and still wins:

    ... -m litereality_agent.pipeline.realism_authoring.author.entrypoint \
        --room <room dir> --surface-ref <dir> --scan <scan dir>
"""

from __future__ import annotations

import os
import time
from pathlib import Path


# Surfaces are DISCOVERED from the room, never assumed: a scan has as many walls as RoomPlan gave it
# (a one-bed flat came back with 30). A hardcoded Wall0..WallN silently drops every wall past the cap
# from the prompt AND from the stitch-coverage check below, so the run reports full coverage while
# most of the room was never referenced. `surface_ids` parses Room.py's SHELL and also excludes
# RoomPlan's sub-0.35m corner-artifact stub walls, whose stitches are useless smears.
def surfaces_for(room: Path) -> list[str]:
    from litereality_agent.room_ops.surfaces import surface_ids

    return surface_ids(room / "Room.py")


# capability tools exposed to the authoring model: real materials + optional self-check
CAPABILITY_TOOLS = ("fetch_material", "render", "critic", "select_views", "grid", "check_collisions")

# Shared by both profiles. Measured on two runs of the same scene, the pass spent its whole budget
# exploring and made every edit at the very end: the first reached its first edit 16.5 min in (69%
# of the session, 85 of 91 tool calls already spent), the second was still on Read/Bash/
# fetch_material after 83 calls with no edit at all. Both called `render` at most once, before any
# edit existed — so "author, render, look, correct" never actually ran a cycle. Nothing in the old
# wording forbade that, and a self-paced model under a step budget will always bank the cheap calls
# first. Hence a stated cadence rather than encouragement, and an explicit first target: the floor
# and walls fill most of every frame, so they are what makes progress visible at all.
RHYTHM = """\
HOW TO WORK — SHIP CHANGES AS YOU GO, NOT IN ONE BATCH AT THE END.
Someone may be watching this room rebuild live, and an unedited `Room.py` shows them nothing. A run
that is cut short keeps only what you have already saved.
• EDIT `Room.py` AT LEAST ONCE EVERY 20-30 TOOL CALLS. If you are nearing 30 calls since your last
  edit, stop investigating: make the best change you can justify from what you already know, save
  it, and keep investigating afterwards. An imperfect saved change beats a perfect unsaved one.
• BIGGEST VISIBLE WIN FIRST — the FLOOR, then the WALLS. Their material fills most of every frame,
  so the floor PBR set and the wall paints change the room more than any fixture can. Work in this
  order, editing at each step: floor -> walls -> ceiling -> fixtures.
• Read for your NEXT edit, not for the whole room. `Room.md` and the `SHELL` dict up front, then ONE
  surface's head-on stitch at a time — measure that surface, edit it, save, move to the next.
• Every edit must leave `Room.py` compiling. Many small complete changes, never one large one.
"""

PROMPT = """\
You are reconstructing a REAL room as an editable Python program, self-paced (no fixed steps).

{rhythm}

The room is `Room.py` in your working directory: a builder + a `SHELL` dict (walls, openings,
object boxes). FIRST Read `Room.md` and `Room.py` IN FULL to learn the helper API (how shell
materials are assigned, how geometry is added). Then edit `Room.py` IN PLACE so the rebuilt room
matches the real room in the images.

Two jobs, IN THIS ORDER — do (1) fully first, then (2):
1) SHELL STRUCTURE + MATERIALS — the FOUNDATION. Get the room itself right and render-check the bare
   room BEFORE you add any fixture, so fixtures land on a room that already reads correctly.
   • STRUCTURE FIRST: check the room's SHAPE against the photos and fix clear scan errors in the
     `SHELL` dict — a missing or spurious wall, a wall with wrong endpoints/length, a door/window at
     the wrong offset/width/height/sill (or one the scan missed or hallucinated), a wrong floor/ceiling
     height. Edit the numbers in `SHELL` (see Room.md "Editing the scene"). Be CONSERVATIVE and
     evidence-based: correct obvious errors against the photos, keep it metric, and do NOT redesign a
     room that is already right. Do NOT move/resize the furniture object boxes
     (Table*/Chair*/reconstructed objects) — structure means walls / openings / floor+ceiling height only.
     Keep this pass SHORT — correct the obvious errors and move on to materials; do not audit every
     number before your first edit.
   • THEN MATERIALS, in the order FLOOR -> WALLS -> CEILING, covering every surface
     ({surface_list}): base colour (hue+lightness), finish, pattern. Do the FLOOR first — it is the
     single largest visible surface. Save after each surface rather than batching them all into one
     edit. READ PAST anything mounted on a wall — material is the exposed wall. Don't skip the
     CEILING. Don't touch fixtures until the structure is corrected and every surface has a material.
2) WALL FIXTURES the scan missed but the photos show (sockets, switches, trunking, boards, signs,
   radiators, shelves, skirting, ceiling vents, rugs) — simple procedural geometry flush to the
   wall, anchored to the SHELL's opening offsets / wall lengths. Never over a Door/Window opening.
   Only start this once (1) is done.

GROUP EVERY FIXTURE AS ONE UNIT. A fixture is usually several boxes (a whiteboard = frame bars +
face + tray; a radiator = panel + fins + valves). Those parts MUST be bundled into a single named,
hide-as-a-unit group — never left as a loose pile of boxes. Use the helper
`group_fixture(name, category, parts)` (defined near the collection helpers in `Room.py`; if your
`Room.py` doesn't have it, add it once: an empty named `name` with `room_id`/`category` props, each
part parented to it and moved into a collection named `name`). Collect the objects your box helpers
return into a list and pass them, e.g.
    parts = [_wall_box(...), _wall_box(...), ...]   # all boxes of THIS one fixture
    group_fixture("Whiteboard0", "whiteboard", parts)
Name each fixture like the furniture handles — `Whiteboard0`, `Radiator0`, `Shelf0`, `Socket0`,
`Trunking0`, `Vent0` (increment per instance). The whole fixture then selects/hides/toggles as one
object in the viewer, exactly like `Table0`/`Chair0`. This grouping is REQUIRED, not optional.

The DECISIVE references are the HEAD-ON STITCHES (each shows one surface square-on). Read every one:
{stitch_lines}
Raw oblique frames are in {scan} (optional context).

You have TOOLS beyond editing — use them:
- `fetch_material(query, name, color_hex, pattern_strength)` — for any PATTERNED surface (carpet,
  tile, wood, brick, strong plaster) pull a REAL Poly Haven PBR set (diffuse+roughness+normal),
  LAB-recoloured to your measured colour. It saves into the room's `materials/` + `textures.json`
  and returns a wiring snippet — wire it into `Room.py` (real UV scale in metres). Prefer this over
  a flat colour whenever the photo shows a pattern; keep flat colour+roughness only for PLAIN paint.
  For the categories Poly Haven covers poorly — CARPET / RUG / FABRIC / PLASTER and patterned
  TILE / BRICK / WOOD floors — you can instead use the code-native procedural materials:
  `from litereality_agent.room_ops.procedural_materials import make` then `make("carpet"|"fabric"|"plaster"|
  "tile"|"brick"|"wood_planks", color=(r,g,b))` returns a Blender material to assign directly in
  `Room.py` (parametric, tileable, no download). Use whichever matches the photo better.
- `render(target)` — render `Room.py` for 'room' or a wall, paired with the real photo; it returns
  PNG path(s). READ the returned PNG with your own eyes to check colour/placement against the photo,
  then fix `Room.py`. Use this to CATCH COLOUR DRIFT especially — author, render, look, correct.
- `critic(images, goal)` — a strict pass/score verdict if you want a second opinion.
- `select_views(target)` — best frames (render auto-picks if you omit frames).
Render-check the surfaces you are least sure about, especially for COLOUR DRIFT — but never let a
render replace an edit: it is your SAVED `Room.py` that the room is rebuilt from, so save first and
check afterwards.

Constraints:
- Edit ONLY `Room.py`; it MUST stay valid Python that compiles. You MAY correct SHELL STRUCTURE
  (walls, openings, floor/ceiling height) when the scan is clearly wrong vs the photos — conservatively
  and metric, per job (1). But do NOT move/resize the furniture object boxes (Table*/Chair*/reconstructed
  objects). Assign everything in CODE (materials via the fetched recipe so a fresh rebuild reproduces them).

When done, summarise per surface: the material (flat vs fetched PBR + which asset) and the fixtures.
"""

# "go harder" profile — same tools, but fixtures MUST be detailed multi-part geometry, render-verified.
DETAIL_PROMPT = """\
You are reconstructing a REAL room as an editable Python program, self-paced. The priority THIS run
is FIXTURE DETAIL: every mounted object must read as a real 3D thing, never a flat slab.

{rhythm}

The room is `Room.py` in your working directory: a builder + a `SHELL` dict (walls, openings, object
boxes). FIRST Read `Room.md` and `Room.py` IN FULL to learn the helper API (how materials are
assigned, how geometry/boxes are added). Then edit `Room.py` IN PLACE to match the real room.

Two jobs, IN THIS ORDER — finish (1) FIRST, then spend the bulk of the run on (2):
1) SHELL STRUCTURE + MATERIALS — the foundation. Do this FIRST and render-check the bare room BEFORE
   any fixture, so fixtures land on a room that already reads correctly.
   • STRUCTURE: check the room's SHAPE against the photos and fix clear scan errors in the `SHELL`
     dict — a missing/spurious wall, wrong wall endpoints/length, a door/window at the wrong
     offset/width/height/sill (or one the scan missed or hallucinated), a wrong floor/ceiling height.
     Edit the numbers in `SHELL` (see Room.md "Editing the scene"). Be conservative, evidence-based and
     metric — correct obvious errors, don't redesign a room that's already right, and do NOT move/resize
     the furniture object boxes (Table*/Chair*/reconstructed objects).
   • MATERIALS for every surface ({surface_list}): match colour/finish/pattern. Read
     PAST anything mounted on a wall. Use `fetch_material` for genuinely patterned surfaces
     (carpet/tile/wood/brick); flat colour+roughness for plain paint. Keep it efficient (don't
     over-polish) but do NOT skip the CEILING. Don't start fixtures until structure is fixed and every
     surface has a material.
2) WALL FIXTURES the photos show — modelled as DETAILED MULTI-PART geometry. This is where most of
   your effort goes, but only AFTER (1) is done.

## THE RULE: no fixture is a single flat box. Build each from parts so its SILHOUETTE matches the stitch.
Model every non-trivial fixture as a small assembly (write a Python loop where it repeats):
- **shelving / bookshelf** → vertical uprights/standards + individual brackets per shelf + each board
  as its own slab + a few representative ITEMS sitting on the boards (books/boxes as small blocks).
- **radiator** → back panel + a LOOP of vertical fins (10–20 thin slats) + top grille + end caps +
  the two pipe valves at the bottom.
- **whiteboard / notice board** → frame border (4 thin bars) + recessed inset face (its own material)
  + a pen tray / bottom lip; cork boards get a visibly thicker frame.
- **AC / split unit** → main body box + a sloped/louvred front vent (a loop of thin louvre slats) +
  the pipe/conduit run leaving it.
- **cable trunking / conduit** → a channel base + a proud lid (two stacked thin boxes), not one strip.
- **sockets / switches** → a faceplate plate + the raised outlet/rocker on it (2 parts min), not a dot.
- **skirting / dado** → a board with a slightly proud top lip if the photo shows a moulding profile.
Match each part's real proportions off the stitch; give parts distinct materials (metal vs plastic vs
board). 10 lines of a `for`-loop beats one box.

## MANDATORY grouping: every fixture's parts bundle into ONE named, hide-as-a-unit group.
The whole point of multi-part fixtures is defeated if the parts are a loose pile you must hide one
box at a time. So EACH fixture (all its bars/fins/slats/boards/items) MUST be wrapped into a single
named group via the helper `group_fixture(name, category, parts)` (defined near the collection
helpers in `Room.py`; if it's not already there, add it once — an empty named `name` with
`room_id`/`category` props, each part parented to it and moved into a collection named `name`):
    parts = []
    parts.append(_wall_box(...))          # back panel
    for i in range(n_fins):               # the fin loop
        parts.append(_wall_box(...))
    parts += [_wall_box(...), _wall_box(...)]   # valves
    group_fixture("Radiator0", "radiator", parts)
Every box helper returns its object — collect them and pass the list. Name each instance like the
furniture handles: `Whiteboard0`, `Radiator0`, `Shelf0`, `Socket0`, `Trunking0`, `AC0`, `Vent0`.
The grouped fixture then selects, HIDES and toggles as ONE object in the viewer / outliner / glTF,
exactly like `Table0`/`Chair0`. This is REQUIRED for every fixture — a whiteboard's frame+face+tray
hide together, a radiator's fins hide together. Do NOT leave fixture parts ungrouped in the flat
`Fixtures` bucket.

## MANDATORY render-verify per wall (this is required, not optional)
After you add the fixtures on a wall, you MUST `render(target='Wall<N>')`, then READ the returned PNG
with your own eyes and check EACH fixture's SILHOUETTE against the real photo/stitch: does the
radiator show fins? the shelf show brackets + boards + items? the whiteboard show a frame + tray?
If any fixture reads as a flat rectangle/slab, ADD the missing parts and re-render that wall. Do NOT
mark a wall done until its fixtures read as 3D assemblies, not cut-outs. Anchor everything to the
SHELL's opening offsets / wall lengths; never place a fixture over a Door/Window opening; horizontal
runs (trunking, skirting) BREAK at openings.

The DECISIVE references are the HEAD-ON STITCHES (each surface square-on) — read every one:
{stitch_lines}
Raw oblique frames are in {scan}. Beware: a stitch can be mirrored/warped — cross-check fixture
positions against a raw frame before committing.

Tools: `fetch_material`, `render` (returns PNG paths — Read them), `critic(images, goal)`,
`select_views`. Use `render` liberally this run — it is how you confirm the silhouettes.

Constraints: edit ONLY `Room.py`; it MUST compile. You MAY correct SHELL STRUCTURE (walls, openings,
floor/ceiling height) when the scan is clearly wrong vs the photos — conservatively and metric, per
job (1) — but do NOT move/resize the furniture object boxes (Table*/Chair*/reconstructed objects).
Everything in CODE so a fresh rebuild reproduces it.

When done, summarise per wall: each fixture, its GROUP NAME (e.g. `Radiator0`), the PARTS you built
it from, and note which walls you render-verified.
"""


# "simulation" profile — the room as a place, not a shell: real light, real clutter, and every
# object stating what holds it up. The base profile stops at materials and wall fixtures, which is
# what makes a render read as a MODEL of a room rather than a room: the surfaces are right and the
# room is empty. A desk with nothing on it is the single strongest cue that a scene is synthetic,
# and a scene whose objects do not say what they rest on cannot be simulated no matter how good it
# looks. Both are asked for here, explicitly, because a self-paced model under a budget will always
# spend it on the surfaces it was told about first.
SIMULATION_PROMPT = """\
You are rebuilding a REAL room as an editable Python program, self-paced (no fixed steps), and the
bar THIS run is a room a person would believe they had walked into — and that a physics engine
could pick up and simulate without being told anything else.

{rhythm}

The room is `Room.py` in your working directory: a builder + a `SHELL` dict (walls, openings,
object boxes). FIRST Read `Room.md` and `Room.py` IN FULL to learn the helper API. Then edit
`Room.py` IN PLACE. LOOK AT THE PHOTOGRAPHS PROPERLY — read the head-on stitches AND a good spread
of the raw frames, and keep going back to them. Everything below is an observation to be made from
those images, not a thing to be invented: author what THIS room contains, not what a room like it
usually contains.

FOUR JOBS, IN THIS ORDER. Do not start one before the previous is done.

1) SHELL STRUCTURE + MATERIALS — the foundation.
   • Fix clear scan errors in `SHELL` first: a missing or spurious wall, wrong endpoints, a
     door/window at the wrong offset/width/sill, a wrong floor/ceiling height. Conservative,
     evidence-based, metric. Do NOT move or resize the furniture object boxes.
   • Then every surface ({surface_list}) in the order FLOOR -> WALLS -> CEILING: base colour,
     finish, pattern. Save after each. Use `fetch_material` or the procedural materials for
     anything patterned; flat colour only for plain paint.

2) LIGHT. A room lit by a default lamp reads as a render of a room no matter how good the
   materials are, and every material judgement you make afterwards is made under the wrong light.
   • Put the REAL luminaires in, where the photos show them: ceiling panels, downlights, pendants,
     desk and floor lamps, under-cabinet strips. Geometry AND an actual Blender light — the fitting
     you can see, and the light it casts.
   • Match what the photographs show: colour temperature (warm domestic ~2700-3000K, office
     fluorescent/LED ~4000-5000K, daylight through a window ~6500K), rough intensity, and the
     direction the shadows in the photo tell you. If a window is the dominant source, light it as a
     window — the frames will show you whether it is blown out or soft.
   • Emissive material on the visible fitting so it reads as ON in a render if the photo shows it on.
   • `render` after the lighting pass and LOOK at it against the photo. Light is the one thing
     where your first guess is usually a stop or two out, and it is cheap to correct.

3) FIXTURES the scan missed but the photos show — sockets, switches, trunking, boards, signs,
   radiators, shelves, skirting, ceiling vents, rugs, blinds/curtains, door furniture. Simple
   procedural geometry flush to the wall, anchored to the SHELL's opening offsets and wall lengths.
   Never over a Door/Window opening.

4) THE SMALL OBJECTS. This is the job that decides whether the room is believable, and it is the
   one you must NOT run out of budget before reaching — start it while you still have room to work.
   An empty desk is the loudest possible statement that a scene is synthetic. Real rooms are full
   of small, specific, slightly untidy things, and the photographs are full of them: mugs, glasses,
   bottles, papers, folders, notebooks, pens and pen pots, books, laptops, monitors, keyboards,
   mice, cables and chargers, phones, boxes, bins, bags, plants and pots, cushions, throws, remote
   controls, tissue boxes, jars, bowls, fruit, cleaning bottles, chopping boards, kettles, toasters.
   • Work SURFACE BY SURFACE. For each placeable surface in the room (the desks, tables, counters,
     shelves, cabinet tops) find it in the photos and author WHAT IS ACTUALLY ON IT, in roughly the
     positions the photos show.
   • Build them as small multi-part procedural geometry — a mug is a cylinder plus a handle, a
     laptop is a base plus a screen at an angle, a plant is a pot plus foliage. They are small in
     frame, so simple honest shapes at the right SIZE, COLOUR and POSITION beat detailed shapes at
     the wrong ones. Metric: a mug is ~8 cm across and ~10 cm tall, an A4 sheet is 210x297 mm, a
     laptop is ~32 cm wide.
   • Do not tidy the room. Things sit at angles, papers overlap, cables trail, a chair is pushed
     out. Alignment to the axes is the giveaway of a generated scene — rotate them.
   • Do not invent a prop you cannot point at in a photograph. A believable room is a SPECIFIC
     room; generic clutter is just a different kind of wrong.

SIMULATION READINESS — REQUIRED FOR EVERY OBJECT YOU ADD, NOT AN EXTRA.
The room has to be something a physics engine can accept, and that needs two things from you:
• GROUP each object as ONE unit with `group_fixture(name, category, parts, rests_on=..., attached_to=...)`
  (near the collection helpers in `Room.py`). Collect the parts of ONE object into a list and pass
  them. Name them like the furniture handles — `Mug0`, `Laptop0`, `Plant0`, `Radiator0`, `Lamp0`
  (increment per instance). A loose pile of boxes is not an object.
• STATE WHAT HOLDS IT UP. `rests_on="Table0"` for anything standing on a surface, `rests_on="Floor0"`
  for anything on the floor, `attached_to="Wall3"` (or `"Ceiling0"`) for anything fixed to the
  structure. Every single added object gets one or the other — no exceptions, no guesses left
  implicit. It is written into the object and comes back out in `room_layout.json`, which is what a
  simulator reads.
• AND MAKE IT TRUE GEOMETRICALLY. An object that says it rests on `Table0` must actually have its
  underside ON that table's top surface — not 2 cm above it, not sunk 1 cm into it. Read the
  table's top from the SHELL / the object's own bbox and place the prop from that number rather
  than by eye. A prop that floats is a prop that will fall on the first simulated frame, and a prop
  that intersects is one the engine will fire across the room.
• Nothing may interpenetrate anything else. Use `check_collisions` to verify, and fix what it
  reports.

The DECISIVE references are the HEAD-ON STITCHES (each shows one surface square-on). Read every one:
{stitch_lines}
The raw frames in {scan} are where the SMALL OBJECTS are legible — the stitches flatten them. Read a
good spread of frames before job 4, and again while you work through it.

TOOLS — use them, they are not decoration:
- `fetch_material(query, name, color_hex, pattern_strength)` — real Poly Haven PBR (diffuse +
  roughness + normal) LAB-recoloured to your measured colour, saved into `materials/` +
  `textures.json` with a wiring snippet. Prefer it over flat colour for any patterned surface. The
  code-native alternative for carpet/fabric/plaster/tile/brick/wood:
  `from litereality_agent.room_ops.procedural_materials import make`.
- `render(target)` — render `Room.py` for 'room' or a wall, paired with the real photo. READ the
  returned PNG with your own eyes and correct what is wrong. Render after LIGHT, and again after
  the props are in.
- `critic(images, goal)` — a strict second opinion when you want one.
- `select_views(target)` — best frames for a target.
- `check_collisions()` — REQUIRED once the props are in. Nothing may interpenetrate.

Constraints:
- Edit ONLY `Room.py`; it MUST stay valid Python that compiles. You MAY correct SHELL structure
  (walls, openings, floor/ceiling height) conservatively. Do NOT move or resize the furniture
  object boxes (Table*/Chair*/reconstructed objects) — those are measured and already corrected.
- Everything in CODE, so a fresh rebuild reproduces it exactly.

Finish with a summary: the material per surface, the lights (type, colour temperature, where), the
fixtures, and a list of every small object with what it rests on or is attached to.
"""


PROFILES = {"base": PROMPT, "detail": DETAIL_PROMPT, "simulation": SIMULATION_PROMPT}


def room_compiles(room: Path) -> str:
    """"" if `Room.py` is valid Python, else the error. A syntax check, not a build.

    Deliberately cheap — `compile()`, no Blender. It runs after every edit, and the failure it
    guards against is syntactic: an `Edit` applied to a file the model was mid-thought about.
    """
    src = room / "Room.py"
    try:
        compile(src.read_text(encoding="utf-8"), str(src), "exec")
    except SyntaxError as exc:
        return f"line {exc.lineno}: {exc.msg}"
    except OSError as exc:
        return str(exc)
    return ""


def checkpoint(room: Path, where: Path) -> bool:
    """Save `Room.py` as last-known-good if it compiles. Returns True when saved.

    A long session that dies mid-edit used to leave whatever the last write happened to be —
    possibly unparseable, which fails every downstream stage. Keeping the newest COMPILING
    version means a break costs one edit, not the run.
    """
    if room_compiles(room):
        return False
    try:
        where.parent.mkdir(parents=True, exist_ok=True)
        where.write_bytes((room / "Room.py").read_bytes())
        return True
    except OSError:
        return False


# The in-process capability server and the step-budget hook are Claude Code machinery, so they
# live with that harness now (`agent/providers/claude.py`). Re-exported because this module has
# been their import site since before there was a second harness.
from litereality_agent.agent.providers.claude import build_capability_server  # noqa: E402,F401


async def run(room: Path, surface_ref: Path, scan: Path, model: str, max_turns: int, profile: str = "base",
              step_budget: int = 100, step_reserve: int = 15, provider: str | None = None):
    from litereality_agent.agent import providers

    surfaces = surfaces_for(room)
    stitches = [surface_ref / f"{s}_stitched.jpg" for s in surfaces]
    stitch_lines = "\n".join(f"  - {s} (head-on): {p}" for s, p in zip(surfaces, stitches) if p.is_file())
    prompt = PROFILES.get(profile, PROMPT).format(stitch_lines=stitch_lines, scan=scan,
                                                  surface_list=", ".join(surfaces), rhythm=RHYTHM)
    # Images the model MAKES to look at are evidence; give it somewhere durable to put them.
    from litereality_agent.agent import scratch
    scratch_at = scratch.bind(near=room)
    prompt += scratch.prompt_line()
    from litereality_agent.agent.tool_narration import describe_tools_line
    prompt += describe_tools_line()

    # readable roots: the room, the stitches, the scan, and the resolved output tree (render PNGs +
    # fetched textures land under the realpath of the output symlink) + the repo root.
    from litereality_agent import REPO_ROOT as repo_root
    dirs = {str(repo_root), str(os.path.realpath(room.parents[2])), str(surface_ref), str(scan),
            str(os.path.realpath(surface_ref))}
    if scratch_at is not None:
        dirs.add(str(scratch_at))
    # Step budget: a graceful landing at `step_budget` tool-calls (wind-down then stop). The Claude
    # Code harness enforces it with a PreToolUse hook; Codex has no such hook and degrades to a hard
    # stop (`providers.describe` says which you got). `--max-turns` is the backstop below it.
    harness = providers.resolve("author", provider)
    spec = providers.SessionSpec(
        prompt=prompt,
        cwd=room,
        read_roots=tuple(Path(d) for d in dirs),
        capability_tools=CAPABILITY_TOOLS,
        model=model,
        max_turns=max_turns,
        step_budget=step_budget or 0,
        step_reserve=step_reserve,
        log=lambda s: print(s, flush=True),
    )

    # verify-reads guardrail: the stitches are the DECISIVE reference but are handed over by path,
    # so the model MUST choose to Read each one. Track which stitches it actually opened and report
    # per-surface coverage — a stitch that exists but was never read is a silent quality risk.
    present = {s: p for s, p in zip(surfaces, stitches) if p.is_file()}
    missing = [s for s in surfaces if s not in present]  # surface exists in Room.py, stage 2 gave no stitch
    stitch_names = {p.name: s for s, p in present.items()}  # basename → surface (symlink-proof match)
    read_surfaces: set[str] = set()

    print(f"== one-shot authoring + capability tools [profile={profile}] ==\n  room={room}\n"
          f"  tools=Read,Edit,Write,Glob + {list(CAPABILITY_TOOLS)}\n"
          f"  surfaces={len(surfaces)} stitches={len(present)}/{len(surfaces)}\n"
          f"  {providers.describe(harness, spec)}, max-turns={max_turns}\n",
          flush=True)
    t0 = time.monotonic()
    result_text, cost = "", None
    terminal_error = ""
    # Names and inputs live on ToolUseBlock only; the narrator holds the id→call map so a
    # ToolResultBlock can be attributed back. See tool_narration.py for what reading them off the
    # result block cost us.
    from litereality_agent.agent.tool_narration import ToolNarrator
    nar = ToolNarrator()
    # Structured record of the loop, so a one-shot run is documented the way staged runs were.
    from litereality_agent.agent.trace import AgentTrace
    tr = AgentTrace("author", room=room, scan=os.environ.get("LITEREALITY_SCAN"))
    # The prompt is the other half of "what happened": a tool choice only makes sense
    # against what the session was actually asked to do, and profiles change that.
    tr.start(model=model, room=str(room), profile=profile, stitches=len(present),
             max_turns=max_turns, scratch=str(scratch_at) if scratch_at else None, prompt=prompt)
    # Kept OUTSIDE the room: the room dir is copied and scanned wholesale downstream, and a
    # stray second Room-ish file in it is a trap.
    last_good = room.parent / ".room_checkpoint.py"
    checkpoint(room, last_good)  # the seed is valid by construction — start from it
    ended_early = ""
    stopped = ""
    try:
      async for m in harness.run(spec):
        # The raw sidecar wants the HARNESS's own object, not our normalised view of it.
        tr.raw(getattr(m, "raw", None) if getattr(m, "raw", None) is not None else m)
        for block in getattr(m, "content", []) or []:
            b = type(block).__name__
            if b == "TextBlock" and getattr(block, "text", "").strip():
                print(f"  …{block.text.strip()[:150]}", flush=True)
                tr.think(block.text)
            elif b == "ToolUseBlock":
                print(nar.use(block), flush=True)
                tr.tool(getattr(block, "name", "?"), getattr(block, "input", {}) or {},
                        tool_id=getattr(block, "id", "") or "")
            elif b == "ToolResultBlock":
                tr.result(block)
                name, inp = nar.result(block)
                # Stitch coverage: the model was handed its head-on references by PATH, so
                # opening them is a choice. Credit the surface only once the Read comes back
                # — a use block that errored is not evidence the reference was seen.
                if name == "Read" and not getattr(block, "is_error", False):
                    surf = stitch_names.get(Path(str(inp.get("file_path") or "")).name)
                    if surf:
                        read_surfaces.add(surf)
                # Safety net: an image the model made outside the run (habitually /tmp) is copied
                # in now, while it still exists. Result time, not use time — a Bash that creates
                # the file has not run yet when its use block arrives.
                for kept in scratch.rescue(inp, getattr(block, "content", None)):
                    print(f"      ↳ kept {kept.name}", flush=True)
                failed = nar.error_line(name, block)
                if failed:
                    print(failed, flush=True)
                # Newest COMPILING Room.py wins. Cheap enough to run per edit, and it is what
                # makes an interrupted session cost one edit instead of the whole run.
                elif name in ("Edit", "Write", "Bash"):
                    checkpoint(room, last_good)
        if isinstance(m, providers.SessionResult):
            result_text = m.result or ""
            cost = m.total_cost_usd
            stopped = m.stopped
            if m.is_error:
                terminal_error = result_text or "provider reported a terminal error"
    except Exception as exc:  # noqa: BLE001 — including the SDK's turn-cap Exception
        # Running out of turns is not a failed run. `Room.py` is edited IN PLACE, so by this
        # point the session's work is already on disk; raising here threw it away, because
        # the CLI runs authoring as a HARD stage and aborts the pipeline on a non-zero exit.
        # Two hours of paid authoring were discarded for hitting a cap that means "time's up",
        # not "this is broken". What actually matters is whether the room still compiles —
        # checked below.
        ended_early = f"{type(exc).__name__}: {exc}"
        print(f"\n  ⚠ session ended early — {ended_early}", flush=True)
        tr.think(f"[session ended early] {ended_early}")
    # A budget stop is a deliberate, graceful end, not a crash — record it the same way as a
    # turn-cap so the summary and exit logic treat it as "time's up".
    if stopped and not ended_early:
        ended_early = stopped
        print(f"\n  ⏹ authoring stopped on {ended_early} "
              f"({nar.calls} tool-calls) — Room.py kept as-is.", flush=True)
        tr.think(f"[step budget] {ended_early}")
    dt = round(time.monotonic() - t0, 1)
    calls, counts = nar.calls, nar.counts

    # The ONE thing that must hold when this returns: Room.py is valid Python. Every downstream
    # stage (materials, refine, qc, export) builds it, so a broken file fails all of them.
    broken = room_compiles(room)
    if broken and last_good.is_file():
        (room / "Room.py").write_bytes(last_good.read_bytes())
        print(f"  ⚠ Room.py did not compile ({broken}) — restored the last good version",
              flush=True)
        tr.think(f"[restored checkpoint] {broken}")
        broken = room_compiles(room)
    tr.end(calls=calls, cost_usd=cost, summary=result_text)
    print(f"\n== done {dt}s | calls={calls} {counts} | cost=${cost} ==\n", flush=True)
    if tr.ok:
        print(f"   trace → {tr.path}", flush=True)

    # per-surface stitch coverage — over EVERY surface in the room, not just the stitched ones, so a
    # surface stage 2 failed to stitch shows up as a gap instead of vanishing from the denominator.
    # Only meaningful where file reads are observable tool calls. Codex reads internally, so an
    # absent Read event says nothing about whether the reference was seen — reporting it as
    # "NEVER OPENED" would be a fabricated quality warning on every run.
    if "read_events" not in harness.supports:
        print(f"STITCH COVERAGE: not observable on the {harness.name} harness "
              f"(file reads are not reported as tool calls) — "
              f"{len(present)}/{len(surfaces)} surface(s) had a stitch available.", flush=True)
        if missing:
            print(f"  ⚠ {len(missing)}/{len(surfaces)} surface(s) have NO stitch: {', '.join(missing)} — "
                  f"stage 2 (surface stitches) did not produce one.", flush=True)
    else:
        skipped = [s for s in present if s not in read_surfaces]
        print("STITCH COVERAGE (head-on references the model actually opened):", flush=True)
        for s in surfaces:
            mark = "✓" if s in read_surfaces else ("✗ NO STITCH" if s in missing else "✗ NEVER OPENED")
            print(f"  {mark} {s}", flush=True)
        if missing:
            print(f"  ⚠ {len(missing)}/{len(surfaces)} surface(s) have NO stitch: {', '.join(missing)} — "
                  f"stage 2 (surface stitches) did not produce one.", flush=True)
        if skipped:
            print(f"  ⚠ {len(skipped)}/{len(present)} stitch(es) never read: {', '.join(skipped)} — "
                  f"those surfaces were authored WITHOUT looking at their head-on reference.", flush=True)
        elif not missing:
            print(f"  all {len(present)} stitch(es) read ✓", flush=True)

    print("\nSUMMARY:\n" + result_text[:2500], flush=True)

    # Exit status = "is the room usable", not "did the session run to completion". Downstream
    # stages need valid Python and nothing else; a turn cap with a compiling room is a partial
    # success worth keeping, and only an unrecoverable room is worth aborting the pipeline for.
    if ended_early:
        print(f"  note: ended early ({ended_early}) — Room.py is valid, continuing.", flush=True)
    if broken:
        print(f"  ✗ Room.py does not compile and no checkpoint could be restored: {broken}",
              flush=True)
    if terminal_error:
        print(f"  ✗ authoring provider failed: {terminal_error}", flush=True)
    return 1 if broken or terminal_error else 0
