---
name: image-to-articulated-glb
description: Generate an articulated (pull-out drawers / openable doors) 3D GLB model from a reference image, with real PBR textures (diffuse/roughness/normal) and headless render verification. Use when the user asks to build an articulated model (GLB/glTF) from an image, add textures or physical material parameters (roughness etc.) to a model, or render-check/preview a GLB file.
argument-hint: <reference_image> [output.glb]
---

# Image → Articulated GLB (with PBR textures + render verification)

Pipeline for turning a reference image of furniture/objects into a GLB with
movable parts, embedded PBR textures, and rendered visual proof.

## Invocation arguments
When invoked as `/image-to-articulated-glb <args>`, parse them as:
- **arg 1** — reference image path (required). If missing or the file doesn't
  exist, ask for it before doing anything else.
- **arg 2** (optional) — output GLB path, or a directory (then use
  `<dir>/<image_basename>.glb`). Default when omitted:
  `<image_dir>/<image_basename>_glb/<image_basename>.glb`.

With valid args, run the FULL pipeline end-to-end without pausing for
confirmation: analyze image → fetch textures → build → validate → render-check
→ emit a self-contained `object.py` + `object.md` → final report showing both
preview PNGs, the GLB path, and the articulation summary. Put every artifact
(textures/, previews/, the build recipe, `object.py`, `object.md`) in the output
GLB's directory so each job is self-contained.

Deliverables per object (in the GLB's directory):
- `<stem>.glb` — the model (embedded textures + articulation extras + animations)
- `object.py` — a **self-contained** Blender script that rebuilds it on its own
  (no external imports / sys.path hacks); runs with `blender -b --python object.py`
- `object.md` — how to edit the object (constants, articulation, rebuild command)
- `previews/`, `textures/`, and the readable build recipe

All reusable scripts live in `scripts/` next to this file. Blender binary:
`/opt/homebrew/bin/blender` (fall back to `blender` on PATH; the desktop app is
at /Applications/Blender.app). The blender-mcp addon is usually NOT connected —
go headless CLI by default; only try MCP tools if the user says Blender is open.

## Workflow

### 1. Analyze the reference image
Read the image. Identify: overall object type and real-world dimensions
(estimate, meters), which parts are movable (drawers → prismatic slide,
doors/lids → revolute hinge), handle/hardware style, material (wood tone,
metal, plastic), and small details (casters, grommets, vents).
State the analysis to the user before building.

### 2. Get PBR textures (internet first, procedural fallback)
```bash
python3 scripts/fetch_polyhaven.py <asset_id> <texture_dir> [1k|2k]
```
Known-good Poly Haven asset ids: `plywood` (light wood), `wood_table_001`,
`oak_veneer_01`, `fabric_pattern_07`, `brushed_concrete`. Downloads
`<id>_diff_*.jpg`, `<id>_rough_*.jpg`, `<id>_nor_gl_*.png`. If download fails
(offline), generate a procedural texture with PIL instead and say so.

### 3. Write the build script
Copy `scripts/example_build_desk.py` **into this object's OUTPUT directory** as
`<glb_dir>/build_<stem>.py` — do **NOT** write per-object build scripts into the
shared `scripts/` folder. That folder is the committed engine only (`blender_lib.py`
+ the `example_build_desk.py` template + the utilities); per-object recipes live in
the output dir next to their `object.py`. Import helpers from `scripts/blender_lib.py`
(add it to sys.path). This authored recipe stays readable (it imports the shared
lib); step 6 bundles it into the self-contained `object.py` deliverable. Conventions:

- Units meters, Blender Z-up (exporter converts to glTF Y-up). **CANONICAL
  ORIENTATION (mandatory):** the object stands upright, base flat on the ground at
  Z=0, centred at X=0/Y=0, and its FRONT — the face a person faces and uses
  (screen / doors / drawers / handles / seat-front) — faces **-Y** (the Blender
  Front view face). Drawers slide toward -Y; doors hinge around Z and swing toward
  -Y. Do NOT bake a correcting rotation into the export — the GLB's local axes must
  BE this frame. A reversed build (front on +Y, joints opening into the body) must
  be rotated 180° about Z and rebuilt, never shipped — wrong orientation makes the
  object face backwards when placed in the scene.
- Every movable part = ONE joined object (its own glTF node), origin placed at
  a meaningful point (drawer front center / hinge line). Parent it to its
  carcass with `parent_keep_world()`.
- Build drawers as real open-top boxes (front + bottom + sides + back), not a
  single slab — they must look right when pulled out.
- Tag every movable part with `set_articulation(ob, 'prismatic'|'revolute',
  axis, limit_min, limit_max, origin=<hinge point>)` → exported as glTF node
  `extras` for simulators. **Pass `origin` for every revolute joint** — it is the
  same point you already handed to `join_parts()`, and without it every consumer
  has to re-derive the pivot from the mesh and guess.
- State the physics you actually know, and only that:
  - `set_physics(ob, mass=…)` when the reference tells you the weight, or
    `set_physics(ob, density=…)` (kg/m³ of that part's own volume) when the
    material is obvious but the weight is not. Never both.
  - `set_physics(ob, friction=…, restitution=…)` for a surface that is clearly
    not ordinary — glass, rubber feet, a felt pad. Leave them out and a default
    is keyed off the material NAME, so name materials honestly (`Brushed_Steel`,
    `Oak_Veneer`, `Glazing`) and most of this comes for free.
  - `set_object_mass(ob, kg)` on any one part when you know what the WHOLE object
    weighs. That stops the compiler deriving a total from category density.
  Guessing is worse than silence here: an unstated value is derived from the mesh
  and the category, and those defaults are calibrated. A wrong stated value is not.
- Animate with `animate_prismatic()` / `animate_revolute()` — staggered
  open-then-close clips, one action per part (one glTF animation per part).
- Materials: `make_pbr_material()` wires diffuse(+optional tint) / roughness
  (Non-Color) / normal (Non-Color, NormalMap node). Plain parts use
  `make_plain_material(name, color, metallic, roughness)`.
- **GLASS (windows, glazed doors, glass partitions/walls) — MUST be clearly
  see-through, never an opaque or textured panel.** Build the glazing as its own thin
  pane with a Principled BSDF: **Transmission Weight ≥ 0.9**, **Roughness ≤ 0.06**,
  **IOR 1.45**, base colour a near-white faint cool tint (~0.9, 0.93, 0.96), metallic 0.
  Do NOT paint the reference image onto the glass and do NOT leave it solid — it should
  read as empty, luminous, real glass. Keep the pane thin and set back behind the
  sash/frame front. (`make_glass_material()` in the example scripts is the template; bump
  its transmission to ~0.9.) glTF carries this via KHR_materials_transmission.
- UV: call `uv_cube_project(ob)` on every wood/textured part BEFORE joining.
- Export via `export_glb(path)` (GLB, embedded images, extras + animations).

Run: `blender -b --python your_build.py -- <texture_dir> <out.glb>`

### 3b. Curved surfaces — don't deliver an all-boxes model
Pure axis-aligned boxes read as CAD blockouts, not real objects. While
analyzing the image (step 1), explicitly note every curved feature, then use
the matching helper:

- **Softened edges** (almost every real tabletop, drawer front, side panel
  has a 2–6 mm edge radius): `add_rounded_box(...)`, or `bevel(ob, width,
  segments)` on any existing box. Cheap — segments=2 is enough.
- **Bowed / curved fronts** (curved drawer & door faces, rounded aprons,
  appliance fascias): `add_arc_panel(name, width, height, thickness, bulge,
  loc)` — face bows toward -Y (front) by `bulge` at center.
- **Turned / rotational parts** (round legs, knobs, cylindrical pulls, lamp
  bases, bowl feet, vases): `add_lathe(name, profile, loc)` with a
  (radius, z) profile; radius 0 at the ends closes the shape.
- **Shading**: curved helpers call `shade_smooth_auto()` themselves (smooth
  with sharp edges preserved); call it manually after joining curved parts.

Match the reference, don't decorate: bevel radii and bulges should come from
what the photo actually shows — if the reference really is sharp-edged
panel furniture, light edge bevels (~2 mm) are still right, but skip arcs
and lathes. `uv_cube_project()` still works fine on beveled/curved parts.

### 4. Validate GLB structure (no Blender needed)
```bash
python3 scripts/validate_glb.py <out.glb>
```
Check the printout: expected node names/hierarchy, one animation per movable
part, articulation extras present, 3 materials with textures embedded.

### 5. Render-check (mandatory — never deliver unseen geometry)
```bash
blender -b --factory-startup --python scripts/render_glb_preview.py -- <out.glb> <out_dir>
```
Renders `preview_closed.png` (frame 1) and `preview_open.png` (auto-picked
mid-animation frame) with an auto-framed camera, and prints every mesh's world
bounding box. Then **Read both PNGs** and compare against the reference image
and the bbox numbers. **Also render the front view and confirm the canonical
orientation: the front view must show the object's real FRONT (screen / doors /
seat-front / handle), not the back or a side. If it shows the back, the front was
built on +Y — rotate the whole object 180° about Z and rebuild.** Iterate until
correct. Extra views: `-- <glb> <dir> --frames 1:closed,80:open --view front`.

### 5b. Probe QC (semantic gate — must pass before you finish)
```bash
python3 scripts/probe_glb.py <out.glb>          # add --json for structured output
```
This is the objective gate — it checks what a render can't measure: parts are
**connected** (no floating/orphaned parts), joints are **sane** (real axis +
non-degenerate limits), moving parts **open toward the front (−Y)** not into the
body, and (advisory) parts don't grossly interpenetrate or swing through the body.
- **`FAIL` / any `[hard]` line** = a real defect you MUST fix, then rebuild and
  re-probe. The most important is `orientation`: a part opening toward +Y means the
  whole object was built 180° backwards — rotate it 180° about Z and rebuild (do
  NOT ship it; this is exactly the "reversed object" failure). `connectivity` means
  a part isn't parented into the tree; `articulation_sanity` means a mover has a
  bad/zero axis or equal limits.
- **`[soft]` lines** (rest overlap, swing clearance) = review and fix if they're
  real interpenetration; they don't block delivery but usually indicate sloppy
  geometry.
Do not emit `object.py` or say DONE until `probe_glb.py` exits 0 (no `[hard]`).

### 5b-ii. Physics gate (does it hold together in a solver?)
```bash
python3 -m lrauthor.models.object_generation.sim check <out.glb>
```
Compiles the object's mass, inertia and convex colliders, exports the URDF and
MJCF, then runs three scenarios on the object ALONE — drop it just above a floor,
open each joint one at a time, and lean gravity over until it slides. This catches
what a render and a probe cannot: a part that was never really attached, colliders
that start interpenetrating, an inertia the solver refuses, a joint whose axis or
origin drives its part through its own carcass.
- **`HARD`** = fix and rebuild. `joint_self_collision` and `drop_came_apart` are
  the two that mean the build is wrong rather than merely unusual.
- **`SOFT`** = read it. `sibling_sweep` (two leaves that cannot both be open) is
  usually a real mechanism, not a defect — a dishwasher rack does slide through
  the space its closed door occupies, and no joint format can record that
  dependency.

### 5c. Completeness check (does it MATCH the reference?)
```bash
python3 scripts/completeness_check.py --ref <reference.png> \
    --render previews/preview_front.png --render previews/preview_closed.png \
    --render previews/preview_open.png
```
`probe_glb.py` proves the model is *valid*; this checks it is *complete* — that every salient
feature the reference shows is actually built. It lists what's **missing** (e.g. "two coat hooks
above the window", "wrong number of drawers", "no handle"). For each missing feature: **add it to
the build recipe**, rebuild, re-render, and re-run — until it reports `COMPLETE`. Do NOT drop
features that are already right while adding the missing ones. This is what stops a
geometrically-valid-but-sparse model (a bare door when the reference has hooks + trim) from
shipping.

### 6. Emit the self-contained `object.py`
Bundle the readable recipe + `blender_lib.py` into ONE portable file, so the
object's definition runs anywhere with no import/sys.path fragility:
```bash
python3 scripts/make_selfcontained.py <your_build.py> <glb_dir>/object.py --name <stem>
```
This inlines the helper library, drops the recipe's `from blender_lib import …`
block, and writes a portable `__file__`-relative arg block (defaults to
`./textures` and `./<stem>.glb`). Then run `object.py` to **export the preview
GLB into the output directory** — it must reproduce the GLB you validated, and it
is the model people open to preview the object:
```bash
blender -b --python <glb_dir>/object.py -- <glb_dir>/textures <glb_dir>/<stem>.glb
python3 scripts/validate_glb.py <glb_dir>/<stem>.glb
```
Keep the readable recipe, `object.py`, and the built `<stem>.glb` together in the
output directory.

### 7. Write `object.md` (how to edit the object)
A short companion doc next to `object.py` so a person/agent can change the object
without re-deriving anything. Cover, in this order:
- **What it is** — one paragraph + a pointer to the reference image.
- **Rebuild** — the `blender -b --python object.py [-- <texdir> <out.glb>]` command.
- **Layout of object.py** — note the two sections (inlined helpers = don't edit;
  `OBJECT DEFINITION` = edit here).
- **Coordinate conventions** — meters, Z-up, front faces −Y.
- **Articulation** — each movable part: joint type, axis, limits, animation name.
- **Key constants** — a table mapping each editable constant → its effect (pull
  these from the top of the OBJECT DEFINITION section).
- **Materials** — which textures/materials and how to swap or tint them.

## Pitfalls (learned the hard way)
- `primitive_cube_add(size=1)` makes a UNIT cube → `ob.scale = dims`
  (NOT dims/2). Wrong scale renders as an "exploded" model: parts at correct
  centers but too small, gaps everywhere. `blender_lib.add_box` already does
  this correctly — use it instead of raw bpy ops.
- Always `reset_scene()` / `read_factory_settings(use_empty=True)` before
  building AND before importing for render — otherwise the default startup
  cube ends up as a giant white box in renders.
- Roughness/normal images must be set to `Non-Color` colorspace.
- Headless EEVEE is unreliable on macOS — render previews with Cycles
  (the render script defaults to Cycles, ~48 samples is enough).
- glTF exporter needs `export_extras=True` or articulation metadata is lost.
- Check world bbox numbers (printed by the render script) against intended
  dimensions — catches scale/offset bugs even when the render "looks ok".
- glTF export silently DROPS Mix/MapRange shader nodes — bake color tints /
  roughness remaps into the texture files with PIL and wire maps directly to
  the Principled BSDF, or the exported GLB won't match the Blender look.
  (`make_pbr_material(tint=...)` uses a Mix node: fine for Blender renders,
  but bake the tint into the diffuse before export if GLB color must match.)
- Blender's glTF importer only activates the FIRST animation clip; per-part
  clips import as unassigned actions, so a render shows only the first part
  moving. render_glb_preview.py now reassigns every action to its object by
  name prefix (`animation_data.action_slot = act.slots[0]`, Blender 5.x
  slotted actions) right after import — keep that block if editing it.
- Blender 5.1 `transform_apply(scale=True)` ALSO bakes the object's LOCATION
  into the mesh (object jumps to world origin). Harmless until you rotate
  the object afterward — then the part orbits the world origin and lands
  far away (huge bbox is the tell). `blender_lib.add_box` now scales mesh
  data directly (`ob.data.transform`) instead; never call transform_apply
  on a part you'll later rotate/animate.
- render_glb_preview.py needs ABSOLUTE paths or Cycles fails to save.
- Poly Haven has no clean appliance metal; `beige_wall_001` rough+normal
  with a PIL white-blended diffuse works well for white enamel.
