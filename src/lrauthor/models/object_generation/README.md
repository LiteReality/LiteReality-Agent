# Object generation

For objects the complexity router
([`classify_complexity.py`](../../pipeline/author/reconstruct/classify/classify_complexity.py))
sends to the **procedural** path (simple/regular geometry — tables, storage,
appliances, TVs, sinks), build them in Blender from primitives + PBR via the
`image-to-articulated-glb` agent — and make them **articulate correctly**.

The key idea: RoomPlan only detects a **fixed set of object categories**, so we
hand-author a detailed spec per category — geometry, materials, and exactly how
each part moves — in [`category_specs.py`](category_specs.py). The generator
injects the matching spec + the object's **real RoomPlan dimensions** + its clean
reference image into the agent prompt, so the asset both looks right and comes
alive the right way (the dishwasher door drops to horizontal, the drawer pulls
out along +Y, the cabinet door swings on its outer vertical edge).

```
routing.json (procedural) ─► category_specs[category] + bbox + reference ─► agent ─► <name>.glb (+ articulation)
```

## Run

Needs **Blender 4.x/5.x on PATH** and `claude_agent_sdk` (uses logged-in Claude
Code creds if `ANTHROPIC_API_KEY` is unset). From the repo root:

```bash
export PATH="$LITEREALITY_BLENDER:$PATH"   # your Blender install dir

# preview the composed prompts (free, no agent calls)
uv run python -m lrauthor.models.object_generation.generate --only storage --dry-run

# the articulated categories (highest value — correct motion)
uv run python -m lrauthor.models.object_generation.generate --only storage dishwasher

# everything on the procedural route
uv run python -m lrauthor.models.object_generation.generate

# one scan
uv run python -m lrauthor.models.object_generation.generate --scan tea_room
```

Flags: `--scan`, `--only <categories>`, `--concurrency` (default 2),
`--retries`, `--skip-existing`, `--max-budget-usd` (per attempt), `--model`.

## Outputs

```
procedural/glb/<scan>/<name>/
    <name>.glb            primitive build + PBR + glTF articulation extras + open/close animation
    object.py             self-contained Blender rebuild (blender -b --python object.py)
    object.md             how to edit it (constants, joints, rebuild)
    previews/             render-check (open/closed for movers, iso/front for static)
    textures/
procedural/glb/procedural_report.json   objects pass
procedural/glb/openings_report.json     doors/windows pass (--openings)
```

Each moving part carries glTF node extras `articulation_type`
(revolute|prismatic), `articulation_axis`, `limit_min`, `limit_max` — readable
straight by a simulator.

## Files

```
object_generation/
  category_specs.py        ★ per-RoomPlan-category geometry + articulation specs (the domain knowledge)
  generate.py              category-aware launcher (routing -> composed prompt -> agent -> sorted GLBs)
  README.md
```

The agent engine + Blender helper scripts live in
[`articulated-glb-agent`](articulated-glb-agent/) (the
`image-to-articulated-glb` skill). `batch_glb.py` there is the generic
(non-category) batch driver; this module adds the category intelligence.

## Routing recap

`object_init` makes clean references → `classify_complexity` routes each object
(chairs/sofas/abstract → TRELLIS; regular box/appliance geometry → procedural,
per the user policy) → **procedural ones come here**, TRELLIS ones go to
[`models/trellis/`](../../models/trellis/).
