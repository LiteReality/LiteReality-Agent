# Support-first authoring

The normal `run` / `stage author` path uses the evidence-rich `open` brief. Author,
materials, object refinement, and the visual critic/reviewer default to Codex with
`LR_CODEX_MODEL=gpt-6-astra`, high reasoning. The per-object render tool now works
over stdio MCP on Codex. Reconstruction, image generation and procedural object
initialization keep their existing model choices. Explicit per-role overrides
remain possible; there is no silent model fallback. Model identifier/effort were
checked against [official OpenAI guidance](https://developers.openai.com/api/docs/guides/latest-model).

## Acceptance, not just generation

Author → compile → mesh support/export validation → independent capture-pair review
→ bounded repair/recheck → accepted source. Publish rebuilds/bakes and repeats the
checks on that final build. Missing dependencies, missing evidence, failed checks,
budget stops and incomplete reviews do not produce an accepted scene. A plain
`Room.py` or `Room.glb` on disk is no longer sufficient to reuse a completed stage.

`room_preview/author_acceptance.json` and `acceptance.json` bind acceptance to SHA256
hashes of the source files, GLB, layout and manifest. Changing any of them invalidates
completion. Old stage-state records cannot bypass this check. Failed candidates
remain on disk for inspection; the stage reports `failed` with `needs_repair`.
They may be deliberately shared for review, but are not accepted deliverables.

Checks include:

- actual upward-facing mesh contact (5 mm float/penetration tolerance), not a
  bounding-box top; attachments within 1 cm of the real target mesh;
- known support targets, rooted acyclic support chains, all manifest instances
  including `represents_prims`, preserved bundled source GLBs;
- disconnected members of grouped meshes must have a contact path to support;
  glTF UV/normal seams are welded for analysis only; more than 10,000 components
  or 20,000 candidate contact pairs is an incomplete check, never a pass;
- animated source assets keep animation (fixed picture windows are not required to
  move), and supported contents follow the specified support part in the exported hierarchy;
- independent visual score ≥8, no outstanding issues, and explicit object coverage.

These are conservative geometric and visual checks, **not a physics stability or
force certificate**. Component connectivity cannot establish semantic correctness,
structural strength or centre-of-mass stability. Uncertain/invisible objects should
fail visual review. Simulation remains a separate optional stage.

## Moving supports

```python
group_fixture("Keyboard", "keyboard", parts,
              rests_on="Table1", support_part="desk_top_lift")
```

Use the exact node name in this asset. Export preserves the rest pose and parents
the group to that link. Nested props keep their own IDs and are excluded from their
support's bounding box. Unknown parts, cycles and independently animated followers
are rejected. An animated support without an explicit part fails validation; the
compiler does not guess whether the object belongs on the fixed base or moving top.
Old authored Room.py copies do not acquire the new helper automatically: rebuild
the seed or port the helper explicitly. The final gate rejects missing propagation.

## Bounded work and resumption

```bash
litereality run /path/to/capture --through publish \
  --repair-rounds 2 --repair-steps 40 --repair-seconds 600
```

Maximum three repair rounds; zero runs checks only. Defaults: two rounds, 40 tool
calls / 600 seconds each; independent reviews 240 seconds. Other Codex sessions
default to 1800 wall seconds, configurable via `LR_CODEX_SESSION_SECONDS`.
These are **tool/time caps, not guaranteed token or dollar caps**. Model calls
consume account usage. Tests marked `live` are not run by the offline test suite.

Every repair round retains source/build snapshots and its findings under
`realism_authoring/acceptance/attempt-*/`. Rerunning author preserves its editable
room instead of deleting it; `--force author` archives it before reseeding.
Repair never lowers tolerances or removes failed objects to obtain a pass.

Verification: offline geometry/orchestration/provider tests plus a Blender test
sampling a moving tabletop at six frames and checking the exported GLB hierarchy.
Live scan results are reported separately: code tests are not evidence that a real
scene has passed reconstruction or visual acceptance.

`scripts/smoke_support_first.py` makes an isolated capture copy for an end-to-end test.
After a finished/failed attempt, `--resume` reuses completed stages without regenerating
assets, retaining attempt history and checking both capture copies against the original
hashes. Codex MCP tools explicitly inherit the run's scene/output/model settings so an
isolated output directory resolves just as the default `run/` does.
