# Fresh scan test: lab_scan_latest_2

Date: 2026-09-21. Branch: `feat/gpt6-support-first-authoring`.
Outcome: **initialization complete; final saved candidate passes geometric support
validation; authoring/visual acceptance incomplete. Nothing published.**

## Test isolation and execution

Source: `/scratch2/iclr_2027/LiteReality-Agent/lab_scan_latest_2`.
Run: `/scratch2/iclr_2027/LiteReality-Agent/run-support-first/lab-scan-20260921-dKBL2I`.
The supplied capture was copied into this run. Ingest, DINO refinement, reference
generation, reconstruction and seed assembly ran from scratch, without reusing an
older scene's assets. SHA256 verification confirms the original capture is unchanged.

- Ingest: 115.5 seconds; 53 DINO-refined boxes.
- Reconstruction: 2305.6 seconds; seven furniture assets, two chair templates
  representing seven chair instances, and two windows. One chair template and one
  table required regeneration.
- Seed: 20.8 seconds; all 11 assets compiled into the initialized room.
- First author attempt was interrupted after an isolated-run MCP context bug was
  identified. Initialization was retained, not regenerated.
- Resumed GPT-6 Astra authoring hit the enforced 1800-second limit at 70 tool calls.
  Its source was retained and the stage failed with `needs_repair`, as intended.

## Results

The corrected seed audit had 12 blocking findings. After authoring, the final saved
source was rebuilt with `--regenerate` into `diagnostic-final/room_preview` and
checked independently of the stopped session:

- **99 objects, 89 declared support relationships; validation PASS.**
- Zero blocking findings and zero mesh clashes.
- Disconnected members, finite support contact, required source instances,
  source-asset animation and moving-support hierarchy checks passed.
- Six bounding-box support notes, 76 box overlaps and 12 open-mesh diagnostics
  remain advisory. This is not a physics stability certificate.

The last in-session visual critic scored **6/10**, flagging cool lighting, pale
carpet, opaque blinds and material colours. The final saved edits include further
appearance changes, but those changes have **not** received a new visual verdict.
Desk-base and chair-shape fidelity also remain concerns. Neither the formal
author-acceptance loop nor publish completed after the authoring timeout.

Artifacts relative to the run directory:

- `smoke_status.json`: terminal state and original/resumed attempt history.
- `source_checksums.json`: original capture hashes.
- `diagnostic-final/room_preview/Room.glb`: rebuilt final saved candidate.
- `diagnostic-final/room_preview/validation.json`: final geometric audit.
- `output/lab_scan_latest_2/realism_authoring/room`: editable authored source.
- `output/lab_scan_latest_2/realism_authoring/_scratch/run_002`: render comparisons.

## Integration findings fixed on the branch

The test exposed and helped fix glTF seam vertices being mistaken for disconnected
members, missing scene/output/model environment in Codex MCP children, fixed picture
windows incorrectly requiring animation, and a nested animated cabinet making its
floor appear animated. Rear object previews and safe isolated-test resumption were
also added. Contact tolerances were not loosened to obtain a pass.

This was an iterative integration test, not an uninterrupted run of one revision:
the fresh run began at `f92e15e`, resumed at `186a392`, and the final diagnostic
rebuild/audit used `cbfb6a5`. The deadline-awareness prompt was added after the capped
author session and is covered by offline tests, not a second complete live run.

Verification at that code revision: **734 offline tests passed, 2 skipped, 10
deselected**, plus the real Blender support-motion/export test. There is one existing
trimesh inertia warning. Next work should be bounded visual refinement of this saved
candidate, followed by full acceptance—not another from-scratch reconstruction.
