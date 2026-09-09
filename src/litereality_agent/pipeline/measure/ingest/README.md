# Ingest

Ingest turns a RoomPlan capture into the evidence reconstruction needs:

```text
capture → scene extraction → merged boxes → crops → DINO refinement → OpenAI references
```

The public boundary is `ingest.run(context, options)`. Generated data stays under the scan's
`scene_init/obj_stage/object_init` tree so an interrupted run can reuse it.

- `extract/` reads RoomPlan and RGB/depth capture data.
- `crop/` and `detect/` produce/refine object views.
- `references/` selects evidence and generates clean object/opening references.
- `merge_boxes.py` combines RoomPlan boxes that represent one physical object.

GroundingDINO, DINOv2, and reference-image providers live under their named `models/` packages.
Ingest coordinates those implementations but does not own their execution runtime.

Old runs may contain provider-specific artifact names. `pipeline.scene_init.paths.ref_artifact`
reads those as fallbacks; new code uses provider-neutral names.
