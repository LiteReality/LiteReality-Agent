"""The evidence kit — plain-python helpers handed to an authoring model as FILES, not tools.

A tool answers one question the way its author framed it. A helper is a function the model can
combine, loop over and adapt. The kit exists because the measured GPT-6 one-shot run got its
proportions right by probing depth, slicing the point cloud and triangulating photo pixels — none
of which our tool set offered — and it did all of that with numpy in a scratch script. So the kit
is what that script needed: readers for the scan, measurement helpers, a photo→texture rectifier,
contact sheets, and the ARKit→Blender camera math.

`evidence_pack.build()` copies this package next to the room as `evidence/helpers/` (a copy, so the
run record is self-contained), alongside links to the scan and the stitches and a README that
states the data formats and coordinate conventions once.

Modules (none import Blender except `arkit_cameras`, which only runs inside `bpy`):
  read_scan        Scan(dir): frames, poses, intrinsics, RGB, depth (m), confidence, point cloud
  measure          probe_depth, pcd_slice, height_profile, triangulate — all in Blender Z-up metres
  rectify          rectify_region: a photo quad → a straight-on texture image (screens, boards)
  contact_sheets   every frame on a few sheets, so the whole capture is seen before anything is built
  arkit_cameras    (bpy) one Blender camera per photograph; render the scene from them
"""
