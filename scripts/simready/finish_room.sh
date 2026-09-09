#!/usr/bin/env bash
# finish_room.sh <scan> — everything after `reconstruct`, without re-authoring.
#
# The four rooms in run-simready/ are being rebuilt from freshly generated objects, but their
# AUTHORING is being reused: the brief that produces `rests_on`/`attached_to`, the small objects
# and the real luminaires is not on this branch, so re-running the author agent would return a
# strictly worse room. So the seed room is exported fresh (which is what carries each object's
# new `sim/` physics sidecar into the Room package) and the authored `Room.py` — plus the
# materials it fetched — is laid over the top of it.
#
#   seed        -> scene_init/scene_stage/room_init/room       (Objects/ + sim/, from the new glbs)
#   merge       -> realism_authoring/room                      (that, + the authored Room.py)
#   publish     -> realism_authoring/room_preview/Room.glb + room_layout.json
#   simulate    -> realism_authoring/mujoco/scene.xml
set -euo pipefail

SCAN="${1:?usage: finish_room.sh <scan>}"
ROOT="run-simready"
PY="${PY:-.venv/bin/python}"
SCENE="$ROOT/$SCAN"
AUTHORED="$SCENE/realism_authoring/_authored_room"
ROOM="$SCENE/realism_authoring/room"

[ -f "$AUTHORED/Room.py" ] || { echo "!! no stashed authored room at $AUTHORED"; exit 1; }

echo "── seed · $SCAN ───────────────────────────────────────────"
$PY -m lrauthor stage seed "$SCENE" --output-root "$ROOT" --force

SEED="$SCENE/scene_init/scene_stage/room_init/room"
[ -f "$SEED/Room.py" ] || { echo "!! seed produced no Room.py"; exit 1; }

echo "── merge · the new objects under the authored room ────────"
rm -rf "$ROOM"; mkdir -p "$(dirname "$ROOM")"
cp -a "$SEED" "$ROOM"                      # Objects/ (with sim/), manifest.json, Room.md
cp -a "$AUTHORED/Room.py" "$ROOM/Room.py"  # the authored shell, fixtures and props
[ -d "$AUTHORED/materials" ] && cp -a "$AUTHORED/materials" "$ROOM/materials"
[ -f "$AUTHORED/textures.json" ] && cp -a "$AUTHORED/textures.json" "$ROOM/textures.json"
echo "   objects: $(ls "$ROOM/Objects"/*/ -d 2>/dev/null | wc -l) dirs, \
$(ls -d "$ROOM/Objects"/*/*/sim 2>/dev/null | wc -l) with physics; \
support declarations in Room.py: $(grep -c 'rests_on=\|attached_to=' "$ROOM/Room.py" || true)"

echo "── publish · build the room ───────────────────────────────"
$PY -m lrauthor stage publish "$SCENE" --output-root "$ROOT" --compare-frames 0

echo "── simulate · MJCF ────────────────────────────────────────"
$PY -m lrauthor stage simulate "$SCENE" --output-root "$ROOT"

echo "── done · $SCAN ───────────────────────────────────────────"
cat "$SCENE/realism_authoring/mujoco/export_report.json" 2>/dev/null || true
