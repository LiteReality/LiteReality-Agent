"""Central paths + knobs for the room-optimization harness (stage 1).

Single source of truth; every other harness module imports this. The harness is
**per scan**: the scan is taken from ``$LITEREALITY_SCAN`` (run_harness sets it from
``--scan``). Everything reads/writes under ``run/<scan>/scene_init/scene_stage/`` so each
harness pass is a self-contained, viewable iteration:

    scene_stage/room_init/{room,room_preview}            the baseline white-shell room
    scene_stage/_scene_assets/                           packed object GLBs the room places
    scene_stage/_harness/                                per-scan scratch (overlay, view_selection)
    scene_stage/stage_<N>/iteration_<M>/
        room/                                            the Room.py THIS iteration edits
        room_preview/                                    THIS iteration's render (PNGs, room.glb)
        .harness_tracking/                               prompt, trajectory, verify.json, version.json
    scans_uploaded/<scan>/                               ground-truth capture (frame_*.jpg/json + usdz)

The LIVE room/preview paths (ROOM_DIR, ROOM_PY, RENDER_DIR, …) are DYNAMIC: the
runner calls :func:`set_room` at the top of each iteration to repoint them (and the
``LITEREALITY_ROOM_DIR`` / ``LITEREALITY_ROOM_PREVIEW`` env vars so the room-ops
CLIs follow). All render + view-selection tooling lives inside the SDK.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from lrauthor import PACKAGE_ROOT
from lrauthor.settings import load_settings

SETTINGS = load_settings()

# --- locations -------------------------------------------------------------- #
# Two roots, and the difference matters: PACKAGE_ROOT is where OUR files are (src/lrauthor —
# the view tools and Blender helpers a subprocess must exec), REPO_ROOT is
# where the USER's data is (output/, scans_uploaded/, .key, .venv). They coincide in a source
# checkout, which is exactly why each path below has to say which one it means.
AGENT_DIR = SETTINGS.repo_root
ROOT = SETTINGS.repo_root
SDK = PACKAGE_ROOT / "room_ops" / "rendering"   # what stayed: room_render
TOOLS = PACKAGE_ROOT / "agent" / "tools"           # what the tools own

# --- where the scan context comes from -------------------------------------- #
# Historically this had exactly one source: the caller exported $LITEREALITY_SCAN (plus
# $LITEREALITY_OUTPUT / $LR_SCANS_DIR / $LITEREALITY_FINAL, which all had to agree). The scene
# PACKAGE init writes carries the same contract as data, so it can be resolved from a folder
# instead — from `--scene <dir>`, $LR_SCENE, or simply being inside one. This is the choke point
# every harness module imports, so bootstrapping here covers all of them at once.
#
# An already-exported environment WINS: a discovered package only fills gaps, so the CLI's
# behaviour is byte-for-byte what it was. Only an explicit --scene/$LR_SCENE overrides.
SCENE = None
SCAN = os.environ.get("LITEREALITY_SCAN", "").strip() or "_unbound"


def _output_root() -> Path:
    return SETTINGS.resolved_output_root()


SCAN_OUT = _output_root() / SCAN
# Stage 1 writes everything under scene_init/; stage 2 reads and extends its scene_stage half.
SCENE_STAGE = SCAN_OUT / "scene_init" / "scene_stage"   # stage 1's tree — read-only from here

# Stage 2's own root, beside stage 1's. `$LR_AUTHORING` is what the scene package exports; the
# harness used to derive every path from SCENE_STAGE, so its stitches and renders landed inside
# the tree stage 1 owns, and the two halves could not be deleted or re-run independently.
AUTHORING = Path(os.environ.get("LR_AUTHORING") or (SCAN_OUT / "realism_authoring"))

# the baseline white-shell room the harness seeds stage 1 from (built by export_room):
ROOM_INIT = SCENE_STAGE / "room_init"

# the packed object GLBs the room places (shared across iterations) — must match
# room_ops.paths.assets_dir (scene_stage/_scene_assets).
ASSETS_DIR = SCENE_STAGE / "_scene_assets"
ASSET_GLB_DIR = ASSETS_DIR / "glb"

# ground-truth capture (override the scans location with $LR_SCANS_DIR)
SCANS_ROOT = SETTINGS.resolved_scans_dir()
SCAN_DIR = SCANS_ROOT / SCAN

# per-scan harness SCRATCH (view_selection/ranking overwritten each render). NOT
# per-iteration — the iteration owns its room/ + room_preview/.
HARNESS_OUT = Path(os.environ.get("LR_HARNESS") or (AUTHORING / "_harness"))
VIEW_SELECTION = HARNESS_OUT / "view_selection.json"
OUTPUT = HARNESS_OUT  # where view_selection/ranking land
# NOT created here. Importing a config module used to mkdir under $LITEREALITY_OUTPUT, so merely
# importing it — from a test, a --help, a tab-completion — left an empty scan tree in the user's
# data directory (`run/test-scan-Room/…` came from the test suite's fixture scan name). Callers
# that write here call ensure_dirs() first.

# --- the LIVE room/preview (DYNAMIC — repointed per iteration by set_room) --- #
# These module globals are reassigned by set_room(); every harness module reads them
# as ``config.X`` at call time, so the reassignment is picked up everywhere.
ROOM_DIR = RENDER_DIR = ROOM_PY = ROOM_MD = OBJECT_PROVENANCE = None
RENDER_MANIFEST = FULL_MANIFEST = ROOM_GLB = None


def set_room(room_dir, preview_dir) -> None:
    """Repoint the LIVE room definition + render dir at THIS iteration, and export the
    matching env so the room-ops CLIs and render-tool subprocesses follow.

    ``room_dir``    = scene_stage/stage_<N>/iteration_<M>/room   (Room.py the agent edits)
    ``preview_dir`` = scene_stage/stage_<N>/iteration_<M>/room_preview  (renders + room.glb)
    """
    global ROOM_DIR, RENDER_DIR, ROOM_PY, ROOM_MD, OBJECT_PROVENANCE
    global RENDER_MANIFEST, FULL_MANIFEST, ROOM_GLB
    ROOM_DIR = Path(room_dir)
    RENDER_DIR = Path(preview_dir)
    ROOM_PY = ROOM_DIR / "Room.py"
    ROOM_MD = ROOM_DIR / "Room.md"
    OBJECT_PROVENANCE = ROOM_DIR / "Objects"
    RENDER_MANIFEST = RENDER_DIR / "render_manifest.json"
    FULL_MANIFEST = RENDER_DIR / "full_manifest.json"
    ROOM_GLB = RENDER_DIR / "room.glb"
    # render_room_cameras.py execs $SB_ROOM_PY as the scene + renders into $SB_RENDER_DIR;
    # room_ops.paths.room_dir/room_preview_dir read the LITEREALITY_* overrides.
    os.environ["SB_ROOM_PY"] = str(ROOM_PY)
    os.environ["SB_RENDER_DIR"] = str(RENDER_DIR)
    os.environ["LITEREALITY_ROOM_DIR"] = str(ROOM_DIR)
    os.environ["LITEREALITY_ROOM_PREVIEW"] = str(RENDER_DIR)


# Initialize pure path values without exporting environment variables at import time.
ROOM_DIR = ROOM_INIT / "room"
RENDER_DIR = ROOM_INIT / "room_preview"
ROOM_PY = ROOM_DIR / "Room.py"
ROOM_MD = ROOM_DIR / "Room.md"
OBJECT_PROVENANCE = ROOM_DIR / "Objects"
RENDER_MANIFEST = RENDER_DIR / "render_manifest.json"
FULL_MANIFEST = RENDER_DIR / "full_manifest.json"
ROOM_GLB = RENDER_DIR / "room.glb"


def ensure_dirs() -> None:
    """Create the per-scan scratch this harness writes into.

    Separate from import on purpose — see the note by HARNESS_OUT. Cheap and idempotent, so
    anything about to write calls it rather than assuming.
    """
    HARNESS_OUT.mkdir(parents=True, exist_ok=True)
    if RENDER_DIR is not None:
        RENDER_DIR.mkdir(parents=True, exist_ok=True)

# tools — all inside the SDK now (render + view selection consolidated here)
RENDER_TOOL = SDK / "room_render" / "render_room_cameras.py"
ANNOTATE_TOOL = SDK / "room_render" / "annotate_views.py"
SELECT_TOOL = TOOLS / "select_views" / "source" / "select_views.py"
RANK_TOOL = SDK / "room_render" / "rank_views.py"  # moved into the SDK

# stage 1 — rectified head-on "ortho" reference per surface (the agent's material
# ground truth). The SDK tool stitches walls; surface_reference.py extends it to the
# floor + ceiling and adds triage. Outputs land in SURFACE_REF (per-scan scratch,
# shared across iterations — it is ground truth, not iteration state).
STITCH_TOOL = TOOLS / "shared" / "stitch_wall_image" / "stitch_wall.py"
SURFACE_REF = Path(os.environ.get("LR_SURFACE_REF") or (AUTHORING / "surface_ref"))
SURFACE_REF_MANIFEST = SURFACE_REF / "surface_ref_manifest.json"
STITCH_PPM = int(os.environ.get("HARNESS_STITCH_PPM", "160"))  # ortho resolution (px/m)

# colour-adaptation: shift a fetched PBR's DIFFUSE to a target RGB in LAB space while
# PRESERVING the pattern (and the untouched roughness/normal stay physical). This is how
# a surface gets the right colour without ever degrading to a flat colour-only material.
RECOLOR_TOOL = PACKAGE_ROOT / "room_ops" / "compile" / "recolor.py"


def blender() -> str:
    """Delegates to the canonical room-ops Blender resolver; see the note about the six
    copies that disagreed about macOS."""
    from lrauthor.room_ops.paths import find_blender as _canonical

    return _canonical()

PY = os.environ.get("HARNESS_PY") or sys.executable

# --- knobs ------------------------------------------------------------------ #
# Stage worker — the EDITOR that edits Room.py (and the planner falls back to this).
# Opus 4.8. Override with $HARNESS_MODEL.
MODEL = SETTINGS.harness_model

MAX_VIEWS = int(os.environ.get("HARNESS_VIEWS", "10"))
SMALL_WALL_PCT = float(os.environ.get("HARNESS_SMALL_WALL_PCT", "12"))
VARY_VIEWS = os.environ.get("HARNESS_VARY_VIEWS", "1") != "0"
MAX_ROUNDS = int(os.environ.get("HARNESS_ROUNDS", "5"))
TURNS_PER_ROUND = int(os.environ.get("HARNESS_TURNS_PER_ROUND", "16"))
MAX_TURNS = MAX_ROUNDS * TURNS_PER_ROUND
MAX_BUDGET_USD = float(os.environ.get("HARNESS_BUDGET", "8"))
MAX_REFINE = int(os.environ.get("HARNESS_REFINE", "2"))
SSIM_SOFT_FLOOR = float(os.environ.get("HARNESS_SSIM", "0.45"))

ALLOWED_TOOLS = ["Bash", "Read", "Write", "Edit", "Glob", "Grep", "TodoWrite", "WebFetch"]
