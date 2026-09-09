"""Paths for assembling the room format from reconstruction outputs.

Everything is rooted at the same single output tree object_init uses, grouped into
two top-level stages:

    run/<scan>/
        scene_init/                      # STAGE 1 — everything the deterministic stage produces
            obj_stage/                   # the per-object half
                object_init/             #   routing + chair groups + scene_data
                reconstructed_objs/      #   generated GLBs consumed by room export
                    <name>.glb           #     flat = TRELLIS (neural) static objects/chairs
                    <name>/<name>.glb    #     dir  = procedural/articulated (has object.py)
            scene_stage/                 # the room-build half
                _scene_assets/           #   flat packed glbs + manifest (build_room input)
                room_init/               #   the baseline white-shell room
                    room/                #     ★ the room DEFINITION (Room.py + SHELL + Objects/)
                    room_preview/        #     ★ the BUILD (Room.glb + room_layout.json)
                stage_<N>/iteration_<M>/ #   each harness pass (room/ + room_preview/ + .harness_tracking/)

The LIVE room/preview default to ``room_init`` but the harness repoints them at the
current iteration via ``$LITEREALITY_ROOM_DIR`` / ``$LITEREALITY_ROOM_PREVIEW`` (so the
stage-2 CLIs follow with no call-site changes). Override the output root with
``$LITEREALITY_OUTPUT``. Blender via ``$BLENDER`` / ``$LITEREALITY_BLENDER``.
"""

from __future__ import annotations

import os
from pathlib import Path

from lrauthor import PACKAGE_ROOT, REPO_ROOT
from lrauthor.settings import load_settings

HERE = Path(__file__).resolve().parent
# REPO_ROOT is the CHECKOUT (output/, scans_uploaded/, .env) — not this package. PACKAGE_ROOT is
# where our own files live; the two coincide in a source checkout, so keep them distinct here.
AGENT_DIR = PACKAGE_ROOT  # back-compat alias, used for paths to OUR tools


def output_root() -> Path:
    return load_settings().resolved_output_root()


def scan_name(value: str) -> str:
    """Accept a scan NAME or a path to the scan folder; return the bare name."""
    return Path(str(value).rstrip("/")).name


def scan_output_dir(scan: str) -> Path:
    return output_root() / scan


# --- stage 1: scene_init ----------------------------------------------------- #
# Everything stage 1 produces lives under one folder, `scene_init/`, named after the package that
# preserves its historical artifact name. Inside it are the seed's two halves:
#   obj_stage/    the per-object work (object_init + the reconstructed GLBs)
#   scene_stage/  the room build (baseline room_init + each harness stage's iterations)
def scene_init_dir(scan: str) -> Path:
    """Stage 1's output root for a scan — the parent of both stage halves."""
    return scan_output_dir(scan) / "scene_init"


def obj_stage_dir(scan: str) -> Path:
    return scene_init_dir(scan) / "obj_stage"


def scene_stage_dir(scan: str) -> Path:
    return scene_init_dir(scan) / "scene_stage"


def reconstruct_dir(scan: str) -> Path:
    return obj_stage_dir(scan) / "reconstructed_objs"


def object_init_dir(scan: str) -> Path:
    return obj_stage_dir(scan) / "object_init"


def scene_data_dir(scan: str) -> Path:
    return object_init_dir(scan) / "input" / "scene_data" / scan


def chair_clusters_json(scan: str) -> Path:
    return object_init_dir(scan) / "chair_clusters" / scan / "chair_clusters.json"


def room_init_dir(scan: str) -> Path:
    """The baseline white-shell room (definition + preview) the harness seeds from."""
    return scene_stage_dir(scan) / "room_init"


def room_dir(scan: str) -> Path:
    """The LIVE room definition Room.py lives in.

    Defaults to the baseline ``scene_stage/room_init/room``; the harness points it at
    the current ``scene_stage/stage_N/iteration_M/room`` via ``$LITEREALITY_ROOM_DIR``
    so the stage-2 CLIs (wall_nano/wall_objects/wall_mount) and build/render tools all
    follow the live iteration with no call-site changes.
    """
    env = os.environ.get("LITEREALITY_ROOM_DIR")
    return Path(env).expanduser().resolve() if env else (room_init_dir(scan) / "room")


def room_preview_dir(scan: str) -> Path:
    """The render/preview of the live room (sibling of room_dir). Overridable with
    ``$LITEREALITY_ROOM_PREVIEW`` (the harness sets it per iteration)."""
    env = os.environ.get("LITEREALITY_ROOM_PREVIEW")
    return Path(env).expanduser().resolve() if env else (room_init_dir(scan) / "room_preview")


def assets_dir(scan: str) -> Path:
    # staging area for build_room (flat glb/ + manifest); shared across iterations, so it
    # sits at the scene_stage root, NOT inside any room definition (so export's reset of
    # a room/ dir doesn't wipe it).
    return scene_stage_dir(scan) / "_scene_assets"


def scans_root() -> Path:
    """Where captures live. `$LR_SCANS_DIR` is the contract every other module resolves against."""
    return load_settings().resolved_scans_dir()


def resolve_scan_dir(scan: str) -> Path:
    """The raw RoomPlan capture dir (room.usdz + frame_*) for ``scan``.

    Honours `$LR_SCANS_DIR`. It used to hardcode `<repo>/scans_uploaded`, so a capture given as a
    folder — `scene_init example_scans/Office_room`, which points $LR_SCANS_DIR at its parent —
    was never found. That does not fail loudly: the room assembles without the capture's ARKit
    cameras and comes out "geometry only", a quietly worse build.
    """
    for cand in (
        scans_root() / scan,
        REPO_ROOT / "scans_uploaded" / scan,  # the historical location, still valid
        object_init_dir(scan) / "input" / "usdz_files",
    ):
        if (cand / "room.usdz").exists() or (cand / "roomplan" / "room.usdz").exists():
            return cand
        # object_init stages the capture's usdz as <scan>.usdz rather than room.usdz
        if (cand / f"{scan}.usdz").exists():
            return cand
    return scans_root() / scan


def resolve_usdz(scan: str) -> Path:
    d = resolve_scan_dir(scan)
    for cand in (
        d / "room.usdz",
        d / "roomplan" / "room.usdz",
        object_init_dir(scan) / "input" / "usdz_files" / f"{scan}.usdz",
    ):
        if cand.exists():
            return cand
    return d / "room.usdz"


def find_blender() -> str:
    """Locate the Blender binary. THE resolver — everything else delegates here.

    There used to be six copies of this, and only the authoring one knew that a normal macOS
    install puts the binary inside `/Applications/Blender.app/Contents/MacOS/`. So on a stock Mac
    with Blender installed and `$BLENDER_PATH` blank, stage 2 found Blender and stage 1 did not:
    `scene_init` packed every object, then died reading the RoomPlan usdz, and left a room
    directory with no `Room.py` in it.

    Accepts either the binary or the install directory (the README documents
    `$LITEREALITY_BLENDER` as the directory), and tries the platform's usual home last.
    """
    import shutil as _sh

    configured = load_settings().blender
    candidates = [
        str(configured) if configured else None,
        _sh.which("blender"),
        "/Applications/Blender.app/Contents/MacOS/Blender",  # stock macOS install
        "/usr/local/bin/blender",
        "/opt/blender/blender",
    ]
    for c in candidates:
        if not c:
            continue
        p = Path(c)
        if p.is_dir():  # an install dir: the binary is inside it
            p = p / ("Blender" if (p / "Blender").is_file() else "blender")
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
    raise SystemExit(
        "Blender not found. Set BLENDER_PATH in .env to your Blender install directory "
        "(the folder containing the `blender` binary), or $BLENDER to the binary itself.\n"
        "  macOS default: /Applications/Blender.app/Contents/MacOS\n"
        "  checked: " + ", ".join(str(c) for c in candidates if c)
    )

def wall_cache_dir() -> Path:
    """Flat cache for nano intermediate images/JSON (files namespaced by scan)."""
    d = output_root() / "_wall_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def wall_fixtures_json(scan: str) -> Path:
    """Detection output, written next to Room.py so the placement layer finds it."""
    return room_dir(scan) / "nano_fixtures.json"


def wall_src_dir(scan: str) -> Path:
    """Per-object procedural object.py sources for wall fixtures."""
    return room_dir(scan) / "Objects" / "WallMounted"


def wall_glb_dir(scan: str) -> Path:
    """Compiled wall-fixture GLBs (never under room/; the placement layer imports these)."""
    return room_preview_dir(scan) / "wall_objects"


def room_layout_json(scan: str) -> Path:
    """Camera intrinsics/extrinsics from the build (room_preview/room_layout.json)."""
    p = room_preview_dir(scan) / "room_layout.json"
    if p.is_file():
        return p
    return room_preview_dir(scan) / "capture" / "room_layout.json"
