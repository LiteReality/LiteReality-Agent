"""Turn what init just produced into a SCENE PACKAGE — the self-describing folder stage 2 needs.

`scene_init` exports the seed `Room.py`; this finalizes the *folder around it* so nothing
downstream has to be told where anything is. It is the last thing `init_scene` does, and the
reason `uv run litereality run <dir> --from evidence` can start authoring with no
scan name, no `$LR_SCANS_DIR`, and no per-stage path flags.

What it writes is described by :mod:`litereality_agent.room_ops.manifest` (layer 0, stdlib only); everything
scan-specific — which output root, which final root, where the capture actually is — is resolved
HERE, where the pipeline and scene path contracts meet.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from litereality_agent.room_ops import manifest

# Stage 1 writes everything under this one folder, named for the package that produces it.
STAGE_ROOT = "scene_init"


def scene_root(scan: str) -> Path:
    """The package root for `scan` — the deliverables tree the CLI already collects into."""
    from litereality_agent.pipeline.measure import paths as oi_config

    return oi_config.final_root() / scan


def ensure_stage_links(scan: str) -> Path:
    """Make the two spellings of each stage dir agree, and return the package root.

    **Call this BEFORE anything writes or reads a stage dir** — `init_scene` does, first thing.
    The object half writes via object_init.config.obj_stage_dir while the room export reads via
    scene paths. Both resolve to `run/<scan>/scene_init/<stage>`, but the two roots
    ($LITEREALITY_FINAL and $LITEREALITY_OUTPUT) can still differ. Without links, export can look in the other
    root, found nothing, and returned None — a whole reconstruction thrown away at the last step,
    reported only as "export_initial_scene returned None".

    The links are therefore created here: the deliverables side is the real directory, the staging side a symlink at
    it. An existing real directory on either side is left strictly alone — a tree from an older
    layout still resolves, it is only less portable.
    """
    from litereality_agent.pipeline.measure import paths as oi_config

    root = scene_root(scan)
    out_scan = oi_config.output_root() / scan
    # Both halves sit under stage 1's own root, so the link is made one level down.
    for stage in (f"{STAGE_ROOT}/obj_stage", f"{STAGE_ROOT}/scene_stage"):
        final_side, out_side = root / stage, out_scan / stage
        try:
            if out_side.is_dir() and not out_side.is_symlink():
                # A real directory under output/ predates the link; adopt it as the target rather
                # than moving data, so both spellings still land on the same tree.
                if not final_side.exists() and not final_side.is_symlink():
                    final_side.parent.mkdir(parents=True, exist_ok=True)
                    final_side.symlink_to(os.path.abspath(out_side), target_is_directory=True)
                continue
            final_side.mkdir(parents=True, exist_ok=True)
            if not out_side.exists() and not out_side.is_symlink():
                out_side.parent.mkdir(parents=True, exist_ok=True)
                out_side.symlink_to(final_side, target_is_directory=True)
        except OSError:
            pass  # a package with absolute paths still works; it is only less portable
    return root


def _capture_source(scan: str) -> Path:
    """Where this scan's RoomPlan capture actually is.

    `$LR_SCANS_DIR/<scan>` is the contract every other module resolves against, so it wins;
    The scene path resolver is the fallback for trees where the capture was staged into the
    object_init input dir instead.
    """
    from litereality_agent.pipeline.measure import paths as oi_config

    candidate = oi_config.SCANS_ROOT / scan
    if candidate.is_dir():
        return candidate
    from litereality_agent.room_ops import paths as scene_paths

    return scene_paths.resolve_scan_dir(scan)


def _objects_in(recon: Path) -> list[str]:
    """Names of the reconstructed assets: `<name>.glb` (TRELLIS) and `<name>/` (procedural)."""
    if not recon.is_dir():
        return []
    flat = {p.stem for p in recon.glob("*.glb")}
    nested = {p.name for p in recon.iterdir() if p.is_dir()}
    return sorted(flat | nested)


def finalize(
    scan: str,
    *,
    capture_mode: str = "link",
    init: dict[str, Any] | None = None,
) -> manifest.ScenePackage:
    """Write `run/<scan>/scene.json` and return the package.

    `capture_mode` — `link` (default; a symlink, so the package is launchable but cheap), `copy`
    (real bytes, so the folder can be moved to another machine), or `reference` (record the
    absolute source path and embed nothing).
    """
    from litereality_agent.pipeline.measure import paths as oi_config
    from litereality_agent.room_ops import paths as scene_paths

    root = ensure_stage_links(scan)

    # Prefer the in-package spelling of each stage dir (after _link_stage_dirs, both spellings
    # point at the same tree) so the manifest records a RELATIVE path and the folder stays movable.
    obj_stage = oi_config.obj_stage_dir(scan)
    scene_stage = root / STAGE_ROOT / "scene_stage"
    if not scene_stage.exists() and not scene_stage.is_symlink():
        scene_stage = scene_paths.scene_stage_dir(scan)
    room_init = scene_stage / "room_init"
    harness = scene_stage / "_harness"
    # Stage 2 writes everything under one directory of its own, named for the command that makes
    # it. Stage 1 owns obj_stage/ and the seed in scene_stage/; keeping stage 2's work out of
    # both is what lets you delete and re-run one half without touching the other.
    authoring = root / "realism_authoring"

    capture = manifest.embed_capture(root, _capture_source(scan), scan, capture_mode)

    paths = {
        "obj_stage": obj_stage,
        "refroot": obj_stage / "object_init",
        "reconstructed_objs": obj_stage / "reconstructed_objs",
        "traces": obj_stage / "traces",
        "scene_stage": scene_stage,
        "room_init": room_init / "room",
        "room_init_preview": room_init / "room_preview",
        "room_py": room_init / "room" / "Room.py",
        "room_glb": room_init / "room_preview" / "Room.glb",
        "assets": scene_stage / "_scene_assets",
        "harness": harness,
        "surface_ref": authoring / "surface_ref",
        "authoring": authoring,
        "authoring_room": authoring / "room",
        "authoring_preview": authoring / "room_preview",
        "authoring_scratch": authoring / "_scratch",
        "capture": capture,
    }
    roots = {
        "output": oi_config.output_root(),
        "final": oi_config.final_root(),
        "scans": oi_config.SCANS_ROOT,
    }

    objects = _objects_in(paths["reconstructed_objs"])
    manifest.write(
        root,
        scan=scan,
        paths=paths,
        roots=roots,
        capture={
            "mode": capture_mode,
            "source": str(_capture_source(scan)),
            **manifest.capture_stats(capture),
        },
        init={**(init or {}), "objects": objects, "n_objects": len(objects)},
        providers={
            "image": "openai",
            "classify": os.environ.get("LR_CLASSIFY_PROVIDER", "claude"),
            "vlm": os.environ.get("HARNESS_VLM", "claude"),
        },
        models={
            k: v
            for k, v in {
                "harness": os.environ.get("HARNESS_MODEL"),
                "chair_judge": os.environ.get("LR_CHAIR_JUDGE_MODEL"),
                "procedural": os.environ.get("LR_PROCEDURAL_MODEL"),
                "image": os.environ.get("LR_OPENAI_IMAGE_MODEL"),
            }.items()
            if v
        },
    )
    return manifest.read(root)
