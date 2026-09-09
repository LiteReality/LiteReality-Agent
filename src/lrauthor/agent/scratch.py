"""Keep the working images the model makes to LOOK at.

An authoring session does not only read the stitches we hand it — it *manufactures* evidence.
It shells out to PIL to crop a wall band, white-balance a frame, or tile two views side by
side, then reads the result back and decides something from it. Those images are the closest
thing we have to a record of what it was actually looking at when it made a call.

They were being written to `/tmp` and lost. `/tmp/f02.png`, `/tmp/f02_dado.png`,
`/tmp/f02_wb.png` — a crop of a dado line and its white-balanced twin — survived exactly as
long as the OS let them, and the trace recorded only the shell command that produced them.

Two mechanisms here, because one is not enough:

* `scratch_dir()` names a durable directory INSIDE the scene package and the prompt points the
  model at it. This is the cheap path — the image is simply written where we can find it.
* `rescue()` is the safety net for when it ignores that and writes to `/tmp` anyway (it is a
  strong habit, and a prompt is a request, not a constraint). Any image path a tool touched
  that lands outside the workspace is copied in at RESULT time — the first moment a file the
  tool created is guaranteed to exist.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

# Anything an agent might make and read back. Deliberately narrow: this copies files, so a
# loose pattern would start hoovering up meshes and logs.
_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})
# Paths as they appear inside a shell command or a tool result. Absolute only — a bare name has
# no anchor we can trust, and guessing a root is how images get copied from the wrong place.
_PATH_RE = re.compile(r"(/[\w./+-]+\.(?:png|jpe?g|webp))")
# Big enough to hold a montage, small enough that a runaway loop cannot fill the disk.
_MAX_BYTES = 40 * 1024 * 1024
_MAX_FILES = 400


def scratch_dir(create: bool = True, near: Path | None = None) -> Path | None:
    """The durable scratch directory for this run, or None when nothing anchors it.

    `$LR_SCRATCH` comes from the scene package (`manifest.stage_env`), so this is
    `run/<scan>/realism_authoring/_scratch` for a the CLI run and follows `$RUN_TAG` with it.
    `apply_env` exports only the `$LITEREALITY_*` contract, though, so a stage launched
    directly with `--scene` has neither variable — hence `near`: pass the work room and the
    scratch lands beside it, which is the same place either way.
    """
    raw = os.environ.get("LR_SCRATCH") or ""
    if not raw:
        authoring = os.environ.get("LR_AUTHORING") or ""
        if authoring:
            raw = str(Path(authoring) / "_scratch")
        elif near is not None:
            raw = str(Path(near).parent / "_scratch")
        else:
            return None
    path = Path(raw).expanduser()
    if create:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            return None
    return path


_RUN_DIR_RE = re.compile(r"^run_(\d+)$")


def scratch_root(near: Path | None = None) -> Path | None:
    """The `_scratch` base that holds every run's directory.

    `$LR_SCRATCH` may already point at a specific run (this process re-binding, or a child
    inheriting the export), so normalise back to the base rather than nesting `run_002/run_003`.
    """
    where = scratch_dir(create=False, near=near)
    if where is None:
        return None
    return where.parent if _RUN_DIR_RE.match(where.name) else where


def bind(near: Path | None = None) -> Path | None:
    """Allocate THIS run its own scratch directory and export it.

    One shared `_scratch` meant each run silently overwrote the last: the model reuses names
    (`f02.png`, `ceil_overlay.png`) every session, so yesterday's evidence was gone and the
    images no longer matched the trace sitting beside them. Runs are numbered instead —
    `_scratch/run_001`, `run_002`, … — with `latest` pointing at the newest.

    Numbered subdirectories rather than sibling `_scratch2`/`_scratch3`: after thirty runs the
    sibling form buries `realism_authoring/`, and grouping them under one base keeps a single
    thing to glob, browse, or delete.

    Exported rather than returned-and-threaded: `rescue()` runs deep in the message loop and
    `prompt_line()` at setup, and they must not disagree about where scratch is.
    """
    base = scratch_root(near=near)
    if base is None:
        return None
    used = [int(m.group(1)) for p in base.glob("run_*")
            if p.is_dir() and (m := _RUN_DIR_RE.match(p.name))]
    where = base / f"run_{max(used, default=0) + 1:03d}"
    try:
        where.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    os.environ["LR_SCRATCH"] = str(where)
    os.environ["LR_RUN_ID"] = where.name  # the trace archives itself under the same id
    _point_latest(base, where)
    return where


def collect_strays(where: Path | None = None) -> list[Path]:
    """Move anything written to the `_scratch` ROOT into this run's directory.

    The prompt names the run directory exactly and the model still writes to the parent — it
    shortens the path. Observed: a session told to use `_scratch/run_002` left `Wall0_cmp.jpg`,
    `Wall1_cmp.jpg` and a `jview.py` helper directly in `_scratch/`, where the next run's files
    would sit beside them with no way to tell the two apart. Asking harder does not fix that;
    sweeping does. Loose files at the root belong to whichever run was active.
    """
    where = where or scratch_dir(create=False)
    if where is None or not _RUN_DIR_RE.match(where.name):
        return []
    base = where.parent
    moved: list[Path] = []
    try:
        entries = list(base.iterdir())
    except OSError:
        return []
    for item in entries:
        if item.is_dir() or item.is_symlink() or _RUN_DIR_RE.match(item.name):
            continue  # run dirs and `latest` are structure, not strays
        try:
            where.mkdir(parents=True, exist_ok=True)
            dest = where / item.name
            n = 2
            while dest.exists() and n < 1000:
                dest = where / f"{item.stem}_{n}{item.suffix}"
                n += 1
            item.replace(dest)
            moved.append(dest)
        except OSError:
            continue
    return moved


def finish(where: Path | None = None) -> None:
    """End-of-session tidy: claim strays, then drop the run dir if it stayed empty.

    Must run even when the session raises — a turn cap, a budget stop and an SDK error all
    exit through the exception path, and that is precisely when the empty directories pile up.
    """
    where = where or scratch_dir(create=False)
    collect_strays(where)
    prune_if_empty(where)


def prune_if_empty(where: Path | None = None) -> bool:
    """Remove this run's scratch dir if it produced nothing. Returns True if removed.

    A short run that only reads stitches makes no images, and one empty `run_NNN` per attempt
    is exactly the clutter that per-run directories were meant to avoid. Call at the end of a
    session; the next `bind()` simply reuses the number.
    """
    where = where or scratch_dir(create=False)
    if where is None or not where.is_dir() or not _RUN_DIR_RE.match(where.name):
        return False
    try:
        if any(where.rglob("*")):
            return False
        where.rmdir()
    except OSError:
        return False
    base = where.parent
    link = base / "latest"
    # `latest` now dangles; repoint it at the newest surviving run, or drop it.
    try:
        if link.is_symlink() and not link.resolve().exists():
            link.unlink()
            survivors = sorted(p for p in base.glob("run_*") if p.is_dir())
            if survivors:
                link.symlink_to(survivors[-1].name)
    except OSError:
        pass
    return True


def _point_latest(base: Path, where: Path) -> None:
    """`_scratch/latest` → the newest run. Convenience only; never fatal."""
    link = base / "latest"
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(where.name)
    except OSError:
        pass


def prompt_line() -> str:
    """The instruction to splice into a stage prompt. Empty when there is nowhere to point."""
    where = scratch_dir()
    if where is None:
        return ""
    return (
        f"\nSCRATCH IMAGES: when you crop, white-balance, tile or otherwise MAKE an image to look\n"
        f"at, write it under `{where}` — NOT /tmp. That directory is kept with the scene, so the\n"
        f"evidence behind your decisions stays reviewable. Name files for what they show\n"
        f"(`Wall2_dado_crop.png`, not `f02.png`).\n"
    )


def _run_root() -> Path | None:
    """The scene package root — everything under it is already durable.

    `$LR_SCENE` is the package's own contract and is exact. Without it, derive from the scratch
    dir: `<scene>/realism_authoring/_scratch` → two levels up.
    """
    scene = os.environ.get("LR_SCENE")
    if scene:
        return Path(scene).expanduser()
    where = scratch_dir(create=False)
    return where.parent.parent if where is not None else None


def _candidates(*sources) -> list[Path]:
    """Absolute image paths mentioned across tool input/result text."""
    found: list[Path] = []
    for src in sources:
        if src is None:
            continue
        if isinstance(src, dict):
            text = " ".join(str(v) for v in src.values())
        elif isinstance(src, (list, tuple)):
            text = " ".join(str(v) for v in src)
        else:
            text = str(src)
        for match in _PATH_RE.findall(text):
            path = Path(match)
            if path.suffix.lower() in _IMAGE_SUFFIXES:
                found.append(path)
    return found


def rescue(*sources, dest: Path | None = None) -> list[Path]:
    """Copy any out-of-workspace image referenced in `sources` into `<scratch>/rescued/`.

    Call at tool-RESULT time: a `Bash` that creates an image has not run when its use block
    arrives, so scanning the input any earlier finds nothing on disk.

    Silent by design — losing a scratch image is a shame, but failing a paid authoring run
    because a copy did not work would be worse.
    """
    if dest is None:
        base = scratch_dir()
        if base is None:
            return []
        dest = base / "rescued"
    # "Already safe" means anywhere in the SCENE PACKAGE, not just the authoring subtree. Using
    # the authoring dir as the boundary copied every capture frame the model opened — those live
    # in `<scene>/capture/` and are as durable as the scratch dir itself.
    root = _run_root()
    copied: list[Path] = []
    seen: set[str] = set()
    for src in _candidates(*sources):
        key = str(src)
        if key in seen:
            continue
        seen.add(key)
        try:
            if root is not None and src.is_relative_to(root):
                continue  # already inside the run — nothing to rescue
            if not src.is_file() or src.stat().st_size > _MAX_BYTES:
                continue
            dest.mkdir(parents=True, exist_ok=True)
            if sum(1 for _ in dest.iterdir()) >= _MAX_FILES:
                continue
            target = dest / src.name
            # A session reuses names (`f02.png` rewritten each crop); keep every version, since
            # which one it was looking at is exactly the thing worth knowing.
            if target.exists():
                stem, suffix = target.stem, target.suffix
                n = 2
                while target.exists() and n < 1000:
                    target = dest / f"{stem}_{n}{suffix}"
                    n += 1
            shutil.copy2(src, target)
            copied.append(target)
        except OSError:
            continue
    return copied
