"""Turn a room scan into an editable, realistic 3D program.

The package has five feature-oriented areas: ``pipeline`` coordinates the scene-init and realism-
authoring phases, ``agent`` owns extensible agent behavior, ``room_ops`` owns the portable room
representation, ``models`` owns inference, and ``runtimes`` owns execution transports. The
supported entrypoint is ``uv run litereality``.

## The two roots

Code and data live in different places, and conflating them is the classic src-layout bug:

    PACKAGE_ROOT   src/lrauthor — where the CODE is. Use for anything shipped in the wheel:
                   a prompt template, a Blender helper script, a tool a subprocess must exec.
    REPO_ROOT      the checkout — where the DATA is: run/, scans_uploaded/, and .env.
                   Override with $LR_REPO_ROOT when the package is installed
                   somewhere other than beside its data.

Reach for `REPO_ROOT` when you mean "where this user's scans and results live" and `PACKAGE_ROOT`
when you mean "where our own files are". They are the same directory in a source checkout, which
is exactly why the distinction has to be made deliberately rather than discovered later.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["PACKAGE_ROOT", "SRC_ROOT", "REPO_ROOT", "repo_root"]

PACKAGE_ROOT = Path(__file__).resolve().parent  # src/lrauthor
SRC_ROOT = PACKAGE_ROOT.parent  # src


def repo_root() -> Path:
    """The checkout holding run/ · scans_uploaded/ · .env.

    Resolved fresh on each call so `$LR_REPO_ROOT` can be set after import (the launchers and
    Blender subprocesses set it on the way in). `REPO_ROOT` below is the import-time snapshot,
    which is what almost every call site wants.
    """
    env = os.environ.get("LR_REPO_ROOT")
    return Path(env).expanduser().resolve() if env else SRC_ROOT.parent


REPO_ROOT = repo_root()
