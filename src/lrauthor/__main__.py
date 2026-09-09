"""Legacy module entrypoint; prefer ``uv run litereality``.

The console script and this compatibility module both dispatch to :mod:`lrauthor.cli`.
"""

from __future__ import annotations

from lrauthor.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
