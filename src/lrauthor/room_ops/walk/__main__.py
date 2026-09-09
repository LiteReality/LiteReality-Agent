"""Walk any compiled room: `python -m lrauthor.room_ops.walk <Room.glb>`.

Takes a glb rather than a run, so it works on a room folder that never went through the pipeline.
`litereality view <scan>` is the same thing with the run tree resolved for you.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lrauthor.room_ops import walk


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="walk a compiled room in the browser")
    ap.add_argument("glb", type=Path, help="a compiled Room.glb")
    ap.add_argument("--label", default=None, help="name shown in the panel (default: the filename)")
    ap.add_argument(
        "--port", type=int, default=8780,
        help="where to start looking for a free port; taken ones are skipped",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-open", dest="open_browser", action="store_false")
    args = ap.parse_args(argv)
    if not args.glb.is_file():
        raise SystemExit(f"no such room: {args.glb}")
    return walk.serve(args.glb, args.label or args.glb.stem, port=args.port, host=args.host,
                      open_browser=args.open_browser)


if __name__ == "__main__":
    raise SystemExit(main())
