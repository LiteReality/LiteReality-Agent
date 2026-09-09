"""CLI for the image-render-tool — four modes.

  # frame side-by-side (render | real), annotated on BOTH sides:
  python -m lrauthor.agent.tools.render.source scene  --room R --frames 10,20        # all objects
  python -m lrauthor.agent.tools.render.source wall   --room R --frames 10,20        # walls only
  python -m lrauthor.agent.tools.render.source wall-focus --room R --frames 10 --wall Wall2   # ONE wall
  python -m lrauthor.agent.tools.render.source object --room R --frames 10 --object Table0

  # wall reference (ortho render | stitch + wall-focused ref frames):
  python -m lrauthor.agent.tools.render.source ref    --room R --walls Wall0,Wall2
"""

from __future__ import annotations

import argparse

from .compose import (
    render_object,
    render_scene,
    render_wall,
    render_wall_focus,
    render_wall_reference,
)


def _frames(spec: str):
    return spec if ("-" in spec or spec == "all") else [int(x) for x in spec.split(",")]


def main() -> int:
    ap = argparse.ArgumentParser(prog="image_render_tool", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("scene", "wall", "wall-focus", "object"):
        p = sub.add_parser(name, help=f"{name} mode: render | real, per frame")
        p.add_argument("--room", required=True)
        p.add_argument("--frames", required=True, help="'10,20' | '0-40' | 'all'")
        p.add_argument("--scan", default=None)
        p.add_argument("--out", default=None)
        if name == "object":
            p.add_argument("--object", required=True, help="object id, e.g. Table0")
        if name == "wall-focus":
            p.add_argument("--wall", required=True, help="the ONE wall to highlight, e.g. Wall2")
    pr = sub.add_parser("ref", help="wall reference: ortho render | stitch + ref frames")
    pr.add_argument("--room", required=True)
    pr.add_argument("--walls", default=None, help="'Wall0,Wall2' or omit for all")
    pr.add_argument("--scan", default=None)
    pr.add_argument("--out", default=None)

    a = ap.parse_args()
    if a.cmd == "scene":
        res = render_scene(a.room, _frames(a.frames), scan=a.scan, out_dir=a.out)
    elif a.cmd == "wall":
        res = render_wall(a.room, _frames(a.frames), scan=a.scan, out_dir=a.out)
    elif a.cmd == "wall-focus":
        res = render_wall_focus(a.room, _frames(a.frames), a.wall, scan=a.scan, out_dir=a.out)
    elif a.cmd == "object":
        res = render_object(a.room, _frames(a.frames), a.object, scan=a.scan, out_dir=a.out)
    else:
        res = render_wall_reference(a.room, a.walls, scan=a.scan, out_dir=a.out)
    print(f"[{a.cmd}] {len(res)} image(s):")
    for k, v in res.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
