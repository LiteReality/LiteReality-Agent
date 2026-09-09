"""image_render_tool — build render-vs-real comparison images on demand.

Frame side-by-side (annotates BOTH render + real, which share the camera pose):
  render_scene(room, frames)          # every object labelled (chips)
  render_wall(room, frames)           # walls outlined + named
  render_wall_focus(room, frames, w)  # ONLY wall `w` outlined + named (focus)
  render_object(room, frames, obj)    # a bbox on ONE object + its name

Wall reference:
  render_wall_reference(room, walls)  # ortho render | stitch, + wall-focused ref frames

    from lrauthor.agent.tools.render.source import render_scene, render_wall, render_wall_focus, render_object, render_wall_reference
"""

from .compose import (
    render_frames,  # back-compat alias of render_scene
    render_object,
    render_scene,
    render_surface,  # back-compat alias of render_wall_reference
    render_wall,
    render_wall_focus,
    render_wall_reference,
    render_walls_focus_batch,
)

__all__ = [
    "render_walls_focus_batch",
    "render_scene",
    "render_wall",
    "render_wall_focus",
    "render_object",
    "render_wall_reference",
    "render_frames",
    "render_surface",
]
