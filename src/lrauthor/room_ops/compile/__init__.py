"""Room-ops compilation — editable Room.py definition → built Room.glb.

The room compiler. ``build_from_room`` materializes each object.py in Blender and assembles the
SHELL into ``Room.glb`` (+ ``Room.blend``); ``bake_glb`` flattens the SHELL's node-graph
materials so the exported glb matches the render. Helpers (``pack_assets``, ``fetch_textures``,
``texture_recipe``, ``recolor``) prepare the per-object inputs and material recipes.

The public entry is ``room_ops.compile_room(room_dir)`` — it wraps both steps.
"""
