#!/usr/bin/env python3
"""bake_glb.py — make a room's exported glb FAITHFUL to its Blender materials.

The shell materials (Floor/Wall/Ceiling/dado) are authored in Room.py as Blender
node graphs: a CC0 texture MIXED with a solid tint (`base_color_srgb` +
`tint_strength`) and driven by OBJECT-coordinate projection. glTF can only store
``baseColorTexture × baseColorFactor`` with UV coords, so the exporter DROPS the
tint and the projection — the glb ends up with the raw, un-tinted PolyHaven texture
(e.g. a cream houndstooth instead of the blue carpet you see in the render).

This pass fixes that: it opens the assembled ``Room.blend``, BAKES each shell
material's node graph (albedo, i.e. the DIFFUSE colour pass — no lighting) into a
flat per-object UV texture, rewires that material to the baked image, and re-exports
the glb. Furniture/opening materials already ship baked image textures from their
object.py, so they are left untouched.

    blender -b Room.blend --python bake_glb.py -- <out.glb> [res]

Run via :func:`bake_room_glb` (spawns Blender) from the room-ops API.
"""

from __future__ import annotations

import os
import sys

# meshes whose material is a node-tint shell surface (the ones glTF can't export)
SHELL_PREFIXES = ("Floor", "Ceiling", "Wall")


def _is_shell(name: str) -> bool:
    return any(name.startswith(p) for p in SHELL_PREFIXES)


def _bake_in_blender(out_glb: str, res: int) -> None:
    import bpy

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    # GPU if the build allows it, else CPU — albedo bake is cheap (few samples).
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.get_devices()
        if any(d.type != "CPU" for d in prefs.devices):
            scene.cycles.device = "GPU"
    except Exception:
        scene.cycles.device = "CPU"
    scene.cycles.samples = 4
    bake = scene.render.bake
    bake.use_pass_direct = False
    bake.use_pass_indirect = False
    bake.use_pass_color = True  # DIFFUSE colour pass = pure albedo (no lighting)
    bake.margin = 6

    shell = [
        o for o in bpy.data.objects if o.type == "MESH" and o.data.materials and _is_shell(o.name)
    ]
    print(f"  baking {len(shell)} shell mesh(es): {[o.name for o in shell]}")

    for o in shell:
        me = o.data
        # Existing UV layers can be raw WORLD-coordinate dumps (procedural tiling writes metres
        # into UVs) — far outside 0-1, so baking through them writes NOTHING (empty/patchy glb
        # walls). Always bake into a FRESH smart-projected layer: `uv_layers.active` is the bake
        # WRITE target, while the material keeps sampling its own coords (Object/original UVs).
        bake_uv = me.uv_layers.new(name="BakeUV")
        me.uv_layers.active = bake_uv
        bpy.context.view_layer.objects.active = o
        bpy.ops.object.select_all(action="DESELECT")
        o.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project(angle_limit=1.15, island_margin=0.02)
        bpy.ops.object.mode_set(mode="OBJECT")

        # Materials are often SHARED across walls (one Wall_Paint_Cream on all of them). We bake
        # and rewire per OBJECT, which mutates the material — after the first wall is rewired the
        # shared node graph is gone and every later wall bakes black. Make each slot single-user.
        for i, mat in enumerate(me.materials):
            if mat is not None and mat.users > 1:
                me.materials[i] = mat.copy()

        # Per material slot: THREE bake targets (albedo / roughness / normal) so the glb keeps the
        # material's CHARACTER — a flat roughness+no-normal rewire reads plasticky vs the render.
        targets = []
        for mat in me.materials:
            if mat is None or not mat.use_nodes:
                targets.append(None)
                continue
            imgs = {}
            for kind in ("base", "rough", "norm"):
                img = bpy.data.images.new(f"bake_{o.name}_{mat.name}_{kind}", res, res, alpha=False)
                if kind != "base":
                    img.colorspace_settings.name = "Non-Color"
                imgs[kind] = img
            targets.append((mat, imgs))

        bpy.context.view_layer.objects.active = o
        bpy.ops.object.select_all(action="DESELECT")
        o.select_set(True)

        def _bake_into(kind, bake_type, **kw):
            # point every material's ACTIVE image node at this kind's image, then bake once
            for t in targets:
                if t is None:
                    continue
                mat, imgs = t
                nt = mat.node_tree
                node = nt.nodes.get("_bake_target")
                if node is None:
                    node = nt.nodes.new("ShaderNodeTexImage")
                    node.name = "_bake_target"
                node.image = imgs[kind]
                nt.nodes.active = node
            bpy.ops.object.bake(type=bake_type, margin=6, **kw)

        # NB: the bake OPERATOR ignores scene.render.bake.* (those only feed the UI button) — pass
        # pass_filter explicitly, or DIRECT+INDIRECT lighting gets baked into the albedo and the
        # glb walls come out black with shadow silhouettes burned in.
        _bake_into("base", "DIFFUSE", pass_filter={"COLOR"})
        _bake_into("rough", "ROUGHNESS")
        _bake_into("norm", "NORMAL")  # tangent-space by default

        # display/export must go through the SAME layer the bake wrote to
        me.uv_layers["BakeUV"].active_render = True

        # rewire each baked material to UV-mapped baseColor + roughness + normal — all three are
        # glTF-exportable (the exporter packs roughness into the metallicRoughness texture).
        for t in targets:
            if t is None:
                continue
            mat, imgs = t
            for img in imgs.values():
                img.pack()
            nt = mat.node_tree
            nt.nodes.clear()
            out = nt.nodes.new("ShaderNodeOutputMaterial")
            bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
            tb = nt.nodes.new("ShaderNodeTexImage")
            tb.image = imgs["base"]
            tr = nt.nodes.new("ShaderNodeTexImage")
            tr.image = imgs["rough"]
            tn = nt.nodes.new("ShaderNodeTexImage")
            tn.image = imgs["norm"]
            nm = nt.nodes.new("ShaderNodeNormalMap")
            nt.links.new(tb.outputs["Color"], bsdf.inputs["Base Color"])
            nt.links.new(tr.outputs["Color"], bsdf.inputs["Roughness"])
            nt.links.new(tn.outputs["Color"], nm.inputs["Color"])
            nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
            nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
            if "Metallic" in bsdf.inputs:
                bsdf.inputs["Metallic"].default_value = 0.0

    # export ALL visible geometry (baked shell + untouched furniture/openings)
    bpy.ops.object.select_all(action="DESELECT")
    for o in bpy.data.objects:
        if o.type in {"MESH", "EMPTY"} and not o.hide_get():
            o.select_set(True)
    os.makedirs(os.path.dirname(out_glb), exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=out_glb,
        export_format="GLB",
        use_selection=True,
        export_apply=True,
        export_cameras=False,
        export_lights=False,
    )
    print(f"  BAKED + EXPORTED -> {out_glb} ({os.path.getsize(out_glb) // 1024} KB)")


def main() -> int:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if not argv:
        print("usage: blender -b Room.blend --python bake_glb.py -- <out.glb> [res]")
        return 2
    out_glb = argv[0]
    res = int(argv[1]) if len(argv) > 1 else int(os.environ.get("LR_BAKE_RES", "1024"))
    _bake_in_blender(out_glb, res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
