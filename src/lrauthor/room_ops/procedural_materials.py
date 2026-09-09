"""Procedural surface materials — a code-native complement to ``fetch_material`` (Poly Haven).

Poly Haven is thin exactly where real rooms are richest: carpet / rug / fabric / plaster, and the
patterned tile/wood floors (herringbone, hexagon, brick grid) that stage-1 struggles with. These
are **parametric procedural** materials built from Blender shader nodes — resolution-independent,
tileable, and tunable to a measured colour, and (being code, not binaries) they fit "Room.py is a
reproducible program". They bake to glTF textures through the normal ``bake_glb`` path.

Each ``make_*`` returns a ``bpy.types.Material``; ``make(category, **params)`` dispatches by name.
Pure ``bpy`` — no external dependency; importable both in Blender (Room.py compile) and out.
Design reference: Infinigen's procedural material library (re-implemented natively for Blender 4.x).
"""

from __future__ import annotations

CATEGORIES = ("carpet", "fabric", "plaster", "tile", "brick", "wood_planks")


def _principled(mat):
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    out = nt.nodes.get("Material Output")
    return nt, bsdf, out


def _new(name):
    import bpy

    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    return m


def make_carpet(name="carpet", color=(0.35, 0.32, 0.30), roughness=0.95, fuzz=0.6, scale=180.0):
    """Matte, fuzzy pile with fine colour break-up and a soft bump — the Poly-Haven blind spot."""
    m = _new(name)
    nt, bsdf, out = _principled(m)
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    if "Sheen Weight" in bsdf.inputs:  # fabric sheen (Blender 4.x)
        bsdf.inputs["Sheen Weight"].default_value = fuzz
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = scale
    noise.inputs["Detail"].default_value = 8.0
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.35
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    # subtle per-fibre colour variation
    cr = nt.nodes.new("ShaderNodeValToRGB")
    cr.color_ramp.elements[0].color = (color[0] * 0.85, color[1] * 0.85, color[2] * 0.85, 1)
    cr.color_ramp.elements[1].color = (min(color[0] * 1.15, 1), min(color[1] * 1.15, 1), min(color[2] * 1.15, 1), 1)
    nt.links.new(noise.outputs["Fac"], cr.inputs["Fac"])
    nt.links.new(cr.outputs["Color"], bsdf.inputs["Base Color"])
    return m


def make_fabric(name="fabric", color=(0.4, 0.4, 0.45), roughness=0.85):
    """Woven fabric — like carpet but tighter weave + more sheen."""
    return make_carpet(name, color=color, roughness=roughness, fuzz=0.9, scale=350.0)


def make_plaster(name="plaster", color=(0.86, 0.85, 0.82), roughness=0.8, bumpiness=0.12):
    """Painted plaster wall — flat colour with a faint orange-peel bump (never dead-flat)."""
    m = _new(name)
    nt, bsdf, out = _principled(m)
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 12.0
    noise.inputs["Detail"].default_value = 6.0
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = bumpiness
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return m


def _grid_material(name, tile_color, grout_color, scale, mortar, roughness, bevel=0.02):
    """Shared brick/tile builder — Brick Texture drives tile-vs-grout colour + a groove bump."""
    import math

    m = _new(name)
    nt, bsdf, out = _principled(m)
    bsdf.inputs["Roughness"].default_value = roughness
    brick = nt.nodes.new("ShaderNodeTexBrick")
    brick.inputs["Scale"].default_value = scale
    brick.inputs["Mortar Size"].default_value = mortar
    brick.inputs["Color1"].default_value = (*tile_color, 1)
    brick.inputs["Color2"].default_value = (*[c * 0.92 for c in tile_color], 1)  # slight tile variation
    brick.inputs["Mortar"].default_value = (*grout_color, 1)
    nt.links.new(brick.outputs["Color"], bsdf.inputs["Base Color"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = bevel
    nt.links.new(brick.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    _ = math
    return m


def make_tile(name="tile", color=(0.9, 0.9, 0.88), grout=(0.7, 0.7, 0.68), scale=8.0, roughness=0.25):
    """Ceramic wall/floor tile grid — glossy tiles + matte grout lines."""
    return _grid_material(name, color, grout, scale, mortar=0.02, roughness=roughness)


def make_brick(name="brick", color=(0.55, 0.28, 0.20), grout=(0.8, 0.78, 0.74), scale=6.0, roughness=0.8):
    """Exposed brick wall."""
    return _grid_material(name, color, grout, scale, mortar=0.03, roughness=roughness)


def make_wood_planks(name="wood_planks", color=(0.45, 0.30, 0.16), scale=(4.0, 0.5), roughness=0.35):
    """Wood plank floor — long rectangular planks with grain break-up (herringbone-adjacent)."""
    m = _new(name)
    nt, bsdf, out = _principled(m)
    bsdf.inputs["Roughness"].default_value = roughness
    brick = nt.nodes.new("ShaderNodeTexBrick")
    brick.inputs["Scale"].default_value = 3.0
    brick.inputs["Mortar Size"].default_value = 0.003
    brick.inputs["Color1"].default_value = (*color, 1)
    brick.inputs["Color2"].default_value = (*[c * 1.15 for c in color], 1)
    brick.inputs["Row Height"].default_value = 0.12
    brick.inputs["Brick Width"].default_value = 0.6
    # wood grain: stretched noise along the plank
    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.inputs["Scale"].default_value = 2.0
    wave.inputs["Distortion"].default_value = 12.0
    mix = nt.nodes.new("ShaderNodeMixRGB")
    mix.inputs["Fac"].default_value = 0.3
    nt.links.new(brick.outputs["Color"], mix.inputs["Color1"])
    nt.links.new(wave.outputs["Color"], mix.inputs["Color2"])
    nt.links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])
    _ = scale
    return m


_DISPATCH = {
    "carpet": make_carpet,
    "fabric": make_fabric,
    "plaster": make_plaster,
    "tile": make_tile,
    "brick": make_brick,
    "wood_planks": make_wood_planks,
}


def add_wear(mat, edge=0.35, scratch=0.15):
    """Layer subtle realism onto an EXISTING material: worn edges (rougher where geometry curves,
    via a Bevel-vs-true-normal mask — Infinigen's trick) + fine surface scratches in the bump.
    Breaks the flat 'too-clean CG' look that VLM critics penalise. Idempotent-ish; safe to skip."""
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    if not bsdf:
        return mat
    try:
        # geometric edge mask: dot(bevel_normal, true_normal) ~1 on flats, <1 on edges
        bev = nt.nodes.new("ShaderNodeBevel")
        bev.samples = 16
        bev.inputs["Radius"].default_value = 0.02
        geo = nt.nodes.new("ShaderNodeNewGeometry")
        dot = nt.nodes.new("ShaderNodeVectorMath")
        dot.operation = "DOT_PRODUCT"
        nt.links.new(bev.outputs["Normal"], dot.inputs[0])
        nt.links.new(geo.outputs["Normal"], dot.inputs[1])
        edge_mask = nt.nodes.new("ShaderNodeMath")  # (1 - dot) * gain -> edge amount
        edge_mask.operation = "MULTIPLY"
        edge_mask.inputs[1].default_value = edge * 6.0
        sub = nt.nodes.new("ShaderNodeMath")
        sub.operation = "SUBTRACT"
        sub.inputs[0].default_value = 1.0
        nt.links.new(dot.outputs["Value"], sub.inputs[1])
        nt.links.new(sub.outputs["Value"], edge_mask.inputs[0])
        # push roughness up at edges
        rough = nt.nodes.new("ShaderNodeMath")
        rough.operation = "ADD"
        rough.inputs[1].default_value = float(bsdf.inputs["Roughness"].default_value)
        rough.use_clamp = True
        nt.links.new(edge_mask.outputs["Value"], rough.inputs[0])
        nt.links.new(rough.outputs["Value"], bsdf.inputs["Roughness"])
        # fine scratches into the bump (only if normal isn't already driven)
        if not bsdf.inputs["Normal"].is_linked:
            scr = nt.nodes.new("ShaderNodeTexNoise")
            scr.inputs["Scale"].default_value = 400.0
            scr.inputs["Detail"].default_value = 10.0
            bump = nt.nodes.new("ShaderNodeBump")
            bump.inputs["Strength"].default_value = scratch * 0.1
            nt.links.new(scr.outputs["Fac"], bump.inputs["Height"])
            nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    except Exception:  # noqa: BLE001 — realism is optional; never break the material
        pass
    return mat


def make(category: str, name: str | None = None, **params):
    """Build a procedural material by category name; unknown → matte plaster in the given colour."""
    fn = _DISPATCH.get(category.lower())
    if fn is None:
        return make_plaster(name or category, **{k: v for k, v in params.items() if k in ("color", "roughness")})
    if name:
        params["name"] = name
    return fn(**params)
