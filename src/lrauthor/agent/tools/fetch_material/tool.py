"""fetch_material — PRIMITIVE. Fetch a REAL captured PBR set (Poly Haven) and recolour it.

The closed harness's biggest quality gap vs the original pipeline: procedural node materials look
flat next to real captured textures. This tool restores the original fetch→recolour flow as a
closed tool:

  1. SEARCH Poly Haven textures for the query ("blue carpet", "ceramic tile", "oak wood floor")
  2. download the set's Diffuse + Rough + nor_gl maps (md5-cached at ~/.litereality_texcache)
  3. optionally RECOLOUR the diffuse to a target colour in LAB space (pattern preserved)
  4. save under the room's own materials/ dir + merge the recipe into the room's textures.json
     (the definition stays reproducible — binaries can always be re-materialized from the recipe)

Returns the three file paths + a ready-to-wire Room.py snippet. Backed by the existing
room_ops/compile/{fetch_textures,recolor}.py — no new download machinery.
"""

from __future__ import annotations

import json
import re

from lrauthor.agent.tools._scene import room_dir_from
from lrauthor.agent.tools.base import (
    BaseDeclarativeTool,
    SceneToolInvocation,
    ToolParamsModel,
    ToolResult,
    make_tool_schema,
    validate_tool_params,
)

_ASSETS_CACHE: dict | None = None


def _all_texture_assets() -> dict:
    """{asset_id: meta} for every Poly Haven texture (one API call, cached per process)."""
    global _ASSETS_CACHE
    if _ASSETS_CACHE is None:
        import requests

        r = requests.get("https://api.polyhaven.com/assets?type=textures", timeout=30)
        r.raise_for_status()
        _ASSETS_CACHE = r.json()
    return _ASSETS_CACHE


def search_assets(query: str, k: int = 5) -> list[tuple[str, int]]:
    """Rank texture assets by WHOLE-WORD overlap with name/tags/categories.

    Substring matching burned us: 'blue carpet' matched 'bark_bluegum' (tree bark) via the
    'blue' substring. Only whole tokens count now, and category/tag hits (curated) outweigh
    name hits. MATERIAL-type words (carpet/tile/wood/...) count double — colour words alone
    must never select an asset."""
    material_words = {
        "carpet", "tile", "tiles", "wood", "plank", "planks", "brick", "plaster", "concrete",
        "fabric", "metal", "stone", "marble", "vinyl", "linoleum", "asphalt", "leather",
        "ceramic", "granite", "parquet", "laminate", "rug", "wicker", "denim", "terrazzo",
    }
    terms = [t for t in re.split(r"[^a-z0-9]+", query.lower()) if len(t) > 2]
    scored = []
    for aid, meta in _all_texture_assets().items():
        name_tokens = set(re.split(r"[^a-z0-9]+", (aid + " " + meta.get("name", "")).lower()))
        cat_tokens = set()
        for c in list(meta.get("tags", [])) + list(meta.get("categories", [])):
            cat_tokens |= set(re.split(r"[^a-z0-9]+", str(c).lower()))
        score = 0
        for t in terms:
            w = 2 if t in material_words else 1
            if t in cat_tokens:
                score += 3 * w
            elif t in name_tokens:
                score += 2 * w
        # an asset that matches NO material word when the query names one is almost surely wrong
        wanted = [t for t in terms if t in material_words]
        if wanted and not any(t in cat_tokens or t in name_tokens for t in wanted):
            score = 0
        # PLACEMENT sanity: 'wall tiles' must not return roof tiles. If the query names a
        # placement and the asset names only DIFFERENT placements, it is almost surely wrong.
        placements = {"wall", "floor", "ceiling", "roof", "ground", "road", "facade"}
        q_place = {t for t in terms if t in placements}
        a_place = (cat_tokens | name_tokens) & placements
        if q_place:
            if q_place & a_place:
                score += 4  # right placement — boost above generic material matches
            elif a_place:
                score //= 4  # explicitly a different placement (roof vs wall)
        if score:
            scored.append((aid, score))
    scored.sort(key=lambda x: -x[1])
    return scored[:k]


def _hex_to_rgb(h: str) -> list[int]:
    h = h.lstrip("#")
    return [int(h[i : i + 2], 16) for i in (0, 2, 4)]


class FetchMaterialParams(ToolParamsModel):
    query: str  # what to fetch: "blue low-pile carpet", "white ceramic tile", "painted plaster"
    name: str  # slug the files are saved under, e.g. "floor_carpet" (also the recipe key prefix)
    color_hex: str | None = None  # recolour the diffuse to this colour (pattern preserved)
    pattern_strength: float = 1.0  # 1.0 = full pattern, ~0.6 for smooth paint; never 0
    res: str = "1k"


class FetchMaterialInvocation(SceneToolInvocation):
    def get_description(self) -> str:
        return f"fetch_material: {self.params.query[:50]}"

    async def execute(self) -> ToolResult:
        from lrauthor.room_ops.compile.fetch_textures import materialize

        p = self.params
        try:
            room_dir = room_dir_from(self.scene_path)
        except ValueError as e:
            return ToolResult(error=str(e))

        try:
            hits = search_assets(p.query)
        except Exception as e:  # noqa: BLE001
            return ToolResult(error=f"Poly Haven search failed: {type(e).__name__}: {e}")
        if not hits:
            return ToolResult(error=f"no Poly Haven texture matches {p.query!r} — the library may "
                                    "simply lack this material (it has very few carpets/fabrics). "
                                    "Try other words, or build it procedurally at real-world scale.")
        asset_id = hits[0][0]
        assets = _all_texture_assets()

        def _desc(aid):
            m = assets.get(aid, {})
            return f"{aid} [{', '.join(list(m.get('categories', []))[:3])}]"

        match_note = "match looks good"
        top_meta = assets.get(asset_id, {})
        top_cats = " ".join(str(c) for c in list(top_meta.get("categories", [])) + list(top_meta.get("tags", []))).lower()
        material_terms = [t for t in re.split(r"[^a-z0-9]+", p.query.lower())
                          if t in ("carpet", "rug", "fabric")]
        if material_terms and not any(t in top_cats for t in material_terms):
            match_note = (f"WEAK match — {_desc(asset_id)} may not be a real {material_terms[0]}. "
                          "Inspect the diffuse before wiring; if wrong, a procedural material with "
                          "correct colour+roughness beats a wrong texture.")

        slug = re.sub(r"[^a-z0-9]+", "_", p.name.lower()).strip("_") or "mat"
        recipe: dict = {
            f"{slug}_rough.jpg": {"from": "polyhaven", "id": asset_id, "map": "Rough", "res": p.res},
            f"{slug}_normal.jpg": {"from": "polyhaven", "id": asset_id, "map": "nor_gl", "res": p.res},
        }
        if p.color_hex:
            recipe[f"{slug}_diff.jpg"] = {
                "from": "recolor", "base": asset_id, "map": "Diffuse", "res": p.res,
                "rgb": _hex_to_rgb(p.color_hex),
                "pattern_strength": max(0.05, min(p.pattern_strength, 1.0)),
            }
        else:
            recipe[f"{slug}_diff.jpg"] = {"from": "polyhaven", "id": asset_id, "map": "Diffuse", "res": p.res}

        mat_dir = room_dir / "materials"
        try:
            made = materialize(recipe, mat_dir)
        except Exception as e:  # noqa: BLE001
            return ToolResult(error=f"fetch failed for {asset_id!r}: {type(e).__name__}: {e}")

        # keep the room definition reproducible: merge into the room's textures.json recipe file
        rec_file = room_dir / "textures.json"
        try:
            full = json.loads(rec_file.read_text()) if rec_file.is_file() else {}
            full.update(recipe)
            rec_file.write_text(json.dumps(full, indent=1))
        except Exception:  # noqa: BLE001
            pass

        paths = {k.rsplit("_", 1)[-1].split(".")[0]: str(mat_dir / k) for k in made}
        usage = (
            f'img = bpy.data.images.load(str(ROOM_DIR / "materials/{slug}_diff.jpg"))  # + _rough/_normal\n'
            "# wire: TexImage(diff)->BaseColor · TexImage(rough, Non-Color)->Roughness ·\n"
            "# TexImage(normal, Non-Color)->NormalMap->Normal; set UV/Object mapping scale to real-world metres."
        )
        return ToolResult(output={
            "asset": _desc(asset_id), "match_note": match_note,
            "alternatives": [_desc(a) for a, _ in hits[1:4]],
            "files": paths, "recoloured": bool(p.color_hex), "usage": usage,
            "images": {f"{slug} diffuse ({asset_id}{', recoloured' if p.color_hex else ''})": paths.get("diff", "")},
        })


class FetchMaterialTool(BaseDeclarativeTool):
    def __init__(self) -> None:
        schema = make_tool_schema(
            name="fetch_material",
            description=(
                "Fetch a REAL captured PBR texture set from Poly Haven (diffuse+roughness+normal) and "
                "optionally RECOLOUR the diffuse to a target hex (pattern preserved). Files land in the "
                "room's materials/ dir; returns paths + a wiring snippet. Use for every surface with a "
                "visible pattern (carpet/tile/wood/brick/plaster grain) — a real texture beats "
                "procedural noise. Plain painted walls may stay flat colour + roughness."
            ),
            parameters={
                "query": {"type": "string", "description": "What to fetch: 'blue low-pile carpet', 'white ceramic tile', ..."},
                "name": {"type": "string", "description": "Slug to save under, e.g. 'floor_carpet'."},
                "color_hex": {"type": "string", "description": "Optional target colour, e.g. '#356199' — diffuse is LAB-shifted to it."},
                "pattern_strength": {"type": "number", "description": "1.0 = keep full pattern; ~0.6 for smooth paint (default 1.0)."},
                "res": {"type": "string", "description": "Texture resolution: 1k (default) or 2k."},
            },
            required=["query", "name"],
        )
        super().__init__("fetch_material", schema)

    async def build(self, params: dict) -> FetchMaterialInvocation:
        return FetchMaterialInvocation(validate_tool_params(FetchMaterialParams, params))
