"""fetch_material — fetch a real Poly Haven PBR set, optionally recoloured.

The ranking is a pure function over a cached asset index, so it tests offline with the network
injected out. That is the half worth pinning: the tool's own docstring records that substring
matching once made "blue carpet" select `bark_bluegum` (tree bark), because "blue" appears inside
"bluegum". A material fetched from a colour word alone is how a floor ends up bark-textured.

The network fetch itself is `-m live`.
"""

from __future__ import annotations

import pytest

from lrauthor.agent.tools.fetch_material import tool as fm

# A miniature Poly Haven index carrying the exact trap the whole-word rule exists to stop.
_ASSETS = {
    "bark_bluegum": {"name": "Bark Bluegum", "tags": ["bark", "tree"], "categories": ["nature"]},
    "carpet_blue_01": {"name": "Blue Carpet", "tags": ["carpet", "blue"], "categories": ["fabric"]},
    "wood_planks_02": {"name": "Wood Planks", "tags": ["wood", "planks"], "categories": ["wood"]},
    "white_tiles_03": {"name": "White Tiles", "tags": ["tile", "white"], "categories": ["tiles"]},
}


@pytest.fixture(autouse=True)
def _offline_index(monkeypatch):
    """Pin the cache so ranking is tested without touching api.polyhaven.com."""
    monkeypatch.setattr(fm, "_ASSETS_CACHE", _ASSETS, raising=False)


def test_schema_is_well_formed():
    fn = fm.FetchMaterialTool().schema["function"]
    assert fn["name"] == "fetch_material"
    for required in ("query", "name"):
        assert required in fn["parameters"]["properties"]


def test_colour_word_does_not_select_by_substring():
    """The recorded bug: 'blue carpet' ranked bark_bluegum, because 'blue' ⊂ 'bluegum'."""
    ranked = [asset for asset, _ in fm.search_assets("blue carpet", k=4)]
    assert ranked, "nothing ranked at all"
    assert ranked[0] == "carpet_blue_01", f"expected the carpet first, got {ranked}"
    if "bark_bluegum" in ranked:
        assert ranked.index("bark_bluegum") > ranked.index("carpet_blue_01")


@pytest.mark.parametrize(
    "query,expected",
    [("wood planks", "wood_planks_02"), ("white ceramic tile", "white_tiles_03")],
)
def test_material_words_drive_the_choice(query, expected):
    ranked = [asset for asset, _ in fm.search_assets(query, k=4)]
    assert ranked and ranked[0] == expected, f"{query!r} ranked {ranked}"


def test_unmatchable_query_returns_nothing_rather_than_a_wrong_material():
    """Silence is correct here. A confident wrong PBR set is applied to a whole surface."""
    assert fm.search_assets("xyzzy nonexistent surface", k=4) == []


@pytest.mark.parametrize("strength", [0.0, -0.5])
def test_pattern_strength_floor_is_documented(strength):
    """`pattern_strength` is documented as "never 0" — 0 erases the captured pattern entirely."""
    params = fm.FetchMaterialParams(query="carpet", name="floor", pattern_strength=strength)
    assert params.pattern_strength == strength, (
        "params accept it today; if this starts failing the tool grew validation and the "
        "prompt guidance in the docstring should move into the schema"
    )


@pytest.mark.live
def test_fetches_a_real_asset_from_polyhaven(monkeypatch, tmp_path):
    """Hits api.polyhaven.com. Run with `-m live`."""
    monkeypatch.setattr(fm, "_ASSETS_CACHE", None, raising=False)
    ranked = fm.search_assets("wood planks", k=3)
    assert ranked, "Poly Haven returned no match for a common material"
    assert all(isinstance(a, str) and score > 0 for a, score in ranked)
