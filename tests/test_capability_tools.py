"""The capability tools the authoring model is handed must actually be callable.

Stage 3 offers the model four tools it cannot substitute with file editing (`author.py`
CAPABILITY_TOOLS). If one of them is missing from the registry, mis-schema'd, or dies before it
reaches its real work, the run does not fail — the model simply stops using it and authors the room
blind. That is a silent quality loss, which is exactly the failure mode worth a test.

Offline by design: these check reachability and wiring, not pixels. A test that renders needs
Blender plus a real scan and is marked `blender`.
"""

from __future__ import annotations

import pytest

from lrauthor.agent.author import CAPABILITY_TOOLS, build_capability_server
from lrauthor.agent.tools import build_default_registry


@pytest.fixture(scope="module")
def registry():
    return build_default_registry()


@pytest.mark.parametrize("name", CAPABILITY_TOOLS)
def test_capability_tool_is_registered(registry, name):
    """A renamed or dropped tool must break here, not degrade a $7 authoring run."""
    assert registry.get_tool(name) is not None, f"{name} is offered to the model but not registered"


@pytest.mark.parametrize("name", CAPABILITY_TOOLS)
def test_capability_tool_schema_is_well_formed(registry, name):
    """`build_capability_server` reads `schema["function"]["description"|"parameters"]` directly and
    would raise KeyError at harness start-up; assert the shape the SDK requires."""
    fn = registry.get_tool(name).schema["function"]
    assert fn["name"] == name
    assert fn["description"].strip(), f"{name} has no description — the model cannot know when to call it"
    params = fn["parameters"]
    assert params.get("type") == "object" and "properties" in params


def test_render_is_among_the_capabilities():
    """The image tool specifically: the prompt instructs the model to render and READ the PNG
    (`author.py` "render, look, correct"), so its absence would gut the self-check loop."""
    assert "render" in CAPABILITY_TOOLS
    assert "select_views" in CAPABILITY_TOOLS, "render auto-picks frames through select_views"


def test_capability_server_exposes_prefixed_names(tmp_path):
    """The allow-list must match the MCP names the SDK will emit, or every call is refused."""
    server, allowed = build_capability_server(tmp_path, CAPABILITY_TOOLS)
    assert server is not None
    assert allowed == [f"mcp__cap__{n}" for n in CAPABILITY_TOOLS]

