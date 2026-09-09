"""The registry must hold every capability tool, each registered and well-schema'd.

`test_capability_tools.py` guards the tools `author.py` cannot substitute. This guards the OTHER end:
every tool the default registry advertises must be reachable by name and expose an OpenAI-format
schema the SDK can serialize. A tool that instantiates but ships a malformed schema is advertised to
the model and then rejected at harness start-up — a silent capability loss.

Offline by design: reachability and schema shape, not behaviour. The per-name checks parametrize over
the LIVE registry, so a tool added to `CAPABILITY_TOOLS` is covered automatically. (The old closed-loop
primitives/composites are retired under tools/legacy/ and are not registered.)
"""

from __future__ import annotations

import pytest

from lrauthor.agent.tools import build_default_registry
from lrauthor.agent.tools.base import BaseDeclarativeTool
from lrauthor.agent.tools.default_registry import CAPABILITY_TOOLS

# Built once at import so it can drive parametrization — instantiation is cheap and side-effect free
# (build_default_registry is already used this way by the offline capability test).
_REGISTRY = build_default_registry()
_NAMES = _REGISTRY.get_all_tool_names()


@pytest.fixture(scope="module")
def registry():
    return build_default_registry()


def test_registry_is_exactly_the_capability_tools(registry):
    """The registry is now just the capability set the agent is handed — no closed-loop leftovers.
    If the count drifts, a tool was dropped (the model loses a capability) or added without updating
    author.CAPABILITY_TOOLS / the docs — surface it."""
    from lrauthor.agent.author import CAPABILITY_TOOLS as CAP_NAMES

    assert len(CAPABILITY_TOOLS) == 6, f"expected 6 capability tools, got {len(CAPABILITY_TOOLS)}"
    assert set(registry.get_all_tool_names()) == set(CAP_NAMES)


def test_tool_names_are_unique(registry):
    """Two tools claiming the same name → the second silently shadows the first in the name→tool
    dict, and the model calls whichever won. A collision must fail loudly here instead."""
    names = registry.get_all_tool_names()
    assert len(names) == len(set(names)), f"duplicate tool name(s): {sorted(names)}"


@pytest.mark.parametrize("name", _NAMES)
def test_every_registered_tool_is_reachable(registry, name):
    """`registry.get_tool(name)` is how the harness dispatches every call; None means dead."""
    tool = registry.get_tool(name)
    assert tool is not None, f"{name} is in the registry index but get_tool returned None"
    assert isinstance(tool, BaseDeclarativeTool)
    assert tool.name == name, f"tool indexed as {name!r} reports name {tool.name!r}"


@pytest.mark.parametrize("name", _NAMES)
def test_every_tool_schema_is_well_formed(registry, name):
    """The SDK reads schema['function']['name'|'description'|'parameters'] directly and strict mode
    requires an object schema with properties + additionalProperties=False. A missing key is a
    KeyError at start-up; a blank description leaves the model unable to know when to call the tool."""
    schema = registry.get_tool(name).schema
    assert schema.get("type") == "function"
    fn = schema["function"]
    assert fn["name"] == name
    assert fn["description"].strip(), f"{name} has an empty description"
    params = fn["parameters"]
    assert params.get("type") == "object"
    assert "properties" in params
    assert params.get("additionalProperties") is False, (
        f"{name} params must set additionalProperties=False for OpenAI strict mode"
    )


def test_get_tool_schemas_covers_the_whole_set(registry):
    """The harness serializes get_tool_schemas() as one block; it must match the registry 1:1."""
    schemas = registry.get_tool_schemas()
    assert len(schemas) == len(registry.get_all_tool_names())
    assert {s["function"]["name"] for s in schemas} == set(registry.get_all_tool_names())


def test_unknown_tool_resolves_to_none(registry):
    """Dispatch of a hallucinated tool name must be a clean miss, not a crash."""
    assert registry.get_tool("no_such_tool") is None
