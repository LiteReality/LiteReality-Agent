"""lrauthor.agent.tools — the capability tools handed to the authoring/materials/QC
agent. See README.md.

Each tool lives in its own folder with code + a README; `default_registry.build_default_registry()`
assembles them into a `ToolRegistry` that `author.build_capability_server` exposes as `mcp__cap__*`.
Retired primitives/composites from the old closed-loop driver live under the repo-root
`legacy/lrauthor/realism_authoring/tools/`.
"""

from lrauthor.agent.tools.default_registry import build_default_registry
from lrauthor.agent.tools.registry import ToolRegistry

__all__ = ["build_default_registry", "ToolRegistry"]
