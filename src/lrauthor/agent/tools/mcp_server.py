"""The capability tools over stdio MCP — for harnesses that cannot host them in-process.

Claude Code gets `fetch_material` / `render` / `critic` / `select_views` / `grid` /
`check_collisions` through `create_sdk_mcp_server`, running them inside the Python process that
started the session. Codex runs as a separate program, so the only way it reaches the same tools
is a server it can spawn and speak MCP to:

    python -m lrauthor.agent.tools.mcp_server --scene <room dir> [--tools a,b,c]

Both paths build the SAME `ToolRegistry` (`agent/tools/default_registry.py`) and bind invocations
to the same scene, so a tool behaves identically whichever harness called it. The registry stays
the one source of truth for names and schemas — this module only moves them across a pipe.

Nothing may be written to stdout: it is the MCP transport. Diagnostics go to stderr.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from lrauthor.agent.tools import build_default_registry


async def serve(scene: Path, tool_names: tuple[str, ...]) -> None:
    import mcp.types as types
    from mcp.server import Server
    from mcp.server.stdio import stdio_server

    registry = build_default_registry()
    unknown = [n for n in tool_names if registry.get_tool(n) is None]
    if unknown:
        raise SystemExit(f"unknown capability tool(s): {', '.join(unknown)}")

    server = Server("cap")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        out = []
        for name in tool_names:
            fn = registry.get_tool(name).schema["function"]
            out.append(
                types.Tool(
                    name=name,
                    description=fn["description"],
                    inputSchema=fn["parameters"],
                )
            )
        return out

    @server.call_tool()
    async def call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
        inv = await registry.build_invocation(name, arguments or {})
        if inv is None:
            raise ValueError(f"unknown tool {name}")
        if hasattr(inv, "bind"):
            inv.bind(str(scene), None)
        result = await inv.execute()
        payload = json.dumps(result.to_dict(), default=str)
        if not result.is_success():
            # MCP signals tool failure by exception; keep the payload in the message so the
            # model still sees WHAT failed, matching the in-process server's error content.
            raise RuntimeError(payload)
        return [types.TextContent(type="text", text=payload)]

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main(argv: list[str] | None = None) -> int:
    from lrauthor.agent.tools.default_registry import CAPABILITY_TOOLS

    default_names = ",".join(cls().name for cls in CAPABILITY_TOOLS)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", required=True, help="room directory the tools bind to")
    ap.add_argument(
        "--tools",
        default=default_names,
        help=f"comma-separated subset to expose (default: {default_names})",
    )
    args = ap.parse_args(argv)
    names = tuple(n.strip() for n in args.tools.split(",") if n.strip())
    try:
        asyncio.run(serve(Path(args.scene), names))
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
