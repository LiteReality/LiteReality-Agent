"""Per-object render tool over stdio MCP; same renderer as the Claude tool host."""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
from pathlib import Path


async def serve(obj_dir, results, blender, tag, targets):
    import mcp.types as types
    from mcp.server import Server
    from mcp.server.stdio import stdio_server

    from . import refine_objects

    refine_objects.RESULTS = Path(results)
    counter = {}
    server = Server("obj")

    @server.list_tools()
    async def list_tools():
        return [
            types.Tool(
                name="render_object",
                description="Build this object's recipe and render capture-vs-build views.",
                inputSchema={"type": "object", "properties": {}},
            )
        ]

    @server.call_tool()
    async def call_tool(name, arguments):
        if name != "render_object":
            raise ValueError(name)
        # Blender/helper diagnostics must never corrupt the MCP transport.
        with contextlib.redirect_stdout(sys.stderr):
            result = await asyncio.to_thread(
                refine_objects._render_object_sync,
                Path(obj_dir),
                [Path(p) for p in targets],
                blender,
                tag,
                counter,
            )
        if "error" in result:
            raise RuntimeError(json.dumps(result))
        return [types.TextContent(type="text", text=json.dumps(result))]

    async with stdio_server() as (reader, writer):
        await server.run(reader, writer, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(serve(*sys.argv[1:5], json.loads(sys.argv[5])))
