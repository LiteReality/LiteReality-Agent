"""Tool registry — name → BaseDeclarativeTool, with OpenAI-format schema export."""

from __future__ import annotations

from typing import Any, Optional

from lrauthor.agent.tools.base import (
    BaseDeclarativeTool,
    BaseToolInvocation,
    ToolSchema,
)


class ToolRegistry:
    def __init__(self, tools: list[BaseDeclarativeTool]):
        self.tools = {tool.name: tool for tool in tools}

    def get_tool(self, name: str) -> Optional[BaseDeclarativeTool]:
        return self.tools.get(name)

    def get_all_tool_names(self) -> list[str]:
        return list(self.tools.keys())

    def get_tool_schemas(self) -> list[ToolSchema]:
        return [tool.schema for tool in self.tools.values()]

    async def build_invocation(
        self, name: str, params: dict[str, Any]
    ) -> Optional[BaseToolInvocation]:
        tool = self.get_tool(name)
        if not tool:
            return None
        return await tool.build(params)
