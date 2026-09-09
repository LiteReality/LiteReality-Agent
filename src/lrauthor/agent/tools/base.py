"""Declarative tool base.

A tool = a name + an OpenAI-style JSON schema + a `build(params) -> invocation`. The invocation
does the work in `execute() -> ToolResult`. The harness validates params (pydantic, strict),
builds the invocation, runs it, and feeds the result back into the conversation.

Some tools are *harness-intercepted*: the schema is advertised to the LLM, but the harness
runs the real flow so it can update build freshness / write a revision. Our intercepted tool
is `build_room` (compile `Room.py` → `Room.glb`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict

TParams = TypeVar("TParams")
TResult = TypeVar("TResult")
TToolParamsModel = TypeVar("TToolParamsModel", bound=BaseModel)

ToolSchema = dict[str, Any]


def make_tool_schema(
    name: str,
    description: str,
    parameters: Optional[dict[str, Any]] = None,
    required: Optional[list[str]] = None,
) -> ToolSchema:
    """OpenAI-compatible function-tool schema (strict mode: additionalProperties=False)."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": parameters or {},
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


class ToolResult:
    """Result of a tool execution. `build` carries optional structured build/QC signals
    from the room.glb compile (when the tool is the intercepted `build_room`)."""

    def __init__(
        self,
        output: Any = None,
        error: Optional[str] = None,
        build: Optional[dict] = None,
        tool_call_id: Optional[str] = None,
    ):
        self.output = output
        self.error = error
        self.build = build
        self.tool_call_id = tool_call_id

    def is_success(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        if self.error:
            result["error"] = self.error
        else:
            result["result"] = self.output
        if self.build:
            result["build"] = self.build
        return result


class ToolParamsModel(BaseModel):
    """Strict params model — reject unknown keys."""

    model_config = ConfigDict(extra="forbid")


def validate_tool_params(
    model_cls: type[TToolParamsModel], params: dict[str, Any]
) -> TToolParamsModel:
    """Validate, treating explicit nulls as omitted (so defaults apply)."""
    normalized = {k: v for k, v in params.items() if v is not None}
    return model_cls(**normalized)


class BaseToolInvocation(ABC, Generic[TParams, TResult]):
    def __init__(self, params: TParams):
        self.params = params

    @abstractmethod
    def get_description(self) -> str: ...

    @abstractmethod
    async def execute(self) -> ToolResult: ...


class SceneToolInvocation(BaseToolInvocation[TParams, TResult], ABC):
    """Invocation bound to the active scene (the scene.py path + record context the harness
    injects), so tools like place_object / set_material can edit the right file."""

    def __init__(self, params: TParams):
        super().__init__(params)
        self.scene_path: str | None = None
        self.ctx: Any = None  # harness context: registry, record dir, render dir

    def bind(self, scene_path: str, ctx: Any) -> None:
        self.scene_path = scene_path
        self.ctx = ctx


class BaseDeclarativeTool(ABC):
    """A tool advertised to the LLM. Subclasses set name+schema and build an invocation."""

    def __init__(self, name: str, schema: ToolSchema):
        self.name = name
        self.schema = schema

    @abstractmethod
    async def build(self, params: dict) -> BaseToolInvocation: ...
