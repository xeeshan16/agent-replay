"""Tool registry and execution.

Tool failures are data, not exceptions: an agent sees an error result and can recover, and the
recorder captures exactly what the agent saw. Only infrastructure problems propagate.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pydantic import JsonValue

from agent_replay.model import ToolCall, ToolResult, ToolSpec

ToolFn = Callable[..., JsonValue | Awaitable[JsonValue]]


@runtime_checkable
class ToolExecutor(Protocol):
    def specs(self) -> tuple[ToolSpec, ...]: ...

    async def execute(self, call: ToolCall) -> ToolResult: ...


@dataclass(frozen=True, slots=True)
class Tool:
    spec: ToolSpec
    fn: ToolFn


class Toolbox:
    """An in-process ``ToolExecutor`` backed by Python callables."""

    def __init__(self, *, timeout_s: float = 10.0) -> None:
        self._tools: dict[str, Tool] = {}
        self._timeout_s = timeout_s

    def add(
        self,
        fn: ToolFn,
        *,
        name: str | None = None,
        description: str = "",
        parameters: dict[str, JsonValue] | None = None,
        side_effect: bool = False,
    ) -> None:
        tool_name = name or fn.__name__
        if tool_name in self._tools:
            raise ValueError(f"tool {tool_name!r} is already registered")
        spec = ToolSpec(
            name=tool_name,
            description=description or (inspect.getdoc(fn) or "").split("\n", 1)[0],
            parameters=parameters or {},
            side_effect=side_effect,
        )
        self._tools[tool_name] = Tool(spec=spec, fn=fn)

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(tool.spec for tool in self._tools.values())

    def spec(self, name: str) -> ToolSpec | None:
        tool = self._tools.get(name)
        return tool.spec if tool else None

    async def execute(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                output=f"unknown tool: {call.name}",
                is_error=True,
                error_type="ToolNotFoundError",
            )
        try:
            async with asyncio.timeout(self._timeout_s):
                value = tool.fn(**call.arguments)
                if inspect.isawaitable(value):
                    value = await value
        except TimeoutError:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                output=f"tool timed out after {self._timeout_s}s",
                is_error=True,
                error_type="TimeoutError",
            )
        except Exception as exc:  # tool bugs are reported to the agent, not raised
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                output=str(exc),
                is_error=True,
                error_type=type(exc).__name__,
            )
        return ToolResult(tool_call_id=call.id, name=call.name, output=value)
