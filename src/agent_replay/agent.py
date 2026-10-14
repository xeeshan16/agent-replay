"""A minimal tool-calling agent loop.

The loop is intentionally framework-free: it depends only on the ``LLMProvider`` and
``ToolExecutor`` protocols. Recording and replay work by swapping those two components, so any
agent that exposes them through ``with_components`` can be recorded and replayed the same way.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Protocol, Self

from agent_replay.canonical import canonical_json
from agent_replay.errors import AgentReplayError
from agent_replay.logs import get_logger
from agent_replay.model import LLMRequest, Message, RunStatus
from agent_replay.providers.base import LLMProvider
from agent_replay.tools import ToolExecutor

_log = get_logger("agent")


@dataclass(frozen=True, slots=True)
class AgentRun:
    status: RunStatus
    final_output: str | None = None
    error: str | None = None


class Agent(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def llm(self) -> LLMProvider: ...

    @property
    def tools(self) -> ToolExecutor: ...

    def with_components(self, llm: LLMProvider, tools: ToolExecutor) -> Self: ...

    async def run(self, task: str) -> AgentRun: ...


@dataclass(frozen=True, slots=True)
class ToolCallingAgent:
    name: str
    llm: LLMProvider
    tools: ToolExecutor
    model: str
    system_prompt: str
    max_turns: int = 8

    def with_components(self, llm: LLMProvider, tools: ToolExecutor) -> Self:
        return dataclasses.replace(self, llm=llm, tools=tools)

    async def run(self, task: str) -> AgentRun:
        messages: list[Message] = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=task),
        ]
        specs = self.tools.specs()
        try:
            for _turn in range(self.max_turns):
                request = LLMRequest(model=self.model, messages=tuple(messages), tools=specs)
                response = await self.llm.complete(request)
                messages.append(
                    Message(
                        role="assistant",
                        content=response.content,
                        tool_calls=response.tool_calls,
                    )
                )
                if not response.tool_calls:
                    return AgentRun(status=RunStatus.SUCCESS, final_output=response.content)
                # Tool calls within one turn run sequentially: their order is part of the
                # trajectory and must be reproducible on replay.
                for call in response.tool_calls:
                    result = await self.tools.execute(call)
                    messages.append(
                        Message(
                            role="tool",
                            content=_render_output(result.output, result.is_error),
                            tool_call_id=call.id,
                            name=call.name,
                        )
                    )
        except AgentReplayError as exc:
            _log.warning(
                "agent run failed",
                extra={"ctx": {"agent": self.name, "error_type": type(exc).__name__}},
            )
            return AgentRun(status=RunStatus.ERROR, error=f"{type(exc).__name__}: {exc}")
        return AgentRun(status=RunStatus.MAX_STEPS, error=f"exceeded {self.max_turns} turns")


def _render_output(output: object, is_error: bool) -> str:
    text = output if isinstance(output, str) else canonical_json(output)
    return f"ERROR: {text}" if is_error else text
