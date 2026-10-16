"""The side-effect sandbox that sits between a replayed agent and its tools.

Every tool call made during replay goes through ``SandboxedTools``:

1. If the agent under test does not offer the tool, the call fails exactly as it would live
   (``ToolNotFoundError`` result), even if an older recording used that tool.
2. If the cassette has a recording, it is served. Writes never match fuzzily unless the match
   policy says so (see ``agent_replay.matching``).
3. On a miss, a **read-only** tool may be executed live when ``live_reads`` is enabled, so a new
   agent version that adds a harmless lookup can still be replayed.
4. On a miss, a **side-effecting** tool is never executed. ``unrecorded_writes="fail"`` ends the
   run with ``SideEffectBlockedError``; ``"block"`` returns an error result to the agent so the
   run continues and later scoring sees the attempted write and what the agent did next.

A tool counts as side-effecting if the agent's spec says so *or* the recording flagged it, or if
it has no spec at all: the classification fails closed. Read-only is a claim made by whoever
registered the tool; the sandbox enforces it, it cannot verify it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import JsonValue

from agent_replay.errors import SideEffectBlockedError
from agent_replay.logs import get_logger
from agent_replay.matching import MatchEvent, MatchOutcome
from agent_replay.model import ToolCall, ToolResult, ToolSpec
from agent_replay.tools import ToolExecutor

if TYPE_CHECKING:
    from agent_replay.replay import Cassette

_log = get_logger("sandbox")

BLOCKED_ERROR_TYPE = "SideEffectBlockedError"


class UnrecordedWrites(StrEnum):
    FAIL = "fail"
    BLOCK = "block"


@dataclass(frozen=True, slots=True)
class SandboxPolicy:
    live_reads: bool = False
    unrecorded_writes: UnrecordedWrites = UnrecordedWrites.FAIL

    def describe(self) -> dict[str, JsonValue]:
        return {"live_reads": self.live_reads, "unrecorded_writes": self.unrecorded_writes.value}


class SandboxedTools:
    """A ``ToolExecutor`` that serves the cassette and guards the agent's real tools."""

    def __init__(self, cassette: Cassette, live: ToolExecutor, policy: SandboxPolicy) -> None:
        self._cassette = cassette
        self._live = live
        self._policy = policy
        self._specs = live.specs()
        self._by_name: dict[str, ToolSpec] = {spec.name: spec for spec in self._specs}
        self.live_calls = 0

    def specs(self) -> tuple[ToolSpec, ...]:
        return self._specs

    def is_write(self, name: str) -> bool:
        spec = self._by_name.get(name)
        return spec is None or spec.side_effect or self._cassette.recorded_side_effect(name)

    async def execute(self, call: ToolCall) -> ToolResult:
        if call.name not in self._by_name:
            self._cassette.record(MatchEvent("tool", call.name, MatchOutcome.UNKNOWN_TOOL))
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                output=f"unknown tool: {call.name}",
                is_error=True,
                error_type="ToolNotFoundError",
            )
        write = self.is_write(call.name)
        allow_fuzzy = not write or self._cassette.policy.fuzzy_writes
        served = self._cassette.try_take_tool(call, allow_fuzzy=allow_fuzzy)
        if served is not None:
            return served

        if not write:
            if self._policy.live_reads:
                return await self._execute_live(call)
            self._cassette.record(MatchEvent("tool", call.name, MatchOutcome.MISS))
            raise self._cassette.tool_miss(call)

        self._cassette.record(MatchEvent("tool", call.name, MatchOutcome.BLOCKED))
        _log.warning(
            "blocked unrecorded side-effecting tool call",
            extra={"ctx": {"tool": call.name, "policy": self._policy.unrecorded_writes.value}},
        )
        if self._policy.unrecorded_writes is UnrecordedWrites.FAIL:
            raise SideEffectBlockedError(
                f"unrecorded call to side-effecting tool {call.name}({call.arguments}) "
                "was not executed"
            )
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            output=(
                f"blocked by replay sandbox: {call.name} has side effects and this call "
                "matches no recording"
            ),
            is_error=True,
            error_type=BLOCKED_ERROR_TYPE,
        )

    async def _execute_live(self, call: ToolCall) -> ToolResult:
        # Second line of defence: only reachable for tools classified read-only above.
        if self.is_write(call.name):
            raise SideEffectBlockedError(f"refusing live execution of {call.name}")
        self.live_calls += 1
        self._cassette.record(MatchEvent("tool", call.name, MatchOutcome.LIVE))
        return await self._live.execute(call)
