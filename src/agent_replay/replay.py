"""Deterministic replay of recorded trajectories.

A ``Cassette`` indexes a recorded trajectory by *content*: model responses are keyed by the
canonical request, tool results by tool name plus canonical arguments. Identical keys are served
in recorded order (FIFO), so a run that polls the same tool twice replays both answers.

Two modes:

* ``FULL``: model and tools both come from the cassette. Replaying an unchanged agent must
  reproduce the recording exactly; any difference means the harness or the agent loop is
  non-deterministic.
* ``TOOLS``: the model is live (a new prompt, model or agent version) and tools are served from
  the cassette. This is the regression-testing mode: the new agent faces the same world the old
  one did, and no real tool, including ones with side effects, is ever executed.

Matching is strict in this version: a request that was never recorded raises
``CassetteMissError``, which ends the run with an error status at the exact point of divergence.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import StrEnum

from agent_replay.agent import Agent
from agent_replay.clock import Clock
from agent_replay.errors import CassetteMissError
from agent_replay.model import (
    LLMRequest,
    LLMResponse,
    LLMStep,
    ToolCall,
    ToolResult,
    ToolSpec,
    ToolStep,
    Trajectory,
)
from agent_replay.providers.base import LLMProvider
from agent_replay.recorder import record_run
from agent_replay.tools import ToolExecutor


class ReplayMode(StrEnum):
    FULL = "full"
    TOOLS = "tools"


class Cassette:
    def __init__(self, trajectory: Trajectory) -> None:
        self.source_id = trajectory.trajectory_id
        self._llm: dict[str, deque[LLMResponse]] = {}
        self._tools: dict[str, deque[ToolResult]] = {}
        self._tool_specs: dict[str, ToolSpec] = {}
        for step in trajectory.steps:
            if isinstance(step, LLMStep):
                self._llm.setdefault(step.request.key(), deque()).append(step.response)
                for spec in step.request.tools:
                    self._tool_specs.setdefault(spec.name, spec)
            elif isinstance(step, ToolStep):
                self._tools.setdefault(step.call.key(), deque()).append(step.result)

    @classmethod
    def from_trajectory(cls, trajectory: Trajectory) -> Cassette:
        return cls(trajectory)

    def take_llm(self, request: LLMRequest) -> LLMResponse:
        key = request.key()
        queue = self._llm.get(key)
        if not queue:
            raise CassetteMissError(
                "llm", key, f"model={request.model} after {len(request.messages)} messages"
            )
        return queue.popleft()

    def take_tool(self, call: ToolCall) -> ToolResult:
        key = call.key()
        queue = self._tools.get(key)
        if not queue:
            raise CassetteMissError("tool", key, f"{call.name}({call.arguments})")
        recorded = queue.popleft()
        # The live model may mint new call IDs; the result must answer the call being made.
        return recorded.model_copy(update={"tool_call_id": call.id})

    def recorded_tool_specs(self) -> tuple[ToolSpec, ...]:
        return tuple(self._tool_specs.values())

    def remaining(self) -> dict[str, int]:
        """Unconsumed interactions; non-zero after a FULL replay means the run ended early."""
        return {
            "llm": sum(len(q) for q in self._llm.values()),
            "tool": sum(len(q) for q in self._tools.values()),
        }


class CassetteLLM:
    def __init__(self, cassette: Cassette) -> None:
        self._cassette = cassette

    @property
    def provider_name(self) -> str:
        return "cassette"

    async def complete(self, request: LLMRequest) -> LLMResponse:
        return self._cassette.take_llm(request)


class CassetteTools:
    """Serves recorded tool results; never executes a real tool."""

    def __init__(self, cassette: Cassette, specs: tuple[ToolSpec, ...] | None = None) -> None:
        self._cassette = cassette
        self._specs = specs if specs is not None else cassette.recorded_tool_specs()

    def specs(self) -> tuple[ToolSpec, ...]:
        return self._specs

    async def execute(self, call: ToolCall) -> ToolResult:
        return self._cassette.take_tool(call)


@dataclass(frozen=True, slots=True)
class ReplayResult:
    original: Trajectory
    replayed: Trajectory
    mode: ReplayMode
    unconsumed: dict[str, int]

    @property
    def identical(self) -> bool:
        return self.original.fingerprint() == self.replayed.fingerprint()


async def replay[A: Agent](
    agent: A,
    recording: Trajectory,
    *,
    mode: ReplayMode = ReplayMode.TOOLS,
    llm: LLMProvider | None = None,
    clock: Clock | None = None,
) -> ReplayResult:
    """Re-run ``agent`` on the recorded task against the recording's cassette.

    In ``TOOLS`` mode the model is ``llm`` if given, otherwise the agent's own provider. Tool
    specs come from the agent so a changed tool description is part of what is being tested.
    """
    cassette = Cassette.from_trajectory(recording)
    model: LLMProvider = CassetteLLM(cassette) if mode is ReplayMode.FULL else (llm or agent.llm)
    tools: ToolExecutor = CassetteTools(cassette, agent.tools.specs())
    replayed = await record_run(
        agent.with_components(model, tools),
        recording.task_input,
        clock=clock,
        trajectory_id=f"{recording.trajectory_id}-replay-{mode.value}",
        metadata={"replay_of": recording.trajectory_id, "replay_mode": mode.value},
    )
    return ReplayResult(
        original=recording, replayed=replayed, mode=mode, unconsumed=cassette.remaining()
    )
