"""Deterministic replay of recorded trajectories.

A ``Cassette`` indexes a recorded trajectory by *content*: model responses by the request, tool
results by tool name plus arguments. Lookups go through the tiers described in
``agent_replay.matching`` (exact, normalised, optionally fuzzy); repeated identical requests are
served in recorded order (FIFO), so a run that polls the same tool twice replays both answers.

Two modes:

* ``FULL``: model and tools both come from the cassette. Replaying an unchanged agent must
  reproduce the recording; any difference means the harness or the agent loop is
  non-deterministic.
* ``TOOLS``: the model is live (a new prompt, model or agent version) and tools are served from
  the cassette through the side-effect sandbox (``agent_replay.sandbox``). This is the
  regression-testing mode: the new agent faces the same world the old one did, and no
  side-effecting tool is ever executed for real.

A request that cannot be served raises ``CassetteMissError`` (or ``SideEffectBlockedError`` for an
unrecorded write), which ends the run with an error status at the exact point of divergence.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, cast

from pydantic import JsonValue

from agent_replay.agent import Agent
from agent_replay.canonical import fingerprint
from agent_replay.clock import Clock
from agent_replay.errors import CassetteMissError
from agent_replay.matching import MatchEvent, MatchMode, MatchOutcome, MatchPolicy, similarity
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
from agent_replay.normalize import Normalizer
from agent_replay.providers.base import LLMProvider
from agent_replay.recorder import record_run
from agent_replay.sandbox import SandboxedTools, SandboxPolicy


class ReplayMode(StrEnum):
    FULL = "full"
    TOOLS = "tools"


Kind = Literal["llm", "tool"]


@dataclass(slots=True)
class _Entry:
    step_index: int
    name: str
    raw_key: str
    norm_key: str
    norm_value: JsonValue
    payload: LLMResponse | ToolResult
    side_effect: bool = False
    consumed: bool = False


@dataclass(frozen=True, slots=True)
class _Hit:
    entry: _Entry
    outcome: MatchOutcome
    score: float | None = None


@dataclass(slots=True)
class _Shelf:
    """All recorded interactions of one kind, indexed by normalised key."""

    kind: Kind
    entries: list[_Entry] = field(default_factory=list)
    by_key: dict[str, deque[_Entry]] = field(default_factory=dict)

    def add(self, entry: _Entry) -> None:
        self.entries.append(entry)
        self.by_key.setdefault(entry.norm_key, deque()).append(entry)

    def take(
        self, name: str, raw_key: str, norm_key: str, norm_value: JsonValue, fuzzy: float | None
    ) -> _Hit | None:
        queue = self.by_key.get(norm_key)
        while queue and queue[0].consumed:  # consumed out of order by a fuzzy match
            queue.popleft()
        if queue:
            entry = queue.popleft()
            entry.consumed = True
            exact = entry.raw_key == raw_key
            return _Hit(entry, MatchOutcome.EXACT if exact else MatchOutcome.NORMALIZED)
        if fuzzy is None:
            return None
        best = self.closest(name, norm_value)
        if best is None or best[1] < fuzzy:
            return None
        best[0].consumed = True
        return _Hit(best[0], MatchOutcome.FUZZY, best[1])

    def closest(self, name: str, norm_value: JsonValue) -> tuple[_Entry, float] | None:
        """Most similar unconsumed entry with the same name; ties go to the earliest."""
        best: tuple[_Entry, float] | None = None
        for entry in self.entries:
            if entry.consumed or entry.name != name:
                continue
            score = similarity(norm_value, entry.norm_value)
            if best is None or score > best[1]:
                best = (entry, score)
        return best

    def remaining(self) -> int:
        return sum(1 for e in self.entries if not e.consumed)


class Cassette:
    def __init__(self, trajectory: Trajectory, policy: MatchPolicy | None = None) -> None:
        self.source_id = trajectory.trajectory_id
        self.policy = policy or MatchPolicy()
        self.events: list[MatchEvent] = []
        self._llm = _Shelf("llm")
        self._tools = _Shelf("tool")
        self._tool_specs: dict[str, ToolSpec] = {}
        self._write_tools: set[str] = set()
        for step in trajectory.steps:
            if isinstance(step, LLMStep):
                norm = self._normalize_llm(step.request)
                self._llm.add(
                    _Entry(
                        step_index=step.index,
                        name=step.request.model,
                        raw_key=step.request.key(),
                        norm_key=fingerprint("llm", norm),
                        norm_value=norm,
                        payload=step.response,
                    )
                )
                for spec in step.request.tools:
                    self._tool_specs.setdefault(spec.name, spec)
            elif isinstance(step, ToolStep):
                norm = self._normalize_args(step.call)
                self._tools.add(
                    _Entry(
                        step_index=step.index,
                        name=step.call.name,
                        raw_key=step.call.key(),
                        norm_key=fingerprint("tool", step.call.name, norm),
                        norm_value=norm,
                        payload=step.result,
                        side_effect=step.side_effect,
                    )
                )
                if step.side_effect:
                    self._write_tools.add(step.call.name)

    @classmethod
    def from_trajectory(cls, trajectory: Trajectory, policy: MatchPolicy | None = None) -> Cassette:
        return cls(trajectory, policy)

    def _normalize_llm(self, request: LLMRequest) -> JsonValue:
        return self.policy.normalizer.apply(cast(JsonValue, request.model_dump(mode="json")))

    def _normalize_args(self, call: ToolCall) -> JsonValue:
        return self.policy.normalizer.apply(cast(JsonValue, call.arguments))

    def _fuzzy_threshold(self, *, allow: bool) -> float | None:
        return self.policy.threshold if allow and self.policy.mode is MatchMode.FUZZY else None

    def record(self, event: MatchEvent) -> None:
        self.events.append(event)

    def take_llm(self, request: LLMRequest) -> LLMResponse:
        norm = self._normalize_llm(request)
        hit = self._llm.take(
            request.model,
            request.key(),
            fingerprint("llm", norm),
            norm,
            self._fuzzy_threshold(allow=True),
        )
        if hit is None:
            self.record(MatchEvent("llm", request.model, MatchOutcome.MISS))
            raise CassetteMissError(
                "llm",
                request.key(),
                f"model={request.model} after {len(request.messages)} messages"
                + self._closest_hint(self._llm, request.model, norm),
            )
        self.record(MatchEvent("llm", request.model, hit.outcome, hit.entry.step_index, hit.score))
        assert isinstance(hit.entry.payload, LLMResponse)  # noqa: S101 - shelf invariant
        return hit.entry.payload

    def try_take_tool(self, call: ToolCall, *, allow_fuzzy: bool = True) -> ToolResult | None:
        """Serve a recorded result for ``call``, or ``None`` without recording a miss."""
        norm = self._normalize_args(call)
        hit = self._tools.take(
            call.name,
            call.key(),
            fingerprint("tool", call.name, norm),
            norm,
            self._fuzzy_threshold(allow=allow_fuzzy),
        )
        if hit is None:
            return None
        self.record(MatchEvent("tool", call.name, hit.outcome, hit.entry.step_index, hit.score))
        assert isinstance(hit.entry.payload, ToolResult)  # noqa: S101 - shelf invariant
        # The live model may mint new call IDs; the result must answer the call being made.
        return hit.entry.payload.model_copy(update={"tool_call_id": call.id})

    def take_tool(self, call: ToolCall, *, allow_fuzzy: bool = True) -> ToolResult:
        result = self.try_take_tool(call, allow_fuzzy=allow_fuzzy)
        if result is None:
            self.record(MatchEvent("tool", call.name, MatchOutcome.MISS))
            raise self.tool_miss(call)
        return result

    def tool_miss(self, call: ToolCall) -> CassetteMissError:
        hint = self._closest_hint(self._tools, call.name, self._normalize_args(call))
        return CassetteMissError("tool", call.key(), f"{call.name}({call.arguments}){hint}")

    def _closest_hint(self, shelf: _Shelf, name: str, norm: JsonValue) -> str:
        best = shelf.closest(name, norm)
        if best is None:
            return ""
        return f"; closest unconsumed recording is step {best[0].step_index} (score {best[1]:.3f})"

    def recorded_side_effect(self, tool_name: str) -> bool:
        return tool_name in self._write_tools

    def recorded_tool_specs(self) -> tuple[ToolSpec, ...]:
        return tuple(self._tool_specs.values())

    def remaining(self) -> dict[str, int]:
        """Unconsumed interactions; non-zero after a FULL replay means the run ended early."""
        return {"llm": self._llm.remaining(), "tool": self._tools.remaining()}


class CassetteLLM:
    def __init__(self, cassette: Cassette) -> None:
        self._cassette = cassette

    @property
    def provider_name(self) -> str:
        return "cassette"

    async def complete(self, request: LLMRequest) -> LLMResponse:
        return self._cassette.take_llm(request)


def normalized_fingerprint(trajectory: Trajectory, normalizer: Normalizer) -> str:
    """Behavioural fingerprint after normalisation, in one scope across the whole run."""
    parts: list[JsonValue] = [
        trajectory.task_input,
        trajectory.status.value,
        trajectory.final_output,
        cast(JsonValue, trajectory.semantic_steps()),
    ]
    return fingerprint(normalizer.apply_many(parts))


@dataclass(frozen=True, slots=True)
class ReplayResult:
    original: Trajectory
    replayed: Trajectory
    mode: ReplayMode
    unconsumed: dict[str, int]
    policy: MatchPolicy = field(default_factory=MatchPolicy)
    sandbox: SandboxPolicy = field(default_factory=SandboxPolicy)
    matches: tuple[MatchEvent, ...] = ()

    @property
    def identical(self) -> bool:
        """Same behaviour, modulo the volatile values the normaliser masks."""
        norm = self.policy.normalizer
        return normalized_fingerprint(self.original, norm) == normalized_fingerprint(
            self.replayed, norm
        )

    @property
    def byte_identical(self) -> bool:
        return self.original.fingerprint() == self.replayed.fingerprint()

    def match_summary(self) -> dict[str, int]:
        counts = Counter(event.outcome.value for event in self.matches)
        return dict(sorted(counts.items()))


async def replay[A: Agent](
    agent: A,
    recording: Trajectory,
    *,
    mode: ReplayMode = ReplayMode.TOOLS,
    llm: LLMProvider | None = None,
    clock: Clock | None = None,
    policy: MatchPolicy | None = None,
    sandbox: SandboxPolicy | None = None,
) -> ReplayResult:
    """Re-run ``agent`` on the recorded task against the recording's cassette.

    In ``TOOLS`` mode the model is ``llm`` if given, otherwise the agent's own provider. Tool
    specs come from the agent so a changed tool description is part of what is being tested.
    The agent's own tool executor is only ever reached for read-only tools, and only when
    ``sandbox.live_reads`` is set.
    """
    match_policy = policy or MatchPolicy()
    sandbox_policy = sandbox or SandboxPolicy()
    cassette = Cassette.from_trajectory(recording, match_policy)
    model: LLMProvider = CassetteLLM(cassette) if mode is ReplayMode.FULL else (llm or agent.llm)
    tools = SandboxedTools(cassette, agent.tools, sandbox_policy)
    replayed = await record_run(
        agent.with_components(model, tools),
        recording.task_input,
        clock=clock,
        trajectory_id=f"{recording.trajectory_id}-replay-{mode.value}",
        metadata={
            "replay_of": recording.trajectory_id,
            "replay_mode": mode.value,
            "match_policy": match_policy.describe(),
            "sandbox": sandbox_policy.describe(),
        },
    )
    unconsumed = cassette.remaining()
    if mode is ReplayMode.TOOLS:
        del unconsumed["llm"]  # the model was live; recorded responses are not served
    return ReplayResult(
        original=recording,
        replayed=replayed,
        mode=mode,
        unconsumed=unconsumed,
        policy=match_policy,
        sandbox=sandbox_policy,
        matches=tuple(cassette.events),
    )
