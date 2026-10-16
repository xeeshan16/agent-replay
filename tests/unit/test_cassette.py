from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import JsonValue

from agent_replay.errors import CassetteMissError
from agent_replay.matching import MatchMode, MatchOutcome, MatchPolicy
from agent_replay.model import (
    LLMRequest,
    LLMResponse,
    LLMStep,
    Message,
    RunStatus,
    ToolCall,
    ToolResult,
    ToolStep,
    Trajectory,
)
from agent_replay.normalize import Normalizer
from agent_replay.replay import Cassette

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _tool_step(index: int, output: str) -> ToolStep:
    call = ToolCall(id=f"c{index}", name="poll", arguments={"job": "j1"})
    return ToolStep(
        index=index,
        started_at=T0,
        call=call,
        result=ToolResult(tool_call_id=call.id, name="poll", output=output),
    )


def _trajectory() -> Trajectory:
    request = LLMRequest(model="m", messages=(Message(role="user", content="go"),))
    return Trajectory(
        trajectory_id="t1",
        agent_name="a",
        task_input="go",
        status=RunStatus.SUCCESS,
        started_at=T0,
        steps=(
            LLMStep(index=0, started_at=T0, request=request, response=LLMResponse(model="m")),
            _tool_step(1, "pending"),
            _tool_step(2, "done"),
        ),
    )


def test_repeated_tool_calls_are_served_in_recorded_order() -> None:
    cassette = Cassette.from_trajectory(_trajectory())
    call = ToolCall(id="new-id", name="poll", arguments={"job": "j1"})
    first, second = cassette.take_tool(call), cassette.take_tool(call)
    assert (first.output, second.output) == ("pending", "done")
    assert first.tool_call_id == "new-id"
    with pytest.raises(CassetteMissError) as excinfo:
        cassette.take_tool(call)
    assert excinfo.value.kind == "tool"


def test_unrecorded_arguments_miss() -> None:
    cassette = Cassette.from_trajectory(_trajectory())
    with pytest.raises(CassetteMissError, match="poll"):
        cassette.take_tool(ToolCall(id="x", name="poll", arguments={"job": "j2"}))


def test_llm_requests_match_on_content_and_track_remaining() -> None:
    cassette = Cassette.from_trajectory(_trajectory())
    assert cassette.remaining() == {"llm": 1, "tool": 2}
    request = LLMRequest(model="m", messages=(Message(role="user", content="go"),))
    assert cassette.take_llm(request).model == "m"
    assert cassette.remaining() == {"llm": 0, "tool": 2}
    with pytest.raises(CassetteMissError):
        cassette.take_llm(request.model_copy(update={"temperature": 0.7}))


def _write_trajectory(*calls: dict[str, JsonValue]) -> Trajectory:
    steps = []
    for index, arguments in enumerate(calls):
        call = ToolCall(id=f"c{index}", name="refund", arguments=arguments)
        steps.append(
            ToolStep(
                index=index,
                started_at=T0,
                call=call,
                result=ToolResult(tool_call_id=call.id, name="refund", output=f"r{index}"),
                side_effect=True,
            )
        )
    return Trajectory(
        trajectory_id="w",
        agent_name="a",
        task_input="go",
        status=RunStatus.SUCCESS,
        started_at=T0,
        steps=tuple(steps),
    )


KEY_A = "2b1f7c1e-8d1c-4b8e-9a77-0d3c2f1e5a10"
KEY_B = "9f0e6d5c-4b3a-4291-8f7e-6d5c4b3a2910"


def test_volatile_values_match_after_normalisation() -> None:
    cassette = Cassette(_write_trajectory({"order": "o1", "key": KEY_A}))
    result = cassette.take_tool(
        ToolCall(id="n", name="refund", arguments={"order": "o1", "key": KEY_B})
    )
    assert result.output == "r0"
    assert [(e.outcome, e.recorded_index) for e in cassette.events] == [
        (MatchOutcome.NORMALIZED, 0)
    ]


def test_without_normalisation_volatile_values_miss() -> None:
    cassette = Cassette(
        _write_trajectory({"order": "o1", "key": KEY_A}), MatchPolicy(normalizer=Normalizer.none())
    )
    with pytest.raises(CassetteMissError):
        cassette.take_tool(ToolCall(id="n", name="refund", arguments={"order": "o1", "key": KEY_B}))
    assert cassette.events[-1].outcome is MatchOutcome.MISS


def test_exact_matches_are_reported_as_exact() -> None:
    cassette = Cassette(_trajectory())
    cassette.take_tool(ToolCall(id="n", name="poll", arguments={"job": "j1"}))
    assert cassette.events[0].outcome is MatchOutcome.EXACT


def test_fuzzy_mode_serves_the_closest_recording_above_threshold() -> None:
    recording = _write_trajectory(
        {"order": "ORD-1001", "amount": 10.0}, {"order": "X", "amount": 1}
    )
    cassette = Cassette(recording, MatchPolicy(mode=MatchMode.FUZZY, threshold=0.8))
    result = cassette.take_tool(
        ToolCall(id="n", name="refund", arguments={"order": "ORD-1001", "amount": 10.5})
    )
    assert result.output == "r0"
    event = cassette.events[0]
    assert event.outcome is MatchOutcome.FUZZY
    assert event.score is not None
    assert 0.8 <= event.score < 1.0
    # The fuzzily consumed entry is gone; an exact request for it now misses.
    with pytest.raises(CassetteMissError, match="closest unconsumed recording is step 1"):
        cassette.take_tool(
            ToolCall(id="m", name="refund", arguments={"order": "ORD-1001", "amount": 10.0}),
            allow_fuzzy=False,
        )


def test_fuzzy_mode_respects_threshold_tool_name_and_opt_out() -> None:
    cassette = Cassette(
        _write_trajectory({"order": "ORD-1001"}), MatchPolicy(mode=MatchMode.FUZZY, threshold=0.99)
    )
    near = ToolCall(id="n", name="refund", arguments={"order": "ORD-1002"})
    with pytest.raises(CassetteMissError, match=r"score 0\.875"):
        cassette.take_tool(near)
    assert cassette.try_take_tool(ToolCall(id="n", name="other", arguments={})) is None

    lenient = Cassette(
        _write_trajectory({"order": "ORD-1001"}), MatchPolicy(mode=MatchMode.FUZZY, threshold=0.5)
    )
    assert lenient.try_take_tool(near, allow_fuzzy=False) is None
    assert lenient.try_take_tool(near) is not None


def test_fifo_skips_entries_consumed_by_fuzzy_matches() -> None:
    recording = _write_trajectory({"order": "a"}, {"order": "a"})
    cassette = Cassette(recording, MatchPolicy(mode=MatchMode.FUZZY, threshold=0.5))
    # Fuzzy takes the earliest similar entry (step 0)...
    assert (
        cassette.take_tool(ToolCall(id="1", name="refund", arguments={"order": "ab"})).output
        == "r0"
    )
    # ...so the exact request is served from step 1, not the consumed step 0.
    assert (
        cassette.take_tool(ToolCall(id="2", name="refund", arguments={"order": "a"})).output == "r1"
    )
    assert cassette.remaining() == {"llm": 0, "tool": 0}
    assert cassette.recorded_side_effect("refund")
    assert not cassette.recorded_side_effect("poll")
