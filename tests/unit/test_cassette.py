from __future__ import annotations

from datetime import UTC, datetime

import pytest

from agent_replay.errors import CassetteMissError
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
