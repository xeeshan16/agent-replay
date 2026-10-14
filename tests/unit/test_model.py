from __future__ import annotations

from datetime import UTC, datetime, timedelta

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
    Usage,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _trajectory(*, run_id: str, call_id: str, offset_s: int = 0) -> Trajectory:
    started = T0 + timedelta(seconds=offset_s)
    call = ToolCall(id=call_id, name="lookup_order", arguments={"order_id": "ORD-1001"})
    request = LLMRequest(model="m", messages=(Message(role="user", content="hi"),))
    return Trajectory(
        trajectory_id=run_id,
        agent_name="a",
        task_input="hi",
        final_output="done",
        status=RunStatus.SUCCESS,
        started_at=started,
        duration_ms=float(offset_s),
        steps=(
            LLMStep(
                index=0,
                started_at=started,
                request=request,
                response=LLMResponse(
                    model="m",
                    tool_calls=(call,),
                    usage=Usage(input_tokens=10, output_tokens=2, cost_usd=0.001),
                ),
            ),
            ToolStep(
                index=1,
                started_at=started,
                call=call,
                result=ToolResult(tool_call_id=call_id, name="lookup_order", output={"ok": True}),
            ),
            LLMStep(
                index=2,
                started_at=started,
                request=request,
                response=LLMResponse(
                    model="m",
                    content="done",
                    usage=Usage(input_tokens=20, output_tokens=1, cost_usd=0.002),
                ),
            ),
        ),
    )


def test_json_round_trip_preserves_step_types() -> None:
    original = _trajectory(run_id="r1", call_id="c1")
    restored = Trajectory.from_json(original.to_json())
    assert restored == original
    assert [type(s) for s in restored.steps] == [LLMStep, ToolStep, LLMStep]


def test_fingerprint_ignores_ids_timestamps_and_durations() -> None:
    first = _trajectory(run_id="r1", call_id="c1")
    second = _trajectory(run_id="r2", call_id="c2", offset_s=60)
    assert first.fingerprint() == second.fingerprint()


def test_fingerprint_changes_when_behaviour_changes() -> None:
    first = _trajectory(run_id="r1", call_id="c1")
    changed = first.model_copy(update={"final_output": "something else"})
    assert first.fingerprint() != changed.fingerprint()


def test_usage_is_summed_over_model_steps() -> None:
    usage = _trajectory(run_id="r1", call_id="c1").usage
    assert usage == Usage(input_tokens=30, output_tokens=3, cost_usd=0.003)
    assert usage.total_tokens == 33
