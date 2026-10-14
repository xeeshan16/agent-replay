"""Record an agent run as a ``Trajectory`` while emitting OpenTelemetry GenAI spans.

The recorder wraps the agent's model provider and tool executor; the agent itself needs no
instrumentation. Each run produces one ``invoke_agent`` span with ``chat`` and ``execute_tool``
child spans, and the same information is captured in the returned trajectory.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable

from opentelemetry.trace import Span, Status, StatusCode
from pydantic import JsonValue

from agent_replay import telemetry as tm
from agent_replay.agent import Agent
from agent_replay.clock import Clock, SystemClock
from agent_replay.logs import get_logger
from agent_replay.model import (
    LLMRequest,
    LLMResponse,
    LLMStep,
    RunStatus,
    Step,
    ToolCall,
    ToolResult,
    ToolSpec,
    ToolStep,
    Trajectory,
)
from agent_replay.providers.base import LLMProvider
from agent_replay.tools import ToolExecutor

_log = get_logger("recorder")


class _StepSink:
    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.steps: list[Step] = []

    @property
    def next_index(self) -> int:
        return len(self.steps)


class RecordingLLM:
    def __init__(self, inner: LLMProvider, sink: _StepSink) -> None:
        self._inner = inner
        self._sink = sink

    @property
    def provider_name(self) -> str:
        return self._inner.provider_name

    async def complete(self, request: LLMRequest) -> LLMResponse:
        clock = self._sink.clock
        index = self._sink.next_index
        started_at, t0 = clock.now(), clock.monotonic()
        with tm.get_tracer().start_as_current_span(f"chat {request.model}") as span:
            span.set_attribute(tm.OPERATION_NAME, "chat")
            span.set_attribute(tm.PROVIDER_NAME, self._inner.provider_name)
            span.set_attribute(tm.REQUEST_MODEL, request.model)
            span.set_attribute(tm.STEP_INDEX, index)
            try:
                response = await self._inner.complete(request)
            except Exception as exc:
                _mark_error(span, exc)
                raise
            span.set_attribute(tm.RESPONSE_MODEL, response.model)
            span.set_attribute(tm.RESPONSE_FINISH_REASONS, [response.finish_reason.value])
            span.set_attribute(tm.USAGE_INPUT_TOKENS, response.usage.input_tokens)
            span.set_attribute(tm.USAGE_OUTPUT_TOKENS, response.usage.output_tokens)
            span.set_attribute(tm.COST_USD, response.usage.cost_usd)
        self._sink.steps.append(
            LLMStep(
                index=index,
                started_at=started_at,
                duration_ms=round((clock.monotonic() - t0) * 1000, 3),
                request=request,
                response=response,
            )
        )
        return response


class RecordingTools:
    def __init__(self, inner: ToolExecutor, sink: _StepSink) -> None:
        self._inner = inner
        self._sink = sink
        self._specs = {spec.name: spec for spec in inner.specs()}

    def specs(self) -> tuple[ToolSpec, ...]:
        return self._inner.specs()

    async def execute(self, call: ToolCall) -> ToolResult:
        clock = self._sink.clock
        index = self._sink.next_index
        spec = self._specs.get(call.name)
        side_effect = spec.side_effect if spec else False
        started_at, t0 = clock.now(), clock.monotonic()
        with tm.get_tracer().start_as_current_span(f"execute_tool {call.name}") as span:
            span.set_attribute(tm.OPERATION_NAME, "execute_tool")
            span.set_attribute(tm.TOOL_NAME, call.name)
            span.set_attribute(tm.TOOL_CALL_ID, call.id)
            span.set_attribute(tm.TOOL_SIDE_EFFECT, side_effect)
            span.set_attribute(tm.STEP_INDEX, index)
            try:
                result = await self._inner.execute(call)
            except Exception as exc:
                _mark_error(span, exc)
                raise
            if result.is_error:
                span.set_attribute(tm.ERROR_TYPE, result.error_type or "ToolError")
                span.set_status(Status(StatusCode.ERROR, "tool returned an error"))
        self._sink.steps.append(
            ToolStep(
                index=index,
                started_at=started_at,
                duration_ms=round((clock.monotonic() - t0) * 1000, 3),
                call=call,
                result=result,
                side_effect=side_effect,
            )
        )
        return result


def _mark_error(span: Span, exc: BaseException) -> None:
    span.record_exception(exc)
    span.set_attribute(tm.ERROR_TYPE, type(exc).__name__)
    span.set_status(Status(StatusCode.ERROR, str(exc)))


async def record_run[A: Agent](
    agent: A,
    task: str,
    *,
    clock: Clock | None = None,
    trajectory_id: str | None = None,
    metadata: dict[str, JsonValue] | None = None,
    id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
) -> Trajectory:
    """Run ``agent`` on ``task`` and return the recorded trajectory.

    Agent-level failures (model errors, cassette misses during replay) are captured in the
    trajectory's status rather than raised, so a failing run is still a complete record.
    """
    clock = clock or SystemClock()
    sink = _StepSink(clock)
    recorded = agent.with_components(
        RecordingLLM(agent.llm, sink), RecordingTools(agent.tools, sink)
    )
    run_id = trajectory_id or id_factory()
    started_at, t0 = clock.now(), clock.monotonic()
    with tm.get_tracer().start_as_current_span(f"invoke_agent {agent.name}") as span:
        span.set_attribute(tm.OPERATION_NAME, "invoke_agent")
        span.set_attribute(tm.AGENT_NAME, agent.name)
        span.set_attribute(tm.CONVERSATION_ID, run_id)
        outcome = await recorded.run(task)
        if outcome.status is not RunStatus.SUCCESS:
            span.set_attribute(tm.ERROR_TYPE, outcome.status.value)
            span.set_status(Status(StatusCode.ERROR, outcome.error or outcome.status.value))
    trajectory = Trajectory(
        trajectory_id=run_id,
        agent_name=agent.name,
        task_input=task,
        final_output=outcome.final_output,
        status=outcome.status,
        error=outcome.error,
        started_at=started_at,
        duration_ms=round((clock.monotonic() - t0) * 1000, 3),
        steps=tuple(sink.steps),
        metadata=metadata or {},
    )
    _log.info(
        "recorded trajectory",
        extra={
            "ctx": {
                "trajectory_id": run_id,
                "agent": agent.name,
                "status": trajectory.status.value,
                "steps": len(trajectory.steps),
                "cost_usd": trajectory.usage.cost_usd,
            }
        },
    )
    return trajectory
