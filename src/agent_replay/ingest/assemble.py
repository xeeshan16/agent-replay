"""Assemble trajectories from the spans of one trace.

Every ``invoke_agent`` span becomes one trajectory. Its steps are the ``chat`` and
``execute_tool`` spans whose nearest ``invoke_agent`` ancestor is that span, so nested agents
(an agent calling a sub-agent) produce separate trajectories rather than one interleaved list.

Assembly is a pure function of the span set: running it again on a superset of spans (a late
batch arrived) yields the updated trajectory, which is what makes at-least-once ingest safe.

Spans without content attributes still assemble: steps keep their structure (which tool, which
model, tokens, errors) with empty messages and arguments, and the trajectory is flagged with
``content_captured: false`` so later stages know it cannot be replayed in ``FULL`` mode.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from pydantic import JsonValue

from agent_replay import genai
from agent_replay import telemetry as tm
from agent_replay.ingest.spans import SpanRecord, SpanStatus
from agent_replay.model import (
    FinishReason,
    LLMRequest,
    LLMResponse,
    LLMStep,
    Message,
    RunStatus,
    Step,
    ToolCall,
    ToolResult,
    ToolSpec,
    ToolStep,
    Trajectory,
    Usage,
)

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_STEP_OPERATIONS = frozenset({"chat", "text_completion", "execute_tool"})
_MAX_DEPTH = 256


def _ts(ns: int) -> datetime:
    return _EPOCH + timedelta(microseconds=ns // 1000)


def _duration_ms(span: SpanRecord) -> float:
    return round(max(span.end_ns - span.start_ns, 0) / 1e6, 3)


def _suffix(span: SpanRecord) -> str:
    """``"execute_tool lookup_order"`` -> ``"lookup_order"``."""
    parts = span.name.split(" ", 1)
    return parts[1] if len(parts) == 2 else span.name  # noqa: PLR2004


class _Builder:
    """Accumulates per-trajectory warnings while converting spans."""

    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.content_captured = True

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def llm_step(self, span: SpanRecord, index: int) -> LLMStep:
        model = span.attr_str(tm.REQUEST_MODEL) or span.attr_str(tm.RESPONSE_MODEL) or "unknown"
        messages: tuple[Message, ...] = ()
        tools: tuple[ToolSpec, ...] = ()
        content = ""
        calls: tuple[ToolCall, ...] = ()
        finish: FinishReason | None = None
        history = span.attr_str(tm.INPUT_MESSAGES)
        output = span.attr_str(tm.OUTPUT_MESSAGES)
        if history is None or output is None:
            self.content_captured = False
        try:
            if history is not None:
                messages = genai.decode_input(span.attr_str(tm.SYSTEM_INSTRUCTIONS), history)
            definitions = span.attr_str(tm.TOOL_DEFINITIONS)
            if definitions is not None:
                tools = genai.decode_tool_definitions(definitions)
            if output is not None:
                content, calls, finish = genai.decode_output(output)
        except (genai.ContentDecodeError, ValueError) as exc:
            self.content_captured = False
            self.warn(f"span {span.span_id}: {exc}")
        if finish is None:
            reasons = span.attributes.get(tm.RESPONSE_FINISH_REASONS)
            if isinstance(reasons, list) and reasons:
                finish = genai.parse_finish_reason(reasons[0])
        if finish is None:
            finish = FinishReason.TOOL_CALLS if calls else FinishReason.STOP
        usage = Usage(
            input_tokens=span.attr_int(tm.USAGE_INPUT_TOKENS) or 0,
            output_tokens=span.attr_int(tm.USAGE_OUTPUT_TOKENS) or 0,
            cost_usd=span.attr_float(tm.COST_USD) or 0.0,
        )
        return LLMStep(
            index=index,
            started_at=_ts(span.start_ns),
            duration_ms=_duration_ms(span),
            request=LLMRequest(
                model=model,
                messages=messages,
                tools=tools,
                temperature=span.attr_float(tm.REQUEST_TEMPERATURE) or 0.0,
            ),
            response=LLMResponse(
                model=span.attr_str(tm.RESPONSE_MODEL) or model,
                content=content,
                tool_calls=calls,
                finish_reason=finish,
                usage=usage,
            ),
        )

    def tool_step(self, span: SpanRecord, index: int) -> ToolStep:
        name = span.attr_str(tm.TOOL_NAME) or _suffix(span)
        call_id = span.attr_str(tm.TOOL_CALL_ID) or span.span_id
        arguments: dict[str, JsonValue] = {}
        output: JsonValue = None
        raw_args = span.attr_str(tm.TOOL_CALL_ARGUMENTS)
        raw_result = span.attr_str(tm.TOOL_CALL_RESULT)
        if raw_args is None or raw_result is None:
            self.content_captured = False
        try:
            if raw_args is not None:
                arguments = _arguments(raw_args)
            if raw_result is not None:
                output = genai.loads(raw_result, tm.TOOL_CALL_RESULT)
        except genai.ContentDecodeError as exc:
            self.content_captured = False
            self.warn(f"span {span.span_id}: {exc}")
        is_error = span.status is SpanStatus.ERROR
        side_effect = span.attributes.get(tm.TOOL_SIDE_EFFECT)
        return ToolStep(
            index=index,
            started_at=_ts(span.start_ns),
            duration_ms=_duration_ms(span),
            call=ToolCall(id=call_id, name=name, arguments=arguments),
            result=ToolResult(
                tool_call_id=call_id,
                name=name,
                output=output,
                is_error=is_error,
                error_type=span.attr_str(tm.ERROR_TYPE) if is_error else None,
            ),
            side_effect=side_effect is True,
        )


def _arguments(raw: str) -> dict[str, JsonValue]:
    decoded = genai.loads(raw, tm.TOOL_CALL_ARGUMENTS)
    if not isinstance(decoded, dict):
        raise genai.ContentDecodeError("tool arguments must be a JSON object")
    return decoded


def _owner(
    span: SpanRecord, by_id: dict[str, SpanRecord], roots: dict[str, SpanRecord]
) -> SpanRecord | None:
    """Nearest ``invoke_agent`` ancestor, or ``None`` if the parent chain is broken."""
    current = span.parent_span_id
    for _ in range(_MAX_DEPTH):
        if current is None:
            return None
        if current in roots:
            return roots[current]
        parent = by_id.get(current)
        if parent is None:
            return None
        current = parent.parent_span_id
    return None


def assemble_trace(spans: Sequence[SpanRecord]) -> list[Trajectory]:
    """Build one trajectory per ``invoke_agent`` span in ``spans`` (all from one trace)."""
    if not spans:
        return []
    trace_ids = {s.trace_id for s in spans}
    if len(trace_ids) != 1:
        raise ValueError(f"spans from {len(trace_ids)} traces passed to assemble_trace")
    by_id = {s.span_id: s for s in spans}
    roots = {s.span_id: s for s in spans if s.operation == "invoke_agent"}
    if not roots:
        return []

    owned: dict[str, list[SpanRecord]] = defaultdict(list)
    orphans: list[SpanRecord] = []
    for span in spans:
        if span.operation not in _STEP_OPERATIONS:
            continue
        owner = _owner(span, by_id, roots)
        if owner is None:
            orphans.append(span)
        else:
            owned[owner.span_id].append(span)
    # A broken parent chain usually means an intermediate span was dropped or not exported. With
    # a single agent in the trace the owner is unambiguous; otherwise the step is left out.
    only_root = next(iter(roots)) if len(roots) == 1 else None
    if only_root is not None:
        owned[only_root].extend(orphans)

    trajectories = []
    for root in sorted(roots.values(), key=lambda s: (s.start_ns, s.span_id)):
        builder = _Builder()
        if orphans and only_root is None:
            builder.warn(f"{len(orphans)} step span(s) in this trace have no agent ancestor")
        elif orphans:
            builder.warn(f"{len(orphans)} step span(s) attached through a broken parent chain")
        trajectories.append(_trajectory(root, owned.get(root.span_id, []), builder))
    return trajectories


def _step_order(span: SpanRecord) -> tuple[int, int, str]:
    index = span.attr_int(tm.STEP_INDEX)
    return (span.start_ns, index if index is not None else 1 << 30, span.span_id)


def _trajectory(root: SpanRecord, step_spans: list[SpanRecord], builder: _Builder) -> Trajectory:
    steps: list[Step] = []
    for index, span in enumerate(sorted(step_spans, key=_step_order)):
        if span.operation == "execute_tool":
            steps.append(builder.tool_step(span, index))
        else:
            steps.append(builder.llm_step(span, index))

    task_input = _root_text(root, tm.INPUT_MESSAGES, builder)
    if task_input is None:
        task_input = next(
            (
                m.content
                for s in steps
                if isinstance(s, LLMStep)
                for m in s.request.messages
                if m.role == "user"
            ),
            "",
        )

    if root.status is SpanStatus.ERROR:
        error_type = root.attr_str(tm.ERROR_TYPE)
        status = RunStatus.MAX_STEPS if error_type == RunStatus.MAX_STEPS.value else RunStatus.ERROR
        error: str | None = root.status_message or error_type or "error"
    else:
        status, error = RunStatus.SUCCESS, None

    final_output = _root_text(root, tm.OUTPUT_MESSAGES, builder)
    if final_output is None and status is RunStatus.SUCCESS:
        last = steps[-1] if steps else None
        if isinstance(last, LLMStep) and last.response.finish_reason is FinishReason.STOP:
            final_output = last.response.content

    metadata: dict[str, JsonValue] = {
        "source": "otlp",
        "trace_id": root.trace_id,
        "root_span_id": root.span_id,
        "content_captured": builder.content_captured,
    }
    if root.service_name:
        metadata["service_name"] = root.service_name
    recorded_id = root.attr_str(tm.TRAJECTORY_ID)
    if recorded_id:
        metadata["recorded_trajectory_id"] = recorded_id
    if builder.warnings:
        metadata["ingest_warnings"] = list(builder.warnings)

    return Trajectory(
        trajectory_id=trajectory_id_for(root.trace_id, root.span_id),
        agent_name=root.attr_str(tm.AGENT_NAME) or _suffix(root),
        task_input=task_input,
        final_output=final_output,
        status=status,
        error=error,
        started_at=_ts(root.start_ns),
        duration_ms=_duration_ms(root),
        steps=tuple(steps),
        metadata=metadata,
    )


def _root_text(root: SpanRecord, key: str, builder: _Builder) -> str | None:
    raw = root.attr_str(key)
    if raw is None:
        return None
    try:
        return genai.first_text(raw, key)
    except genai.ContentDecodeError as exc:
        builder.warn(f"span {root.span_id}: {exc}")
        return None


def trajectory_id_for(trace_id: str, root_span_id: str) -> str:
    """Stable ID of an ingested trajectory: the trace plus its ``invoke_agent`` span.

    Keying on span identity (rather than e.g. ``gen_ai.conversation.id``, which can span many
    agent invocations) is what makes a re-delivered or extended trace update the same record.
    """
    return f"{trace_id}-{root_span_id}"
