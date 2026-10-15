from __future__ import annotations

import random

import pytest

from agent_replay import telemetry as tm
from agent_replay.ingest.assemble import assemble_trace, trajectory_id_for
from agent_replay.ingest.spans import SpanRecord, SpanStatus
from agent_replay.model import LLMStep, RunStatus, ToolStep
from agent_replay.reference import support_desk
from agent_replay.replay import ReplayMode, replay
from tests.helpers import record_with_spans, to_records


@pytest.mark.parametrize("task", support_desk.SAMPLE_TASKS)
async def test_spans_with_content_rebuild_a_replayable_trajectory(task: str) -> None:
    recorded, spans = await record_with_spans(task)
    [assembled] = assemble_trace(to_records(spans))

    assert assembled.fingerprint() == recorded.fingerprint()
    assert assembled.metadata["content_captured"] is True
    assert assembled.metadata["recorded_trajectory_id"] == recorded.trajectory_id
    assert assembled.usage == recorded.usage
    # The rebuilt record is a full cassette: replaying the same agent against it is exact.
    result = await replay(support_desk.build_agent(), assembled, mode=ReplayMode.FULL)
    assert result.identical


async def test_assembly_is_independent_of_span_arrival_order() -> None:
    _, spans = await record_with_spans()
    records = to_records(spans)
    shuffled = records[:]
    random.Random(7).shuffle(shuffled)
    assert assemble_trace(shuffled) == assemble_trace(records)


async def test_without_content_the_structure_is_kept_and_flagged() -> None:
    recorded, spans = await record_with_spans(capture_content=False)
    [assembled] = assemble_trace(to_records(spans))

    assert assembled.metadata["content_captured"] is False
    assert [s.call.name for s in assembled.tool_steps] == ["lookup_order", "issue_refund"]
    assert [s.side_effect for s in assembled.tool_steps] == [False, True]
    assert assembled.usage == recorded.usage
    assert assembled.task_input == ""
    assert assembled.status is RunStatus.SUCCESS


async def test_no_agent_span_means_nothing_to_assemble_yet() -> None:
    _, spans = await record_with_spans()
    children = [r for r in to_records(spans) if r.operation != "invoke_agent"]
    assert assemble_trace(children) == []


async def test_error_and_max_steps_statuses_come_from_the_root_span() -> None:
    _, spans = await record_with_spans()
    records = to_records(spans)
    root = next(r for r in records if r.operation == "invoke_agent")
    others = [r for r in records if r is not root]

    errored = root.model_copy(
        update={
            "status": SpanStatus.ERROR,
            "status_message": "model failed",
            "attributes": {**root.attributes, tm.ERROR_TYPE: "error"},
        }
    )
    [traj] = assemble_trace([*others, errored])
    assert (traj.status, traj.error) == (RunStatus.ERROR, "model failed")

    capped = errored.model_copy(
        update={"attributes": {**root.attributes, tm.ERROR_TYPE: "max_steps"}}
    )
    assert assemble_trace([*others, capped])[0].status is RunStatus.MAX_STEPS


def _span(span_id: str, parent: str | None, name: str, start: int, **attrs: object) -> SpanRecord:
    return SpanRecord(
        trace_id="a" * 32,
        span_id=span_id,
        parent_span_id=parent,
        name=name,
        start_ns=start,
        end_ns=start + 1_000_000,
        attributes=attrs,  # type: ignore[arg-type]
    )


def test_nested_agents_produce_separate_trajectories() -> None:
    spans = [
        _span("1" * 16, None, "invoke_agent planner", 0),
        _span("2" * 16, "1" * 16, "execute_tool search", 10),
        _span("3" * 16, "1" * 16, "invoke_agent researcher", 20),
        _span("4" * 16, "3" * 16, "execute_tool fetch", 30),
        _span("5" * 16, "1" * 16, "execute_tool summarise", 40),
    ]
    planner, researcher = assemble_trace(spans)
    assert planner.agent_name == "planner"
    assert [s.call.name for s in planner.tool_steps] == ["search", "summarise"]
    assert researcher.agent_name == "researcher"
    assert [s.call.name for s in researcher.tool_steps] == ["fetch"]
    assert planner.trajectory_id == trajectory_id_for("a" * 32, "1" * 16)


def test_intermediate_non_genai_spans_are_walked_through() -> None:
    spans = [
        _span("1" * 16, "9" * 16, "invoke_agent a", 0),  # parent is an HTTP span, not exported
        _span("2" * 16, "1" * 16, "graph.node lookup", 5),
        _span("3" * 16, "2" * 16, "execute_tool lookup", 10),
    ]
    [traj] = assemble_trace(spans)
    assert [s.call.name for s in traj.tool_steps] == ["lookup"]
    assert "ingest_warnings" not in traj.metadata


def test_broken_parent_chain_with_single_agent_is_attached_with_a_warning() -> None:
    spans = [
        _span("1" * 16, None, "invoke_agent a", 0),
        _span("3" * 16, "8" * 16, "execute_tool lookup", 10),  # parent never arrived
    ]
    [traj] = assemble_trace(spans)
    assert len(traj.tool_steps) == 1
    assert traj.metadata["ingest_warnings"] == [
        "1 step span(s) attached through a broken parent chain"
    ]


def test_truncated_content_attribute_is_a_warning_not_a_failure() -> None:
    spans = [
        _span("1" * 16, None, "invoke_agent a", 0),
        _span(
            "2" * 16,
            "1" * 16,
            "chat m",
            10,
            **{tm.INPUT_MESSAGES: '[{"role":"user","parts":[{"type":"te', tm.OUTPUT_MESSAGES: "[]"},
        ),
    ]
    [traj] = assemble_trace(spans)
    [step] = traj.steps
    assert isinstance(step, LLMStep)
    assert traj.metadata["content_captured"] is False
    warnings = traj.metadata["ingest_warnings"]
    assert isinstance(warnings, list)
    assert "not valid JSON" in str(warnings[0])


def test_tool_error_status_and_order_by_step_index_on_equal_timestamps() -> None:
    spans = [
        _span("1" * 16, None, "invoke_agent a", 0),
        _span("3" * 16, "1" * 16, "execute_tool second", 10, **{tm.STEP_INDEX: 1}),
        _span("2" * 16, "1" * 16, "execute_tool first", 10, **{tm.STEP_INDEX: 0}),
    ]
    spans[1] = spans[1].model_copy(update={"status": SpanStatus.ERROR})
    [traj] = assemble_trace(spans)
    names = [s.call.name for s in traj.steps if isinstance(s, ToolStep)]
    assert names == ["first", "second"]
    assert traj.tool_steps[1].result.is_error


def test_spans_from_several_traces_are_refused() -> None:
    a = _span("1" * 16, None, "invoke_agent a", 0)
    b = a.model_copy(update={"trace_id": "b" * 32})
    with pytest.raises(ValueError, match="2 traces"):
        assemble_trace([a, b])
