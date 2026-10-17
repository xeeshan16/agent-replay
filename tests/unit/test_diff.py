from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import JsonValue

from agent_replay.diff import (
    DiffCosts,
    DiffOp,
    DivergenceReason,
    align,
    diff_trajectories,
    extract_actions,
    json_changes,
)
from agent_replay.matching import MatchEvent, MatchOutcome
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
)
from agent_replay.normalize import Normalizer

T0 = datetime(2026, 10, 17, 9, 0, tzinfo=UTC)
WRITES = {"issue_refund", "create_ticket"}
SPECS = tuple(
    ToolSpec(name=n, description=n, side_effect=n in WRITES)
    for n in ("lookup_order", "get_refund_policy", "issue_refund", "create_ticket")
)

# (tool name, arguments, output) or (tool name, arguments, None, "unexecuted")
Call = tuple[Any, ...]


def build(
    calls: Sequence[Call],
    *,
    answer: str | None = "done",
    status: RunStatus = RunStatus.SUCCESS,
    trajectory_id: str = "t",
    error: str | None = None,
) -> Trajectory:
    """One model step per call (requesting it), then a final answering model step."""
    steps: list[Step] = []
    request = LLMRequest(model="m", messages=(Message(role="user", content="task"),), tools=SPECS)

    def llm(content: str, tool_calls: tuple[ToolCall, ...]) -> None:
        steps.append(
            LLMStep(
                index=len(steps),
                started_at=T0,
                request=request,
                response=LLMResponse(
                    model="m",
                    content=content,
                    tool_calls=tool_calls,
                    finish_reason=FinishReason.TOOL_CALLS if tool_calls else FinishReason.STOP,
                ),
            )
        )

    for n, item in enumerate(calls):
        name, arguments, output = item[0], item[1], item[2]
        call = ToolCall(id=f"call_{trajectory_id}{n:08d}", name=name, arguments=arguments)
        llm("", (call,))
        if len(item) > 3 and item[3] == "unexecuted":
            break
        is_error = isinstance(output, str) and output.startswith("ERROR")
        steps.append(
            ToolStep(
                index=len(steps),
                started_at=T0,
                call=call,
                result=ToolResult(
                    tool_call_id=call.id, name=name, output=output, is_error=is_error
                ),
                side_effect=name in WRITES,
            )
        )
    else:
        if answer is not None:
            llm(answer, ())
    return Trajectory(
        trajectory_id=trajectory_id,
        agent_name="a",
        task_input="task",
        final_output=answer,
        status=status,
        error=error,
        started_at=T0,
        steps=tuple(steps),
    )


LOOKUP: Call = ("lookup_order", {"order_id": "ORD-1"}, {"status": "delivered", "total": 120.0})
POLICY: Call = ("get_refund_policy", {}, {"window_days": 30})
REFUND: Call = ("issue_refund", {"order_id": "ORD-1", "amount": 120.0}, {"refund_id": "RF-1"})


def test_identical_runs_have_no_divergence() -> None:
    diff = diff_trajectories(build([LOOKUP, REFUND]), build([LOOKUP, REFUND]))
    assert diff.identical
    assert diff.first_divergence is None
    assert diff.distance == 0.0
    assert diff.similarity == 1.0
    assert [p.op for p in diff.alignment] == [DiffOp.MATCH, DiffOp.MATCH]


def test_empty_trajectories_are_identical() -> None:
    diff = diff_trajectories(build([]), build([]))
    assert diff.identical
    assert diff.similarity == 1.0


def test_missing_step_points_at_the_decision_that_skipped_it() -> None:
    baseline = build([LOOKUP, REFUND])
    candidate = build([REFUND])
    diff = diff_trajectories(baseline, candidate)
    first = diff.first_divergence
    assert first is not None
    assert first.reason is DivergenceReason.MISSING_STEP
    assert first.tool == "lookup_order"
    assert first.baseline_step == 1
    assert first.candidate_step is None
    # The candidate's first model call is where it should have looked the order up.
    assert first.decision_step == 0
    assert [p.op for p in diff.alignment] == [DiffOp.DELETE, DiffOp.MATCH]


def test_missing_step_later_in_the_run_uses_the_next_model_call() -> None:
    diff = diff_trajectories(build([LOOKUP, POLICY, REFUND]), build([LOOKUP, REFUND]))
    first = diff.first_divergence
    assert first is not None
    assert first.reason is DivergenceReason.MISSING_STEP
    assert first.tool == "get_refund_policy"
    # Candidate steps: 0 llm, 1 lookup, 2 llm (asked for the refund instead of the policy).
    assert first.decision_step == 2


def test_extra_step_and_extra_loop_are_told_apart() -> None:
    diff = diff_trajectories(build([LOOKUP, REFUND]), build([LOOKUP, POLICY, LOOKUP, REFUND]))
    reasons = [d.reason for d in diff.divergences]
    assert reasons == [DivergenceReason.EXTRA_STEP, DivergenceReason.EXTRA_LOOP]
    loop = diff.divergences[1]
    assert loop.tool == "lookup_order"
    assert loop.candidate_step == 5


def test_the_later_of_two_equal_calls_is_the_extra_one() -> None:
    diff = diff_trajectories(build([LOOKUP, REFUND]), build([LOOKUP, LOOKUP, REFUND]))
    assert [p.op for p in diff.alignment] == [DiffOp.MATCH, DiffOp.INSERT, DiffOp.MATCH]
    first = diff.first_divergence
    assert first is not None
    assert first.reason is DivergenceReason.EXTRA_LOOP
    assert first.candidate_step == 3


def test_argument_drift_on_a_write_is_a_substitution_with_paths() -> None:
    drifted = ("issue_refund", {"order_id": "ORD-1", "amount": 121.0}, {"refund_id": "RF-1"})
    diff = diff_trajectories(build([LOOKUP, REFUND]), build([LOOKUP, drifted]))
    first = diff.first_divergence
    assert first is not None
    assert first.reason is DivergenceReason.ARGUMENT_DRIFT
    assert first.side_effect
    assert [(c.path, c.before, c.after) for c in first.changes] == [("amount", 120.0, 121.0)]
    assert diff.alignment[1].op is DiffOp.SUBSTITUTE
    assert 0 < diff.alignment[1].cost < 0.1


def test_wrong_tool_at_the_same_position() -> None:
    ticket = ("create_ticket", {"order_id": "ORD-1"}, {"ticket_id": "T-1"})
    diff = diff_trajectories(build([LOOKUP, REFUND]), build([LOOKUP, ticket]))
    first = diff.first_divergence
    assert first is not None
    assert first.reason is DivergenceReason.WRONG_TOOL
    assert first.summary == "called create_ticket where the baseline called issue_refund"
    assert diff.alignment[1].cost == DiffCosts().rename


def test_output_drift_when_the_same_call_saw_a_different_world() -> None:
    other = ("lookup_order", {"order_id": "ORD-1"}, {"status": "shipped", "total": 120.0})
    diff = diff_trajectories(build([LOOKUP]), build([other]))
    first = diff.first_divergence
    assert first is not None
    assert first.reason is DivergenceReason.OUTPUT_DRIFT
    assert [c.path for c in first.changes] == ["output.status"]


def test_status_and_answer_changes_follow_action_divergences() -> None:
    candidate = build(
        [LOOKUP],
        answer=None,
        status=RunStatus.ERROR,
        error="CassetteMissError: tool miss",
    )
    diff = diff_trajectories(build([LOOKUP, REFUND]), candidate)
    reasons = [d.reason for d in diff.divergences]
    assert reasons == [
        DivergenceReason.MISSING_STEP,
        DivergenceReason.STATUS_CHANGED,
        DivergenceReason.DIFFERENT_ANSWER,
    ]
    assert "CassetteMissError" in diff.divergences[1].summary


def test_answer_only_difference() -> None:
    diff = diff_trajectories(build([LOOKUP], answer="Delivered."), build([LOOKUP], answer="Lost."))
    first = diff.first_divergence
    assert first is not None
    assert first.reason is DivergenceReason.DIFFERENT_ANSWER
    assert first.alignment_index is None
    assert diff.outcome.output_similarity < 1.0


def test_requested_but_unexecuted_call_is_an_action() -> None:
    candidate = build(
        [LOOKUP, (*REFUND[:2], None, "unexecuted")],
        answer=None,
        status=RunStatus.ERROR,
        error="SideEffectBlockedError: unrecorded call",
    )
    actions = extract_actions(candidate)
    assert [a.name for a in actions] == ["lookup_order", "issue_refund"]
    refund = actions[1]
    assert not refund.executed
    assert refund.step_index is None
    assert refund.decision_step == 2
    assert refund.side_effect  # from the offered tool specs
    diff = diff_trajectories(build([LOOKUP, REFUND]), candidate)
    first = diff.first_divergence
    assert first is not None
    assert first.reason is DivergenceReason.OUTPUT_DRIFT
    assert "never executed" in first.summary


def test_normaliser_masks_volatile_values_but_keeps_identity() -> None:
    def refund(key: str, passed: str) -> Call:
        return ("issue_refund", {"key": key, "ref": passed}, {"ok": True})

    a = "0b3c6f0e-8d1a-4f5e-9a39-0c1f7e2d4b6a"
    b = "7e1f2a3b-4c5d-4e6f-8a9b-0c1d2e3f4a5b"
    same_shape = diff_trajectories(build([refund(a, a)]), build([refund(b, b)]))
    assert same_shape.identical
    # Passing back a different ID than the one minted is behaviour, not noise.
    different = diff_trajectories(build([refund(a, a)]), build([refund(b, a)]))
    assert not different.identical
    raw = diff_trajectories(build([refund(a, a)]), build([refund(b, b)]), normalizer=Normalizer())
    assert not raw.identical


def test_provenance_annotates_candidate_actions() -> None:
    candidate = build([LOOKUP, POLICY])
    events = [
        MatchEvent("llm", "m", MatchOutcome.EXACT, 0),
        MatchEvent("tool", "lookup_order", MatchOutcome.NORMALIZED, 1),
        MatchEvent("llm", "m", MatchOutcome.EXACT, 2),
        MatchEvent("tool", "get_refund_policy", MatchOutcome.LIVE),
    ]
    diff = diff_trajectories(build([LOOKUP]), candidate, candidate_events=events)
    extra = diff.first_divergence
    assert extra is not None
    assert extra.reason is DivergenceReason.EXTRA_STEP
    assert extra.summary.endswith("[replay: live]")
    lookup = diff.alignment[0].candidate
    assert lookup is not None
    assert lookup.served is MatchOutcome.NORMALIZED
    assert lookup.served_from == 1


def test_provenance_stops_at_a_name_mismatch() -> None:
    events = [MatchEvent("tool", "something_else", MatchOutcome.EXACT, 1)]
    actions = extract_actions(build([LOOKUP]), events=events)
    assert actions[0].served is None


def test_tool_steps_without_requested_calls_still_become_actions() -> None:
    """Traces ingested without content carry tool steps but no requested calls."""
    full = build([LOOKUP, REFUND])
    stripped_steps: list[Step] = []
    for step in full.steps:
        if isinstance(step, LLMStep):
            response = step.response.model_copy(update={"tool_calls": ()})
            stripped_steps.append(step.model_copy(update={"response": response}))
        else:
            stripped_steps.append(step)
    stripped = full.model_copy(update={"steps": tuple(stripped_steps)})
    actions = extract_actions(stripped)
    assert [a.name for a in actions] == ["lookup_order", "issue_refund"]
    assert [a.decision_step for a in actions] == [0, 2]
    assert diff_trajectories(full, stripped).identical


def test_usage_delta() -> None:
    diff = diff_trajectories(build([LOOKUP]), build([LOOKUP, POLICY]))
    assert diff.usage.tool_calls == 1
    assert diff.usage.steps == 2


def test_report_is_json_serialisable() -> None:
    import json  # noqa: PLC0415

    diff = diff_trajectories(build([LOOKUP, REFUND]), build([REFUND]))
    report = diff.to_report()
    encoded = json.loads(json.dumps(report))
    assert encoded["identical"] is False
    assert encoded["first_divergence"]["reason"] == "missing_step"
    assert encoded["reasons"] == {"missing_step": 1}


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [
        ({"a": 1}, {"a": 1}, []),
        ({"a": 1}, {"a": 2}, [("a", "changed")]),
        ({"a": 1}, {}, [("a", "removed")]),
        ({}, {"b": [1]}, [("b", "added")]),
        (
            {"a": {"b": [1, 2]}},
            {"a": {"b": [1, 3, 4]}},
            [("a.b[1]", "changed"), ("a.b[2]", "added")],
        ),
        ([1, 2], [1], [("[1]", "removed")]),
        (1, "1", [("$", "changed")]),
        (1, 1.0, [("$", "changed")]),
    ],
)
def test_json_changes(before: JsonValue, after: JsonValue, expected: list[tuple[str, str]]) -> None:
    assert [(c.path, c.kind) for c in json_changes(before, after)] == expected


def test_json_changes_limit() -> None:
    before: JsonValue = {str(i): i for i in range(50)}
    after: JsonValue = {str(i): i + 1 for i in range(50)}
    assert len(json_changes(before, after, limit=7)) == 7


def test_cost_validation() -> None:
    with pytest.raises(ValueError, match="rename"):
        DiffCosts(rename=2.0)
    with pytest.raises(ValueError, match="same-tool"):
        DiffCosts(arguments=1.5, output=0.5)
    with pytest.raises(ValueError, match="positive"):
        DiffCosts(insert=0.0)
    with pytest.raises(ValueError, match="non-negative"):
        DiffCosts(output=-0.1)


def test_alignment_size_guard() -> None:
    actions = extract_actions(build([LOOKUP] * 5))
    with pytest.raises(ValueError, match="too long"):
        align(actions, actions, DiffCosts(max_cells=10))


def test_alignment_is_minimal_against_brute_force() -> None:
    """Exhaustively check the DP on small inputs against every possible alignment."""
    import itertools  # noqa: PLC0415
    from functools import cache  # noqa: PLC0415

    pool = [LOOKUP, POLICY, REFUND, ("issue_refund", {"order_id": "ORD-1", "amount": 1.0}, {})]
    costs = DiffCosts()
    for n, m in itertools.product(range(4), repeat=2):
        for left in itertools.product(range(len(pool)), repeat=n):
            for right in list(itertools.product(range(len(pool)), repeat=m))[:12]:
                a = extract_actions(build([pool[k] for k in left]))
                b = extract_actions(build([pool[k] for k in right]))

                @cache
                def best(i: int, j: int, a: Any = a, b: Any = b) -> float:
                    if i == len(a) and j == len(b):
                        return 0.0
                    options = []
                    if i < len(a):
                        options.append(costs.delete + best(i + 1, j))
                    if j < len(b):
                        options.append(costs.insert + best(i, j + 1))
                    if i < len(a) and j < len(b):
                        _, rows = align([a[i]], [b[j]], costs)
                        sub = rows[0][3] if rows[0][0] in {DiffOp.MATCH, DiffOp.SUBSTITUTE} else 2
                        options.append(sub + best(i + 1, j + 1))
                    return min(options)

                distance, rows = align(a, b, costs)
                assert distance == pytest.approx(best(0, 0))
                assert sum(r[3] for r in rows) == pytest.approx(distance)
