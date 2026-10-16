"""The richer reference agent end to end: failure paths, volatile values and the sandbox."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import JsonValue

from agent_replay.clock import TickingClock
from agent_replay.matching import MatchMode, MatchOutcome, MatchPolicy
from agent_replay.model import LLMStep, RunStatus, ToolCall, ToolStep, Trajectory
from agent_replay.normalize import Normalizer
from agent_replay.recorder import record_run
from agent_replay.reference import support_desk as desk
from agent_replay.replay import ReplayMode, replay
from agent_replay.sandbox import SandboxPolicy, UnrecordedWrites

pytestmark = pytest.mark.integration

REFUND_1001 = "I'd like a refund for ORD-1001, it arrived damaged."
REFUND_1004 = "Refund ORD-1004 please, the box was crushed."
REFUND_1005 = "I want a refund for ORD-1005, it's the wrong size."
LATER = datetime(2026, 11, 1, 8, 30, tzinfo=UTC)


async def _record(task: str, store: desk.OrderStore | None = None) -> Trajectory:
    return await record_run(desk.build_agent(store=store), task, clock=TickingClock())


def _outcomes(events: tuple[object, ...]) -> list[str]:
    return [e.outcome.value for e in events]  # type: ignore[attr-defined]


async def test_transient_lookup_failure_is_retried_and_replayed_in_order() -> None:
    recording = await _record(REFUND_1004)
    assert [s.call.name for s in recording.tool_steps] == [
        "lookup_order",
        "lookup_order",
        "issue_refund",
    ]
    first, second = recording.tool_steps[0].result, recording.tool_steps[1].result
    assert first.is_error
    assert first.error_type == "ConnectionError"
    assert not second.is_error
    assert recording.final_output is not None
    assert recording.final_output.startswith("Refund RF-")

    # FULL replay serves the failure first and the success second (FIFO on identical calls).
    full = await replay(desk.build_agent(), recording, mode=ReplayMode.FULL)
    assert full.identical
    assert full.unconsumed == {"llm": 0, "tool": 0}


async def test_refund_over_the_limit_is_escalated_to_a_ticket() -> None:
    store = desk.OrderStore()
    recording = await _record(REFUND_1005, store)
    names = [s.call.name for s in recording.tool_steps]
    assert names == ["lookup_order", "issue_refund", "create_ticket"]
    assert recording.tool_steps[1].result.error_type == "PermissionError"
    assert [s.side_effect for s in recording.tool_steps] == [False, True, True]
    assert store.refunds == {}
    assert len(store.tickets) == 1
    assert recording.final_output is not None
    assert "ticket TCK-" in recording.final_output


async def test_fresh_idempotency_keys_match_only_after_normalisation() -> None:
    recording = await _record(REFUND_1001)
    result = await replay(desk.build_agent(), recording, policy=desk.match_policy())
    assert result.identical
    assert not result.byte_identical  # the live model minted a new idempotency key
    assert _outcomes(result.matches) == ["exact", "normalized"]

    raw = await replay(
        desk.build_agent(), recording, policy=MatchPolicy(normalizer=Normalizer.none())
    )
    assert raw.replayed.status is RunStatus.ERROR
    assert raw.replayed.error is not None
    assert raw.replayed.error.startswith("SideEffectBlockedError")


async def test_full_replay_absorbs_a_different_prompt_time() -> None:
    recording = await _record(REFUND_1001)
    later_agent = desk.build_agent(clock=TickingClock(start=LATER))
    result = await replay(later_agent, recording, mode=ReplayMode.FULL, policy=desk.match_policy())
    assert result.identical
    llm_outcomes = [e.outcome for e in result.matches if e.kind == "llm"]
    assert llm_outcomes == [MatchOutcome.NORMALIZED] * 3

    strict = await replay(
        later_agent,
        recording,
        mode=ReplayMode.FULL,
        policy=MatchPolicy(normalizer=Normalizer.none()),
    )
    assert strict.replayed.status is RunStatus.ERROR
    assert strict.replayed.error is not None
    assert strict.replayed.error.startswith("CassetteMissError: no recorded llm interaction")


def _reverse_tags(recording: Trajectory) -> Trajectory:
    """Simulate a recording made in another process, where the set iterated the other way."""

    def flip(call: ToolCall) -> ToolCall:
        tags = call.arguments.get("tags")
        if not isinstance(tags, list):
            return call
        flipped: list[JsonValue] = list(reversed(tags))
        return call.model_copy(update={"arguments": {**call.arguments, "tags": flipped}})

    steps: list[LLMStep | ToolStep] = []
    for step in recording.steps:
        if isinstance(step, ToolStep):
            steps.append(step.model_copy(update={"call": flip(step.call)}))
        else:
            calls = tuple(flip(c) for c in step.response.tool_calls)
            response = step.response.model_copy(update={"tool_calls": calls})
            steps.append(step.model_copy(update={"response": response}))
    return recording.model_copy(update={"steps": tuple(steps)})


async def test_unordered_ticket_tags_match_only_with_the_agents_normaliser() -> None:
    recording = _reverse_tags(await _record(REFUND_1005))
    ticket = recording.tool_steps[-1].call.arguments["tags"]
    assert isinstance(ticket, list)
    assert ticket != desk._escalation_tags("ORD-1005")

    result = await replay(desk.build_agent(), recording, policy=desk.match_policy())
    assert result.identical
    assert _outcomes(result.matches)[-1] == "normalized"

    generic = await replay(desk.build_agent(), recording, policy=MatchPolicy())
    assert generic.replayed.status is RunStatus.ERROR
    assert _outcomes(generic.matches)[-1] == "blocked"


async def test_benign_new_read_replays_only_with_live_reads() -> None:
    recording = await _record(REFUND_1001)
    changed = desk.build_agent(policy=desk.policy_check_policy)

    strict = await replay(changed, recording, policy=desk.match_policy())
    assert strict.replayed.status is RunStatus.ERROR
    assert strict.replayed.error is not None
    assert "get_refund_policy" in strict.replayed.error

    store = desk.OrderStore()
    live = await replay(
        desk.build_agent(store=store, policy=desk.policy_check_policy),
        recording,
        policy=desk.match_policy(),
        sandbox=SandboxPolicy(live_reads=True),
    )
    assert live.replayed.status is RunStatus.SUCCESS
    assert live.replayed.final_output is not None
    assert live.replayed.final_output.startswith("Refund RF-")
    assert _outcomes(live.matches) == ["exact", "live", "normalized"]
    assert store.refunds == {}  # the refund was served from the cassette, never executed
    assert not live.identical  # an extra step is a behaviour change, even a benign one


async def test_giving_up_on_a_transient_error_is_a_divergence() -> None:
    recording = await _record(REFUND_1004)
    result = await replay(
        desk.build_agent(policy=desk.policy_no_retry), recording, policy=desk.match_policy()
    )
    assert result.replayed.status is RunStatus.SUCCESS
    assert result.replayed.final_output is not None
    assert "unavailable" in result.replayed.final_output
    assert not result.identical
    assert result.unconsumed == {"tool": 2}  # the retry and the refund never happened


@pytest.mark.parametrize("mode", [MatchMode.STRICT, MatchMode.FUZZY])
async def test_amount_drift_on_a_write_is_never_served(mode: MatchMode) -> None:
    recording = await _record(REFUND_1001)
    store = desk.OrderStore()
    result = await replay(
        desk.build_agent(store=store, policy=desk.policy_amount_drift),
        recording,
        policy=desk.match_policy(mode=mode, threshold=0.5),
        sandbox=SandboxPolicy(unrecorded_writes=UnrecordedWrites.BLOCK),
    )
    assert _outcomes(result.matches) == ["exact", "blocked"]
    refund = result.replayed.tool_steps[-1]
    assert refund.call.arguments["amount"] == 60.0
    assert refund.result.error_type == "SideEffectBlockedError"
    assert store.refunds == {}


async def test_refund_tool_is_idempotent_and_validates_amounts() -> None:
    store = desk.OrderStore()
    box = desk.build_toolbox(store)
    args: dict[str, JsonValue] = {"order_id": "ORD-1001", "amount": 59.9, "idempotency_key": "k1"}
    first = await box.execute(ToolCall(id="1", name="issue_refund", arguments=args))
    again = await box.execute(ToolCall(id="2", name="issue_refund", arguments=args))
    assert first.output == again.output
    assert len(store.refunds) == 1

    too_much = {**args, "amount": 61.0, "idempotency_key": "k2"}
    rejected = await box.execute(ToolCall(id="3", name="issue_refund", arguments=too_much))
    assert rejected.error_type == "ValueError"
    policy = await box.execute(ToolCall(id="4", name="get_refund_policy", arguments={}))
    assert isinstance(policy.output, dict)
    assert policy.output["auto_refund_limit"] == desk.AUTO_REFUND_LIMIT


def test_seeded_ids_are_reproducible() -> None:
    a, b = desk.seeded_ids("s"), desk.seeded_ids("s")
    assert [a(), a()] == [b(), b()]
    assert desk.seeded_ids("t")() != desk.seeded_ids("s")()
