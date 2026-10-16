from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import JsonValue

from agent_replay.errors import CassetteMissError, SideEffectBlockedError
from agent_replay.matching import MatchMode, MatchOutcome, MatchPolicy
from agent_replay.model import RunStatus, ToolCall, ToolResult, ToolStep, Trajectory
from agent_replay.replay import Cassette
from agent_replay.sandbox import (
    BLOCKED_ERROR_TYPE,
    SandboxedTools,
    SandboxPolicy,
    UnrecordedWrites,
)
from agent_replay.tools import Toolbox

T0 = datetime(2026, 1, 1, tzinfo=UTC)


class World:
    """Live tools that count every real execution."""

    def __init__(self, *, refund_is_write: bool = True) -> None:
        self.reads: list[str] = []
        self.writes: list[str] = []
        self.box = Toolbox()

        def lookup(order_id: str) -> JsonValue:
            self.reads.append(order_id)
            return {"order_id": order_id, "live": True}

        def refund(order_id: str, amount: float) -> JsonValue:
            self.writes.append(order_id)
            return {"refunded": amount}

        self.box.add(lookup)
        self.box.add(refund, side_effect=refund_is_write)


def _recording() -> Trajectory:
    def step(index: int, name: str, args: dict[str, JsonValue], *, write: bool) -> ToolStep:
        call = ToolCall(id=f"c{index}", name=name, arguments=args)
        result = ToolResult(tool_call_id=call.id, name=name, output={"recorded": index})
        return ToolStep(index=index, started_at=T0, call=call, result=result, side_effect=write)

    return Trajectory(
        trajectory_id="rec",
        agent_name="a",
        task_input="t",
        status=RunStatus.SUCCESS,
        started_at=T0,
        steps=(
            step(0, "lookup", {"order_id": "A"}, write=False),
            step(1, "refund", {"order_id": "A", "amount": 10.0}, write=True),
        ),
    )


def _tools(
    world: World, *, policy: SandboxPolicy | None = None, match: MatchPolicy | None = None
) -> tuple[SandboxedTools, Cassette]:
    cassette = Cassette(_recording(), match)
    return SandboxedTools(cassette, world.box, policy or SandboxPolicy()), cassette


async def test_recorded_calls_are_served_without_touching_live_tools() -> None:
    world = World()
    tools, cassette = _tools(world)
    lookup = await tools.execute(ToolCall(id="x", name="lookup", arguments={"order_id": "A"}))
    refund = await tools.execute(
        ToolCall(id="y", name="refund", arguments={"order_id": "A", "amount": 10.0})
    )
    assert (lookup.output, refund.output) == ({"recorded": 0}, {"recorded": 1})
    assert refund.tool_call_id == "y"
    assert world.reads == world.writes == []
    assert [e.outcome for e in cassette.events] == [MatchOutcome.EXACT, MatchOutcome.EXACT]
    assert tools.specs() == world.box.specs()


async def test_unrecorded_read_misses_unless_live_reads_are_enabled() -> None:
    world = World()
    tools, _ = _tools(world)
    with pytest.raises(CassetteMissError):
        await tools.execute(ToolCall(id="x", name="lookup", arguments={"order_id": "B"}))

    live, cassette = _tools(world, policy=SandboxPolicy(live_reads=True))
    result = await live.execute(ToolCall(id="x", name="lookup", arguments={"order_id": "B"}))
    assert result.output == {"order_id": "B", "live": True}
    assert world.reads == ["B"]
    assert live.live_calls == 1
    assert cassette.events[-1].outcome is MatchOutcome.LIVE


@pytest.mark.parametrize("live_reads", [False, True])
async def test_unrecorded_write_fails_the_run_and_never_executes(live_reads: bool) -> None:
    world = World()
    tools, cassette = _tools(world, policy=SandboxPolicy(live_reads=live_reads))
    call = ToolCall(id="x", name="refund", arguments={"order_id": "A", "amount": 99.0})
    with pytest.raises(SideEffectBlockedError, match="refund"):
        await tools.execute(call)
    assert world.writes == []
    assert cassette.events[-1].outcome is MatchOutcome.BLOCKED


async def test_block_policy_returns_an_error_result_so_the_run_continues() -> None:
    world = World()
    tools, _ = _tools(world, policy=SandboxPolicy(unrecorded_writes=UnrecordedWrites.BLOCK))
    result = await tools.execute(
        ToolCall(id="x", name="refund", arguments={"order_id": "A", "amount": 99.0})
    )
    assert result.is_error
    assert result.error_type == BLOCKED_ERROR_TYPE
    assert result.tool_call_id == "x"
    assert world.writes == []


async def test_writes_never_match_fuzzily_by_default() -> None:
    world = World()
    drifted = ToolCall(id="x", name="refund", arguments={"order_id": "A", "amount": 10.01})
    fuzzy = MatchPolicy(mode=MatchMode.FUZZY, threshold=0.5)
    tools, _ = _tools(world, match=fuzzy)
    with pytest.raises(SideEffectBlockedError):
        await tools.execute(drifted)

    opted_in, cassette = _tools(
        world, match=MatchPolicy(mode=MatchMode.FUZZY, threshold=0.5, fuzzy_writes=True)
    )
    assert (await opted_in.execute(drifted)).output == {"recorded": 1}
    assert cassette.events[-1].outcome is MatchOutcome.FUZZY
    # Reads do match fuzzily in fuzzy mode.
    read, _ = _tools(world, match=fuzzy)
    near = ToolCall(id="z", name="lookup", arguments={"order_id": "A1"})
    assert (await read.execute(near)).output == {"recorded": 0}
    assert world.writes == []


async def test_a_tool_recorded_as_a_write_stays_a_write_even_if_relabelled() -> None:
    world = World(refund_is_write=False)  # the new agent version "forgot" the flag
    tools, _ = _tools(world, policy=SandboxPolicy(live_reads=True))
    assert tools.is_write("refund")
    with pytest.raises(SideEffectBlockedError):
        await tools.execute(
            ToolCall(id="x", name="refund", arguments={"order_id": "Z", "amount": 1.0})
        )
    assert world.writes == []


async def test_tools_the_agent_does_not_offer_fail_like_they_would_live() -> None:
    world = World()
    tools, cassette = _tools(world)
    result = await tools.execute(ToolCall(id="x", name="delete_account", arguments={}))
    assert result.is_error
    assert result.error_type == "ToolNotFoundError"
    assert cassette.events[-1].outcome is MatchOutcome.UNKNOWN_TOOL
    assert tools.is_write("delete_account")  # no spec: classification fails closed


def test_policy_describes_itself() -> None:
    assert SandboxPolicy().describe() == {"live_reads": False, "unrecorded_writes": "fail"}
