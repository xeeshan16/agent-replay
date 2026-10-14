"""End-to-end: record the reference agent, replay it, and check spans and side effects."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agent_replay import telemetry as tm
from agent_replay.cli import EXIT_DIVERGED, EXIT_OK, main
from agent_replay.clock import TickingClock
from agent_replay.errors import ProviderError
from agent_replay.model import LLMRequest, LLMResponse, RunStatus
from agent_replay.recorder import record_run
from agent_replay.reference import support_desk
from agent_replay.replay import ReplayMode, replay

pytestmark = pytest.mark.integration
REFUND_TASK = "I'd like a refund for ORD-1001, it arrived damaged."


@pytest.mark.parametrize("task", support_desk.SAMPLE_TASKS)
async def test_full_and_tools_replay_reproduce_the_recording(task: str) -> None:
    recording = await record_run(support_desk.build_agent(), task, clock=TickingClock())
    assert recording.status is RunStatus.SUCCESS

    full = await replay(support_desk.build_agent(), recording, mode=ReplayMode.FULL)
    assert full.identical
    assert full.unconsumed == {"llm": 0, "tool": 0}

    tools = await replay(support_desk.build_agent(), recording, mode=ReplayMode.TOOLS)
    assert tools.identical
    assert tools.replayed.metadata["replay_of"] == recording.trajectory_id


async def test_replay_never_executes_side_effecting_tools() -> None:
    store = support_desk.OrderStore()
    recording = await record_run(support_desk.build_agent(store=store), REFUND_TASK)
    assert len(store.refunds) == 1
    assert [s.side_effect for s in recording.tool_steps] == [False, True]

    replay_store = support_desk.OrderStore()
    result = await replay(support_desk.build_agent(store=replay_store), recording)
    assert result.identical
    assert replay_store.refunds == []


async def test_regressed_agent_diverges_at_the_unrecorded_call() -> None:
    recording = await record_run(support_desk.build_agent(), REFUND_TASK)
    regressed = support_desk.build_agent(policy=support_desk.policy_skip_lookup)
    result = await replay(regressed, recording, mode=ReplayMode.TOOLS)

    assert not result.identical
    assert result.replayed.status is RunStatus.ERROR
    assert result.replayed.error is not None
    assert result.replayed.error.startswith("CassetteMissError")
    assert "issue_refund" in result.replayed.error
    # The run stopped right after the model asked for the unrecorded refund.
    assert len(result.replayed.steps) == 1


async def test_recording_emits_genai_spans(spans: InMemorySpanExporter) -> None:
    recording = await record_run(support_desk.build_agent(), REFUND_TASK)
    finished = spans.get_finished_spans()
    names = [s.name for s in finished]
    assert names == [
        "chat fake-tool-caller-1",
        "execute_tool lookup_order",
        "chat fake-tool-caller-1",
        "execute_tool issue_refund",
        "chat fake-tool-caller-1",
        "invoke_agent support-desk",
    ]
    root = finished[-1]
    assert root.attributes is not None
    assert root.attributes[tm.CONVERSATION_ID] == recording.trajectory_id
    assert all(s.parent is not None for s in finished[:-1])
    assert {s.context.trace_id for s in finished} == {root.context.trace_id}

    chat = finished[0].attributes
    assert chat is not None
    assert chat[tm.OPERATION_NAME] == "chat"
    assert chat[tm.USAGE_INPUT_TOKENS] == recording.llm_steps[0].response.usage.input_tokens
    refund = finished[3].attributes
    assert refund is not None
    assert refund[tm.TOOL_SIDE_EFFECT] is True


class FailingLLM:
    @property
    def provider_name(self) -> str:
        return "failing"

    async def complete(self, request: LLMRequest) -> LLMResponse:
        raise ProviderError("upstream unavailable")


async def test_failed_runs_are_still_recorded() -> None:
    agent = support_desk.build_agent(llm=FailingLLM())
    recording = await record_run(agent, REFUND_TASK)
    assert recording.status is RunStatus.ERROR
    assert recording.error == "ProviderError: upstream unavailable"
    assert recording.steps == ()


def test_cli_record_and_replay_exit_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = tmp_path / "run.json"
    assert main(["record", "--task", REFUND_TASK, "--out", str(run)]) == EXIT_OK
    assert json.loads(capsys.readouterr().out)["tool_calls"] == ["lookup_order", "issue_refund"]

    assert main(["replay", str(run), "--mode", "full"]) == EXIT_OK
    assert json.loads(capsys.readouterr().out)["identical"] is True

    diverged = main(["replay", str(run), "--policy", "skip-lookup", "--out", str(tmp_path / "r")])
    assert diverged == EXIT_DIVERGED
    assert (tmp_path / "r").exists()

    assert main(["demo", "--out-dir", str(tmp_path / "demo")]) == EXIT_OK
    assert len(list((tmp_path / "demo").glob("demo-*.json"))) == len(support_desk.SAMPLE_TASKS)
