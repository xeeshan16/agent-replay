"""The reference agent recorded through the OpenAI-compatible provider against a local mock
server, then replayed: real providers are interchangeable with the fake behind the interface."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from agent_replay.clock import TickingClock
from agent_replay.model import LLMRequest, Message, ToolCall
from agent_replay.providers import OpenAICompatibleLLM
from agent_replay.recorder import record_run
from agent_replay.reference import support_desk as desk
from agent_replay.replay import ReplayMode, replay

pytestmark = pytest.mark.integration


def _server(request: httpx.Request) -> httpx.Response:
    """A chat-completions endpoint whose "model" is the reference decision policy."""
    body: dict[str, Any] = json.loads(request.content)
    names: dict[str, str] = {}
    messages: list[Message] = []
    for m in body["messages"]:
        calls = tuple(
            ToolCall(
                id=c["id"],
                name=c["function"]["name"],
                arguments=json.loads(c["function"]["arguments"]),
            )
            for c in m.get("tool_calls") or []
        )
        names.update({c.id: c.name for c in calls})
        messages.append(
            Message(
                role=m["role"],
                content=m.get("content") or "",
                tool_calls=calls,
                tool_call_id=m.get("tool_call_id"),
                name=names.get(m.get("tool_call_id") or ""),
            )
        )
    turn = desk.policy_v1(
        LLMRequest(model=body["model"], messages=tuple(messages)),
        new_id=desk.seeded_ids("server"),
    )
    tool_calls = [
        {"id": f"srv_{i}", "type": "function",
         "function": {"name": name, "arguments": json.dumps(args)}}
        for i, (name, args) in enumerate(turn.tool_calls)
    ]  # fmt: skip
    message: dict[str, Any] = {"role": "assistant", "content": turn.content or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return httpx.Response(
        200,
        json={
            "model": "mock-1",
            "choices": [
                {"message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10},
        },
    )


async def test_record_and_replay_through_the_http_provider() -> None:
    client = httpx.AsyncClient(transport=httpx.MockTransport(_server))
    llm = OpenAICompatibleLLM(base_url="http://mock/v1", model="mock-1", client=client)
    task = "I'd like a refund for ORD-1001, it arrived damaged."
    recording = await record_run(desk.build_agent(llm=llm), task, clock=TickingClock())

    assert recording.status.value == "success"
    assert [s.call.name for s in recording.tool_steps] == ["lookup_order", "issue_refund"]
    assert recording.llm_steps[0].response.model == "mock-1"
    assert recording.usage.input_tokens == 100 * len(recording.llm_steps)

    # FULL replay serves the recorded model responses: no further HTTP calls are needed.
    offline = OpenAICompatibleLLM(
        base_url="http://unused/v1",
        client=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500))),
    )
    result = await replay(
        desk.build_agent(llm=offline),
        recording,
        mode=ReplayMode.FULL,
        clock=TickingClock(),
        policy=desk.match_policy(),
    )
    assert result.identical
    await client.aclose()
