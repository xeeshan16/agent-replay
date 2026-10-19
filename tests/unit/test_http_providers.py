from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from agent_replay.accounting import ModelPrice, PriceTable
from agent_replay.errors import ProviderError
from agent_replay.model import FinishReason, LLMRequest, Message, ToolCall, ToolSpec
from agent_replay.providers import (
    AnthropicLLM,
    LLMProvider,
    OpenAICompatibleLLM,
    ResilientLLM,
    RetryPolicy,
    provider_from_env,
)
from agent_replay.providers.config import ProviderConfigError
from agent_replay.providers.http import INVALID_ARGUMENTS_KEY

Handler = Callable[[httpx.Request], httpx.Response]
SECRET = "sk-test-not-a-real-key"  # noqa: S105 - a fixture value
PRICES = PriceTable({"m-1": ModelPrice(input_per_mtok=1.0, output_per_mtok=2.0)})

REQUEST = LLMRequest(
    model="fake-tool-caller-1",
    messages=(
        Message(role="system", content="You are a support agent."),
        Message(role="user", content="Refund ORD-1."),
        Message(
            role="assistant",
            tool_calls=(
                ToolCall(id="c1", name="lookup_order", arguments={"order_id": "ORD-1"}),
                ToolCall(id="c2", name="get_refund_policy", arguments={}),
            ),
        ),
        Message(role="tool", tool_call_id="c1", name="lookup_order", content='{"ok": true}'),
        Message(role="tool", tool_call_id="c2", name="get_refund_policy", content="30 days"),
    ),
    tools=(
        ToolSpec(
            name="lookup_order",
            description="Look up an order",
            parameters={"type": "object", "properties": {"order_id": {"type": "string"}}},
        ),
    ),
)


def _client(handler: Handler, seen: list[httpx.Request]) -> httpx.AsyncClient:
    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    return httpx.AsyncClient(transport=httpx.MockTransport(record))


def _body(request: httpx.Request) -> dict[str, Any]:
    parsed: dict[str, Any] = json.loads(request.content)
    return parsed


# --- OpenAI-compatible --------------------------------------------------------------------------


async def test_openai_payload_and_tool_call_parsing() -> None:
    seen: list[httpx.Request] = []
    reply = {
        "model": "m-1",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": None,
                    "tool_calls": [
                        {"id": "x1", "function": {"name": "issue_refund",
                                                  "arguments": '{"amount": 5}'}},
                        {"id": "x2", "function": {"name": "lookup_order", "arguments": "{oops"}},
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
    }  # fmt: skip
    llm = OpenAICompatibleLLM(
        base_url="https://llm.test/v1",
        api_key=SECRET,
        model="m-1",
        prices=PRICES,
        client=_client(lambda _: httpx.Response(200, json=reply), seen),
    )
    assert isinstance(llm, LLMProvider)
    response = await llm.complete(REQUEST)

    [request] = seen
    assert str(request.url) == "https://llm.test/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {SECRET}"
    body = _body(request)
    assert body["model"] == "m-1"  # configured model overrides the request's
    assert [m["role"] for m in body["messages"]] == ["system", "user", "assistant", "tool", "tool"]
    call = body["messages"][2]["tool_calls"][0]
    assert call["function"] == {"name": "lookup_order", "arguments": '{"order_id": "ORD-1"}'}
    assert body["tools"][0]["function"]["parameters"]["properties"]["order_id"]
    assert response.finish_reason is FinishReason.TOOL_CALLS
    assert response.tool_calls[0].arguments == {"amount": 5}
    assert response.tool_calls[1].arguments == {INVALID_ARGUMENTS_KEY: "{oops"}
    assert (response.usage.input_tokens, response.usage.output_tokens) == (1000, 500)
    assert response.usage.cost_usd == pytest.approx(0.002)


async def test_azure_uses_api_key_header_and_version() -> None:
    seen: list[httpx.Request] = []
    reply = {"choices": [{"message": {"content": "hi"}, "finish_reason": "length"}]}
    llm = OpenAICompatibleLLM(
        base_url="https://res.openai.azure.com/openai/deployments/d1",
        api_key=SECRET,
        azure=True,
        api_version="2024-10-21",
        client=_client(lambda _: httpx.Response(200, json=reply), seen),
    )
    response = await llm.complete(REQUEST)
    assert llm.provider_name == "azure-openai"
    assert seen[0].headers["api-key"] == SECRET
    assert "authorization" not in seen[0].headers
    assert seen[0].url.params["api-version"] == "2024-10-21"
    assert (response.content, response.finish_reason) == ("hi", FinishReason.LENGTH)
    assert response.model == "fake-tool-caller-1"  # no model configured or returned


# --- Anthropic ------------------------------------------------------------------------------------


async def test_anthropic_payload_and_parsing() -> None:
    seen: list[httpx.Request] = []
    reply = {
        "model": "m-1",
        "content": [
            {"type": "text", "text": "Checking. "},
            {"type": "tool_use", "id": "t1", "name": "issue_refund", "input": {"amount": 5}},
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "output_tokens": 4},
    }
    llm = AnthropicLLM(
        api_key=SECRET,
        model="m-1",
        prices=PRICES,
        client=_client(lambda _: httpx.Response(200, json=reply), seen),
    )
    response = await llm.complete(REQUEST)
    request = seen[0]
    assert request.headers["x-api-key"] == SECRET
    assert request.headers["anthropic-version"] == "2023-06-01"
    body = _body(request)
    assert body["system"] == "You are a support agent."
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]
    # Both tool results are merged into one user turn, in order.
    assert [b["tool_use_id"] for b in body["messages"][2]["content"]] == ["c1", "c2"]
    assert body["messages"][1]["content"][0]["type"] == "tool_use"
    assert body["tools"][0]["input_schema"]["type"] == "object"
    assert response.content == "Checking. "
    assert response.tool_calls == (ToolCall(id="t1", name="issue_refund", arguments={"amount": 5}),)
    assert response.finish_reason is FinishReason.TOOL_CALLS


# --- errors and retries ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "retryable"), [(400, False), (401, False), (408, True), (429, True), (503, True)]
)
async def test_http_errors_are_classified_without_leaking_the_key(
    status: int, retryable: bool
) -> None:
    llm = OpenAICompatibleLLM(
        api_key=SECRET,
        client=_client(lambda _: httpx.Response(status, text="x" * 1000), []),
    )
    with pytest.raises(ProviderError) as info:
        await llm.complete(REQUEST)
    assert info.value.retryable is retryable
    assert SECRET not in str(info.value)
    assert len(str(info.value)) < 400  # bodies are truncated


async def test_transport_errors_are_retryable_and_resilient_wrapper_recovers() -> None:
    calls = {"n": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("refused", request=request)
        if calls["n"] == 2:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    async def no_sleep(_: float) -> None:
        return None

    llm = ResilientLLM(
        OpenAICompatibleLLM(client=_client(flaky, [])),
        RetryPolicy(max_attempts=3),
        sleep=no_sleep,
    )
    assert (await llm.complete(REQUEST)).content == "ok"
    assert calls["n"] == 3


def _replying(status: int, **kwargs: Any) -> httpx.AsyncClient:
    return _client(lambda _: httpx.Response(status, **kwargs), [])


async def test_malformed_responses_are_provider_errors() -> None:
    replies: list[dict[str, Any]] = [{"choices": []}, {"nope": 1}]
    for reply in replies:
        llm = OpenAICompatibleLLM(client=_replying(200, json=reply))
        with pytest.raises(ProviderError, match="no choices"):
            await llm.complete(REQUEST)
    text = OpenAICompatibleLLM(client=_replying(200, text="<html>"))
    with pytest.raises(ProviderError, match="not JSON"):
        await text.complete(REQUEST)
    anthropic = AnthropicLLM(api_key="k", client=_replying(200, json={}))
    with pytest.raises(ProviderError, match="no content"):
        await anthropic.complete(REQUEST)


# --- configuration --------------------------------------------------------------------------------


def test_provider_from_env() -> None:
    assert provider_from_env({}) is None
    assert provider_from_env({"AGENT_REPLAY_LLM_PROVIDER": "fake"}) is None
    with pytest.raises(ProviderConfigError, match="expected one of"):
        provider_from_env({"AGENT_REPLAY_LLM_PROVIDER": "gpt"})
    with pytest.raises(ProviderConfigError, match="OPENAI_API_KEY"):
        provider_from_env({"AGENT_REPLAY_LLM_PROVIDER": "openai"})
    with pytest.raises(ProviderConfigError, match="BASE_URL"):
        provider_from_env({"AGENT_REPLAY_LLM_PROVIDER": "azure", "AZURE_OPENAI_API_KEY": "k"})

    anthropic = provider_from_env(
        {"AGENT_REPLAY_LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "k"}
    )
    assert isinstance(anthropic, ResilientLLM)
    assert anthropic.provider_name == "anthropic"
    ollama = provider_from_env({"AGENT_REPLAY_LLM_PROVIDER": "ollama"})  # no key needed
    assert ollama is not None
    assert ollama.provider_name == "openai-compatible"
