"""Real model providers over HTTP: OpenAI-compatible chat completions and Anthropic messages.

Both are optional. Nothing in the test suite, CI or the demo calls them; they exist so the same
recorder, gate and judge can run against a real model. ``OpenAICompatibleLLM`` covers OpenAI,
Azure OpenAI (``api-key`` header plus ``api-version`` query), Ollama, vLLM and any other server
that speaks ``/chat/completions`` with tools.

Error handling is shaped for ``ResilientLLM``: connection errors, timeouts, 408, 409, 429 and
5xx are retryable ``ProviderError``; other 4xx fail fast. API keys only ever travel in request
headers; they are never logged and never appear in error messages, and response bodies quoted in
errors are truncated.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import httpx
from pydantic import JsonValue

from agent_replay.accounting import DEFAULT_PRICES, PriceTable
from agent_replay.errors import ProviderError
from agent_replay.model import FinishReason, LLMRequest, LLMResponse, Message, ToolCall

INVALID_ARGUMENTS_KEY = "__invalid_json__"
_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 529})
_BODY_IN_ERRORS = 300


def _arguments(raw: Any) -> dict[str, JsonValue]:
    """Tool arguments as a dict. A model that emits malformed JSON is recorded as such (the
    trajectory should show what the model did), not turned into a provider failure."""
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw) if isinstance(raw, str) and raw.strip() else {}
    except json.JSONDecodeError:
        return {INVALID_ARGUMENTS_KEY: str(raw)}
    return parsed if isinstance(parsed, dict) else {INVALID_ARGUMENTS_KEY: str(raw)}


class _HttpProvider:
    name = "http"

    def __init__(
        self,
        *,
        base_url: str,
        headers: Mapping[str, str],
        model: str | None,
        prices: PriceTable,
        timeout_s: float,
        params: Mapping[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._model = model
        self._prices = prices
        self._path_params = dict(params or {})
        self._client = client or httpx.AsyncClient(timeout=timeout_s)
        self._owns_client = client is None
        self._base_url = base_url.rstrip("/")
        self._headers = dict(headers)

    @property
    def provider_name(self) -> str:
        return self.name

    def model_for(self, request: LLMRequest) -> str:
        return self._model or request.model

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self._client.post(
                f"{self._base_url}{path}",
                json=payload,
                headers=self._headers,
                params=self._path_params,
            )
        except httpx.TimeoutException as exc:
            raise ProviderError(f"{self.name}: request timed out", retryable=True) from exc
        except httpx.TransportError as exc:
            raise ProviderError(
                f"{self.name}: transport error ({type(exc).__name__})", retryable=True
            ) from exc
        if response.status_code >= 400:  # noqa: PLR2004
            body = response.text[:_BODY_IN_ERRORS]
            raise ProviderError(
                f"{self.name}: HTTP {response.status_code}: {body}",
                retryable=response.status_code in _RETRYABLE_STATUS,
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(f"{self.name}: response is not JSON", retryable=True) from exc
        if not isinstance(data, dict):
            raise ProviderError(f"{self.name}: unexpected response shape")
        return data


class OpenAICompatibleLLM(_HttpProvider):
    name = "openai-compatible"

    def __init__(
        self,
        *,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        model: str | None = None,
        azure: bool = False,
        api_version: str | None = None,
        prices: PriceTable = DEFAULT_PRICES,
        timeout_s: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        headers: dict[str, str] = {}
        if api_key:
            headers["api-key" if azure else "Authorization"] = (
                api_key if azure else f"Bearer {api_key}"
            )
        super().__init__(
            base_url=base_url,
            headers=headers,
            model=model,
            prices=prices,
            timeout_s=timeout_s,
            params={"api-version": api_version} if api_version else None,
            client=client,
        )
        if azure:
            self.name = "azure-openai"

    @staticmethod
    def _message(message: Message) -> dict[str, Any]:
        if message.role == "tool":
            return {
                "role": "tool",
                "tool_call_id": message.tool_call_id,
                "content": message.content,
            }
        out: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.tool_calls:
            out["content"] = message.content or None
            out["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                }
                for call in message.tool_calls
            ]
        return out

    def build_payload(self, request: LLMRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_for(request),
            "messages": [self._message(m) for m in request.messages],
            "temperature": request.temperature,
        }
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters or {"type": "object", "properties": {}},
                    },
                }
                for t in request.tools
            ]
        return payload

    async def complete(self, request: LLMRequest) -> LLMResponse:
        data = await self._post("/chat/completions", self.build_payload(request))
        try:
            choice = data["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"{self.name}: response has no choices") from exc
        calls = tuple(
            ToolCall(
                id=str(call.get("id") or f"call_{index}"),
                name=str(call["function"]["name"]),
                arguments=_arguments(call["function"].get("arguments")),
            )
            for index, call in enumerate(message.get("tool_calls") or [])
        )
        reason = choice.get("finish_reason")
        finish = (
            FinishReason.TOOL_CALLS
            if calls or reason == "tool_calls"
            else FinishReason.LENGTH
            if reason == "length"
            else FinishReason.STOP
        )
        usage = data.get("usage") or {}
        model = str(data.get("model") or self.model_for(request))
        return LLMResponse(
            model=model,
            content=message.get("content") or "",
            tool_calls=calls,
            finish_reason=finish,
            usage=self._prices.usage(
                model,
                input_tokens=int(usage.get("prompt_tokens", 0)),
                output_tokens=int(usage.get("completion_tokens", 0)),
            ),
        )


class AnthropicLLM(_HttpProvider):
    name = "anthropic"

    def __init__(
        self,
        *,
        api_key: str,
        model: str | None = None,
        base_url: str = "https://api.anthropic.com/v1",
        max_tokens: int = 1024,
        version: str = "2023-06-01",
        prices: PriceTable = DEFAULT_PRICES,
        timeout_s: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            base_url=base_url,
            headers={"x-api-key": api_key, "anthropic-version": version},
            model=model,
            prices=prices,
            timeout_s=timeout_s,
            client=client,
        )
        self._max_tokens = max_tokens

    def build_payload(self, request: LLMRequest) -> dict[str, Any]:
        system = "\n\n".join(m.content for m in request.messages if m.role == "system")
        messages: list[dict[str, Any]] = []
        for message in request.messages:
            if message.role == "system":
                continue
            if message.role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id,
                    "content": message.content,
                }
                # Consecutive tool results belong in one user turn.
                last = messages[-1] if messages else None
                if last and last["role"] == "user" and isinstance(last["content"], list):
                    last["content"].append(block)
                else:
                    messages.append({"role": "user", "content": [block]})
                continue
            if message.role == "assistant" and message.tool_calls:
                content: list[dict[str, Any]] = []
                if message.content:
                    content.append({"type": "text", "text": message.content})
                content.extend(
                    {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                    for c in message.tool_calls
                )
                messages.append({"role": "assistant", "content": content})
                continue
            messages.append({"role": message.role, "content": message.content})
        payload: dict[str, Any] = {
            "model": self.model_for(request),
            "max_tokens": self._max_tokens,
            "messages": messages,
            "temperature": request.temperature,
        }
        if system:
            payload["system"] = system
        if request.tools:
            payload["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters or {"type": "object", "properties": {}},
                }
                for t in request.tools
            ]
        return payload

    async def complete(self, request: LLMRequest) -> LLMResponse:
        data = await self._post("/messages", self.build_payload(request))
        blocks = data.get("content")
        if not isinstance(blocks, list):
            raise ProviderError(f"{self.name}: response has no content")
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        calls = tuple(
            ToolCall(id=str(b["id"]), name=str(b["name"]), arguments=_arguments(b.get("input")))
            for b in blocks
            if b.get("type") == "tool_use"
        )
        stop = data.get("stop_reason")
        finish = (
            FinishReason.TOOL_CALLS
            if calls
            else FinishReason.LENGTH
            if stop == "max_tokens"
            else FinishReason.STOP
        )
        usage = data.get("usage") or {}
        model = str(data.get("model") or self.model_for(request))
        return LLMResponse(
            model=model,
            content=text,
            tool_calls=calls,
            finish_reason=finish,
            usage=self._prices.usage(
                model,
                input_tokens=int(usage.get("input_tokens", 0)),
                output_tokens=int(usage.get("output_tokens", 0)),
            ),
        )
