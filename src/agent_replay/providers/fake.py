"""A deterministic fake LLM.

The fake delegates the *decision* to a pure policy function over the request and derives
everything else (tool-call IDs, token usage, cost) deterministically from the request content.
The same request therefore always yields byte-identical responses, which makes the fake suitable
for tests, CI and demos, and lets the eval harness inject controlled regressions by swapping the
policy.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic import JsonValue

from agent_replay.accounting import (
    DEFAULT_PRICES,
    PriceTable,
    estimate_message_tokens,
    estimate_tokens,
)
from agent_replay.canonical import canonical_json, fingerprint
from agent_replay.model import FinishReason, LLMRequest, LLMResponse, ToolCall


@dataclass(frozen=True, slots=True)
class FakeTurn:
    """What the policy decided: either tool calls, or a final text answer."""

    content: str = ""
    tool_calls: tuple[tuple[str, dict[str, JsonValue]], ...] = field(default_factory=tuple)

    @classmethod
    def call(cls, name: str, **arguments: JsonValue) -> FakeTurn:
        return cls(tool_calls=((name, dict(arguments)),))

    @classmethod
    def answer(cls, content: str) -> FakeTurn:
        return cls(content=content)


Policy = Callable[[LLMRequest], FakeTurn]


class FakeLLM:
    def __init__(
        self,
        policy: Policy,
        *,
        model: str = "fake-tool-caller-1",
        prices: PriceTable = DEFAULT_PRICES,
    ) -> None:
        self._policy = policy
        self._model = model
        self._prices = prices
        self.calls = 0

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return self._model

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        turn = self._policy(request)
        request_key = request.key()
        tool_calls = tuple(
            ToolCall(
                id=f"call_{fingerprint(request_key, index, name, args, length=16)}",
                name=name,
                arguments=args,
            )
            for index, (name, args) in enumerate(turn.tool_calls)
        )
        output_tokens = estimate_tokens(turn.content) + sum(
            estimate_tokens(name) + estimate_tokens(canonical_json(args))
            for name, args in turn.tool_calls
        )
        usage = self._prices.usage(
            request.model,
            input_tokens=estimate_message_tokens(request.messages),
            output_tokens=output_tokens,
        )
        return LLMResponse(
            model=request.model,
            content=turn.content,
            tool_calls=tool_calls,
            finish_reason=FinishReason.TOOL_CALLS if tool_calls else FinishReason.STOP,
            usage=usage,
        )
