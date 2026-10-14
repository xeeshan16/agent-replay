"""Token estimation and cost accounting.

Real providers report token usage; the fake provider estimates it with a deterministic
tokenizer so that cost regressions can be exercised in tests without any API key. Prices are
configuration, not constants baked into scoring: the gate compares runs priced with the same
table.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

from pydantic import BaseModel, ConfigDict

from agent_replay.model import Message, Usage

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def estimate_tokens(text: str) -> int:
    """Deterministic token estimate: one token per word or punctuation character."""
    return len(_TOKEN_RE.findall(text))


def estimate_message_tokens(messages: Iterable[Message]) -> int:
    total = 0
    for message in messages:
        # A small fixed overhead per message approximates role/separator tokens.
        total += 4 + estimate_tokens(message.content)
        for call in message.tool_calls:
            total += estimate_tokens(call.name) + estimate_tokens(str(call.arguments))
    return total


class ModelPrice(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_per_mtok: float
    output_per_mtok: float


class PriceTable:
    """USD prices per million tokens, keyed by model name."""

    def __init__(self, prices: Mapping[str, ModelPrice], default: ModelPrice | None = None):
        self._prices = dict(prices)
        self._default = default

    def price_for(self, model: str) -> ModelPrice | None:
        return self._prices.get(model, self._default)

    def usage(self, model: str, input_tokens: int, output_tokens: int) -> Usage:
        price = self.price_for(model)
        cost = 0.0
        if price is not None:
            cost = (
                input_tokens * price.input_per_mtok + output_tokens * price.output_per_mtok
            ) / 1_000_000
        return Usage(
            input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=round(cost, 10)
        )


# Illustrative prices for the fake model so that cost accounting is visible in demos and tests.
# They are not the price of any real model.
DEFAULT_PRICES = PriceTable(
    {"fake-tool-caller-1": ModelPrice(input_per_mtok=0.5, output_per_mtok=1.5)}
)
