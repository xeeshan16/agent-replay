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

from agent_replay.model import Message, Trajectory, Usage

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

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Mapping[str, float]]) -> PriceTable:
        """``{"model": {"input_per_mtok": 0.5, "output_per_mtok": 1.5}, "*": {...}}``; the
        optional ``*`` entry prices models that are not listed."""
        prices = {name: ModelPrice.model_validate(value) for name, value in raw.items()}
        default = prices.pop("*", None)
        return cls(prices, default)

    def merged(self, other: PriceTable) -> PriceTable:
        """This table with ``other``'s entries (and default, if set) taking precedence."""
        return PriceTable({**self._prices, **other._prices}, other._default or self._default)

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


def usage_by_model(trajectory: Trajectory) -> dict[str, Usage]:
    """Recorded usage of each model the run called, as the provider reported it."""
    totals: dict[str, Usage] = {}
    for step in trajectory.llm_steps:
        model = step.response.model or step.request.model
        totals[model] = totals.get(model, Usage()) + step.response.usage
    return dict(sorted(totals.items()))


def reprice(trajectory: Trajectory, prices: PriceTable) -> Usage:
    """The run's token usage priced with ``prices``.

    Recorded costs depend on whichever price table the recording process used, and traces
    ingested from other frameworks usually carry tokens but no cost. Comparing a candidate with
    its baseline is only meaningful when both are priced with the same table, so scoring
    reprices both. Models with no price (and no default) keep their recorded cost.
    """
    total = Usage()
    for step in trajectory.llm_steps:
        usage = step.response.usage
        model = step.response.model or step.request.model
        if prices.price_for(model) is not None:
            usage = prices.usage(model, usage.input_tokens, usage.output_tokens)
        total = total + usage
    return total


# Illustrative prices for the fake models so that cost accounting is visible in demos and tests.
# They are not the price of any real model.
DEFAULT_PRICES = PriceTable(
    {
        "fake-tool-caller-1": ModelPrice(input_per_mtok=0.5, output_per_mtok=1.5),
        "fake-judge-1": ModelPrice(input_per_mtok=1.0, output_per_mtok=4.0),
    }
)
