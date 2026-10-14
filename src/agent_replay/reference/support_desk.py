"""A small customer-support agent with fake tools and a deterministic policy.

It is realistic enough to produce interesting trajectories (lookups, conditional refunds with a
side effect, error handling) while being fully deterministic. ``policy_v1`` is the baseline
behaviour; variants of it are used to seed regressions such as skipping the order lookup before
issuing a refund.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from pydantic import JsonValue

from agent_replay.agent import ToolCallingAgent
from agent_replay.canonical import fingerprint
from agent_replay.model import LLMRequest, Message
from agent_replay.providers.base import LLMProvider
from agent_replay.providers.fake import FakeLLM, FakeTurn, Policy
from agent_replay.tools import Toolbox

AGENT_NAME = "support-desk"
MODEL = "fake-tool-caller-1"
SYSTEM_PROMPT = (
    "You are a support agent for an online store. Look up the order before answering. "
    "Refunds are allowed only for delivered orders within the refund window, for at most the "
    "order total. Never reveal other customers' data."
)
REFUND_WINDOW_DAYS = 30
_ORDER_RE = re.compile(r"\bORD-\d{4}\b")


@dataclass
class OrderStore:
    """In-memory synthetic order data. Mutated by ``issue_refund``."""

    orders: dict[str, dict[str, JsonValue]] = field(
        default_factory=lambda: {
            "ORD-1001": {"status": "delivered", "total": 59.90, "days_since_delivery": 3},
            "ORD-1002": {"status": "shipped", "total": 120.00, "days_since_delivery": None},
            "ORD-1003": {"status": "delivered", "total": 15.50, "days_since_delivery": 45},
        }
    )
    refunds: list[dict[str, JsonValue]] = field(default_factory=list)


def build_toolbox(store: OrderStore) -> Toolbox:
    box = Toolbox()

    def lookup_order(order_id: str) -> JsonValue:
        """Fetch an order's status, total and days since delivery."""
        order = store.orders.get(order_id)
        if order is None:
            raise LookupError(f"order {order_id} not found")
        return {"order_id": order_id, **order}

    def issue_refund(order_id: str, amount: float) -> JsonValue:
        """Refund an amount to the customer's original payment method."""
        refund_id = f"RF-{fingerprint(order_id, amount, length=8)}"
        record: dict[str, JsonValue] = {
            "refund_id": refund_id,
            "order_id": order_id,
            "amount": amount,
        }
        store.refunds.append(record)
        return record

    box.add(
        lookup_order,
        parameters={
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    )
    box.add(
        issue_refund,
        side_effect=True,
        parameters={
            "type": "object",
            "properties": {"order_id": {"type": "string"}, "amount": {"type": "number"}},
            "required": ["order_id", "amount"],
        },
    )
    return box


def _task(request: LLMRequest) -> str:
    return next((m.content for m in request.messages if m.role == "user"), "")


def _tool_outputs(request: LLMRequest, name: str) -> list[Message]:
    return [m for m in request.messages if m.role == "tool" and m.name == name]


def _parse(message: Message) -> dict[str, JsonValue] | None:
    if message.content.startswith("ERROR:"):
        return None
    value = json.loads(message.content)
    return value if isinstance(value, dict) else None


def policy_v1(request: LLMRequest) -> FakeTurn:  # noqa: PLR0911 - a decision table
    """Baseline behaviour: look up, check eligibility, refund only if eligible."""
    task = _task(request)
    match = _ORDER_RE.search(task)
    if match is None:
        return FakeTurn.answer("Please share your order number (format ORD-1234).")
    order_id = match.group(0)
    wants_refund = "refund" in task.lower()

    lookups = _tool_outputs(request, "lookup_order")
    if not lookups:
        return FakeTurn.call("lookup_order", order_id=order_id)
    order = _parse(lookups[-1])
    if order is None:
        return FakeTurn.answer(f"I couldn't find order {order_id}. Please check the number.")

    if not wants_refund:
        return FakeTurn.answer(f"Order {order_id} is currently {order['status']}.")

    refunds = _tool_outputs(request, "issue_refund")
    if refunds:
        refund = _parse(refunds[-1])
        if refund is None:
            return FakeTurn.answer("The refund could not be processed; a human will follow up.")
        return FakeTurn.answer(
            f"Refund {refund['refund_id']} of {refund['amount']} issued for {order_id}."
        )

    days = order.get("days_since_delivery")
    eligible = (
        order.get("status") == "delivered" and isinstance(days, int) and days <= REFUND_WINDOW_DAYS
    )
    if not eligible:
        return FakeTurn.answer(
            f"Order {order_id} is not eligible for a refund "
            f"(status {order['status']}, refund window {REFUND_WINDOW_DAYS} days)."
        )
    total = order["total"]
    assert isinstance(total, int | float)  # noqa: S101 - synthetic data invariant
    return FakeTurn.call("issue_refund", order_id=order_id, amount=float(total))


def policy_skip_lookup(request: LLMRequest) -> FakeTurn:
    """Seeded regression: refunds without looking the order up first."""
    task = _task(request)
    match = _ORDER_RE.search(task)
    if match and "refund" in task.lower() and not _tool_outputs(request, "issue_refund"):
        return FakeTurn.call("issue_refund", order_id=match.group(0), amount=100.0)
    return policy_v1(request)


def build_agent(
    *,
    store: OrderStore | None = None,
    llm: LLMProvider | None = None,
    policy: Policy = policy_v1,
) -> ToolCallingAgent:
    return ToolCallingAgent(
        name=AGENT_NAME,
        llm=llm or FakeLLM(policy, model=MODEL),
        tools=build_toolbox(store or OrderStore()),
        model=MODEL,
        system_prompt=SYSTEM_PROMPT,
    )


SAMPLE_TASKS: tuple[str, ...] = (
    "Hi, where is my order ORD-1002?",
    "I'd like a refund for ORD-1001, it arrived damaged.",
    "Please refund ORD-1003.",
    "Refund ORD-9999 please.",
    "My parcel never came.",
)
