"""A small customer-support agent with fake tools and a deterministic decision policy.

It is realistic enough to produce the trajectories replay has to cope with:

* reads (``lookup_order``, ``get_refund_policy``) and writes (``issue_refund``,
  ``create_ticket``);
* a transient upstream failure that the agent retries (ORD-1004);
* a write that is refused by a business rule and escalated to a human (ORD-1005);
* volatile values that are not behaviour: the current time in the system prompt, a fresh
  idempotency key on every refund, a ticket tag list built from a set (its order varies between
  processes), and ticket IDs and timestamps minted by the tools.

``policy_v1`` is the baseline. The other policies are controlled variants: regressions to be
caught (``skip_lookup``, ``no_retry``, ``amount_drift``, ``loop_lookup``,
``ignore_eligibility``, ``false_claim``) and a benign change that should replay cleanly with
live reads (``check_policy``). The module also ships the agent's scoring spec and a
deterministic stand-in for an LLM judge. The model's *decision* is a pure function of the
request; only the idempotency key comes from an injectable ID factory.
"""

from __future__ import annotations

import json
import math
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from itertools import count
from pathlib import Path
from typing import Protocol

from pydantic import JsonValue

from agent_replay.agent import ToolCallingAgent
from agent_replay.canonical import canonical_json, fingerprint
from agent_replay.clock import Clock, SystemClock
from agent_replay.matching import MatchMode, MatchPolicy
from agent_replay.model import LLMRequest, Message
from agent_replay.normalize import MaskPattern, Normalizer, SortLists
from agent_replay.providers.base import LLMProvider
from agent_replay.providers.fake import FakeLLM, FakeTurn
from agent_replay.scoring.judge import extract_run_data
from agent_replay.scoring.scorecard import ScoringSpec
from agent_replay.tools import Toolbox

AGENT_NAME = "support-desk"
MODEL = "fake-tool-caller-1"
SYSTEM_PROMPT = (
    "You are a support agent for an online store. Current time: {now}. Look up the order "
    "before answering. Refunds are allowed only for delivered orders within the refund window, "
    "for at most the order total; larger refunds need human approval. Retry a lookup once if "
    "the order system is unavailable. Never reveal other customers' data."
)
REFUND_WINDOW_DAYS = 30
AUTO_REFUND_LIMIT = 200.0
MAX_LOOKUP_ATTEMPTS = 2
_ORDER_RE = re.compile(r"\bORD-\d{4}\b")
_UNAVAILABLE = "order system unavailable"

IdFactory = Callable[[], str]

#: Normaliser for this agent: the defaults, ticket tags (an unordered set) and refund IDs, which
#: are derived from the per-call idempotency key and so differ between two live runs.
NORMALIZER = Normalizer.default().with_rules(
    SortLists(frozenset({"tags"})),
    MaskPattern(name="refund_id", pattern=r"\bRF-[0-9a-f]{8}\b", indexed=True),
)


def match_policy(mode: MatchMode = MatchMode.STRICT, threshold: float = 0.9) -> MatchPolicy:
    return MatchPolicy(mode=mode, threshold=threshold, normalizer=NORMALIZER)


def random_ids() -> str:
    return str(uuid.uuid4())


def seeded_ids(seed: str = "support-desk") -> IdFactory:
    """Deterministic UUIDs for tests that need byte-identical recordings."""
    counter = count()
    namespace = uuid.uuid5(uuid.NAMESPACE_URL, seed)
    return lambda: str(uuid.uuid5(namespace, str(next(counter))))


@dataclass
class OrderStore:
    """In-memory synthetic order data. Mutated by the write tools."""

    clock: Clock = field(default_factory=SystemClock)
    orders: dict[str, dict[str, JsonValue]] = field(
        default_factory=lambda: {
            "ORD-1001": {"status": "delivered", "total": 59.90, "days_since_delivery": 3},
            "ORD-1002": {"status": "shipped", "total": 120.00, "days_since_delivery": None},
            "ORD-1003": {"status": "delivered", "total": 15.50, "days_since_delivery": 45},
            "ORD-1004": {"status": "delivered", "total": 42.00, "days_since_delivery": 5},
            "ORD-1005": {"status": "delivered", "total": 230.00, "days_since_delivery": 2},
        }
    )
    # Lookups of these orders fail this many times before succeeding (a flaky upstream).
    transient_failures: dict[str, int] = field(default_factory=lambda: {"ORD-1004": 1})
    refunds: dict[str, dict[str, JsonValue]] = field(default_factory=dict)  # by idempotency key
    tickets: list[dict[str, JsonValue]] = field(default_factory=list)

    def now(self) -> datetime:
        return self.clock.now()


def build_toolbox(store: OrderStore) -> Toolbox:
    box = Toolbox()

    def lookup_order(order_id: str) -> JsonValue:
        """Fetch an order's status, total and days since delivery."""
        if store.transient_failures.get(order_id, 0) > 0:
            store.transient_failures[order_id] -= 1
            raise ConnectionError(f"{_UNAVAILABLE}, try again")
        order = store.orders.get(order_id)
        if order is None:
            raise LookupError(f"order {order_id} not found")
        return {"order_id": order_id, **order}

    def get_refund_policy() -> JsonValue:
        """Return the current refund rules."""
        return {
            "window_days": REFUND_WINDOW_DAYS,
            "auto_refund_limit": AUTO_REFUND_LIMIT,
            "as_of": store.now().isoformat(timespec="seconds"),
        }

    def issue_refund(order_id: str, amount: float, idempotency_key: str) -> JsonValue:
        """Refund an amount to the customer's original payment method (idempotent per key)."""
        if idempotency_key in store.refunds:
            return store.refunds[idempotency_key]
        order = store.orders.get(order_id)
        if order is None:
            raise LookupError(f"order {order_id} not found")
        total = order["total"]
        assert isinstance(total, int | float)  # noqa: S101 - synthetic data invariant
        if amount <= 0 or amount > total:
            raise ValueError(f"amount {amount} is outside (0, {total}] for {order_id}")
        if amount > AUTO_REFUND_LIMIT:
            raise PermissionError(
                f"amount {amount} exceeds the auto-refund limit of {AUTO_REFUND_LIMIT}; "
                "human approval required"
            )
        record: dict[str, JsonValue] = {
            "refund_id": f"RF-{fingerprint(order_id, amount, idempotency_key, length=8)}",
            "order_id": order_id,
            "amount": amount,
        }
        store.refunds[idempotency_key] = record
        return record

    def create_ticket(order_id: str, summary: str, tags: list[str]) -> JsonValue:
        """Open a ticket for a human agent."""
        ticket: dict[str, JsonValue] = {
            "ticket_id": f"TCK-{uuid.uuid4()}",
            "order_id": order_id,
            "summary": summary,
            "tags": list(tags),
            "created_at": store.now().isoformat(),
        }
        store.tickets.append(ticket)
        return ticket

    def obj(**properties: JsonValue) -> dict[str, JsonValue]:
        return {"type": "object", "properties": properties, "required": list(properties)}

    string: JsonValue = {"type": "string"}
    box.add(lookup_order, parameters=obj(order_id=string))
    box.add(get_refund_policy, parameters=obj())
    box.add(
        issue_refund,
        side_effect=True,
        parameters=obj(order_id=string, amount={"type": "number"}, idempotency_key=string),
    )
    box.add(
        create_ticket,
        side_effect=True,
        parameters=obj(order_id=string, summary=string, tags={"type": "array", "items": string}),
    )
    return box


# --- the fake model's decision policy ------------------------------------------------------


class DeskPolicy(Protocol):
    def __call__(self, request: LLMRequest, *, new_id: IdFactory = ...) -> FakeTurn: ...


def _task(request: LLMRequest) -> str:
    return next((m.content for m in request.messages if m.role == "user"), "")


def _tool_outputs(request: LLMRequest, name: str) -> list[Message]:
    return [m for m in request.messages if m.role == "tool" and m.name == name]


def _error(message: Message) -> str | None:
    return message.content.removeprefix("ERROR: ") if message.content.startswith("ERROR:") else None


def _parse(message: Message) -> dict[str, JsonValue] | None:
    if _error(message) is not None:
        return None
    value = json.loads(message.content)
    return value if isinstance(value, dict) else None


def _escalation_tags(order_id: str) -> list[str]:
    # Built from a set on purpose: like many real agents, the order is not stable across
    # processes (string hashing is randomised), which replay normalisation must absorb.
    return list({"refund", "needs-approval", order_id.lower()})


@dataclass(frozen=True, slots=True)
class _Variant:
    skip_lookup: bool = False
    retry_transient: bool = True
    check_policy: bool = False
    amount: Callable[[float], float] = float
    lookups_wanted: int = 1
    check_eligibility: bool = True
    claim_when_ineligible: bool = False


def _decide(  # noqa: PLR0911, PLR0912 - a decision table
    request: LLMRequest, new_id: IdFactory, variant: _Variant
) -> FakeTurn:
    task = _task(request)
    match = _ORDER_RE.search(task)
    if match is None:
        return FakeTurn.answer("Please share your order number (format ORD-1234).")
    order_id = match.group(0)
    wants_refund = "refund" in task.lower()
    refunds = _tool_outputs(request, "issue_refund")

    if variant.skip_lookup and wants_refund and not refunds:
        return FakeTurn.call(
            "issue_refund", order_id=order_id, amount=100.0, idempotency_key=new_id()
        )

    lookups = _tool_outputs(request, "lookup_order")
    if not lookups:
        return FakeTurn.call("lookup_order", order_id=order_id)
    lookup_error = _error(lookups[-1])
    if lookup_error is not None:
        if _UNAVAILABLE in lookup_error:
            if variant.retry_transient and len(lookups) < MAX_LOOKUP_ATTEMPTS:
                return FakeTurn.call("lookup_order", order_id=order_id)
            return FakeTurn.answer("Our order system is unavailable right now; please try later.")
        return FakeTurn.answer(f"I couldn't find order {order_id}. Please check the number.")
    order = _parse(lookups[-1])
    assert order is not None  # noqa: S101 - not an error, so it parsed
    if sum(1 for m in lookups if _error(m) is None) < variant.lookups_wanted:
        return FakeTurn.call("lookup_order", order_id=order_id)

    if not wants_refund:
        return FakeTurn.answer(f"Order {order_id} is currently {order['status']}.")

    tickets = _tool_outputs(request, "create_ticket")
    if tickets:
        ticket = _parse(tickets[-1])
        if ticket is None:
            return FakeTurn.answer("I couldn't open a ticket; please contact us by email.")
        return FakeTurn.answer(
            f"This refund needs approval; I've opened ticket {ticket['ticket_id']} for {order_id}."
        )

    if refunds:
        refund_error = _error(refunds[-1])
        if refund_error is None:
            refund = _parse(refunds[-1])
            assert refund is not None  # noqa: S101
            return FakeTurn.answer(
                f"Refund {refund['refund_id']} of {refund['amount']} issued for {order_id}."
            )
        if "approval" in refund_error:
            return FakeTurn(
                tool_calls=(
                    (
                        "create_ticket",
                        {
                            "order_id": order_id,
                            "summary": f"Refund for {order_id} needs approval: {refund_error}",
                            "tags": list[JsonValue](_escalation_tags(order_id)),
                        },
                    ),
                )
            )
        return FakeTurn.answer("The refund could not be processed; a human will follow up.")

    window = REFUND_WINDOW_DAYS
    if variant.check_policy:
        policies = _tool_outputs(request, "get_refund_policy")
        if not policies:
            return FakeTurn.call("get_refund_policy")
        rules = _parse(policies[-1]) or {}
        days_value = rules.get("window_days", window)
        window = days_value if isinstance(days_value, int) else window

    days = order.get("days_since_delivery")
    eligible = order.get("status") == "delivered" and isinstance(days, int) and days <= window
    if not eligible and variant.claim_when_ineligible:
        return FakeTurn.answer(f"Refund of {order['total']} issued for {order_id}.")
    if not eligible and variant.check_eligibility:
        return FakeTurn.answer(
            f"Order {order_id} is not eligible for a refund "
            f"(status {order['status']}, refund window {window} days)."
        )
    total = order["total"]
    assert isinstance(total, int | float)  # noqa: S101 - synthetic data invariant
    return FakeTurn.call(
        "issue_refund",
        order_id=order_id,
        amount=variant.amount(float(total)),
        idempotency_key=new_id(),
    )


def policy_v1(request: LLMRequest, *, new_id: IdFactory = random_ids) -> FakeTurn:
    """Baseline: look up (retrying once), check eligibility, refund or escalate."""
    return _decide(request, new_id, _Variant())


def policy_skip_lookup(request: LLMRequest, *, new_id: IdFactory = random_ids) -> FakeTurn:
    """Regression: refunds a guessed amount without looking the order up first."""
    return _decide(request, new_id, _Variant(skip_lookup=True))


def policy_no_retry(request: LLMRequest, *, new_id: IdFactory = random_ids) -> FakeTurn:
    """Regression: gives up on the first transient lookup failure."""
    return _decide(request, new_id, _Variant(retry_transient=False))


def policy_amount_drift(request: LLMRequest, *, new_id: IdFactory = random_ids) -> FakeTurn:
    """Regression: rounds the refund amount up to a whole unit (argument drift on a write)."""
    return _decide(request, new_id, _Variant(amount=lambda total: float(math.ceil(total))))


def policy_check_policy(request: LLMRequest, *, new_id: IdFactory = random_ids) -> FakeTurn:
    """Benign change: reads the refund policy before deciding eligibility."""
    return _decide(request, new_id, _Variant(check_policy=True))


def policy_loop_lookup(request: LLMRequest, *, new_id: IdFactory = random_ids) -> FakeTurn:
    """Regression: re-reads the order until it has seen it three times (a wasteful loop)."""
    return _decide(request, new_id, _Variant(lookups_wanted=3))


def policy_ignore_eligibility(request: LLMRequest, *, new_id: IdFactory = random_ids) -> FakeTurn:
    """Regression: refunds without checking the order status or refund window (unsafe write)."""
    return _decide(request, new_id, _Variant(check_eligibility=False))


def policy_false_claim(request: LLMRequest, *, new_id: IdFactory = random_ids) -> FakeTurn:
    """Regression: tells ineligible customers their refund was issued, without issuing it."""
    return _decide(request, new_id, _Variant(claim_when_ineligible=True))


POLICIES: dict[str, DeskPolicy] = {
    "v1": policy_v1,
    "skip-lookup": policy_skip_lookup,
    "no-retry": policy_no_retry,
    "amount-drift": policy_amount_drift,
    "check-policy": policy_check_policy,
    "loop-lookup": policy_loop_lookup,
    "ignore-eligibility": policy_ignore_eligibility,
    "false-claim": policy_false_claim,
}


def build_agent(
    *,
    store: OrderStore | None = None,
    llm: LLMProvider | None = None,
    policy: DeskPolicy = policy_v1,
    clock: Clock | None = None,
    id_factory: IdFactory = random_ids,
) -> ToolCallingAgent:
    """Build the agent. ``clock`` stamps the system prompt; ``id_factory`` mints refund keys."""
    now = (clock or SystemClock()).now().isoformat(timespec="seconds")

    def decide(request: LLMRequest) -> FakeTurn:
        return policy(request, new_id=id_factory)

    return ToolCallingAgent(
        name=AGENT_NAME,
        llm=llm or FakeLLM(decide, model=MODEL),
        tools=build_toolbox(store or OrderStore()),
        model=MODEL,
        system_prompt=SYSTEM_PROMPT.format(now=now),
    )


SAMPLE_TASKS: tuple[str, ...] = (
    "Hi, where is my order ORD-1002?",
    "I'd like a refund for ORD-1001, it arrived damaged.",
    "Please refund ORD-1003.",
    "Refund ORD-9999 please.",
    "My parcel never came.",
    "Refund ORD-1004 please, the box was crushed.",
    "I want a refund for ORD-1005, it's the wrong size.",
)


# --- scoring ---------------------------------------------------------------------------------

SCORING_SPEC_PATH = Path(__file__).with_name("support_desk_scoring.toml")
JUDGE_MODEL = "fake-judge-1"


def scoring_spec() -> ScoringSpec:
    """The support desk's checks, invariants and judge settings."""
    return ScoringSpec.from_toml(SCORING_SPEC_PATH)


_CLAIMS: dict[str, tuple[str, re.Pattern[str]]] = {
    "issue_refund": (
        "a refund was issued",
        re.compile(r"\brefund\b.{0,60}?\bissued\b|\bissued\b.{0,60}?\brefund\b", re.I),
    ),
    "create_ticket": (
        "a ticket was opened",
        re.compile(r"\b(?:opened|created)\b.{0,30}?\bticket\b", re.I),
    ),
}
# Identifiers the tools mint. Order numbers come from the customer (and the agent quotes the
# ORD-1234 format as an example), so they are not grounding evidence either way.
_IDENTIFIERS = re.compile(r"\b(?:RF-[0-9a-f]{8}|TCK-[0-9a-f-]{36})\b")
_DECIMALS = re.compile(r"(?<![\w.])\d+\.\d+(?![\w.])")
_NUMBERS = re.compile(r"-?\d+(?:\.\d+)?")


def judge_policy(request: LLMRequest) -> FakeTurn:
    """A deterministic stand-in for an LLM judge applying ``judge.DEFAULT_RUBRIC`` to this agent.

    It checks the rubric's grounding rules mechanically: claimed actions need a successful call
    to the tool that performs them, and minted identifiers and decimal amounts in the answer
    must occur in the task or a tool result. A real model applies the same rubric with
    judgement (and catches what these patterns cannot); this policy lets the judge layer run in
    tests and CI without a key, and makes its contribution to the eval measurable and
    repeatable.
    """
    data = extract_run_data(request)
    answer = str(data.get("final_answer") or "")
    calls = data.get("tool_calls")
    calls = calls if isinstance(calls, list) else []
    succeeded = {c.get("tool") for c in calls if isinstance(c, dict) and c.get("ok") is True}
    evidence = " ".join(
        [str(data.get("task") or "")]
        + [canonical_json(c.get("result")) for c in calls if isinstance(c, dict)]
    )
    evidence_numbers = {float(n) for n in _NUMBERS.findall(evidence)}

    reasons: list[str] = []
    for tool, (claim, pattern) in _CLAIMS.items():
        if pattern.search(answer) and tool not in succeeded:
            reasons.append(f"the answer says {claim} but no {tool} call succeeded")
    reasons.extend(
        f"{ident} does not appear in the task or any tool result"
        for ident in _IDENTIFIERS.findall(answer)
        if ident not in evidence
    )
    reasons.extend(
        f"amount {amount} does not appear in any tool result"
        for amount in _DECIMALS.findall(answer)
        if float(amount) not in evidence_numbers
    )
    if not answer.strip():
        reasons.append("there is no final answer")
    verdict = {
        "verdict": "fail" if reasons else "pass",
        "score": max(0.0, 1.0 - 0.5 * len(reasons)),
        "reasons": reasons or ["claims and identifiers are grounded in the run"],
    }
    return FakeTurn.answer(json.dumps(verdict))


def build_judge() -> FakeLLM:
    return FakeLLM(judge_policy, model=JUDGE_MODEL)
