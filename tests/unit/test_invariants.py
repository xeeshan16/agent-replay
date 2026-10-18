from __future__ import annotations

import pytest
from pydantic import JsonValue

from agent_replay.diff import Action
from agent_replay.scoring.invariants import (
    AtMost,
    InvariantSyntaxError,
    NoAfterFailure,
    Require,
    evaluate,
    parse_rules,
)


def act(
    name: str,
    args: dict[str, JsonValue] | None = None,
    output: JsonValue = None,
    *,
    error: bool = False,
    executed: bool = True,
    write: bool = False,
) -> Action:
    return Action(
        position=0,
        call_id=f"c-{name}",
        name=name,
        arguments=args or {},
        output=output,
        is_error=error or not executed,
        executed=executed,
        side_effect=write,
    )


def run(*actions: Action) -> tuple[Action, ...]:
    return tuple(
        a.model_copy(update={"position": i, "step_index": 2 * i + 1, "decision_step": 2 * i})
        for i, a in enumerate(actions)
    )


def lookup(order: str, total: float = 59.9, *, error: bool = False, **extra: JsonValue) -> Action:
    output: JsonValue = (
        "order system unavailable" if error else {"order_id": order, "total": total, **extra}
    )
    return act("lookup_order", {"order_id": order}, output, error=error)


def refund(order: str, amount: float, *, executed: bool = True, error: bool = False) -> Action:
    return act(
        "issue_refund",
        {"order_id": order, "amount": amount, "idempotency_key": "k"},
        {"refund_id": "RF-1"} if executed and not error else "refused",
        executed=executed,
        error=error,
        write=True,
    )


RULES = """
# the support desk's refund policy
lookup_before_refund: require lookup_order where ok before issue_refund on order_id
refund_within_total: require lookup_order where ok and output.total >= target.args.amount
    before issue_refund on order_id
bounded_lookups: at_most 2 lookup_order per order_id
one_refund: at_most 1 issue_refund where ok per order_id
no_blind_writes: no write after failed read
"""


def violated(actions: tuple[Action, ...], rules: str = RULES) -> dict[str, list[int]]:
    results = evaluate(parse_rules(rules), actions)
    return {r.rule: [v.position for v in r.violations] for r in results if not r.passed}


def test_parses_statements_and_continuation_lines() -> None:
    rules = parse_rules(RULES)
    assert [r.name for r in rules] == [
        "lookup_before_refund",
        "refund_within_total",
        "bounded_lookups",
        "one_refund",
        "no_blind_writes",
    ]
    assert isinstance(rules[0].statement, Require)
    assert rules[0].statement.on == ("order_id",)
    assert "target.args.amount" in rules[1].source
    assert rules[1].line == 4
    bounded = rules[2].statement
    assert isinstance(bounded, AtMost)
    assert (bounded.limit, bounded.per) == (2, ("order_id",))
    assert isinstance(rules[4].statement, NoAfterFailure)


def test_a_correct_run_passes_every_rule() -> None:
    actions = run(lookup("ORD-1", error=True), lookup("ORD-1"), refund("ORD-1", 59.9))
    assert violated(actions) == {}


def test_refund_without_lookup_is_flagged_even_when_it_never_ran() -> None:
    actions = run(refund("ORD-1", 100.0, executed=False))
    assert violated(actions) == {"lookup_before_refund": [0], "refund_within_total": [0]}
    result = evaluate(parse_rules(RULES), actions)[0]
    assert result.violations[0].executed is False
    assert result.violations[0].decision_step == 0


def test_lookup_of_another_order_does_not_count() -> None:
    actions = run(lookup("ORD-2"), refund("ORD-1", 10.0))
    assert set(violated(actions)) == {"lookup_before_refund", "refund_within_total"}


def test_condition_can_compare_against_the_later_call() -> None:
    assert violated(run(lookup("ORD-1", 59.9), refund("ORD-1", 60.0))) == {
        "refund_within_total": [1]
    }


def test_failed_lookup_does_not_satisfy_where_ok() -> None:
    actions = run(lookup("ORD-1", error=True), refund("ORD-1", 5.0))
    assert set(violated(actions)) == {
        "lookup_before_refund",
        "refund_within_total",
        "no_blind_writes",
    }


def test_at_most_reports_the_call_that_crossed_the_limit_once() -> None:
    actions = run(
        lookup("ORD-1"), lookup("ORD-1"), lookup("ORD-1"), lookup("ORD-1"), lookup("ORD-2")
    )
    assert violated(actions) == {"bounded_lookups": [2]}


def test_at_most_with_condition_counts_only_matching_calls() -> None:
    actions = run(lookup("ORD-1"), refund("ORD-1", 5.0, error=True), refund("ORD-1", 5.0))
    assert violated(actions) == {}
    actions = run(lookup("ORD-1"), refund("ORD-1", 5.0), refund("ORD-1", 5.0))
    assert violated(actions) == {"one_refund": [2]}


def test_a_successful_retry_recovers_from_a_failure() -> None:
    rules = "blind: no write after failed read"
    escalate = act("create_ticket", {"order_id": "ORD-1"}, {"ticket_id": "T"}, write=True)
    assert violated(run(lookup("ORD-1", error=True), escalate), rules) == {"blind": [1]}
    assert violated(run(lookup("ORD-1", error=True), lookup("ORD-1"), escalate), rules) == {}
    # A failed write followed by another write is not a read failure.
    assert violated(run(refund("ORD-1", 5.0, error=True), escalate), rules) == {}


def test_forbid_with_boolean_logic_and_literals() -> None:
    rules = 'big: forbid issue_refund where args.amount > 100 and not (args.order_id == "ORD-9")'
    assert violated(run(refund("ORD-1", 150.0)), rules) == {"big": [0]}
    assert violated(run(refund("ORD-9", 150.0)), rules) == {}
    assert violated(run(refund("ORD-1", 50.0)), rules) == {}


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        ("output.total == 59.9", True),
        ("output.total != 59.9", False),
        ("output.missing == null", False),  # a missing path never compares true
        ("output.missing != 1", False),
        ('output.order_id >= "A"', False),  # ordering is numeric only
        ("output.flags.0 == true", True),
        ("output.flags.5 == true", False),
        ('tool == "lookup_order" or error', True),
        ("executed and not error", True),
    ],
)
def test_condition_semantics(condition: str, expected: bool) -> None:
    actions = run(lookup("ORD-1", flags=[True]))
    result = evaluate(parse_rules(f"r: forbid lookup_order where {condition}"), actions)[0]
    assert result.passed is not expected


def test_selectors_any_read_write() -> None:
    actions = run(lookup("ORD-1"), refund("ORD-1", 5.0))
    rules = "reads: at_most 0 read\nwrites: at_most 0 write\nall: at_most 1 any"
    assert violated(actions, rules) == {"reads": [0], "writes": [1], "all": [1]}


def test_missing_on_field_fails_closed() -> None:
    actions = run(lookup("ORD-1"), act("issue_refund", {"amount": 1.0}, write=True))
    assert violated(actions, "r: require lookup_order before issue_refund on order_id") == {
        "r": [1]
    }


@pytest.mark.parametrize(
    ("text", "message", "line", "column"),
    [
        ("r forbid x", "expected ':'", 1, 3),
        ("r: allow x", "expected 'require', 'forbid', 'at_most' or 'no'", 1, 4),
        ("r: forbid x where", "expected a path", 1, 18),
        ("r: forbid x where args.a >", "expected a path", 1, 27),
        ("r: forbid x where target.args.a == 1", "only meaningful in a require", 1, 19),
        ("r: at_most 1.5 x", "non-negative integer", 1, 12),
        ("r: forbid x extra", "expected end of rule", 1, 13),
        ("r: forbid x where ok.a", "'ok' has no fields", 1, 19),
        ("r: forbid x where 1", "comparison operator", 1, 20),
        ("r: forbid x where args.a == 'y'", "unexpected character", 1, 29),
        ('r: forbid x where args.a == "\\q"', "unknown escape", 1, 29),
        ("r: forbid x\nr: forbid y", "duplicate rule name", 2, 1),
        ("\n\nr: require a before", "expected a tool name", 3, 20),
    ],
)
def test_syntax_errors_point_at_the_problem(
    text: str, message: str, line: int, column: int
) -> None:
    with pytest.raises(InvariantSyntaxError, match=message) as info:
        parse_rules(text)
    assert (info.value.line, info.value.column) == (line, column)


def test_comments_respect_strings() -> None:
    rules = parse_rules('r: forbid x where args.tag == "#1"  # trailing comment')
    assert rules[0].source == 'forbid x where args.tag == "#1"'


def test_render_round_trips_conditions() -> None:
    rule = parse_rules('r: forbid x where not ok or (args.a < 2 and output.b == "c")')[0]
    assert str(rule.statement.condition) == 'not ok or (args.a < 2 and output.b == "c")'  # type: ignore[union-attr]
