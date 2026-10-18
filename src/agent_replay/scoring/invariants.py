"""A small rule language for behavioural invariants over an agent's tool calls.

Deterministic checks look at a run as a whole (status, budgets); invariants look at the *order
and content* of what the agent did, which is where the regressions that matter for
tool-calling agents live: acting before looking, acting on data it never fetched, looping. The
language is deliberately small, so a rule file reads like a policy and can be reviewed by
someone who does not write Python::

    # name: statement
    lookup_before_refund: require lookup_order where ok before issue_refund on order_id
    refund_within_total:  require lookup_order where ok and output.total >= target.args.amount
                          before issue_refund on order_id
    bounded_lookups:      at_most 2 lookup_order per order_id
    one_refund:           at_most 1 issue_refund where ok per order_id
    no_blind_writes:      no write after failed read
    no_deletes:           forbid delete_account

Statements (``SEL`` is a tool name, ``any``, ``read`` or ``write``):

``require SEL_A [where COND] before SEL_B [on FIELD, ...]``
    Every call matching ``SEL_B`` must be preceded by a call matching ``SEL_A`` that satisfies
    ``COND`` and has equal argument values on every ``FIELD``. ``COND`` may refer to the later
    call as ``target`` (``target.args.amount``).
``forbid SEL [where COND]``
    No call may match.
``at_most N SEL [where COND] [per FIELD, ...]``
    At most ``N`` matching calls, counted per distinct value of the fields.
``no SEL_A after failed SEL_B``
    After a call matching ``SEL_B`` fails, no call matching ``SEL_A`` until a later call to
    the same tool as the failure succeeds (a successful retry recovers).

Conditions combine comparisons (``== != < <= > >=``) of paths and literals with ``and``,
``or``, ``not`` and parentheses. Paths start at ``args``, ``output``, ``ok`` (executed without
error), ``executed``, ``error`` or ``tool``; ``target.`` prefixes refer to the later call in a
``require``. A path that does not resolve compares false, so a rule fails closed when the data
it needs is missing. Ordering comparisons apply to numbers only.

Rules are evaluated on every tool call the model *requested*, including calls that never ran
(a write the replay sandbox blocked is still an attempted write).
"""

from __future__ import annotations

import re
import textwrap
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue

from agent_replay.canonical import canonical_json
from agent_replay.diff import Action, extract_actions
from agent_replay.model import Trajectory
from agent_replay.normalize import Normalizer


class InvariantSyntaxError(ValueError):
    """A rule file that does not parse; carries the 1-based line and column."""

    def __init__(self, message: str, line: int, column: int) -> None:
        super().__init__(f"line {line}, column {column}: {message}")
        self.line = line
        self.column = column


# --------------------------------------------------------------------------------------------
# Tokens
# --------------------------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"""
    (?P<ws>[ \t]+)
  | (?P<number>-?\d+(?:\.\d+)?)
  | (?P<string>"(?:[^"\\]|\\.)*")
  | (?P<op>==|!=|<=|>=|<|>)
  | (?P<punct>[(),:.])
  | (?P<ident>[A-Za-z_][A-Za-z0-9_\-]*)
    """,
    re.VERBOSE,
)
_ESCAPES = {'"': '"', "\\": "\\", "n": "\n", "t": "\t"}


@dataclass(frozen=True, slots=True)
class _Token:
    kind: Literal["number", "string", "op", "punct", "ident", "end"]
    text: str
    line: int
    column: int


def _tokenize(text: str, line: int) -> list[_Token]:
    tokens: list[_Token] = []
    pos = 0
    while pos < len(text):
        match = _TOKEN_RE.match(text, pos)
        if match is None:
            raise InvariantSyntaxError(f"unexpected character {text[pos]!r}", line, pos + 1)
        kind = match.lastgroup
        assert kind is not None  # noqa: S101 - every alternative is a named group
        if kind != "ws":
            tokens.append(_Token(kind, match.group(), line, pos + 1))  # type: ignore[arg-type]
        pos = match.end()
    tokens.append(_Token("end", "", line, len(text) + 1))
    return tokens


def _unquote(token: _Token) -> str:
    body = token.text[1:-1]
    out: list[str] = []
    chars = iter(body)
    for char in chars:
        if char == "\\":
            escaped = next(chars)
            if escaped not in _ESCAPES:
                raise InvariantSyntaxError(f"unknown escape \\{escaped}", token.line, token.column)
            out.append(_ESCAPES[escaped])
        else:
            out.append(char)
    return "".join(out)


# --------------------------------------------------------------------------------------------
# AST
# --------------------------------------------------------------------------------------------

_ROOTS = frozenset({"args", "output", "ok", "executed", "error", "tool"})
_KEYWORDS = frozenset(
    {"require", "where", "before", "on", "forbid", "at_most", "per", "no", "after", "failed",
     "and", "or", "not", "true", "false", "null", "target"}
)  # fmt: skip
CompareOp = Literal["==", "!=", "<", "<=", ">", ">="]


@dataclass(frozen=True, slots=True)
class Selector:
    """A tool name, or ``any`` / ``read`` / ``write``."""

    name: str

    def matches(self, action: Action) -> bool:
        if self.name == "any":
            return True
        if self.name == "write":
            return action.side_effect
        if self.name == "read":
            return not action.side_effect
        return action.name == self.name

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class Path:
    target: bool  # resolve against the later call of a ``require``
    root: str
    keys: tuple[str, ...]

    def __str__(self) -> str:
        return ".".join((*(("target",) if self.target else ()), self.root, *self.keys))


@dataclass(frozen=True, slots=True)
class Const:
    value: JsonValue

    def __str__(self) -> str:
        return canonical_json(self.value)


Operand = Path | Const


@dataclass(frozen=True, slots=True)
class Compare:
    left: Operand
    op: CompareOp | None  # None: the operand must be exactly ``true``
    right: Operand | None = None

    def __str__(self) -> str:
        return str(self.left) if self.op is None else f"{self.left} {self.op} {self.right}"


@dataclass(frozen=True, slots=True)
class Not:
    inner: Condition

    def __str__(self) -> str:
        return f"not {self.inner}"


@dataclass(frozen=True, slots=True)
class BoolOp:
    op: Literal["and", "or"]
    parts: tuple[Condition, ...]

    def __str__(self) -> str:
        return f" {self.op} ".join(
            f"({p})" if isinstance(p, BoolOp) else str(p) for p in self.parts
        )


Condition = Compare | Not | BoolOp


@dataclass(frozen=True, slots=True)
class Require:
    prior: Selector
    condition: Condition | None
    subject: Selector
    on: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Forbid:
    subject: Selector
    condition: Condition | None


@dataclass(frozen=True, slots=True)
class AtMost:
    limit: int
    subject: Selector
    condition: Condition | None
    per: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NoAfterFailure:
    subject: Selector
    failed: Selector


Statement = Require | Forbid | AtMost | NoAfterFailure


@dataclass(frozen=True, slots=True)
class Rule:
    name: str
    statement: Statement
    source: str
    line: int


# --------------------------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------------------------


class _Parser:
    def __init__(self, tokens: list[_Token]) -> None:
        self._tokens = tokens
        self._pos = 0
        self.in_require = False

    @property
    def peek(self) -> _Token:
        return self._tokens[self._pos]

    def _error(self, message: str, token: _Token | None = None) -> InvariantSyntaxError:
        token = token or self.peek
        found = "end of rule" if token.kind == "end" else repr(token.text)
        return InvariantSyntaxError(f"{message}, found {found}", token.line, token.column)

    def advance(self) -> _Token:
        token = self.peek
        self._pos += 1
        return token

    def accept(self, text: str) -> bool:
        if self.peek.text == text and self.peek.kind in {"ident", "punct", "op"}:
            self._pos += 1
            return True
        return False

    def expect(self, text: str) -> _Token:
        if not self.accept(text):
            raise self._error(f"expected {text!r}")
        return self._tokens[self._pos - 1]

    def name(self, what: str) -> str:
        token = self.peek
        if token.kind != "ident" or token.text in _KEYWORDS:
            raise self._error(f"expected {what}")
        self._pos += 1
        return token.text

    def end(self) -> None:
        if self.peek.kind != "end":
            raise self._error("expected end of rule")

    # statements ---------------------------------------------------------------------------

    def statement(self) -> Statement:
        if self.accept("require"):
            prior = self.selector()
            self.in_require = True
            condition = self.condition() if self.accept("where") else None
            self.in_require = False
            self.expect("before")
            subject = self.selector()
            on = self.fields() if self.accept("on") else ()
            return Require(prior, condition, subject, on)
        if self.accept("forbid"):
            subject = self.selector()
            return Forbid(subject, self.condition() if self.accept("where") else None)
        if self.accept("at_most"):
            token = self.advance()
            if token.kind != "number" or not token.text.isdigit():
                raise self._error("expected a non-negative integer", token)
            subject = self.selector()
            condition = self.condition() if self.accept("where") else None
            per = self.fields() if self.accept("per") else ()
            return AtMost(int(token.text), subject, condition, per)
        if self.accept("no"):
            subject = self.selector()
            self.expect("after")
            self.expect("failed")
            return NoAfterFailure(subject, self.selector())
        raise self._error("expected 'require', 'forbid', 'at_most' or 'no'")

    def selector(self) -> Selector:
        return Selector(self.name("a tool name, 'any', 'read' or 'write'"))

    def fields(self) -> tuple[str, ...]:
        names = [self.name("an argument name")]
        while self.accept(","):
            names.append(self.name("an argument name"))
        return tuple(names)

    # conditions ---------------------------------------------------------------------------

    def condition(self) -> Condition:
        parts = [self.conjunction()]
        while self.accept("or"):
            parts.append(self.conjunction())
        return parts[0] if len(parts) == 1 else BoolOp("or", tuple(parts))

    def conjunction(self) -> Condition:
        parts = [self.unary()]
        while self.accept("and"):
            parts.append(self.unary())
        return parts[0] if len(parts) == 1 else BoolOp("and", tuple(parts))

    def unary(self) -> Condition:
        if self.accept("not"):
            return Not(self.unary())
        if self.accept("("):
            inner = self.condition()
            self.expect(")")
            return inner
        left = self.operand()
        if self.peek.kind == "op":
            op: CompareOp = self.advance().text  # type: ignore[assignment]
            return Compare(left, op, self.operand())
        if isinstance(left, Const):
            raise self._error("expected a comparison operator after a literal")
        return Compare(left, None)

    def operand(self) -> Operand:
        token = self.peek
        if token.kind == "number":
            self.advance()
            number = float(token.text)
            return Const(int(number) if number.is_integer() and "." not in token.text else number)
        if token.kind == "string":
            self.advance()
            return Const(_unquote(token))
        if token.kind == "ident" and token.text in {"true", "false", "null"}:
            self.advance()
            return Const({"true": True, "false": False, "null": None}[token.text])
        return self.path()

    def path(self) -> Path:
        token = self.peek
        target = False
        if self.accept("target"):
            if not self.in_require:
                raise self._error("'target' is only meaningful in a require condition", token)
            target = True
            self.expect(".")
            token = self.peek
        if token.kind != "ident" or token.text not in _ROOTS:
            raise self._error(f"expected a path starting with one of {', '.join(sorted(_ROOTS))}")
        self.advance()
        keys: list[str] = []
        while self.accept("."):
            key = self.peek
            if key.kind not in {"ident", "number"}:
                raise self._error("expected a key after '.'")
            self.advance()
            keys.append(key.text)
        if keys and token.text not in {"args", "output"}:
            raise self._error(f"'{token.text}' has no fields", token)
        return Path(target, token.text, tuple(keys))


def _logical_lines(text: str) -> Iterator[tuple[int, str]]:
    """Rules, with continuation lines (indented, no ``name:``) joined to the rule above."""
    start, buffer = 0, ""
    for number, raw in enumerate(textwrap.dedent(text).splitlines(), start=1):
        line = _strip_comment(raw)
        if not line.strip():
            continue
        if line[0] in " \t" and buffer:
            buffer += " " + line.strip()
            continue
        if buffer:
            yield start, buffer
        start, buffer = number, line.strip()
    if buffer:
        yield start, buffer


def _strip_comment(raw: str) -> str:
    quoted = False
    for index, char in enumerate(raw):
        if char == '"' and (index == 0 or raw[index - 1] != "\\"):
            quoted = not quoted
        elif char == "#" and not quoted:
            return raw[:index].rstrip()
    return raw.rstrip()


def parse_rules(text: str) -> tuple[Rule, ...]:
    """Parse a rule file. Raises ``InvariantSyntaxError`` with the position of the problem."""
    rules: list[Rule] = []
    seen: set[str] = set()
    for line, source in _logical_lines(text):
        parser = _Parser(_tokenize(source, line))
        name = parser.name("a rule name")
        if name in seen:
            raise InvariantSyntaxError(f"duplicate rule name {name!r}", line, 1)
        parser.expect(":")
        statement = parser.statement()
        parser.end()
        seen.add(name)
        rules.append(Rule(name, statement, source.split(":", 1)[1].strip(), line))
    return tuple(rules)


# --------------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------------

_MISSING = object()


def _resolve(path: Path, action: Action, target: Action | None) -> object:
    subject = target if path.target else action
    if subject is None:
        return _MISSING
    flags: dict[str, object] = {
        "ok": subject.executed and not subject.is_error,
        "executed": subject.executed,
        "error": subject.is_error,
        "tool": subject.name,
    }
    if path.root in flags:
        return flags[path.root]
    value: object = subject.arguments if path.root == "args" else subject.output
    for key in path.keys:
        if isinstance(value, Mapping) and key in value:
            value = value[key]
        elif isinstance(value, list) and key.isdigit() and int(key) < len(value):
            value = value[int(key)]
        else:
            return _MISSING
    return value


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _compare(left: object, op: CompareOp, right: object) -> bool:  # noqa: PLR0911
    if left is _MISSING or right is _MISSING:
        return False
    if op in {"==", "!="}:
        a, b = _number(left), _number(right)
        equal = a == b if a is not None and b is not None else left == right
        return equal if op == "==" else not equal
    a, b = _number(left), _number(right)
    if a is None or b is None:
        return False
    if op == "<":
        return a < b
    if op == "<=":
        return a <= b
    if op == ">":
        return a > b
    return a >= b


def _holds(condition: Condition | None, action: Action, target: Action | None = None) -> bool:
    if condition is None:
        return True
    if isinstance(condition, Not):
        return not _holds(condition.inner, action, target)
    if isinstance(condition, BoolOp):
        results = (_holds(part, action, target) for part in condition.parts)
        return all(results) if condition.op == "and" else any(results)

    def value(operand: Operand | None) -> object:
        if isinstance(operand, Const):
            return operand.value
        assert operand is not None  # noqa: S101 - comparisons always carry a right operand
        return _resolve(operand, action, target)

    if condition.op is None:
        return value(condition.left) is True
    return _compare(value(condition.left), condition.op, value(condition.right))


def _key(action: Action, fields: Sequence[str]) -> tuple[str, ...] | None:
    args = action.arguments if isinstance(action.arguments, Mapping) else {}
    if any(name not in args for name in fields):
        return None
    return tuple(canonical_json(args[name]) for name in fields)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Violation(_Frozen):
    rule: str
    message: str
    tool: str
    position: int  # in the run's action list
    step_index: int | None = None  # the tool step, when it ran
    decision_step: int | None = None  # the model step that requested the call
    executed: bool = True


class RuleResult(_Frozen):
    rule: str
    source: str
    checked: int  # calls the rule applied to
    violations: tuple[Violation, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.violations


def _violation(rule: Rule, action: Action, message: str) -> Violation:
    return Violation(
        rule=rule.name,
        message=message,
        tool=action.name,
        position=action.position,
        step_index=action.step_index,
        decision_step=action.decision_step,
        executed=action.executed,
    )


def _check_require(rule: Rule, stmt: Require, actions: Sequence[Action]) -> RuleResult:
    checked, violations = 0, []
    for target in actions:
        if not stmt.subject.matches(target):
            continue
        checked += 1
        key = _key(target, stmt.on)
        satisfied = key is not None and any(
            stmt.prior.matches(prior)
            and _key(prior, stmt.on) == key
            and _holds(stmt.condition, prior, target)
            for prior in actions[: target.position]
        )
        if not satisfied:
            scope = f" with the same {', '.join(stmt.on)}" if stmt.on else ""
            where = f" where {stmt.condition}" if stmt.condition else ""
            violations.append(
                _violation(
                    rule, target, f"{target.name} without a prior {stmt.prior}{where}{scope}"
                )
            )
    return RuleResult(
        rule=rule.name, source=rule.source, checked=checked, violations=tuple(violations)
    )


def _check_forbid(rule: Rule, stmt: Forbid, actions: Sequence[Action]) -> RuleResult:
    matching = [a for a in actions if stmt.subject.matches(a)]
    violations = tuple(
        _violation(rule, a, f"forbidden call to {a.name}")
        for a in matching
        if _holds(stmt.condition, a)
    )
    return RuleResult(
        rule=rule.name, source=rule.source, checked=len(matching), violations=violations
    )


def _check_at_most(rule: Rule, stmt: AtMost, actions: Sequence[Action]) -> RuleResult:
    counts: dict[tuple[str, ...] | None, int] = {}
    checked, violations = 0, []
    for action in actions:
        if not stmt.subject.matches(action) or not _holds(stmt.condition, action):
            continue
        checked += 1
        key = _key(action, stmt.per)
        counts[key] = counts.get(key, 0) + 1
        if counts[key] == stmt.limit + 1:  # report the call that crossed the limit, once
            scope = f" for the same {', '.join(stmt.per)}" if stmt.per else ""
            violations.append(
                _violation(rule, action, f"more than {stmt.limit} {stmt.subject} calls{scope}")
            )
    return RuleResult(
        rule=rule.name, source=rule.source, checked=checked, violations=tuple(violations)
    )


def _check_no_after_failure(
    rule: Rule, stmt: NoAfterFailure, actions: Sequence[Action]
) -> RuleResult:
    unrecovered: dict[str, Action] = {}  # failing tool -> the failure
    checked, violations = 0, []
    for action in actions:
        ok = action.executed and not action.is_error
        if stmt.subject.matches(action):
            checked += 1
            if unrecovered:
                failure = next(iter(unrecovered.values()))
                violations.append(
                    _violation(
                        rule,
                        action,
                        f"{action.name} after {failure.name} failed (call {failure.position}) "
                        "without a successful retry",
                    )
                )
        if ok:
            unrecovered.pop(action.name, None)
        elif stmt.failed.matches(action):
            unrecovered.setdefault(action.name, action)
    return RuleResult(
        rule=rule.name, source=rule.source, checked=checked, violations=tuple(violations)
    )


def evaluate_rule(rule: Rule, actions: Sequence[Action]) -> RuleResult:
    stmt = rule.statement
    if isinstance(stmt, Require):
        return _check_require(rule, stmt, actions)
    if isinstance(stmt, Forbid):
        return _check_forbid(rule, stmt, actions)
    if isinstance(stmt, AtMost):
        return _check_at_most(rule, stmt, actions)
    return _check_no_after_failure(rule, stmt, actions)


def raw_actions(trajectory: Trajectory) -> tuple[Action, ...]:
    """Requested tool calls with their *raw* arguments and outputs (no masking), so rules can
    compare real amounts and IDs."""
    return extract_actions(trajectory, Normalizer.none())


def evaluate(
    rules: Sequence[Rule], trajectory: Trajectory | Sequence[Action]
) -> tuple[RuleResult, ...]:
    actions = raw_actions(trajectory) if isinstance(trajectory, Trajectory) else trajectory
    return tuple(evaluate_rule(rule, actions) for rule in rules)
