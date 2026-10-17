"""Behavioural diff of two trajectories: alignment, divergence reasons, first divergence.

Replay says *whether* a run changed; the diff says *where* and *why*. It compares what two runs
did, not how they were worded: the sequence of tool calls the model requested (name, arguments,
result) and the outcome (status and final answer). Intermediate model text is deliberately not
aligned; it changes with every prompt tweak and is the job of scoring, not of localisation.

**Actions.** A trajectory becomes a list of actions, one per tool call the model *requested*, in
request order. Each action carries its result when the call was executed, or none when the run
ended before it ran (a cassette miss, a write refused by the sandbox). Building actions from the
requests rather than from executed tool steps is what lets the diff show the refund the sandbox
refused, not only the steps that happened before it.

**Normalisation.** Each side is normalised in one scope, in action order, with the same
``Normalizer`` replay used, so a fresh idempotency key or ticket timestamp is not drift while an
agent passing back a *different* ID than it was given still is.

**Alignment.** A weighted edit distance over the two action lists:

* deleting a baseline action (the candidate never made it) costs ``delete``;
* inserting a candidate action costs ``insert``;
* substituting a same-named action costs ``arguments * (1 - sim(args)) + output * (1 - sim(out))``
  with the structural ``similarity`` from ``agent_replay.matching``, so a refund whose amount
  drifted aligns with the recorded refund instead of reading as one removed and one added;
* substituting a differently-named action costs ``rename`` (below ``insert + delete``, so "called
  the wrong tool at this point" wins over "dropped one tool and added another").

The table is filled from the end (cost of aligning the two suffixes) and traced forward,
preferring a substitution, then a deletion, then an insertion on ties. Tracing forward makes the
*later* of two equal calls the extra one, which is how a retry loop reads to a human.

**Reasons.** Every non-matching aligned pair is a ``Divergence``: ``wrong_tool``,
``argument_drift`` (with the JSON paths that changed), ``output_drift`` (same call, different
result: the world differed, e.g. a live read), ``missing_step``, ``extra_step`` or
``extra_loop`` (an inserted call that repeats an earlier one exactly). The outcome adds
``status_changed`` and ``different_answer``. The first divergence also names the *decision
step*: the candidate's model call that made (or should have made) the divergent choice, which
is the step to look at first when debugging a regression.

The cost is O(n * m) time and memory in the number of actions; ``DiffCosts.max_cells`` bounds it
so a pathological pair of trajectories fails fast instead of exhausting memory.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, JsonValue

from agent_replay.canonical import canonical_json
from agent_replay.matching import MatchEvent, MatchOutcome, similarity
from agent_replay.model import LLMStep, ToolCall, ToolStep, Trajectory
from agent_replay.normalize import Normalizer

_EPS = 1e-9
_MISSING = object()  # "no such key / index" in ``json_changes``


class DiffOp(StrEnum):
    MATCH = "match"
    SUBSTITUTE = "substitute"
    DELETE = "delete"  # in the baseline only
    INSERT = "insert"  # in the candidate only


class DivergenceReason(StrEnum):
    WRONG_TOOL = "wrong_tool"
    ARGUMENT_DRIFT = "argument_drift"
    OUTPUT_DRIFT = "output_drift"
    MISSING_STEP = "missing_step"
    EXTRA_STEP = "extra_step"
    EXTRA_LOOP = "extra_loop"
    STATUS_CHANGED = "status_changed"
    DIFFERENT_ANSWER = "different_answer"


@dataclass(frozen=True, slots=True)
class DiffCosts:
    """Edit costs for the alignment. See the module docstring for how they interact."""

    insert: float = 1.0
    delete: float = 1.0
    rename: float = 1.5
    arguments: float = 0.8
    output: float = 0.2
    max_cells: int = 1_000_000

    def __post_init__(self) -> None:
        values = (self.insert, self.delete, self.rename, self.arguments, self.output)
        if any(v < 0 for v in values):
            raise ValueError("diff costs must be non-negative")
        if self.insert <= 0 or self.delete <= 0:
            raise ValueError("insert and delete costs must be positive")
        if self.rename >= self.insert + self.delete:
            raise ValueError("rename must cost less than a delete plus an insert")
        if self.arguments + self.output >= self.insert + self.delete:
            raise ValueError("a same-tool substitution must cost less than a delete plus an insert")

    def describe(self) -> dict[str, JsonValue]:
        return {
            "insert": self.insert,
            "delete": self.delete,
            "rename": self.rename,
            "arguments": self.arguments,
            "output": self.output,
        }


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Action(_Frozen):
    """One tool call the model requested, normalised, with its result if it was executed."""

    position: int  # index in this trajectory's action list
    call_id: str
    name: str
    arguments: JsonValue
    output: JsonValue = None
    is_error: bool = False
    executed: bool = True
    side_effect: bool = False
    step_index: int | None = None  # the ToolStep, when executed
    decision_step: int | None = None  # the LLMStep that requested the call
    # How replay served the call (candidate side of a replay diff only).
    served: MatchOutcome | None = None
    served_from: int | None = None
    match_score: float | None = None

    def label(self) -> str:
        return f"{self.name}({_compact_args(self.arguments)})"

    def result_value(self) -> JsonValue:
        return {"executed": self.executed, "is_error": self.is_error, "output": self.output}


class Change(_Frozen):
    path: str
    kind: Literal["changed", "added", "removed"]
    before: JsonValue = None
    after: JsonValue = None


class AlignedPair(_Frozen):
    op: DiffOp
    cost: float
    baseline: Action | None = None
    candidate: Action | None = None
    reason: DivergenceReason | None = None
    changes: tuple[Change, ...] = ()


class Divergence(_Frozen):
    reason: DivergenceReason
    summary: str
    tool: str | None = None
    baseline_step: int | None = None
    candidate_step: int | None = None
    decision_step: int | None = None  # candidate LLM step that made (or missed) the choice
    alignment_index: int | None = None  # row in ``TrajectoryDiff.alignment``; None for outcome
    side_effect: bool = False
    changes: tuple[Change, ...] = ()


class OutcomeDiff(_Frozen):
    baseline_status: str
    candidate_status: str
    baseline_output: str | None
    candidate_output: str | None
    output_similarity: float
    candidate_error: str | None = None

    @property
    def status_changed(self) -> bool:
        return self.baseline_status != self.candidate_status

    @property
    def answer_changed(self) -> bool:
        return self.output_similarity < 1.0


class UsageDelta(_Frozen):
    """Candidate minus baseline."""

    steps: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float


class TrajectoryDiff(_Frozen):
    baseline_id: str
    candidate_id: str
    normalizer: tuple[str, ...]
    costs: dict[str, JsonValue]
    distance: float
    similarity: float  # 1 - distance / (cost of deleting everything and inserting everything)
    alignment: tuple[AlignedPair, ...]
    divergences: tuple[Divergence, ...]
    outcome: OutcomeDiff
    usage: UsageDelta

    @property
    def identical(self) -> bool:
        return not self.divergences

    @property
    def first_divergence(self) -> Divergence | None:
        return self.divergences[0] if self.divergences else None

    def reason_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for divergence in self.divergences:
            counts[divergence.reason.value] = counts.get(divergence.reason.value, 0) + 1
        return dict(sorted(counts.items()))

    def to_report(self) -> dict[str, JsonValue]:
        first = self.first_divergence
        return {
            "identical": self.identical,
            "first_divergence": None if first is None else first.model_dump(mode="json"),
            "reasons": cast(JsonValue, self.reason_counts()),
            **self.model_dump(mode="json"),
        }


# --------------------------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------------------------


@dataclass(slots=True)
class _RawAction:
    call: ToolCall
    decision_step: int | None
    step: ToolStep | None
    side_effect: bool = False


def _raw_actions(trajectory: Trajectory) -> list[_RawAction]:
    """Requested tool calls in request order, joined to the tool steps that executed them.

    Executed calls appear where their tool step is. Calls a model requested but that never
    produced a tool step (the run ended first) follow the executed calls of the same turn.
    Traces ingested without content have no requested calls on their model steps; their tool
    steps still become actions, attributed to the closest preceding model step.
    """
    executed = {step.call.id for step in trajectory.tool_steps}
    requested_by: dict[str, int] = {}
    writes: set[str] = set()
    for llm in trajectory.llm_steps:
        writes.update(spec.name for spec in llm.request.tools if spec.side_effect)
        for call in llm.response.tool_calls:
            requested_by.setdefault(call.id, llm.index)

    actions: list[_RawAction] = []
    pending: list[_RawAction] = []
    last_llm: int | None = None
    for step in trajectory.steps:
        if isinstance(step, LLMStep):
            actions.extend(pending)
            # Never executed, so no tool step says whether it writes: ask the offered specs.
            pending = [
                _RawAction(call, step.index, None, call.name in writes)
                for call in step.response.tool_calls
                if call.id not in executed
            ]
            last_llm = step.index
        else:
            decision = requested_by.get(step.call.id, last_llm)
            actions.append(_RawAction(step.call, decision, step, step.side_effect))
    actions.extend(pending)
    return actions


def extract_actions(
    trajectory: Trajectory,
    normalizer: Normalizer | None = None,
    events: Sequence[MatchEvent] = (),
) -> tuple[Action, ...]:
    """Normalised actions of ``trajectory``; ``events`` annotates how replay served each call."""
    norm = normalizer or Normalizer.default()
    raw = _raw_actions(trajectory)
    normalised = _normalised(raw, norm)
    provenance = _provenance(raw, events)
    actions: list[Action] = []
    for position, item in enumerate(raw):
        event = provenance[position]
        step = item.step
        actions.append(
            Action(
                position=position,
                call_id=item.call.id,
                name=item.call.name,
                arguments=normalised[2 * position],
                output=normalised[2 * position + 1],
                is_error=step.result.is_error if step is not None else True,
                executed=step is not None,
                side_effect=item.side_effect,
                step_index=None if step is None else step.index,
                decision_step=item.decision_step,
                served=None if event is None else event.outcome,
                served_from=None if event is None else event.recorded_index,
                match_score=None if event is None else event.score,
            )
        )
    return tuple(actions)


def _normalised(raw: Sequence[_RawAction], norm: Normalizer, *extra: JsonValue) -> list[JsonValue]:
    """Arguments and outputs of every action (then ``extra``) normalised in one scope."""
    values: list[JsonValue] = []
    for item in raw:
        values.append(cast(JsonValue, item.call.arguments))
        values.append(None if item.step is None else item.step.result.output)
    values.extend(extra)
    return norm.apply_many(values)


def _provenance(raw: Sequence[_RawAction], events: Sequence[MatchEvent]) -> list[MatchEvent | None]:
    """Pair tool match events with actions in order; every call that reached the sandbox
    produced exactly one event. Stops at the first name mismatch rather than guessing."""
    out: list[MatchEvent | None] = [None] * len(raw)
    tool_events = [e for e in events if e.kind == "tool"]
    for position, (item, event) in enumerate(zip(raw, tool_events, strict=False)):
        if event.name != item.call.name:
            break
        out[position] = event
    return out


# --------------------------------------------------------------------------------------------
# JSON changes
# --------------------------------------------------------------------------------------------


def json_changes(
    before: JsonValue, after: JsonValue, *, path: str = "", limit: int = 20
) -> tuple[Change, ...]:
    """Leaf-level differences between two JSON values, as paths (``amount``, ``tags[1]``).

    Objects are compared key by key and lists position by position; at most ``limit`` changes
    are returned, in a deterministic order (sorted keys, ascending indices).
    """
    out: list[Change] = []
    _walk_changes(before, after, path, out, limit)
    return tuple(out)


def _walk_changes(
    before: JsonValue, after: JsonValue, path: str, out: list[Change], limit: int
) -> None:
    if len(out) >= limit or (before == after and type(before) is type(after)):
        return
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        keys = sorted(before.keys() | after.keys())
        entries: list[tuple[str, object, object]] = [
            (f"{path}.{k}" if path else str(k), before.get(k, _MISSING), after.get(k, _MISSING))
            for k in keys
        ]
    elif isinstance(before, list) and isinstance(after, list):
        entries = [
            (
                f"{path}[{i}]",
                before[i] if i < len(before) else _MISSING,
                after[i] if i < len(after) else _MISSING,
            )
            for i in range(max(len(before), len(after)))
        ]
    else:
        out.append(Change(path=path or "$", kind="changed", before=before, after=after))
        return
    for sub, old, new in entries:
        if len(out) >= limit:
            return
        if new is _MISSING:
            out.append(Change(path=sub, kind="removed", before=cast(JsonValue, old)))
        elif old is _MISSING:
            out.append(Change(path=sub, kind="added", after=cast(JsonValue, new)))
        else:
            _walk_changes(cast(JsonValue, old), cast(JsonValue, new), sub, out, limit)


# --------------------------------------------------------------------------------------------
# Alignment
# --------------------------------------------------------------------------------------------


def align(
    baseline: Sequence[Action], candidate: Sequence[Action], costs: DiffCosts | None = None
) -> tuple[float, list[tuple[DiffOp, int | None, int | None, float]]]:
    """Minimum-cost alignment; returns the distance and ``(op, i, j, cost)`` rows in order."""
    cost = costs or DiffCosts()
    n, m = len(baseline), len(candidate)
    if (n + 1) * (m + 1) > cost.max_cells:
        raise ValueError(
            f"trajectories too long to diff: {n} x {m} actions exceeds max_cells={cost.max_cells}"
        )
    sub = [[_substitution(a, b, cost) for b in candidate] for a in baseline]
    # suffix[i][j]: cost of aligning baseline[i:] with candidate[j:].
    suffix = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n, -1, -1):
        for j in range(m, -1, -1):
            if i == n and j == m:
                continue
            best = float("inf")
            if i < n and j < m:
                best = sub[i][j] + suffix[i + 1][j + 1]
            if i < n:
                best = min(best, cost.delete + suffix[i + 1][j])
            if j < m:
                best = min(best, cost.insert + suffix[i][j + 1])
            suffix[i][j] = best

    rows: list[tuple[DiffOp, int | None, int | None, float]] = []
    i = j = 0
    while i < n or j < m:
        here = suffix[i][j]
        if i < n and j < m and abs(sub[i][j] + suffix[i + 1][j + 1] - here) < _EPS:
            op = DiffOp.MATCH if sub[i][j] < _EPS else DiffOp.SUBSTITUTE
            rows.append((op, i, j, sub[i][j]))
            i, j = i + 1, j + 1
        elif i < n and abs(cost.delete + suffix[i + 1][j] - here) < _EPS:
            rows.append((DiffOp.DELETE, i, None, cost.delete))
            i += 1
        else:
            rows.append((DiffOp.INSERT, None, j, cost.insert))
            j += 1
    return suffix[0][0], rows


def _substitution(a: Action, b: Action, cost: DiffCosts) -> float:
    if a.name != b.name:
        return cost.rename
    args = 1.0 - similarity(a.arguments, b.arguments)
    result = 1.0 - similarity(a.result_value(), b.result_value())
    value = cost.arguments * args + cost.output * result
    return 0.0 if value < _EPS else value


# --------------------------------------------------------------------------------------------
# Diff
# --------------------------------------------------------------------------------------------


def diff_trajectories(
    baseline: Trajectory,
    candidate: Trajectory,
    *,
    normalizer: Normalizer | None = None,
    costs: DiffCosts | None = None,
    candidate_events: Sequence[MatchEvent] = (),
) -> TrajectoryDiff:
    """Align ``candidate`` against ``baseline`` and explain every difference."""
    norm = normalizer or Normalizer.default()
    cost = costs or DiffCosts()
    base_actions = extract_actions(baseline, norm)
    cand_actions = extract_actions(candidate, norm, candidate_events)
    distance, rows = align(base_actions, cand_actions, cost)

    alignment: list[AlignedPair] = []
    divergences: list[Divergence] = []
    last_candidate_step: int | None = None
    for row_index, (op, i, j, row_cost) in enumerate(rows):
        a = None if i is None else base_actions[i]
        b = None if j is None else cand_actions[j]
        reason, changes = _classify(op, a, b, cand_actions)
        alignment.append(
            AlignedPair(
                op=op, cost=round(row_cost, 6), baseline=a, candidate=b, reason=reason,
                changes=changes,
            )
        )  # fmt: skip
        if reason is not None:
            decision = (
                b.decision_step
                if b is not None
                else _next_decision(candidate, after_step=last_candidate_step)
            )
            divergences.append(
                _divergence(
                    reason, a=a, b=b, decision=decision, row_index=row_index, changes=changes
                )
            )
        if b is not None:
            last_candidate_step = b.step_index if b.step_index is not None else b.decision_step

    outcome = _outcome(baseline, candidate, norm)
    divergences.extend(_outcome_divergences(outcome, candidate))

    worst = cost.delete * len(base_actions) + cost.insert * len(cand_actions)
    return TrajectoryDiff(
        baseline_id=baseline.trajectory_id,
        candidate_id=candidate.trajectory_id,
        normalizer=tuple(norm.describe()),
        costs=cost.describe(),
        distance=round(distance, 6),
        similarity=1.0 if worst == 0 else round(max(0.0, 1.0 - distance / worst), 6),
        alignment=tuple(alignment),
        divergences=tuple(divergences),
        outcome=outcome,
        usage=_usage_delta(baseline, candidate, base_actions, cand_actions),
    )


def _classify(
    op: DiffOp, a: Action | None, b: Action | None, candidate: Sequence[Action]
) -> tuple[DivergenceReason | None, tuple[Change, ...]]:
    if op is DiffOp.MATCH:
        return None, ()
    if op is DiffOp.DELETE:
        return DivergenceReason.MISSING_STEP, ()
    if op is DiffOp.INSERT:
        assert b is not None  # noqa: S101 - insert rows always carry a candidate action
        repeat = any(
            prior.name == b.name and prior.arguments == b.arguments
            for prior in candidate[: b.position]
        )
        return (DivergenceReason.EXTRA_LOOP if repeat else DivergenceReason.EXTRA_STEP), ()
    assert a is not None  # noqa: S101 - substitutions pair two actions
    assert b is not None  # noqa: S101
    if a.name != b.name:
        return DivergenceReason.WRONG_TOOL, ()
    arg_changes = json_changes(a.arguments, b.arguments)
    if arg_changes:
        return DivergenceReason.ARGUMENT_DRIFT, arg_changes
    return DivergenceReason.OUTPUT_DRIFT, json_changes(a.result_value(), b.result_value())


def _next_decision(trajectory: Trajectory, *, after_step: int | None) -> int | None:
    """The first model call after ``after_step``: where a missing call should have been made."""
    for llm in trajectory.llm_steps:
        if after_step is None or llm.index > after_step:
            return llm.index
    return None


_SUMMARIES: dict[DivergenceReason, str] = {
    DivergenceReason.WRONG_TOOL: "called {b} where the baseline called {a}",
    DivergenceReason.ARGUMENT_DRIFT: "called {name} with different arguments",
    DivergenceReason.OUTPUT_DRIFT: "{name} returned a different result for the same call",
    DivergenceReason.MISSING_STEP: "did not call {a}",
    DivergenceReason.EXTRA_STEP: "made an extra call to {b}",
    DivergenceReason.EXTRA_LOOP: "repeated {b}",
}


def _divergence(
    reason: DivergenceReason,
    *,
    a: Action | None,
    b: Action | None,
    decision: int | None,
    row_index: int,
    changes: tuple[Change, ...],
) -> Divergence:
    subject = b or a
    tool = subject.name if subject is not None else None
    summary = _SUMMARIES[reason].format(
        a=a.name if a else "", b=b.name if b else "", name=tool or ""
    )
    if b is not None and not b.executed:
        summary += " (requested, never executed)"
    if b is not None and b.served in {MatchOutcome.BLOCKED, MatchOutcome.LIVE, MatchOutcome.FUZZY}:
        summary += f" [replay: {b.served.value}]"
    return Divergence(
        reason=reason,
        summary=summary,
        tool=tool,
        baseline_step=None if a is None else a.step_index,
        candidate_step=None if b is None else b.step_index,
        decision_step=decision,
        alignment_index=row_index,
        side_effect=bool((a and a.side_effect) or (b and b.side_effect)),
        changes=changes,
    )


def _outcome(baseline: Trajectory, candidate: Trajectory, norm: Normalizer) -> OutcomeDiff:
    # Each answer is normalised in the scope of its own run, after its actions (the order
    # ``extract_actions`` uses), so an ID a tool minted gets the same placeholder in the answer.
    def scoped(trajectory: Trajectory) -> JsonValue:
        return _normalised(_raw_actions(trajectory), norm, trajectory.final_output)[-1]

    base_answer, cand_answer = scoped(baseline), scoped(candidate)
    return OutcomeDiff(
        baseline_status=baseline.status.value,
        candidate_status=candidate.status.value,
        baseline_output=baseline.final_output,
        candidate_output=candidate.final_output,
        output_similarity=round(similarity(base_answer, cand_answer), 6),
        candidate_error=candidate.error,
    )


def _outcome_divergences(outcome: OutcomeDiff, candidate: Trajectory) -> Iterable[Divergence]:
    last_llm = candidate.llm_steps[-1].index if candidate.llm_steps else None
    if outcome.status_changed:
        detail = f": {outcome.candidate_error}" if outcome.candidate_error else ""
        yield Divergence(
            reason=DivergenceReason.STATUS_CHANGED,
            summary=(
                f"run ended {outcome.candidate_status} instead of {outcome.baseline_status}{detail}"
            ),
            decision_step=last_llm,
        )
    if outcome.answer_changed:
        yield Divergence(
            reason=DivergenceReason.DIFFERENT_ANSWER,
            summary=f"final answer differs (similarity {outcome.output_similarity:.3f})",
            decision_step=last_llm,
            changes=(
                Change(
                    path="final_output",
                    kind="changed",
                    before=outcome.baseline_output,
                    after=outcome.candidate_output,
                ),
            ),
        )


def _usage_delta(
    baseline: Trajectory,
    candidate: Trajectory,
    base_actions: Sequence[Action],
    cand_actions: Sequence[Action],
) -> UsageDelta:
    a, b = baseline.usage, candidate.usage
    return UsageDelta(
        steps=len(candidate.steps) - len(baseline.steps),
        tool_calls=len(cand_actions) - len(base_actions),
        input_tokens=b.input_tokens - a.input_tokens,
        output_tokens=b.output_tokens - a.output_tokens,
        cost_usd=round(b.cost_usd - a.cost_usd, 10),
    )


def _compact_args(arguments: JsonValue, width: int = 60) -> str:
    if isinstance(arguments, Mapping):
        text = ", ".join(f"{k}={_short(v)}" for k, v in arguments.items())
    else:
        text = _short(arguments)
    return text if len(text) <= width else text[: width - 1] + "…"


def _short(value: JsonValue) -> str:
    return value if isinstance(value, str) else canonical_json(value)
