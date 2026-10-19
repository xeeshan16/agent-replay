"""The CI gate: replay a candidate agent against a baseline suite and decide pass or breach.

For every case in the suite, the candidate is replayed against the recorded baseline (``TOOLS``
mode by default: the candidate's model decides, recorded tool outputs are served, unrecorded
reads may run live and unrecorded writes are blocked), ``repeats`` times. Each replay is diffed
against the baseline and scored with the layered scorecard. A sample's outcome is:

* ``fail``: the scorecard failed (a check, an invariant or the judge);
* ``inconclusive``: nothing failed, but the judge could not decide;
* ``changed``: the scorecard passed but the diff found a divergence. The candidate behaves
  differently and nothing says it is worse. This needs a human look but does not breach the
  gate (the Day 5 eval showed the diff alone flags every benign change);
* ``pass``: scorecard passed and no divergence;
* ``error``: the harness itself failed on this case (reported, never silently dropped).

The suite-level decision is made on thresholds, each of which is reported with its measured
value whether or not it breached:

* ``success_drop``: the lower bound of the paired bootstrap CI on
  ``success(candidate) - success(baseline)`` must not fall below ``-max_success_drop``;
* ``success_rate`` (optional): the conservative lower bound (the lower of the cluster bootstrap
  and the Wilson interval) must be at least ``min_success_rate``;
* ``cost_ratio`` and ``tool_call_ratio``: suite totals against the baseline's;
* ``unsafe_actions``: side-effecting calls the sandbox had to block, summed over all samples;
* ``errors`` and (optionally) ``inconclusive`` samples.

Cases run concurrently up to ``concurrency``; results are ordered by case and repeat, and every
random choice is seeded, so the report is reproducible.
"""

from __future__ import annotations

import asyncio
import tomllib
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Literal

from opentelemetry.trace import TracerProvider
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from agent_replay.agent import Agent
from agent_replay.clock import TickingClock
from agent_replay.diff import diff_trajectories
from agent_replay.errors import AgentReplayError
from agent_replay.logs import get_logger
from agent_replay.matching import MatchPolicy
from agent_replay.model import Trajectory
from agent_replay.providers.base import LLMProvider
from agent_replay.replay import ReplayMode, replay
from agent_replay.sandbox import SandboxPolicy, UnrecordedWrites
from agent_replay.scoring import ScoringSpec, Verdict, score
from agent_replay.scoring.invariants import raw_actions
from agent_replay.scoring.scorecard import blocked_side_effects
from agent_replay.stats import Interval, bootstrap_mean, bootstrap_paired_delta, wilson
from agent_replay.suite import Suite, SuiteCase
from agent_replay.telemetry import get_tracer

_log = get_logger("gate")

AgentFactory = Callable[[str, int], Agent]  # (case id, repeat) -> a fresh candidate agent


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Outcome(StrEnum):
    PASS = "pass"  # noqa: S105 - an outcome, not a password
    CHANGED = "changed"
    INCONCLUSIVE = "inconclusive"
    FAIL = "fail"
    ERROR = "error"


_SEVERITY = {o: i for i, o in enumerate(Outcome)}  # declaration order is severity order


class GateThresholds(_Frozen):
    """Loadable from the ``[gate]`` table of a TOML file."""

    max_success_drop: float | None = Field(default=0.10, ge=0, le=1)
    min_success_rate: float | None = Field(default=None, ge=0, le=1)
    max_cost_ratio: float | None = Field(default=1.25, gt=0)
    max_tool_call_ratio: float | None = Field(default=1.5, gt=0)
    max_unsafe_actions: int | None = Field(default=0, ge=0)
    max_errors: int = Field(default=0, ge=0)
    fail_on_inconclusive: bool = False
    confidence: float = Field(default=0.95, gt=0, lt=1)
    resamples: int = Field(default=2000, ge=100, le=100_000)
    seed: int = 0

    @classmethod
    def from_toml(cls, path: str | Path) -> GateThresholds:
        with Path(path).open("rb") as handle:
            data = tomllib.load(handle)
        return cls.model_validate(data.get("gate", {}))


class ReplaySettings(_Frozen):
    mode: ReplayMode = ReplayMode.TOOLS
    live_reads: bool = True
    unrecorded_writes: UnrecordedWrites = UnrecordedWrites.BLOCK

    def describe(self) -> str:
        reads = "live reads" if self.live_reads else "no live reads"
        return f"{self.mode.value} mode, {reads}, unrecorded writes {self.unrecorded_writes.value}"


class SampleResult(_Frozen):
    repeat: int
    outcome: Outcome
    verdict: Verdict | None = None
    first_divergence: str | None = None  # ``reason:tool @ step``
    failures: tuple[str, ...] = ()  # ``layer:name: message``
    unsafe_actions: int = 0
    tool_calls: int = 0
    cost_usd: float = 0.0  # agent spend, priced with the spec's table
    tokens: int = 0
    judge_cost_usd: float = 0.0
    fingerprint: str | None = None
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.verdict is Verdict.PASS


class CaseResult(_Frozen):
    id: str
    task: str
    baseline_verdict: Verdict
    baseline_tool_calls: int
    baseline_cost_usd: float
    baseline_tokens: int
    samples: tuple[SampleResult, ...]

    @property
    def outcome(self) -> Outcome:
        return max((s.outcome for s in self.samples), key=_SEVERITY.__getitem__)

    @property
    def success_values(self) -> list[float]:
        return [1.0 if s.success else 0.0 for s in self.samples]

    @property
    def baseline_success(self) -> float:
        return 1.0 if self.baseline_verdict is Verdict.PASS else 0.0


class ThresholdCheck(_Frozen):
    name: str
    passed: bool
    value: float | int | None
    limit: float | int | None
    message: str


class GateReport(_Frozen):
    suite: str
    candidate: str
    provider: str
    replay: str
    repeats: int
    thresholds: GateThresholds
    cases: tuple[CaseResult, ...]
    success_rate: Interval
    success_rate_wilson: Interval
    baseline_success_rate: float
    success_delta: Interval
    cost_ratio: float | None
    tool_call_ratio: float | None
    checks: tuple[ThresholdCheck, ...]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def verdict(self) -> Literal["pass", "breach"]:
        return "pass" if self.passed else "breach"

    @property
    def success_lower_bound(self) -> float:
        return min(self.success_rate.low, self.success_rate_wilson.low)

    def outcome_counts(self) -> dict[str, int]:
        counts = {o.value: 0 for o in Outcome}
        for case in self.cases:
            counts[case.outcome.value] += 1
        return counts

    def cases_with(self, outcome: Outcome) -> list[CaseResult]:
        return [c for c in self.cases if c.outcome is outcome]

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "verdict": self.verdict,
            "outcomes": dict(self.outcome_counts()),
            "success_lower_bound": self.success_lower_bound,
            **self.model_dump(mode="json"),
        }


# --------------------------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------------------------


def _first(divergence_reason: str, tool: str | None, step: int | None) -> str:
    where = "" if step is None else f" @ {step}"
    return f"{divergence_reason}:{tool or '-'}{where}"


async def _run_sample(
    case: SuiteCase,
    baseline: Trajectory,
    repeat: int,
    *,
    agent_factory: AgentFactory,
    spec: ScoringSpec,
    judge: LLMProvider | None,
    match: MatchPolicy,
    settings: ReplaySettings,
    tracer_provider: TracerProvider | None,
) -> SampleResult:
    try:
        result = await replay(
            agent_factory(case.id, repeat),
            baseline,
            mode=settings.mode,
            clock=TickingClock(),
            policy=match,
            sandbox=SandboxPolicy(
                live_reads=settings.live_reads, unrecorded_writes=settings.unrecorded_writes
            ),
        )
        candidate = result.replayed
        diff = diff_trajectories(
            baseline, candidate, normalizer=match.normalizer, candidate_events=result.matches
        )
        card = await score(
            candidate, spec, baseline=baseline, judge=judge, tracer_provider=tracer_provider
        )
    except (AgentReplayError, ValueError, TimeoutError) as exc:
        _log.warning(
            "gate sample failed",
            extra={"ctx": {"case": case.id, "repeat": repeat, "error": repr(exc)}},
        )
        return SampleResult(
            repeat=repeat, outcome=Outcome.ERROR, error=f"{type(exc).__name__}: {exc}"
        )

    actions = raw_actions(candidate)
    if card.verdict is Verdict.FAIL:
        outcome = Outcome.FAIL
    elif card.verdict is Verdict.INCONCLUSIVE:
        outcome = Outcome.INCONCLUSIVE
    elif not diff.identical:
        outcome = Outcome.CHANGED
    else:
        outcome = Outcome.PASS
    first = diff.first_divergence
    return SampleResult(
        repeat=repeat,
        outcome=outcome,
        verdict=card.verdict,
        first_divergence=None
        if first is None
        else _first(first.reason.value, first.tool, first.decision_step),
        failures=tuple(f"{f.layer.value}:{f.name}: {f.message}" for f in card.failures),
        unsafe_actions=len(blocked_side_effects(actions, candidate)),
        tool_calls=len(actions),
        cost_usd=card.cost.agent.cost_usd,
        tokens=card.cost.agent.total_tokens,
        judge_cost_usd=card.cost.judge.cost_usd,
        fingerprint=candidate.fingerprint(),
    )


async def run_gate(
    suite: Suite,
    agent_factory: AgentFactory,
    spec: ScoringSpec,
    *,
    match: MatchPolicy,
    thresholds: GateThresholds | None = None,
    judge: LLMProvider | None = None,
    repeats: int = 1,
    concurrency: int = 4,
    settings: ReplaySettings | None = None,
    candidate: str = "candidate",
    provider: str = "unknown",
    tracer_provider: TracerProvider | None = None,
) -> GateReport:
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    thresholds = thresholds or GateThresholds()
    settings = settings or ReplaySettings()
    tracer = get_tracer(tracer_provider)
    semaphore = asyncio.Semaphore(concurrency)

    async def sample(case: SuiteCase, baseline: Trajectory, repeat: int) -> SampleResult:
        async with semaphore:
            with tracer.start_as_current_span("agent_replay.gate.sample") as span:
                span.set_attribute("agent_replay.gate.case", case.id)
                span.set_attribute("agent_replay.gate.repeat", repeat)
                result = await _run_sample(
                    case,
                    baseline,
                    repeat,
                    agent_factory=agent_factory,
                    spec=spec,
                    judge=judge,
                    match=match,
                    settings=settings,
                    tracer_provider=tracer_provider,
                )
                span.set_attribute("agent_replay.gate.outcome", result.outcome.value)
                return result

    with tracer.start_as_current_span("agent_replay.gate") as span:
        span.set_attribute("agent_replay.gate.suite", suite.name)
        span.set_attribute("agent_replay.gate.repeats", repeats)
        items = suite.items()
        # Score each baseline against itself, without the judge: it is the reference point.
        no_judge = spec.model_copy(
            update={"judge": spec.judge.model_copy(update={"enabled": False})}
        )
        baseline_cards = [
            await score(b, no_judge, baseline=b, tracer_provider=tracer_provider) for _, b in items
        ]
        async with asyncio.TaskGroup() as group:
            tasks = [
                [group.create_task(sample(case, baseline, r)) for r in range(repeats)]
                for case, baseline in items
            ]
        cases = tuple(
            CaseResult(
                id=case.id,
                task=case.task,
                baseline_verdict=card.verdict,
                baseline_tool_calls=len(raw_actions(baseline)),
                baseline_cost_usd=card.cost.agent.cost_usd,
                baseline_tokens=card.cost.agent.total_tokens,
                samples=tuple(t.result() for t in row),
            )
            for (case, baseline), card, row in zip(items, baseline_cards, tasks, strict=True)
        )
        report = summarise(
            suite=suite.name,
            cases=cases,
            thresholds=thresholds,
            repeats=repeats,
            candidate=candidate,
            provider=provider,
            replay=settings.describe(),
        )
        span.set_attribute("agent_replay.gate.verdict", report.verdict)
        span.set_attribute("agent_replay.gate.success_lower_bound", report.success_lower_bound)
        _log.info(
            "gate finished",
            extra={
                "ctx": {
                    "suite": suite.name,
                    "candidate": candidate,
                    "verdict": report.verdict,
                    "outcomes": report.outcome_counts(),
                    "breached": [c.name for c in report.checks if not c.passed],
                }
            },
        )
        return report


# --------------------------------------------------------------------------------------------
# Deciding
# --------------------------------------------------------------------------------------------


def _ratio(candidate: float, baseline: float) -> float | None:
    return None if baseline <= 0 else round(candidate / baseline, 4)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarise(
    *,
    suite: str,
    cases: tuple[CaseResult, ...],
    thresholds: GateThresholds,
    repeats: int,
    candidate: str,
    provider: str,
    replay: str,
) -> GateReport:
    """Turn per-case results into intervals and threshold checks (pure; no I/O)."""
    if not cases:
        raise ValueError("a gate needs at least one case")
    conf, resamples, seed = thresholds.confidence, thresholds.resamples, thresholds.seed
    candidate_values = [c.success_values for c in cases]
    baseline_values = [[c.baseline_success] for c in cases]
    rate = bootstrap_mean(candidate_values, confidence=conf, resamples=resamples, seed=seed)
    rate_wilson = wilson(sum(_mean(v) for v in candidate_values), len(cases), confidence=conf)
    delta = bootstrap_paired_delta(
        baseline_values, candidate_values, confidence=conf, resamples=resamples, seed=seed
    )

    # Errors carry no cost or tool data; average the samples that ran.
    def per_case(case: CaseResult, field: str) -> float:
        ran = [s for s in case.samples if s.outcome is not Outcome.ERROR]
        return _mean([float(getattr(s, field)) for s in ran])

    cost = sum(per_case(c, "cost_usd") for c in cases)
    base_cost = sum(c.baseline_cost_usd for c in cases)
    if base_cost <= 0:  # unpriced models: fall back to tokens
        cost = sum(per_case(c, "tokens") for c in cases)
        base_cost = float(sum(c.baseline_tokens for c in cases))
    cost_ratio = _ratio(cost, base_cost)
    tool_ratio = _ratio(
        sum(per_case(c, "tool_calls") for c in cases),
        float(sum(c.baseline_tool_calls for c in cases)),
    )
    samples = [s for c in cases for s in c.samples]
    unsafe = sum(s.unsafe_actions for s in samples)
    errors = sum(s.outcome is Outcome.ERROR for s in samples)
    inconclusive = sum(s.outcome is Outcome.INCONCLUSIVE for s in samples)

    checks: list[ThresholdCheck] = []
    lower = min(rate.low, rate_wilson.low)
    pct = f"{conf:.0%}"
    if thresholds.max_success_drop is not None:
        floor = -thresholds.max_success_drop
        checks.append(
            ThresholdCheck(
                name="success_drop",
                passed=delta.low >= floor - 1e-9,
                value=delta.low,
                limit=floor,
                message=f"success change {delta.estimate:+.3f}, {pct} CI "
                f"[{delta.low:+.3f}, {delta.high:+.3f}]; lower bound must be >= {floor:+.3f}",
            )
        )
    if thresholds.min_success_rate is not None:
        checks.append(
            ThresholdCheck(
                name="success_rate",
                passed=lower >= thresholds.min_success_rate - 1e-9,
                value=lower,
                limit=thresholds.min_success_rate,
                message=f"success rate {rate.estimate:.3f}, conservative {pct} lower bound "
                f"{lower:.3f} (bootstrap {rate.low:.3f}, Wilson {rate_wilson.low:.3f}); "
                f"must be >= {thresholds.min_success_rate:.3f}",
            )
        )
    if thresholds.max_cost_ratio is not None and cost_ratio is not None:
        checks.append(
            ThresholdCheck(
                name="cost_ratio",
                passed=cost_ratio <= thresholds.max_cost_ratio,
                value=cost_ratio,
                limit=thresholds.max_cost_ratio,
                message=f"suite cost {cost_ratio:.2f}x the baseline's "
                f"(max {thresholds.max_cost_ratio:.2f}x)",
            )
        )
    if thresholds.max_tool_call_ratio is not None and tool_ratio is not None:
        checks.append(
            ThresholdCheck(
                name="tool_call_ratio",
                passed=tool_ratio <= thresholds.max_tool_call_ratio,
                value=tool_ratio,
                limit=thresholds.max_tool_call_ratio,
                message=f"{tool_ratio:.2f}x the baseline's tool calls "
                f"(max {thresholds.max_tool_call_ratio:.2f}x)",
            )
        )
    if thresholds.max_unsafe_actions is not None:
        checks.append(
            ThresholdCheck(
                name="unsafe_actions",
                passed=unsafe <= thresholds.max_unsafe_actions,
                value=unsafe,
                limit=thresholds.max_unsafe_actions,
                message=f"{unsafe} side-effecting call(s) matched no recording and were blocked "
                f"(max {thresholds.max_unsafe_actions})",
            )
        )
    checks.append(
        ThresholdCheck(
            name="errors",
            passed=errors <= thresholds.max_errors,
            value=errors,
            limit=thresholds.max_errors,
            message=f"{errors} sample(s) could not be run (max {thresholds.max_errors})",
        )
    )
    if thresholds.fail_on_inconclusive:
        checks.append(
            ThresholdCheck(
                name="inconclusive",
                passed=inconclusive == 0,
                value=inconclusive,
                limit=0,
                message=f"{inconclusive} sample(s) the judge could not decide",
            )
        )
    return GateReport(
        suite=suite,
        candidate=candidate,
        provider=provider,
        replay=replay,
        repeats=repeats,
        thresholds=thresholds,
        cases=cases,
        success_rate=rate,
        success_rate_wilson=rate_wilson,
        baseline_success_rate=round(_mean([c.baseline_success for c in cases]), 6),
        success_delta=delta,
        cost_ratio=cost_ratio,
        tool_call_ratio=tool_ratio,
        checks=tuple(checks),
    )
