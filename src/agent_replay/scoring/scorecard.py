"""Layered scoring of one agent run, optionally against its baseline.

Layers run from cheapest and most trustworthy to most expensive and least trustworthy:

1. **checks**: deterministic facts about the run: it ended successfully with an answer, stayed
   within step, tool-call and cost budgets (absolute, and relative to the baseline when one is
   given), and made no side-effecting call that the replay sandbox had to block.
2. **invariants**: the rule file (``agent_replay.scoring.invariants``) over the requested tool
   calls.
3. **judge** (optional): an LLM verdict on the final answer, behind the provider interface. By
   default it only runs when layers 1 and 2 passed: a run that already failed a deterministic
   layer does not need a paid opinion, and the judge cannot overturn a deterministic failure.

The verdict is ``fail`` if any layer failed, ``inconclusive`` if nothing failed but the judge
could not decide, otherwise ``pass``. Every finding names its layer, so the report says which
layer caught a regression, and the cost report separates the agent's spend from the judge's.
"""

from __future__ import annotations

import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Literal

from opentelemetry.trace import TracerProvider
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from agent_replay.accounting import DEFAULT_PRICES, PriceTable, reprice, usage_by_model
from agent_replay.diff import Action
from agent_replay.logs import get_logger
from agent_replay.model import RunStatus, Trajectory, Usage
from agent_replay.providers.base import LLMProvider
from agent_replay.sandbox import BLOCKED_ERROR_TYPE
from agent_replay.scoring.invariants import Rule, evaluate, parse_rules, raw_actions
from agent_replay.scoring.judge import JudgeConfig, JudgeOutcome, judge_run
from agent_replay.telemetry import get_tracer

_log = get_logger(__name__)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Layer(StrEnum):
    CHECKS = "checks"
    INVARIANTS = "invariants"
    JUDGE = "judge"


class Verdict(StrEnum):
    PASS = "pass"  # noqa: S105 - a verdict, not a password
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


Severity = Literal["error", "warning"]


class CheckConfig(_Frozen):
    require_success: bool = True
    require_answer: bool = True
    max_llm_steps: int | None = Field(default=12, ge=1)
    max_tool_calls: int | None = Field(default=12, ge=0)
    max_cost_usd: float | None = Field(default=None, ge=0)
    # Candidate cost (or tokens, when unpriced) over the baseline's; None disables the check.
    max_cost_ratio: float | None = Field(default=1.5, gt=0)
    blocked_writes: Severity | Literal["off"] = "error"


class ScoringSpec(_Frozen):
    """What "good" means for one agent. Loadable from TOML (``[checks]``, ``invariants``,
    ``[judge]``, ``[prices.<model>]``)."""

    checks: CheckConfig = CheckConfig()
    invariants: str = ""
    judge: JudgeConfig = JudgeConfig()
    prices: dict[str, dict[str, float]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _rules_parse(self) -> ScoringSpec:
        parse_rules(self.invariants)  # fail at load time, with the line and column
        return self

    # Plain properties, not cached: ``model_copy(update=...)`` copies cached values along with
    # the fields, so a copy with new rules would keep evaluating the old ones.
    @property
    def rules(self) -> tuple[Rule, ...]:
        return parse_rules(self.invariants)

    @property
    def price_table(self) -> PriceTable:
        return DEFAULT_PRICES.merged(PriceTable.from_mapping(self.prices))

    @classmethod
    def from_toml(cls, path: str | Path) -> ScoringSpec:
        with Path(path).open("rb") as handle:
            return cls.model_validate(tomllib.load(handle))


class Finding(_Frozen):
    layer: Layer
    name: str
    passed: bool
    severity: Severity = "error"
    message: str
    step: int | None = None  # the step to look at first (a model decision, usually)
    detail: dict[str, JsonValue] = Field(default_factory=dict)

    @property
    def failing(self) -> bool:
        return not self.passed and self.severity == "error"


LayerStatus = Literal["passed", "failed", "skipped", "inconclusive"]


class LayerResult(_Frozen):
    layer: Layer
    status: LayerStatus
    findings: tuple[Finding, ...] = ()
    note: str | None = None


class CostReport(_Frozen):
    agent: Usage  # priced with the spec's table
    judge: Usage = Usage()
    recorded: Usage  # as the provider reported it
    by_model: dict[str, Usage] = Field(default_factory=dict)
    baseline: Usage | None = None
    ratio: float | None = None  # agent / baseline (cost, or tokens when unpriced)

    @property
    def total(self) -> Usage:
        return self.agent + self.judge


class Scorecard(_Frozen):
    trajectory_id: str
    baseline_id: str | None = None
    verdict: Verdict
    layers: tuple[LayerResult, ...]
    cost: CostReport
    judge_attempts: int = 0

    @property
    def failed_layers(self) -> tuple[Layer, ...]:
        return tuple(r.layer for r in self.layers if r.status == "failed")

    @property
    def failures(self) -> tuple[Finding, ...]:
        return tuple(f for r in self.layers for f in r.findings if f.failing)

    def layer(self, layer: Layer) -> LayerResult:
        return next(r for r in self.layers if r.layer is layer)

    def to_report(self) -> dict[str, JsonValue]:
        return {
            "verdict": self.verdict.value,
            "failed_layers": [layer.value for layer in self.failed_layers],
            **self.model_dump(mode="json"),
            "total_cost_usd": self.cost.total.cost_usd,
        }


# --------------------------------------------------------------------------------------------
# Layer 1: deterministic checks
# --------------------------------------------------------------------------------------------


def _ratio(candidate: Usage, baseline: Usage) -> float | None:
    if baseline.cost_usd > 0 and candidate.cost_usd > 0:
        return round(candidate.cost_usd / baseline.cost_usd, 4)
    if baseline.total_tokens > 0:
        return round(candidate.total_tokens / baseline.total_tokens, 4)
    return None


def _blocked(actions: tuple[Action, ...], trajectory: Trajectory) -> list[Action]:
    blocked_steps = {
        step.index for step in trajectory.tool_steps if step.result.error_type == BLOCKED_ERROR_TYPE
    }
    return [
        a for a in actions if a.side_effect and (not a.executed or a.step_index in blocked_steps)
    ]


def run_checks(
    trajectory: Trajectory,
    actions: tuple[Action, ...],
    config: CheckConfig,
    cost: CostReport,
) -> LayerResult:
    findings: list[Finding] = []

    def add(name: str, passed: bool, message: str, **extra: JsonValue) -> None:
        severity: Severity = "error"
        if name == "blocked_writes" and config.blocked_writes == "warning":
            severity = "warning"
        step = extra.pop("step", None)
        findings.append(
            Finding(
                layer=Layer.CHECKS,
                name=name,
                passed=passed,
                severity=severity,
                message=message,
                step=step if isinstance(step, int) else None,
                detail=extra,
            )
        )

    last_llm = trajectory.llm_steps[-1].index if trajectory.llm_steps else None
    if config.require_success:
        ok = trajectory.status is RunStatus.SUCCESS
        add(
            "status",
            ok,
            f"run ended {trajectory.status.value}"
            + (f": {trajectory.error}" if trajectory.error else ""),
            step=last_llm,
        )
    if config.require_answer:
        answered = bool((trajectory.final_output or "").strip())
        add("answer", answered, "final answer present" if answered else "no final answer")
    if config.max_llm_steps is not None:
        n = len(trajectory.llm_steps)
        add("llm_steps", n <= config.max_llm_steps, f"{n} model calls (max {config.max_llm_steps})")
    if config.max_tool_calls is not None:
        n = len(actions)
        add(
            "tool_calls",
            n <= config.max_tool_calls,
            f"{n} tool calls requested (max {config.max_tool_calls})",
        )
    if config.max_cost_usd is not None:
        spent = cost.agent.cost_usd
        add(
            "cost",
            spent <= config.max_cost_usd,
            f"cost ${spent:.6f} (max ${config.max_cost_usd:.6f})",
        )
    if config.max_cost_ratio is not None and cost.ratio is not None:
        add(
            "cost_ratio",
            cost.ratio <= config.max_cost_ratio,
            f"{cost.ratio:.2f}x the baseline's cost (max {config.max_cost_ratio:.2f}x)",
            ratio=cost.ratio,
        )
    if config.blocked_writes != "off":
        blocked = _blocked(actions, trajectory)
        first = blocked[0] if blocked else None
        add(
            "blocked_writes",
            not blocked,
            "no unrecorded side-effecting calls"
            if first is None
            else f"{len(blocked)} side-effecting call(s) matched no recording and were not "
            f"executed, first {first.label()}",
            step=None if first is None else first.decision_step,
            tools=[a.name for a in blocked],
        )
    failed = any(f.failing for f in findings)
    return LayerResult(
        layer=Layer.CHECKS, status="failed" if failed else "passed", findings=tuple(findings)
    )


# --------------------------------------------------------------------------------------------
# Layers 2 and 3
# --------------------------------------------------------------------------------------------


def run_invariants(rules: tuple[Rule, ...], actions: tuple[Action, ...]) -> LayerResult:
    if not rules:
        return LayerResult(layer=Layer.INVARIANTS, status="skipped", note="no rules")
    findings: list[Finding] = []
    for result in evaluate(rules, actions):
        first = result.violations[0] if result.violations else None
        findings.append(
            Finding(
                layer=Layer.INVARIANTS,
                name=result.rule,
                passed=result.passed,
                message=(
                    f"held on {result.checked} call(s)"
                    if first is None
                    else f"{len(result.violations)} violation(s), first: {first.message}"
                ),
                step=None if first is None else first.decision_step,
                detail={
                    "rule": result.source,
                    "checked": result.checked,
                    "violations": [v.model_dump(mode="json") for v in result.violations],
                },
            )
        )
    failed = any(f.failing for f in findings)
    return LayerResult(
        layer=Layer.INVARIANTS, status="failed" if failed else "passed", findings=tuple(findings)
    )


def _judge_layer(outcome: JudgeOutcome, trajectory: Trajectory) -> LayerResult:
    last_llm = trajectory.llm_steps[-1].index if trajectory.llm_steps else None
    if outcome.verdict is None:
        return LayerResult(layer=Layer.JUDGE, status="inconclusive", note=outcome.error)
    verdict = outcome.verdict
    finding = Finding(
        layer=Layer.JUDGE,
        name="judge",
        passed=verdict.verdict == "pass",
        message="; ".join(verdict.reasons) or verdict.verdict,
        step=last_llm,
        detail={"score": verdict.score, "reasons": list(verdict.reasons)},
    )
    return LayerResult(
        layer=Layer.JUDGE, status="passed" if finding.passed else "failed", findings=(finding,)
    )


async def score(
    trajectory: Trajectory,
    spec: ScoringSpec,
    *,
    baseline: Trajectory | None = None,
    judge: LLMProvider | None = None,
    tracer_provider: TracerProvider | None = None,
) -> Scorecard:
    """Score ``trajectory`` layer by layer (see the module docstring)."""
    tracer = get_tracer(tracer_provider)
    with tracer.start_as_current_span("agent_replay.score") as span:
        span.set_attribute("agent_replay.trajectory.id", trajectory.trajectory_id)
        actions = raw_actions(trajectory)
        prices = spec.price_table
        agent_usage = reprice(trajectory, prices)
        base_usage = None if baseline is None else reprice(baseline, prices)
        cost = CostReport(
            agent=agent_usage,
            recorded=trajectory.usage,
            by_model=usage_by_model(trajectory),
            baseline=base_usage,
            ratio=None if base_usage is None else _ratio(agent_usage, base_usage),
        )

        layers = [
            run_checks(trajectory, actions, spec.checks, cost),
            run_invariants(spec.rules, actions),
        ]
        cheap_failed = any(layer.status == "failed" for layer in layers)
        attempts = 0
        if not spec.judge.enabled:
            layers.append(LayerResult(layer=Layer.JUDGE, status="skipped", note="disabled"))
        elif judge is None:
            layers.append(
                LayerResult(layer=Layer.JUDGE, status="skipped", note="no judge provider")
            )
        elif cheap_failed and spec.judge.when == "passed":
            layers.append(
                LayerResult(
                    layer=Layer.JUDGE, status="skipped", note="an earlier layer already failed"
                )
            )
        else:
            outcome = await judge_run(judge, trajectory, actions, spec.judge)
            attempts = outcome.attempts
            cost = cost.model_copy(update={"judge": outcome.usage})
            layers.append(_judge_layer(outcome, trajectory))

        if any(layer.status == "failed" for layer in layers):
            verdict = Verdict.FAIL
        elif any(layer.status == "inconclusive" for layer in layers):
            verdict = Verdict.INCONCLUSIVE
        else:
            verdict = Verdict.PASS
        card = Scorecard(
            trajectory_id=trajectory.trajectory_id,
            baseline_id=None if baseline is None else baseline.trajectory_id,
            verdict=verdict,
            layers=tuple(layers),
            cost=cost,
            judge_attempts=attempts,
        )
        span.set_attribute("agent_replay.score.verdict", verdict.value)
        span.set_attribute(
            "agent_replay.score.failed_layers", [layer.value for layer in card.failed_layers]
        )
        span.set_attribute("agent_replay.score.judge_cost_usd", cost.judge.cost_usd)
        _log.info(
            "scored run",
            extra={
                "ctx": {
                    "trajectory_id": trajectory.trajectory_id,
                    "verdict": verdict.value,
                    "failed_layers": [layer.value for layer in card.failed_layers],
                    "agent_cost_usd": cost.agent.cost_usd,
                    "judge_cost_usd": cost.judge.cost_usd,
                }
            },
        )
        return card
