"""The regression-detection eval: does each layer catch seeded regressions without flagging
benign changes?

Every case in the labelled dataset (``evals/regressions.jsonl``) names a task, a candidate agent
policy and a human label: ``regression`` (the candidate behaves worse), ``benign`` (it behaves
differently but acceptably) or ``no_effect`` (the change does not apply to this task). Labels
come from what each policy is written to do, not from running any detector, and regressions and
benign changes also carry the candidate model step that made the decisive choice.

For each case the harness records the baseline agent (``v1``) on the task, replays the
recording with the candidate policy in ``TOOLS`` mode (live reads, unrecorded writes blocked,
the way a CI gate would run it), then diffs and scores the replay against the baseline. Each
detector is measured as a binary classifier for "is a regression":

* ``diff``: the trajectory diff found any divergence;
* ``checks``, ``invariants``, ``judge``: that layer failed (scored with the judge always on, so
  each layer is measured on every case);
* ``layered``: the production verdict is ``fail`` (judge only when the cheaper layers passed).

Everything is deterministic with the fake providers: seeded IDs, a ticking clock, a fresh
order store per run.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue

from agent_replay.clock import TickingClock
from agent_replay.diff import diff_trajectories
from agent_replay.model import Usage
from agent_replay.recorder import record_run
from agent_replay.reference import support_desk as desk
from agent_replay.replay import ReplayMode, replay
from agent_replay.report import cell
from agent_replay.sandbox import SandboxPolicy, UnrecordedWrites
from agent_replay.scoring import Layer, ScoringSpec, Verdict, score

DEFAULT_DATASET = Path("evals/regressions.jsonl")
DETECTORS: tuple[str, ...] = ("diff", "checks", "invariants", "judge", "layered")
Label = Literal["regression", "benign", "no_effect"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class EvalCase(_Frozen):
    id: str
    task: str
    policy: str
    label: Label
    category: str
    decision_step: int | None = None
    note: str = ""


def load_cases(path: str | Path = DEFAULT_DATASET) -> tuple[EvalCase, ...]:
    cases: list[EvalCase] = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        case = EvalCase.model_validate_json(line)
        if case.policy not in desk.POLICIES:
            raise ValueError(f"{path}:{number}: unknown policy {case.policy!r}")
        if case.label != "no_effect" and case.decision_step is None:
            raise ValueError(f"{path}:{number}: {case.id} changes behaviour but has no step")
        cases.append(case)
    ids = [c.id for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate case ids")
    return tuple(cases)


class CaseResult(_Frozen):
    id: str
    policy: str
    label: Label
    category: str
    flags: dict[str, bool]  # detector -> flagged as a regression
    verdict: Verdict
    failures: tuple[str, ...]  # failing findings, ``layer:name``
    first_divergence: str | None  # ``reason:tool``
    expected_step: int | None
    predicted_step: int | None  # the diff's decision step for its first divergence
    agent_cost_usd: float
    judge_calls: int  # in the layered (production) configuration
    judge_cost_usd: float
    judge_cost_always_usd: float


async def run_case(case: EvalCase, spec: ScoringSpec) -> CaseResult:
    baseline_agent = desk.build_agent(id_factory=desk.seeded_ids(f"{case.id}/baseline"))
    baseline = await record_run(
        baseline_agent, case.task, clock=TickingClock(), trajectory_id=f"{case.id}/baseline"
    )
    candidate_agent = desk.build_agent(
        policy=desk.POLICIES[case.policy], id_factory=desk.seeded_ids(f"{case.id}/candidate")
    )
    result = await replay(
        candidate_agent,
        baseline,
        mode=ReplayMode.TOOLS,
        clock=TickingClock(),
        policy=desk.match_policy(),
        sandbox=SandboxPolicy(live_reads=True, unrecorded_writes=UnrecordedWrites.BLOCK),
    )
    candidate = result.replayed
    diff = diff_trajectories(
        baseline, candidate, normalizer=desk.NORMALIZER, candidate_events=result.matches
    )
    judge = desk.build_judge()
    layered = await score(candidate, spec, baseline=baseline, judge=judge)
    always_spec = spec.model_copy(
        update={"judge": spec.judge.model_copy(update={"enabled": True, "when": "always"})}
    )
    always = await score(candidate, always_spec, baseline=baseline, judge=judge)
    first = diff.first_divergence
    return CaseResult(
        id=case.id,
        policy=case.policy,
        label=case.label,
        category=case.category,
        flags={
            "diff": not diff.identical,
            "checks": always.layer(Layer.CHECKS).status == "failed",
            "invariants": always.layer(Layer.INVARIANTS).status == "failed",
            "judge": always.layer(Layer.JUDGE).status == "failed",
            "layered": layered.verdict is Verdict.FAIL,
        },
        verdict=layered.verdict,
        failures=tuple(f"{f.layer.value}:{f.name}" for f in always.failures),
        first_divergence=None if first is None else f"{first.reason.value}:{first.tool or '-'}",
        expected_step=case.decision_step,
        predicted_step=None if first is None else first.decision_step,
        agent_cost_usd=layered.cost.agent.cost_usd,
        judge_calls=layered.judge_attempts,
        judge_cost_usd=layered.cost.judge.cost_usd,
        judge_cost_always_usd=always.cost.judge.cost_usd,
    )


class DetectorMetrics(_Frozen):
    tp: int
    fp: int
    fn: int
    tn: int
    benign_flagged: int  # false positives that are real behaviour changes

    @property
    def precision(self) -> float | None:
        return None if self.tp + self.fp == 0 else round(self.tp / (self.tp + self.fp), 4)

    @property
    def recall(self) -> float | None:
        return None if self.tp + self.fn == 0 else round(self.tp / (self.tp + self.fn), 4)

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if p is None or r is None or p + r == 0:
            return None
        return round(2 * p * r / (p + r), 4)


class EvalReport(_Frozen):
    dataset: str
    provider: str
    settings: dict[str, JsonValue]
    cases: tuple[CaseResult, ...]

    @property
    def label_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for case in self.cases:
            counts[case.label] = counts.get(case.label, 0) + 1
        return dict(sorted(counts.items()))

    def metrics(self, detector: str) -> DetectorMetrics:
        tp = fp = fn = tn = benign = 0
        for case in self.cases:
            flagged, positive = case.flags[detector], case.label == "regression"
            tp += flagged and positive
            fp += flagged and not positive
            fn += positive and not flagged
            tn += not positive and not flagged
            benign += flagged and case.label == "benign"
        return DetectorMetrics(tp=tp, fp=fp, fn=fn, tn=tn, benign_flagged=benign)

    def recall_by_category(self) -> dict[str, dict[str, str]]:
        """``category -> detector -> "caught/total"`` over regression cases."""
        table: dict[str, dict[str, str]] = {}
        categories = sorted({c.category for c in self.cases if c.label == "regression"})
        for category in categories:
            cases = [c for c in self.cases if c.category == category]
            table[category] = {
                d: f"{sum(c.flags[d] for c in cases)}/{len(cases)}" for d in DETECTORS
            }
        return table

    @property
    def localisation(self) -> tuple[int, int]:
        """Changed cases whose first divergence names the labelled decision step."""
        labelled = [c for c in self.cases if c.expected_step is not None]
        hits = sum(c.predicted_step == c.expected_step for c in labelled)
        return hits, len(labelled)

    @property
    def judge_cost(self) -> tuple[Usage, Usage]:
        """Judge spend (cost only) in the layered configuration and with the judge always on."""
        layered = sum(c.judge_cost_usd for c in self.cases)
        always = sum(c.judge_cost_always_usd for c in self.cases)
        return Usage(cost_usd=round(layered, 10)), Usage(cost_usd=round(always, 10))

    def to_json(self) -> dict[str, JsonValue]:
        hits, total = self.localisation
        layered, always = self.judge_cost
        return {
            "dataset": self.dataset,
            "provider": self.provider,
            "settings": self.settings,
            "labels": dict(self.label_counts),
            "metrics": {
                d: {
                    **self.metrics(d).model_dump(),
                    "precision": self.metrics(d).precision,
                    "recall": self.metrics(d).recall,
                    "f1": self.metrics(d).f1,
                }
                for d in DETECTORS
            },
            "recall_by_category": dict(self.recall_by_category()),  # type: ignore[arg-type]
            "localisation": {"correct": hits, "labelled": total},
            "judge": {
                "calls_layered": sum(c.judge_calls for c in self.cases),
                "cost_usd_layered": layered.cost_usd,
                "cost_usd_always": always.cost_usd,
            },
            "cases": [c.model_dump(mode="json") for c in self.cases],
        }


async def run_eval(
    cases: Sequence[EvalCase], spec: ScoringSpec, *, dataset: str = str(DEFAULT_DATASET)
) -> EvalReport:
    results = [await run_case(case, spec) for case in cases]
    return EvalReport(
        dataset=dataset,
        provider=f"fake (agent {desk.MODEL}, judge {spec.judge.model}); no network",
        settings={
            "replay": "tools mode, live reads, unrecorded writes blocked",
            "match": "strict, support-desk normaliser",
            "judge_when": spec.judge.when,
            "max_cost_ratio": spec.checks.max_cost_ratio,
            "rules": [rule.name for rule in spec.rules],
        },
        cases=tuple(results),
    )


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def render_markdown(report: EvalReport) -> str:
    labels = ", ".join(f"{n} {label}" for label, n in report.label_counts.items())
    lines = [
        "### Regression-detection eval",
        "",
        f"Dataset `{cell(report.dataset, 80)}` ({len(report.cases)} cases: {labels}). "
        f"Provider: {cell(report.provider, 120)}.",
        "",
        "| Detector | TP | FP | FN | TN | Benign flagged | Precision | Recall | F1 |",
        "|----------|----|----|----|----|----------------|-----------|--------|----|",
    ]
    benign_total = report.label_counts.get("benign", 0)
    for detector in DETECTORS:
        m = report.metrics(detector)
        lines.append(
            f"| {detector} | {m.tp} | {m.fp} | {m.fn} | {m.tn} | {m.benign_flagged}/{benign_total} "
            f"| {_fmt(m.precision)} | {_fmt(m.recall)} | {_fmt(m.f1)} |"
        )
    lines.extend(
        [
            "",
            "| Regression category | " + " | ".join(DETECTORS) + " |",
            "|---------------------|" + "|".join("---" for _ in DETECTORS) + "|",
        ]
    )
    for category, row in report.recall_by_category().items():
        lines.append(f"| {category} | " + " | ".join(row[d] for d in DETECTORS) + " |")
    hits, total = report.localisation
    layered, always = report.judge_cost
    calls = sum(c.judge_calls for c in report.cases)
    lines.extend(
        [
            "",
            f"First-divergence decision step matches the label: {hits}/{total} changed cases.",
            f"Judge: {calls} calls in the layered configuration, ${layered.cost_usd:.6f} "
            f"(${always.cost_usd:.6f} with the judge on every case).",
            "",
        ]
    )
    missed = [c for c in report.cases if c.label == "regression" and not c.flags["layered"]]
    flagged = [c for c in report.cases if c.label != "regression" and c.flags["layered"]]
    for title, rows in (("Regressions the layered verdict missed", missed),
                        ("Non-regressions the layered verdict failed", flagged)):  # fmt: skip
        if rows:
            lines.append(f"**{title}:** " + ", ".join(f"`{c.id}`" for c in rows))
            lines.append("")
    return "\n".join(lines)
