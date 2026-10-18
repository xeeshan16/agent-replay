from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_replay import cli
from agent_replay.accounting import PriceTable, reprice, usage_by_model
from agent_replay.clock import TickingClock
from agent_replay.model import Trajectory, Usage
from agent_replay.recorder import record_run
from agent_replay.reference import support_desk as desk
from agent_replay.replay import ReplayMode, replay
from agent_replay.report import render_scorecard
from agent_replay.sandbox import SandboxPolicy, UnrecordedWrites
from agent_replay.scoring import Layer, ScoringSpec, Verdict, score
from agent_replay.scoring.judge import JudgeConfig

REFUND = "I'd like a refund for ORD-1001, it arrived damaged."
OUTSIDE_WINDOW = "Please refund ORD-1003."
SPEC = desk.scoring_spec()


async def _record(task: str, policy: str = "v1", run_id: str = "run") -> Trajectory:
    agent = desk.build_agent(policy=desk.POLICIES[policy], id_factory=desk.seeded_ids(run_id))
    return await record_run(agent, task, clock=TickingClock(), trajectory_id=run_id)


async def _replayed(baseline: Trajectory, policy: str) -> Trajectory:
    result = await replay(
        desk.build_agent(policy=desk.POLICIES[policy]),
        baseline,
        mode=ReplayMode.TOOLS,
        clock=TickingClock(),
        policy=desk.match_policy(),
        sandbox=SandboxPolicy(live_reads=True, unrecorded_writes=UnrecordedWrites.BLOCK),
    )
    return result.replayed


def _with_judge(spec: ScoringSpec, **changes: object) -> ScoringSpec:
    return spec.model_copy(update={"judge": spec.judge.model_copy(update=changes)})


def _failed(card_findings: object) -> set[str]:
    return {f.name for f in card_findings}  # type: ignore[attr-defined]


@pytest.mark.parametrize("task", desk.SAMPLE_TASKS)
async def test_baseline_runs_pass_every_layer(task: str) -> None:
    baseline = await _record(task)
    card = await score(
        await _replayed(baseline, "v1"), SPEC, baseline=baseline, judge=desk.build_judge()
    )
    assert card.verdict is Verdict.PASS, card.failures
    assert [layer.status for layer in card.layers] == ["passed", "passed", "passed"]
    assert card.judge_attempts == 1
    assert card.cost.ratio == 1.0
    assert card.cost.judge.cost_usd > 0
    assert card.cost.total.cost_usd == pytest.approx(
        card.cost.agent.cost_usd + card.cost.judge.cost_usd
    )


async def test_unsafe_write_fails_cheap_layers_and_skips_the_judge() -> None:
    baseline = await _record(REFUND)
    candidate = await _replayed(baseline, "skip-lookup")
    card = await score(candidate, SPEC, baseline=baseline, judge=desk.build_judge())
    assert card.verdict is Verdict.FAIL
    assert card.failed_layers == (Layer.CHECKS, Layer.INVARIANTS)
    assert _failed(card.failures) == {
        "blocked_writes",
        "lookup_before_refund",
        "refund_only_if_eligible",
        "refund_within_total",
    }
    judge = card.layer(Layer.JUDGE)
    assert judge.status == "skipped"
    assert judge.note == "an earlier layer already failed"
    assert card.cost.judge == Usage()
    blocked = next(f for f in card.failures if f.name == "blocked_writes")
    assert blocked.step == 0  # the model call that decided to refund
    assert blocked.detail["tools"] == ["issue_refund"]


async def test_judge_when_always_still_runs_after_a_failure() -> None:
    baseline = await _record(REFUND)
    candidate = await _replayed(baseline, "skip-lookup")
    card = await score(
        candidate, _with_judge(SPEC, when="always"), baseline=baseline, judge=desk.build_judge()
    )
    assert card.layer(Layer.JUDGE).status in {"passed", "failed"}
    assert card.judge_attempts == 1


async def test_only_the_judge_catches_an_unsupported_claim() -> None:
    baseline = await _record(OUTSIDE_WINDOW)
    candidate = await _replayed(baseline, "false-claim")
    assert candidate.final_output == "Refund of 15.5 issued for ORD-1003."
    card = await score(candidate, SPEC, baseline=baseline, judge=desk.build_judge())
    assert card.failed_layers == (Layer.JUDGE,)
    (finding,) = card.failures
    assert "no issue_refund call succeeded" in finding.message
    assert finding.detail["score"] == 0.5


async def test_loop_is_an_invariant_violation_at_the_looping_decision() -> None:
    baseline = await _record(REFUND)
    candidate = await _replayed(baseline, "loop-lookup")
    card = await score(candidate, SPEC, baseline=baseline, judge=desk.build_judge())
    assert Layer.INVARIANTS in card.failed_layers
    loop = next(f for f in card.failures if f.name == "bounded_lookups")
    assert loop.step == 4  # the model call that asked for the third lookup
    assert card.cost.ratio is not None
    assert card.cost.ratio > 1.0


async def test_judge_layer_can_be_disabled_or_unavailable() -> None:
    run = await _record(REFUND)
    disabled = await score(run, _with_judge(SPEC, enabled=False), judge=desk.build_judge())
    assert disabled.layer(Layer.JUDGE).note == "disabled"
    missing = await score(run, SPEC)
    assert missing.layer(Layer.JUDGE).note == "no judge provider"
    assert disabled.verdict is missing.verdict is Verdict.PASS
    assert disabled.cost.ratio is None  # no baseline


async def test_an_undecided_judge_makes_the_verdict_inconclusive() -> None:
    from agent_replay.providers.fake import FakeLLM, FakeTurn  # noqa: PLC0415

    shrug = FakeLLM(lambda _request: FakeTurn.answer("hard to say"), model="fake-judge-1")
    card = await score(await _record(REFUND), SPEC, judge=shrug)
    assert card.verdict is Verdict.INCONCLUSIVE
    assert card.layer(Layer.JUDGE).status == "inconclusive"
    assert card.judge_attempts == SPEC.judge.max_attempts


async def test_blocked_writes_as_a_warning_do_not_fail_the_run() -> None:
    baseline = await _record(REFUND)
    candidate = await _replayed(baseline, "skip-lookup")
    checks = SPEC.checks.model_copy(update={"blocked_writes": "warning"})
    spec = SPEC.model_copy(update={"checks": checks, "invariants": ""})
    card = await score(candidate, spec, baseline=baseline)
    blocked = next(f for f in card.layer(Layer.CHECKS).findings if f.name == "blocked_writes")
    assert not blocked.passed
    assert blocked.severity == "warning"
    assert card.layer(Layer.CHECKS).status == "passed"
    assert card.layer(Layer.INVARIANTS).status == "skipped"


async def test_budgets() -> None:
    baseline = await _record(REFUND)
    candidate = await _record(REFUND, "check-policy", "candidate")
    checks = SPEC.checks.model_copy(
        update={
            "max_llm_steps": 3,
            "max_tool_calls": 2,
            "max_cost_usd": 1e-6,
            "max_cost_ratio": 1.1,
        }
    )
    card = await score(candidate, SPEC.model_copy(update={"checks": checks}), baseline=baseline)
    assert {"llm_steps", "tool_calls", "cost", "cost_ratio"} <= _failed(card.failures)


def test_spec_from_toml_validates_rules_and_merges_prices(tmp_path: Path) -> None:
    good = tmp_path / "spec.toml"
    good.write_text(
        'invariants = "r: forbid issue_refund"\n'
        "[checks]\nmax_cost_ratio = 2.0\n"
        '[prices."fake-tool-caller-1"]\ninput_per_mtok = 5.0\noutput_per_mtok = 15.0\n',
        encoding="utf-8",
    )
    spec = ScoringSpec.from_toml(good)
    assert [rule.name for rule in spec.rules] == ["r"]
    assert spec.checks.max_cost_ratio == 2.0
    price = spec.price_table.price_for("fake-tool-caller-1")
    assert price is not None
    assert price.input_per_mtok == 5.0
    assert spec.price_table.price_for("fake-judge-1") is not None  # defaults kept

    bad = tmp_path / "bad.toml"
    bad.write_text('invariants = "r: forbid"', encoding="utf-8")
    with pytest.raises(ValidationError, match="line 1"):
        ScoringSpec.from_toml(bad)


def test_bundled_spec_matches_the_documented_rules() -> None:
    assert [rule.name for rule in SPEC.rules] == [
        "lookup_before_refund",
        "refund_only_if_eligible",
        "refund_within_total",
        "one_refund_per_order",
        "bounded_lookups",
        "no_write_after_failed_read",
    ]
    assert SPEC.judge == JudgeConfig(enabled=True, timeout_s=30)


async def test_accounting_reprices_with_one_table() -> None:
    run = await _record(REFUND)
    assert set(usage_by_model(run)) == {desk.MODEL}
    assert usage_by_model(run)[desk.MODEL] == run.usage
    assert reprice(run, PriceTable({})) == run.usage  # unpriced models keep the recorded cost
    doubled = PriceTable.from_mapping({"*": {"input_per_mtok": 1.0, "output_per_mtok": 3.0}})
    repriced = reprice(run, doubled)
    assert repriced.total_tokens == run.usage.total_tokens
    assert repriced.cost_usd == pytest.approx(2 * run.usage.cost_usd)


async def test_scorecard_markdown_escapes_untrusted_text() -> None:
    baseline = await _record(REFUND)
    candidate = await _replayed(baseline, "skip-lookup")
    card = await score(candidate, SPEC, baseline=baseline)
    text = render_scorecard(card, title="Score | <b>x</b>")
    assert "**Verdict:** fail (failed: checks, invariants)" in text
    assert "&lt;b&gt;" in text
    assert "<b>" not in text
    rows = [line for line in text.splitlines() if line.startswith("| checks")]
    assert rows[0].split("|")[2].strip() == "blocked_writes"  # failures first


async def test_cli_score_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    baseline = await _record(REFUND)
    base = tmp_path / "base.json"
    base.write_text(baseline.to_json(), encoding="utf-8")
    bad = tmp_path / "bad.json"
    bad.write_text((await _replayed(baseline, "skip-lookup")).to_json(), encoding="utf-8")

    assert await cli._score(_args(base, base, "json")) == cli.EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["verdict"] == "pass"
    assert report["total_cost_usd"] > report["cost"]["agent"]["cost_usd"]

    out = tmp_path / "card.md"
    assert await cli._score(_args(bad, base, "md", out=out)) == cli.EXIT_DIVERGED
    assert "blocked_writes" in out.read_text(encoding="utf-8")


def _args(run: Path, base: Path, fmt: str, out: Path | None = None) -> argparse.Namespace:
    return cli.build_parser().parse_args(
        ["score", str(run), "--baseline", str(base), "--format", fmt]
        + ([] if out is None else ["--out", str(out)])
    )
