from __future__ import annotations

import asyncio
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from agent_replay import cli
from agent_replay.agent import ToolCallingAgent
from agent_replay.errors import AgentReplayError
from agent_replay.gate import (
    CaseResult,
    GateReport,
    GateThresholds,
    Outcome,
    SampleResult,
    run_gate,
    summarise,
)
from agent_replay.gate_report import render_junit, render_markdown
from agent_replay.reference import support_desk as desk
from agent_replay.scoring import Verdict
from agent_replay.suite import Suite, SuiteError, load_suite, record_suite, write_suite


def _factory(policy: str) -> object:
    def build(case_id: str, repeat: int) -> ToolCallingAgent:
        return desk.build_agent(
            policy=desk.POLICIES[policy], id_factory=desk.seeded_ids(f"{case_id}/{repeat}")
        )

    return build


@pytest.fixture(scope="module")
def suite() -> Suite:
    # Recorded once per module, outside the per-test event loops.
    return asyncio.run(
        record_suite(
            "desk-v1",
            desk.SAMPLE_TASKS,
            lambda case_id: desk.build_agent(id_factory=desk.seeded_ids(f"desk-v1/{case_id}")),
        )
    )


async def _gate(suite: Suite, policy: str, **kwargs: object) -> GateReport:
    return await run_gate(
        suite,
        _factory(policy),  # type: ignore[arg-type]
        desk.scoring_spec(),
        match=desk.match_policy(),
        judge=desk.build_judge(),
        candidate=policy,
        **kwargs,  # type: ignore[arg-type]
    )


# --- suites ------------------------------------------------------------------------------------


async def test_suite_round_trips_and_detects_tampering(suite: Suite, tmp_path: Path) -> None:
    root = write_suite(suite, tmp_path / "suite")
    loaded = load_suite(root)
    assert loaded.baselines == suite.baselines
    assert [c.model_copy(update={"file": None}) for c in loaded.manifest.cases] == list(
        suite.manifest.cases
    )
    assert next(c.file for c in loaded.manifest.cases) == "cases/00-hi-where-is-my-order.json"

    victim = root / "cases" / "01-i-d-like-a-refund.json"
    data = json.loads(victim.read_text())
    data["final_output"] = "Refund issued for $1,000,000."
    victim.write_text(json.dumps(data))
    with pytest.raises(SuiteError, match="does not match the fingerprint"):
        load_suite(root)


async def test_suite_rejects_paths_outside_its_directory(suite: Suite, tmp_path: Path) -> None:
    root = write_suite(suite, tmp_path / "suite")
    manifest = json.loads((root / "suite.json").read_text())
    manifest["cases"][0]["file"] = "../../etc/passwd"
    (root / "suite.json").write_text(json.dumps(manifest))
    with pytest.raises(SuiteError, match="outside"):
        load_suite(root)
    with pytest.raises(SuiteError, match="not a suite"):
        load_suite(tmp_path)


# --- the gate on the reference agent ---------------------------------------------------------


async def test_identical_candidate_passes_every_threshold(suite: Suite) -> None:
    report = await _gate(suite, "v1")
    assert report.passed
    assert report.outcome_counts()["pass"] == len(desk.SAMPLE_TASKS)
    assert report.success_delta.model_dump()["low"] == 0.0
    assert report.cost_ratio == 1.0
    assert report.tool_call_ratio == 1.0


async def test_unsafe_regression_breaches_with_named_causes(suite: Suite) -> None:
    report = await _gate(suite, "skip-lookup")
    assert not report.passed
    breached = {c.name for c in report.checks if not c.passed}
    assert breached == {"success_drop", "unsafe_actions"}
    failing = {c.id for c in report.cases_with(Outcome.FAIL)}
    assert "01-i-d-like-a-refund" in failing
    refund = next(c for c in report.cases if c.id == "01-i-d-like-a-refund")
    assert any("lookup_before_refund" in f for f in refund.samples[0].failures)
    assert refund.samples[0].first_divergence == "extra_step:issue_refund @ 0"


async def test_benign_change_is_reported_for_review(suite: Suite) -> None:
    report = await _gate(suite, "check-policy", thresholds=GateThresholds(max_cost_ratio=2.0))
    changed = {c.id for c in report.cases_with(Outcome.CHANGED)}
    assert "01-i-d-like-a-refund" in changed
    assert all(s.success for c in report.cases_with(Outcome.CHANGED) for s in c.samples)
    assert "Changed, needs review" in render_markdown(report)


async def test_repeats_concurrency_and_determinism(suite: Suite) -> None:
    one = await _gate(suite, "loop-lookup", repeats=3, concurrency=1)
    many = await _gate(suite, "loop-lookup", repeats=3, concurrency=8)
    assert one == many
    assert all(len(c.samples) == 3 for c in one.cases)
    assert [s.repeat for s in one.cases[0].samples] == [0, 1, 2]


async def test_harness_errors_are_reported_not_dropped(suite: Suite) -> None:
    def broken(case_id: str, repeat: int) -> ToolCallingAgent:
        if case_id.startswith("02-"):
            raise AgentReplayError("could not build the candidate")
        return desk.build_agent(id_factory=desk.seeded_ids(f"{case_id}/{repeat}"))

    report = await run_gate(suite, broken, desk.scoring_spec(), match=desk.match_policy())
    errored = report.cases_with(Outcome.ERROR)
    assert [c.id for c in errored] == ["02-please-refund-ord-1003"]
    assert errored[0].samples[0].error == "AgentReplayError: could not build the candidate"
    assert {c.name for c in report.checks if not c.passed} == {"success_drop", "errors"}


async def test_argument_validation(suite: Suite) -> None:
    with pytest.raises(ValueError, match="repeats"):
        await _gate(suite, "v1", repeats=0)
    with pytest.raises(ValueError, match="concurrency"):
        await _gate(suite, "v1", concurrency=0)


# --- thresholds on synthetic results ---------------------------------------------------------


def _case(
    case_id: str, outcomes: list[Outcome], *, cost: float = 1.0, unsafe: int = 0
) -> CaseResult:
    verdicts = {Outcome.PASS: Verdict.PASS, Outcome.CHANGED: Verdict.PASS,
                Outcome.FAIL: Verdict.FAIL, Outcome.INCONCLUSIVE: Verdict.INCONCLUSIVE}  # fmt: skip
    return CaseResult(
        id=case_id,
        task=f"task {case_id}",
        baseline_verdict=Verdict.PASS,
        baseline_tool_calls=2,
        baseline_cost_usd=1.0,
        baseline_tokens=100,
        samples=tuple(
            SampleResult(
                repeat=i,
                outcome=o,
                verdict=verdicts.get(o),
                tool_calls=2,
                cost_usd=cost,
                tokens=100,
                unsafe_actions=unsafe,
            )
            for i, o in enumerate(outcomes)
        ),
    )


def _summarise(cases: list[CaseResult], **thresholds: object) -> GateReport:
    return summarise(
        suite="s",
        cases=tuple(cases),
        thresholds=GateThresholds.model_validate(thresholds),
        repeats=len(cases[0].samples) if cases else 1,
        candidate="c",
        provider="p",
        replay="r",
    )


def test_success_rate_threshold_uses_the_conservative_bound() -> None:
    cases = [_case(f"c{i}", [Outcome.PASS]) for i in range(7)]
    report = _summarise(cases, min_success_rate=0.7)
    check = next(c for c in report.checks if c.name == "success_rate")
    # Bootstrap says [1, 1]; Wilson says 7/7 is only good for ~0.65 at 95%.
    assert report.success_rate.low == 1.0
    assert check.value == pytest.approx(0.6457, abs=1e-4)
    assert not check.passed
    assert _summarise(cases, min_success_rate=0.6).passed


def test_flaky_task_counts_as_one_task_not_many_runs() -> None:
    cases = [_case(f"c{i}", [Outcome.PASS] * 4) for i in range(19)]
    cases.append(_case("flaky", [Outcome.PASS, Outcome.FAIL, Outcome.PASS, Outcome.FAIL]))
    report = _summarise(cases, max_success_drop=0.1)
    assert report.success_rate.estimate == pytest.approx(0.975)
    assert report.success_delta.low >= -0.1
    assert report.passed
    assert report.cases[-1].outcome is Outcome.FAIL


def test_cost_unsafe_and_inconclusive_thresholds() -> None:
    cases = [_case("a", [Outcome.PASS], cost=1.5), _case("b", [Outcome.INCONCLUSIVE], unsafe=1)]
    report = _summarise(cases, max_success_drop=None, fail_on_inconclusive=True)
    by_name = {c.name: c for c in report.checks}
    assert by_name["cost_ratio"].value == 1.25
    assert by_name["cost_ratio"].passed
    assert not by_name["unsafe_actions"].passed
    assert not by_name["inconclusive"].passed
    assert "success_drop" not in by_name
    with pytest.raises(ValueError, match="at least one"):
        _summarise([])


def test_thresholds_load_from_toml(tmp_path: Path) -> None:
    path = tmp_path / "gate.toml"
    path.write_text("[gate]\nmax_success_drop = 0.0\nmin_success_rate = 0.5\n", encoding="utf-8")
    loaded = GateThresholds.from_toml(path)
    assert (loaded.max_success_drop, loaded.min_success_rate) == (0.0, 0.5)
    path.write_text("[gate]\nmax_sucess_drop = 0.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="max_sucess_drop"):
        GateThresholds.from_toml(path)


# --- reports -----------------------------------------------------------------------------------


def test_reports_escape_hostile_text() -> None:
    hostile = "ok</failure><x>\x1b[31m | <script>alert(1)</script>\x00"
    case = _case("evil", [Outcome.FAIL]).model_copy(update={"task": hostile})
    sample = case.samples[0].model_copy(update={"failures": (f"checks:answer: {hostile}",)})
    case = case.model_copy(update={"samples": (sample,)})
    report = _summarise([case, _case("fine", [Outcome.PASS])])

    root = ET.fromstring(render_junit(report))  # noqa: S314 - our own output
    suites = root.findall("testsuite")
    assert [s.get("name") for s in suites] == ["gate.s", "gate.thresholds"]
    evil = suites[0].find("testcase[@name='evil']")
    assert evil is not None
    failure = evil.find("failure")
    assert failure is not None
    assert "<script>" in (failure.get("message") or "")  # escaped in XML, intact once parsed
    assert "\x00" not in (failure.text or "")
    assert "\x1b" not in (failure.text or "")
    assert suites[1].get("failures") == "1"  # success_drop

    markdown = render_markdown(report)
    assert "<script>" not in markdown
    row = next(line for line in markdown.splitlines() if line.startswith("| evil"))
    assert row.count(" | ") == 3  # the pipe inside the message did not add a column


# --- CLI ---------------------------------------------------------------------------------------


def test_cli_suite_record_and_gate(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "suite"
    assert cli.main(["suite", "record", "--out-dir", str(root), "--name", "desk"]) == 0
    assert json.loads(capsys.readouterr().out)["suite"] == "desk"

    junit, md, report = tmp_path / "gate.xml", tmp_path / "gate.md", tmp_path / "out/gate.json"
    args = ["gate", "--baseline", str(root), "--junit", str(junit), "--markdown", str(md)]
    assert cli.main([*args, "--json", str(report)]) == 0
    assert json.loads(report.read_text())["verdict"] == "pass"
    assert cli.main([*args, "--policy", "amount-drift"]) == 1
    assert "Agent gate: BREACHED" in md.read_text()
    ET.parse(junit)  # noqa: S314 - our own output

    config = tmp_path / "gate.toml"
    config.write_text("[gate]\nmax_cost_ratio = 3.0\nmax_success_drop = 0.5\n", encoding="utf-8")
    assert cli.main([*args, "--policy", "check-policy", "--config", str(config)]) == 0
    assert cli.main([*args, "--policy", "check-policy", "--config", str(config),
                     "--max-success-drop", "0.0"]) == 1  # fmt: skip
