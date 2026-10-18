from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_replay import cli
from agent_replay.evaluation import (
    DEFAULT_DATASET,
    DETECTORS,
    EvalCase,
    load_cases,
    render_markdown,
    run_case,
    run_eval,
)
from agent_replay.reference import support_desk as desk

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / DEFAULT_DATASET


@pytest.fixture(scope="module")
def cases() -> tuple[EvalCase, ...]:
    return load_cases(DATASET)


def test_dataset_is_well_formed(cases: tuple[EvalCase, ...]) -> None:
    assert len(cases) == len(desk.POLICIES) * 8  # every policy on every task
    assert {c.policy for c in cases} == set(desk.POLICIES)
    assert all(c.label == "no_effect" for c in cases if c.policy == "v1")
    assert {c.label for c in cases if c.policy == "check-policy"} == {"benign", "no_effect"}
    labels = [c.label for c in cases]
    assert (labels.count("regression"), labels.count("benign")) == (18, 5)


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ('{"id": "a", "task": "t", "policy": "nope", "label": "no_effect", "category": "x"}',
         "unknown policy"),
        ('{"id": "a", "task": "t", "policy": "v1", "label": "regression", "category": "x"}',
         "has no step"),
    ],
)  # fmt: skip
def test_dataset_validation(tmp_path: Path, line: str, message: str) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_cases(path)


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    line = '{"id": "a", "task": "t", "policy": "v1", "label": "no_effect", "category": "none"}'
    path = tmp_path / "cases.jsonl"
    path.write_text(f"{line}\n\n{line}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(path)


async def test_controls_are_never_flagged(cases: tuple[EvalCase, ...]) -> None:
    for case in (c for c in cases if c.policy == "v1"):
        result = await run_case(case, desk.scoring_spec())
        assert not any(result.flags.values()), (case.id, result.failures)
        assert result.first_divergence is None
        assert result.judge_calls == 1


async def test_full_eval_is_deterministic_and_consistent(cases: tuple[EvalCase, ...]) -> None:
    spec = desk.scoring_spec()
    first = await run_eval(cases, spec)
    second = await run_eval(cases, spec)
    assert first.to_json() == second.to_json()

    for detector in DETECTORS:
        m = first.metrics(detector)
        assert m.tp + m.fp + m.fn + m.tn == len(cases)
    # The diff sees every behaviour change, so it flags all benign changes too.
    assert first.metrics("diff").fn == 0
    assert first.metrics("diff").benign_flagged == 5
    # Cases whose policy change does not apply replay identically.
    assert not any(c.flags["diff"] for c in first.cases if c.label == "no_effect")
    # The judge only runs when the cheaper layers passed, so it costs less than always-on.
    layered, always = first.judge_cost
    assert 0 < layered.cost_usd < always.cost_usd

    text = render_markdown(first)
    assert "| layered |" in text
    assert "First-divergence decision step matches the label" in text


async def test_cli_eval_writes_reports(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    subset = tmp_path / "subset.jsonl"
    lines = DATASET.read_text(encoding="utf-8").splitlines()
    subset.write_text(
        "\n".join(row for row in lines if '"amount-drift/' in row) + "\n", encoding="utf-8"
    )
    args = cli.build_parser().parse_args(
        ["eval", "--dataset", str(subset), "--out-dir", str(tmp_path / "out")]
    )
    assert await cli._eval(args) == cli.EXIT_OK
    report = json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))
    assert report["labels"] == {"no_effect": 7, "regression": 1}
    assert report["metrics"]["layered"]["tp"] == 1
    assert "Regression-detection eval" in capsys.readouterr().out
