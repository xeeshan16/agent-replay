from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_replay import determinism
from agent_replay.cli import main
from agent_replay.determinism import (
    DeterminismReport,
    PairResult,
    render_markdown,
    run_determinism,
)
from agent_replay.reference import support_desk as desk
from agent_replay.replay import ReplayMode

TASKS = len(desk.SAMPLE_TASKS)


def _pair(pair: int, *, otlp: int = TASKS, tools: int = TASKS) -> PairResult:
    return PairResult(
        pair=pair,
        record_seed=2 * pair + 1,
        replay_seed=2 * pair + 2,
        tasks=TASKS,
        live={"none": 1, "default": 2, "support-desk": TASKS},
        replay={
            "none": {"full": 0, "tools": 3},
            "default": {"full": TASKS, "tools": TASKS - 1},
            "support-desk": {"full": TASKS, "tools": tools},
        },
        otlp_fingerprint=otlp,
        otlp_replay=otlp,
    )


def _report(*pairs: PairResult) -> DeterminismReport:
    return DeterminismReport(provider="fake", settings={"tasks": TASKS}, pairs=pairs)


def test_report_sums_pairs_and_requires_every_cell_of_the_agent_normaliser() -> None:
    report = _report(_pair(0), _pair(1))
    assert report.runs == 2 * TASKS
    assert report.live("default") == 4
    assert report.replayed("none", ReplayMode.TOOLS) == 6
    assert report.otlp == (2 * TASKS, 2 * TASKS)
    assert report.stable
    assert not _report(_pair(0), _pair(1, tools=TASKS - 1)).stable
    assert not _report(_pair(0), _pair(1, otlp=TASKS - 1)).stable

    text = render_markdown(report)
    assert f"| none | 2/{2 * TASKS} (14%) | 0/{2 * TASKS} (0%) |" in text
    assert "**yes**" in text


def test_one_cross_process_pair_is_stable_only_with_the_agent_normaliser(tmp_path: Path) -> None:
    report = run_determinism(1, work_dir=tmp_path)

    assert report.runs == TASKS
    assert report.stable
    # The two workers run in different wall-clock seconds and the system prompt carries the
    # time, so an un-normalised FULL replay can never match the recorded model calls.
    assert report.replayed("none", ReplayMode.FULL) == 0
    assert report.replayed("default", ReplayMode.FULL) == TASKS
    # Live re-runs mint fresh ticket and refund IDs; only the agent's normaliser masks both.
    assert report.live("support-desk") == TASKS
    assert report.live("none") < TASKS
    [pair] = report.pairs
    assert pair.record_seed != pair.replay_seed
    recorded = sorted(p.name for p in (tmp_path / "pair-00").iterdir())
    assert len(recorded) == 2 * TASKS  # a trajectory and its OTLP spans per task


def test_a_failing_worker_is_reported_with_its_stderr(tmp_path: Path) -> None:
    missing = tmp_path / "nothing-recorded"
    with pytest.raises(RuntimeError, match="replay worker failed"):
        determinism._run_worker("replay", missing, seed=7)


def test_pairs_must_be_positive() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        run_determinism(0)


def test_cli_writes_reports_and_exits_zero_when_stable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["determinism", "--pairs", "1", "--out-dir", str(tmp_path)]) == 0
    payload = json.loads((tmp_path / "determinism.json").read_text())
    assert payload["stable"] is True
    assert payload["runs"] == TASKS
    assert payload["otlp"] == {"fingerprint": TASKS, "replay_full": TASKS}
    assert (tmp_path / "determinism.md").read_text() in capsys.readouterr().out


def test_workers_run_in_process_too(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert determinism._worker_main(["record", str(tmp_path)]) == 0
    assert determinism._worker_main(["replay", str(tmp_path)]) == 0
    raw = json.loads(capsys.readouterr().out)
    # Same process and (almost always) the same second: only the minted IDs differ.
    assert raw["tasks"] == TASKS
    assert raw["replay"]["support-desk"] == {"full": TASKS, "tools": TASKS}
    assert raw["otlp_fingerprint"] == raw["otlp_replay"] == TASKS
