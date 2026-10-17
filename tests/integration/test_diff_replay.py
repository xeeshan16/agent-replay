"""The diff on real replays of the reference agent: every seeded regression is localised."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agent_replay.cli import EXIT_DIVERGED, EXIT_OK, main
from agent_replay.clock import TickingClock
from agent_replay.diff import DivergenceReason, TrajectoryDiff, diff_trajectories
from agent_replay.matching import MatchOutcome
from agent_replay.model import Trajectory
from agent_replay.recorder import record_run
from agent_replay.reference import support_desk as desk
from agent_replay.replay import ReplayMode, replay
from agent_replay.sandbox import SandboxPolicy, UnrecordedWrites

pytestmark = pytest.mark.integration

REFUND_1001 = "I'd like a refund for ORD-1001, it arrived damaged."
REFUND_1004 = "Refund ORD-1004 please, the box was crushed."
LATER = datetime(2026, 11, 1, 8, 30, tzinfo=UTC)


async def _replay_diff(
    task: str,
    policy: str,
    *,
    live_reads: bool = False,
    unrecorded_writes: UnrecordedWrites = UnrecordedWrites.FAIL,
) -> TrajectoryDiff:
    recording = await record_run(desk.build_agent(), task, clock=TickingClock())
    result = await replay(
        desk.build_agent(policy=desk.POLICIES[policy], clock=TickingClock(LATER)),
        recording,
        mode=ReplayMode.TOOLS,
        clock=TickingClock(LATER),
        policy=desk.match_policy(),
        sandbox=SandboxPolicy(live_reads=live_reads, unrecorded_writes=unrecorded_writes),
    )
    return diff_trajectories(
        recording, result.replayed, normalizer=desk.NORMALIZER, candidate_events=result.matches
    )


@pytest.mark.parametrize("task", desk.SAMPLE_TASKS)
async def test_unchanged_agent_diffs_identical(task: str) -> None:
    diff = await _replay_diff(task, "v1")
    assert diff.identical, diff.divergences
    assert diff.similarity == 1.0


async def test_skipped_lookup_is_localised_to_the_first_model_call() -> None:
    diff = await _replay_diff(REFUND_1001, "skip-lookup")
    first = diff.first_divergence
    assert first is not None
    assert first.reason is DivergenceReason.MISSING_STEP
    assert first.tool == "lookup_order"
    assert first.decision_step == 0
    refund = next(d for d in diff.divergences if d.tool == "issue_refund")
    assert refund.reason is DivergenceReason.ARGUMENT_DRIFT
    assert refund.side_effect
    assert "never executed" in refund.summary
    assert "[replay: blocked]" in refund.summary
    assert DivergenceReason.STATUS_CHANGED in {d.reason for d in diff.divergences}


async def test_amount_drift_reports_the_changed_argument() -> None:
    diff = await _replay_diff(REFUND_1001, "amount-drift")
    first = diff.first_divergence
    assert first is not None
    assert first.reason is DivergenceReason.ARGUMENT_DRIFT
    assert first.tool == "issue_refund"
    assert [(c.path, c.before, c.after) for c in first.changes] == [("amount", 59.9, 60.0)]
    # The fresh idempotency key is normalised away; only the amount is drift.
    assert first.decision_step == 2


async def test_missing_retry_is_a_missing_lookup_after_the_failure() -> None:
    diff = await _replay_diff(REFUND_1004, "no-retry")
    first = diff.first_divergence
    assert first is not None
    assert first.reason is DivergenceReason.MISSING_STEP
    assert first.tool == "lookup_order"
    assert first.baseline_step == 3
    assert first.decision_step == 2  # the model call that saw the failure and gave up
    assert [d.reason for d in diff.divergences] == [
        DivergenceReason.MISSING_STEP,
        DivergenceReason.MISSING_STEP,
        DivergenceReason.DIFFERENT_ANSWER,
    ]


async def test_benign_extra_read_is_the_only_difference() -> None:
    diff = await _replay_diff(REFUND_1001, "check-policy", live_reads=True)
    assert [d.reason for d in diff.divergences] == [DivergenceReason.EXTRA_STEP]
    extra = diff.divergences[0]
    assert extra.tool == "get_refund_policy"
    assert not extra.side_effect
    served = [p.candidate.served for p in diff.alignment if p.candidate is not None]
    assert served[1] is MatchOutcome.LIVE
    assert {served[0], served[2]} <= {MatchOutcome.EXACT, MatchOutcome.NORMALIZED}


async def test_blocked_write_keeps_the_rest_of_the_run_visible() -> None:
    diff = await _replay_diff(REFUND_1001, "skip-lookup", unrecorded_writes=UnrecordedWrites.BLOCK)
    reasons = [d.reason for d in diff.divergences]
    assert reasons == [
        DivergenceReason.EXTRA_STEP,
        DivergenceReason.MISSING_STEP,
        DivergenceReason.DIFFERENT_ANSWER,
    ]
    blocked = diff.alignment[0].candidate
    assert blocked is not None
    assert blocked.served is MatchOutcome.BLOCKED
    assert blocked.is_error


def _record_cli(path: Path, policy: str = "v1") -> None:
    assert (
        main(["record", "--task", REFUND_1001, "--out", str(path), "--policy", policy]) == EXIT_OK
    )


def test_cli_diff_exit_codes_and_formats(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base, same, drift = tmp_path / "base.json", tmp_path / "same.json", tmp_path / "drift.json"
    _record_cli(base)
    _record_cli(same)
    _record_cli(drift, "amount-drift")
    capsys.readouterr()

    assert main(["diff", str(base), str(same)]) == EXIT_OK
    assert "**Verdict:** identical" in capsys.readouterr().out

    out = tmp_path / "diff.json"
    assert main(["diff", str(base), str(drift), "--format", "json", "--out", str(out)]) == (
        EXIT_DIVERGED
    )
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["identical"] is False
    assert report["first_divergence"]["reason"] == "argument_drift"
    assert report["first_divergence"]["changes"][0]["path"] == "amount"

    # Without normalisation the fresh idempotency keys alone are a difference.
    assert main(["diff", str(base), str(same), "--normalizer", "none"]) == EXIT_DIVERGED


def test_cli_replay_writes_a_markdown_diff(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run, md = tmp_path / "run.json", tmp_path / "diff.md"
    _record_cli(run)
    capsys.readouterr()
    code = main(["replay", str(run), "--policy", "skip-lookup", "--diff-out", str(md)])
    assert code == EXIT_DIVERGED
    report = json.loads(capsys.readouterr().out)
    assert report["first_divergence"]["reason"] == "missing_step"
    assert report["divergences"]["missing_step"] == 1
    text = md.read_text(encoding="utf-8")
    assert "**First divergence:** `missing_step`" in text
    assert "Involves a side-effecting tool" not in text.split("**All differences")[0]


def test_trajectory_json_round_trip_diffs_identical(tmp_path: Path) -> None:
    run = tmp_path / "run.json"
    _record_cli(run)
    trajectory = Trajectory.from_json(run.read_bytes())
    assert diff_trajectories(trajectory, trajectory, normalizer=desk.NORMALIZER).identical
