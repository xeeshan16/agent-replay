from __future__ import annotations

import pytest

from agent_replay.stats import bootstrap_mean, bootstrap_paired_delta, wilson


def test_wilson_matches_reference_values() -> None:
    # Reference values from the closed form (z = 1.959964).
    seven = wilson(7, 7)
    assert (seven.estimate, seven.high) == (1.0, 1.0)
    assert seven.low == pytest.approx(0.6457, abs=1e-4)
    eight = wilson(8, 10)
    assert eight.low == pytest.approx(0.4902, abs=1e-4)
    assert eight.high == pytest.approx(0.9433, abs=1e-4)
    zero = wilson(0, 10)
    assert zero.low == 0.0
    assert zero.high == pytest.approx(0.2775, abs=1e-4)


def test_wilson_accepts_fractional_successes_and_validates() -> None:
    assert wilson(2.5, 5).estimate == 0.5
    with pytest.raises(ValueError, match="within"):
        wilson(6, 5)
    with pytest.raises(ValueError, match="n must"):
        wilson(0, 0)
    with pytest.raises(ValueError, match="confidence"):
        wilson(1, 2, confidence=1.0)


def test_bootstrap_collapses_at_the_boundary_where_wilson_does_not() -> None:
    clusters = [[1.0]] * 7
    boot = bootstrap_mean(clusters)
    assert (boot.low, boot.high) == (1.0, 1.0)
    assert wilson(7, 7).low < 0.7  # the reason the gate takes the lower of the two


def test_bootstrap_is_seeded_and_reproducible() -> None:
    clusters = [[1.0], [0.0], [1.0], [1.0], [0.0], [1.0]]
    assert bootstrap_mean(clusters, seed=3) == bootstrap_mean(clusters, seed=3)
    interval = bootstrap_mean(clusters, seed=3)
    assert interval.low < interval.estimate < interval.high
    assert interval.contains(interval.estimate)


def test_repeats_inside_a_task_do_not_narrow_the_interval() -> None:
    # Two tasks, ten repeats each: one always passes, one always fails. Resampling runs would
    # treat this as 20 independent coin flips; resampling tasks keeps the honest width.
    clustered = bootstrap_mean([[1.0] * 10, [0.0] * 10], seed=1)
    assert clustered.estimate == 0.5
    assert (clustered.low, clustered.high) == (0.0, 1.0)
    flat = bootstrap_mean([[1.0]] * 10 + [[0.0]] * 10, seed=1)
    assert flat.high - flat.low < 0.5


def test_paired_delta() -> None:
    same = bootstrap_paired_delta([[1.0], [0.0], [1.0]], [[1.0], [0.0], [1.0]])
    assert (same.estimate, same.low, same.high) == (0.0, 0.0, 0.0)
    one_lost = bootstrap_paired_delta([[1.0]] * 7, [[1.0]] * 6 + [[0.0]])
    assert one_lost.estimate == pytest.approx(-1 / 7, abs=1e-6)
    assert one_lost.low < one_lost.estimate <= one_lost.high <= 0.0
    with pytest.raises(ValueError, match="same tasks"):
        bootstrap_paired_delta([[1.0]], [[1.0], [0.0]])
    with pytest.raises(ValueError, match="at least one"):
        bootstrap_paired_delta([[1.0], []], [[1.0], [1.0]])
