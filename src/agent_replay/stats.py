"""Small-sample statistics for the CI gate: cluster bootstrap and Wilson intervals.

A gate suite is usually a few dozen tasks, often fewer, and each task may be run several times
against a non-deterministic model. Two facts shape the statistics:

* **Runs of the same task are not independent.** A task the candidate gets wrong it tends to get
  wrong every time. Resampling individual runs would treat ten repeats of one task as ten pieces
  of evidence and produce intervals that are far too narrow. The bootstrap here resamples
  *tasks* (clusters) and uses each task's mean over its repeats.
* **Percentile bootstrap collapses at the boundary.** When every task passes, every resample
  has a success rate of 1.0 and the interval is ``[1.0, 1.0]``: no uncertainty at all from
  seven tasks, which is wrong. The Wilson score interval does not have that problem (seven out
  of seven gives a 95% lower bound of about 0.65). The gate therefore uses the *lower* of the
  two lower bounds (see ADR 0005).

Everything is stdlib-only and seeded, so a gate report is reproducible bit for bit.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from statistics import NormalDist, fmean

from pydantic import BaseModel, ConfigDict

DEFAULT_RESAMPLES = 2000


class Interval(BaseModel):
    model_config = ConfigDict(frozen=True)

    estimate: float
    low: float
    high: float
    confidence: float
    method: str

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high


def _check_confidence(confidence: float) -> None:
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (``q`` in [0, 1]) of already sorted values."""
    if not sorted_values:
        raise ValueError("no values")
    position = q * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _round(value: float) -> float:
    return round(value, 6)


def cluster_means(clusters: Sequence[Sequence[float]]) -> list[float]:
    if not clusters or any(not c for c in clusters):
        raise ValueError("every cluster needs at least one value")
    return [fmean(c) for c in clusters]


def bootstrap_mean(
    clusters: Sequence[Sequence[float]],
    *,
    confidence: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap CI for the mean, resampling clusters (tasks) with replacement."""
    _check_confidence(confidence)
    if resamples < 1:
        raise ValueError("resamples must be >= 1")
    means = cluster_means(clusters)
    rng = random.Random(seed)  # noqa: S311 - resampling, not cryptography
    n = len(means)
    stats = sorted(fmean(rng.choices(means, k=n)) for _ in range(resamples))
    alpha = 1 - confidence
    return Interval(
        estimate=_round(fmean(means)),
        low=_round(_percentile(stats, alpha / 2)),
        high=_round(_percentile(stats, 1 - alpha / 2)),
        confidence=confidence,
        method="cluster bootstrap (percentile)",
    )


def bootstrap_paired_delta(
    baseline: Sequence[Sequence[float]],
    candidate: Sequence[Sequence[float]],
    *,
    confidence: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> Interval:
    """CI for ``mean(candidate) - mean(baseline)`` over the same tasks.

    Pairing matters: a hard task that both versions fail contributes zero to the difference
    instead of widening both marginal intervals. Tasks are resampled together.
    """
    _check_confidence(confidence)
    if len(baseline) != len(candidate):
        raise ValueError("baseline and candidate must cover the same tasks")
    deltas = [c - b for b, c in zip(cluster_means(baseline), cluster_means(candidate), strict=True)]
    rng = random.Random(seed)  # noqa: S311 - resampling, not cryptography
    n = len(deltas)
    stats = sorted(fmean(rng.choices(deltas, k=n)) for _ in range(resamples))
    alpha = 1 - confidence
    return Interval(
        estimate=_round(fmean(deltas)),
        low=_round(_percentile(stats, alpha / 2)),
        high=_round(_percentile(stats, 1 - alpha / 2)),
        confidence=confidence,
        method="paired cluster bootstrap (percentile)",
    )


def wilson(successes: float, n: int, *, confidence: float = 0.95) -> Interval:
    """Wilson score interval for a proportion. ``successes`` may be fractional (a sum of
    per-task success rates), which keeps the effective sample size at the number of tasks."""
    _check_confidence(confidence)
    if n < 1:
        raise ValueError("n must be >= 1")
    if not 0 <= successes <= n:
        raise ValueError("successes must be within [0, n]")
    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return Interval(
        estimate=_round(p),
        low=_round(max(0.0, centre - half)),
        high=_round(min(1.0, centre + half)),
        confidence=confidence,
        method="Wilson score",
    )
