"""Injectable time sources so recordings and tests are reproducible."""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime: ...

    def monotonic(self) -> float: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)

    def monotonic(self) -> float:
        return time.monotonic()


class TickingClock:
    """Deterministic clock that advances by a fixed step on every read."""

    def __init__(
        self,
        start: datetime = datetime(2026, 1, 1, tzinfo=UTC),
        step: timedelta = timedelta(milliseconds=10),
    ) -> None:
        self._current = start
        self._step = step
        self._mono = 0.0

    def now(self) -> datetime:
        value = self._current
        self._current += self._step
        return value

    def monotonic(self) -> float:
        value = self._mono
        self._mono += self._step.total_seconds()
        return value
