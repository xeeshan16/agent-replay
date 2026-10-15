"""The storage interface shared by the in-memory and PostgreSQL stores."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from agent_replay.ingest.spans import SpanRecord
from agent_replay.model import Trajectory

TraceBuilder = Callable[[Sequence[SpanRecord]], list[Trajectory]]


class SaveOutcome(StrEnum):
    CREATED = "created"
    UPDATED = "updated"
    UNCHANGED = "unchanged"


@dataclass(frozen=True, slots=True)
class SaveResult:
    trajectory_id: str
    outcome: SaveOutcome
    revision: int


@dataclass(frozen=True, slots=True)
class AddSpansResult:
    inserted: int
    duplicates: int
    traces: frozenset[str]  # traces that received at least one new span


class TrajectorySummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    trajectory_id: str
    agent_name: str
    status: str
    source: str
    started_at: datetime
    step_count: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    fingerprint: str
    revision: int


class StoreStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    trajectories: int
    steps: int
    blobs: int
    blob_bytes: int
    logical_bytes: int
    buffered_spans: int

    @property
    def dedup_ratio(self) -> float:
        """Logical trajectory size over stored blob size (higher is better)."""
        return round(self.logical_bytes / self.blob_bytes, 2) if self.blob_bytes else 0.0


class TrajectoryStore(Protocol):
    async def save(self, trajectory: Trajectory) -> SaveResult:
        """Insert or update by trajectory ID. Saving identical content is a no-op."""

    async def get(self, trajectory_id: str) -> Trajectory | None: ...

    async def list_trajectories(
        self, *, agent_name: str | None = None, limit: int = 50
    ) -> list[TrajectorySummary]: ...

    async def known_traces(self, trace_ids: Sequence[str]) -> set[str]:
        """The subset of ``trace_ids`` that already has buffered spans."""

    async def add_spans(self, spans: Sequence[SpanRecord]) -> AddSpansResult:
        """Buffer spans, ignoring ones already stored (same trace and span ID)."""

    async def rebuild_trace(self, trace_id: str, build: TraceBuilder) -> list[SaveResult]:
        """Re-assemble a trace from all its buffered spans and save the result.

        Implementations serialise rebuilds of the same trace so concurrent batches cannot
        overwrite a newer assembly with an older one.
        """

    async def stats(self) -> StoreStats: ...

    async def close(self) -> None: ...
