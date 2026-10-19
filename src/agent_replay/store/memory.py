"""In-memory store with the same semantics as the PostgreSQL store.

Used by tests, the demo and ``serve`` without a database. It goes through the same codec as the
PostgreSQL store, so blob deduplication and integrity checks behave identically.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from agent_replay.ingest.spans import SpanRecord
from agent_replay.model import Trajectory
from agent_replay.store import codec
from agent_replay.store.base import (
    AddSpansResult,
    PruneResult,
    SaveOutcome,
    SaveResult,
    StoreStats,
    SuiteSummary,
    SweepResult,
    TraceBuilder,
    TrajectorySummary,
)
from agent_replay.suite import SuiteManifest, check_stored


@dataclass(slots=True)
class _Row:
    header: codec.Header
    steps: tuple[codec.StepRow, ...]
    revision: int


class InMemoryStore:
    def __init__(self) -> None:
        self._rows: dict[str, _Row] = {}
        self._blobs: dict[str, str] = {}
        self._spans: dict[str, dict[str, SpanRecord]] = defaultdict(dict)
        self._trace_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._blob_created: dict[str, datetime] = {}
        self._span_received: dict[str, datetime] = {}  # trace -> newest span arrival
        self._pruned: set[str] = set()
        self._suites: dict[str, tuple[SuiteManifest, int]] = {}

    async def save(self, trajectory: Trajectory) -> SaveResult:
        return self._save(codec.encode(trajectory))

    def _save(self, encoded: codec.EncodedTrajectory) -> SaveResult:
        header = encoded.header
        current = self._rows.get(header.trajectory_id)
        if current is not None and current.header.content_digest == header.content_digest:
            return SaveResult(header.trajectory_id, SaveOutcome.UNCHANGED, current.revision)
        now = datetime.now(UTC)
        for digest, text in encoded.blobs.items():
            self._blobs.setdefault(digest, text)
            self._blob_created.setdefault(digest, now)
        revision = 1 if current is None else current.revision + 1
        self._rows[header.trajectory_id] = _Row(header, encoded.steps, revision)
        outcome = SaveOutcome.CREATED if current is None else SaveOutcome.UPDATED
        return SaveResult(header.trajectory_id, outcome, revision)

    async def get(self, trajectory_id: str) -> Trajectory | None:
        row = self._rows.get(trajectory_id)
        if row is None:
            return None
        return codec.decode(row.header, row.steps, self._blobs)

    async def list_trajectories(
        self, *, agent_name: str | None = None, limit: int = 50
    ) -> list[TrajectorySummary]:
        rows = [
            r
            for r in self._rows.values()
            if agent_name is None or r.header.agent_name == agent_name
        ]
        rows.sort(key=lambda r: (r.header.started_at, r.header.trajectory_id), reverse=True)
        return [_summary(r) for r in rows[:limit]]

    async def known_traces(self, trace_ids: Sequence[str]) -> set[str]:
        return {t for t in trace_ids if t in self._spans}

    async def add_spans(self, spans: Sequence[SpanRecord]) -> AddSpansResult:
        inserted, expired, traces = 0, 0, set()
        now = datetime.now(UTC)
        for span in spans:
            if span.trace_id in self._pruned:
                expired += 1
                continue
            bucket = self._spans[span.trace_id]
            if span.span_id in bucket:
                continue
            bucket[span.span_id] = span
            self._span_received[span.trace_id] = now
            inserted += 1
            traces.add(span.trace_id)
        return AddSpansResult(
            inserted, len(spans) - inserted - expired, frozenset(traces), expired=expired
        )

    async def prune_spans(self, received_before: datetime) -> PruneResult:
        old = [t for t, at in self._span_received.items() if at < received_before]
        deleted = 0
        for trace_id in old:
            async with self._trace_locks[trace_id]:
                deleted += len(self._spans.pop(trace_id, {}))
                self._span_received.pop(trace_id, None)
                self._pruned.add(trace_id)
        return PruneResult(traces=len(old), spans=deleted)

    async def sweep_blobs(self, created_before: datetime) -> SweepResult:
        # Single-threaded event loop and no awaits below: nothing can save in between.
        step_level = codec.step_digests(s for r in self._rows.values() for s in r.steps)
        bodies = {d: self._blobs[d] for d in step_level if d in self._blobs}
        live = step_level | codec.nested_digests(bodies)
        candidates = [d for d, at in self._blob_created.items() if at < created_before]
        dead = [d for d in candidates if d not in live]
        freed = sum(len(self._blobs[d].encode("utf-8")) for d in dead)
        for digest in dead:
            del self._blobs[digest]
            del self._blob_created[digest]
        return SweepResult(
            scanned=len(candidates), deleted=len(dead), deleted_bytes=freed, live=len(live)
        )

    async def save_suite(self, manifest: SuiteManifest) -> SaveResult:
        fingerprints = {
            case.trajectory_id: self._rows[case.trajectory_id].header.fingerprint
            for case in manifest.cases
            if case.trajectory_id in self._rows
        }
        stored = check_stored(manifest, fingerprints)
        current = self._suites.get(stored.name)
        if current is None:
            self._suites[stored.name] = (stored, 1)
            return SaveResult(stored.name, SaveOutcome.CREATED, 1)
        if current[0] == stored:
            return SaveResult(stored.name, SaveOutcome.UNCHANGED, current[1])
        self._suites[stored.name] = (stored, current[1] + 1)
        return SaveResult(stored.name, SaveOutcome.UPDATED, current[1] + 1)

    async def get_suite(self, name: str) -> SuiteManifest | None:
        found = self._suites.get(name)
        return None if found is None else found[0]

    async def list_suites(self) -> list[SuiteSummary]:
        return [
            SuiteSummary(
                name=m.name,
                description=m.description,
                agent_name=m.agent_name,
                case_count=len(m.cases),
                revision=revision,
                created_at=m.created_at,
            )
            for m, revision in sorted(self._suites.values(), key=lambda item: item[0].name)
        ]

    async def rebuild_trace(self, trace_id: str, build: TraceBuilder) -> list[SaveResult]:
        async with self._trace_locks[trace_id]:
            if trace_id in self._pruned:
                return []
            spans = list(self._spans.get(trace_id, {}).values())
            if not any(s.operation == "invoke_agent" for s in spans):
                return []
            return [self._save(codec.encode(t)) for t in build(spans)]

    async def stats(self) -> StoreStats:
        return StoreStats(
            trajectories=len(self._rows),
            steps=sum(len(r.steps) for r in self._rows.values()),
            blobs=len(self._blobs),
            blob_bytes=sum(len(t.encode("utf-8")) for t in self._blobs.values()),
            logical_bytes=sum(r.header.logical_bytes for r in self._rows.values()),
            buffered_spans=sum(len(b) for b in self._spans.values()),
        )

    async def close(self) -> None:
        return None


def _summary(row: _Row) -> TrajectorySummary:
    h = row.header
    return TrajectorySummary(
        trajectory_id=h.trajectory_id,
        agent_name=h.agent_name,
        status=h.status,
        source=h.source,
        started_at=h.started_at,
        step_count=h.step_count,
        input_tokens=h.input_tokens,
        output_tokens=h.output_tokens,
        cost_usd=h.cost_usd,
        fingerprint=h.fingerprint,
        revision=row.revision,
    )
