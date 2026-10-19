"""PostgreSQL store (async SQLAlchemy Core on asyncpg).

Schema (see ``tables`` below):

* ``blobs``: content-addressed canonical JSON, insert-if-absent. Shared by every trajectory.
* ``trajectories``: one row per trajectory with denormalised summary columns (status, tokens,
  cost, fingerprint) so listing and gating never touch blobs; ``revision`` counts content changes.
* ``trajectory_steps``: ordered step rows pointing at their input and output blobs.
* ``otlp_spans``: the raw span buffer, unique on (trace_id, span_id), which is what makes
  re-delivered OTLP batches harmless.
* ``pruned_traces``: tombstones for traces whose buffered spans were deleted by retention, so a
  straggler span is dropped instead of re-assembling a partial trace over a complete one.
* ``suites``: named baseline suites (a manifest of trajectory IDs and pinned fingerprints).

Blob sweep: blobs are shared and referenced from inside other blobs, so they cannot be deleted
with a foreign key cascade. ``sweep_blobs`` marks every digest reachable from a stored step and
deletes the rest under an exclusive advisory lock; every save takes the same lock in shared
mode. A save therefore either commits before the sweep marks (its blobs are live) or starts
after the sweep committed (its blobs are re-inserted), and a blob a running save just reused can
never be swept from under it.

Concurrency: rebuilding a trace takes a transaction-scoped advisory lock on the trace ID, so two
workers ingesting different batches of the same trace serialise instead of racing, and each save
locks its trajectory row (``SELECT ... FOR UPDATE``) before comparing content digests.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Double,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    PrimaryKeyConstraint,
    Table,
    Text,
    func,
    select,
    text,
    update,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from agent_replay.ingest.spans import SpanRecord
from agent_replay.logs import get_logger
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
from agent_replay.suite import SuiteError, SuiteManifest, check_stored

_log = get_logger("store.postgres")
metadata = MetaData()

blobs = Table(
    "blobs",
    metadata,
    Column("digest", Text, primary_key=True),
    Column("body", Text, nullable=False),
    Column("size_bytes", Integer, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)

trajectories = Table(
    "trajectories",
    metadata,
    Column("trajectory_id", Text, primary_key=True),
    Column("schema_version", Integer, nullable=False),
    Column("source", Text, nullable=False),
    Column("trace_id", Text),
    Column("agent_name", Text, nullable=False),
    Column("task_input", Text, nullable=False),
    Column("final_output", Text),
    Column("status", Text, nullable=False),
    Column("error", Text),
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("duration_ms", Double, nullable=False),
    Column("step_count", Integer, nullable=False),
    Column("input_tokens", BigInteger, nullable=False),
    Column("output_tokens", BigInteger, nullable=False),
    Column("cost_usd", Double, nullable=False),
    Column("fingerprint", Text, nullable=False),
    Column("content_digest", Text, nullable=False),
    Column("logical_bytes", BigInteger, nullable=False),
    Column("metadata", JSONB, nullable=False),
    Column("revision", Integer, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Index("ix_trajectories_agent_started", "agent_name", "started_at"),
    Index("ix_trajectories_trace", "trace_id"),
    Index("ix_trajectories_fingerprint", "fingerprint"),
)

steps = Table(
    "trajectory_steps",
    metadata,
    Column(
        "trajectory_id",
        Text,
        ForeignKey("trajectories.trajectory_id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("idx", Integer, nullable=False),
    Column("kind", Text, nullable=False),
    Column("name", Text, nullable=False),
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("duration_ms", Double, nullable=False),
    Column("input_digest", Text, ForeignKey("blobs.digest"), nullable=False),
    Column("output_digest", Text, ForeignKey("blobs.digest"), nullable=False),
    Column("call_id", Text),
    Column("side_effect", Boolean, nullable=False),
    Column("is_error", Boolean, nullable=False),
    PrimaryKeyConstraint("trajectory_id", "idx"),
    Index("ix_steps_tool_name", "kind", "name"),
)

spans = Table(
    "otlp_spans",
    metadata,
    Column("trace_id", Text, nullable=False),
    Column("span_id", Text, nullable=False),
    Column("parent_span_id", Text),
    Column("operation", Text),
    Column("start_ns", BigInteger, nullable=False),
    Column("record", Text, nullable=False),
    Column("received_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    PrimaryKeyConstraint("trace_id", "span_id"),
)

pruned_traces = Table(
    "pruned_traces",
    metadata,
    Column("trace_id", Text, primary_key=True),
    Column("pruned_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)

suites = Table(
    "suites",
    metadata,
    Column("name", Text, primary_key=True),
    Column("description", Text, nullable=False),
    Column("agent_name", Text, nullable=False),
    Column("case_count", Integer, nullable=False),
    Column("manifest", Text, nullable=False),
    Column("revision", Integer, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)

# One advisory lock key guards blob reuse against the sweep (shared: saves, exclusive: sweep).
_SWEEP_LOCK_KEY = "agent_replay:blob-sweep"

# asyncpg caps a statement at 32767 bind parameters; stay well below for multi-row inserts.
_CHUNK = 500


def normalise_url(url: str) -> str:
    """Accept plain ``postgresql://`` URLs and select the asyncpg driver."""
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+asyncpg://" + url[len(prefix) :]
    return url


def _chunks[T](items: Sequence[T], size: int = _CHUNK) -> list[Sequence[T]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


class PostgresStore:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        pool_size: int = 5,
        connect_timeout_s: float = 5.0,
        statement_timeout_s: float = 30.0,
    ) -> PostgresStore:
        engine = create_async_engine(
            normalise_url(url),
            pool_size=pool_size,
            pool_pre_ping=True,
            connect_args={"timeout": connect_timeout_s, "command_timeout": statement_timeout_s},
        )
        return cls(engine)

    async def migrate(self) -> None:
        """Create tables and indexes if they do not exist (idempotent)."""
        async with self._engine.begin() as conn:
            await conn.run_sync(metadata.create_all)

    async def close(self) -> None:
        await self._engine.dispose()

    # --- trajectories -----------------------------------------------------------------------

    async def save(self, trajectory: Trajectory) -> SaveResult:
        encoded = codec.encode(trajectory)
        async with self._engine.begin() as conn:
            return await self._save(conn, encoded)

    async def _save(self, conn: AsyncConnection, encoded: codec.EncodedTrajectory) -> SaveResult:
        header = encoded.header
        await conn.execute(
            text("SELECT pg_advisory_xact_lock_shared(hashtextextended(:key, 0))"),
            {"key": _SWEEP_LOCK_KEY},
        )
        current = await self._lock_row(conn, header.trajectory_id)
        if current is not None and current[0] == header.content_digest:
            return SaveResult(header.trajectory_id, SaveOutcome.UNCHANGED, current[1])

        await self._put_blobs(conn, encoded.blobs)
        values = _header_values(header)
        if current is None:
            inserted = await conn.execute(
                pg_insert(trajectories)
                .values(**values, revision=1)
                .on_conflict_do_nothing(index_elements=["trajectory_id"])
                .returning(trajectories.c.revision)
            )
            if inserted.first() is not None:
                await self._put_steps(conn, header.trajectory_id, encoded.steps)
                return SaveResult(header.trajectory_id, SaveOutcome.CREATED, 1)
            # A concurrent transaction inserted the row first; lock it and update instead.
            current = await self._lock_row(conn, header.trajectory_id)
            assert current is not None  # noqa: S101 - the conflicting row exists and is locked
            if current[0] == header.content_digest:
                return SaveResult(header.trajectory_id, SaveOutcome.UNCHANGED, current[1])

        revision = current[1] + 1
        await conn.execute(
            update(trajectories)
            .where(trajectories.c.trajectory_id == header.trajectory_id)
            .values(**values, revision=revision, updated_at=func.now())
        )
        await conn.execute(steps.delete().where(steps.c.trajectory_id == header.trajectory_id))
        await self._put_steps(conn, header.trajectory_id, encoded.steps)
        return SaveResult(header.trajectory_id, SaveOutcome.UPDATED, revision)

    @staticmethod
    async def _lock_row(conn: AsyncConnection, trajectory_id: str) -> tuple[str, int] | None:
        row = (
            await conn.execute(
                select(trajectories.c.content_digest, trajectories.c.revision)
                .where(trajectories.c.trajectory_id == trajectory_id)
                .with_for_update()
            )
        ).first()
        return None if row is None else (row[0], row[1])

    @staticmethod
    async def _put_blobs(conn: AsyncConnection, items: dict[str, str]) -> None:
        rows = [
            {"digest": d, "body": t, "size_bytes": len(t.encode("utf-8"))}
            for d, t in sorted(items.items())  # stable order avoids lock-order deadlocks
        ]
        for chunk in _chunks(rows):
            await conn.execute(
                pg_insert(blobs)
                .values(list(chunk))
                .on_conflict_do_nothing(index_elements=["digest"])
            )

    @staticmethod
    async def _put_steps(
        conn: AsyncConnection, trajectory_id: str, rows: Sequence[codec.StepRow]
    ) -> None:
        values = [
            {
                "trajectory_id": trajectory_id,
                "idx": r.index,
                "kind": r.kind,
                "name": r.name,
                "started_at": r.started_at,
                "duration_ms": r.duration_ms,
                "input_digest": r.input_digest,
                "output_digest": r.output_digest,
                "call_id": r.call_id,
                "side_effect": r.side_effect,
                "is_error": r.is_error,
            }
            for r in rows
        ]
        for chunk in _chunks(values):
            await conn.execute(steps.insert().values(list(chunk)))

    async def get(self, trajectory_id: str) -> Trajectory | None:
        async with self._engine.connect() as conn:
            head = (
                await conn.execute(
                    select(trajectories).where(trajectories.c.trajectory_id == trajectory_id)
                )
            ).first()
            if head is None:
                return None
            step_rows = [
                codec.StepRow(
                    index=r.idx,
                    kind=r.kind,
                    name=r.name,
                    started_at=r.started_at,
                    duration_ms=r.duration_ms,
                    input_digest=r.input_digest,
                    output_digest=r.output_digest,
                    call_id=r.call_id,
                    side_effect=r.side_effect,
                    is_error=r.is_error,
                )
                for r in await conn.execute(
                    select(steps)
                    .where(steps.c.trajectory_id == trajectory_id)
                    .order_by(steps.c.idx)
                )
            ]
            # Two round trips: step-level blobs, then the messages and tool specs they reference.
            bodies = await self._fetch_blobs(conn, codec.step_digests(step_rows))
            bodies.update(await self._fetch_blobs(conn, codec.nested_digests(bodies) - set(bodies)))
        return codec.decode(_header_from_row(head._mapping), step_rows, bodies)

    @staticmethod
    async def _fetch_blobs(conn: AsyncConnection, digests: set[str]) -> dict[str, str]:
        found: dict[str, str] = {}
        ordered = sorted(digests)
        for chunk in _chunks(ordered, 5000):
            result = await conn.execute(
                select(blobs.c.digest, blobs.c.body).where(blobs.c.digest.in_(list(chunk)))
            )
            found.update({row.digest: row.body for row in result})
        return found

    async def list_trajectories(
        self, *, agent_name: str | None = None, limit: int = 50
    ) -> list[TrajectorySummary]:
        query = select(trajectories).order_by(
            trajectories.c.started_at.desc(), trajectories.c.trajectory_id.desc()
        )
        if agent_name is not None:
            query = query.where(trajectories.c.agent_name == agent_name)
        async with self._engine.connect() as conn:
            rows = await conn.execute(query.limit(limit))
            return [TrajectorySummary.model_validate(dict(r._mapping)) for r in rows]

    # --- span buffer ------------------------------------------------------------------------

    async def known_traces(self, trace_ids: Sequence[str]) -> set[str]:
        if not trace_ids:
            return set()
        async with self._engine.connect() as conn:
            result = await conn.execute(
                select(spans.c.trace_id).distinct().where(spans.c.trace_id.in_(list(trace_ids)))
            )
            return {row[0] for row in result}

    async def add_spans(self, records: Sequence[SpanRecord]) -> AddSpansResult:
        traces: set[str] = set()
        inserted = 0
        # Deduplicate within the batch too: ON CONFLICT cannot handle a key twice in one VALUES.
        unique = list({(s.trace_id, s.span_id): s for s in records}.values())
        async with self._engine.begin() as conn:
            closed = await self._pruned(conn, {s.trace_id for s in unique})
            live = [s for s in unique if s.trace_id not in closed]
            expired = len(unique) - len(live)
            unique = live
            for chunk in _chunks(unique):
                result = await conn.execute(
                    pg_insert(spans)
                    .values(
                        [
                            {
                                "trace_id": s.trace_id,
                                "span_id": s.span_id,
                                "parent_span_id": s.parent_span_id,
                                "operation": s.operation,
                                "start_ns": s.start_ns,
                                "record": s.model_dump_json(),
                            }
                            for s in chunk
                        ]
                    )
                    .on_conflict_do_nothing(index_elements=["trace_id", "span_id"])
                    .returning(spans.c.trace_id)
                )
                new = [row[0] for row in result]
                inserted += len(new)
                traces.update(new)
        return AddSpansResult(
            inserted, len(records) - inserted - expired, frozenset(traces), expired=expired
        )

    @staticmethod
    async def _pruned(conn: AsyncConnection, trace_ids: set[str]) -> set[str]:
        if not trace_ids:
            return set()
        result = await conn.execute(
            select(pruned_traces.c.trace_id).where(pruned_traces.c.trace_id.in_(sorted(trace_ids)))
        )
        return {row[0] for row in result}

    async def prune_spans(self, received_before: datetime) -> PruneResult:
        async with self._engine.begin() as conn:
            old = [
                row[0]
                for row in await conn.execute(
                    select(spans.c.trace_id)
                    .group_by(spans.c.trace_id)
                    .having(func.max(spans.c.received_at) < received_before)
                    .order_by(spans.c.trace_id)
                )
            ]
            deleted = 0
            for chunk in _chunks(old):
                ids = list(chunk)
                await conn.execute(
                    pg_insert(pruned_traces)
                    .values([{"trace_id": t} for t in ids])
                    .on_conflict_do_nothing(index_elements=["trace_id"])
                )
                result = await conn.execute(spans.delete().where(spans.c.trace_id.in_(ids)))
                deleted += result.rowcount or 0
        _log.info("pruned span buffer", extra={"ctx": {"traces": len(old), "spans": deleted}})
        return PruneResult(traces=len(old), spans=deleted)

    async def sweep_blobs(self, created_before: datetime) -> SweepResult:
        async with self._engine.begin() as conn:
            await conn.execute(
                text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"),
                {"key": _SWEEP_LOCK_KEY},
            )
            step_level = {
                row[0]
                for row in await conn.execute(
                    text(
                        "SELECT input_digest FROM trajectory_steps"
                        " UNION SELECT output_digest FROM trajectory_steps"
                    )
                )
            }
            # Only LLM-request blobs reference other blobs; scan those in bounded batches.
            live = set(step_level)
            for chunk in _chunks(sorted(step_level), 2000):
                bodies = await self._fetch_blobs(conn, set(chunk))
                live.update(codec.nested_digests(bodies))
            candidates = (
                await conn.execute(
                    select(blobs.c.digest, blobs.c.size_bytes).where(
                        blobs.c.created_at < created_before
                    )
                )
            ).all()
            dead = [(row.digest, row.size_bytes) for row in candidates if row.digest not in live]
            for chunk in _chunks(dead, 5000):
                await conn.execute(blobs.delete().where(blobs.c.digest.in_([d for d, _ in chunk])))
        result = SweepResult(
            scanned=len(candidates),
            deleted=len(dead),
            deleted_bytes=sum(size for _, size in dead),
            live=len(live),
        )
        _log.info("swept blobs", extra={"ctx": dataclasses.asdict(result)})
        return result

    # --- suites -----------------------------------------------------------------------------

    async def save_suite(self, manifest: SuiteManifest) -> SaveResult:
        ids = sorted({c.trajectory_id for c in manifest.cases})
        async with self._engine.begin() as conn:
            rows = await conn.execute(
                select(trajectories.c.trajectory_id, trajectories.c.fingerprint).where(
                    trajectories.c.trajectory_id.in_(ids)
                )
            )
            stored = check_stored(manifest, {r.trajectory_id: r.fingerprint for r in rows})
            body = stored.model_dump_json()
            current = (
                await conn.execute(
                    select(suites.c.manifest, suites.c.revision)
                    .where(suites.c.name == stored.name)
                    .with_for_update()
                )
            ).first()
            values = {
                "description": stored.description,
                "agent_name": stored.agent_name,
                "case_count": len(stored.cases),
                "manifest": body,
                "created_at": stored.created_at,
            }
            if current is None:
                inserted = await conn.execute(
                    pg_insert(suites)
                    .values(name=stored.name, revision=1, **values)
                    .on_conflict_do_nothing(index_elements=["name"])
                    .returning(suites.c.revision)
                )
                if inserted.first() is not None:
                    return SaveResult(stored.name, SaveOutcome.CREATED, 1)
                raise SuiteError(f"suite {stored.name} was created concurrently; retry")
            if current.manifest == body:
                return SaveResult(stored.name, SaveOutcome.UNCHANGED, current.revision)
            revision = current.revision + 1
            await conn.execute(
                update(suites)
                .where(suites.c.name == stored.name)
                .values(**values, revision=revision, updated_at=func.now())
            )
            return SaveResult(stored.name, SaveOutcome.UPDATED, revision)

    async def get_suite(self, name: str) -> SuiteManifest | None:
        async with self._engine.connect() as conn:
            row = (
                await conn.execute(select(suites.c.manifest).where(suites.c.name == name))
            ).first()
        return None if row is None else SuiteManifest.model_validate_json(row[0])

    async def list_suites(self) -> list[SuiteSummary]:
        async with self._engine.connect() as conn:
            rows = await conn.execute(
                select(
                    suites.c.name,
                    suites.c.description,
                    suites.c.agent_name,
                    suites.c.case_count,
                    suites.c.revision,
                    suites.c.created_at,
                ).order_by(suites.c.name)
            )
            return [SuiteSummary.model_validate(dict(r._mapping)) for r in rows]

    async def rebuild_trace(self, trace_id: str, build: TraceBuilder) -> list[SaveResult]:
        async with self._engine.begin() as conn:
            await conn.execute(
                text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"),
                {"key": f"agent_replay:trace:{trace_id}"},
            )
            if await self._pruned(conn, {trace_id}):
                return []
            rows = (
                await conn.execute(
                    select(spans.c.operation, spans.c.record).where(spans.c.trace_id == trace_id)
                )
            ).all()
            if not any(r.operation == "invoke_agent" for r in rows):
                return []
            records = [SpanRecord.model_validate_json(r.record) for r in rows]
            return [await self._save(conn, codec.encode(t)) for t in build(records)]

    async def stats(self) -> StoreStats:
        async with self._engine.connect() as conn:
            row = (
                await conn.execute(
                    text(
                        "SELECT"
                        " (SELECT count(*) FROM trajectories) AS trajectories,"
                        " (SELECT count(*) FROM trajectory_steps) AS steps,"
                        " (SELECT count(*) FROM blobs) AS blobs,"
                        " (SELECT coalesce(sum(size_bytes), 0) FROM blobs) AS blob_bytes,"
                        " (SELECT coalesce(sum(logical_bytes), 0) FROM trajectories)"
                        "   AS logical_bytes,"
                        " (SELECT count(*) FROM otlp_spans) AS buffered_spans"
                    )
                )
            ).one()
            return StoreStats.model_validate(dict(row._mapping))

    async def truncate(self) -> None:
        """Delete all data. Intended for tests."""
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    "TRUNCATE trajectory_steps, trajectories, blobs, otlp_spans, pruned_traces,"
                    " suites CASCADE"
                )
            )


def _header_values(header: codec.Header) -> dict[str, Any]:
    return {
        "trajectory_id": header.trajectory_id,
        "schema_version": header.schema_version,
        "source": header.source,
        "trace_id": header.trace_id,
        "agent_name": header.agent_name,
        "task_input": header.task_input,
        "final_output": header.final_output,
        "status": header.status,
        "error": header.error,
        "started_at": header.started_at,
        "duration_ms": header.duration_ms,
        "step_count": header.step_count,
        "input_tokens": header.input_tokens,
        "output_tokens": header.output_tokens,
        "cost_usd": header.cost_usd,
        "fingerprint": header.fingerprint,
        "content_digest": header.content_digest,
        "logical_bytes": header.logical_bytes,
        "metadata": header.metadata,
    }


def _header_from_row(row: Any) -> codec.Header:
    meta = row["metadata"]
    return codec.Header(
        trajectory_id=row["trajectory_id"],
        schema_version=row["schema_version"],
        agent_name=row["agent_name"],
        task_input=row["task_input"],
        final_output=row["final_output"],
        status=row["status"],
        error=row["error"],
        started_at=row["started_at"],
        duration_ms=row["duration_ms"],
        metadata=json.loads(meta) if isinstance(meta, str) else meta,
        source=row["source"],
        trace_id=row["trace_id"],
        step_count=row["step_count"],
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        cost_usd=row["cost_usd"],
        fingerprint=row["fingerprint"],
        content_digest=row["content_digest"],
        logical_bytes=row["logical_bytes"],
    )
