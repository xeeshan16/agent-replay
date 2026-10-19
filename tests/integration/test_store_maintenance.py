"""Suites, span-buffer retention and blob sweep; runs against memory and PostgreSQL."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from agent_replay.ingest.service import ingest_batch
from agent_replay.ingest.spans import DecodedBatch
from agent_replay.reference import support_desk as desk
from agent_replay.store import PruneResult, SaveOutcome, TrajectoryStore
from agent_replay.store.postgres import PostgresStore
from agent_replay.suite import Suite, SuiteError, load_stored_suite, record_suite
from tests.helpers import record_with_spans, to_records

pytestmark = pytest.mark.integration

PAST = datetime(2000, 1, 1, tzinfo=UTC)


def _future() -> datetime:
    return datetime.now(UTC) + timedelta(minutes=1)


async def _suite(name: str = "desk", tasks: tuple[str, ...] = desk.SAMPLE_TASKS[:3]) -> Suite:
    return await record_suite(
        name, tasks, lambda case_id: desk.build_agent(id_factory=desk.seeded_ids(case_id))
    )


# --- suites -----------------------------------------------------------------------------------


async def test_suite_round_trip_and_revisions(store: TrajectoryStore) -> None:
    suite = await _suite()
    with pytest.raises(SuiteError, match="is not stored"):
        await store.save_suite(suite.manifest)
    for _, trajectory in suite.items():
        await store.save(trajectory)

    created = await store.save_suite(suite.manifest)
    again = await store.save_suite(suite.manifest)
    assert (created.outcome, again.outcome) == (SaveOutcome.CREATED, SaveOutcome.UNCHANGED)
    loaded = await load_stored_suite(store, "desk")
    assert loaded.baselines == suite.baselines

    edited = suite.manifest.model_copy(update={"description": "three cases"})
    updated = await store.save_suite(edited)
    assert (updated.outcome, updated.revision) == (SaveOutcome.UPDATED, 2)
    [summary] = await store.list_suites()
    assert (summary.name, summary.case_count, summary.revision) == ("desk", 3, 2)
    assert summary.description == "three cases"
    assert await store.get_suite("nope") is None
    with pytest.raises(SuiteError, match="no stored suite"):
        await load_stored_suite(store, "nope")


async def test_suite_with_a_mismatched_fingerprint_is_refused(store: TrajectoryStore) -> None:
    suite = await _suite()
    first_case, first = suite.items()[0]
    await store.save(first.model_copy(update={"final_output": "something else"}))
    for _, trajectory in suite.items()[1:]:
        await store.save(trajectory)
    with pytest.raises(SuiteError, match="does not match"):
        await store.save_suite(suite.manifest)
    assert first_case.id.startswith("00-")


# --- span-buffer retention ---------------------------------------------------------------------


async def test_retention_closes_idle_traces_and_drops_stragglers(store: TrajectoryStore) -> None:
    recorded, spans = await record_with_spans()
    records = to_records(spans)
    straggler, rest = records[0], records[1:]
    report = await ingest_batch(store, DecodedBatch(spans=list(records)))
    [trajectory_id] = report.created

    assert await store.prune_spans(PAST) == PruneResult(traces=0, spans=0)
    pruned = await store.prune_spans(_future())
    assert (pruned.traces, pruned.spans) == (1, len(records))
    assert (await store.stats()).buffered_spans == 0

    # A late re-delivery of one span must not rebuild the trace from that span alone.
    late = await ingest_batch(store, DecodedBatch(spans=[straggler]))
    assert (late.spans_inserted, late.spans_expired) == (0, 1)
    assert late.created == late.updated == []
    stored = await store.get(trajectory_id)
    assert stored is not None
    assert stored.fingerprint() == recorded.fingerprint()
    assert len(rest) > 0


async def test_retention_keeps_active_traces(store: TrajectoryStore) -> None:
    _, spans = await record_with_spans()
    await ingest_batch(store, DecodedBatch(spans=to_records(spans)))
    assert (await store.prune_spans(PAST)).traces == 0
    assert (await store.stats()).buffered_spans == len(spans)


# --- blob sweep ---------------------------------------------------------------------------------


async def test_sweep_deletes_only_unreferenced_blobs(store: TrajectoryStore) -> None:
    suite = await _suite(tasks=desk.SAMPLE_TASKS[:2])
    keep, drop = (t for _, t in suite.items())
    await store.save(keep)
    await store.save(drop)
    before = await store.stats()
    # Replace the second run with a truncated version: the blobs only it used become orphans.
    truncated = drop.model_copy(update={"steps": drop.steps[:1], "final_output": None})
    await store.save(truncated)

    assert (await store.sweep_blobs(PAST)).deleted == 0  # inside the grace period
    swept = await store.sweep_blobs(_future())
    after = await store.stats()
    assert swept.deleted > 0
    assert swept.deleted_bytes > 0
    assert after.blobs == before.blobs - swept.deleted
    assert swept.live == after.blobs
    # Everything still stored decodes (the codec verifies every blob it reads).
    assert await store.get(keep.trajectory_id) == keep
    assert await store.get(drop.trajectory_id) == truncated
    assert (await store.sweep_blobs(_future())).deleted == 0


@pytest.mark.postgres
async def test_sweep_and_concurrent_saves_never_lose_a_blob(pg_store: PostgresStore) -> None:
    suite = await _suite(tasks=desk.SAMPLE_TASKS)
    runs = [t for _, t in suite.items()]
    # Orphan everything, then race a sweep with re-saving runs that reuse the same blobs.
    await asyncio.gather(*(pg_store.save(t) for t in runs))
    for t in runs:
        await pg_store.save(t.model_copy(update={"steps": (), "final_output": None}))
    copies = [t.model_copy(update={"trajectory_id": f"copy/{t.trajectory_id}"}) for t in runs]
    await asyncio.gather(
        pg_store.sweep_blobs(_future()),
        *(pg_store.save(t) for t in copies),
        pg_store.sweep_blobs(_future()),
    )
    for t in copies:
        assert await pg_store.get(t.trajectory_id) == t
