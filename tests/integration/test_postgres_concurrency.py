"""Races that only a real database can exercise."""

from __future__ import annotations

import asyncio

import pytest

from agent_replay.ingest.assemble import assemble_trace
from agent_replay.ingest.service import ingest_batch
from agent_replay.ingest.spans import DecodedBatch
from agent_replay.store import SaveOutcome
from agent_replay.store.postgres import PostgresStore
from tests.helpers import record_with_spans, to_records

pytestmark = [pytest.mark.integration, pytest.mark.postgres]


async def test_concurrent_saves_of_one_trajectory_create_it_once(pg_store: PostgresStore) -> None:
    recorded, _ = await record_with_spans()
    results = await asyncio.gather(*(pg_store.save(recorded) for _ in range(8)))
    outcomes = sorted(r.outcome.value for r in results)
    assert outcomes == ["created"] + ["unchanged"] * 7
    assert (await pg_store.stats()).trajectories == 1


async def test_concurrent_batches_of_one_trace_converge(pg_store: PostgresStore) -> None:
    recorded, spans = await record_with_spans()
    records = to_records(spans)
    # Every span delivered in its own batch, plus a full re-delivery, all at once.
    batches = [DecodedBatch(spans=[r]) for r in records] + [DecodedBatch(spans=records)]
    reports = await asyncio.gather(*(ingest_batch(pg_store, b) for b in batches))

    assert sum(r.spans_inserted for r in reports) == len(records)
    created = [t for r in reports for t in r.created]
    assert len(created) == 1
    stored = await pg_store.get(created[0])
    assert stored is not None
    assert stored.fingerprint() == recorded.fingerprint()
    [final] = await pg_store.rebuild_trace(records[0].trace_id, assemble_trace)
    assert final.outcome is SaveOutcome.UNCHANGED
