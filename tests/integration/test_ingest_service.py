from __future__ import annotations

import pytest

from agent_replay.ingest.service import ingest_batch
from agent_replay.ingest.spans import DecodedBatch, SpanRecord
from agent_replay.store import TrajectoryStore
from tests.helpers import record_with_spans, to_records

pytestmark = pytest.mark.integration


def _http_span(trace_id: str, span_id: str, parent: str | None = None) -> SpanRecord:
    return SpanRecord(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent,
        name="GET /",
        start_ns=0,
        end_ns=1,
    )


async def test_children_first_then_root_creates_then_late_span_updates(
    store: TrajectoryStore,
) -> None:
    recorded, spans = await record_with_spans()
    records = to_records(spans)
    root = records[-1]
    last_chat = records[-2]
    early = [r for r in records if r not in (root, last_chat)]

    first = await ingest_batch(store, DecodedBatch(spans=early))
    assert (first.spans_inserted, first.pending_traces, first.created) == (4, 1, [])

    second = await ingest_batch(store, DecodedBatch(spans=[root]))
    assert len(second.created) == 1
    partial = await store.get(second.created[0])
    assert partial is not None
    assert len(partial.steps) == 4

    third = await ingest_batch(store, DecodedBatch(spans=[last_chat]))
    assert third.updated == second.created
    complete = await store.get(second.created[0])
    assert complete is not None
    assert complete.fingerprint() == recorded.fingerprint()

    duplicate = await ingest_batch(store, DecodedBatch(spans=records))
    assert (duplicate.spans_inserted, duplicate.spans_duplicate) == (0, len(records))
    assert duplicate.created == duplicate.updated == duplicate.unchanged == []


async def test_traces_without_genai_spans_are_ignored(store: TrajectoryStore) -> None:
    noise = [_http_span("c" * 32, "1" * 16), _http_span("c" * 32, "2" * 16, "1" * 16)]
    report = await ingest_batch(store, DecodedBatch(spans=noise, rejected=1))
    assert (report.spans_ignored, report.spans_inserted, report.spans_rejected) == (2, 0, 1)
    assert report.spans_received == 3
    assert (await store.stats()).buffered_spans == 0


async def test_non_genai_spans_of_a_known_trace_are_kept(store: TrajectoryStore) -> None:
    _, spans = await record_with_spans()
    records = to_records(spans)
    await ingest_batch(store, DecodedBatch(spans=records))
    extra = _http_span(records[0].trace_id, "9" * 16)
    report = await ingest_batch(store, DecodedBatch(spans=[extra]))
    assert (report.spans_inserted, report.spans_ignored) == (1, 0)
    assert report.unchanged  # re-assembled, nothing changed
