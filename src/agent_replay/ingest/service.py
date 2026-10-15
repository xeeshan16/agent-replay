"""Ingest pipeline: buffer decoded spans, then (re)assemble the traces they touched.

OTLP exporters deliver at least once and in arbitrary batch boundaries: a trace's child spans
usually arrive before its root (the root ends last), batches are retried on timeouts, and one
batch can hold spans from many traces. The pipeline is shaped around that:

1. Spans are buffered keyed by (trace_id, span_id); re-delivered spans are counted and dropped.
2. Traces without any GenAI span are ignored, so pointing a general-purpose collector at the
   receiver does not fill the buffer with HTTP and database spans. Spans of a trace that is
   already known are kept even if they are not GenAI spans (they may be intermediate parents).
3. Every trace that received a new span and has an ``invoke_agent`` span is re-assembled from
   *all* its buffered spans and saved. Saving identical content is a no-op, and a late child
   span produces a new revision of the same trajectory.
"""

from __future__ import annotations

from collections import defaultdict

from pydantic import BaseModel, ConfigDict

from agent_replay import telemetry as tm
from agent_replay.ingest.assemble import assemble_trace
from agent_replay.ingest.spans import DecodedBatch, SpanRecord
from agent_replay.logs import get_logger
from agent_replay.store.base import SaveOutcome, TrajectoryStore

_log = get_logger("ingest")


class IngestReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    spans_received: int
    spans_inserted: int
    spans_duplicate: int
    spans_ignored: int
    spans_rejected: int
    created: list[str]
    updated: list[str]
    unchanged: list[str]
    pending_traces: int  # traces with new spans whose agent span has not arrived yet


async def ingest_batch(store: TrajectoryStore, batch: DecodedBatch) -> IngestReport:
    with tm.get_tracer().start_as_current_span("agent_replay.ingest") as span:
        by_trace: dict[str, list[SpanRecord]] = defaultdict(list)
        for record in batch.spans:
            by_trace[record.trace_id].append(record)
        genai_traces = {t for t, group in by_trace.items() if any(s.is_genai for s in group)}
        others = [t for t in by_trace if t not in genai_traces]
        keep = genai_traces | await store.known_traces(others)
        kept = [s for t in keep for s in by_trace[t]]
        ignored = len(batch.spans) - len(kept)

        added = await store.add_spans(kept)
        created: list[str] = []
        updated: list[str] = []
        unchanged: list[str] = []
        pending = 0
        buckets = {
            SaveOutcome.CREATED: created,
            SaveOutcome.UPDATED: updated,
            SaveOutcome.UNCHANGED: unchanged,
        }
        for trace_id in sorted(added.traces):
            results = await store.rebuild_trace(trace_id, assemble_trace)
            if not results:
                pending += 1
            for result in results:
                buckets[result.outcome].append(result.trajectory_id)

        report = IngestReport(
            spans_received=len(batch.spans) + batch.rejected,
            spans_inserted=added.inserted,
            spans_duplicate=added.duplicates,
            spans_ignored=ignored,
            spans_rejected=batch.rejected,
            created=created,
            updated=updated,
            unchanged=unchanged,
            pending_traces=pending,
        )
        span.set_attribute("agent_replay.ingest.spans_received", report.spans_received)
        span.set_attribute("agent_replay.ingest.spans_inserted", report.spans_inserted)
        span.set_attribute("agent_replay.ingest.spans_rejected", report.spans_rejected)
        span.set_attribute("agent_replay.ingest.trajectories_saved", len(created) + len(updated))
        _log.info(
            "ingested batch",
            extra={
                "ctx": {
                    "spans_received": report.spans_received,
                    "spans_inserted": report.spans_inserted,
                    "spans_duplicate": report.spans_duplicate,
                    "spans_ignored": report.spans_ignored,
                    "spans_rejected": report.spans_rejected,
                    "created": len(created),
                    "updated": len(updated),
                    "pending_traces": pending,
                }
            },
        )
        return report
