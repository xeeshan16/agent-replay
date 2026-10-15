# ADR 0002: OTLP ingest through a span buffer with whole-trace re-assembly

- Status: accepted
- Date: 2026-10-15

## Context

Trajectories from other frameworks arrive as OpenTelemetry spans over OTLP/HTTP. Exporters batch
by time and size, not by trace, so:

- a trace's spans are spread over several requests, and the `invoke_agent` span (which ends last)
  usually arrives after its children;
- a request that times out is retried, so the same spans can arrive twice (at-least-once);
- several receiver workers can process batches of the same trace concurrently.

Options considered:

1. **Stream processing**: update the trajectory incrementally as each span arrives.
2. **Wait for completeness**: hold spans until the trace looks finished (root seen plus a quiet
   period), then assemble once.
3. **Buffer and re-assemble**: keep every span, keyed by `(trace_id, span_id)`, and re-run a pure
   assembly function over the whole trace whenever it receives a new span.

## Decision

Option 3. Spans go into `otlp_spans` with `ON CONFLICT DO NOTHING`, which makes re-delivery a
no-op. Every trace that received a new span and has an agent span is re-assembled from all its
spans under a transaction-scoped advisory lock on the trace ID, and saved by content digest:
identical content is `UNCHANGED`, different content is a new `revision` of the same trajectory.
Trajectory IDs are `<trace_id>-<invoke_agent span_id>`, so the same trace always maps to the
same record.

## Consequences

- Correctness does not depend on arrival order or batch boundaries: assembly is a pure function
  of the span set, tested by shuffling spans and by delivering one span per batch concurrently
  against PostgreSQL.
- No timers or "trace complete" heuristics. A trajectory can be visible before its last child
  span arrives; the late span bumps the revision. Consumers that need finality (the CI gate) read
  the latest revision after the export has been flushed.
- Re-assembly re-reads the whole trace, so cost grows with spans per trace times batches per
  trace. Agent traces are small (tens to hundreds of spans), which keeps this cheap; very long
  sessions would need incremental assembly.
- The buffer grows with every ingested span. Traces with no GenAI span are ignored at the door,
  but a retention job for old buffered spans is still needed (tracked in the plan).
- The advisory lock serialises work per trace only; different traces ingest in parallel.
