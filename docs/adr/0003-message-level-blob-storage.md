# ADR 0003: Message-level content-addressed blobs in PostgreSQL

- Status: accepted
- Date: 2026-10-15

## Context

A recorded LLM step contains the full request: system prompt, tool definitions and the whole
conversation so far. Storing each request as one document repeats the history at every turn
(quadratic in conversation length) and repeats the system prompt and tool definitions in every
run. Tool outputs also recur across runs (the same fixture order, the same error).

## Decision

Store content in a `blobs` table keyed by the SHA-256 of canonical JSON text, insert-if-absent.
An LLM request blob holds only the model, temperature and the digests of its message and tool
definition blobs; each message and each tool definition is its own blob. Tool calls are stored as
`{name, arguments}` without the per-run call ID (which lives on the step row) so equal calls share
a blob. Step rows and a denormalised trajectory header (status, tokens, cost, fingerprint) are
ordinary relational rows.

Blob bodies are `TEXT` holding canonical JSON rather than `JSONB`: the stored bytes are exactly
the hashed bytes (any blob can be verified against its key), and JSON containing `\u0000`, which
`JSONB` rejects, still stores. SQL queries over content can cast with `body::jsonb`.

## Consequences

- An N-turn conversation stores N message blobs instead of O(N²) message copies, and shared
  prompts and tool definitions are stored once across all runs. On the five reference runs the
  store measured 19,805 logical bytes in 11,179 blob bytes (1.77x); the ratio grows with run
  count and conversation length, since this sample is tiny.
- Reading a trajectory takes three queries (header, steps, then blobs in two rounds: step blobs,
  then the messages they reference). Listing and gating use the header only.
- Decoding re-checks the trajectory fingerprint against the header, so a missing or altered blob
  fails loudly with `CorruptRecordError` instead of returning a subtly different run.
- Blobs are never deleted on update; orphaned blobs need a mark-and-sweep job (not built yet).
