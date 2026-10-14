# ADR 0001: Content-addressed cassette with strict matching

- Status: accepted
- Date: 2026-10-14

## Context

Replay must serve a new agent version the same tool results the recorded run saw. Options for
locating the recorded answer to a request:

1. **Positional**: the n-th tool call gets the n-th recorded result.
2. **By call ID**: match on the provider-issued tool-call ID.
3. **By content**: match on tool name plus canonical arguments (and, for model calls, the
   canonical request).

## Decision

Match by content. Keys are SHA-256 digests of canonical JSON (sorted keys, compact separators,
NaN rejected). Repeated identical keys are served first-in-first-out, so polling loops replay
faithfully. Matching is strict: an unrecorded request raises `CassetteMissError`, which ends the
run with an error status at the exact step where behaviour diverged.

## Consequences

- Positional matching would silently hand the wrong result to a reordered agent, producing
  confusing downstream failures far from the cause. Call IDs are minted by the model and differ
  between versions, so they cannot be keys. Content keys survive both.
- Any argument change, however small (a timestamp, a reworded query), is a miss. That is the
  right default for a regression gate, but too brittle for real traces; Day 3 adds a normaliser
  chain applied before hashing and an opt-in fuzzy mode.
- Served results are rewritten to carry the live call ID so the agent's message history stays
  consistent.
- A miss is a signal, not a crash: the partial trajectory is kept and can be diffed and scored.
