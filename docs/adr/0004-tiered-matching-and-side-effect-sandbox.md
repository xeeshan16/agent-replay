# ADR 0004: Tiered matching and a fail-closed side-effect sandbox

- Status: accepted
- Date: 2026-10-16
- Amends: [ADR 0001](0001-content-addressed-cassette.md)

## Context

Strict content matching (ADR 0001) is too brittle for real agents. A replay of an unchanged
agent misses when the system prompt carries the current time, when a write carries a freshly
minted idempotency key, when the provider mints new tool-call IDs, or when a list built from a
set comes out in another order. Conversely, a replay of a *changed* agent should be able to
continue past harmless differences (a new read-only lookup, a slightly reworded search query)
so the rest of the run can be diffed and scored.

Loosening matching has an obvious failure mode: serving a recorded result for a request that is
*not* equivalent. For reads that produces a plausible but stale answer. For writes it is worse:
serving the recorded refund for a request with a different amount reports success for a call
that never happened, which is exactly the regression replay exists to catch.

## Decision

1. **Three match tiers**, cheapest first, each reported per request as a `MatchEvent`:
   `exact` (raw content key), `normalized` (equal after the normaliser chain), and opt-in
   `fuzzy` (the most similar unconsumed recording of the same tool or model with structural
   similarity at or above a threshold, default 0.9; ties go to the earliest recording).
2. **Normalisation is a configurable chain** applied in one walk over the whole request, with
   indexed placeholders by first appearance (`<uuid:0>`). The default masks ISO-8601
   timestamps, UUIDs and provider call IDs; agents add rules for their own volatile fields
   (the reference agent declares `tags` unordered). Bare dates are not masked by default,
   because a cut-off date in an argument is usually meaningful.
3. **Writes never match fuzzily** unless `fuzzy_writes` is set explicitly. A write matches at
   the `normalized` tier or not at all.
4. **A sandbox owns every tool call during replay.** Recorded calls are served from the
   cassette. An unrecorded read-only call is a miss, or runs live with `live_reads`. An
   unrecorded side-effecting call is never executed: it either fails the run
   (`SideEffectBlockedError`, the default) or returns an error result to the agent (`block`) so
   the run continues and scoring sees what the agent did next.
5. **Classification fails closed.** A tool is a write if the agent's spec says so, if the
   recording flagged it, or if it has no spec. Relabelling a tool as read-only in a new agent
   version cannot unlock live execution of something that was recorded as a write.
6. **"Identical" means equal after normalisation.** `ReplayResult.identical` compares
   normalised behavioural fingerprints; `byte_identical` keeps the raw comparison.

## Consequences

- Unchanged agents with volatile inputs replay as identical, and the event log shows which
  steps needed normalisation, so masking that is too aggressive is visible rather than silent.
- A fuzzy-served step is still a behaviour change: the replayed step keeps the new arguments,
  so the normalised fingerprint differs and the run is not reported identical. Fuzzy mode buys
  a complete trajectory for the diff, not a pass.
- Similarity is structural and model-free (edit ratio for strings, relative difference for
  numbers, key-union mean for objects). It is cheap and deterministic but has no notion of
  meaning: "ORD-1001" and "ORD-1007" score 0.875. Thresholds should stay high, and the score is
  reported with every fuzzy match.
- Live reads trade determinism for coverage: a live read sees today's world, not the recorded
  one. Runs that used one report a `live` event so a gate can refuse to compare them.
- The sandbox enforces the read-only claim made at tool registration; it cannot verify it. A
  tool that writes but is registered without `side_effect=True` and never recorded as a write
  can still run live under `live_reads`. This is documented in the security notes.
