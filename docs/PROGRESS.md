# Progress log

## Day 1 — 2026-10-14: foundations and first vertical slice

**Shipped**

- Project layout with `pyproject.toml` (hatchling), ruff (incl.
  bandit, async and pylint rule sets), mypy `--strict` with the pydantic plugin, pytest +
  pytest-asyncio + coverage.
- `model.py`: frozen pydantic v2 trajectory model (`LLMStep` / `ToolStep` discriminated union,
  `Usage` arithmetic, content keys, a behavioural fingerprint that ignores IDs and timing).
- `providers/`: async `LLMProvider` protocol; deterministic `FakeLLM` driven by a policy function
  (stable call IDs derived from request content, estimated tokens, priced usage);
  `ResilientLLM` with per-attempt timeouts and full-jitter exponential backoff.
- `recorder.py`: wraps any agent's model and tools, emits `invoke_agent` / `chat` /
  `execute_tool` spans with `gen_ai.*` attributes, records failed runs too.
- `replay.py`: content-addressed cassette with FIFO for repeated calls; `FULL` and `TOOLS`
  replay; strict matching raising `CassetteMissError` at the divergence point; replay never runs
  real tools.
- Reference support-desk agent (lookup + side-effecting refund) with a seeded "skip lookup"
  regression; CLI (`record`, `replay`, `demo`); structured JSON logging with trace correlation.
- CI workflow: lint, format check, mypy, tests with coverage, replay smoke test.

**Verified locally**: ruff clean, mypy strict clean (30 files), 35 tests passing, 99% line
coverage; `agent-replay demo` reports 5/5 sample runs identical under both replay modes (fake
provider).

**Next (Day 2)**: OTLP JSON ingest and the PostgreSQL store with idempotent ingest.

## Day 1 (follow-up) — 2026-10-14: standalone repository

- Moved from `ai-engineering-lab/projects/01-agent-replay` into its own repository with full
  history; CI now runs from the repository root (`.github/workflows/ci.yml`).

## Day 2 — 2026-10-15: ingestion and storage

**Shipped**

- `genai.py` + recorder: opt-in GenAI content attributes (`gen_ai.input.messages`,
  `gen_ai.output.messages`, `gen_ai.system_instructions`, `gen_ai.tool.definitions`,
  `gen_ai.tool.call.arguments` / `.result`). Off by default; `AGENT_REPLAY_CAPTURE_CONTENT` or
  `capture_content=True` turns it on. The recorder now takes an injectable `TracerProvider`.
- `ingest/spans.py`: OTLP decoders for HTTP/JSON (hex or base64 IDs, every `AnyValue` kind) and
  HTTP/protobuf into one `SpanRecord`; per-span rejection reported as OTLP partial success.
- `ingest/assemble.py`: pure trace → trajectories assembly (one per `invoke_agent` span, nested
  agents kept separate, intermediate non-GenAI spans walked through, broken parent chains
  attached with a warning, truncated content downgraded to a warning).
- `store/`: `TrajectoryStore` protocol, in-memory and PostgreSQL implementations sharing one
  codec. Message-level content-addressed blobs, step rows, denormalised headers, revisions,
  fingerprint check on read; advisory-locked per-trace re-assembly and row-locked saves.
- `ingest/service.py`: buffer → dedupe → re-assemble pipeline (ADR 0002).
- `server.py`: FastAPI OTLP/HTTP receiver (`/v1/traces`, protobuf + JSON, gzip, 8 MiB cap incl.
  gzip-bomb guard, bearer token, 503 + `Retry-After` on store outage, `google.rpc.Status` error
  bodies) and `GET /v1/trajectories[/{id}]`.
- CLI: `record --otlp-out`, `ingest` (JSON, JSON lines, `.pb`), `stats`, `serve`.
- CI: PostgreSQL 16 service; PostgreSQL tests are required in CI (they skip locally without a
  database URL); OTLP-file-to-PostgreSQL smoke test. `docker-compose.yml` for local Postgres.
- ADR 0002 (span buffer and re-assembly) and ADR 0003 (message-level blob storage).

**Verified locally** (Python 3.12, PostgreSQL 16, fake provider): ruff and format clean, mypy
strict clean (53 files), 112 tests passing including the PostgreSQL contract and concurrency
tests, 97% line coverage. The headline test exports the five sample runs through the stock
`OTLPSpanExporter` (batches of 4 spans, so traces split across requests) to a live receiver;
every stored trajectory has the recorded fingerprint and replays identically in `FULL` mode.
Measured on the five sample runs ingested into PostgreSQL: 19,805 logical bytes stored as 11,179
blob bytes (1.77x); a second ingest of the same files inserted 0 spans and changed nothing.

**Next (Day 3)**: normaliser chain, fuzzy matching and the side-effect sandbox for replay.

## Day 3 — 2026-10-16: replay engine, hardened

**Shipped**

- `normalize.py`: an immutable rule chain applied in one walk: masks for ISO-8601 timestamps
  (bare dates kept by default), UUIDs and provider tool-call IDs, `DropKeys`, and `SortLists`
  for lists declared unordered. Indexed placeholders by first appearance keep identity
  (`<uuid:0>` passed back stays `<uuid:0>`) while ignoring the value; numbering follows sorted
  keys and, inside unordered lists, the sorted order, so it never depends on arrival order.
- `matching.py`: `MatchPolicy` (strict or fuzzy, threshold, normaliser, `fuzzy_writes`), a
  model-free structural `similarity` over JSON, and `MatchEvent` records.
- `replay.py`: the cassette matches exact → normalised → (opt-in) fuzzy, keeps FIFO for
  repeated calls even when a fuzzy match consumed an entry out of order, logs one event per
  request, and names the closest unconsumed recording and its score on a miss.
  `ReplayResult.identical` compares normalised fingerprints; `byte_identical` keeps the raw one.
- `sandbox.py`: every replayed tool call goes through `SandboxedTools`. Unrecorded reads miss or
  run live (`live_reads`); unrecorded writes are never executed and either fail the run
  (`SideEffectBlockedError`) or return an error result so the run continues. Tools the agent
  does not offer fail as they would live. Write classification fails closed (spec, recording,
  or no spec).
- Reference agent: current time in the system prompt, an idempotent `issue_refund` with a fresh
  key per call, `get_refund_policy`, `create_ticket` (minted ticket ID and timestamp, tags from a
  set), a retried transient lookup failure (ORD-1004) and a refund over the auto-refund limit
  escalated to a ticket (ORD-1005). Variants `no-retry`, `amount-drift` and `check-policy` join
  `skip-lookup`; the agent ships its own normaliser (defaults + unordered `tags`).
- CLI: `replay --match {strict,fuzzy} --threshold --fuzzy-writes --live-reads
  --unrecorded-writes {fail,block}`; the report includes the match policy, sandbox policy, an
  outcome summary and the per-request events.
- ADR 0004 (tiered matching and the fail-closed sandbox).

**Verified locally** (Python 3.12, PostgreSQL 16, fake provider): ruff and format clean, mypy
strict clean (60 files), 177 tests passing including the PostgreSQL tests, 97% line coverage.

Measured on the 7 sample tasks, recording in one process and replaying in another (different
`PYTHONHASHSEED`, at least one second apart), 5 seed pairs, so 35 replays per cell, counting
runs reported identical:

| Normaliser | FULL replay | TOOLS replay |
|------------|-------------|--------------|
| none (Day 1 strict matching) | 0 / 35 | 20 / 35 |
| default (timestamps, UUIDs, call IDs) | 35 / 35 | 31 / 35 |
| support-desk (default + unordered `tags`) | 35 / 35 | 35 / 35 |

FULL replay without normalisation always misses because the system prompt carries the time;
TOOLS replay misses on the fresh idempotency key; the 4 remaining default-normaliser misses are
the ticket tags coming out of the set in another order (the tags matched by chance in one seed
pair).

**Plan adjustment**: none to the milestones. The match events are designed as Day 4's input:
the diff can align on `recorded_index` and report fuzzy, live and blocked steps as divergence
reasons.

**Next (Day 4)**: trajectory diff — alignment of tool-call sequences with argument-aware
substitution cost, first-divergence detection with a reason, CLI and Markdown rendering.

## Day 4 — 2026-10-17: trajectory diff

**Shipped**

- `diff.py`: a trajectory becomes *actions*, one per tool call the model requested, joined to
  the tool step that executed it. Calls that never ran (a cassette miss, a write the sandbox
  refused) are still actions, marked `executed: false`, so the diff shows the refused refund and
  not only what happened before it. Traces ingested without content (no requested calls on model
  steps) fall back to their tool steps.
- Both sides are normalised in one scope each, in action order, with the replay normaliser.
- Alignment: weighted edit distance (delete 1, insert 1, rename 1.5, same-tool substitution
  `0.8·(1 − sim(args)) + 0.2·(1 − sim(result))` using the Day 3 structural similarity), filled
  from the end and traced forward so the later of two equal calls is the extra one. `DiffCosts`
  validates that a rename and a same-tool substitution always beat delete + insert, and caps the
  table size.
- Reasons: `wrong_tool`, `argument_drift` (leaf JSON paths with before/after), `output_drift`,
  `missing_step`, `extra_step`, `extra_loop` (an inserted exact repeat), `status_changed`,
  `different_answer`. Each divergence carries baseline and candidate step indices, whether a
  write is involved, and the *decision step*: the candidate model call that made the choice (for
  a missing call, the first model call after the last aligned candidate action).
- Replay provenance: match events annotate candidate actions (exact, normalised, fuzzy, live,
  blocked, unknown tool) and appear in divergence summaries.
- `report.py`: GitHub-flavoured Markdown (verdict, first divergence, aligned table, outcome,
  usage delta). Cells are flattened, truncated and escaped so untrusted tool output cannot break
  the table or inject markup; covered by a test with a hostile tool output.
- CLI: `agent-replay diff BASELINE CANDIDATE [--format md|json] [--normalizer ...] [--out]`
  (exit 1 on divergence); `replay` reports `first_divergence` and per-reason counts and writes
  the Markdown diff with `--diff-out`. CI runs a diff smoke test and publishes the diff to the
  job summary.
- The support-desk normaliser now also masks refund IDs (`RF-xxxxxxxx`, indexed): they are
  derived from the per-call idempotency key, so two independent live runs always differed on
  them. Replay results are unaffected (replayed tool outputs come from the cassette).

**Verified locally** (Python 3.12, PostgreSQL 16, fake provider): ruff and format clean, mypy
strict clean (65 files), 225 tests passing including the PostgreSQL tests, 97% line coverage
(`diff.py` 99%). A brute-force test checks the alignment is minimal for every pair of action
sequences up to length 3 over a 4-call pool.

Measured on the 7 sample tasks x 5 policies in TOOLS replay with live reads, recording in one
process and diffing in another (3 seed pairs, different `PYTHONHASHSEED`): the first divergence
was the same in all 3 runs for all 35 cells. Every `v1` replay diffs identical; the regressions
are localised as follows (reason: tool @ candidate decision step):

| Task | skip-lookup | no-retry | amount-drift | check-policy |
|------|-------------|----------|--------------|--------------|
| refund ORD-1001 | missing_step: lookup_order @0 | identical | argument_drift: issue_refund @2 (`amount` 59.9 → 60.0) | extra_step: get_refund_policy @2 |
| refund ORD-1003 (outside window) | wrong_tool: issue_refund @0 | identical | identical | extra_step: get_refund_policy @2 |
| refund ORD-9999 (unknown) | wrong_tool: issue_refund @0 | identical | identical | identical |
| refund ORD-1004 (transient failure) | missing_step: lookup_order @0 | missing_step: lookup_order @2 | identical | extra_step: get_refund_policy @4 |
| refund ORD-1005 (over limit) | missing_step: lookup_order @0 | identical | identical | extra_step: get_refund_policy @2 |

The two non-refund tasks diff identical under every policy, as they should: no variant changes
their behaviour. "Identical" cells are policies whose change does not apply to that task (e.g.
`amount-drift` on an order whose total is already whole, or that is never refunded). Diff time
for 200 x 200 synthetic actions: about 40 ms on this machine.

**Observation for Day 5**: `skip-lookup` reads as `wrong_tool` when the baseline made only the
lookup (ORD-1003, ORD-9999) and as `missing_step` + `argument_drift` when the baseline also
refunded. Both are correct alignments of the same regression; the Day 5 labelled dataset will
label first divergences by position and tool, and accept either reason where the sequences make
them equivalent, rather than tune costs to one reading.

**Next (Day 5)**: layered scoring (deterministic checks → invariants DSL → optional LLM judge),
per-run token and cost accounting, and the labelled dataset of seeded regressions.
