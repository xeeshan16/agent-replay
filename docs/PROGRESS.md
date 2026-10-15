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
