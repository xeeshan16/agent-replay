# Progress log

## Day 1 — 2026-10-14: foundations and first vertical slice

**Shipped**

- Project layout under `projects/01-agent-replay` with `pyproject.toml` (hatchling), ruff (incl.
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
