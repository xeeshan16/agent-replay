# Agent Replay

[![CI](https://github.com/xeeshan16/agent-replay/actions/workflows/ci.yml/badge.svg)](https://github.com/xeeshan16/agent-replay/actions/workflows/ci.yml)

**Record, replay and regression-gate AI agent trajectories.**

> Status: 🚧 in progress (day 2 of 7). See [docs/PLAN.md](docs/PLAN.md) and
> [docs/PROGRESS.md](docs/PROGRESS.md).

## The problem

Agents fail across multi-step tool chains, not single prompts. A prompt tweak, a model upgrade
or a changed tool description can make an agent skip a lookup, loop, call a write tool with
drifted arguments, or burn twice the tokens, while single-turn evals stay green. Testing this
against live tools is slow, flaky and unsafe: you can't re-issue a refund every time CI runs.

Agent Replay records what an agent did as a typed *trajectory* (every model call and tool call),
then replays new agent versions against the **recorded world**: tool results come from a
cassette, so runs are deterministic, free and side-effect-safe. Trajectories are diffed to the
first divergence, scored in layers, and a CI gate fails the build on regressions.

## Architecture (target)

```mermaid
flowchart LR
    subgraph Capture
        A[Agent under test] -- wrapped LLM + tools --> R[Recorder]
        OT[OTel GenAI spans<br/>any framework] -- OTLP/HTTP<br/>protobuf or JSON --> I[Receiver<br/>POST /v1/traces]
        R -- OTLP exporter --> I
    end
    I --> B[(Span buffer<br/>dedup on trace+span ID)]
    B --> AS[Assembler<br/>per trace, advisory lock]
    R --> T[(Trajectory)]
    AS --> T
    T --> S[(PostgreSQL<br/>headers, steps,<br/>content-addressed blobs)]
    S --> C[Cassette]
    C --> RP[Replayer<br/>FULL / TOOLS mode]
    A2[New agent version] --> RP
    RP --> T2[(Replayed trajectory)]
    T2 -.-> D[Diff: first divergence]
    T2 -.-> SC[Layered scoring<br/>checks → invariants → judge]
    D -.-> G[CI gate]
    SC -.-> G
    G -.-> J[JUnit + Markdown report]
```

Solid lines exist today; dotted lines are planned (see the plan).

## What works today

- **Trajectory model** (`model.py`): pydantic v2, frozen, mirrors the OpenTelemetry GenAI
  `invoke_agent` → `chat` / `execute_tool` span shape; content fingerprints that ignore IDs,
  timestamps and durations.
- **Provider interface** (`providers/`): an async `LLMProvider` protocol, a deterministic
  `FakeLLM` driven by a pure policy function, and `ResilientLLM` (per-attempt timeout, bounded
  retries with full-jitter exponential backoff, fail-fast on non-retryable errors).
- **Recorder** (`recorder.py`): wraps any agent's provider and tool executor, emits OTel GenAI
  spans with token and cost attributes, and returns a `Trajectory`, also for failed runs.
- **Replayer** (`replay.py`): content-addressed cassette (canonical request → response, FIFO for
  repeats), `FULL` and `TOOLS` modes, strict matching that fails at the exact divergence point,
  and no real tool is executed during replay.
- **Reference agent** (`reference/support_desk.py`): a support-desk agent with a read tool and a
  side-effecting refund tool, plus a seeded regression policy.
- **Cost accounting**: deterministic token estimation for the fake provider and a configurable
  price table.
- **OTLP receiver** (`server.py`, `ingest/`): `POST /v1/traces` accepts OTLP/HTTP protobuf and
  JSON (optionally gzipped), answers with OTLP partial success for bad spans, 503 + `Retry-After`
  when the store is down, and enforces a body limit (also after decompression) and an optional
  bearer token. The standard OpenTelemetry `OTLPSpanExporter` works against it unchanged.
- **Trace assembly** (`ingest/assemble.py`): every `invoke_agent` span becomes a trajectory built
  from its `chat` / `execute_tool` descendants (nested agents stay separate). With the opt-in
  GenAI content attributes (`gen_ai.input.messages`, `gen_ai.tool.call.arguments`, ...) the
  result replays exactly like the in-process recording; without them the structure is kept and
  the run is flagged `content_captured: false`.
- **Idempotent ingest** (`ingest/service.py`): spans are buffered by `(trace_id, span_id)`,
  re-delivered spans are dropped, and each touched trace is re-assembled from all its spans, so
  out-of-order batches converge and a late child span becomes a new revision of the same record.
  See [ADR 0002](docs/adr/0002-otlp-ingest-span-buffer.md).
- **Storage** (`store/`): one `TrajectoryStore` interface, an in-memory implementation and a
  PostgreSQL one (async SQLAlchemy Core + asyncpg). Messages, tool definitions and tool payloads
  are content-addressed blobs shared across turns and runs; reads verify the fingerprint. See
  [ADR 0003](docs/adr/0003-message-level-blob-storage.md).

## Quick start

```bash
git clone https://github.com/xeeshan16/agent-replay && cd agent-replay
uv venv -p 3.12 && uv pip install -e '.[dev]'   # or: python -m venv .venv && pip install -e '.[dev]'

# Record a run of the reference agent
agent-replay record --task "I'd like a refund for ORD-1001, it arrived damaged." --out run.json

# Replay it unchanged: identical, exit 0
agent-replay replay run.json --mode full

# Replay a regressed agent version against the recorded tools: diverges, exit 1
agent-replay replay run.json --mode tools --policy skip-lookup

# Record and replay every sample task
agent-replay demo
```

No API keys are needed; the default provider is the deterministic fake.

### Ingest traces into PostgreSQL

```bash
docker compose up -d postgres
export AGENT_REPLAY_DATABASE_URL=postgresql://agent_replay:agent_replay@localhost:5432/agent_replay

# Offline: write a run's spans as an OTLP/JSON request and ingest the file
agent-replay record --task "Please refund ORD-1003." --out run.json --otlp-out spans.json
agent-replay ingest spans.json      # also accepts .pb and collector JSON-lines files
agent-replay stats                  # trajectories, blobs, deduplication ratio

# Online: run the receiver on the standard OTLP/HTTP port
agent-replay serve --port 4318
```

Point any OpenTelemetry SDK at it with
`OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://localhost:4318/v1/traces`; enable content capture in
the instrumentation to get replayable trajectories. Stored runs are readable at
`GET /v1/trajectories` and `GET /v1/trajectories/{id}`.

### Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `AGENT_REPLAY_DATABASE_URL` | unset (in-memory) | PostgreSQL URL for `serve`, `ingest`, `stats` |
| `AGENT_REPLAY_INGEST_TOKEN` | unset (open) | Require `Authorization: Bearer <token>` on `POST /v1/traces` |
| `AGENT_REPLAY_CAPTURE_CONTENT` | `false` | Put prompts, tool arguments and results on recorder spans |
| `AGENT_REPLAY_LOG_LEVEL` | `WARNING` | JSON log level |

## Development

```bash
ruff check . && ruff format --check .
mypy
pytest                                    # PostgreSQL tests skip unless configured:
AGENT_REPLAY_TEST_DATABASE_URL=postgresql://agent_replay:agent_replay@localhost:5432/agent_replay \
  pytest
```

## Security notes

- Replay never executes real tools; side-effecting tools (`side_effect=True`) are flagged in the
  trajectory and on spans so later scoring can treat any new write as unsafe.
- Recorded trajectories contain full prompts and tool outputs. Treat stored recordings with the
  same data classification as the production data they came from; redaction at ingest is planned.
- Content capture on spans is **off by default** (`AGENT_REPLAY_CAPTURE_CONTENT`), matching the
  OpenTelemetry GenAI guidance; without it, only structure, tokens and errors leave the process.
- The receiver is a network-facing parser of untrusted input: bodies are capped at 8 MiB before
  and after gzip decompression (a gzip bomb is refused with 413, not inflated), malformed spans
  are rejected individually, and an optional bearer token is compared in constant time. Run it
  behind TLS termination; it does not terminate TLS itself.
- Traces with no GenAI spans are dropped at the door so a misrouted collector cannot fill the
  span buffer with unrelated telemetry.
- Recorded tool outputs are untrusted input to the agent under test (they may contain prompt
  injection captured from the real world); replay reproduces them faithfully on purpose.

## Results

Evaluation results will be added on day 7, with measured numbers only.

## License

MIT, see [LICENSE](LICENSE).
