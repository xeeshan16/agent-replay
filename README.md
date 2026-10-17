# Agent Replay

[![CI](https://github.com/xeeshan16/agent-replay/actions/workflows/ci.yml/badge.svg)](https://github.com/xeeshan16/agent-replay/actions/workflows/ci.yml)

**Record, replay and regression-gate AI agent trajectories.**

> Status: 🚧 in progress (day 4 of 7). See [docs/PLAN.md](docs/PLAN.md) and
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
    N[Normaliser chain] --> M[Matcher<br/>exact → normalized → fuzzy]
    C --> M
    M --> RP[Replayer<br/>FULL / TOOLS mode]
    A2[New agent version] --> RP
    RP -- every tool call --> SB{Sandbox}
    SB -- recorded --> M
    SB -- unrecorded read<br/>--live-reads --> LT[Live read-only tools]
    SB -- unrecorded write --> BL[Blocked: fail or error result]
    RP --> T2[(Replayed trajectory)]
    T --> D[Diff<br/>alignment + reasons]
    T2 --> D
    D --> MD[Markdown / JSON<br/>first divergence]
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
- **Replayer** (`replay.py`): content-addressed cassette (FIFO for repeated identical calls),
  `FULL` and `TOOLS` modes, and a per-request match log. A request that cannot be served ends
  the run at the exact divergence point.
- **Normaliser chain** (`normalize.py`): masks timestamps, UUIDs and provider call IDs, drops
  volatile keys and sorts lists declared unordered, in one walk with placeholders indexed by
  first appearance (`<uuid:0>`), so identity survives while the concrete value is ignored.
- **Tiered matching** (`matching.py`): `exact` → `normalized` → opt-in `fuzzy` (closest
  unconsumed recording of the same tool or model by structural similarity, threshold 0.9 by
  default). Writes never match fuzzily unless `fuzzy_writes` is set. "Identical" means equal
  after normalisation; `byte_identical` keeps the raw comparison.
- **Side-effect sandbox** (`sandbox.py`): every replayed tool call goes through it. Unrecorded
  reads miss, or run live with `--live-reads`; unrecorded writes are never executed and either
  fail the run or return an error result to the agent (`--unrecorded-writes block`). A tool is a
  write if its spec, the recording, or the absence of a spec says so. See
  [ADR 0004](docs/adr/0004-tiered-matching-and-side-effect-sandbox.md).
- **Reference agent** (`reference/support_desk.py`): a support-desk agent with two reads
  (`lookup_order`, `get_refund_policy`) and two writes (`issue_refund`, idempotent per key, and
  `create_ticket`). It retries a transient lookup failure, escalates refunds over a limit to a
  ticket, and carries realistic volatile values (the time in its system prompt, a fresh
  idempotency key per refund, ticket tags built from a set). Controlled variants: `skip-lookup`,
  `no-retry` and `amount-drift` (regressions) and `check-policy` (a benign extra read).
- **Trajectory diff** (`diff.py`, `report.py`): turns both runs into the tool calls the model
  requested (including calls the sandbox refused, which never became tool steps), normalises
  them with the replay normaliser and aligns them with a weighted edit distance whose
  substitution cost is argument-aware, so a refund with a drifted amount lines up with the
  recorded refund instead of reading as one call removed and another added. Each difference gets
  a reason (`wrong_tool`, `argument_drift` with the JSON paths that changed, `output_drift`,
  `missing_step`, `extra_step`, `extra_loop`, `status_changed`, `different_answer`), and the
  first divergence names the candidate model call that made the choice. Rendered as Markdown
  (escaped for PR comments and CI summaries) or JSON; `replay` annotates each call with how it
  was served (normalised, fuzzy, live, blocked).
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

# Let the run continue past the blocked write so the whole trajectory can be inspected
agent-replay replay run.json --policy skip-lookup --unrecorded-writes block

# A benign change that adds a read: replays with live reads (the refund is still never executed)
agent-replay replay run.json --policy check-policy --live-reads

# Tolerate small argument drift on reads; writes still match exactly
agent-replay replay run.json --match fuzzy --threshold 0.9

# Record and replay every sample task
agent-replay demo

# Where and why did it diverge? Markdown diff with the first divergence and an aligned table
agent-replay replay run.json --policy amount-drift --diff-out diff.md

# Diff any two saved trajectories (exit 1 if they differ; --format json for machines)
agent-replay record --task "I'd like a refund for ORD-1001, it arrived damaged." --out again.json
agent-replay diff run.json again.json
```

Excerpt of `diff.md` from the `amount-drift` replay above (real output, line-wrapped):

```markdown
**First divergence:** `argument_drift`: called issue_refund with different arguments
(requested, never executed) [replay: blocked] (baseline step 3, decided at candidate model step 2)

> **Involves a side-effecting tool.**
- amount: 59.9 → 60.0
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

## Design trade-offs

- **Diff actions, not messages.** The diff aligns tool calls and compares the outcome; it does
  not align intermediate model text, which changes with every prompt edit. Wording quality is a
  scoring concern (Day 5), localisation is the diff's.
- **Edit distance with forward tracing.** The alignment table is filled from the end and traced
  from the start, preferring a substitution, then a deletion, then an insertion on ties. That
  makes the *later* of two equal calls the extra one, which is how a retry loop reads. A rename
  costs 1.5 (less than delete + insert = 2), so "called the wrong tool here" wins over "dropped
  one tool and added another"; same-tool substitutions cost at most 1. The costs are
  configurable (`DiffCosts`) and validated so these orderings always hold.
- **O(n·m) alignment** with a cell cap (`max_cells`, 1M by default) so a pathological pair fails
  fast. Measured on this project's synthetic actions: 200 x 200 actions diff in about 40 ms.

## Development

```bash
ruff check . && ruff format --check .
mypy
pytest                                    # PostgreSQL tests skip unless configured:
AGENT_REPLAY_TEST_DATABASE_URL=postgresql://agent_replay:agent_replay@localhost:5432/agent_replay \
  pytest
```

## Security notes

- Replay never executes a side-effecting tool. Read-only tools run live only with
  `--live-reads` and only on a cassette miss. Classification fails closed: a tool is treated as
  a write if its spec says so, if the recording flagged it, or if it has no spec, so relabelling
  a recorded write as read-only in a new agent version does not unlock it. The sandbox enforces
  the `side_effect` flag; it cannot verify it, so a tool that writes must be registered as such.
- Fuzzy matching is off by default and never applies to writes unless `--fuzzy-writes` is given:
  serving a recorded refund for a request with a different amount would hide argument drift on
  exactly the calls that matter. Every fuzzy match is reported with its similarity score.
- Normalisation can hide real differences if a rule is too broad (masking every number, say).
  The defaults only mask timestamps, UUIDs and provider call IDs, bare dates are kept, and the
  match log shows which steps needed normalisation.
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
- Diff reports quote tool arguments and outputs, which are untrusted. The Markdown renderer
  flattens, truncates and escapes every cell (pipes, angle brackets, backticks), so recorded
  content cannot break the table or inject markup into a PR comment or CI summary. Reports carry
  recorded data: publish them with the same care as the recordings.
- Recorded tool outputs are untrusted input to the agent under test (they may contain prompt
  injection captured from the real world); replay reproduces them faithfully on purpose.

## Results

Evaluation results will be added on day 7, with measured numbers only.

## License

MIT, see [LICENSE](LICENSE).
