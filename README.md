# Agent Replay

[![CI](https://github.com/xeeshan16/agent-replay/actions/workflows/ci.yml/badge.svg)](https://github.com/xeeshan16/agent-replay/actions/workflows/ci.yml)

**Record, replay and regression-gate AI agent trajectories.**

> Status: ✅ complete (built 2026-10-14 → 2026-10-20). See [docs/PLAN.md](docs/PLAN.md),
> [docs/PROGRESS.md](docs/PROGRESS.md) and the [ADRs](docs/adr/README.md).

## The problem

Agents fail across multi-step tool chains, not single prompts. A prompt tweak, a model upgrade
or a changed tool description can make an agent skip a lookup, loop, call a write tool with
drifted arguments, or burn twice the tokens, while single-turn evals stay green. Testing this
against live tools is slow, flaky and unsafe: you can't re-issue a refund every time CI runs.

Agent Replay records what an agent did as a typed *trajectory* (every model call and tool call),
then replays new agent versions against the **recorded world**: tool results come from a
cassette, so runs are deterministic, free and side-effect-safe. Trajectories are diffed to the
first divergence, scored in layers, and a CI gate fails the build on regressions.

## Architecture

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
    T2 --> SC[Layered scoring<br/>checks → invariants → judge]
    T --> SC
    SP[Scoring spec TOML<br/>budgets, rules, judge] --> SC
    JD[Judge LLMProvider] -. only if cheaper layers pass .-> SC
    SC --> SCD[Scorecard<br/>verdict, findings, costs]
    SU[(Baseline suite<br/>pinned fingerprints)] --> G
    S --> SU
    D --> G[CI gate<br/>bootstrap + Wilson CIs,<br/>thresholds]
    SCD --> G
    G --> J[JUnit / Markdown / JSON<br/>exit 1 on breach]
    S --> API[Read API<br/>/v1/trajectories, /v1/suites, /v1/diff]
    MT[maintain<br/>retention + blob sweep] --> S
```

## How it works

1. **Record.** The recorder wraps an agent's model provider and tool executor. Every model call
   and tool call becomes a typed step of a `Trajectory` and an OpenTelemetry GenAI span; agents
   in other frameworks that emit OTel GenAI spans arrive over OTLP/HTTP instead and are
   re-assembled from their spans.
2. **Store.** Trajectories go to PostgreSQL with messages and tool payloads as content-addressed
   blobs, so a 20-turn conversation stores its shared prefix once. Ingest is idempotent per span
   and re-assembles a trace whenever late spans arrive.
3. **Replay.** A new agent version runs against the recording's cassette. Each request is
   normalised (timestamps, UUIDs, minted IDs, unordered lists) and matched exact → normalised →
   optionally fuzzy. In `TOOLS` mode the new model runs for real and only tools come from the
   cassette; in `FULL` mode model calls are served too. Every tool call passes a fail-closed
   sandbox: an unrecorded write is never executed.
4. **Diff.** Both runs become sequences of requested tool calls, aligned with a weighted edit
   distance whose substitution cost looks at arguments and results. The first divergence gets a
   reason (`missing_step`, `argument_drift`, `extra_loop`, ...) and the model step that made the
   choice.
5. **Score.** Deterministic checks (status, budgets, cost ratio, blocked writes), then
   invariants written in a small rule language, then, only if both passed, an LLM judge.
6. **Gate.** Over a baseline suite, optionally with repeats: per-case classification, paired
   cluster-bootstrap and Wilson intervals, cost and tool-call ratios, and a non-zero exit on a
   breach, with Markdown, JUnit and JSON reports.

## Components

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
  idempotency key per refund, ticket tags built from a set). Controlled variants: six
  regressions (`skip-lookup`, `no-retry`, `amount-drift`, `loop-lookup`, `ignore-eligibility`,
  `false-claim`) and one benign change (`check-policy`, an extra read).
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
- **Layered scoring** (`scoring/`): a scorecard per run, optionally against its baseline, in
  three layers ordered by cost and trust. *Checks*: ended successfully with an answer, within
  model-call, tool-call and cost budgets (absolute, and relative to the baseline), and no
  side-effecting call the sandbox had to block. *Invariants*: a small rule language over the tool
  calls the model requested (below). *Judge*: an LLM verdict behind the provider interface,
  which by default runs only when the first two layers passed. Verdict `pass`, `fail` or
  `inconclusive`; every finding names its layer and the model step to look at.
- **Invariants DSL** (`scoring/invariants.py`): rules a reviewer can read, with parse errors
  that point at the line and column. Rules see calls that never executed (a blocked write is
  still an attempted write), and a path that does not resolve compares false, so they fail
  closed. The support desk's rules:

  ```text
  lookup_before_refund: require lookup_order where ok before issue_refund on order_id
  refund_only_if_eligible: require lookup_order where ok and output.status == "delivered"
      and output.days_since_delivery <= 30 before issue_refund on order_id
  refund_within_total: require lookup_order where ok and output.total >= target.args.amount
      before issue_refund on order_id
  one_refund_per_order: at_most 1 issue_refund where ok per order_id
  bounded_lookups: at_most 2 lookup_order per order_id
  no_write_after_failed_read: no write after failed read
  ```
- **Judge** (`scoring/judge.py`): run data goes in as escaped JSON inside `<run-data>` tags it
  cannot close, results are truncated to a character budget, a malformed reply gets one repair
  attempt, and a timeout, provider error or unparseable reply is `inconclusive`, never `pass`.
  CI uses a deterministic judge policy for the support desk that applies the rubric's grounding
  rules (claimed actions need a successful tool call; minted IDs and amounts must appear in a
  tool result); any `LLMProvider` can take its place.
- **Cost accounting** (`accounting.py`): deterministic token estimation for the fake provider,
  per-model usage, and repricing with one price table (from the scoring spec) so a candidate and
  its baseline are compared on the same prices; the judge's spend is reported separately.
- **Regression-detection eval** (`evaluation.py`, [evals/](evals/README.md)): 64 labelled
  synthetic cases (every policy on eight tasks) measuring the diff alone, each scoring layer
  alone and the layered verdict as regression detectors.
- **Determinism harness** (`determinism.py`): records in one interpreter and replays in
  another, a wall-clock second later and with another hash seed, and reports identical-run
  rates per normaliser and replay mode plus the OTLP round trip; `agent-replay determinism`
  exits 1 unless the agent's own normaliser is stable everywhere, and CI runs it.
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
- **CI gate** (`gate.py`, `stats.py`, `gate_report.py`): replays a candidate against every case
  of a baseline suite (optionally `--repeats N` times, concurrently), diffs and scores each
  replay, and classifies it `pass`, `changed` (diverged but the scorecard passed: needs a human
  look, does not breach), `inconclusive`, `fail` or `error`. Suite-level thresholds: the lower
  bound of a paired cluster-bootstrap CI on the success change, an optional success-rate floor
  taken as the lower of the bootstrap and Wilson bounds, suite cost and tool-call ratios, blocked
  side effects and harness errors. Reports as Markdown (job summary), JUnit XML (one test per
  case plus one per threshold) and JSON; exit 1 on a breach. Seeded, so reports reproduce.
- **Baseline suites** (`suite.py`): a directory with a `suite.json` manifest that pins each
  case's behavioural fingerprint; an edited or truncated baseline is refused. Suites can be
  pushed to the store and gated from there (`gate --suite NAME`).
- **Read API** (`server.py`): `GET /v1/trajectories[/{id}]`, `/v1/suites[/{name}]` and
  `/v1/diff?baseline=&candidate=&format=json|md`, optionally behind its own bearer token.
- **Retention and blob sweep** (`store/`, `agent-replay maintain`): traces idle past the
  retention are closed with a tombstone (a straggler span is dropped instead of rebuilding a
  fragment over the complete run), and blobs no stored trajectory reaches are swept under an
  advisory lock that saves share, so a sweep is safe during ingest.
- **Real providers, optional** (`providers/http.py`, `providers/config.py`): an
  OpenAI-compatible client (OpenAI, Azure OpenAI, Ollama, vLLM) and an Anthropic client behind
  the same interface, wrapped in the retry/timeout layer and selected with `AGENT_REPLAY_LLM_*`
  (`record --llm env`, `gate --llm env`, `--judge env`). Never used by tests, CI or the demo.
- **Packaging**: a multi-stage, non-root Docker image with a health check, and docker-compose
  with the app, PostgreSQL and a maintenance job; CI builds the image and sends a run through
  the running container.
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

# Is the change acceptable? Score a replay against its baseline: exit 0 pass, 1 fail, 3 undecided
agent-replay replay run.json --policy skip-lookup --unrecorded-writes block --out unsafe.json
agent-replay score unsafe.json --baseline run.json          # --spec my.toml, --format json

# Run the labelled regression-detection eval (about a second, no keys)
agent-replay eval --out-dir evals/results

# Does a recording replay identically in another process, another second, another hash seed?
agent-replay determinism --pairs 5 --out-dir evals/results     # about 5 s, exit 1 if unstable

# Gate a candidate against a baseline suite: exit 0 if every threshold holds, 1 on a breach
agent-replay suite record --out-dir suites/desk-v1 --name desk-v1   # 7 sample tasks, v1
agent-replay gate --baseline suites/desk-v1 --policy skip-lookup \
  --markdown gate.md --junit gate.xml --json gate.json        # --repeats 5, --config gate.toml
```

Excerpt of `gate.md` for the `skip-lookup` candidate (real output):

```markdown
### Agent gate: BREACHED

Success rate 0.286 (baseline 1.000); 95% CI bootstrap [0.000, 0.571], Wilson [0.082, 0.641].
Change vs baseline -0.714, 95% CI [-1.000, -0.429].

| Threshold | Result | Value | Limit | Detail |
|-----------|--------|-------|-------|--------|
| success_drop | BREACH | -1.0 | -0.1 | success change -0.714, 95% CI [-1.000, -0.429]; lower bound must be &gt;= -0.100 |
| cost_ratio | ok | 1.0015 | 1.25 | suite cost 1.00x the baseline's (max 1.25x) |
| tool_call_ratio | ok | 1.0909 | 1.5 | 1.09x the baseline's tool calls (max 1.50x) |
| unsafe_actions | BREACH | 5 | 0 | 5 side-effecting call(s) matched no recording and were blocked (max 0) |
| errors | ok | 0 | 0 | 0 sample(s) could not be run (max 0) |
```

Thresholds live in a `[gate]` table (all optional; shown with their defaults):

```toml
[gate]
max_success_drop = 0.10     # lower CI bound of (candidate - baseline) success must be >= -0.10
min_success_rate = 0.90     # off by default; checked against the conservative lower bound
max_cost_ratio = 1.25       # suite cost vs the baseline's (tokens when models are unpriced)
max_tool_call_ratio = 1.5
max_unsafe_actions = 0      # side-effecting calls the sandbox had to block
max_errors = 0
fail_on_inconclusive = false
confidence = 0.95
resamples = 2000
seed = 0
```

Excerpt of `diff.md` from the `amount-drift` replay above (real output, line-wrapped):

```markdown
**First divergence:** `argument_drift`: called issue_refund with different arguments
(requested, never executed) [replay: blocked] (baseline step 3, decided at candidate model step 2)

> **Involves a side-effecting tool.**
- amount: 59.9 → 60.0
```

Excerpt of the scorecard for `unsafe.json` above (real output, long cells truncated by the
renderer):

```markdown
**Verdict:** fail (failed: checks, invariants)

| Layer | Check | Result | Step | Detail |
|-------|-------|--------|------|--------|
| checks | blocked_writes | **FAIL** | 0 | 1 side-effecting call(s) matched no recording and were not executed, first issue_refund(o… |
| invariants | lookup_before_refund | **FAIL** | 0 | 1 violation(s), first: issue_refund without a prior lookup_order where ok with the same o… |
| judge | | skipped | | an earlier layer already failed |
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

### Run the service with Docker

```bash
docker compose up -d --build            # app on 127.0.0.1:4318 + PostgreSQL
curl -s localhost:4318/healthz
curl -s -H 'Content-Type: application/json' --data-binary @spans.json localhost:4318/v1/traces

# Push a suite, then gate against it from any CI job that can reach the database
agent-replay suite push suites/desk-v1
agent-replay gate --suite desk-v1 --policy amount-drift
curl -s "localhost:4318/v1/diff?baseline=desk-v1/01-i-d-like-a-refund&candidate=<id>&format=md"

# Housekeeping: close traces idle for 72 h, sweep unreferenced blobs older than 1 h
docker compose run --rm maintain        # or: agent-replay maintain --span-retention-hours 72
```

### Real model providers (optional)

```bash
export AGENT_REPLAY_LLM_PROVIDER=ollama AGENT_REPLAY_LLM_MODEL=llama3.1   # or openai, azure, anthropic
agent-replay record --llm env --task "Please refund ORD-1003." --out real.json
agent-replay gate --baseline suites/desk-v1 --llm env --judge env --repeats 5
```

Prices for real models are not bundled; add a `[prices.<model>]` table to the scoring spec to
get costs (ratios fall back to tokens otherwise). Nothing in this README's results used a real
provider.

### Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `AGENT_REPLAY_DATABASE_URL` | unset (in-memory) | PostgreSQL URL for `serve`, `ingest`, `stats` |
| `AGENT_REPLAY_INGEST_TOKEN` | unset (open) | Require `Authorization: Bearer <token>` on `POST /v1/traces` |
| `AGENT_REPLAY_CAPTURE_CONTENT` | `false` | Put prompts, tool arguments and results on recorder spans |
| `AGENT_REPLAY_READ_TOKEN` | unset (open) | Require a bearer token on the read API |
| `AGENT_REPLAY_LOG_LEVEL` | `WARNING` | JSON log level |
| `AGENT_REPLAY_LLM_PROVIDER` | `fake` | `fake`, `openai`, `azure`, `ollama` or `anthropic` |
| `AGENT_REPLAY_LLM_MODEL`, `_BASE_URL`, `_API_KEY`, `_API_VERSION` | unset | Real-provider settings (see `.env.example`) |
| `AGENT_REPLAY_LLM_TIMEOUT_S`, `_MAX_ATTEMPTS` | `60`, `3` | Per-attempt timeout and retries |

## Design trade-offs

- **Diff actions, not messages.** The diff aligns tool calls and compares the outcome; it does
  not align intermediate model text, which changes with every prompt edit. Wording quality is a
  scoring concern (the judge layer), localisation is the diff's.
- **Edit distance with forward tracing.** The alignment table is filled from the end and traced
  from the start, preferring a substitution, then a deletion, then an insertion on ties. That
  makes the *later* of two equal calls the extra one, which is how a retry loop reads. A rename
  costs 1.5 (less than delete + insert = 2), so "called the wrong tool here" wins over "dropped
  one tool and added another"; same-tool substitutions cost at most 1. The costs are
  configurable (`DiffCosts`) and validated so these orderings always hold.
- **Scoring is layered, and the judge goes last.** Deterministic checks and invariants are
  free, repeatable and explainable; a judge is none of those. Running it only when the cheaper
  layers passed means it never overturns a deterministic failure and is never paid for a run
  that already failed (on the eval set: 47 judge calls instead of 64, 29% less judge spend).
  `when = "always"` exists for measuring the layers separately.
- **A rule language instead of Python predicates.** Invariants are policy; a reviewer who does
  not write Python should be able to read and change them, and a parser can reject a typo with a
  line and column instead of a stack trace. The language stays small (four statement forms,
  comparisons, boolean logic); anything richer belongs in a check written in code.
- **Diff and scoring answer different questions.** The diff flags every behaviour change, so it
  also flags benign ones (5 of 5 in the eval); scoring judges whether a run is acceptable, and
  misses regressions it has no rule for (`no-retry`: giving up on a transient failure produces a
  polite, grounded answer that breaks no rule). The CI gate combines them: a scorecard failure
  fails the case, a divergence with a passing scorecard is reported as `changed` for review.
- **Resample tasks, not runs.** With `--repeats`, runs of one task are correlated (a task the
  candidate gets wrong it tends to get wrong every time). The gate's bootstrap resamples tasks
  and uses each task's mean, so ten repeats of one flaky task do not count as ten pieces of
  evidence; the success change is bootstrapped *paired* by task, so a task both versions fail
  adds nothing.
- **Take the lower of two lower bounds.** The percentile bootstrap collapses to `[1, 1]` when
  every task passes; Wilson does not (7/7 gives about 0.65 at 95%). The success-rate floor uses
  the lower of the two, which is why it is off by default: on a 7-task suite even a perfect run
  cannot prove 0.9. Small suites should rely on `max_success_drop` and per-case failures, and
  grow the suite before tightening the floor.
- **Retention closes traces instead of trimming them.** Deleting old spans alone would let a
  late re-delivered span re-assemble a one-span fragment over the complete trajectory; a
  tombstone makes the late span a counted, harmless drop (`spans_expired`).
- **Sweep with a lock, not reference counts.** Blobs are referenced from steps and from inside
  other blobs, so there is no cascade to lean on and counts would have to be maintained on every
  save. Mark-and-sweep under an exclusive advisory lock, with every save holding it shared, is
  simple and provably safe against a concurrent save reusing a blob; the cost is that saves
  wait for a running sweep, and marking reads every request blob (fine at this scale, to be
  made incremental for large stores).
- **Relative cost budgets have a price.** `max_cost_ratio = 1.5` caught a lookup loop the
  invariants also caught, and failed two benign changes whose extra read is a large share of a
  short baseline run. The threshold is set by policy, not tuned on the eval.
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
- The judge reads untrusted text (the customer's task, tool outputs, the agent's answer). It is
  serialised as JSON with `<` and `>` escaped inside `<run-data>` tags, so content cannot close
  the block, and the rubric says to treat it as evidence only. That narrows prompt injection
  without removing it, which is why the judge runs last, cannot overturn a deterministic
  failure, and anything but a valid verdict counts as inconclusive. Judge requests carry
  recorded content to the judge's provider: use a provider approved for that data.
- The read API returns prompts and tool outputs. Set `AGENT_REPLAY_READ_TOKEN`; it is separate
  from the ingest token so a span exporter's credential cannot read data back.
- Real-provider API keys are read from the environment, sent only as request headers, and never
  logged or included in error messages (response bodies quoted in errors are truncated). `--llm
  env` and `--judge env` send recorded content to that provider.
- Gate reports quote untrusted text too: Markdown cells are escaped as above, and JUnit XML is
  built with ElementTree after stripping control characters XML 1.0 cannot carry, so a hostile
  tool output cannot produce a report the CI test viewer refuses to parse.
- Suites pin each baseline's fingerprint and refuse case files that resolve outside the suite
  directory, so a tampered or path-traversing manifest fails loudly instead of changing what the
  gate compares against.
- The container runs as an unprivileged user with a read-only root filesystem in compose, and
  compose binds ports to localhost only.
- Recorded tool outputs are untrusted input to the agent under test (they may contain prompt
  injection captured from the real world); replay reproduces them faithfully on purpose.

## Results

Every number below was measured on 2026-10-20 (the gate table on 2026-10-19, with no gate code
changed since; the verdicts were re-checked on 2026-10-20) by the commands shown, with the
deterministic fake agent model `fake-tool-caller-1` and fake judge `fake-judge-1`, on Python
3.12, with no network. They describe this project's reference support-desk agent and seeded
regressions, not the behaviour of a real model.

### 1. Regression detection

`agent-replay eval --out-dir evals/results` on [`evals/regressions.jsonl`](evals/README.md):
64 synthetic cases (every agent policy on eight tasks: 18 regression, 5 benign, 41 no-effect),
labelled from each policy's definition before any detector ran. Settings: TOOLS replay, live
reads, unrecorded writes blocked, strict matching with the support-desk normaliser, the bundled
scoring spec (`max_cost_ratio = 1.5`, six invariants), judge only when checks and invariants
pass.

| Detector | TP | FP | FN | TN | Benign flagged | Precision | Recall | F1 |
|----------|----|----|----|----|----------------|-----------|--------|----|
| diff (any divergence) | 18 | 5 | 0 | 41 | 5/5 | 0.78 | 1.00 | 0.88 |
| checks only | 14 | 2 | 4 | 44 | 2/5 | 0.88 | 0.78 | 0.82 |
| invariants only | 15 | 0 | 3 | 46 | 0/5 | 1.00 | 0.83 | 0.91 |
| judge only | 2 | 0 | 16 | 46 | 0/5 | 1.00 | 0.11 | 0.20 |
| **layered verdict** | 17 | 2 | 1 | 44 | 2/5 | **0.89** | **0.94** | **0.92** |

Recall of the layered verdict by regression category: skipped lookup 6/6, extra loop 6/6,
unsafe write 2/2, argument drift 1/1, unsupported claim 2/2 (only the judge sees these),
missing retry 0/1. Its two false positives are the benign extra read (`check-policy`) on two
short runs, where one more read exceeds the 1.5x cost budget.

**Localisation:** the diff's first divergence named the labelled decision step in **23/23**
changed cases. **Judge spend:** 47 calls and $0.023266 at the illustrative fake prices in the
layered configuration, against 64 calls and $0.032889 with the judge on every case (29% less).

### 2. Replay determinism across processes

`agent-replay determinism --pairs 10`: each pair records the 7 sample tasks live in one
interpreter and replays them in another, started in a later wall-clock second with a different
`PYTHONHASHSEED` (70 runs per cell). "Live re-run" compares two independent live runs of the
same task, which is what you get without a cassette.

| Normaliser | Live re-run | FULL replay | TOOLS replay |
|------------|-------------|-------------|--------------|
| none | 40/70 (57%) | 0/70 (0%) | 40/70 (57%) |
| default (timestamps, UUIDs, call IDs) | 42/70 (60%) | 70/70 (100%) | 62/70 (89%) |
| support-desk (default + unordered `tags` + refund IDs) | 70/70 (100%) | 70/70 (100%) | 70/70 (100%) |

OTLP round trip: 70/70 trajectories rebuilt from their exported spans had the recording's exact
fingerprint, and 70/70 of those replayed identically in FULL mode.

Without normalisation, FULL replay never matches (the system prompt carries the time), and
TOOLS replay and live re-runs fail whenever a ticket ID or refund key was minted. The default
normaliser fixes the time and UUIDs; the remaining TOOLS misses are ticket tags coming out of a
set in hash-seed order. The agent's own normaliser is stable everywhere. With a deterministic
fake model, a live re-run is as stable as a replay once normalised; what the replay adds here is
that no tool is ever executed and no order store is needed. With a real model, the live column
would also carry sampling variance, which this table does not measure.

### 3. CI gate

`agent-replay gate` against a 7-task suite recorded from `v1` (default thresholds, one repeat;
the fake model is deterministic, so repeats change nothing here):

| Candidate | Verdict | pass / changed / fail | Success [95% Wilson] | Change vs baseline [95% CI] | Cost ratio | Breached |
|---|---|---|---|---|---|---|
| `v1` | pass | 7 / 0 / 0 | 1.00 [0.65, 1.00] | +0.00 [+0.00, +0.00] | 1.00 | none |
| `check-policy` (benign) | breach | 3 / 3 / 1 | 0.86 [0.49, 0.97] | -0.14 [-0.43, +0.00] | 1.27 | success_drop, cost_ratio |
| `no-retry` | pass | 6 / 1 / 0 | 1.00 [0.65, 1.00] | +0.00 [+0.00, +0.00] | 0.85 | none |
| `amount-drift` | breach | 6 / 0 / 1 | 0.86 [0.49, 0.97] | -0.14 [-0.43, +0.00] | 1.00 | success_drop, unsafe_actions |
| `skip-lookup` | breach | 2 / 0 / 5 | 0.29 [0.08, 0.64] | -0.71 [-1.00, -0.43] | 1.00 | success_drop, unsafe_actions |
| `loop-lookup` | breach | 2 / 0 / 5 | 0.29 [0.08, 0.64] | -0.71 [-1.00, -0.29] | 1.60 | success_drop, cost_ratio, tool_call_ratio |
| `ignore-eligibility` | breach | 6 / 0 / 1 | 0.86 [0.49, 0.97] | -0.14 [-0.43, +0.00] | 1.08 | success_drop, unsafe_actions |
| `false-claim` | breach | 6 / 0 / 1 | 0.86 [0.49, 0.97] | -0.14 [-0.43, +0.00] | 0.99 | success_drop |

Every seeded regression but `no-retry` breaches; `no-retry` passes with its changed case listed
for review, the honest outcome when no rule describes the regression. The benign `check-policy`
breaches on cost, the same trade-off as in the eval. A gate with `--repeats 5` takes about
0.5 s.

**Performance notes** (this machine, not a benchmark): diffing 200 x 200 synthetic actions takes
about 40 ms; the full regression eval about a second; ten determinism pairs about 10 s, most of
it interpreter start-up. Storage on the five Day 2 sample runs: 19,805 logical bytes stored as
11,179 blob bytes (1.77x), and re-ingesting the same spans inserted nothing.

## Limitations

- **One reference agent, fake models.** Every number above comes from a small deterministic
  support-desk agent with seeded regressions. Nothing here measures how a real model's
  run-to-run variance interacts with strict matching, how often TOOLS replay misses on a real
  model that rephrases tool arguments, or how a real LLM judge agrees with humans.
- **Small, synthetic eval set.** 64 cases from 8 policies on 8 tasks; categories have 1 to 6
  regressions each, so per-category rates are anecdotes, not estimates. Labels were written by
  the same author as the detectors.
- **Replay needs content.** Traces ingested without the opt-in GenAI content attributes keep
  their structure but cannot be replayed; most instrumentations default content capture off.
- **Normalisers are per agent.** The default rules cover timestamps, UUIDs and call IDs; an
  agent with other volatile values (the refund IDs and tag sets here) needs its own rules, and
  an over-broad rule can hide a real difference.
- **The sandbox trusts the `side_effect` flag.** A tool that writes but is declared read-only
  will run live under `--live-reads`.
- **Streaming and multimodal content** are not modelled: the provider interface returns one
  complete text-and-tool-calls message per model call.
- **Scale.** Alignment is O(n·m) with a cell cap; the blob sweep marks by reading every request
  blob; the span buffer re-assembles a whole trace on every batch. All fine at the sizes tested,
  none designed for millions of runs.
- **No redaction at ingest.** Stored recordings hold full prompts and tool outputs.

## What I'd do next

- A real-model study: record a few hundred runs with a small local model (Ollama) at a fixed
  temperature, and measure TOOLS-replay miss rates, fuzzy-match precision and gate flakiness
  with repeats, so thresholds can be set from data rather than policy.
- Judge calibration: label a set of runs by hand and report the judge's agreement (Cohen's
  kappa) before trusting its verdicts in the gate.
- Redaction at ingest (PII patterns and per-attribute allow lists) applied before blobs are
  written, so recordings can leave production.
- An adapter that records LangGraph and OpenAI Agents SDK runs in-process (not only via OTLP),
  plus importers for their trace formats.
- Incremental blob marking and partitioned span buffers for large stores; a GitHub Action that
  posts the gate's Markdown as a PR comment.
- Minimised counterexamples: when the gate fails, shrink the suite to the smallest set of cases
  that still breaches.

## License

MIT, see [LICENSE](LICENSE).
