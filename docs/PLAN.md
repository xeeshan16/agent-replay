# Agent Replay: 7-day plan

One milestone per day. Each day ships working, tested code; Day 7 is evaluation, polish and docs.

| Day | Milestone | Done when |
|-----|-----------|-----------|
| 1 | Foundations and first vertical slice | Package layout, ruff + mypy strict + pytest, async provider interface with a deterministic fake and a retry/timeout wrapper, the `Trajectory` model, a recorder that emits OTel GenAI spans, strict cassette replay (`FULL` and `TOOLS` modes), a reference agent, CLI, CI workflow, docs skeleton |
| 2 | Ingestion and storage | OTLP/HTTP JSON receiver mapping `invoke_agent` / `chat` / `execute_tool` spans to `Trajectory`; PostgreSQL store (async SQLAlchemy) with content-addressed tool I/O blobs and idempotent ingest keyed by trace ID; integration tests against a Postgres service in CI |
| 3 | Replay engine, hardened | Normaliser chain (timestamps, UUIDs, list ordering) applied before matching; strict and fuzzy match modes with a similarity threshold; side-effect sandbox (read-only tools may run live, writes are always blocked); richer reference agent with more tools and failure paths |
| 4 | Trajectory diff | Alignment of tool-call sequences (edit-distance variant with argument-aware substitution cost), first-divergence detection with a reason (wrong tool, argument drift, extra loop, missing step, different answer), CLI and Markdown rendering |
| 5 | Layered scoring and the eval dataset | Deterministic checks → invariants DSL (e.g. "lookup before refund", "no write after error") → optional LLM judge behind the provider interface; token and cost accounting per run; labelled synthetic dataset of seeded regressions (wrong tool, extra loops, unsafe write, argument drift) |
| 6 | CI gate, API and packaging | `agent-replay gate --baseline <suite>` with thresholds, bootstrap confidence intervals on success rate, JUnit and Markdown reports, non-zero exit on breach; FastAPI read API; Dockerfile and docker-compose (app + Postgres); optional real providers (OpenAI-compatible, Anthropic, Ollama) |
| 7 | Evaluation, polish, docs | Measured eval: regression-detection precision/recall, first-divergence localisation accuracy, replay determinism rate; README results table, Mermaid diagram, 2–4 ADRs, limitations, clean-clone quick start check |

## Adjustments from the original proposal

- The reference agent is a small framework-free tool-calling loop rather than a LangGraph graph.
  Recording and replay only need the provider and tool-executor seams, so a framework would add
  a heavy dependency without exercising anything new. Day 2's OTLP ingest is how traces from
  LangGraph or other instrumented frameworks come in.
- Strict cassette replay and the reference agent moved into Day 1 to have a true end-to-end
  slice (record → replay → divergence) from the first day. Day 3 now focuses on normalisation,
  fuzzy matching and the side-effect sandbox.
