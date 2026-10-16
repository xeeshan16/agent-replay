"""Command-line interface.

    agent-replay record --task "Please refund ORD-1001" --out run.json
    agent-replay replay run.json --mode tools --policy skip-lookup
    agent-replay replay run.json --match fuzzy --threshold 0.9 --live-reads
    agent-replay demo --out-dir runs/
    agent-replay record --task "..." --out run.json --otlp-out spans.json
    agent-replay ingest spans.json            # into AGENT_REPLAY_DATABASE_URL, or memory
    agent-replay serve --port 4318            # OTLP/HTTP receiver + read API
    agent-replay stats

``replay`` exits 0 when the replayed run is semantically identical to the recording and 1 when
it diverged, so it can be used directly in CI.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agent_replay.clock import TickingClock
from agent_replay.ingest.export import spans_to_otlp_json
from agent_replay.ingest.service import ingest_batch
from agent_replay.ingest.spans import decode_json, decode_protobuf
from agent_replay.logs import configure_logging
from agent_replay.matching import MatchMode, MatchPolicy
from agent_replay.model import Trajectory
from agent_replay.providers.fake import Policy
from agent_replay.recorder import record_run
from agent_replay.reference import support_desk
from agent_replay.replay import ReplayMode, ReplayResult, replay
from agent_replay.sandbox import SandboxPolicy, UnrecordedWrites
from agent_replay.store import open_store

POLICIES: dict[str, Policy] = {
    "v1": support_desk.policy_v1,
    "skip-lookup": support_desk.policy_skip_lookup,
}

EXIT_OK = 0
EXIT_DIVERGED = 1
EXIT_USAGE = 2


def _summary(trajectory: Trajectory) -> dict[str, object]:
    return {
        "trajectory_id": trajectory.trajectory_id,
        "status": trajectory.status.value,
        "steps": len(trajectory.steps),
        "tool_calls": [s.call.name for s in trajectory.tool_steps],
        "final_output": trajectory.final_output,
        "input_tokens": trajectory.usage.input_tokens,
        "output_tokens": trajectory.usage.output_tokens,
        "cost_usd": trajectory.usage.cost_usd,
        "fingerprint": trajectory.fingerprint(),
    }


def _replay_report(result: ReplayResult) -> dict[str, object]:
    return {
        "mode": result.mode.value,
        "identical": result.identical,
        "byte_identical": result.byte_identical,
        "unconsumed": result.unconsumed,
        "match_policy": result.policy.describe(),
        "sandbox": result.sandbox.describe(),
        "matches": result.match_summary(),
        "events": [event.to_json() for event in result.matches],
        "original": _summary(result.original),
        "replayed": _summary(result.replayed),
    }


async def _record(args: argparse.Namespace) -> int:
    agent = support_desk.build_agent(policy=POLICIES[args.policy])
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trajectory = await record_run(
        agent,
        args.task,
        clock=TickingClock(),
        tracer_provider=provider,
        capture_content=True if args.otlp_out else None,
    )
    Path(args.out).write_text(trajectory.to_json(), encoding="utf-8")
    if args.otlp_out:
        payload = spans_to_otlp_json(exporter.get_finished_spans())
        Path(args.otlp_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(_summary(trajectory), indent=2))
    return EXIT_OK


async def _replay(args: argparse.Namespace) -> int:
    recording = Trajectory.from_json(Path(args.recording).read_bytes())
    agent = support_desk.build_agent(policy=POLICIES[args.policy])
    policy = MatchPolicy(
        mode=MatchMode(args.match),
        threshold=args.threshold,
        fuzzy_writes=args.fuzzy_writes,
    )
    sandbox = SandboxPolicy(
        live_reads=args.live_reads, unrecorded_writes=UnrecordedWrites(args.unrecorded_writes)
    )
    result = await replay(
        agent,
        recording,
        mode=ReplayMode(args.mode),
        clock=TickingClock(),
        policy=policy,
        sandbox=sandbox,
    )
    if args.out:
        Path(args.out).write_text(result.replayed.to_json(), encoding="utf-8")
    print(json.dumps(_replay_report(result), indent=2))
    return EXIT_OK if result.identical else EXIT_DIVERGED


async def _demo(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    failures = 0
    for index, task in enumerate(support_desk.SAMPLE_TASKS):
        agent = support_desk.build_agent()
        recording = await record_run(
            agent, task, clock=TickingClock(), trajectory_id=f"demo-{index:02d}"
        )
        (out_dir / f"demo-{index:02d}.json").write_text(recording.to_json(), encoding="utf-8")
        full = await replay(
            support_desk.build_agent(), recording, mode=ReplayMode.FULL, clock=TickingClock()
        )
        tools = await replay(
            support_desk.build_agent(), recording, mode=ReplayMode.TOOLS, clock=TickingClock()
        )
        ok = full.identical and tools.identical
        failures += 0 if ok else 1
        rows.append(
            {
                "task": task,
                "status": recording.status.value,
                "tools": [s.call.name for s in recording.tool_steps],
                "full_replay_identical": full.identical,
                "tools_replay_identical": tools.identical,
                "cost_usd": recording.usage.cost_usd,
            }
        )
    print(json.dumps({"runs": rows, "non_deterministic": failures}, indent=2))
    return EXIT_OK if failures == 0 else EXIT_DIVERGED


async def _ingest(args: argparse.Namespace) -> int:
    """Ingest OTLP files: protobuf (``.pb``), a JSON request, or JSON lines (collector output)."""
    store = await open_store(args.database_url)
    try:
        reports = []
        for name in args.files:
            raw = Path(name).read_bytes()
            if name.endswith(".pb"):
                batches = [decode_protobuf(raw)]
            elif raw.lstrip().startswith(b"{") and b"\n{" in raw.strip():
                batches = [decode_json(line) for line in raw.splitlines() if line.strip()]
            else:
                batches = [decode_json(raw)]
            for batch in batches:
                report = await ingest_batch(store, batch)
                reports.append({"file": name, **report.model_dump()})
        print(json.dumps(reports, indent=2))
        rejected = sum(int(str(r["spans_rejected"])) for r in reports)
    finally:
        await store.close()
    return EXIT_OK if rejected == 0 else EXIT_DIVERGED


async def _stats(args: argparse.Namespace) -> int:
    store = await open_store(args.database_url)
    try:
        stats = await store.stats()
    finally:
        await store.close()
    print(json.dumps({**stats.model_dump(), "dedup_ratio": stats.dedup_ratio}, indent=2))
    return EXIT_OK


def _serve(args: argparse.Namespace) -> int:
    import uvicorn  # noqa: PLC0415 - only needed for this command

    from agent_replay.server import create_app  # noqa: PLC0415

    if args.database_url:
        os.environ["AGENT_REPLAY_DATABASE_URL"] = args.database_url
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level=args.log_level.lower())
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-replay", description=__doc__.split("\n")[0])
    parser.add_argument("--log-level", default=os.environ.get("AGENT_REPLAY_LOG_LEVEL", "WARNING"))
    sub = parser.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("record", help="run the reference agent and save its trajectory")
    rec.add_argument("--task", required=True)
    rec.add_argument("--out", required=True)
    rec.add_argument("--policy", choices=sorted(POLICIES), default="v1")
    rec.add_argument(
        "--otlp-out", help="also write the run's spans (with content) as an OTLP/JSON request"
    )

    rep = sub.add_parser("replay", help="replay a recorded trajectory")
    rep.add_argument("recording")
    rep.add_argument("--mode", choices=[m.value for m in ReplayMode], default="tools")
    rep.add_argument("--policy", choices=sorted(POLICIES), default="v1")
    rep.add_argument("--out", help="write the replayed trajectory here")
    rep.add_argument("--match", choices=[m.value for m in MatchMode], default="strict")
    rep.add_argument(
        "--threshold", type=float, default=0.9, help="minimum similarity for --match fuzzy"
    )
    rep.add_argument(
        "--fuzzy-writes",
        action="store_true",
        help="let side-effecting tools match fuzzily too (unsafe: hides argument drift)",
    )
    rep.add_argument(
        "--live-reads", action="store_true", help="run unrecorded read-only tools live"
    )
    rep.add_argument(
        "--unrecorded-writes",
        choices=[u.value for u in UnrecordedWrites],
        default="fail",
        help="fail the run, or block the call and let the agent continue",
    )

    demo = sub.add_parser("demo", help="record and replay every sample task")
    demo.add_argument("--out-dir", default=".agent-replay/demo")

    db_help = "PostgreSQL URL (default: $AGENT_REPLAY_DATABASE_URL, else in-memory)"
    ing = sub.add_parser("ingest", help="ingest OTLP trace files into the store")
    ing.add_argument("files", nargs="+")
    ing.add_argument("--database-url", help=db_help)

    stats = sub.add_parser("stats", help="show store size and blob deduplication")
    stats.add_argument("--database-url", help=db_help)

    serve = sub.add_parser("serve", help="run the OTLP/HTTP receiver and read API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=4318)
    serve.add_argument("--database-url", help=db_help)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    if args.command == "serve":
        return _serve(args)
    handlers = {
        "record": _record,
        "replay": _replay,
        "demo": _demo,
        "ingest": _ingest,
        "stats": _stats,
    }
    return asyncio.run(handlers[args.command](args))


if __name__ == "__main__":
    sys.exit(main())
