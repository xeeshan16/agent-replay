"""Command-line interface.

    agent-replay record --task "Please refund ORD-1001" --out run.json
    agent-replay replay run.json --mode tools --policy skip-lookup
    agent-replay replay run.json --match fuzzy --threshold 0.9 --live-reads
    agent-replay replay run.json --policy amount-drift --diff-out diff.md
    agent-replay diff baseline.json candidate.json --format md
    agent-replay score candidate.json --baseline baseline.json --format md
    agent-replay eval --out-dir evals/results   # the labelled regression-detection eval
    agent-replay suite record --out-dir suites/support-desk-v1
    agent-replay gate --baseline suites/support-desk-v1 --policy v2 --junit gate.xml
    agent-replay demo --out-dir runs/
    agent-replay record --task "..." --out run.json --otlp-out spans.json
    agent-replay ingest spans.json            # into AGENT_REPLAY_DATABASE_URL, or memory
    agent-replay serve --port 4318            # OTLP/HTTP receiver + read API
    agent-replay stats

``replay`` and ``diff`` exit 0 when the runs are behaviourally identical and 1 when they
diverged; ``score`` exits 0 on pass, 1 on fail and 3 when the judge could not decide; ``gate``
exits 0 when every threshold held and 1 on a breach. All of them can be used directly in CI.
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

from agent_replay.agent import ToolCallingAgent
from agent_replay.clock import TickingClock
from agent_replay.diff import TrajectoryDiff, diff_trajectories
from agent_replay.gate import GateThresholds, ReplaySettings, run_gate
from agent_replay.gate_report import render_junit
from agent_replay.gate_report import render_markdown as render_gate_markdown
from agent_replay.ingest.export import spans_to_otlp_json
from agent_replay.ingest.service import ingest_batch
from agent_replay.ingest.spans import decode_json, decode_protobuf
from agent_replay.logs import configure_logging
from agent_replay.matching import MatchMode, MatchPolicy
from agent_replay.model import Trajectory
from agent_replay.normalize import Normalizer
from agent_replay.recorder import record_run
from agent_replay.reference import support_desk
from agent_replay.replay import ReplayMode, ReplayResult, replay
from agent_replay.report import render_markdown, render_scorecard
from agent_replay.sandbox import SandboxPolicy, UnrecordedWrites
from agent_replay.scoring import ScoringSpec, Verdict, score
from agent_replay.store import open_store
from agent_replay.suite import load_suite, record_suite, write_suite

POLICIES = support_desk.POLICIES
NORMALIZERS: dict[str, Normalizer] = {
    "none": Normalizer.none(),
    "default": Normalizer.default(),
    "support-desk": support_desk.NORMALIZER,
}

EXIT_OK = 0
EXIT_DIVERGED = 1
EXIT_USAGE = 2
EXIT_INCONCLUSIVE = 3


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


def _first_divergence(diff: TrajectoryDiff) -> dict[str, object] | None:
    first = diff.first_divergence
    return None if first is None else first.model_dump(mode="json", exclude_defaults=True)


def _replay_report(result: ReplayResult, diff: TrajectoryDiff) -> dict[str, object]:
    return {
        "mode": result.mode.value,
        "identical": result.identical,
        "byte_identical": result.byte_identical,
        "unconsumed": result.unconsumed,
        "match_policy": result.policy.describe(),
        "sandbox": result.sandbox.describe(),
        "matches": result.match_summary(),
        "events": [event.to_json() for event in result.matches],
        "first_divergence": _first_divergence(diff),
        "divergences": diff.reason_counts(),
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
        normalizer=support_desk.NORMALIZER,
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
    diff = diff_trajectories(
        recording, result.replayed, normalizer=policy.normalizer, candidate_events=result.matches
    )
    if args.out:
        Path(args.out).write_text(result.replayed.to_json(), encoding="utf-8")
    if args.diff_out:
        title = f"Replay ({result.mode.value} mode, policy {args.policy})"
        Path(args.diff_out).write_text(render_markdown(diff, title=title), encoding="utf-8")
    print(json.dumps(_replay_report(result, diff), indent=2))
    return EXIT_OK if result.identical else EXIT_DIVERGED


async def _diff(args: argparse.Namespace) -> int:
    baseline = Trajectory.from_json(Path(args.baseline).read_bytes())
    candidate = Trajectory.from_json(Path(args.candidate).read_bytes())
    diff = diff_trajectories(baseline, candidate, normalizer=NORMALIZERS[args.normalizer])
    text = render_markdown(diff) if args.format == "md" else json.dumps(diff.to_report(), indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(text)
    return EXIT_OK if diff.identical else EXIT_DIVERGED


def _load_spec(path: str | None) -> ScoringSpec:
    return support_desk.scoring_spec() if path is None else ScoringSpec.from_toml(path)


async def _score(args: argparse.Namespace) -> int:
    candidate = Trajectory.from_json(Path(args.trajectory).read_bytes())
    baseline = Trajectory.from_json(Path(args.baseline).read_bytes()) if args.baseline else None
    spec = _load_spec(args.spec)
    if args.no_judge:
        spec = spec.model_copy(update={"judge": spec.judge.model_copy(update={"enabled": False})})
    card = await score(candidate, spec, baseline=baseline, judge=support_desk.build_judge())
    text = render_scorecard(card) if args.format == "md" else json.dumps(card.to_report(), indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(text)
    return {Verdict.PASS: EXIT_OK, Verdict.FAIL: EXIT_DIVERGED}.get(card.verdict, EXIT_INCONCLUSIVE)


async def _eval(args: argparse.Namespace) -> int:
    from agent_replay import evaluation  # noqa: PLC0415 - only needed for this command

    cases = evaluation.load_cases(args.dataset)
    report = await evaluation.run_eval(cases, _load_spec(args.spec), dataset=args.dataset)
    text = evaluation.render_markdown(report)
    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(
            json.dumps(report.to_json(), indent=2) + "\n", encoding="utf-8"
        )
        (out / "report.md").write_text(text, encoding="utf-8")
    print(text)
    return EXIT_OK


def _write(path: str | None, text: str) -> None:
    if path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")


async def _suite_record(args: argparse.Namespace) -> int:
    tasks = (
        [t for t in Path(args.tasks).read_text(encoding="utf-8").splitlines() if t.strip()]
        if args.tasks
        else list(support_desk.SAMPLE_TASKS)
    )
    policy = POLICIES[args.policy]
    suite = await record_suite(
        args.name,
        tasks,
        lambda case_id: support_desk.build_agent(
            policy=policy, id_factory=support_desk.seeded_ids(f"{args.name}/{case_id}")
        ),
        description=args.description,
        metadata={"policy": args.policy, "model": support_desk.MODEL},
    )
    root = write_suite(suite, args.out_dir)
    print(
        json.dumps(
            {
                "suite": suite.name,
                "path": str(root),
                "cases": [c.model_dump(exclude_none=True) for c in suite.manifest.cases],
            },
            indent=2,
        )
    )
    return EXIT_OK


def _thresholds(args: argparse.Namespace) -> GateThresholds:
    thresholds = GateThresholds.from_toml(args.config) if args.config else GateThresholds()
    overrides = {
        name: getattr(args, name)
        for name in ("max_success_drop", "min_success_rate", "max_cost_ratio")
        if getattr(args, name) is not None
    }
    return GateThresholds.model_validate({**thresholds.model_dump(), **overrides})


async def _gate(args: argparse.Namespace) -> int:
    suite = load_suite(args.baseline)
    spec = _load_spec(args.spec)
    judge = None if args.no_judge else support_desk.build_judge()
    policy = POLICIES[args.policy]

    def candidate(case_id: str, repeat: int) -> ToolCallingAgent:
        return support_desk.build_agent(
            policy=policy, id_factory=support_desk.seeded_ids(f"{case_id}/{repeat}")
        )

    report = await run_gate(
        suite,
        candidate,
        spec,
        match=support_desk.match_policy(),
        thresholds=_thresholds(args),
        judge=judge,
        repeats=args.repeats,
        concurrency=args.concurrency,
        settings=ReplaySettings(),
        candidate=f"support-desk policy {args.policy}",
        provider=f"fake (agent {support_desk.MODEL}"
        + ("" if judge is None else f", judge {spec.judge.model}")
        + "); no network",
    )
    markdown = render_gate_markdown(report)
    _write(args.markdown, markdown)
    _write(args.junit, render_junit(report))
    _write(args.json, json.dumps(report.to_json(), indent=2) + "\n")
    print(markdown)
    return EXIT_OK if report.passed else EXIT_DIVERGED


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
        policy = support_desk.match_policy()
        full = await replay(
            support_desk.build_agent(),
            recording,
            mode=ReplayMode.FULL,
            clock=TickingClock(),
            policy=policy,
        )
        tools = await replay(
            support_desk.build_agent(),
            recording,
            mode=ReplayMode.TOOLS,
            clock=TickingClock(),
            policy=policy,
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
                "tools_replay_matches": tools.match_summary(),
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


def _add_suite_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    suite = sub.add_parser("suite", help="manage baseline suites")
    suite_sub = suite.add_subparsers(dest="suite_command", required=True)
    srec = suite_sub.add_parser("record", help="record a baseline suite from a task list")
    srec.add_argument("--out-dir", required=True)
    srec.add_argument("--name", default="support-desk-v1")
    srec.add_argument("--description", default="")
    srec.add_argument("--tasks", help="file with one task per line (default: the sample tasks)")
    srec.add_argument("--policy", choices=sorted(POLICIES), default="v1")


def _add_gate_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    gate = sub.add_parser("gate", help="gate a candidate against a baseline suite")
    gate.add_argument("--baseline", required=True, help="suite directory (with suite.json)")
    gate.add_argument("--policy", choices=sorted(POLICIES), default="v1", help="candidate")
    gate.add_argument("--repeats", type=int, default=1, help="replays per case")
    gate.add_argument("--concurrency", type=int, default=4)
    gate.add_argument("--spec", help="scoring spec TOML (default: the support-desk spec)")
    gate.add_argument("--config", help="TOML file with a [gate] table of thresholds")
    gate.add_argument("--max-success-drop", type=float)
    gate.add_argument("--min-success-rate", type=float)
    gate.add_argument("--max-cost-ratio", type=float)
    gate.add_argument("--no-judge", action="store_true", help="skip the judge layer")
    gate.add_argument("--markdown", help="write the Markdown report here")
    gate.add_argument("--junit", help="write a JUnit XML report here")
    gate.add_argument("--json", help="write the full JSON report here")


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
    rep.add_argument("--diff-out", help="write a Markdown diff against the recording here")

    dif = sub.add_parser("diff", help="align two trajectories and explain the differences")
    dif.add_argument("baseline")
    dif.add_argument("candidate")
    dif.add_argument("--format", choices=["md", "json"], default="md")
    dif.add_argument("--normalizer", choices=sorted(NORMALIZERS), default="support-desk")
    dif.add_argument("--out", help="write the diff here instead of standard output")

    sco = sub.add_parser("score", help="score a run: checks, invariants, then the judge")
    sco.add_argument("trajectory")
    sco.add_argument("--baseline", help="baseline run, for cost-regression checks")
    sco.add_argument("--spec", help="scoring spec TOML (default: the support-desk spec)")
    sco.add_argument("--no-judge", action="store_true", help="skip the judge layer")
    sco.add_argument("--format", choices=["md", "json"], default="md")
    sco.add_argument("--out", help="write the scorecard here instead of standard output")

    ev = sub.add_parser("eval", help="run the labelled regression-detection eval")
    ev.add_argument("--dataset", default="evals/regressions.jsonl")
    ev.add_argument("--spec", help="scoring spec TOML (default: the support-desk spec)")
    ev.add_argument("--out-dir", help="also write report.json and report.md here")

    _add_suite_parser(sub)
    _add_gate_parser(sub)

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
    if args.command == "suite":
        suite_handlers = {"record": _suite_record}
        return asyncio.run(suite_handlers[args.suite_command](args))
    handlers = {
        "record": _record,
        "replay": _replay,
        "diff": _diff,
        "score": _score,
        "eval": _eval,
        "gate": _gate,
        "demo": _demo,
        "ingest": _ingest,
        "stats": _stats,
    }
    return asyncio.run(handlers[args.command](args))


if __name__ == "__main__":
    sys.exit(main())
