"""Command-line interface.

    agent-replay record --task "Please refund ORD-1001" --out run.json
    agent-replay replay run.json --mode tools --policy skip-lookup
    agent-replay demo --out-dir runs/

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

from agent_replay.clock import TickingClock
from agent_replay.logs import configure_logging
from agent_replay.model import Trajectory
from agent_replay.providers.fake import Policy
from agent_replay.recorder import record_run
from agent_replay.reference import support_desk
from agent_replay.replay import ReplayMode, ReplayResult, replay

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
        "unconsumed": result.unconsumed,
        "original": _summary(result.original),
        "replayed": _summary(result.replayed),
    }


async def _record(args: argparse.Namespace) -> int:
    agent = support_desk.build_agent(policy=POLICIES[args.policy])
    trajectory = await record_run(agent, args.task, clock=TickingClock())
    Path(args.out).write_text(trajectory.to_json(), encoding="utf-8")
    print(json.dumps(_summary(trajectory), indent=2))
    return EXIT_OK


async def _replay(args: argparse.Namespace) -> int:
    recording = Trajectory.from_json(Path(args.recording).read_bytes())
    agent = support_desk.build_agent(policy=POLICIES[args.policy])
    result = await replay(agent, recording, mode=ReplayMode(args.mode), clock=TickingClock())
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-replay", description=__doc__.split("\n")[0])
    parser.add_argument("--log-level", default=os.environ.get("AGENT_REPLAY_LOG_LEVEL", "WARNING"))
    sub = parser.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("record", help="run the reference agent and save its trajectory")
    rec.add_argument("--task", required=True)
    rec.add_argument("--out", required=True)
    rec.add_argument("--policy", choices=sorted(POLICIES), default="v1")

    rep = sub.add_parser("replay", help="replay a recorded trajectory")
    rep.add_argument("recording")
    rep.add_argument("--mode", choices=[m.value for m in ReplayMode], default="tools")
    rep.add_argument("--policy", choices=sorted(POLICIES), default="v1")
    rep.add_argument("--out", help="write the replayed trajectory here")

    demo = sub.add_parser("demo", help="record and replay every sample task")
    demo.add_argument("--out-dir", default=".agent-replay/demo")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    handlers = {"record": _record, "replay": _replay, "demo": _demo}
    return asyncio.run(handlers[args.command](args))


if __name__ == "__main__":
    sys.exit(main())
