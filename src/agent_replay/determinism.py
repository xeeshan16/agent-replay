"""Replay determinism across processes: does a recording replay identically somewhere else?

A replay that only reproduces inside the process that recorded it is worth little: CI replays
yesterday's recording in a fresh interpreter, at a different wall-clock time, with a different
hash seed. This harness measures exactly that. Each *pair* runs two worker processes one after
the other, with different ``PYTHONHASHSEED`` values and at least one wall-clock second apart
(the agent stamps the time into its system prompt at second resolution):

1. the *record* worker runs the reference agent live on every sample task, with the system
   clock and random IDs, and writes each trajectory plus its OTLP/JSON spans (content on);
2. the *replay* worker runs the same tasks live once more, then replays the first worker's
   recordings in ``FULL`` and ``TOOLS`` mode under each normaliser, and rebuilds every
   recording from its OTLP spans alone.

Three rates come out of it, each counted over tasks x pairs:

* ``live``: two independent live runs have the same normalised fingerprint. This is what
  "re-run the agent and compare" gives you without a cassette.
* ``replay``: the replayed run is identical to the recording (``ReplayResult.identical``).
* ``otlp``: the trajectory assembled from the exported spans has the recording's exact
  fingerprint, and it replays identically in ``FULL`` mode.

Nothing is mocked inside the workers: the clock, the ID factories and the hash seed are the
real, process-local ones. The fake model is deterministic by construction, so every difference
measured here comes from the volatile values a real agent also carries (timestamps, minted IDs,
idempotency keys, set ordering).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel, ConfigDict, JsonValue

from agent_replay.clock import SystemClock
from agent_replay.ingest.assemble import assemble_trace
from agent_replay.ingest.export import spans_to_otlp_json
from agent_replay.ingest.spans import decode_json
from agent_replay.matching import MatchMode, MatchPolicy
from agent_replay.model import Trajectory
from agent_replay.normalize import Normalizer
from agent_replay.recorder import record_run
from agent_replay.reference import support_desk as desk
from agent_replay.replay import ReplayMode, normalized_fingerprint, replay
from agent_replay.report import cell

NORMALIZERS: dict[str, Normalizer] = {
    "none": Normalizer.none(),
    "default": Normalizer.default(),
    "support-desk": desk.NORMALIZER,
}
MODES: tuple[ReplayMode, ...] = (ReplayMode.FULL, ReplayMode.TOOLS)
#: The configuration a user of this agent would run; the CLI fails unless it is fully stable.
REQUIRED_NORMALIZER = "support-desk"
WORKER_TIMEOUT_S = 120.0


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PairResult(_Frozen):
    """Counts of identical runs for one record/replay process pair."""

    pair: int
    record_seed: int
    replay_seed: int
    tasks: int
    live: dict[str, int]  # normaliser -> live re-runs with the same normalised fingerprint
    replay: dict[str, dict[str, int]]  # normaliser -> mode -> identical replays
    otlp_fingerprint: int  # assembled from spans with the recording's exact fingerprint
    otlp_replay: int  # assembled trajectory replays identically (FULL, agent normaliser)


# --- workers -----------------------------------------------------------------------------------


async def _record_live(task: str) -> tuple[Trajectory, dict[str, JsonValue]]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trajectory = await record_run(
        desk.build_agent(), task, clock=SystemClock(), tracer_provider=provider,
        capture_content=True,
    )  # fmt: skip
    return trajectory, spans_to_otlp_json(exporter.get_finished_spans())


async def _record_all() -> list[tuple[Trajectory, dict[str, JsonValue]]]:
    return [await _record_live(task) for task in desk.SAMPLE_TASKS]


def _worker_record(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for index, (trajectory, otlp) in enumerate(asyncio.run(_record_all())):
        (out_dir / f"{index:02d}.json").write_text(trajectory.to_json(), encoding="utf-8")
        (out_dir / f"{index:02d}.otlp.json").write_text(json.dumps(otlp), encoding="utf-8")


def _policy(normalizer: Normalizer) -> MatchPolicy:
    return MatchPolicy(mode=MatchMode.STRICT, normalizer=normalizer)


def _load_recordings(rec_dir: Path) -> list[tuple[Trajectory, bytes]]:
    return [
        (
            Trajectory.from_json((rec_dir / f"{index:02d}.json").read_bytes()),
            (rec_dir / f"{index:02d}.otlp.json").read_bytes(),
        )
        for index in range(len(desk.SAMPLE_TASKS))
    ]


async def _worker_replay(recordings: Sequence[tuple[Trajectory, bytes]]) -> dict[str, JsonValue]:
    live = dict.fromkeys(NORMALIZERS, 0)
    replayed = {name: dict.fromkeys((m.value for m in MODES), 0) for name in NORMALIZERS}
    otlp_fingerprint = otlp_replay = 0
    for task, (recording, otlp_body) in zip(desk.SAMPLE_TASKS, recordings, strict=True):
        again, _ = await _record_live(task)
        for name, normalizer in NORMALIZERS.items():
            same = normalized_fingerprint(recording, normalizer) == normalized_fingerprint(
                again, normalizer
            )
            live[name] += same
            for mode in MODES:
                result = await replay(
                    desk.build_agent(), recording, mode=mode, clock=SystemClock(),
                    policy=_policy(normalizer),
                )  # fmt: skip
                replayed[name][mode.value] += result.identical
        batch = decode_json(otlp_body)
        assembled = assemble_trace(batch.spans)
        if len(assembled) == 1 and assembled[0].fingerprint() == recording.fingerprint():
            otlp_fingerprint += 1
            result = await replay(
                desk.build_agent(), assembled[0], mode=ReplayMode.FULL, clock=SystemClock(),
                policy=_policy(NORMALIZERS[REQUIRED_NORMALIZER]),
            )  # fmt: skip
            otlp_replay += result.identical
    return {
        "tasks": len(desk.SAMPLE_TASKS),
        "live": dict(live),
        "replay": {name: dict(counts) for name, counts in replayed.items()},
        "otlp_fingerprint": otlp_fingerprint,
        "otlp_replay": otlp_replay,
    }


def _worker_main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent_replay.determinism")
    parser.add_argument("role", choices=["record", "replay"])
    parser.add_argument("directory")
    args = parser.parse_args(argv)
    if args.role == "record":
        _worker_record(Path(args.directory))
    else:
        recordings = _load_recordings(Path(args.directory))
        print(json.dumps(asyncio.run(_worker_replay(recordings))))
    return 0


# --- orchestration -----------------------------------------------------------------------------


def _run_worker(role: str, directory: Path, seed: int) -> str:
    env = {**os.environ, "PYTHONHASHSEED": str(seed)}
    completed = subprocess.run(  # noqa: S603 - argv is this interpreter and this module
        [sys.executable, "-m", "agent_replay.determinism", role, str(directory)],
        env=env,
        capture_output=True,
        text=True,
        timeout=WORKER_TIMEOUT_S,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{role} worker failed ({completed.returncode}): {completed.stderr}")
    return completed.stdout


def _next_second() -> None:
    """Sleep past the next wall-clock second so the two workers never share a timestamp."""
    time.sleep(1.0 - (time.time() % 1.0) + 0.01)


def run_pair(pair: int, work_dir: Path) -> PairResult:
    record_seed, replay_seed = 2 * pair + 1, 2 * pair + 2  # 0 would disable hash randomisation
    rec_dir = work_dir / f"pair-{pair:02d}"
    _run_worker("record", rec_dir, record_seed)
    _next_second()
    raw = json.loads(_run_worker("replay", rec_dir, replay_seed))
    return PairResult(pair=pair, record_seed=record_seed, replay_seed=replay_seed, **raw)


class DeterminismReport(_Frozen):
    provider: str
    settings: dict[str, JsonValue]
    pairs: tuple[PairResult, ...]

    @property
    def runs(self) -> int:
        return sum(p.tasks for p in self.pairs)

    def live(self, normalizer: str) -> int:
        return sum(p.live[normalizer] for p in self.pairs)

    def replayed(self, normalizer: str, mode: ReplayMode) -> int:
        return sum(p.replay[normalizer][mode.value] for p in self.pairs)

    @property
    def otlp(self) -> tuple[int, int]:
        return sum(p.otlp_fingerprint for p in self.pairs), sum(p.otlp_replay for p in self.pairs)

    @property
    def stable(self) -> bool:
        """Every replay under the agent's own normaliser, and every OTLP rebuild, is identical."""
        fingerprints, replays = self.otlp
        return (
            all(self.replayed(REQUIRED_NORMALIZER, m) == self.runs for m in MODES)
            and fingerprints == self.runs
            and replays == self.runs
        )

    def to_json(self) -> dict[str, JsonValue]:
        fingerprints, replays = self.otlp
        return {
            "provider": self.provider,
            "settings": self.settings,
            "runs": self.runs,
            "live": {n: self.live(n) for n in NORMALIZERS},
            "replay": {n: {m.value: self.replayed(n, m) for m in MODES} for n in NORMALIZERS},
            "otlp": {"fingerprint": fingerprints, "replay_full": replays},
            "stable": self.stable,
            "pairs": [p.model_dump(mode="json") for p in self.pairs],
        }


def run_determinism(pairs: int, *, work_dir: Path | None = None) -> DeterminismReport:
    if pairs < 1:
        raise ValueError("pairs must be at least 1")
    with TemporaryDirectory(prefix="agent-replay-determinism-") as tmp:
        base = work_dir or Path(tmp)
        results = tuple(run_pair(pair, base) for pair in range(pairs))
    return DeterminismReport(
        provider=f"fake (agent {desk.MODEL}); no network",
        settings={
            "tasks": len(desk.SAMPLE_TASKS),
            "pairs": pairs,
            "processes": "record and replay in separate interpreters, different PYTHONHASHSEED, "
            "at least 1 s apart",
            "match": "strict",
            "otlp": "content capture on, OTLP/JSON, FULL replay with the support-desk normaliser",
        },
        pairs=results,
    )


def _rate(hits: int, total: int) -> str:
    return f"{hits}/{total} ({hits / total:.0%})" if total else "n/a"


def render_markdown(report: DeterminismReport) -> str:
    total = report.runs
    lines = [
        "### Replay determinism across processes",
        "",
        f"{len(report.pairs)} process pairs x {report.settings['tasks']} tasks = {total} runs per "
        f"cell. Provider: {cell(report.provider, 120)}.",
        "",
        "| Normaliser | Live re-run | FULL replay | TOOLS replay |",
        "|------------|-------------|-------------|--------------|",
    ]
    for name in NORMALIZERS:
        lines.append(
            f"| {name} | {_rate(report.live(name), total)} "
            f"| {_rate(report.replayed(name, ReplayMode.FULL), total)} "
            f"| {_rate(report.replayed(name, ReplayMode.TOOLS), total)} |"
        )
    fingerprints, replays = report.otlp
    lines.extend(
        [
            "",
            f"OTLP round trip: {_rate(fingerprints, total)} rebuilt with the exact fingerprint, "
            f"{_rate(replays, total)} replayed identically in FULL mode.",
            "",
            f"Stable under the `{REQUIRED_NORMALIZER}` normaliser: "
            f"**{'yes' if report.stable else 'no'}**.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(_worker_main(sys.argv[1:]))
