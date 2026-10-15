"""The real OpenTelemetry OTLP/HTTP exporter talking to a real receiver over a socket."""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator

import httpx
import pytest
import uvicorn
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from agent_replay.model import Trajectory
from agent_replay.recorder import record_run
from agent_replay.reference import support_desk
from agent_replay.replay import ReplayMode, replay
from agent_replay.server import create_app
from agent_replay.store import InMemoryStore

pytestmark = pytest.mark.integration


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port: int = sock.getsockname()[1]
        return port


@pytest.fixture
def receiver() -> Iterator[str]:
    port = _free_port()
    config = uvicorn.Config(
        create_app(InMemoryStore(), ingest_token="t0ken"), port=port, log_level="error"
    )
    srv = uvicorn.Server(config)
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 10
    while not srv.started:
        if time.monotonic() > deadline:
            pytest.fail("receiver did not start")
        time.sleep(0.02)
    try:
        yield base
    finally:
        srv.should_exit = True
        thread.join(timeout=5)


async def test_exported_runs_are_stored_and_replay_exactly(receiver: str) -> None:
    provider = TracerProvider()
    exporter = OTLPSpanExporter(
        endpoint=f"{receiver}/v1/traces", headers={"Authorization": "Bearer t0ken"}, timeout=5
    )
    provider.add_span_processor(BatchSpanProcessor(exporter, max_export_batch_size=4))
    recorded = [
        await record_run(
            support_desk.build_agent(), task, tracer_provider=provider, capture_content=True
        )
        for task in support_desk.SAMPLE_TASKS
    ]
    assert provider.force_flush(timeout_millis=10_000)
    provider.shutdown()

    async with httpx.AsyncClient(base_url=receiver) as client:
        listing = (await client.get("/v1/trajectories")).json()
        assert len(listing) == len(recorded)
        stored = [
            Trajectory.model_validate(
                (await client.get(f"/v1/trajectories/{s['trajectory_id']}")).json()
            )
            for s in listing
        ]

    by_recorded_id = {t.metadata["recorded_trajectory_id"]: t for t in stored}
    for original in recorded:
        ingested = by_recorded_id[original.trajectory_id]
        assert ingested.fingerprint() == original.fingerprint()
        result = await replay(support_desk.build_agent(), ingested, mode=ReplayMode.FULL)
        assert result.identical
