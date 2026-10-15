from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agent_replay.store import InMemoryStore, TrajectoryStore
from agent_replay.store.postgres import PostgresStore

_EXPORTER = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_EXPORTER))
trace.set_tracer_provider(_provider)


@pytest.fixture
def spans() -> Iterator[InMemorySpanExporter]:
    _EXPORTER.clear()
    yield _EXPORTER
    _EXPORTER.clear()


DATABASE_URL_ENV = "AGENT_REPLAY_TEST_DATABASE_URL"


def _postgres_url() -> str:
    url = os.environ.get(DATABASE_URL_ENV)
    if not url:
        if os.environ.get("AGENT_REPLAY_REQUIRE_POSTGRES") == "1":
            pytest.fail(f"{DATABASE_URL_ENV} must be set when AGENT_REPLAY_REQUIRE_POSTGRES=1")
        pytest.skip(f"set {DATABASE_URL_ENV} to run PostgreSQL tests")
    return url


@pytest.fixture
async def pg_store() -> AsyncIterator[PostgresStore]:
    store = PostgresStore.from_url(_postgres_url())
    await store.migrate()
    await store.truncate()
    try:
        yield store
    finally:
        await store.close()


@pytest.fixture(params=["memory", pytest.param("postgres", marks=pytest.mark.postgres)])
async def store(request: pytest.FixtureRequest) -> AsyncIterator[TrajectoryStore]:
    """Runs a test against every store implementation (PostgreSQL only when configured)."""
    if request.param == "memory":
        yield InMemoryStore()
        return
    pg = PostgresStore.from_url(_postgres_url())
    await pg.migrate()
    await pg.truncate()
    try:
        yield pg
    finally:
        await pg.close()
