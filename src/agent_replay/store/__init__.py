"""Trajectory storage: an interface, an in-memory implementation and a PostgreSQL one."""

from __future__ import annotations

import os

from agent_replay.store.base import (
    AddSpansResult,
    PruneResult,
    SaveOutcome,
    SaveResult,
    StoreStats,
    SuiteSummary,
    SweepResult,
    TrajectoryStore,
    TrajectorySummary,
)
from agent_replay.store.memory import InMemoryStore

DATABASE_URL_ENV = "AGENT_REPLAY_DATABASE_URL"

__all__ = [
    "DATABASE_URL_ENV",
    "AddSpansResult",
    "InMemoryStore",
    "PruneResult",
    "SaveOutcome",
    "SaveResult",
    "StoreStats",
    "SuiteSummary",
    "SweepResult",
    "TrajectoryStore",
    "TrajectorySummary",
    "open_store",
]


async def open_store(url: str | None = None) -> TrajectoryStore:
    """Open the configured store: PostgreSQL when a URL is given or set in the environment.

    The PostgreSQL import is deferred so the in-memory path does not need the driver loaded.
    """
    url = url or os.environ.get(DATABASE_URL_ENV)
    if not url:
        return InMemoryStore()
    from agent_replay.store.postgres import PostgresStore  # noqa: PLC0415

    store = PostgresStore.from_url(url)
    await store.migrate()
    return store
