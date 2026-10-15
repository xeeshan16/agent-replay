"""Behaviour every ``TrajectoryStore`` must have; runs against memory and PostgreSQL."""

from __future__ import annotations

import pytest

from agent_replay.clock import TickingClock
from agent_replay.ingest.assemble import assemble_trace
from agent_replay.model import Trajectory
from agent_replay.recorder import record_run
from agent_replay.reference import support_desk
from agent_replay.store import SaveOutcome, TrajectoryStore
from tests.helpers import record_with_spans, to_records

pytestmark = pytest.mark.integration


async def _record(task: str, trajectory_id: str) -> Trajectory:
    return await record_run(
        support_desk.build_agent(), task, clock=TickingClock(), trajectory_id=trajectory_id
    )


async def test_save_get_round_trip_and_idempotency(store: TrajectoryStore) -> None:
    trajectory = await _record(support_desk.SAMPLE_TASKS[1], "run-1")
    first = await store.save(trajectory)
    again = await store.save(trajectory)
    assert (first.outcome, first.revision) == (SaveOutcome.CREATED, 1)
    assert (again.outcome, again.revision) == (SaveOutcome.UNCHANGED, 1)
    assert await store.get("run-1") == trajectory
    assert await store.get("missing") is None


async def test_changed_content_bumps_the_revision(store: TrajectoryStore) -> None:
    trajectory = await _record(support_desk.SAMPLE_TASKS[1], "run-1")
    await store.save(trajectory)
    changed = trajectory.model_copy(update={"steps": trajectory.steps[:2], "final_output": None})
    result = await store.save(changed)
    assert (result.outcome, result.revision) == (SaveOutcome.UPDATED, 2)
    assert await store.get("run-1") == changed


async def test_listing_and_stats(store: TrajectoryStore) -> None:
    for index, task in enumerate(support_desk.SAMPLE_TASKS):
        await store.save(await _record(task, f"run-{index}"))
    listed = await store.list_trajectories(agent_name="support-desk", limit=3)
    assert len(listed) == 3
    assert await store.list_trajectories(agent_name="nobody") == []
    stats = await store.stats()
    assert stats.trajectories == len(support_desk.SAMPLE_TASKS)
    # Shared system prompts, tool definitions and outputs are stored once.
    assert stats.blob_bytes < stats.logical_bytes
    assert stats.dedup_ratio > 1


async def test_span_buffer_deduplicates_and_rebuilds(store: TrajectoryStore) -> None:
    recorded, spans = await record_with_spans()
    records = to_records(spans)
    trace_id = records[0].trace_id

    children = records[:-1]  # the agent (root) span ends, and is exported, last
    added = await store.add_spans(children)
    assert (added.inserted, added.duplicates, added.traces) == (5, 0, frozenset({trace_id}))
    assert await store.rebuild_trace(trace_id, assemble_trace) == []  # root not here yet
    assert await store.known_traces([trace_id, "f" * 32]) == {trace_id}

    redelivered = await store.add_spans(records)
    assert (redelivered.inserted, redelivered.duplicates) == (1, 5)
    [result] = await store.rebuild_trace(trace_id, assemble_trace)
    assert result.outcome is SaveOutcome.CREATED
    stored = await store.get(result.trajectory_id)
    assert stored is not None
    assert stored.fingerprint() == recorded.fingerprint()

    [again] = await store.rebuild_trace(trace_id, assemble_trace)
    assert again.outcome is SaveOutcome.UNCHANGED
