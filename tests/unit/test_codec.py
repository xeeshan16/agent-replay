from __future__ import annotations

import json

import pytest

from agent_replay.clock import TickingClock
from agent_replay.model import Trajectory
from agent_replay.recorder import record_run
from agent_replay.reference import support_desk
from agent_replay.store import codec


async def _record(task: str, trajectory_id: str) -> Trajectory:
    return await record_run(
        support_desk.build_agent(), task, clock=TickingClock(), trajectory_id=trajectory_id
    )


async def test_encode_decode_is_lossless() -> None:
    original = await _record(support_desk.SAMPLE_TASKS[1], "t1")
    encoded = codec.encode(original)
    blobs = dict(encoded.blobs)
    assert codec.decode(encoded.header, encoded.steps, blobs) == original
    assert encoded.header.step_count == len(original.steps)
    assert encoded.header.fingerprint == original.fingerprint()


async def test_messages_are_stored_once_across_turns_and_runs() -> None:
    first = codec.encode(await _record(support_desk.SAMPLE_TASKS[1], "a"))
    second = codec.encode(await _record(support_desk.SAMPLE_TASKS[1], "b"))
    # Three model calls resend the system prompt and the growing history, but each distinct
    # message is a single blob.
    message_blobs = [t for t in first.blobs.values() if json.loads(t)["t"] == "message"]
    assert len(message_blobs) == 6  # system, user, 2 x (assistant, tool)
    # A repeat of the same run adds no new content: every blob is shared.
    assert second.blobs.keys() == first.blobs.keys()
    assert second.header.content_digest != first.header.content_digest  # IDs differ


async def test_digests_cover_nested_references() -> None:
    encoded = codec.encode(await _record(support_desk.SAMPLE_TASKS[1], "t"))
    top = codec.step_digests(encoded.steps)
    nested = codec.nested_digests({d: encoded.blobs[d] for d in top})
    assert top | nested == set(encoded.blobs)


async def test_missing_or_mistyped_blobs_are_detected() -> None:
    encoded = codec.encode(await _record(support_desk.SAMPLE_TASKS[1], "t"))
    first_input = encoded.steps[0].input_digest
    missing = {d: t for d, t in encoded.blobs.items() if d != first_input}
    with pytest.raises(codec.CorruptRecordError, match="missing"):
        codec.decode(encoded.header, encoded.steps, missing)
    swapped = dict(encoded.blobs)
    swapped[first_input] = encoded.blobs[encoded.steps[0].output_digest]
    with pytest.raises(codec.CorruptRecordError, match="not a llm_request"):
        codec.decode(encoded.header, encoded.steps, swapped)


async def test_fingerprint_check_catches_tampered_content() -> None:
    encoded = codec.encode(await _record(support_desk.SAMPLE_TASKS[1], "t"))
    tool_row = next(r for r in encoded.steps if r.kind == "tool")
    tampered = dict(encoded.blobs)
    body = json.loads(tampered[tool_row.output_digest])
    body["output"] = {"tampered": True}
    tampered[tool_row.output_digest] = json.dumps(body)
    with pytest.raises(codec.CorruptRecordError, match="fingerprint"):
        codec.decode(encoded.header, encoded.steps, tampered)
