"""Split trajectories into a header, step rows and content-addressed blobs, and back.

Agent traffic is extremely repetitive: every model call resends the system prompt, the tool
definitions and the whole conversation so far, and the same tool outputs recur across runs. The
codec therefore stores each *message* and each *tool definition* as its own blob, and an LLM
request as a small blob that references them by digest. A ten-turn conversation then costs ten
message blobs instead of 1 + 2 + ... + 10 copies, and a system prompt shared by thousands of runs
is stored once.

Blobs are canonical JSON text, addressed by the SHA-256 of that text, so writes are idempotent
(insert-if-absent) and any blob can be verified against its key. The codec is pure and shared by
every store implementation.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic import JsonValue

from agent_replay.canonical import canonical_json, fingerprint
from agent_replay.model import (
    LLMRequest,
    LLMResponse,
    LLMStep,
    Message,
    RunStatus,
    Step,
    ToolCall,
    ToolResult,
    ToolSpec,
    ToolStep,
    Trajectory,
)

DIGEST_LENGTH = 64


class CorruptRecordError(ValueError):
    """Stored rows cannot be turned back into a trajectory (missing or mismatched blob)."""


def blob_digest(text: str) -> str:
    return fingerprint("blob", text, length=DIGEST_LENGTH)


@dataclass(frozen=True, slots=True)
class StepRow:
    index: int
    kind: str
    name: str
    started_at: datetime
    duration_ms: float
    input_digest: str
    output_digest: str
    call_id: str | None
    side_effect: bool
    is_error: bool


@dataclass(frozen=True, slots=True)
class Header:
    trajectory_id: str
    schema_version: int
    agent_name: str
    task_input: str
    final_output: str | None
    status: str
    error: str | None
    started_at: datetime
    duration_ms: float
    metadata: dict[str, JsonValue]
    source: str
    trace_id: str | None
    step_count: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    fingerprint: str
    content_digest: str
    logical_bytes: int


@dataclass(frozen=True, slots=True)
class EncodedTrajectory:
    header: Header
    steps: tuple[StepRow, ...]
    blobs: dict[str, str]  # digest -> canonical JSON text


class _BlobSink:
    def __init__(self) -> None:
        self.blobs: dict[str, str] = {}

    def put(self, body: Mapping[str, Any]) -> str:
        text = canonical_json(body)
        digest = blob_digest(text)
        self.blobs[digest] = text
        return digest


def _encode_step(step: Step, sink: _BlobSink) -> StepRow:
    if isinstance(step, LLMStep):
        request = step.request
        input_digest = sink.put(
            {
                "t": "llm_request",
                "model": request.model,
                "temperature": request.temperature,
                "messages": [
                    sink.put({"t": "message", **m.model_dump(mode="json")})
                    for m in request.messages
                ],
                "tools": [
                    sink.put({"t": "tool_spec", **s.model_dump(mode="json")}) for s in request.tools
                ],
            }
        )
        output_digest = sink.put({"t": "llm_response", **step.response.model_dump(mode="json")})
        return StepRow(
            index=step.index,
            kind=step.kind,
            name=request.model,
            started_at=step.started_at,
            duration_ms=step.duration_ms,
            input_digest=input_digest,
            output_digest=output_digest,
            call_id=None,
            side_effect=False,
            is_error=False,
        )
    call, result = step.call, step.result
    if result.tool_call_id != call.id or result.name != call.name:
        raise ValueError(f"step {step.index}: tool result does not belong to its call")
    return StepRow(
        index=step.index,
        kind=step.kind,
        name=call.name,
        started_at=step.started_at,
        duration_ms=step.duration_ms,
        # The call ID is per-run noise; keeping it out of the blob lets equal calls dedupe.
        input_digest=sink.put({"t": "tool_call", "name": call.name, "arguments": call.arguments}),
        output_digest=sink.put(
            {
                "t": "tool_result",
                "output": result.output,
                "is_error": result.is_error,
                "error_type": result.error_type,
            }
        ),
        call_id=call.id,
        side_effect=step.side_effect,
        is_error=result.is_error,
    )


def encode(trajectory: Trajectory) -> EncodedTrajectory:
    sink = _BlobSink()
    steps = tuple(_encode_step(step, sink) for step in trajectory.steps)
    document = trajectory.model_dump_json()
    usage = trajectory.usage
    source = trajectory.metadata.get("source")
    trace_id = trajectory.metadata.get("trace_id")
    header = Header(
        trajectory_id=trajectory.trajectory_id,
        schema_version=trajectory.schema_version,
        agent_name=trajectory.agent_name,
        task_input=trajectory.task_input,
        final_output=trajectory.final_output,
        status=trajectory.status.value,
        error=trajectory.error,
        started_at=trajectory.started_at,
        duration_ms=trajectory.duration_ms,
        metadata=dict(trajectory.metadata),
        source=source if isinstance(source, str) else "recorder",
        trace_id=trace_id if isinstance(trace_id, str) else None,
        step_count=len(steps),
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost_usd=usage.cost_usd,
        fingerprint=trajectory.fingerprint(),
        content_digest=fingerprint(json.loads(document), length=DIGEST_LENGTH),
        logical_bytes=len(document.encode("utf-8")),
    )
    return EncodedTrajectory(header=header, steps=steps, blobs=sink.blobs)


# --- decoding -------------------------------------------------------------------------------


def step_digests(steps: Iterable[StepRow]) -> set[str]:
    return {d for row in steps for d in (row.input_digest, row.output_digest)}


def nested_digests(bodies: Mapping[str, str]) -> set[str]:
    """Digests referenced from inside already-fetched blobs (messages and tool specs)."""
    found: set[str] = set()
    for text in bodies.values():
        body = json.loads(text)
        if isinstance(body, dict) and body.get("t") == "llm_request":
            found.update(body["messages"])
            found.update(body["tools"])
    return found


class _BlobReader:
    def __init__(self, blobs: Mapping[str, str]) -> None:
        self._blobs = blobs

    def get(self, digest: str, expected_type: str) -> dict[str, Any]:
        text = self._blobs.get(digest)
        if text is None:
            raise CorruptRecordError(f"blob {digest} is missing")
        body = json.loads(text)
        if not isinstance(body, dict) or body.get("t") != expected_type:
            raise CorruptRecordError(f"blob {digest} is not a {expected_type}")
        body.pop("t")
        return body


def _decode_step(row: StepRow, reader: _BlobReader) -> Step:
    if row.kind == "llm":
        request = reader.get(row.input_digest, "llm_request")
        return LLMStep(
            index=row.index,
            started_at=row.started_at,
            duration_ms=row.duration_ms,
            request=LLMRequest(
                model=request["model"],
                temperature=request["temperature"],
                messages=tuple(
                    Message.model_validate(reader.get(d, "message")) for d in request["messages"]
                ),
                tools=tuple(
                    ToolSpec.model_validate(reader.get(d, "tool_spec")) for d in request["tools"]
                ),
            ),
            response=LLMResponse.model_validate(reader.get(row.output_digest, "llm_response")),
        )
    if row.kind == "tool":
        if row.call_id is None:
            raise CorruptRecordError(f"tool step {row.index} has no call ID")
        call = reader.get(row.input_digest, "tool_call")
        result = reader.get(row.output_digest, "tool_result")
        return ToolStep(
            index=row.index,
            started_at=row.started_at,
            duration_ms=row.duration_ms,
            call=ToolCall(id=row.call_id, name=call["name"], arguments=call["arguments"]),
            result=ToolResult(tool_call_id=row.call_id, name=call["name"], **result),
            side_effect=row.side_effect,
        )
    raise CorruptRecordError(f"unknown step kind {row.kind!r}")


def decode(header: Header, steps: Iterable[StepRow], blobs: Mapping[str, str]) -> Trajectory:
    reader = _BlobReader(blobs)
    trajectory = Trajectory(
        schema_version=header.schema_version,
        trajectory_id=header.trajectory_id,
        agent_name=header.agent_name,
        task_input=header.task_input,
        final_output=header.final_output,
        status=RunStatus(header.status),
        error=header.error,
        started_at=header.started_at,
        duration_ms=header.duration_ms,
        steps=tuple(_decode_step(row, reader) for row in sorted(steps, key=lambda r: r.index)),
        metadata=header.metadata,
    )
    if trajectory.fingerprint() != header.fingerprint:
        raise CorruptRecordError(f"trajectory {header.trajectory_id} fails its fingerprint check")
    return trajectory
