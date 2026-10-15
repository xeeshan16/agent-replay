"""Shared test helpers: record the reference agent and capture its spans as OTLP records."""

from __future__ import annotations

from collections.abc import Sequence

from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agent_replay.ingest.spans import SpanRecord, decode_protobuf
from agent_replay.model import Trajectory
from agent_replay.providers.fake import Policy
from agent_replay.recorder import record_run
from agent_replay.reference import support_desk

REFUND_TASK = "I'd like a refund for ORD-1001, it arrived damaged."


async def record_with_spans(
    task: str = REFUND_TASK,
    *,
    policy: Policy = support_desk.policy_v1,
    capture_content: bool = True,
) -> tuple[Trajectory, tuple[ReadableSpan, ...]]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trajectory = await record_run(
        support_desk.build_agent(policy=policy),
        task,
        tracer_provider=provider,
        capture_content=capture_content,
    )
    return trajectory, exporter.get_finished_spans()


def otlp_bytes(spans: Sequence[ReadableSpan]) -> bytes:
    return encode_spans(spans).SerializeToString()


def to_records(spans: Sequence[ReadableSpan]) -> list[SpanRecord]:
    batch = decode_protobuf(otlp_bytes(spans))
    assert batch.rejected == 0
    return batch.spans
