"""OTLP trace ingest: decode exported spans and assemble them into trajectories."""

from agent_replay.ingest.assemble import assemble_trace, trajectory_id_for
from agent_replay.ingest.spans import (
    DecodedBatch,
    OtlpDecodeError,
    SpanRecord,
    SpanStatus,
    decode_json,
    decode_protobuf,
)

__all__ = [
    "DecodedBatch",
    "OtlpDecodeError",
    "SpanRecord",
    "SpanStatus",
    "assemble_trace",
    "decode_json",
    "decode_protobuf",
    "trajectory_id_for",
]
