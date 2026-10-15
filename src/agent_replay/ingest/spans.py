"""Decode OTLP trace exports (HTTP/JSON and HTTP/protobuf) into flat span records.

Both encodings decode into the same ``SpanRecord`` so everything downstream (buffering,
assembly, storage) is encoding-agnostic. Decoding is strict about structure but tolerant per
span: a span with malformed IDs is rejected and counted, the rest of the batch is accepted, and
the receiver reports the count back as an OTLP partial success.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from google.protobuf.message import DecodeError
from opentelemetry.proto.collector.trace.v1 import trace_service_pb2
from opentelemetry.proto.common.v1 import common_pb2
from pydantic import BaseModel, ConfigDict, JsonValue

from agent_replay import telemetry as tm

AttrValue = JsonValue

_TRACE_ID = re.compile(r"^[0-9a-f]{32}$")
_SPAN_ID = re.compile(r"^[0-9a-f]{16}$")
_ZERO_TRACE = "0" * 32
_ZERO_SPAN = "0" * 16

_MAX_ERRORS = 5

GENAI_OPERATIONS = frozenset({"invoke_agent", "chat", "text_completion", "execute_tool"})


class OtlpDecodeError(ValueError):
    """The payload as a whole could not be decoded."""


class SpanStatus(IntEnum):
    UNSET = 0
    OK = 1
    ERROR = 2


class SpanRecord(BaseModel):
    """One span, flattened. IDs are lowercase hex; times are Unix nanoseconds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    name: str
    start_ns: int
    end_ns: int
    attributes: dict[str, AttrValue] = {}
    status: SpanStatus = SpanStatus.UNSET
    status_message: str = ""
    service_name: str | None = None

    @property
    def operation(self) -> str | None:
        """GenAI operation name, falling back to the span-name prefix used by the conventions."""
        value = self.attributes.get(tm.OPERATION_NAME)
        if isinstance(value, str):
            return value
        head = self.name.split(" ", 1)[0]
        return head if head in GENAI_OPERATIONS else None

    @property
    def is_genai(self) -> bool:
        return self.operation in GENAI_OPERATIONS

    def attr_str(self, key: str) -> str | None:
        value = self.attributes.get(key)
        return value if isinstance(value, str) else None

    def attr_int(self, key: str) -> int | None:
        value = self.attributes.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    def attr_float(self, key: str) -> float | None:
        value = self.attributes.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        return float(value)


@dataclass(slots=True)
class DecodedBatch:
    spans: list[SpanRecord] = field(default_factory=list)
    rejected: int = 0
    errors: list[str] = field(default_factory=list)

    def reject(self, reason: str) -> None:
        self.rejected += 1
        if len(self.errors) < _MAX_ERRORS:  # keep the partial-success message bounded
            self.errors.append(reason)


# --- JSON -----------------------------------------------------------------------------------


def _json_any(value: Mapping[str, Any]) -> AttrValue:  # noqa: PLR0911 - one branch per type
    """Decode an OTLP ``AnyValue`` in its proto3 JSON mapping."""
    if "stringValue" in value:
        return str(value["stringValue"])
    if "boolValue" in value:
        return bool(value["boolValue"])
    if "intValue" in value:  # int64 is a JSON string in proto3, but accept numbers too
        return int(value["intValue"])
    if "doubleValue" in value:
        return float(value["doubleValue"])
    if "arrayValue" in value:
        return [_json_any(v) for v in value["arrayValue"].get("values", [])]
    if "kvlistValue" in value:
        return _json_attrs(value["kvlistValue"].get("values", []))
    if "bytesValue" in value:
        return str(value["bytesValue"])  # kept base64-encoded
    return None


def _json_attrs(items: Iterable[Mapping[str, Any]]) -> dict[str, AttrValue]:
    return {str(kv["key"]): _json_any(kv.get("value") or {}) for kv in items}


def _json_id(raw: object, pattern: re.Pattern[str]) -> str | None:
    """OTLP/JSON encodes IDs as hex; accept base64 too, which some exporters emit."""
    if not isinstance(raw, str) or not raw:
        return None
    lowered = raw.lower()
    if pattern.match(lowered):
        return lowered
    try:
        decoded = base64.b64decode(raw, validate=True).hex()
    except (binascii.Error, ValueError):
        return None
    return decoded if pattern.match(decoded) else None


def decode_json(body: bytes) -> DecodedBatch:
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise OtlpDecodeError(f"invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise OtlpDecodeError("payload must be an ExportTraceServiceRequest object")
    batch = DecodedBatch()
    for resource_spans in payload.get("resourceSpans") or []:
        resource = _json_attrs((resource_spans.get("resource") or {}).get("attributes") or [])
        service = resource.get("service.name")
        for scope_spans in resource_spans.get("scopeSpans") or []:
            for raw in scope_spans.get("spans") or []:
                _append_json_span(batch, raw, service if isinstance(service, str) else None)
    return batch


def _append_json_span(batch: DecodedBatch, raw: Mapping[str, Any], service: str | None) -> None:
    trace_id = _json_id(raw.get("traceId"), _TRACE_ID)
    span_id = _json_id(raw.get("spanId"), _SPAN_ID)
    if trace_id is None or span_id is None or trace_id == _ZERO_TRACE or span_id == _ZERO_SPAN:
        batch.reject(f"span {raw.get('name')!r}: invalid traceId/spanId")
        return
    parent = _json_id(raw.get("parentSpanId"), _SPAN_ID)
    status = raw.get("status") or {}
    try:
        batch.spans.append(
            SpanRecord(
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=parent if parent != _ZERO_SPAN else None,
                name=str(raw.get("name") or ""),
                start_ns=int(raw.get("startTimeUnixNano") or 0),
                end_ns=int(raw.get("endTimeUnixNano") or 0),
                attributes=_json_attrs(raw.get("attributes") or []),
                status=_json_status(status.get("code")),
                status_message=str(status.get("message") or ""),
                service_name=service,
            )
        )
    except (TypeError, ValueError, KeyError) as exc:
        batch.reject(f"span {raw.get('name')!r}: {exc}")


def _json_status(code: object) -> SpanStatus:
    if isinstance(code, str):  # proto3 JSON allows enum names
        return {
            "STATUS_CODE_OK": SpanStatus.OK,
            "STATUS_CODE_ERROR": SpanStatus.ERROR,
        }.get(code, SpanStatus.UNSET)
    if isinstance(code, int) and code in (1, 2):
        return SpanStatus(code)
    return SpanStatus.UNSET


# --- protobuf -------------------------------------------------------------------------------


def _pb_any(value: common_pb2.AnyValue) -> AttrValue:  # noqa: PLR0911 - one branch per type
    kind = value.WhichOneof("value")
    if kind == "string_value":
        return value.string_value
    if kind == "bool_value":
        return value.bool_value
    if kind == "int_value":
        return value.int_value
    if kind == "double_value":
        return value.double_value
    if kind == "array_value":
        return [_pb_any(v) for v in value.array_value.values]
    if kind == "kvlist_value":
        return {kv.key: _pb_any(kv.value) for kv in value.kvlist_value.values}
    if kind == "bytes_value":
        return base64.b64encode(value.bytes_value).decode("ascii")
    return None


def decode_protobuf(body: bytes) -> DecodedBatch:
    request = trace_service_pb2.ExportTraceServiceRequest()
    try:
        request.ParseFromString(body)
    except DecodeError as exc:
        raise OtlpDecodeError(f"invalid protobuf: {exc}") from exc
    batch = DecodedBatch()
    for resource_spans in request.resource_spans:
        resource = {kv.key: _pb_any(kv.value) for kv in resource_spans.resource.attributes}
        service = resource.get("service.name")
        for scope_spans in resource_spans.scope_spans:
            for span in scope_spans.spans:
                trace_id, span_id = span.trace_id.hex(), span.span_id.hex()
                if not _TRACE_ID.match(trace_id) or not _SPAN_ID.match(span_id):
                    batch.reject(f"span {span.name!r}: invalid trace_id/span_id length")
                    continue
                if trace_id == _ZERO_TRACE or span_id == _ZERO_SPAN:
                    batch.reject(f"span {span.name!r}: all-zero trace_id/span_id")
                    continue
                parent = span.parent_span_id.hex()
                batch.spans.append(
                    SpanRecord(
                        trace_id=trace_id,
                        span_id=span_id,
                        parent_span_id=parent if _SPAN_ID.match(parent) else None,
                        name=span.name,
                        start_ns=span.start_time_unix_nano,
                        end_ns=span.end_time_unix_nano,
                        attributes={kv.key: _pb_any(kv.value) for kv in span.attributes},
                        status=_json_status(int(span.status.code)),
                        status_message=span.status.message,
                        service_name=service if isinstance(service, str) else None,
                    )
                )
    return batch


def export_response(rejected: int, message: str, *, protobuf: bool) -> tuple[bytes, str]:
    """Encode an ``ExportTraceServiceResponse`` in the request's encoding."""
    response = trace_service_pb2.ExportTraceServiceResponse()
    if rejected:
        response.partial_success.rejected_spans = rejected
        response.partial_success.error_message = message
    if protobuf:
        return response.SerializeToString(), "application/x-protobuf"
    body: dict[str, Any] = {}
    if rejected:
        body["partialSuccess"] = {"rejectedSpans": str(rejected), "errorMessage": message}
    return json.dumps(body).encode(), "application/json"
