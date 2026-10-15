"""Encode finished SDK spans as an OTLP/JSON ``ExportTraceServiceRequest``.

Used by ``agent-replay record --otlp-out`` to produce the same payload an OTLP/HTTP JSON exporter
would send, so recordings can be ingested offline (``agent-replay ingest``) or fed to a receiver.
Trace and span IDs are written as lowercase hex, as the OTLP/JSON mapping requires, rather than
the base64 that the generic protobuf JSON printer would produce.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from google.protobuf.json_format import MessageToDict
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.sdk.trace import ReadableSpan

_ID_FIELDS = ("traceId", "spanId", "parentSpanId")


def spans_to_otlp_json(spans: Sequence[ReadableSpan]) -> dict[str, Any]:
    request = encode_spans(spans)
    payload: dict[str, Any] = MessageToDict(request)
    pb_spans = [
        span
        for resource_spans in request.resource_spans
        for scope_spans in resource_spans.scope_spans
        for span in scope_spans.spans
    ]
    json_spans = [
        span
        for resource_spans in payload.get("resourceSpans", [])
        for scope_spans in resource_spans.get("scopeSpans", [])
        for span in scope_spans.get("spans", [])
    ]
    for pb_span, json_span in zip(pb_spans, json_spans, strict=True):
        json_span["traceId"] = pb_span.trace_id.hex()
        json_span["spanId"] = pb_span.span_id.hex()
        if pb_span.parent_span_id:
            json_span["parentSpanId"] = pb_span.parent_span_id.hex()
    return payload
