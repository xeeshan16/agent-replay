from __future__ import annotations

import base64
import json

import pytest
from opentelemetry.proto.collector.trace.v1 import trace_service_pb2

from agent_replay.ingest.export import spans_to_otlp_json
from agent_replay.ingest.spans import (
    OtlpDecodeError,
    SpanStatus,
    decode_json,
    decode_protobuf,
    export_response,
)
from tests.helpers import otlp_bytes, record_with_spans

TRACE = "5b8efff798038103d269b633813fc60c"
SPAN = "eee19b7ec3c1b174"


def _request(*spans: dict[str, object]) -> bytes:
    return json.dumps(
        {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [{"key": "service.name", "value": {"stringValue": "svc"}}]
                    },
                    "scopeSpans": [{"spans": list(spans)}],
                }
            ]
        }
    ).encode()


def _span(**overrides: object) -> dict[str, object]:
    span: dict[str, object] = {
        "traceId": TRACE,
        "spanId": SPAN,
        "name": "execute_tool lookup",
        "startTimeUnixNano": "1700000000000000000",
        "endTimeUnixNano": "1700000000500000000",
        "attributes": [],
    }
    span.update(overrides)
    return span


def test_json_decodes_every_any_value_kind() -> None:
    attributes = [
        {"key": "s", "value": {"stringValue": "x"}},
        {"key": "b", "value": {"boolValue": True}},
        {"key": "i", "value": {"intValue": "42"}},
        {"key": "d", "value": {"doubleValue": 1.5}},
        {"key": "a", "value": {"arrayValue": {"values": [{"stringValue": "stop"}]}}},
        {
            "key": "kv",
            "value": {"kvlistValue": {"values": [{"key": "k", "value": {"intValue": 1}}]}},
        },
        {"key": "empty", "value": {}},
    ]
    batch = decode_json(
        _request(_span(attributes=attributes, status={"code": 2, "message": "boom"}))
    )
    [span] = batch.spans
    assert span.attributes == {
        "s": "x",
        "b": True,
        "i": 42,
        "d": 1.5,
        "a": ["stop"],
        "kv": {"k": 1},
        "empty": None,
    }
    assert span.status is SpanStatus.ERROR
    assert span.status_message == "boom"
    assert span.service_name == "svc"
    assert span.operation == "execute_tool"  # derived from the span name


def test_json_accepts_base64_ids_and_enum_status_names() -> None:
    b64_trace = base64.b64encode(bytes.fromhex(TRACE)).decode()
    batch = decode_json(_request(_span(traceId=b64_trace, status={"code": "STATUS_CODE_OK"})))
    assert batch.spans[0].trace_id == TRACE
    assert batch.spans[0].status is SpanStatus.OK


@pytest.mark.parametrize(
    "bad",
    [
        {"traceId": "nothex"},
        {"traceId": "0" * 32},
        {"spanId": "abc"},
        {"startTimeUnixNano": "not-a-number"},
    ],
)
def test_bad_spans_are_rejected_individually(bad: dict[str, object]) -> None:
    good = _span(spanId="1111111111111111")
    batch = decode_json(_request(good, _span(**bad)))
    assert len(batch.spans) == 1
    assert batch.rejected == 1
    assert batch.errors


@pytest.mark.parametrize("body", [b"not json", b"[]", b"\xff\xfe"])
def test_undecodable_json_payload_raises(body: bytes) -> None:
    with pytest.raises(OtlpDecodeError):
        decode_json(body)


def test_invalid_protobuf_raises() -> None:
    with pytest.raises(OtlpDecodeError):
        decode_protobuf(b"\xff\xff\xff\xff")


async def test_protobuf_and_json_decode_to_the_same_records() -> None:
    _, spans = await record_with_spans()
    from_pb = decode_protobuf(otlp_bytes(spans)).spans
    assert [s.operation for s in from_pb] == [
        "chat",
        "execute_tool",
        "chat",
        "execute_tool",
        "chat",
        "invoke_agent",
    ]
    from_json = decode_json(json.dumps(spans_to_otlp_json(spans)).encode()).spans
    assert from_json == from_pb


def test_export_response_reports_partial_success_in_both_encodings() -> None:
    body, media = export_response(2, "bad ids", protobuf=False)
    assert media == "application/json"
    assert json.loads(body) == {"partialSuccess": {"rejectedSpans": "2", "errorMessage": "bad ids"}}
    raw, media = export_response(2, "bad ids", protobuf=True)
    parsed = trace_service_pb2.ExportTraceServiceResponse.FromString(raw)
    assert media == "application/x-protobuf"
    assert parsed.partial_success.rejected_spans == 2
    assert export_response(0, "", protobuf=False)[0] == b"{}"
