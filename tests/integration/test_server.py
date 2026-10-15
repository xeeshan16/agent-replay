from __future__ import annotations

import gzip
import json
from collections.abc import Iterator, Sequence

import pytest
from fastapi.testclient import TestClient
from opentelemetry.proto.collector.trace.v1 import trace_service_pb2
from sqlalchemy.exc import OperationalError

from agent_replay import server
from agent_replay.ingest.export import spans_to_otlp_json
from agent_replay.ingest.spans import SpanRecord
from agent_replay.store import InMemoryStore
from agent_replay.store.base import AddSpansResult
from tests.helpers import otlp_bytes, record_with_spans

pytestmark = pytest.mark.integration
PB = {"Content-Type": "application/x-protobuf"}
JSON = {"Content-Type": "application/json"}


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(server.create_app(InMemoryStore(), ingest_token="")) as c:
        yield c


async def test_protobuf_export_is_ingested_and_readable(client: TestClient) -> None:
    recorded, spans = await record_with_spans()
    response = client.post("/v1/traces", content=otlp_bytes(spans), headers=PB)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/x-protobuf"
    parsed = trace_service_pb2.ExportTraceServiceResponse.FromString(response.content)
    assert parsed.partial_success.rejected_spans == 0
    assert "created=1" in response.headers["x-agent-replay-ingest"]

    [summary] = client.get("/v1/trajectories").json()
    detail = client.get(f"/v1/trajectories/{summary['trajectory_id']}").json()
    assert detail["task_input"] == recorded.task_input
    assert [s["kind"] for s in detail["steps"]] == ["llm", "tool", "llm", "tool", "llm"]
    assert client.get("/v1/trajectories/nope").status_code == 404


async def test_gzipped_json_with_a_bad_span_is_a_partial_success(client: TestClient) -> None:
    _, spans = await record_with_spans()
    payload = spans_to_otlp_json(spans)
    payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["spanId"] = "zz"
    body = gzip.compress(json.dumps(payload).encode())
    response = client.post("/v1/traces", content=body, headers={**JSON, "Content-Encoding": "gzip"})
    assert response.status_code == 200
    partial = response.json()["partialSuccess"]
    assert partial["rejectedSpans"] == "1"
    assert "invalid traceId/spanId" in partial["errorMessage"]


@pytest.mark.parametrize(
    ("headers", "body", "status"),
    [
        ({"Content-Type": "text/plain"}, b"x", 415),
        (JSON, b"{not json", 400),
        (PB, b"\xff\xff\xff", 400),
        ({**JSON, "Content-Encoding": "gzip"}, b"not gzip", 400),
    ],
)
def test_bad_requests_get_otlp_error_codes(
    client: TestClient, headers: dict[str, str], body: bytes, status: int
) -> None:
    response = client.post("/v1/traces", content=body, headers=headers)
    assert response.status_code == status


def test_protobuf_errors_are_encoded_as_rpc_status(client: TestClient) -> None:
    response = client.post("/v1/traces", content=b"\xff\xff\xff", headers=PB)
    assert response.headers["content-type"] == "application/x-protobuf"
    assert response.content.startswith(b"\x08\x03\x12")  # code=INVALID_ARGUMENT, then message


def test_oversized_and_gzip_bomb_bodies_are_refused(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "MAX_BODY_BYTES", 1024)
    assert client.post("/v1/traces", content=b"{}" * 1024, headers=JSON).status_code == 413
    bomb = gzip.compress(b" " * 100_000)  # ~100 bytes on the wire, 100 kB inflated
    assert len(bomb) < 1024
    response = client.post("/v1/traces", content=bomb, headers={**JSON, "Content-Encoding": "gzip"})
    assert response.status_code == 413


def test_bearer_token_is_enforced_when_configured() -> None:
    app = server.create_app(InMemoryStore(), ingest_token="s3cret")
    with TestClient(app) as c:
        assert c.post("/v1/traces", content=b"{}", headers=JSON).status_code == 401
        wrong = {**JSON, "Authorization": "Bearer nope"}
        assert c.post("/v1/traces", content=b"{}", headers=wrong).status_code == 401
        right = {**JSON, "Authorization": "Bearer s3cret"}
        assert c.post("/v1/traces", content=b"{}", headers=right).status_code == 200
        assert c.get("/healthz").status_code == 200  # health stays open


class _DownStore(InMemoryStore):
    async def add_spans(self, spans: Sequence[SpanRecord]) -> AddSpansResult:
        raise OperationalError("INSERT", {}, ConnectionRefusedError("db down"))


async def test_store_outage_returns_retryable_503() -> None:
    _, spans = await record_with_spans()
    with TestClient(server.create_app(_DownStore(), ingest_token="")) as c:
        response = c.post("/v1/traces", content=otlp_bytes(spans), headers=PB)
    assert response.status_code == 503
    assert response.headers["retry-after"] == str(server.RETRY_AFTER_S)
