"""HTTP service: an OTLP/HTTP trace receiver plus a small read API.

``POST /v1/traces`` implements the OTLP/HTTP trace endpoint for both encodings
(``application/x-protobuf`` and ``application/json``), with optional gzip. It follows the OTLP
response contract so standard exporters behave correctly:

* 200 with an ``ExportTraceServiceResponse`` (in the request's encoding), carrying
  ``partial_success`` when individual spans were rejected;
* 400 for undecodable payloads (exporters do not retry these);
* 503 with ``Retry-After`` when the store is unavailable (exporters retry; ingest is idempotent,
  so a retried batch cannot duplicate data);
* 401, 413 and 415 for auth, size and content-type problems.

The read API (``/v1/trajectories``, ``/v1/suites``, ``/v1/diff``) serves stored data, which may
include prompts and tool outputs. Set ``AGENT_REPLAY_READ_TOKEN`` to require a bearer token for
it; the ingest token is separate so a span exporter's credential cannot read anything back.
"""

from __future__ import annotations

import hmac
import os
import zlib
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import JsonValue
from sqlalchemy.exc import DBAPIError, InterfaceError, OperationalError

from agent_replay import __version__
from agent_replay.diff import diff_trajectories
from agent_replay.ingest.service import IngestReport, ingest_batch
from agent_replay.ingest.spans import (
    DecodedBatch,
    OtlpDecodeError,
    decode_json,
    decode_protobuf,
    export_response,
)
from agent_replay.logs import get_logger
from agent_replay.model import Trajectory
from agent_replay.normalize import Normalizer
from agent_replay.report import render_markdown
from agent_replay.store import SuiteSummary, TrajectoryStore, TrajectorySummary, open_store
from agent_replay.suite import SuiteManifest

INGEST_TOKEN_ENV = "AGENT_REPLAY_INGEST_TOKEN"  # noqa: S105 - the variable name, not a secret
READ_TOKEN_ENV = "AGENT_REPLAY_READ_TOKEN"  # noqa: S105 - the variable name, not a secret
MAX_BODY_BYTES = 8 * 1024 * 1024
RETRY_AFTER_S = 5

_log = get_logger("server")
_PROTOBUF = "application/x-protobuf"
_JSON = "application/json"


class PayloadTooLargeError(Exception):
    pass


async def _read_body(request: Request, limit: int) -> bytes:
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > limit:
        raise PayloadTooLargeError
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise PayloadTooLargeError
        chunks.append(chunk)
    return b"".join(chunks)


def _gunzip(body: bytes, limit: int) -> bytes:
    """Decompress with a hard output cap so a small gzip bomb cannot exhaust memory."""
    inflater = zlib.decompressobj(wbits=16 + zlib.MAX_WBITS)
    try:
        data = inflater.decompress(body, limit + 1)
    except zlib.error as exc:
        raise OtlpDecodeError(f"invalid gzip body: {exc}") from exc
    if len(data) > limit or inflater.unconsumed_tail:
        raise PayloadTooLargeError
    return data


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _rpc_status(code: int, message: str) -> bytes:
    """Serialise ``google.rpc.Status{code, message}`` without depending on googleapis protos."""
    encoded = message.encode("utf-8")
    return b"\x08" + _varint(code) + b"\x12" + _varint(len(encoded)) + encoded


# OTLP asks for google.rpc.Status bodies; codes per google.rpc.Code.
_RPC_CODES = {400: 3, 401: 16, 413: 8, 415: 3, 503: 14}


def _otlp_error(status: int, message: str, *, protobuf: bool) -> Response:
    """Error response in the request's encoding, as the OTLP/HTTP spec requires."""
    code = _RPC_CODES.get(status, 2)
    if protobuf:
        return Response(_rpc_status(code, message), status_code=status, media_type=_PROTOBUF)
    return JSONResponse({"code": code, "message": message}, status_code=status)


def _unavailable(protobuf: bool) -> Response:
    response = _otlp_error(503, "store unavailable, retry later", protobuf=protobuf)
    response.headers["Retry-After"] = str(RETRY_AFTER_S)
    return response


def _get_store(request: Request) -> TrajectoryStore:
    resolved: TrajectoryStore = request.app.state.store
    return resolved


StoreDep = Annotated[TrajectoryStore, Depends(_get_store)]


def _bearer_check(token: str | None) -> Callable[[str | None], None]:
    def check(authorization: Annotated[str | None, Header()] = None) -> None:
        if not token:
            return
        supplied = (authorization or "").removeprefix("Bearer ").strip()
        if not hmac.compare_digest(supplied.encode(), token.encode()):
            raise HTTPException(status_code=401, detail="invalid or missing bearer token")

    return check


def create_app(
    store: TrajectoryStore | None = None,
    *,
    ingest_token: str | None = None,
    read_token: str | None = None,
    normalizer: Normalizer | None = None,
) -> FastAPI:
    """Build the app. Without ``store`` it opens one from the environment on startup.

    ``normalizer`` masks volatile values before ``/v1/diff`` aligns two runs (default: the
    built-in timestamp, UUID and call-ID masks).
    """
    token = ingest_token if ingest_token is not None else os.environ.get(INGEST_TOKEN_ENV)
    reader = read_token if read_token is not None else os.environ.get(READ_TOKEN_ENV)
    diff_normalizer = normalizer or Normalizer.default()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned = store is None
        app.state.store = store if store is not None else await open_store()
        try:
            yield
        finally:
            if owned:
                await app.state.store.close()

    app = FastAPI(title="agent-replay", version=__version__, lifespan=lifespan)

    check_token = _bearer_check(token)
    read = [Depends(_bearer_check(reader))]

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/traces", dependencies=[Depends(check_token)])
    async def receive_traces(request: Request, store: StoreDep) -> Response:
        content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
        protobuf = content_type == _PROTOBUF
        if content_type not in (_PROTOBUF, _JSON):
            return _otlp_error(415, f"unsupported content type {content_type!r}", protobuf=False)
        try:
            body = await _read_body(request, MAX_BODY_BYTES)
            if request.headers.get("content-encoding", "").lower() == "gzip":
                body = _gunzip(body, MAX_BODY_BYTES)
            batch: DecodedBatch = decode_protobuf(body) if protobuf else decode_json(body)
        except PayloadTooLargeError:
            return _otlp_error(413, f"payload exceeds {MAX_BODY_BYTES} bytes", protobuf=protobuf)
        except OtlpDecodeError as exc:
            return _otlp_error(400, str(exc), protobuf=protobuf)

        try:
            report = await ingest_batch(store, batch)
        except (OSError, TimeoutError, OperationalError, InterfaceError) as exc:
            # The store is unreachable or timed out: tell the exporter to retry later.
            _log.warning("store unavailable", extra={"ctx": {"error": type(exc).__name__}})
            return _unavailable(protobuf)
        except DBAPIError as exc:
            if not exc.connection_invalidated:
                raise
            _log.warning("store connection lost", extra={"ctx": {"error": type(exc).__name__}})
            return _unavailable(protobuf)

        message = "; ".join(batch.errors)
        payload, media_type = export_response(report.spans_rejected, message, protobuf=protobuf)
        headers = {"X-Agent-Replay-Ingest": _report_header(report)}
        return Response(content=payload, media_type=media_type, headers=headers)

    app.include_router(_read_router(diff_normalizer), dependencies=read)
    return app


def _read_router(diff_normalizer: Normalizer) -> APIRouter:
    router = APIRouter()

    @router.get("/v1/trajectories")
    async def list_trajectories(
        store: StoreDep,
        agent: str | None = None,
        limit: Annotated[int, Query(ge=1, le=500)] = 50,
    ) -> list[TrajectorySummary]:
        return await store.list_trajectories(agent_name=agent, limit=limit)

    # ``:path`` because trajectory IDs may contain slashes (suite cases are ``suite/case``).
    @router.get("/v1/trajectories/{trajectory_id:path}")
    async def get_trajectory(trajectory_id: str, store: StoreDep) -> Trajectory:
        return await _require(store, trajectory_id)

    @router.get("/v1/suites")
    async def list_suites(store: StoreDep) -> list[SuiteSummary]:
        return await store.list_suites()

    @router.get("/v1/suites/{name}")
    async def get_suite(name: str, store: StoreDep) -> SuiteManifest:
        manifest = await store.get_suite(name)
        if manifest is None:
            raise HTTPException(status_code=404, detail="suite not found")
        return manifest

    @router.get("/v1/diff", response_model=None)
    async def diff(
        store: StoreDep,
        baseline: str,
        candidate: str,
        format: Annotated[str, Query(pattern="^(json|md)$")] = "json",
    ) -> dict[str, JsonValue] | PlainTextResponse:
        """Align two stored trajectories and explain the first divergence."""
        result = diff_trajectories(
            await _require(store, baseline),
            await _require(store, candidate),
            normalizer=diff_normalizer,
        )
        if format == "md":
            return PlainTextResponse(render_markdown(result), media_type="text/markdown")
        return result.to_report()

    return router


async def _require(store: TrajectoryStore, trajectory_id: str) -> Trajectory:
    trajectory = await store.get(trajectory_id)
    if trajectory is None:
        raise HTTPException(status_code=404, detail=f"trajectory {trajectory_id!r} not found")
    return trajectory


def _report_header(report: IngestReport) -> str:
    return (
        f"inserted={report.spans_inserted};duplicate={report.spans_duplicate};"
        f"ignored={report.spans_ignored};expired={report.spans_expired};created={len(report.created)};"
        f"updated={len(report.updated)};pending={report.pending_traces}"
    )
