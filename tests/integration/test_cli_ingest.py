from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_replay.cli import EXIT_OK, main
from tests.conftest import _postgres_url

pytestmark = pytest.mark.integration
TASK = "I'd like a refund for ORD-1001, it arrived damaged."


def _record(tmp_path: Path) -> Path:
    spans = tmp_path / "spans.json"
    code = main(
        ["record", "--task", TASK, "--out", str(tmp_path / "run.json"), "--otlp-out", str(spans)]
    )
    assert code == EXIT_OK
    return spans


def test_record_otlp_out_then_ingest_in_memory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AGENT_REPLAY_DATABASE_URL", raising=False)
    spans = _record(tmp_path)
    payload = json.loads(spans.read_text())
    first = payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    assert len(first["traceId"]) == 32  # hex, as OTLP/JSON requires
    capsys.readouterr()

    # The same request twice as JSON lines, the way a collector file exporter writes it.
    lines = tmp_path / "spans.jsonl"
    compact = json.dumps(payload)
    lines.write_text(f"{compact}\n{compact}\n")
    assert main(["ingest", str(lines)]) == EXIT_OK
    reports = json.loads(capsys.readouterr().out)
    assert [len(r["created"]) for r in reports] == [1, 0]
    assert [r["spans_duplicate"] for r in reports] == [0, 6]


@pytest.mark.postgres
def test_ingest_and_stats_against_postgres(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    url = _postgres_url()
    spans = _record(tmp_path)
    assert main(["ingest", str(spans), "--database-url", url]) == EXIT_OK
    assert main(["stats", "--database-url", url]) == EXIT_OK
    out = capsys.readouterr().out
    stats = json.loads(out[out.rindex("{\n") :])
    assert stats["trajectories"] >= 1
