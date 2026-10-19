from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent_replay import cli, server
from agent_replay.reference import support_desk as desk
from agent_replay.replay import replay
from agent_replay.sandbox import SandboxPolicy, UnrecordedWrites
from agent_replay.store import InMemoryStore
from agent_replay.suite import record_suite

pytestmark = pytest.mark.integration


async def _seeded_store() -> InMemoryStore:
    store = InMemoryStore()
    suite = await record_suite(
        "desk", desk.SAMPLE_TASKS[1:3], lambda c: desk.build_agent(id_factory=desk.seeded_ids(c))
    )
    for _, trajectory in suite.items():
        await store.save(trajectory)
    await store.save_suite(suite.manifest)
    _, baseline = suite.items()[0]
    drifted = await replay(
        desk.build_agent(policy=desk.policy_amount_drift),
        baseline,
        policy=desk.match_policy(),
        sandbox=SandboxPolicy(live_reads=True, unrecorded_writes=UnrecordedWrites.BLOCK),
    )
    await store.save(drifted.replayed.model_copy(update={"trajectory_id": "candidate-1"}))
    return store


async def test_suites_trajectories_and_diffs_are_readable() -> None:
    store = await _seeded_store()
    app = server.create_app(store, ingest_token="", read_token="", normalizer=desk.NORMALIZER)
    with TestClient(app) as client:
        [summary] = client.get("/v1/suites").json()
        assert (summary["name"], summary["case_count"]) == ("desk", 2)
        manifest = client.get("/v1/suites/desk").json()
        baseline_id = manifest["cases"][0]["trajectory_id"]
        assert baseline_id == "desk/00-i-d-like-a-refund"
        assert client.get("/v1/suites/nope").status_code == 404

        # Trajectory IDs with slashes resolve.
        assert client.get(f"/v1/trajectories/{baseline_id}").json()["trajectory_id"] == baseline_id

        same = client.get("/v1/diff", params={"baseline": baseline_id, "candidate": baseline_id})
        assert same.json()["identical"] is True
        params = {"baseline": baseline_id, "candidate": "candidate-1"}
        drift = client.get("/v1/diff", params=params).json()
        assert drift["identical"] is False
        assert drift["first_divergence"]["reason"] == "argument_drift"
        md = client.get("/v1/diff", params={**params, "format": "md"})
        assert md.headers["content-type"].startswith("text/markdown")
        assert "argument_drift" in md.text
        assert client.get("/v1/diff", params={**params, "format": "html"}).status_code == 422
        missing = client.get("/v1/diff", params={"baseline": "x", "candidate": "candidate-1"})
        assert missing.status_code == 404


async def test_read_token_is_separate_from_the_ingest_token() -> None:
    store = await _seeded_store()
    app = server.create_app(store, ingest_token="ingest", read_token="reader")
    with TestClient(app) as client:
        assert client.get("/v1/suites").status_code == 401
        ingest = {"Authorization": "Bearer ingest"}
        assert client.get("/v1/trajectories", headers=ingest).status_code == 401
        reader = {"Authorization": "Bearer reader"}
        assert client.get("/v1/suites", headers=reader).status_code == 200
        assert client.get("/v1/trajectories", headers=reader).status_code == 200
        assert client.get("/healthz").status_code == 200


@pytest.mark.postgres
def test_cli_push_gate_from_store_and_maintain(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], pg_url: str
) -> None:
    root = tmp_path / "suite"
    assert cli.main(["suite", "record", "--out-dir", str(root), "--name", "pushed"]) == 0
    capsys.readouterr()
    assert cli.main(["suite", "push", str(root), "--database-url", pg_url]) == 0
    assert json.loads(capsys.readouterr().out)["outcome"] == "created"
    assert cli.main(["suite", "push", str(root), "--database-url", pg_url]) == 0
    assert json.loads(capsys.readouterr().out)["outcome"] == "unchanged"

    gate = ["gate", "--suite", "pushed", "--database-url", pg_url]
    assert cli.main(gate) == 0
    assert cli.main([*gate, "--policy", "skip-lookup"]) == 1
    capsys.readouterr()

    assert cli.main(["maintain", "--database-url", pg_url]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["blobs"]["deleted"] == 0  # every blob is referenced and inside the grace
    assert set(report["span_buffer"]) == {"traces", "spans"}
