from __future__ import annotations

import pytest

from agent_replay.canonical import canonical_json, fingerprint
from agent_replay.model import ToolCall


def test_key_order_and_whitespace_do_not_matter() -> None:
    assert canonical_json({"b": 1, "a": {"y": 2, "x": 1}}) == '{"a":{"x":1,"y":2},"b":1}'
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})


def test_different_values_produce_different_fingerprints() -> None:
    assert fingerprint("tool", {"id": 1}) != fingerprint("tool", {"id": "1"})


def test_pydantic_models_and_sets_are_encoded() -> None:
    call = ToolCall(id="c1", name="lookup", arguments={"k": "v"})
    assert '"name":"lookup"' in canonical_json({"call": call})
    assert canonical_json({3, 1, 2}) == "[1,2,3]"


def test_unserialisable_values_are_rejected() -> None:
    with pytest.raises(TypeError, match="not JSON serialisable"):
        canonical_json({"x": object()})
    with pytest.raises(ValueError, match="Out of range float"):
        canonical_json(float("nan"))


def test_tool_call_key_ignores_call_id() -> None:
    first = ToolCall(id="call_a", name="lookup", arguments={"order_id": "ORD-1", "x": 1})
    second = ToolCall(id="call_b", name="lookup", arguments={"x": 1, "order_id": "ORD-1"})
    assert first.key() == second.key()
