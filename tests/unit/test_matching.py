from __future__ import annotations

import pytest
from pydantic import JsonValue

from agent_replay.matching import MatchEvent, MatchMode, MatchOutcome, MatchPolicy, similarity


@pytest.mark.parametrize(
    "value",
    [None, True, 3, 2.5, "text", [], {}, [1, "a"], {"a": {"b": [1, 2]}}],
)
def test_equal_values_score_one(value: JsonValue) -> None:
    assert similarity(value, value) == 1.0


@pytest.mark.parametrize(
    ("a", "b"),
    [("1", 1), (True, 1), (None, 0), ([1], {"0": 1}), (False, 0.0)],
)
def test_different_json_types_score_zero(a: JsonValue, b: JsonValue) -> None:
    assert similarity(a, b) == 0.0


def test_numbers_use_relative_difference() -> None:
    assert similarity(100, 50) == pytest.approx(0.5)
    assert similarity(59.9, 60) == pytest.approx(1 - 0.1 / 60)
    assert similarity(0, 0.0) == 1.0
    assert similarity(-1, 1) == 0.0
    assert similarity(float("inf"), float("inf")) == 1.0
    assert similarity(float("inf"), 1.0) == 0.0


def test_objects_average_over_the_union_of_keys() -> None:
    assert similarity({"a": 1, "b": 2}, {"a": 1}) == pytest.approx(0.5)
    assert similarity({"a": 1, "b": "x"}, {"a": 1, "b": "y"}) == pytest.approx(0.5)


def test_lists_compare_positionally_over_the_longer_length() -> None:
    assert similarity([1, 2, 3], [1, 2]) == pytest.approx(2 / 3)
    assert similarity([1, 2], [2, 1]) == pytest.approx(0.5)


def test_strings_use_edit_similarity_and_long_strings_token_overlap() -> None:
    assert similarity("ORD-1001", "ord-1001") == pytest.approx(10 / 16)
    assert 0.8 < similarity("refund order", "refund orders") < 1.0
    long_a = " ".join(f"w{i}" for i in range(600))
    long_b = " ".join(f"w{i}" for i in range(300, 900))
    assert similarity(long_a, long_b) == pytest.approx(300 / 900)


def test_similarity_is_symmetric_on_mixed_documents() -> None:
    a: JsonValue = {"order_id": "ORD-1001", "tags": ["a", "b"], "n": 3}
    b: JsonValue = {"order_id": "ORD-1002", "tags": ["a"], "n": 4, "extra": None}
    assert similarity(a, b) == pytest.approx(similarity(b, a))
    assert 0.0 < similarity(a, b) < 1.0


@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.01])
def test_policy_rejects_out_of_range_thresholds(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        MatchPolicy(mode=MatchMode.FUZZY, threshold=threshold)


def test_policy_and_events_serialise() -> None:
    policy = MatchPolicy(mode=MatchMode.FUZZY, threshold=0.8)
    assert policy.describe() == {
        "mode": "fuzzy",
        "threshold": 0.8,
        "normalizer": ["mask:timestamp", "mask:uuid:indexed", "mask:call_id:indexed"],
        "fuzzy_writes": False,
    }
    event = MatchEvent("tool", "lookup", MatchOutcome.FUZZY, 3, 0.912345)
    assert event.to_json() == {
        "kind": "tool",
        "name": "lookup",
        "outcome": "fuzzy",
        "recorded_index": 3,
        "score": 0.9123,
    }
