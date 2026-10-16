from __future__ import annotations

import uuid

from pydantic import JsonValue

from agent_replay.normalize import (
    DropKeys,
    MaskPattern,
    Normalizer,
    SortLists,
    call_ids,
    timestamps,
    uuids,
)

U1 = str(uuid.UUID(int=1))
U2 = str(uuid.UUID(int=2))


def test_timestamps_are_masked_in_any_position_but_dates_are_kept_by_default() -> None:
    norm = Normalizer([timestamps()])
    text = "Now 2026-10-16T09:15:02.123456+00:00, earlier 2026-10-15 08:00Z, cut-off 2026-12-31."
    assert norm.apply(text) == "Now <timestamp>, earlier <timestamp>, cut-off 2026-12-31."
    assert Normalizer([timestamps(include_dates=True)]).apply("by 2026-12-31") == "by <timestamp>"


def test_timestamp_mask_does_not_eat_longer_digit_runs() -> None:
    norm = Normalizer([timestamps()])
    assert norm.apply("2026-10-16T09:15:0299") == "2026-10-16T09:15:0299"
    assert norm.apply("Done at 2026-10-16T09:15:02.") == "Done at <timestamp>."


def test_indexed_uuids_keep_identity_but_not_value() -> None:
    norm = Normalizer([uuids()])
    same = norm.apply({"a": U1, "b": f"ref {U1}", "c": U2})
    other = norm.apply({"a": U2, "b": f"ref {U2}", "c": U1})
    assert same == other == {"a": "<uuid:0>", "b": "ref <uuid:0>", "c": "<uuid:1>"}
    # "Passed back the ID it was given" is different from "used two different IDs".
    assert norm.apply({"a": U1, "b": f"ref {U2}", "c": U2}) != same


def test_plain_masks_collapse_all_values() -> None:
    norm = Normalizer([uuids(indexed=False)])
    assert norm.apply([U1, U2]) == ["<uuid>", "<uuid>"]


def test_numbering_follows_sorted_keys_not_insertion_order() -> None:
    norm = Normalizer([uuids()])
    assert norm.apply({"b": U1, "a": U2}) == norm.apply({"a": U2, "b": U1})
    assert norm.apply({"b": U1, "a": U2}) == {"a": "<uuid:0>", "b": "<uuid:1>"}


def test_apply_many_shares_one_scope() -> None:
    norm = Normalizer([uuids()])
    assert norm.apply_many([U2, U1, U2]) == ["<uuid:0>", "<uuid:1>", "<uuid:0>"]
    assert [norm.apply(v) for v in (U2, U1)] == ["<uuid:0>", "<uuid:0>"]


def test_drop_keys_at_any_depth() -> None:
    norm = Normalizer([DropKeys(frozenset({"request_id"}))])
    value: JsonValue = {"request_id": "r1", "body": [{"request_id": "r2", "x": 1}]}
    assert norm.apply(value) == {"body": [{"x": 1}]}


def test_sort_lists_only_under_named_keys() -> None:
    norm = Normalizer([SortLists(frozenset({"tags"}))])
    assert norm.apply({"tags": ["b", "a"], "steps": ["b", "a"]}) == {
        "tags": ["a", "b"],
        "steps": ["b", "a"],
    }


def test_unordered_lists_number_placeholders_in_sorted_order() -> None:
    norm = Normalizer([uuids(), SortLists(frozenset({"ids"}))])
    first = norm.apply({"ids": [{"id": U1, "n": 2}, {"id": U2, "n": 1}]})
    second = norm.apply({"ids": [{"id": U2, "n": 1}, {"id": U1, "n": 2}]})
    assert first == second


def test_call_ids_and_custom_patterns() -> None:
    norm = Normalizer([call_ids(), MaskPattern(name="ticket", pattern=r"TCK-\d+")])
    assert norm.apply("call_ab12cd34ef and toolu_01XyZ987abc, TCK-42") == (
        "<call_id:0> and <call_id:1>, <ticket>"
    )


def test_chain_is_described_and_keyed() -> None:
    default = Normalizer.default()
    assert default.describe() == ["mask:timestamp", "mask:uuid:indexed", "mask:call_id:indexed"]
    extended = default.with_rules(SortLists(frozenset({"tags"})))
    assert extended.describe()[-1] == "sort:tags"
    assert extended.key() != default.key()
    assert default.rules == Normalizer.default().rules
    assert Normalizer.none().apply({"t": "2026-10-16T09:15:02Z"}) == {"t": "2026-10-16T09:15:02Z"}
