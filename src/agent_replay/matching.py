"""How a replayed request is matched to a recorded interaction.

Matching happens in three tiers, cheapest first:

1. **exact**: the raw content key is equal (the Day 1 behaviour).
2. **normalized**: equal after the ``Normalizer`` chain (timestamps, UUIDs, call IDs, ...).
3. **fuzzy** (opt-in): the most similar unconsumed recording of the *same* tool or model whose
   normalised content scores at or above ``threshold``.

Fuzzy matching is a tool for replaying a changed agent against an old recording when small,
irrelevant argument drift should not end the run. It is never allowed to paper over a write:
side-effecting tools match at most at tier 2 unless ``fuzzy_writes`` is set explicitly, because
serving a recorded refund for a request with a different amount would hide exactly the kind of
regression replay exists to catch. Every match is reported as a ``MatchEvent`` so a gate can
treat fuzzy-served steps as drift rather than as identical behaviour.
"""

from __future__ import annotations

import difflib
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

from pydantic import JsonValue

from agent_replay.normalize import Normalizer

# Above this length strings are compared by token-set overlap: ``SequenceMatcher`` is quadratic
# in the worst case and full conversations can be long.
_LONG_STRING = 2_000


class MatchMode(StrEnum):
    STRICT = "strict"
    FUZZY = "fuzzy"


class MatchOutcome(StrEnum):
    EXACT = "exact"
    NORMALIZED = "normalized"
    FUZZY = "fuzzy"
    LIVE = "live"  # cassette miss on a read-only tool, executed live by the sandbox
    BLOCKED = "blocked"  # cassette miss on a side-effecting tool, refused by the sandbox
    MISS = "miss"  # cassette miss that ended the run
    UNKNOWN_TOOL = "unknown_tool"  # the agent under test does not offer this tool


@dataclass(frozen=True, slots=True)
class MatchPolicy:
    mode: MatchMode = MatchMode.STRICT
    threshold: float = 0.9
    normalizer: Normalizer = field(default_factory=Normalizer.default)
    fuzzy_writes: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {self.threshold}")

    def describe(self) -> dict[str, JsonValue]:
        return {
            "mode": self.mode.value,
            "threshold": self.threshold,
            "normalizer": list(self.normalizer.describe()),
            "fuzzy_writes": self.fuzzy_writes,
        }


@dataclass(frozen=True, slots=True)
class MatchEvent:
    """One request served (or refused) during replay."""

    kind: Literal["llm", "tool"]
    name: str  # model for LLM requests, tool name for tool calls
    outcome: MatchOutcome
    recorded_index: int | None = None  # step index in the recording that served it
    score: float | None = None  # similarity, for fuzzy matches

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "kind": self.kind,
            "name": self.name,
            "outcome": self.outcome.value,
            "recorded_index": self.recorded_index,
            "score": None if self.score is None else round(self.score, 4),
        }


def similarity(a: JsonValue, b: JsonValue) -> float:  # noqa: PLR0911 - type dispatch
    """Structural similarity of two JSON values in [0, 1]; 1.0 iff they are equal.

    Objects score the mean over the union of keys (a missing key scores 0), lists compare
    position by position over the longer length, strings use edit similarity, numbers use
    relative difference. Values of different JSON types score 0.
    """
    if a == b and type(a) is type(b):
        return 1.0
    if isinstance(a, bool) or isinstance(b, bool):
        return 0.0
    if isinstance(a, int | float) and isinstance(b, int | float):
        return _number_similarity(float(a), float(b))
    if isinstance(a, str) and isinstance(b, str):
        return _string_similarity(a, b)
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        keys = a.keys() | b.keys()
        if not keys:
            return 1.0
        return sum(similarity(a[k], b[k]) if k in a and k in b else 0.0 for k in keys) / len(keys)
    if isinstance(a, list) and isinstance(b, list):
        longest = max(len(a), len(b))
        if longest == 0:
            return 1.0
        return sum(similarity(x, y) for x, y in zip(a, b, strict=False)) / longest
    return 0.0


def _number_similarity(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)):
        return 1.0 if a == b else 0.0
    scale = max(abs(a), abs(b))
    if scale == 0.0:
        return 1.0
    return max(0.0, 1.0 - abs(a - b) / scale)


def _string_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if max(len(a), len(b)) > _LONG_STRING:
        ta, tb = set(a.split()), set(b.split())
        union = ta | tb
        return len(ta & tb) / len(union) if union else 1.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
