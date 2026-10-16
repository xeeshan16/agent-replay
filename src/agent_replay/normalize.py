"""Normalisation of volatile values before replay matching.

A replayed agent rarely produces byte-identical requests: the system prompt carries the current
time, writes carry freshly minted idempotency keys, providers mint new tool-call IDs, and a list
built from a set comes out in a different order in another process. None of these are behaviour
changes, but all of them change a content hash.

A ``Normalizer`` is an ordered chain of rules applied in one walk over a JSON value. Rules can
rewrite strings (mask timestamps, UUIDs, arbitrary patterns), drop object keys, and declare lists
under given keys as unordered. Masks are either plain (``<timestamp>``) or *indexed* by first
appearance (``<uuid:0>``, ``<uuid:1>``): an indexed mask keeps identity, so "the agent passed back
the ID it was given" still differs from "the agent invented a different ID", while the concrete
value is ignored.

Indexing is scoped to one ``apply`` call. Callers normalise a whole request (or a whole
trajectory) at once so the same value maps to the same placeholder everywhere in it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from pydantic import JsonValue

from agent_replay.canonical import canonical_json, fingerprint

_ISO_TIMESTAMP = (
    r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}"
    r"(?::\d{2}(?:\.\d{1,9})?)?"
    r"(?:Z|[+-]\d{2}:?\d{2})?"
    # Not followed by more time digits, so a malformed value is never partially masked.
    r"(?![\d:]|\.\d)"
)
_ISO_DATE = r"\b\d{4}-\d{2}-\d{2}\b"
_UUID = r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
# OpenAI-style ("call_..."), Anthropic-style ("toolu_...") and similar provider call IDs.
_CALL_ID = r"\b(?:call|toolu|fc)_[A-Za-z0-9]{8,}\b"


@dataclass(slots=True)
class _Scope:
    """Placeholder assignments for one ``apply`` call: (rule name, original) -> index."""

    seen: dict[tuple[str, str], int] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)

    def index(self, rule: str, original: str) -> int:
        key = (rule, original)
        if key not in self.seen:
            self.seen[key] = self.counters.get(rule, 0)
            self.counters[rule] = self.seen[key] + 1
        return self.seen[key]

    def copy(self) -> _Scope:
        return _Scope(seen=dict(self.seen), counters=dict(self.counters))


class Rule:
    """One normalisation rule. Subclasses override the hooks they need; defaults are identity."""

    def describe(self) -> str:
        return type(self).__name__

    def string(self, text: str, scope: _Scope) -> str:
        return text

    def drops(self, key: str) -> bool:
        return False

    def unordered(self, key: str | None) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class MaskPattern(Rule):
    """Replace every match of ``pattern`` inside string values with a placeholder."""

    name: str
    pattern: str
    indexed: bool = False
    _regex: re.Pattern[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_regex", re.compile(self.pattern))

    def describe(self) -> str:
        return f"mask:{self.name}{':indexed' if self.indexed else ''}"

    def string(self, text: str, scope: _Scope) -> str:
        if self.indexed:
            return self._regex.sub(
                lambda m: f"<{self.name}:{scope.index(self.name, m.group(0))}>", text
            )
        return self._regex.sub(f"<{self.name}>", text)


@dataclass(frozen=True, slots=True)
class DropKeys(Rule):
    """Remove object members with these keys, at any depth (e.g. ``request_id``)."""

    keys: frozenset[str]

    def describe(self) -> str:
        return f"drop:{','.join(sorted(self.keys))}"

    def drops(self, key: str) -> bool:
        return key in self.keys


@dataclass(frozen=True, slots=True)
class SortLists(Rule):
    """Treat lists stored under these object keys as unordered sets."""

    keys: frozenset[str]

    def describe(self) -> str:
        return f"sort:{','.join(sorted(self.keys))}"

    def unordered(self, key: str | None) -> bool:
        return key is not None and key in self.keys


def timestamps(*, include_dates: bool = False) -> MaskPattern:
    """ISO-8601 date-times; bare dates only when asked (a cut-off date is often meaningful)."""
    pattern = f"{_ISO_TIMESTAMP}|{_ISO_DATE}" if include_dates else _ISO_TIMESTAMP
    return MaskPattern(name="timestamp", pattern=pattern)


def uuids(*, indexed: bool = True) -> MaskPattern:
    return MaskPattern(name="uuid", pattern=_UUID, indexed=indexed)


def call_ids() -> MaskPattern:
    return MaskPattern(name="call_id", pattern=_CALL_ID, indexed=True)


class Normalizer:
    """An ordered, immutable chain of rules."""

    __slots__ = ("_rules",)

    def __init__(self, rules: Iterable[Rule] = ()) -> None:
        self._rules: tuple[Rule, ...] = tuple(rules)

    @classmethod
    def none(cls) -> Normalizer:
        return cls()

    @classmethod
    def default(cls) -> Normalizer:
        """Timestamps, UUIDs (indexed) and provider tool-call IDs (indexed)."""
        return cls((timestamps(), uuids(), call_ids()))

    @property
    def rules(self) -> tuple[Rule, ...]:
        return self._rules

    def with_rules(self, *rules: Rule) -> Normalizer:
        return Normalizer((*self._rules, *rules))

    def describe(self) -> list[str]:
        return [rule.describe() for rule in self._rules]

    def key(self) -> str:
        """Identifies the configuration; recorded with replay results for reproducibility."""
        return fingerprint("normalizer", self.describe(), length=16)

    def apply(self, value: JsonValue) -> JsonValue:
        if not self._rules:
            return value
        return self._walk(value, None, _Scope())

    def apply_many(self, values: Sequence[JsonValue]) -> list[JsonValue]:
        """Normalise several values in one scope, so indexed placeholders agree across them."""
        if not self._rules:
            return list(values)
        scope = _Scope()
        return [self._walk(v, None, scope) for v in values]

    def _walk(self, value: JsonValue, parent_key: str | None, scope: _Scope) -> JsonValue:
        if isinstance(value, str):
            for rule in self._rules:
                value = rule.string(value, scope)
            return value
        if isinstance(value, Mapping):
            # Sorted keys: placeholder numbering must not depend on dict insertion order.
            return {
                k: self._walk(value[k], k, scope)
                for k in sorted(value)
                if not any(rule.drops(k) for rule in self._rules)
            }
        if isinstance(value, list):
            items: list[JsonValue] = list(value)
            if any(rule.unordered(parent_key) for rule in self._rules):
                # Order by each element's normalised form, computed in a throwaway scope so the
                # numbering of the real walk follows the sorted order, not the arrival order.
                items.sort(key=lambda item: canonical_json(self._walk(item, None, scope.copy())))
            return [self._walk(item, None, scope) for item in items]
        return value
