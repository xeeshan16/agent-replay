"""Canonical JSON encoding and content fingerprints.

Replay matches recorded interactions by *content*, not by position or ID, so two requests that
are semantically equal must hash to the same key regardless of dict ordering or whitespace.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel


def _default(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, set | frozenset):
        return sorted(value, key=repr)
    raise TypeError(f"value of type {type(value).__name__} is not JSON serialisable")


def canonical_json(value: Any) -> str:
    """Encode ``value`` as compact JSON with sorted keys and no insignificant whitespace."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_default,
    )


def fingerprint(*parts: Any, length: int = 32) -> str:
    """Return a stable hex digest of the canonical encoding of ``parts``."""
    digest = hashlib.sha256(canonical_json(list(parts)).encode("utf-8")).hexdigest()
    return digest[:length]
